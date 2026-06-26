import { gaussianSimilarityShader, weightSimilarityShader } from '../shaders';
import { GPURSSorter } from '../sort';

function toArrayBuffer(view: ArrayBufferView): ArrayBuffer {
  // WebGPU types can be strict about accepting only ArrayBuffer-backed views.
  // Copy to a fresh ArrayBuffer to avoid SharedArrayBuffer-related type mismatches.
  const out = new ArrayBuffer(view.byteLength);
  new Uint8Array(out).set(new Uint8Array(view.buffer, view.byteOffset, view.byteLength));
  return out;
}

export interface WeightComputeDispatchArgs {
  weightsBuffer: GPUBuffer | null;
  similarityBuffer: GPUBuffer;
  baseOffset: number;
  numPoints: number;
}

export interface WeightRenderArgs {
  splat2DBuffer: GPUBuffer;
  similarityBuffer: GPUBuffer;
  sourceIndicesBuffer: GPUBuffer;
  sortRenderBindGroup: GPUBindGroup;
  drawIndirectBuffer: GPUBuffer;
  width: number;
  height: number;
  depthView?: GPUTextureView;
  depthFormat?: GPUTextureFormat;
  useDepth?: boolean;
}

export class WeightSimilarityPass {
  private device: GPUDevice;
  private format: GPUTextureFormat;

  private computePipeline!: GPUComputePipeline;
  private computeBindGroupLayout!: GPUBindGroupLayout;
  private renderPipeline!: GPURenderPipeline;
  private renderPipelineDepth: GPURenderPipeline | null = null;
  private renderBindGroupLayout!: GPUBindGroupLayout;
  private renderBindGroup: GPUBindGroup | null = null;
  private lastRenderBindings: { splat2d?: GPUBuffer; similarity?: GPUBuffer; source?: GPUBuffer } = {};

  private queryUniform: GPUBuffer;
  private gramUniform: GPUBuffer;
  private softmaxUniform: GPUBuffer;
  private paramsUniform: GPUBuffer;
  private dummyWeights: GPUBuffer;

  private renderTarget: GPUTexture | null = null;
  private renderTargetView: GPUTextureView | null = null;
  private renderTargetSize: [number, number] = [0, 0];

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.format = format;

    this.queryUniform = device.createBuffer({
      label: 'Weight query uniform',
      size: 16 * 16, // 16 * vec4<f32>
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.gramUniform = device.createBuffer({
      label: 'Weight gram uniform (C*C^T)',
      size: 64 * 64 * 4, // 4096 * f32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.softmaxUniform = device.createBuffer({
      label: 'Weight softmax params uniform',
      size: 16, // vec4<f32>
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.paramsUniform = device.createBuffer({
      label: 'Weight params uniform',
      size: 16, // 4 * u32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.dummyWeights = device.createBuffer({
      label: 'Weight dummy buffer',
      size: 256,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.queryUniform, 0, new Float32Array(64));
    device.queue.writeBuffer(this.gramUniform, 0, new Float32Array(64 * 64));
    // default: temperature=1.0, useSoftmax=0 (disabled)
    device.queue.writeBuffer(this.softmaxUniform, 0, new Float32Array([1.0, 0.0, 0.0, 0.0]));
    device.queue.writeBuffer(this.paramsUniform, 0, new Uint32Array([0, 0, 0, 0]));
  }

  initialize(): void {
    this.createComputePipeline();
    this.createRenderPipeline();
  }

  setQueryWeights(weights: Float32Array): void {
    if (weights.length !== 64) {
      console.warn('[WeightSimilarityPass] Query weights must be length 64.');
      return;
    }
    this.device.queue.writeBuffer(this.queryUniform, 0, toArrayBuffer(weights));
  }

  setGramMatrix(gram: Float32Array): void {
    if (gram.length !== 64 * 64) {
      console.warn('[WeightSimilarityPass] Gram matrix must be length 4096 (64x64).');
      return;
    }
    this.device.queue.writeBuffer(this.gramUniform, 0, toArrayBuffer(gram));
  }

  setSoftmaxConfig(useSoftmax: boolean, temperature: number): void {
    const t = Number.isFinite(temperature) ? Math.max(1e-6, temperature) : 1.0;
    const u = useSoftmax ? 1.0 : 0.0;
    this.device.queue.writeBuffer(this.softmaxUniform, 0, new Float32Array([t, u, 0.0, 0.0]));
  }

  getRenderTarget(): GPUTexture | null {
    return this.renderTarget;
  }

  ensureRenderTarget(width: number, height: number): GPUTextureView {
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    if (this.renderTarget && this.renderTargetSize[0] === w && this.renderTargetSize[1] === h) {
      return this.renderTargetView!;
    }

    if (this.renderTarget) {
      this.renderTarget.destroy();
    }

    this.renderTarget = this.device.createTexture({
      label: 'Weight map render target',
      size: { width: w, height: h },
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.renderTargetView = this.renderTarget.createView();
    this.renderTargetSize = [w, h];
    return this.renderTargetView;
  }

  recordComputePass(encoder: GPUCommandEncoder, args: WeightComputeDispatchArgs): void {
    const hasWeights = args.weightsBuffer ? 1 : 0;
    this.device.queue.writeBuffer(
      this.paramsUniform,
      0,
      new Uint32Array([args.baseOffset, args.numPoints, hasWeights, 0])
    );

    const bg = this.device.createBindGroup({
      label: 'Weight similarity compute bg',
      layout: this.computeBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.queryUniform } },
        { binding: 1, resource: { buffer: args.weightsBuffer ?? this.dummyWeights } },
        { binding: 2, resource: { buffer: args.similarityBuffer } },
        { binding: 3, resource: { buffer: this.paramsUniform } },
        { binding: 4, resource: { buffer: this.gramUniform } },
        { binding: 5, resource: { buffer: this.softmaxUniform } },
      ],
    });

    const pass = encoder.beginComputePass({ label: 'weight similarity compute' });
    pass.setPipeline(this.computePipeline);
    pass.setBindGroup(0, bg);
    const workgroups = Math.ceil(args.numPoints / 256);
    pass.dispatchWorkgroups(workgroups, 1, 1);
    pass.end();
  }

  recordRenderPass(encoder: GPUCommandEncoder, args: WeightRenderArgs): GPUTextureView {
    const targetView = this.ensureRenderTarget(args.width, args.height);
    const useDepth = !!args.useDepth && !!args.depthView;

    if (!this.renderBindGroup || this.lastRenderBindings.splat2d !== args.splat2DBuffer ||
        this.lastRenderBindings.similarity !== args.similarityBuffer ||
        this.lastRenderBindings.source !== args.sourceIndicesBuffer) {
      this.renderBindGroup = this.device.createBindGroup({
        label: 'Weight map render bg',
        layout: this.renderBindGroupLayout,
        entries: [
          { binding: 2, resource: { buffer: args.splat2DBuffer } },
          { binding: 3, resource: { buffer: args.similarityBuffer } },
          { binding: 4, resource: { buffer: args.sourceIndicesBuffer } },
        ],
      });
      this.lastRenderBindings = {
        splat2d: args.splat2DBuffer,
        similarity: args.similarityBuffer,
        source: args.sourceIndicesBuffer,
      };
    }

    const passDesc: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: targetView,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    };

    if (useDepth && args.depthView && args.depthFormat) {
      (passDesc as any).depthStencilAttachment = {
        view: args.depthView,
        depthLoadOp: 'load',
        depthStoreOp: 'store',
        depthClearValue: 1.0,
      };
      if (!this.renderPipelineDepth || (this.renderPipelineDepth as any)._depthFormat !== args.depthFormat) {
        this.renderPipelineDepth = this.createDepthPipeline(args.depthFormat);
        (this.renderPipelineDepth as any)._depthFormat = args.depthFormat;
      }
    }

    const pass = encoder.beginRenderPass(passDesc);
    pass.setBindGroup(0, this.renderBindGroup);
    pass.setBindGroup(1, args.sortRenderBindGroup);
    pass.setPipeline(useDepth && this.renderPipelineDepth ? this.renderPipelineDepth : this.renderPipeline);
    pass.drawIndirect(args.drawIndirectBuffer, 0);
    pass.end();

    return targetView;
  }

  private createComputePipeline(): void {
    this.computeBindGroupLayout = this.device.createBindGroupLayout({
      label: 'Weight similarity compute BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ],
    });

    const pipelineLayout = this.device.createPipelineLayout({
      label: 'Weight similarity compute layout',
      bindGroupLayouts: [this.computeBindGroupLayout],
    });

    const shaderModule = this.device.createShaderModule({
      label: 'Weight similarity compute shader',
      code: weightSimilarityShader,
    });

    this.computePipeline = this.device.createComputePipeline({
      label: 'Weight similarity compute pipeline',
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: 'main' },
    });
  }

  private createRenderPipeline(): void {
    this.renderBindGroupLayout = this.device.createBindGroupLayout({
      label: 'Weight map render BGL',
      entries: [
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });

    const shaderModule = this.device.createShaderModule({
      label: 'Gaussian similarity shader',
      code: gaussianSimilarityShader,
    });

    const pipelineLayout = this.device.createPipelineLayout({
      label: 'Weight map render pipeline layout',
      bindGroupLayouts: [
        this.renderBindGroupLayout,
        GPURSSorter.createRenderBindGroupLayout(this.device),
      ],
    });

    this.renderPipeline = this.device.createRenderPipeline({
      label: 'Weight map render pipeline',
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
        buffers: [],
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{
          format: 'rgba16float',
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: {
        topology: 'triangle-strip',
        frontFace: 'ccw',
      },
      multisample: {},
    });
  }

  private createDepthPipeline(depthFormat: GPUTextureFormat): GPURenderPipeline {
    const shaderModule = this.device.createShaderModule({
      label: 'Gaussian similarity shader (depth)',
      code: gaussianSimilarityShader,
    });

    const pipelineLayout = this.device.createPipelineLayout({
      label: 'Weight map render pipeline layout (depth)',
      bindGroupLayouts: [
        this.renderBindGroupLayout,
        GPURSSorter.createRenderBindGroupLayout(this.device),
      ],
    });

    return this.device.createRenderPipeline({
      label: `Weight map render pipeline (depth-${depthFormat})`,
      layout: pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
        buffers: [],
      },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{
          format: 'rgba16float',
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: {
        topology: 'triangle-strip',
        frontFace: 'ccw',
      },
      depthStencil: {
        format: depthFormat,
        depthWriteEnabled: false,
        depthCompare: 'less',
      },
      multisample: {},
    });
  }
}

