import { gaussianNumDen2Shader, weightNumDen2Shader } from '../shaders';
import { GPURSSorter } from '../sort';

function toArrayBuffer(view: ArrayBufferView): ArrayBuffer {
  const out = new ArrayBuffer(view.byteLength);
  new Uint8Array(out).set(new Uint8Array(view.buffer, view.byteOffset, view.byteLength));
  return out;
}

export interface NumDen2ComputeDispatchArgs {
  weightsBuffer: GPUBuffer | null;
  numDen2Buffer: GPUBuffer;
  baseOffset: number;
  numPoints: number;
}

export interface NumDen2RenderArgs {
  splat2DBuffer: GPUBuffer;
  numDen2Buffer: GPUBuffer;
  sourceIndicesBuffer: GPUBuffer;
  sortRenderBindGroup: GPUBindGroup;
  drawIndirectBuffer: GPUBuffer;
  width: number;
  height: number;
  depthView?: GPUTextureView;
  depthFormat?: GPUTextureFormat;
  useDepth?: boolean;
}

export class WeightNumDen2Pass {
  private device: GPUDevice;
  private format: GPUTextureFormat;

  private computePipeline!: GPUComputePipeline;
  private computeBindGroupLayout!: GPUBindGroupLayout;
  private renderPipeline!: GPURenderPipeline;
  private renderPipelineDepth: GPURenderPipeline | null = null;
  private renderBindGroupLayout!: GPUBindGroupLayout;
  private renderBindGroup: GPUBindGroup | null = null;
  private lastRenderBindings: { splat2d?: GPUBuffer; numden2?: GPUBuffer; source?: GPUBuffer } = {};

  private queryUniform: GPUBuffer;
  private gramUniform: GPUBuffer;
  private paramsUniform: GPUBuffer;
  private dummyWeights: GPUBuffer;

  private renderTarget: GPUTexture | null = null;
  private renderTargetView: GPUTextureView | null = null;
  private renderTargetSize: [number, number] = [0, 0];

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.format = format;

    this.queryUniform = device.createBuffer({
      label: 'NumDen2 query uniform',
      size: 16 * 16, // 16 * vec4<f32>
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.gramUniform = device.createBuffer({
      label: 'NumDen2 gram uniform (C*C^T)',
      size: 64 * 64 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.paramsUniform = device.createBuffer({
      label: 'NumDen2 params uniform',
      size: 16, // 4 * u32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.dummyWeights = device.createBuffer({
      label: 'NumDen2 dummy buffer',
      size: 256,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(this.queryUniform, 0, new Float32Array(64));
    device.queue.writeBuffer(this.gramUniform, 0, new Float32Array(64 * 64));
    device.queue.writeBuffer(this.paramsUniform, 0, new Uint32Array([0, 0, 0, 0]));
  }

  initialize(): void {
    this.createComputePipeline();
    this.createRenderPipeline();
  }

  setQueryWeights(weights: Float32Array): void {
    if (weights.length !== 64) {
      console.warn('[WeightNumDen2Pass] Query weights must be length 64.');
      return;
    }
    this.device.queue.writeBuffer(this.queryUniform, 0, toArrayBuffer(weights));
  }

  setGramMatrix(gram: Float32Array): void {
    if (gram.length !== 64 * 64) {
      console.warn('[WeightNumDen2Pass] Gram matrix must be length 4096 (64x64).');
      return;
    }
    this.device.queue.writeBuffer(this.gramUniform, 0, toArrayBuffer(gram));
  }

  getRenderTarget(): GPUTexture | null {
    return this.renderTarget;
  }

  recordComputePass(encoder: GPUCommandEncoder, args: NumDen2ComputeDispatchArgs): void {
    if (!this.computePipeline) return;
    const hasWeights = args.weightsBuffer ? 1 : 0;
    this.device.queue.writeBuffer(this.paramsUniform, 0, new Uint32Array([args.baseOffset, args.numPoints, hasWeights, 0]));

    const bg = this.device.createBindGroup({
      label: 'NumDen2 compute bg',
      layout: this.computeBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.queryUniform } },
        { binding: 1, resource: { buffer: args.weightsBuffer ?? this.dummyWeights } },
        { binding: 2, resource: { buffer: args.numDen2Buffer } },
        { binding: 3, resource: { buffer: this.paramsUniform } },
        { binding: 4, resource: { buffer: this.gramUniform } },
      ],
    });

    const pass = encoder.beginComputePass({ label: 'NumDen2 compute pass' });
    pass.setPipeline(this.computePipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(args.numPoints / 256));
    pass.end();
  }

  recordRenderPass(encoder: GPUCommandEncoder, args: NumDen2RenderArgs): void {
    this.ensureRenderTarget(args.width, args.height);
    if (!this.renderTargetView) return;

    // Create or reuse bind group (per global buffers).
    if (
      !this.renderBindGroup ||
      this.lastRenderBindings.splat2d !== args.splat2DBuffer ||
      this.lastRenderBindings.numden2 !== args.numDen2Buffer ||
      this.lastRenderBindings.source !== args.sourceIndicesBuffer
    ) {
      this.renderBindGroup = this.device.createBindGroup({
        label: 'NumDen2 render bg',
        layout: this.renderBindGroupLayout,
        entries: [
          { binding: 2, resource: { buffer: args.splat2DBuffer } },
          { binding: 3, resource: { buffer: args.numDen2Buffer } },
          { binding: 4, resource: { buffer: args.sourceIndicesBuffer } },
        ],
      });
      this.lastRenderBindings = { splat2d: args.splat2DBuffer, numden2: args.numDen2Buffer, source: args.sourceIndicesBuffer };
    }

    const rpDesc: GPURenderPassDescriptor = {
      label: 'NumDen2 render pass',
      colorAttachments: [{
        view: this.renderTargetView,
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    };

    const useDepth = !!args.useDepth && !!args.depthView && !!args.depthFormat;
    if (useDepth) {
      (rpDesc as any).depthStencilAttachment = {
        view: args.depthView!,
        depthLoadOp: 'load',
        depthStoreOp: 'store',
      };
    }

    const pass = encoder.beginRenderPass(rpDesc);
    pass.setBindGroup(0, this.renderBindGroup!);
    pass.setBindGroup(1, args.sortRenderBindGroup);
    pass.setPipeline(useDepth ? this.getDepthPipeline(args.depthFormat!) : this.renderPipeline);
    pass.drawIndirect(args.drawIndirectBuffer, 0);
    pass.end();
  }

  private createComputePipeline(): void {
    this.computeBindGroupLayout = this.device.createBindGroupLayout({
      label: 'NumDen2 compute BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }, // query_weights
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // weights
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // numden2_out
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }, // params
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }, // gram
      ],
    });

    const pipelineLayout = this.device.createPipelineLayout({
      label: 'NumDen2 compute layout',
      bindGroupLayouts: [this.computeBindGroupLayout],
    });

    const shaderModule = this.device.createShaderModule({
      label: 'NumDen2 compute shader',
      code: weightNumDen2Shader,
    });

    this.computePipeline = this.device.createComputePipeline({
      label: 'NumDen2 compute pipeline',
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: 'main' },
    });
  }

  private createRenderPipeline(): void {
    this.renderBindGroupLayout = this.device.createBindGroupLayout({
      label: 'NumDen2 render BGL',
      entries: [
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });

    const shaderModule = this.device.createShaderModule({
      label: 'Gaussian numden2 shader',
      code: gaussianNumDen2Shader,
    });

    const pipelineLayout = this.device.createPipelineLayout({
      label: 'NumDen2 render pipeline layout',
      bindGroupLayouts: [
        this.renderBindGroupLayout,
        GPURSSorter.createRenderBindGroupLayout(this.device),
      ],
    });

    this.renderPipeline = this.device.createRenderPipeline({
      label: 'NumDen2 render pipeline',
      layout: pipelineLayout,
      vertex: { module: shaderModule, entryPoint: 'vs_main', buffers: [] },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-strip', frontFace: 'ccw' },
      multisample: {},
    });
  }

  private getDepthPipeline(depthFormat: GPUTextureFormat): GPURenderPipeline {
    if (this.renderPipelineDepth && (this.renderPipelineDepth as any).__depthFormat === depthFormat) {
      return this.renderPipelineDepth;
    }
    const shaderModule = this.device.createShaderModule({
      label: 'Gaussian numden2 shader (depth)',
      code: gaussianNumDen2Shader,
    });

    const pipelineLayout = this.device.createPipelineLayout({
      label: 'NumDen2 render pipeline layout (depth)',
      bindGroupLayouts: [
        this.renderBindGroupLayout,
        GPURSSorter.createRenderBindGroupLayout(this.device),
      ],
    });

    const pipe = this.device.createRenderPipeline({
      label: `NumDen2 render pipeline (depth-${depthFormat})`,
      layout: pipelineLayout,
      vertex: { module: shaderModule, entryPoint: 'vs_main', buffers: [] },
      fragment: {
        module: shaderModule,
        entryPoint: 'fs_main',
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { topology: 'triangle-strip', frontFace: 'ccw' },
      depthStencil: {
        format: depthFormat,
        depthWriteEnabled: false,
        depthCompare: 'less',
      },
      multisample: {},
    });

    (pipe as any).__depthFormat = depthFormat;
    this.renderPipelineDepth = pipe;
    return pipe;
  }

  private ensureRenderTarget(width: number, height: number): void {
    if (this.renderTarget && this.renderTargetView && this.renderTargetSize[0] === width && this.renderTargetSize[1] === height) {
      return;
    }
    this.renderTarget?.destroy();
    this.renderTarget = this.device.createTexture({
      label: 'NumDen2 render target',
      size: { width, height },
      format: this.format,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.renderTargetView = this.renderTarget.createView();
    this.renderTargetSize = [width, height];
  }
}

