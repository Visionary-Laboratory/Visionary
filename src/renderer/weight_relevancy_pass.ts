import { gaussianSimilarityShader, weightRelevancyShader } from '../shaders';
import { GPURSSorter } from '../sort';

function toArrayBuffer(view: ArrayBufferView): ArrayBuffer {
  const out = new ArrayBuffer(view.byteLength);
  new Uint8Array(out).set(new Uint8Array(view.buffer, view.byteOffset, view.byteLength));
  return out;
}

export interface RelevancyComputeDispatchArgs {
  weightsBuffer: GPUBuffer | null;
  similarityBuffer: GPUBuffer;
  baseOffset: number;
  numPoints: number;
}

export interface RelevancyRenderArgs {
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

export class WeightRelevancyPass {
  private device: GPUDevice;
  private format: GPUTextureFormat;

  private computePipeline!: GPUComputePipeline;
  private computeBindGroupLayout!: GPUBindGroupLayout;
  private renderPipeline!: GPURenderPipeline;
  private renderPipelineDepth: GPURenderPipeline | null = null;
  private renderBindGroupLayout!: GPUBindGroupLayout;
  private renderBindGroup: GPUBindGroup | null = null;
  private lastRenderBindings: { splat2d?: GPUBuffer; similarity?: GPUBuffer; source?: GPUBuffer } = {};

  private queriesUniform: GPUBuffer; // 5 * 16 vec4 = 80 vec4
  private gramUniform: GPUBuffer;
  private cfgUniform: GPUBuffer;     // vec4<f32>: tau, negMask(u32-as-f32), enabled, reserved
  private paramsUniform: GPUBuffer;  // 4 * u32
  private dummyWeights: GPUBuffer;

  private renderTarget: GPUTexture | null = null;
  private renderTargetView: GPUTextureView | null = null;
  private renderTargetSize: [number, number] = [0, 0];

  constructor(device: GPUDevice, format: GPUTextureFormat) {
    this.device = device;
    this.format = format;

    this.queriesUniform = device.createBuffer({
      label: 'Relevancy queries uniform (pos+4 neg)',
      size: 80 * 16, // 80 * vec4<f32>
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.gramUniform = device.createBuffer({
      label: 'Relevancy gram uniform (C*C^T)',
      size: 64 * 64 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.cfgUniform = device.createBuffer({
      label: 'Relevancy cfg uniform',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.paramsUniform = device.createBuffer({
      label: 'Relevancy params uniform',
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.dummyWeights = device.createBuffer({
      label: 'Relevancy dummy weights buffer',
      size: 256,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    device.queue.writeBuffer(this.queriesUniform, 0, new Float32Array(80 * 4));
    device.queue.writeBuffer(this.gramUniform, 0, new Float32Array(64 * 64));
    // default: tau=10, negMask=0, enabled=0
    device.queue.writeBuffer(this.cfgUniform, 0, new Float32Array([10.0, 0.0, 0.0, 0.0]));
    device.queue.writeBuffer(this.paramsUniform, 0, new Uint32Array([0, 0, 0, 0]));
  }

  initialize(): void {
    this.createComputePipeline();
    this.createRenderPipeline();
  }

  setGramMatrix(gram: Float32Array): void {
    if (gram.length !== 64 * 64) {
      console.warn('[WeightRelevancyPass] Gram matrix must be length 4096 (64x64).');
      return;
    }
    this.device.queue.writeBuffer(this.gramUniform, 0, toArrayBuffer(gram));
  }

  /** qPacked must be 5*64 floats: [pos64, neg0..neg3]. */
  setQueriesPacked(qPacked: Float32Array): void {
    if (qPacked.length !== 5 * 64) {
      console.warn('[WeightRelevancyPass] Queries packed must be length 320 (5x64).');
      return;
    }
    // pack f32[5*64] into vec4 array of length 80
    const vec4s = new Float32Array(80 * 4);
    vec4s.set(qPacked);
    this.device.queue.writeBuffer(this.queriesUniform, 0, toArrayBuffer(vec4s));
  }

  setConfig(enabled: boolean, tau: number, negMask: number): void {
    const t = Number.isFinite(tau) ? Math.max(1e-6, tau) : 10.0;
    const maskU32 = (negMask >>> 0);
    const maskF32 = new Float32Array(new Uint32Array([maskU32]).buffer)[0];
    this.device.queue.writeBuffer(this.cfgUniform, 0, new Float32Array([t, maskF32, enabled ? 1.0 : 0.0, 0.0]));
  }

  getRenderTarget(): GPUTexture | null {
    return this.renderTarget;
  }

  recordComputePass(encoder: GPUCommandEncoder, args: RelevancyComputeDispatchArgs): void {
    if (!this.computePipeline) return;
    const hasWeights = args.weightsBuffer ? 1 : 0;
    this.device.queue.writeBuffer(this.paramsUniform, 0, new Uint32Array([args.baseOffset, args.numPoints, hasWeights, 0]));

    const bg = this.device.createBindGroup({
      label: 'Relevancy compute bg',
      layout: this.computeBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.queriesUniform } },
        { binding: 1, resource: { buffer: args.weightsBuffer ?? this.dummyWeights } },
        { binding: 2, resource: { buffer: args.similarityBuffer } },
        { binding: 3, resource: { buffer: this.paramsUniform } },
        { binding: 4, resource: { buffer: this.gramUniform } },
        { binding: 6, resource: { buffer: this.cfgUniform } },
      ],
    });

    const pass = encoder.beginComputePass({ label: 'Relevancy compute pass' });
    pass.setPipeline(this.computePipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(args.numPoints / 256));
    pass.end();
  }

  recordRenderPass(encoder: GPUCommandEncoder, args: RelevancyRenderArgs): void {
    this.ensureRenderTarget(args.width, args.height);
    if (!this.renderTargetView) return;

    if (
      !this.renderBindGroup ||
      this.lastRenderBindings.splat2d !== args.splat2DBuffer ||
      this.lastRenderBindings.similarity !== args.similarityBuffer ||
      this.lastRenderBindings.source !== args.sourceIndicesBuffer
    ) {
      this.renderBindGroup = this.device.createBindGroup({
        label: 'Relevancy render bg',
        layout: this.renderBindGroupLayout,
        entries: [
          { binding: 2, resource: { buffer: args.splat2DBuffer } },
          { binding: 3, resource: { buffer: args.similarityBuffer } },
          { binding: 4, resource: { buffer: args.sourceIndicesBuffer } },
        ],
      });
      this.lastRenderBindings = { splat2d: args.splat2DBuffer, similarity: args.similarityBuffer, source: args.sourceIndicesBuffer };
    }

    const rpDesc: GPURenderPassDescriptor = {
      label: 'Relevancy render pass',
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
      label: 'Relevancy compute BGL',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }, // queries
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // weights
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }, // similarity
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }, // params
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }, // gram
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }, // cfg
      ],
    });

    const pipelineLayout = this.device.createPipelineLayout({
      label: 'Relevancy compute layout',
      bindGroupLayouts: [this.computeBindGroupLayout],
    });

    const shaderModule = this.device.createShaderModule({
      label: 'Relevancy compute shader',
      code: weightRelevancyShader,
    });

    this.computePipeline = this.device.createComputePipeline({
      label: 'Relevancy compute pipeline',
      layout: pipelineLayout,
      compute: { module: shaderModule, entryPoint: 'main' },
    });
  }

  private createRenderPipeline(): void {
    this.renderBindGroupLayout = this.device.createBindGroupLayout({
      label: 'Relevancy render BGL',
      entries: [
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });

    const shaderModule = this.device.createShaderModule({
      label: 'Gaussian similarity shader (relevancy map)',
      code: gaussianSimilarityShader,
    });

    const pipelineLayout = this.device.createPipelineLayout({
      label: 'Relevancy render pipeline layout',
      bindGroupLayouts: [
        this.renderBindGroupLayout,
        GPURSSorter.createRenderBindGroupLayout(this.device),
      ],
    });

    this.renderPipeline = this.device.createRenderPipeline({
      label: 'Relevancy render pipeline',
      layout: pipelineLayout,
      vertex: { module: shaderModule, entryPoint: 'vs_main', buffers: [] },
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
      primitive: { topology: 'triangle-strip', frontFace: 'ccw' },
      multisample: {},
    });
  }

  private getDepthPipeline(depthFormat: GPUTextureFormat): GPURenderPipeline {
    if (this.renderPipelineDepth && (this.renderPipelineDepth as any).__depthFormat === depthFormat) {
      return this.renderPipelineDepth;
    }
    const shaderModule = this.device.createShaderModule({
      label: 'Gaussian similarity shader (relevancy map, depth)',
      code: gaussianSimilarityShader,
    });

    const pipelineLayout = this.device.createPipelineLayout({
      label: 'Relevancy render pipeline layout (depth)',
      bindGroupLayouts: [
        this.renderBindGroupLayout,
        GPURSSorter.createRenderBindGroupLayout(this.device),
      ],
    });

    const pipe = this.device.createRenderPipeline({
      label: `Relevancy render pipeline (depth-${depthFormat})`,
      layout: pipelineLayout,
      vertex: { module: shaderModule, entryPoint: 'vs_main', buffers: [] },
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
      label: 'Relevancy render target',
      size: { width, height },
      format: 'rgba16float',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.renderTargetView = this.renderTarget.createView();
    this.renderTargetSize = [width, height];
  }
}

