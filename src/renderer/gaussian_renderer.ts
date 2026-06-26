// WebGPU Gaussian Splatting Renderer Implementation

import { IRenderer, RenderArgs, RendererConfig, RenderStats } from './index';
import { PointCloud } from '../point_cloud';
import { PerspectiveCamera } from '../camera';
import { GPURSSorter } from '../sort';
import { PointCloudSortStuff } from '../sort/radix_sort';
import { BUFFER_CONFIG } from '../point_cloud/layouts';
import { GaussianPreprocessor } from '../preprocess';
import { gaussianShader } from '../shaders/index';
import { WeightSimilarityPass } from './weight_similarity_pass';
import { WeightNumDen2Pass } from './weight_numden2_pass';
import { WeightRelevancyPass } from './weight_relevancy_pass';

export const DEFAULT_KERNEL_SIZE = 0.3;

/**
 * High-performance WebGPU Gaussian Splatting renderer
 * Features static GPU resource caching to eliminate per-frame allocations
 */
export class GaussianRenderer implements IRenderer {
  private device: GPUDevice;
  private format: GPUTextureFormat;
  private shDegree: number;
  private compressed: boolean;
  private debug: boolean;
  
  // Core GPU resources (static, cached)
  private pipeline!: GPURenderPipeline;
  private pipelineDepth!: GPURenderPipeline;
  private useDepth: boolean = false;
  private depthFormat: GPUTextureFormat = 'depth24plus'; // Configurable depth format
  private pipelineLayout!: GPUPipelineLayout;
  private drawIndirectBuffer!: GPUBuffer;
  
  // Subsystem instances
  private sorter!: GPURSSorter;
  private preprocessorSH!: GaussianPreprocessor;  // For spherical harmonics models
  private preprocessorRGB!: GaussianPreprocessor; // For direct RGB models
  
  // Per-point-cloud resources (cached by point count)
  private sortResourcesCache = new WeakMap<PointCloud, any>();

  // Global-sorting scaffolding
  private globalCapacity = 0; // total allocated splats capacity
  private globalBuffers: {
    splat2D: GPUBuffer;
    similarity: GPUBuffer;
    numDen2: GPUBuffer;
    renderBG: GPUBindGroup;
    sortStuff: PointCloudSortStuff;
  } | null = null;

  // Weight map pass (optional)
  private weightPass: WeightSimilarityPass | null = null;
  private weightMapEnabled = false;
  private numDen2Pass: WeightNumDen2Pass | null = null;
  private weightMapApprox2Enabled = false;
  private relevancyPass: WeightRelevancyPass | null = null;
  private weightMapRelevancyEnabled = false;
  private weightSoftmaxEnabled = false;
  private weightSoftmaxTemperature = 1.0;
  
  // Legacy constructor support
  constructor(device: GPUDevice, format: GPUTextureFormat, shDeg: number, compressed?: boolean);
  // Modern constructor with config object
  constructor(config: RendererConfig);
  // Implementation
  constructor(deviceOrConfig: GPUDevice | RendererConfig, format?: GPUTextureFormat, shDeg?: number, compressed = false) {
    if ('device' in deviceOrConfig) {
      // Modern config-based constructor
      this.device = deviceOrConfig.device;
      this.format = deviceOrConfig.format;
      this.shDegree = deviceOrConfig.shDegree;
      this.compressed = deviceOrConfig.compressed ?? false;
      this.debug = deviceOrConfig.debug ?? false;
    } else {
      // Legacy constructor for backward compatibility
      this.device = deviceOrConfig;
      this.format = format!;
      this.shDegree = shDeg!;
      this.compressed = compressed;
      this.debug = false;
    }
  }
  
  /**
   * Ensure sorter is initialized (legacy method name for compatibility)
   */
  async ensureSorter(): Promise<void> {
    await this.initialize();
  }
  
  /**
   * Initialize all GPU resources asynchronously
   * Called once during renderer setup
   */
  async initialize(): Promise<void> {
    // Initialize subsystems
    await this.initializeSorter();
    await this.initializePreprocessor();
    
    // Create static GPU resources
    this.createPipelineLayout();
    this.createRenderPipeline();
    this.createIndirectDrawBuffer();

    // Initialize global buffers with a reasonable default capacity
    // This ensures global buffers are always available
    const defaultCapacity = 1000000; // 1M splats initial capacity
    this.ensureGlobalCapacity(defaultCapacity);

    // Expose a global handle for console debugging (safe in dev)
    try { (globalThis as any).gaussianRenderer = this; } catch {}

    if (this.debug) {
      console.log(`GaussianRenderer initialized: ${this.format}, SH degree ${this.shDegree}, global capacity ${this.globalCapacity}`);
    }
  }
  
  
  /**
   * Record render commands to the provided render pass
   */
  render(pass: GPURenderPassEncoder, pointCloud: PointCloud): void {
    const sortStuff = this.getSortResources(pointCloud);
    
    // Bind resources and pipeline
    pass.setBindGroup(0, pointCloud.renderBindGroup()); // Point cloud data
    pass.setBindGroup(1, sortStuff.sorter_render_bg);    // Sorted indices
    pass.setPipeline(this.useDepth && this.pipelineDepth ? this.pipelineDepth : this.pipeline);
    
    // Indirect draw using pre-computed instance count
    pass.drawIndirect(this.drawIndirectBuffer, 0);
  }
  
  /**
   * Get pipeline information for external integrations
   */
  /** Enable or disable depth testing for GS render pipeline */
  public setDepthEnabled(enabled: boolean) { this.useDepth = !!enabled; }
  
  /**
   * Set depth format and recreate depth pipeline if format changed
   * @param format - WebGPU depth texture format (e.g., 'depth24plus', 'depth32float')
   */
  public setDepthFormat(format: GPUTextureFormat): void {
    if (this.depthFormat !== format) {
      this.depthFormat = format;
      // Recreate depth pipeline with new format
      this.createDepthPipeline();
      
      if ((globalThis as any).GS_DEPTH_DEBUG) {
        console.log('[GaussianRenderer] Depth format changed to:', format);
        console.log('[GaussianRenderer] Depth pipeline recreated');
      }
    }
  }
  
  /**
   * Create depth-enabled pipeline variant
   */
  private createDepthPipeline(): void {
    const shaderModule = this.device.createShaderModule({
      label: 'Gaussian Shader Module',
      code: gaussianShader,
    });
    
    this.pipelineDepth = this.device.createRenderPipeline({
      label: `Gaussian Render Pipeline (Depth-${this.depthFormat})`,
      layout: this.pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
        buffers: [],
      },
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
      primitive: {
        topology: 'triangle-strip',
        frontFace: 'ccw',
      },
      depthStencil: {
        format: this.depthFormat,
        depthWriteEnabled: false,
        depthCompare: 'less',
      },
      multisample: {},
    });
  }

  getPipelineInfo() {
    return {
      format: this.format,
      bindGroupLayouts: [
        PointCloud.renderBindGroupLayout(this.device),
        GPURSSorter.createRenderBindGroupLayout(this.device),
      ],
    };
  }
  
  /**
   * Get rendering statistics for profiling
   */
  getRenderStats(pointCloud: PointCloud): RenderStats {
    const sortStuff = this.sortResourcesCache.get(pointCloud);
    return {
      gaussianCount: pointCloud.numPoints,
      visibleSplats: sortStuff?.num_points ?? 0,
      memoryUsage: this.estimateMemoryUsage(pointCloud),
    };
  }


  /**
   * prepare multiple point clouds.
   * Current turn: falls back to per-model prepare to keep behavior unchanged.
   */
  prepareMulti(
    encoder: GPUCommandEncoder,
    queue: GPUQueue,
    pointClouds: PointCloud[],
    args: RenderArgs
  ): void {

    if (pointClouds.length === 0) return;

    // Compute total points and per-model offsets
    const offsets: number[] = [];
    let total = 0;
    for (const pc of pointClouds) {
      offsets.push(total);
      total += pc.numPoints;
    }
    this._dlog('[prepareMulti] total points =', total, 'offsets =', offsets);

    // Ensure global buffers capacity
    this.ensureGlobalCapacity(total);
    if (!this.globalBuffers) return;
    this._dlog('[prepareMulti] using global capacity =', this.globalCapacity);

    if (this.weightMapEnabled) {
      this.ensureWeightPass();
    }
    if (this.weightMapApprox2Enabled) {
      this.ensureNumDen2Pass();
    }
    if (this.weightMapRelevancyEnabled) {
      this.ensureRelevancyPass();
    }

    // Reset global sorter state
    this.sorter.recordResetIndirectBuffer(this.globalBuffers.sortStuff.sorter_dis, this.globalBuffers.sortStuff.sorter_uni, queue);

    // Identity model matrix (Phase B 后可传 per-model 变换)
    // Dispatch preprocess for each model into its [offset, offset+count) slice
    for (let i = 0; i < pointClouds.length; i++) {
      const pc = pointClouds[i];
      const base = offsets[i];
      this._dlog(`[prepareMulti] dispatch model #${i} baseOffset=${base} count=${pc.numPoints}`);
      
      // Check if this point cloud has a count buffer (ONNX models)
      let countBuffer: GPUBuffer | undefined;
      if ('countBuffer' in pc && typeof pc.countBuffer === 'function') {
        countBuffer = pc.countBuffer();
        if (countBuffer) {
          this._dlog(`[prepareMulti] Model #${i} has ONNX count buffer`);
        }
      }
      
      // Select appropriate preprocessor for this model
      const colorMode = this.getColorMode(pc);
      const preprocessor = colorMode === 'rgb' ? this.preprocessorRGB : this.preprocessorSH;
      
      const renderSettings = this.buildRenderSettings(pc, args);
      
      // Debug: Log what matrix we're passing
      // console.log(`Dispatching model #${i} (baseOffset=${base}) with transform:`);
      // const transform = Array.from(pc.transform);
      // console.log(`   [${transform.slice(0, 4).map(x => x.toFixed(3)).join(', ')}]`);
      // console.log(`   [${transform.slice(4, 8).map(x => x.toFixed(3)).join(', ')}]`);
      // console.log(`   [${transform.slice(8, 12).map(x => x.toFixed(3)).join(', ')}]`);
      // console.log(`   [${transform.slice(12, 16).map(x => x.toFixed(3)).join(', ')}]`);
      
      (preprocessor as any).dispatchModel({
        camera: args.camera,
        viewport: args.viewport,
        pointCloud: pc,
        sortStuff: this.globalBuffers.sortStuff,
        settings: renderSettings,
        modelMatrix: pc.transform,  // Use the point cloud's transform matrix
        baseOffset: base,
        global: { splat2D: this.globalBuffers.splat2D },
        countBuffer: countBuffer,  // Pass the count buffer if available
      }, encoder);

      if (this.weightMapEnabled && this.weightPass && this.globalBuffers) {
        const weightBuffer = (pc as any).getWeightBuffer?.() ?? null;
        this.weightPass.recordComputePass(encoder, {
          weightsBuffer: weightBuffer,
          similarityBuffer: this.globalBuffers.similarity,
          baseOffset: base,
          numPoints: pc.numPoints,
        });
      }

      if (this.weightMapApprox2Enabled && this.numDen2Pass && this.globalBuffers) {
        const weightBuffer = (pc as any).getWeightBuffer?.() ?? null;
        this.numDen2Pass.recordComputePass(encoder, {
          weightsBuffer: weightBuffer,
          numDen2Buffer: this.globalBuffers.numDen2,
          baseOffset: base,
          numPoints: pc.numPoints,
        });
      }

      if (this.weightMapRelevancyEnabled && this.relevancyPass && this.globalBuffers) {
        const weightBuffer = (pc as any).getWeightBuffer?.() ?? null;
        this.relevancyPass.recordComputePass(encoder, {
          weightsBuffer: weightBuffer,
          similarityBuffer: this.globalBuffers.similarity,
          baseOffset: base,
          numPoints: pc.numPoints,
        });
      }
    }

    // One global sort using indirect dispatch produced during preprocess
    this.sorter.recordSortIndirect(this.globalBuffers.sortStuff, this.globalBuffers.sortStuff.sorter_dis, encoder);

    // Update indirect draw argument instanceCount from sort uniform (keys_size)
    encoder.copyBufferToBuffer(this.globalBuffers.sortStuff.sorter_uni, 0, this.drawIndirectBuffer, 4, 4);
    this._dlog('[prepareMulti] recorded global sort & updated instanceCount from sorter_uni');
  }

  /**
   * Phase B API (M1 scaffolding): record render for multiple point clouds.
   * Current turn: falls back to per-model render to keep behavior unchanged.
   */
  renderMulti(pass: GPURenderPassEncoder, _pointClouds: PointCloud[]): void {
    if (!this.globalBuffers) return;

    // Bind global buffers
    pass.setBindGroup(0, this.globalBuffers.renderBG);
    pass.setBindGroup(1, this.globalBuffers.sortStuff.sorter_render_bg);
    pass.setPipeline(this.useDepth && this.pipelineDepth ? this.pipelineDepth : this.pipeline);
    pass.drawIndirect(this.drawIndirectBuffer, 0);
  }

  /**
   * Enable or disable weight similarity map generation.
   */
  public setWeightMapEnabled(enabled: boolean): void {
    this.weightMapEnabled = !!enabled;
    if (this.weightMapEnabled) {
      this.ensureWeightPass();
    }
  }

  /** Enable or disable pixel-level approx2 (num/den2) weight map generation. */
  public setWeightMapApprox2Enabled(enabled: boolean): void {
    this.weightMapApprox2Enabled = !!enabled;
    if (this.weightMapApprox2Enabled) {
      this.ensureNumDen2Pass();
    }
  }

  /** Enable or disable per-GS neg relevancy generation (writes into similarity buffer). */
  public setWeightMapRelevancyEnabled(enabled: boolean): void {
    this.weightMapRelevancyEnabled = !!enabled;
    if (this.weightMapRelevancyEnabled) {
      this.ensureRelevancyPass();
    }
  }

  /** Control whether per-GS weights are softmaxed in the similarity compute, and softmax temperature. */
  public setWeightSoftmaxConfig(enabled: boolean, temperature: number = 1.0): void {
    this.weightSoftmaxEnabled = !!enabled;
    this.weightSoftmaxTemperature = Number.isFinite(temperature) ? Math.max(1e-6, temperature) : 1.0;
    this.ensureWeightPass();
    this.weightPass?.setSoftmaxConfig(this.weightSoftmaxEnabled, this.weightSoftmaxTemperature);
  }

  /**
   * Update query weights (length 64).
   */
  public setWeightQueryWeights(weights: Float32Array, gram?: Float32Array): void {
    this.ensureWeightPass();
    this.weightPass?.setQueryWeights(weights);
    if (gram) {
      this.weightPass?.setGramMatrix(gram);
    }
  }

  /** Update query weights (length 64) and gram matrix for approx2 (num/den2) path. */
  public setWeightQueryNumDen2(weights: Float32Array, gram?: Float32Array): void {
    this.ensureNumDen2Pass();
    this.numDen2Pass?.setQueryWeights(weights);
    if (gram) {
      this.numDen2Pass?.setGramMatrix(gram);
    }
  }

  public setWeightRelevancyData(packedQueries: Float32Array, gram?: Float32Array, enabled: boolean = true, tau: number = 10.0, negMask: number = 0xF): void {
    this.ensureRelevancyPass();
    if (gram) this.relevancyPass?.setGramMatrix(gram);
    this.relevancyPass?.setQueriesPacked(packedQueries);
    this.relevancyPass?.setConfig(enabled, tau, negMask);
  }

  /**
   * Record a weight map render pass into an offscreen texture.
   */
  public recordWeightMapPass(
    encoder: GPUCommandEncoder,
    viewport: [number, number],
    depthView?: GPUTextureView
  ): GPUTexture | null {
    if (!this.weightMapEnabled || !this.weightPass || !this.globalBuffers) return null;
    this.weightPass.recordRenderPass(encoder, {
      splat2DBuffer: this.globalBuffers.splat2D,
      similarityBuffer: this.globalBuffers.similarity,
      sourceIndicesBuffer: this.globalBuffers.sortStuff.source_indices,
      sortRenderBindGroup: this.globalBuffers.sortStuff.sorter_render_bg,
      drawIndirectBuffer: this.drawIndirectBuffer,
      width: viewport[0],
      height: viewport[1],
      depthView,
      depthFormat: this.depthFormat,
      useDepth: this.useDepth,
    });
    return this.weightPass.getRenderTarget();
  }

  /** Record pixel-level approx2 num/den2 pass into an offscreen texture. */
  public recordWeightMapApprox2Pass(
    encoder: GPUCommandEncoder,
    viewport: [number, number],
    depthView?: GPUTextureView
  ): GPUTexture | null {
    if (!this.weightMapApprox2Enabled || !this.numDen2Pass || !this.globalBuffers) return null;
    this.numDen2Pass.recordRenderPass(encoder, {
      splat2DBuffer: this.globalBuffers.splat2D,
      numDen2Buffer: this.globalBuffers.numDen2,
      sourceIndicesBuffer: this.globalBuffers.sortStuff.source_indices,
      sortRenderBindGroup: this.globalBuffers.sortStuff.sorter_render_bg,
      drawIndirectBuffer: this.drawIndirectBuffer,
      width: viewport[0],
      height: viewport[1],
      depthView,
      depthFormat: this.depthFormat,
      useDepth: this.useDepth,
    });
    return this.numDen2Pass.getRenderTarget();
  }

  public recordWeightMapRelevancyPass(
    encoder: GPUCommandEncoder,
    viewport: [number, number],
    depthView?: GPUTextureView
  ): GPUTexture | null {
    if (!this.weightMapRelevancyEnabled || !this.relevancyPass || !this.globalBuffers) return null;
    this.relevancyPass.recordRenderPass(encoder, {
      splat2DBuffer: this.globalBuffers.splat2D,
      similarityBuffer: this.globalBuffers.similarity,
      sourceIndicesBuffer: this.globalBuffers.sortStuff.source_indices,
      sortRenderBindGroup: this.globalBuffers.sortStuff.sorter_render_bg,
      drawIndirectBuffer: this.drawIndirectBuffer,
      width: viewport[0],
      height: viewport[1],
      depthView,
      depthFormat: this.depthFormat,
      useDepth: this.useDepth,
    });
    return this.relevancyPass.getRenderTarget();
  }
  
  // ========== Private Implementation ==========
  
  /**
   * Initialize the radix sorter
   */
  private async initializeSorter(): Promise<void> {
    this.sorter = await GPURSSorter.create(this.device, this.device.queue);
  }
  
  /**
   * Initialize the preprocessing pipeline
   */
  private async initializePreprocessor(): Promise<void> {
    // Create SH preprocessor (default)
    this.preprocessorSH = new GaussianPreprocessor();
    await this.preprocessorSH.initialize(this.device, this.shDegree, false);
    
    // Create RGB preprocessor  
    this.preprocessorRGB = new GaussianPreprocessor();
    await this.preprocessorRGB.initialize(this.device, 0, true); // degree 0 for RGB
    
    console.log('Initialized dual preprocessors: SH and RGB modes');
  }
  
  /**
   * Detect color mode from point cloud to select appropriate preprocessor
   */
  private getColorMode(pointCloud: PointCloud): 'sh' | 'rgb' {
    // Use the point cloud's color mode if available
    return pointCloud.colorMode;
  }
  
  /**
   * Create the pipeline layout (static resource)
   */
  private createPipelineLayout(): void {
    this.pipelineLayout = this.device.createPipelineLayout({
      label: 'Gaussian Renderer Pipeline Layout',
      bindGroupLayouts: [
        PointCloud.renderBindGroupLayout(this.device),      // @group(0)
        GPURSSorter.createRenderBindGroupLayout(this.device), // @group(1)
      ],
    });
  }
  
  /**
   * Create the render pipeline (static resource)
   */
  private createRenderPipeline(): void {
    const shaderModule = this.device.createShaderModule({
      label: 'Gaussian Shader Module',
      code: gaussianShader,
    });
    
    this.pipeline = this.device.createRenderPipeline({
      label: 'Gaussian Render Pipeline',
      layout: this.pipelineLayout,
      vertex: {
        module: shaderModule,
        entryPoint: 'vs_main',
        buffers: [], // No vertex buffers - everything comes from storage
      },
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
      primitive: {
        topology: 'triangle-strip',
        frontFace: 'ccw',
      },
      multisample: {},
    });

    // Create depth pipeline with initial format
    this.createDepthPipeline();
  }
  
  /**
   * Create indirect draw buffer (static resource)
   */
  private createIndirectDrawBuffer(): void {
    this.drawIndirectBuffer = this.device.createBuffer({
      label: 'Gaussian Indirect Draw Buffer',
      size: 16, // DrawIndirect struct: 4 × u32
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    
    // Initialize with default values: {vertexCount=4, instanceCount=0, firstVertex=0, firstInstance=0}
    this.device.queue.writeBuffer(this.drawIndirectBuffer, 0, new Uint32Array([4, 0, 0, 0]));
  }
  
  /**
   * Get sort resources for a point cloud (cached to avoid per-frame allocation)
   */
  private getSortResources(pointCloud: PointCloud): any {
    let sortStuff = this.sortResourcesCache.get(pointCloud);
    
    if (!sortStuff || sortStuff.num_points !== pointCloud.numPoints) {
      // Create new sort resources for this point cloud size
      sortStuff = this.sorter.createSortStuff(this.device, pointCloud.numPoints);
      this.sortResourcesCache.set(pointCloud, sortStuff);
      
      if (this.debug) {
        console.log(`Created sort resources for ${pointCloud.numPoints} points`);
      }
    }
    
    return sortStuff;
  }
  
  /**
   * Build render settings from args and point cloud properties
   */
  private buildRenderSettings(pointCloud: PointCloud, args: RenderArgs) {
    const bbox = pointCloud.bbox;
    const center = pointCloud.center;
    
    // Calculate scene bounds
    const sceneMin = bbox.min;
    const sceneMax = bbox.max;
    const sceneSize = Math.max(
      Math.abs(sceneMax[0] - sceneMin[0]),
      Math.abs(sceneMax[1] - sceneMin[1]),
      Math.abs(sceneMax[2] - sceneMin[2])
    );
    
    return {
      maxSHDegree: Math.min(args.maxSHDegree ?? pointCloud.shDeg, this.shDegree),
      showEnvMap: args.showEnvMap ?? true,
      mipSplatting: args.mipSplatting ?? pointCloud.mipSplatting ?? false,
      kernelSize: args.kernelSize ?? pointCloud.kernelSize ?? DEFAULT_KERNEL_SIZE,
      walltime: args.walltime ?? 1.0,
      sceneExtend: args.sceneExtend ?? sceneSize,
      center: new Float32Array([
        args.sceneCenter?.[0] ?? center[0],
        args.sceneCenter?.[1] ?? center[1],
        args.sceneCenter?.[2] ?? center[2]
      ]),
      clippingBoxMin: new Float32Array([
        args.clippingBox?.min[0] ?? sceneMin[0],
        args.clippingBox?.min[1] ?? sceneMin[1],
        args.clippingBox?.min[2] ?? sceneMin[2]
      ]),
      clippingBoxMax: new Float32Array([
        args.clippingBox?.max[0] ?? sceneMax[0],
        args.clippingBox?.max[1] ?? sceneMax[1],
        args.clippingBox?.max[2] ?? sceneMax[2]
      ])
    };
  }
  
  /**
   * Estimate memory usage for debugging
   */
  private estimateMemoryUsage(pointCloud: PointCloud): number {
    const baseSize = pointCloud.numPoints * (4 * 8 + 4 * 24); // Gaussians + SH coeffs
    const sortSize = pointCloud.numPoints * (4 + 4) * 2; // Keys + payloads, ping-pong
    return baseSize + sortSize;
  }

  /**
   * Ensure global buffers capacity (placeholder for upcoming M1 implementation)
   */
  private async ensureGlobalCapacity(total: number) {
    const needed = Math.max(1, total);
    if (this.globalBuffers && needed <= this.globalCapacity) {
      // Hot-reload / migration safety: older globalBuffers might not have the new numDen2 buffer yet.
      const gbAny = this.globalBuffers as any;
      if (!gbAny.numDen2) {
        gbAny.numDen2 = this.device.createBuffer({
          label: `global/numDen2(cap=${this.globalCapacity})`,
          size: this.globalCapacity * 8, // vec2<f32> per point
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
      }
      return;
    }
    this._dlog('[ensureGlobalCapacity] grow needed. needed=', needed, 'oldCap=', this.globalCapacity);

    // Grow with 1.25x factor to reduce realloc frequency
    const newCapacity = Math.ceil(needed * 1.25);

    // Release old resources if any
    if (this.globalBuffers) {
      try { this.globalBuffers.splat2D.destroy(); } catch {}
      try { (this.globalBuffers as any).similarity?.destroy?.(); } catch {}
      try { (this.globalBuffers as any).numDen2?.destroy?.(); } catch {}
      // sortStuff buffers are owned by sorter; they will be GC'd when unused
      this.globalBuffers = null;
    }

    while(!this.sorter){ // 等待sorter初始化完成
      await new Promise(resolve => setTimeout(resolve, 100));
    }

    // Allocate new global sort stuff and splat2D buffer
    const sortStuff = this.sorter.createSortStuff(this.device, newCapacity);
    const splat2D = this.device.createBuffer({
      label: `global/splat2d(cap=${newCapacity})`,
      size: newCapacity * BUFFER_CONFIG.SPLAT_STRIDE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const similarity = this.device.createBuffer({
      label: `global/similarity(cap=${newCapacity})`,
      size: newCapacity * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    const numDen2 = this.device.createBuffer({
      label: `global/numDen2(cap=${newCapacity})`,
      size: newCapacity * 8, // vec2<f32> per point
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    // Create render bind group compatible with PointCloud.renderBindGroupLayout
    const rbl = PointCloud.renderBindGroupLayout(this.device);
    const renderBG = this.device.createBindGroup({
      label: 'global/render/bg',
      layout: rbl,
      entries: [ { binding: 2, resource: { buffer: splat2D } } ],
    });

    this.globalBuffers = { splat2D, similarity, numDen2, renderBG, sortStuff };
    this.globalCapacity = newCapacity;
    this._dlog('[ensureGlobalCapacity] new capacity =', this.globalCapacity);
  }

  private ensureWeightPass(): void {
    if (!this.weightPass) {
      this.weightPass = new WeightSimilarityPass(this.device, this.format);
      this.weightPass.initialize();
    }
    // Keep compute shader params in sync
    this.weightPass.setSoftmaxConfig(this.weightSoftmaxEnabled, this.weightSoftmaxTemperature);
  }

  private ensureNumDen2Pass(): void {
    if (!this.numDen2Pass) {
      this.numDen2Pass = new WeightNumDen2Pass(this.device, 'rgba16float');
      this.numDen2Pass.initialize();
    }
  }

  private ensureRelevancyPass(): void {
    if (!this.relevancyPass) {
      this.relevancyPass = new WeightRelevancyPass(this.device, this.format);
      this.relevancyPass.initialize();
    }
  }

  /**
   * Debug helper: read current instanceCount from drawIndirectBuffer.
   * Usage in Console: await gaussianRenderer.readInstanceCountDebug()
   */
  public async readInstanceCountDebug(): Promise<number> {
    const staging = this.device.createBuffer({
      label: 'debug/instanceCount',
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = this.device.createCommandEncoder({ label: 'debug/enc' });
    enc.copyBufferToBuffer(this.drawIndirectBuffer, 4, staging, 0, 4);
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    await staging.mapAsync(GPUMapMode.READ);
    const v = new Uint32Array(staging.getMappedRange())[0];
    staging.unmap(); staging.destroy();
    console.log('[debug] instanceCount =', v);
    return v;
  }

  /**
   * Debug ONNX count values through the pipeline
   */
  public async debugONNXCount(): Promise<void> {
    console.log('=== RENDERER DEBUG: ONNX Count Pipeline ===');
    
    if ((this as any)._debugCountBuffer) {
      // Use SH preprocessor for debug by default (both have the same debug methods)
      const preprocessor = this.preprocessorSH;
      
      // Call preprocessor's debug method
      if ('debugCountValues' in preprocessor && typeof preprocessor.debugCountValues === 'function') {
        await (preprocessor as any).debugCountValues();
      }
      
      // Also show point cloud info
      const pc = (this as any)._debugPointCloud;
      if (pc) {
        console.log(`PointCloud.numPoints = ${pc.numPoints}`);
      }
    } else {
      console.log('No ONNX count buffer to debug');
    }
  }
  
  /**
   * Debug: read a small sample of payload indices (global) to inspect ranges.
   * Usage: await gaussianRenderer.readPayloadSampleDebug(8)
   */
  public async readPayloadSampleDebug(n: number = 8): Promise<Uint32Array> {
    if (!this.globalBuffers) throw new Error('globalBuffers not ready');
    const payload = this.globalBuffers.sortStuff.payload_a;
    const size = Math.min(payload.size, n * 4);
    const staging = this.device.createBuffer({ label: 'debug/payloadSample', size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder({ label: 'debug/payloadEnc' });
    enc.copyBufferToBuffer(payload, 0, staging, 0, size);
    this.device.queue.submit([enc.finish()]);
    await this.device.queue.onSubmittedWorkDone();
    await staging.mapAsync(GPUMapMode.READ);
    const data = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap(); staging.destroy();
    console.log('[debug] payload[0..', n, ')=', Array.from(data));
    return data;
  }

  // Dev-only debug logger (disabled by default). Enable via: window.GS_DEBUG_LOGS = true
  private _dlog(...args: any[]): void {
    try { if ((globalThis as any).GS_DEBUG_LOGS) console.log(...args); } catch {}
  }
}
