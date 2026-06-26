// PointCloud class - GPU buffer management and point cloud data

import { vec3 } from "gl-matrix";
import { UniformBuffer } from "../uniform";
import { Aabb } from "../utils";
import { GenericGaussianPointCloudTS } from "../utils";
import { SplatBuffer } from "./splat_buffer";
import { getBindGroupLayout, getRenderBindGroupLayout, BUFFER_CONFIG } from "./layouts";

function toArrayBuffer(view: ArrayBufferView): ArrayBuffer {
  // WebGPU types can be strict about accepting only ArrayBuffer-backed views.
  // Copy to a fresh ArrayBuffer to avoid SharedArrayBuffer-related type mismatches.
  const out = new ArrayBuffer(view.byteLength);
  new Uint8Array(out).set(new Uint8Array(view.buffer, view.byteOffset, view.byteLength));
  return out;
}

export class PointCloud {
  private splat2DBuffer: GPUBuffer;            // storage for projected splats / sort keys
  protected gaussianBufferGPU: GPUBuffer;        // 3DGS attributes
  protected shBufferGPU: GPUBuffer;              // SH coefficients
  protected extraPcaBufferGPU: GPUBuffer;        // optional extra per-point attributes (vec4<f32>)
  protected weightBufferGPU: GPUBuffer | null = null; // optional per-point 64D weights
  protected weightChannels: number = 64;

  private _bindGroup: GPUBindGroup;            // general compute/prepare BG (if used)
  private _renderBindGroup: GPUBindGroup;      // render pass BG (textures/samplers + storage)

  readonly numPoints: number;
  readonly shDeg: number;
  readonly bbox: Aabb;
  readonly compressed: boolean;
  
  // Color mode: 'sh' for spherical harmonics, 'rgb' for direct RGB color
  public readonly colorMode: 'sh' | 'rgb' = 'sh';

  center: vec3;
  up: vec3 | null;
  
  // Transform matrix for GPU rendering (4x4 column-major)
  // @deprecated - Transform should be managed externally (e.g., by GaussianModel/Object3D)
  // This property is kept for backward compatibility but should not be relied upon.
  // New code should pass transform explicitly to updateModelParamsBuffer()
  readonly transform: Float32Array = new Float32Array([
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
  ]);

  // Optional raster params (mirroring fields in Rust)
  mipSplatting?: boolean;
  kernelSize?: number;
  backgroundColor?: GPUColor;

  // 高斯点缩放参数（独立于Three.js的scale属性）
  private _gaussianScaling: number = 1.0;
  private _maxShDeg: number = 3;
  private _kernelSize: number = 0.1;
  private _opacityScale: number = 1.0;
  private _cutoffScale: number = 1.0;
  private _rendermode: number = 0;

  // Uniforms used by both compute & render stages
  readonly uniforms: UniformBuffer;

  // Per-model parameters uniform buffer (transform matrix + metadata)
  readonly modelParamsUniforms: UniformBuffer;

  /**
   * Create bind group layout for point cloud compute shaders
   * @deprecated Use getBindGroupLayout from layouts.ts instead
   */
  static bindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
    return getBindGroupLayout(device);
  }

  /**
   * Create bind group layout for point cloud render passes
   * @deprecated Use getRenderBindGroupLayout from layouts.ts instead
   */
  static renderBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
    return getRenderBindGroupLayout(device);
  }

  constructor(device: GPUDevice, pc: GenericGaussianPointCloudTS, externalBuffers?: { gaussianBuffer: GPUBuffer, shBuffer: GPUBuffer }) {
    this.numPoints = pc.numPoints();
    this.shDeg = pc.shDegree();
    this._maxShDeg = this.shDeg; // Initialize from source SH degree
    
    this.bbox = new Aabb(pc.bbox().min, pc.bbox().max);
    this.center = (pc.center ? vec3.fromValues(pc.center[0], pc.center[1], pc.center[2]) : vec3.fromValues(0, 0, 0)) as vec3;
    this.up = pc.up ? (vec3.fromValues(pc.up[0], pc.up[1], pc.up[2]) as vec3) : null;
    this.compressed = false; // TODO: set based on source type

    // Use external buffers if provided, otherwise create from CPU data
    if (externalBuffers) {
      // Direct GPU buffer injection for ONNX/dynamic loading
      this.gaussianBufferGPU = externalBuffers.gaussianBuffer;
      this.shBufferGPU = externalBuffers.shBuffer;
      console.log('🌟 PointCloud created with external GPU buffers (no CPU upload)');
    } else {
      // Legacy path: create buffers from CPU data for PLY files
      this.gaussianBufferGPU = device.createBuffer({
        label: "gaussians/storage",
        size: pc.gaussianBuffer().byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(this.gaussianBufferGPU, 0, pc.gaussianBuffer());

      this.shBufferGPU = device.createBuffer({
        label: "sh/storage",
        size: pc.shCoefsBuffer().byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(this.shBufferGPU, 0, pc.shCoefsBuffer());
    }

    // Optional extra PCA RGB buffer (vec4<f32> per point); allocate dummy zeros if missing
    const extraPcaBytes = Math.max(1, this.numPoints) * 16;
    let extraPcaCPU: ArrayBuffer | null = null;
    try {
      if (typeof (pc as any).extraPcaBuffer === 'function') {
        const buf = (pc as any).extraPcaBuffer() as ArrayBuffer;
        if (buf && buf.byteLength >= extraPcaBytes) extraPcaCPU = buf;
      }
    } catch {}
    this.extraPcaBufferGPU = device.createBuffer({
      label: "extra_pca/storage",
      size: extraPcaBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    if (extraPcaCPU) {
      device.queue.writeBuffer(this.extraPcaBufferGPU, 0, extraPcaCPU);
    } else {
      device.queue.writeBuffer(this.extraPcaBufferGPU, 0, new Uint8Array(extraPcaBytes));
    }

    // 2D splat buffer (projected attributes + sort keys). Size depends on pipeline; allocate numPoints * stride.
    this.splat2DBuffer = device.createBuffer({
      label: "splat2d/storage",
      size: Math.max(1, this.numPoints) * BUFFER_CONFIG.SPLAT_STRIDE,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
    });

    // Uniforms (per draw). Layout should mirror Rust `PointCloudUniforms` if present.
    const u32 = new Uint32Array([this.numPoints, this.shDeg, 0, 0]);
    this.uniforms = new UniformBuffer(device, u32, "pointcloud uniforms");
    
    // Model parameters uniform buffer (128 bytes, 16B aligned)
    const modelParamsBuffer = new ArrayBuffer(128);
    const modelParamsF32 = new Float32Array(modelParamsBuffer);
    const modelParamsU32 = new Uint32Array(modelParamsBuffer);
    
    // Initialize with identity matrix
    for (let i = 0; i < 16; i++) {
      modelParamsF32[i] = (i % 5 === 0) ? 1 : 0; // Identity matrix
    }
    // baseOffset at byte offset 64
    modelParamsU32[16] = 0;
    // num_points at byte offset 68  
    modelParamsU32[17] = this.numPoints;
    // gaussianScaling at byte offset 72
    modelParamsF32[18] = this._gaussianScaling;
    // maxShDeg at byte offset 76
    modelParamsU32[19] = this._maxShDeg;
    // kernelSize at byte offset 80
    modelParamsF32[20] = this._kernelSize;
    // opacityScale at byte offset 84
    modelParamsF32[21] = this._opacityScale;
    // cutoffScale at byte offset 88
    modelParamsF32[22] = this._cutoffScale;
    // rendermode at byte offset 92
    modelParamsU32[23] = this._rendermode;
    // --- Extended fields for multi-precision support (defaults) ---
    // gaussDataType at byte offset 96 (0=f32,1=f16,2=i8,3=u8)
    modelParamsU32[24] = 1; // default FP16
    // colorDataType at byte offset 100
    modelParamsU32[25] = 1; // default FP16
    // gaussScale at byte offset 104
    modelParamsF32[26] = 1.0;
    // gaussZeroPoint at byte offset 108
    modelParamsF32[27] = 0.0;
    // colorScale at byte offset 112
    modelParamsF32[28] = 1.0;
    // colorZeroPoint at byte offset 116
    modelParamsF32[29] = 0.0;
    
    this.modelParamsUniforms = new UniformBuffer(device, modelParamsBuffer, "model params");

    // Bind groups
    const bgl = getBindGroupLayout(device);
    this._bindGroup = device.createBindGroup({
      label: "pointcloud/bg",
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: this.gaussianBufferGPU } },
        { binding: 1, resource: { buffer: this.shBufferGPU } },
        { binding: 2, resource: { buffer: this.splat2DBuffer } },
        { binding: 3, resource: { buffer: this.uniforms.buffer } },
        { binding: 4, resource: { buffer: this.extraPcaBufferGPU } },
      ],
    });

    const rbl = getRenderBindGroupLayout(device);
    this._renderBindGroup = device.createBindGroup({
      label: "pointcloud/render/bg",
      layout: rbl,
      entries: [
        { binding: 2, resource: { buffer: this.splat2DBuffer } },
      ],
    });

    // Initial identity matrix is already set above
  }

  bindGroup(): GPUBindGroup { return this._bindGroup; }
  renderBindGroup(): GPUBindGroup { return this._renderBindGroup; }

  /**
   * Replace gaussian/sh storage buffers and rebuild bind group.
   * Intended for runtime precision conversion (e.g., FP32→FP16).
   */
  public replaceStorageBuffers(device: GPUDevice, buffers: { gaussianBuffer: GPUBuffer, shBuffer: GPUBuffer }): void {
    this.gaussianBufferGPU = buffers.gaussianBuffer;
    this.shBufferGPU = buffers.shBuffer;
    const bgl = getBindGroupLayout(device);
    this._bindGroup = device.createBindGroup({
      label: "pointcloud/bg",
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: this.gaussianBufferGPU } },
        { binding: 1, resource: { buffer: this.shBufferGPU } },
        { binding: 2, resource: { buffer: this.splat2DBuffer } },
        { binding: 3, resource: { buffer: this.uniforms.buffer } },
        { binding: 4, resource: { buffer: this.extraPcaBufferGPU } },
      ],
    });
  }

  /** Expose extra PCA buffer for preprocess binding (read-only storage) */
  getExtraPcaBuffer(): GPUBuffer { return this.extraPcaBufferGPU; }

  /**
   * Attach per-point weight buffer (64 channels by default).
   */
  setWeightBufferGPU(buffer: GPUBuffer, channels: number = 64): void {
    this.weightBufferGPU = buffer;
    this.weightChannels = channels;
  }

  /**
   * Create and upload per-point weight buffer from CPU data.
   */
  setWeightBufferFromArray(device: GPUDevice, weights: Float32Array, channels: number = 64): void {
    this.weightChannels = channels;
    this.weightBufferGPU = device.createBuffer({
      label: "weights/storage",
      size: weights.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.weightBufferGPU, 0, toArrayBuffer(weights));
  }

  hasWeightBuffer(): boolean { return !!this.weightBufferGPU; }
  getWeightBuffer(): GPUBuffer | null { return this.weightBufferGPU; }
  getWeightChannels(): number { return this.weightChannels; }

  /**
   * Get SplatBuffer interface for this point cloud
   */
  getSplatBuffer(): SplatBuffer {
    return {
      gaussianBuffer: this.gaussianBufferGPU,
      shBuffer: this.shBufferGPU,
      numPoints: this.numPoints,
      shDegree: this.shDeg,
      bbox: this.bbox
    };
  }
  
  /**
   * Update the model parameters uniform buffer with provided transform matrix
   * @param transformMatrix - 4x4 transform matrix
   * @param baseOffset - base offset for the model in global buffers
   */
  updateModelParamsBuffer(transformMatrix: Float32Array, baseOffset: number = 0): void {
    // ModelParams uniform must be 16B-aligned; current struct requires 120B, so allocate 128B
    const buffer = new ArrayBuffer(128);
    const f32 = new Float32Array(buffer);
    const u32 = new Uint32Array(buffer);

    // Copy transform matrix (16 floats)
    for (let i = 0; i < 16; i++) {
      f32[i] = transformMatrix[i];
    }

    // baseOffset at byte offset 64
    u32[16] = baseOffset;
    // num_points at byte offset 68
    u32[17] = this.numPoints;
    // gaussianScaling at byte offset 72
    f32[18] = this._gaussianScaling;
    // maxShDeg at byte offset 76
    u32[19] = this._maxShDeg;
    // kernelSize at byte offset 80
    f32[20] = this._kernelSize;
    // opacityScale at byte offset 84
    f32[21] = this._opacityScale;
    // cutoffScale at byte offset 88
    f32[22] = this._cutoffScale;
    // rendermode at byte offset 92
    u32[23] = this._rendermode;

    // --- Extended fields for multi-precision support ---
    // gaussDataType at byte offset 96 (0=f32, 1=f16, 2=i8, 3=u8)
    u32[24] = 1; // default FP16
    // colorDataType at byte offset 100
    u32[25] = 1; // default FP16
    // gaussScale at byte offset 104 (for INT8 dequant)
    f32[26] = 1.0;
    // gaussZeroPoint at byte offset 108
    f32[27] = 0.0;
    // colorScale at byte offset 112
    f32[28] = 1.0;
    // colorZeroPoint at byte offset 116
    f32[29] = 0.0;

    this.modelParamsUniforms.setData(new DataView(buffer));

    // console.log(`📤 Updated model params buffer: baseOffset=${baseOffset}, numPoints=${this.numPoints}, gaussianScaling=${this._gaussianScaling}`);
    // console.log(`   Transform: [${transformMatrix[12].toFixed(3)}, ${transformMatrix[13].toFixed(3)}, ${transformMatrix[14].toFixed(3)}]`);
  }

  /**
   * Set the transform matrix for this point cloud
   * @param matrix - 4x4 transform matrix (Float32Array or number[])
   */
  setTransform(matrix: Float32Array | number[]): void {
    const m = matrix instanceof Float32Array ? matrix : new Float32Array(matrix);
    (this.transform as Float32Array).set(m);
    // Update GPU buffer with identity matrix since transform will be applied by caller
    this.updateModelParamsBuffer(this.transform, 0);
  }

  /**
   * Update model parameters buffer with a specific baseOffset (called by preprocessor)
   * @param transformMatrix - 4x4 transform matrix
   * @param baseOffset - base offset for the model in global buffers
   */
  updateModelParamsWithOffset(transformMatrix: Float32Array, baseOffset: number): void {
    this.updateModelParamsBuffer(transformMatrix, baseOffset);
  }

  /**
   * 设置高斯缩放参数
   * @param scale - 缩放因子
   */
  public setGaussianScaling(scale: number): void {
    this._gaussianScaling = scale;
    console.log(`[PointCloud] Gaussian scaling set to: ${scale}`);
  }
  
  /**
   * 获取当前高斯缩放参数
   * @returns 当前缩放值
   */
  public getGaussianScaling(): number {
    return this._gaussianScaling;
  }

  /**
   * 设置球谐等级
   * @param deg - 球谐等级 (0-3)
   */
  public setMaxShDeg(deg: number): void {
    this._maxShDeg = Math.max(0, Math.min(3, deg));
    console.log(`[PointCloud] Max SH degree set to: ${this._maxShDeg}`);
  }
  
  /**
   * 获取当前球谐等级
   * @returns 当前球谐等级
   */
  public getMaxShDeg(): number {
    return this._maxShDeg;
  }

  /**
   * 设置二维核大小
   * @param size - 核大小
   */
  public setKernelSize(size: number): void {
    this._kernelSize = Math.max(0, size);
    console.log(`[PointCloud] Kernel size set to: ${this._kernelSize}`);
  }
  
  /**
   * 获取当前二维核大小
   * @returns 当前核大小
   */
  public getKernelSize(): number {
    return this._kernelSize;
  }

  /**
   * 设置透明度倍数
   * @param scale - 透明度倍数
   */
  public setOpacityScale(scale: number): void {
    this._opacityScale = Math.max(0, scale);
    console.log(`[PointCloud] Opacity scale set to: ${this._opacityScale}`);
  }
  
  /**
   * 获取当前透明度倍数
   * @returns 当前透明度倍数
   */
  public getOpacityScale(): number {
    return this._opacityScale;
  }

  /**
   * 设置最大像素比例倍数
   * @param scale - 像素比例倍数
   */
  public setCutoffScale(scale: number): void {
    this._cutoffScale = Math.max(0.1, scale);
    console.log(`[PointCloud] Cutoff scale set to: ${this._cutoffScale}`);
  }
  
  /**
   * 获取当前最大像素比例倍数
   * @returns 当前像素比例倍数
   */
  public getCutoffScale(): number {
    return this._cutoffScale;
  }

  /**
   * 设置渲染模式
   * @param mode - 渲染模式 (0=颜色, 1=法线, 2=深度, 3=PCA)
   */
  public setRenderMode(mode: number): void {
    this._rendermode = Math.max(0, Math.min(3, mode));
    console.log(`[PointCloud] Render mode set to: ${this._rendermode}`);
  }
  
  /**
   * 获取当前渲染模式
   * @returns 当前渲染模式
   */
  public getRenderMode(): number {
    return this._rendermode;
  }
}