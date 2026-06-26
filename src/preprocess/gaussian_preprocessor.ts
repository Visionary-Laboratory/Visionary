// 3D Gaussian to 2D screen space preprocessing implementation

import { preprocessShader } from '../shaders/index';
import { UniformBuffer } from '../uniform';
import { PerspectiveCamera } from '../camera';
import { IPreprocessor, PreprocessArgs, PreprocessResults } from './index';
import { PointCloud } from '../point_cloud';
import { readModelParamsNumPoints, debugCountPipeline } from '../utils/debug-gpu-buffers';
import { VIEWPORT_Y_FLIP } from '../utils/transforms';

/**
 * GPU-accelerated 3D Gaussian preprocessing pipeline
 * Projects 3D Gaussians to 2D screen-space splats with spherical harmonics evaluation
 */
export class GaussianPreprocessor implements IPreprocessor {
  private pipeline!: GPUComputePipeline;
  private pipelineLayout!: GPUPipelineLayout;
  private cameraUniforms!: UniformBuffer;
  private settingsUniforms!: UniformBuffer;
  
  private shDegree: number = 3;
  private device!: GPUDevice;
  private m_useRawColor: boolean = false;
  
  // Pre-allocated scratch buffers to avoid per-frame allocations
  private scratchCameraBuffer = new ArrayBuffer(272);
  private scratchCameraView = new Float32Array(this.scratchCameraBuffer);
  private scratchSettingsBuffer = new ArrayBuffer(80);
  private scratchSettingsView = new DataView(this.scratchSettingsBuffer);
  
  /**
   * Initialize GPU resources for preprocessing
   */
  async initialize(device: GPUDevice, shDegree: number, useRawColor: boolean = false): Promise<void> {
    this.device = device;
    this.shDegree = shDegree;
    
    // Create uniform buffers
    this.cameraUniforms = new UniformBuffer(device, new ArrayBuffer(272), "Camera Uniforms");
    this.settingsUniforms = new UniformBuffer(device, new ArrayBuffer(80), "Settings Uniforms");
    
    // Create pipeline layout (add group(4) for per-model params)
    this.pipelineLayout = device.createPipelineLayout({
      label: 'preprocess pipeline layout',
      bindGroupLayouts: [
        UniformBuffer.bindGroupLayout(device),           // @group(0) - Camera
        this.getPointCloudBindGroupLayout(device),       // @group(1) - PointCloud
        this.getSortBindGroupLayout(device),             // @group(2) - Sort
        // Merge model params into group(3) as binding(1) to keep BGL count ≤ 4
        // @group(3) will host: binding(0) RenderSettings, binding(1) ModelParams
        this.getSettingsAndModelParamsBGL(device),       // @group(3)
      ],
    });
    
    // Inject SH degree into shader
    const shaderCode = preprocessShader.replace('<injected>', shDegree.toString());
    
    const shaderModule = device.createShaderModule({
      label: 'preprocess.wgsl',
      code: shaderCode,
    });
    
    // Create compute pipeline with constants
    this.pipeline = device.createComputePipeline({
      label: 'preprocess pipeline',
      layout: this.pipelineLayout,
      compute: { 
        module: shaderModule, 
        entryPoint: 'preprocess',
        constants: {
          USE_RAW_COLOR: useRawColor ? 1 : 0
        }
      },
    });
    this.m_useRawColor = useRawColor
    console.log(`📐 Preprocessor initialized with SH degree ${shDegree}, raw color: ${useRawColor}`);
  }
  
  /**
   * Execute preprocessing compute pass (Phase A 单模型路径，保留兼容)
   */
  // should be throwed
  // preprocess(args: PreprocessArgs, encoder: GPUCommandEncoder): void {
  //   // Pack uniforms
  //   this.packCameraUniforms(args.camera, args.viewport);
  //   this.packSettingsUniforms(args.pointCloud, args.settings);
  //   const initialNumPoints = args.pointCloud.numPoints;
  //   this.packModelParams(this.identityMat4(), 0, initialNumPoints); // 默认单位矩阵 + baseOffset=0 + numPoints
  //   console.log(`📐 Initial num_points packed in CPU buffer: ${initialNumPoints}` + ` using RAWRGB? ${this.m_useRawColor}`);

  //   if (args.countBuffer) {
  //     console.log(`ONNX count buffer detected - will overwrite num_points after flush`);
  //   }
    
  //   // Flush camera and settings uniforms first
  //   this.cameraUniforms.flush(this.device);
  //   this.settingsUniforms.flush(this.device);

  //   // Handle ONNX count buffer if provided
  //   if (args.countBuffer) {

  //     // IMPORTANT: Flush FIRST to write model matrix and baseOffset
  //     this.modelParamsUniforms.flush(this.device);
      
  //     // THEN copy ONNX count to overwrite ONLY the num_points field at offset 68
  //     encoder.copyBufferToBuffer(
  //       args.countBuffer,                    // src (ONNX count buffer)
  //       0,                                   // srcOffset
  //       this.modelParamsUniforms.buffer,     // dst (ModelParams buffer)
  //       68,                                  // dstOffset (num_points field at byte 68)
  //       4                                    // size (4 bytes for u32)
  //     );
  //     console.log('Copied ONNX count to ModelParams.num_points at offset 68 (AFTER flush)');
      
  //     // Store for debugging later
  //     (this as any)._debugCountBuffer = args.countBuffer;
  //     (this as any)._debugMaxPoints = initialNumPoints;
  //   } else {
  //     // Normal path: just flush when no ONNX count buffer
  //     this.modelParamsUniforms.flush(this.device);
  //   }
    
  //   // Begin compute pass
  //   const computePass = encoder.beginComputePass({ 
  //     label: 'preprocess compute pass' 
  //   });
    
  //   // Set pipeline and bind groups
  //   computePass.setPipeline(this.pipeline);
  //   computePass.setBindGroup(0, this.cameraUniforms.bindGroup);
  //   computePass.setBindGroup(1, args.pointCloud.bindGroup());
  //   computePass.setBindGroup(2, this.getSortBindGroup(args.sortStuff));
  //   // group(3): settings + model params packed in one bind group layout
  //   const smBGL = this.pipeline.getBindGroupLayout(3);
  //   const smBG = this.device.createBindGroup({
  //     layout: smBGL,
  //     entries: [
  //       { binding: 0, resource: { buffer: this.settingsUniforms.buffer } },
  //       { binding: 1, resource: { buffer: this.modelParamsUniforms.buffer } },
  //     ],
  //   });
  //   computePass.setBindGroup(3, smBG);
    
  //   // Dispatch workgroups (256 threads per workgroup)
  //   const workgroupSize = 256;
  //   const workgroups = Math.ceil(args.pointCloud.numPoints / workgroupSize);
  //   computePass.dispatchWorkgroups(workgroups, 1, 1);
    
  //   computePass.end();
  // }

  /**
   * Phase B / M1：多模型全局路径的单模型调度
   * 将当前模型写入全局缓冲的指定区段 [baseOffset, baseOffset + count)
   */
  dispatchModel(args: {
    camera: PerspectiveCamera,
    viewport: [number, number],
    pointCloud: PointCloud,
    sortStuff: any,
    settings: any,
    modelMatrix: Float32Array,
    baseOffset: number,
    global: { splat2D: GPUBuffer },
    countBuffer?: GPUBuffer,  // Optional ONNX count buffer
  }, encoder: GPUCommandEncoder): void {
    // Pack uniforms
    this.packCameraUniforms(args.camera, args.viewport);
    this.packSettingsUniforms(args.pointCloud, args.settings);
    
    // Update point cloud's model params buffer with current baseOffset and transform
    args.pointCloud.updateModelParamsWithOffset(args.modelMatrix, args.baseOffset);

    // console.log( ` using RAWRGB? ${this.m_useRawColor}`)

    // Flush uniform data
    this.cameraUniforms.flush(this.device);
    this.settingsUniforms.flush(this.device);
    
    // Handle ONNX count buffer if provided (same as in preprocess)
    // Before dispatch, allow dynamic point cloud to inject precision metadata
    if ('setPrecisionForShader' in args.pointCloud && typeof (args.pointCloud as any).setPrecisionForShader === 'function') {
      try { (args.pointCloud as any).setPrecisionForShader(); } catch {}
    }

    if (args.countBuffer) {
      // IMPORTANT: Flush FIRST to write model matrix and baseOffset
      args.pointCloud.modelParamsUniforms.flush(this.device);
      
      // THEN copy ONNX count to overwrite ONLY the num_points field at offset 68
      encoder.copyBufferToBuffer(
        args.countBuffer,                    // src (ONNX count buffer)
        0,                                   // srcOffset
        args.pointCloud.modelParamsUniforms.buffer,     // dst (ModelParams buffer)
        68,                                  // dstOffset (num_points field at byte 68)
        4                                    // size (4 bytes for u32)
      );
      // console.log(`[dispatchModel] Copied ONNX count to offset 68 for model at baseOffset ${args.baseOffset}`);
    } else {
      // Normal path: just flush when no ONNX count buffer (PLY files, etc.)
      args.pointCloud.modelParamsUniforms.flush(this.device);
    }

    // Create compute pass
    const computePass = encoder.beginComputePass({ label: 'preprocess compute pass (global/M1)' });
    computePass.setPipeline(this.pipeline);

    // group(0): camera
    computePass.setBindGroup(0, this.cameraUniforms.bindGroup);

    // group(1): point cloud inputs + 全局 splat2D 输出
    const pcBGL = this.pipeline.getBindGroupLayout(1);
    const pcBuf = args.pointCloud.getSplatBuffer();
    const pcBG = this.device.createBindGroup({
      label: 'preprocess/pc-global-bg',
      layout: pcBGL,
      entries: [
        { binding: 0, resource: { buffer: pcBuf.gaussianBuffer } },
        { binding: 1, resource: { buffer: pcBuf.shBuffer } },
        { binding: 2, resource: { buffer: args.global.splat2D } }, // 全局输出
        { binding: 3, resource: { buffer: args.pointCloud.uniforms.buffer } },
        { binding: 4, resource: { buffer: args.pointCloud.getExtraPcaBuffer() } },
      ],
    });
    computePass.setBindGroup(1, pcBG);

    // group(2): global sort buffers (传入的 sortStuff 应为全局资源)
    computePass.setBindGroup(2, this.getSortBindGroup(args.sortStuff));

    // group(3): settings + model params组合绑定
    const smBGL = this.pipeline.getBindGroupLayout(3);
    const smBG = this.device.createBindGroup({
      layout: smBGL,
      entries: [
        { binding: 0, resource: { buffer: this.settingsUniforms.buffer } },
        { binding: 1, resource: { buffer: args.pointCloud.modelParamsUniforms.buffer } },
      ],
    });
    computePass.setBindGroup(3, smBG);

    // Dispatch
    const workgroupSize = 256;
    const workgroups = Math.ceil(args.pointCloud.numPoints / workgroupSize);
    computePass.dispatchWorkgroups(workgroups, 1, 1);
    computePass.end();
  }
  
  /**
   * Get bind group layout (required by interface)
   */
  getBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
    return this.pipeline.getBindGroupLayout(0);
  }
  
  /**
   * Pack camera matrices and parameters into uniform buffer
   */
  private packCameraUniforms(camera: PerspectiveCamera, viewport: [number, number]): void {
    // Use pre-allocated scratch buffer to avoid per-frame allocation
    const f32View = this.scratchCameraView;
    
    // View matrix
    const viewMatrix = camera.viewMatrix();
    f32View.set(viewMatrix, 0);
    
    // View inverse matrix
    const viewInverse = this.invertMatrix4(viewMatrix);
    f32View.set(viewInverse, 16);
    
    // Projection matrix (with Y flip for WebGPU)
    const projMatrix = camera.projMatrix();
    const projY = this.multiplyMatrix4(VIEWPORT_Y_FLIP, projMatrix);
    f32View.set(projY, 32);
    
    // Projection inverse matrix (invert original, not Y-flipped)
    const projInverse = this.invertMatrix4(projMatrix);
    f32View.set(projInverse, 48);
    
    // Viewport dimensions
    f32View[64] = viewport[0];
    f32View[65] = viewport[1];
    
    // Focal length
    const focal = camera.projection.focal(viewport);
    f32View[66] = focal[0];
    f32View[67] = focal[1];
    
    this.cameraUniforms.setData(f32View);
  }
  
  /**
   * Pack render settings into uniform buffer
   */
  private packSettingsUniforms(pointCloud: any, settings: any): void {
    // Use pre-allocated scratch buffer to avoid per-frame allocation
    const dataView = this.scratchSettingsView;
    
    let offset = 0;
    
    // Clipping box min (vec4)
    dataView.setFloat32(offset + 0, settings.clippingBoxMin[0], true);
    dataView.setFloat32(offset + 4, settings.clippingBoxMin[1], true);
    dataView.setFloat32(offset + 8, settings.clippingBoxMin[2], true);
    dataView.setFloat32(offset + 12, 0, true);
    offset += 16;
    
    // Clipping box max (vec4)
    dataView.setFloat32(offset + 0, settings.clippingBoxMax[0], true);
    dataView.setFloat32(offset + 4, settings.clippingBoxMax[1], true);
    dataView.setFloat32(offset + 8, settings.clippingBoxMax[2], true);
    dataView.setFloat32(offset + 12, 0, true);
    offset += 16;
    
    // Gaussian scaling
    dataView.setFloat32(offset, settings.gaussianScaling, true);
    offset += 4;
    
    // Max SH degree
    dataView.setUint32(offset, settings.maxSHDegree, true);
    offset += 4;
    
    // Show env map
    dataView.setUint32(offset, settings.showEnvMap ? 1 : 0, true);
    offset += 4;
    
    // Mip splatting
    dataView.setUint32(offset, settings.mipSplatting ? 1 : 0, true);
    offset += 4;
    
    // Kernel size
    dataView.setFloat32(offset, settings.kernelSize, true);
    offset += 4;
    
    // Wall time
    dataView.setFloat32(offset, settings.walltime, true);
    offset += 4;
    
    // Scene extend
    dataView.setFloat32(offset, settings.sceneExtend, true);
    offset += 4;
    
    // Padding to align to 64 bytes
    offset = 64;
    
    // Scene center (vec3 + padding)
    dataView.setFloat32(offset + 0, settings.center[0], true);
    dataView.setFloat32(offset + 4, settings.center[1], true);
    dataView.setFloat32(offset + 8, settings.center[2], true);
    dataView.setFloat32(offset + 12, 0, true);
    
    this.settingsUniforms.setData(dataView);
  }

  /**
   * Pack per-model params (model matrix + baseOffset + num_points)
   */

  private identityMat4(): Float32Array {
    return new Float32Array([
      1,0,0,0,
      0,1,0,0,
      0,0,1,0,
      0,0,0,1,
    ]);
  }
  
  /**
   * Debug method to verify count values after preprocessing
   */
  async debugCountValues(): Promise<void> {
    if ((this as any)._debugCountBuffer) {
      console.log('=== PREPROCESSOR DEBUG ===');
      await debugCountPipeline(
        this.device, 
        (this as any)._debugCountBuffer,
        (this as any).modelParamsUniforms?.buffer || null,
        (this as any)._debugMaxPoints || 0
      );
    }
  }
  
  /**
   * Get point cloud bind group layout  
   */
  private getPointCloudBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
    return device.createBindGroupLayout({
      label: "Point Cloud Bind Group Layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // gaussians
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // sh coeffs
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },           // splat2d output
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },           // uniforms
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } }, // extra PCA
      ],
    });
  }
  
  /**
   * Get sort bind group layout
   */
  private getSortBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
    return device.createBindGroupLayout({
      label: "Sort Preprocess Bind Group Layout", 
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // sort infos
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // keys
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // payloads  
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // dispatch
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }, // source indices
      ],
    });
  }

  /**
   * Settings + ModelParams 合并后的 BGL（group 3）
   */
  private getSettingsAndModelParamsBGL(device: GPUDevice): GPUBindGroupLayout {
    return device.createBindGroupLayout({
      label: "Settings + ModelParams BGL",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }, // RenderSettings
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } }, // ModelParams
      ],
    });
  }
  
  /**
   * Get sort bind group from sort stuff
   */
  private getSortBindGroup(sortStuff: any): GPUBindGroup {
    // This should come from the sortStuff object
    return (sortStuff as any).sorter_bg_pre;
  }
  
  /**
   * Invert a 4x4 matrix
   */
  private invertMatrix4(matrix: ArrayLike<number>): Float32Array {
    const result = new Float32Array(16);
    const m = matrix as any as number[];
    
    result[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + 
               m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    result[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - 
                m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    result[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + 
               m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    result[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - 
                 m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    
    result[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - 
                m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    result[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + 
               m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    result[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - 
                m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    result[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + 
                 m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    
    result[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + 
               m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    result[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - 
                m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    result[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + 
                 m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    result[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - 
                 m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    
    result[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - 
                m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    result[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + 
               m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    result[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - 
                 m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    result[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + 
                 m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    
    let det = m[0] * result[0] + m[1] * result[4] + m[2] * result[8] + m[3] * result[12];
    
    if (Math.abs(det) < 1e-8) {
      throw new Error("Matrix not invertible");
    }
    
    det = 1.0 / det;
    
    for (let i = 0; i < 16; i++) {
      result[i] *= det;
    }
    
    return result;
  }

  /**
   * Multiply two 4x4 matrices
   */
  private multiplyMatrix4(a: ArrayLike<number>, b: ArrayLike<number>): Float32Array {
    const A = a as any as number[];
    const B = b as any as number[];
    const result = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      const ai0 = A[i];
      const ai1 = A[i + 4];
      const ai2 = A[i + 8];
      const ai3 = A[i + 12];
      result[i] = ai0 * B[0] + ai1 * B[1] + ai2 * B[2] + ai3 * B[3];
      result[i + 4] = ai0 * B[4] + ai1 * B[5] + ai2 * B[6] + ai3 * B[7];
      result[i + 8] = ai0 * B[8] + ai1 * B[9] + ai2 * B[10] + ai3 * B[11];
      result[i + 12] = ai0 * B[12] + ai1 * B[13] + ai2 * B[14] + ai3 * B[15];
    }
    return result;
  }
}