// onnx_gpu_io.ts
import * as ort from 'onnxruntime-web/webgpu';
import { readONNXCountBuffer } from '../utils/debug-gpu-buffers';
import { mat4 } from "gl-matrix";
import { PrecisionMetadata, PrecisionConfig, calcSizeInBytes, dataTypeToOrtString } from './precision-types';
import { PrecisionDetector } from './precision-detector';
export type OnnxGpuIOConfig = {
  modelUrl: string;            // ONNX 模型路径
  maxPoints?: number;          // 预设最大点数（可选，会从metadata推断）
  device: GPUDevice;           // Use app's existing WebGPU device to avoid device mismatch
  verbose?: boolean;           // Enable verbose debug logging
};


let debug_graph = false
// —— 工具：从任意元信息对象里拿到 dims —— //
function parseDimsFromAnything(m: any): number[] | undefined {
  if (!m) return undefined;
  // 1) onnxruntime-web TensorMetadata
  if (Array.isArray(m.dimensions)) return m.dimensions as number[];
  // 2) 你截图这种：{ isTensor, name, shape, type }
  if (Array.isArray(m.shape)) return m.shape as number[];
  // 3) ValueInfoProto
  const dims = m?.type?.tensorType?.shape?.dim;
  if (Array.isArray(dims)) {
    const out: number[] = [];
    for (const d of dims) {
      if (typeof d?.dimValue === 'number') out.push(Number(d.dimValue));
      else if (typeof d?.dimParam === 'string') out.push(-1);
      else return undefined;
    }
    return out;
  }
  return undefined;
}



export class OnnxGpuIO {
  public device!: GPUDevice;
  public session!: ort.InferenceSession;
  private verbose = false;

  // 全局推理串行化协调器，避免 ORT WebGPU IOBinding 并发导致的 Session 冲突
  private static _runChain: Promise<void> = Promise.resolve();
  public static async runExclusive<T>(fn: () => Promise<T>): Promise<T> {
    // 将任务串联到全局链上，保证同一时刻只有一个推理在执行
    const prev = OnnxGpuIO._runChain;
    let release!: () => void;
    OnnxGpuIO._runChain = new Promise<void>(r => (release = r));
    // 等待前序任务完成（忽略其异常，以免阻塞后续）
    try { await prev; } catch (_) {}
    try {
      const result = await fn();
      release();
      return result;
    } catch (e) {
      release();
      throw e;
    }
  }

  // Color detection fields
  private colorMode: 'sh' | 'rgb' = 'sh';
  private colorDim = 48;  // 48 for SH, 3 for RGB
  private colorOutputName: string | null = null;

  // Capacity detection fields
  private capacity!: number;          // Detected maxPoints from metadata
  private gaussOutputName: string | null = null;
  private gaussFields: number = 10;   // Usually 10 for gaussian attributes

  // 直接暴露给渲染阶段使用的 GPUBuffer
  public gaussBuf!: GPUBuffer; // (maxPoints, 10) f16 → 20B/pt
  public shBuf!: GPUBuffer;    // (maxPoints, colorDim) f16 → variable size
  public countBuf!: GPUBuffer; // i32[1]：本次推理实际点数（不回读 CPU）

  // Input GPU buffers for enableGraphCapture compatibility
  public cameraMatrixBuf!: GPUBuffer; // 4x4 float32 → 64B
  public projMatrixBuf!: GPUBuffer;   // 4x4 float32 → 64B
  public timeBuf!: GPUBuffer;         // 1 float32 → 4B

  public maxPoints!: number;
  public get detectedCapacity(): number { return this.capacity; }
  public get detectedGaussOutputName(): string | null { return this.gaussOutputName; }
  public get detectedGaussFields(): number { return this.gaussFields; }
  public actualPoints!: number; // Actual points returned by the model
  public inputNames!: readonly string[]; // Model's expected input names
  
  // Precision information (detected or overridden)
  private gaussianPrecision!: PrecisionMetadata; // data type for gaussian output
  private colorPrecision!: PrecisionMetadata;    // data type for color output
  
  // Public getters for color detection results
  public get detectedColorMode(): 'sh' | 'rgb' { return this.colorMode; }
  public get detectedColorDim(): number { return this.colorDim; }
  public get detectedColorOutputName(): string | null { return this.colorOutputName; }

  // Conditional logging helpers
  private log(...args: any[]) {
    // this.verbose = true;
    if (this.verbose) console.log(...args);
  }

  private warn(...args: any[]) {
    if (this.verbose) console.warn(...args);
  }

  private table(data: any) {
    if (this.verbose) console.table(data);
  }

  async init(cfg: OnnxGpuIOConfig & { precisionConfig?: PrecisionConfig }) {
    this.device = cfg.device; // Use the app's device to avoid device mismatch
    this.verbose = cfg.verbose ?? false;

    // 0) Initialize ONNX Runtime environment
    this.log('Initializing ONNX Runtime environment...');
    try {
      // Configure ONNX Runtime
      ort.env.wasm.numThreads = 1;
      // ort.env.logLevel = 'verbose';
      ort.env.logLevel = 'warning'; // 设置为warning，避免输出太多日志

      // ort.env.webgpu.profiling = {
      //   mode: 'default',
      //   ondata: (d) => console.log('[profiling]', d) // 临时观察
      // };


      // this.log("profiling enbabled")
      ort.env.wasm.wasmPaths = '/src/ort/'; // Ensure WASM files are accessible
      // 路径配置现在通过 initOrtEnvironment 函数处理
      
      // Set the WebGPU device for ONNX Runtime to use the same device
      // ort.env.webgpu.device =  cfg.device ;

    // 任何时候你要用到 device 时，先做这两句
      // Set env.device for builds/paths that read it. After the dummy session, JSEP defines
      // ort.env.webgpu.device with Object.defineProperty(..., { writable: false }), so assign
      // can throw. Rely on executionProviders.device for session creation when env is read-only.
      try {
        const webgpuEnv = (ort.env as any).webgpu;
        if (webgpuEnv) {
          const desc = Object.getOwnPropertyDescriptor(webgpuEnv, 'device');
          if (!desc || desc.writable !== false) {
            webgpuEnv.device = cfg.device;
          }
        }
      } catch {
        // env.webgpu.device may be read-only (JSEP sets it non-writable); session options pass device
      }    

    this.log('isGPUDevice?', cfg.device && typeof cfg.device.createBuffer === 'function' && !!cfg.device.queue);

      this.log('ONNX Runtime environment configured with provided WebGPU device');
    } catch (error) {
      this.warn('ONNX Runtime environment configuration failed:', error);
    }

    // 1) Debug the modelUrl and try different approaches
    this.log(`Attempting to create ONNX session with model: ${cfg.modelUrl}`);
    this.log(`Model URL type: ${typeof cfg.modelUrl}`);
    if (cfg.modelUrl && cfg.modelUrl.constructor) {
      this.log(`Model URL constructor: ${cfg.modelUrl.constructor.name}`);
    }
    if (cfg.modelUrl && typeof cfg.modelUrl.toString === 'function') {
      this.log(`Model URL toString: ${cfg.modelUrl.toString()}`);
    }
    
    // Ensure modelUrl is a proper string path
    let modelPath: string;
    if (!cfg.modelUrl) {
      throw new Error(`modelUrl is required but was ${cfg.modelUrl}`);
    } else if (typeof cfg.modelUrl === 'string') {
      modelPath = cfg.modelUrl;
    } else if (typeof cfg.modelUrl === 'object' && cfg.modelUrl && 'toString' in cfg.modelUrl && typeof (cfg.modelUrl as any).toString === 'function') {
      modelPath = (cfg.modelUrl as any).toString();
    } else {
      throw new Error(`Invalid modelUrl type: ${typeof cfg.modelUrl}. Expected string path.`);
    }
    
    if (!modelPath || modelPath.trim() === '') {
      throw new Error(`modelUrl cannot be empty. Got: "${modelPath}"`);
    }
    
    // Pass app's GPUDevice in EP options so native WebGPU EP uses it (avoids adapter.requestDevice
    // which fails with "adapter is consumed" when Three.js already created a device from the same adapter).
    const buildSessionOptions = (graphCaptureEnabled: boolean): ort.InferenceSession.SessionOptions => ({
      executionProviders: [{
        name: 'webgpu',
        device: cfg.device, // helen add 0206
        deviceId: 0,
        powerPreference: 'high-performance'
      } as ort.InferenceSession.WebGpuExecutionProviderOption & { device: GPUDevice }], // helen add 0206
      graphOptimizationLevel: 'extended' as const,
      preferredOutputLocation: 'gpu-buffer' as const,
      enableGraphCapture: graphCaptureEnabled && (!debug_graph),
      enableProfiling: debug_graph
    });

    let cachedModelBuffer: Uint8Array | null = null;
    const fetchModelBuffer = async () => {
      if (cachedModelBuffer) return cachedModelBuffer;
      this.log(` Fetching model as ArrayBuffer from: ${modelPath}`);
      const response = await fetch(modelPath);
      if (!response.ok) {
        throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
      }
      const buffer = await response.arrayBuffer();
      this.log(` Model buffer size: ${buffer.byteLength} bytes`);
      cachedModelBuffer = new Uint8Array(buffer);
      return cachedModelBuffer;
    };

    const createSession = async (graphCaptureEnabled: boolean) => {
      const sessionOptions = buildSessionOptions(graphCaptureEnabled);
      this.log(`Creating WebGPU-only ONNX session (graphCapture=${graphCaptureEnabled})...`);
      this.log(`Using model path: "${modelPath}"`);
      this.log(`Session options:`, sessionOptions);

      try {
        this.session = await ort.InferenceSession.create(modelPath as any, sessionOptions);
        this.log(` ONNX session created successfully with WebGPU provider`);
        return;
      } catch (pathError) {
        this.warn(` WebGPU session creation failed, trying ArrayBuffer approach:`, pathError);
        try {
          const buffer = await fetchModelBuffer();
          this.session = await ort.InferenceSession.create(buffer, sessionOptions);
          this.log(` ONNX session created successfully with WebGPU provider (ArrayBuffer)`);
          return;
        } catch (bufferError) {
          console.error(` WebGPU session creation failed with both path and buffer approaches`);
          console.error(`Path error:`, pathError);
          console.error(`Buffer error:`, bufferError);
          const combinedError = new Error(`WebGPU execution provider required but failed to initialize (graphCapture=${graphCaptureEnabled}). Ensure WebGPU is supported and enabled.`);
          (combinedError as any).pathError = pathError;
          (combinedError as any).bufferError = bufferError;
          throw combinedError;
        }
      }
    };

    try {
      await createSession(true);
    } catch (graphCaptureError) {
      const errorMessage = graphCaptureError instanceof Error ? graphCaptureError.message : String(graphCaptureError);
      const alertMessage = `Onnx can not enable WebGPU Graph Capture, system will automatically close this feature and re-initialize.\nError details: ${errorMessage}`;
      const maybeAlert = (globalThis as any)?.alert;
      if (typeof maybeAlert === 'function') {
        maybeAlert(alertMessage);
      } else {
        console.error(alertMessage);
      }
      this.warn(` Graph Capture initialization failed, retrying without it`, graphCaptureError);
      await createSession(false);
    }

    this.log(' Using provided WebGPU device to avoid device mismatch');

    // Log and store model's input/output information
    this.log('📋 Model Input Names:', this.session.inputNames);
    this.log('📋 Model Output Names:', this.session.outputNames);
    this.inputNames = this.session.inputNames;

    // Detect capacity and color output from metadata
    await this.detectFromMetadata();
    
    // Set maxPoints from detected capacity or fallback to config
    this.maxPoints = this.capacity || cfg.maxPoints || 2000000;
    this.log(`📏 Using maxPoints: ${this.maxPoints} (detected: ${this.capacity}, config: ${cfg.maxPoints})`);

    // 2) 精度检测（并允许手动覆盖）
    await this.detectPrecisions(cfg.precisionConfig);

    console.log('gaussianPrecision', this.gaussianPrecision);
    console.log('gaussianPrecision', this.colorPrecision);
    // 3) 预分配 GPUBuffer（按精度与维度计算，16B 对齐）
    const bytesGauss = calcSizeInBytes([this.maxPoints, 10], this.gaussianPrecision);
    const bytesSH    = calcSizeInBytes([this.maxPoints, this.colorDim], this.colorPrecision);
    this.log(` Allocating buffers (gauss ${this.gaussianPrecision.dataType}): ${bytesGauss}B, (color ${this.colorPrecision.dataType}): ${bytesSH}B, channels=${this.colorDim}`);
    this.log(` Color mode: ${this.colorMode}, output name: '${this.colorOutputName}'`);
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.VERTEX;

    this.gaussBuf = this.device.createBuffer({ size: bytesGauss, usage, label: `gaussian_${this.gaussianPrecision.dataType}` });
    this.shBuf    = this.device.createBuffer({ size: bytesSH,    usage, label: `color_${this.colorPrecision.dataType}` });
    this.countBuf = this.device.createBuffer({
      size: Math.ceil(4 / 16) * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, label: 'num_points'
    });

    // 3) 预分配输入的 GPUBuffer（enableGraphCapture 要求所有输入都是 GPU buffer）
    const inputUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    const align16 = (n: number) => Math.ceil(n / 16) * 16;
    this.cameraMatrixBuf = this.device.createBuffer({
      size: align16(16 * 4), // 4x4 float32
      usage: inputUsage,
      label: 'camera_matrix'
    });
    this.projMatrixBuf = this.device.createBuffer({
      size: align16(16 * 4), // 4x4 float32  
      usage: inputUsage,
      label: 'projection_matrix'
    });
    this.timeBuf = this.device.createBuffer({
      size: align16(4), // 1 float32
      usage: inputUsage,
      label: 'time_input'
    });
  }

  /**
   * Detect capacity and color output from ONNX model metadata
   */
  private async detectFromMetadata(): Promise<void> {
    this.log(' Detecting capacity and color output from model metadata...');
    
    try {
      // Detect capacity from gaussian output metadata
      if (await this.pickCapacityFromMetadataOrProbe()) {
        this.log(` Capacity detected from metadata: ${this.capacity}`);
      } else {
        this.log(' Could not detect capacity from metadata');
      }

      // Detect color output from metadata  
      if (this.pickColorOutputFromMetadata()) {
        this.log(` Color mode detected from metadata: ${this.colorMode} (${this.colorDim} channels)`);
      } else {
        this.log(' Could not detect color type from metadata, using defaults');
        // Use safe defaults
        this.colorMode = 'sh';
        this.colorDim = 48;
        this.colorOutputName = 'sh_f16';
      }
      
    } catch (error) {
      console.error(' Metadata detection failed:', error);
      this.warn(' Falling back to defaults');
      
      // Ensure we have valid defaults
      this.colorMode = 'sh';
      this.colorDim = 48;
      this.colorOutputName = 'sh_f16';
      
      this.log(` Detection completed with defaults: capacity=${this.capacity || 'none'}, color=${this.colorMode} (${this.colorDim} channels)`);
    }
  }

  private async detectPrecisions(config?: import('./precision-types').PrecisionConfig): Promise<void> {
    // Debug: print output metadata to help diagnose naming/type
    try {
      const names = this.session.outputNames || [];
      const meta: any = (this.session as any).outputMetadata;
      if (this.verbose) {
        console.log('[ONNX][Debug] outputNames =', names);
        if (meta) {
          for (const k in meta) {
            const m = meta[k];
            const shape = (m?.shape ? `[${m.shape.join(', ')}]` : 'unknown');
            console.log(`[ONNX][Meta] idx=${k} name='${m?.name}' type='${m?.type ?? m?.dataType}' shape=${shape}`);
          }
        }
      }
    } catch {}

    const gaussName = this.gaussOutputName || (this.session.outputNames?.find(n => /gauss|gaussian/i.test(n)) ?? 'gaussian_f16');
    // 优先根据 outputMetadata 的 type/dataType 判断；没有 meta 再 fallback 名称后缀
    const gp = PrecisionDetector.detectFromMetadataPreferringNameSuffix(this.session, gaussName);
    this.gaussianPrecision = gp as PrecisionMetadata;
    this.colorPrecision = { ...gp } as PrecisionMetadata; // 颜色跟随高斯
    
    // INT8/UINT8 量化参数
    if (gp.dataType === 'int8' || gp.dataType === 'uint8') {
      const q = PrecisionDetector.extractQuantizationParams(this.session, gaussName);
      this.gaussianPrecision.scale = q.scale ?? 1.0;
      this.gaussianPrecision.zeroPoint = q.zeroPoint ?? 0;

      const colorName = this.colorOutputName || 'sh_f16';
      const cq = PrecisionDetector.extractQuantizationParams(this.session, colorName);
      this.colorPrecision.scale = cq.scale ?? 1.0;
      this.colorPrecision.zeroPoint = cq.zeroPoint ?? 0;
    }

    this.log(`Precision detected: gaussian=${this.gaussianPrecision.dataType} (${this.gaussianPrecision.bytesPerElement}B), color=${this.colorPrecision.dataType}`);
    // Store a short label for UI/debuggers
    (this as any).detectedPrecisionLabel = this.gaussianPrecision?.dataType || 'float16';
  }

  /**
   * Detect capacity from metadata using the approach provided by user
   */
  private async pickCapacityFromMetadataOrProbe(): Promise<boolean> {
    this.log('🧭 Detecting capacity from gaussian output metadata...');
    
    try {
      // 1) Try to access metadata in different ways
      const session = this.session as any;
      this.log(' DEBUG: session object keys:', Object.keys(session));
      
      const metaMap1 = session.outputMetadata;
      this.log(' DEBUG: outputMetadata:', metaMap1);

      for (const key of Object.keys(metaMap1)) {
        console.log(` Found gaussian candidate in outputMetadata: '${key}' dims=${metaMap1[key]?.shape ? `[${metaMap1[key]?.shape.join(', ')}]` : 'undefined'}`);
        if (metaMap1[key].name.startsWith('gauss') || metaMap1[key].name.startsWith('gaussian')) {
          const dims = metaMap1[key]?.shape;
          console.log(` Found gaussian candidate in outputMetadata: '${key}' dims=${dims ? `[${dims.join(', ')}]` : 'undefined'}`);
          this.gaussOutputName = metaMap1[key].name;
          this.gaussFields = dims[-1];
          this.capacity = dims[0] as number;
          this.log(`🦭 Capacity from metadata: ${metaMap1[key].name} -> N=${this.capacity}, fields=${this.gaussFields}`);
          return true;
        
        }
      }
      // Fallback: try minimal CPU inference to get shapes
      console.log('🔬 Fallback: Running minimal CPU inference to detect shapes...');
      return await this.detectCapacityFromCPUInference();
      
    } catch (error) {
      console.warn(' Error accessing output metadata:', error);
      // Fallback: try minimal CPU inference to get shapes
      return await this.detectCapacityFromCPUInference();
    }
  }

  /**
   * Fallback: Use minimal CPU inference to detect capacity and color dimensions
   */
  private async detectCapacityFromCPUInference(): Promise<boolean> {
    this.log('🔬 Running minimal CPU inference to detect model dimensions...');
    
    try {
      // Create minimal dummy inputs using already imported ort
      const dummyFeeds: Record<string, ort.Tensor> = {};
      
      for (const inputName of this.inputNames) {
        if (inputName.toLowerCase().includes('camera') || 
            inputName.toLowerCase().includes('view') || 
            inputName.toLowerCase().includes('matrix')) {
          dummyFeeds[inputName] = new ort.Tensor('float32', new Float32Array(16).fill(0), [4, 4]);
        } else if (inputName.toLowerCase().includes('time') || inputName === 't') {
          dummyFeeds[inputName] = new ort.Tensor('float32', new Float32Array([0.0]), [1]);
        } else if (inputName.toLowerCase().includes('projection') || 
                   inputName.toLowerCase().includes('proj')) {
          dummyFeeds[inputName] = new ort.Tensor('float32', new Float32Array(16).fill(0), [4, 4]);
        }
      }

      this.log(` Running CPU inference with inputs: ${Object.keys(dummyFeeds).join(', ')}`);

      // Run inference to get output shapes
      const result = await this.session.run(dummyFeeds);
      
      // Look for gaussian outputs first
      const candidates = this.session.outputNames
        .filter(n => /gauss|gaussian|means|cov|gaussian_f16/i.test(n));

      for (const outputName of candidates) {
        const tensor = result[outputName];
        if (tensor && tensor.dims.length >= 2) {
          const dims = tensor.dims;
          const last = dims[dims.length - 1];
          const first = dims[0];
          
          this.log(` Found gaussian output '${outputName}': shape [${dims.join(', ')}]`);
          
          if ((last === 10 || last === 9) && Number.isFinite(first) && first > 0) {
            this.gaussOutputName = outputName;
            this.gaussFields = last as number;
            this.capacity = first as number;
            this.log(` Capacity detected from CPU inference: ${outputName} -> N=${this.capacity}, fields=${this.gaussFields}`);
            
            // Also detect color output while we're here
            this.detectColorFromCPUResult(result);
            return true;
          }
        }
      }
      
      this.log(' No gaussian output found with expected dimensions');
      return false;
      
    } catch (error) {
      this.warn(' CPU inference detection failed:', error);
      return false;
    }
  }

  /**
   * Detect color output from CPU inference result
   */
  private detectColorFromCPUResult(result: Record<string, any>): void {
    for (const [outputName, tensor] of Object.entries(result)) {
      const dims = tensor.dims;
      const lowerName = outputName.toLowerCase();
      
      // Check if this looks like a color output
      if ((lowerName.includes('color') || lowerName.includes('sh') || lowerName.includes('rgb')) && 
          dims.length >= 2) {
        const channels = dims[dims.length - 1];
        
        if (channels === 48) {
          this.colorMode = 'sh';
          this.colorDim = 48;
          this.colorOutputName = outputName;
          this.log(` Color mode detected from CPU inference: SH (${channels} channels) - ${outputName}`);
          return;
        } else if (channels === 3 || channels === 4) {
          this.colorMode = 'rgb';
          this.colorDim = channels === 3 ? 4 : channels; // Align to 4
          this.colorOutputName = outputName;
          this.log(` Color mode detected from CPU inference: RGB (${channels} channels) - ${outputName}`);
          return;
        } else if (channels === 12 || channels === 27) {
          this.colorMode = 'sh';
          this.colorDim = channels;
          this.colorOutputName = outputName;
          this.log(` Color mode detected from CPU inference: SH (${channels} channels) - ${outputName}`);
          return;
        }
      }
    }
  }

  /**
   * Try to detect color output type from session metadata using dimensions
   */
  private pickColorOutputFromMetadata(): boolean {
    this.log(' Detecting color output type from metadata dimensions...');
    
    try {
      // Access metadata directly 
      const metaMap: any = (this.session as any).outputMetadata ?? {};
      
      for (const outputName of this.session.outputNames) {
        const lowerName = outputName.toLowerCase();
 
        // Check for color-related names
        if (lowerName.includes('color') || lowerName.includes('sh') || lowerName.includes('rgb')) {
          let used_index = -1;
          for (const i in metaMap) {
            if (metaMap[i]?.name === outputName) {
              used_index = Number(i);
              break;
            }
          }
          if (used_index === -1) continue;
          const meta = metaMap[used_index];
          const dims = meta?.shape as (number[] | undefined);
          
          this.log(` Found potential color output: '${outputName}' dims=${dims ? `[${dims.join(', ')}]` : 'undefined'}`);
          
          if (dims && dims.length >= 2) {
            const channels = dims[dims.length - 1]; // Last dimension is channels
            
            // Use ACTUAL dimensions to determine color mode (not names)
            if (channels === 48) {
              this.colorMode = 'sh';
              this.colorDim = 48;
              this.colorOutputName = outputName;
              this.log(` Detected SH from dimensions: '${outputName}' → ${channels} channels`);
              return true;
            } else if (channels === 3 || channels === 4) {
              this.colorMode = 'rgb';
              this.colorDim = channels; // Store as channels for alignment
              this.colorOutputName = outputName;
              this.log(` Detected RGB from dimensions: '${outputName}' → ${channels} channels`);
              return true;
            } else if (channels === 12 || channels === 27) {
              // Support other SH degrees
              this.colorMode = 'sh';
              this.colorDim = channels;
              this.colorOutputName = outputName;
              this.log(` Detected SH from dimensions: '${outputName}' → ${channels} channels`);
              return true;
            } else {
              this.warn(` Found color output '${outputName}' with unexpected ${channels} channels`);
            }
          }
        }
      }
      
    } catch (error) {
      this.warn(' Error accessing color output metadata:', error);
    }

    // Check for standard names as fallback (SH only - no RGB assumptions)
    const standardSHOutputs = ['sh_f16', 'spherical_harmonics', 'color_sh'];
    for (const standardName of standardSHOutputs) {
      if (this.session.outputNames.includes(standardName)) {
        this.colorMode = 'sh';
        this.colorDim = 48;
        this.colorOutputName = standardName;
        this.log(`📝 Found standard SH output: '${standardName}' → 48 channels (name-based fallback)`);
        return true;
      }
    }

    this.log(' No color output detected from metadata');
    return false;
  }


  /**
   * Update input GPU buffers from CPU data
   * Required for enableGraphCapture compatibility
   */
  updateInputBuffers(cameraMatrix?: Float32Array, projectionMatrix?: Float32Array, time?: number): void {
    if (cameraMatrix) {
      this.log(' DEBUG: Camera Matrix passed to ONNX:');
      this.log(cameraMatrix);
      this.table([
        cameraMatrix.slice(0, 4),
        cameraMatrix.slice(4, 8),
        cameraMatrix.slice(8, 12),
        cameraMatrix.slice(12, 16)
      ]);
      // mat4.transpose(cameraMatrix.buffer, cameraMatrix.buffer);
      this.device.queue.writeBuffer(this.cameraMatrixBuf, 0, cameraMatrix.buffer);
    }
    if (projectionMatrix) {
      this.log(' DEBUG: Projection Matrix passed to ONNX:');
      this.log(projectionMatrix);
      this.table([
        projectionMatrix.slice(0, 4),
        projectionMatrix.slice(4, 8),
        projectionMatrix.slice(8, 12),
        projectionMatrix.slice(12, 16)
      ]);

     // mat4.transpose(projectionMatrix.buffer, projectionMatrix.buffer);
      this.device.queue.writeBuffer(this.projMatrixBuf, 0, projectionMatrix.buffer);
    }
    if (time !== undefined) {
      this.log(` DEBUG: Time passed to ONNX: ${time}`);
      const timeData = new Float32Array([time]);
      this.device.queue.writeBuffer(this.timeBuf, 0, timeData.buffer);
    }
  }

  // 执行推理，将结果直接写入 GPUBuffer
  async runInference(inputData: { cameraMatrix?: Float32Array, projectionMatrix?: Float32Array, time?: number } = {}) {
    return OnnxGpuIO.runExclusive(async () => {
      this.log(` GPU DIRECT: Running WebGPU inference with IOBinding...`);
      this.log(`📏 Using pre-allocated buffers for ${this.maxPoints} points`);

      // Update input buffers with new data
      this.updateInputBuffers(inputData.cameraMatrix, inputData.projectionMatrix, inputData.time);

      // Use maxPoints for IOBinding dimensions (no probing needed)
      const N = this.maxPoints;

      // Create GPU buffer inputs for all model inputs
      const feeds: Record<string, ort.Tensor> = {};
      for (const inputName of this.inputNames) {
        if (inputName.toLowerCase().includes('camera') ||
            inputName.toLowerCase().includes('view') ||
            inputName.toLowerCase().includes('matrix')) {
          feeds[inputName] = ort.Tensor.fromGpuBuffer(
            this.cameraMatrixBuf,
            { dataType: 'float32', dims: [4, 4] }
          );
          this.log(`  📷 Created GPU camera matrix for '${inputName}'`);
        }
        else if (inputName.toLowerCase().includes('time') || inputName === 't') {
          feeds[inputName] = ort.Tensor.fromGpuBuffer(
            this.timeBuf,
            { dataType: 'float32', dims: [1] }
          );
          this.log(`  ⏰ Created GPU time input for '${inputName}'`);
        }
        else if (inputName.toLowerCase().includes('projection') ||
                 inputName.toLowerCase().includes('proj')) {
          feeds[inputName] = ort.Tensor.fromGpuBuffer(
            this.projMatrixBuf,
            { dataType: 'float32', dims: [4, 4] }
          );
          this.log(`  📐 Created GPU projection matrix for '${inputName}'`);
        }
      }

      // Create IOBinding fetches with pre-allocated GPU buffers
      const gaussName = this.gaussOutputName || 'gaussian_f16';
      const fetches: Record<string, ort.Tensor> = {};
      fetches[gaussName] = ort.Tensor.fromGpuBuffer(
        this.gaussBuf,
        { dataType: dataTypeToOrtString(this.gaussianPrecision) as any, dims: [N, 10] }
      );
      fetches['num_points'] = ort.Tensor.fromGpuBuffer(
        this.countBuf,
        { dataType: 'int32', dims: [1] }
      );

      // Add color output with detected name and dimensions
      const colorOutputName = this.colorOutputName || 'sh_f16';
      this.log("----- real needed color channels "+( this.colorDim))
      if (!this.session.outputNames.includes(colorOutputName)) {
        this.warn(` Color output '${colorOutputName}' not found in model outputs`);
        this.warn(`Available outputs: ${this.session.outputNames.join(', ')}`);

        const possibleColorOutput = this.session.outputNames.find(name =>
          name.toLowerCase().includes('color') ||
          name.toLowerCase().includes('sh') ||
          name.toLowerCase().includes('rgb')
        );

        if (possibleColorOutput) {
          this.warn(` Using possible color output: '${possibleColorOutput}'`);
          fetches[possibleColorOutput] = ort.Tensor.fromGpuBuffer(
            this.shBuf,
            { dataType: dataTypeToOrtString(this.colorPrecision) as any, dims: [N, this.colorDim] }
          );
        } else {
          throw new Error(`No suitable color output found in model. Available outputs: ${this.session.outputNames.join(', ')}`);
        }
      } else {
        fetches[colorOutputName] = ort.Tensor.fromGpuBuffer(
          this.shBuf,
          { dataType: dataTypeToOrtString(this.colorPrecision) as any, dims: [N, this.colorDim] }
        );
      }

      this.log(` IOBinding configured: gaussian[${N}x10], ${this.colorMode}[${N}x${this.colorDim}]`);
      this.log(` Input feeds: ${Object.keys(feeds).join(', ')}`);
      this.log(` Output fetches: ${Object.keys(fetches).join(', ')}`);

      try {
        await this.session.run(feeds, fetches);
        if (debug_graph) {
          console.log("this is the table")
          const profRaw = await this.session.endProfiling();
          const prof = typeof profRaw === 'string' ? JSON.parse(profRaw) : profRaw;
          const nodes = (prof.events || [])
            .filter((e: any) => e.cat === 'Node');
          const pick = (e: any) => e.args?.provider || e.args?.execution_provider || 'unknown';
          const nonGpu = nodes
            .map((e: any) => ({name:e.name, op:e.args?.op_name, provider:pick(e)}))
            .filter((x: any) => x.provider !== 'webgpu');
          console.table(nonGpu);
        }

        this.log(` GPU DIRECT SUCCESS: Inference completed with full GPU pipeline`);
        try {
          const actualCount = await readONNXCountBuffer(this.device, this.countBuf);
          this.log(` DEBUG: ONNX wrote count=${actualCount} to GPU buffer`);
          this.actualPoints = actualCount;
        } catch (e) {
          this.warn(' Could not read count buffer for debugging:', e);
        }
      } catch (error) {
        console.error(` WebGPU IOBinding failed:`, error);
        const errorMsg = error instanceof Error ? error.message : String(error);
        console.error('name:', (error as any)?.name, 'message:', (error as any)?.message, 'stack:', (error as any)?.stack);
        throw new Error(`WebGPU inference required but failed: ${errorMsg}`);
      }
    });
  }

  destroy() {
    this.gaussBuf?.destroy?.();
    this.shBuf?.destroy?.();
    this.countBuf?.destroy?.();
    this.cameraMatrixBuf?.destroy?.();
    this.projMatrixBuf?.destroy?.();
    this.timeBuf?.destroy?.();
  }
}