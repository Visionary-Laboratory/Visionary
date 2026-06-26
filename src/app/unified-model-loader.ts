import * as THREE from "three/webgpu";
import { defaultLoader, ThreeJSModelData, ThreeJSDataSource, GaussianDataSource, LoadingOptions, DataSource } from '../io';
import { GaussianModel } from './GaussianModel';
import { GaussianThreeJSRenderer } from './GaussianThreeJSRenderer';
import { GaussianLoader, GaussianLoadOptions } from './managers/gaussian-loader';
import { FileLoader } from './managers/file-loader';
import { ONNXManager } from './managers/onnx-manager';
import { ModelManager } from './managers/model-manager';
import { FBXLoaderManager } from './managers/fbx-loader';
import { FBXLoadOptions } from '../models/fbx-model-wrapper';

// 支持的模型类型
export type ModelType = 'ply' | 'onnx' | 'gltf' | 'glb' | 'obj' | 'fbx' | 'stl' | 'spz' | 'gaussian';

export interface UnifiedLoadOptions {
  /** 模型类型，如果不指定会自动检测 */
  type?: ModelType;
  /** 自定义模型名称 */
  name?: string;

  isGaussian?: boolean;
  /** 相机矩阵（高斯模型需要） */
  cameraMatrix?: Float32Array;
  /** 投影矩阵（高斯模型需要） */
  projectionMatrix?: Float32Array;
  /** 高斯模型特定选项 */
  gaussianOptions?: GaussianLoadOptions;
  /** FBX 模型特定选项 */
  fbxOptions?: FBXLoadOptions;
  /** 原始文件对象（用于blob URL情况） */
  sourceFile?: File;

  onProgress?: (progress: number) => void;
  onError?: (error: Error) => void;
}

export interface LoadResult {
  models: THREE.Object3D[];
  gaussianRenderer?: GaussianThreeJSRenderer;
  sourceFile?: File;
  info: {
    type: ModelType;
    name: string;
    count: number;
    isGaussian: boolean;
  };
}

export class UnifiedModelLoader {
  private renderer: THREE.WebGPURenderer;
  private scene: THREE.Scene;
  private gaussianLoader: GaussianLoader;
  private fbxLoader: FBXLoaderManager;
  private modelManager: ModelManager;

  constructor(renderer: THREE.WebGPURenderer, scene: THREE.Scene) {
    this.renderer = renderer;
    this.scene = scene;

    // 初始化模型管理器
    this.modelManager = new ModelManager();
    const fileLoader = new FileLoader(this.modelManager);
    const onnxManager = new ONNXManager(this.modelManager);
    this.gaussianLoader = new GaussianLoader(fileLoader, onnxManager);
    this.fbxLoader = new FBXLoaderManager(this.modelManager);
  }

  private detectFileType(filename: string): ModelType {
    const lowerName = filename.toLowerCase();

    // 1. 优先检测复合后缀 (compressed.ply)
    if (lowerName.endsWith('.compressed.ply')) {
      return 'gaussian'; // 直接归类为 gaussian，确保走高斯加载流程
    }

    const ext = lowerName.split('.').pop();

    // 2. 检测明确的高斯格式
    // 只要是这些后缀，都应当作高斯模型处理
    if (['onnx', 'sog', 'ksplat', 'splat', 'spz'].includes(ext || '')) {
      return 'gaussian';
    }

    // 3. 常规格式
    if (ext === 'ply') return 'ply'; // ply 可能是点云也可能是高斯，后续逻辑处理
    if (ext === 'fbx') return 'fbx';

    // 其他类型让 UniversalLoader 处理
    return (ext as ModelType) || 'unknown';
  }

  public async loadModel(
    input: File | string,
    options: UnifiedLoadOptions = {}
  ): Promise<LoadResult> {
    try {
      const filename = input instanceof File ? input.name : input;
      const fileType = options.type || this.detectFileType(filename);

      console.log(`开始加载模型: ${filename}, 类型: ${fileType}`);
      let ext = filename.toLowerCase().split('.').pop();
      let isGaussian = ext === 'onnx';
      if (ext === 'ply') {
        if (input instanceof File) {
          // 本地文件判断
          isGaussian = await this.is3dgsPly(input as File);
        } else {
          // 在线 URL 判断
          isGaussian = await this.isGaussianPlyUrl(input as string);
        }
      }
      options.isGaussian = isGaussian;

      // 根据模型类型选择加载方式
      const result = await this.loadModelByType(input, fileType, options);

      console.log(`模型加载完成: ${result.info.name}, 数量: ${result.info.count}`);
      return result;

    } catch (error) {
      const err = error as Error;
      console.error(`模型加载失败: ${err.message}`);
      if (options.onError) {
        options.onError(err);
      }
      throw err;
    }
  }

  /**
   * 根据类型加载模型
   */
  private async loadModelByType(
    input: File | string,
    fileType: ModelType,
    options: UnifiedLoadOptions
  ): Promise<LoadResult> {

    console.log('fileType:', fileType, 'is Gaussian?', options.isGaussian);

    // 明确的高斯格式 -> 走高斯流程
    if (fileType === 'gaussian' || fileType === 'onnx') {
      return await this.loadGaussianModel(input, options);
    }

    // 如果是 PLY，判断 options.isGaussian
    if (fileType === 'ply') {
      if (options.isGaussian) {
        // 是 3DGS PLY -> 走高斯流程
        return await this.loadGaussianModel(input, options);
      } else {
        // 是 Mesh PLY -> 走通用模型流程 (ThreeJS PLYLoader)
        console.log('UnifiedModelLoader: 检测到普通 Mesh PLY');
        // 直接跳到底部去调用 loadWithUniversalLoader
      }
    } else if (options.isGaussian) {
      // 其他情况，如果强行指定了 isGaussian
      return await this.loadGaussianModel(input, options);
    }

    // 处理 FBX 文件
    if (fileType === 'fbx') {
      return await this.loadFBXModel(input, options);
    }

    // 使用 UniversalLoader 统一加载
    const modelData = await this.loadWithUniversalLoader(input, options);
    return await this.processLoadedData(modelData, fileType, options);
  }

  /**
   * 使用 UniversalLoader 统一加载
   */
  private async loadWithUniversalLoader(
    input: File | string,
    options: UnifiedLoadOptions
  ): Promise<DataSource> {
    const loadingOptions: LoadingOptions = {
      onProgress: (progress) => {
        if (options.onProgress) {
          options.onProgress(progress.progress);
        }
      },
      isGaussian: options.isGaussian
    };

    // 使用 UniversalLoader 统一加载
    return input instanceof File
      ? await defaultLoader.loadFile(input, loadingOptions)
      : await defaultLoader.loadUrl(input, loadingOptions);
  }

  /**
   * 处理加载的数据
   */
  private async processLoadedData(
    modelData: DataSource,
    fileType: ModelType,
    options: UnifiedLoadOptions
  ): Promise<LoadResult> {
    const filename = options.sourceFile?.name || 'model';
    const modelName = options.name || filename.split('/').pop()?.split('.')[0] || 'model';

    if (modelData instanceof ThreeJSModelData) {
      // Three.js 模型处理
      console.log(`处理 Three.js 模型`);
      return this.processThreeJSModel(modelData, fileType, modelName, options);
    } else {
      // 高斯模型处理（需要特殊处理）
      console.log(`处理 高斯模型`);
      return this.processGaussianModel(modelData as GaussianDataSource, fileType, modelName, options);
    }
  }

  /**
   * 处理 Three.js 模型
   */
  private processThreeJSModel(
    modelData: ThreeJSModelData,
    fileType: ModelType,
    modelName: string,
    options: UnifiedLoadOptions
  ): LoadResult {
    const object3D = modelData.object3D();
    const models: THREE.Object3D[] = [];

    // 处理不同类型的 Object3D
    if (object3D instanceof THREE.Group) {
      models.push(...object3D.children);
    } else {
      models.push(object3D);
    }

    // 添加到场景
    models.forEach(model => {
      this.scene.add(model);
    });

    // 打印 Three.js 模型信息
    console.log('=== Three.js 模型加载完成 ===');
    console.log('模型名称:', modelName);
    console.log('模型数量:', models.length);
    models.forEach((model, index) => {
      console.log(`模型 ${index + 1}:`);
      console.log('  Object3D UUID:', model.uuid);
      console.log('  Object3D 类型:', model.constructor.name);
      console.log('  Object3D 名称:', model.name || '未命名');
    });
    if (options.sourceFile) {
      console.log('原始文件路径:', options.sourceFile.name);
      console.log('文件大小:', options.sourceFile.size, 'bytes');
      console.log('文件类型:', options.sourceFile.type);
    } else {
      console.log('原始文件: URL 加载，无 File 对象');
    }

    return {
      models,
      sourceFile: options.sourceFile,
      info: {
        type: fileType,
        name: modelName,
        count: models.length,
        isGaussian: false
      }
    };
  }

  /**
   * 处理高斯模型
   */
  private async processGaussianModel(
    modelData: GaussianDataSource,
    fileType: ModelType,
    modelName: string,
    options: UnifiedLoadOptions
  ): Promise<LoadResult> {
    // 对于高斯模型，我们需要特殊处理
    // 这里需要将 GaussianDataSource 转换为 GaussianModel
    // 暂时抛出错误，因为需要更复杂的处理
    throw new Error('高斯模型处理需要特殊实现，请使用 loadGaussianModel 方法');
  }

  /**
   * 加载高斯模型
   * 简化版本 - 直接使用 Core 的 GaussianLoader
   */
  private async loadGaussianModel(
    input: File | string,
    options: UnifiedLoadOptions
  ): Promise<LoadResult> {
    const filename = input instanceof File ? input.name : input;
    const modelName = options.name || filename.split('/').pop()?.split('.')[0] || 'gaussian_model';

    console.log('=== 进入 loadGaussianModel ===');
    console.log('文件名:', filename);

    let specificFormat = 'ply'; // 默认为 ply
    const lowerName = filename.toLowerCase();

    if (lowerName.endsWith('.compressed.ply')) specificFormat = 'compressed.ply';
    else if (lowerName.endsWith('.sog')) specificFormat = 'sog';
    else if (lowerName.endsWith('.ksplat')) specificFormat = 'ksplat';
    else if (lowerName.endsWith('.splat')) specificFormat = 'splat';
    else if (lowerName.endsWith('.spz')) specificFormat = 'spz';
    else if (lowerName.endsWith('.onnx')) specificFormat = 'onnx';

    console.log('传递给加载器的具体格式:', specificFormat);

    // 保存原始 File 对象
    const sourceFile = options.sourceFile || (input instanceof File ? input : undefined);

    // 直接使用 GaussianLoader，它已经处理了 File/URL/Blob 的转换
    const gaussianModel = await this.gaussianLoader.createFromFile(
      this.renderer,
      input instanceof File ? URL.createObjectURL(input) : input,
      {
        camMat: options.cameraMatrix || new Float32Array(16),
        projMat: options.projectionMatrix || new Float32Array(16),
        ...options.gaussianOptions
      },
      options.gaussianOptions,
      specificFormat as any
    );

    // 清理 blob URL
    if (input instanceof File) {
      URL.revokeObjectURL(URL.createObjectURL(input));
    }

    // 添加到场景
    this.scene.add(gaussianModel);

    // 创建高斯渲染器
    const gaussianRenderer = new GaussianThreeJSRenderer(
      this.renderer,
      this.scene,
      [gaussianModel]
    );
    await gaussianRenderer.init();
    this.scene.add(gaussianRenderer);

    // 打印高斯模型信息
    console.log('=== 高斯模型加载完成 ===');
    console.log('模型名称:', modelName);
    console.log('Object3D UUID:', gaussianModel.uuid);
    console.log('Object3D 类型:', gaussianModel.constructor.name);
    if (sourceFile) {
      console.log('原始文件路径:', sourceFile.name);
      console.log('文件大小:', sourceFile.size, 'bytes');
      console.log('文件类型:', sourceFile.type);
    } else {
      console.log('原始文件: URL 加载，无 File 对象');
    }
    console.log('高斯渲染器 UUID:', gaussianRenderer.uuid);

    return {
      models: [gaussianModel],
      gaussianRenderer,
      sourceFile: sourceFile,
      info: {
        type: 'gaussian',
        name: modelName,
        count: 1,
        isGaussian: true
      }
    };
  }

  private async readFileHeader(file: File, maxBytes = 4096): Promise<string> {
    const slice = file.slice(0, maxBytes);
    const arrayBuffer = await slice.arrayBuffer();
    return new TextDecoder('utf-8').decode(arrayBuffer || new ArrayBuffer(0));
  }

  private async is3dgsPly(file: File): Promise<boolean> {
    try {
      const header = (await this.readFileHeader(file)).toLowerCase();
      if (!header.startsWith('ply')) {
        console.log('不是 PLY 文件:', file.name);
        return false;
      }

      const requiredKeywords = [
        'property float opacity',
        'property float scale_0',
        'property float scale_1',
        'property float scale_2',
        'property float rot_0',
        'property float rot_1',
        'property float rot_2',
        'property float rot_3'
      ];

      const hasAllBaseProps = requiredKeywords.every(keyword => header.includes(keyword));
      const hasShCoefficients = /property\s+float\s+sh_\d+/.test(header);
      console.log(`PLY 文件 ${file.name} 3DGS 检测结果: 基础属性=${hasAllBaseProps}, SH 系数=${hasShCoefficients}`);

      return hasAllBaseProps;
    } catch (error) {
      console.warn('读取 PLY 头信息失败，按非 3DGS 处理:', file.name, error);
      return false;
    }
  }

  /**
   * 通过读取 URL 的文件头判断是否为 3DGS 高斯模型
   * 只下载前 4KB 数据，检查是否包含高斯特有的属性字段
   */
  private async isGaussianPlyUrl(modelurl: string): Promise<boolean> {
    try {
      // 1. 发起 Range 请求，只获取前 4KB 数据 (Header 通常在最前面)
      const separator = modelurl.includes('?') ? '&' : '?';
      const url = `${modelurl}${separator}temp=${new Date().getTime()}`;
      const response = await fetch(url, {
        method: 'GET',
        mode: 'cors', // 必须支持跨域
        headers: {
          'Range': 'bytes=0-4095'
        }
      });

      // 如果服务器不支持 Range (返回 200) 或者请求成功 (返回 206)，都尝试读取文本
      if (!response.ok && response.status !== 206) {
        console.warn(`[UnifiedModelLoader] Range 请求未按预期返回 206，状态码: ${response.status}。尝试继续解析...`);
        // 注意：如果文件巨大且不支持 Range，这里可能会下载较多数据，但通常浏览器流式处理能读取头部
      }

      // 2. 读取头部文本并转小写
      const text = await response.text();
      // 截取 4096 字符以防服务器返回了整个大文件
      const header = text.slice(0, 4096).toLowerCase();

      // 3. 复用 is3dgsPly 的判断逻辑
      if (!header.startsWith('ply')) {
        console.log(`[UnifiedModelLoader] URL 资源不是 PLY 格式: ${url}`);
        return false;
      }

      // 核心特征字段（与 is3dgsPly(file: File) 一致）
      const requiredKeywords = [
        'property float opacity',
        'property float scale_0',
        'property float scale_1',
        'property float scale_2',
        'property float rot_0',
        'property float rot_1',
        'property float rot_2',
        'property float rot_3'
      ];

      // 4. 检查是否包含所有核心字段
      const hasAllBaseProps = requiredKeywords.every(keyword => header.includes(keyword));

      // 额外的 SH 系数检测 (仅用于日志)
      const hasShCoefficients = /property\s+float\s+sh_\d+/.test(header) || /property\s+float\s+f_dc_0/.test(header);

      console.log(`[UnifiedModelLoader] 线上 PLY 检测结果: 基础属性=${hasAllBaseProps}, SH系数=${hasShCoefficients}, URL=${url}`);

      return hasAllBaseProps;

    } catch (error) {
      console.warn('[UnifiedModelLoader] 无法检测线上 PLY 类型 (可能是跨域或网络问题)，默认按 False 处理:', error);
      // 和本地文件失败一致，返回 false
      return false;
    }
  }

  /**
   * 批量加载模型
   */
  public async loadModels(
    inputs: (File | string)[],
    options: UnifiedLoadOptions = {}
  ): Promise<LoadResult[]> {
    const results: LoadResult[] = [];

    for (let i = 0; i < inputs.length; i++) {
      const input = inputs[i];
      const currentOptions = {
        ...options,
        onProgress: (progress: number) => {
          if (options.onProgress) {
            // 计算整体进度
            const overallProgress = (i + progress) / inputs.length;
            options.onProgress(overallProgress);
          }
        }
      };

      try {
        const result = await this.loadModel(input, currentOptions);
        results.push(result);
      } catch (error) {
        console.error(`加载模型失败: ${input}`, error);
        if (options.onError) {
          options.onError(error as Error);
        }
        // 继续加载其他模型
      }
    }

    return results;
  }

  /**
   * 加载 FBX 模型
   */
  private async loadFBXModel(
    input: File | string,
    options: UnifiedLoadOptions
  ): Promise<LoadResult> {
    const filename = input instanceof File ? input.name : input;
    const modelName = options.name || filename.split('/').pop()?.split('.')[0] || 'fbx_model';

    console.log('=== 进入 loadFBXModel ===');
    console.log('文件名:', filename);

    try {
      let modelEntry: any;

      if (input instanceof File) {
        // 从文件加载
        modelEntry = await this.fbxLoader.loadFromFile(input, options.fbxOptions);
      } else {
        // 从 URL 加载
        modelEntry = await this.fbxLoader.loadFromURL(input, options.fbxOptions);
      }

      // 获取 FBX 模型包装器
      const fbxWrapper = modelEntry.pointCloud as any; // FBXModelWrapper
      const object3D = fbxWrapper.object3D;

      // 添加到场景
      this.scene.add(object3D);

      // 打印 FBX 模型信息
      console.log('=== FBX 模型加载完成 ===');
      console.log('模型名称:', modelName);
      console.log('Object3D UUID:', object3D.uuid);
      console.log('Object3D 类型:', object3D.constructor.name);
      console.log('动画数量:', fbxWrapper.clips.length);
      console.log('顶点数量:', modelEntry.pointCount);

      return {
        models: [object3D],
        sourceFile: input instanceof File ? input : undefined,
        info: {
          type: 'fbx',
          name: modelName,
          count: 1,
          isGaussian: false
        }
      };

    } catch (error) {
      console.error('FBX 模型加载失败:', error);
      throw error;
    }
  }

  /**
   * 清理资源
   */
  public dispose(): void {
    // UniversalLoader 会自动管理资源，无需手动清理
  }
}

/**
 * 这是提供给 Editor 使用的主要接口
 */
export async function loadUnifiedModel(
  renderer: THREE.WebGPURenderer,
  scene: THREE.Scene,
  input: File | string | (File | string)[],
  options: UnifiedLoadOptions = {}
): Promise<LoadResult | LoadResult[]> {
  const loader = new UnifiedModelLoader(renderer, scene);

  try {
    if (Array.isArray(input)) {
      return await loader.loadModels(input, options);
    } else {
      return await loader.loadModel(input, options);
    }
  } finally {
    loader.dispose();
  }
}