/**
 * File Loader - Enhanced to support all Gaussian formats
 * Handles file loading operations (drag & drop, browse, URL loading)
 * Now supports: PLY, SPZ, KSplat, SPLAT, SOG, Compressed PLY, ONNX, FBX
 */

import { 
  defaultLoader, 
  isGaussianFormat, 
  detectGaussianFormat, 
  isGaussianDataSource as isGaussianDataSourceFromIO,
  type GaussianDataSource 
} from "../../io";
import { ModelManager } from "./model-manager";
import { ModelEntry } from "../../models/model-entry";
import { FBXLoaderManager } from "./fbx-loader";

export interface LoadProgress {
  stage: string;
  progress: number;
}

export interface LoadingCallbacks {
  onProgress?: (show: boolean, text?: string, pct?: number) => void;
  onError?: (message: string) => void;
}

export class FileLoader {
  private modelManager: ModelManager;
  private callbacks: LoadingCallbacks;

  constructor(modelManager: ModelManager, callbacks: LoadingCallbacks = {}) {
    this.modelManager = modelManager;
    this.callbacks = callbacks;
  }

  /**
   * Load a file from a File object (drag & drop or file picker)
   * Now supports all Gaussian formats via defaultLoader
   */
  async loadFile(file: File, device: GPUDevice): Promise<ModelEntry | null> {
    try {
      this.showProgress(true, "Reading file...", 10);

      // Debug: print first 16 bytes
      file.arrayBuffer().then(buf => {
        console.log("[FileLoader] First 16 bytes:", new Uint8Array(buf).slice(0, 16));
      });

      // Check capacity
      if (this.modelManager.isAtCapacity()) {
        this.showProgress(false);
        this.showError(`Reached model limit (${this.modelManager.getRemainingCapacity()}). Remove models before adding more.`);
        return null;
      }
      
      const fileName = file.name.toLowerCase();
      
      // Check if it's a Gaussian format (PLY, SPZ, KSplat, SPLAT, SOG, etc.)
      if (isGaussianFormat(fileName)) {
        const format = detectGaussianFormat(fileName);
        console.log(`[FileLoader] Loading Gaussian format: ${format}`);
        return await this.loadGaussianFile(file, device);
      } else if (fileName.endsWith('.onnx')) {
        this.showError('ONNX files should be loaded through ONNXManager, not FileLoader');
        return null;
      } else if (fileName.endsWith('.fbx')) {
        return await this.loadFBXFile(file);
      } else {
        this.showProgress(false);
        this.showError(`Unsupported file type: ${fileName}\nSupported formats: ${this.getSupportedExtensions().join(', ')}`);
        return null;
      }
    } catch (e) {
      this.showProgress(false);
      this.showError((e as Error).message);
      return null;
    }
  }

  /**
   * Load a sample file from URL
   * Now supports all Gaussian formats
   */
  async loadSample(filename: string, device: GPUDevice, expectedType?: 'ply' | 'onnx' | 'gaussian' | string): Promise<ModelEntry | null> {
    try {
      // Check capacity
      if (this.modelManager.isAtCapacity()) {
        this.showError(`Reached model limit. Remove models before adding more.`);
        return null;
      }

      console.log('[FileLoader] Loading sample:', filename, expectedType ? `(expected: ${expectedType})` : '');
      
      // Determine file type
      let fileType = expectedType;
      if (!fileType) {
          fileType = this.detectTypeFromFilename(filename);
      }
      
      // 定义所有视为高斯的格式
      const gaussianTypes = ['ply', 'gaussian', 'sog', 'splat', 'ksplat', 'spz', 'compressed.ply'];

      // Handle Gaussian types
      // 确保这里包含了 detectedFileType，并且将其传给 loadGaussianUrl
      if (fileType && gaussianTypes.includes(fileType)) {
          // 这里的 fileType 就是 'sog', 'splat' 等
          return await this.loadGaussianUrl(filename, device, fileType);
      } else if (fileType === 'onnx') {
          this.showError('ONNX files should be loaded through ONNXManager, not FileLoader');
          return null;
      } else {
          this.showError(`Unsupported file type: ${filename}`);
          return null;
      }
    } catch (e) {
      console.error(`[FileLoader] Failed to load sample ${filename}:`, e);
      this.showProgress(false);
      this.showError((e as Error).message);
      return null;
    }
  }

  /**
   * Detect file type from filename
   */
  private detectTypeFromFilename(filename: string): 'gaussian' | 'onnx' | undefined {
    const fileName = filename.toLowerCase();
    
    if (isGaussianFormat(fileName)) return 'gaussian';
    if (fileName.endsWith('.onnx')) return 'onnx';
    
    return undefined;
  }

  /**
   * Load Gaussian format file from File object
   * Supports: PLY, SPZ, KSplat, SPLAT, SOG, Compressed PLY
   */
  private async loadGaussianFile(file: File, device: GPUDevice): Promise<ModelEntry | null> {
    const format = detectGaussianFormat(file.name) || 'unknown';
    console.log(`[FileLoader] Loading ${format.toUpperCase()} file:`, file.name);
    
    // Use defaultLoader to load any Gaussian format
    const data = await defaultLoader.loadFile(file, {
      onProgress: (progress) => {
        this.showProgress(true, progress.stage, progress.progress * 100);
      },
      isGaussian: true
    });

    // Type guard: ensure it's a GaussianDataSource
    if (!this.isGaussianDataSource(data)) {
      throw new Error(`Loaded data is not a valid Gaussian format: ${file.name}`);
    }

    // Convert to ModelEntry
    return await this.createGaussianModel(data, file.name, device);
  }

  /**
   * Load Gaussian format file from URL
   * Supports: PLY, SPZ, KSplat, SPLAT, SOG, Compressed PLY
   */
  private async loadGaussianUrl(filename: string, device: GPUDevice, formatHint?: string): Promise<ModelEntry | null> {
    // Handle blob URLs specially
    if (filename.startsWith('blob:')) {
      console.log('[FileLoader] Detected blob URL, using blob-to-file loading path');
      // 传入 formatHint
      return await this.loadGaussianFromBlob(filename, device, formatHint);
    }

    const format = detectGaussianFormat(filename) || 'unknown';
    console.log(`[FileLoader] Loading ${format.toUpperCase()} from URL:`, filename);

    // Use defaultLoader to load any Gaussian format
    const data = await defaultLoader.loadUrl(filename, {
      onProgress: (progress) => {
        this.showProgress(true, progress.stage, progress.progress * 100);
      }
    });

    // Type guard: ensure it's a GaussianDataSource
    if (!this.isGaussianDataSource(data)) {
      throw new Error(`Loaded data is not a valid Gaussian format: ${filename}`);
    }

    const modelName = filename.split('/').pop() || filename;
    return await this.createGaussianModel(data, modelName, device);
  }

  /**
   * Load Gaussian file from blob URL by converting to File object
   */
  private async loadGaussianFromBlob(blobUrl: string, device: GPUDevice, formatHint?: string): Promise<ModelEntry | null> {
    console.log(`[FileLoader] Converting blob URL to File object. Hint: ${formatHint}`);
    
    try {
      // Fetch the blob content
      const response = await fetch(blobUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch blob: ${response.status} ${response.statusText}`);
      }
      
      const blob = await response.blob();
      
      // 核心逻辑：根据 hint 构造正确的文件名
      // 如果 hint 是 'gaussian' 或者空，回退到 'ply'
      // 如果 hint 是 'sog', 'splat' 等，直接使用
      let extension = 'ply';
      if (formatHint && formatHint !== 'gaussian') {
          extension = formatHint;
          console.log('type:', extension)
      }
      
      const fakeFileName = `scene-model.${extension}`;
      
      // Create a File object with proper name and type
      // 这里的 fakeFileName 会是 "scene-model.sog"，这会让 UniversalLoader 能够识别
      const file = new File([blob], fakeFileName, { type: 'application/octet-stream' });
      
      console.log(`[FileLoader] Created File object '${fakeFileName}' from blob, delegating to loadGaussianFile`);
      
      // Use existing File-based loading
      return await this.loadGaussianFile(file, device);
    } catch (error) {
      console.error('[FileLoader] Failed to load from blob URL:', error);
      throw error;
    }
  }

  /**
   * Create a model from Gaussian data (unified for all formats)
   * This is the key method that converts GaussianDataSource to ModelEntry
   */
  private async createGaussianModel(
    gaussianData: GaussianDataSource, 
    name: string, 
    device: GPUDevice
  ): Promise<ModelEntry> {
    this.showProgress(true, "Creating GPU buffers...", 60);
    
    // Import PointCloud class
    const { PointCloud } = await import("../../point_cloud");
    
    // Create point cloud from GaussianDataSource
    // The PointCloud constructor should accept GaussianDataSource
    const pc = new PointCloud(device, gaussianData);

    if (typeof gaussianData.weightBuffer === 'function') {
      const weightBuf = gaussianData.weightBuffer();
      if (weightBuf && weightBuf.byteLength > 0) {
        const weightArray = new Float32Array(weightBuf);
        pc.setWeightBufferFromArray(device, weightArray, 64);
      }
    }
    
    // Generate unique name if needed
    const uniqueName = this.modelManager.generateUniqueName(name);
    
    // Detect format for metadata
    const format = detectGaussianFormat(name) || 'ply';
    
    // Add to model manager
    const entry = this.modelManager.addModel({
      name: uniqueName,
      visible: true,
      pointCloud: pc,
      pointCount: pc.numPoints,
      isDynamic: false,
      modelType: format, // Store the actual format (ply, spz, ksplat, etc.)
    });

    this.showProgress(true, "Initializing renderer...", 90);
    this.showProgress(false);
    
    console.log(`[FileLoader] Successfully created ${format.toUpperCase()} model:`, uniqueName);
    
    return entry;
  }

  /**
   * Load FBX file
   */
  private async loadFBXFile(file: File): Promise<ModelEntry | null> {
    try {
      const fbxLoader = new FBXLoaderManager(this.modelManager, {
        onProgress: (progress, message) => this.showProgress(true, message, progress),
        onError: (error) => this.showError(error),
        onSuccess: (model) => console.log('[FileLoader] FBX loaded successfully:', model.name)
      });

      return await fbxLoader.loadFromFile(file);
    } catch (error) {
      this.showError(`Failed to load FBX file: ${(error as Error).message}`);
      return null;
    }
  }

  /**
   * Check if file type is supported (updated to include all Gaussian formats)
   */
  isFileTypeSupported(filename: string): boolean {
    const lower = filename.toLowerCase();
    return isGaussianFormat(lower) || lower.endsWith('.onnx') || lower.endsWith('.fbx');
  }

  /**
   * Get supported file extensions (updated to include all Gaussian formats)
   */
  getSupportedExtensions(): string[] {
    return [
      '.ply', '.spz', '.ksplat', '.splat', '.sog', '.compressed.ply', // Gaussian formats
      '.onnx',  // ONNX
      '.fbx'    // FBX
    ];
  }

  /**
   * Get file type from filename (updated)
   */
  getFileType(filename: string): 'gaussian' | 'onnx' | 'fbx' | 'unknown' {
    const lower = filename.toLowerCase();
    if (isGaussianFormat(lower)) return 'gaussian';
    if (lower.endsWith('.onnx')) return 'onnx';
    if (lower.endsWith('.fbx')) return 'fbx';
    return 'unknown';
  }

  /**
   * Get specific Gaussian format
   */
  getGaussianFormat(filename: string): string | null {
    return detectGaussianFormat(filename);
  }

  /**
   * Update loading callbacks
   */
  setCallbacks(callbacks: LoadingCallbacks): void {
    this.callbacks = { ...this.callbacks, ...callbacks };
  }

  /**
   * Show loading progress
   */
  private showProgress(show: boolean, text?: string, pct?: number): void {
    if (this.callbacks.onProgress) {
      this.callbacks.onProgress(show, text, pct);
    }
  }

  /**
   * Show error message
   */
  private showError(message: string): void {
    if (this.callbacks.onError) {
      this.callbacks.onError(message);
    } else {
      console.error('[FileLoader] Error:', message);
    }
  }

  /**
   * Validate file before loading
   */
  validateFile(file: File): { valid: boolean; error?: string } {
    // Check file size (optional limit)
    const maxSize = 1024 * 1024 * 1024; // 1GB limit
    if (file.size > maxSize) {
      return { valid: false, error: 'File too large (max 1GB)' };
    }

    // Check file type
    if (!this.isFileTypeSupported(file.name)) {
      return { 
        valid: false, 
        error: `Unsupported file type. Supported: ${this.getSupportedExtensions().join(', ')}` 
      };
    }

    // Check capacity
    if (this.modelManager.isAtCapacity()) {
      return { 
        valid: false, 
        error: `Model limit reached. Remove models before adding more.` 
      };
    }

    return { valid: true };
  }

  /**
   * Type guard to check if data is GaussianDataSource
   * Uses the helper from io module
   */
  private isGaussianDataSource(data: any): data is GaussianDataSource {
    return isGaussianDataSourceFromIO(data);
  }
}