// IO interfaces and loading contracts

/**
 * Base interface for all loaded data sources
 */
export interface DataSource {
  /** Bounding box of the data */
  bbox(): { min: [number, number, number]; max: [number, number, number] };
  /** Optional scene center */
  center?: [number, number, number];
  /** Optional up vector */
  up?: [number, number, number] | null;
  /** Optional rendering settings */
  kernelSize?: number;
  mipSplatting?: boolean;
  backgroundColor?: [number, number, number];
}

/**
 * Interface for Gaussian point cloud data
 */
export interface GaussianDataSource extends DataSource {
  /** Get the Gaussian data buffer (packed format) */
  gaussianBuffer(): ArrayBuffer;
  /** Get spherical harmonics coefficients buffer */
  shCoefsBuffer(): ArrayBuffer;
  /** Optional extra per-GS PCA buffer (vec3 packed as vec4) */
  extraPcaBuffer?(): ArrayBuffer;
  /** Optional per-point weight buffer (64 channels, float32) */
  weightBuffer?(): ArrayBuffer;
  /** Number of points in the dataset */
  numPoints(): number;
  /** Spherical harmonics degree */
  shDegree(): number;
}

/**
 * Interface for Three.js model data
 */
export interface ThreeJSDataSource extends DataSource {
  /** Get the Three.js Object3D */
  object3D(): any; // 使用 any 避免 THREE 命名空间依赖
  /** Get the model type */
  modelType(): string;
}

/**
 * Progress callback for loading operations
 */
export interface LoadingProgress {
  /** Current progress 0.0 - 1.0 */
  progress: number;
  /** Current stage description */
  stage: string;
  /** Optional detailed message */
  message?: string;
}

/**
 * Options for loading operations
 */
export interface LoadingOptions {
  /** Progress callback */
  onProgress?: (progress: LoadingProgress) => void;
  /** Abort signal */
  signal?: AbortSignal;
  /** Enable debug logging */
  debug?: boolean;
  isGaussian?: boolean;
}

/**
 * Generic interface for file format loaders
 */
export interface ILoader<T extends DataSource = DataSource> {
  /**
   * Load from a file
   */
  loadFile(file: File, options?: LoadingOptions): Promise<T>;
  
  /**
   * Load from a URL
   */
  loadUrl(url: string, options?: LoadingOptions): Promise<T>;
  
  /**
   * Load from raw buffer data
   */
  loadBuffer(buffer: ArrayBuffer, options?: LoadingOptions): Promise<T>;
  
  /**
   * Check if this loader can handle the given file type
   */
  canHandle(filename: string, mimeType?: string): boolean;
  
  /**
   * Get supported file extensions
   */
  getSupportedExtensions(): string[];
}

/**
 * Registry for different file format loaders
 */
export interface LoaderRegistry {
  /**
   * Register a loader for specific file types
   */
  register<T extends DataSource>(loader: ILoader<T>, extensions: string[]): void;
  
  /**
   * Get appropriate loader for a file
   */
  getLoader(filename: string, mimeType?: string): ILoader | null;
  
  /**
   * Get all supported extensions
   */
  getAllSupportedExtensions(): string[];
}

// ============= Export Gaussian Splatting Loaders =============
export { PLYGaussianData } from './GaussianData'

export { PLYLoader } from './ply_loader';
export { SPZLoader } from './spz_loader';
export { KSplatLoader } from './ksplat_loader';
export { SplatLoader } from './splat_loader';
export { SOGLoader } from './sog_loader';
export { CompressedPLYLoader } from './compressed_ply_loader';

// Universal loader that auto-detects format
export { UniversalGaussianLoader } from './universal_gaussian_loader';

// ============= Export Three.js Adapters =============
export { 
  ThreeJSModelData, 
  GLTFLoaderAdapter, 
  OBJLoaderAdapter, 
  FBXLoaderAdapter, 
  STLLoaderAdapter, 
  ThreeJSPLYLoaderAdapter,
  createThreeJSAdapters 
} from './threejs_adapters';

// Weight-only loader
export { WeightPLYLoader, WeightPLYData } from './weight_ply_loader';

// ============= Export Universal Loader (if exists) =============
export { UniversalLoader, createUniversalLoader, defaultLoader } from './universal_loader';

// ============= Supported Format Types =============

/**
 * Enum of supported Gaussian Splatting formats
 */
export enum GaussianFormat {
  /** Standard PLY format */
  PLY = 'ply',
  /** GZIP compressed custom format */
  SPZ = 'spz',
  /** Luma AI optimized format */
  KSPLAT = 'ksplat',
  /** AntiSplat simplified format */
  SPLAT = 'splat',
  /** SuperOrdered Gaussians format */
  SOG = 'sog',
  /** GZIP compressed PLY */
  COMPRESSED_PLY = 'compressed.ply'
}

/**
 * Get all supported Gaussian format extensions
 */
export function getSupportedGaussianFormats(): string[] {
  return ['.ply', '.spz', '.ksplat', '.splat', '.sog', '.compressed.ply'];
}

/**
 * Check if a filename has a supported Gaussian format
 */
export function isGaussianFormat(filename: string): boolean {
  const lower = filename.toLowerCase();
  return getSupportedGaussianFormats().some(ext => lower.endsWith(ext));
}

/**
 * Detect Gaussian format from filename
 */
export function detectGaussianFormat(filename: string): GaussianFormat | null {
  const lower = filename.toLowerCase();
  
  if (lower.endsWith('.compressed.ply')) return GaussianFormat.COMPRESSED_PLY;
  if (lower.endsWith('.ksplat')) return GaussianFormat.KSPLAT;
  if (lower.endsWith('.splat')) return GaussianFormat.SPLAT;
  if (lower.endsWith('.spz')) return GaussianFormat.SPZ;
  if (lower.endsWith('.sog')) return GaussianFormat.SOG;
  if (lower.endsWith('.ply')) return GaussianFormat.PLY;
  
  return null;
}

// ============= Type Guards =============

/**
 * Type guard to check if data source is Gaussian data
 */
export function isGaussianDataSource(data: DataSource): data is GaussianDataSource {
  return (
    'gaussianBuffer' in data &&
    'shCoefsBuffer' in data &&
    'numPoints' in data &&
    'shDegree' in data
  );
}

/**
 * Type guard to check if data source is Three.js model
 */
export function isThreeJSDataSource(data: DataSource): data is ThreeJSDataSource {
  return 'object3D' in data && 'modelType' in data;
}