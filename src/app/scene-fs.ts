/**
 * SceneFS - 3D Gaussian Splatting Scene File System Manager
 * 
 * Handles directory-based scene loading/saving with scene.json manifests.
 * Uses File System Access API for local folder authorization and file access.
 */

import type { App } from './app';
import type { ModelType } from '../models/model-entry';

// File System Access API types
declare global {
  interface Window {
    showDirectoryPicker(options?: {
      mode?: 'read' | 'readwrite';
    }): Promise<FileSystemDirectoryHandle>;
  }

  interface FileSystemDirectoryHandle {
    queryPermission(descriptor?: { mode?: 'read' | 'readwrite' }): Promise<PermissionState>;
    requestPermission(descriptor?: { mode?: 'read' | 'readwrite' }): Promise<PermissionState>;
  }
}

// Scene manifest schema
export interface SceneManifest {
  version: 1;
  meta?: {
    app?: string;
    createdAt?: string;
    unit?: 'meter' | 'centimeter' | string;
  };
  env?: {
    bgColor?: [number, number, number, number];   // RGBA 0..1
    gaussianScale?: number;
    camera?: {
      position: [number, number, number];
      target: [number, number, number];
      up: [number, number, number];
      fov: number;
    } | null;
  };
  assets: AssetEntry[];
}

export type AssetType = ModelType | 'url';

const MODEL_TYPE_VALUES = [
  'ply',
  'spz',
  'ksplat',
  'splat',
  'sog',
  'compressed.ply',
  'onnx',
  'fbx'
] as const satisfies readonly ModelType[];

const isModelType = (value: string): value is ModelType => {
  return (MODEL_TYPE_VALUES as readonly string[]).includes(value);
};

export interface AssetEntry {
  name: string;
  type: AssetType;
  path: string;                  // Root-relative path like "models/a/bunny.ply"
  dynamic?: boolean;             // ONNX: true = per-frame updates
  transform?: {
    position?: [number, number, number];
    rotationEulerRad?: [number, number, number];   // Radians
    scale?: [number, number, number];              // Uniform stored as 3D
  };
  extras?: {
    urlFallback?: string;        // Backup URL if local file missing
  };
}

/**
 * Scene File System Manager
 * 
 * Manages directory permissions, scene manifest reading/writing,
 * and integration with the main App instance for loading/saving scenes.
 */
export class SceneFS {
  private rootHandle: FileSystemDirectoryHandle | null = null;
  private permissions: 'read' | 'readwrite' | null = null;
  
  constructor() {}

  /**
   * Select root directory for read-only access
   */
  async pickFolderRead(): Promise<void> {
    try {
      const handle = await window.showDirectoryPicker({
        mode: 'read'
      });
      
      await this.ensurePermission(handle, 'read');
      this.rootHandle = handle;
      this.permissions = 'read';
      
      console.log('SceneFS: Root folder selected (read-only)', handle.name);
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        throw new Error('Folder selection cancelled by user');
      }
      throw new Error(`Failed to select read folder: ${(error as Error).message}`);
    }
  }

  async setRootHandle(handle: FileSystemDirectoryHandle,permissions: 'read' | 'readwrite') {
    try {
      await this.ensurePermission(handle, permissions);
      this.rootHandle = handle;
      this.permissions = permissions;
      
      console.log('SceneFS: Root folder selected (read-only)', handle.name);
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        throw new Error('Folder selection cancelled by user');
      }
      throw new Error(`Failed to select read folder: ${(error as Error).message}`);
    }
  }

  /**
   * Select root directory for read-write access
   */
  async pickFolderWrite(): Promise<void> {
    try {
      const handle = await window.showDirectoryPicker({
        mode: 'readwrite'
      });
      
      await this.ensurePermission(handle, 'readwrite');
      this.rootHandle = handle;
      this.permissions = 'readwrite';
      
      console.log('SceneFS: Root folder selected (read-write)', handle.name);
    } catch (error) {
      if ((error as Error).name === 'AbortError') {
        throw new Error('Folder selection cancelled by user');
      }
      throw new Error(`Failed to select write folder: ${(error as Error).message}`);
    }
  }

  /**
   * Load scene from root directory or provided JSON data
   */
  async loadScene(app: App, options?: { sceneData?: any }): Promise<any> {
    const sceneData = options?.sceneData;

    try {
      if (sceneData !== undefined) {
        console.log('SceneFS: Loading scene from provided data...');
        await this.loadSceneDataIntoApp(app, sceneData);
        console.log('SceneFS: Scene (provided data) loaded successfully');
        return sceneData;
      }

      if (!this.rootHandle) {
        throw new Error('No root directory selected. Use pickFolderRead() first.');
      }

      console.log('SceneFS: Loading scene from folder...');

      // 先读取原始 scene.json
      const raw = await this.readJSON('scene.json').catch(() => null);
      if (raw && Array.isArray((raw as any).scenes)) {
        console.log('SceneFS: Detected unified scenes schema; loading per view');
        await this.loadScenesArrayIntoApp(app, raw as any);
        console.log('SceneFS: Scene (scenes[]) loaded successfully');
        return raw;
      }

      if (raw) {
        const normalizedManifest = this.normalizeSceneManifest(raw);
        if (normalizedManifest) {
          await this.loadManifestIntoApp(app, normalizedManifest);
          console.log('SceneFS: Scene loaded successfully');
          return raw;
        }
      }

      // 退回到标准 assets 清单流程
      const manifest = await this.findSceneManifestAtRoot();
      if (!manifest) {
        throw new Error('No scene.json found in root directory');
      }

      console.log(`SceneFS: Found scene manifest with ${manifest.assets.length} assets`);

      // Load manifest into app
      await this.loadManifestIntoApp(app, manifest);

      console.log('SceneFS: Scene loaded successfully');
      return manifest;
    } catch (error) {
      console.error('SceneFS: Failed to load scene:', error);
      throw error;
    }
  }

  /**
   * 从 scenes[]（统一场景格式）加载到双视窗
   * 修改：支持加载 url 类型的模型，typeTag=url 时，直接加载 url 资源
   */
  private async loadScenesArrayIntoApp(app: any, raw: { scenes: Array<{ models: Array<any>; keyframes?: any[] }>, meta?: any, env?: any }): Promise<void> {
    const sceneCount = raw.scenes.length;

    const viewIdForIndex = (index: number): 'left' | 'right' => (index === 0 ? 'left' : 'right');

    // 先应用环境（若 app 提供）
    if (raw.env) {
      if (raw.env.gaussianScale !== undefined && typeof app.setGaussianScale === 'function') {
        app.setGaussianScale(raw.env.gaussianScale);
      }
      if (raw.env.bgColor && typeof app.setBackgroundColor === 'function') {
        app.setBackgroundColor(raw.env.bgColor);
      }
    }

    const fileCache = new Map<string, File>();

    for (let i = 0; i < sceneCount; i++) {
      const viewId = viewIdForIndex(i);
      const sceneEntry = raw.scenes[i] || {};
      const models = Array.isArray(sceneEntry?.models) ? sceneEntry.models : [];

      const sceneView = typeof app.getSceneView === 'function' ? app.getSceneView(viewId) : null;
      const animationController = typeof app.getAnimationControllerForView === 'function' ? app.getAnimationControllerForView(viewId) : null;

      if (sceneView && typeof sceneView.clearScene === 'function') {
        sceneView.clearScene();
      }
      if (sceneView && typeof sceneView.clearSelection === 'function') {
        sceneView.clearSelection();
      }
      if (animationController && typeof animationController.replaceGlobalKeyframes === 'function') {
        animationController.replaceGlobalKeyframes([]);
      }

      for (const m of models) {
        if (!m) continue;
        const typeTag: string = m.typeTag || 'fileModel';

        if (typeTag === 'fileModel') {
          const source = this.resolveModelSource(m);
          if (!source) {
            console.warn('SceneFS: fileModel 缺少可识别的资源路径，跳过', m);
            continue;
          }

          const modelType = typeof m?.type === 'string' ? m.type.toLowerCase() : '';
          const loadAsUrl = modelType === 'url' || this.isHttpUrl(source);

          if (loadAsUrl) {
            try {
              await this.loadModelFromUrl(app, sceneView, viewId, source, m);
            } catch (error) {
              console.warn('SceneFS: 通过 URL 加载模型失败:', error);
              continue;
            }
          } else {
            let file = fileCache.get(source);
            if (!file) {
              try {
                file = await this.fileFromRelativePath(source);
                fileCache.set(source, file);
              } catch (e) {
                console.warn(`SceneFS: 无法加载模型 ${source}`, e);
                continue;
              }
            }

            if (typeof app.loadSerializedFileModel === 'function') {
              await app.loadSerializedFileModel(viewId, file, m);
            } else {
              const fallbackType: 'onnx' | 'ply' = (m?.type === 'onnx' || m?.type === 'ply') ? m.type : 'ply';
              const modelName = m?.name ?? this.extractFileNameFromPath(source);
              if (fallbackType === 'onnx') {
                if (typeof app.loadONNXModelToView === 'function') {
                  await app.loadONNXModelToView(viewId, file, modelName, !(m?.dynamic));
                } else if (typeof app.loadONNXModel === 'function') {
                  await app.loadONNXModel(file, modelName, !(m?.dynamic));
                }
              } else {
                if (typeof app.loadPLYToView === 'function') {
                  await app.loadPLYToView(viewId, file);
                } else if (typeof app.loadPLY === 'function') {
                  await app.loadPLY(file);
                }
              }

              const trsFallback = Array.isArray(m?.trs) ? m.trs : undefined;
              if (trsFallback) {
                const mgr = (typeof app.getModelManagerForView === 'function') ? app.getModelManagerForView(viewId) : app.getModelManager?.();
                if (mgr) {
                  const pos = Array.isArray(trsFallback[0]) ? trsFallback[0] : undefined;
                  const rot = Array.isArray(trsFallback[1]) ? trsFallback[1] : undefined;
                  const scl = Array.isArray(trsFallback[2]) ? trsFallback[2] : undefined;
                  if (pos && typeof mgr.setModelPosition === 'function') mgr.setModelPosition(modelName, pos[0]||0, pos[1]||0, pos[2]||0);
                  if (rot && typeof mgr.setModelRotation === 'function') mgr.setModelRotation(modelName, rot[0]||0, rot[1]||0, rot[2]||0);
                  if (scl && typeof mgr.setModelScale === 'function') {
                    if (Array.isArray(scl)) mgr.setModelScale(modelName, scl);
                    else mgr.setModelScale(modelName, scl as any);
                  }
                }
              }
            }
          }

          if (sceneView && m.gaussianParams && typeof sceneView.applyGaussianParams === 'function') {
            const targetId = m.id || m.name;
            sceneView.applyGaussianParams(targetId, m.gaussianParams);
          }
          if (sceneView && Array.isArray(m.trs) && typeof sceneView.applyTRSToObject === 'function') {
            sceneView.applyTRSToObject(m.id || m.name, m.trs);
          }
        }else if(typeTag==='url'){ // typeTag=url 时，直接加载 url 资源
          try {
            const sourceurl = this.resolveModelSource(m);
            if (!sourceurl) {
              console.warn('SceneFS: url 缺少可识别的资源路径，跳过', m);
              continue;
            }
            await this.loadModelFromUrl(app, sceneView, viewId, sourceurl, m);
          } catch (error) {
            console.warn('SceneFS: 通过 URL 加载模型失败:', error);
            continue;
          }
        } else if (typeTag === 'recordingCamera') {
          if (sceneView && typeof sceneView.restoreRecordingCameraFromSerialized === 'function') {
            sceneView.restoreRecordingCameraFromSerialized(m.id, m.name, Array.isArray(m.trs) ? m.trs : undefined, m.params);
          } else {
            console.warn('SceneFS: 当前视窗不支持恢复录制相机对象', m);
          }
        } else {
          if (sceneView && typeof sceneView.restorePrimitiveFromSerialized === 'function') {
            sceneView.restorePrimitiveFromSerialized(typeTag, m.id, m.name, m.params);
            if (Array.isArray(m.trs) && typeof sceneView.applyTRSToObject === 'function') {
              sceneView.applyTRSToObject(m.id, m.trs);
            }
          } else {
            console.warn('SceneFS: 未知或暂不支持的对象类型标签', typeTag, m);
          }
        }
      }

      if (Array.isArray(sceneEntry.keyframes) && animationController && typeof animationController.loadKeyframesFromSerialized === 'function') {
        animationController.loadKeyframesFromSerialized(sceneEntry.keyframes);
      }
    }
  }

  private async loadSceneDataIntoApp(app: App, data: any): Promise<void> {
    if (!data || typeof data !== 'object') {
      throw new Error('SceneFS: Provided scene data is empty or invalid');
    }

    if (Array.isArray((data as any).scenes)) {
      await this.loadScenesArrayIntoApp(app, data as any);
      return;
    }

    const manifest = this.normalizeSceneManifest(data);
    if (!manifest) {
      throw new Error('SceneFS: Unsupported scene data format. Expect scenes[] or assets[].');
    }

    await this.loadManifestIntoApp(app, manifest);
  }

  private async loadModelFromUrl(app: App, sceneView: any, viewId: 'left' | 'right', source: string, meta: any): Promise<void> {
    const forcedId = meta?.id;
    const displayName = meta?.name;
    const url = source;

    if (sceneView && typeof sceneView.loadModel === 'function') {
      const options: Record<string, any> = {};
      if (forcedId) options.forcedId = forcedId;
      if (displayName) options.displayName = displayName;
      await sceneView.loadModel(url, options);
      return;
    }

    const lower = url.toLowerCase();
    if (lower.endsWith('.onnx')) {
      if (typeof app.loadONNXModel === 'function') {
        await app.loadONNXModel(url, displayName ?? undefined, !(meta?.dynamic));
        return;
      }
    } else if (lower.endsWith('.ply') || lower.endsWith('.glb') || lower.endsWith('.gltf')) {
      if (typeof app.loadPLY === 'function') {
        await app.loadPLY(url);
        return;
      }
      if (typeof (app as any).loadSample === 'function') {
        await (app as any).loadSample(url);
        return;
      }
    } else if (typeof (app as any).loadSample === 'function') {
      await (app as any).loadSample(url);
      return;
    }

    throw new Error(`SceneFS: 无法通过 URL 加载模型，缺少匹配的加载接口 (${url})`);
  }

  private resolveModelSource(model: any): string | null {
    if (!model || typeof model !== 'object') return null;

    const candidates: Array<unknown> = [
      model.assetName,
      model.path,
      model.url,
      model.source,
      model?.extras?.urlFallback,
      model.name
    ];

    for (const candidate of candidates) {
      if (typeof candidate === 'string' && candidate.trim().length > 0) {
        return candidate.trim();
      }
    }

    return null;
  }

  private normalizeSceneManifest(raw: any): SceneManifest | null {
    if (!raw || typeof raw !== 'object') {
      return null;
    }

    if (Array.isArray((raw as any).scenes)) {
      const scenes = (raw as any).scenes as Array<{ models: Array<any> }>;
      const assets: AssetEntry[] = [];

      for (const scene of scenes) {
        const models = Array.isArray(scene?.models) ? scene.models : [];
        for (const m of models) {
          const source = this.resolveModelSource(m);
          if (!source) {
            console.warn('SceneFS: 无法从 scenes[] 中推断模型路径，跳过', m);
            continue;
          }

          const typeValue = typeof m?.type === 'string' ? m.type.toLowerCase() : '';
          const type: AssetType = isModelType(typeValue)
            ? typeValue
            : typeValue === 'url'
              ? 'url'
              : 'ply';
          const nameCandidate = typeof m?.name === 'string' && m.name.length > 0
            ? m.name
            : this.extractFileNameFromPath(source);

          const entry: AssetEntry = {
            name: nameCandidate,
            type,
            path: source
          };

          if (m?.dynamic !== undefined) {
            entry.dynamic = !!m.dynamic;
          }

          const trs = Array.isArray(m?.trs) ? m.trs : undefined;
          if (trs) {
            entry.transform = {
              position: Array.isArray(trs[0]) ? [
                Number(trs[0][0] ?? 0),
                Number(trs[0][1] ?? 0),
                Number(trs[0][2] ?? 0)
              ] as [number, number, number] : undefined,
              rotationEulerRad: Array.isArray(trs[1]) ? [
                Number(trs[1][0] ?? 0),
                Number(trs[1][1] ?? 0),
                Number(trs[1][2] ?? 0)
              ] as [number, number, number] : undefined,
              scale: Array.isArray(trs[2]) ? [
                Number(trs[2][0] ?? 1),
                Number(trs[2][1] ?? 1),
                Number(trs[2][2] ?? 1)
              ] as [number, number, number] : undefined
            };
          }

          if (m?.extras) {
            entry.extras = m.extras;
          } else if (this.isHttpUrl(source)) {
            entry.extras = { urlFallback: source };
          }

          assets.push(entry);
        }
      }

      if (assets.length === 0) {
        return null;
      }

      return {
        version: 1,
        meta: raw.meta ?? { app: 'VisionaryEditor', createdAt: new Date().toISOString() },
        env: raw.env ?? {},
        assets
      };
    }

    if (Array.isArray((raw as any).assets)) {
      const normalizedAssets: AssetEntry[] = [];
      for (const asset of raw.assets) {
        if (!asset || typeof asset !== 'object') continue;
        const source = this.resolveModelSource(asset);
        if (!source) {
          console.warn('SceneFS: 资产缺少路径/URL，跳过', asset);
          continue;
        }

        const typeValue = typeof asset.type === 'string' ? asset.type.toLowerCase() : '';
        const type: AssetType = isModelType(typeValue)
          ? typeValue
          : typeValue === 'url'
            ? 'url'
            : 'ply';
        const nameCandidate = typeof asset.name === 'string' && asset.name.length > 0
          ? asset.name
          : this.extractFileNameFromPath(source);

        const entry: AssetEntry = {
          name: nameCandidate,
          type,
          path: source
        };

        if (asset.dynamic !== undefined) {
          entry.dynamic = !!asset.dynamic;
        }

        if (asset.transform) {
          entry.transform = asset.transform;
        }

        if (asset.extras) {
          entry.extras = asset.extras;
        } else if (this.isHttpUrl(source)) {
          entry.extras = { urlFallback: source };
        }

        normalizedAssets.push(entry);
      }

      if (normalizedAssets.length === 0) {
        return null;
      }

      return {
        version: 1,
        meta: raw.meta ?? { app: 'VisionaryEditor', createdAt: new Date().toISOString() },
        env: raw.env ?? {},
        assets: normalizedAssets
      };
    }

    return null;
  }

  private isHttpUrl(value: string): boolean {
    return typeof value === 'string' && /^https?:\/\//i.test(value);
  }

  private async fetchFileFromUrl(url: string): Promise<File> {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
    }
    const blob = await response.blob();
    const fileName = this.extractFileNameFromPath(url);
    const mime = blob.type && blob.type.length > 0 ? blob.type : this.guessMimeType(fileName);
    return new File([blob], fileName, { type: mime });
  }

  private extractFileNameFromPath(path: string): string {
    if (!path) return 'model';

    try {
      const parsed = new URL(path);
      const pathname = parsed.pathname;
      const lastSegment = pathname.split('/').filter(Boolean).pop();
      if (lastSegment) {
        return lastSegment.split('?')[0].split('#')[0];
      }
    } catch {
      // Not a valid URL, fall back to simple split
    }

    const segments = path.split('/').filter(Boolean);
    const candidate = segments.pop();
    if (!candidate || candidate.length === 0) {
      return path;
    }
    return candidate.split('?')[0].split('#')[0];
  }

  private guessMimeType(fileName: string): string {
    const ext = fileName.split('.').pop()?.toLowerCase();
    switch (ext) {
      case 'onnx':
        return 'application/octet-stream';
      case 'ply':
        return 'application/octet-stream';
      case 'fbx':
        return 'application/octet-stream';
      case 'json':
        return 'application/json';
      default:
        return 'application/octet-stream';
    }
  }

  private async loadUrlAsset(app: App, url: string, asset: AssetEntry): Promise<void> {
    const lower = url.toLowerCase();

    if (lower.endsWith('.onnx')) {
      if (typeof app.loadONNXModel === 'function') {
        await app.loadONNXModel(url, asset.name, !asset.dynamic);
        return;
      }
    } else if (lower.endsWith('.ply') || lower.endsWith('.glb') || lower.endsWith('.gltf')) {
      if (typeof app.loadPLY === 'function') {
        await app.loadPLY(url);
        return;
      }
      if (typeof (app as any).loadSample === 'function') {
        await (app as any).loadSample(url);
        return;
      }
    } else if (typeof (app as any).loadSample === 'function') {
      await (app as any).loadSample(url);
      return;
    }

    throw new Error(`SceneFS: 无法通过 URL 加载资产 ${asset.name} (${url})`);
  }

  /**
   * Save current scene to root directory
   */
  async saveToFolder(app: App): Promise<void> {
    // Use the new smart save method that handles file conversion
    return this.saveSceneToFolderSmart(app);
  }

  /**
   * Check if permission is available and request if needed
   */
  public async ensurePermission(handle: FileSystemDirectoryHandle, mode: 'read' | 'readwrite'): Promise<void> {
    const permission = await handle.queryPermission({ mode });
    
    if (permission === 'granted') {
      return;
    }
    
    const requestResult = await handle.requestPermission({ mode });
    if (requestResult !== 'granted') {
      throw new Error(`${mode} permission denied for directory`);
    }
  }

  /**
   * Get file from relative path by navigating directory structure
   */
  private async fileFromRelativePath(relativePath: string): Promise<File> {
    if (this.isHttpUrl(relativePath)) {
      return this.fetchFileFromUrl(relativePath);
    }

    if (!this.rootHandle) {
      throw new Error('No root directory handle');
    }

    const pathParts = relativePath.split('/').filter(p => p.length > 0);
    let currentHandle: FileSystemDirectoryHandle = this.rootHandle;

    // Navigate to parent directory
    for (let i = 0; i < pathParts.length - 1; i++) {
      const part = pathParts[i];
      try {
        currentHandle = await currentHandle.getDirectoryHandle(part);
      } catch (error) {
        throw new Error(`Directory not found: ${pathParts.slice(0, i + 1).join('/')}`);
      }
    }

    // Get the file
    const fileName = pathParts[pathParts.length - 1];
    try {
      const fileHandle = await currentHandle.getFileHandle(fileName);
      return await fileHandle.getFile();
    } catch (error) {
      throw new Error(`File not found: ${relativePath}`);
    }
  }

  /**
   * Read JSON file from root directory
   */
  private async readJSON(fileName: string): Promise<any> {
    if (!this.rootHandle) {
      throw new Error('No root directory handle');
    }

    try {
      const fileHandle = await this.rootHandle.getFileHandle(fileName);
      const file = await fileHandle.getFile();
      const text = await file.text();
      return JSON.parse(text);
    } catch (error) {
      throw new Error(`Failed to read ${fileName}: ${(error as Error).message}`);
    }
  }

  /**
   * Write JSON file to root directory
   */
  private async writeJSONToRoot(fileName: string, data: any): Promise<void> {
    if (!this.rootHandle) {
      throw new Error('No root directory handle');
    }

    try {
      const fileHandle = await this.rootHandle.getFileHandle(fileName, {
        create: true
      });
      const writable = await fileHandle.createWritable();
      await writable.write(JSON.stringify(data, null, 2));
      await writable.close();
    } catch (error) {
      throw new Error(`Failed to write ${fileName}: ${(error as Error).message}`);
    }
  }

  /**
   * Find scene.json manifest in root directory
   */
  private async findSceneManifestAtRoot(): Promise<SceneManifest | null> {
    try {
      const raw = await this.readJSON('scene.json');
      const manifest = this.normalizeSceneManifest(raw);
      if (manifest) {
        return manifest;
      }
      throw new Error('Invalid scene.json format');
    } catch (error) {
      console.warn('SceneFS: No valid scene.json found or parse failed:', (error as Error).message);
      return null;
    }
  }

  /**
   * Load manifest into app instance
   */
  private async loadManifestIntoApp(app: App, manifest: SceneManifest): Promise<void> {
    console.log('SceneFS: Loading manifest into app...');
    
    // Store model IDs before loading to track new additions
    const modelsBefore = new Set(app.getModels().map(m => m.id));
    
    // First, restore environment settings
    if (manifest.env) {
      if (manifest.env.gaussianScale !== undefined) {
        app.setGaussianScale(manifest.env.gaussianScale);
        console.log(`SceneFS: Set gaussian scale to ${manifest.env.gaussianScale}`);
      }
      
      if (manifest.env.bgColor) {
        app.setBackgroundColor(manifest.env.bgColor);
        console.log(`SceneFS: Set background color to ${manifest.env.bgColor}`);
      }
    }

    // Load each asset
    const loadPromises = manifest.assets.map(async (asset, index) => {
      try {
        console.log(`SceneFS: Loading asset ${index + 1}/${manifest.assets.length}: ${asset.name} (${asset.type})`);
        
        if (asset.type === 'url' || this.isHttpUrl(asset.path)) {
          const url = asset.type === 'url' ? asset.path : (this.isHttpUrl(asset.path) ? asset.path : asset.extras?.urlFallback);
          if (!url) {
            throw new Error(`Invalid URL asset path for ${asset.name}`);
          }
          await this.loadUrlAsset(app, url, asset);
        } else {
          let file: File | null = null;
          let loadFromUrlDirect = false;

          try {
            file = await this.fileFromRelativePath(asset.path);
          } catch (pathError) {
            if (asset.extras?.urlFallback) {
              console.warn(`SceneFS: Local file not found, trying fallback URL: ${asset.extras.urlFallback}`);
              loadFromUrlDirect = true;
              file = null;
            } else {
              throw new Error(`File not found: ${asset.path} (${(pathError as Error).message})`);
            }
          }

          if (asset.type === 'onnx') {
            if (loadFromUrlDirect && asset.extras?.urlFallback) {
              console.log(`SceneFS: Loading ONNX asset from URL: ${asset.extras.urlFallback}`);
              await app.loadONNXModel(asset.extras.urlFallback, asset.name, !asset.dynamic);
            } else {
              console.log(`SceneFS: Loading ONNX asset from File: ${asset.name}`);
              await app.loadONNXModel(file as unknown as any, asset.name, !asset.dynamic);
            }
          } else if (asset.type === 'ply' || asset.type === 'fbx') {
            if (loadFromUrlDirect && asset.extras?.urlFallback) {
              console.log(`SceneFS: Loading ${asset.type.toUpperCase()} asset from URL: ${asset.extras.urlFallback}`);
              if (asset.type === 'fbx' && typeof (app as any).loadFBX === 'function') {
                await (app as any).loadFBX(asset.extras.urlFallback);
              } else {
                await app.loadPLY(asset.extras.urlFallback);
              }
            } else {
              console.log(`SceneFS: Loading ${asset.type.toUpperCase()} asset from File: ${asset.name}`);
              if (asset.type === 'fbx' && typeof (app as any).loadFBX === 'function') {
                await (app as any).loadFBX(file);
              } else {
                await app.loadPLY(file as unknown as any, { debugLogging: true });
              }
            }
          } else {
            console.warn(`SceneFS: Unknown asset type: ${asset.type}`);
            return;
          }
        }

        console.log(`SceneFS: Successfully loaded ${asset.name}`);

      } catch (error) {
        console.error(`SceneFS: Failed to load asset ${asset.name}:`, error);
        // Continue with other assets rather than failing completely
      }
    });

    // Wait for all assets to load
    await Promise.allSettled(loadPromises);
    
    // Find newly loaded models and apply transforms
    await this.applyTransforms(app, manifest.assets, modelsBefore);
  }

  /**
   * Apply transforms to newly loaded models
   */
  private async applyTransforms(app: App, assets: AssetEntry[], modelsBefore: Set<string>): Promise<void> {
    // Wait a bit for models to be fully registered
    await new Promise(resolve => setTimeout(resolve, 100));
    
    const modelsAfter = app.getModels();
    const newModels = modelsAfter.filter(m => !modelsBefore.has(m.id));
    
    console.log(`SceneFS: Applying transforms to ${newModels.length} newly loaded models`);
    
    // Match models to assets by name and apply transforms
    for (const asset of assets) {
      if (!asset.transform) continue;
      
      // Find matching model by name (prefer exact match, fallback to partial)
      let matchingModel = newModels.find(m => m.name === asset.name);
      if (!matchingModel) {
        matchingModel = newModels.find(m => m.name.includes(asset.name) || asset.name.includes(m.name));
      }
      
      if (!matchingModel) {
        console.warn(`SceneFS: Could not find model to apply transform: ${asset.name}`);
        continue;
      }
      
      const modelManager = app.getModelManager();
      const transform = asset.transform;
      
      // Apply position
      if (transform.position) {
        modelManager.setModelPosition(matchingModel.id, ...transform.position);
      }
      
      // Apply rotation (convert from radians)
      if (transform.rotationEulerRad) {
        modelManager.setModelRotation(matchingModel.id, ...transform.rotationEulerRad);
      }
      
      // Apply scale
      if (transform.scale) {
        // Use uniform scale if all components are equal, otherwise use vector scale
        const [sx, sy, sz] = transform.scale;
        if (sx === sy && sy === sz) {
          modelManager.setModelScale(matchingModel.id, sx);
        } else {
          modelManager.setModelScale(matchingModel.id, transform.scale);
        }
      }
      
      console.log(`SceneFS: Applied transforms to ${matchingModel.name}`);
    }
  }

  /**
   * Build scene manifest from current app state
   */
  private async buildManifestFromApp(app: App): Promise<{manifest: SceneManifest, convertibleModels?: Array<{model: any, source: any, suggestedPath: string}>}> {
    const models = app.getModels();
    
    // Get transform tracker if available (will be injected by TransformTracker)
    const transformTracker = (app as any).__transformTracker;
    
    // Build assets array and track what gets included/skipped
    const assets: AssetEntry[] = [];
    const skippedModels: Array<{name: string, reason: string, model?: any, source?: any}> = [];
    const convertibleModels: Array<{model: any, source: any, suggestedPath: string}> = [];
    
    for (const model of models) {
      const source = transformTracker?.getSource(model.id);
      const transform = transformTracker?.getTransform(model.id);
      
      // Only include models with relative path sources
      if (!source) {
        skippedModels.push({
          name: model.name,
          reason: 'No source tracking information available'
        });
        continue;
      }
      
      if (source.kind !== 'relative') {
        // Check if this uploaded file might actually be in the scene directory
        if (source.url === '<file-input>' && this.rootHandle && this.permissions === 'readwrite') {
          const suggestedPath = await this.findFileInSceneDirectory(source.originalName);
          if (suggestedPath) {
            // This uploaded file exists in the scene directory - it's convertible!
            convertibleModels.push({
              model,
              source,
              suggestedPath
            });
            
            // For now, skip it but with a different reason
            skippedModels.push({
              name: model.name,
              reason: 'Uploaded file (found in scene directory - can be converted)',
              model,
              source
            });
            continue;
          }
        }
        
        const reason = source.url === '<file-input>' 
          ? 'Uploaded file (cannot be referenced by relative path)'
          : 'Loaded from URL (not a local file)';
        skippedModels.push({
          name: model.name,
          reason: reason,
          model,
          source
        });
        continue;
      }
      
      // 统一 name 与 path：优先使用 source.originalName 或 source.path 的文件名部分，确保与根目录同名文件一致
      const fileNameFromPath = typeof source.path === 'string' ? source.path.split('/').filter(Boolean).pop() : undefined;
      const unifiedName = (source as any).originalName || fileNameFromPath || model.name;

      const asset: AssetEntry = {
        name: unifiedName,
        type: model.modelType,
        path: unifiedName // 按当前规范，模型文件与 scene.json 同目录
      };
      
      // Add dynamic flag for ONNX models
      if (model.modelType === 'onnx' && model.isDynamic) {
        asset.dynamic = true;
      }
      
      // Add transform if available and not default
      if (transform) {
        const hasNonDefaultTransform = 
          (transform.position && (transform.position[0] !== 0 || transform.position[1] !== 0 || transform.position[2] !== 0)) ||
          (transform.rotationEulerRad && (transform.rotationEulerRad[0] !== 0 || transform.rotationEulerRad[1] !== 0 || transform.rotationEulerRad[2] !== 0)) ||
          (transform.scale && (transform.scale[0] !== 1 || transform.scale[1] !== 1 || transform.scale[2] !== 1));
          
        if (hasNonDefaultTransform) {
          asset.transform = transform;
        }
      }
      
      assets.push(asset);
    }
    
    // Build complete manifest
    const manifest: SceneManifest = {
      version: 1,
      meta: {
        app: 'WebGaussianJS',
        createdAt: new Date().toISOString(),
        unit: 'meter'
      },
      env: {
        bgColor: app.getBackgroundColor(),
        gaussianScale: app.getGaussianScale()
      },
      assets
    };
    
    // Provide informative summary about what was included/excluded
    console.log(`SceneFS: Scene manifest built - ${assets.length} models included, ${skippedModels.length} models skipped`);
    
    if (convertibleModels.length > 0) {
      console.log(`SceneFS: Found ${convertibleModels.length} uploaded files that exist in the scene directory:`);
      convertibleModels.forEach(({model, source, suggestedPath}) => {
        console.log(`  • ${model.name}: ${suggestedPath}`);
      });
    }
    
    if (skippedModels.length > 0) {
      console.log('SceneFS: Skipped models (cannot be included in reproducible scenes):');
      skippedModels.forEach(({name, reason}) => {
        console.log(`  • ${name}: ${reason}`);
      });
      console.log('SceneFS: Note - Only models loaded from local files with relative paths can be saved to scenes');
    }
    
    if (assets.length > 0) {
      console.log('SceneFS: Included models:');
      assets.forEach(asset => {
        console.log(`  • ${asset.name} (${asset.type}): ${asset.path}`);
      });
    }
    
    return { manifest, convertibleModels: convertibleModels.length > 0 ? convertibleModels : undefined };
  }

  /**
   * Find a file in the scene directory by name
   */
  private async findFileInSceneDirectory(filename: string): Promise<string | null> {
    if (!this.rootHandle) return null;
    
    try {
      // First try to find exact match in root
      try {
        const fileHandle = await this.rootHandle.getFileHandle(filename);
        return filename; // Found in root directory
      } catch (e) {
        // File not in root, continue searching subdirectories
      }
      
      // Search recursively in subdirectories
      return await this.searchDirectoryRecursively(this.rootHandle, filename, '');
    } catch (error) {
      console.warn(`SceneFS: Error searching for file ${filename}:`, error);
      return null;
    }
  }
  
  /**
   * Recursively search directories for a file
   */
  private async searchDirectoryRecursively(dirHandle: FileSystemDirectoryHandle, filename: string, currentPath: string): Promise<string | null> {
    try {
      for await (const [name, handle] of dirHandle as any) {
        const fullPath = currentPath ? `${currentPath}/${name}` : name;
        
        if (handle.kind === 'file' && name === filename) {
          return fullPath;
        } else if (handle.kind === 'directory') {
          // Search recursively in subdirectory (limit depth to avoid infinite loops)
          const pathDepth = fullPath.split('/').length;
          if (pathDepth < 10) { // Max 10 levels deep
            const result = await this.searchDirectoryRecursively(handle as FileSystemDirectoryHandle, filename, fullPath);
            if (result) return result;
          }
        }
      }
    } catch (error) {
      // Directory might not be accessible, skip it
      console.warn(`SceneFS: Could not search directory ${currentPath}:`, error);
    }
    
    return null;
  }
  
  /**
   * Convert uploaded file models to relative path models
   */
  async convertUploadedFilesToRelative(app: App, conversions: Array<{modelId: string, relativePath: string}>): Promise<void> {
    const transformTracker = (app as any).__transformTracker;
    if (!transformTracker) throw new Error('Transform tracker not available');
    
    for (const {modelId, relativePath} of conversions) {
      // Update the source in the tracker
      const currentSource = transformTracker.getSource(modelId);
      if (currentSource && currentSource.url === '<file-input>') {
        // Convert from file-input to relative path
        const newSource = {
          kind: 'relative' as const,
          path: relativePath,
          originalName: currentSource.originalName
        };
        
        // Update the tracker
        transformTracker.updateSource(modelId, newSource);
        
        console.log(`SceneFS: Converted model ${modelId} to relative path: ${relativePath}`);
      }
    }
  }
  
  /**
   * Save scene to folder with smart conversion of uploadedfiles
   */
  async saveSceneToFolderSmart(app: App): Promise<void> {
    if (!this.rootHandle || this.permissions !== 'readwrite') {
      throw new Error('No writable folder selected. Use pickFolderWrite() first.');
    }
    
    console.log('SceneFS: Building manifest and checking for convertible files...');
    
    // Build manifest and get convertible models
    const {manifest, convertibleModels} = await this.buildManifestFromApp(app);
    
    // If there are convertible models, ask user what to do
    if (convertibleModels && convertibleModels.length > 0) {
      const shouldConvert = await this.promptUserForConversion(convertibleModels);
      
      if (shouldConvert) {
        // Convert the models
        const conversions = convertibleModels.map(({model, suggestedPath}) => ({
          modelId: model.id,
          relativePath: suggestedPath
        }));
        
        await this.convertUploadedFilesToRelative(app, conversions);
        
        // Rebuild manifest with converted models
        const {manifest: updatedManifest} = await this.buildManifestFromApp(app);
        await this.writeManifest(updatedManifest);
      } else {
        // Save without converting
        await this.writeManifest(manifest);
      }
    } else {
      // No convertible models, save normally
      await this.writeManifest(manifest);
    }
    
    console.log('SceneFS: Scene saved successfully');
  }
  
  /**
   * Prompt user whether to convert uploaded files to relative paths
   */
  private async promptUserForConversion(convertibleModels: Array<{model: any, source: any, suggestedPath: string}>): Promise<boolean> {
    const modelList = convertibleModels.map(({model, suggestedPath}) => `• ${model.name} → ${suggestedPath}`).join('\n');
    
    return confirm(`Found ${convertibleModels.length} uploaded file(s) that exist in the scene directory:\n\n${modelList}\n\nWould you like to convert these to use relative paths so they can be included in the scene?`);
  }
  
  /**
   * Write manifest to scene.json
   */
  private async writeManifest(manifest: SceneManifest): Promise<void> {
    if (!this.rootHandle) throw new Error('No root handle available');
    
    const manifestJson = JSON.stringify(manifest, null, 2);
    const fileHandle = await this.rootHandle.getFileHandle('scene.json', { create: true });
    const writable = await fileHandle.createWritable();
    
    await writable.write(manifestJson);
    await writable.close();
    
    console.log('SceneFS: Wrote scene.json with', manifest.assets.length, 'assets');
  }
  
  /**
   * Get current permissions
   */
  getPermissions(): 'read' | 'readwrite' | null {
    return this.permissions;
  }

  /**
   * Check if folder is selected
   */
  hasFolder(): boolean {
    return this.rootHandle !== null;
  }

  /**
   * Get root folder name
   */
  getFolderName(): string | null {
    return this.rootHandle?.name || null;
  }
}