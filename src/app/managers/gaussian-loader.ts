/**
 * Gaussian Loader Service - Enhanced for multi-format support
 * High-level convenience service that wraps FileLoader and ONNXManager
 * Now supports: PLY, SPZ, KSplat, SPLAT, SOG, Compressed PLY, ONNX
 */

import * as THREE from "three/webgpu";
import { GaussianModel } from "../GaussianModel";
import { ModelEntry } from "../../models/model-entry";
import { FileLoader } from "./file-loader";
import { ONNXManager, ONNXLoadOptions } from "./onnx-manager";
import { isGaussianFormat, detectGaussianFormat, WeightPLYLoader } from "../../io";
import { PointCloud } from "../../point_cloud";

export interface GaussianLoadOptions {
    name?: string;
    onnxOptions?: ONNXLoadOptions;
}

/**
 * Simplified service for creating GaussianModel instances
 * Delegates actual loading to FileLoader/ONNXManager, focuses on GaussianModel creation
 */
export class GaussianLoader {
    constructor(
        private fileLoader: FileLoader,
        private onnxManager: ONNXManager
    ) {}

    /**
     * Create GaussianModel from Gaussian format file (PLY, SPZ, KSplat, SPLAT, SOG, etc.)
     * @param renderer - Three.js WebGPU renderer
     * @param modelPath - Path to Gaussian format file
     * @param options - Optional loading options
     * @returns GaussianModel instance
     */
    async createFromGaussian(
        renderer: THREE.WebGPURenderer,
        modelPath: string,
        options?: GaussianLoadOptions,
        formatHint?: string
    ): Promise<GaussianModel> {
        const device = (renderer.backend as any).device as GPUDevice;
        
        // Detect format
        const format = detectGaussianFormat(modelPath);
        console.log(`[GaussianLoader] Loading ${format?.toUpperCase() || 'GAUSSIAN'} file:`, modelPath);
        
        // Delegate to FileLoader (now supports all Gaussian formats)
        const entry = await this.fileLoader.loadSample(
            modelPath, 
            device, 
            formatHint || 'gaussian'
        );
        if (!entry) {
            throw new Error(`Failed to load Gaussian file: ${modelPath}`);
        }
        
        // Apply custom name if provided
        if (options?.name) {
            entry.name = options.name;
        }
        
        return new GaussianModel(entry);
    }

    /**
     * Create GaussianModel from PLY file
     * @param renderer - Three.js WebGPU renderer
     * @param modelPath - Path to PLY file
     * @param options - Optional loading options
     * @returns GaussianModel instance
     */
    async createFromPLY(
        renderer: THREE.WebGPURenderer,
        modelPath: string,
        options?: GaussianLoadOptions
    ): Promise<GaussianModel> {
        const device = (renderer.backend as any).device as GPUDevice;
        
        // Delegate to FileLoader
        const entry = await this.fileLoader.loadSample(modelPath, device, 'ply');
        if (!entry) {
            throw new Error(`Failed to load PLY file: ${modelPath}`);
        }
        
        // Apply custom name if provided
        if (options?.name) {
            entry.name = options.name;
        }
        
        return new GaussianModel(entry);
    }

    /**
     * Create GaussianModel from SPZ file
     */
    async createFromSPZ(
        renderer: THREE.WebGPURenderer,
        modelPath: string,
        options?: GaussianLoadOptions
    ): Promise<GaussianModel> {
        return this.createFromGaussian(renderer, modelPath, options);
    }

    /**
     * Create GaussianModel from KSplat file
     */
    async createFromKSplat(
        renderer: THREE.WebGPURenderer,
        modelPath: string,
        options?: GaussianLoadOptions
    ): Promise<GaussianModel> {
        return this.createFromGaussian(renderer, modelPath, options);
    }

    /**
     * Create GaussianModel from SPLAT file
     */
    async createFromSplat(
        renderer: THREE.WebGPURenderer,
        modelPath: string,
        options?: GaussianLoadOptions
    ): Promise<GaussianModel> {
        return this.createFromGaussian(renderer, modelPath, options);
    }

    /**
     * Create GaussianModel from SOG file
     */
    async createFromSOG(
        renderer: THREE.WebGPURenderer,
        modelPath: string,
        options?: GaussianLoadOptions
    ): Promise<GaussianModel> {
        return this.createFromGaussian(renderer, modelPath, options);
    }

    /**
     * Create GaussianModel from ONNX file
     * @param renderer - Three.js WebGPU renderer
     * @param modelPath - Path to ONNX file
     * @param cameraMatrix - Initial camera view matrix
     * @param projectionMatrix - Initial projection matrix
     * @param options - Optional loading options
     * @returns GaussianModel instance
     */
    async createFromONNX(
        renderer: THREE.WebGPURenderer,
        modelPath: string,
        cameraMatrix: Float32Array,
        projectionMatrix: Float32Array,
        options?: GaussianLoadOptions
    ): Promise<GaussianModel> {
        const device = (renderer.backend as any).device as GPUDevice;
        
        // Prepare ONNX loading options
        const onnxOptions: ONNXLoadOptions = {
            staticInference: false,  // Default to dynamic mode
            debugLogging: true,
            ...options?.onnxOptions
        };
        
        // Delegate to ONNXManager
        const entry = await this.onnxManager.loadONNXModel(
            device,
            modelPath,
            cameraMatrix,
            projectionMatrix,
            options?.name,
            onnxOptions
        );
        
        return new GaussianModel(entry);
    }

    /**
     * Attach per-point weights from a separate PLY file.
     * The weight PLY must contain weight_0..weight_63 for each vertex.
     */
    async attachWeightsFromPLY(
        renderer: THREE.WebGPURenderer,
        model: GaussianModel,
        weightsPath: string
    ): Promise<void> {
        const device = (renderer.backend as any).device as GPUDevice;
        const loader = new WeightPLYLoader();
        const data = await loader.loadUrl(weightsPath);
        const pc = model.getPointCloud();
        if (!(pc instanceof PointCloud)) {
            throw new Error('attachWeightsFromPLY: model is not a PointCloud');
        }
        if (data.numPoints() !== pc.numPoints) {
            console.warn(
                `[GaussianLoader] Weight PLY point count (${data.numPoints()}) does not match model (${pc.numPoints})`
            );
        }
        pc.setWeightBufferFromArray(device, new Float32Array(data.weightsBuffer()), 64);
    }

    /**
     * Auto-detect file type and create GaussianModel
     * Uses FileLoader's detection logic
     * @param renderer - Three.js WebGPU renderer
     * @param modelPath - Path to model file
     * @param cameraMatrices - Camera matrices (required for ONNX)
     * @param options - Optional loading options
     * @param fileType - Optional explicit file type
     * @returns GaussianModel instance
     */
    async createFromFile(
        renderer: THREE.WebGPURenderer,
        modelPath: string,
        cameraMatrices?: { camMat: Float32Array; projMat: Float32Array },
        options?: GaussianLoadOptions,
        fileType?: string
    ): Promise<GaussianModel> {
        // Use provided fileType or detect from path
        const detectedFileType = fileType || this.fileLoader.getFileType(modelPath);
        // console.log('detectedFileType', detectedFileType);
        // console.log('modelPath', modelPath);
        // console.log('fileType', fileType);

        // fallback just due to not refactored yet
        if (detectedFileType === 'ply') {
            return this.createFromPLY(renderer, modelPath, options);
        }
        if (detectedFileType === 'onnx') {
            if (!cameraMatrices) {
                throw new Error(`ONNX file ${modelPath} requires camera matrices`);
            }
            return this.createFromONNX(
                renderer,
                modelPath,
                cameraMatrices.camMat,
                cameraMatrices.projMat,
                options
            );
        } 
        // 剩下的所有高斯相关格式 (sog, splat, ksplat, spz, compressed.ply, gaussian)
        // 全部交给 createFromGaussian，并传入具体的 detectedFileType
        const gaussianFormats = ['gaussian', 'sog', 'splat', 'ksplat', 'spz', 'compressed.ply'];
        
        if (gaussianFormats.includes(detectedFileType)) {
            return this.createFromGaussian(renderer, modelPath, options, detectedFileType);
        } else {
            throw new Error(`Unsupported file type: ${detectedFileType}`);
        }
    }






    /**
     * Create GaussianModel from existing ModelEntry
     * Useful when you already have a ModelEntry from other sources
     */
    createFromEntry(entry: ModelEntry): GaussianModel {
        return new GaussianModel(entry);
    }

    /**
     * Check if a file format is supported
     */
    isFormatSupported(filename: string): boolean {
        return this.fileLoader.isFileTypeSupported(filename);
    }

    /**
     * Get supported file formats
     */
    getSupportedFormats(): string[] {
        return this.fileLoader.getSupportedExtensions();
    }

    /**
     * Detect the Gaussian format of a file
     */
    detectFormat(filename: string): string | null {
        return this.fileLoader.getGaussianFormat(filename);
    }
}