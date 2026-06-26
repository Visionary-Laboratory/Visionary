import * as THREE from "three/webgpu";
import {GaussianModel} from "./GaussianModel.ts";
import {PerspectiveCamera} from "../camera";
import {GaussianRenderer} from "../renderer";
import {PointCloud, DynamicPointCloud} from "../point_cloud";
import { CameraAdapter } from "../camera/CameraAdapter";
import { FBXModelWrapper } from "../models/fbx-model-wrapper";

export class GaussianThreeJSRenderer extends THREE.Mesh {
    private renderer: GaussianRenderer;
    private gaussianModels: GaussianModel[];
    private pcs: (PointCloud | DynamicPointCloud)[] | null = null;
    
    // Three.js integration
    private threeRenderer: THREE.WebGPURenderer;
    private threeScene: THREE.Scene;
    private device: GPUDevice;
    private canvasFormat: GPUTextureFormat;
    
    // Depth capture from full scene (new architecture)
    private sceneDepthRT: THREE.RenderTarget | null = null;  // Three.js scene renders here for depth
    private sceneDepthTexture: THREE.DepthTexture | null = null;
    private autoDepthMode: boolean = true; // Auto capture depth from full scene
    
    // Legacy occluder support (kept for compatibility)
    private occluderMeshes: THREE.Mesh[] = [];
    private occluderScene: THREE.Scene = new THREE.Scene();

    // Gizmo overlay support
    private gizmoOverlayRT: THREE.RenderTarget | null = null;
    private overlaySampler: GPUSampler | null = null;
    private overlayBindGroupLayout: GPUBindGroupLayout | null = null;
    private overlayPipeline: GPURenderPipeline | null = null;
    private overlayRenderedThisFrame = false;

    // Post-process pass support
    private postProcessEnabled = false;
    private postProcessTarget: GPUTexture | null = null;
    private postProcessTargetView: GPUTextureView | null = null;
    private postProcessTargetWidth = 0;
    private postProcessTargetHeight = 0;
    private postProcessTargetWrittenThisFrame = false;
    private postProcessSampler: GPUSampler | null = null;
    private postProcessBindGroupLayout: GPUBindGroupLayout | null = null;
    private postProcessPipeline: GPURenderPipeline | null = null;
    private postProcessFragmentWGSL: string | null = null;
    private postProcessUsesDepth = false;
    private postProcessUsesExtraTexture = false;
    private postProcessUsesUniform = false;
    private postProcessExtraTexture: GPUTexture | null = null;
    private postProcessExtraSampler: GPUSampler | null = null;
    private postProcessUniformBuffer: GPUBuffer | null = null;
    private dummyPostProcessExtraTexture: GPUTexture | null = null;

    // Weight overlay support
    private weightOverlayEnabled = false;
    private weightOverlayApprox2Enabled = false;
    private weightOverlayUseRelevancy = false;
    private weightOverlayRelevancyTau = 10.0;
    private weightOverlayThreshold = 0.25;
    private weightOverlayOpacity = 0.6;
    private weightOverlayBlendWithBase = true;
    private weightOverlayGain = 1.5;
    private weightOverlayColormap: number = 0; // 0:Turbo,1:Viridis,2:Inferno,3:Magma,4:Plasma,5:Jet

    public constructor(renderer: THREE.WebGPURenderer, scene: THREE.Scene, gaussianModels: GaussianModel[]) {
        super();

        // Important: ensure this helper mesh is never frustum-culled by Three.js.
        // If it's culled (common because it has no geometry bounds), onBeforeRender
        // won't fire when the camera is very close, making GS appear "frozen".
        this.frustumCulled = false;

        this.threeRenderer = renderer;
        this.threeScene = scene;
        this.device = (renderer.backend as any).device as GPUDevice;
        // Surface WebGPU validation errors (otherwise failures can look like "frozen" frames).
        try {
            this.device.onuncapturederror = (ev: GPUUncapturedErrorEvent) => {
                const msg = (ev.error as any)?.message;
                console.error('[WebGPU] uncaptured error:', msg ?? ev.error, ev.error);
            };
        } catch {}
        const format = navigator.gpu.getPreferredCanvasFormat();
        this.renderer = new GaussianRenderer(this.device, format, 3);
        this.gaussianModels = gaussianModels;
        this.canvasFormat = format;
    }

    public onResize(width: number, height: number, _forceUpdate?: boolean): void {
        return; // editor在window.resize时调用会有深度问题，猜测在其他地方有冲突，暂时关闭,需要优化 by helen 2025-11-28
    }
    
    /**
     * Render Three.js scene internally to capture depth
     * This replaces the manual renderer.render(scene, camera) in demo
     */
    public renderThreeScene(camera: THREE.Camera): void {
        // Tracks whether we refreshed postProcessTarget this frame (important to avoid "frozen" stale contents).
        this.postProcessTargetWrittenThisFrame = false;
        if (!this.autoDepthMode) return;
        
        // Get actual drawing buffer size (matches GPU context)
        const dbSize = new THREE.Vector2();
        (this.threeRenderer as any).getDrawingBufferSize?.(dbSize);
        const width = dbSize.x || this.threeRenderer.domElement.width || 1;
        const height = dbSize.y || this.threeRenderer.domElement.height || 1;
        
        // Ensure scene depth RT is created with correct size
        if (!this.sceneDepthRT || this.sceneDepthRT.width !== width || this.sceneDepthRT.height !== height) {
            // Dispose old RT if exists
            if (this.sceneDepthRT) {
                this.sceneDepthRT.dispose();
            }
            
            // 关键修复：使用HalfFloatType（16位浮点）平衡精度和性能
            // FloatType（32位）不支持可过滤采样（UnfilterableFloat），会导致WebGPU验证错误
            // HalfFloatType（16位）支持过滤，精度通常足够，且性能更好
            // 使用LinearSRGBColorSpace，让Three.js保持线性空间，避免环境贴图被错误转换
            this.sceneDepthRT = new THREE.RenderTarget(width, height, {
                format: THREE.RGBAFormat,
                type: THREE.HalfFloatType, // 16位浮点，支持可过滤采样，精度通常足够
                samples: 1,
                depthBuffer: true,
            });
            // 使用LinearSRGBColorSpace，Three.js不会进行颜色空间转换
            // RT中存储的是线性值，环境贴图（HDR，线性空间）不会被错误转换
            // 在blit时再进行线性到sRGB的转换
            this.sceneDepthRT.texture.colorSpace = THREE.LinearSRGBColorSpace;
            this.sceneDepthTexture = new THREE.DepthTexture(width, height, THREE.FloatType);
            this.sceneDepthRT.depthTexture = this.sceneDepthTexture;
            
            if ((globalThis as any).GS_DEPTH_DEBUG) {
                //console.log(`[Depth] Created scene depth RT: ${width}x${height} (drawing buffer size)`);
            }
        }
        
        // Single render pass: Scene → RT (captures both color and depth)
        this.threeRenderer.setRenderTarget(this.sceneDepthRT);
        this.threeRenderer.clear(true, true, false);
        this.threeRenderer.render(this.threeScene, camera);
        this.threeRenderer.setRenderTarget(null);
        
        if ((globalThis as any).GS_DEPTH_DEBUG) {
            //console.log('[Depth] Rendered Three scene to RT (color + depth captured)');
        }
        
        // Blit RT color to canvas (no need to re-render scene)
        const postProcessView = this.ensurePostProcessTarget(width, height);
        this.blitRenderTargetToTarget(camera, postProcessView ?? undefined);
        this.postProcessTargetWrittenThisFrame = !!postProcessView;
    }

    /**
     * Blit RenderTarget color buffer to canvas using render pass
     * Uses a fullscreen quad shader to handle format conversion
     */
    private blitRenderTargetToTarget(camera: THREE.Camera, targetView?: GPUTextureView): void {
        if (!this.sceneDepthRT) return;

        try {
            // Get the GPU device
            const device = (this.threeRenderer as any).backend?.device as GPUDevice;
            if (!device) {
                console.warn('[Depth] No GPU device available for blit');
                this.threeRenderer.render(this.threeScene, camera);
                return;
            }

            let outputView = targetView;
            if (!outputView) {
                // Get canvas context
                const canvas = this.threeRenderer.domElement;
                const context = canvas.getContext('webgpu');
                if (!context) {
                    console.warn('[Depth] No WebGPU context available for blit');
                    this.threeRenderer.render(this.threeScene, camera);
                    return;
                }

                // Get current texture view (canvas)
                const currentTexture = context.getCurrentTexture();
                outputView = currentTexture.createView();
            }

            // Get RT color texture
            const backend = (this.threeRenderer as any).backend;
            const rtColorInfo = backend?.get?.(this.sceneDepthRT.texture);
            const rtColorTexture = rtColorInfo?.texture as GPUTexture;
            
            if (!rtColorTexture) {
                console.warn('[Depth] Could not access RT color texture for blit');
                this.threeRenderer.render(this.threeScene, camera);
                return;
            }

            // Check format compatibility
            const rtFormat = rtColorInfo?.format;
            const targetFormat = this.canvasFormat;

            if ((globalThis as any).GS_DEPTH_DEBUG) {
                //console.log('[Depth] RT format:', rtFormat, 'Target format:', targetFormat);
            }

            // Use render pass for format conversion (more reliable than copy)
            this.blitWithRenderPass(device, rtColorTexture, outputView, this.sceneDepthRT.width, this.sceneDepthRT.height);

            if ((globalThis as any).GS_DEPTH_DEBUG) {
                //console.log('[Depth] ✅ Blitted RT to canvas via render pass (format conversion)');
            }

        } catch (error) {
            console.warn('[Depth] Blit with render pass failed, falling back to re-render:', error);
            // Fallback: re-render scene to canvas
            this.threeRenderer.render(this.threeScene, camera);
        }
    }

    /**
     * Blit using render pass with fullscreen quad shader
     * This handles format conversion reliably
     */
    private blitWithRenderPass(device: GPUDevice, sourceTexture: GPUTexture, targetView: GPUTextureView, width: number, height: number): void {
        // 关键修复：改进采样器配置，使用更好的过滤方式以保持高光细节
        // 对于高动态范围内容（如环境贴图反射），使用linear过滤保持细节
        const sampler = device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
            mipmapFilter: 'linear', // 如果纹理有mipmap，使用线性过滤
            // 不使用anisotropic filtering，因为这是全屏blit，不需要
        });

        // Create bind group layout
        const bindGroupLayout = device.createBindGroupLayout({
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
                texture: { viewDimension: '2d' }
            }, {
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                sampler: {}
            }]
        });

        // Create bind group
        // 使用FloatType（32位浮点）格式的纹理视图，WebGPU会自动处理高精度采样
        // FloatType提供足够精度保持高动态范围内容（如环境贴图反射）的细节
        const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [{
                binding: 0,
                resource: sourceTexture.createView() // FloatType格式，提供高精度
            }, {
                binding: 1,
                resource: sampler
            }]
        });

        // Create render pipeline
        const pipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
            vertex: {
                module: device.createShaderModule({
                    code: `
                        @vertex
                        fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> @builtin(position) vec4f {
                            var pos = array<vec2f, 6>(
                                vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
                                vec2f(-1.0, 1.0),  vec2f(1.0, -1.0), vec2f(1.0, 1.0)
                            );
                            return vec4f(pos[vertexIndex], 0.0, 1.0);
                        }
                    `
                }),
                entryPoint: 'vs_main'
            },
            fragment: {
                module: device.createShaderModule({
                    code: `
                        @group(0) @binding(0) var sourceTexture: texture_2d<f32>;
                        @group(0) @binding(1) var sourceSampler: sampler;

                        // 线性空间到sRGB空间的转换函数（标准sRGB gamma校正）
                        fn linearToSRGB(linear: vec3<f32>) -> vec3<f32> {
                            return select(
                                linear * 12.92,
                                pow(max(linear, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) * 1.055 - 0.055,
                                linear > vec3<f32>(0.0031308)
                            );
                        }

                        @fragment
                        fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
                            let texCoord = fragCoord.xy / vec2f(${width}.0, ${height}.0);
                            // RT使用HalfFloatType（16位浮点），存储的是线性空间的值
                            // HalfFloatType支持可过滤采样，精度通常足够保持高动态范围内容
                            let linearColor = textureSample(sourceTexture, sourceSampler, texCoord);
                            
                            // 关键修复：将线性空间的值转换为sRGB空间输出到canvas
                            // 使用HalfFloatType（16位浮点）支持WebGPU的过滤采样，避免验证错误
                            // 在输出时进行线性到sRGB的转换，确保颜色正确显示
                            let srgbColor = linearToSRGB(linearColor.rgb);
                            
                            return vec4<f32>(srgbColor, linearColor.a);
                        }
                    `
                }),
                entryPoint: 'fs_main',
                targets: [{
                    format: this.canvasFormat // Canvas format (sRGB显示)
                }]
            },
            primitive: {
                topology: 'triangle-list'
            }
        });

        // Create command encoder and render pass
        const encoder = device.createCommandEncoder({ label: 'RT-to-Canvas render pass' });
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: targetView,
                loadOp: 'clear',
                storeOp: 'store',
                clearValue: { r: 0, g: 0, b: 0, a: 1 }
            }]
        });

        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(6, 1, 0, 0);
        pass.end();

        device.queue.submit([encoder.finish()]);
    }

    public onBeforeRender(renderer: any, scene: THREE.Scene, camera: THREE.Camera, _geometry?: any, _material?: any, _group?: any) {
        // 检查相机类型，兼容不同的Three.js导入方式
        if (!(camera instanceof THREE.PerspectiveCamera) && camera.type !== 'PerspectiveCamera') {
            console.log("Only THREE.PerspectiveCamera is supported!", camera);
            return;
        }


        
        const cam = this.convertCamera(camera as any, renderer as any);
        const visibleModels = this.gaussianModels.filter(model => model.isVisible(camera));
        // 只过滤高斯模型（PointCloud/DynamicPointCloud），FBX模型不参与高斯渲染
        // 注意：使用鸭子类型检查而不是instanceof，因为构建后的类名会被minify
        this.pcs = visibleModels.map(m => m.getPointCloud()).filter(
            (pc): pc is PointCloud | DynamicPointCloud => {
                // 检查是否有PointCloud的特征方法/属性
                return pc && typeof pc === 'object' && 
                       ('numPoints' in pc || 'countBuffer' in pc) && 
                       !('skeletalAnimation' in pc || 'fbxMesh' in pc); // 排除FBX模型
            }
        );
        if (!this.pcs || this.pcs.length === 0) {
            if ((globalThis as any).GS_VIDEO_EXPORT_DEBUG) {
                console.warn('[GaussianThreeJSRenderer] onBeforeRender: 没有可见的高斯点云');
                console.log('[GaussianThreeJSRenderer] - gaussianModels数量:', this.gaussianModels.length);
                console.log('[GaussianThreeJSRenderer] - visibleModels数量:', visibleModels.length);
                visibleModels.forEach((model, index) => {
                    const pc = model.getPointCloud();
                    console.log(`[GaussianThreeJSRenderer] - 模型${index} getPointCloud返回类型:`, 
                        pc.constructor.name, 
                        'is PointCloud:', pc instanceof PointCloud,
                        'is DynamicPointCloud:', pc instanceof DynamicPointCloud,
                        'is FBXModelWrapper:', pc instanceof FBXModelWrapper
                    );
                });
            }
            return;
        }

        // 同步每个可见模型的变换矩阵到GPU
        visibleModels.forEach((model, index) => {
            model.syncTransformToGPU();
            if ((globalThis as any).GS_DEBUG_FLAG) {
                //console.log(`[GaussianThreeJSRenderer] Synced transform for model ${index}: ${model.name}`);
            }
        });

        const device = (renderer.backend as any).device as GPUDevice;
        const encoder = device.createCommandEncoder({label: "frame"});
        // Use drawing buffer size for viewport in pixels
        const db = new THREE.Vector2();
        (renderer as any).getDrawingBufferSize?.(db);
        const vw = db.x || (renderer.getSize(new THREE.Vector2()).x);
        const vh = db.y || (renderer.getSize(new THREE.Vector2()).y);
        // 单次调用，利用全局批处理优化
        // 每个模型的 gaussianScaling 参数现在存储在各自的 ModelParams 中
        this.renderer.prepareMulti(encoder, device.queue, this.pcs, {
            camera: cam,
            viewport: [vw, vh],
            // 不再需要全局 gaussianScaling，每个模型使用自己的参数
        } as any);
        device.queue.submit([encoder.finish()]);

        // Note: In auto depth mode, scene rendering is done via renderThreeScene()
        if (this.autoDepthMode && (globalThis as any).GS_DEPTH_DEBUG) {
            //console.log('[Depth] Pass A: Skipped (auto depth mode - depth captured in renderThreeScene)');
        }
    }

    public drawSplats(renderer: any, scene: THREE.Scene, camera: THREE.Camera, _geometry?: any, _material?: any, _group?: any) {
        if (this.pcs == null || this.pcs.length === 0) {
            if ((globalThis as any).GS_VIDEO_EXPORT_DEBUG) {
                console.warn('[GaussianThreeJSRenderer] drawSplats: pcs为空或长度为0');
            }
          
            return false;
        }

        // NOTE: We intentionally do not force-call onBeforeRender() here.
        // The correct per-frame prepare is driven by the normal render flow; forcing it can hide real bugs.
        
        // 确保相机是PerspectiveCamera类型
        if (!(camera instanceof THREE.PerspectiveCamera) && camera.type !== 'PerspectiveCamera') {
            console.warn("drawSplats: Only THREE.PerspectiveCamera is supported!", camera);
            return false;
        }

        const device = (renderer.backend as any).device as GPUDevice;
        const context = (renderer.backend as any).context as GPUCanvasContext;
        const canvasView = context.getCurrentTexture().createView();
        let colorView = canvasView;
        if (this.postProcessEnabled) {
            const [vw, vh] = this.getViewport();
            const postProcessView = this.ensurePostProcessTarget(vw, vh);
            if (postProcessView) {
                colorView = postProcessView;
            }
        }
        const encoder = device.createCommandEncoder({label: "GS-render"});
        
        // Pass B: Try to get depth view from Three.js RenderTarget
        let depthView: GPUTextureView | undefined;
        
        // Use scene depth texture from renderThreeScene
        if (this.sceneDepthTexture) {
            // Get viewport size
            const db = new THREE.Vector2();
            (renderer as any).getDrawingBufferSize?.(db);
            const vw = db.x || (renderer.getSize(new THREE.Vector2()).x);
            const vh = db.y || (renderer.getSize(new THREE.Vector2()).y);
            
            if ((globalThis as any).GS_DEPTH_DEBUG) {
                //console.log('[Depth] Pass B: Attempting to get depth view');
                //console.log('[Depth] - Mode: auto (full scene)');
                //console.log('[Depth] - Viewport size:', vw, 'x', vh);
                //console.log('[Depth] - Scene depth RT size:', this.sceneDepthRT?.width, 'x', this.sceneDepthRT?.height);
                //console.log('[Depth] - Scene depth texture exists:', !!this.sceneDepthTexture);
            }
            
            try {
                // Try to access via renderer.backend
                const backend = (this.threeRenderer as any).backend;
                const backendInfo = backend?.get?.(this.sceneDepthTexture);
                
                if ((globalThis as any).GS_DEPTH_DEBUG) {
                    //console.log('[Depth] - Backend info:', backendInfo);
                }
                
                const depthGpuTexture = backendInfo?.texture as GPUTexture;
                const threeDepthFormat = backendInfo?.format as GPUTextureFormat;
                
                if (depthGpuTexture && threeDepthFormat) {
                    // Set GaussianRenderer to use the same depth format as Three.js
                    this.renderer.setDepthFormat(threeDepthFormat);
                    
                    depthView = depthGpuTexture.createView();
                    
                    if ((globalThis as any).GS_DEPTH_DEBUG) {
                        //console.log('[Depth] ✅ Pass B: Successfully got depth view from Three.js RT');
                        //console.log('[Depth] - Depth Format:', threeDepthFormat);
                        //console.log('[Depth] - Source: Full scene (auto depth mode)');
                    }
                } else {
                    if ((globalThis as any).GS_DEPTH_DEBUG) {
                        console.warn('[Depth] ⚠️ Could not access depth GPU texture from Three.js backend');
                    }
                }
                
                // Enable depth testing when we have depth texture
                if (depthView) {
                    this.renderer.setDepthEnabled(true);
                    
                    if ((globalThis as any).GS_DEPTH_DEBUG) {
                        //console.log('[Depth] - Depth testing enabled');
                    }
                }
                
            } catch (e) {
                if ((globalThis as any).GS_DEPTH_DEBUG) {
                    console.error('[Depth] ❌ Error accessing depth texture:', e);
                }
            }
        } else {
            // No depth source, disable depth testing
            this.renderer.setDepthEnabled(false);
            if ((globalThis as any).GS_DEPTH_DEBUG) {
                //console.log('[Depth] Pass B: No depth source, depth testing disabled');
            }
        }

        // Build render pass descriptor
        const passDescriptor: GPURenderPassDescriptor = {
            colorAttachments: [{
                view: colorView,
                clearValue: { r: 0, g: 0, b: 0, a: 1 },
                // If postprocess is enabled but the target wasn't refreshed this frame, don't "load" stale pixels.
                loadOp: (this.postProcessEnabled && colorView !== canvasView && !this.postProcessTargetWrittenThisFrame) ? 'clear' : 'load',
                storeOp: "store",
            }]
        };

        // Add depth attachment if available
        if (depthView) {
            (passDescriptor as any).depthStencilAttachment = {
                view: depthView,
                depthLoadOp: 'load', // Always load depth from RT
                depthStoreOp: 'store',
                depthClearValue: 1.0,
            };
            
            if ((globalThis as any).GS_DEPTH_DEBUG) {
                //console.log('[Depth] ✅ Depth attachment added to render pass');
                //console.log('[Depth] - depthLoadOp: load');
                //console.log('[Depth] - depthStoreOp: store');
            }
        } else {
            if ((globalThis as any).GS_DEPTH_DEBUG) {
                console.warn('[Depth] ⚠️ No depth view available - render pass has no depth attachment');
            }
        }

        const pass = encoder.beginRenderPass(passDescriptor);
        this.renderer.renderMulti(pass, this.pcs);
        pass.end();
        this.compositeOverlayToCanvas(device, encoder, colorView);

        if (this.weightOverlayEnabled) {
            const [vw, vh] = this.getViewport();
            const weightTexture = this.weightOverlayUseRelevancy
                ? ((this.renderer as any).recordWeightMapRelevancyPass?.(encoder, [vw, vh], depthView) as GPUTexture | null)
                : this.renderer.recordWeightMapPass(encoder, [vw, vh], depthView);
            if (weightTexture) {
                this.setPostProcessExtraTexture(weightTexture);
            }
        }
        if (this.weightOverlayApprox2Enabled) {
            const [vw, vh] = this.getViewport();
            const tex = (this.renderer as any).recordWeightMapApprox2Pass?.(encoder, [vw, vh], depthView) as GPUTexture | null;
            if (tex) {
                this.setPostProcessExtraTexture(tex);
            }
        }

        if (this.postProcessEnabled && this.postProcessTarget && colorView !== canvasView) {
            this.runPostProcessPass(device, encoder, this.postProcessTarget, canvasView, depthView);
        }
        device.queue.submit([encoder.finish()]);

        return true;
    }

    public renderOverlayScene(scene: THREE.Scene, camera: THREE.Camera): void {
        const [width, height] = this.getViewport();
        this.ensureGizmoOverlayRenderTarget(width, height);
        if (!this.gizmoOverlayRT) return;

        const prevRenderTarget = this.threeRenderer.getRenderTarget();
        const prevClearColor = new THREE.Color();
        (this.threeRenderer as any).getClearColor?.(prevClearColor);
        const prevClearAlpha = (this.threeRenderer as any).getClearAlpha?.() ?? 1;

        this.threeRenderer.setRenderTarget(this.gizmoOverlayRT);
        (this.threeRenderer as any).setClearColor?.(new THREE.Color(0x00000000), 0);
        this.threeRenderer.clear(true, false, false);
        this.threeRenderer.render(scene, camera);

        (this.threeRenderer as any).setClearColor?.(prevClearColor, prevClearAlpha);
        this.threeRenderer.setRenderTarget(prevRenderTarget);

        this.overlayRenderedThisFrame = true;
    }

    private ensureGizmoOverlayRenderTarget(width: number, height: number): void {
        const w = Math.max(1, Math.floor(width));
        const h = Math.max(1, Math.floor(height));

        if (this.gizmoOverlayRT && this.gizmoOverlayRT.width === w && this.gizmoOverlayRT.height === h) {
            return;
        }

        if (this.gizmoOverlayRT) {
            this.gizmoOverlayRT.dispose();
        }

        const RenderTargetClass =
            (THREE as any).WebGPURenderTarget ??
            (THREE as any).WebGLRenderTarget ??
            THREE.RenderTarget;

        this.gizmoOverlayRT = new RenderTargetClass(w, h);
        if (this.gizmoOverlayRT && this.gizmoOverlayRT.texture) {
            this.gizmoOverlayRT.texture.colorSpace = THREE.SRGBColorSpace;
        }
    }

    private compositeOverlayToCanvas(device: GPUDevice, encoder: GPUCommandEncoder, targetView: GPUTextureView): void {
        if (!this.overlayRenderedThisFrame || !this.gizmoOverlayRT) {
            return;
        }

        const backend = (this.threeRenderer as any).backend;
        const overlayInfo = backend?.get?.(this.gizmoOverlayRT.texture);
        const overlayTexture = overlayInfo?.texture as GPUTexture | undefined;

        if (!overlayTexture) {
            this.overlayRenderedThisFrame = false;
            return;
        }

        if (!this.overlaySampler) {
            this.overlaySampler = device.createSampler({
                magFilter: 'linear',
                minFilter: 'linear',
            });
        }

        if (!this.overlayBindGroupLayout) {
            this.overlayBindGroupLayout = device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.FRAGMENT,
                        texture: { viewDimension: '2d' },
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.FRAGMENT,
                        sampler: {},
                    },
                ],
            });
        }

        const bindGroup = device.createBindGroup({
            layout: this.overlayBindGroupLayout,
            entries: [
                { binding: 0, resource: overlayTexture.createView() },
                { binding: 1, resource: this.overlaySampler! },
            ],
        });

        if (!this.overlayPipeline) {
            const shaderModule = device.createShaderModule({
                code: `
                    struct VertexOutput {
                        @builtin(position) position : vec4f,
                        @location(0) uv : vec2f,
                    };

                    @vertex
                    fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
                        var positions = array<vec2f, 6>(
                            vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
                            vec2f(-1.0, 1.0),  vec2f(1.0, -1.0), vec2f(1.0, 1.0)
                        );
                        var uvs = array<vec2f, 6>(
                            vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(0.0, 0.0),
                            vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(1.0, 0.0)
                        );

                        var output : VertexOutput;
                        output.position = vec4f(positions[vertexIndex], 0.0, 1.0);
                        output.uv = uvs[vertexIndex];
                        return output;
                    }

                    @group(0) @binding(0) var overlayTexture : texture_2d<f32>;
                    @group(0) @binding(1) var overlaySampler : sampler;

                    @fragment
                    fn fs_main(@location(0) uv : vec2f) -> @location(0) vec4f {
                        return textureSample(overlayTexture, overlaySampler, uv);
                    }
                `,
            });

            const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [this.overlayBindGroupLayout] });
            this.overlayPipeline = device.createRenderPipeline({
                layout: pipelineLayout,
                vertex: {
                    module: shaderModule,
                    entryPoint: 'vs_main',
                },
                fragment: {
                    module: shaderModule,
                    entryPoint: 'fs_main',
                    targets: [
                        {
                            format: this.canvasFormat,
                            blend: {
                                color: {
                                    srcFactor: 'one',
                                    dstFactor: 'one-minus-src-alpha',
                                    operation: 'add',
                                },
                                alpha: {
                                    srcFactor: 'one',
                                    dstFactor: 'one-minus-src-alpha',
                                    operation: 'add',
                                },
                            },
                        },
                    ],
                },
                primitive: {
                    topology: 'triangle-list',
                },
            });
        }

        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: targetView,
                    loadOp: 'load',
                    storeOp: 'store',
                },
            ],
        });

        pass.setPipeline(this.overlayPipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(6, 1, 0, 0);
        pass.end();

        this.overlayRenderedThisFrame = false;
    }

    public async init() {
        await this.renderer.ensureSorter();
        console.log("GaussianThreeJSRenderer.init() Done!");
    }


    /**
     * Set occluder meshes for depth rendering (DEPRECATED - use auto depth mode instead)
     * Calling this will disable auto depth mode
     */
    public setOccluderMeshes(meshes: THREE.Mesh[]): void {
        console.warn('[GaussianThreeJSRenderer] setOccluderMeshes is deprecated. Auto depth mode captures the full scene automatically.');
        console.warn('[GaussianThreeJSRenderer] To use manual occluders, set autoDepthMode = false');
        this.autoDepthMode = false; // Disable auto mode when using manual occluders
        this.occluderMeshes = meshes;
        this.occluderScene.clear();
        meshes.forEach(m => this.occluderScene.add(m));
    }

    /**
     * Enable or disable auto depth mode
     * Auto mode: Captures depth from full scene automatically
     * Manual mode: Requires setOccluderMeshes()
     */
    public setAutoDepthMode(enabled: boolean): void {
        this.autoDepthMode = enabled;
        if ((globalThis as any).GS_DEPTH_DEBUG) {
            //console.log(`[Depth] Auto depth mode: ${enabled ? 'enabled (full scene)' : 'disabled (manual occluders)'}`);
        }
    }

    /**
     * Enable or disable the post-process pass.
     * When enabled, scene color + GS are rendered to an offscreen target,
     * then post-processed into the canvas.
     */
    public setPostProcessEnabled(enabled: boolean): void {
        this.postProcessEnabled = enabled;
    }

    /**
     * Set custom post-process fragment WGSL.
     * The fragment should define:
     *   @fragment fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f
     * Bindings:
     *   @group(0) @binding(0) var sourceTexture: texture_2d<f32>;
     *   @group(0) @binding(1) var sourceSampler: sampler;
     * Optional (when usesDepth=true):
     *   @group(0) @binding(2) var sourceDepth: texture_depth_2d;
     */
    public setPostProcessShader(fragmentWGSL: string | null, usesDepth: boolean = false): void {
        this.setPostProcessShaderEx(fragmentWGSL, { usesDepth });
    }

    public setPostProcessShaderEx(
        fragmentWGSL: string | null,
        options?: { usesDepth?: boolean; usesExtraTexture?: boolean; usesUniform?: boolean }
    ): void {
        this.postProcessFragmentWGSL = fragmentWGSL;
        this.postProcessUsesDepth = !!options?.usesDepth;
        this.postProcessUsesExtraTexture = !!options?.usesExtraTexture;
        this.postProcessUsesUniform = !!options?.usesUniform;
        this.postProcessPipeline = null;
        this.postProcessBindGroupLayout = null;
    }

    public setPostProcessExtraTexture(texture: GPUTexture | null): void {
        this.postProcessExtraTexture = texture;
    }

    public setPostProcessUniformData(data: Float32Array): void {
        this.ensurePostProcessUniformBuffer();
        if (!this.postProcessUniformBuffer) return;
        this.device.queue.writeBuffer(
            this.postProcessUniformBuffer,
            0,
            data.buffer,
            data.byteOffset,
            data.byteLength
        );
    }

    public setWeightQueryWeights(weights: Float32Array): void {
        this.renderer.setWeightQueryWeights(weights);
    }

    public setWeightQueryCosineData(query64: Float32Array, gram64x64: Float32Array): void {
        // Keep both heatmap algorithms fed with the latest query data.
        this.renderer.setWeightQueryWeights(query64, gram64x64);
        try { (this.renderer as any).setWeightQueryNumDen2?.(query64, gram64x64); } catch {}
    }

    public setLanguageWeightsSoftmax(enabled: boolean, temperature: number = 1.0): void {
        (this.renderer as any).setWeightSoftmaxConfig?.(enabled, temperature);
    }

    public async loadWeightQueryFromUrls(
        codebookUrl: string,
        queryListUrl: string,
        queryName?: string
    ): Promise<void> {
        const [codebookBuffer, queryJson] = await Promise.all([
            fetch(codebookUrl).then(async (res) => {
                if (!res.ok) {
                    throw new Error(`Failed to fetch codebook: ${res.status} ${res.statusText}`);
                }
                return res.arrayBuffer();
            }),
            fetch(queryListUrl).then(async (res) => {
                if (!res.ok) {
                    throw new Error(`Failed to fetch query list: ${res.status} ${res.statusText}`);
                }
                return res.json();
            }),
        ]);

        const languageCodebook = new Float32Array(codebookBuffer);
        const languageQuery512 = this.extractQueryVector(queryJson, queryName);
        const languageQuery64 = this.computeQueryWeights(languageCodebook, languageQuery512);
        const gram = this.computeGramMatrix(languageCodebook);
        this.setWeightQueryCosineData(languageQuery64, gram);
    }

    public setWeightQueryFromData(codebook: Float32Array, queryVector: Float32Array): void {
        const languageCodebook = codebook;
        const languageQuery512 = queryVector;
        const languageQuery64 = this.computeQueryWeights(languageCodebook, languageQuery512);
        const gram = this.computeGramMatrix(languageCodebook);
        this.setWeightQueryCosineData(languageQuery64, gram);
    }

    public setWeightOverlayEnabled(enabled: boolean, options?: { threshold?: number; opacity?: number }): void {
        this.weightOverlayEnabled = !!enabled;
        if (this.weightOverlayEnabled) this.weightOverlayApprox2Enabled = false;
        if (options?.threshold !== undefined) this.weightOverlayThreshold = options.threshold;
        if (options?.opacity !== undefined) this.weightOverlayOpacity = options.opacity;

        if (this.weightOverlayEnabled) {
            // Enable the correct per-GS generator (plain similarity vs neg relevancy).
            if (this.weightOverlayUseRelevancy) {
                try { (this.renderer as any).setWeightMapRelevancyEnabled?.(true); } catch {}
                try { (this.renderer as any).setWeightMapEnabled?.(false); } catch {}
            } else {
                this.renderer.setWeightMapEnabled(true);
                try { (this.renderer as any).setWeightMapRelevancyEnabled?.(false); } catch {}
            }
            this.setPostProcessEnabled(true);
            this.setPostProcessShaderEx(this.getWeightOverlayShader(), {
                usesExtraTexture: true,
                usesUniform: true,
            });
            this.ensureDummyPostProcessExtraTexture();
            // Fallback to dummy until the weight map RT is produced this frame.
            if (!this.postProcessExtraTexture) this.setPostProcessExtraTexture(this.dummyPostProcessExtraTexture);
            this.updateWeightOverlayUniform();
        } else {
            // Hard-disable the entire heatmap pipeline when not in use.
            // (No extra similarity pass, no postprocess blit path.)
            try { (this.renderer as any).setWeightMapEnabled?.(false); } catch {}
            try { (this.renderer as any).setWeightMapRelevancyEnabled?.(false); } catch {}
            this.setPostProcessEnabled(false);
            this.setPostProcessExtraTexture(null);
        }
    }

    public setWeightOverlayApprox2Enabled(enabled: boolean, options?: { threshold?: number; opacity?: number }): void {
        this.weightOverlayApprox2Enabled = !!enabled;
        if (this.weightOverlayApprox2Enabled) this.weightOverlayEnabled = false;
        // Reuse same UI params
        if (options?.threshold !== undefined) this.weightOverlayThreshold = options.threshold;
        if (options?.opacity !== undefined) this.weightOverlayOpacity = options.opacity;

        if (!!enabled) {
            // Ensure per-GS similarity path is off (mutual exclusion)
            try { (this.renderer as any).setWeightMapEnabled?.(false); } catch {}
            try { (this.renderer as any).setWeightMapRelevancyEnabled?.(false); } catch {}
            try { (this.renderer as any).setWeightMapApprox2Enabled?.(true); } catch {}

            this.setPostProcessEnabled(true);
            this.setPostProcessShaderEx(this.getWeightOverlayApprox2Shader(), {
                usesExtraTexture: true,
                usesUniform: true,
            });
            this.ensureDummyPostProcessExtraTexture();
            if (!this.postProcessExtraTexture) this.setPostProcessExtraTexture(this.dummyPostProcessExtraTexture);
            this.updateWeightOverlayUniform();
        } else {
            try { (this.renderer as any).setWeightMapApprox2Enabled?.(false); } catch {}
            this.setPostProcessEnabled(false);
            this.setPostProcessExtraTexture(null);
        }
    }

    private ensureDummyPostProcessExtraTexture(): void {
        if (this.dummyPostProcessExtraTexture) return;
        // 1x1 rgba16float is compatible with texture_2d<f32> sampling.
        this.dummyPostProcessExtraTexture = this.device.createTexture({
            label: 'postprocess/dummy-extra',
            size: { width: 1, height: 1 },
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        // Write zeros.
        // WebGPU requires bytesPerRow to be a multiple of 256. For a 1x1 RGBA16F pixel,
        // we upload one padded row (256 bytes) with the first 8 bytes = 0.
        const padded = new Uint8Array(256);
        this.device.queue.writeTexture(
            { texture: this.dummyPostProcessExtraTexture },
            padded,
            { bytesPerRow: 256, rowsPerImage: 1 },
            { width: 1, height: 1, depthOrArrayLayers: 1 }
        );
    }

    private getWeightOverlayApprox2Shader(): string {
        // extraTexture stores (num, den2) in rg, accumulated with premultiplied over.
        // IMPORTANT: Do NOT declare bindings here. `ensurePostProcessPipeline()` injects:
        // - sourceTexture/sourceSampler at bindings 0/1
        // - extraTexture/extraSampler at bindings 3/4 (when usesExtraTexture=true)
        // - postParams uniform at binding 5 (when usesUniform=true)
        return /* wgsl */ `
            fn sample_lut(t: f32, lut: array<vec3f, 16>) -> vec3f {
                let x = clamp(t, 0.0, 1.0) * 15.0;
                let i0 = u32(floor(x));
                let i1 = min(15u, i0 + 1u);
                let f = fract(x);
                return mix(lut[i0], lut[i1], f);
            }

            const TURBO : array<vec3f, 16> = array<vec3f, 16>(
                vec3f(0.189,0.071,0.232), vec3f(0.259,0.275,0.518), vec3f(0.241,0.488,0.706), vec3f(0.149,0.690,0.741),
                vec3f(0.240,0.824,0.580), vec3f(0.508,0.902,0.329), vec3f(0.792,0.888,0.165), vec3f(0.961,0.787,0.173),
                vec3f(0.992,0.620,0.208), vec3f(0.955,0.412,0.223), vec3f(0.843,0.231,0.233), vec3f(0.678,0.111,0.242),
                vec3f(0.497,0.047,0.256), vec3f(0.330,0.034,0.271), vec3f(0.208,0.049,0.291), vec3f(0.140,0.078,0.299)
            );
            const VIRIDIS : array<vec3f, 16> = array<vec3f, 16>(
                vec3f(0.267,0.005,0.329), vec3f(0.283,0.141,0.458), vec3f(0.254,0.265,0.530), vec3f(0.207,0.372,0.553),
                vec3f(0.164,0.471,0.558), vec3f(0.128,0.567,0.551), vec3f(0.135,0.659,0.518), vec3f(0.267,0.749,0.441),
                vec3f(0.478,0.821,0.318), vec3f(0.678,0.863,0.189), vec3f(0.824,0.884,0.106), vec3f(0.914,0.896,0.119),
                vec3f(0.965,0.909,0.190), vec3f(0.988,0.933,0.308), vec3f(0.993,0.959,0.439), vec3f(0.993,0.984,0.603)
            );
            const INFERNO : array<vec3f, 16> = array<vec3f, 16>(
                vec3f(0.002,0.005,0.013), vec3f(0.081,0.017,0.174), vec3f(0.206,0.016,0.318), vec3f(0.340,0.057,0.431),
                vec3f(0.472,0.112,0.428), vec3f(0.604,0.176,0.401), vec3f(0.734,0.254,0.352), vec3f(0.845,0.353,0.285),
                vec3f(0.930,0.466,0.220), vec3f(0.972,0.584,0.165), vec3f(0.988,0.708,0.153), vec3f(0.975,0.822,0.216),
                vec3f(0.936,0.914,0.330), vec3f(0.885,0.967,0.476), vec3f(0.839,0.993,0.646), vec3f(0.988,0.998,0.905)
            );
            const MAGMA : array<vec3f, 16> = array<vec3f, 16>(
                vec3f(0.001,0.000,0.014), vec3f(0.088,0.028,0.156), vec3f(0.212,0.060,0.304), vec3f(0.341,0.091,0.427),
                vec3f(0.469,0.117,0.514), vec3f(0.595,0.157,0.558), vec3f(0.713,0.216,0.565), vec3f(0.815,0.298,0.540),
                vec3f(0.895,0.404,0.490), vec3f(0.948,0.533,0.427), vec3f(0.976,0.678,0.363), vec3f(0.983,0.827,0.320),
                vec3f(0.965,0.949,0.400), vec3f(0.992,0.991,0.545), vec3f(0.996,0.999,0.741), vec3f(0.998,0.999,0.918)
            );
            const PLASMA : array<vec3f, 16> = array<vec3f, 16>(
                vec3f(0.050,0.030,0.528), vec3f(0.209,0.027,0.675), vec3f(0.353,0.049,0.725), vec3f(0.488,0.079,0.708),
                vec3f(0.610,0.116,0.650), vec3f(0.710,0.164,0.563), vec3f(0.794,0.222,0.460), vec3f(0.861,0.292,0.357),
                vec3f(0.909,0.376,0.266), vec3f(0.941,0.474,0.199), vec3f(0.958,0.585,0.163), vec3f(0.961,0.709,0.167),
                vec3f(0.940,0.843,0.240), vec3f(0.883,0.945,0.375), vec3f(0.790,0.996,0.557), vec3f(0.637,0.996,0.714)
            );
            const JET : array<vec3f, 16> = array<vec3f, 16>(
                vec3f(0.000,0.000,0.500), vec3f(0.000,0.000,1.000), vec3f(0.000,0.500,1.000), vec3f(0.000,1.000,1.000),
                vec3f(0.500,1.000,0.500), vec3f(1.000,1.000,0.000), vec3f(1.000,0.500,0.000), vec3f(1.000,0.000,0.000),
                vec3f(0.500,0.000,0.000), vec3f(0.500,0.000,0.000), vec3f(0.500,0.000,0.000), vec3f(0.500,0.000,0.000),
                vec3f(0.500,0.000,0.000), vec3f(0.500,0.000,0.000), vec3f(0.500,0.000,0.000), vec3f(0.500,0.000,0.000)
            );

            fn colormap(t: f32, id: u32) -> vec3f {
                if (id == 1u) { return sample_lut(t, VIRIDIS); }
                if (id == 2u) { return sample_lut(t, INFERNO); }
                if (id == 3u) { return sample_lut(t, MAGMA); }
                if (id == 4u) { return sample_lut(t, PLASMA); }
                if (id == 5u) { return sample_lut(t, JET); }
                return sample_lut(t, TURBO);
            }

            @fragment
            fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
                let base = textureSample(sourceTexture, sourceSampler, uv);
                let nd = textureSample(extraTexture, extraSampler, uv);
                let num = nd.r;
                let den2 = nd.g;
                let sim = num / sqrt(max(1e-12, den2));
                let similarity = clamp(max(0.0, sim), 0.0, 1.0);

                let s = clamp(similarity, 0.0, 1.0);
                let gateSmooth = max(1e-6, postParams.gateSmooth);
                let gate = smoothstep(postParams.threshold, min(1.0, postParams.threshold + gateSmooth), s);
                let hm = colormap(s, u32(postParams.colormap + 0.5));
                let alpha = clamp(gate * postParams.opacity * postParams.gain, 0.0, 1.0);
                let blendWithBase = postParams.blendWithBase > 0.5;
                let out_rgb = select(hm * alpha, mix(base.rgb, hm, alpha), blendWithBase);
                return vec4f(out_rgb, base.a);
            }
        `;
    }

    public setWeightOverlayParams(options: { threshold?: number; opacity?: number }): void {
        if (options.threshold !== undefined) this.weightOverlayThreshold = options.threshold;
        if (options.opacity !== undefined) this.weightOverlayOpacity = options.opacity;
        this.updateWeightOverlayUniform();
    }

    public setWeightOverlayBlendWithBase(blendWithBase: boolean): void {
        this.weightOverlayBlendWithBase = !!blendWithBase;
        this.updateWeightOverlayUniform();
    }

    public setWeightOverlayVisualOptions(options: { gain?: number; colormap?: number }): void {
        if (options.gain !== undefined) {
            const g = Number(options.gain);
            this.weightOverlayGain = Number.isFinite(g) ? Math.max(0.0, g) : this.weightOverlayGain;
        }
        if (options.colormap !== undefined) {
            const c = Number(options.colormap);
            this.weightOverlayColormap = Number.isFinite(c) ? Math.max(0, Math.floor(c)) : this.weightOverlayColormap;
        }
        this.updateWeightOverlayUniform();
    }

    public setWeightOverlayUseRelevancy(enabled: boolean, tau: number = 10.0): void {
        this.weightOverlayUseRelevancy = !!enabled;
        this.weightOverlayRelevancyTau = Number.isFinite(tau) ? Math.max(1e-6, tau) : 10.0;
    }

    /** Provide packed queries for relevancy: 5x64 floats [pos, neg0..3]. */
    public setWeightOverlayRelevancyData(packedQueries: Float32Array, gram64x64: Float32Array, negMask: number = 0xF): void {
        try {
            (this.renderer as any).setWeightRelevancyData?.(
                packedQueries,
                gram64x64,
                true,
                this.weightOverlayRelevancyTau,
                negMask
            );
        } catch {}
    }

    /**
     * Diagnostic: Check if depth is properly configured
     */
    public diagnoseDepth(): void {
        console.group('[Depth Diagnostic]');
        console.log('Auto depth mode:', this.autoDepthMode);
        console.log('Scene depth RT exists:', !!this.sceneDepthRT);
        console.log('Scene depth texture exists:', !!this.sceneDepthTexture);
        
        if (this.sceneDepthRT) {
            console.log('Scene depth RT size:', this.sceneDepthRT.width, 'x', this.sceneDepthRT.height);
            console.log('Scene depth RT format:', this.sceneDepthRT.texture.format);
        }
        
        if (this.renderer) {
            console.log('GaussianRenderer depth enabled:', (this.renderer as any).useDepth);
            console.log('GaussianRenderer depth format:', (this.renderer as any).depthFormat);
        }
        
        console.groupEnd();
    }

    /**
     * Clean up depth resources
     */
    public disposeDepthResources(): void {
        if (this.sceneDepthRT) {
            this.sceneDepthRT.dispose();
            this.sceneDepthRT = null;
        }
        
        this.sceneDepthTexture = null;
        
        if ((globalThis as any).GS_DEPTH_DEBUG) {
            console.log('[Depth] Cleaned up depth resources');
        }

        if (this.gizmoOverlayRT) {
            this.gizmoOverlayRT.dispose();
            this.gizmoOverlayRT = null;
        }
        this.overlayPipeline = null;
        this.overlayBindGroupLayout = null;
        this.overlaySampler = null;
        this.overlayRenderedThisFrame = false;

        if (this.postProcessTarget) {
            this.postProcessTarget.destroy();
            this.postProcessTarget = null;
        }
        this.postProcessTargetView = null;
        this.postProcessTargetWidth = 0;
        this.postProcessTargetHeight = 0;
        this.postProcessSampler = null;
        this.postProcessBindGroupLayout = null;
        this.postProcessPipeline = null;
        this.postProcessExtraTexture = null;
        this.postProcessExtraSampler = null;
        this.postProcessUsesExtraTexture = false;
        this.postProcessUsesUniform = false;
        this.postProcessUniformBuffer = null;
        this.weightOverlayEnabled = false;
        this.weightOverlayApprox2Enabled = false;
    }

    private ensurePostProcessTarget(width: number, height: number): GPUTextureView | null {
        if (!this.postProcessEnabled) return null;
        if (!this.device) return null;

        const w = Math.max(1, Math.floor(width));
        const h = Math.max(1, Math.floor(height));

        if (this.postProcessTarget && this.postProcessTargetWidth === w && this.postProcessTargetHeight === h) {
            return this.postProcessTargetView;
        }

        if (this.postProcessTarget) {
            this.postProcessTarget.destroy();
        }

        this.postProcessTarget = this.device.createTexture({
            label: 'GS postprocess target',
            size: { width: w, height: h },
            format: this.canvasFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });
        this.postProcessTargetView = this.postProcessTarget.createView();
        this.postProcessTargetWidth = w;
        this.postProcessTargetHeight = h;
        return this.postProcessTargetView;
    }

    private ensurePostProcessUniformBuffer(): void {
        if (this.postProcessUniformBuffer) return;
        this.postProcessUniformBuffer = this.device.createBuffer({
            label: 'postprocess params',
            size: 32, // 2 * vec4<f32>
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    private updateWeightOverlayUniform(): void {
        // Both heatmap algorithms share the same postprocess uniform.
        if (!this.weightOverlayEnabled && !this.weightOverlayApprox2Enabled) return;
        this.ensurePostProcessUniformBuffer();
        const data = new Float32Array([
            this.weightOverlayThreshold,
            this.weightOverlayOpacity,
            this.weightOverlayBlendWithBase ? 1.0 : 0.0,
            this.weightOverlayGain,
            // vec4 #2
            this.weightOverlayColormap,
            0, // reserved (gamma etc.)
            0.02, // gate smoothness in [0,1] similarity space
            0,
        ]);
        this.device.queue.writeBuffer(
            this.postProcessUniformBuffer!,
            0,
            data.buffer,
            data.byteOffset,
            data.byteLength
        );
    }

    private getWeightOverlayShader(): string {
        return `
            fn sample_lut(t: f32, lut: array<vec3f, 16>) -> vec3f {
                let x = clamp(t, 0.0, 1.0) * 15.0;
                let i0 = u32(floor(x));
                let i1 = min(15u, i0 + 1u);
                let f = fract(x);
                return mix(lut[i0], lut[i1], f);
            }

            const TURBO : array<vec3f, 16> = array<vec3f, 16>(
                vec3f(0.189,0.071,0.232), vec3f(0.259,0.275,0.518), vec3f(0.241,0.488,0.706), vec3f(0.149,0.690,0.741),
                vec3f(0.240,0.824,0.580), vec3f(0.508,0.902,0.329), vec3f(0.792,0.888,0.165), vec3f(0.961,0.787,0.173),
                vec3f(0.992,0.620,0.208), vec3f(0.955,0.412,0.223), vec3f(0.843,0.231,0.233), vec3f(0.678,0.111,0.242),
                vec3f(0.497,0.047,0.256), vec3f(0.330,0.034,0.271), vec3f(0.208,0.049,0.291), vec3f(0.140,0.078,0.299)
            );
            const VIRIDIS : array<vec3f, 16> = array<vec3f, 16>(
                vec3f(0.267,0.005,0.329), vec3f(0.283,0.141,0.458), vec3f(0.254,0.265,0.530), vec3f(0.207,0.372,0.553),
                vec3f(0.164,0.471,0.558), vec3f(0.128,0.567,0.551), vec3f(0.135,0.659,0.518), vec3f(0.267,0.749,0.441),
                vec3f(0.478,0.821,0.318), vec3f(0.678,0.863,0.189), vec3f(0.824,0.884,0.106), vec3f(0.914,0.896,0.119),
                vec3f(0.965,0.909,0.190), vec3f(0.988,0.933,0.308), vec3f(0.993,0.959,0.439), vec3f(0.993,0.984,0.603)
            );
            const INFERNO : array<vec3f, 16> = array<vec3f, 16>(
                vec3f(0.002,0.005,0.013), vec3f(0.081,0.017,0.174), vec3f(0.206,0.016,0.318), vec3f(0.340,0.057,0.431),
                vec3f(0.472,0.112,0.428), vec3f(0.604,0.176,0.401), vec3f(0.734,0.254,0.352), vec3f(0.845,0.353,0.285),
                vec3f(0.930,0.466,0.220), vec3f(0.972,0.584,0.165), vec3f(0.988,0.708,0.153), vec3f(0.975,0.822,0.216),
                vec3f(0.936,0.914,0.330), vec3f(0.885,0.967,0.476), vec3f(0.839,0.993,0.646), vec3f(0.988,0.998,0.905)
            );
            const MAGMA : array<vec3f, 16> = array<vec3f, 16>(
                vec3f(0.001,0.000,0.014), vec3f(0.088,0.028,0.156), vec3f(0.212,0.060,0.304), vec3f(0.341,0.091,0.427),
                vec3f(0.469,0.117,0.514), vec3f(0.595,0.157,0.558), vec3f(0.713,0.216,0.565), vec3f(0.815,0.298,0.540),
                vec3f(0.895,0.404,0.490), vec3f(0.948,0.533,0.427), vec3f(0.976,0.678,0.363), vec3f(0.983,0.827,0.320),
                vec3f(0.965,0.949,0.400), vec3f(0.992,0.991,0.545), vec3f(0.996,0.999,0.741), vec3f(0.998,0.999,0.918)
            );
            const PLASMA : array<vec3f, 16> = array<vec3f, 16>(
                vec3f(0.050,0.030,0.528), vec3f(0.209,0.027,0.675), vec3f(0.353,0.049,0.725), vec3f(0.488,0.079,0.708),
                vec3f(0.610,0.116,0.650), vec3f(0.710,0.164,0.563), vec3f(0.794,0.222,0.460), vec3f(0.861,0.292,0.357),
                vec3f(0.909,0.376,0.266), vec3f(0.941,0.474,0.199), vec3f(0.958,0.585,0.163), vec3f(0.961,0.709,0.167),
                vec3f(0.940,0.843,0.240), vec3f(0.883,0.945,0.375), vec3f(0.790,0.996,0.557), vec3f(0.637,0.996,0.714)
            );
            const JET : array<vec3f, 16> = array<vec3f, 16>(
                vec3f(0.000,0.000,0.500), vec3f(0.000,0.000,1.000), vec3f(0.000,0.500,1.000), vec3f(0.000,1.000,1.000),
                vec3f(0.500,1.000,0.500), vec3f(1.000,1.000,0.000), vec3f(1.000,0.500,0.000), vec3f(1.000,0.000,0.000),
                vec3f(0.500,0.000,0.000), vec3f(0.500,0.000,0.000), vec3f(0.500,0.000,0.000), vec3f(0.500,0.000,0.000),
                vec3f(0.500,0.000,0.000), vec3f(0.500,0.000,0.000), vec3f(0.500,0.000,0.000), vec3f(0.500,0.000,0.000)
            );

            fn colormap(t: f32, id: u32) -> vec3f {
                if (id == 1u) { return sample_lut(t, VIRIDIS); }
                if (id == 2u) { return sample_lut(t, INFERNO); }
                if (id == 3u) { return sample_lut(t, MAGMA); }
                if (id == 4u) { return sample_lut(t, PLASMA); }
                if (id == 5u) { return sample_lut(t, JET); }
                return sample_lut(t, TURBO);
            }

            @fragment
            fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
                let base = textureSample(sourceTexture, sourceSampler, uv);
                // The extraTexture stores per-pixel language similarity in [0, 1].
                let similarity = textureSample(extraTexture, extraSampler, uv).r;
                let s = clamp(similarity, 0.0, 1.0);
                let gateSmooth = max(1e-6, postParams.gateSmooth);
                let gate = smoothstep(postParams.threshold, min(1.0, postParams.threshold + gateSmooth), s);
                let hm = colormap(s, u32(postParams.colormap + 0.5));
                let alpha = clamp(gate * postParams.opacity * postParams.gain, 0.0, 1.0);
                let blendWithBase = postParams.blendWithBase > 0.5;
                let out_rgb = select(hm * alpha, mix(base.rgb, hm, alpha), blendWithBase);
                return vec4<f32>(out_rgb, base.a);
            }
        `;
    }

    private async computeQueryWeightsFromUrls(
        codebookUrl: string,
        queryListUrl: string,
        queryName?: string
    ): Promise<Float32Array> {
        const [codebookBuffer, queryJson] = await Promise.all([
            fetch(codebookUrl).then(async (res) => {
                if (!res.ok) {
                    throw new Error(`Failed to fetch codebook: ${res.status} ${res.statusText}`);
                }
                return res.arrayBuffer();
            }),
            fetch(queryListUrl).then(async (res) => {
                if (!res.ok) {
                    throw new Error(`Failed to fetch query list: ${res.status} ${res.statusText}`);
                }
                return res.json();
            }),
        ]);

        const codebook = new Float32Array(codebookBuffer);
        const queryVector = this.extractQueryVector(queryJson, queryName);
        return this.computeQueryWeights(codebook, queryVector);
    }

    private extractQueryVector(queryJson: any, queryName?: string): Float32Array {
        const queries = queryJson?.queries;
        if (!Array.isArray(queries) || queries.length === 0) {
            throw new Error('Query list JSON has no queries');
        }
        const selected = queryName
            ? queries.find((q: any) => q?.name === queryName)
            : queries[0];
        if (!selected || !Array.isArray(selected.vector)) {
            throw new Error('Query vector not found in query list JSON');
        }
        return new Float32Array(selected.vector);
    }

    private computeQueryWeights(codebook: Float32Array, queryVector: Float32Array): Float32Array {
        const rows = 64;
        const cols = 512;
        if (codebook.length !== rows * cols) {
            throw new Error(`Codebook length mismatch: expected ${rows * cols}, got ${codebook.length}`);
        }
        if (queryVector.length !== cols) {
            throw new Error(`Query vector length mismatch: expected ${cols}, got ${queryVector.length}`);
        }
        // L2 normalize query (defensive; exporter may or may not have normalized)
        const q = new Float32Array(queryVector);
        {
            let n2 = 0;
            for (let i = 0; i < q.length; i++) n2 += q[i] * q[i];
            const inv = n2 > 1e-12 ? 1.0 / Math.sqrt(n2) : 1.0;
            for (let i = 0; i < q.length; i++) q[i] *= inv;
        }

        const out = new Float32Array(rows);
        for (let r = 0; r < rows; r++) {
            let sum = 0;
            const rowBase = r * cols;
            for (let c = 0; c < cols; c++) {
                sum += codebook[rowBase + c] * q[c];
            }
            out[r] = sum;
        }
        return out;
    }

    private computeGramMatrix(codebook: Float32Array): Float32Array {
        const rows = 64;
        const cols = 512;
        if (codebook.length !== rows * cols) {
            throw new Error(`Codebook length mismatch: expected ${rows * cols}, got ${codebook.length}`);
        }
        const gram = new Float32Array(rows * rows);
        for (let i = 0; i < rows; i++) {
            const iBase = i * cols;
            for (let j = i; j < rows; j++) {
                const jBase = j * cols;
                let sum = 0;
                for (let k = 0; k < cols; k++) {
                    sum += codebook[iBase + k] * codebook[jBase + k];
                }
                gram[i * rows + j] = sum;
                gram[j * rows + i] = sum;
            }
        }
        return gram;
    }

    private ensurePostProcessPipeline(device: GPUDevice): GPURenderPipeline {
        if (this.postProcessPipeline && this.postProcessBindGroupLayout) {
            return this.postProcessPipeline;
        }

        const bindGroupEntries: GPUBindGroupLayoutEntry[] = [
            {
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
                texture: { viewDimension: '2d' },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.FRAGMENT,
                sampler: {},
            },
        ];

        if (this.postProcessUsesDepth) {
            bindGroupEntries.push({
                binding: 2,
                visibility: GPUShaderStage.FRAGMENT,
                texture: { sampleType: 'depth' },
            });
        }

        if (this.postProcessUsesExtraTexture) {
            bindGroupEntries.push({
                binding: 3,
                visibility: GPUShaderStage.FRAGMENT,
                texture: { viewDimension: '2d' },
            });
            bindGroupEntries.push({
                binding: 4,
                visibility: GPUShaderStage.FRAGMENT,
                sampler: {},
            });
        }

        if (this.postProcessUsesUniform) {
            bindGroupEntries.push({
                binding: 5,
                visibility: GPUShaderStage.FRAGMENT,
                buffer: { type: 'uniform' },
            });
        }

        this.postProcessBindGroupLayout = device.createBindGroupLayout({
            entries: bindGroupEntries,
        });

        const fragmentCode = `
            @group(0) @binding(0) var sourceTexture: texture_2d<f32>;
            @group(0) @binding(1) var sourceSampler: sampler;
            ${this.postProcessUsesDepth ? '@group(0) @binding(2) var sourceDepth: texture_depth_2d;' : ''}
            ${this.postProcessUsesExtraTexture ? '@group(0) @binding(3) var extraTexture: texture_2d<f32>;\n@group(0) @binding(4) var extraSampler: sampler;' : ''}
            ${this.postProcessUsesUniform ? 'struct PostProcessParams { threshold: f32, opacity: f32, blendWithBase: f32, gain: f32, colormap: f32, gamma: f32, gateSmooth: f32, _pad0: f32 };\n@group(0) @binding(5) var<uniform> postParams: PostProcessParams;' : ''}

            ${this.postProcessFragmentWGSL ?? `
            @fragment
            fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
                return textureSample(sourceTexture, sourceSampler, uv);
            }
            `}
        `;

        const shaderModule = device.createShaderModule({
            code: `
                struct VertexOutput {
                    @builtin(position) position : vec4f,
                    @location(0) uv : vec2f,
                };

                @vertex
                fn vs_main(@builtin(vertex_index) vertexIndex : u32) -> VertexOutput {
                    var positions = array<vec2f, 6>(
                        vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
                        vec2f(-1.0, 1.0),  vec2f(1.0, -1.0), vec2f(1.0, 1.0)
                    );
                    var uvs = array<vec2f, 6>(
                        vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(0.0, 0.0),
                        vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(1.0, 0.0)
                    );

                    var output : VertexOutput;
                    output.position = vec4f(positions[vertexIndex], 0.0, 1.0);
                    output.uv = uvs[vertexIndex];
                    return output;
                }

                ${fragmentCode}
            `,
        });

        this.postProcessPipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.postProcessBindGroupLayout] }),
            vertex: {
                module: shaderModule,
                entryPoint: 'vs_main',
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_main',
                targets: [
                    {
                        format: this.canvasFormat,
                    },
                ],
            },
            primitive: {
                topology: 'triangle-list',
            },
        });

        return this.postProcessPipeline;
    }

    private runPostProcessPass(
        device: GPUDevice,
        encoder: GPUCommandEncoder,
        sourceTexture: GPUTexture,
        targetView: GPUTextureView,
        depthView?: GPUTextureView
    ): void {
        if (!this.postProcessEnabled) return;
        if (this.postProcessUsesDepth && !depthView) {
            console.warn('[PostProcess] Depth requested but not available, skipping pass.');
            return;
        }
        if (this.postProcessUsesExtraTexture && !this.postProcessExtraTexture) {
            console.warn('[PostProcess] Extra texture requested but not available, skipping pass.');
            return;
        }
        if (this.postProcessUsesUniform && !this.postProcessUniformBuffer) {
            this.ensurePostProcessUniformBuffer();
        }

        if (!this.postProcessSampler) {
            this.postProcessSampler = device.createSampler({
                magFilter: 'linear',
                minFilter: 'linear',
            });
        }

        const pipeline = this.ensurePostProcessPipeline(device);
        const bindGroupEntries: GPUBindGroupEntry[] = [
            { binding: 0, resource: sourceTexture.createView() },
            { binding: 1, resource: this.postProcessSampler },
        ];
        if (this.postProcessUsesDepth && depthView) {
            bindGroupEntries.push({ binding: 2, resource: depthView });
        }
        if (this.postProcessUsesExtraTexture && this.postProcessExtraTexture) {
            if (!this.postProcessExtraSampler) {
                this.postProcessExtraSampler = device.createSampler({
                    magFilter: 'linear',
                    minFilter: 'linear',
                });
            }
            bindGroupEntries.push({ binding: 3, resource: this.postProcessExtraTexture.createView() });
            bindGroupEntries.push({ binding: 4, resource: this.postProcessExtraSampler });
        }
        if (this.postProcessUsesUniform && this.postProcessUniformBuffer) {
            bindGroupEntries.push({ binding: 5, resource: { buffer: this.postProcessUniformBuffer } });
        }

        const bindGroup = device.createBindGroup({
            layout: this.postProcessBindGroupLayout!,
            entries: bindGroupEntries,
        });

        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: targetView,
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                },
            ],
        });

        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(6, 1, 0, 0);
        pass.end();
    }

    private getViewport(): [number, number] {
        const dbSize = new THREE.Vector2();
        (this.threeRenderer as any).getDrawingBufferSize?.(dbSize);
        const width = dbSize.x || (this.threeRenderer.domElement?.width ?? 0) || (this.threeRenderer.getSize(new THREE.Vector2()).x);
        const height = dbSize.y || (this.threeRenderer.domElement?.height ?? 0) || (this.threeRenderer.getSize(new THREE.Vector2()).y);
        return [width, height];
    }

    private convertCamera(camera: THREE.Camera, renderer: THREE.WebGPURenderer): PerspectiveCamera {
        const viewport = this.getViewport();
        const adapter = new CameraAdapter();
        // Use unified adapter API (mirrors DirectCameraAdapter)
        adapter.update(camera as any, viewport);
        return adapter as unknown as PerspectiveCamera;
    }

        /** 
     * Update all dynamic models with current camera and time
     */
        public async updateDynamicModels(camera: THREE.Camera, time?: number): Promise<void> {
            // 使用 CameraAdapter 获取转换后的矩阵（与渲染管线一致）
            const adapter = new CameraAdapter();
            const viewport = this.getViewport();
            adapter.update(camera as any, viewport);
            
            const viewMatrix = adapter.viewMatrix();
            const projMatrix = adapter.projMatrix();
    
            // Update each model with transformed matrices
            // 串行更新，确保每个模型的推理完成后再更新下一个
            for (const model of this.gaussianModels) {
                try {
                    await model.update(viewMatrix as any, time, projMatrix as any);
                } catch (error) {
                    console.warn(`Failed to update model:`, error);
                }
            }
        }

    // ==================== 3DGS Parameters Management ====================
    
    /**
     * Set Gaussian scaling for a specific model
     */
    public setModelGaussianScale(modelId: string, scale: number): void {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            this.gaussianModels[modelIndex].setGaussianScale(scale);
            console.log(`[GaussianThreeJSRenderer] Model ${modelId} Gaussian scale set to: ${scale}`);
        } else {
            console.warn(`[GaussianThreeJSRenderer] Model ${modelId} not found`);
        }
    }
    
    /**
     * Get Gaussian scale for a specific model
     */
    public getModelGaussianScale(modelId: string): number {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            return this.gaussianModels[modelIndex].getGaussianScale();
        }
        return 1.0; // default value
    }

    /** Get model visibility */
    public getModelVisible(modelId: string): boolean {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            return this.gaussianModels[modelIndex].getModelVisible();
        }
        return false;
    }

    /**
     * Set maximum spherical harmonics degree for a specific model
     */
    public setModelMaxShDeg(modelId: string, deg: number): void {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            this.gaussianModels[modelIndex].setMaxShDeg(deg);
            console.log(`[GaussianThreeJSRenderer] Model ${modelId} Max SH degree set to: ${deg}`);
        } else {
            console.warn(`[GaussianThreeJSRenderer] Model ${modelId} not found`);
        }
    }

    /** Get max SH degree for a specific model */
    public getModelMaxShDeg(modelId: string): number {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            return this.gaussianModels[modelIndex].getMaxShDeg();
        }
        return 0;
    }

    /**
     * Set kernel size for a specific model
     */
    public setModelKernelSize(modelId: string, size: number): void {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            this.gaussianModels[modelIndex].setKernelSize(size);
            console.log(`[GaussianThreeJSRenderer] Model ${modelId} Kernel size set to: ${size}`);
        } else {
            console.warn(`[GaussianThreeJSRenderer] Model ${modelId} not found`);
        }
    }

    /** Get kernel size for a specific model */
    public getModelKernelSize(modelId: string): number {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            return this.gaussianModels[modelIndex].getKernelSize();
        }
        return 0;
    }

    /**
     * Set opacity scale for a specific model
     */
    public setModelOpacityScale(modelId: string, scale: number): void {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            this.gaussianModels[modelIndex].setOpacityScale(scale);
            console.log(`[GaussianThreeJSRenderer] Model ${modelId} Opacity scale set to: ${scale}`);
        } else {
            console.warn(`[GaussianThreeJSRenderer] Model ${modelId} not found`);
        }
    }

    /** Get opacity scale for a specific model */
    public getModelOpacityScale(modelId: string): number {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            return this.gaussianModels[modelIndex].getOpacityScale();
        }
        return 1.0;
    }

    /**
     * Set cutoff scale for a specific model
     */
    public setModelCutoffScale(modelId: string, scale: number): void {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            this.gaussianModels[modelIndex].setCutoffScale(scale);
            console.log(`[GaussianThreeJSRenderer] Model ${modelId} Cutoff scale set to: ${scale}`);
        } else {
            console.warn(`[GaussianThreeJSRenderer] Model ${modelId} not found`);
        }
    }

    /** Get cutoff scale for a specific model */
    public getModelCutoffScale(modelId: string): number {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            return this.gaussianModels[modelIndex].getCutoffScale();
        }
        return 1.0;
    }

    /**
     * Set time scale for a specific model
     */
    public setModelTimeScale(modelId: string, scale: number): void {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            this.gaussianModels[modelIndex].setTimeScale(scale);
            console.log(`[GaussianThreeJSRenderer] Model ${modelId} Time scale set to: ${scale}`);
        } else {
            console.warn(`[GaussianThreeJSRenderer] Model ${modelId} not found`);
        }
    }

    /**
     * Set time offset for a specific model
     */
    public setModelTimeOffset(modelId: string, offset: number): void {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            this.gaussianModels[modelIndex].setTimeOffset(offset);
            console.log(`[GaussianThreeJSRenderer] Model ${modelId} Time offset set to: ${offset}`);
        } else {
            console.warn(`[GaussianThreeJSRenderer] Model ${modelId} not found`);
        }
    }

    /**
     * Set time offset for a specific model
     */
    public setModelAnimationIsLoop(modelId: string, is_loop: boolean): void {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            this.gaussianModels[modelIndex].setAnimationIsLoop(is_loop);
            console.log(`[GaussianThreeJSRenderer] Model ${modelId} Animation is loop set to: ${is_loop}`);
        } else {
            console.warn(`[GaussianThreeJSRenderer] Model ${modelId} not found`);
        }
    }


    /**
     * Set time update mode for a specific model
     */
    public setModelTimeUpdateMode(modelId: string, mode: any): void {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            this.gaussianModels[modelIndex].setTimeUpdateMode(mode);
            console.log(`[GaussianThreeJSRenderer] Model ${modelId} Time update mode set to: ${mode}`);
        } else {
            console.warn(`[GaussianThreeJSRenderer] Model ${modelId} not found`);
        }
    }

    /**
     * Set render mode for a specific model
     */
    public setModelRenderMode(modelId: string, mode: number): void {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            this.gaussianModels[modelIndex].setRenderMode(mode);
            console.log(`[GaussianThreeJSRenderer] Model ${modelId} Render mode set to: ${mode}`);
        } else {
            console.warn(`[GaussianThreeJSRenderer] Model ${modelId} not found`);
        }
    }

    /** Get render mode for a specific model */
    public getModelRenderMode(modelId: string): number {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            return this.gaussianModels[modelIndex].getRenderMode();
        }
        return 0;
    }

    /**
     * Start animation for a specific model
     */
    public startModelAnimation(modelId: string, speed: number = 1.0): void {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            this.gaussianModels[modelIndex].startAnimation(speed);
            console.log(`[GaussianThreeJSRenderer] Model ${modelId} Animation started at ${speed}x speed`);
        } else {
            console.warn(`[GaussianThreeJSRenderer] Model ${modelId} not found`);
        }
    }

    /**
     * Pause animation for a specific model
     */
    public pauseModelAnimation(modelId: string): void {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            this.gaussianModels[modelIndex].pauseAnimation();
            console.log(`[GaussianThreeJSRenderer] Model ${modelId} Animation paused`);
        } else {
            console.warn(`[GaussianThreeJSRenderer] Model ${modelId} not found`);
        }
    }

    /**
     * Resume animation for a specific model
     */
    public resumeModelAnimation(modelId: string): void {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            this.gaussianModels[modelIndex].resumeAnimation();
            console.log(`[GaussianThreeJSRenderer] Model ${modelId} Animation resumed`);
        } else {
            console.warn(`[GaussianThreeJSRenderer] Model ${modelId} not found`);
        }
    }

    /**
     * Stop animation for a specific model
     */
    public stopModelAnimation(modelId: string): void {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            this.gaussianModels[modelIndex].stopAnimation();
            console.log(`[GaussianThreeJSRenderer] Model ${modelId} Animation stopped`);
        } else {
            console.warn(`[GaussianThreeJSRenderer] Model ${modelId} not found`);
        }
    }

    /**
     * Set animation time for a specific model
     */
    public setModelAnimationTime(modelId: string, time: number): void {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            this.gaussianModels[modelIndex].setAnimationTime(time);
            console.log(`[GaussianThreeJSRenderer] Model ${modelId} Animation time set to: ${time.toFixed(3)}s`);
        } else {
            console.warn(`[GaussianThreeJSRenderer] Model ${modelId} not found`);
        }
    }

    /**
     * Set animation speed for a specific model
     */
    public setModelAnimationSpeed(modelId: string, speed: number): void {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            this.gaussianModels[modelIndex].setAnimationSpeed(speed);
            console.log(`[GaussianThreeJSRenderer] Model ${modelId} Animation speed set to: ${speed}x`);
        } else {
            console.warn(`[GaussianThreeJSRenderer] Model ${modelId} not found`);
        }
    }
    
    /**
     * Get all model parameters
     */
    public getModelParams(): any {
        const result: any = {
            models: {}
        };
        
        this.gaussianModels.forEach((model, index) => {
            const modelId = `model_${index}`;
            result.models[modelId] = {
                id: modelId,
                name: model.name,
                visible: model.getModelVisible(),
                gaussianScale: model.getGaussianScale(),
                maxShDeg: model.getMaxShDeg(),
                kernelSize: model.getKernelSize(),
                opacityScale: model.getOpacityScale(),
                cutoffScale: model.getCutoffScale(),
                timeScale: model.getTimeScale(),
                timeOffset: model.getTimeOffset(),
                timeUpdateMode: model.getTimeUpdateMode(),
                animationSpeed: model.getAnimationSpeed(),
                isAnimationRunning: model.isAnimationRunning(),
                isAnimationPaused: model.isAnimationPaused()
            };
        });
        
        return result;
    }
    
    /**
     * 获取所有高斯模型
     * @returns 高斯模型数组的副本
     */
    public getGaussianModels(): GaussianModel[] {
        return [...this.gaussianModels]; // 返回副本以防止外部修改
    }

    /**
     * Append a GaussianModel at runtime
     */
    public appendGaussianModel(model: GaussianModel): void {
        this.gaussianModels.push(model);
    }

    /**
     * Remove a GaussianModel by id (e.g., 'model_2')
     * Returns true if removed
     */
    public removeModelById(modelId: string): boolean {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (isNaN(modelIndex) || modelIndex < 0 || modelIndex >= this.gaussianModels.length) {
            console.warn(`[GaussianThreeJSRenderer] removeModelById: invalid id ${modelId}`);
            return false;
        }
        const model = this.gaussianModels[modelIndex];
        try {
            // Remove from scene graph
            this.threeScene.remove(model);
            // Optional: dispose hooks
            model.dispose?.();
        } catch (e) {
            console.warn('[GaussianThreeJSRenderer] Error removing model from scene:', e);
        }
        // Remove from internal list
        this.gaussianModels.splice(modelIndex, 1);
        console.log(`[GaussianThreeJSRenderer] Removed ${modelId} (${model.name})`);
        return true;
    }

    /**
     * Set visibility for a specific model
     */
    public setModelVisible(modelId: string, visible: boolean): void {
        const modelIndex = parseInt(modelId.replace('model_', ''));
        if (modelIndex >= 0 && modelIndex < this.gaussianModels.length) {
            this.gaussianModels[modelIndex].setModelVisible(visible);
            console.log(`[GaussianThreeJSRenderer] Model ${modelId} visible: ${visible}`);
        } else {
            console.warn(`[GaussianThreeJSRenderer] Model ${modelId} not found`);
        }
    }

    /**
     * Reset all parameters to defaults
     */
    public resetParameters(): void {
        this.gaussianModels.forEach(model => {
            model.setGaussianScale(1.0);
            model.setMaxShDeg(3);
            model.setKernelSize(0.1);
            model.setOpacityScale(1.0);
            model.setCutoffScale(1.0);
            model.setTimeScale(1.0);
            model.setTimeOffset(0.0);
            model.setTimeUpdateMode('fixed_delta');
        });
        console.log('[GaussianThreeJSRenderer] All parameters reset to defaults');
    }

    // ==================== Global Animation Controls ====================

    /** Set time scale for all models */
    public setGlobalTimeScale(scale: number): void {
        this.gaussianModels.forEach(model => model.setTimeScale(scale));
        console.log(`[GaussianThreeJSRenderer] Global time scale set: ${scale}`);
    }

    /** Set time offset for all models */
    public setGlobalTimeOffset(offset: number): void {
        this.gaussianModels.forEach(model => model.setTimeOffset(offset));
        console.log(`[GaussianThreeJSRenderer] Global time offset set: ${offset}`);
    }

    /** Set time update mode for all models */
    public setGlobalTimeUpdateMode(mode: any): void {
        this.gaussianModels.forEach(model => model.setTimeUpdateMode(mode));
        console.log(`[GaussianThreeJSRenderer] Global time update mode: ${mode}`);
    }

    /** Start animation on all models */
    public startAllAnimations(speed: number = 1.0): void {
        this.gaussianModels.forEach(model => model.startAnimation(speed));
        console.log(`[GaussianThreeJSRenderer] All animations started at ${speed}x`);
    }

    /** Pause animation on all models */
    public pauseAllAnimations(): void {
        this.gaussianModels.forEach(model => model.pauseAnimation());
        console.log(`[GaussianThreeJSRenderer] All animations paused`);
    }

    /** Resume animation on all models */
    public resumeAllAnimations(): void {
        this.gaussianModels.forEach(model => model.resumeAnimation());
        console.log(`[GaussianThreeJSRenderer] All animations resumed`);
    }

    /** Stop animation on all models */
    public stopAllAnimations(): void {
        this.gaussianModels.forEach(model => model.stopAnimation());
        console.log(`[GaussianThreeJSRenderer] All animations stopped`);
    }

    /** Set animation time for all models */
    public setAllAnimationTime(time: number): void {
        this.gaussianModels.forEach(model => model.setAnimationTime(time));
        console.log(`[GaussianThreeJSRenderer] Global animation time set: ${time.toFixed(3)}s`);
    }

    /** Set animation speed for all models */
    public setAllAnimationSpeed(speed: number): void {
        this.gaussianModels.forEach(model => model.setAnimationSpeed(speed));
        console.log(`[GaussianThreeJSRenderer] Global animation speed set: ${speed}x`);
    }
}
