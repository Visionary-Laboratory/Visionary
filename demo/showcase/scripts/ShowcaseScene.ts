import * as THREE from 'three/webgpu';
import { initThreeContext } from '@/app/three-context';
import { loadUnifiedModel, LoadResult } from '@/app/unified-model-loader';
import { getDefaultOrtWasmPaths, initOrtEnvironment } from '@/config/ort-config';
import { CameraViewConfig, EasingName, LocalTransform } from './types';
import { SCENE1_CAROUSEL_ITEMS, getScene2CameraViews, SCENE3_MULTI_ONNX_CONFIGS, SCENE3_CAMERA_BASE, SCENE3_CAMERA_SWING } from './sceneConfigs';

// Easing Functions Type Definition
type EasingFunction = (t: number) => number;

// Easing Functions Collection
const EasingFunctions: Record<EasingName, EasingFunction> = {
    linear: (t: number) => t,
    easeInQuad: (t: number) => t * t,
    easeOutQuad: (t: number) => t * (2 - t),
    easeInOutQuad: (t: number) => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
    easeInCubic: (t: number) => t * t * t,
    easeOutCubic: (t: number) => (--t) * t * t + 1,
    easeInOutCubic: (t: number) => t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1
};

interface AnimationSegment {
    startPos: THREE.Vector3;
    endPos: THREE.Vector3;
    startTarget: THREE.Vector3;
    endTarget: THREE.Vector3;
    duration: number;
    easing: EasingFunction;
    type: 'transition' | 'idle';
}

export class ShowcaseScene {
    container: HTMLElement;
    renderer: THREE.WebGPURenderer | null = null;
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    
    // State
    models: THREE.Object3D[] = [];
    currentSceneIndex: number = 1;
    
    // Carousel Configuration
    carouselSpeed: number = 2.0;      // Units per second
    carouselRotationSpeed: number = 0.5; // Radians per second
    carouselSpacing: number = 4.0;    // Distance between objects
    carouselTotalWidth: number = 0;   // Calculated at runtime
    carouselItemTransforms: LocalTransform[] = [];
    scene3CameraBasePos: THREE.Vector3 | null = null;
    scene3CameraBaseTarget: THREE.Vector3 | null = null;
    scene3SwingAmplitude = SCENE3_CAMERA_SWING.amplitude;
    scene3SwingFrequency = SCENE3_CAMERA_SWING.frequency;
    scene3TimeScale = SCENE3_CAMERA_SWING.timeScale;
    
    // Scene 2 Sequence State
    animationSegments: AnimationSegment[] = [];
    currentSegmentIndex: number = 0;
    segmentStartTime: number = 0;
    
    // Renderer specific (from visionaryScene.ts reference)
    gaussianThreeJSRenderer: any = null;
    startTime: number = Date.now();
    
    // Animation
    clock: THREE.Clock;

    // Shared axes/vectors
    private readonly worldYAxis = new THREE.Vector3(0, 1, 0);

    constructor(containerId: string) {
        this.container = document.getElementById(containerId)!;
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.clock = new THREE.Clock();
        
        // Initial camera position
        this.camera.position.set(0, 0, 5);
        
        // Handle Resize
        window.addEventListener('resize', this.onResize.bind(this));
    }
    
    async init() {
        // 1. Init ORT
        const wasmPaths = getDefaultOrtWasmPaths();
        initOrtEnvironment(wasmPaths);
        
        // 2. Create Canvas & Init WebGPU Renderer
        const canvas = document.createElement('canvas');
        canvas.style.display = 'block';
        canvas.style.width = '100%';
        canvas.style.height = '100%';
        this.container.appendChild(canvas);
        
        this.renderer = await initThreeContext(canvas);
        
        if (!this.renderer) {
            console.error('Failed to initialize WebGPU renderer');
            return;
        }

        // Set clear color to transparent to let CSS background show through
        // This fixes the "grayish" look from the default context initialization
        this.renderer.setClearColor(0x000000, 0);
        
        // Setup Lighting (optional, but good for 3D models)
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);
        const dirLight = new THREE.DirectionalLight(0xffffff, 1);
        dirLight.position.set(5, 5, 5);
        this.scene.add(dirLight);

        // Start Loop
        this.renderer.setAnimationLoop(this.animate.bind(this));
        
        await this.loadScene1();
    }
    
    clearModels() {
        // Remove existing models from scene
        this.models.forEach(m => {
            this.scene.remove(m);
            // Dispose geometries/materials if standard Mesh
            if ((m as any).geometry) (m as any).geometry.dispose();
            // Gaussian Splatting disposal might be handled by renderer/loader, 
            // but for simple switching we just remove from scene graph.
        });
        this.models = [];
        this.carouselItemTransforms = [];
        
        // Also need to clear any specific GaussianRenderers if added to scene separately?
        // UnifiedLoader adds `gaussianRenderer` to scene.
        // We should find it and remove it.
        const childrenToRemove: THREE.Object3D[] = [];
        this.scene.traverse((child) => {
            if (child.constructor.name === 'GaussianThreeJSRenderer' || 
                (child as any).isGaussianSplatting) {
                childrenToRemove.push(child);
            }
        });
        childrenToRemove.forEach(c => this.scene.remove(c));

        // Reset the renderer reference
        if (this.gaussianThreeJSRenderer) {
            // If it has dispose method, call it (assuming it might have)
            if (typeof this.gaussianThreeJSRenderer.dispose === 'function') {
                this.gaussianThreeJSRenderer.dispose();
            }
            this.gaussianThreeJSRenderer = null;
        }
    }
    
    private handleLoadResult(result: LoadResult | LoadResult[]) {
        const results = Array.isArray(result) ? result : [result];
        
        for (const singleResult of results) {
             // Handle models
             if (singleResult.models && singleResult.models.length > 0) {
                 const mainModel = singleResult.models[0];
                 this.models.push(mainModel);
             }

            // Handle Gaussian Renderer Logic (Reference from visionaryScene.ts)
            if (singleResult.gaussianRenderer) {
                // If existing renderer, try to append
                if (this.gaussianThreeJSRenderer && singleResult.models.length > 0) {
                    const newModel = singleResult.models.find((m: any) => m.isObject3D);
                    if (newModel && this.gaussianThreeJSRenderer.appendGaussianModel) {
                        this.gaussianThreeJSRenderer.appendGaussianModel(newModel);
                        console.log('[Showcase] Added new model to existing GaussianThreeJSRenderer');
                    } else {
                         // No append method, replace
                        this.gaussianThreeJSRenderer = singleResult.gaussianRenderer;
                    }
                } else {
                    // No existing renderer, use new one
                    this.gaussianThreeJSRenderer = singleResult.gaussianRenderer;
                }
            }
        }
    }

    private applyLocalRotation(model: THREE.Object3D | undefined, transform?: LocalTransform) {
        if (!model || !transform?.rotation) return;
        const { rotation } = transform;
        if (rotation.x) model.rotateX(rotation.x);
        if (rotation.y) model.rotateY(rotation.y);
        if (rotation.z) model.rotateZ(rotation.z);
    }

    private applyImmediateTransform(model: THREE.Object3D | undefined, transform?: LocalTransform) {
        if (!model || !transform) return;
        if (transform.position) {
            model.position.add(transform.position);
        }
        this.applyLocalRotation(model, transform);
    }

    private registerCarouselModel(model: THREE.Object3D | undefined, transform?: LocalTransform) {
        this.carouselItemTransforms.push(transform ?? {});
        this.applyLocalRotation(model, transform);
    }

    async loadScene1() {
        this.clearModels();
        
        const carouselItems = SCENE1_CAROUSEL_ITEMS;

        this.carouselSpacing = 5.0;
        this.carouselSpeed = 3.0;
        this.carouselRotationSpeed = 0.8;
        this.carouselTotalWidth = carouselItems.length * this.carouselSpacing;

        // Position camera for Scene 1
        this.camera.position.set(0, 2, 8);
        this.camera.lookAt(0, 0, 0);

        // Load Items sequentially to maintain order
        try {
            for (let i = 0; i < carouselItems.length; i++) {
                const config = carouselItems[i];
                
                if (config.type === 'mesh') {
                    // Create Standard Mesh
                    const mesh = new THREE.Mesh(config.geometry, config.material);
                    mesh.castShadow = true;
                    mesh.receiveShadow = true;
                    this.scene.add(mesh);
                    this.models.push(mesh);
                    this.registerCarouselModel(mesh, config.transform);
                } else if (config.type === 'file') {
                    // Load Model
                    try {
                        const result = await loadUnifiedModel(this.renderer!, this.scene, config.url!, config.loadOptions);
                        this.handleLoadResult(result);
                        
                        // Apply specific scale if needed for the just-loaded model
                        const loadedModel = this.models[this.models.length - 1];
                        if (loadedModel && config.scale) {
                            loadedModel.scale.setScalar(config.scale);
                        }
                        this.registerCarouselModel(loadedModel, config.transform);
                    } catch (err) {
                        console.error(`Failed to load carousel item: ${config.url}`, err);
                        // Add a fallback mesh placeholder to keep layout intact
                        const fallback = new THREE.Mesh(
                            new THREE.BoxGeometry(1, 1, 1), 
                            new THREE.MeshBasicMaterial({ color: 0x555555, wireframe: true })
                        );
                        this.scene.add(fallback);
                        this.models.push(fallback);
                        this.registerCarouselModel(fallback, config.transform);
                    }
                }
            }
            
            // Initial Layout
            this.layoutCarousel();
            
        } catch (e) {
            console.error('Error constructing carousel', e);
        }
    }

    /**
     * Positions models in a line based on spacing
     */
    layoutCarousel() {
        // Start centering: offset so the group is somewhat centered or starting from right
        // Strategy: Place them starting from x=0 going right, or centered.
        // For infinite scroll left, we just need them spaced.
        this.models.forEach((model, index) => {
            model.position.set(index * this.carouselSpacing, 0, 0);
            const transform = this.carouselItemTransforms[index];
            if (transform?.position) {
                model.position.add(transform.position);
            }
        });
    }

    /**
     * Generate animation segments from view configurations
     */
    private generateSegments(views: CameraViewConfig[]) {
        this.animationSegments = [];
        if (views.length === 0) return;

        // Helper to get view cyclic
        const getView = (i: number) => views[i % views.length];

        // We need to calculate exact start/end positions for consistency
        // Let's assume we start at View 0's base position.
        
        let currentPos = getView(0).position.clone();
        
        // Loop through views to create segments
        // Sequence: 
        // 1. Idle at View 0 (Start -> End)
        // 2. Trans View 0 End -> View 1 Start
        // 3. Idle at View 1 (Start -> End)
        // ...
        
        for (let i = 0; i < views.length; i++) {
            const currentView = getView(i);
            const nextView = getView(i + 1);
            
            // --- 1. Idle Segment (Micro-movement at current view) ---
            const idleStartPos = currentPos.clone(); // Start where we arrived
            const idleStartTarget = currentView.target.clone();
            
            // Calculate Idle End Position based on type
            let idleEndPos = idleStartPos.clone();
            
            if (currentView.idleType === 'orbit') {
                // Orbit Logic with Composition Offset correction
                // We want to orbit around the OBJECT CENTER (0,0,0), even if the camera is shifted for composition.
                
                // 1. Recover the "un-shifted" position relative to the object center
                // Assuming the shift is constant `focusOffset` used in config, but we can infer it.
                // Let's assume rotation is always around WORLD Y axis passing through (0,0,0).
                
                // Current Camera Pos
                const startPos = idleStartPos.clone();
                
                // Pivot is World Origin (Object Center)
                const pivot = new THREE.Vector3(0, 0, 0);
                
                // Vector from Pivot to Camera
                const offsetFromPivot = new THREE.Vector3().subVectors(startPos, pivot);
                
                // Rotate this vector
                const angle = currentView.idleSpeed * currentView.idleDuration;
                offsetFromPivot.applyAxisAngle(new THREE.Vector3(0, 1, 0), angle);
                
                // New Camera Pos = Pivot + Rotated Vector
                idleEndPos = new THREE.Vector3().addVectors(pivot, offsetFromPivot);
            }
            // Add 'static' or other types here (end = start)

            this.animationSegments.push({
                type: 'idle',
                startPos: idleStartPos,
                endPos: idleEndPos,
                startTarget: idleStartTarget,
                endTarget: idleStartTarget, // Assuming target stays same during idle
                duration: currentView.idleDuration,
                easing: EasingFunctions.linear // Idle usually linear
            });
            
            // Update currentPos for next step
            currentPos = idleEndPos;
            
            // --- 2. Transition Segment (To Next View) ---
            // Target Position is the Next View's BASE position (before its idle starts)
            const transEndPos = nextView.position.clone();
            const transEndTarget = nextView.target.clone();
            
            this.animationSegments.push({
                type: 'transition',
                startPos: currentPos.clone(),
                endPos: transEndPos,
                startTarget: idleStartTarget.clone(),
                endTarget: transEndTarget,
                duration: nextView.transitionDuration,
                easing: EasingFunctions[nextView.transitionEasing] || EasingFunctions.easeInOutQuad
            });
            
            // Update currentPos
            currentPos = transEndPos;
        }
    }

    async loadScene2() {
        this.clearModels();
        
        // Load ONNX 4DGS
        const onnxUrl = '/models/qiewu/gaussianA.onnx';
        
        try {
            const result = await loadUnifiedModel(this.renderer!, this.scene, onnxUrl, {
                type: 'onnx',
                name: '4DGS Demo',
            });
            this.handleLoadResult(result);
        } catch (e) {
            console.warn('Failed to load ONNX, falling back to PLY', e);
             try {
                const result = await loadUnifiedModel(this.renderer!, this.scene, '/models/test.ply', {
                    isGaussian: true
                });
                this.handleLoadResult(result);
             } catch(e2) {
                 console.error("Fallback failed", e2);
             }
        }
        
        const viewConfigs = getScene2CameraViews();
        this.generateSegments(viewConfigs);
        
        // Initialize Sequence State
        this.currentSegmentIndex = 0;
        this.segmentStartTime = this.clock.getElapsedTime();
        
        // Set Initial Position
        if (this.animationSegments.length > 0) {
            const firstSeg = this.animationSegments[0];
            this.camera.position.copy(firstSeg.startPos);
            this.camera.lookAt(firstSeg.startTarget);
        }
    }
    
    async loadScene3() {
        this.clearModels();

        // Scene 3 is static: no camera animation segments
        this.animationSegments = [];
        this.currentSegmentIndex = 0;
        this.segmentStartTime = this.clock.getElapsedTime();

        // Static camera framing baseline
        this.scene3CameraBasePos = SCENE3_CAMERA_BASE.position.clone();
        this.scene3CameraBaseTarget = SCENE3_CAMERA_BASE.target.clone();
        this.camera.position.copy(this.scene3CameraBasePos);
        this.camera.lookAt(this.scene3CameraBaseTarget);

        const multiOnnxConfigs = SCENE3_MULTI_ONNX_CONFIGS;

        const spacing = 4.5;
        const startX = -((multiOnnxConfigs.length - 1) * spacing) / 2;

        for (let i = 0; i < multiOnnxConfigs.length; i++) {
            const config = multiOnnxConfigs[i];
            try {
                const result = await loadUnifiedModel(this.renderer!, this.scene, config.url!, config.loadOptions);
                this.handleLoadResult(result);

                const loadedModel = this.models[this.models.length - 1];
                if (!loadedModel) continue;

                if (config.scale) {
                    loadedModel.scale.setScalar(config.scale);
                }

                loadedModel.position.set(startX + i * spacing, 0, 0);
                this.applyImmediateTransform(loadedModel, config.transform);
            } catch (err) {
                console.error(`Failed to load Scene 3 ONNX: ${config.url}`, err);

                const fallback = new THREE.Mesh(
                    new THREE.BoxGeometry(1, 1, 1),
                    new THREE.MeshBasicMaterial({ color: 0x555555, wireframe: true })
                );
                fallback.position.set(startX + i * spacing, 0, 0);
                this.applyImmediateTransform(fallback, config.transform);
                this.scene.add(fallback);
                this.models.push(fallback);
            }
        }
    }
    
    animate() {
        const delta = this.clock.getDelta();
        const time = this.clock.getElapsedTime();
        let timeScale = 1000;
        
        if (this.currentSceneIndex === 1) {
            // Scene 1: Infinite Carousel
            const moveDist = this.carouselSpeed * delta;
            const wrapThreshold = -10;
            
            this.models.forEach((model) => {
                model.position.x -= moveDist;
                
                if (model.position.x < wrapThreshold) {
                    model.position.x += this.carouselTotalWidth;
                }
                
                // 3. 自转：使用世界坐标系的垂直轴，确保不同坐标系的模型都围绕同一轴旋转
                model.rotateOnWorldAxis(this.worldYAxis, this.carouselRotationSpeed * delta);
            });

            timeScale = 1000;
            
        } else if (this.currentSceneIndex === 2) {
            // Scene 3: Static multi-ONNX with subtle sway
            timeScale = this.scene3TimeScale;
            if (this.scene3CameraBasePos && this.scene3CameraBaseTarget) {
                const swingOffset = Math.sin(time * this.scene3SwingFrequency) * this.scene3SwingAmplitude;
                this.camera.position.copy(this.scene3CameraBasePos);
                this.camera.position.x += swingOffset;

                const swingTarget = this.scene3CameraBaseTarget.clone();
                swingTarget.x += swingOffset * 0.3;
                this.camera.lookAt(swingTarget);
            }
        } else {
            // Scene 2: Unified Animation Sequence (legacy)
            if (this.animationSegments.length > 0) {
                const currentSegment = this.animationSegments[this.currentSegmentIndex];
                
                const elapsed = time - this.segmentStartTime;
                const progress = Math.min(elapsed / currentSegment.duration, 1.0);
                
                const t = currentSegment.easing(progress);
                
                this.camera.position.lerpVectors(currentSegment.startPos, currentSegment.endPos, t);
                
                const currentTarget = new THREE.Vector3().lerpVectors(currentSegment.startTarget, currentSegment.endTarget, t);
                this.camera.lookAt(currentTarget);
                
                if (progress >= 1.0) {
                    this.currentSegmentIndex = (this.currentSegmentIndex + 1) % this.animationSegments.length;
                    this.segmentStartTime = time;
                }
            }

            timeScale = 15000;
        }
        
        // Render Logic (Reference from visionaryScene.ts)
        if (this.renderer && this.gaussianThreeJSRenderer) {
            const currentTime = (Date.now() - this.startTime) / timeScale;
            this.gaussianThreeJSRenderer.updateDynamicModels(this.camera, currentTime);
            
            this.gaussianThreeJSRenderer.renderThreeScene(this.camera);
            this.gaussianThreeJSRenderer.drawSplats(this.renderer, this.scene, this.camera);
        } else if (this.renderer) {
            this.renderer.render(this.scene, this.camera);
        }
    }
    
    onResize() {
        if (!this.renderer || !this.camera) return;
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
    
    switchToScene(index: number) {
        if (index === this.currentSceneIndex) return;

        if (index === 1) {
            this.currentSceneIndex = 1;
            this.loadScene1();
        } else if (index === 2) {
            this.currentSceneIndex = 2;
            this.loadScene3();
        } else if (index === 3) {
            this.currentSceneIndex = 3;
            this.loadScene2();
        }
    }
}

