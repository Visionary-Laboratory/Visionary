import * as THREE from "three/webgpu";
import { GaussianThreeJSRenderer } from "../app/GaussianThreeJSRenderer";
import { GaussianModel } from "../app/GaussianModel";
import { RendererInitHelper } from "../utils/renderer-init-helper";

// 录制相机类 - 独立的录制相机对象
export class RecordingCamera {
    private parentNodeId: string;
    private domId: string;
    private isPreviewShow: boolean = false;
    public camera: THREE.PerspectiveCamera;
    public canvas!: HTMLCanvasElement;
    public overlayContainer!: HTMLElement;
    public renderer: THREE.WebGPURenderer | null = null;
    private gaussianRenderer: GaussianThreeJSRenderer | null = null;
    private gizmo: THREE.Group | null = null; // 相机视锥gizmo
    private gizmoVisible: boolean = true; // gizmo可见性
    private gizmoColor: number = 0x00ff00; // gizmo颜色
    private gizmoLength: number = 5; // gizmo长度（默认10单位）
    private width: number;
    private height: number;
    private showPreview: boolean;
    private sceneWrapper: THREE.Group | null = null;
    private tempPosition: THREE.Vector3 = new THREE.Vector3();
    private tempQuaternion: THREE.Quaternion = new THREE.Quaternion();
    private tempScale: THREE.Vector3 = new THREE.Vector3();
    private editorHelperVisibilityCache: Array<{ object: THREE.Object3D, visible: boolean }> = [];
    // 环境贴图相关成员
    private recordingEnvMap: THREE.Texture | null = null; // 录制渲染器专用的环境贴图
    private recordingBackground: THREE.Texture | null = null; // 录制渲染器专用的背景
    private originalEnvMap: THREE.Texture | null = null; // 保存原始环境贴图
    private originalBackground: THREE.Texture | THREE.Color | null = null; // 保存原始背景
    private skyboxEnabled: boolean = true; // 是否在录制时替换天空盒

    private statusDom: HTMLElement | null = null;
    private cameraInfoDom: HTMLElement | null = null;
    private titleDom: HTMLElement | null = null;
    private cameraName: string = '';
    private isSync: boolean = false;

    constructor(id: string, width: number = 1920, height: number = 1080, fov: number = 55, showPreview: boolean = true, cameraName: string = '', isSync: boolean = false) {
        this.parentNodeId = id;
        this.domId = `recordingOverlay_${this.parentNodeId}`;
        // 创建独立的录制相机
        this.cameraName = cameraName;
        this.camera = new THREE.PerspectiveCamera(fov, width / height, 0.1, 1000);
        this.width = width;
        this.height = height;
        this.showPreview = showPreview;
        this.isSync = isSync;
        const overlayDom = document.getElementById(this.domId);
        if (overlayDom) {
            this.overlayContainer = overlayDom;
            this.canvas = this.overlayContainer.querySelector('canvas') as HTMLCanvasElement;
            console.log("overlayContainer----", this.overlayContainer, this.canvas);
            this.titleDom = this.overlayContainer.querySelector('.recording-overlay-title-name') as HTMLElement;
        }
    }

    public isInitialized(): boolean {
        return !!this.renderer;
    }

    public ensurePreviewWindow(currLang: string = 'zh') {
        if (this.overlayContainer && document.body.contains(this.overlayContainer)) {
            // this.overlayContainer.style.display = 'block';
            this.overlayContainer.style.visibility = 'visible';
            this.overlayContainer.style.opacity = '1';
            this.showPreview = true;
            if(!this.canvas) { // 如果canvas不存在，则重新获取
                this.canvas = this.overlayContainer.querySelector('canvas') as HTMLCanvasElement;
            }
            if(!this.titleDom) {
                this.titleDom = this.overlayContainer.querySelector('.recording-overlay-title-name') as HTMLElement;
            }
            console.log('[RecordingCamera] 预览窗口已存在，显示窗口');
            return;
        }

        this.showPreview = true;
        console.log('[RecordingCamera] 创建新的预览窗口');
        this.createOverlayCanvas(this.width, this.height,currLang);

        // 确保窗口创建后立即可见
        if (this.overlayContainer) {
            this.overlayContainer.style.display = 'none'; // 创建后隐藏窗口
            this.overlayContainer.style.visibility = 'visible';
            this.overlayContainer.style.opacity = '1';
        }
    }

    public hidePreviewWindow() {
        if (this.overlayContainer) {
            this.overlayContainer.style.display = 'none';
            this.isPreviewShow = false;
        }
    }
    public showPreviewWindow(skyboxEnabled: boolean = true) {
        if (this.titleDom) {
            this.titleDom.textContent = ` - ${this.cameraName}`; // 多语言
        }
        if (this.overlayContainer) {
            this.overlayContainer.style.display = 'block';
            this.isPreviewShow = true;
        }
        this.setSkyboxEnabled(skyboxEnabled); 
    }
    public setCameraName(name: string) {
        // console.log("设置名字------set CameraName----", name);
        this.cameraName = name;
        if (this.titleDom) {
            this.titleDom.textContent = ` - ${this.cameraName}`; // 多语言
        }
    }

    /**
     * 开关录制时的天空盒替换
     */
    public setSkyboxEnabled(enabled: boolean) {
        this.skyboxEnabled = enabled;
    }

    public isPreviewVisible(): boolean {
        if (!this.overlayContainer) {
            return false;
        }

        if (!document.body.contains(this.overlayContainer)) {
            return false;
        }

        const computed = window.getComputedStyle(this.overlayContainer);
        const isVisible = computed.display !== 'none' && computed.visibility !== 'hidden' && computed.opacity !== '0';

        // 调试日志（仅在不可见时输出，避免日志过多）
        if (!isVisible && this.showPreview) {
            console.warn('[RecordingCamera] 预览窗口应该可见但检测为不可见', {
                display: computed.display,
                visibility: computed.visibility,
                opacity: computed.opacity,
                showPreview: this.showPreview
            });
        }

        return isVisible;
    }

    private cleanupCanvasElements() {
        // if (this.overlayContainer && this.overlayContainer.parentNode) {
        //     this.overlayContainer.parentNode.removeChild(this.overlayContainer);
        // }
        // if (this.canvas && this.canvas.parentNode) {
        //     this.canvas.parentNode.removeChild(this.canvas);
        // }

        if (this.gaussianRenderer) {
            this.gaussianRenderer = null;
        }

        if (this.renderer) {
            this.renderer.dispose();
            this.renderer = null;
        }
        console.log("dom----", document.getElementById(this.domId));
    }

    private hideEditorHelpers(scene: THREE.Scene) {
        this.editorHelperVisibilityCache.length = 0;
        scene.traverse((object) => {
            if (object.userData && object.userData.__visionaryEditorHelper) {
                this.editorHelperVisibilityCache.push({ object, visible: object.visible });
                object.visible = false;
            }
        });
    }

    private restoreEditorHelpers() {
        for (const entry of this.editorHelperVisibilityCache) {
            entry.object.visible = entry.visible;
        }
        this.editorHelperVisibilityCache.length = 0;
    }

    private createOverlayCanvas(width: number, height: number, currLang: string = 'zh') {
        this.cleanupCanvasElements();
        this.showPreview = true;
        const overlayDom = document.getElementById(this.domId);
        if (overlayDom) {
            this.overlayContainer = overlayDom;
        } else {
            console.log('创建录制相机预览窗口dom');
            // 创建覆盖层容器
            this.overlayContainer = document.createElement('div');
            this.overlayContainer.id = this.domId;
            this.overlayContainer.classList.add('recording-overlay-container');

            let titleTextStr='录制预览',syncTextStr=' - 同步录制相机';

            if(currLang === 'zh') {
                titleTextStr = '录制预览';
                syncTextStr = ' - 同步录制相机';
            } else if(currLang === 'en') {
                titleTextStr = 'Preview';
                syncTextStr = ' - Sync Camera';
            }

            // 创建标题栏（可拖动）
            const titleBar = document.createElement('div');
            titleBar.classList.add('title-bar');

            const titleContent = document.createElement('h3');
            titleContent.classList.add('recording-overlay-title');
            const titleText=document.createElement('span');
            titleText.textContent = titleTextStr;
            titleText.setAttribute('data-i18n', 'recordingCamera.previewWindowTitle');
            titleContent.appendChild(titleText);
            if(this.isSync) {    
                const syncText=document.createElement('span');
                syncText.textContent = syncTextStr;
                syncText.setAttribute('data-i18n', 'recordingCamera.syncPreviewTitle');
                titleContent.appendChild(syncText);
            }
            const titleName=document.createElement('span');
            titleName.classList.add('recording-overlay-title-name');
            titleName.textContent = this.cameraName;
            titleContent.appendChild(titleName);
            this.titleDom=titleName;

            const closeBtn = document.createElement('button');
            closeBtn.textContent = '×';
            closeBtn.classList.add('recording-overlay-close-btn');
            closeBtn.onclick = () => this.hidePreviewWindow();

            titleBar.appendChild(titleContent);
            titleBar.appendChild(closeBtn);
            this.overlayContainer.appendChild(titleBar);

            // 创建Canvas容器
            const canvasContainer = document.createElement('div');
            canvasContainer.classList.add('canvas-container');

            // 创建Canvas - 保持原始分辨率用于录制
            this.canvas = document.createElement('canvas');
            this.canvas.width = width;   // 使用完整的1920
            this.canvas.height = height; // 使用完整的1080

            // 计算预览窗口的显示尺寸（保持宽高比）
            const maxDisplayWidth = 400;
            const maxDisplayHeight = 300;
            const aspectRatio = width / height;

            let displayWidth = maxDisplayWidth;
            let displayHeight = maxDisplayWidth / aspectRatio;

            if (displayHeight > maxDisplayHeight) {
                displayHeight = maxDisplayHeight;
                displayWidth = maxDisplayHeight * aspectRatio;
            }

            this.canvas.classList.add('recording-overlay-canvas');
            this.canvas.style.cssText = `
            width: ${displayWidth}px;
            height: ${displayHeight}px;
        `;
            canvasContainer.appendChild(this.canvas);
            this.overlayContainer.appendChild(canvasContainer);

            // 添加相机信息
            const cameraInfo = document.createElement('div');
            cameraInfo.classList.add('camera-info');
            this.cameraInfoDom=cameraInfo;
            this.overlayContainer.appendChild(cameraInfo);

            // 将覆盖层添加到主窗口
            // document.body.appendChild(this.overlayContainer);
            const parentNode = document.getElementById(this.parentNodeId);
            if (parentNode) {
                parentNode.appendChild(this.overlayContainer);
            }

            // 添加拖动功能
            this.makeDraggable(this.overlayContainer, titleBar);
            // 只使用 WebGPU 上下文，不使用 2D 上下文
        }

    }

    private createHiddenCanvas(width: number, height: number) {
        this.cleanupCanvasElements();
        this.showPreview = false;

        // 创建隐藏的Canvas用于录制（无预览）
        this.canvas = document.createElement('canvas');
        this.canvas.width = width;
        this.canvas.height = height;

        // 隐藏Canvas，不显示在页面上
        this.canvas.style.cssText = `
            position: fixed;
            top: -9999px;
            left: -9999px;
            width: ${width}px;
            height: ${height}px;
            visibility: hidden;
            pointer-events: none;
        `;

        // 添加到DOM（虽然不可见）
        document.body.appendChild(this.canvas);

        // 创建简单的状态容器（可选）
        this.overlayContainer = document.createElement('div');
        this.overlayContainer.id = 'recordingStatus';
        this.overlayContainer.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: rgba(0, 0, 0, 0.8);
            color: #fff;
            padding: 10px 15px;
            border-radius: 5px;
            font-family: Arial, sans-serif;
            font-size: 14px;
            z-index: 10000;
            pointer-events: none;
        `;
        this.overlayContainer.textContent = '录制中...';
        document.body.appendChild(this.overlayContainer);
    }


    // 添加拖动功能
    private makeDraggable(element: HTMLElement, handle: HTMLElement) {
        let isDragging = false;
        let startX = 0;
        let startY = 0;
        let startLeft = 0;
        let startTop = 0;

        // 获取父容器元素
        const parentNode = document.getElementById(this.parentNodeId);
        if (!parentNode) {
            console.warn('[RecordingCamera] 未找到父容器元素，无法启用拖动功能');
            return;
        }

        const handleMouseMove = (e: MouseEvent) => {
            if (!isDragging) return;

            const deltaX = e.clientX - startX;
            const deltaY = e.clientY - startY;

            const newLeft = startLeft + deltaX;
            const newTop = startTop + deltaY;

            // 获取父容器的边界范围
            const parentRect = parentNode.getBoundingClientRect();
            const maxLeft = parentRect.width - element.offsetWidth;
            const maxTop = parentRect.height - element.offsetHeight;

            // 限制在父容器范围内（相对于父容器的位置）
            const relativeLeft = newLeft - parentRect.left;
            const relativeTop = newTop - parentRect.top;

            element.style.left = Math.max(0, Math.min(relativeLeft, maxLeft)) + 'px';
            element.style.top = Math.max(0, Math.min(relativeTop, maxTop)) + 'px';
            element.style.right = 'auto';
            element.style.bottom = 'auto';
        };

        // 清理拖拽状态和事件监听器
        const cleanupDrag = () => {
            if (isDragging) {
                isDragging = false;
                element.style.cursor = 'move';
                parentNode.removeEventListener('mousemove', handleMouseMove);
                parentNode.removeEventListener('mouseup', handleMouseUp);
                parentNode.removeEventListener('mouseleave', handleMouseLeave);
                // 同时移除document上的监听，以防鼠标移出父容器
                document.removeEventListener('mousemove', handleMouseMove);
                document.removeEventListener('mouseup', handleMouseUp);
            }
        };

        const handleMouseUp = () => {
            cleanupDrag();
        };

        const handleMouseLeave = () => {
            // 当鼠标移出父容器范围时，视同mouseup
            cleanupDrag();
        };

        handle.addEventListener('mousedown', (e) => {
            isDragging = true;
            startX = e.clientX;
            startY = e.clientY;

            const rect = element.getBoundingClientRect();
            startLeft = rect.left;
            startTop = rect.top;

            element.style.cursor = 'grabbing';
            e.preventDefault();

            // 在父容器上添加事件监听
            parentNode.addEventListener('mousemove', handleMouseMove);
            parentNode.addEventListener('mouseup', handleMouseUp);
            parentNode.addEventListener('mouseleave', handleMouseLeave);
            // 同时在document上添加监听，以防鼠标移出父容器
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
        });
    }

    // 设置相机位置和朝向
    setPosition(x: number, y: number, z: number) {
        this.camera.position.set(x, y, z);
        this.camera.updateMatrixWorld();
        this.syncWrapperTransform();
        // 更新gizmo位置
        this.updateGizmo();
    }

    // 设置相机朝向目标
    lookAt(x: number, y: number, z: number) {
        this.camera.lookAt(x, y, z);
        this.camera.updateMatrixWorld();
        this.syncWrapperTransform();
        // 更新gizmo位置和朝向
        this.updateGizmo();
    }

    public syncWrapperTransform() {
        if (!this.sceneWrapper) return;
        this.sceneWrapper.position.copy(this.camera.position);
        this.sceneWrapper.quaternion.copy(this.camera.quaternion);
        this.sceneWrapper.scale.set(1, 1, 1);
        if (this.gizmo && this.gizmo.parent !== this.sceneWrapper) {
            this.sceneWrapper.add(this.gizmo);
        }
        this.sceneWrapper.updateMatrixWorld(true);
    }

    public syncCameraFromWrapper() {
        if (!this.sceneWrapper) return;
        this.sceneWrapper.updateMatrixWorld(true);
        this.sceneWrapper.matrixWorld.decompose(this.tempPosition, this.tempQuaternion, this.tempScale);
        this.camera.position.copy(this.tempPosition);
        this.camera.quaternion.copy(this.tempQuaternion);
        this.camera.scale.copy(this.tempScale); // 增加录制相机的缩放
        this.camera.updateMatrixWorld(true);
        this.updateGizmo();
    }

    public getSceneWrapper(): THREE.Group | null {
        return this.sceneWrapper;
    }

    public attachSceneObject(wrapper: THREE.Group) {
        if (!wrapper.name) {
            wrapper.name = 'RecordingCameraWrapper';
        }
        this.sceneWrapper = wrapper;
        if (!this.sceneWrapper.userData) {
            this.sceneWrapper.userData = {};
        }
        this.sceneWrapper.userData.recordingCamera = this;
        this.sceneWrapper.userData.type = 'recordingCamera';

        if (this.gizmo && this.gizmo.parent !== this.sceneWrapper) {
            this.sceneWrapper.add(this.gizmo);
        }

        this.syncWrapperTransform();
    }

    // 创建相机视锥gizmo（线框四边形棱台）
    createGizmo(color: number = 0x00ff00, length?: number): THREE.Group {
        if (this.gizmo) {
            return this.gizmo;
        }

        this.gizmoColor = color;
        if (length !== undefined) {
            this.gizmoLength = length;
        }

        const group = new THREE.Group();
        group.name = 'RecordingCameraGizmo';
        group.position.set(0, 0, 0);
        group.layers.set(0);

        // 计算视锥的8个顶点（近平面和远平面的4个角）
        const updateVertices = () => {
            const camera = this.camera;
            const near = camera.near;
            const far = near + this.gizmoLength; // 使用配置的长度
            const fov = camera.fov * (Math.PI / 180);
            const aspect = camera.aspect;

            // 计算近平面和远平面的高度和宽度
            const nearHeight = 2 * Math.tan(fov / 2) * near;
            const nearWidth = nearHeight * aspect;
            const farHeight = 2 * Math.tan(fov / 2) * far;
            const farWidth = farHeight * aspect;

            // 近平面4个顶点（相对于相机局部坐标系）
            const nearTopLeft = new THREE.Vector3(-nearWidth / 2, nearHeight / 2, -near);
            const nearTopRight = new THREE.Vector3(nearWidth / 2, nearHeight / 2, -near);
            const nearBottomLeft = new THREE.Vector3(-nearWidth / 2, -nearHeight / 2, -near);
            const nearBottomRight = new THREE.Vector3(nearWidth / 2, -nearHeight / 2, -near);

            // 远平面4个顶点
            const farTopLeft = new THREE.Vector3(-farWidth / 2, farHeight / 2, -far);
            const farTopRight = new THREE.Vector3(farWidth / 2, farHeight / 2, -far);
            const farBottomLeft = new THREE.Vector3(-farWidth / 2, -farHeight / 2, -far);
            const farBottomRight = new THREE.Vector3(farWidth / 2, -farHeight / 2, -far);

            return {
                near: { topLeft: nearTopLeft, topRight: nearTopRight, bottomLeft: nearBottomLeft, bottomRight: nearBottomRight },
                far: { topLeft: farTopLeft, topRight: farTopRight, bottomLeft: farBottomLeft, bottomRight: farBottomRight }
            };
        };

        const vertices = updateVertices();

        // 创建线框材质
        const lineMaterial = new THREE.LineBasicMaterial({
            color: color,
            linewidth: 2,
            transparent: true,
            opacity: 0.8
        });

        // 创建近平面矩形
        const nearGeometry = new THREE.BufferGeometry().setFromPoints([
            vertices.near.topLeft,
            vertices.near.topRight,
            vertices.near.bottomRight,
            vertices.near.bottomLeft,
            vertices.near.topLeft
        ]);
        const nearLines = new THREE.Line(nearGeometry, lineMaterial);
        group.add(nearLines);

        // 创建远平面矩形
        const farGeometry = new THREE.BufferGeometry().setFromPoints([
            vertices.far.topLeft,
            vertices.far.topRight,
            vertices.far.bottomRight,
            vertices.far.bottomLeft,
            vertices.far.topLeft
        ]);
        const farLines = new THREE.Line(farGeometry, lineMaterial);
        group.add(farLines);

        // 创建连接近远平面的4条边
        const edge1 = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([vertices.near.topLeft, vertices.far.topLeft]),
            lineMaterial
        );
        const edge2 = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([vertices.near.topRight, vertices.far.topRight]),
            lineMaterial
        );
        const edge3 = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([vertices.near.bottomLeft, vertices.far.bottomLeft]),
            lineMaterial
        );
        const edge4 = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([vertices.near.bottomRight, vertices.far.bottomRight]),
            lineMaterial
        );
        group.add(edge1, edge2, edge3, edge4);

        // 添加中心线（从相机位置到视锥中心）
        const far = this.camera.near + this.gizmoLength;
        const centerLine = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(0, 0, 0),
                new THREE.Vector3(0, 0, -(this.camera.near + far) / 2)
            ]),
            new THREE.LineBasicMaterial({ color: color, linewidth: 1, transparent: true, opacity: 0.5 })
        );
        group.add(centerLine);

        // 将gizmo添加到相机组中，跟随相机位置和旋转
        this.gizmo = group;

        if (this.sceneWrapper && group.parent !== this.sceneWrapper) {
            this.sceneWrapper.add(group);
        }

        this.updateGizmo();

        return group;
    }

    // 更新gizmo位置和朝向
    updateGizmo() {
        if (!this.gizmo) return;

        // 更新gizmo的几何体以反映相机参数变化
        // 清除旧的几何体
        const oldChildren = [...this.gizmo.children];
        oldChildren.forEach(child => {
            if (child instanceof THREE.Line) {
                child.geometry.dispose();
                if (child.material instanceof THREE.Material) {
                    child.material.dispose();
                }
                this.gizmo!.remove(child);
            }
        });

        // 重新创建gizmo几何体
        const camera = this.camera;
        const near = camera.near;
        const far = near + this.gizmoLength; // 使用配置的长度
        const fov = camera.fov * (Math.PI / 180);
        const aspect = camera.aspect;

        // 计算近平面和远平面的高度和宽度
        const nearHeight = 2 * Math.tan(fov / 2) * near;
        const nearWidth = nearHeight * aspect;
        const farHeight = 2 * Math.tan(fov / 2) * far;
        const farWidth = farHeight * aspect;

        // 近平面4个顶点
        const nearTopLeft = new THREE.Vector3(-nearWidth / 2, nearHeight / 2, -near);
        const nearTopRight = new THREE.Vector3(nearWidth / 2, nearHeight / 2, -near);
        const nearBottomLeft = new THREE.Vector3(-nearWidth / 2, -nearHeight / 2, -near);
        const nearBottomRight = new THREE.Vector3(nearWidth / 2, -nearHeight / 2, -near);

        // 远平面4个顶点
        const farTopLeft = new THREE.Vector3(-farWidth / 2, farHeight / 2, -far);
        const farTopRight = new THREE.Vector3(farWidth / 2, farHeight / 2, -far);
        const farBottomLeft = new THREE.Vector3(-farWidth / 2, -farHeight / 2, -far);
        const farBottomRight = new THREE.Vector3(farWidth / 2, -farHeight / 2, -far);

        // 创建线框材质
        const lineMaterial = new THREE.LineBasicMaterial({
            color: this.gizmoColor,
            linewidth: 2,
            transparent: true,
            opacity: 0.8
        });

        // 创建近平面矩形
        const nearGeometry = new THREE.BufferGeometry().setFromPoints([
            nearTopLeft, nearTopRight, nearBottomRight, nearBottomLeft, nearTopLeft
        ]);
        const nearLines = new THREE.Line(nearGeometry, lineMaterial);
        this.gizmo.add(nearLines);

        // 创建远平面矩形
        const farGeometry = new THREE.BufferGeometry().setFromPoints([
            farTopLeft, farTopRight, farBottomRight, farBottomLeft, farTopLeft
        ]);
        const farLines = new THREE.Line(farGeometry, lineMaterial);
        this.gizmo.add(farLines);

        // 创建连接近远平面的4条边
        const edge1 = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([nearTopLeft, farTopLeft]),
            lineMaterial
        );
        const edge2 = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([nearTopRight, farTopRight]),
            lineMaterial
        );
        const edge3 = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([nearBottomLeft, farBottomLeft]),
            lineMaterial
        );
        const edge4 = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([nearBottomRight, farBottomRight]),
            lineMaterial
        );
        this.gizmo.add(edge1, edge2, edge3, edge4);

        // 添加中心线
        const centerLine = new THREE.Line(
            new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(0, 0, 0),
                new THREE.Vector3(0, 0, -(near + far) / 2)
            ]),
            new THREE.LineBasicMaterial({ color: this.gizmoColor, linewidth: 1, transparent: true, opacity: 0.5 })
        );
        this.gizmo.add(centerLine);

        if (this.sceneWrapper) {
            this.gizmo.position.set(0, 0, 0);
            this.gizmo.quaternion.identity();
            this.gizmo.rotation.set(0, 0, 0);
        } else {
            this.gizmo.position.copy(this.camera.position);
            this.gizmo.quaternion.copy(this.camera.quaternion);
        }
        this.gizmo.visible = this.gizmoVisible;
        this.gizmo.updateMatrixWorld(true);
    }

    // 设置gizmo可见性
    setGizmoVisible(visible: boolean) {
        this.gizmoVisible = visible;
        if (this.gizmo) {
            this.gizmo.visible = visible;
        }
    }

    // 设置gizmo长度
    setGizmoLength(length: number) {
        if (length <= 0) {
            console.warn('[RecordingCamera] gizmo长度必须大于0');
            return;
        }
        this.gizmoLength = length;
        // 如果gizmo已创建，更新它
        if (this.gizmo) {
            this.updateGizmo();
        }
    }

    // 获取gizmo长度
    getGizmoLength(): number {
        return this.gizmoLength;
    }

    // 获取gizmo对象（用于添加到场景）
    getGizmo(): THREE.Group | null {
        return this.gizmo;
    }

    // 获取场景对象（包装对象，用于handleObjectsPush）
    getSceneObject(): THREE.Object3D {
        if (!this.sceneWrapper) {
            const wrapper = new THREE.Group();
            this.attachSceneObject(wrapper);
        }

        const wrapper = this.sceneWrapper!;

        if (!this.gizmo) {
            this.createGizmo(0x00ff00);
        }

        if (this.gizmo && wrapper.children.indexOf(this.gizmo) === -1) {
            wrapper.add(this.gizmo);
        }

        this.syncWrapperTransform();

        return wrapper;
    }

    // 更新状态信息（使用 HTML overlay）
    public updateStatusInfo(status: string) {
        if (this.statusDom) {
            this.statusDom.textContent = status;
        }

        // 更新相机信息（仅在预览模式下）
        // if (this.cameraInfoDom) {
        //     this.cameraInfoDom.innerHTML = `
        //         Position: (${this.camera.position.x.toFixed(2)}, ${this.camera.position.y.toFixed(2)}, ${this.camera.position.z.toFixed(2)})<br>
        //         Rotation: (${this.camera.rotation.x.toFixed(3)}, ${this.camera.rotation.y.toFixed(3)}, ${this.camera.rotation.z.toFixed(3)})<br>
        //         Scale: (${this.camera.scale.x.toFixed(3)}, ${this.camera.scale.y.toFixed(3)}, ${this.camera.scale.z.toFixed(3)})<br>
        //         FOV: ${this.camera.fov}°
        //     `;
        // }
    }

    // 初始化录制相机的渲染器
    async initializeRenderer(
        mainRenderer: THREE.WebGPURenderer,
        scene: THREE.Scene,
        gaussianModels?: GaussianModel[],
        originalTexture?: THREE.DataTexture
    ) {
        try {
            // 获取主渲染器的WebGPU设备
            const backend = (mainRenderer as any).backend;
            const device = backend?.device;

            if (!device) {
                throw new Error('无法获取WebGPU设备');
            }

            // 为录制Canvas获取WebGPU上下文
            const context = this.canvas.getContext('webgpu') as GPUCanvasContext;
            if (!context) {
                throw new Error('无法获取WebGPU上下文');
            }

            // 配置WebGPU上下文，使用共享的device
            const format = navigator.gpu.getPreferredCanvasFormat();
            context.configure({
                device: device,
                format: format,
                alphaMode: 'premultiplied'
            });

            // 创建独立的Three.js渲染器
            this.renderer = new THREE.WebGPURenderer({
                canvas: this.canvas,
                antialias: true,
                forceWebGL: false,
                context: context,
                device: device
            });

            await this.renderer.init();
            
            // ✅ 使用 RendererInitHelper 初始化渲染器（一站式方法）
            // 如果提供了 mainRenderer，则从中同步配置；否则使用默认配置
            const initResult = await RendererInitHelper.initializeRenderer(
                this.renderer,
                scene,
                {
                    sourceRenderer: mainRenderer,
                    originalTexture: originalTexture || null,
                    width: this.canvas.width,
                    height: this.canvas.height,
                    pixelRatio: 1
                }
            );
            
            // 保存录制渲染器专用的环境贴图和背景
            // 不在初始化时替换 scene 的属性，而是在渲染时临时替换
            this.recordingEnvMap = initResult.envMap;
            this.recordingBackground = initResult.background;

            // 如果提供了高斯模型，为RecordingCamera创建独立的GaussianThreeJSRenderer实例
            console.log('[RecordingCamera] 开始初始化 GaussianThreeJSRenderer...');
            console.log('[RecordingCamera] gaussianModels:', gaussianModels ? `${gaussianModels.length}个` : '无');
            console.log('[RecordingCamera] this.renderer:', this.renderer ? '存在' : '不存在');
            
            if (gaussianModels && gaussianModels.length > 0 && this.renderer) {
                console.log('[RecordingCamera] 创建 GaussianThreeJSRenderer...');
                this.gaussianRenderer = new GaussianThreeJSRenderer(
                    this.renderer,  // 使用录制渲染器
                    scene,          // 共享场景
                    gaussianModels  // 共享高斯模型
                );
                console.log('[RecordingCamera] 调用 gaussianRenderer.init()...');
                await this.gaussianRenderer.init();  // ✅ 确保初始化完成

                // ✅ 触发首次深度纹理创建
                // 必须传入 forceUpdate=true，确保在录制场景下更新深度纹理
                console.log('[RecordingCamera] 调用 onResize...', this.canvas.width, this.canvas.height);
                this.gaussianRenderer.onResize(this.canvas.width, this.canvas.height, true);

                console.log('[RecordingCamera] 录制相机Gaussian渲染器初始化成功');
            } else {
                if (!gaussianModels || gaussianModels.length === 0) {
                    console.log('[RecordingCamera] 未提供高斯模型，使用标准 Three.js 渲染流程');
                }
                if (!this.renderer) {
                    console.warn('[RecordingCamera] 录制渲染器未初始化');
                }
            }

            console.log('录制相机渲染器初始化成功');
            return true;

        } catch (error) {
            console.warn('录制相机渲染器初始化失败:', error);
            this.renderer = null;
            this.gaussianRenderer = null;
            return false;
        }
    }

    public onResize(isRecording: boolean = false,resolution: { width: number, height: number }) {
        if (!this.renderer||!this.overlayContainer) return;

        this.overlayContainer.style.inset='';

        // const canvasContainer=this.overlayContainer.querySelector('.canvas-container') as HTMLElement;
        // if(!canvasContainer) return;
        // const width=Math.floor(canvasContainer.clientWidth/2)*2;
        // const height=Math.floor(canvasContainer.clientHeight/2)*2;
        // if(width===0 || height===0) return;
        // // if(this.canvas){
        // //     this.canvas.width=width;
        // //     this.canvas.height=height;
        // // }
        // console.log('[RecordingCamera] 录制相机调用 onResize...', width, height);
                // 如果在录制中，使用canvas的实际像素尺寸，而不是CSS显示尺寸
        // 这样可以避免CSS resize导致的尺寸不匹配问题
        let width: number;
        let height: number;
        console.log('[RecordingCamera] onResize...', isRecording, resolution);
        if (isRecording && this.canvas) {
            // 录制模式：使用canvas的实际像素尺寸
            // width = this.canvas.width;
            // height = this.canvas.height;
            width = resolution.width||this.canvas.width;
            height = resolution.height||this.canvas.height;
            console.log('[RecordingCamera] 录制模式 onResize，使用实际像素尺寸:', width, height);
        } else {
            // 预览模式：使用CSS显示尺寸
            const canvasContainer=this.overlayContainer.querySelector('.canvas-container') as HTMLElement;
            if(!canvasContainer) return;
            width=Math.floor(canvasContainer.clientWidth/2)*2;
            height=Math.floor(canvasContainer.clientHeight/2)*2;
            if(width===0 || height===0) return;
            console.log('[RecordingCamera] 预览模式 onResize，使用CSS显示尺寸:', width, height);
        }
        
        if (this.renderer) {
            this.renderer.setSize(width, height, false);
        }
        if (this.gaussianRenderer) {
            // 如果在录制模式，强制更新深度纹理；否则使用默认行为
            // this.gaussianRenderer.onResize(this.canvas.width, this.canvas.height, isRecording);
            this.gaussianRenderer.onResize(width, height, isRecording);
        }
    }

    public async updateGaussianModels(gaussianModels: GaussianModel[], scene: THREE.Scene) {
        return new Promise((resolve, reject) => {
            let oldShow = this.isPreviewShow;
            if (this.gaussianRenderer) {
                this.gaussianRenderer = null;
            }
            if (gaussianModels && gaussianModels.length > 0) {
                this.isPreviewShow = false;
                this.gaussianRenderer = new GaussianThreeJSRenderer(
                    this.renderer as THREE.WebGPURenderer,  // 使用录制渲染器
                    scene,          // 共享场景
                    gaussianModels  // 共享高斯模型
                );
                console.log('init 高斯renderer', oldShow)
                this.gaussianRenderer.init().then(() => {
                    this.isPreviewShow = oldShow;
                    resolve(true);
                }).catch((error) => {
                    console.warn('高斯renderer初始化失败:', error);
                    this.isPreviewShow = oldShow;
                    reject(error);
                });
            } else {
                console.warn('没有提供高斯模型');
                resolve(true);
            }
        });
    }

    // 使用录制相机渲染到录制相机的Canvas
    async render(scene: THREE.Scene) {
        if (!this.renderer) {
            const errorMsg = '录制渲染器未初始化';
            console.error(`[RecordingCamera] ${errorMsg}`);
            this.updateStatusInfo(errorMsg);
            return;
        }
        if (!this.isPreviewShow) return;
        if (!this.isPreviewVisible()) return;
        this.hideEditorHelpers(scene);

        try {
            // ✅ 关键修复：更新相机矩阵，确保isVisible()能正确判断可见性
            this.camera.updateMatrixWorld();

            // ✅ 修复：每次渲染前都重新保存当前场景状态，而不是只在第一次保存
            // 这样可以确保如果场景状态在导出过程中被外部修改（如 setSkyboxEnabled），
            // 我们保存的是最新的状态，而不是第一次调用时的旧状态
            this.originalEnvMap = scene.environment;
            this.originalBackground = scene.background;
            
            // ✅ 如果开启天空盒替换且已准备独立环境，临时替换
            if (this.skyboxEnabled && this.recordingEnvMap && this.recordingBackground) {
                scene.environment = this.recordingEnvMap;
                scene.background = this.recordingBackground;
            }
            
            try {
                if (this.gaussianRenderer) {
                    // ✅ 关键修复：先调用 onBeforeRender 设置 this.pcs
                    this.gaussianRenderer.onBeforeRender(
                        this.renderer,  // 录制渲染器
                        scene,          // 共享场景
                        this.camera     // 录制相机
                    );

                    // ✅ 渲染 Three.js 场景到 RenderTarget（捕获深度）
                    this.gaussianRenderer.renderThreeScene(this.camera);

                    // ✅ 渲染高斯 splats 到录制 canvas
                    this.gaussianRenderer.drawSplats(
                        this.renderer,  // 录制渲染器
                        scene,          // 共享场景
                        this.camera     // 录制相机
                    );
                } else {
                // 无 Gaussian 渲染器时，直接使用 Three.js 渲染
                // ✅ 关键修复：在 WebGPU 中，PMREM 纹理绑定到特定渲染器
                // 如果为录制渲染器创建了独立的环境贴图，需要临时替换 scene 的属性
                
                // 保存原始值（如果还没有保存）
                if (this.originalEnvMap === null) {
                    this.originalEnvMap = scene.environment;
                }
                if (this.originalBackground === null) {
                    this.originalBackground = scene.background;
                }
                
                // ✅ 如果为录制渲染器创建了独立的环境贴图和背景，临时替换
                if (this.skyboxEnabled && this.recordingEnvMap && this.recordingBackground) {
                    scene.environment = this.recordingEnvMap;
                    scene.background = this.recordingBackground;
                }
                
                    // 直接渲染场景，Three.js 应该自动处理背景和环境贴图
                    this.renderer.render(scene, this.camera);
                }
            } finally {
                // ✅ 修复：始终恢复原始值，无论 skyboxEnabled 状态如何
                // 这样可以确保场景状态在渲染后正确恢复，避免被临时替换的值覆盖
                if (this.originalEnvMap !== null) {
                    scene.environment = this.originalEnvMap;
                }
                if (this.originalBackground !== null) {
                    scene.background = this.originalBackground;
                }
            }

            // 等待GPU操作完成，确保渲染完成
            const backend = (this.renderer as any).backend;
            const device = backend?.device as GPUDevice;
            if (device && device.queue) {
                await device.queue.onSubmittedWorkDone();
            }
        } catch (error) {
            console.error('[RecordingCamera] 录制相机渲染失败error:', error);
            this.updateStatusInfo("Error: Rendering failed");
            // 不抛出错误，避免中断录制流程
        } finally {
            this.restoreEditorHelpers();
        }
    }

    // 获取canvas流用于录制
    getStream(fps: number = 30) {
        if (!this.canvas) {
            console.error('[RecordingCamera] getStream Canvas 未初始化，无法获取流');
            return null;
        }

        try {
            const stream = this.canvas.captureStream(fps);
            if (!stream) {
                console.error('[RecordingCamera] getStream captureStream 返回 null');
                return null;
            }

            // 验证流是否有效
            if (stream.getVideoTracks().length === 0) {
                console.error('[RecordingCamera] getStream 流中没有视频轨道');
                return null;
            }

            return stream;
        } catch (error) {
            console.error('获取录制流失败:', error);
            return null;
        }
    }

    getParameters(): any {
        return { width: this.width, height: this.height, fov: this.camera.fov };
    }

    // 销毁资源
    dispose() {
        // dispose gizmo
        if (this.gizmo) {
            this.gizmo.traverse((child) => {
                if (child instanceof THREE.Line) {
                    child.geometry.dispose();
                    if (child.material instanceof THREE.Material) {
                        child.material.dispose();
                    }
                }
            });
            this.gizmo = null;
        }

        if (this.sceneWrapper) {
            if (this.sceneWrapper.parent) {
                this.sceneWrapper.parent.remove(this.sceneWrapper);
            }
            this.sceneWrapper = null;
        }

        // dispose Gaussian渲染器
        if (this.gaussianRenderer) {
            this.gaussianRenderer = null;
        }

        // dispose独立的录制渲染器
        if (this.renderer) {
            this.renderer.dispose();
            this.renderer = null;
        }

        // 从主窗口移除覆盖层和Canvas
        if (this.overlayContainer && this.overlayContainer.parentNode) {
            this.overlayContainer.parentNode.removeChild(this.overlayContainer);
            this.overlayContainer = null!;
        }

        if (this.canvas && this.canvas.parentNode) {
            this.canvas.parentNode.removeChild(this.canvas);
            this.canvas = null!;
        }
    }

    /**
     * Actively renders the current scene from this camera's perspective into a new, stable canvas.
     * @returns A Promise that resolves with an HTMLCanvasElement containing the rendered image.
     */
    public async renderToCanvas(scene: THREE.Scene): Promise<HTMLCanvasElement> {
        if (!this.renderer || !this.sceneWrapper) {
            throw new Error("RecordingCamera is not fully initialized. Renderer or Scene is missing.");
        }

        const renderer = this.renderer;
        const camera = this.camera;
        // 使用 canvas 的实际尺寸，而不是内部存储的 width/height
        // 这样可以确保与 RecordingManager 设置的分辨率一致
        const width = this.canvas.width || this.width;
        const height = this.canvas.height || this.height;

        // 确保渲染尺寸和相机宽高比正确
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();

        // ✅ 修复：每次渲染前都重新保存当前场景状态，而不是只在第一次保存
        // 这样可以确保如果场景状态在导出过程中被外部修改（如 setSkyboxEnabled），
        // 我们保存的是最新的状态，而不是第一次调用时的旧状态
        this.originalEnvMap = scene.environment;
        this.originalBackground = scene.background;

        try {
            // 为本次录制渲染设置专用的环境贴图和背景
            if (this.skyboxEnabled && this.recordingEnvMap) {
                scene.environment = this.recordingEnvMap;
            }
            if (this.skyboxEnabled && this.recordingBackground) {
                scene.background = this.recordingBackground;
            }
            var render_success = false;
            // 执行渲染
            if (this.gaussianRenderer) {
                const currentTime = (Date.now() - (window as any).startTime) / 1000.0;
                await this.gaussianRenderer.updateDynamicModels(camera, currentTime);
                this.gaussianRenderer.onBeforeRender(
                    renderer,  // 录制渲染器
                    scene,          // 共享场景
                    camera     // 录制相机
                );

                const backend = (this.renderer as any).backend;
                const device = backend?.device as GPUDevice;
                if (device && device.queue) {
                    await device.queue.onSubmittedWorkDone();
                }

                this.gaussianRenderer.renderThreeScene(camera);
                render_success = this.gaussianRenderer.drawSplats(renderer, scene, camera);
            } 
            
            if (!render_success) {
                renderer.render(scene, camera);
                console.warn('fall back in three js render camera recording camera');
            }

            const backend = (this.renderer as any).backend;
            const device = backend?.device as GPUDevice;
            // if (device && device.queue) {
            //     await device.queue.onSubmittedWorkDone();
            //     console.log('onSubmittedWorkDone');
            // }

            // 将渲染结果绘制到新的Canvas上并返回
            const exportCanvas = document.createElement('canvas');
            exportCanvas.width = width;
            exportCanvas.height = height;
            const exportCtx = exportCanvas.getContext('2d');
            if (!exportCtx) {
                throw new Error('Failed to create context for export canvas');
            }

            const sourceCanvas = renderer.domElement;
            exportCtx.drawImage(sourceCanvas, 0, 0, width, height);
            if (device && device.queue) {
                await device.queue.onSubmittedWorkDone();
            }
            return exportCanvas;

        } catch (error) {
            // 重新抛出错误，让调用者知道渲染失败了
            throw new Error(`RecordingCamera failed to render frame: ${(error as Error).message}`);
        } finally {
            // ✅ 修复：始终恢复原始值，无论 skyboxEnabled 状态如何
            // 这样可以确保场景状态在渲染后正确恢复，避免被临时替换的值覆盖
            if (this.originalEnvMap !== null) {
                scene.environment = this.originalEnvMap;
            }
            if (this.originalBackground !== null) {
                scene.background = this.originalBackground;
            }
            
            // 清理引用，避免内存占用
            this.originalEnvMap = null;
            this.originalBackground = null;
        }
    }
}
