import * as THREE from "three/webgpu";
import {OrbitControls} from "three/examples/jsm/controls/OrbitControls.js";
import {initThreeContext, loadGaussianModels} from "../app";
import {exportImage, exportVideoWithRecordingCamera, RecordingCamera, VideoCodec, VideoQuality, VideoExportConfig} from "../exportMedia";
import {GridHelper} from "./GridHelper";


// 获取录制配置的通用函数
function getRecordingConfig(title: string = '录制配置'): { showPreview: boolean; config: VideoExportConfig } {
    // 询问是否显示预览
    const showPreview = confirm(`${title}\n\n是否显示录制预览窗口？\n\n点击"确定"显示画中画预览\n点击"取消"后台录制（无预览）`);
    
    // 询问质量预设
    const qualityChoice = prompt(
        '选择视频质量:\n\n' +
        '1. 低质量 (2 Mbps, 文件小)\n' +
        '2. 中质量 (5 Mbps, 平衡)\n' +
        '3. 高质量 (15 Mbps, 推荐)\n' +
        '4. 接近无损 (50 Mbps, 文件大)\n\n' +
        '请输入数字 (1-4):'
    );
    
    let quality = VideoQuality.MEDIUM;
    switch(qualityChoice) {
        case '1': quality = VideoQuality.LOW; break;
        case '2': quality = VideoQuality.MEDIUM; break;
        case '3': quality = VideoQuality.HIGH; break;
        case '4': quality = VideoQuality.NEAR_LOSSLESS; break;
    }
    
    const config: VideoExportConfig = {
        codec: VideoCodec.H264,
        quality: quality
    };
    
    return { showPreview, config };
}

async function main() {
    const canvasElement = document.querySelector("#canvas") as HTMLCanvasElement;
    if (canvasElement == null) {
        throw new Error("Can't find the canvas element!");
    }

    const renderer = await initThreeContext(canvasElement);
    if (renderer == null) {
        throw new Error("Failed to initialize the three context!");
    }

    const scene = new THREE.Scene();
    const frame = new THREE.Group();
    // frame.quaternion.set(0, 0, 0, 1);
    frame.quaternion.set(1, 0, 0, 0);
    scene.add(frame);
    const camera = new THREE.PerspectiveCamera(75, canvasElement.clientWidth / canvasElement.clientHeight, 0.1, 1000);
    camera.position.z = 5;

    const controls = new OrbitControls( camera, renderer.domElement );

    // Add Cube
    const cube = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial({color: 0xffffff}))
    cube.position.set(0, 0, 0);
    frame.add(cube);

    const cube_x = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial({color: 0xff0000}))
    cube_x.position.set(1, 0, 0);
    frame.add(cube_x);

    const cube_y = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial({color: 0x00ff00}))
    cube_y.position.set(0, 1, 0);
    frame.add(cube_y);

    const cube_z = new THREE.Mesh(new THREE.BoxGeometry(), new THREE.MeshStandardMaterial({color: 0x0000ff}))
    cube_z.position.set(0, 0, 1);
    frame.add(cube_z);

    // Add Light
    const light = new THREE.DirectionalLight(0xffffff, 1)
    light.position.set(-1, 2, 4)
    frame.add(light)

    // Add Infinite Grid Helper
    const gridHelper = new GridHelper(10, 10, 0x444444, 0x888888);
    frame.add(gridHelper.getGrid());
    
    // 配置网格参数
    gridHelper.setGridSizeRange(0.005, 50); // 密度范围: 0.005倍(超远) 到 50倍(极近)
    gridHelper.setFadeDistance(200); // 淡出距离
    gridHelper.setOpacity(0.8); // 透明度

    // Update camera matrices initially
    camera.updateMatrixWorld();
    camera.updateProjectionMatrix();

    // Add Gaussians with proper matrices
    const gaussianThreeJSRenderer = await loadGaussianModels(renderer, scene, frame,
        [
            "/models/gaussians3d_lhm_split.onnx",
            "/models/test.ply"
        ],
        {
            camMat: new Float32Array(camera.matrixWorldInverse.elements),
            projMat: new Float32Array(camera.projectionMatrix.elements)
        }
    );

    // 示例后处理：轻微提饱和 + 暗角
    if (gaussianThreeJSRenderer) {
        gaussianThreeJSRenderer.setPostProcessEnabled(true);
        gaussianThreeJSRenderer.setPostProcessShader(`
            @fragment
            fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
                let color = textureSample(sourceTexture, sourceSampler, uv);
                let centered = uv - vec2f(0.5, 0.5);
                let vignette = clamp(1.0 - dot(centered, centered) * 1.8, 0.0, 1.0);
                let luma = dot(color.rgb, vec3f(0.299, 0.587, 0.114));
                let sat = 1.15;
                let rgb = vec3f(luma) + (color.rgb - vec3f(luma)) * sat;
                return vec4f(rgb * vignette, color.a);
            }
        `);
    }

    window.addEventListener("resize", () => {
        const width = canvasElement.clientWidth;
        const height = canvasElement.clientHeight;

        renderer.setSize(width, height, false);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

        camera.aspect = width / height;
        camera.updateProjectionMatrix();

        gaussianThreeJSRenderer?.onResize(width, height);
    });

    // Track time for dynamic updates
    const startTime = Date.now();

    // 添加导出按钮事件监听器
    const exportImageButton = document.getElementById('exportImageButton');
    if (exportImageButton) {
        exportImageButton.addEventListener('click', () => {
            exportImage(renderer);
        });
    }

    // const exportVideoButton = document.getElementById('exportVideoButton');
    // if (exportVideoButton) {
    //     exportVideoButton.addEventListener('click', () => {
    //         // 获取录制配置
    //         const { showPreview, config } = getRecordingConfig();
            
    //         // 创建录制相机，使用当前主相机的位置
    //         const recordingCamera = new RecordingCamera(1920, 1080, 75, showPreview);
    //         recordingCamera.setPosition(camera.position.x, camera.position.y, camera.position.z);
    //         recordingCamera.lookAt(0, 0, 0);
            
    //         // 使用真实时间模式录制（不传入 timelineController）
    //         exportVideoWithRecordingCamera(
    //             renderer, 
    //             scene, 
    //             recordingCamera, 
    //             5,  // duration: 5秒
    //             30, // fps: 30
    //             gaussianThreeJSRenderer, 
    //             showPreview, 
    //             config
    //         );
    //     });
    // }

    // 添加网格显示/隐藏按钮事件监听器
    const toggleGridButton = document.getElementById('toggleGridButton');
    if (toggleGridButton) {
        // 初始化按钮状态
        updateGridButtonState();
        
        toggleGridButton.addEventListener('click', () => {
            gridHelper.toggle();
            updateGridButtonState();
            console.log('Grid visibility:', gridHelper.isVisible() ? 'shown' : 'hidden');
        });
    }
    
    // 更新网格按钮状态
    function updateGridButtonState() {
        if (toggleGridButton) {
            if (gridHelper.isVisible()) {
                toggleGridButton.textContent = '隐藏网格';
                toggleGridButton.classList.add('active');
            } else {
                toggleGridButton.textContent = '显示网格';
                toggleGridButton.classList.remove('active');
            }
        }
    }

    renderer.setAnimationLoop(async () => {
        cube.rotation.x += 0.01;
        cube.rotation.y += 0.01;

        controls.update();

        // Update camera matrices for dynamic models
        camera.updateMatrixWorld();

        // Update infinite grid with camera position
        gridHelper.updateCamera(camera);

        // Update dynamic ONNX models with current time and camera
        if (gaussianThreeJSRenderer) {
            const currentTime = (Date.now() - startTime) / 1000.0; // Time in seconds
            await gaussianThreeJSRenderer.updateDynamicModels(camera, currentTime);
        }


        // New architecture: GaussianThreeJSRenderer handles Three.js rendering internally
        // This automatically captures depth from the full scene
        if (gaussianThreeJSRenderer) {
            gaussianThreeJSRenderer.renderThreeScene(camera);  // Replaces renderer.render(scene, camera)
            gaussianThreeJSRenderer.drawSplats(renderer, scene, camera);
        } else {
            // Fallback if no GaussianRenderer
            renderer.render(scene, camera);
        }
    });
}

main().then();