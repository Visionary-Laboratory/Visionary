import * as THREE from 'three/webgpu';
import { CarouselItemConfig, CameraViewConfig } from './types';

export const SCENE1_CAROUSEL_ITEMS: CarouselItemConfig[] = [
    { 
        type: 'file', 
        url: '/models/册方彝.glb', 
        loadOptions: { type: 'glb' as const, name: '册方彝' },
        scale: 2.5,
        transform: {
            position: new THREE.Vector3(0, -1, 0)
        }
    },
    { 
        type: 'file', 
        url: '/models/fox.ply', 
        loadOptions: { type: 'ply' as const, name: 'fox' },
        scale: 2.5
    },
    { 
        type: 'file', 
        url: '/models/dyn/mutant.onnx', 
        loadOptions: { type: 'onnx' as const, name: '0007_07' },
        scale: 2.5,
        transform: {
            rotation: new THREE.Euler(Math.PI / 2, 0, 0)
        }
    },
    { 
        type: 'file', 
        url: '/models/谷纹青玉璧.glb', 
        loadOptions: { type: 'glb' as const, name: '谷纹青玉璧' },
        scale: 2.5,
        transform: {
            position: new THREE.Vector3(0, -1, 0)
        }
    },
    { 
        type: 'file', 
        url: '/models/dyn/hellwarrior.onnx', 
        loadOptions: { type: 'onnx' as const, name: '0172_05' },
        scale: 2.5,
        transform: {
            rotation: new THREE.Euler(Math.PI / 2, 0, 0)
        }
    },
    { 
        type: 'file', 
        url: '/models/白玉卧鹿.glb', 
        loadOptions: { type: 'glb' as const, name: '白玉卧鹿' },
        scale: 2.5,
        transform: {
            position: new THREE.Vector3(0, -1, 0)
        }
    },
    { 
        type: 'file', 
        url: '/models/dyn/trex.onnx', 
        loadOptions: { type: 'onnx' as const, name: '0007_07' },
        scale: 2.5,
        transform: {
            rotation: new THREE.Euler(Math.PI / 2, 0, 0)
        }
    }
];

const SCENE2_POSITIONS = [
    new THREE.Vector3(0.26733479586726827, 0.30708254148041925, 1.7688451394840126),
    new THREE.Vector3(0.6188313777428768, 1.0811383545304041, 0.3726244060904826),
    new THREE.Vector3(0.019308701905451073, -0.39441113787133447, -1.5096615317410889),
    new THREE.Vector3(-1.9120670085314637, 0.2511638313745907, 1.4427727541792363)
];

const SCENE2_TARGETS = [
    new THREE.Vector3(0, 0.2, 0),
    new THREE.Vector3(0, 0.5, 0),
    new THREE.Vector3(0, 0.2, 0),
    new THREE.Vector3(0, 0.2, 0)
];

export function getScene2CameraViews(): CameraViewConfig[] {
    return [
        {
            position: SCENE2_POSITIONS[0].clone(),
            target: SCENE2_TARGETS[0].clone(),
            transitionDuration: 2,
            transitionEasing: 'easeInOutCubic',
            idleDuration: 4.0,
            idleSpeed: 0.05, 
            idleType: 'orbit'
        },
        {
            position: SCENE2_POSITIONS[1].clone(),
            target: SCENE2_TARGETS[1].clone(),
            transitionDuration: 2.0, 
            transitionEasing: 'easeOutCubic',
            idleDuration: 4.0,
            idleSpeed: 0.05, 
            idleType: 'orbit'
        },
        {
            position: SCENE2_POSITIONS[2].clone(),
            target: SCENE2_TARGETS[2].clone(),
            transitionDuration: 2.0,
            transitionEasing: 'easeInOutQuad',
            idleDuration: 4,
            idleSpeed: 0.05,
            idleType: 'orbit'
        },
        {
            position: SCENE2_POSITIONS[3].clone(),
            target: SCENE2_TARGETS[3].clone(),
            transitionDuration: 2.0,
            transitionEasing: 'easeInOutQuad',
            idleDuration: 4,
            idleSpeed: 0.05,
            idleType: 'orbit'
        }
    ];
}

const SCENE3_GAUSSIAN_IDS = ['gaussianA', 'gaussianB', 'gaussianC', 'gaussianD', 'gaussianE', 'gaussianF', 'gaussianG'];

export const SCENE3_MULTI_ONNX_CONFIGS: CarouselItemConfig[] = SCENE3_GAUSSIAN_IDS.map((id) => ({
    type: 'file',
    url: `/models/qiewu/${id}.onnx`,
    loadOptions: { type: 'onnx' as const, name: id },
    scale: 2.5,
    transform: {
        position: new THREE.Vector3(0, 0, 0),
        rotation: new THREE.Euler(0, 0, 0)
    }
}));

export const SCENE3_CAMERA_BASE = {
    position: new THREE.Vector3(0, 3, 12),
    target: new THREE.Vector3(0, 1.5, 0)
};

export const SCENE3_CAMERA_SWING = {
    amplitude: 8,
    frequency: 1,
    timeScale: 25000
};


