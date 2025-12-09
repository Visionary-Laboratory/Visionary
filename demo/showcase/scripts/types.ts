import * as THREE from 'three/webgpu';

export type EasingName =
    | 'linear'
    | 'easeInQuad'
    | 'easeOutQuad'
    | 'easeInOutQuad'
    | 'easeInCubic'
    | 'easeOutCubic'
    | 'easeInOutCubic';

export interface CameraViewConfig {
    position: THREE.Vector3;
    target: THREE.Vector3;
    transitionDuration: number;
    transitionEasing: EasingName;
    idleDuration: number;
    idleSpeed: number;
    idleType: 'orbit' | 'static';
}

export interface LocalTransform {
    position?: THREE.Vector3;
    rotation?: THREE.Euler;
}

export interface CarouselItemConfig {
    type: 'file' | 'mesh';
    url?: string;
    loadOptions?: Record<string, any>;
    scale?: number;
    geometry?: THREE.BufferGeometry;
    material?: THREE.Material;
    transform?: LocalTransform;
}


