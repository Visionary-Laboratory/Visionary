import { GaussianDataSource } from './index';

// ============= 通用Gaussian数据源 =============
export class PLYGaussianData implements GaussianDataSource {
  private _gaussianBuffer: ArrayBuffer;
  private _shCoefsBuffer: ArrayBuffer;
  private _extraPcaBuffer?: ArrayBuffer;
  private _weightBuffer?: ArrayBuffer;
  private _numPoints: number;
  private _shDegree: number;
  private _bbox: { min: [number, number, number]; max: [number, number, number] };
  public center?: [number, number, number];
  public up?: [number, number, number] | null;
  public kernelSize?: number;
  public mipSplatting?: boolean;
  public backgroundColor?: [number, number, number];

  constructor(data: {
    gaussianBuffer: ArrayBuffer;
    shCoefsBuffer: ArrayBuffer;
    extraPcaBuffer?: ArrayBuffer;
    weightBuffer?: ArrayBuffer;
    numPoints: number;
    shDegree: number;
    bbox: { min: [number, number, number]; max: [number, number, number] };
    center?: [number, number, number];
    up?: [number, number, number] | null;
    kernelSize?: number;
    mipSplatting?: boolean;
    backgroundColor?: [number, number, number];
  }) {
    this._gaussianBuffer = data.gaussianBuffer;
    this._shCoefsBuffer = data.shCoefsBuffer;
    this._extraPcaBuffer = data.extraPcaBuffer;
    this._weightBuffer = data.weightBuffer;
    this._numPoints = data.numPoints;
    this._shDegree = data.shDegree;
    this._bbox = data.bbox;
    this.center = data.center;
    this.up = data.up;
    this.kernelSize = data.kernelSize;
    this.mipSplatting = data.mipSplatting;
    this.backgroundColor = data.backgroundColor;
  }

  gaussianBuffer(): ArrayBuffer { return this._gaussianBuffer; }
  shCoefsBuffer(): ArrayBuffer { return this._shCoefsBuffer; }
  extraPcaBuffer(): ArrayBuffer {
    if (!this._extraPcaBuffer) return new ArrayBuffer(0);
    return this._extraPcaBuffer;
  }
  weightBuffer(): ArrayBuffer {
    if (!this._weightBuffer) return new ArrayBuffer(0);
    return this._weightBuffer;
  }
  numPoints(): number { return this._numPoints; }
  shDegree(): number { return this._shDegree; }
  bbox() { return this._bbox; }
}