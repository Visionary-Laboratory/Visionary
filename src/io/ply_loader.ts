import { f32_to_f16, makeCopySH_PackedF16 } from '../utils/half';
import { buildCov, sigmoid, shDegreeFromNumCoeffs } from '../utils';
import { planeFromPoints } from '../utils/vector-math';
import { quat } from 'gl-matrix';
import { ILoader, LoadingOptions } from './index';
import { PLYGaussianData } from './GaussianData';

// Helper Types
type PlyFormat = "ascii" | "binary_little_endian" | "binary_big_endian";

interface PlyHeader {
  format: PlyFormat;
  vertices: number;
  props: string[];
  headerByteLength: number;
}

interface VertexData {
  props: string[];
  rows: (i: number) => number[];
}

/**
 * PLY file format loader for 3D Gaussian Splatting data
 */
export class PLYLoader implements ILoader<PLYGaussianData> {
  
  async loadFile(file: File, options?: LoadingOptions): Promise<PLYGaussianData> {
    const buffer = await file.arrayBuffer();
    return this.loadBuffer(buffer, options);
  }
  
  async loadUrl(url: string, options?: LoadingOptions): Promise<PLYGaussianData> {
    const response = await fetch(url, { signal: options?.signal });
    if (!response.ok) {
      throw new Error(`Failed to fetch PLY file: ${response.status} ${response.statusText}`);
    }
    
    const buffer = await response.arrayBuffer();
    return this.loadBuffer(buffer, options);
  }
  
  async loadBuffer(buffer: ArrayBuffer, options?: LoadingOptions): Promise<PLYGaussianData> {
    const progress = (stage: string, progress: number, message?: string) => {
      options?.onProgress?.({ stage, progress, message });
    };
    
    progress('Parsing PLY header', 0.1);
    const header = this.parseHeader(buffer);
    
    progress('Parsing vertex data', 0.2);
    const data = this.parseVertices(buffer, header);
    
    progress('Processing Gaussian data', 0.4);
    const result = this.processGaussianData(header, data, progress);
    
    progress('Complete', 1.0);
    return result;
  }
  
  canHandle(filename: string, mimeType?: string): boolean {
    return filename.toLowerCase().endsWith('.ply') || 
           mimeType === 'application/octet-stream';
  }
  
  getSupportedExtensions(): string[] {
    return ['.ply'];
  }
  
  // ========== Private Implementation ==========
  
  private processGaussianData(
    header: PlyHeader, 
    data: VertexData, 
    progress?: (stage: string, progress: number, message?: string) => void
  ): PLYGaussianData {
    
    const dcCount = 3; 
    const restCount = data.props.filter((p) => p.startsWith("f_rest_")).length;
    const totalPerPoint = dcCount + restCount; 
    const perChannel = totalPerPoint / 3;
    const L = shDegreeFromNumCoeffs(perChannel) ?? 0;

    const fieldIndices = this.getFieldIndices(data.props);
    this.validateRequiredFields(fieldIndices);

    const N = header.vertices;
    const GAUSS_STRIDE = 10; 
    const gaussHalf = new Uint16Array(N * GAUSS_STRIDE);

    const SH_FIXED_STRIDE_U32 = 24; 
    const shPacked = new Uint32Array(N * SH_FIXED_STRIDE_U32);

    // Optional extra PCA color (0..1) as vec4<f32> aligned buffer (r,g,b,0)
    const hasExtraPca = fieldIndices.iExtraPcaR >= 0 && fieldIndices.iExtraPcaG >= 0 && fieldIndices.iExtraPcaB >= 0;
    const extraPca = hasExtraPca ? new Float32Array(N * 4) : null;

    const weightIndices = fieldIndices.weightIndices;
    const hasAnyWeight = weightIndices.some((idx) => idx >= 0);
    const hasWeights = weightIndices.every((idx) => idx >= 0);
    if (hasAnyWeight && !hasWeights) {
      throw new Error("PLY has partial weight properties (expected weight_0..weight_63)");
    }
    const weights = hasWeights ? new Float32Array(N * 64) : null;
    
    const gaussFloat: [number, number, number][] = [];
    
    const min: [number, number, number] = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
    const max: [number, number, number] = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];

    const { copySH, wordsPerPoint } = makeCopySH_PackedF16({
      props: data.props, 
      iDC0: fieldIndices.iDC0, 
      iDC1: fieldIndices.iDC1, 
      iDC2: fieldIndices.iDC2, 
      k: 3, 
      shU32: shPacked
    });

    progress?.('Processing Gaussians', 0.5);
    for (let i = 0; i < N; i++) {
      if (i % 10000 === 0) {
        progress?.('Processing Gaussians', 0.5 + 0.3 * (i / N), `${i}/${N} points`);
      }
      
      const row = data.rows(i);
      const gaussian = this.processGaussian(row, fieldIndices, i, GAUSS_STRIDE, gaussHalf);
      
      gaussFloat.push([gaussian.x, gaussian.y, gaussian.z]);
      copySH(i * wordsPerPoint, row, false);

      if (extraPca) {
        const base4 = i * 4;
        extraPca[base4 + 0] = row[fieldIndices.iExtraPcaR];
        extraPca[base4 + 1] = row[fieldIndices.iExtraPcaG];
        extraPca[base4 + 2] = row[fieldIndices.iExtraPcaB];
        extraPca[base4 + 3] = 0.0;
      }

      if (weights) {
        const base64 = i * 64;
        for (let w = 0; w < 64; w++) {
          weights[base64 + w] = row[weightIndices[w]];
        }
      }
      
      if (gaussian.x < min[0]) min[0] = gaussian.x;
      if (gaussian.y < min[1]) min[1] = gaussian.y;
      if (gaussian.z < min[2]) min[2] = gaussian.z;
      if (gaussian.x > max[0]) max[0] = gaussian.x;
      if (gaussian.y > max[1]) max[1] = gaussian.y;
      if (gaussian.z > max[2]) max[2] = gaussian.z;
    }
    
    progress?.('Computing scene geometry', 0.8);
    const { centroid, normal } = this.computeSceneGeometry(gaussFloat);
    
    const center: [number, number, number] = [centroid[0], centroid[1], centroid[2]];
    const up: [number, number, number] = normal ? [normal[0], normal[1], normal[2]] : [1.0, 0.0, 0.0];
    
    return new PLYGaussianData({
      gaussianBuffer: gaussHalf.buffer,
      shCoefsBuffer: shPacked.buffer,
      extraPcaBuffer: extraPca ? extraPca.buffer : undefined,
      weightBuffer: weights ? weights.buffer : undefined,
      numPoints: N,
      shDegree: L,
      bbox: { min, max },
      center,
      up,
      mipSplatting: undefined,
      kernelSize: undefined,
      backgroundColor: undefined,
    });
  }
  
  private getFieldIndices(props: string[]) {
    const weightIndices = new Array(64).fill(-1);
    for (let i = 0; i < 64; i++) {
      weightIndices[i] = props.indexOf(`weight_${i}`);
    }
    return {
      ix: props.indexOf("x"),
      iy: props.indexOf("y"),
      iz: props.indexOf("z"),
      iOpacity: props.indexOf("opacity"),
      iS0: props.indexOf("scale_0"),
      iS1: props.indexOf("scale_1"),
      iS2: props.indexOf("scale_2"),
      iR0: props.indexOf("rot_0"),
      iR1: props.indexOf("rot_1"),
      iR2: props.indexOf("rot_2"),
      iR3: props.indexOf("rot_3"),
      iDC0: props.indexOf("f_dc_0"),
      iDC1: props.indexOf("f_dc_1"),
      iDC2: props.indexOf("f_dc_2"),
      iExtraPcaR: props.indexOf("extra_pca_r"),
      iExtraPcaG: props.indexOf("extra_pca_g"),
      iExtraPcaB: props.indexOf("extra_pca_b"),
      weightIndices,
    };
  }
  
  private validateRequiredFields(indices: ReturnType<typeof this.getFieldIndices>) {
    const required = ['ix', 'iy', 'iz', 'iOpacity', 'iS0', 'iS1', 'iS2', 'iR0', 'iR1', 'iR2', 'iR3', 'iDC0', 'iDC1', 'iDC2'];
    for (const field of required) {
      if ((indices as any)[field] < 0) {
        throw new Error(`PLY missing required field: ${field.slice(1)}`);
      }
    }
  }
  
  private processGaussian(
    row: number[], 
    indices: ReturnType<typeof this.getFieldIndices>,
    pointIndex: number,
    stride: number,
    output: Uint16Array
  ) {
    const px = row[indices.ix];
    const py = row[indices.iy]; 
    const pz = row[indices.iz];
    const opacity = sigmoid(row[indices.iOpacity]);
    
    const s: [number, number, number] = [
      Math.exp(row[indices.iS0]), 
      Math.exp(row[indices.iS1]), 
      Math.exp(row[indices.iS2])
    ];
    
    const q = quat.fromValues(row[indices.iR1], row[indices.iR2], row[indices.iR3], row[indices.iR0]);
    quat.normalize(q, q);
    
    const [m00, m01, m02, m11, m12, m22] = buildCov(q as any, new Float32Array(s) as any);
    
    const base = pointIndex * stride;
    output[base + 0] = f32_to_f16(px);
    output[base + 1] = f32_to_f16(py);
    output[base + 2] = f32_to_f16(pz);
    output[base + 3] = f32_to_f16(opacity);
    output[base + 4] = f32_to_f16(m00);
    output[base + 5] = f32_to_f16(m01);
    output[base + 6] = f32_to_f16(m02);
    output[base + 7] = f32_to_f16(m11);
    output[base + 8] = f32_to_f16(m12);
    output[base + 9] = f32_to_f16(m22);
    
    return { x: px, y: py, z: pz };
  }
  
  private computeSceneGeometry(points: [number, number, number][]) {
    return planeFromPoints(points);
  }
  
  private parseHeader(buffer: ArrayBuffer): PlyHeader {
    const text = new TextDecoder().decode(buffer.slice(0, Math.min(1 << 20, buffer.byteLength)));
    const lines = text.split(/\r?\n/);

    if (!/^ply\b/.test(lines[0])) throw new Error("Not a PLY file");

    let format: PlyFormat | null = null;
    let vertices = 0;
    const props: string[] = [];
    let headerEndOffset = 0;
    let inVertex = false;

    for (let i = 1; i < lines.length; i++) {
      const line = lines[i];

      if (line === "end_header") {
        let pos = text.indexOf("end_header");
        if (pos < 0) throw new Error("Malformed PLY: missing end_header");
        const nl = text.indexOf("\n", pos + "end_header".length);
        headerEndOffset = (nl >= 0 ? nl + 1 : pos + "end_header".length + 1);
        break;
      }

      if (line.startsWith("format ")) {
        const f = line.split(/\s+/)[1];
        if (f === "ascii" || f === "binary_little_endian" || f === "binary_big_endian") {
          format = f as PlyFormat;
        } else {
          throw new Error(`Unsupported PLY format: ${f}`);
        }
      } else if (line.startsWith("element ")) {
        inVertex = line.startsWith("element vertex ");
        if (inVertex) vertices = parseInt(line.split(/\s+/)[2], 10);
      } else if (inVertex && line.startsWith("property ")) {
        const parts = line.trim().split(/\s+/);
        if (parts[1] === "list") throw new Error("Unexpected list property in vertex");
        const name = parts[parts.length - 1];
        props.push(name);
      }
    }

    if (!format) throw new Error("PLY header missing format");
    if (vertices <= 0) throw new Error("PLY has no vertices element");

    return { format, vertices, props, headerByteLength: headerEndOffset };
  }
  
  private parseVertices(buffer: ArrayBuffer, header: PlyHeader): VertexData {
    const props = header.props.slice();

    if (header.format === "ascii") {
      return this.parseASCIIVertices(buffer, header, props);
    } else {
      return this.parseBinaryVertices(buffer, header, props);
    }
  }
  
  private parseASCIIVertices(buffer: ArrayBuffer, header: PlyHeader, props: string[]): VertexData {
    const text = new TextDecoder().decode(buffer.slice(header.headerByteLength));
    const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
    
    return {
      props,
      rows: (i: number) => {
        const parts = lines[i].trim().split(/\s+/);
        if (parts.length < props.length) throw new Error("Malformed PLY ASCII row");
        return parts.map(parseFloat);
      },
    };
  }
  
  private parseBinaryVertices(buffer: ArrayBuffer, header: PlyHeader, props: string[]): VertexData {
    const little = header.format === "binary_little_endian";
    const view = new DataView(buffer, header.headerByteLength);
    const stride = props.length * 4; 
    
    return {
      props,
      rows: (i: number) => {
        const base = i * stride;
        const out: number[] = new Array(props.length);
        for (let p = 0; p < props.length; p++) {
          out[p] = view.getFloat32(base + p * 4, little);
        }
        return out;
      },
    };
  }
}