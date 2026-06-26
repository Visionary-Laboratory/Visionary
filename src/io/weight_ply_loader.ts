import { ILoader, LoadingOptions } from './index';

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

export class WeightPLYData {
  private _weights: ArrayBuffer;
  private _numPoints: number;
  private _channels: number;
  // Weight-only PLY has no positions; provide a default bbox to satisfy DataSource contract.
  private _bbox: { min: [number, number, number]; max: [number, number, number] } = {
    min: [0, 0, 0],
    max: [0, 0, 0],
  };

  constructor(data: { weightsBuffer: ArrayBuffer; numPoints: number; channels?: number }) {
    this._weights = data.weightsBuffer;
    this._numPoints = data.numPoints;
    this._channels = data.channels ?? 64;
  }

  weightsBuffer(): ArrayBuffer { return this._weights; }
  numPoints(): number { return this._numPoints; }
  weightChannels(): number { return this._channels; }
  bbox(): { min: [number, number, number]; max: [number, number, number] } { return this._bbox; }
}

/**
 * Loader for weight-only PLY files (expects weight_0..weight_63 float32 properties).
 */
export class WeightPLYLoader implements ILoader<WeightPLYData> {
  async loadFile(file: File, options?: LoadingOptions): Promise<WeightPLYData> {
    const buffer = await file.arrayBuffer();
    return this.loadBuffer(buffer, options);
  }

  async loadUrl(url: string, options?: LoadingOptions): Promise<WeightPLYData> {
    const response = await fetch(url, { signal: options?.signal });
    if (!response.ok) {
      throw new Error(`Failed to fetch weight PLY: ${response.status} ${response.statusText}`);
    }
    const buffer = await response.arrayBuffer();
    return this.loadBuffer(buffer, options);
  }

  async loadBuffer(buffer: ArrayBuffer, options?: LoadingOptions): Promise<WeightPLYData> {
    const progress = (stage: string, progress: number, message?: string) => {
      options?.onProgress?.({ stage, progress, message });
    };

    progress('Parsing PLY header', 0.1);
    const header = this.parseHeader(buffer);

    progress('Parsing vertex data', 0.2);
    const data = this.parseVertices(buffer, header);

    progress('Processing weights', 0.5);
    const result = this.processWeights(header, data, progress);

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

  private processWeights(
    header: PlyHeader,
    data: VertexData,
    progress?: (stage: string, progress: number, message?: string) => void
  ): WeightPLYData {
    const weightIndices = new Array(64).fill(-1);
    for (let i = 0; i < 64; i++) {
      weightIndices[i] = data.props.indexOf(`weight_${i}`);
    }

    const hasWeights = weightIndices.every((idx) => idx >= 0);
    if (!hasWeights) {
      throw new Error("Weight PLY missing required properties weight_0..weight_63");
    }

    const N = header.vertices;
    const weights = new Float32Array(N * 64);

    for (let i = 0; i < N; i++) {
      if (i % 10000 === 0) {
        progress?.('Processing weights', 0.5 + 0.4 * (i / N), `${i}/${N} points`);
      }
      const row = data.rows(i);
      const base64 = i * 64;
      for (let w = 0; w < 64; w++) {
        weights[base64 + w] = row[weightIndices[w]];
      }
    }

    return new WeightPLYData({
      weightsBuffer: weights.buffer,
      numPoints: N,
      channels: 64,
    });
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

