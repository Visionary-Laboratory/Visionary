// GPU utility functions and WebGPU helpers

import { mat3, vec3, quat } from "gl-matrix";

export function align4(n: number): number { 
  return (n + 3) & ~3; 
}

export function align8(n: number): number { 
  return (n + 7) & ~7; 
}

/**
 * Read entire buffer (or a segment from an offset).
 * @param device GPUDevice
 * @param src Source GPUBuffer (must have COPY_SRC usage)
 * @param byteLength Number of bytes to read
 * @param srcOffset Source offset (default 0)
 * @returns Uint8Array (length = byteLength)
 */
export async function readWholeBuffer(
  device: GPUDevice,
  src: GPUBuffer,
  byteLength: number,
  srcOffset = 0
): Promise<Uint8Array> {
  // Constraints: copy must be 4-aligned; mapping must be 8-aligned
  const copySize = align4(byteLength);
  const mapSize = align8(copySize);

  const staging = device.createBuffer({
    size: mapSize,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(src, srcOffset, staging, 0, copySize);
  device.queue.submit([enc.finish()]);

  // Optional but recommended: ensure GPU has completed submitted work
  if ('onSubmittedWorkDone' in device.queue) {
    // @ts-ignore
    await device.queue.onSubmittedWorkDone();
  }

  await staging.mapAsync(GPUMapMode.READ, 0, mapSize);
  const mapped = staging.getMappedRange(0, mapSize);

  // Only take the first byteLength bytes that we actually need
  const out = new Uint8Array(byteLength);
  out.set(new Uint8Array(mapped).subarray(0, byteLength));

  staging.unmap();
  staging.destroy();
  return out;
}

export function dumpU32(bytes: Uint8Array): void {
  const u32 = new Uint32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 4));
  console.log('u32:', Array.from(u32));
}

export function dumpHex(bytes: Uint8Array): void {
  const hex = Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join(' ');
  console.log('hex:', hex);
}

export function keyToNum(code: string): number | undefined {
  switch (code) {
    case "Digit0": return 0;
    case "Digit1": return 1;
    case "Digit2": return 2;
    case "Digit3": return 3;
    case "Digit4": return 4;
    case "Digit5": return 5;
    case "Digit6": return 6;
    case "Digit7": return 7;
    case "Digit8": return 8;
    case "Digit9": return 9;
    default: return undefined;
  }
}

/**
 * GPU stopwatch based on timestamp queries.
 * WebGPU: device must support `timestamp-query` feature.
 */
export class GPUStopwatch {
  private device: GPUDevice;
  private querySet: GPUQuerySet;
  private queryBuffer: GPUBuffer;
  private capacity: number; // number of *pairs* (start, stop)
  private index = 0; // current measurement index
  private labels = new Map<string, number>();

  constructor(device: GPUDevice, capacity?: number) {
    this.device = device;
    const pairs = capacity ?? 64; // choose a sane default if not provided
    this.capacity = pairs;

    this.querySet = device.createQuerySet({
      type: "timestamp",
      count: pairs * 2,
      label: "time stamp query set",
    });

    this.queryBuffer = device.createBuffer({
      size: pairs * 2 * 8, // bytes for u64 timestamps (number, not BigInt)
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_READ,
      mappedAtCreation: false,
      label: "query set buffer",
    });
  }

  /** Begin a measurement region. Must be paired with `stop(label)` */
  start(encoder: GPUCommandEncoder, label: string): void {
    if (this.labels.has(label)) {
      throw new Error("cannot start measurement for same label twice");
    }
    if (this.labels.size >= this.capacity) {
      throw new Error(`query set capacity (${this.capacity}) reached`);
    }
    this.labels.set(label, this.index);
    // position within the query set is index*2 (start)
    (encoder as any).writeTimestamp?.(this.querySet, this.index * 2);
    this.index += 1;
  }

  /** End a measurement region previously started with the same label. */
  stop(encoder: GPUCommandEncoder, label: string): void {
    const idx = this.labels.get(label);
    if (idx === undefined) throw new Error(`start was not yet called for label ${label}`);
    (encoder as any).writeTimestamp?.(this.querySet, idx * 2 + 1);
  }

  /** Resolve all queries for the current frame into the query buffer. */
  end(encoder: GPUCommandEncoder): void {
    // Resolve the full range we allocated; unused timestamps are fine.
    (encoder as any).resolveQuerySet?.(this.querySet, 0, this.capacity * 2, this.queryBuffer, 0);
  }

  /**
   * Read back all started/finished measurements as a map of label -> Duration (nanoseconds).
   * `timestampPeriodNS` is the timestamp period in *nanoseconds per tick*.
   * If omitted, we try `queue.getTimestampPeriod()` (Chrome) and fall back to 1.
   */
  async takeMeasurements(queue: GPUQueue, timestampPeriodNS?: number): Promise<Record<string, number>> {
    const period = timestampPeriodNS ?? (queue as any).getTimestampPeriod?.() ?? 1;

    // Ensure work is done so timestamps are available
    await queue.onSubmittedWorkDone?.();

    await this.queryBuffer.mapAsync(GPUMapMode.READ);
    const range = this.queryBuffer.getMappedRange();
    const timestamps = new BigUint64Array(range);

    const out: Record<string, number> = {};
    for (const [label, idx] of this.labels) {
      const start = timestamps[idx * 2];
      const stop = timestamps[idx * 2 + 1];
      if (start !== undefined && stop !== undefined) {
        const diffTicks = stop - start;
        // Convert to ns; BigInt math then to Number (safe for typical frame timings)
        const ns = Number(diffTicks) * period;
        out[label] = ns; // nanoseconds
      }
    }

    this.queryBuffer.unmap();
    // reset for next frame
    this.labels.clear();
    this.index = 0;
    return out;
  }
}

/** Number of SH coefficients for degree `l` */
export function shNumCoefficients(l: number): number { return (l + 1) * (l + 1); }

/** Inverse of `shNumCoefficients`; returns degree if `n` is a perfect square minus 1 */
export function shDegreeFromNumCoeffs(n: number): number | undefined {
  const sqrt = Math.sqrt(n);
  return Number.isInteger(sqrt) ? (sqrt | 0) - 1 : undefined;
}

/**
 * Build symmetric 3×3 covariance (upper triangular order) from rotation `rot` and diagonal scale `scale`.
 * Matches 3D Gaussian Splatting formulation.
 */
export function buildCov(rot: quat, scale: vec3): [number, number, number, number, number, number] {
  const R = mat3.create();
  mat3.fromQuat(R, rot);
  const S = mat3.fromValues(
    scale[0], 0, 0,
    0, scale[1], 0,
    0, 0, scale[2]
  );
  const L = mat3.create();
  mat3.multiply(L, R, S);
  const Lt = mat3.create();
  mat3.transpose(Lt, L);
  const M = mat3.create();
  mat3.multiply(M, L, Lt);
  // Return upper-triangular elements in the same order as Rust: [m00, m01, m02, m11, m12, m22]
  return [M[0], M[1], M[2], M[4], M[5], M[8]];
}

/** Numerically stable sigmoid */
export function sigmoid(x: number): number {
  if (x >= 0) return 1 / (1 + Math.exp(-x));
  const ex = Math.exp(x);
  return ex / (1 + ex);
}