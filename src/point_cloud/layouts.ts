// GPU buffer layout definitions and bind group management

/**
 * WebGPU bind group layout cache to avoid recreating layouts
 */
const bindGroupLayoutCache = new WeakMap<GPUDevice, GPUBindGroupLayout>();
const renderBindGroupLayoutCache = new WeakMap<GPUDevice, GPUBindGroupLayout>();

/**
 * Create or retrieve cached bind group layout for point cloud compute shaders
 */
export function getBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
  const cached = bindGroupLayoutCache.get(device);
  if (cached) return cached;

  const layout = device.createBindGroupLayout({
    label: "point cloud bind group layout",
    entries: [
      // 0: gaussians storage
      { binding: 0, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" } },
      // 1: SH coeffs storage
      { binding: 1, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" } },
      // 2: 2D splat buffer (output/indirect)
      { binding: 2, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" } },
      // 3: uniforms
      { binding: 3, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" } },
      // 4: optional extra PCA buffer
      { binding: 4, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" } },
    ],
  });

  bindGroupLayoutCache.set(device, layout);
  return layout;
}

/**
 * Create or retrieve cached bind group layout for point cloud render passes
 */
export function getRenderBindGroupLayout(device: GPUDevice): GPUBindGroupLayout {
  const cached = renderBindGroupLayoutCache.get(device);
  if (cached) return cached;

  const layout = device.createBindGroupLayout({
    label: "Point Cloud Render Bind Group Layout",
    entries: [
      // Keep these entries EXACTLY matching your render shader's @group(0) expectations
      { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      // If you later add more bindings, append them here:
      // { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      // { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "storage" } },
      // { binding: 3, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
    ],
  });

  renderBindGroupLayoutCache.set(device, layout);
  return layout;
}

/**
 * GPU buffer configuration constants
 */
export const BUFFER_CONFIG = {
  SPLAT_STRIDE: 32, // conservative placeholder; match WGSL struct size
} as const;