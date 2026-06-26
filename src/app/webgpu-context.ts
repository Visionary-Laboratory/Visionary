import * as ort from 'onnxruntime-web/webgpu';

// 默认 dummy 模型的 URL（打包时随包发布）
// 注意：visionary 开源版无 public/ 目录，根路径 '/models/...' 在生产构建中无法解析，
// 故使用 bundler 友好的 import.meta.url 形式（vite 会哈希并打包该资源）。
export const DEFAULT_DUMMY_MODEL_URL = new URL('../../models/onnx_dummy.onnx', import.meta.url).toString();
// export const DEFAULT_DUMMY_MODEL_URL = '/models/onnx_dummy.onnx';

export async function initWebGPU(canvas: HTMLCanvasElement) {
  if (!navigator.gpu) throw new Error('WebGPU not supported');

  // 1) 先拿 adapter，并检查 f16 能力
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter found');

  const want: GPUFeatureName[] = [];
  if (adapter.features.has('shader-f16')) want.push('shader-f16');
  else {
    // 你“必须用 f16”的场景下，直接显式失败；如果想兜底到 f32，这里改成返回 null / 走回退
    throw new Error('Adapter does not support shader-f16');
  }





  const L = adapter.limits;
  const maxStorageBuffers = (L as any).maxStorageBuffersPerShaderStage ?? 8;
  const requestedStorageBuffers = Math.min(10, maxStorageBuffers);
  if (requestedStorageBuffers < 9) {
    console.warn(
      `[WebGPU] maxStorageBuffersPerShaderStage=${requestedStorageBuffers} < 9; preprocess pipeline may fail on this adapter.`
    );
  }

  const requiredLimits: GPUDeviceDescriptor["requiredLimits"] = {
    maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
    maxBufferSize: (L as any).maxBufferSize ?? L.maxStorageBufferBindingSize,
    maxComputeWorkgroupStorageSize: Math.min(32768, L.maxComputeWorkgroupStorageSize),
    maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: L.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: L.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: L.maxComputeWorkgroupSizeZ,
    maxStorageBuffersPerShaderStage: requestedStorageBuffers,
  } as any;

  const device = await adapter.requestDevice({
    requiredFeatures: want,
    requiredLimits,
  });


  // // 2) 由你来 requestDevice，把 f16 feature 打开
  // const device = await adapter.requestDevice({ requiredFeatures: want });
  console.log('[WebGPU] shader-f16 enabled on device?', device.features.has('shader-f16'));

  // 3) **在创建任何 ORT Session 之前**注入 adapter/device（关键！）
  (ort.env as any).webgpu = (ort.env as any).webgpu || {};
  (ort.env as any).webgpu.adapter = adapter;
  (ort.env as any).webgpu.device  = device;

  // 4) 现在用同一个 device 配置你的 canvas
  const context = canvas.getContext('webgpu')!;
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'premultiplied' });

  // 5) （可选）立即构造一个“空模型” session，强制 ORT 绑定到这个 device（避免后续懒创建）
  // const bytes = await (await fetch('/models/onnx_dummy.onnx')).arrayBuffer();
  // await ort.InferenceSession.create(bytes, { executionProviders: ['webgpu'] });

  // 6) 断言 ORT 的 device 确实就是我们注入的这一个
  const ortDev = await (ort.env as any).webgpu.device;
  if (ortDev !== device) throw new Error('ORT device mismatch after injection');

  return { device, context, format };
}



/**
 * WebGPU initialization and context management — ORT-friendly (strict) version
 *
 * Key changes vs previous draft:
 * 1) **Never** create our own GPUDevice if onnxruntime-web (ORT) is present and we intend to share.
 *    This avoids the classic "two devices" race where ORT lazily creates a different device later.
 * 2) Provide a strict dummy-session path to force ORT to materialize its device.
 * 3) Robust dynamic import + HMR-safe singleton; no silent env.webgpu.device setters.
 * 4) Helpful diagnostics & a runtime assertion to verify device sharing.
 */



export interface WebGPUContext {
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
}

export interface WebGPUInitOptions {
  /** Prefer to share the device with onnxruntime-web if available (default: true). */
  preferShareWithOrt?: boolean;
  /**
   * When ORT is present and sharing is preferred, attempt to force-init ORT's device by
   * spinning up a tiny dummy session from this URL if needed. Example: "/models/onnx_dummy.onnx".
   */
  dummyModelUrl?: string | null;
  /** Pass-through to navigator.gpu.requestAdapter */
  adapterPowerPreference?: GPURequestAdapterOptions["powerPreference"];
  /**
   * If ORT is present but we still cannot obtain its device, should we fall back to our own device?
   * - false (default): throw with a clear error so you don't accidentally end up with two devices later.
   * - true: proceed with our own device, but you must ensure the app will not use ORT afterward.
   */
  allowOwnDeviceWhenOrtPresent?: boolean;
}

/**
 * Single place to (soft) import onnxruntime-web and memoize it across HMR/reloads.
 */
async function getOrtModule(): Promise<null | typeof import("onnxruntime-web/webgpu")> {
  const g = globalThis as any;
  if (g.__ORT_WEBGPU_SINGLETON__) return g.__ORT_WEBGPU_SINGLETON__;
  try {
    const mod = await import("onnxruntime-web/webgpu");
    g.__ORT_WEBGPU_SINGLETON__ = mod;
    return mod;
  } catch {
    return null; // ORT not installed / not available
  }
}

/** Try to create ORT's device by creating a tiny dummy session (strict path). */
async function createOrtDummySession(dummyModelUrl: string): Promise<void> {
  const ort = await getOrtModule();



  if (!ort) return;
  ort.env.wasm.numThreads = 1;
  ort.env.logLevel = 'warning';

  // ort.env.debug = true;
  //ort.env.logLevel = 'verbose'
  // 确保WASM路径已设置
  if (!ort.env.wasm.wasmPaths) {
    ort.env.wasm.wasmPaths = '/src/ort/';
    console.log('[WebGPU] Setting default WASM paths:', ort.env.wasm.wasmPaths);
  } else {
    console.log('[WebGPU] Using existing WASM paths:', ort.env.wasm.wasmPaths);
  }

  const resp = await fetch(dummyModelUrl);
  if (!resp.ok) throw new Error(`[ORT] Failed to fetch dummy model: ${dummyModelUrl}`);
  const bytes = await resp.arrayBuffer();
  await ort.InferenceSession.create(bytes, { executionProviders: ["webgpu"] });
}

/** Obtain ORT's GPUDevice, optionally forcing initialization via a dummy session. */
async function obtainOrtDevice(opts: { adapter?: GPUAdapter | null; dummyModelUrl?: string | null }): Promise<GPUDevice | null> {
  const ort = await getOrtModule();
  if (!ort) return null;

  // First, check if device already exists - if so, return it immediately
  // This avoids trying to set adapter when it's already read-only
  try {
    const webgpuEnv = (ort.env as any).webgpu;
    if (webgpuEnv) {
      const existingDevice = webgpuEnv.device;
      if (existingDevice) {
        const dev = existingDevice instanceof Promise ? await existingDevice : existingDevice;
        if (dev) {
          console.log('[WebGPU] Reusing existing ORT device from obtainOrtDevice');
          // If adapter exists and is the same, we're good
          // If adapter is read-only or different, we can't change it, but that's OK if device is valid
          return dev;
        }
      }
    }
  } catch (e) {
    // Continue to initialization path
    console.warn('[WebGPU] Could not check existing ORT device:', e);
  }

  // If an adapter is provided, supply it to ORT **before** any session creation.
  // But only if it hasn't been set yet (to avoid "read only property" errors when creating multiple scenes)
  if (opts.adapter) {
    try {
      const webgpuEnv = (ort.env as any).webgpu || {};
      (ort.env as any).webgpu = webgpuEnv;
      
      // Check if adapter is already set and accessible
      const existingAdapter = webgpuEnv.adapter;
      
      // Only set adapter if it's not already set (or if it's different)
      if (!existingAdapter) {
        try {
          webgpuEnv.adapter = opts.adapter;
        } catch (e) {
          // If adapter is read-only, try to continue with existing setup
          console.warn('[WebGPU] Could not set adapter (may be read-only):', e);
        }
      } else if (existingAdapter !== opts.adapter) {
        // Adapter exists but is different - try to update only if it's writable
        try {
          // Check if property descriptor shows it's writable
          const descriptor = Object.getOwnPropertyDescriptor(webgpuEnv, 'adapter');
          if (descriptor && descriptor.writable !== false) {
            webgpuEnv.adapter = opts.adapter;
          } else {
            console.warn('[WebGPU] Adapter is read-only, keeping existing adapter');
          }
        } catch (e) {
          // If setting fails, that's OK - existing adapter should work
          console.warn('[WebGPU] Could not update adapter (may be read-only, which is OK):', e);
        }
      }
      // If adapter is the same, no action needed
    } catch (e) {
      // If we can't access webgpu env at all, try to continue
      console.warn('[WebGPU] Could not access ORT webgpu environment:', e);
    }
  }

  // Prefer creating our own device with requiredLimits (e.g. maxStorageBuffersPerShaderStage >= 9)
  // and injecting into ORT. The patched onnxruntime-web backend will use env.webgpu.device when set,
  // so it will not call adapter.requestDevice() again (avoids "adapter is consumed").
  if (opts.adapter) {
    try {
      const webgpuEnv = (ort.env as any).webgpu || {};
      (ort.env as any).webgpu = webgpuEnv;

      if (!webgpuEnv.device) {
        const device = await createOwnDevice(opts.adapter);
        webgpuEnv.device = device;
        if (!webgpuEnv.adapter) webgpuEnv.adapter = opts.adapter;
        console.log('[WebGPU] Created app-owned device with requiredLimits; ORT backend will use it from env.webgpu.device.');
        return device;
      }
    } catch (e) {
      console.warn('[WebGPU] Failed to create shared device for ORT:', e);
    }
  }

  // helen add 0206
  // Fallback: let ORT create the device via dummy session (requires dummyModelUrl).
  if (opts.adapter && opts.dummyModelUrl) {
    try {
      const webgpuEnv = (ort.env as any).webgpu || {};
      (ort.env as any).webgpu = webgpuEnv;
      if (!webgpuEnv.adapter) webgpuEnv.adapter = opts.adapter;
      await createOrtDummySession(opts.dummyModelUrl);
      const dev = (ort.env as any).webgpu?.device;
      if (dev) {
        const finalDev = dev instanceof Promise ? await dev : dev;
        if (finalDev) {
          console.log('[WebGPU] ORT created device via dummy session.');
          return finalDev;
        }
      }
    } catch (e) {
      console.warn('[WebGPU] Failed to obtain device via ORT dummy session:', e);
    }
  }

  // Reuse existing ORT device if set (e.g. by a previous init).
  try {
    const webgpuEnv = (ort.env as any).webgpu;
    if (webgpuEnv) {
      const devPromise = webgpuEnv.device;
      if (devPromise) {
        const dev = devPromise instanceof Promise ? await devPromise : devPromise;
        if (dev) return dev;
      }
    }
  } catch (e) {
    // Continue to dummy path
    console.warn('[WebGPU] Could not get device from ORT env:', e);
  }

  // ********************helen add 0206******************************************** */
  // Strict fallback: force-init via dummy session if URL provided
  // if (opts.dummyModelUrl) {
  //   try {
  //     await createOrtDummySession(opts.dummyModelUrl);
  //     const ort2 = await getOrtModule();
  //     if (ort2) {
  //       try {
  //         const dev = (ort2.env as any).webgpu?.device;
  //         if (dev) {
  //           const finalDev = dev instanceof Promise ? await dev : dev;
  //           if (finalDev) return finalDev;
  //         }
  //       } catch {/* ignore */}
  //     }
  //   } catch (e) {
  //     console.warn('[WebGPU] Failed to create dummy session:', e);
  //   }
  // }
//****************************************************************************************** */
  return null;
}

/** Request our own GPUDevice from a chosen adapter, with sensible features/limits. */
async function createOwnDevice(adapter: GPUAdapter): Promise<GPUDevice> {
  const want: GPUFeatureName[] = [];
  if (adapter.features.has("shader-f16")) want.push("shader-f16");
  if (adapter.features.has("timestamp-query")) want.push("timestamp-query");
  if (adapter.features.has("chromium-experimental-timestamp-query-inside-passes")) {
    want.push("chromium-experimental-timestamp-query-inside-passes" as GPUFeatureName);
  }

  const L = adapter.limits;
  const maxStorageBuffers = (L as any).maxStorageBuffersPerShaderStage ?? 8;
  const requestedStorageBuffers = Math.min(10, maxStorageBuffers);
  if (requestedStorageBuffers < 9) {
    console.warn(
      `[WebGPU] maxStorageBuffersPerShaderStage=${requestedStorageBuffers} < 9; preprocess pipeline may fail on this adapter.`
    );
  }

  const requiredLimits: GPUDeviceDescriptor["requiredLimits"] = {
    maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
    maxBufferSize: (L as any).maxBufferSize ?? L.maxStorageBufferBindingSize,
    maxComputeWorkgroupStorageSize: Math.min(32768, L.maxComputeWorkgroupStorageSize),
    maxComputeInvocationsPerWorkgroup: L.maxComputeInvocationsPerWorkgroup,
    maxComputeWorkgroupSizeX: L.maxComputeWorkgroupSizeX,
    maxComputeWorkgroupSizeY: L.maxComputeWorkgroupSizeY,
    maxComputeWorkgroupSizeZ: L.maxComputeWorkgroupSizeZ,
    maxStorageBuffersPerShaderStage: requestedStorageBuffers,
  } as any;

  const device = await adapter.requestDevice({
    requiredFeatures: want,
    requiredLimits,
  });

  if (!device.label) device.label = "app-device";
  return device;
}

/**
 * Runtime assertion helper to verify we're truly sharing the same device with ORT.
 * Call it once after init in dev builds.
 */
export async function assertSharedWithOrt(appDevice: GPUDevice): Promise<void> {
  const ort = await getOrtModule();
  if (!ort) {
    console.warn("[assertSharedWithOrt] ORT not present; nothing to verify.");
    return;
  }
  try {
    const ortDev = await (ort.env as any).webgpu.device as GPUDevice;
    if (ortDev && ortDev !== appDevice) {
      console.error("[assertSharedWithOrt] MISMATCH: appDevice !== ortDevice", { appDevice, ortDev });
      throw new Error("WebGPU device mismatch between app and ORT.");
    }
    console.log("[assertSharedWithOrt] OK: app shares the same device with ORT.");
  } catch (e) {
    console.error("[assertSharedWithOrt] Failed to obtain ORT device:", e);
  }
}

/**
 * Initialize WebGPU canvas + device, sharing one device with ORT and Three.js.
 *
 * Strategy:
 * 1) Get adapter
 * 2) If preferShareWithOrt && ORT is available: create our own device with requiredLimits
 *    (e.g. maxStorageBuffersPerShaderStage >= 9 for Gaussian preprocess), inject into ort.env.webgpu
 * 3) If ORT absent: create our own device the same way
 * 4) Configure canvas and return context. ONNX session creation must pass this device in EP options.
 */
export async function initWebGPU_onnx(
  canvas: HTMLCanvasElement,
  opts: WebGPUInitOptions = {}
): Promise<WebGPUContext | null> {
  if (!navigator.gpu) {
    console.error("WebGPU not supported in this environment.");
    return null;
  }

  const ort = await getOrtModule();
  let device: GPUDevice | null = null;
  let adapter: GPUAdapter | null = null;

  // First, try to get existing ORT device (for subsequent scene initializations)
  if (opts.preferShareWithOrt !== false && ort) {
    try {
      const webgpuEnv = (ort.env as any).webgpu;
      if (webgpuEnv) {
        // Check if device already exists (from previous initialization)
        const existingDevice = webgpuEnv.device;
        if (existingDevice) {
          // If it's a Promise, await it
          device = existingDevice instanceof Promise ? await existingDevice : existingDevice;
          if (device) {
            const maxSB = (device.limits as any).maxStorageBuffersPerShaderStage ?? 8;
            if (maxSB < 9) {
              console.warn(`[WebGPU] Existing ORT device maxStorageBuffersPerShaderStage=${maxSB} < 9; ignore and re-create.`);
              device = null;
            } else {
              console.log('[WebGPU] Reusing existing ORT device for new canvas');
              // Try to get the adapter from ORT env (if available)
              adapter = webgpuEnv.adapter;
            }
          }
        }
      }
    } catch (e) {
      console.warn('[WebGPU] Could not get existing ORT device:', e);
    }
  }

  // If no existing device, get adapter and initialize
  if (!device) {
    adapter = await navigator.gpu.requestAdapter({
      powerPreference: opts.adapterPowerPreference,
    });
    if (!adapter) throw new Error("No WebGPU adapter found");

    if (opts.preferShareWithOrt !== false && ort) {
      device = await obtainOrtDevice({ adapter, dummyModelUrl: opts.dummyModelUrl ?? null });
      if (!device) {
        const allow = !!opts.allowOwnDeviceWhenOrtPresent;
        const msg = "[WebGPU init] ORT detected but failed to obtain its device. " +
                    (allow ? "Proceeding with app-owned device (do NOT use ORT later)." :
                               "Refusing to create a separate device to avoid future mismatch." +
                               "- Provide a valid dummyModelUrl, ORdisable preferShareWithOrt, ORensure a single ORT import.");
        console.warn(msg);
        if (!allow) throw new Error("ORT present but cannot acquire ORT device (strict mode)");
      } else if (device && !adapter) {
        // If we got device from ORT but no adapter, try to get it from ORT env
        try {
          const webgpuEnv = (ort.env as any)?.webgpu;
          adapter = webgpuEnv?.adapter;
        } catch {}
      }
    }

    // If no device yet (no ORT or allowed fallback), create our own
    if (!device) {
      // Ensure adapter is available before creating device
      // Adapter should already be set at line 343, but TypeScript needs this check
      if (!adapter) {
        adapter = await navigator.gpu.requestAdapter({
          powerPreference: opts.adapterPowerPreference,
        });
        if (!adapter) throw new Error("No WebGPU adapter found");
      }
      // At this point, adapter is guaranteed to be non-null (type narrowing)
      device = await createOwnDevice(adapter);
    }
  } else if (!adapter) {
    // If we have device but no adapter, try to get it from ORT or create a new one
    try {
      const webgpuEnv = (ort?.env as any)?.webgpu;
      adapter = webgpuEnv?.adapter;
    } catch {}
    
    if (!adapter) {
      adapter = await navigator.gpu.requestAdapter({
        powerPreference: opts.adapterPowerPreference,
      });
    }
  }

  device.pushErrorScope('out-of-memory');
  device.pushErrorScope('validation');
  // 运行一次
  await device.popErrorScope().then(e => console.warn('validation:', e));
  await device.popErrorScope().then(e => console.warn('oom:', e));
  device.lost.then(info => console.error('device lost:', info.message, info.reason));


  const context = canvas.getContext("webgpu");
  if (!context) throw new Error("Failed to get WebGPU canvas context");

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: "premultiplied" });

  console.log(`[WebGPU] initialized. format=${format}, sharedWithORT=${!!(ort && opts.preferShareWithOrt !== false)}`);
  return { device, context, format };
}
