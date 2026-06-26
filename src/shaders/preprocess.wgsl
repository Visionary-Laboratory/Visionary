// enable f16;
const KERNEL_SIZE:f32 = 0.3;
//const MAX_SH_DEG:u32 = <injected>u;


override SH_LAYOUT_CHANNEL_MAJOR : bool = false;
override USE_RAW_COLOR : bool = false;

const SH_C0:f32 = 0.28209479177387814;

const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);

const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);


struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct Gaussian {
    pos_opacity: array<u32,2>,
    cov: array<u32,3>
}

struct Splat {
     // 4x f16 packed as u32
    v_0: u32, v_1: u32,
    // 2x f16 packed as u32 (NDC x,y)
    pos: u32,
    // NDC z (high precision)
    posz: f32,
    // rgba packed as f16
    color_0: u32,color_1: u32
};

struct DrawIndirect {
    /// The number of gaussians to draw.
    vertex_count: u32,
    /// The number of instances to draw.
    instance_count: atomic<u32>,
    /// The Index of the first vertex to draw.
    base_vertex: u32,
    /// The instance ID of the first instance to draw.
    /// Has to be 0, unless [`Features::INDIRECT_FIRST_INSTANCE`](crate::Features::INDIRECT_FIRST_INSTANCE) is enabled.
    base_instance: u32,
}

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,     // essentially contains the same info as instance_count in DrawIndirect
    padded_size: u32,
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

@group(2) @binding(4)
var<storage, read_write> sort_source_indices : array<u32>;

struct RenderSettings {
    clipping_box_min: vec4<f32>,
    clipping_box_max: vec4<f32>,
    max_sh_deg: u32,
    show_env_map: u32,
    mip_spatting: u32,
    kernel_size: f32,
    walltime: f32,
    scene_extend: f32,
    center: vec3<f32>,
}

override DISCARD_BY_WORLD_TRACE   : bool = false;  // 可选：world-space 近似阈值
override MAX_WORLD_TRACE          : f32  = 0.25;   // 协方差迹上限（单位≈米^2，按你的尺度调）

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

// Shared buffer - read as different types based on gaussDataType
@group(1) @binding(0) 
var<storage,read> gaussians_packed : array<u32>; // Uint32Array backing

@group(1) @binding(1)
var<storage, read> color_buffer : array<u32>; // Uint32Array backing

@group(1) @binding(2) 
var<storage,read_write> points_2d : array<Splat>;

@group(1) @binding(4)
var<storage, read> extra_pca : array<vec4<f32>>;



@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

@group(3) @binding(0)
var<uniform> render_settings: RenderSettings;

// Phase B M1: per-model params
struct ModelParams {
    model: mat4x4<f32>,
    baseOffset: u32,
    num_points: u32,  // Dynamic point count from ONNX (was _pad0)
    gaussianScaling: f32,  // 每个模型的独立高斯缩放参数
    maxShDeg: u32,        // 球谐等级
    kernelSize: f32,      // 二维核大小
    opacityScale: f32,    // 透明度倍数
    cutoffScale: f32,     // 最大像素比例倍数
    rendermode: u32,      // 渲染模式: 0=颜色, 1=法线, 2=深度, 3=PCA
    // 多精度支持
    gaussDataType: u32,   // 0=f32, 1=f16, 2=i8, 3=u8
    colorDataType: u32,
    gaussScale: f32,
    gaussZeroPoint: f32,
    colorScale: f32,
    colorZeroPoint: f32,
}

@group(3) @binding(1)
var<uniform> uModel: ModelParams;






// Helper: read gaussian pos+opacity based on precision
fn read_gaussian_pos_opacity(idx: u32) -> vec4<f32> {
  if (uModel.gaussDataType == 0u) {
    // FP32: 4 consecutive f32s
    let base = idx * 10u;
    return vec4<f32>(
      bitcast<f32>(gaussians_packed[base + 0u]),
      bitcast<f32>(gaussians_packed[base + 1u]),
      bitcast<f32>(gaussians_packed[base + 2u]),
      bitcast<f32>(gaussians_packed[base + 3u])
    );
  } else {
    // FP16: packed as 2 u32s (pos_xy + pos_z_opacity)
    let w0 = gaussians_packed[idx * 5u + 0u];
    let w1 = gaussians_packed[idx * 5u + 1u];
    let a = unpack2x16float(w0);
    let b = unpack2x16float(w1);
    return vec4<f32>(a.x, a.y, b.x, b.y);
  }
}

// Helper: read gaussian covariance (6 floats) based on precision
fn read_gaussian_cov(idx: u32) -> array<f32,6> {
  if (uModel.gaussDataType == 0u) {
    // FP32: 6 consecutive f32s starting at idx*10+4
    let base = idx * 10u + 4u;
    return array<f32,6>(
      bitcast<f32>(gaussians_packed[base + 0u]),
      bitcast<f32>(gaussians_packed[base + 1u]),
      bitcast<f32>(gaussians_packed[base + 2u]),
      bitcast<f32>(gaussians_packed[base + 3u]),
      bitcast<f32>(gaussians_packed[base + 4u]),
      bitcast<f32>(gaussians_packed[base + 5u])
    );
  } else {
    // FP16: packed as 3 u32s
    let a = unpack2x16float(gaussians_packed[idx * 5u + 2u]);
    let b = unpack2x16float(gaussians_packed[idx * 5u + 3u]);
    let c = unpack2x16float(gaussians_packed[idx * 5u + 4u]);
    return array<f32,6>(a.x, a.y, b.x, b.y, c.x, c.y);
  }
}

// ---- 小工具：计算该点的起始 word 下标 ----
fn base_word_of(splat_idx: u32) -> u32 {
    if (USE_RAW_COLOR) {
        // RGB 直接颜色：每点的存储字数取决于颜色精度
        if (uModel.colorDataType == 0u) { // fp32: 3 words
            return splat_idx * 3u;
        } else if (uModel.colorDataType == 1u) { // fp16: 2 words (4 halfs)
            return splat_idx * 2u;
        } else { // int8/uint8: 1 word (4 bytes)
            return splat_idx * 1u;
        }
    }
    // SH：按精度和通道数计算
    if (uModel.colorDataType == 0u) {
      // FP32: 48 channels (degree 3) = 48 words
      return splat_idx * 48u;
    } else {
      // FP16: 48 channels = 48 halfs = 24 words
      return splat_idx * 24u;
    }
}

// 读第 word_idx 个 u32
fn read_word(splat_idx: u32, word_idx: u32) -> u32 {
  return color_buffer[base_word_of(splat_idx) + word_idx];
}

// ===== 读半精“标量”：线性 half 下标（0,1,2,3, ...）=====
fn read_half_at(splat_idx: u32, half_idx: u32) -> f32 {
  let w = read_word(splat_idx, half_idx >> 1u);
  let p = unpack2x16float(w);                 // vec2<f32>，低/高 half
  // 如果 half_idx 是奇数取高位，否则取低位
  return select(p.x, p.y, (half_idx & 1u) == 1u);
}

// ===== 连续颜色：直接读取前三个 half 作为最终 RGB =====
// 读取颜色分量（依据数据类型）
fn read_color_channel(splat_idx: u32, channel_idx: u32) -> f32 {
  if (uModel.colorDataType == 0u) {
    // fp32：每通道 1 word
    let w = read_word(splat_idx, channel_idx);
    return bitcast<f32>(w);
  } else if (uModel.colorDataType == 1u) {
    // fp16：按 half 读取
    return read_half_at(splat_idx, channel_idx);
  } else {
    // int8/uint8：从 word 中提取 8-bit，然后反量化
    let packed = read_word(splat_idx, channel_idx >> 2u);
    let byte_off = (channel_idx & 3u) * 8u;
    let q = extractBits(i32(packed), byte_off, 8u);
    return f32(q) * uModel.colorScale + uModel.colorZeroPoint;
  }
}

fn fetch_rgb_no_sh(splat_idx: u32) -> vec3<f32> {
  return vec3<f32>(
    read_color_channel(splat_idx, 0u),
    read_color_channel(splat_idx, 1u),
    read_color_channel(splat_idx, 2u)
  );
}

fn sh_coef_interleaved(splat_idx: u32, c_idx: u32) -> vec3<f32> {
  // c_idx ∈ [0 .. (deg+1)^2-1]，每个系数 3 个 half 连着
  let h0 = c_idx * 3u;
  return vec3<f32>(
    read_color_channel(splat_idx, h0 + 0u),
    read_color_channel(splat_idx, h0 + 1u),
    read_color_channel(splat_idx, h0 + 2u)
  );
}

// 新的：channel-major（[Rdc,Gdc,Bdc, R1..Rm, G1..Gm, B1..Bm]）
fn sh_coef_channel_major(splat_idx: u32, c_idx: u32) -> vec3<f32> {
  if (c_idx == 0u) {
    // DC
    return vec3<f32>(
      read_color_channel(splat_idx, 0u),
      read_color_channel(splat_idx, 1u),
      read_color_channel(splat_idx, 2u)
    );
  }
  // AC
  let m  = (uModel.maxShDeg + 1u) * (uModel.maxShDeg + 1u) - 1u; // 每通道 AC 数
  let k  = c_idx - 1u;                   // 第 k 个 AC，k ∈ [0..m-1]
  let r  = read_color_channel(splat_idx, 3u + k);
  let g  = read_color_channel(splat_idx, 3u + m + k);
  let b  = read_color_channel(splat_idx, 3u + 2u*m + k);
  return vec3<f32>(r, g, b);
}

// 统一入口：根据布局选择
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
  return select(
    sh_coef_interleaved(splat_idx, c_idx),
    sh_coef_channel_major(splat_idx, c_idx),
    SH_LAYOUT_CHANNEL_MAJOR
  );
}




fn evaluate_sh(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u) ;
    // sh_deg = 0;
    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return result;
}


fn evaluate_color(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    if (USE_RAW_COLOR) {
        // 直接颜色（0..1），不做 +0.5
        return fetch_rgb_no_sh(v_idx);
    } else {
        // 球谐路径：evaluate_sh already adds 0.5 at the end
        return evaluate_sh(dir, v_idx, sh_deg);
    }
}


fn cov_coefs(v_idx: u32) -> array<f32,6> {
    return read_gaussian_cov(v_idx);
}


// normal calculation
fn inverse_sym3(m: mat3x3<f32>) -> mat3x3<f32> {
    // m = [[a,b,c],[b,d,e],[c,e,f]]
    let a = m[0][0]; let b = m[0][1]; let c = m[0][2];
    let d = m[1][1]; let e = m[1][2];
    let f = m[2][2];

    let co00 = d*f - e*e;
    let co01 = c*e - b*f;
    let co02 = b*e - c*d;
    let co11 = a*f - c*c;
    let co12 = c*b - a*e;
    let co22 = a*d - b*b;

    let det = a*co00 + b*co01 + c*co02;
    let eps = 1e-12;
    let inv_det = select(1.0/det, 1.0/eps, abs(det) < eps);

    // 对称：只需填上三角
    var inv = mat3x3<f32>(
        vec3<f32>(co00, co01, co02),
        vec3<f32>(co01, co11, co12),
        vec3<f32>(co02, co12, co22)
    );
    return inv * inv_det;
}

fn smallest_evec_via_power(Sigma_world: mat3x3<f32>) -> vec3<f32> {
    let invS = inverse_sym3(Sigma_world);
    // 选个稳定的初始向量（取列和可以避免退化）
    var v = normalize(invS[0] + invS[1] + invS[2]);
    // 少量迭代即可（3~5 次）
    v = normalize(invS * v);
    v = normalize(invS * v);
    v = normalize(invS * v);
    return v; // 未定向，之后可按相机翻转
}

fn normal_view_dependent(Sigma_world: mat3x3<f32>, cam_world: vec3<f32>, x_world: vec3<f32>) -> vec3<f32> {
    let v = normalize(cam_world - x_world);                // 从点指向相机
    let invS = inverse_sym3(Sigma_world);
    var n = normalize(invS * v);                           // ∝ Σ^{-1} v
    // 使法线朝向相机（可选）
    if (dot(n, v) < 0.0) { n = -n; }
    return n;
}


@compute @workgroup_size(256,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    // Use dynamic point count from ONNX instead of full gaussian array length
    if idx >= uModel.num_points  {
   //     return;
    }
    if idx > 500000  {
       // return;
    }
    let focal = camera.focal;
    let viewport = camera.viewport;
    let pos_op = read_gaussian_pos_opacity(idx);
    let xyz_local = pos_op.xyz;
    let xyz = (uModel.model * vec4<f32>(xyz_local, 1.)).xyz;
    var opacity = pos_op.w * uModel.opacityScale;

    
    // uModel.maxShDeg = 0;

    var camspace = camera.view * vec4<f32>(xyz, 1.);
    let pos2d = camera.proj * camspace;
    let bounds = 1.2 * pos2d.w;
    let z = pos2d.z / pos2d.w;

    if uModel.baseOffset == 0u && idx == 0u {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);   // safety addition to always have an unfull block at the end of the buffer
    }

    // TODO bring back frustrum culling
    // M1: no world-space clipping here to avoid sparse writes
    // if any(xyz < render_settings.clipping_box_min.xyz) || any(xyz > render_settings.clipping_box_max.xyz) { return; }


    // M1: disable frustum culling to keep dense writes
    if z <= 0. || z >= 1. || pos2d.x < -bounds || pos2d.x > bounds || pos2d.y < -bounds || pos2d.y > bounds { return; }

    if (opacity <0.02)
    {
        return;
    }


    if (opacity > 0.98)
    {
      //  return;
    }

    let cov_sparse = cov_coefs(idx);

    
    var scale_mod = 0.;

    scale_mod = 1.0;
    let scaling = uModel.gaussianScaling * scale_mod * 1.0f;

    // --- 1) 局部协方差（保持不变）
    let Sigma_local = mat3x3<f32>(
        cov_sparse[0], cov_sparse[1], cov_sparse[2],
        cov_sparse[1], cov_sparse[3], cov_sparse[4],
        cov_sparse[2], cov_sparse[4], cov_sparse[5]
    ) * scaling * scaling;

    // --- 2) 用模型矩阵的线性部分把协方差从 local 变到 world
    // 注意：WGSL 按列存储，这里取 model 的前三列作为 3x3 线性部分
    let A = mat3x3<f32>(
        uModel.model[0].xyz,  // 第0列
        uModel.model[1].xyz,  // 第1列
        uModel.model[2].xyz   // 第2列
    );
    let Sigma_world = A * Sigma_local * transpose(A);

    // --- 3 你的 J（cam → NDC）的写法可以沿用
    let J = mat3x3<f32>(
        focal.x / camspace.z,  0.0,                         -(focal.x * camspace.x) / (camspace.z * camspace.z),
        0.0,                  -focal.y / camspace.z,        (focal.y * camspace.y) / (camspace.z * camspace.z),
        0.0,                   0.0,                          0.0
    );

    // --- 4) 取 view 的 3x3 线性部分（world → camera）
    let W = transpose(mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz
    ));

    // --- 5) 正确的组合：T = J * V，然后 Σ_ndc = T Σ_world T^T
    let T   = W * J;
    let cov = transpose(T) * Sigma_world * T;



    if (true) {
        let world_trace = Sigma_local[0][0] + Sigma_local[1][1] + Sigma_local[2][2];
        if (world_trace > 1000.0000002) {
            //return;
        }
    }


    let kernel_size = uModel.kernelSize;
    if bool(render_settings.mip_spatting) {
        // according to Mip-Splatting by Yu et al. 2023
        let det_0 = max(1e-6, cov[0][0] * cov[1][1] - cov[0][1] * cov[0][1]);
        let det_1 = max(1e-6, (cov[0][0] + kernel_size) * (cov[1][1] + kernel_size) - cov[0][1] * cov[0][1]);
        var coef = sqrt(det_0 / (det_1 + 1e-6) + 1e-6);

        if det_0 <= 1e-6 || det_1 <= 1e-6 {
            coef = 0.0;
        }
        opacity *= coef;
    }

    //opacity = 0.1;

    let diagonal1 = cov[0][0] + kernel_size;
    let offDiagonal = cov[0][1];
    let diagonal2 = cov[1][1] + kernel_size;

    let mid = 0.5 * (diagonal1 + diagonal2);
    let radius = length(vec2<f32>((diagonal1 - diagonal2) / 2.0, offDiagonal));
    // eigenvalues of the 2D screen space splat
    let lambda1 = mid + radius;
    let lambda2 = max(mid - radius, 0.1);

    let diagonalVector = normalize(vec2<f32>(offDiagonal, lambda1 - diagonal1));
    // scaled eigenvectors in screen space 
    let v1 = sqrt(2.0 * lambda1) * diagonalVector * uModel.cutoffScale;
    let v2 = sqrt(2.0 * lambda2) * vec2<f32>(diagonalVector.y, -diagonalVector.x) * uModel.cutoffScale;

    let v_center = pos2d.xyzw / pos2d.w;

    let camera_pos = camera.view_inv[3].xyz;
    // let dir = normalize(xyz - camera_pos);
    // DEBUG: prepare color var (assigned after store_idx is known)



    let t = uModel.model[3].xyz;

    // --- 世界相机位置
    let cam_world = camera.view_inv[3].xyz;

    // --- 计算 s^2 （等比缩放下三列长度平方相等，取均值更稳）
    let s2 = max(
        1e-12,
        (dot(A[0], A[0]) + dot(A[1], A[1]) + dot(A[2], A[2])) / 3.0
    );

    // --- cam_local = A^{-1} * (cam_world - t) = (A^T / s^2) * (cam_world - t)
    let cam_local = (transpose(A) * (cam_world - t)) / s2;

    // --- 用局部方向评估 SH
    let dir_local = normalize(xyz_local - cam_local);



    var color: vec4<f32>;
    
    
    // color = vec4<f32>(sh_coef(0, 1u),opacity);

    // M1 (revised): use global contiguous index to avoid per-dispatch uniform dependency
    let store_idx = atomicAdd(&sort_infos.keys_size, 1u);
    let global_index = store_idx;
    // 根据渲染模式选择不同的颜色
    if (uModel.rendermode == 0u) {
        // 模式0: 正常颜色 (SH evaluation or direct RGB)
        color = vec4<f32>(
            max(vec3<f32>(0.), evaluate_color(dir_local, idx, uModel.maxShDeg)),
            opacity
        );
    } else if (uModel.rendermode == 1u) {
        // 模式1: 法线可视化（视角无关：最小特征向量，使用自身分量确定符号，避免视角触发翻转）
        var n_world = smallest_evec_via_power(Sigma_world);

        // 选最大幅值分量的符号作为锚点，保证符号在不同视角下保持一致
        let abs_n = abs(n_world);
        if (abs_n.x >= abs_n.y && abs_n.x >= abs_n.z) {
            if (n_world.x < 0.0) { n_world = -n_world; }
        } else if (abs_n.y >= abs_n.z) {
            if (n_world.y < 0.0) { n_world = -n_world; }
        } else {
            if (n_world.z < 0.0) { n_world = -n_world; }
        }

        // 归一化，避免长度漂移/数值噪声；退化时给默认方向
        let n_len = length(n_world);
        if (n_len < 1e-8) {
            n_world = vec3<f32>(0.0, 0.0, 1.0);
        } else {
            n_world = n_world / n_len;
        }

        // 可视化时编码到颜色 [0,1]
        let n_rgb = clamp(0.5 * (n_world + vec3<f32>(1.0, 1.0, 1.0)), vec3<f32>(0.0), vec3<f32>(1.0));
        color = vec4<f32>(n_rgb, opacity);
    } else if (uModel.rendermode == 2u) {
        // 模式2: 深度可视化（使用透视除法后的 NDC 深度，0..1）
        let depth_ndc = 1.0 - clamp(pos2d.z / pos2d.w, 0.0, 1.0);
        color = vec4<f32>(depth_ndc, depth_ndc, depth_ndc, opacity);
    } else if (uModel.rendermode == 3u) {
        // 模式3: PCA可视化（extra_pca_r/g/b）
        let p = extra_pca[idx].xyz;
        color = vec4<f32>(clamp(p, vec3<f32>(0.0), vec3<f32>(1.0)), opacity);
    } else {
        // 默认: 正常颜色
        color = vec4<f32>(
            max(vec3<f32>(0.), evaluate_color(dir_local, idx, uModel.maxShDeg)),
            1
        );
    }

    let v = vec4<f32>(v1 / viewport, v2 / viewport);
    points_2d[store_idx] = Splat(
        pack2x16float(v.xy), pack2x16float(v.zw),
        pack2x16float(v_center.xy),
        v_center.z,
        pack2x16float(color.rg), pack2x16float(color.ba),
    );
    // filling the sorting buffers and the indirect sort dispatch buffer
    let znear = -camera.proj[3][2] / camera.proj[2][2];
    let zfar = -camera.proj[3][2] / (camera.proj[2][2] - (1.));
    // filling the sorting buffers and the indirect sort dispatch buffer
    sort_depths[store_idx] = bitcast<u32>(zfar - pos2d.z) ;//u32(f32(0xffffffu) - pos2d.z / zfar * f32(0xffffffu));
    sort_indices[store_idx] = store_idx;
    sort_source_indices[store_idx] = uModel.baseOffset + idx;

    let keys_per_wg = 256u * 15u;         // Caution: if workgroup size (256) or keys per thread (15) changes the dispatch is wrong!!
    if (global_index % keys_per_wg) == 0u {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}