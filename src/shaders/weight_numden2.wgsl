// Compute per-Gaussian scalar numerator/denominator terms for pixel-level cosine approximation.
// Output per-point:
//   num  = w · q64
//   den2 = w^T (C C^T) w = w^T G w
//
// Notes:
// - This is an approximation when used with pixel compositing because it ignores cross-terms between
//   different splats inside the same pixel. It is still useful and much cheaper than HxWx64.

struct Params {
  baseOffset: u32,
  numPoints: u32,
  hasWeights: u32,
  _pad0: u32,
};

@group(0) @binding(0)
var<uniform> query_weights : array<vec4<f32>, 16>; // q64 packed as 16 vec4

@group(0) @binding(1)
var<storage, read> language_weights : array<vec4<f32>>; // 16 vec4 per point

@group(0) @binding(2)
var<storage, read_write> numden2_out : array<vec2<f32>>; // per-point (num, den2)

@group(0) @binding(3)
var<uniform> params : Params;

@group(0) @binding(4)
// Gram matrix G = C * C^T, row-major packed as 64 rows × 16 vec4s (= 64*64 floats)
var<uniform> gram : array<vec4<f32>, 64 * 16>;

fn lane_f32(v: vec4<f32>, lane: u32) -> f32 {
  if (lane == 0u) { return v.x; }
  if (lane == 1u) { return v.y; }
  if (lane == 2u) { return v.z; }
  return v.w;
}

fn gram_at(r: u32, c: u32) -> f32 {
  let v = gram[r * 16u + (c >> 2u)];
  return lane_f32(v, c & 3u);
}

fn w_at(wv: ptr<function, array<vec4<f32>, 16>>, i: u32) -> f32 {
  let vi = i >> 2u;
  let li = i & 3u;
  if (li == 0u) { return (*wv)[vi].x; }
  if (li == 1u) { return (*wv)[vi].y; }
  if (li == 2u) { return (*wv)[vi].z; }
  return (*wv)[vi].w;
}

fn q_at(i: u32) -> f32 {
  let vi = i >> 2u;
  let li = i & 3u;
  return lane_f32(query_weights[vi], li);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.numPoints) { return; }
  let out_index = params.baseOffset + idx;

  if (params.hasWeights == 0u) {
    numden2_out[out_index] = vec2<f32>(0.0, 1.0);
    return;
  }

  // Load 64D weights (16 vec4)
  let base = idx * 16u;
  var wv: array<vec4<f32>, 16>;
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    wv[i] = language_weights[base + i];
  }

  // num = w · q64
  var num: f32 = 0.0;
  for (var i: u32 = 0u; i < 64u; i = i + 1u) {
    num = num + w_at(&wv, i) * q_at(i);
  }

  // den2 = w^T G w
  var den2: f32 = 0.0;
  for (var r: u32 = 0u; r < 64u; r = r + 1u) {
    var acc: f32 = 0.0;
    for (var c: u32 = 0u; c < 64u; c = c + 1u) {
      acc = acc + gram_at(r, c) * w_at(&wv, c);
    }
    den2 = den2 + w_at(&wv, r) * acc;
  }

  numden2_out[out_index] = vec2<f32>(num, den2);
}

