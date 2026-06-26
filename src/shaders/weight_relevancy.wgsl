// Compute per-Gaussian neg relevancy scores.
//
// We compute cosine similarities using precomputed q64 vectors and gram matrix G=C*C^T:
//   s(q) = (w · q64) / sqrt(w^T G w)
//
// Neg relevancy (binary softmax) against each negative:
//   p_pos_given_neg = softmax(tau * [s_pos, s_neg])[0] = sigmoid(tau * (s_pos - s_neg))
//
// Output similarity := min_j p_pos_given_neg_j (hardest negative), clamped to [0,1].

struct Params {
  baseOffset: u32,
  numPoints: u32,
  hasWeights: u32,
  _pad0: u32,
};

// cfg.x = tau
// cfg.y = negMask bits (0..3) encoded as f32 but treated as u32
// cfg.z = useRelevancy (0.0/1.0)
// cfg.w = reserved
@group(0) @binding(6)
var<uniform> cfg : vec4<f32>;

@group(0) @binding(0)
var<uniform> queries : array<vec4<f32>, 16u * 5u>; // [pos, neg0..3] each is 16 vec4

@group(0) @binding(1)
var<storage, read> language_weights : array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> similarity : array<f32>;

@group(0) @binding(3)
var<uniform> params : Params;

// Gram matrix G = C * C^T, row-major packed as 64 rows × 16 vec4s (= 64*64 floats)
@group(0) @binding(4)
var<uniform> gram : array<vec4<f32>, 64u * 16u>;

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
  let v = (*wv)[i >> 2u];
  return lane_f32(v, i & 3u);
}

fn q_at(qi: u32, i: u32) -> f32 {
  let base = qi * 16u;
  let v = queries[base + (i >> 2u)];
  return lane_f32(v, i & 3u);
}

fn sigmoid(x: f32) -> f32 {
  // Stable-ish sigmoid
  return 1.0 / (1.0 + exp(-x));
}

fn cosine_for_query(wv: ptr<function, array<vec4<f32>, 16>>, qi: u32, denom: f32) -> f32 {
  var num: f32 = 0.0;
  for (var i: u32 = 0u; i < 64u; i = i + 1u) {
    num = num + w_at(wv, i) * q_at(qi, i);
  }
  return num / denom;
}

fn denom_from_gram(wv: ptr<function, array<vec4<f32>, 16>>) -> f32 {
  var den2: f32 = 0.0;
  for (var r: u32 = 0u; r < 64u; r = r + 1u) {
    var acc: f32 = 0.0;
    for (var c: u32 = 0u; c < 64u; c = c + 1u) {
      acc = acc + gram_at(r, c) * w_at(wv, c);
    }
    den2 = den2 + w_at(wv, r) * acc;
  }
  return sqrt(max(1e-12, den2));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.numPoints) { return; }
  let out_index = params.baseOffset + idx;

  if (params.hasWeights == 0u) {
    similarity[out_index] = 0.0;
    return;
  }

  let useRel = cfg.z > 0.5;

  // Load 64D weights (16 vec4)
  let base = idx * 16u;
  var wv: array<vec4<f32>, 16>;
  for (var i: u32 = 0u; i < 16u; i = i + 1u) {
    wv[i] = language_weights[base + i];
  }

  let denom = denom_from_gram(&wv);
  let s_pos = cosine_for_query(&wv, 0u, denom);

  if (!useRel) {
    similarity[out_index] = clamp(max(0.0, s_pos), 0.0, 1.0);
    return;
  }

  let tau = max(1e-6, cfg.x);
  let negMask = bitcast<u32>(cfg.y);

  var best: f32 = 1.0;
  var hasAny: bool = false;
  for (var j: u32 = 0u; j < 4u; j = j + 1u) {
    if ((negMask & (1u << j)) == 0u) { continue; }
    hasAny = true;
    let s_neg = cosine_for_query(&wv, 1u + j, denom);
    let p = sigmoid(tau * (s_pos - s_neg));
    best = min(best, p);
  }
  if (!hasAny) {
    best = clamp(max(0.0, s_pos), 0.0, 1.0);
  }

  similarity[out_index] = clamp(best, 0.0, 1.0);
}

