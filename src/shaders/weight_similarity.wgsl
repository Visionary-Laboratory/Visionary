struct WeightParams {
    baseOffset: u32,
    numPoints: u32,
    hasWeights: u32,
    _pad0: u32,
};

@group(0) @binding(0)
var<uniform> query_weights : array<vec4<f32>, 16>;

// Gram matrix G = C * C^T, row-major packed as 64 rows × 16 vec4s (= 64*64 floats)
@group(0) @binding(4)
var<uniform> gram : array<vec4<f32>, 64 * 16>;

@group(0) @binding(1)
var<storage, read> language_weights : array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> similarity : array<f32>;

@group(0) @binding(3)
var<uniform> params : WeightParams;

// softmaxParams.x = temperature (>= 1e-6)
// softmaxParams.y = useSoftmax (0.0 or 1.0)
@group(0) @binding(5)
var<uniform> softmaxParams : vec4<f32>;

fn lane_f32(v: vec4<f32>, lane: u32) -> f32 {
    if (lane == 0u) { return v.x; }
    if (lane == 1u) { return v.y; }
    if (lane == 2u) { return v.z; }
    return v.w;
}

fn gram_at(r: u32, c: u32) -> f32 {
    // r in [0,63], c in [0,63]
    let v = gram[r * 16u + (c >> 2u)];
    return lane_f32(v, c & 3u);
}

fn w_at(wv: ptr<function, array<vec4<f32>, 16>>, i: u32) -> f32 {
    let v = (*wv)[i >> 2u];
    return lane_f32(v, i & 3u);
}

fn q_at(i: u32) -> f32 {
    let v = query_weights[i >> 2u];
    return lane_f32(v, i & 3u);
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.numPoints) {
        return;
    }

    let out_index = params.baseOffset + idx;
    if (params.hasWeights == 0u) {
        similarity[out_index] = 0.0;
        return;
    }

    let base = idx * 16u;
    // Load 64D weights (16 vec4) for this gaussian
    var wv: array<vec4<f32>, 16>;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        wv[i] = language_weights[base + i];
    }

    var ws: array<f32, 64>;
    let useSoftmax = softmaxParams.y > 0.5;
    let temperature = max(1e-6, softmaxParams.x);
    let invTemp = 1.0 / temperature;

    if (useSoftmax) {
        // Stable softmax over 64D weights (optionally temperature-scaled).
        var maxv: f32 = -1e30;
        for (var i: u32 = 0u; i < 64u; i = i + 1u) {
            maxv = max(maxv, w_at(&wv, i));
        }
        var sumexp: f32 = 0.0;
        for (var i: u32 = 0u; i < 64u; i = i + 1u) {
            let e = exp((w_at(&wv, i) - maxv) * invTemp);
            ws[i] = e;
            sumexp = sumexp + e;
        }
        let invsum = 1.0 / max(1e-12, sumexp);
        for (var i: u32 = 0u; i < 64u; i = i + 1u) {
            ws[i] = ws[i] * invsum;
        }
    } else {
        // No softmax: interpret weights as already-normalized (or user-controlled).
        for (var i: u32 = 0u; i < 64u; i = i + 1u) {
            ws[i] = w_at(&wv, i);
        }
    }

    // Numerator: softmax(w) · (Cq)
    var num: f32 = 0.0;
    for (var i: u32 = 0u; i < 64u; i = i + 1u) {
        num = num + ws[i] * q_at(i);
    }

    // Denominator: || w^T C || = sqrt(w^T (C C^T) w) = sqrt(w^T G w)
    var den2: f32 = 0.0;
    for (var r: u32 = 0u; r < 64u; r = r + 1u) {
        var acc: f32 = 0.0;
        for (var c: u32 = 0u; c < 64u; c = c + 1u) {
            acc = acc + gram_at(r, c) * ws[c];
        }
        den2 = den2 + ws[r] * acc;
    }

    let denom = sqrt(max(1e-12, den2));
    // We keep only positive cosine for visualization/thresholding (range ~[0,1])
    let cos_sim = num / denom;
    similarity[out_index] = clamp(max(0.0, cos_sim), 0.0, 1.0);
}

