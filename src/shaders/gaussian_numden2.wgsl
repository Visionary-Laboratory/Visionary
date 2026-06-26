// Render a per-splat (num, den2) map using the same splat shape as gaussian.wgsl.
// The map is accumulated with premultiplied alpha over blending.
//
// Output target format: rgba16float
//   r = num (premultiplied by alpha in shader)
//   g = den2 (premultiplied by alpha in shader)
//   b = 0
//   a = alpha

const CUTOFF:f32 = 2.3539888583335364; // = sqrt(log(255))

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) screen_pos: vec2<f32>,
    @location(1) num: f32,
    @location(2) den2: f32,
    @location(3) opacity: f32,
};

struct Splat {
    v_0: u32, v_1: u32,
    pos: u32,
    posz: f32,
    color_0: u32, color_1: u32,
};

@group(0) @binding(2)
var<storage, read> points_2d : array<Splat>;

@group(0) @binding(3)
var<storage, read> numden2_buffer : array<vec2<f32>>;

@group(0) @binding(4)
var<storage, read> source_indices : array<u32>;

@group(1) @binding(4)
var<storage, read> indices : array<u32>;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) in_instance_index: u32
) -> VertexOutput {
    var out: VertexOutput;
    let splat_index = indices[in_instance_index];
    let vertex = points_2d[splat_index];

    let v1 = unpack2x16float(vertex.v_0);
    let v2 = unpack2x16float(vertex.v_1);

    let v_center_xy = unpack2x16float(vertex.pos);
    let v_center_z = vertex.posz;

    let x = f32(in_vertex_index % 2u == 0u) * 2. - (1.);
    let y = f32(in_vertex_index < 2u) * 2. - (1.);
    let position = vec2<f32>(x, y) * CUTOFF;

    let offset = 2. * mat2x2<f32>(v1, v2) * position;
    let z_ndc = clamp(v_center_z, 0.0, 1.0);
    out.position = vec4<f32>(v_center_xy + offset, z_ndc, 1.);
    out.screen_pos = position;

    let src_index = source_indices[splat_index];
    let nd = numden2_buffer[src_index];
    out.num = nd.x;
    out.den2 = nd.y;

    let c1 = unpack2x16float(vertex.color_1);
    out.opacity = c1.y;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let a = dot(in.screen_pos, in.screen_pos);
    if a > 2. * CUTOFF {
        discard;
    }
    let alpha = min(0.99, exp(-a) * in.opacity);
    // Premultiply by alpha; blending does the over composition.
    return vec4<f32>(in.num, in.den2, 0.0, 1.0) * alpha;
}

