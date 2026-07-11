// Fullscreen-triangle blit used to fill each mip level from the one above.

@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VsOut {
    // one triangle covering the whole target: (-1,-1) (3,-1) (-1,3)
    let x = f32(i32(vi / 2u)) * 4.0 - 1.0;
    let y = f32(i32(vi % 2u)) * 4.0 - 1.0;

    var out: VsOut;
    out.pos = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(t, s, in.uv);
}
