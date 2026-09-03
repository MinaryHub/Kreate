// The depth pyramid: a mip chain where each texel holds the *farthest* depth in
// the region it covers.
//
// Farthest, not nearest, and that is the whole subtlety. In reversed-Z a larger
// value is closer, so "farthest" is a **minimum** reduction — the opposite of
// the max-reduction every hi-z article written for standard Z describes. Getting
// this backwards does not produce a wrong-looking pyramid; it produces one that
// culls geometry the user can see, and holes in a model read as corrupt data
// rather than as a renderer bug.
//
// Why farthest is the right answer: a cluster is hidden only if *every* pixel it
// covers already has something closer in front of it. So the test is
//
//     min(depth over the region) > cluster's nearest depth   =>   hidden
//
// and for that to be sound the pyramid has to report the smallest depth in the
// region, which is the farthest thing drawn there. Cleared background sits at
// 0.0, the far end, so any cluster overlapping empty space is never culled.

struct Params {
    // x, y: destination size in texels. z, w: source size in texels.
    size: vec4<u32>,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var src_depth: texture_depth_2d;
@group(0) @binding(2) var dst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn from_depth(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.size.x || id.y >= p.size.y) {
        return;
    }
    // Level zero is the depth buffer as it stands, at full resolution. A copy
    // rather than a first reduction, so that every level above it is one
    // uniform rule and the CPU reference has one rule to mirror.
    let d = textureLoad(src_depth, vec2<i32>(id.xy), 0);
    textureStore(dst, vec2<i32>(id.xy), vec4<f32>(d, 0.0, 0.0, 0.0));
}

@group(1) @binding(0) var<uniform> rp: Params;
@group(1) @binding(1) var src_level: texture_2d<f32>;
@group(1) @binding(2) var rdst: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn reduce(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= rp.size.x || id.y >= rp.size.y) {
        return;
    }
    let src = rp.size.zw;
    let base = id.xy * 2u;

    var m = textureLoad(src_level, vec2<i32>(base), 0).r;
    // Clamped rather than skipped: reading the same texel twice is harmless for
    // a minimum, and a branch per sample is not.
    let x1 = min(base.x + 1u, src.x - 1u);
    let y1 = min(base.y + 1u, src.y - 1u);
    m = min(m, textureLoad(src_level, vec2<i32>(i32(x1), i32(base.y)), 0).r);
    m = min(m, textureLoad(src_level, vec2<i32>(i32(base.x), i32(y1)), 0).r);
    m = min(m, textureLoad(src_level, vec2<i32>(i32(x1), i32(y1)), 0).r);

    // An odd source dimension leaves a row or column that the 2x2 blocks never
    // reach, and a mip chain's sizes are floored, so the last destination texel
    // has to swallow it. Without this the pyramid would silently omit the far
    // edge of the screen: its minimum would come out too large, and the cull
    // would reject clusters that are actually visible there.
    //
    // Widening a minimum's coverage is always safe — it can only make the
    // reported depth farther, and a farther depth culls less.
    if ((src.x & 1u) == 1u && id.x + 1u == rp.size.x) {
        let extra = src.x - 1u;
        m = min(m, textureLoad(src_level, vec2<i32>(i32(extra), i32(base.y)), 0).r);
        m = min(m, textureLoad(src_level, vec2<i32>(i32(extra), i32(y1)), 0).r);
    }
    if ((src.y & 1u) == 1u && id.y + 1u == rp.size.y) {
        let extra = src.y - 1u;
        m = min(m, textureLoad(src_level, vec2<i32>(i32(base.x), i32(extra)), 0).r);
        m = min(m, textureLoad(src_level, vec2<i32>(i32(x1), i32(extra)), 0).r);
    }
    // The corner, reachable only when both dimensions are odd.
    if ((src.x & 1u) == 1u && (src.y & 1u) == 1u
        && id.x + 1u == rp.size.x && id.y + 1u == rp.size.y) {
        m = min(
            m,
            textureLoad(src_level, vec2<i32>(i32(src.x - 1u), i32(src.y - 1u)), 0).r,
        );
    }

    textureStore(rdst, vec2<i32>(id.xy), vec4<f32>(m, 0.0, 0.0, 0.0));
}
