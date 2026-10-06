// Box, circle and lasso selection, as a compute reduction over the id buffer.
//
// The obvious implementation reads the id texture back and scans it on the CPU.
// At 1080p an `Rg32Uint` frame is 16 MB: roughly ten milliseconds of PCIe plus a
// two-million-texel scan, on *every* frame of a rubber-band drag. That is a
// visible stall on the one interaction where the user is watching a rectangle
// follow their pointer.
//
// So the scan happens where the data already is. One thread per texel, each
// testing its own pixel against the selection shape and folding a hit into a
// bitset with an atomic. What comes back is one bit per element — 125 KB for a
// million vertices, under a millisecond — and the bitset is also usable directly
// as GPU selection state, so the highlight can update in the same frame with no
// upload at all.
//
// # The lasso is a point-in-polygon test, not a stencil texture
//
// Rasterising the lasso into a 1-bit mask first is the textbook approach and
// needs an extra target, an extra pass and a size that has to track the
// viewport. A crossing-number test against the polygon costs one loop over its
// points per texel, is exact, needs no memory, and at a few dozen points is
// nothing next to the texel fetch. The point cap is what keeps it bounded.

struct SelectParams {
    // xy: region origin in pixels. zw: region size in pixels.
    region: vec4<u32>,
    // Box: min.xy, max.xy. Circle: centre.xy, radius in z.
    shape: vec4<f32>,
    // x: shape kind — 0 box, 1 circle, 2 lasso. y: lasso point count.
    // z: how many entries the bitset covers. w: the object whose elements are
    // being collected, stored plus one to match the id buffer, or zero to
    // collect whole objects instead.
    mode: vec4<u32>,
};

const SHAPE_BOX: u32 = 0u;
const SHAPE_CIRCLE: u32 = 1u;
const SHAPE_LASSO: u32 = 2u;

/// The most points a lasso may have.
///
/// A freehand drag produces far more than this, so the caller decimates. That is
/// not a loss: a lasso is a coarse gesture, and its own line is drawn from the
/// same decimated points, so what is tested is exactly what the user was shown.
const MAX_LASSO: u32 = 256u;

@group(0) @binding(0) var<uniform> p: SelectParams;
@group(0) @binding(1) var ids: texture_2d<u32>;
@group(0) @binding(2) var<storage, read_write> bits: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> lasso: array<vec2<f32>>;
// Multi-object edit mode: where each object's elements start, by object id;
// all ones for an object not being collected.
@group(0) @binding(4) var<storage, read> bases: array<u32>;

const MANY: u32 = 0xffffffffu;

/// Crossing-number point-in-polygon.
///
/// Counts how many times a ray from the point crosses the outline; odd means
/// inside. The `!=` on the y comparisons is what makes a vertex exactly at the
/// test height count once rather than twice or not at all — the classic source
/// of single-pixel holes along a lasso's edge.
fn in_lasso(at: vec2<f32>, count: u32) -> bool {
    var inside = false;
    var j = count - 1u;
    for (var i = 0u; i < count; i = i + 1u) {
        let a = lasso[i];
        let b = lasso[j];
        if ((a.y > at.y) != (b.y > at.y)) {
            let t = (at.y - a.y) / (b.y - a.y);
            if (at.x < a.x + t * (b.x - a.x)) {
                inside = !inside;
            }
        }
        j = i;
    }
    return inside;
}

fn inside(at: vec2<f32>) -> bool {
    switch p.mode.x {
        case SHAPE_CIRCLE: {
            return distance(at, p.shape.xy) <= p.shape.z;
        }
        case SHAPE_LASSO: {
            let count = min(p.mode.y, MAX_LASSO);
            // Fewer than three points is not a shape. Returning false rather
            // than something arbitrary means a stray click that registers as a
            // one-point lasso selects nothing instead of everything.
            if (count < 3u) {
                return false;
            }
            return in_lasso(at, count);
        }
        default: {
            return all(at >= p.shape.xy) && all(at <= p.shape.zw);
        }
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= p.region.z || gid.y >= p.region.w) {
        return;
    }
    let texel = vec2<i32>(vec2<u32>(gid.x + p.region.x, gid.y + p.region.y));

    // The pixel centre, so a shape's edge falls between pixels the same way the
    // rasteriser puts it. Testing the corner biases every selection half a pixel
    // up and left, which shows up as a rubber band that seems to miss the row it
    // is drawn over.
    if (!inside(vec2<f32>(texel) + vec2<f32>(0.5, 0.5))) {
        return;
    }

    let id = textureLoad(ids, texel, 0);
    // Zero is "nothing here": the object id is stored plus one precisely so that
    // an untouched pixel is a miss rather than object zero.
    if (id.x == 0u) {
        return;
    }

    // Two things a drag can collect, and the same reduction does both.
    //
    // In edit mode it collects *elements* of one object, and only that object:
    // every mesh numbers its elements from zero, so folding a neighbour's in
    // would select unrelated geometry wherever the box crossed it.
    //
    // In object mode it collects *objects*, from the channel that already names
    // them. Reusing the reduction is the point — an object-mode box has exactly
    // the same shapes, the same occlusion and the same cost characteristics, and
    // a second pipeline for it would be the same code with one index changed.
    var entry: u32;
    if (p.mode.w == 0u) {
        entry = id.x - 1u;
    } else if (p.mode.w == MANY) {
        // Several objects, each element at its object's base.
        let object = id.x - 1u;
        if (object >= arrayLength(&bases)) {
            return;
        }
        let base = bases[object];
        if (base == MANY) {
            return;
        }
        entry = base + id.y;
    } else {
        if (id.x != p.mode.w) {
            return;
        }
        entry = id.y;
    }
    if (entry >= p.mode.z) {
        return;
    }
    // One bit per entry. The atomic is what lets thousands of texels covering
    // one face fold into a single bit with no ordering and no duplicates —
    // which is also why the result is a *set* rather than a list that would need
    // sorting and deduplicating on the way back.
    atomicOr(&bits[entry >> 5u], 1u << (entry & 31u));
}
