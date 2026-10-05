//! What a light sees.
//!
//! A shadow map is a depth buffer rendered from a light, so the first question
//! is where the light's camera is and what it frames. That question is
//! answerable without a device, which is why it is here on its own: a frustum
//! that is too small crops shadows at its edge, and one that is too large
//! spends its precision on empty space and stipples the contact shadows. Both
//! look like a shadow bug and neither is.
//!
//! # Eye-relative, like everything else
//!
//! The main pass has the fragment's position in eye-relative metres and the
//! lights packed the same way (see [`crate::lights`]), so the shadow matrix is
//! built there too. A shadow map in absolute world space would mean the sampling
//! shader converting back, in `f32`, at the exact magnitudes eye-relative space
//! exists to avoid.
//!
//! # Reversed-Z, like everything else
//!
//! Depth increases toward the light, cleared to zero, compared with
//! `GreaterEqual`. The float depth buffer then has its precision where the
//! geometry is, which is the whole reason §v1 chose it — and a shadow pass that
//! used the other convention would be a depth test that always passed.
//!
//! # What is not here
//!
//! Cascades. One frustum over the whole scene is the first step and it is
//! enough to see a shadow; a sun over a large scene wants three or four of
//! them, split by view distance, and the split is a policy question rather than
//! a geometric one. The shape below takes a bounding box, so a cascade is the
//! same function called with a smaller one.

use kreate_math::Aabb;

/// The camera a shadow map is rendered from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShadowView {
    /// Eye-relative world to the light's clip space, reversed-Z.
    pub view_proj: glam::Mat4,
    /// The near and far planes the projection was built with, in metres.
    ///
    /// Carried because the sampling pass needs them to turn a stored depth
    /// back into a distance, and recomputing them there would be a second
    /// place for the numbers to live.
    pub near: f32,
    pub far: f32,
    /// How many texels of penumbra one unit of stored-depth difference makes.
    ///
    /// See [`GpuShadow::penumbra`]. Here because it needs the projection's near
    /// and far planes, which exist only while the frame is being fitted —
    /// packing them for the sampler instead would be two more numbers to keep
    /// in step for a value that never changes once the frame is built.
    pub penumbra: f32,
    /// How wide one shadow texel is in metres, per unit of clip `w`.
    ///
    /// The number the acne fix is built on, and the one thing about the map
    /// the sampling shader cannot work out for itself: it can ask the texture
    /// how many texels there are, but not how much world each one covers.
    ///
    /// Per unit `w` so that one field covers both projections. An orthographic
    /// clip `w` is 1 and the texel is the same size everywhere; a perspective
    /// one is the distance along the light's axis, and a texel grows with it
    /// in exact proportion.
    pub texel_world: f32,
}

/// A little room around the fitted box.
///
/// Geometry exactly on the boundary is geometry that half of its pixels fall
/// outside, and the edge of a shadow map is where a sample reads nothing.
const MARGIN: f32 = 1.0e-3;

/// The smallest near plane a perspective shadow will use, in metres.
///
/// Reversed-Z puts its precision near the near plane, so a near plane at a
/// millimetre wastes most of the buffer on the first centimetre. A tenth of a
/// metre is closer than any light usefully sits to what it lights.
const MIN_NEAR: f32 = 0.1;

/// Builds an orthographic projection in reversed-Z.
///
/// Written out rather than taken from `glam`, because the reversed-Z form is
/// not what any library's `orthographic_rh` produces and a silently normal-Z
/// matrix is a depth test that always passes.
fn ortho_reversed(half: glam::Vec3, near: f32, far: f32) -> glam::Mat4 {
    let depth = (far - near).max(1e-6);
    // `near` and `far` are distances along the looking direction, so the view
    // coordinates they name are `-near` and `-far`. This maps the first to 1
    // and the second to 0:
    //
    //     depth = (z_view + far) / (far - near)
    glam::Mat4::from_cols(
        glam::Vec4::new(1.0 / half.x, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, 1.0 / half.y, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, 1.0 / depth, 0.0),
        glam::Vec4::new(0.0, 0.0, far / depth, 1.0),
    )
}

/// A perspective projection in reversed-Z, with an infinite far plane.
///
/// The same form the main camera uses, so a shadow's depth means what the
/// viewport's does.
fn perspective_reversed(fov_y: f32, near: f32) -> glam::Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    glam::Mat4::from_cols(
        glam::Vec4::new(f, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, f, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, 0.0, -1.0),
        glam::Vec4::new(0.0, 0.0, near, 0.0),
    )
}

/// A rotation that looks along `forward`.
///
/// Picks the up vector rather than taking one, because a shadow map has no
/// preferred roll — and a caller that had to supply one would have to know to
/// avoid the degenerate case, which is the whole difficulty.
fn look_along(forward: glam::Vec3) -> glam::Mat4 {
    let forward = forward.try_normalize().unwrap_or(glam::Vec3::NEG_Z);
    // Any axis not parallel to the direction. Z unless the light points along
    // it, which is the common case for a sun and exactly where a fixed up
    // vector degenerates.
    let up = if forward.z.abs() > 0.99 {
        glam::Vec3::Y
    } else {
        glam::Vec3::Z
    };
    let right = up.cross(forward).normalize();
    let up = forward.cross(right);
    // World to light: the transpose of the light's basis.
    glam::Mat4::from_cols(
        glam::Vec4::new(right.x, up.x, -forward.x, 0.0),
        glam::Vec4::new(right.y, up.y, -forward.y, 0.0),
        glam::Vec4::new(right.z, up.z, -forward.z, 0.0),
        glam::Vec4::W,
    )
}

/// What a parallel light sees: an orthographic box around the scene.
///
/// Fitted to the bounds *rotated into the light's frame*, not to the bounds
/// themselves. A box fitted to the world-axis extents is bigger than it needs
/// to be at every angle but one, and the excess comes out of the shadow's
/// resolution.
///
/// `None` for an empty scene, which is not an error: there is nothing to cast
/// and nothing to catch, and a pass that ran anyway would render a map nothing
/// samples.
pub fn sun_view(direction: glam::Vec3, bounds: Aabb) -> Option<ShadowView> {
    sun_view_of(
        direction,
        &bounds.corners(),
        bounds,
        SourceSize::Angular(0.0),
    )
}

/// The same, fitted across `wanted` rather than across the whole scene.
///
/// `wanted` is what the map has to *cover* — for a cascade, the corners of a
/// slice of the view frustum. `scene` is what exists, and it bounds the fit in
/// two different ways:
///
/// - **Across the map**, it is an intersection. A slice of the frustum reaching
///   a kilometre past the last object would spend the map's resolution on empty
///   space, which is the whole thing cascades exist to stop.
/// - **In depth**, it is a replacement. The near plane has to reach back to the
///   light past everything in the scene, or an object between the lamp and the
///   slice is clipped out of the map and stops casting into it — a wall's
///   shadow disappearing because the wall itself is behind the camera.
pub fn sun_view_of(
    direction: glam::Vec3,
    wanted: &[glam::Vec3],
    scene: Aabb,
    source: SourceSize,
) -> Option<ShadowView> {
    if scene.is_empty() || wanted.is_empty() {
        return None;
    }
    let view = look_along(direction);
    let (mut min, mut max) = (
        glam::Vec3::splat(f32::INFINITY),
        glam::Vec3::splat(f32::NEG_INFINITY),
    );
    for corner in wanted {
        let p = view.transform_point3(*corner);
        min = min.min(p);
        max = max.max(p);
    }
    let (mut scene_min, mut scene_max) = (
        glam::Vec3::splat(f32::INFINITY),
        glam::Vec3::splat(f32::NEG_INFINITY),
    );
    for corner in scene.corners() {
        let p = view.transform_point3(corner);
        scene_min = scene_min.min(p);
        scene_max = scene_max.max(p);
    }
    // Across the map: the part of what was asked for that has anything in it.
    min = min.max(scene_min);
    max = max.min(scene_max);
    if min.x > max.x || min.y > max.y {
        // The slice is outside the scene entirely. Not an error — a cascade
        // covering distance the scene does not reach has nothing to draw.
        return None;
    }
    // In depth: the scene's, so that casters outside the slice still cast into
    // it.
    min.z = scene_min.z;
    max.z = scene_max.z;
    // The camera looks down -Z, so in front is negative and *nearer* is the
    // larger z. The distances the projection wants are the negated extents.
    //
    // Not clamped to zero, which was the first attempt and was wrong: a
    // parallel light has no position, so the frame's origin is arbitrary and
    // the scene can sit at any z at all — including entirely "behind" it. What
    // matters is only that the range covers the scene, and `far - near` is the
    // extent either way.
    let near = -max.z - MARGIN;
    let far = -min.z + MARGIN;
    let half = glam::Vec3::new(
        (max.x - min.x).max(1e-4) * 0.5 + MARGIN,
        (max.y - min.y).max(1e-4) * 0.5 + MARGIN,
        0.0,
    );
    let texel = half.x.max(half.y) * 2.0 / SHADOW_RESOLUTION as f32;
    // Centred on the scene, which is what stops the box wasting half its width
    // when the scene is off to one side.
    let centre = glam::Vec3::new((min.x + max.x) * 0.5, (min.y + max.y) * 0.5, 0.0);
    let recentre = glam::Mat4::from_translation(-centre);
    Some(ShadowView {
        view_proj: ortho_reversed(half, near, far) * recentre * view,
        near,
        far,
        // The box is `half` either side, so it spans `2 * half` across the
        // map's width. The wider of the two axes, because the bias has to
        // cover the worst texel and not the average one.
        texel_world: texel,
        penumbra: penumbra_scale(source, near, far, texel),
    })
}

/// What a spot sees: a perspective frustum as wide as its cone.
///
/// `cone_angle` is the full angle, as [`crate::lights::PlacedLight`] stores it.
/// The frustum is square and covers the cone's circle, so the corners of the
/// map are outside the cone and shade nothing — which is the price of a square
/// texture and is cheaper than the alternatives.
pub fn spot_view(
    position: glam::Vec3,
    direction: glam::Vec3,
    cone_angle: f32,
    bounds: Aabb,
    source: SourceSize,
) -> Option<ShadowView> {
    if bounds.is_empty() {
        return None;
    }
    let view = look_along(direction) * glam::Mat4::from_translation(-position);
    // How far the scene reaches in front of the light, which is what the far
    // plane has to cover.
    let mut far = 0.0f32;
    for corner in bounds.corners() {
        far = far.max(-view.transform_point3(corner).z);
    }
    if far <= 0.0 {
        // Every corner is behind the light. Nothing it lights is in the scene.
        return None;
    }
    let near = MIN_NEAR.min(far * 0.5);
    // A square frustum that contains the cone's circle: the diagonal of the
    // square is what the cone's full angle has to fit inside.
    let fov = (cone_angle.clamp(1e-3, std::f32::consts::PI * 0.99)).min(3.0);
    Some(ShadowView {
        view_proj: perspective_reversed(fov, near) * view,
        near,
        far: far + MARGIN,
        texel_world: perspective_texel(fov),
        penumbra: penumbra_scale(source, near, far + MARGIN, perspective_texel(fov)),
    })
}

/// One face of a point light's cube.
///
/// Six 90° frustums. `face` is 0 to 5 in the order `+X -X +Y -Y +Z -Z`, which
/// is the order a cube texture's layers are.
pub fn point_view(
    position: glam::Vec3,
    face: usize,
    bounds: Aabb,
    source: SourceSize,
) -> Option<ShadowView> {
    if bounds.is_empty() {
        return None;
    }
    let direction = match face % 6 {
        0 => glam::Vec3::X,
        1 => glam::Vec3::NEG_X,
        2 => glam::Vec3::Y,
        3 => glam::Vec3::NEG_Y,
        4 => glam::Vec3::Z,
        _ => glam::Vec3::NEG_Z,
    };
    let view = look_along(direction) * glam::Mat4::from_translation(-position);
    let mut far = 0.0f32;
    for corner in bounds.corners() {
        far = far.max(-view.transform_point3(corner).z);
    }
    if far <= 0.0 {
        return None;
    }
    let near = MIN_NEAR.min(far * 0.5);
    let fov = cube_face_fov();
    Some(ShadowView {
        view_proj: perspective_reversed(fov, near) * view,
        near,
        far: far + MARGIN,
        texel_world: perspective_texel(fov),
        penumbra: penumbra_scale(source, near, far + MARGIN, perspective_texel(fov)),
    })
}

/// How many texels past a cube face's 90° a face's map reaches.
///
/// A point is shaded from the face its direction points through, and the
/// filter around it — the soft-shadow kernel, up to eight texels, and the
/// normal offset — reaches past where that face stops. At exactly 90° those
/// taps read outside the map, which is "lit": a bright line ran through every
/// shadow along the seam between two faces. With the faces a little wider
/// than the cube they are chosen by, every tap lands on rendered depth.
const CUBE_FACE_OVERLAP: f32 = 24.0;

/// A cube face's field of view: 90° and `CUBE_FACE_OVERLAP` texels each side.
fn cube_face_fov() -> f32 {
    let inner = 1.0 - 2.0 * CUBE_FACE_OVERLAP / SHADOW_RESOLUTION as f32;
    2.0 * (1.0 / inner).atan()
}

/// The frame a packed light should render its shadow from.
///
/// Dispatches on what the light is, and the light says so in the two fields it
/// already has: the falloff distinguishes a parallel light — which has no
/// position and gets the orthographic box — and the cone's outer cosine
/// distinguishes a spot from a bare point light, because a light with no cone
/// packs that at -1.
///
/// Read off the packed form rather than passed a kind, so the shadow frame and
/// the shading cannot disagree about what a light is.
///
/// Empty when the light has nothing to shadow: an empty scene, a spot aimed
/// away from everything, a point light with the scene entirely behind it.
///
/// # Why a list
///
/// A point light sees in every direction and one frustum cannot. It gets the
/// faces of a cube — but only the ones with anything in them, because
/// [`point_view`] answers `None` for a face that frames nothing. A lamp hung
/// over a model spends one or two; a lamp in the middle of a room spends six.
/// The cost follows the scene rather than the kind, which is what makes six
/// faces affordable at all.
///
/// The index in each pair is the cube face, in the order `+X -X +Y -Y +Z -Z`.
/// A light with a single frustum returns face zero, which is not a direction
/// for it — it is the slot the sampler looks in when the light is not a cube.
pub fn shadow_views(
    light: &crate::lights::GpuLight,
    bounds: Aabb,
    camera: Option<CameraFrustum>,
) -> Vec<(usize, ShadowView)> {
    let position = glam::Vec3::from_slice(&light.position[..3]);
    let direction = glam::Vec3::from_slice(&light.direction[..3]);
    // The two sizes a light can have, each in the unit its own projection
    // works in. See `SourceSize`.
    let source = if light.falloff == PARALLEL_FALLOFF {
        SourceSize::Angular(light.direction[3])
    } else {
        SourceSize::Radius(light.position[3])
    };
    if light.falloff == PARALLEL_FALLOFF {
        let Some(camera) = camera else {
            // No camera to split against — a probe, or a test asking what a
            // sun's frame is. One slice over the whole scene, which is what
            // this did before cascades and is still the right answer when
            // there is no view to spend resolution along.
            return sun_view_of(direction, &bounds.corners(), bounds, source)
                .map(|v| vec![(0, v)])
                .unwrap_or_default();
        };
        // How far the slices have to go: to the far end of the scene, which
        // is a *distance from the eye* and not the scene's own size. Sizing it
        // by the extent was the first attempt and it gives a small scene two
        // hundred metres away no shadow at all — the slices stop a few metres
        // in front of the camera, where there is nothing.
        let reach = CASCADE_REACH.min(
            bounds
                .corners()
                .iter()
                .fold(camera.near * 4.0, |acc, c| acc.max(c.length())),
        );
        let mut out = Vec::with_capacity(CASCADES);
        let mut near = camera.near;
        for (index, fraction) in SPLITS.iter().enumerate() {
            let far = reach * fraction;
            if far <= near {
                continue;
            }
            let slice = frustum_slice(camera.rotation, camera.fov_y, camera.aspect, near, far);
            if let Some(view) = sun_view_of(direction, &slice, bounds, source) {
                out.push((index, view));
            }
            // The next slice starts where this one ended, whether or not this
            // one framed anything: a gap would be a band of the view with no
            // shadow in it, and a slice that found nothing is a slice with
            // nothing in that band anyway.
            near = far;
        }
        return out;
    }
    if light.cone[0] > -1.0 {
        // The stored outer cosine back to a full angle.
        let cone = light.cone[0].clamp(-1.0, 1.0).acos() * 2.0;
        return spot_view(position, direction, cone, bounds, source)
            .map(|v| vec![(0, v)])
            .unwrap_or_default();
    }
    (0..6)
        .filter_map(|face| point_view(position, face, bounds, source).map(|v| (face, v)))
        .collect()
}

/// How many slices a parallel light's shadow is split into.
///
/// One frustum over a whole scene spends its resolution evenly, and a camera
/// does not look evenly: what is near fills the screen and what is far does
/// not. Four slices put four times the texels where the pixels are.
///
/// Four rather than three because the last one is nearly free — it covers the
/// distance where a shadow is a few pixels wide — and rather than five because
/// each is a pass over the scene's geometry and a sun with five is as expensive
/// as five lamps.
pub const CASCADES: usize = 4;

/// Where the slices meet, as fractions of the shadowed distance.
///
/// A blend of the two schemes everybody uses. Splitting by equal distance gives
/// the near slices too little and splitting logarithmically gives the far ones
/// too little; the halfway mix is the practical scheme, and these are what it
/// produces for four slices over a range of a thousand.
///
/// Written out rather than computed, because they are a decision about how the
/// resolution is spent and not a formula anybody should have to re-derive to
/// change it.
const SPLITS: [f32; CASCADES] = [0.06, 0.18, 0.45, 1.0];

/// How far a parallel light's cascades reach, in metres.
///
/// Past this a sun casts no shadow at all. A cap rather than the scene's own
/// extent, because a scene can be a city and the last slice would then be a
/// kilometre of map at a metre a texel — which is not a shadow, it is a stain.
const CASCADE_REACH: f32 = 300.0;

/// The corners of the slice of `camera`'s frustum between two distances.
///
/// Eye-relative, like everything the shadow pass works in — so the eye is the
/// origin and only the rotation matters.
pub fn frustum_slice(
    rotation: glam::Quat,
    fov_y: f32,
    aspect: f32,
    near: f32,
    far: f32,
) -> [glam::Vec3; 8] {
    let t = (fov_y * 0.5).tan();
    let mut out = [glam::Vec3::ZERO; 8];
    for (i, d) in [near, far].into_iter().enumerate() {
        let h = d * t;
        let w = h * aspect;
        for (j, (sx, sy)) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)]
            .into_iter()
            .enumerate()
        {
            // The camera looks down its own -Z, which is why the distance is
            // negated here and nowhere else.
            let local = glam::Vec3::new(sx * w, sy * h, -d);
            out[i * 4 + j] = rotation * local;
        }
    }
    out
}

/// Whether a packed light needs a cube's worth of frames.
///
/// The same two fields [`shadow_views`] dispatches on, asked separately because
/// the sampler needs to know as well: a light with one frustum uses face zero
/// whatever direction the fragment is in, and a cube picks the face.
pub fn is_cube(light: &crate::lights::GpuLight) -> bool {
    light.falloff != PARALLEL_FALLOFF && light.cone[0] <= -1.0
}

/// How many texels of penumbra one unit of stored-depth difference makes.
///
/// A source with size casts a penumbra whose width grows with how far the
/// blocker is from what it shadows. Both projections reduce to the same shape
/// — a constant times the difference in stored depth — and the constants come
/// from different algebra.
///
/// **Orthographic.** Depth is linear in distance, so a depth difference `Δd` is
/// a distance difference `Δd · (far - near)`. A source of angular radius `θ`
/// spreads that into a penumbra `Δdist · tan θ` wide. Divided by a texel, which
/// for an orthographic frame is one size everywhere.
///
/// **Perspective.** Depth is `near / w`, so a blocker at `w_b` and a receiver at
/// `w_r` put a source of radius `r` into a penumbra `r · (w_r - w_b) / w_b`
/// wide at the receiver. Measured in texels there — which are `texel · w_r`
/// across — the `w`s cancel exactly and what is left is
/// `r / (near · texel) · (d_b - d_r)`.
///
/// Exact in both cases, which is worth saying because the usual presentation of
/// this is a similar-triangles sketch and an approximation.
fn penumbra_scale(source: SourceSize, near: f32, far: f32, texel_world: f32) -> f32 {
    if texel_world <= 0.0 {
        return 0.0;
    }
    match source {
        SourceSize::Angular(theta) if theta > 0.0 => {
            (far - near).max(0.0) * theta.tan() / texel_world
        }
        SourceSize::Radius(r) if r > 0.0 && near > 0.0 => r / (near * texel_world),
        // No size. A hard shadow, which nothing real casts and which is exactly
        // what a radius of nothing asks for.
        _ => 0.0,
    }
}

/// How big a light's source is, in the unit its projection can use.
///
/// Two units and not one, because they are different quantities: a sun is
/// infinitely far away and has an apparent size, and a lamp is somewhere and
/// has a physical one. Converting between them needs a distance, and a sun does
/// not have one — which is the same argument `PlacedLight` makes for keeping
/// `radius` and `angular_radius` apart.
#[derive(Clone, Copy, Debug)]
pub enum SourceSize {
    /// Half the apparent size, in radians. A sun's own is 0.0046.
    Angular(f32),
    /// Metres.
    Radius(f32),
}

/// One texel's width in metres at unit distance, for a perspective frustum.
///
/// The frustum spans `2 * tan(fov / 2)` across the map at a distance of one,
/// and proportionally more further out — which is what the shader's multiply
/// by clip `w` supplies.
fn perspective_texel(fov: f32) -> f32 {
    2.0 * (fov * 0.5).tan() / SHADOW_RESOLUTION as f32
}

/// What a shadow pass needs to know about the view it is shadowing.
///
/// Only a parallel light uses it, and only to decide where to spend its
/// resolution. Passed rather than taken from a `Camera` because this module
/// deliberately does not depend on one — a cascade split is geometry, and the
/// path tracer will want the same function with a frustum it made up.
#[derive(Clone, Copy, Debug)]
pub struct CameraFrustum {
    /// Eye-relative, so there is no position: the eye is the origin.
    pub rotation: glam::Quat,
    pub fov_y: f32,
    /// Width over height.
    pub aspect: f32,
    pub near: f32,
}

/// The falloff id a parallel light packs. See `lights::Falloff`.
const PARALLEL_FALLOFF: u32 = 1;

/// One shadow, as the shading shader reads it.
///
/// Parallel to the light buffer: entry `i` belongs to light `i`, and a light
/// that casts nothing has `enabled` at zero. Parallel rather than indexed,
/// because an index would be a second thing to keep in step with a light
/// buffer that is already grown and rewritten every frame — and most scenes
/// have few enough lights that the empty entries cost nothing.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuShadow {
    /// Eye-relative world to the light's clip space, reversed-Z, per cube face.
    ///
    /// Six because a point light sees in every direction. Everything else uses
    /// the first and leaves the rest at zero, which costs three hundred and
    /// twenty bytes a light — a hundred kilobytes at the largest light count
    /// this renderer draws, and the price of the sampler indexing a frame
    /// instead of deriving one.
    ///
    /// Deriving it was the alternative and it is the more tempting one: the six
    /// frames differ only by a rotation, so a shader could build the right one
    /// from the direction. That puts the same construction in two languages,
    /// and the two must agree about which axis is up on each face — which is
    /// exactly the kind of duplication this renderer keeps buying parity tests
    /// to guard. Here the storage is cheap enough that there is nothing to
    /// guard.
    pub view_proj: [[[f32; 4]; 4]; 6],
    /// Which layer of the depth array each face landed in, or -1 for a face
    /// with nothing in it.
    ///
    /// Six used and two of padding, because a storage buffer wants sixteen-byte
    /// alignment and eight `i32` is how that is reached without a separate pad
    /// field to explain.
    pub layers: [i32; 8],
    /// One texel in metres per unit clip `w`, per face. See
    /// [`ShadowView::texel_world`].
    ///
    /// Per face and not per light, which the cube did not need and the cascades
    /// do: the six faces of a cube share a projection and so share a texel,
    /// while the nearest slice of a view covers a tenth of the world the
    /// farthest does. One value for all of them gives the far slices an offset
    /// sized for the near one, which is acne everywhere except close up.
    pub texels: [f32; 8],
    /// How many texels of penumbra one unit of stored-depth difference makes,
    /// per face.
    ///
    /// The whole of what a source's size does to a shadow, in one number per
    /// frame. Multiply it by how much nearer the blocker is than the receiver
    /// and the answer is how wide to filter — which is why a contact is sharp
    /// and a shadow cast from far away is not, with no separate rule for
    /// either.
    ///
    /// Exact for both projections rather than fitted, and they arrive at the
    /// same shape from different algebra. See [`penumbra_scale`].
    ///
    /// Zero is a hard shadow: the source has no size, nothing real does, and it
    /// is what a file asks for by leaving the radius at nothing.
    pub penumbra: [f32; 8],
    /// x: unused. y: one texel in metres per unit clip `w`, see
    /// [`ShadowView::texel_world`]. z: the depth bias. w: nonzero if this light
    /// casts.
    ///
    /// The layer used to live in `x` and now lives per face in `layers`.
    pub params: [f32; 4],
}

impl GpuShadow {
    /// A light that casts nothing.
    ///
    /// Zero twice over, and the second is redundant on purpose. The matrix is
    /// zero, so a position through it comes out with `w` at zero and the
    /// sampler's own near-plane test already returns "lit" — and a sweep that
    /// turned the flag on, or deleted the shader's check for it, changed no
    /// picture anywhere. It stays because "this light casts nothing" is what is
    /// meant, and leaning on a zero matrix to imply it makes the next entry
    /// somebody adds depend on a coincidence.
    pub const NONE: Self = Self {
        view_proj: [[[0.0; 4]; 4]; 6],
        // Every face empty, which is what the sampler tests before it does
        // anything else.
        layers: [-1; 8],
        texels: [0.0; 8],
        penumbra: [0.0; 8],
        params: [0.0; 4],
    };

    /// Packs the frames a light was given, each with the layer it was drawn
    /// into.
    ///
    /// `frames` is what [`shadow_views`] returned, paired with the layer each
    /// went to. An empty list is [`GpuShadow::NONE`]: a light that casts but
    /// framed nothing.
    pub fn pack(frames: &[(usize, ShadowView, u32)], bias: f32) -> Self {
        let mut out = Self::NONE;
        if frames.is_empty() {
            return out;
        }
        for (face, view, layer) in frames {
            let face = face.min(&5);
            out.view_proj[*face] = view.view_proj.to_cols_array_2d();
            out.layers[*face] = *layer as i32;
            out.texels[*face] = view.texel_world;
            out.penumbra[*face] = view.penumbra;
        }
        out.params = [0.0, 0.0, bias, 1.0];
        out
    }
}

/// How much to move a comparison toward the light before making it.
///
/// The smaller half of the acne fix and the one that works in depth. A surface
/// lit at a glancing angle spans more depth across one shadow texel than it
/// does head-on, so the shader scales this by the slope.
///
/// It cannot be the whole fix, and the first version of this file assumed it
/// was. A depth bias is in clip units, and how much depth one texel spans is a
/// property of the *scene* — a frustum around a large scene puts more metres
/// behind each unit of depth, so the number that was ample around a small one
/// leaves the large one crawling with acne. A falsification sweep did not find
/// that, because the fixture lit a floor square-on, where a stored depth and
/// the depth compared against it are equal and reversed-Z's `>=` passes. A
/// fixture with a tilted sun found it in one run.
///
/// The larger half is the normal offset the shader applies, which is measured
/// in texels and so scales with the scene by construction. See
/// [`ShadowView::texel_world`].
///
/// In reversed-Z the bias moves depth *up*, toward the light.
///
/// # The two halves cover for each other, and that is measured
///
/// Setting this to zero changes no test's result. Neither does removing the
/// slope scaling, nor its clamp. But setting this to zero *and* the offset to
/// zero together puts acne over 5,842 of 5,845 lit pixels — so the pair is
/// necessary and neither one is, in any scene there is a test for.
///
/// The reason is that they dominate on different projections. A sun's
/// orthographic depth is linear across the scene, so a texel's worth of depth
/// is tiny in clip units and the lateral offset is what saves it. A spot's is
/// `near / w`, which compresses a distant scene into almost no depth at all, so
/// this constant is enormous out there and covers the acne by itself.
///
/// Both stay. Leaving one out would work today and fail on the first scene that
/// mixes the two kinds of lamp at a distance neither test uses — and a test
/// that passes whether or not a line is present is not worth adding, which is
/// what any test aimed at one half alone would be.
pub const DEPTH_BIAS: f32 = 2.0e-4;

/// The side of one shadow map, in texels.
///
/// A kilobyte square per light at `Depth32Float` is four megabytes a light,
/// which is affordable for the handful of shadow casters a viewport shows and
/// is the resolution below which a contact shadow starts to look like stairs.
pub const SHADOW_RESOLUTION: u32 = 1024;

/// The depth maps, and the pass that fills them.
pub struct ShadowMaps {
    depth: wgpu::Texture,
    /// One view per layer, to render into.
    layers: Vec<wgpu::TextureView>,
    /// The whole array, to sample.
    array: wgpu::TextureView,
    pipeline: wgpu::RenderPipeline,
    object_layout: wgpu::BindGroupLayout,
    /// One globals buffer per layer, written before that layer's pass.
    globals: Vec<wgpu::Buffer>,
    globals_binds: Vec<wgpu::BindGroup>,
    objects: wgpu::Buffer,
    object_bind: wgpu::BindGroup,
    /// The same pass for a see-through object: its texels drop out by its
    /// transmission. See `shadow.wgsl`.
    hashed: wgpu::RenderPipeline,
    object_stride: u32,
    object_capacity: u32,
}

/// The most shadow maps a frame will draw.
///
/// Layers, not lights: a point light in the middle of a room spends six of
/// these and a sun spends one. A cap rather than a growing array, because each
/// layer is a full pass over the scene's geometry — the cost is in the draws,
/// and a scene that wants a hundred shadowed lights wants the clustered path
/// and a budget rather than a hundred passes.
///
/// Sixteen at `Depth32Float` and a kilobyte square is sixty-four megabytes,
/// which is the real reason this is not larger. Eight was enough while a point
/// light took one layer and is not enough for a single lamp in a room.
/// One draw's uniform: a model matrix, then how see-through it is and the
/// frame's seed.
const OBJECT_SIZE: u64 = 80;

pub const MAX_SHADOWS: usize = 16;

impl ShadowMaps {
    pub fn new(device: &wgpu::Device) -> Self {
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow maps"),
            size: wgpu::Extent3d {
                width: SHADOW_RESOLUTION,
                height: SHADOW_RESOLUTION,
                depth_or_array_layers: MAX_SHADOWS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: crate::gpu::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let layers = (0..MAX_SHADOWS)
            .map(|i| {
                depth.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("shadow layer"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i as u32,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let array = depth.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let uniform = |label: &str, dynamic: bool, size: u64| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX.union(wgpu::ShaderStages::FRAGMENT),
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: dynamic,
                        min_binding_size: wgpu::BufferSize::new(size),
                    },
                    count: None,
                }],
            })
        };
        let globals_layout = uniform("shadow globals", false, 64);
        let object_layout = uniform("shadow object", true, OBJECT_SIZE);

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shadow.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow"),
            bind_group_layouts: &[Some(&globals_layout), Some(&object_layout)],
            immediate_size: 0,
        });
        let pipeline_of = |label: &str, fragment: Option<&str>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_shadow"),
                    compilation_options: Default::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: 12,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        }],
                    }],
                },
                // None for a solid: a shadow map is a depth buffer. See the
                // shader's header. A see-through object's texels drop out.
                fragment: fragment.map(|entry| wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    targets: &[],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(crate::gpu::depth_write_state()),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let pipeline = pipeline_of("shadow", None);
        let hashed = pipeline_of("shadow see-through", Some("fs_hashed"));

        let stride = Self::stride(device);
        let (objects, object_bind) = Self::make_objects(device, &object_layout, stride, 8);
        let globals: Vec<wgpu::Buffer> = (0..MAX_SHADOWS)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("shadow globals"),
                    size: 64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let globals_binds = globals
            .iter()
            .map(|buffer| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("shadow globals"),
                    layout: &globals_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                })
            })
            .collect();

        Self {
            depth,
            layers,
            array,
            pipeline,
            hashed,
            object_layout,
            globals,
            globals_binds,
            objects,
            object_bind,
            object_stride: stride,
            object_capacity: 8,
        }
    }

    /// The distance between per-draw uniform slots.
    ///
    /// The device's minimum dynamic offset alignment, which is 256 on most
    /// hardware and is not a number to assume.
    fn stride(device: &wgpu::Device) -> u32 {
        let align = device.limits().min_uniform_buffer_offset_alignment.max(1);
        (OBJECT_SIZE as u32).div_ceil(align) * align
    }

    fn make_objects(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        stride: u32,
        capacity: u32,
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow objects"),
            size: (stride as u64) * capacity.max(1) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow objects"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(OBJECT_SIZE),
                }),
            }],
        });
        (buffer, bind)
    }

    /// The array, for the shading pass to sample.
    pub fn array_view(&self) -> &wgpu::TextureView {
        &self.array
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.depth
    }

    /// Makes room for `count` draws.
    pub fn reserve(&mut self, device: &wgpu::Device, count: u32) {
        if count <= self.object_capacity {
            return;
        }
        let capacity = count.next_power_of_two();
        let (buffer, bind) =
            Self::make_objects(device, &self.object_layout, self.object_stride, capacity);
        self.objects = buffer;
        self.object_bind = bind;
        self.object_capacity = capacity;
    }

    /// Writes one draw's model matrix, how much light passes through it, and
    /// the frame's seed for the texels that drop out.
    pub fn set_object(
        &self,
        queue: &wgpu::Queue,
        index: u32,
        model: &glam::Mat4,
        see_through: f32,
        seed: u32,
    ) {
        let mut data = [0.0f32; 20];
        data[..16].copy_from_slice(&model.to_cols_array());
        data[16] = see_through.clamp(0.0, 1.0);
        data[17] = (seed % 4096) as f32;
        queue.write_buffer(
            &self.objects,
            (self.object_stride * index) as u64,
            bytemuck::cast_slice(&data),
        );
    }

    /// Writes the light's view-projection for one layer.
    pub fn set_view(&self, queue: &wgpu::Queue, layer: usize, view: &ShadowView) {
        if let Some(buffer) = self.globals.get(layer) {
            queue.write_buffer(
                buffer,
                0,
                bytemuck::cast_slice(&view.view_proj.to_cols_array()),
            );
        }
    }

    /// Renders one layer's depth.
    ///
    /// `draws` yields `(index, positions, indices, triangles)` — the index is
    /// the slot [`ShadowMaps::set_object`] wrote, so the two cannot disagree
    /// about which matrix belongs to which draw.
    pub fn encode<'a>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layer: usize,
        draws: impl IntoIterator<Item = (u32, &'a wgpu::Buffer, &'a wgpu::Buffer, u32, bool)>,
    ) {
        let Some(view) = self.layers.get(layer) else {
            return;
        };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shadow"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view,
                depth_ops: Some(wgpu::Operations {
                    // Reversed-Z clears to zero: nothing has been drawn, so
                    // everything is infinitely far from the light.
                    load: wgpu::LoadOp::Clear(crate::gpu::DEPTH_CLEAR as f32),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        let Some(globals) = self.globals_binds.get(layer) else {
            return;
        };
        pass.set_bind_group(0, globals, &[]);
        let mut hashing = false;
        for (index, positions, indices, triangles, see_through) in draws {
            if triangles == 0 {
                continue;
            }
            if see_through != hashing {
                pass.set_pipeline(if see_through {
                    &self.hashed
                } else {
                    &self.pipeline
                });
                hashing = see_through;
            }
            pass.set_bind_group(1, &self.object_bind, &[self.object_stride * index]);
            pass.set_vertex_buffer(0, positions.slice(..));
            pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..triangles * 3, 0, 0..1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene() -> Aabb {
        Aabb::new(
            glam::Vec3::new(-3.0, -2.0, 0.0),
            glam::Vec3::new(4.0, 5.0, 2.0),
        )
    }

    /// Where a point lands in the shadow's clip space, after the divide.
    fn project(view: &ShadowView, p: glam::Vec3) -> glam::Vec3 {
        let clip = view.view_proj * p.extend(1.0);
        clip.truncate() / clip.w
    }

    fn inside(p: glam::Vec3) -> bool {
        p.x.abs() <= 1.0001 && p.y.abs() <= 1.0001 && (-0.0001..=1.0001).contains(&p.z)
    }

    use crate::lights::{Falloff, GpuLight, PlacedLight};

    fn packed(light: PlacedLight) -> GpuLight {
        GpuLight::pack(&light, glam::DVec3::ZERO)
    }

    #[test]
    fn a_lights_own_fields_say_which_frame_it_wants() {
        // Read off the packed form rather than passed a kind, so the shadow
        // frame and the shading cannot disagree about what a light is.
        let sun = packed(PlacedLight {
            falloff: Falloff::Parallel,
            direction: glam::Vec3::NEG_Z,
            ..PlacedLight::default()
        });
        let spot = packed(PlacedLight {
            position: glam::DVec3::new(0.0, 0.0, 8.0),
            direction: glam::Vec3::NEG_Z,
            cone: Some((60f32.to_radians(), 0.2)),
            ..PlacedLight::default()
        });
        let point = packed(PlacedLight {
            position: glam::DVec3::new(0.0, 0.0, 8.0),
            ..PlacedLight::default()
        });

        /// The one frame a light with a single frustum gets.
        fn only(frames: Vec<(usize, ShadowView)>) -> ShadowView {
            assert_eq!(frames.len(), 1, "expected a single frustum");
            assert_eq!(frames[0].0, 0, "a single frustum belongs in face zero");
            frames[0].1
        }

        // A sun's frame is orthographic: parallel lines stay parallel, so two
        // points at different depths on the same axis land at the same place.
        let view = only(shadow_views(&sun, scene(), None));
        let near = project(&view, glam::Vec3::new(1.0, 1.0, 2.0));
        let far = project(&view, glam::Vec3::new(1.0, 1.0, 0.0));
        assert!((near.x - far.x).abs() < 1e-5, "a sun is not orthographic");

        // A spot's is perspective: they do not.
        let view = only(shadow_views(&spot, scene(), None));
        let near = project(&view, glam::Vec3::new(1.0, 1.0, 2.0));
        let far = project(&view, glam::Vec3::new(1.0, 1.0, 0.0));
        assert!((near.x - far.x).abs() > 1e-3, "a spot is not perspective");

        // And a point light gets several — one per face with anything in it,
        // which is what a single frustum could never cover.
        let faces = shadow_views(&point, scene(), None);
        assert!(faces.len() > 1, "a point light got {} frames", faces.len());
        assert!(is_cube(&point) && !is_cube(&sun) && !is_cube(&spot));
    }

    #[test]
    fn a_point_light_in_the_middle_of_a_scene_gets_all_six_faces() {
        // A lamp with scene on every side of it. Six frames, and between them
        // they cover every direction — which is the whole difference between
        // this and the one face the dispatch used to hand back.
        let light = packed(PlacedLight {
            position: glam::DVec3::ZERO,
            ..PlacedLight::default()
        });
        let big = Aabb::new(glam::Vec3::splat(-10.0), glam::Vec3::splat(10.0));
        let frames = shadow_views(&light, big, None);
        assert_eq!(frames.len(), 6, "a lamp in a room needs every face");

        // Each probe lands inside the face a cube map would send it to.
        for (probe, face) in [
            (glam::Vec3::new(5.0, 0.0, 0.0), 0),
            (glam::Vec3::new(-5.0, 0.0, 0.0), 1),
            (glam::Vec3::new(0.0, 5.0, 0.0), 2),
            (glam::Vec3::new(0.0, -5.0, 0.0), 3),
            (glam::Vec3::new(0.0, 0.0, 5.0), 4),
            (glam::Vec3::new(0.0, 0.0, -5.0), 5),
        ] {
            let view = frames.iter().find(|(f, _)| *f == face).expect("the face").1;
            assert!(inside(project(&view, probe)), "face {face} missed {probe}");
        }
    }

    #[test]
    fn a_lamp_over_a_model_pays_for_the_faces_it_uses() {
        // The reason six faces are affordable: `point_view` answers nothing for
        // a face that frames nothing, so the cost follows the scene rather than
        // the kind. A lamp hung over a model looks down at it and the faces
        // pointing at empty sky cost nothing.
        let light = packed(PlacedLight {
            position: glam::DVec3::new(0.0, 0.0, 9.0),
            ..PlacedLight::default()
        });
        // A floor, well below the lamp.
        let floor = Aabb::new(
            glam::Vec3::new(-8.0, -8.0, -0.1),
            glam::Vec3::new(8.0, 8.0, 0.1),
        );
        let frames = shadow_views(&light, floor, None);
        assert!(
            frames.len() < 6,
            "a lamp over a floor paid for {} faces",
            frames.len()
        );
        // And the one looking down is among them, or it shadows nothing at all.
        assert!(
            frames.iter().any(|(face, _)| *face == 5),
            "no downward face"
        );
    }

    #[test]
    fn a_penumbra_is_the_source_size_carried_to_the_receiver() {
        // Both halves of `penumbra_scale`, against numbers worked out by hand.
        // A closed form deserves an exact test, and the two halves arrive at
        // the same shape from different algebra — which is exactly the sort of
        // pair where one can be quietly wrong.
        let texel = 0.01;

        // Orthographic. A depth difference of one spans the whole range, so a
        // source of angular radius θ spreads it into `(far - near) · tan θ`
        // of world, and that many texels.
        let theta = 0.05f32;
        let ortho = penumbra_scale(SourceSize::Angular(theta), 1.0, 51.0, texel);
        let expected = 50.0 * theta.tan() / texel;
        assert!(
            (ortho - expected).abs() < expected * 1e-4,
            "{ortho} against {expected}"
        );
        // Twice the range is twice the penumbra, and so is twice the angle at
        // this size — tan is very nearly linear here.
        let wider = penumbra_scale(SourceSize::Angular(theta), 1.0, 101.0, texel);
        assert!((wider / ortho - 2.0).abs() < 0.01);

        // Perspective. The `w`s cancel and what is left is `r / (near · texel)`
        // — no far plane in it at all, which is the part that is easy to get
        // wrong by writing the orthographic form twice.
        let perspective = penumbra_scale(SourceSize::Radius(0.5), 0.1, 40.0, texel);
        let expected = 0.5 / (0.1 * texel);
        assert!(
            (perspective - expected).abs() < expected * 1e-4,
            "{perspective} against {expected}"
        );
        // Changing the far plane changes nothing.
        assert_eq!(
            perspective,
            penumbra_scale(SourceSize::Radius(0.5), 0.1, 400.0, texel)
        );
        // A bulb twice as wide is twice the penumbra; a near plane twice as
        // far is half of it.
        assert!(
            (penumbra_scale(SourceSize::Radius(1.0), 0.1, 40.0, texel) / perspective - 2.0).abs()
                < 1e-4
        );
        assert!(
            (penumbra_scale(SourceSize::Radius(0.5), 0.2, 40.0, texel) / perspective - 0.5).abs()
                < 1e-4
        );

        // And no size is a hard shadow, whichever unit the nothing is in.
        assert_eq!(
            penumbra_scale(SourceSize::Angular(0.0), 1.0, 51.0, texel),
            0.0
        );
        assert_eq!(
            penumbra_scale(SourceSize::Radius(0.0), 0.1, 40.0, texel),
            0.0
        );
    }

    #[test]
    fn each_kind_of_light_reads_its_size_from_its_own_field() {
        // A sun has an apparent size and a lamp has a physical one, and they
        // are not interchangeable: converting between them needs a distance and
        // a sun does not have one. So each projection reads the field that is
        // in its own unit, and a value in the other one has to be ignored
        // rather than used.
        let scene = Aabb::new(glam::Vec3::splat(-5.0), glam::Vec3::splat(5.0));
        let sun = packed(PlacedLight {
            falloff: Falloff::Parallel,
            direction: glam::Vec3::NEG_Z,
            angular_radius: 0.05,
            // A metre-wide bulb on a sun is a caller's mistake, not a size.
            radius: 1.0,
            ..PlacedLight::default()
        });
        let soft = shadow_views(&sun, scene, None);
        assert!(soft[0].1.penumbra > 0.0);

        let hard_sun = packed(PlacedLight {
            falloff: Falloff::Parallel,
            direction: glam::Vec3::NEG_Z,
            angular_radius: 0.0,
            radius: 1.0,
            ..PlacedLight::default()
        });
        assert_eq!(
            shadow_views(&hard_sun, scene, None)[0].1.penumbra,
            0.0,
            "a sun took its size from a bulb"
        );

        let lamp = packed(PlacedLight {
            position: glam::DVec3::new(0.0, 0.0, 8.0),
            radius: 0.0,
            // And an apparent size on a lamp is the same mistake reversed.
            angular_radius: 0.05,
            ..PlacedLight::default()
        });
        for (_, view) in shadow_views(&lamp, scene, None) {
            assert_eq!(view.penumbra, 0.0, "a lamp took its size from an angle");
        }
    }

    #[test]
    fn a_texel_is_measured_across_the_wider_axis() {
        // The map is square and the box it covers is not, so one texel is wider
        // on one axis than the other. The normal offset has to clear the texel
        // the surface itself wrote, so the number that matters is the worst of
        // the two and not the average or the better one.
        //
        // A square scene cannot tell the two apart, and the device tests use
        // one: a sweep that swapped `max` for `min` walked through all of them.
        let long = Aabb::new(
            glam::Vec3::new(-50.0, -1.0, 0.0),
            glam::Vec3::new(50.0, 1.0, 1.0),
        );
        let view = sun_view(glam::Vec3::NEG_Z, long).expect("a scene");
        // A hundred metres across a thousand texels.
        let expected = 100.0 / SHADOW_RESOLUTION as f32;
        assert!(
            (view.texel_world - expected).abs() < expected * 0.05,
            "texel is {} across a 100 x 2 scene",
            view.texel_world
        );
    }

    #[test]
    fn a_perspective_texel_grows_with_the_cone_and_with_distance() {
        // Per unit of clip `w`, which for a perspective projection is the
        // distance along the light's axis — so one field covers both the
        // widening of the frustum and the sampler's multiply by `w`.
        //
        // A wider cone spreads the same thousand texels over more world, so
        // each one covers more of it.
        let narrow = spot_view(
            glam::Vec3::new(0.0, 0.0, 8.0),
            glam::Vec3::NEG_Z,
            20f32.to_radians(),
            scene(),
            SourceSize::Radius(0.0),
        )
        .expect("a scene");
        let wide = spot_view(
            glam::Vec3::new(0.0, 0.0, 8.0),
            glam::Vec3::NEG_Z,
            100f32.to_radians(),
            scene(),
            SourceSize::Radius(0.0),
        )
        .expect("a scene");
        assert!(
            wide.texel_world > narrow.texel_world * 4.0,
            "{} against {}",
            wide.texel_world,
            narrow.texel_world
        );

        // And an orthographic one does not: a parallel light's texel is the
        // same size everywhere, which is why the field is "per unit `w`" and
        // an orthographic `w` is one.
        let sun = sun_view(glam::Vec3::NEG_Z, scene()).expect("a scene");
        assert!(sun.texel_world > 0.0);
    }

    /// A camera looking out across the ground rather than down at it.
    ///
    /// A camera turns its own -Z toward what it is looking at, and in this
    /// Z-up world an unrotated one is therefore pointing at the floor — the
    /// view leaves the scene within a metre and every slice past the first has
    /// nothing in it. Which is correct, and measures nothing.
    fn a_camera(near: f32) -> CameraFrustum {
        CameraFrustum {
            // Ninety degrees about X sends local -Z to world +Y.
            rotation: glam::Quat::from_rotation_x(std::f32::consts::FRAC_PI_2),
            fov_y: 45f32.to_radians(),
            aspect: 16.0 / 9.0,
            near,
        }
    }

    /// Ground stretching away in front of `a_camera`, with room above it.
    fn a_landscape() -> Aabb {
        Aabb::new(
            glam::Vec3::new(-150.0, 0.0, -2.0),
            glam::Vec3::new(150.0, 300.0, 20.0),
        )
    }

    #[test]
    fn a_sun_spends_its_resolution_where_the_pixels_are() {
        // The whole of why cascades exist, as a number. One frustum over a
        // large scene gives every metre the same share of a map, and a camera
        // does not look that way: what is near fills the screen.
        //
        // Measured as the width each slice covers. The near one has to be much
        // smaller than the whole scene, or the resolution has not moved.
        let big = a_landscape();
        let sun = packed(PlacedLight {
            falloff: Falloff::Parallel,
            direction: glam::Vec3::new(0.3, 0.4, -0.87).normalize(),
            ..PlacedLight::default()
        });
        let cascades = shadow_views(&sun, big, Some(a_camera(0.1)));
        assert!(
            cascades.len() >= 3,
            "a large scene got {} cascades",
            cascades.len()
        );

        // Each covers less world than the next, which is what "nested" means
        // and what makes searching for the first that contains a point find the
        // tightest.
        let widths: Vec<f32> = cascades
            .iter()
            .map(|(_, v)| 1.0 / v.view_proj.x_axis.length())
            .collect();
        for pair in widths.windows(2) {
            assert!(pair[0] < pair[1], "cascades are not nested: {widths:?}");
        }
        // And the nearest is a small fraction of the whole, or nothing was
        // gained: one frustum over this scene is three hundred metres across.
        assert!(
            widths[0] * 10.0 < widths[widths.len() - 1],
            "the near cascade is {} against {}",
            widths[0],
            widths[widths.len() - 1]
        );
    }

    #[test]
    fn the_near_cascade_has_far_finer_texels_than_one_frustum_would() {
        // The number cascades exist to move, measured in the unit that matters:
        // how much world one texel of the map covers. Pixel comparisons cannot
        // see this — a coarser map still puts the shadow in the right place, it
        // just puts it there in steps — so it is measured where it is exact.
        let big = a_landscape();
        let sun = packed(PlacedLight {
            falloff: Falloff::Parallel,
            direction: glam::Vec3::new(0.3, 0.4, -0.87).normalize(),
            ..PlacedLight::default()
        });
        let one = sun_view(glam::Vec3::new(0.3, 0.4, -0.87).normalize(), big).expect("one frustum");
        let cascades = shadow_views(&sun, big, Some(a_camera(0.1)));
        let near = cascades.first().expect("a near cascade").1;

        assert!(
            near.texel_world * 10.0 < one.texel_world,
            "the near cascade's texel is {} against one frustum's {}",
            near.texel_world,
            one.texel_world
        );

        // And each slice carries its own, because they differ by that much: one
        // texel size for the whole light gives the far slices an offset sized
        // for the near one, which is acne everywhere but close up.
        let packed_shadow = GpuShadow::pack(
            &cascades
                .iter()
                .enumerate()
                .map(|(i, (face, view))| (*face, *view, i as u32))
                .collect::<Vec<_>>(),
            DEPTH_BIAS,
        );
        let used: Vec<f32> = cascades
            .iter()
            .map(|(face, _)| packed_shadow.texels[*face])
            .collect();
        assert!(
            used[0] * 5.0 < used[used.len() - 1],
            "every slice was packed the same texel: {used:?}"
        );
    }

    #[test]
    fn each_cascade_covers_its_own_band_and_not_all_of_them() {
        // Slices that all started at the camera would cover correctly and waste
        // most of what cascades buy: the second would span everything the first
        // does plus its own band, and the third everything again.
        //
        // The test for "its own band" is a point the *first* slice covers: the
        // ones after it must not. Asking about the eye instead does not work,
        // because every slice starts at the camera's near plane and so none of
        // them contains it either way.
        let big = a_landscape();
        let sun = packed(PlacedLight {
            falloff: Falloff::Parallel,
            // Across the view: along it, every slice projects onto the same
            // patch of the map and the eye is inside all of them, correctly.
            direction: glam::Vec3::X,
            ..PlacedLight::default()
        });
        let cascades = shadow_views(&sun, big, Some(a_camera(0.1)));
        assert!(cascades.len() >= 3);
        // Five metres in front of the camera, which is well inside the first
        // slice and well short of the second.
        let near_probe = glam::Vec3::new(0.0, 5.0, 0.0);
        assert!(
            inside(project(&cascades[0].1, near_probe)),
            "the first cascade does not cover its own band"
        );
        for (index, view) in cascades.iter().skip(1) {
            assert!(
                !inside(project(view, near_probe)),
                "cascade {index} reaches back over the first one's band"
            );
        }
    }

    #[test]
    fn a_small_scene_far_away_still_gets_a_cascade() {
        // How far the slices reach is a distance from the eye, not the scene's
        // own size. Sizing it by the extent leaves a two-metre object two
        // hundred metres away with no shadow at all, because the slices stop a
        // few metres in front of the camera where there is nothing.
        let sun = packed(PlacedLight {
            falloff: Falloff::Parallel,
            direction: glam::Vec3::X,
            ..PlacedLight::default()
        });
        let far_away = Aabb::new(
            glam::Vec3::new(-1.0, 198.0, -1.0),
            glam::Vec3::new(1.0, 200.0, 1.0),
        );
        let frames = shadow_views(&sun, far_away, Some(a_camera(0.1)));
        assert!(!frames.is_empty(), "a distant scene got no cascade at all");
        for corner in far_away.corners() {
            let covered = frames.iter().any(|(_, v)| inside(project(v, corner)));
            assert!(covered, "no cascade covers {corner}");
        }
    }

    #[test]
    fn a_cascade_reaches_back_to_the_light_past_the_whole_scene() {
        // A slice of the view is a box in the middle of the air, and a wall
        // between it and the sun is *outside* the slice. If the near plane were
        // fitted to the slice, that wall would be clipped out of the map and
        // its shadow would vanish — which is the failure people describe as
        // "shadows disappear when I look away from the thing casting them".
        let scene = Aabb::new(
            glam::Vec3::new(-40.0, 0.0, -1.0),
            glam::Vec3::new(40.0, 80.0, 30.0),
        );
        let sun = packed(PlacedLight {
            falloff: Falloff::Parallel,
            direction: glam::Vec3::NEG_Z,
            ..PlacedLight::default()
        });
        let cascades = shadow_views(&sun, scene, Some(a_camera(0.1)));
        let near = cascades.first().expect("a cascade").1;
        // Something at the very top of the scene, far above any near slice, is
        // still inside the map's depth range.
        let high = glam::Vec3::new(0.0, 4.0, 29.5);
        let p = project(&near, high);
        assert!(
            (-0.0001..=1.0001).contains(&p.z),
            "a caster at the top of the scene is outside the near cascade's              depth at {p}"
        );
    }

    #[test]
    fn without_a_camera_a_sun_gets_one_frustum_over_everything() {
        // A probe, or anything asking what a sun's frame is with no view to
        // split along. One slice over the whole scene, which is what this did
        // before cascades and is still the only answer available.
        let sun = packed(PlacedLight {
            falloff: Falloff::Parallel,
            direction: glam::Vec3::NEG_Z,
            ..PlacedLight::default()
        });
        let frames = shadow_views(&sun, scene(), None);
        assert_eq!(frames.len(), 1);
        for corner in scene().corners() {
            assert!(inside(project(&frames[0].1, corner)));
        }
    }

    #[test]
    fn a_slice_outside_the_scene_is_no_cascade_at_all() {
        // A camera a long way from a small scene: the near slices are in empty
        // air. They are skipped rather than drawn, because a pass over the
        // geometry that frames none of it is a pass for nothing — and the
        // slices that follow still line up, because each starts where the last
        // ended whether or not it framed anything.
        // Across the view rather than along it: a sun looking the same way the
        // camera does projects every slice onto the same patch of its map, so
        // they all overlap the scene and none is skipped. That is correct and
        // it is not what this is measuring.
        let sun = packed(PlacedLight {
            falloff: Falloff::Parallel,
            direction: glam::Vec3::X,
            ..PlacedLight::default()
        });
        let far_away = Aabb::new(
            glam::Vec3::new(-1.0, 198.0, -1.0),
            glam::Vec3::new(1.0, 200.0, 1.0),
        );
        let frames = shadow_views(&sun, far_away, Some(a_camera(0.1)));
        assert!(
            frames.len() < CASCADES,
            "every slice framed something: {}",
            frames.len()
        );
    }

    #[test]
    fn a_suns_box_contains_the_whole_scene() {
        // A frustum too small crops the shadow at its edge, and the symptom is
        // a shadow that ends in mid-air.
        for direction in [
            glam::Vec3::NEG_Z,
            glam::Vec3::new(0.3, -0.5, -0.81).normalize(),
            glam::Vec3::X,
            glam::Vec3::new(-0.7, 0.7, 0.1).normalize(),
        ] {
            let view = sun_view(direction, scene()).expect("a scene");
            for corner in scene().corners() {
                let p = project(&view, corner);
                assert!(
                    inside(p),
                    "{direction} left {corner} at {p} outside the shadow"
                );
            }
        }
    }

    #[test]
    fn a_suns_box_is_no_bigger_than_it_needs_to_be() {
        // The other half, and the one that shows as stippled contact shadows
        // rather than as a missing shadow: a box fitted to the world-axis
        // extents is larger than the rotated scene at every angle but one, and
        // the excess comes out of the resolution.
        //
        // Measured by how far the scene actually spreads across the map. A
        // snug fit touches both edges.
        let view =
            sun_view(glam::Vec3::new(0.3, -0.5, -0.81).normalize(), scene()).expect("a scene");
        let (mut lo, mut hi) = (
            glam::Vec2::splat(f32::INFINITY),
            glam::Vec2::splat(f32::NEG_INFINITY),
        );
        for corner in scene().corners() {
            let p = project(&view, corner);
            lo = lo.min(p.truncate());
            hi = hi.max(p.truncate());
        }
        assert!(lo.x < -0.98 && lo.y < -0.98, "the box is loose: {lo}");
        assert!(hi.x > 0.98 && hi.y > 0.98, "the box is loose: {hi}");
    }

    #[test]
    fn depth_increases_toward_the_light() {
        // Reversed-Z, like the rest of the renderer. The other convention is a
        // depth test that always passes, and a scene with no shadows in it at
        // all.
        let direction = glam::Vec3::NEG_Z;
        let view = sun_view(direction, scene()).expect("a scene");
        // The light looks down -Z, so the top of the scene is nearer to it.
        let near_point = glam::Vec3::new(0.0, 0.0, 2.0);
        let far_point = glam::Vec3::new(0.0, 0.0, 0.0);
        assert!(
            project(&view, near_point).z > project(&view, far_point).z,
            "depth does not increase toward the light"
        );
        // And the ends land on the range's ends, or the precision is being
        // spent somewhere the geometry is not.
        assert!(project(&view, near_point).z > 0.99);
        assert!(project(&view, far_point).z < 0.01);
    }

    #[test]
    fn a_sun_has_no_position_and_its_view_does_not_depend_on_one() {
        // A parallel light is a direction. A `sun_view` that took a position
        // would be a frustum somebody had to place, and placing it wrongly
        // crops the scene.
        let a = sun_view(glam::Vec3::NEG_Z, scene()).expect("a scene");
        let shifted = Aabb::new(
            glam::Vec3::new(-3.0, -2.0, 10.0),
            glam::Vec3::new(4.0, 5.0, 12.0),
        );
        let b = sun_view(glam::Vec3::NEG_Z, shifted).expect("a scene");
        // The two frames differ only in depth: the scene moved along the
        // light's axis, so the box slid rather than resized.
        for corner in shifted.corners() {
            assert!(inside(project(&b, corner)));
        }
        assert!(a.far - a.near > 0.0 && b.far - b.near > 0.0);
    }

    #[test]
    fn an_empty_scene_gives_no_shadow_camera() {
        // Nothing to cast and nothing to catch. A pass that ran anyway would
        // render a map nothing samples, every frame.
        assert!(sun_view(glam::Vec3::NEG_Z, Aabb::EMPTY).is_none());
        assert!(spot_view(
            glam::Vec3::ZERO,
            glam::Vec3::NEG_Z,
            1.0,
            Aabb::EMPTY,
            SourceSize::Radius(0.0)
        )
        .is_none());
        assert!(point_view(glam::Vec3::ZERO, 0, Aabb::EMPTY, SourceSize::Radius(0.0)).is_none());
    }

    #[test]
    fn a_spot_frames_what_is_in_front_of_it() {
        // On the axis at any distance the scene reaches, and not behind.
        let position = glam::Vec3::new(0.0, 0.0, 8.0);
        let view = spot_view(
            position,
            glam::Vec3::NEG_Z,
            60f32.to_radians(),
            scene(),
            SourceSize::Radius(0.0),
        )
        .expect("a scene");
        for z in [6.0f32, 4.0, 1.0] {
            let p = project(&view, glam::Vec3::new(0.0, 0.0, z));
            assert!(inside(p), "a point on the axis at {z} landed at {p}");
        }
        // Behind the light is outside, and the divide does not make it look
        // inside — which is what a missing `w` check does.
        let behind = view.view_proj * glam::Vec3::new(0.0, 0.0, 10.0).extend(1.0);
        assert!(
            behind.w <= 0.0,
            "a point behind the light had w {}",
            behind.w
        );
    }

    #[test]
    fn a_spot_pointing_away_from_everything_has_nothing_to_draw() {
        // Reachable: a lamp aimed out of a window. Rendering a map for it is
        // a pass whose every fragment is clipped.
        let view = spot_view(
            glam::Vec3::new(0.0, 0.0, 8.0),
            glam::Vec3::Z,
            1.0,
            scene(),
            SourceSize::Radius(0.0),
        );
        assert!(view.is_none());
    }

    #[test]
    fn a_wider_cone_frames_more() {
        // The cone angle reaches the projection. A spot whose map ignored it
        // would have the same resolution however narrow the beam, which is
        // the resolution being wasted where it is most visible.
        let position = glam::Vec3::new(0.0, 0.0, 8.0);
        let off_axis = glam::Vec3::new(2.0, 0.0, 4.0);
        let narrow = spot_view(
            position,
            glam::Vec3::NEG_Z,
            20f32.to_radians(),
            scene(),
            SourceSize::Radius(0.0),
        )
        .expect("a scene");
        let wide = spot_view(
            position,
            glam::Vec3::NEG_Z,
            100f32.to_radians(),
            scene(),
            SourceSize::Radius(0.0),
        )
        .expect("a scene");
        let n = project(&narrow, off_axis).x.abs();
        let w = project(&wide, off_axis).x.abs();
        assert!(w < n, "the wider cone did not frame more: {w} against {n}");
    }

    #[test]
    fn the_six_faces_of_a_point_light_cover_every_direction() {
        // Six 90° frustums meeting without a gap. A gap is a wedge of the
        // scene that casts no shadow, and it moves with the light.
        let position = glam::Vec3::new(0.5, 1.0, 1.0);
        let big = Aabb::new(glam::Vec3::splat(-20.0), glam::Vec3::splat(20.0));
        let mut probes = Vec::new();
        for i in 0..200 {
            // A spread of directions from a deterministic spiral, so the test
            // is not testing a random number generator.
            let t = i as f32 / 200.0;
            let phi = t * std::f32::consts::TAU * 7.0;
            let z = 1.0 - 2.0 * t;
            let r = (1.0 - z * z).max(0.0).sqrt();
            probes.push(position + glam::Vec3::new(r * phi.cos(), r * phi.sin(), z) * 5.0);
        }
        for probe in probes {
            let seen = (0..6).any(|face| {
                point_view(position, face, big, SourceSize::Radius(0.0))
                    .map(|view| inside(project(&view, probe)))
                    .unwrap_or(false)
            });
            assert!(seen, "no face of the cube sees {probe}");
        }
    }

    #[test]
    fn the_near_plane_is_not_a_millimetre() {
        // Reversed-Z puts its precision near the near plane, so a near plane
        // at a millimetre spends most of the buffer on the first centimetre —
        // which is exactly where nothing is.
        let view = spot_view(
            glam::Vec3::new(0.0, 0.0, 8.0),
            glam::Vec3::NEG_Z,
            1.0,
            scene(),
            SourceSize::Radius(0.0),
        )
        .expect("a scene");
        // A literal and not `MIN_NEAR`: the claim is about a physical
        // magnitude, and an assertion written against the constant is
        // satisfied by whatever the constant happens to be — which is no
        // assertion at all. A sweep that set it to a millimetre survived
        // this test until the number was written out.
        assert!(view.near >= 0.099, "near is {}", view.near);
        assert!(view.far > view.near);
    }

    #[test]
    fn a_light_pointing_along_the_up_axis_still_has_a_frame() {
        // The degenerate case a fixed up vector produces, and the commonest
        // light in any scene: a sun pointing straight down.
        for direction in [glam::Vec3::Z, glam::Vec3::NEG_Z] {
            let view = sun_view(direction, scene()).expect("a scene");
            for corner in scene().corners() {
                let p = project(&view, corner);
                assert!(p.x.is_finite() && p.y.is_finite() && p.z.is_finite());
                assert!(inside(p), "{direction} left {corner} at {p}");
            }
        }
    }

    #[test]
    fn a_point_on_the_seam_between_two_cube_faces_is_well_inside_both_maps() {
        // The seam is where a bright line ran through point-light shadows:
        // the filter around a point on it read past the edge of its face's
        // map, and past the edge is "lit". Each face reaches far enough past
        // 90° that the widest filter still lands on rendered depth.
        let light = glam::Vec3::new(0.0, 0.0, 3.0);
        let scene = Aabb::new(glam::Vec3::splat(-5.0), glam::Vec3::splat(5.0));
        // Straight down and out along +X at 45°: the edge of -Z and +X.
        let point = light + glam::Vec3::new(1.0, 0.0, -1.0);
        for face in [0, 5] {
            let view = point_view(light, face, scene, SourceSize::Radius(0.1))
                .expect("the face sees the scene");
            let clip = view.view_proj * point.extend(1.0);
            let ndc = clip.truncate() / clip.w;
            let texels_in = (1.0 - ndc.x.abs().max(ndc.y.abs())) * 0.5 * SHADOW_RESOLUTION as f32;
            assert!(
                texels_in >= 16.0,
                "face {face}: only {texels_in} texels inside the map"
            );
        }
    }
}
