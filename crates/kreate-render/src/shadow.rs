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
    if bounds.is_empty() {
        return None;
    }
    let view = look_along(direction);
    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
    for corner in bounds.corners() {
        let p = view.transform_point3(corner);
        min = min.min(p);
        max = max.max(p);
    }
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
        texel_world: half.x.max(half.y) * 2.0 / SHADOW_RESOLUTION as f32,
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
    })
}

/// One face of a point light's cube.
///
/// Six 90° frustums. `face` is 0 to 5 in the order `+X -X +Y -Y +Z -Z`, which
/// is the order a cube texture's layers are.
pub fn point_view(position: glam::Vec3, face: usize, bounds: Aabb) -> Option<ShadowView> {
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
    Some(ShadowView {
        // Exactly 90°, so the six faces meet without a gap or an overlap.
        view_proj: perspective_reversed(std::f32::consts::FRAC_PI_2, near) * view,
        near,
        far: far + MARGIN,
        texel_world: perspective_texel(std::f32::consts::FRAC_PI_2),
    })
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
/// `None` when the light has nothing to shadow: an empty scene, a spot aimed
/// away from everything, a point light with the scene behind the face.
///
/// # A point light gets one face, for now
///
/// Six would be correct and is what `point_view` is shaped for. One is what
/// this renders, so a bare point light shadows correctly in a sixth of the
/// scene and not at all in the rest. That is worse than nothing if it is not
/// said out loud, so it is said here, in the plan, and in a test.
pub fn shadow_view(light: &crate::lights::GpuLight, bounds: Aabb) -> Option<ShadowView> {
    let position = glam::Vec3::from_slice(&light.position[..3]);
    let direction = glam::Vec3::from_slice(&light.direction[..3]);
    if light.falloff == PARALLEL_FALLOFF {
        return sun_view(direction, bounds);
    }
    if light.cone[0] > -1.0 {
        // The stored outer cosine back to a full angle.
        let cone = light.cone[0].clamp(-1.0, 1.0).acos() * 2.0;
        return spot_view(position, direction, cone, bounds);
    }
    point_view(position, 0, bounds)
}

/// One texel's width in metres at unit distance, for a perspective frustum.
///
/// The frustum spans `2 * tan(fov / 2)` across the map at a distance of one,
/// and proportionally more further out — which is what the shader's multiply
/// by clip `w` supplies.
fn perspective_texel(fov: f32) -> f32 {
    2.0 * (fov * 0.5).tan() / SHADOW_RESOLUTION as f32
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
    /// Eye-relative world to the light's clip space, reversed-Z.
    pub view_proj: [[f32; 4]; 4],
    /// x: which layer of the array. y: one texel in metres per unit clip `w`,
    /// see [`ShadowView::texel_world`]. z: the depth bias. w: nonzero if this
    /// light casts.
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
        view_proj: [[0.0; 4]; 4],
        params: [0.0; 4],
    };

    pub fn pack(view: &ShadowView, layer: u32, bias: f32) -> Self {
        Self {
            view_proj: view.view_proj.to_cols_array_2d(),
            params: [layer as f32, view.texel_world, bias, 1.0],
        }
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
    object_stride: u32,
    object_capacity: u32,
}

/// The most shadow-casting lights a frame will draw.
///
/// A cap rather than a growing array, because each one is a full pass over the
/// scene's geometry — the cost is in the draws, not the memory, and a scene
/// that wants a hundred shadowed lights wants the clustered path and a budget
/// rather than a hundred passes.
pub const MAX_SHADOWS: usize = 8;

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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
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
                    visibility: wgpu::ShaderStages::VERTEX,
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
        let object_layout = uniform("shadow object", true, 64);

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shadow.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow"),
            bind_group_layouts: &[Some(&globals_layout), Some(&object_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow"),
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
            // None: a shadow map is a depth buffer. See the shader's header.
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(crate::gpu::depth_write_state()),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

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
        64u32.div_ceil(align) * align
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
                    size: wgpu::BufferSize::new(64),
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

    /// Writes one draw's model matrix.
    pub fn set_object(&self, queue: &wgpu::Queue, index: u32, model: &glam::Mat4) {
        queue.write_buffer(
            &self.objects,
            (self.object_stride * index) as u64,
            bytemuck::cast_slice(&model.to_cols_array()),
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
        draws: impl IntoIterator<Item = (u32, &'a wgpu::Buffer, &'a wgpu::Buffer, u32)>,
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
        for (index, positions, indices, triangles) in draws {
            if triangles == 0 {
                continue;
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

        // A sun's frame is orthographic: parallel lines stay parallel, so two
        // points at different depths on the same axis land at the same place.
        let view = shadow_view(&sun, scene()).expect("a sun");
        let near = project(&view, glam::Vec3::new(1.0, 1.0, 2.0));
        let far = project(&view, glam::Vec3::new(1.0, 1.0, 0.0));
        assert!((near.x - far.x).abs() < 1e-5, "a sun is not orthographic");

        // A spot's is perspective: they do not.
        let view = shadow_view(&spot, scene()).expect("a spot");
        let near = project(&view, glam::Vec3::new(1.0, 1.0, 2.0));
        let far = project(&view, glam::Vec3::new(1.0, 1.0, 0.0));
        assert!((near.x - far.x).abs() > 1e-3, "a spot is not perspective");

        // And a point light gets a frame at all, which is the part that would
        // otherwise be silently missing.
        assert!(shadow_view(&point, scene()).is_some());
    }

    #[test]
    fn a_point_light_shadows_one_sixth_of_the_scene() {
        // Recorded, not hidden. Six faces are what `point_view` is shaped for
        // and one is what the dispatch renders, so a bare point light shadows
        // correctly in front of itself and not at all behind. A test that
        // pinned only the working sixth would let this be forgotten.
        let light = packed(PlacedLight {
            position: glam::DVec3::ZERO,
            ..PlacedLight::default()
        });
        let big = Aabb::new(glam::Vec3::splat(-10.0), glam::Vec3::splat(10.0));
        let view = shadow_view(&light, big).expect("a frame");
        // Face zero looks along +X.
        assert!(inside(project(&view, glam::Vec3::new(5.0, 0.0, 0.0))));
        assert!(!inside(project(&view, glam::Vec3::new(-5.0, 0.0, 0.0))));
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
        )
        .expect("a scene");
        let wide = spot_view(
            glam::Vec3::new(0.0, 0.0, 8.0),
            glam::Vec3::NEG_Z,
            100f32.to_radians(),
            scene(),
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
        let view = sun_view(glam::Vec3::new(0.3, -0.5, -0.81).normalize(), scene())
            .expect("a scene");
        let (mut lo, mut hi) = (glam::Vec2::splat(f32::INFINITY), glam::Vec2::splat(f32::NEG_INFINITY));
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
        assert!(spot_view(glam::Vec3::ZERO, glam::Vec3::NEG_Z, 1.0, Aabb::EMPTY).is_none());
        assert!(point_view(glam::Vec3::ZERO, 0, Aabb::EMPTY).is_none());
    }

    #[test]
    fn a_spot_frames_what_is_in_front_of_it() {
        // On the axis at any distance the scene reaches, and not behind.
        let position = glam::Vec3::new(0.0, 0.0, 8.0);
        let view = spot_view(position, glam::Vec3::NEG_Z, 60f32.to_radians(), scene())
            .expect("a scene");
        for z in [6.0f32, 4.0, 1.0] {
            let p = project(&view, glam::Vec3::new(0.0, 0.0, z));
            assert!(inside(p), "a point on the axis at {z} landed at {p}");
        }
        // Behind the light is outside, and the divide does not make it look
        // inside — which is what a missing `w` check does.
        let behind = view.view_proj * glam::Vec3::new(0.0, 0.0, 10.0).extend(1.0);
        assert!(behind.w <= 0.0, "a point behind the light had w {}", behind.w);
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
        let narrow = spot_view(position, glam::Vec3::NEG_Z, 20f32.to_radians(), scene())
            .expect("a scene");
        let wide = spot_view(position, glam::Vec3::NEG_Z, 100f32.to_radians(), scene())
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
                point_view(position, face, big)
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
}
