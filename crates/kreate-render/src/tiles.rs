//! Which lights reach which square of the screen.
//!
//! The shading pass walks every light for every fragment, which is a quarter of
//! a billion light evaluations at two hundred and fifty-six lamps and a million
//! pixels — measured at 11.9 ms a frame in `light_cost`, against the 1.5 ms
//! §2.2 allows for the whole of the shading overhead.
//!
//! This walks every light for every *tile* instead, once, and hands the shading
//! pass the short list. Eight thousand tiles rather than a million fragments is
//! the whole of the saving, and it is worth having only because most lights do
//! not reach most of the screen — which is a fact about the light's range, and
//! why [`crate::lights::PlacedLight::range`] had to exist first.

/// The side of a tile, in pixels.
///
/// Sixteen, as §2.2 asks. Smaller tiles cull more tightly and cost more to
/// build: the cull is per tile and the win is per fragment, so the useful size
/// is where the two meet, and sixteen is where everybody who has measured it
/// has landed.
pub const TILE_SIZE: u32 = 16;

/// The most lights one tile will shade with.
///
/// A cap rather than a growing list, because a tile's list is written by one
/// thread into a slot it owns: growing would need an allocator and an atomic,
/// and would order the list differently every frame — a picture that shimmers
/// while nothing moves.
///
/// Lights past the cap are dropped from that tile, which is a **wrong picture**
/// and not merely a slow one — and it is silent, which is worse. That is why
/// this is as large as §2.2's own target rather than as small as memory would
/// like: at sixty-four, the two hundred and fifty-six lamps the plan budgets
/// for lost three quarters of their light the moment they overlapped, and the
/// first measurement of the pass read as a speedup because a quarter of the
/// work was missing.
///
/// The cost is memory: a tile's list is this many indices plus a count, so at
/// 1080p it is eight megabytes and at 4K thirty-three. The cost of it being
/// smaller is a scene that quietly loses lamps.
pub const MAX_LIGHTS_PER_TILE: u32 = 256;

/// Where this library expects its bindings.
pub const TILES_GROUP: u32 = 2;
pub const TILES_BINDING: u32 = 5;

/// What the cull pass needs to know about the camera.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TileParams {
    view: [[f32; 4]; 4],
    counts: [u32; 4],
    lens: [f32; 4],
    ortho: [f32; 4],
}

/// The per-tile light lists, and the pass that fills them.
pub struct Tiles {
    params: wgpu::Buffer,
    lists: wgpu::Buffer,
    layout: wgpu::BindGroupLayout,
    bind: Option<wgpu::BindGroup>,
    pipeline: wgpu::ComputePipeline,
    /// How many tiles the current buffer holds.
    capacity: u32,
    across: u32,
    down: u32,
}

impl Tiles {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile cull"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tiles"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/tiles.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tile cull"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tile cull"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("cull"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile params"),
            size: std::mem::size_of::<TileParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let capacity = 1;
        let lists = Self::make_lists(device, capacity);
        Self {
            params,
            lists,
            layout,
            bind: None,
            pipeline,
            capacity,
            across: 1,
            down: 1,
        }
    }

    fn make_lists(device: &wgpu::Device, tiles: u32) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile lights"),
            size: (tiles as u64) * (MAX_LIGHTS_PER_TILE as u64 + 1) * 4,
            // `COPY_SRC` so a test can read a tile's list back. What is in a
            // list is invisible in a render — a light wrongly *in* one costs
            // time and shows nothing — so the only way to check it is to look.
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// How many tiles a target of this size needs.
    pub fn grid(width: u32, height: u32) -> (u32, u32) {
        (
            width.div_ceil(TILE_SIZE).max(1),
            height.div_ceil(TILE_SIZE).max(1),
        )
    }

    /// The lists, for the shading pass to read.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.lists
    }

    pub fn across(&self) -> u32 {
        self.across
    }

    /// Makes room for a target of this size, and says whether the buffer moved.
    ///
    /// A caller holding a bind group over the lists has to rebuild it when this
    /// answers true — the same contract [`crate::lights::Lights`] has, for the
    /// same reason: a bind group holds the old buffer and nothing tells it.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) -> bool {
        let (across, down) = Self::grid(width, height);
        self.across = across;
        self.down = down;
        let wanted = across * down;
        if wanted <= self.capacity {
            return false;
        }
        // Rounded up, so that dragging a window edge does not reallocate on
        // every pixel.
        let capacity = wanted.next_power_of_two();
        self.lists = Self::make_lists(device, capacity);
        self.capacity = capacity;
        self.bind = None;
        true
    }

    /// Works out which lights reach which tile.
    ///
    /// `lights` is the packed light buffer, and `count` how many of it is live
    /// — the buffer does not shrink, so its length is not the answer.
    #[allow(clippy::too_many_arguments)]
    pub fn cull(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        lights: &wgpu::Buffer,
        count: u32,
        camera: &crate::Camera,
        aspect: f32,
    ) {
        // The rotation only. The lights are already eye-relative, so the eye is
        // the origin and a translation would move them twice.
        let view = glam::Mat4::from_quat(camera.rotation.as_quat().inverse());
        let orthographic = matches!(camera.projection, crate::Projection::Orthographic);
        let half_height = (camera.ortho_height() * 0.5) as f32;
        let params = TileParams {
            view: view.to_cols_array_2d(),
            counts: [self.across, self.down, count, MAX_LIGHTS_PER_TILE],
            lens: [
                1.0 / (camera.fov_y as f32 * 0.5).tan(),
                aspect.max(1e-4),
                camera.near() as f32,
                if orthographic { 1.0 } else { 0.0 },
            ],
            ortho: [half_height * aspect, half_height, 0.0, 0.0],
        };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));

        // Rebuilt when either buffer moved. The light buffer grows by being
        // replaced, so its identity is not stable across frames either.
        let bind = self.bind.get_or_insert_with(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tile cull"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: lights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.lists.as_entire_binding(),
                    },
                ],
            })
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tile cull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &*bind, &[]);
        pass.dispatch_workgroups(self.across.div_ceil(8), self.down.div_ceil(8), 1);
    }

    /// Forgets the bind group, so the next cull rebuilds it.
    ///
    /// Called when the light buffer is replaced. The lists' own growth clears
    /// this itself; a light buffer that moved is invisible from here.
    pub fn forget_bindings(&mut self) {
        self.bind = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_grid_covers_every_pixel() {
        // A tile that is not a whole tile still has to exist, or the pixels in
        // it read a tile that was never written.
        for (w, h) in [(1u32, 1u32), (16, 16), (17, 16), (1920, 1080), (1023, 1)] {
            let (across, down) = Tiles::grid(w, h);
            assert!(across * TILE_SIZE >= w, "{w}x{h} left a column uncovered");
            assert!(down * TILE_SIZE >= h, "{w}x{h} left a row uncovered");
            assert!(across >= 1 && down >= 1);
        }
    }

    #[test]
    fn a_zero_sized_target_still_has_a_tile() {
        // wgpu refuses a zero-sized binding, and a viewport can be laid out at
        // no size for a frame while a panel is being dragged.
        let (across, down) = Tiles::grid(0, 0);
        assert_eq!((across, down), (1, 1));
    }
}
