//! Box, circle and lasso selection: a compute reduction over the id buffer.
//!
//! The naive implementation reads the id texture back and scans it on the CPU. At
//! 1080p that is 16 MB an `Rg32Uint` frame — around ten milliseconds of PCIe plus
//! a two-million-texel scan, on *every* frame of a rubber-band drag. So the scan
//! runs where the data already is, and what comes back is one bit per element:
//! 125 KB for a million vertices, under a millisecond.
//!
//! # Live feedback and the commit are the same dispatch
//!
//! During the drag the bitset is what the highlight is drawn from, so the
//! rubber band updates in the same frame with no upload. On release the same
//! bitset is read back once and turned into the element list an operator takes —
//! which is why the operator's parameter is a resolved *set* rather than a screen
//! rectangle: a set replays exactly, in a script or from a file, and does not
//! depend on where the camera was.
//!
//! # Sized for the mesh, not the viewport
//!
//! The bitset is one bit per element of the mesh being edited, so it grows with
//! the mesh and not with the window. That is the whole reason this is affordable
//! at every viewport size, and it is why the buffer is reallocated on the element
//! count rather than on a resize.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

/// What a drag collects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Collect {
    /// Elements of one object, in the domain the id pass was drawn for.
    ///
    /// Only that object's: every mesh numbers its elements from zero, so folding
    /// a neighbour's in would select unrelated geometry wherever the box crossed
    /// it.
    Elements {
        /// The object, as the id pass was told it.
        object: u32,
        /// How many elements that domain has, bounding the bitset.
        count: u32,
    },
    /// Whole objects, from the channel that already names them.
    ///
    /// The same reduction: an object-mode box has the same shapes, the same
    /// occlusion and the same cost, and a second pipeline for it would be this
    /// one with a different index.
    Objects {
        /// One past the largest object id in the scene.
        count: u32,
    },
}

impl Collect {
    fn count(self) -> u32 {
        match self {
            Collect::Elements { count, .. } | Collect::Objects { count } => count,
        }
    }

    /// The object filter, as the shader wants it: plus one, or zero for "all".
    fn filter(self) -> u32 {
        match self {
            Collect::Elements { object, .. } => object + 1,
            Collect::Objects { .. } => 0,
        }
    }
}

/// Which shape a drag is testing against.
#[derive(Clone, Debug, PartialEq)]
pub enum SelectShape {
    /// Opposite corners, in viewport pixels. Either order.
    Box {
        from: (f32, f32),
        to: (f32, f32),
    },
    Circle {
        centre: (f32, f32),
        radius: f32,
    },
    /// A freehand outline, in viewport pixels.
    ///
    /// Decimated to [`MAX_LASSO`] points by [`SelectShape::lasso`], because the
    /// test is a loop over the outline per texel and a raw drag produces
    /// thousands of points. The drawn line uses the same points, so what is
    /// tested is what the user was shown.
    Lasso(Vec<(f32, f32)>),
}

/// The most points a lasso outline may have.
pub const MAX_LASSO: usize = 256;

impl SelectShape {
    /// A lasso from a raw drag path, decimated to fit.
    ///
    /// Evenly sampled rather than truncated: cutting the tail off would test the
    /// first part of a gesture and silently ignore the rest.
    pub fn lasso(path: &[(f32, f32)]) -> Self {
        if path.len() <= MAX_LASSO {
            return SelectShape::Lasso(path.to_vec());
        }
        let step = path.len() as f32 / MAX_LASSO as f32;
        let points = (0..MAX_LASSO)
            .map(|i| path[((i as f32 * step) as usize).min(path.len() - 1)])
            .collect();
        SelectShape::Lasso(points)
    }

    /// The pixel rectangle that has to be scanned.
    ///
    /// Only the shape's own bounds, not the frame: a small box in the corner of a
    /// 4K viewport dispatches a few hundred threads rather than eight million.
    pub fn bounds(&self) -> ((f32, f32), (f32, f32)) {
        match self {
            SelectShape::Box { from, to } => (
                (from.0.min(to.0), from.1.min(to.1)),
                (from.0.max(to.0), from.1.max(to.1)),
            ),
            SelectShape::Circle { centre, radius } => (
                (centre.0 - radius, centre.1 - radius),
                (centre.0 + radius, centre.1 + radius),
            ),
            SelectShape::Lasso(points) => {
                let mut min = (f32::INFINITY, f32::INFINITY);
                let mut max = (f32::NEG_INFINITY, f32::NEG_INFINITY);
                for (x, y) in points {
                    min = (min.0.min(*x), min.1.min(*y));
                    max = (max.0.max(*x), max.1.max(*y));
                }
                if points.is_empty() {
                    return ((0.0, 0.0), (0.0, 0.0));
                }
                (min, max)
            }
        }
    }

    fn kind(&self) -> u32 {
        match self {
            SelectShape::Box { .. } => 0,
            SelectShape::Circle { .. } => 1,
            SelectShape::Lasso(_) => 2,
        }
    }

    fn shape_params(&self) -> [f32; 4] {
        match self {
            SelectShape::Box { .. } => {
                let (min, max) = self.bounds();
                [min.0, min.1, max.0, max.1]
            }
            SelectShape::Circle { centre, radius } => [centre.0, centre.1, *radius, 0.0],
            SelectShape::Lasso(_) => [0.0; 4],
        }
    }

    fn points(&self) -> &[(f32, f32)] {
        match self {
            SelectShape::Lasso(p) => p,
            _ => &[],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Params {
    region: [u32; 4],
    shape: [f32; 4],
    mode: [u32; 4],
}

const IDLE: u32 = 0;
const IN_FLIGHT: u32 = 1;
const READY: u32 = 2;
const FAILED: u32 = 3;
/// `map_async` has been called and its callback has not run yet. See the same
/// state in `picking`: without it a second frame maps the buffer twice.
const MAPPING: u32 = 4;

/// The bitset, the dispatch, and the readback behind region selection.
pub struct RegionSelect {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    lasso: wgpu::Buffer,
    /// One bit per element, rounded up to whole words.
    bits: wgpu::Buffer,
    readback: wgpu::Buffer,
    /// How many elements the current buffers cover.
    capacity: u32,
    state: Arc<AtomicU32>,
    /// The last set read back, as element indices.
    resolved: Option<Vec<u32>>,
    /// Bumped whenever the bitset is reallocated.
    ///
    /// Exposed so a consumer holding a bind group over the bitset knows when to
    /// rebuild it. An explicit counter rather than comparing buffer identity:
    /// wgpu offers no stable handle to compare, and a raw address can be reused
    /// by the next allocation — which would leave a bind group pointing at a
    /// freed buffer with nothing to notice it.
    generation: u64,
}

/// Words needed for `elements` bits.
fn words_for(elements: u32) -> u32 {
    elements.div_ceil(32).max(1)
}

impl RegionSelect {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("region select"),
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
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("select_region"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/select_region.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("region select layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("region select"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("region select params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lasso = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("region select lasso"),
            size: (MAX_LASSO * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let (bits, readback) = allocate_bits(device, 1);
        Self {
            pipeline,
            layout,
            params,
            lasso,
            bits,
            readback,
            capacity: 32,
            state: Arc::new(AtomicU32::new(IDLE)),
            resolved: None,
            generation: 0,
        }
    }

    /// How many times the bitset has been reallocated.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// How many elements the bitset currently covers.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// The bitset buffer, for a pass that draws the highlight straight from it.
    pub fn bits(&self) -> &wgpu::Buffer {
        &self.bits
    }

    /// Grows the bitset to cover `elements`, if it does not already.
    ///
    /// Sized on the *mesh*, so it grows with the model and not with the window —
    /// which is what makes the whole approach affordable at any viewport size.
    pub fn reserve(&mut self, device: &wgpu::Device, elements: u32) {
        if elements <= self.capacity {
            return;
        }
        // Doubled rather than fitted exactly, so a mesh growing an element at a
        // time does not reallocate on every edit.
        let words = words_for(elements).next_power_of_two();
        let (bits, readback) = allocate_bits(device, words);
        self.bits = bits;
        self.readback = readback;
        self.capacity = words * 32;
        // A pending map refers to a buffer that no longer exists.
        self.state.store(IDLE, Ordering::Release);
        self.resolved = None;
        self.generation += 1;
    }

    /// Records the reduction, and the copy that brings the result back.
    ///
    /// `collect` says whether the drag is gathering elements of one object or
    /// whole objects. The region scanned is the shape's own bounds clipped to the
    /// viewport, so a small box in the corner of a 4K viewport dispatches a few
    /// hundred threads rather than eight million.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ids: &wgpu::TextureView,
        viewport: (u32, u32),
        shape: &SelectShape,
        collect: Collect,
    ) {
        self.reserve(device, collect.count());

        let (min, max) = shape.bounds();
        let x0 = (min.0.floor().max(0.0) as u32).min(viewport.0);
        let y0 = (min.1.floor().max(0.0) as u32).min(viewport.1);
        let x1 = (max.0.ceil().max(0.0) as u32).min(viewport.0);
        let y1 = (max.1.ceil().max(0.0) as u32).min(viewport.1);
        let (w, h) = (x1.saturating_sub(x0), y1.saturating_sub(y0));

        queue.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&Params {
                region: [x0, y0, w, h],
                shape: shape.shape_params(),
                mode: [
                    shape.kind(),
                    shape.points().len() as u32,
                    collect.count(),
                    collect.filter(),
                ],
            }),
        );
        if !shape.points().is_empty() {
            let flat: Vec<f32> = shape.points().iter().flat_map(|(x, y)| [*x, *y]).collect();
            queue.write_buffer(&self.lasso, 0, bytemuck::cast_slice(&flat));
        }

        // Cleared every dispatch. The reduction only ever sets bits — an atomic
        // `or` has no way to clear one — so a stale bit would make every
        // subsequent drag select everything the previous ones did.
        encoder.clear_buffer(&self.bits, 0, None);

        if w > 0 && h > 0 {
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("region select bind"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(ids),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.bits.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.lasso.as_entire_binding(),
                    },
                ],
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("region select"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }

        // Copied whether or not anything was dispatched: an empty region has to
        // produce an empty set rather than leaving the previous answer standing.
        if self.state.load(Ordering::Acquire) == IDLE {
            let words = words_for(self.capacity) as u64;
            encoder.copy_buffer_to_buffer(&self.bits, 0, &self.readback, 0, words * 4);
            self.state.store(IN_FLIGHT, Ordering::Release);
        }
    }

    /// Starts the map for a copy recorded this frame. Call after `submit`.
    pub fn after_submit(&self) {
        if self
            .state
            .compare_exchange(IN_FLIGHT, MAPPING, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        let state = self.state.clone();
        self.readback.map_async(wgpu::MapMode::Read, .., move |r| {
            state.store(if r.is_ok() { READY } else { FAILED }, Ordering::Release);
        });
    }

    /// Collects a completed readback, without blocking.
    pub fn poll(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::Poll);
        match self.state.load(Ordering::Acquire) {
            READY => {
                let elements = decode(&self.readback.get_mapped_range(..));
                self.readback.unmap();
                self.state.store(IDLE, Ordering::Release);
                self.resolved = Some(elements);
            }
            FAILED => {
                // A failed map leaves nothing to report. Dropping the answer is
                // recoverable — the next frame of the drag asks again — and a
                // panic inside frame construction is not.
                self.state.store(IDLE, Ordering::Release);
            }
            _ => {}
        }
    }

    /// The element indices from the last completed reduction.
    ///
    /// Ascending, because they come out of a bitset. That is worth relying on:
    /// an operator recording them produces the same script line whichever way
    /// the user dragged the box.
    pub fn resolved(&self) -> Option<&[u32]> {
        self.resolved.as_deref()
    }

    /// Forgets the last result.
    pub fn clear(&mut self) {
        self.resolved = None;
    }
}

fn allocate_bits(device: &wgpu::Device, words: u32) -> (wgpu::Buffer, wgpu::Buffer) {
    let size = (words as u64 * 4).max(4);
    let bits = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("selection bits"),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("selection bits readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    (bits, readback)
}

/// Expands a bitset into ascending element indices.
fn decode(bytes: &[u8]) -> Vec<u32> {
    let mut out = Vec::new();
    for (word_index, word) in bytes.as_chunks::<4>().0.iter().enumerate() {
        let mut word = u32::from_le_bytes(*word);
        while word != 0 {
            let bit = word.trailing_zeros();
            out.push(word_index as u32 * 32 + bit);
            word &= word - 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_box_scans_only_its_own_bounds() {
        // The reason the dispatch is sized on the shape: a small box in the
        // corner of a 4K viewport should cost a few hundred threads, not eight
        // million.
        let shape = SelectShape::Box {
            from: (100.0, 200.0),
            to: (140.0, 260.0),
        };
        let (min, max) = shape.bounds();
        assert_eq!(min, (100.0, 200.0));
        assert_eq!(max, (140.0, 260.0));
    }

    #[test]
    fn a_box_dragged_backwards_is_the_same_box() {
        // Dragging up and to the left is as common as down and to the right, and
        // an unnormalised rectangle selects nothing at all.
        let forward = SelectShape::Box {
            from: (10.0, 10.0),
            to: (50.0, 40.0),
        };
        let backward = SelectShape::Box {
            from: (50.0, 40.0),
            to: (10.0, 10.0),
        };
        assert_eq!(forward.bounds(), backward.bounds());
        assert_eq!(forward.shape_params(), backward.shape_params());
    }

    #[test]
    fn a_circle_bounds_its_radius() {
        let shape = SelectShape::Circle {
            centre: (100.0, 100.0),
            radius: 12.0,
        };
        assert_eq!(shape.bounds(), ((88.0, 88.0), (112.0, 112.0)));
    }

    #[test]
    fn a_short_lasso_is_kept_whole() {
        let path: Vec<(f32, f32)> = (0..10).map(|i| (i as f32, 0.0)).collect();
        match SelectShape::lasso(&path) {
            SelectShape::Lasso(points) => assert_eq!(points.len(), 10),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn a_long_lasso_is_sampled_rather_than_truncated() {
        // Cutting the tail off would test the first part of the gesture and
        // silently ignore the rest — a lasso that selects half of what it
        // enclosed.
        let path: Vec<(f32, f32)> = (0..5000).map(|i| (i as f32, 0.0)).collect();
        match SelectShape::lasso(&path) {
            SelectShape::Lasso(points) => {
                assert_eq!(points.len(), MAX_LASSO);
                assert_eq!(points[0].0, 0.0, "it starts where the drag started");
                assert!(
                    points[MAX_LASSO - 1].0 > 4900.0,
                    "and ends where the drag ended, not a fiftieth of the way in: {}",
                    points[MAX_LASSO - 1].0
                );
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn an_empty_lasso_bounds_to_nothing() {
        // A stray click can register as a zero-point gesture. An infinite
        // bounding box would dispatch over the whole viewport.
        let shape = SelectShape::Lasso(Vec::new());
        assert_eq!(shape.bounds(), ((0.0, 0.0), (0.0, 0.0)));
    }

    #[test]
    fn a_bitset_decodes_to_ascending_indices() {
        // Bit 0 of word 0, bit 5 of word 0, bit 1 of word 2 — element 0, 5, 65.
        let mut bytes = vec![0u8; 12];
        bytes[0..4].copy_from_slice(&0b100001u32.to_le_bytes());
        bytes[8..12].copy_from_slice(&0b10u32.to_le_bytes());
        assert_eq!(decode(&bytes), vec![0, 5, 65]);
    }

    #[test]
    fn an_empty_bitset_decodes_to_nothing() {
        assert!(decode(&[0u8; 64]).is_empty());
    }

    #[test]
    fn every_bit_of_a_full_word_decodes() {
        let bytes = u32::MAX.to_le_bytes();
        assert_eq!(decode(&bytes).len(), 32);
        assert_eq!(decode(&bytes)[31], 31);
    }

    #[test]
    fn the_bitset_is_sized_in_whole_words() {
        assert_eq!(words_for(1), 1);
        assert_eq!(words_for(32), 1);
        assert_eq!(words_for(33), 2);
        assert_eq!(words_for(0), 1, "a zero-element mesh still needs a buffer");
    }
}
