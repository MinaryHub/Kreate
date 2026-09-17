//! Markers: world-space lines and constant-size rings, drawn over the picture
//! and depth-tested against it.
//!
//! What lamps and cameras are drawn with. They have no surface, so the solid
//! pass has nothing for them. The interface painted them over the frame for a
//! while, which put a lamp's circle in front of the wall it was behind; these
//! are drawn inside the viewport's own pass, against its depth, so a lamp
//! behind a wall is behind the wall — Blender's behaviour, and the only one a
//! modeller can navigate by.
//!
//! **What the renderer knows and does not.** It knows lines and rings. It does
//! not know what a lamp is: the application turns a light's kind and transform
//! into segments and a ring, the way it turns a light into a `PlacedLight`, and
//! hands them over each frame. A handful of objects, so the upload is a few
//! hundred bytes and nothing is cached.
//!
//! Positions arrive in world space as `f64` and are rebased onto the camera
//! before narrowing, for the reason every world-space thing in this crate is:
//! the subtraction is what keeps a glyph on its object a kilometre from the
//! origin.

use bytemuck::{Pod, Zeroable};
use glam::DVec3;

/// A line segment to draw over the picture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OverlayLine {
    pub a: DVec3,
    pub b: DVec3,
    /// Display-referred RGBA, the interface palette.
    pub color: [f32; 4],
    pub width_px: f32,
    /// Six pixels on, four off, along the screen. For the line a lamp drops
    /// to the ground, which reads as a measurement whatever the distance.
    pub dashed: bool,
}

/// The shape a marker's outline is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MarkerShape {
    Ring,
    Square,
}

/// A constant-size mark at a world position — a lamp's body.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OverlayPoint {
    pub position: DVec3,
    pub color: [f32; 4],
    /// Outer radius on screen, in pixels.
    pub radius_px: f32,
    pub shape: MarkerShape,
    /// A dot at the centre, which is what says "a point light" rather than
    /// "a circle".
    pub dot: bool,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuLine {
    a: [f32; 4],
    b: [f32; 4],
    color: [f32; 4],
    style: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct GpuPoint {
    position: [f32; 4],
    color: [f32; 4],
    style: [f32; 4],
}

impl GpuLine {
    pub(crate) fn pack(line: &OverlayLine, eye: DVec3) -> Self {
        let a = (line.a - eye).as_vec3();
        let b = (line.b - eye).as_vec3();
        Self {
            a: [a.x, a.y, a.z, if line.dashed { 1.0 } else { 0.0 }],
            b: [b.x, b.y, b.z, 0.0],
            color: line.color,
            style: [line.width_px, 0.0, 0.0, 0.0],
        }
    }
}

impl GpuPoint {
    pub(crate) fn pack(point: &OverlayPoint, eye: DVec3) -> Self {
        let p = (point.position - eye).as_vec3();
        Self {
            position: [p.x, p.y, p.z, 0.0],
            color: point.color,
            style: [
                point.radius_px,
                match point.shape {
                    MarkerShape::Ring => 0.0,
                    MarkerShape::Square => 1.0,
                },
                if point.dot { 1.0 } else { 0.0 },
                0.0,
            ],
        }
    }
}

/// A storage buffer that grows to fit and is never shrunk.
///
/// A handful of markers most frames, so the buffer is small; regrown only
/// when a frame has more than any before it, because a reallocation is a new
/// bind group and that is not a per-frame cost worth paying.
pub(crate) struct GrowingStorage {
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) bind: wgpu::BindGroup,
    capacity: u64,
    pub(crate) count: u32,
    label: &'static str,
}

impl GrowingStorage {
    pub(crate) fn new(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        label: &'static str,
        stride: u64,
    ) -> Self {
        let capacity = stride * 16;
        let (buffer, bind) = Self::allocate(device, layout, label, capacity);
        Self {
            buffer,
            bind,
            capacity,
            count: 0,
            label,
        }
    }

    fn allocate(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        label: &str,
        capacity: u64,
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: capacity,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        (buffer, bind)
    }

    /// Uploads `bytes` as `count` items, regrowing first if they do not fit.
    pub(crate) fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        bytes: &[u8],
        count: u32,
    ) {
        let needed = bytes.len() as u64;
        if needed > self.capacity {
            let capacity = needed.next_power_of_two();
            let (buffer, bind) = Self::allocate(device, layout, self.label, capacity);
            self.buffer = buffer;
            self.bind = bind;
            self.capacity = capacity;
        }
        if !bytes.is_empty() {
            queue.write_buffer(&self.buffer, 0, bytes);
        }
        self.count = count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_are_rebased_onto_the_camera_before_narrowing() {
        // The reason the API takes f64 and an eye: a lamp a kilometre out with
        // the camera beside it must pack to a small, exact number.
        let eye = DVec3::new(1000.0, 0.0, 0.0);
        let line = OverlayLine {
            a: DVec3::new(1000.25, 0.0, 0.0),
            b: DVec3::new(1000.5, 0.0, 0.0),
            color: [1.0; 4],
            width_px: 1.0,
            dashed: true,
        };
        let packed = GpuLine::pack(&line, eye);
        assert_eq!(packed.a[0], 0.25);
        assert_eq!(packed.b[0], 0.5);
        assert_eq!(packed.a[3], 1.0, "the dash flag rides in a.w");
    }

    #[test]
    fn a_point_carries_its_shape_and_dot_in_the_style_word() {
        let p = OverlayPoint {
            position: DVec3::ZERO,
            color: [0.0; 4],
            radius_px: 7.0,
            shape: MarkerShape::Square,
            dot: true,
        };
        let packed = GpuPoint::pack(&p, DVec3::ZERO);
        assert_eq!(packed.style, [7.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn the_gpu_layouts_are_sixteen_byte_aligned() {
        assert_eq!(std::mem::size_of::<GpuLine>() % 16, 0);
        assert_eq!(std::mem::size_of::<GpuPoint>() % 16, 0);
    }
}
