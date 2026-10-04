//! The id buffer and the readback ring behind it.
//!
//! This is the GPU half of picking. The CPU BVH answers a different question —
//! "where in continuous space does this ray meet the surface", with barycentric
//! coordinates and full precision, which is what snapping, knife projection and
//! measurement need. This answers "what did the user click, as they see it",
//! with occlusion for free and a screen-space tolerance. Both are needed; trying
//! to make one serve both is the mistake.
//!
//! # Read every frame, not on the click
//!
//! The obvious flow — click, render ids, `map_async`, poll, interpret — costs two
//! to four frames *after* the click. Thirty to sixty milliseconds of dead time
//! between pressing the button and the selection changing, which reads as the
//! application being sluggish however high the frame rate is.
//!
//! So the id pass runs every frame in edit mode, and a small region around the
//! cursor is copied into a ring of readback buffers each frame. A poll one or two
//! frames later always has a recent map of what is under the cursor:
//!
//! - **hover highlighting** uses it directly, and one frame of lag is not
//!   perceptible;
//! - **a click** consumes data already in flight, so it costs *no* extra
//!   latency and the selection changes on the same frame as the click.
//!
//! # A region, not a pixel
//!
//! Reading a 33×33 window rather than one pixel is what lets a near miss resolve
//! to the nearest element instead of to nothing, and it is what makes the pick
//! feel forgiving. It costs 17 KB a frame.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// `.r` is the object id plus one, `.g` the element id.
///
/// Sixty-four bits: no packing, and no id-space exhaustion on a scene with a
/// hundred million elements. Eight bytes a pixel is 16 MB at 1080p, which is
/// affordable; the alternative — packing an object and element into 32 bits —
/// caps one of them at a number a real scene reaches.
pub const ID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;

/// Width and height of the region read back each frame.
///
/// Odd, so that it has an exact centre pixel.
pub const REGION: u32 = 33;

/// Bytes per row in a readback slot.
///
/// `copy_texture_to_buffer` requires a multiple of 256, and 33 pixels is 264
/// bytes, so each row is padded. The padding is why a reader must index by
/// [`ROW_BYTES`] rather than by the region width.
pub const ROW_BYTES: u32 = 512;

/// How many buffers the ring holds.
///
/// Three: one being written by the GPU, one in flight, one being read. Two
/// stalls whenever a map has not completed; more than three only adds staleness.
const SLOTS: usize = 3;

const SLOT_BYTES: u64 = (ROW_BYTES * REGION) as u64;

// Checked at compile time rather than in a test. These are relationships between
// constants, and a build that cannot satisfy them should not produce a binary
// whose picking is subtly offset — the failure mode is a region decoded through
// the wrong stride, which reads as the cursor being a few pixels off rather than
// as anything broken.
const _: () = assert!(
    ROW_BYTES.is_multiple_of(256),
    "copy_texture_to_buffer requires a row stride that is a multiple of 256"
);
const _: () = assert!(
    ROW_BYTES >= REGION * 8,
    "the padded row has to hold a whole region row of Rg32Uint pixels"
);
const _: () = assert!(
    !REGION.is_multiple_of(2),
    "an even region has no centre pixel to rank distances from"
);

/// Which element domain an id pass was drawn for.
///
/// One domain per pass. Drawing all three at once would need a channel saying
/// which domain a pixel belongs to, and a modeller in vertex mode is not picking
/// faces — so the domain is a property of the pass, recorded here, rather than
/// something the reader has to work out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PickDomain {
    /// Faces, but reported as the object that owns them.
    Object,
    Vert,
    Edge,
    Face,
}

impl PickDomain {
    /// Pick radius in pixels, at 1× scale.
    ///
    /// Vertices get the widest tolerance and faces none at all: a face is a
    /// large target already, while a vertex is a few pixels of dot that the user
    /// is expected to hit exactly. Each is a little larger than the thing is
    /// drawn, which is what turns a near miss into a hit.
    pub fn radius_px(self) -> f32 {
        match self {
            PickDomain::Object | PickDomain::Face => 0.0,
            PickDomain::Vert => 8.0,
            PickDomain::Edge => 5.0,
        }
    }

    /// Relative depth offset toward the viewer.
    ///
    /// The same values the visible overlays use, and that is the point: the z
    /// order in the id buffer has to be the z order on screen, or the thing that
    /// looks clickable is not the thing that gets clicked.
    ///
    /// **Never zero.** The id pass tests against the depth the solid pass wrote
    /// and does not write its own, so a fragment at exactly that depth fails a
    /// `Greater` comparison and the pass produces nothing at all. Object mode
    /// draws the same faces as face mode and takes the same nudge.
    pub fn depth_offset(self) -> f32 {
        match self {
            PickDomain::Object | PickDomain::Face => 1e-4,
            PickDomain::Edge => 3e-4,
            PickDomain::Vert => 6e-4,
        }
    }
}

/// What the id buffer says is at one pixel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PickHit {
    /// Index of the object, already decoded from the stored plus-one.
    pub object: u32,
    pub element: u32,
    pub domain: PickDomain,
    /// Distance in pixels from where the caller asked.
    pub distance_px: u32,
}

/// A decoded region of the id buffer.
///
/// Owned CPU data rather than a mapped buffer view: the copy is 17 KB, and it
/// frees the slot immediately instead of holding a map open across frames while
/// something reads from it.
#[derive(Clone, Debug)]
pub struct PickRegion {
    /// Where the region's centre pixel sits in the viewport.
    pub centre: (i32, i32),
    pub domain: PickDomain,
    /// `REGION * REGION` pairs, row-major from the region's top-left.
    ids: Vec<[u32; 2]>,
}

impl PickRegion {
    /// The nearest hit to a viewport pixel, if the region covers one.
    ///
    /// Searched outward from `at` rather than from the captured centre: the
    /// region may be a frame or two old and the cursor will have moved a little,
    /// and ranking from where the cursor *is* is what makes a click land on what
    /// the user is looking at. A cursor that has left the captured region
    /// entirely returns nothing rather than a stale answer from its edge.
    pub fn nearest(&self, at: (i32, i32)) -> Option<PickHit> {
        let half = REGION as i32 / 2;
        let local = (at.0 - self.centre.0 + half, at.1 - self.centre.1 + half);
        if !(0..REGION as i32).contains(&local.0) || !(0..REGION as i32).contains(&local.1) {
            return None;
        }

        let mut best: Option<PickHit> = None;
        for y in 0..REGION as i32 {
            for x in 0..REGION as i32 {
                let [object, element] = self.ids[(y * REGION as i32 + x) as usize];
                // Zero is "nothing here", which is why the object id is stored
                // plus one.
                if object == 0 {
                    continue;
                }
                let dx = x - local.0;
                let dy = y - local.1;
                let distance = (dx * dx + dy * dy) as u32;
                if best.is_some_and(|b| b.distance_px <= distance) {
                    continue;
                }
                best = Some(PickHit {
                    object: object - 1,
                    element,
                    domain: self.domain,
                    distance_px: distance,
                });
            }
        }
        // Reported as pixels rather than squared pixels; squared distance is
        // only used to rank, because a square root per pixel of a 1089-pixel
        // region buys nothing.
        best.map(|mut hit| {
            hit.distance_px = (hit.distance_px as f32).sqrt() as u32;
            hit
        })
    }

    /// The id exactly at a viewport pixel, with no tolerance.
    ///
    /// For a caller that wants "is this element under the cursor" rather than
    /// "what would a click select" — a drag that must not snap to a neighbour.
    pub fn at(&self, at: (i32, i32)) -> Option<PickHit> {
        let hit = self.nearest(at)?;
        (hit.distance_px == 0).then_some(hit)
    }
}

/// One readback buffer and what it is holding.
struct Slot {
    buffer: wgpu::Buffer,
    /// Written by the map callback, read on the next frame. `0` idle, `1` in
    /// flight, `2` ready, `3` failed, `4` mapping.
    state: Arc<AtomicU32>,
    centre: (i32, i32),
    domain: PickDomain,
}

const IDLE: u32 = 0;
const IN_FLIGHT: u32 = 1;
const READY: u32 = 2;
const FAILED: u32 = 3;
/// `map_async` has been called and its callback has not run yet. Distinct from
/// `IN_FLIGHT` so a frame submitted before the callback fires does not map the
/// buffer a second time, which wgpu treats as a fatal error.
const MAPPING: u32 = 4;

/// The id render target and the ring of readback buffers behind it.
pub struct PickTarget {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
    slots: Vec<Slot>,
    /// Which slot the next copy goes into.
    next: usize,
    /// The most recent decoded region, kept until a newer one arrives.
    ///
    /// Held rather than consumed so that a frame with no completed map still has
    /// an answer for hover. It is at most a frame or two old.
    latest: Option<PickRegion>,
    /// How many copies have been requested and how many came back, so that "the
    /// readback is quietly never completing" is answerable rather than
    /// presenting as picking that does not respond.
    requested: u64,
    received: u64,
}

impl PickTarget {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let (texture, view) = create_target(device, width, height);
        let slots = (0..SLOTS)
            .map(|i| Slot {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("pick readback {i}")),
                    size: SLOT_BYTES,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU32::new(IDLE)),
                centre: (0, 0),
                domain: PickDomain::Object,
            })
            .collect();
        Self {
            texture,
            view,
            size: (width.max(1), height.max(1)),
            slots,
            next: 0,
            latest: None,
            requested: 0,
            received: 0,
        }
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Reallocates the target if the viewport grew.
    ///
    /// The readback slots are a fixed size and never need reallocating, which is
    /// most of the reason for reading a small region rather than the frame.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let wanted = (width.max(1), height.max(1));
        if wanted == self.size {
            return;
        }
        let (texture, view) = create_target(device, wanted.0, wanted.1);
        self.texture = texture;
        self.view = view;
        self.size = wanted;
        // Whatever was captured describes a differently sized viewport, so its
        // pixel coordinates no longer mean anything.
        self.latest = None;
    }

    /// Copies the region around `cursor` into the next slot.
    ///
    /// The region is clamped to the target, so the centre recorded is not always
    /// the cursor: near an edge of the viewport the window slides inward rather
    /// than reading outside the texture. A reader asks by viewport pixel and does
    /// not have to know.
    pub fn capture(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        cursor: (i32, i32),
        domain: PickDomain,
    ) {
        let half = REGION as i32 / 2;
        let max_x = self.size.0.saturating_sub(REGION) as i32;
        let max_y = self.size.1.saturating_sub(REGION) as i32;
        let origin = (
            (cursor.0 - half).clamp(0, max_x.max(0)),
            (cursor.1 - half).clamp(0, max_y.max(0)),
        );

        let slot = &mut self.slots[self.next];
        // A slot still in flight cannot be written. Dropping this frame's capture
        // is correct: the next frame will try again, and the alternative is
        // waiting on the GPU inside frame construction.
        if slot.state.load(Ordering::Acquire) != IDLE {
            return;
        }
        slot.centre = (origin.0 + half, origin.1 + half);
        slot.domain = domain;

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: origin.0 as u32,
                    y: origin.1 as u32,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &slot.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(ROW_BYTES),
                    rows_per_image: Some(REGION),
                },
            },
            wgpu::Extent3d {
                width: REGION.min(self.size.0),
                height: REGION.min(self.size.1),
                depth_or_array_layers: 1,
            },
        );
        slot.state.store(IN_FLIGHT, Ordering::Release);
        self.requested += 1;
        self.next = (self.next + 1) % SLOTS;
    }

    /// Starts the map for anything captured this frame.
    ///
    /// Called after `submit`: mapping a buffer whose copy has not been submitted
    /// waits for a command that will never arrive.
    pub fn after_submit(&self) {
        for slot in &self.slots {
            if slot
                .state
                .compare_exchange(IN_FLIGHT, MAPPING, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                continue;
            }
            let state = slot.state.clone();
            slot.buffer.map_async(wgpu::MapMode::Read, .., move |r| {
                state.store(if r.is_ok() { READY } else { FAILED }, Ordering::Release);
            });
        }
    }

    /// Collects whatever has completed, without blocking.
    ///
    /// Non-blocking on purpose. Waiting here is exactly the stall this whole ring
    /// exists to avoid, and there is always the previous region to answer from.
    pub fn poll(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::Poll);
        let mut fresh = None;
        for slot in &self.slots {
            match slot.state.load(Ordering::Acquire) {
                READY => {
                    let ids = decode(&slot.buffer.get_mapped_range(..));
                    slot.buffer.unmap();
                    slot.state.store(IDLE, Ordering::Release);
                    self.received += 1;
                    fresh = Some(PickRegion {
                        centre: slot.centre,
                        domain: slot.domain,
                        ids,
                    });
                }
                FAILED => {
                    // A failed map leaves the slot unusable until it is released.
                    // Dropping the frame's answer is the right outcome: picking
                    // going quiet is recoverable, and a panic in frame
                    // construction is not.
                    slot.state.store(IDLE, Ordering::Release);
                }
                _ => {}
            }
        }
        // Newest wins, and only if something completed: a frame with no
        // completed map keeps answering from the previous region rather than
        // going blank, which is what makes hover feel continuous.
        if fresh.is_some() {
            self.latest = fresh;
        }
    }

    /// The most recent region read back, if any.
    pub fn latest(&self) -> Option<&PickRegion> {
        self.latest.as_ref()
    }

    /// Throws away what was read back.
    ///
    /// For when the scene has changed underneath it — leaving edit mode, or an
    /// operator running — and the ids no longer mean what they meant.
    pub fn invalidate(&mut self) {
        self.latest = None;
    }

    /// Copies requested and regions received.
    pub fn stats(&self) -> (u64, u64) {
        (self.requested, self.received)
    }
}

fn create_target(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ids"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: ID_FORMAT,
        // TEXTURE_BINDING as well: box and lasso selection read this texture in
        // a compute pass rather than copying it back, which is the difference
        // between a rubber-band drag that keeps up and one that stalls on ten
        // milliseconds of PCIe every frame.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Unpacks a padded readback into `REGION * REGION` id pairs.
fn decode(bytes: &[u8]) -> Vec<[u32; 2]> {
    let mut ids = Vec::with_capacity((REGION * REGION) as usize);
    for y in 0..REGION as usize {
        let row = y * ROW_BYTES as usize;
        for x in 0..REGION as usize {
            let at = row + x * 8;
            // Indexed through the padded row stride, not the region width: the
            // 256-byte alignment leaves 248 bytes of slack on every row, and
            // reading straight through them shifts the whole image diagonally.
            let object = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap_or([0; 4]));
            let element = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap_or([0; 4]));
            ids.push([object, element]);
        }
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A region with one element at a given offset from the centre.
    fn region_with(offsets: &[((i32, i32), u32)]) -> PickRegion {
        let half = REGION as i32 / 2;
        let mut ids = vec![[0u32; 2]; (REGION * REGION) as usize];
        for ((dx, dy), element) in offsets {
            let x = half + dx;
            let y = half + dy;
            ids[(y * REGION as i32 + x) as usize] = [1, *element];
        }
        PickRegion {
            centre: (100, 100),
            domain: PickDomain::Vert,
            ids,
        }
    }

    #[test]
    fn an_element_under_the_cursor_is_found() {
        let region = region_with(&[((0, 0), 7)]);
        let hit = region.nearest((100, 100)).expect("something is there");
        assert_eq!(hit.element, 7);
        assert_eq!(hit.object, 0, "the stored plus-one is decoded");
        assert_eq!(hit.distance_px, 0);
    }

    #[test]
    fn a_near_miss_still_hits() {
        // The whole reason for reading a region rather than a pixel: a click a
        // few pixels off the vertex has to select it.
        let region = region_with(&[((4, 3), 12)]);
        let hit = region.nearest((100, 100)).expect("within the region");
        assert_eq!(hit.element, 12);
        assert_eq!(hit.distance_px, 5, "3-4-5");
    }

    #[test]
    fn the_nearest_of_several_wins() {
        let region = region_with(&[((10, 0), 1), ((2, 0), 2), ((-6, 0), 3)]);
        let hit = region.nearest((100, 100)).unwrap();
        assert_eq!(hit.element, 2);
    }

    #[test]
    fn ranking_follows_the_cursor_rather_than_the_captured_centre() {
        // The region is a frame or two old, so the cursor has moved. Ranking from
        // where it *was* would pick the wrong element during any drag.
        let region = region_with(&[((-8, 0), 10), ((8, 0), 20)]);
        assert_eq!(
            region.nearest((100, 100)).unwrap().element,
            10,
            "a tie goes to the first found"
        );
        assert_eq!(region.nearest((106, 100)).unwrap().element, 20);
        assert_eq!(region.nearest((94, 100)).unwrap().element, 10);
    }

    #[test]
    fn an_empty_region_finds_nothing() {
        assert_eq!(region_with(&[]).nearest((100, 100)), None);
    }

    #[test]
    fn a_cursor_outside_the_captured_region_finds_nothing() {
        // Better than answering from the edge of a stale window: a wrong hover
        // that follows the cursor at a distance is more confusing than none.
        let region = region_with(&[((0, 0), 1)]);
        assert_eq!(region.nearest((200, 100)), None);
        assert_eq!(region.nearest((100, 60)), None);
    }

    #[test]
    fn the_exact_query_ignores_a_near_miss() {
        let region = region_with(&[((3, 0), 5)]);
        assert_eq!(region.at((100, 100)), None);
        assert_eq!(region.at((103, 100)).unwrap().element, 5);
    }

    #[test]
    fn a_decoded_row_is_read_through_the_padded_stride() {
        // 33 pixels is 264 bytes and the copy needs a multiple of 256, so every
        // row carries slack. Reading straight through it shifts the whole image
        // diagonally — which looks like picking being subtly offset, not like a
        // decoding bug.
        let mut bytes = vec![0u8; (ROW_BYTES * REGION) as usize];
        for y in 0..REGION as usize {
            let at = y * ROW_BYTES as usize;
            bytes[at..at + 4].copy_from_slice(&1u32.to_le_bytes());
            bytes[at + 4..at + 8].copy_from_slice(&(y as u32).to_le_bytes());
        }
        let ids = decode(&bytes);
        assert_eq!(ids.len(), (REGION * REGION) as usize);
        for y in 0..REGION as usize {
            assert_eq!(
                ids[y * REGION as usize],
                [1, y as u32],
                "row {y} decoded from the wrong offset"
            );
        }
    }

    #[test]
    fn the_pick_radius_widens_as_the_target_gets_smaller() {
        // A face is a big target and needs no tolerance; a vertex is a few pixels
        // of dot. Getting this backwards makes vertex selection feel broken.
        assert!(PickDomain::Vert.radius_px() > PickDomain::Edge.radius_px());
        assert!(PickDomain::Edge.radius_px() > PickDomain::Face.radius_px());
    }

    #[test]
    fn the_id_layering_matches_the_overlay_layering() {
        // The z order in the id buffer has to be the z order on screen, or the
        // thing that looks clickable is not the thing that gets clicked.
        assert!(PickDomain::Vert.depth_offset() > PickDomain::Edge.depth_offset());
        assert!(PickDomain::Edge.depth_offset() > PickDomain::Face.depth_offset());
    }

    #[test]
    fn no_domain_has_a_zero_depth_offset() {
        // Zero is not "the default layer", it is *nothing renders*: the id pass
        // tests against the solid pass's depth without writing its own, so a
        // fragment at exactly that depth fails `Greater` and the whole pass comes
        // back empty. It looks like picking being switched off.
        for domain in [
            PickDomain::Object,
            PickDomain::Face,
            PickDomain::Edge,
            PickDomain::Vert,
        ] {
            assert!(
                domain.depth_offset() > 0.0,
                "{domain:?} would render nothing into the id buffer"
            );
        }
    }
}
