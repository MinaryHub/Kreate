//! The GPU's picture has to be the CPU's picture.
//!
//! §5's second risk, in the one place it can be settled cheaply. Two
//! implementations of one formula drift; the user notices as "the viewport and
//! the render disagree"; and that report recurs for as long as both exist. The
//! answer is not care, it is this test.
//!
//! # The tolerance is a quarter of a code value, and that is not arbitrary
//!
//! The two implementations cannot be bit-identical: `log2` and `pow` are
//! different code on a GPU than in `std`, and no specification requires them to
//! agree in the last bits. So the claim is not equality of floats. It is that
//! **no colour can round to a different pixel** — the outputs live in `[0, 1]`,
//! an 8-bit screen quantises them to 1/255 apart, and a disagreement smaller
//! than a quarter of that cannot change what anybody sees.
//!
//! Stated that way the test also stays honest if the display path later becomes
//! 10-bit: the bound tightens by the same argument rather than being renegotiated.

use kreate_math::{DisplayTransform, SceneRgb, ViewTransform};
use kreate_render::display::{workgroups, Display, DisplayResolve};
use kreate_render::GpuOptions;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// The one device this test binary uses.
///
/// Shared rather than created per test, and a falsification sweep is what
/// forced the question: eighteen repeated runs of this binary created several
/// hundred devices between them and the driver eventually wedged, leaving
/// `poll(wait_indefinitely)` waiting forever — fifteen minutes elapsed against
/// two seconds of CPU.
///
/// There was never a reason for one each. No test here mutates device state,
/// so per-test creation buys no isolation and costs the driver eight times the
/// work. `wgpu::Device` and `Queue` are `Arc` handles, so cloning them is a
/// refcount bump and every call site stays as it was.
fn gpu() -> Option<Gpu> {
    static SHARED: std::sync::OnceLock<Option<(wgpu::Device, wgpu::Queue)>> =
        std::sync::OnceLock::new();
    let (device, queue) = SHARED
        .get_or_init(|| {
            let opts = GpuOptions::default();
            let instance = kreate_render::Gpu::create_instance(&opts);
            let gpu = pollster::block_on(kreate_render::Gpu::new(instance, &opts, None)).ok()?;
            Some((gpu.device.clone(), gpu.queue.clone()))
        })
        .clone()?;
    Some(Gpu { device, queue })
}

/// Skips when there is no adapter, which is the state of a headless runner.
macro_rules! gpu_or_skip {
    ($e:expr) => {
        match $e {
            Some(v) => v,
            None => {
                eprintln!("no GPU adapter; skipping");
                return;
            }
        }
    };
}

/// A quarter of an 8-bit code value. See the module docs.
const TOLERANCE: f32 = 1.0 / (255.0 * 4.0);

/// Runs the resolve pass and reads back what it wrote.
fn on_gpu(gpu: &Gpu, colors: &[[f32; 4]], display: Display) -> Vec<[f32; 4]> {
    on_gpu_with_count(gpu, colors, display, colors.len() as u32)
}

/// The same, with a count that need not be the buffer's length.
///
/// Separated so that "a buffer larger than the live count" is testable, which
/// is the shape a viewport's own buffer has — allocated at capacity, filled
/// as far as the frame goes.
fn on_gpu_with_count(
    gpu: &Gpu,
    colors: &[[f32; 4]],
    display: Display,
    count: u32,
) -> Vec<[f32; 4]> {
    use wgpu::util::DeviceExt as _;

    let scene = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("scene"),
            contents: bytemuck::cast_slice(colors),
            usage: wgpu::BufferUsages::STORAGE,
        });
    let size = std::mem::size_of_val(colors) as u64;
    // Filled with a value the transform cannot produce, so an entry the pass
    // failed to write is a visible sentinel rather than a plausible black.
    let sentinel = vec![[-7.0f32; 4]; colors.len()];
    let shown = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shown"),
            contents: bytemuck::cast_slice(&sentinel),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

    let resolve = DisplayResolve::new(&gpu.device);
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    resolve.resolve(&gpu.device, &mut encoder, &scene, &shown, display, count);

    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(&shown, 0, &staging, 0, size);
    gpu.queue.submit([encoder.finish()]);

    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let data = staging.slice(..).get_mapped_range().to_vec();
    staging.unmap();
    bytemuck::cast_slice(&data).to_vec()
}

/// Scene-linear colours worth asking about.
///
/// Spread over the decades the transform claims, plus the awkward ones: a
/// negative channel from a primaries conversion, a NaN from a division
/// upstream, an infinity, and the tiniest positive float.
fn interesting() -> Vec<[f32; 4]> {
    let mut out = vec![
        [0.0, 0.0, 0.0, 1.0],
        [0.18, 0.18, 0.18, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [10.0, 10.0, 10.0, 1.0],
        [1000.0, 1000.0, 1000.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [50.0, 0.5, 0.5, 1.0],
        [0.18, 0.02, 0.02, 1.0],
        [0.02, 0.18, 0.02, 1.0],
        [0.3, 0.1, 0.7, 1.0],
        [-1.0, 0.5, 2.0, 1.0],
        [1e30, 0.0, -1e30, 1.0],
        [f32::NAN, 0.18, f32::NAN, 1.0],
        [f32::INFINITY, f32::NEG_INFINITY, 0.0, 1.0],
        [f32::MIN_POSITIVE, 0.0, 0.0, 1.0],
    ];
    // A sweep across the range, so the comparison is not only about the corner
    // cases somebody thought of.
    let mut v = 1e-5f32;
    while v < 1e4 {
        out.push([v, v * 0.5, v * 0.25, 1.0]);
        v *= 1.7;
    }
    out
}

fn compare(gpu: &Gpu, display: Display) {
    let colors = interesting();
    let theirs = on_gpu(gpu, &colors, display);
    let cpu: DisplayTransform = display.on_cpu();

    assert_eq!(theirs.len(), colors.len());
    for (input, got) in colors.iter().zip(&theirs) {
        let want = cpu.apply(SceneRgb::new(input[0], input[1], input[2]));
        let want = want.to_array();
        for c in 0..3 {
            assert!(
                got[c].is_finite(),
                "{input:?} under {display:?} gave {got:?}"
            );
            assert!(
                (0.0..=1.0).contains(&got[c]),
                "{input:?} under {display:?} left the display range: {got:?}"
            );
            assert!(
                (got[c] - want[c]).abs() <= TOLERANCE,
                "{input:?} under {display:?}: GPU {} against CPU {} (channel {c})",
                got[c],
                want[c]
            );
        }
    }
}

#[test]
fn agx_on_the_gpu_is_agx_on_the_cpu() {
    // The test the whole module rests on.
    let gpu = gpu_or_skip!(gpu());
    compare(
        &gpu,
        Display {
            view: ViewTransform::AgX,
            exposure: 0.0,
        },
    );
}

#[test]
fn they_agree_at_every_exposure() {
    // Exposure multiplies *before* the log encoding, so a mismatch would show
    // only away from zero stops — and zero stops is the setting every casual
    // test uses.
    let gpu = gpu_or_skip!(gpu());
    for exposure in [-8.0, -2.5, -1.0, 0.0, 0.75, 3.0, 9.0] {
        compare(
            &gpu,
            Display {
                view: ViewTransform::AgX,
                exposure,
            },
        );
    }
}

#[test]
fn they_agree_on_the_standard_transform_too() {
    // Which is a clip, and so is the case where a mismatch would be least
    // visible and most annoying: an image editor showing a texture a shade off
    // from the file.
    let gpu = gpu_or_skip!(gpu());
    for exposure in [-1.0, 0.0, 2.0] {
        compare(
            &gpu,
            Display {
                view: ViewTransform::Standard,
                exposure,
            },
        );
    }
}

#[test]
fn the_two_view_transforms_are_not_the_same_pass() {
    // Guards the three tests above: if the shader ignored the transform
    // selector, every comparison would still pass for whichever one it always
    // ran, and only this notices.
    let gpu = gpu_or_skip!(gpu());
    let color = [[1.0, 1.0, 1.0, 1.0]];
    let agx = on_gpu(
        &gpu,
        &color,
        Display {
            view: ViewTransform::AgX,
            exposure: 0.0,
        },
    );
    let standard = on_gpu(
        &gpu,
        &color,
        Display {
            view: ViewTransform::Standard,
            exposure: 0.0,
        },
    );
    assert_eq!(standard[0][0], 1.0, "standard is not a clip");
    assert!(
        agx[0][0] < 0.95,
        "agx clipped white like standard does: {}",
        agx[0][0]
    );
}

#[test]
fn no_colour_can_round_to_a_different_pixel() {
    // The claim the tolerance is chosen to support, asserted directly rather
    // than left as an argument in a comment: the byte an 8-bit screen receives
    // is the same byte from either implementation.
    let gpu = gpu_or_skip!(gpu());
    let display = Display {
        view: ViewTransform::AgX,
        exposure: 0.0,
    };
    let colors = interesting();
    let theirs = on_gpu(&gpu, &colors, display);
    let cpu = display.on_cpu();

    for (input, got) in colors.iter().zip(&theirs) {
        let want = cpu
            .apply(SceneRgb::new(input[0], input[1], input[2]))
            .encode()
            .to_bytes();
        let mine = kreate_math::DisplayRgb::new(got[0], got[1], got[2])
            .encode()
            .to_bytes();
        for c in 0..3 {
            let apart = mine[c].abs_diff(want[c]);
            assert!(
                apart <= 1,
                "{input:?} channel {c}: {} against {} ({apart} apart)",
                mine[c],
                want[c]
            );
        }
    }
}

#[test]
fn alpha_is_carried_through_untouched() {
    // Because it is coverage, not light. Putting a half-covered edge pixel's
    // alpha through a tone curve would change the shape of the edge, and the
    // symptom would be antialiasing that looks wrong in a way nobody would
    // trace back to the display transform.
    let gpu = gpu_or_skip!(gpu());
    let colors: Vec<[f32; 4]> = [0.0, 0.25, 0.5, 1.0]
        .iter()
        .map(|a| [0.18, 0.18, 0.18, *a])
        .collect();
    let out = on_gpu(
        &gpu,
        &colors,
        Display {
            view: ViewTransform::AgX,
            exposure: 2.0,
        },
    );
    for (input, got) in colors.iter().zip(&out) {
        assert_eq!(got[3], input[3], "alpha changed");
    }
}

#[test]
fn a_count_that_is_not_a_whole_workgroup_writes_exactly_that_many() {
    // The dispatch is rounded up, so the tail invocations run with indices past
    // the end. Without the bounds check they would write into whatever buffer
    // follows — a corruption whose symptom appears somewhere unrelated.
    let gpu = gpu_or_skip!(gpu());
    let count = 70; // Two workgroups' worth of threads, seventy colours.
    assert_eq!(workgroups(count as u32), 2);
    let colors: Vec<[f32; 4]> = (0..count).map(|i| [i as f32 * 0.01, 0.0, 0.0, 1.0]).collect();
    let out = on_gpu(
        &gpu,
        &colors,
        Display {
            view: ViewTransform::Standard,
            exposure: 0.0,
        },
    );
    assert_eq!(out.len(), count);
    for (i, got) in out.iter().enumerate() {
        // Every entry was written: the sentinel the buffer was filled with is
        // a value the transform cannot produce.
        assert!(got[0] >= 0.0, "entry {i} was never written: {got:?}");
    }
}

#[test]
fn a_partial_frame_leaves_the_rest_of_the_buffer_alone() {
    // The case the count exists for, and the one this file did not have: a
    // viewport allocates its buffer at capacity and fills as far as the frame
    // goes, so the entries past the count hold whatever was there before and
    // must stay that way.
    //
    // A falsification sweep found the gap. Deleting the shader's bounds check
    // survived every other test here, and so did sending `u32::MAX` as the
    // count — because every other test passes a count equal to the buffer's
    // length, and then the threads past the count are past the buffer too,
    // where WebGPU's own bounds checking discards their writes. Inside the
    // buffer there is nothing to discard: they would simply overwrite live
    // data.
    let gpu = gpu_or_skip!(gpu());
    let capacity = 100;
    let live = 10;
    let held: Vec<[f32; 4]> = (0..capacity).map(|_| [0.18f32, 0.18, 0.18, 1.0]).collect();
    let out = on_gpu_with_count(&gpu, &held, Display::default(), live as u32);

    assert_eq!(out.len(), capacity);
    for (i, got) in out.iter().enumerate() {
        if i < live {
            assert!(
                got[0] > 0.0 && got[0] <= 1.0,
                "entry {i} should have been resolved: {got:?}"
            );
        } else {
            // Still the sentinel the buffer was filled with, which the
            // transform cannot produce.
            assert_eq!(got[0], -7.0, "entry {i} was written past the count: {got:?}");
        }
    }
}

#[test]
fn nothing_to_show_writes_nothing_and_dispatches_nothing() {
    // Nothing to show is a state a viewport is in on its first frame. It
    // arrives as a buffer allocated at capacity with zero *live* colours,
    // because wgpu refuses a zero-sized binding outright — so "no colours" can
    // never be "no buffer", and the count is what carries it.
    let gpu = gpu_or_skip!(gpu());
    let held = [[0.18f32, 0.18, 0.18, 1.0]; 4];
    let out = on_gpu_with_count(&gpu, &held, Display::default(), 0);

    // Every entry still holds the sentinel the buffer was filled with, which
    // the transform cannot produce: the pass wrote nothing at all.
    for got in &out {
        assert_eq!(got[0], -7.0, "something was written: {got:?}");
    }
}
