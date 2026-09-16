//! What the prefilter actually put in the cube.
//!
//! Read back and inspected rather than looked at. A prefiltered environment
//! shows up as the brightness of a reflection, and every mistake it can make —
//! a face mirrored, a level normalised wrongly, a lobe sampled about the wrong
//! axis — produces a picture that still looks like a reflection of a sky.
//!
//! The claim that matters most is an identity: **a uniform environment must
//! come back as the constant at every roughness.** That is what binds the image
//! path to the flat one, and if it drifts, picking a plain grey sky lights a
//! scene differently from setting the same grey as a flat colour — a difference
//! nobody would attribute to the prefilter.

use kreate_render::environment::{self, Environment, FACE, MIPS};
use kreate_render::GpuOptions;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// The one device this binary uses. See `deform.rs` for why it is shared.
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

/// An equirectangular `Rgba16Float` image, coloured by a function of its
/// texture coordinates.
fn image(width: u32, height: u32, mut at: impl FnMut(f32, f32) -> [f32; 3]) -> Vec<u8> {
    let mut out = Vec::with_capacity((width * height) as usize * 8);
    for y in 0..height {
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let v = (y as f32 + 0.5) / height as f32;
            let rgb = at(u, v);
            for c in rgb {
                out.extend_from_slice(&half::f16::from_f32(c).to_le_bytes());
            }
            out.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        }
    }
    out
}

/// One mip of one face, as floats.
fn face(gpu: &Gpu, environment: &Environment, level: u32, layer: u32) -> Vec<[f32; 4]> {
    let size = (FACE >> level).max(1);
    // 256-byte rows, which is what a texture-to-buffer copy requires. Padded
    // and then unpadded below rather than assumed away: at the smallest mip a
    // face is 4 texels, which is 32 bytes, and the padding is most of the row.
    let unpadded = size * 8;
    let padded = unpadded.div_ceil(256) * 256;
    let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("face readback"),
        size: (padded * size) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: environment.texture(),
            mip_level: level,
            origin: wgpu::Origin3d {
                x: 0,
                y: 0,
                z: layer,
            },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(size),
            },
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit([encoder.finish()]);

    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    let data = slice.get_mapped_range();
    let mut out = Vec::with_capacity((size * size) as usize);
    for row in 0..size {
        let start = (row * padded) as usize;
        for column in 0..size {
            let at = start + column as usize * 8;
            let channel = |i: usize| {
                half::f16::from_le_bytes([data[at + i * 2], data[at + i * 2 + 1]]).to_f32()
            };
            out.push([channel(0), channel(1), channel(2), channel(3)]);
        }
    }
    drop(data);
    buffer.unmap();
    out
}

/// The mean of a face's red channel.
fn mean(texels: &[[f32; 4]]) -> f32 {
    if texels.is_empty() {
        return 0.0;
    }
    texels.iter().map(|t| t[0]).sum::<f32>() / texels.len() as f32
}

#[test]
fn a_uniform_environment_comes_back_as_the_constant_at_every_roughness() {
    // The identity the whole image path rests on. A prefilter normalised by the
    // sample count rather than the accumulated weight fails this, and fails it
    // by a factor that depends on roughness — so a scene's rough metals come
    // out dark and its smooth ones do not, which reads as a material problem.
    let gpu = gpu_or_skip!(gpu());
    let pixels = image(64, 32, |_, _| [3.0, 3.0, 3.0]);
    let env = Environment::new(&gpu.device, &gpu.queue, 64, 32, &pixels);

    for level in 0..MIPS {
        for layer in 0..6 {
            let texels = face(&gpu, &env, level, layer);
            for t in &texels {
                for c in 0..3 {
                    assert!(
                        (t[c] - 3.0).abs() < 0.05,
                        "mip {level} face {layer} holds {t:?} under a uniform 3.0"
                    );
                }
            }
        }
    }
}

#[test]
fn the_faces_look_where_the_cube_sampler_will_look_for_them() {
    // A cube written with a self-consistent but permuted mapping produces a sky
    // that is a sky, rotated — and nothing about a rendered picture says so.
    // The only way to check it is to put something recognisable in one place
    // and ask which face it landed on.
    //
    // A bright band around the top of the image is up, so it must be face four
    // (+Z in Kreate) and not any other.
    let gpu = gpu_or_skip!(gpu());
    let pixels = image(128, 64, |_, v| if v < 0.25 { [8.0; 3] } else { [0.0; 3] });
    let env = Environment::new(&gpu.device, &gpu.queue, 128, 64, &pixels);

    let up = mean(&face(&gpu, &env, 0, 4));
    let down = mean(&face(&gpu, &env, 0, 5));
    assert!(
        up > 6.0,
        "the +Z face averages {up} under a sky bright overhead"
    );
    assert!(down < 0.5, "the -Z face averages {down}");
    // And the four sides are in between, because the band reaches partway down
    // them. A mapping that put up on a side face would make one of these as
    // bright as the top.
    for layer in 0..4 {
        let side = mean(&face(&gpu, &env, 0, layer));
        assert!(
            side < up * 0.8,
            "face {layer} averages {side} against {up} overhead"
        );
    }
}

#[test]
fn an_image_is_not_mirrored_on_its_way_onto_the_cube() {
    // The mistake a cube mapping most easily makes and a picture most easily
    // hides. A sky is mostly symmetric enough that a mirrored one still reads
    // as a sky — so the fixture is deliberately not: one bright quadrant, put
    // where only one face can hold it.
    //
    // The centre column of an equirectangular image faces -Y (see
    // `environment::equirect_uv`), and `u` increases to the right. So a bright
    // stripe at u ≈ 0.75 is +X, which is face zero.
    let gpu = gpu_or_skip!(gpu());
    let pixels = image(128, 64, |u, v| {
        let horizon = (v - 0.5).abs() < 0.1;
        if horizon && (u - 0.75).abs() < 0.06 {
            [10.0; 3]
        } else {
            [0.0; 3]
        }
    });
    let env = Environment::new(&gpu.device, &gpu.queue, 128, 64, &pixels);

    let plus_x = mean(&face(&gpu, &env, 0, 0));
    let minus_x = mean(&face(&gpu, &env, 0, 1));
    assert!(
        plus_x > minus_x * 10.0 && plus_x > 0.1,
        "+X averages {plus_x} and -X {minus_x} for a stripe on the +X side"
    );
}

#[test]
fn a_rough_level_is_blurrier_than_a_smooth_one() {
    // What the prefilter is *for*. The sharp level keeps the stripe and the
    // rough one spreads it, and the way to say that without measuring a shape
    // is that the brightest texel falls while the mean does not.
    let gpu = gpu_or_skip!(gpu());
    let pixels = image(128, 64, |u, v| {
        if (v - 0.5).abs() < 0.05 && (u - 0.75).abs() < 0.03 {
            [40.0; 3]
        } else {
            [0.1; 3]
        }
    });
    let env = Environment::new(&gpu.device, &gpu.queue, 128, 64, &pixels);

    let peak = |level: u32| {
        face(&gpu, &env, level, 0)
            .iter()
            .map(|t| t[0])
            .fold(0.0f32, f32::max)
    };
    let sharp = peak(0);
    let rough = peak(MIPS - 1);
    assert!(
        rough < sharp * 0.5,
        "the sharp level peaks at {sharp} and the roughest at {rough}"
    );
    // And the energy did not leave with the sharpness: the roughest level is
    // very nearly the whole environment's average, because a lobe at roughness
    // one covers the hemisphere. A prefilter that lost energy would fail this
    // while passing the blur test above.
    let flat = mean(&face(&gpu, &env, MIPS - 1, 0));
    assert!(
        flat > 0.1,
        "the roughest level averages {flat}, which is less than the sky it came from"
    );
}

#[test]
fn the_nine_coefficients_and_the_cube_describe_the_same_sky() {
    // The two halves are computed by different code in different places — nine
    // numbers on the CPU, six faces on the GPU — and nothing else would notice
    // if they disagreed about which way is up. A surface lit by a diffuse term
    // pointing one way and a specular term pointing the other is a surface
    // nobody can debug from the picture.
    let gpu = gpu_or_skip!(gpu());
    let pixels = image(128, 64, |_, v| if v < 0.25 { [8.0; 3] } else { [0.0; 3] });
    let env = Environment::new(&gpu.device, &gpu.queue, 128, 64, &pixels);

    let sh = env.sh9();
    let up = environment::sh9_irradiance(sh, glam::Vec3::Z).x;
    let down = environment::sh9_irradiance(sh, glam::Vec3::NEG_Z).x;
    assert!(
        up > down * 3.0,
        "the coefficients give {up} up and {down} down for a sky bright overhead"
    );
    // The cube says the same, in its own units: irradiance against radiance, so
    // the two are not comparable directly — what is comparable is which way
    // each of them says the light comes from.
    let cube_up = mean(&face(&gpu, &env, 0, 4));
    let cube_down = mean(&face(&gpu, &env, 0, 5));
    assert!(
        cube_up > cube_down,
        "the cube disagrees with the coefficients about which way is up"
    );
}

#[test]
fn an_environment_with_no_pixels_is_black_rather_than_whatever_was_there() {
    // A half-loaded image is a real state — §3.6 made images sparse on purpose
    // — and the tiles that are not resident read as nothing. What must not
    // happen is that they read as whatever the allocator handed over.
    let gpu = gpu_or_skip!(gpu());
    let env = Environment::new(&gpu.device, &gpu.queue, 64, 32, &[]);
    for layer in 0..6 {
        let texels = face(&gpu, &env, 0, layer);
        assert!(
            texels.iter().all(|t| t[0] == 0.0 && t[1] == 0.0 && t[2] == 0.0),
            "face {layer} of an empty environment is not black"
        );
    }
    for c in env.sh9() {
        assert_eq!(*c, glam::Vec3::ZERO);
    }
}
