//! How long a sculpt stroke takes on a large mesh, as it grows dab by dab.
//! `cargo run --release -p kreate-mesh --example sculpt_cost`

use std::time::Instant;

use kreate_math::Vec3;
use kreate_mesh::primitives;
use kreate_mesh::sculpt::{apply, Brush, Dab, Stroke, StrokeCache};

fn main() {
    let base = primitives::uv_sphere(1.0, 512, 256);
    println!("verts {}", base.num_verts());
    for brush in [Brush::Draw, Brush::Smooth, Brush::Inflate, Brush::Grab] {
        let stroke = Stroke {
            brush,
            strength: 0.5,
            invert: false,
            grab: Vec3::new(0.0, 0.0, 0.2),
            mirror_x: true,
        };
        for n in [1usize, 10, 100] {
            let dabs: Vec<Dab> = (0..n)
                .map(|i| {
                    let a = i as f32 * 0.02;
                    Dab {
                        centre: Vec3::new(a.cos(), a.sin(), 0.0),
                        radius: 0.1,
                    }
                })
                .collect();
            let mut m = base.clone();
            let t = Instant::now();
            apply(&mut m, &dabs, &stroke);
            println!("{brush:?} {n:>3} dabs: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        }
        // As a stroke is drawn: the same stroke re-run with one more dab.
        let dabs: Vec<Dab> = (0..100)
            .map(|i| Dab {
                centre: Vec3::new((i as f32 * 0.02).cos(), (i as f32 * 0.02).sin(), 0.0),
                radius: 0.1,
            })
            .collect();
        let mut cache = StrokeCache::new(&base, &stroke, 0.1);
        let t = Instant::now();
        let mut worst = 0.0f64;
        for n in 1..=100 {
            let mut m = base.clone();
            let f = Instant::now();
            cache.run(&mut m, &dabs[..n], &stroke);
            worst = worst.max(f.elapsed().as_secs_f64() * 1e3);
        }
        println!(
            "{brush:?} drawn dab by dab: {:.1} ms for 100, worst {worst:.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }
}
