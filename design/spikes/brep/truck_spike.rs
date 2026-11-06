//! P0 spike, truck: test 1 of design/brep-kernel.md §4 (the plate with four
//! holes cut and a boss joined), its time and the volume of its mesh.
use std::time::Instant;
use truck_meshalgo::prelude::*;
use truck_modeling::*;

fn cylinder(c: Point3, r: f64, h: f64) -> Solid {
    let v = builder::vertex(Point3::new(c.x + r, c.y, c.z));
    let w = builder::rsweep(&v, c, Vector3::unit_z(), Rad(7.0));
    let f = builder::try_attach_plane(&[w]).unwrap();
    builder::tsweep(&f, Vector3::unit_z() * h)
}

fn run(k: f64, tol: f64) {
    let t0 = Instant::now();
    let v = builder::vertex(Point3::origin());
    let e = builder::tsweep(&v, Vector3::unit_x() * 0.1 * k);
    let f = builder::tsweep(&e, Vector3::unit_y() * 0.06 * k);
    let mut part = builder::tsweep(&f, Vector3::unit_z() * 0.01 * k);
    let mut steps = Vec::new();
    for (x, y) in [(0.01, 0.01), (0.09, 0.01), (0.01, 0.05), (0.09, 0.05)] {
        let mut hole = cylinder(Point3::new(x * k, y * k, -0.001 * k), 0.0033 * k, 0.012 * k);
        hole.not();
        match truck_shapeops::and(&part, &hole, tol) {
            Some(s) => {
                part = s;
                steps.push("cut ok");
            }
            None => {
                steps.push("cut FAILED");
            }
        }
    }
    let boss = cylinder(Point3::new(0.05 * k, 0.03 * k, 0.01 * k), 0.01 * k, 0.015 * k);
    match truck_shapeops::or(&part, &boss, tol) {
        Some(s) => {
            part = s;
            steps.push("join ok");
        }
        None => steps.push("join FAILED"),
    }
    let built = t0.elapsed();
    let mesh = part.triangulation(1e-5 * k).to_polygon();
    let v = mesh.volume();
    let want = k * k * k * (0.1 * 0.06 * 0.01 - 4.0 * std::f64::consts::PI * 0.0033f64.powi(2) * 0.01
        + std::f64::consts::PI * 0.01f64.powi(2) * 0.015);
    println!(
        "k={k} tol={tol}: {:?}; built in {:.1} ms; mesh volume err {:.2e}; faces {}",
        steps,
        built.as_secs_f64() * 1e3,
        (v - want).abs() / want,
        part.face_iter().count()
    );
}

fn main() {
    for (k, tol) in [(1.0, 1e-5), (1000.0, 0.05), (1000.0, 0.01), (1000.0, 0.001)] {
        run(k, tol);
    }
}
