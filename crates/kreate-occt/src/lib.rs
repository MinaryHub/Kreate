//! Exact solids through OpenCASCADE: what Kreate's features are built with
//! when the kernel is there. See design/brep-kernel.md.
//!
//! The kernel is linked only with the `occt` feature. Without it this crate
//! still builds and answers every question — [`available`] is false and
//! each operation `None` — so the rest of Kreate takes its mesh path, as it
//! does when the kernel gives up on a shape.
//!
//! Everything is in metres, as Kreate keeps it. A face is made in the plane
//! z = 0 from loops of [`Piece`]s; solids are made from faces; a solid is
//! moved by an affine map, joined, cut, measured, meshed for the screen
//! (each triangle knowing the face it is on) and written to STEP.

/// One piece of a loop, in the plane z = 0.
#[derive(Clone, Debug, PartialEq)]
pub enum Piece {
    Line {
        a: [f64; 2],
        b: [f64; 2],
    },
    /// From `start` through `sweep` radians, anticlockwise when positive.
    Arc {
        centre: [f64; 2],
        radius: f64,
        start: f64,
        sweep: f64,
    },
    Circle {
        centre: [f64; 2],
        radius: f64,
    },
    /// Its flat knot vector, `ctrl.len() + degree + 1` long; weights for a
    /// rational one.
    Spline {
        degree: usize,
        ctrl: Vec<[f64; 2]>,
        knots: Vec<f64>,
        weights: Option<Vec<f64>>,
    },
    /// `major` the half-axis from the centre, `ratio` the minor over the
    /// major, the part from `start` through `sweep` of its own angle.
    Ellipse {
        centre: [f64; 2],
        major: [f64; 2],
        ratio: f64,
        start: f64,
        sweep: f64,
    },
}

/// A closed loop of pieces, end to end.
pub type Loop = Vec<Piece>;

/// The numbers the shim reads a face from: see `kocc_face`.
#[cfg_attr(not(feature = "occt"), allow(dead_code))]
fn encode(outer: &Loop, holes: &[Loop]) -> Vec<f64> {
    let mut out = vec![(1 + holes.len()) as f64];
    for l in std::iter::once(outer).chain(holes) {
        out.push(l.len() as f64);
        for p in l {
            match p {
                Piece::Line { a, b } => out.extend([0.0, a[0], a[1], b[0], b[1]]),
                Piece::Arc {
                    centre,
                    radius,
                    start,
                    sweep,
                } => out.extend([1.0, centre[0], centre[1], *radius, *start, *sweep]),
                Piece::Circle { centre, radius } => {
                    out.extend([2.0, centre[0], centre[1], *radius])
                }
                Piece::Spline {
                    degree,
                    ctrl,
                    knots,
                    weights,
                } => {
                    out.extend([
                        3.0,
                        *degree as f64,
                        ctrl.len() as f64,
                        knots.len() as f64,
                        if weights.is_some() { 1.0 } else { 0.0 },
                    ]);
                    out.extend(ctrl.iter().flatten());
                    out.extend(knots);
                    if let Some(w) = weights {
                        out.extend(w);
                    }
                }
                Piece::Ellipse {
                    centre,
                    major,
                    ratio,
                    start,
                    sweep,
                } => out.extend([
                    4.0, centre[0], centre[1], major[0], major[1], *ratio, *start, *sweep,
                ]),
            }
        }
    }
    out
}

/// How two solids are put together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boolean {
    Join,
    Cut,
    Common,
}

/// Triangles for the screen: positions, three indices a triangle, and the
/// B-rep face (from 0) each triangle lies on.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mesh {
    pub positions: Vec<[f64; 3]>,
    pub triangles: Vec<[u32; 3]>,
    pub faces: Vec<u32>,
}

/// Whether the kernel is linked in.
pub fn available() -> bool {
    cfg!(feature = "occt")
}

pub use imp::Shape;

#[cfg(feature = "occt")]
mod imp {
    use super::{encode, Boolean, Loop, Mesh};
    use std::ffi::{c_char, c_int, CString};

    #[repr(C)]
    struct KShape {
        _private: [u8; 0],
    }

    #[repr(C)]
    struct KMesh {
        verts: usize,
        positions: *mut f64,
        tris: usize,
        indices: *mut u32,
        faces: *mut u32,
    }

    extern "C" {
        fn kocc_face(data: *const f64, len: usize) -> *mut KShape;
        fn kocc_prism(face: *const KShape, dx: f64, dy: f64, dz: f64) -> *mut KShape;
        fn kocc_revolve(
            face: *const KShape,
            px: f64,
            py: f64,
            pz: f64,
            dx: f64,
            dy: f64,
            dz: f64,
            angle: f64,
        ) -> *mut KShape;
        fn kocc_boolean(a: *const KShape, b: *const KShape, op: c_int) -> *mut KShape;
        fn kocc_transform(s: *const KShape, m: *const f64) -> *mut KShape;
        fn kocc_volume(s: *const KShape) -> f64;
        fn kocc_valid(s: *const KShape) -> c_int;
        fn kocc_count_faces(s: *const KShape) -> usize;
        fn kocc_mesh(s: *const KShape, deflection: f64, angle: f64) -> *mut KMesh;
        fn kocc_mesh_free(m: *mut KMesh);
        fn kocc_write_step(
            shapes: *const *const KShape,
            n: usize,
            path: *const c_char,
            millimetres: c_int,
        ) -> c_int;
        fn kocc_free(s: *mut KShape);
    }

    /// A shape the kernel holds: a face, a solid, or several.
    pub struct Shape(*mut KShape);

    // SAFETY: a shape is owned by its handle alone and OCCT's shapes are
    // immutable values; the kernel's algorithms build new ones and read the
    // old, which OCCT allows from several threads.
    unsafe impl Send for Shape {}
    unsafe impl Sync for Shape {}

    impl Drop for Shape {
        fn drop(&mut self) {
            // SAFETY: made by the shim, freed once, here.
            unsafe { kocc_free(self.0) }
        }
    }

    fn own(p: *mut KShape) -> Option<Shape> {
        (!p.is_null()).then_some(Shape(p))
    }

    impl Shape {
        /// A face in the plane z = 0: `outer` its outside, `holes` taken out.
        pub fn face(outer: &Loop, holes: &[Loop]) -> Option<Shape> {
            let data = encode(outer, holes);
            // SAFETY: the shim reads `len` numbers and keeps none of them.
            own(unsafe { kocc_face(data.as_ptr(), data.len()) })
        }

        /// The face pushed along `d`: a solid.
        pub fn extrude(&self, d: [f64; 3]) -> Option<Shape> {
            // SAFETY: a live shape, read only.
            own(unsafe { kocc_prism(self.0, d[0], d[1], d[2]) })
        }

        /// The face turned `angle` radians about the line through `at`
        /// along `axis`.
        pub fn revolve(&self, at: [f64; 3], axis: [f64; 3], angle: f64) -> Option<Shape> {
            // SAFETY: a live shape, read only.
            own(unsafe {
                kocc_revolve(
                    self.0, at[0], at[1], at[2], axis[0], axis[1], axis[2], angle,
                )
            })
        }

        pub fn boolean(&self, other: &Shape, op: Boolean) -> Option<Shape> {
            let op = match op {
                Boolean::Join => 0,
                Boolean::Cut => 1,
                Boolean::Common => 2,
            };
            // SAFETY: two live shapes, read only.
            own(unsafe { kocc_boolean(self.0, other.0, op) })
        }

        /// Moved by the affine map whose rows are given.
        pub fn transformed(&self, rows: [[f64; 4]; 3]) -> Option<Shape> {
            let m: Vec<f64> = rows.iter().flatten().copied().collect();
            // SAFETY: twelve numbers, read only.
            own(unsafe { kocc_transform(self.0, m.as_ptr()) })
        }

        /// Cubic metres, NaN when it cannot be measured.
        pub fn volume(&self) -> f64 {
            // SAFETY: a live shape, read only.
            unsafe { kocc_volume(self.0) }
        }

        /// Whether the kernel's checker finds nothing wrong with it.
        pub fn is_valid(&self) -> bool {
            // SAFETY: a live shape, read only.
            unsafe { kocc_valid(self.0) != 0 }
        }

        pub fn face_count(&self) -> usize {
            // SAFETY: a live shape, read only.
            unsafe { kocc_count_faces(self.0) }
        }

        /// Triangles no further than `deflection` metres from the surface,
        /// nor more than `angle` radians turned from one to the next.
        pub fn mesh(&self, deflection: f64, angle: f64) -> Option<Mesh> {
            // SAFETY: the shim hands back arrays of the lengths it says,
            // copied out here, then freed by it.
            unsafe {
                let m = kocc_mesh(self.0, deflection, angle);
                if m.is_null() {
                    return None;
                }
                let k = &*m;
                let positions = std::slice::from_raw_parts(k.positions, k.verts * 3)
                    .chunks_exact(3)
                    .map(|c| [c[0], c[1], c[2]])
                    .collect();
                let triangles = std::slice::from_raw_parts(k.indices, k.tris * 3)
                    .chunks_exact(3)
                    .map(|c| [c[0], c[1], c[2]])
                    .collect();
                let faces = std::slice::from_raw_parts(k.faces, k.tris).to_vec();
                kocc_mesh_free(m);
                Some(Mesh {
                    positions,
                    triangles,
                    faces,
                })
            }
        }
    }

    /// The shapes as one STEP AP214 file, its lengths in millimetres or
    /// metres. Whether it was written.
    pub fn write_step(shapes: &[&Shape], path: &std::path::Path, millimetres: bool) -> bool {
        let Some(path) = path.to_str().and_then(|p| CString::new(p).ok()) else {
            return false;
        };
        let ptrs: Vec<*const KShape> = shapes.iter().map(|s| s.0 as *const KShape).collect();
        // SAFETY: live shapes and a NUL-terminated path, read only.
        unsafe {
            kocc_write_step(
                ptrs.as_ptr(),
                ptrs.len(),
                path.as_ptr(),
                c_int::from(millimetres),
            ) != 0
        }
    }
}

#[cfg(not(feature = "occt"))]
mod imp {
    use super::{Boolean, Loop, Mesh};

    /// No kernel: there is never a shape, and every operation says so.
    pub struct Shape(());

    impl Shape {
        pub fn face(_outer: &Loop, _holes: &[Loop]) -> Option<Shape> {
            None
        }
        pub fn extrude(&self, _d: [f64; 3]) -> Option<Shape> {
            None
        }
        pub fn revolve(&self, _at: [f64; 3], _axis: [f64; 3], _angle: f64) -> Option<Shape> {
            None
        }
        pub fn boolean(&self, _other: &Shape, _op: Boolean) -> Option<Shape> {
            None
        }
        pub fn transformed(&self, _rows: [[f64; 4]; 3]) -> Option<Shape> {
            None
        }
        pub fn volume(&self) -> f64 {
            f64::NAN
        }
        pub fn is_valid(&self) -> bool {
            false
        }
        pub fn face_count(&self) -> usize {
            0
        }
        pub fn mesh(&self, _deflection: f64, _angle: f64) -> Option<Mesh> {
            None
        }
    }

    pub fn write_step(_shapes: &[&Shape], _path: &std::path::Path, _millimetres: bool) -> bool {
        false
    }
}

pub use imp::write_step;

#[cfg(all(test, feature = "occt"))]
mod tests {
    use super::*;

    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Loop {
        let c = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
        (0..4)
            .map(|i| Piece::Line {
                a: c[i],
                b: c[(i + 1) % 4],
            })
            .collect()
    }

    /// The spike's first test through the crate: a plate with four holes
    /// (as circles in its face) extruded, and a boss joined — exact.
    #[test]
    fn a_plate_with_holes_and_a_boss_is_exact() {
        let holes: Vec<Loop> = [(0.01, 0.01), (0.09, 0.01), (0.01, 0.05), (0.09, 0.05)]
            .iter()
            .map(|&(x, y)| {
                vec![Piece::Circle {
                    centre: [x, y],
                    radius: 0.0033,
                }]
            })
            .collect();
        let plate = Shape::face(&rect(0.0, 0.0, 0.1, 0.06), &holes)
            .unwrap()
            .extrude([0.0, 0.0, 0.01])
            .unwrap();
        let boss = Shape::face(
            &vec![Piece::Circle {
                centre: [0.0, 0.0],
                radius: 0.01,
            }],
            &[],
        )
        .unwrap()
        .extrude([0.0, 0.0, 0.015])
        .unwrap()
        .transformed([
            [1.0, 0.0, 0.0, 0.05],
            [0.0, 1.0, 0.0, 0.03],
            [0.0, 0.0, 1.0, 0.01],
        ])
        .unwrap();
        let part = plate.boolean(&boss, Boolean::Join).unwrap();
        let pi = std::f64::consts::PI;
        let want =
            0.1 * 0.06 * 0.01 - 4.0 * pi * 0.0033f64.powi(2) * 0.01 + pi * 0.01f64.powi(2) * 0.015;
        assert!(part.is_valid());
        assert!(
            (part.volume() - want).abs() < want * 1e-9,
            "{}",
            part.volume()
        );
        let mesh = part.mesh(1e-5, 0.2).unwrap();
        assert!(!mesh.triangles.is_empty());
        assert_eq!(mesh.faces.len(), mesh.triangles.len());
        assert!(*mesh.faces.iter().max().unwrap() as usize + 1 == part.face_count());
    }

    /// An arc, a spline and an ellipse make a face, and a half turn of a
    /// rectangle beside the axis is half a tube.
    #[test]
    fn curved_loops_make_faces_and_revolve() {
        let d = Shape::face(
            &vec![
                Piece::Line {
                    a: [-0.01, 0.0],
                    b: [0.01, 0.0],
                },
                Piece::Arc {
                    centre: [0.0, 0.0],
                    radius: 0.01,
                    start: 0.0,
                    sweep: std::f64::consts::PI,
                },
            ],
            &[],
        )
        .unwrap();
        let half_disc = d.extrude([0.0, 0.0, 0.001]).unwrap();
        let want = std::f64::consts::PI * 1e-4 * 0.5 * 0.001;
        assert!((half_disc.volume() - want).abs() < want * 1e-9);
        let e = Shape::face(
            &vec![Piece::Ellipse {
                centre: [0.0, 0.0],
                major: [0.02, 0.0],
                ratio: 0.5,
                start: 0.0,
                sweep: std::f64::consts::TAU,
            }],
            &[],
        )
        .unwrap();
        let v = e.extrude([0.0, 0.0, 0.001]).unwrap().volume();
        let want = std::f64::consts::PI * 0.02 * 0.01 * 0.001;
        assert!((v - want).abs() < want * 1e-9, "{v}");
        let ring = Shape::face(&rect(0.01, 0.0, 0.02, 0.005), &[])
            .unwrap()
            .revolve([0.0; 3], [0.0, 1.0, 0.0], std::f64::consts::PI)
            .unwrap();
        let want = std::f64::consts::PI * (0.02f64.powi(2) - 0.01f64.powi(2)) * 0.005 * 0.5;
        assert!(
            (ring.volume() - want).abs() < want * 1e-9,
            "{}",
            ring.volume()
        );
    }
}
