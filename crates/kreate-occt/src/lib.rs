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
//! moved by an affine map, joined, cut, filleted, chamfered, shelled, swept,
//! lofted, drafted, projected with hidden-line removal, measured, meshed for
//! the screen (each triangle knowing the face it is on), and read/written as
//! STEP AP214/AP242.

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

/// Geometric signature of one B-rep face, used for persistent topological
/// naming across feature regeneration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceSignature {
    pub index: u32,
    pub center: [f64; 3],
    pub normal: [f64; 3],
    pub area: f64,
    pub surface_kind: u8,
}

/// Geometric signature of one B-rep edge, used for persistent topological
/// naming across feature regeneration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EdgeSignature {
    pub index: u32,
    pub start: [f64; 3],
    pub mid: [f64; 3],
    pub end: [f64; 3],
    pub length: f64,
    pub curve_kind: u8,
}

/// One projected 2D curve produced by OCCT's exact Hidden-Line Removal engine.
#[derive(Clone, Debug, PartialEq)]
pub enum HlrCurve {
    Line {
        a: [f64; 2],
        b: [f64; 2],
    },
    Arc {
        centre: [f64; 2],
        radius: f64,
        start: f64,
        sweep: f64,
    },
    Polyline(Vec<[f64; 2]>),
}

/// A projected 2D edge tagged with visibility and outline classification.
#[derive(Clone, Debug, PartialEq)]
pub struct HlrEdge {
    pub curve: HlrCurve,
    pub hidden: bool,
    pub silhouette: bool,
}

/// Whether the kernel is linked in.
pub fn available() -> bool {
    cfg!(feature = "occt")
}

pub use imp::{read_step, write_step, write_step_named, Shape};

#[cfg(feature = "occt")]
mod imp {
    use super::{encode, Boolean, EdgeSignature, FaceSignature, HlrCurve, HlrEdge, Loop, Mesh};
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

    #[repr(C)]
    struct KHlrBuffer {
        len: usize,
        data: *mut f64,
    }

    #[repr(C)]
    struct KTopoSig {
        count: usize,
        data: *mut f64,
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
        fn kocc_fillet(
            s: *const KShape,
            radius: f64,
            edge_pts: *const f64,
            n_pts: usize,
        ) -> *mut KShape;
        fn kocc_chamfer(
            s: *const KShape,
            dist: f64,
            edge_pts: *const f64,
            n_pts: usize,
        ) -> *mut KShape;
        fn kocc_shell(
            s: *const KShape,
            thickness: f64,
            open_pts: *const f64,
            n_pts: usize,
            has_dir: c_int,
            dx: f64,
            dy: f64,
            dz: f64,
        ) -> *mut KShape;
        fn kocc_sweep(
            profile: *const KShape,
            spine_pts: *const f64,
            n_pts: usize,
            closed: c_int,
        ) -> *mut KShape;
        fn kocc_loft(
            sections: *const *const KShape,
            n: usize,
            solid: c_int,
            ruled: c_int,
        ) -> *mut KShape;
        fn kocc_draft(
            s: *const KShape,
            angle: f64,
            dx: f64,
            dy: f64,
            dz: f64,
            nx: f64,
            ny: f64,
            nz: f64,
        ) -> *mut KShape;
        fn kocc_transform(s: *const KShape, m: *const f64) -> *mut KShape;
        fn kocc_volume(s: *const KShape) -> f64;
        fn kocc_valid(s: *const KShape) -> c_int;
        fn kocc_count_faces(s: *const KShape) -> usize;
        fn kocc_count_edges(s: *const KShape) -> usize;
        fn kocc_face_signatures(s: *const KShape) -> *mut KTopoSig;
        fn kocc_edge_signatures(s: *const KShape) -> *mut KTopoSig;
        fn kocc_topo_sig_free(sig: *mut KTopoSig);
        fn kocc_project_hlr(
            s: *const KShape,
            vx: f64,
            vy: f64,
            vz: f64,
            rx: f64,
            ry: f64,
            rz: f64,
        ) -> *mut KHlrBuffer;
        fn kocc_hlr_free(b: *mut KHlrBuffer);
        fn kocc_mesh(s: *const KShape, deflection: f64, angle: f64) -> *mut KMesh;
        fn kocc_mesh_free(m: *mut KMesh);
        fn kocc_read_step(path: *const c_char) -> *mut KShape;
        fn kocc_explode_solids(
            s: *const KShape,
            out_shapes: *mut *mut *mut KShape,
            out_count: *mut usize,
        ) -> c_int;
        fn kocc_shape_array_free(arr: *mut *mut KShape);
        fn kocc_write_step(
            shapes: *const *const KShape,
            n: usize,
            path: *const c_char,
            millimetres: c_int,
        ) -> c_int;
        fn kocc_write_step_named(
            shapes: *const *const KShape,
            names: *const *const c_char,
            n: usize,
            path: *const c_char,
            millimetres: c_int,
            ap242: c_int,
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

        /// Rounds the edges nearest `edges` (or all edges when empty) to `radius`.
        pub fn fillet(&self, radius: f64, edges: &[[f64; 3]]) -> Option<Shape> {
            let flat: Vec<f64> = edges.iter().flatten().copied().collect();
            let ptr = if flat.is_empty() {
                std::ptr::null()
            } else {
                flat.as_ptr()
            };
            // SAFETY: live shape and contiguous flat xyz array.
            own(unsafe { kocc_fillet(self.0, radius, ptr, edges.len()) })
        }

        /// Chamfers the edges nearest `edges` (or all edges when empty) by `distance`.
        pub fn chamfer(&self, distance: f64, edges: &[[f64; 3]]) -> Option<Shape> {
            let flat: Vec<f64> = edges.iter().flatten().copied().collect();
            let ptr = if flat.is_empty() {
                std::ptr::null()
            } else {
                flat.as_ptr()
            };
            // SAFETY: live shape and contiguous flat xyz array.
            own(unsafe { kocc_chamfer(self.0, distance, ptr, edges.len()) })
        }

        /// Hollows the solid inward by `thickness`, opening either the faces
        /// nearest `open_faces` or (when empty) the extreme face along `open_dir`.
        pub fn shell(
            &self,
            thickness: f64,
            open_faces: &[[f64; 3]],
            open_dir: Option<[f64; 3]>,
        ) -> Option<Shape> {
            let flat: Vec<f64> = open_faces.iter().flatten().copied().collect();
            let ptr = if flat.is_empty() {
                std::ptr::null()
            } else {
                flat.as_ptr()
            };
            let (has_dir, d) = match open_dir {
                Some(v) => (1, v),
                None => (0, [0.0; 3]),
            };
            // SAFETY: live shape and contiguous flat xyz array.
            own(unsafe {
                kocc_shell(
                    self.0,
                    thickness,
                    ptr,
                    open_faces.len(),
                    has_dir,
                    d[0],
                    d[1],
                    d[2],
                )
            })
        }

        /// Sweeps this planar face/profile along a 3D spine polyline.
        pub fn sweep(&self, spine: &[[f64; 3]], closed: bool) -> Option<Shape> {
            if spine.len() < 2 {
                return None;
            }
            let flat: Vec<f64> = spine.iter().flatten().copied().collect();
            // SAFETY: live profile shape and contiguous spine xyz array.
            own(unsafe { kocc_sweep(self.0, flat.as_ptr(), spine.len(), c_int::from(closed)) })
        }

        /// Lofts a solid (or shell) smoothly through two or more planar section shapes.
        pub fn loft(sections: &[&Shape], solid: bool, ruled: bool) -> Option<Shape> {
            if sections.len() < 2 {
                return None;
            }
            let ptrs: Vec<*const KShape> = sections.iter().map(|s| s.0 as *const KShape).collect();
            // SAFETY: live shapes, read only.
            own(unsafe {
                kocc_loft(
                    ptrs.as_ptr(),
                    ptrs.len(),
                    c_int::from(solid),
                    c_int::from(ruled),
                )
            })
        }

        /// Applies a mold draft angle (radians) to side faces parallel to `pull_dir`
        /// anchored at neutral plane point `neutral`.
        pub fn draft(&self, angle: f64, pull_dir: [f64; 3], neutral: [f64; 3]) -> Option<Shape> {
            // SAFETY: live shape, read only.
            own(unsafe {
                kocc_draft(
                    self.0,
                    angle,
                    pull_dir[0],
                    pull_dir[1],
                    pull_dir[2],
                    neutral[0],
                    neutral[1],
                    neutral[2],
                )
            })
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

        pub fn edge_count(&self) -> usize {
            // SAFETY: a live shape, read only.
            unsafe { kocc_count_edges(self.0) }
        }

        /// Geometric signatures of all faces in this shape.
        pub fn face_signatures(&self) -> Vec<FaceSignature> {
            // SAFETY: shim allocates KTopoSig and frees via kocc_topo_sig_free.
            unsafe {
                let ptr = kocc_face_signatures(self.0);
                if ptr.is_null() {
                    return Vec::new();
                }
                let sig = &*ptr;
                let raw = std::slice::from_raw_parts(sig.data, sig.count * 8);
                let mut out = Vec::with_capacity(sig.count);
                for (i, chunk) in raw.as_chunks::<8>().0.iter().enumerate() {
                    out.push(FaceSignature {
                        index: i as u32,
                        center: [chunk[0], chunk[1], chunk[2]],
                        normal: [chunk[3], chunk[4], chunk[5]],
                        area: chunk[6],
                        surface_kind: chunk[7] as u8,
                    });
                }
                kocc_topo_sig_free(ptr);
                out
            }
        }

        /// Geometric signatures of all edges in this shape.
        pub fn edge_signatures(&self) -> Vec<EdgeSignature> {
            // SAFETY: shim allocates KTopoSig and frees via kocc_topo_sig_free.
            unsafe {
                let ptr = kocc_edge_signatures(self.0);
                if ptr.is_null() {
                    return Vec::new();
                }
                let sig = &*ptr;
                let raw = std::slice::from_raw_parts(sig.data, sig.count * 11);
                let mut out = Vec::with_capacity(sig.count);
                for (i, chunk) in raw.as_chunks::<11>().0.iter().enumerate() {
                    out.push(EdgeSignature {
                        index: i as u32,
                        start: [chunk[0], chunk[1], chunk[2]],
                        mid: [chunk[3], chunk[4], chunk[5]],
                        end: [chunk[6], chunk[7], chunk[8]],
                        length: chunk[9],
                        curve_kind: chunk[10] as u8,
                    });
                }
                kocc_topo_sig_free(ptr);
                out
            }
        }

        /// Exact analytic hidden-line removal projection along `view_dir` with
        /// horizontal axis `right_dir`.
        pub fn project_hlr(&self, view_dir: [f64; 3], right_dir: [f64; 3]) -> Option<Vec<HlrEdge>> {
            // SAFETY: shim allocates KHlrBuffer and frees via kocc_hlr_free.
            unsafe {
                let ptr = kocc_project_hlr(
                    self.0,
                    view_dir[0],
                    view_dir[1],
                    view_dir[2],
                    right_dir[0],
                    right_dir[1],
                    right_dir[2],
                );
                if ptr.is_null() {
                    return None;
                }
                let buf = &*ptr;
                let raw = std::slice::from_raw_parts(buf.data, buf.len);
                let mut out = Vec::new();
                let mut i = 0usize;
                while i + 2 <= raw.len() {
                    let kind = raw[i] as i32;
                    let flags = raw[i + 1] as u32;
                    i += 2;
                    let hidden = (flags & 1) != 0;
                    let silhouette = (flags & 2) != 0;
                    match kind {
                        0 if i + 4 <= raw.len() => {
                            out.push(HlrEdge {
                                curve: HlrCurve::Line {
                                    a: [raw[i], raw[i + 1]],
                                    b: [raw[i + 2], raw[i + 3]],
                                },
                                hidden,
                                silhouette,
                            });
                            i += 4;
                        }
                        1 if i + 5 <= raw.len() => {
                            out.push(HlrEdge {
                                curve: HlrCurve::Arc {
                                    centre: [raw[i], raw[i + 1]],
                                    radius: raw[i + 2],
                                    start: raw[i + 3],
                                    sweep: raw[i + 4],
                                },
                                hidden,
                                silhouette,
                            });
                            i += 5;
                        }
                        2 if i < raw.len() => {
                            let n = raw[i] as usize;
                            i += 1;
                            if i + 2 * n <= raw.len() {
                                let mut pts = Vec::with_capacity(n);
                                for k in 0..n {
                                    pts.push([raw[i + 2 * k], raw[i + 2 * k + 1]]);
                                }
                                out.push(HlrEdge {
                                    curve: HlrCurve::Polyline(pts),
                                    hidden,
                                    silhouette,
                                });
                                i += 2 * n;
                            } else {
                                break;
                            }
                        }
                        _ => break,
                    }
                }
                kocc_hlr_free(ptr);
                Some(out)
            }
        }

        /// Splits a compound shape into its constituent solids (or shells).
        pub fn explode_solids(&self) -> Vec<Shape> {
            // SAFETY: shim allocates array of owned KShape pointers.
            unsafe {
                let mut arr: *mut *mut KShape = std::ptr::null_mut();
                let mut count: usize = 0;
                if kocc_explode_solids(self.0, &mut arr, &mut count) == 0 || arr.is_null() {
                    return Vec::new();
                }
                let slice = std::slice::from_raw_parts(arr, count);
                let out: Vec<Shape> = slice.iter().filter_map(|&p| own(p)).collect();
                kocc_shape_array_free(arr);
                out
            }
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
                    .as_chunks::<3>()
                    .0
                    .to_vec();
                let triangles = std::slice::from_raw_parts(k.indices, k.tris * 3)
                    .as_chunks::<3>()
                    .0
                    .to_vec();
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

    /// Reads a STEP AP203/AP214/AP242 file into an exact B-rep shape (in metres).
    pub fn read_step(path: &std::path::Path) -> Option<Shape> {
        let path = path.to_str().and_then(|p| CString::new(p).ok())?;
        // SAFETY: NUL-terminated path string, read only.
        own(unsafe { kocc_read_step(path.as_ptr()) })
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

    /// Writes named shapes to a STEP AP214 or AP242 file.
    pub fn write_step_named(
        shapes: &[(&str, &Shape)],
        path: &std::path::Path,
        millimetres: bool,
        ap242: bool,
    ) -> bool {
        let Some(path) = path.to_str().and_then(|p| CString::new(p).ok()) else {
            return false;
        };
        let c_names: Vec<CString> = shapes
            .iter()
            .map(|(n, _)| CString::new(n.replace('\0', "_")).unwrap_or_default())
            .collect();
        let name_ptrs: Vec<*const c_char> = c_names.iter().map(|c| c.as_ptr()).collect();
        let shape_ptrs: Vec<*const KShape> =
            shapes.iter().map(|(_, s)| s.0 as *const KShape).collect();
        // SAFETY: live shapes, NUL-terminated names & path, read only.
        unsafe {
            kocc_write_step_named(
                shape_ptrs.as_ptr(),
                name_ptrs.as_ptr(),
                shape_ptrs.len(),
                path.as_ptr(),
                c_int::from(millimetres),
                c_int::from(ap242),
            ) != 0
        }
    }
}

#[cfg(not(feature = "occt"))]
mod imp {
    use super::{Boolean, EdgeSignature, FaceSignature, HlrEdge, Loop, Mesh};

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
        pub fn fillet(&self, _radius: f64, _edges: &[[f64; 3]]) -> Option<Shape> {
            None
        }
        pub fn chamfer(&self, _distance: f64, _edges: &[[f64; 3]]) -> Option<Shape> {
            None
        }
        pub fn shell(
            &self,
            _thickness: f64,
            _open_faces: &[[f64; 3]],
            _open_dir: Option<[f64; 3]>,
        ) -> Option<Shape> {
            None
        }
        pub fn sweep(&self, _spine: &[[f64; 3]], _closed: bool) -> Option<Shape> {
            None
        }
        pub fn loft(_sections: &[&Shape], _solid: bool, _ruled: bool) -> Option<Shape> {
            None
        }
        pub fn draft(&self, _angle: f64, _pull_dir: [f64; 3], _neutral: [f64; 3]) -> Option<Shape> {
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
        pub fn edge_count(&self) -> usize {
            0
        }
        pub fn face_signatures(&self) -> Vec<FaceSignature> {
            Vec::new()
        }
        pub fn edge_signatures(&self) -> Vec<EdgeSignature> {
            Vec::new()
        }
        pub fn project_hlr(
            &self,
            _view_dir: [f64; 3],
            _right_dir: [f64; 3],
        ) -> Option<Vec<HlrEdge>> {
            None
        }
        pub fn explode_solids(&self) -> Vec<Shape> {
            Vec::new()
        }
        pub fn mesh(&self, _deflection: f64, _angle: f64) -> Option<Mesh> {
            None
        }
    }

    pub fn read_step(_path: &std::path::Path) -> Option<Shape> {
        None
    }

    pub fn write_step(_shapes: &[&Shape], _path: &std::path::Path, _millimetres: bool) -> bool {
        false
    }

    pub fn write_step_named(
        _shapes: &[(&str, &Shape)],
        _path: &std::path::Path,
        _millimetres: bool,
        _ap242: bool,
    ) -> bool {
        false
    }
}

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

    #[test]
    fn exact_fillet_chamfer_shell_loft_sweep_draft_step_and_hlr() {
        let box_solid = Shape::face(&rect(0.0, 0.0, 0.1, 0.1), &[])
            .unwrap()
            .extrude([0.0, 0.0, 0.1])
            .unwrap();
        let box_vol = box_solid.volume();

        // Fillet one edge
        let filleted = box_solid.fillet(0.01, &[[0.05, 0.0, 0.1]]).unwrap();
        assert!(filleted.is_valid());
        assert!(filleted.volume() < box_vol);
        assert!(filleted.face_count() > box_solid.face_count());

        // Chamfer one edge
        let chamfered = box_solid.chamfer(0.01, &[[0.05, 0.0, 0.1]]).unwrap();
        assert!(chamfered.is_valid());
        let expected_chamfer_vol = box_vol - 0.5 * 0.01 * 0.01 * 0.1;
        assert!((chamfered.volume() - expected_chamfer_vol).abs() < 1e-9);

        // Shell opening top +Z face
        let tray = box_solid.shell(0.01, &[], Some([0.0, 0.0, 1.0])).unwrap();
        assert!(tray.is_valid());
        let expected_tray_vol = box_vol - 0.08 * 0.08 * 0.09;
        assert!(
            (tray.volume() - expected_tray_vol).abs() < 1e-6,
            "tray vol {} vs {}",
            tray.volume(),
            expected_tray_vol
        );

        // Loft between square and smaller square
        let bot = Shape::face(&rect(-0.05, -0.05, 0.05, 0.05), &[]).unwrap();
        let top = Shape::face(&rect(-0.025, -0.025, 0.025, 0.025), &[])
            .unwrap()
            .transformed([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.1],
            ])
            .unwrap();
        let frustum = Shape::loft(&[&bot, &top], true, true).unwrap();
        assert!(frustum.is_valid());
        let want_frustum = (0.1 / 3.0) * (0.01 + 0.0025 + (0.01 * 0.0025f64).sqrt());
        assert!((frustum.volume() - want_frustum).abs() < 1e-7);

        // Sweep circular profile along straight spine
        let prof = Shape::face(
            &vec![Piece::Circle {
                centre: [0.0, 0.0],
                radius: 0.01,
            }],
            &[],
        )
        .unwrap();
        let swept = prof
            .sweep(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.08]], false)
            .unwrap();
        assert!(swept.is_valid());
        let want_swept = std::f64::consts::PI * 0.01 * 0.01 * 0.08;
        assert!((swept.volume() - want_swept).abs() < 1e-7);

        // Draft side faces by 5 degrees
        let drafted = box_solid
            .draft(5.0f64.to_radians(), [0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
            .unwrap();
        assert!(drafted.is_valid());
        assert!((drafted.volume() - box_vol).abs() > 1e-5);

        // Named STEP AP242 write + STEP read round-trip
        let step_path =
            std::env::temp_dir().join(format!("kreate-occt-test-{}.step", std::process::id()));
        assert!(write_step_named(
            &[("Bracket", &filleted)],
            &step_path,
            true,
            true
        ));
        let reloaded = read_step(&step_path).expect("STEP read back");
        assert!(reloaded.is_valid());
        assert!(
            (reloaded.volume() - filleted.volume()).abs() < 1e-8,
            "reloaded={} filleted={}",
            reloaded.volume(),
            filleted.volume()
        );
        let solids = reloaded.explode_solids();
        assert_eq!(solids.len(), 1);
        let _ = std::fs::remove_file(&step_path);

        // Topology signatures
        assert_eq!(box_solid.face_signatures().len(), 6);
        assert_eq!(box_solid.edge_signatures().len(), 12);

        // Exact HLR projection of a cylinder contains a true circular arc
        let cyl = Shape::face(
            &vec![Piece::Circle {
                centre: [0.0, 0.0],
                radius: 0.02,
            }],
            &[],
        )
        .unwrap()
        .extrude([0.0, 0.0, 0.05])
        .unwrap();
        let hlr = cyl.project_hlr([0.0, 0.0, 1.0], [1.0, 0.0, 0.0]).unwrap();
        assert!(
            hlr.iter().any(|e| !e.hidden
                && matches!(e.curve, HlrCurve::Arc { radius, .. } if (radius - 0.02).abs() < 1e-5)),
            "top view of cylinder must contain exact circular arc: {hlr:?}"
        );
    }
}
