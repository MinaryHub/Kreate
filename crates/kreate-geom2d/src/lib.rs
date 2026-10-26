//! Exact 2D geometry for drawings: curves as they are drawn — a line, an
//! arc, a circle, an ellipse, a polyline with bulges, a spline — and what
//! drafting does with them: where one meets another, the point on one
//! nearest a click, the polyline that draws one on screen to a given error.
//!
//! # Why curves, and not points
//!
//! A sketch used to be points and lines, with a circle as a constraint on a
//! ring of points. That is enough to extrude, and not enough to draft: a
//! trim lands where two *chords* cross instead of where the curves do, an
//! offset of a polygon is not the offset of a circle, and a DXF reader on
//! the other side gets forty lines where an arc was drawn. Here a curve is
//! what it is, and it becomes points only where something consumes points
//! — the screen, a region, a mesh — to an error that consumer chooses.
//!
//! # Parameters
//!
//! Every curve is a function of one parameter over [`Curve2D::domain`]:
//! `[0, 1]` for a line, an arc, a circle and an ellipse; `[0, n]` for a
//! polyline of `n` segments, segment `i` being `[i, i + 1]`; the knot
//! domain for a spline. One convention for every kind is what lets trim,
//! split and intersection be written once.
//!
//! # Standing alone
//!
//! No dependency but `glam`, so it builds for wasm as the mesh kernel does
//! and is tested without a window or a GPU.

mod curve;
mod edit;
mod intersect;
mod prim;
mod regions;
mod spline;

pub use curve::Curve2D;
pub use edit::{fillet, Fillet};
pub use intersect::{intersect, Intersection};
pub use prim::Prim;
pub use regions::{regions, BoundaryPiece, Loop, Region};
pub use spline::Spline;

pub use glam::DVec2;

/// How close two points are to be one, in the drawing's unit (metres).
///
/// A nanometre: far below anything a drawing shows or a machine cuts, far
/// above the rounding of `f64` arithmetic on coordinates of a building.
pub const POINT_EPS: f64 = 1e-9;

/// The full turn.
pub const TAU: f64 = std::f64::consts::TAU;
