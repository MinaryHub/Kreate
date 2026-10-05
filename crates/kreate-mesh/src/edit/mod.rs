//! Mesh editing operations on [`BMesh`](crate::BMesh).
//!
//! These are the topology half of the operators the plan lists for v1. They take
//! a `&mut BMesh` and element handles and know nothing about documents, undo or
//! parameters — `kreate-ops` wraps each one in an [`Operator`] and supplies
//! those. The split is what lets the whole modelling toolset be tested without a
//! document, a GPU or a mouse.
//!
//! [`Operator`]: https://docs.rs/kreate-ops

pub mod bevel;
pub mod bridge;
pub mod delete;
pub mod extrude;
pub mod fill;
pub mod inset;
pub mod join;
pub mod knife;
pub mod loopcut;
pub mod merge;
pub mod normals;
pub mod region;
pub mod spin;
pub mod split;
pub mod subdivide;
pub mod uv_edit;
pub mod uv_pack;
pub mod uv_project;
pub mod uv_unwrap;
pub mod walk;

pub use bevel::{bevel_edges, bevel_edges_segments, bevel_selected, bevel_selected_segments, Bevel};
pub use bridge::{bridge_loops, bridge_selected, open_borders, Bridge, BridgeError};
pub use delete::{delete, dissolve_edges, dissolve_faces, dissolve_verts, DeleteMode, Dissolution};
pub use extrude::{extrude_edges_and_verts, extrude_faces, extrude_selected_faces, Extrusion, WireExtrusion};
pub use fill::{fill, fill_selected, grid_fill, the_only_border, Fill, FillError};
pub use inset::{inset_individual, inset_region, inset_selected, Inset};
pub use knife::{knife, Knife};
pub use join::{join, separate_selected, Part, Report as JoinReport};
pub use loopcut::{loop_cut, loop_cut_selected, LoopCut};
pub use merge::{merge_groups, merge_selected, weld_all, weld_by_distance, Merge, MergeTarget};
pub use normals::{
    auto_smooth, flip, recalculate, set_crease, set_shade_smooth, set_sharp, set_sharp_selected, Recalculation,
};
pub use region::Region;
pub use spin::{close_spin, spin, spin_selected, Spin};
pub use split::{rip_selected_edges, split_edges, split_region, split_selected_faces, Split};
pub use subdivide::{subdivide_faces, subdivide_selected, Subdivision};
pub use uv_edit::{islands as uv_islands, SelectAction as UvSelectAction, Transform2};
pub use uv_pack::{cluster_by_angle, pack as uv_pack, smart_project};
pub use uv_project::{project as uv_project, project_with as uv_project_with, Fit, Projection};
pub use walk::{edge_loop, edge_ring, face_loop};
