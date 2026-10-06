//! Geometry nodes: a mesh made by a graph.
//!
//! Blender's Geometry Nodes, its first slice. An object can carry a graph that
//! runs after its modifier stack: Group Input is the mesh as the stack left
//! it, Group Output is what is drawn, and between them nodes transform, repeat,
//! subdivide, mirror and join geometry, or make some from nothing. One kind of
//! wire — geometry — and each node one output.
//!
//! The graph is data on the object, saved and copied with it, and evaluated
//! by the modifier cache like the stack, keyed on its input and its own value.

use kreate_math::{DAffine3, DQuat, DVec3, Vec3};
use kreate_mesh::edit::join::{join, Part};
use kreate_mesh::modifiers::{array, mirror, subdivide, ArrayParams, MirrorParams, SubdivideParams};
use kreate_mesh::MeshData;

/// What a node does.
#[derive(Clone, Debug, PartialEq)]
pub enum GeoNodeKind {
    /// The object's mesh, as its modifier stack left it.
    Input,
    /// What the object shows.
    Output,
    /// Moved, turned (degrees, XYZ) and scaled.
    Transform {
        translate: [f32; 3],
        rotate: [f32; 3],
        scale: [f32; 3],
    },
    /// Copies along an offset, in metres.
    Array { count: u32, offset: [f32; 3] },
    /// Each level splits every edge; smoothed toward the limit surface.
    Subdivide { levels: u32, smooth: bool },
    /// Reflected across one plane: 0 X, 1 Y, 2 Z.
    Mirror { axis: u32 },
    /// Two geometries as one.
    Join,
    /// A shape from nothing: 0 cube, 1 UV sphere, 2 cylinder, 3 cone,
    /// 4 torus, 5 icosphere.
    Primitive { shape: u32, size: f32 },
}

/// The primitive shapes, in [`GeoNodeKind::Primitive`]'s order.
pub const PRIMITIVES: [&str; 6] = ["cube", "uv_sphere", "cylinder", "cone", "torus", "ico_sphere"];

impl GeoNodeKind {
    /// How many geometry inputs it has.
    pub fn inputs(&self) -> usize {
        match self {
            GeoNodeKind::Input | GeoNodeKind::Primitive { .. } => 0,
            GeoNodeKind::Join => 2,
            _ => 1,
        }
    }

    /// Whether it has an output: every kind but the group's output.
    pub fn has_output(&self) -> bool {
        !matches!(self, GeoNodeKind::Output)
    }

    /// A stable name for files and scripts.
    pub fn tag(&self) -> &'static str {
        match self {
            GeoNodeKind::Input => "input",
            GeoNodeKind::Output => "output",
            GeoNodeKind::Transform { .. } => "transform",
            GeoNodeKind::Array { .. } => "array",
            GeoNodeKind::Subdivide { .. } => "subdivide",
            GeoNodeKind::Mirror { .. } => "mirror",
            GeoNodeKind::Join => "join",
            GeoNodeKind::Primitive { .. } => "primitive",
        }
    }

    /// The kinds the Add menu offers, in its order. Not the group's input or
    /// output: a graph has one of each.
    pub const ADDABLE: [&'static str; 6] =
        ["transform", "array", "subdivide", "mirror", "join", "primitive"];

    /// The node's numbers, flat — for a file, and for its fields by index.
    pub fn values(&self) -> Vec<f32> {
        match self {
            GeoNodeKind::Input | GeoNodeKind::Output | GeoNodeKind::Join => Vec::new(),
            GeoNodeKind::Transform {
                translate,
                rotate,
                scale,
            } => translate.iter().chain(rotate).chain(scale).copied().collect(),
            GeoNodeKind::Array { count, offset } => {
                vec![*count as f32, offset[0], offset[1], offset[2]]
            }
            GeoNodeKind::Subdivide { levels, smooth } => {
                vec![*levels as f32, if *smooth { 1.0 } else { 0.0 }]
            }
            GeoNodeKind::Mirror { axis } => vec![*axis as f32],
            GeoNodeKind::Primitive { shape, size } => vec![*shape as f32, *size],
        }
    }

    /// A kind from its tag and numbers; missing numbers take the defaults.
    /// `None` for a tag this build does not know.
    pub fn from_parts(tag: &str, values: &[f32]) -> Option<Self> {
        let at = |i: usize, d: f32| values.get(i).copied().filter(|v| v.is_finite()).unwrap_or(d);
        Some(match tag {
            "input" => GeoNodeKind::Input,
            "output" => GeoNodeKind::Output,
            "transform" => GeoNodeKind::Transform {
                translate: [at(0, 0.0), at(1, 0.0), at(2, 0.0)],
                rotate: [at(3, 0.0), at(4, 0.0), at(5, 0.0)],
                scale: [at(6, 1.0), at(7, 1.0), at(8, 1.0)],
            },
            "array" => GeoNodeKind::Array {
                count: at(0, 3.0).clamp(1.0, kreate_mesh::modifiers::array::MAX_COUNT as f32) as u32,
                offset: [at(1, 2.5), at(2, 0.0), at(3, 0.0)],
            },
            "subdivide" => GeoNodeKind::Subdivide {
                levels: at(0, 1.0).clamp(0.0, 4.0) as u32,
                smooth: at(1, 1.0) >= 0.5,
            },
            "mirror" => GeoNodeKind::Mirror {
                axis: at(0, 0.0).clamp(0.0, 2.0) as u32,
            },
            "join" => GeoNodeKind::Join,
            "primitive" => GeoNodeKind::Primitive {
                shape: at(0, 0.0).clamp(0.0, (PRIMITIVES.len() - 1) as f32) as u32,
                size: at(1, 1.0).max(0.001),
            },
            _ => return None,
        })
    }

    /// The editable numbers: a label key's tail, the value, its range, and
    /// whether it picks from a list.
    pub fn fields(&self) -> Vec<GeoField> {
        let v = self.values();
        let number = |name: &'static str, i: usize, min: f32, max: f32| GeoField {
            name,
            index: i as u32,
            value: v[i],
            min,
            max,
            choice: None,
        };
        match self {
            GeoNodeKind::Input | GeoNodeKind::Output | GeoNodeKind::Join => Vec::new(),
            GeoNodeKind::Transform { .. } => vec![
                number("translate-x", 0, -1000.0, 1000.0),
                number("translate-y", 1, -1000.0, 1000.0),
                number("translate-z", 2, -1000.0, 1000.0),
                number("rotate-x", 3, -360.0, 360.0),
                number("rotate-y", 4, -360.0, 360.0),
                number("rotate-z", 5, -360.0, 360.0),
                number("scale-x", 6, -100.0, 100.0),
                number("scale-y", 7, -100.0, 100.0),
                number("scale-z", 8, -100.0, 100.0),
            ],
            GeoNodeKind::Array { .. } => vec![
                number("count", 0, 1.0, 100.0),
                number("offset-x", 1, -100.0, 100.0),
                number("offset-y", 2, -100.0, 100.0),
                number("offset-z", 3, -100.0, 100.0),
            ],
            GeoNodeKind::Subdivide { .. } => vec![
                number("levels", 0, 0.0, 4.0),
                GeoField {
                    choice: Some(&["off", "on"]),
                    ..number("smooth", 1, 0.0, 1.0)
                },
            ],
            GeoNodeKind::Mirror { .. } => vec![GeoField {
                choice: Some(&["x", "y", "z"]),
                ..number("axis", 0, 0.0, 2.0)
            }],
            GeoNodeKind::Primitive { .. } => vec![
                GeoField {
                    choice: Some(&PRIMITIVES),
                    ..number("shape", 0, 0.0, 5.0)
                },
                number("size", 1, 0.001, 1000.0),
            ],
        }
    }

    /// The same kind with number `index` set to `value`, kept in range.
    pub fn with_value(&self, index: u32, value: f32) -> Option<Self> {
        let mut values = self.values();
        *values.get_mut(index as usize)? = value;
        GeoNodeKind::from_parts(self.tag(), &values)
    }
}

/// One editable number on a node.
#[derive(Clone, Debug, PartialEq)]
pub struct GeoField {
    /// The tail of its label's catalogue key: `geo-field-<name>`.
    pub name: &'static str,
    /// Its place in [`GeoNodeKind::values`].
    pub index: u32,
    pub value: f32,
    pub min: f32,
    pub max: f32,
    /// For a pick from a list: the options' key tails, `geo-option-<name>`.
    pub choice: Option<&'static [&'static str]>,
}

/// One node.
#[derive(Clone, Debug, PartialEq)]
pub struct GeoNode {
    pub name: String,
    pub kind: GeoNodeKind,
    /// Where it sits in the editor.
    pub position: [f32; 2],
}

/// A wire: `from_node`'s geometry into `to_node`'s input `to_socket`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeoLink {
    pub from_node: usize,
    pub to_node: usize,
    pub to_socket: usize,
}

/// Why a wire was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GeoLinkError {
    NoSuchSocket,
    /// It would make the graph loop.
    Cycle,
}

/// A geometry graph.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GeoGraph {
    pub nodes: Vec<GeoNode>,
    pub links: Vec<GeoLink>,
}

impl GeoGraph {
    /// Group Input straight into Group Output: the mesh as it was, which is
    /// what turning geometry nodes on must look like before anything is added.
    pub fn passthrough() -> Self {
        Self {
            nodes: vec![
                GeoNode {
                    name: "Group Input".into(),
                    kind: GeoNodeKind::Input,
                    position: [-300.0, 0.0],
                },
                GeoNode {
                    name: "Group Output".into(),
                    kind: GeoNodeKind::Output,
                    position: [300.0, 0.0],
                },
            ],
            links: vec![GeoLink {
                from_node: 0,
                to_node: 1,
                to_socket: 0,
            }],
        }
    }

    /// The wire into `node`'s input `socket`, if one is there.
    pub fn incoming(&self, node: usize, socket: usize) -> Option<&GeoLink> {
        self.links
            .iter()
            .find(|l| l.to_node == node && l.to_socket == socket)
    }

    /// Wires `from` into `to`'s input `socket`, replacing what was there.
    pub fn link(&mut self, from: usize, to: usize, socket: usize) -> Result<(), GeoLinkError> {
        let (Some(f), Some(t)) = (self.nodes.get(from), self.nodes.get(to)) else {
            return Err(GeoLinkError::NoSuchSocket);
        };
        if !f.kind.has_output() || socket >= t.kind.inputs() {
            return Err(GeoLinkError::NoSuchSocket);
        }
        if from == to || self.reaches(to, from) {
            return Err(GeoLinkError::Cycle);
        }
        self.links
            .retain(|l| !(l.to_node == to && l.to_socket == socket));
        self.links.push(GeoLink {
            from_node: from,
            to_node: to,
            to_socket: socket,
        });
        Ok(())
    }

    /// Whether `from`'s output can reach `to` along the wires.
    fn reaches(&self, from: usize, to: usize) -> bool {
        let mut stack = vec![from];
        let mut seen = vec![false; self.nodes.len()];
        while let Some(n) = stack.pop() {
            if n == to {
                return true;
            }
            if std::mem::replace(&mut seen[n], true) {
                continue;
            }
            stack.extend(self.links.iter().filter(|l| l.from_node == n).map(|l| l.to_node));
        }
        false
    }

    /// Removes nodes — never the group's input or output — and their wires,
    /// renumbering what is left.
    pub fn remove(&mut self, doomed: &[usize]) -> bool {
        let doomed: Vec<usize> = doomed
            .iter()
            .copied()
            .filter(|n| {
                self.nodes
                    .get(*n)
                    .is_some_and(|node| !matches!(node.kind, GeoNodeKind::Input | GeoNodeKind::Output))
            })
            .collect();
        if doomed.is_empty() {
            return false;
        }
        let mut map = vec![None; self.nodes.len()];
        let mut kept = Vec::new();
        for (i, n) in self.nodes.drain(..).enumerate() {
            if !doomed.contains(&i) {
                map[i] = Some(kept.len());
                kept.push(n);
            }
        }
        self.nodes = kept;
        self.links = self
            .links
            .iter()
            .filter_map(|l| {
                Some(GeoLink {
                    from_node: map[l.from_node]?,
                    to_node: map[l.to_node]?,
                    to_socket: l.to_socket,
                })
            })
            .collect();
        true
    }

    /// What the graph makes of `input`, or `None` when it has no output.
    pub fn evaluate(&self, input: &MeshData) -> Option<MeshData> {
        let output = self
            .nodes
            .iter()
            .position(|n| matches!(n.kind, GeoNodeKind::Output))?;
        let mut memo: Vec<Option<MeshData>> = vec![None; self.nodes.len()];
        Some(self.socket(output, 0, input, &mut memo, 0))
    }

    /// What arrives at `node`'s input `socket`: the wired node's result, or
    /// nothing.
    fn socket(
        &self,
        node: usize,
        socket: usize,
        input: &MeshData,
        memo: &mut Vec<Option<MeshData>>,
        depth: usize,
    ) -> MeshData {
        match self.incoming(node, socket) {
            Some(l) if depth <= self.nodes.len() => self.node(l.from_node, input, memo, depth + 1),
            _ => MeshData::empty(),
        }
    }

    fn node(
        &self,
        n: usize,
        input: &MeshData,
        memo: &mut Vec<Option<MeshData>>,
        depth: usize,
    ) -> MeshData {
        if let Some(done) = &memo[n] {
            return done.clone();
        }
        let out = match &self.nodes[n].kind {
            GeoNodeKind::Input => input.clone(),
            GeoNodeKind::Output => self.socket(n, 0, input, memo, depth),
            GeoNodeKind::Transform {
                translate,
                rotate,
                scale,
            } => {
                let mut m = self.socket(n, 0, input, memo, depth);
                let to = DAffine3::from_scale_rotation_translation(
                    Vec3::from_array(*scale).as_dvec3(),
                    DQuat::from_euler(
                        kreate_math::glam::EulerRot::ZYX,
                        (rotate[2] as f64).to_radians(),
                        (rotate[1] as f64).to_radians(),
                        (rotate[0] as f64).to_radians(),
                    ),
                    DVec3::from_array(translate.map(|v| v as f64)),
                );
                for p in m.positions_mut() {
                    *p = to.transform_point3(p.as_dvec3()).as_vec3();
                }
                m
            }
            GeoNodeKind::Array { count, offset } => {
                let m = self.socket(n, 0, input, memo, depth);
                array(
                    &m,
                    &ArrayParams {
                        count: *count,
                        relative: Vec3::ZERO,
                        constant: Vec3::from_array(*offset),
                    },
                )
            }
            GeoNodeKind::Subdivide { levels, smooth } => {
                let m = self.socket(n, 0, input, memo, depth);
                subdivide(
                    &m,
                    &SubdivideParams {
                        levels: *levels,
                        smooth: *smooth,
                    },
                )
            }
            GeoNodeKind::Mirror { axis } => {
                let m = self.socket(n, 0, input, memo, depth);
                mirror(
                    &m,
                    &MirrorParams {
                        x: *axis == 0,
                        y: *axis == 1,
                        z: *axis == 2,
                        ..Default::default()
                    },
                )
            }
            GeoNodeKind::Join => {
                let a = self.socket(n, 0, input, memo, depth);
                let b = self.socket(n, 1, input, memo, depth);
                let parts: Vec<Part<'_>> = [&a, &b]
                    .into_iter()
                    .filter(|m| m.num_verts() > 0)
                    .map(|mesh| Part {
                        mesh,
                        transform: DAffine3::IDENTITY,
                    })
                    .collect();
                join(&parts).0
            }
            GeoNodeKind::Primitive { shape, size } => {
                use kreate_mesh::primitives as p;
                match shape {
                    1 => p::uv_sphere(*size * 0.5, 24, 12),
                    2 => p::cylinder(*size * 0.5, *size, 24),
                    3 => p::cone(*size * 0.5, *size, 24),
                    4 => p::torus(*size * 0.5, *size * 0.15, 32, 12),
                    5 => p::ico_sphere(*size * 0.5, 2),
                    _ => p::cube(*size),
                }
            }
        };
        memo[n] = Some(out.clone());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add(g: &mut GeoGraph, kind: GeoNodeKind) -> usize {
        g.nodes.push(GeoNode {
            name: kind.tag().into(),
            kind,
            position: [0.0, 0.0],
        });
        g.nodes.len() - 1
    }

    #[test]
    fn a_passthrough_gives_the_mesh_back() {
        let cube = kreate_mesh::primitives::cube(2.0);
        let out = GeoGraph::passthrough().evaluate(&cube).unwrap();
        assert_eq!(out.positions(), cube.positions());
        assert_eq!(out.num_faces(), cube.num_faces());
    }

    #[test]
    fn transform_array_and_join_compose() {
        let cube = kreate_mesh::primitives::cube(1.0);
        let mut g = GeoGraph::passthrough();
        // Input → Transform (up 5) → Array ×3 → Join with a fresh cube → Output.
        let t = add(
            &mut g,
            GeoNodeKind::Transform {
                translate: [0.0, 0.0, 5.0],
                rotate: [0.0, 0.0, 90.0],
                scale: [1.0, 1.0, 2.0],
            },
        );
        let a = add(&mut g, GeoNodeKind::Array { count: 3, offset: [2.0, 0.0, 0.0] });
        let p = add(&mut g, GeoNodeKind::Primitive { shape: 0, size: 1.0 });
        let j = add(&mut g, GeoNodeKind::Join);
        g.link(0, t, 0).unwrap();
        g.link(t, a, 0).unwrap();
        g.link(a, j, 0).unwrap();
        g.link(p, j, 1).unwrap();
        g.link(j, 1, 0).unwrap();
        let out = g.evaluate(&cube).unwrap();
        assert_eq!(out.num_faces(), 6 * 3 + 6);
        let top = out.positions().iter().map(|p| p.z).fold(f32::MIN, f32::max);
        assert!((top - 6.0).abs() < 1e-5, "{top}");
        let right = out.positions().iter().map(|p| p.x).fold(f32::MIN, f32::max);
        assert!((right - 4.5).abs() < 1e-5, "{right}");
    }

    #[test]
    fn a_wire_that_would_loop_is_refused_and_removing_keeps_the_group_ends() {
        let mut g = GeoGraph::passthrough();
        let t = add(&mut g, GeoNodeKind::Mirror { axis: 0 });
        let u = add(&mut g, GeoNodeKind::Subdivide { levels: 1, smooth: false });
        g.link(t, u, 0).unwrap();
        assert_eq!(g.link(u, t, 0), Err(GeoLinkError::Cycle));
        assert_eq!(g.link(1, t, 0), Err(GeoLinkError::NoSuchSocket));
        assert!(!g.remove(&[0, 1]));
        assert!(g.remove(&[t]));
        assert_eq!(g.nodes.len(), 3);
        assert!(g.links.iter().all(|l| l.from_node < 3 && l.to_node < 3));
    }

    #[test]
    fn every_kind_round_trips_through_its_numbers() {
        for tag in GeoNodeKind::ADDABLE.iter().chain(&["input", "output"]) {
            let kind = GeoNodeKind::from_parts(tag, &[]).unwrap();
            assert_eq!(GeoNodeKind::from_parts(kind.tag(), &kind.values()), Some(kind.clone()));
            for f in kind.fields() {
                assert!(kind.with_value(f.index, f.value).is_some());
            }
        }
    }
}
