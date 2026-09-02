//! The `BMesh` validator.
//!
//! The highest-return item in the whole plan, and the cheapest. A mesh kernel
//! fails by degrees: an operator leaves a subtly invalid cycle, three operators
//! later something crashes or the data is quietly wrong, and by then the trail
//! back to the cause is gone. Every failed modelling application has died this
//! way.
//!
//! So this runs after **every** operator in debug builds, and it reports rather
//! than panics, so a caller can decide whether a given intermediate state is
//! legitimately half-built.
//!
//! It must be safe on a *corrupt* mesh, which is the only kind it will ever be
//! asked about: every cycle walk here is bounded by an arena's slot count rather
//! than by "walk until you return to the start".

use kreate_math::Vec3;

use crate::arena::{ArenaIndex, EdgeId, FaceId, LoopId, VertId};
use crate::attribute::{names, Domain};

use super::BMesh;

/// A structural problem found by [`BMesh::validate`].
#[derive(Clone, Debug, PartialEq)]
pub enum BmViolation {
    /// A face's loop cycle does not return to its start in `len` steps.
    FaceCycleNotClosed { face: FaceId, len: u32 },
    /// The loop cycle's real length differs from the recorded one.
    FaceLengthMismatch {
        face: FaceId,
        declared: u32,
        walked: usize,
    },
    /// A face has fewer than three corners.
    FaceTooShort { face: FaceId, len: u32 },
    /// A loop in a face's cycle claims to belong to another face.
    LoopFaceMismatch {
        loop_: LoopId,
        expected: FaceId,
        found: FaceId,
    },
    /// `next.prev` does not lead back, so the loop cycle is not a valid doubly
    /// linked list.
    LoopCycleBroken { loop_: LoopId },
    /// `radial_next.radial_prev` does not lead back.
    RadialCycleBroken { loop_: LoopId },
    /// A loop in an edge's radial cycle references a different edge.
    RadialEdgeMismatch {
        loop_: LoopId,
        expected: EdgeId,
        found: EdgeId,
    },
    /// A loop's edge does not join the loop's vertex to the next loop's vertex.
    LoopEdgeEndpointsMismatch {
        loop_: LoopId,
        edge: EdgeId,
        from: VertId,
        to: VertId,
    },
    /// A disk cycle does not return to its start.
    DiskCycleNotClosed { vert: VertId },
    /// `disk.prev`/`disk.next` do not lead back to each other.
    DiskCycleBroken { vert: VertId, edge: EdgeId },
    /// An edge in a vertex's disk cycle does not have that vertex as an endpoint.
    DiskEdgeNotIncident { vert: VertId, edge: EdgeId },
    /// An edge joins a vertex to itself.
    SelfEdge { edge: EdgeId, vert: VertId },
    /// Two edges join the same pair of vertices, which makes every disk walk
    /// ambiguous.
    DuplicateEdge { edge: EdgeId, other: EdgeId },
    /// A live loop that no face's cycle reaches — leaked by a failed removal.
    OrphanLoop { loop_: LoopId },
    /// A handle that no longer refers to a live element.
    DanglingReference { from: String, to: String },
    /// An attribute layer's length does not match its domain's slot count.
    AttrLengthMismatch {
        layer: String,
        domain: Domain,
        len: usize,
        expected: usize,
    },
    /// A position is NaN or infinite. Cheap to check and it poisons everything
    /// downstream — bounds, normals, the BVH, the camera framing.
    NonFinitePosition { vert: VertId, value: Vec3 },
}

impl BMesh {
    /// Checks every structural invariant, returning what is wrong.
    ///
    /// Empty means the mesh is sound. Ordering is deterministic — arena slot
    /// order — so a golden test over the output is stable.
    pub fn validate(&self) -> Vec<BmViolation> {
        let mut out = Vec::new();
        self.check_faces(&mut out);
        self.check_edges(&mut out);
        self.check_verts(&mut out);
        self.check_orphan_loops(&mut out);
        self.check_attributes(&mut out);
        out
    }

    /// Panics with a readable report if anything is wrong.
    ///
    /// The form an operator's debug-build assertion takes.
    #[track_caller]
    pub fn assert_valid(&self) {
        let violations = self.validate();
        assert!(
            violations.is_empty(),
            "BMesh is invalid ({} problem(s)):\n{}",
            violations.len(),
            violations
                .iter()
                .map(|v| format!("  - {v:?}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    fn check_faces(&self, out: &mut Vec<BmViolation>) {
        for (f, face) in self.faces.iter() {
            if face.len < 3 {
                out.push(BmViolation::FaceTooShort {
                    face: f,
                    len: face.len,
                });
            }

            // Bounded by the loop arena rather than by the declared length, so
            // that a cycle which closes at the wrong place is reported as a
            // length mismatch rather than as a broken cycle. The two are
            // different bugs and the distinction is most of the diagnosis.
            let mut l = face.first_loop;
            let mut walked = 0usize;
            let limit = self.loops.slot_count() + 1;
            let mut closed = false;
            while walked < limit {
                let Some(node) = self.loops.get(l) else {
                    out.push(BmViolation::DanglingReference {
                        from: format!("{f:?}"),
                        to: format!("{l:?}"),
                    });
                    break;
                };
                if node.face != f {
                    out.push(BmViolation::LoopFaceMismatch {
                        loop_: l,
                        expected: f,
                        found: node.face,
                    });
                }
                match self.loops.get(node.next) {
                    Some(next) if next.prev == l => {}
                    _ => out.push(BmViolation::LoopCycleBroken { loop_: l }),
                }
                self.check_loop_edge(l, out);

                walked += 1;
                l = node.next;
                if l == face.first_loop {
                    closed = true;
                    break;
                }
            }

            if !closed {
                out.push(BmViolation::FaceCycleNotClosed {
                    face: f,
                    len: face.len,
                });
            } else if walked != face.len as usize {
                out.push(BmViolation::FaceLengthMismatch {
                    face: f,
                    declared: face.len,
                    walked,
                });
            }
        }
    }

    /// A loop's edge must join its vertex to the next loop's vertex.
    ///
    /// The invariant that ties the loop cycle to the edge list. When it breaks,
    /// the face still draws correctly and the wireframe does not, which is a
    /// deeply confusing way to find out.
    fn check_loop_edge(&self, l: LoopId, out: &mut Vec<BmViolation>) {
        let Some(node) = self.loops.get(l) else {
            return;
        };
        let Some(next) = self.loops.get(node.next) else {
            return;
        };
        let Some(edge) = self.edges.get(node.edge) else {
            out.push(BmViolation::DanglingReference {
                from: format!("{l:?}"),
                to: format!("{:?}", node.edge),
            });
            return;
        };
        let want = [node.vert, next.vert];
        if edge.verts != want && edge.verts != [want[1], want[0]] {
            out.push(BmViolation::LoopEdgeEndpointsMismatch {
                loop_: l,
                edge: node.edge,
                from: node.vert,
                to: next.vert,
            });
        }
    }

    fn check_edges(&self, out: &mut Vec<BmViolation>) {
        // Duplicate detection over normalised endpoint pairs. Sorted rather than
        // hashed so the reported pair is the same on every run.
        let mut pairs: Vec<([u32; 2], EdgeId)> = Vec::with_capacity(self.edges.len());

        for (e, edge) in self.edges.iter() {
            if edge.verts[0] == edge.verts[1] {
                out.push(BmViolation::SelfEdge {
                    edge: e,
                    vert: edge.verts[0],
                });
            }
            for v in edge.verts {
                if !self.verts.contains(v) {
                    out.push(BmViolation::DanglingReference {
                        from: format!("{e:?}"),
                        to: format!("{v:?}"),
                    });
                }
            }
            let mut key = [edge.verts[0].index(), edge.verts[1].index()];
            key.sort_unstable();
            pairs.push((key, e));

            // The radial cycle.
            let Some(first) = edge.first_loop else {
                continue;
            };
            let mut l = first;
            let limit = self.loops.slot_count() + 1;
            for _ in 0..limit {
                let Some(node) = self.loops.get(l) else {
                    out.push(BmViolation::DanglingReference {
                        from: format!("{e:?}"),
                        to: format!("{l:?}"),
                    });
                    break;
                };
                if node.edge != e {
                    out.push(BmViolation::RadialEdgeMismatch {
                        loop_: l,
                        expected: e,
                        found: node.edge,
                    });
                }
                match self.loops.get(node.radial_next) {
                    Some(next) if next.radial_prev == l => {}
                    _ => out.push(BmViolation::RadialCycleBroken { loop_: l }),
                }
                l = node.radial_next;
                if l == first {
                    break;
                }
            }
        }

        pairs.sort_unstable();
        for w in pairs.windows(2) {
            if w[0].0 == w[1].0 {
                out.push(BmViolation::DuplicateEdge {
                    edge: w[0].1,
                    other: w[1].1,
                });
            }
        }
    }

    fn check_verts(&self, out: &mut Vec<BmViolation>) {
        for (v, vert) in self.verts.iter() {
            if !vert.co.is_finite() {
                out.push(BmViolation::NonFinitePosition {
                    vert: v,
                    value: vert.co,
                });
            }
            let Some(first) = vert.edge else {
                continue;
            };
            let mut e = first;
            let limit = self.edges.slot_count() + 1;
            let mut closed = false;
            for _ in 0..limit {
                let Some(edge) = self.edges.get(e) else {
                    out.push(BmViolation::DanglingReference {
                        from: format!("{v:?}"),
                        to: format!("{e:?}"),
                    });
                    break;
                };
                let Some(i) = self.disk_index(e, v) else {
                    out.push(BmViolation::DiskEdgeNotIncident { vert: v, edge: e });
                    break;
                };
                let link = edge.disk[i];
                let back_ok = self
                    .disk_index(link.next, v)
                    .and_then(|j| self.edges.get(link.next).map(|n| n.disk[j].prev == e))
                    .unwrap_or(false);
                if !back_ok {
                    out.push(BmViolation::DiskCycleBroken { vert: v, edge: e });
                }
                e = link.next;
                if e == first {
                    closed = true;
                    break;
                }
            }
            if !closed {
                out.push(BmViolation::DiskCycleNotClosed { vert: v });
            }
        }
    }

    fn check_orphan_loops(&self, out: &mut Vec<BmViolation>) {
        let mut reached = vec![false; self.loops.slot_count()];
        for f in self.faces.ids() {
            for l in self.face_loops(f) {
                if let Some(slot) = reached.get_mut(l.index() as usize) {
                    *slot = true;
                }
            }
        }
        for l in self.loops.ids() {
            if !reached[l.index() as usize] {
                out.push(BmViolation::OrphanLoop { loop_: l });
            }
        }
    }

    fn check_attributes(&self, out: &mut Vec<BmViolation>) {
        for layer in self.attributes.iter() {
            let expected = self.slot_count(layer.domain);
            if layer.data.len() != expected {
                out.push(BmViolation::AttrLengthMismatch {
                    layer: layer.name.to_string(),
                    domain: layer.domain,
                    len: layer.data.len(),
                    expected,
                });
            }
        }
        // Position lives inline in `Vert`, so a layer by that name is not merely
        // redundant: whichever copy an operator happens to read decides the
        // answer, and the two silently diverge.
        if self.attributes.contains(names::POSITION) {
            out.push(BmViolation::DanglingReference {
                from: "attributes".into(),
                to: format!("{} (positions live in Vert::co)", names::POSITION),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attribute::{AttrData, AttrType};
    use std::sync::Arc;

    fn quad() -> BMesh {
        let mut m = BMesh::new();
        let v: Vec<VertId> = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ]
        .iter()
        .map(|p| m.add_vert(*p))
        .collect();
        m.add_face(&v).unwrap();
        m
    }

    #[test]
    fn a_well_formed_mesh_reports_nothing() {
        assert_eq!(quad().validate(), Vec::new());
        assert_eq!(
            BMesh::from_mesh_data(&crate::primitives::uv_sphere(1.0, 12, 6)).validate(),
            Vec::new()
        );
    }

    // Each of the following corrupts one field directly, which is the only way
    // to reach these states — the public API cannot produce them. That is the
    // point: the validator has to catch what a *buggy operator* would do, and a
    // test that can only build valid meshes proves nothing about it.

    #[test]
    fn a_broken_loop_cycle_is_caught() {
        let mut m = quad();
        let f = m.face_ids().next().unwrap();
        let l = m.faces[f].first_loop;
        let next = m.loops[l].next;
        m.loops[l].next = l;
        assert!(
            m.validate()
                .iter()
                .any(|v| matches!(v, BmViolation::LoopCycleBroken { .. })),
            "got {:?}",
            m.validate()
        );
        m.loops[l].next = next;
        assert_eq!(m.validate(), Vec::new(), "and clean again once repaired");
    }

    #[test]
    fn a_wrong_face_length_is_caught() {
        let mut m = quad();
        let f = m.face_ids().next().unwrap();
        m.faces[f].len = 3;
        let found = m.validate();
        assert!(
            found.contains(&BmViolation::FaceLengthMismatch {
                face: f,
                declared: 3,
                walked: 4,
            }),
            "the cycle still closes, just not where the face says it does; got {found:?}"
        );
    }

    #[test]
    fn a_loop_pointing_at_the_wrong_face_is_caught() {
        let mut m = quad();
        let v = m.add_vert(Vec3::Z);
        let w = m.add_vert(Vec3::new(1.0, 0.0, 1.0));
        let x = m.add_vert(Vec3::new(0.0, 1.0, 1.0));
        let other = m.add_face(&[v, w, x]).unwrap();
        let f = m.face_ids().next().unwrap();
        let l = m.faces[f].first_loop;
        m.loops[l].face = other;
        assert!(m
            .validate()
            .iter()
            .any(|v| matches!(v, BmViolation::LoopFaceMismatch { .. })));
    }

    #[test]
    fn a_radial_cycle_that_does_not_lead_back_is_caught() {
        let mut m = quad();
        let f = m.face_ids().next().unwrap();
        let l = m.faces[f].first_loop;
        m.loops[l].radial_next = m.loops[l].next;
        let found = m.validate();
        assert!(
            found
                .iter()
                .any(|v| matches!(v, BmViolation::RadialCycleBroken { .. }))
                || found
                    .iter()
                    .any(|v| matches!(v, BmViolation::RadialEdgeMismatch { .. })),
            "got {found:?}"
        );
    }

    #[test]
    fn a_loop_whose_edge_joins_the_wrong_vertices_is_caught() {
        let mut m = quad();
        let f = m.face_ids().next().unwrap();
        let l = m.faces[f].first_loop;
        let opposite = m.loops[m.loops[l].next].next;
        m.loops[l].edge = m.loops[opposite].edge;
        assert!(m
            .validate()
            .iter()
            .any(|v| matches!(v, BmViolation::LoopEdgeEndpointsMismatch { .. })));
    }

    #[test]
    fn a_broken_disk_cycle_is_caught() {
        let mut m = quad();
        let v = m.vert_ids().next().unwrap();
        let e = m.vert_edges(v)[0];
        let i = m.disk_index(e, v).unwrap();
        m.edges[e].disk[i].next = e;
        let found = m.validate();
        assert!(
            found
                .iter()
                .any(|x| matches!(x, BmViolation::DiskCycleBroken { .. })),
            "got {found:?}"
        );
    }

    #[test]
    fn an_orphaned_loop_is_caught() {
        let mut m = quad();
        let f = m.face_ids().next().unwrap();
        // Drop the face without unlinking its loops: exactly what a removal that
        // forgets a step leaves behind.
        m.faces.remove(f);
        let found = m.validate();
        assert_eq!(
            found
                .iter()
                .filter(|v| matches!(v, BmViolation::OrphanLoop { .. }))
                .count(),
            4,
            "got {found:?}"
        );
    }

    #[test]
    fn a_non_finite_position_is_caught() {
        let mut m = quad();
        let v = m.vert_ids().next().unwrap();
        m.verts[v].co = Vec3::new(f32::NAN, 0.0, 0.0);
        assert!(m
            .validate()
            .iter()
            .any(|x| matches!(x, BmViolation::NonFinitePosition { .. })));
    }

    #[test]
    fn a_desynchronised_attribute_layer_is_caught() {
        let mut m = quad();
        m.attributes_mut().insert(
            "uv",
            Domain::Corner,
            AttrData::F32x2(Arc::new(vec![kreate_math::Vec2::ZERO; 2])),
        );
        let found = m.validate();
        assert!(
            found.iter().any(|v| matches!(
                v,
                BmViolation::AttrLengthMismatch {
                    domain: Domain::Corner,
                    len: 2,
                    expected: 4,
                    ..
                }
            )),
            "got {found:?}"
        );
    }

    #[test]
    fn a_shadow_position_layer_is_caught() {
        let mut m = quad();
        let points = m.slot_count(Domain::Point);
        m.attributes_mut()
            .ensure(names::POSITION, Domain::Point, AttrType::F32x3, points);
        assert!(
            !m.validate().is_empty(),
            "two copies of position must be an error"
        );
    }

    #[test]
    fn a_duplicate_edge_is_caught() {
        let mut m = quad();
        let vs: Vec<VertId> = m.vert_ids().collect();
        // Slipped past `add_edge`'s find-or-create, which is what a bad splice
        // inside an operator amounts to.
        let placeholder = m.edges[m.vert_edges(vs[0])[0]];
        let dup = m.edges.insert(placeholder);
        let edges = m.slot_count(Domain::Edge);
        m.attributes_mut().resize_domain(Domain::Edge, edges);
        let found = m.validate();
        assert!(
            found
                .iter()
                .any(|v| matches!(v, BmViolation::DuplicateEdge { .. })),
            "inserted {dup:?}, got {found:?}"
        );
    }

    #[test]
    fn the_validator_terminates_on_a_cycle_that_never_closes() {
        // The validator's own failure mode. A corrupt mesh is the only kind it
        // ever runs on, so "walk until you return to the start" would hang the
        // editor at exactly the moment the user most needs it to say something.
        //
        // The disk here is a lasso: it runs into a sub-cycle that never comes
        // back to the entry point.
        let mut m = BMesh::new();
        let hub = m.add_vert(Vec3::ZERO);
        for i in 0..3 {
            let s = m.add_vert(Vec3::new(i as f32, 1.0, 0.0));
            m.add_edge(hub, s).unwrap();
        }
        let e: Vec<EdgeId> = m.vert_edges(hub);
        assert_eq!(e.len(), 3);
        let last = e[2];
        let middle = e[1];
        let j = m.disk_index(last, hub).unwrap();
        m.edges[last].disk[j].next = middle;

        let found = m.validate();
        assert!(
            found.contains(&BmViolation::DiskCycleNotClosed { vert: hub }),
            "got {found:?}"
        );
    }
}
