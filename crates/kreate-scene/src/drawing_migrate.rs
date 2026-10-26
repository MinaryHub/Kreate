//! A sketch of format 1.4 as a drawing.
//!
//! A sketch was points (its mesh's vertices), lines between them, and
//! constraints on the points by their place in the list — a circle among
//! them, as a ring of points held round a centre. A drawing is curves with
//! ids. This takes one to the other:
//!
//! - a circle's, an arc's or a spline's ring becomes that curve, exactly,
//!   and the short lines along the ring go with it;
//! - a polygon becomes its sides, a construction circle through its
//!   corners and the sides held equal — the regular polygon Fusion draws;
//! - every other line is a line, and a point two things share is one of
//!   them, the rest held to it as coincident;
//! - a point nothing draws through stays a point;
//! - each constraint is the same relation on what its points became, a
//!   distance from a centre to its ring is the circle's radius, and a
//!   distance or an angle is a driving dimension.
//!
//! What it cannot place — a constraint on a line the sketch never drew —
//! is dropped and counted, never guessed at.

use crate::drawing::{
    Constraint, DimKind, Dimension, Drawing, EntityId, EntityKind, EntityRef, Part,
};
use crate::sketch::Constraint as Old;
use kreate_geom2d::{Curve2D, DVec2, Spline};
use std::collections::BTreeMap;

/// The drawing, and the old constraints that could not be carried.
#[derive(Clone, Debug)]
pub struct Migrated {
    pub drawing: Drawing,
    /// The old constraints, by their place, that could not be carried.
    pub dropped: Vec<u32>,
    /// Where each old point went, for the features and bindings that named
    /// them.
    pub points: Vec<Option<EntityRef>>,
    /// Where each old constraint went: the dimension a distance or angle
    /// became, for the parameters bound to it by its place.
    pub dimensions: BTreeMap<u32, EntityId>,
}

/// The sketch whose points are `points`, lines `lines` and constraints
/// `old`, as a drawing.
pub fn drawing_from_sketch(points: &[DVec2], lines: &[[u32; 2]], old: &[Old]) -> Migrated {
    let mut d = Drawing::new();
    let mut owners: Vec<Vec<EntityRef>> = vec![Vec::new(); points.len()];
    let p = |i: u32| points.get(i as usize).copied().unwrap_or(DVec2::ZERO);
    // The short lines along a ring, absorbed into its curve.
    let mut absorbed: std::collections::BTreeSet<(u32, u32)> = Default::default();
    let key = |a: u32, b: u32| (a.min(b), a.max(b));
    let absorb_chain =
        |chain: &[u32], closed: bool, absorbed: &mut std::collections::BTreeSet<(u32, u32)>| {
            for w in chain.windows(2) {
                absorbed.insert(key(w[0], w[1]));
            }
            if closed && chain.len() > 2 {
                absorbed.insert(key(chain[chain.len() - 1], chain[0]));
            }
        };
    // The curve each centre belongs to, for tangents and radii.
    let mut centre_of: BTreeMap<u32, EntityId> = BTreeMap::new();
    let mut ring_of: BTreeMap<u32, EntityId> = BTreeMap::new();
    let mut construction_lines: std::collections::BTreeSet<(u32, u32)> = Default::default();
    for c in old {
        if let Old::Construction { a, b } = c {
            construction_lines.insert(key(*a, *b));
        }
    }
    // Curves first: their rings decide which lines are theirs.
    let mut polygons: Vec<(u32, Vec<u32>)> = Vec::new();
    for c in old {
        match c {
            Old::Circle { centre, ring } if !ring.is_empty() => {
                let r = p(*centre).distance(p(ring[0]));
                let id = d.add_curve(Curve2D::Circle { c: p(*centre), r });
                owners[*centre as usize].push(EntityRef::at(id, Part::Center));
                for q in ring {
                    ring_of.insert(*q, id);
                }
                centre_of.insert(*centre, id);
                absorb_chain(ring, true, &mut absorbed);
            }
            Old::Arc { centre, ring } if ring.len() >= 2 => {
                let c0 = p(*centre);
                let (first, last) = (ring[0], ring[ring.len() - 1]);
                let r = c0.distance(p(first));
                let start = (p(first) - c0).to_angle();
                let end = (p(last) - c0).to_angle();
                let sweep = (end - start).rem_euclid(std::f64::consts::TAU);
                let id = d.add_curve(Curve2D::Arc {
                    c: c0,
                    r,
                    start,
                    sweep,
                });
                owners[*centre as usize].push(EntityRef::at(id, Part::Center));
                owners[first as usize].push(EntityRef::at(id, Part::Start));
                owners[last as usize].push(EntityRef::at(id, Part::End));
                for q in &ring[1..ring.len() - 1] {
                    ring_of.insert(*q, id);
                }
                centre_of.insert(*centre, id);
                absorb_chain(ring, false, &mut absorbed);
            }
            Old::Spline {
                through,
                ring,
                closed,
            } if through.len() >= 2 => {
                let fit: Vec<DVec2> = through.iter().map(|i| p(*i)).collect();
                let id = d.add_curve(Curve2D::Spline(Spline::catmull_rom(&fit, *closed)));
                for (k, q) in through.iter().enumerate() {
                    owners[*q as usize].push(EntityRef::at(id, Part::Vertex(k as u32)));
                }
                for q in ring {
                    ring_of.insert(*q, id);
                }
                // The chain the sketch drew: each fit point, then the ring
                // points of the span after it.
                let spans = if *closed {
                    through.len()
                } else {
                    through.len() - 1
                };
                let per = ring.len().checked_div(spans).unwrap_or(0);
                let mut chain = Vec::new();
                for (k, t) in through.iter().enumerate() {
                    chain.push(*t);
                    if k < spans {
                        chain.extend(&ring[k * per..((k + 1) * per).min(ring.len())]);
                    }
                }
                absorb_chain(&chain, *closed, &mut absorbed);
            }
            Old::Polygon { centre, ring } if ring.len() >= 3 => {
                polygons.push((*centre, ring.clone()));
            }
            _ => {}
        }
    }
    // Lines: every line not along a ring.
    let mut line_of: BTreeMap<(u32, u32), (EntityId, u32)> = BTreeMap::new();
    for [a, b] in lines {
        if a == b || absorbed.contains(&key(*a, *b)) || line_of.contains_key(&key(*a, *b)) {
            continue;
        }
        let id = d.add_curve(Curve2D::Line { a: p(*a), b: p(*b) });
        if construction_lines.contains(&key(*a, *b)) {
            if let Some(e) = d.entity_mut(id) {
                e.construction = true;
            }
        }
        owners[*a as usize].push(EntityRef::at(id, Part::Start));
        owners[*b as usize].push(EntityRef::at(id, Part::End));
        line_of.insert(key(*a, *b), (id, *a));
    }
    // Polygons: their sides are lines already; a construction circle
    // through the corners and the sides equal keep them regular.
    for (centre, ring) in &polygons {
        let r = p(*centre).distance(p(ring[0]));
        let circle = d.add_curve(Curve2D::Circle { c: p(*centre), r });
        if let Some(e) = d.entity_mut(circle) {
            e.construction = true;
        }
        owners[*centre as usize].push(EntityRef::at(circle, Part::Center));
        centre_of.insert(*centre, circle);
        let sides: Vec<EntityId> = (0..ring.len())
            .filter_map(|k| {
                line_of
                    .get(&key(ring[k], ring[(k + 1) % ring.len()]))
                    .map(|l| l.0)
            })
            .collect();
        for w in sides.windows(2) {
            d.constraints.push(Constraint::Equal(w[0], w[1]));
        }
        for q in ring {
            if let Some(r) = owners[*q as usize].first().copied() {
                d.constraints.push(Constraint::PointOn(r, circle));
            }
        }
    }
    // Points nothing draws through stay points; points two things share
    // are held together.
    for (i, own) in owners.iter_mut().enumerate() {
        if own.is_empty() && !ring_of.contains_key(&(i as u32)) {
            let id = d.add(EntityKind::Point(points[i]));
            own.push(EntityRef::whole(id));
        }
    }
    for own in &owners {
        for other in own.iter().skip(1) {
            d.constraints.push(Constraint::Coincident(own[0], *other));
        }
    }
    let refer = |i: u32| owners.get(i as usize).and_then(|o| o.first().copied());
    // A line between two points, the way round it was stored.
    let line = |a: u32, b: u32| -> Option<(EntityId, bool)> {
        line_of
            .get(&key(a, b))
            .map(|(id, start)| (*id, *start == a))
    };
    let mut dropped = Vec::new();
    let mut dimensions = BTreeMap::new();
    let dim = |kind: DimKind, refs: Vec<EntityRef>, value: f64, at: DVec2| {
        EntityKind::Dimension(Dimension {
            kind,
            refs,
            text_at: at,
            style: String::new(),
            driving: true,
            value,
            text: None,
            tolerance: None,
        })
    };
    // From a centre to a point of its own curve: that curve, whose radius
    // the distance between them is.
    let radius_of = |x: u32, y: u32| -> Option<EntityId> {
        let curve = *centre_of.get(&x)?;
        let on = ring_of.get(&y).copied() == Some(curve)
            || owners[y as usize].iter().any(|r| r.entity == curve);
        on.then_some(curve)
    };
    let radius_either = |x: u32, y: u32| radius_of(x, y).or_else(|| radius_of(y, x));
    for (index, c) in old.iter().enumerate() {
        let made: Option<()> = (|| {
            match c {
                Old::Circle { .. }
                | Old::Arc { .. }
                | Old::Spline { .. }
                | Old::Polygon { .. }
                | Old::Construction { .. } => {}
                Old::Distance { a, b, value } => {
                    // From a centre to its own ring: the radius.
                    let at = (p(*a) + p(*b)) / 2.0;
                    let kind = if let Some(curve) = radius_of(*a, *b).or_else(|| radius_of(*b, *a))
                    {
                        dim(DimKind::Radius, vec![EntityRef::whole(curve)], *value, at)
                    } else if let Some((l, _)) = line(*a, *b) {
                        dim(DimKind::Aligned, vec![EntityRef::whole(l)], *value, at)
                    } else {
                        dim(DimKind::Aligned, vec![refer(*a)?, refer(*b)?], *value, at)
                    };
                    let id = d.add(kind);
                    dimensions.insert(index as u32, id);
                }
                Old::Horizontal { a, b } | Old::Vertical { a, b } => {
                    let level = matches!(c, Old::Horizontal { .. });
                    let made = match line(*a, *b) {
                        Some((l, _)) => (EntityRef::whole(l), None),
                        None => (refer(*a)?, Some(refer(*b)?)),
                    };
                    d.constraints.push(if level {
                        Constraint::Horizontal(made.0, made.1)
                    } else {
                        Constraint::Vertical(made.0, made.1)
                    });
                }
                Old::Fixed { a, at } => {
                    d.constraints
                        .push(Constraint::Fixed(refer(*a)?, DVec2::new(at[0], at[1])));
                }
                Old::Angle {
                    a,
                    b,
                    c: c3,
                    d: d4,
                    value,
                } => {
                    let (l1, f1) = line(*a, *b)?;
                    let (l2, f2) = line(*c3, *d4)?;
                    // The drawing measures each line start to end; a line
                    // stored the other way round turns the angle by π.
                    let mut v = *value;
                    if !f1 {
                        v += std::f64::consts::PI;
                    }
                    if !f2 {
                        v += std::f64::consts::PI;
                    }
                    let v = (v + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU)
                        - std::f64::consts::PI;
                    let id = d.add(dim(
                        DimKind::Angular,
                        vec![EntityRef::whole(l1), EntityRef::whole(l2)],
                        v,
                        p(*b),
                    ));
                    dimensions.insert(index as u32, id);
                }
                Old::Parallel { a, b, c: c3, d: d4 }
                | Old::Perpendicular { a, b, c: c3, d: d4 }
                | Old::Equal { a, b, c: c3, d: d4 } => {
                    // Two radii, each a centre to its own curve: the curves
                    // the same size.
                    if let (Old::Equal { .. }, Some(k1), Some(k2)) =
                        (c, radius_either(*a, *b), radius_either(*c3, *d4))
                    {
                        d.constraints.push(Constraint::Equal(k1, k2));
                        return Some(());
                    }
                    let (l1, _) = line(*a, *b)?;
                    let (l2, _) = line(*c3, *d4)?;
                    d.constraints.push(match c {
                        Old::Parallel { .. } => Constraint::Parallel(l1, l2),
                        Old::Perpendicular { .. } => Constraint::Perpendicular(l1, l2),
                        _ => Constraint::Equal(l1, l2),
                    });
                }
                Old::Coincident { a, b } => {
                    d.constraints
                        .push(Constraint::Coincident(refer(*a)?, refer(*b)?));
                }
                Old::OnLine { p: q, a, b } => {
                    let (l, _) = line(*a, *b)?;
                    d.constraints.push(Constraint::PointOn(refer(*q)?, l));
                }
                Old::Tangent { a, b, centre, .. } => {
                    let (l, _) = line(*a, *b)?;
                    let curve = *centre_of.get(centre)?;
                    d.constraints.push(Constraint::Tangent(l, curve));
                }
                Old::Symmetric { a, b, axis } => {
                    let (l, _) = line(axis[0], axis[1])?;
                    d.constraints
                        .push(Constraint::Symmetric(refer(*a)?, refer(*b)?, l));
                }
            }
            Some(())
        })();
        if made.is_none() {
            dropped.push(index as u32);
        }
    }
    let points = (0..points.len() as u32).map(refer).collect();
    Migrated {
        drawing: d,
        dropped,
        points,
        dimensions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64, y: f64) -> DVec2 {
        DVec2::new(x, y)
    }

    /// A circle drawn as forty-eight points becomes one circle, and its
    /// forty-eight short lines go with it.
    #[test]
    fn a_ring_of_points_becomes_its_circle() {
        let n = 48;
        let mut points = vec![v(0.02, 0.01)];
        let ring: Vec<u32> = (1..=n).collect();
        for k in 0..n {
            let a = std::f64::consts::TAU * k as f64 / n as f64;
            points.push(v(0.02, 0.01) + DVec2::from_angle(a) * 0.005);
        }
        let lines: Vec<[u32; 2]> = (0..n).map(|k| [1 + k, 1 + (k + 1) % n]).collect();
        let m = drawing_from_sketch(
            &points,
            &lines,
            &[
                Old::Circle {
                    centre: 0,
                    ring: ring.clone(),
                },
                Old::Distance {
                    a: 0,
                    b: 1,
                    value: 0.005,
                },
            ],
        );
        assert!(m.dropped.is_empty(), "{:?}", m.dropped);
        let curves: Vec<_> = m.drawing.profile_curves();
        assert_eq!(curves.len(), 1, "one circle, no lines");
        assert!(matches!(curves[0].1, Curve2D::Circle { r, .. } if (r - 0.005).abs() < 1e-12));
        let radius = m.drawing.entities.iter().find_map(|e| match &e.kind {
            EntityKind::Dimension(d) => Some(d.clone()),
            _ => None,
        });
        assert_eq!(radius.map(|d| d.kind), Some(DimKind::Radius));
        assert!(
            m.dimensions.contains_key(&1),
            "the distance's place is kept"
        );
    }

    /// A rectangle's shared corners are held together, and its sides keep
    /// their level and upright.
    #[test]
    fn a_rectangle_keeps_its_corners_and_its_sides() {
        let points = vec![v(0.0, 0.0), v(0.04, 0.0), v(0.04, 0.02), v(0.0, 0.02)];
        let lines = vec![[0, 1], [1, 2], [2, 3], [3, 0]];
        let old = vec![
            Old::Horizontal { a: 0, b: 1 },
            Old::Vertical { a: 1, b: 2 },
            Old::Horizontal { a: 2, b: 3 },
            Old::Vertical { a: 3, b: 0 },
            Old::Distance {
                a: 0,
                b: 1,
                value: 0.04,
            },
        ];
        let mut m = drawing_from_sketch(&points, &lines, &old);
        assert!(m.dropped.is_empty(), "{:?}", m.dropped);
        let coincident = m
            .drawing
            .constraints
            .iter()
            .filter(|c| matches!(c, Constraint::Coincident(..)))
            .count();
        assert_eq!(coincident, 4, "each corner is two line ends held as one");
        let out = m.drawing.solve();
        assert!(out.iter().all(|p| p.converged), "{out:?}");
        assert_eq!(m.drawing.regions(1e-9).len(), 1);
        assert!((m.drawing.regions(1e-9)[0].area - 0.0008).abs() < 1e-12);
    }

    /// An angle from a line stored the other way round turns by half a
    /// turn, so the drawing holds the same lines where they were.
    #[test]
    fn an_angle_on_a_reversed_line_holds_the_same_lines() {
        let points = vec![v(0.0, 0.0), v(0.03, 0.0), v(0.0, 0.03)];
        // The second line stored from its far end.
        let lines = vec![[0, 1], [2, 0]];
        let old = vec![Old::Angle {
            a: 0,
            b: 1,
            c: 0,
            d: 2,
            value: std::f64::consts::FRAC_PI_2,
        }];
        let mut m = drawing_from_sketch(&points, &lines, &old);
        assert!(m.dropped.is_empty(), "{:?}", m.dropped);
        let before: Vec<_> = m
            .drawing
            .profile_curves()
            .iter()
            .map(|(_, c)| (*c).clone())
            .collect();
        m.drawing.solve();
        let after: Vec<_> = m
            .drawing
            .profile_curves()
            .iter()
            .map(|(_, c)| (*c).clone())
            .collect();
        for (b, a) in before.iter().zip(&after) {
            assert!(b.start().distance(a.start()) < 1e-9 && b.end().distance(a.end()) < 1e-9);
        }
    }
}
