//! The path array: copies of the mesh set along another object's edges —
//! a row of posts along a fence line, bolts round a flange, lamps along a
//! road — each turned to follow the path if asked.
//!
//! The path is a chain of edges: its points walked end to end, or round
//! when it closes on itself. Where the edges branch, the longest chain is
//! the path.
//!
//! # Where the copies go
//!
//! Spread evenly over the whole path unless a spacing is given; slid along
//! it by `start`, which wraps round a closed path and drops the copies that
//! slide off an open one; and moved off it by `offset`, measured along the
//! path, across it and up from it at each copy — so a row can sit beside the
//! line rather than on it, and stay beside it round the bends.

use kreate_math::Vec3;

use crate::MeshData;

/// How many, how far apart, and where along and beside the path.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PathArrayParams {
    /// Copies, the first at the path's start.
    pub count: u32,
    /// The step along the path, in metres; zero spreads the copies over the
    /// whole of it.
    pub spacing: f32,
    /// Where the first copy sits, in metres along the path from its start.
    pub start: f32,
    /// Each copy moved from its place on the path: along it, across it and
    /// up from it, in metres.
    pub offset: Vec3,
    /// Each copy turned so its X runs along the path; otherwise only moved.
    pub align: bool,
}

impl Default for PathArrayParams {
    fn default() -> Self {
        Self {
            count: 5,
            spacing: 0.0,
            start: 0.0,
            offset: Vec3::ZERO,
            align: true,
        }
    }
}

/// The longest chain `edges` make between `points`, as the points along it,
/// and whether it closes on itself. `None` when there are no edges.
pub fn path_chain(points: &[Vec3], edges: &[[u32; 2]]) -> Option<(Vec<Vec3>, bool)> {
    let n = points.len();
    let mut around: Vec<Vec<u32>> = vec![Vec::new(); n];
    for [a, b] in edges {
        let (a, b) = (*a as usize, *b as usize);
        if a < n && b < n && a != b {
            around[a].push(b as u32);
            around[b].push(a as u32);
        }
    }
    let mut used = std::collections::HashSet::new();
    let key = |a: u32, b: u32| (a.min(b), a.max(b));
    let mut best: Option<(Vec<u32>, bool, f32)> = None;
    // Ends first, so an open chain is walked from one end, not its middle;
    // then whatever is left, which is loops.
    let mut starts: Vec<u32> = (0..n as u32)
        .filter(|v| around[*v as usize].len() == 1)
        .collect();
    starts.extend((0..n as u32).filter(|v| !around[*v as usize].is_empty()));
    for s in starts {
        let Some(&first) = around[s as usize]
            .iter()
            .find(|w| !used.contains(&key(s, **w)))
        else {
            continue;
        };
        let mut chain = vec![s];
        let (mut prev, mut at) = (s, first);
        used.insert(key(s, first));
        let mut closed = false;
        loop {
            chain.push(at);
            if at == s {
                chain.pop();
                closed = true;
                break;
            }
            // Straight on where it can: the least turn.
            let came = (points[at as usize] - points[prev as usize]).normalize_or_zero();
            let next = around[at as usize]
                .iter()
                .copied()
                .filter(|w| !used.contains(&key(at, *w)))
                .max_by(|a, b| {
                    let da = (points[*a as usize] - points[at as usize]).normalize_or_zero();
                    let db = (points[*b as usize] - points[at as usize]).normalize_or_zero();
                    came.dot(da).total_cmp(&came.dot(db))
                });
            let Some(next) = next else { break };
            used.insert(key(at, next));
            prev = at;
            at = next;
        }
        let mut length: f32 = chain
            .windows(2)
            .map(|w| points[w[0] as usize].distance(points[w[1] as usize]))
            .sum();
        if closed && chain.len() > 1 {
            length += points[chain[0] as usize].distance(points[*chain.last().unwrap() as usize]);
        }
        if best.as_ref().is_none_or(|b| length > b.2) {
            best = Some((chain, closed, length));
        }
    }
    let (chain, closed, length) = best?;
    if chain.len() < 2 || length <= 0.0 {
        return None;
    }
    Some((chain.iter().map(|v| points[*v as usize]).collect(), closed))
}

/// The place `s` metres along `path`, and the way it runs there.
fn along(path: &[Vec3], closed: bool, s: f32) -> (Vec3, Vec3) {
    let count = if closed { path.len() } else { path.len() - 1 };
    let mut left = s;
    for i in 0..count {
        let (a, b) = (path[i], path[(i + 1) % path.len()]);
        let len = a.distance(b);
        if left <= len || i + 1 == count {
            let t = if len > 0.0 {
                (left / len).clamp(0.0, 1.0)
            } else {
                0.0
            };
            return (a.lerp(b, t), (b - a).normalize_or_zero());
        }
        left -= len;
    }
    (path[0], Vec3::X)
}

/// Where each copy goes: a place and a turn (its X, Y and Z), in order.
pub fn placements(path: &[Vec3], closed: bool, params: &PathArrayParams) -> Vec<(Vec3, [Vec3; 3])> {
    let mut out = Vec::new();
    if path.len() < 2 {
        return out;
    }
    let segments = if closed { path.len() } else { path.len() - 1 };
    let length: f32 = (0..segments)
        .map(|i| path[i].distance(path[(i + 1) % path.len()]))
        .sum();
    if length <= 0.0 {
        return out;
    }
    let count = params.count.clamp(1, super::array::MAX_COUNT);
    let step = if params.spacing > 0.0 {
        params.spacing
    } else if closed {
        length / count as f32
    } else if count > 1 {
        length / (count - 1) as f32
    } else {
        0.0
    };
    for k in 0..count {
        let mut s = params.start + step * k as f32;
        if closed {
            s = s.rem_euclid(length);
        } else if s < -1e-5 || s > length + 1e-5 {
            continue;
        }
        let (at, x) = along(path, closed, s.clamp(0.0, length));
        let x = if x == Vec3::ZERO { Vec3::X } else { x };
        // Up is the object's Z, unless the path runs up it.
        let up = if x.dot(Vec3::Z).abs() > 0.99 {
            Vec3::Y
        } else {
            Vec3::Z
        };
        let y = up.cross(x).normalize_or_zero();
        let z = x.cross(y);
        let place = at + x * params.offset.x + y * params.offset.y + z * params.offset.z;
        let turn = if params.align {
            [x, y, z]
        } else {
            [Vec3::X, Vec3::Y, Vec3::Z]
        };
        out.push((place, turn));
    }
    out
}

/// Copies of `input` along `path`, each about its own origin.
pub fn path_array(
    input: &MeshData,
    path: &[Vec3],
    closed: bool,
    params: &PathArrayParams,
) -> MeshData {
    let places = placements(path, closed, params);
    if places.is_empty() || input.is_empty() {
        return input.clone();
    }
    let source_verts = input.num_verts() as usize;
    let mut positions: Vec<Vec3> = Vec::with_capacity(source_verts * places.len());
    let mut point_order: Vec<u32> = Vec::with_capacity(positions.capacity());
    let mut faces: Vec<Vec<u32>> = Vec::new();
    let mut face_order: Vec<u32> = Vec::new();
    let mut corner_order: Vec<u32> = Vec::new();
    for (copy, (place, [x, y, z])) in places.iter().enumerate() {
        let base = (copy * source_verts) as u32;
        for (i, p) in input.positions().iter().enumerate() {
            positions.push(*place + *x * p.x + *y * p.y + *z * p.z);
            point_order.push(i as u32);
        }
        for (index, face) in input.faces().enumerate() {
            faces.push(face.iter().map(|v| v + base).collect());
            face_order.push(index as u32);
            corner_order.extend(input.face_corner_range(index as u32).map(|c| c as u32));
        }
    }
    let mut out = MeshData::from_faces(positions, &faces);
    super::carry_attributes(&mut out, input, &point_order, &corner_order, &face_order);
    super::gather_edges_by_pair(&mut out, input, &point_order);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    fn square() -> (Vec<Vec3>, Vec<[u32; 2]>) {
        (
            vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(4.0, 0.0, 0.0),
                Vec3::new(4.0, 4.0, 0.0),
                Vec3::new(0.0, 4.0, 0.0),
            ],
            vec![[0, 1], [1, 2], [2, 3], [3, 0]],
        )
    }

    #[test]
    fn a_square_of_edges_is_a_closed_path_and_a_line_an_open_one() {
        let (p, e) = square();
        let (path, closed) = path_chain(&p, &e).unwrap();
        assert!(closed);
        assert_eq!(path.len(), 4);
        let (path, closed) = path_chain(&p, &e[..3]).unwrap();
        assert!(!closed);
        assert_eq!(path.len(), 4);
        assert_eq!(path[0], Vec3::ZERO, "walked from an end");
        assert!(path_chain(&p, &[]).is_none());
    }

    #[test]
    fn copies_spread_round_a_closed_path_and_slide_with_the_start() {
        let (p, e) = square();
        let (path, closed) = path_chain(&p, &e).unwrap();
        // Eight round a 16 m square: one every 2 m, at the corners and the
        // middles of the sides.
        let params = PathArrayParams {
            count: 8,
            ..Default::default()
        };
        let at: Vec<Vec3> = placements(&path, closed, &params)
            .iter()
            .map(|p| p.0)
            .collect();
        assert_eq!(at.len(), 8);
        assert!(at[0].distance(Vec3::ZERO) < 1e-5);
        assert!(at[1].distance(Vec3::new(2.0, 0.0, 0.0)) < 1e-5);
        assert!(at[3].distance(Vec3::new(4.0, 2.0, 0.0)) < 1e-5);
        // Slid a metre along: every copy a metre further round.
        let slid = PathArrayParams {
            start: 1.0,
            ..params
        };
        let at: Vec<Vec3> = placements(&path, closed, &slid)
            .iter()
            .map(|p| p.0)
            .collect();
        assert!(at[0].distance(Vec3::new(1.0, 0.0, 0.0)) < 1e-5);
        // The last wraps past the start: 1 + 7 × 2 = 15 m round, the fourth
        // side's middle.
        assert!(
            at[7].distance(Vec3::new(0.0, 1.0, 0.0)) < 1e-5,
            "{:?}",
            at[7]
        );
    }

    #[test]
    fn an_offset_keeps_beside_the_path_round_its_bends() {
        let (p, e) = square();
        let (path, closed) = path_chain(&p, &e).unwrap();
        let params = PathArrayParams {
            count: 4,
            start: 2.0,
            offset: Vec3::new(0.0, -0.5, 0.25),
            ..Default::default()
        };
        // Each side's middle, half a metre outside the square (to the
        // right of the way round), a quarter up.
        let at: Vec<Vec3> = placements(&path, closed, &params)
            .iter()
            .map(|p| p.0)
            .collect();
        let want = [
            Vec3::new(2.0, -0.5, 0.25),
            Vec3::new(4.5, 2.0, 0.25),
            Vec3::new(2.0, 4.5, 0.25),
            Vec3::new(-0.5, 2.0, 0.25),
        ];
        for (a, w) in at.iter().zip(want) {
            assert!(a.distance(w) < 1e-5, "{a:?} against {w:?}");
        }
    }

    #[test]
    fn an_open_path_drops_what_slides_off_it_and_spacing_is_honoured() {
        let (p, e) = square();
        let (path, closed) = path_chain(&p, &e[..1]).unwrap();
        let params = PathArrayParams {
            count: 5,
            spacing: 1.0,
            start: 1.5,
            ..Default::default()
        };
        let at: Vec<Vec3> = placements(&path, closed, &params)
            .iter()
            .map(|p| p.0)
            .collect();
        // 1.5, 2.5, 3.5 on a 4 m line; 4.5 and 5.5 are off it.
        assert_eq!(at.len(), 3);
        assert!(at[2].distance(Vec3::new(3.5, 0.0, 0.0)) < 1e-5);
    }

    #[test]
    fn copies_turn_with_the_path_when_aligned() {
        let (p, e) = square();
        let (path, closed) = path_chain(&p, &e).unwrap();
        let bar = {
            // A bar along X: 1 m long.
            let mut m = primitives::cube(1.0);
            for q in m.positions_mut() {
                q.y *= 0.1;
                q.z *= 0.1;
            }
            m
        };
        let params = PathArrayParams {
            count: 4,
            start: 2.0,
            ..Default::default()
        };
        let out = path_array(&bar, &path, closed, &params);
        assert_eq!(out.num_verts(), 32);
        out.assert_valid();
        // The second copy, on the side running up Y, lies along Y.
        let second: Vec<Vec3> = out.positions()[8..16].to_vec();
        let span = |f: fn(&Vec3) -> f32| {
            let v: Vec<f32> = second.iter().map(f).collect();
            v.iter().cloned().fold(f32::MIN, f32::max) - v.iter().cloned().fold(f32::MAX, f32::min)
        };
        assert!((span(|q| q.y) - 1.0).abs() < 1e-5);
        assert!((span(|q| q.x) - 0.1).abs() < 1e-5);
        // Not aligned: every copy as it was, only moved.
        let still = path_array(
            &bar,
            &path,
            closed,
            &PathArrayParams {
                align: false,
                ..params
            },
        );
        let second: Vec<f32> = still.positions()[8..16].iter().map(|q| q.x).collect();
        let w = second.iter().cloned().fold(f32::MIN, f32::max)
            - second.iter().cloned().fold(f32::MAX, f32::min);
        assert!((w - 1.0).abs() < 1e-5);
    }
}
