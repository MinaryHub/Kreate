//! A model's outline against a picture's: how much they agree, and where
//! the model has too much or too little — the measure an assistant shaping a
//! character to concept art needs, side by side and front on.
//!
//! Both are made the same height and set on the same middle before they are
//! compared, so a picture of any size, framed any way, measures the shape and
//! its proportions, not where the camera stood.

use glam::DVec3;

/// Which way the model is looked at, as the viewport's sides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Front,
    Back,
    Left,
    Right,
    Top,
}

impl Side {
    pub fn from_name(name: &str) -> Option<Side> {
        Some(match name {
            "front" => Side::Front,
            "back" => Side::Back,
            "left" => Side::Left,
            "right" | "side" => Side::Right,
            "top" => Side::Top,
            _ => return None,
        })
    }

    /// The picture plane's own third axis, right × up: a point's depth is
    /// its position along it.
    pub fn depth_axis(self) -> DVec3 {
        let (right, up) = self.axes();
        right.cross(up)
    }

    /// The world directions of the picture's right and up.
    fn axes(self) -> (DVec3, DVec3) {
        match self {
            Side::Front => (DVec3::X, DVec3::Z),
            Side::Back => (DVec3::NEG_X, DVec3::Z),
            Side::Right => (DVec3::Y, DVec3::Z),
            Side::Left => (DVec3::NEG_Y, DVec3::Z),
            Side::Top => (DVec3::X, DVec3::Y),
        }
    }
}

/// An outline: which cells of a grid are inside, row 0 at the top.
#[derive(Clone, Debug, PartialEq)]
pub struct Mask {
    pub width: usize,
    pub height: usize,
    pub inside: Vec<bool>,
}

impl Mask {
    fn at(&self, x: usize, y: usize) -> bool {
        x < self.width && y < self.height && self.inside[y * self.width + x]
    }

    /// The columns and rows something is in: left, top, right, bottom.
    fn extent(&self) -> Option<(usize, usize, usize, usize)> {
        let mut out: Option<(usize, usize, usize, usize)> = None;
        for y in 0..self.height {
            for x in 0..self.width {
                if self.at(x, y) {
                    out = Some(match out {
                        None => (x, y, x, y),
                        Some((l, t, r, b)) => (l.min(x), t.min(y), r.max(x), b.max(y)),
                    });
                }
            }
        }
        out
    }
}

/// How many rows the outlines are compared at.
const ROWS: usize = 256;

/// The model's outline seen from `side`: its triangles, in the world, laid
/// flat. With it, the picture-plane box it fills (left, bottom, right, top)
/// in metres, to say where a difference is on the model.
pub fn outline(triangles: &[[DVec3; 3]], side: Side) -> Option<(Mask, [f64; 4])> {
    let (right, up) = side.axes();
    let flat: Vec<[(f64, f64); 3]> = triangles
        .iter()
        .map(|t| t.map(|p| (p.dot(right), p.dot(up))))
        .collect();
    let (mut lo, mut hi) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
    for t in &flat {
        for (u, v) in t {
            lo = (lo.0.min(*u), lo.1.min(*v));
            hi = (hi.0.max(*u), hi.1.max(*v));
        }
    }
    let (w, h) = (hi.0 - lo.0, hi.1 - lo.1);
    if flat.is_empty() || w <= 0.0 || h <= 0.0 {
        return None;
    }
    let cell = h / ROWS as f64;
    let width = ((w / cell).ceil() as usize).clamp(1, ROWS * 8);
    let mut mask = Mask {
        width,
        height: ROWS,
        inside: vec![false; width * ROWS],
    };
    // Each cell whose middle a triangle covers.
    for t in &flat {
        let to_cell = |(u, v): (f64, f64)| ((u - lo.0) / cell, (hi.1 - v) / cell);
        let [a, b, c] = t.map(to_cell);
        let (x0, x1) = (
            a.0.min(b.0).min(c.0).floor().max(0.0) as usize,
            a.0.max(b.0).max(c.0).ceil() as usize,
        );
        let (y0, y1) = (
            a.1.min(b.1).min(c.1).floor().max(0.0) as usize,
            a.1.max(b.1).max(c.1).ceil() as usize,
        );
        let area = (b.0 - a.0) * (c.1 - a.1) - (c.0 - a.0) * (b.1 - a.1);
        if area.abs() < 1e-12 {
            continue;
        }
        for y in y0..y1.min(ROWS) {
            for x in x0..x1.min(width) {
                let p = (x as f64 + 0.5, y as f64 + 0.5);
                let edge = |s: (f64, f64), e: (f64, f64)| {
                    (e.0 - s.0) * (p.1 - s.1) - (e.1 - s.1) * (p.0 - s.0)
                };
                let (e0, e1, e2) = (edge(a, b), edge(b, c), edge(c, a));
                let inside = if area > 0.0 {
                    e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0
                } else {
                    e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0
                };
                if inside {
                    mask.inside[y * width + x] = true;
                }
            }
        }
    }
    Some((mask, [lo.0, lo.1, hi.0, hi.1]))
}

/// A picture's outline: what is opaque, when it has see-through parts, or
/// else what differs from the colour round its edge.
pub fn picture_outline(bytes: &[u8]) -> Result<Mask, String> {
    let image = image::load_from_memory(bytes)
        .map_err(|e| format!("cannot read the picture: {e}"))?
        .to_rgba8();
    let (w, h) = (image.width() as usize, image.height() as usize);
    if w < 4 || h < 4 {
        return Err("the picture is too small".into());
    }
    let pixel = |x: usize, y: usize| image.get_pixel(x as u32, y as u32).0;
    let see_through = (0..h)
        .step_by(4)
        .any(|y| (0..w).step_by(4).any(|x| pixel(x, y)[3] < 128));
    // The background: the middle colour of the border.
    let mut border: Vec<[u8; 4]> = (0..w)
        .flat_map(|x| [pixel(x, 0), pixel(x, h - 1)])
        .chain((0..h).flat_map(|y| [pixel(0, y), pixel(w - 1, y)]))
        .collect();
    let mut channel = |i: usize| {
        border.sort_by_key(|p| p[i]);
        border[border.len() / 2][i] as i32
    };
    let background = [channel(0), channel(1), channel(2)];
    let inside: Vec<bool> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| {
            let p = pixel(x, y);
            if see_through {
                p[3] >= 128
            } else {
                let d: i32 = (0..3).map(|i| (p[i] as i32 - background[i]).abs()).sum();
                d > 60
            }
        })
        .collect();
    Ok(Mask {
        width: w,
        height: h,
        inside,
    })
}

/// A place where the outlines differ.
#[derive(Clone, Debug, PartialEq)]
pub struct Difference {
    /// The model sticks out where the picture does not; or the other way.
    pub model_has_more: bool,
    /// Where, in the world, on the picture plane through the model's middle.
    pub at: [f64; 3],
    /// How big, as a share of the outline's area.
    pub share: f64,
}

/// What the comparison found.
#[derive(Clone, Debug)]
pub struct Comparison {
    /// How much they overlap: shared over either, 0 to 1.
    pub agreement: f64,
    /// Width over height of each.
    pub model_aspect: f64,
    pub picture_aspect: f64,
    /// The biggest differences, biggest first.
    pub differences: Vec<Difference>,
    /// A picture of it: both in grey, the model's extra in red, its
    /// shortfall in blue — as a PNG.
    pub png: Vec<u8>,
}

/// The model's outline against the picture's, both scaled to the same
/// height and set side to side where they overlap most. `plane` is the model's box from [`outline`],
/// and `side` and `depth` put a difference back in the world.
pub fn compare(
    model: &Mask,
    plane: [f64; 4],
    picture: &Mask,
    side: Side,
    depth: f64,
) -> Result<Comparison, String> {
    let fit = |m: &Mask| -> Result<Mask, String> {
        let (l, t, r, b) = m.extent().ok_or("an outline is empty")?;
        let (w, h) = (r - l + 1, b - t + 1);
        let scale = ROWS as f64 / h as f64;
        let width = ((w as f64 * scale).round() as usize).max(1);
        let mut out = Mask {
            width,
            height: ROWS,
            inside: vec![false; width * ROWS],
        };
        for y in 0..ROWS {
            for x in 0..width {
                let sx = l + ((x as f64 + 0.5) / scale) as usize;
                let sy = t + ((y as f64 + 0.5) / scale) as usize;
                out.inside[y * width + x] = m.at(sx.min(r), sy.min(b));
            }
        }
        Ok(out)
    };
    let (a, b) = (fit(model)?, fit(picture)?);
    // Side to side, where they overlap most: centring their boxes would
    // put a body beside itself when only the picture has an arm out.
    // The model in the middle, room for the picture on either side of it.
    let oa = b.width;
    let width = oa + a.width + b.width;
    let overlap = |ob: usize| -> usize {
        (0..ROWS)
            .map(|y| {
                (0..b.width)
                    .filter(|x| {
                        b.at(*x, y) && (ob + x).checked_sub(oa).is_some_and(|ax| a.at(ax, y))
                    })
                    .count()
            })
            .sum()
    };
    let ob = (0..=oa + a.width)
        .max_by_key(|ob| {
            (
                overlap(*ob),
                std::cmp::Reverse(ob.abs_diff(oa + a.width / 2)),
            )
        })
        .unwrap_or(oa);
    let (mut shared, mut either) = (0usize, 0usize);
    let mut more = vec![false; width * ROWS];
    let mut less = vec![false; width * ROWS];
    let mut png = image::RgbImage::new(width as u32, ROWS as u32);
    for y in 0..ROWS {
        for x in 0..width {
            let m = x >= oa && a.at(x - oa, y);
            let p = x >= ob && b.at(x - ob, y);
            shared += (m && p) as usize;
            either += (m || p) as usize;
            more[y * width + x] = m && !p;
            less[y * width + x] = p && !m;
            let colour = match (m, p) {
                (true, true) => [150, 150, 150],
                (true, false) => [220, 40, 40],
                (false, true) => [40, 90, 230],
                (false, false) => [255, 255, 255],
            };
            png.put_pixel(x as u32, y as u32, image::Rgb(colour));
        }
    }
    // Differences in blocks a sixteenth of the height across.
    let block = ROWS / 16;
    let (right, up) = side.axes();
    let normal = right.cross(up);
    let model_h = plane[3] - plane[1];
    let mut found = Vec::new();
    for (cells, model_has_more) in [(&more, true), (&less, false)] {
        for by in (0..ROWS).step_by(block) {
            for bx in (0..width).step_by(block) {
                let mut count = 0;
                let (mut sx, mut sy) = (0.0, 0.0);
                for y in by..(by + block).min(ROWS) {
                    for x in bx..(bx + block).min(width) {
                        if cells[y * width + x] {
                            count += 1;
                            sx += x as f64;
                            sy += y as f64;
                        }
                    }
                }
                if count * 4 < block * block {
                    continue;
                }
                let (cx, cy) = (sx / count as f64, sy / count as f64);
                // Back to the model's plane: its height is the grid's.
                let metres = model_h / ROWS as f64;
                let u = plane[0] + (cx - oa as f64) * metres;
                let v = plane[3] - cy * metres;
                let at = right * u + up * v + normal * depth;
                found.push(Difference {
                    model_has_more,
                    at: at.to_array(),
                    share: count as f64 / either.max(1) as f64,
                });
            }
        }
    }
    found.sort_by(|a, b| b.share.total_cmp(&a.share));
    found.truncate(8);
    // Only the columns either is in, and a little room.
    let used: Vec<usize> = (0..width)
        .filter(|x| {
            (0..ROWS).any(|y| {
                more[y * width + x] || less[y * width + x] || (*x >= oa && a.at(x - oa, y))
            })
        })
        .collect();
    let (first, last) = (
        used.first().copied().unwrap_or(0).saturating_sub(4),
        (used.last().copied().unwrap_or(width - 1) + 4).min(width - 1),
    );
    let png = image::imageops::crop_imm(
        &png,
        first as u32,
        0,
        (last - first + 1) as u32,
        ROWS as u32,
    )
    .to_image();
    let mut bytes = Vec::new();
    png.write_to(
        &mut std::io::Cursor::new(&mut bytes),
        image::ImageFormat::Png,
    )
    .map_err(|e| e.to_string())?;
    let aspect = |m: &Mask| {
        m.extent()
            .map(|(l, t, r, b)| (r - l + 1) as f64 / (b - t + 1) as f64)
            .unwrap_or(0.0)
    };
    Ok(Comparison {
        agreement: shared as f64 / either.max(1) as f64,
        model_aspect: aspect(model),
        picture_aspect: aspect(picture),
        differences: found,
        png: bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A box's twelve triangles.
    fn boxed(lo: DVec3, hi: DVec3) -> Vec<[DVec3; 3]> {
        let c = |i: usize| {
            DVec3::new(
                if i & 1 == 0 { lo.x } else { hi.x },
                if i & 2 == 0 { lo.y } else { hi.y },
                if i & 4 == 0 { lo.z } else { hi.z },
            )
        };
        let quads = [
            [0, 1, 3, 2],
            [4, 6, 7, 5],
            [0, 4, 5, 1],
            [2, 3, 7, 6],
            [0, 2, 6, 4],
            [1, 5, 7, 3],
        ];
        quads
            .iter()
            .flat_map(|q| [[c(q[0]), c(q[1]), c(q[2])], [c(q[0]), c(q[2]), c(q[3])]])
            .collect()
    }

    /// A picture of a figure: a dark body and, if asked, a dark arm out to
    /// the right, on a light ground.
    fn picture(with_arm: bool) -> Vec<u8> {
        let mut img = image::RgbImage::from_pixel(200, 300, image::Rgb([240, 240, 240]));
        for y in 50..250 {
            for x in 80..120 {
                img.put_pixel(x, y, image::Rgb([30, 30, 30]));
            }
        }
        if with_arm {
            for y in 80..100 {
                for x in 120..180 {
                    img.put_pixel(x, y, image::Rgb([30, 30, 30]));
                }
            }
        }
        let mut bytes = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .expect("written");
        bytes
    }

    #[test]
    fn a_model_like_its_picture_agrees_and_one_missing_an_arm_is_told_where() {
        // A body 0.4 wide and 2 tall, front on: the picture's 40 by 200.
        let body = boxed(DVec3::new(-0.2, -0.1, 0.0), DVec3::new(0.2, 0.1, 2.0));
        let (model, plane) = outline(&body, Side::Front).expect("an outline");
        let same = compare(
            &model,
            plane,
            &picture_outline(&picture(false)).expect("read"),
            Side::Front,
            0.0,
        )
        .expect("compared");
        assert!(same.agreement > 0.95, "{}", same.agreement);

        // The picture has an arm the model lacks: up on the model's right.
        let armed = compare(
            &model,
            plane,
            &picture_outline(&picture(true)).expect("read"),
            Side::Front,
            0.0,
        )
        .expect("compared");
        // Body 8000 cells, arm 1200: 8000 / 9200.
        assert!((armed.agreement - 0.87).abs() < 0.02, "{}", armed.agreement);
        let missing = armed
            .differences
            .iter()
            .find(|d| !d.model_has_more)
            .expect("the arm is missing");
        assert!(missing.at[2] > 1.3, "high on the body: {:?}", missing.at);
        assert!(armed.picture_aspect > armed.model_aspect);
        assert!(!armed.png.is_empty());
    }

    #[test]
    fn a_see_through_picture_is_read_by_its_alpha() {
        let mut img = image::RgbaImage::from_pixel(50, 50, image::Rgba([255, 255, 255, 0]));
        for y in 10..40 {
            for x in 20..30 {
                img.put_pixel(x, y, image::Rgba([255, 255, 255, 255]));
            }
        }
        let mut bytes = Vec::new();
        img.write_to(
            &mut std::io::Cursor::new(&mut bytes),
            image::ImageFormat::Png,
        )
        .expect("written");
        let mask = picture_outline(&bytes).expect("read");
        assert_eq!(mask.extent(), Some((20, 10, 29, 39)));
    }
}
