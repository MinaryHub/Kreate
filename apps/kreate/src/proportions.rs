//! A standing figure measured against a realistic adult's proportions: how
//! many heads tall, how long its legs and arms, how broad its chest and
//! hips — each with the usual range and whether it is in it. A stylised
//! figure departs on purpose; this says by how much, so the departure is a
//! choice.
//!
//! Only what can be measured is: the neck as the thinnest place between
//! the shoulders and the crown, the crotch as where the legs meet, the
//! span from fingertip to fingertip.

use glam::DVec3;

/// One measure.
#[derive(Clone, Debug, PartialEq)]
pub struct Measure {
    pub name: &'static str,
    pub value: f64,
    /// A realistic adult's range.
    pub usual: (f64, f64),
    /// What it is measured from, said plainly.
    pub how: &'static str,
}

impl Measure {
    /// Below, within or above the usual range.
    pub fn verdict(&self) -> &'static str {
        if self.value < self.usual.0 {
            "below"
        } else if self.value > self.usual.1 {
            "above"
        } else {
            "within"
        }
    }
}

/// The figure's measures; empty if it is too small to read.
pub fn measure(points: &[DVec3]) -> Vec<Measure> {
    let (lo, hi) = points.iter().fold(
        (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN)),
        |(lo, hi), p| (lo.min(*p), hi.max(*p)),
    );
    let height = hi.z - lo.z;
    if points.len() < 16 || height <= 0.0 {
        return Vec::new();
    }
    let at = |f: f64| lo.z + height * f;
    // The width across X of what is near the middle at a height: the body,
    // not an arm held out beside it.
    let body_width = |f: f64, reach: f64| -> Option<f64> {
        let band = height * 0.01;
        let xs: Vec<f64> = points
            .iter()
            .filter(|p| (p.z - at(f)).abs() <= band && p.x.abs() <= reach)
            .map(|p| p.x)
            .collect();
        let lo = xs.iter().copied().reduce(f64::min)?;
        let hi = xs.iter().copied().reduce(f64::max)?;
        Some(hi - lo)
    };
    let mut out = Vec::new();

    // The chin: the top of the neck — of the run of heights, between three
    // quarters up and the crown, where the body is about as thin as it gets
    // there — where the head begins to widen.
    let widths: Vec<(f64, f64)> = (75..=94)
        .filter_map(|i| {
            let f = i as f64 / 100.0;
            body_width(f, height * 0.12).map(|w| (f, w))
        })
        .collect();
    if let Some(thinnest) = widths.iter().map(|(_, w)| *w).reduce(f64::min) {
        let chin = widths
            .iter()
            .filter(|(_, w)| *w <= thinnest * 1.1)
            .map(|(f, _)| *f)
            .fold(0.0, f64::max);
        out.push(Measure {
            name: "heads_tall",
            value: 1.0 / (1.0 - chin).max(1e-3),
            usual: (6.8, 8.2),
            how: "height over the head's, crown to chin — the chin where the neck stops being thin",
        });
    }

    // The crotch: the lowest height where something is on the middle line.
    let crotch = (2..70).map(|i| i as f64 / 100.0).find(|f| {
        let band = height * 0.006;
        points
            .iter()
            .any(|p| (p.z - at(*f)).abs() <= band && p.x.abs() <= height * 0.01)
    });
    if let Some(f) = crotch {
        out.push(Measure {
            name: "leg_length_share",
            value: f,
            usual: (0.44, 0.52),
            how: "the crotch's height over the whole height",
        });
    }

    // The span: tip to tip, when the arms are out far enough to read it.
    let span = hi.x - lo.x;
    if span > height * 0.8 {
        out.push(Measure {
            name: "arm_span_share",
            value: span / height,
            usual: (0.95, 1.06),
            how: "fingertip to fingertip over the height, arms out",
        });
    }

    if let Some(chest) = body_width(0.72, height * 0.16) {
        out.push(Measure {
            name: "chest_width_share",
            value: chest / height,
            usual: (0.14, 0.21),
            how: "the body's width at the chest over the height",
        });
    }
    if let Some(hips) = body_width(0.52, height * 0.16) {
        out.push(Measure {
            name: "hip_width_share",
            value: hips / height,
            usual: (0.14, 0.21),
            how: "the body's width at the hips over the height",
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grid of points filling a box, every few centimetres.
    fn fill(lo: DVec3, hi: DVec3, out: &mut Vec<DVec3>) {
        let step = 0.02;
        let mut z = lo.z;
        while z <= hi.z + 1e-9 {
            let mut x = lo.x;
            while x <= hi.x + 1e-9 {
                for y in [lo.y, hi.y] {
                    out.push(DVec3::new(x, y, z));
                }
                x += step;
            }
            z += step;
        }
    }

    #[test]
    fn a_realistic_figure_is_within_and_a_big_headed_one_is_told() {
        let figure = |head: f64| {
            let mut p = Vec::new();
            let top = 1.8;
            let neck = top - head;
            fill(
                DVec3::new(-0.16, -0.1, 0.86),
                DVec3::new(0.16, 0.1, neck - 0.08),
                &mut p,
            ); // torso
            fill(
                DVec3::new(-0.04, -0.04, neck - 0.08),
                DVec3::new(0.04, 0.04, neck + 0.02),
                &mut p,
            ); // neck
            fill(
                DVec3::new(-0.09, -0.1, neck + 0.02),
                DVec3::new(0.09, 0.1, top),
                &mut p,
            ); // head
            fill(
                DVec3::new(0.03, -0.07, 0.0),
                DVec3::new(0.15, 0.07, 0.86),
                &mut p,
            ); // legs
            fill(
                DVec3::new(-0.15, -0.07, 0.0),
                DVec3::new(-0.03, 0.07, 0.86),
                &mut p,
            );
            fill(
                DVec3::new(0.16, -0.05, 1.38),
                DVec3::new(0.9, 0.05, 1.44),
                &mut p,
            ); // arms
            fill(
                DVec3::new(-0.9, -0.05, 1.38),
                DVec3::new(-0.16, 0.05, 1.44),
                &mut p,
            );
            p
        };
        let real = measure(&figure(0.23));
        let get =
            |m: &[Measure], name: &str| m.iter().find(|x| x.name == name).cloned().expect(name);
        assert_eq!(get(&real, "heads_tall").verdict(), "within", "{real:?}");
        assert_eq!(
            get(&real, "leg_length_share").verdict(),
            "within",
            "{real:?}"
        );
        assert_eq!(get(&real, "arm_span_share").verdict(), "within", "{real:?}");

        let chibi = measure(&figure(0.45));
        let heads = get(&chibi, "heads_tall");
        assert_eq!(heads.verdict(), "below", "{heads:?}");
        assert!(heads.value < 4.5, "{heads:?}");
    }
}
