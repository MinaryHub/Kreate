//! ISO 286 limits and fits: how far a hole or a shaft may be from its
//! nominal size, and what a pair of them makes — a clearance, a transition
//! or an interference fit. For sizes up to 500 mm, grades IT5 to IT11, and
//! the letters a machine designer reaches for: e f g h js k m n p, and the
//! holes' E F G H JS K M N P.
//!
//! Every number here is in micrometres, as the standard's tables are.

/// The size ranges' upper bounds, mm: "over 3 up to and including 6".
const RANGES: [f64; 13] = [
    3.0, 6.0, 10.0, 18.0, 30.0, 50.0, 80.0, 120.0, 180.0, 250.0, 315.0, 400.0, 500.0,
];

/// The standard tolerance grades IT4 to IT11, µm, by range.
const IT: [[f64; 8]; 13] = [
    [3.0, 4.0, 6.0, 10.0, 14.0, 25.0, 40.0, 60.0],
    [4.0, 5.0, 8.0, 12.0, 18.0, 30.0, 48.0, 75.0],
    [4.0, 6.0, 9.0, 15.0, 22.0, 36.0, 58.0, 90.0],
    [5.0, 8.0, 11.0, 18.0, 27.0, 43.0, 70.0, 110.0],
    [6.0, 9.0, 13.0, 21.0, 33.0, 52.0, 84.0, 130.0],
    [7.0, 11.0, 16.0, 25.0, 39.0, 62.0, 100.0, 160.0],
    [8.0, 13.0, 19.0, 30.0, 46.0, 74.0, 120.0, 190.0],
    [10.0, 15.0, 22.0, 35.0, 54.0, 87.0, 140.0, 220.0],
    [12.0, 18.0, 25.0, 40.0, 63.0, 100.0, 160.0, 250.0],
    [14.0, 20.0, 29.0, 46.0, 72.0, 115.0, 185.0, 290.0],
    [16.0, 23.0, 32.0, 52.0, 81.0, 130.0, 210.0, 320.0],
    [18.0, 25.0, 36.0, 57.0, 89.0, 140.0, 230.0, 360.0],
    [20.0, 27.0, 40.0, 63.0, 97.0, 155.0, 250.0, 400.0],
];

/// The lowest and highest grade the tables give.
pub const GRADES: std::ops::RangeInclusive<u32> = 5..=11;

/// The letters offered, shafts' and holes' alike (a hole's is upper case).
pub const LETTERS: [&str; 9] = ["e", "f", "g", "h", "js", "k", "m", "n", "p"];

/// Shafts' fundamental deviations, µm, by range: the upper deviation `es`
/// for e f g (h is 0), the lower `ei` for k m n p.
const E: [f64; 13] = [
    -14.0, -20.0, -25.0, -32.0, -40.0, -50.0, -60.0, -72.0, -85.0, -100.0, -110.0, -125.0, -135.0,
];
const F: [f64; 13] = [
    -6.0, -10.0, -13.0, -16.0, -20.0, -25.0, -30.0, -36.0, -43.0, -50.0, -56.0, -62.0, -68.0,
];
const G: [f64; 13] = [
    -2.0, -4.0, -5.0, -6.0, -7.0, -9.0, -10.0, -12.0, -14.0, -15.0, -17.0, -18.0, -20.0,
];
/// k, for grades IT4 to IT7; 0 for the others.
const K: [f64; 13] = [
    0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0,
];
const M: [f64; 13] = [
    2.0, 4.0, 6.0, 7.0, 8.0, 9.0, 11.0, 13.0, 15.0, 17.0, 20.0, 21.0, 23.0,
];
const N: [f64; 13] = [
    4.0, 8.0, 10.0, 12.0, 15.0, 17.0, 20.0, 23.0, 27.0, 31.0, 34.0, 37.0, 40.0,
];
const P: [f64; 13] = [
    6.0, 12.0, 15.0, 18.0, 22.0, 26.0, 32.0, 37.0, 43.0, 50.0, 56.0, 62.0, 68.0,
];

/// How far a size may be from nominal, µm: the most and the least.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Limits {
    pub upper: f64,
    pub lower: f64,
}

/// What a hole and a shaft make together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitKind {
    /// The shaft always smaller: it turns or slides.
    Clearance,
    /// Either, by where in their limits the two fall: located, pressed lightly.
    Transition,
    /// The shaft always larger: pressed or shrunk in.
    Interference,
}

/// A hole and a shaft and what they make, µm. Clearance negative is
/// interference.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fit {
    pub hole: Limits,
    pub shaft: Limits,
    pub max_clearance: f64,
    pub min_clearance: f64,
    pub kind: FitKind,
}

fn range(size: f64) -> Option<usize> {
    if size.is_nan() || size <= 0.0 {
        return None;
    }
    RANGES.iter().position(|top| size <= *top)
}

/// The standard tolerance of `grade` at `size` mm, µm.
pub fn it(grade: u32, size: f64) -> Option<f64> {
    let r = range(size)?;
    (4..=11)
        .contains(&grade)
        .then(|| IT[r][(grade - 4) as usize])
}

/// A shaft's limits: `letter` lower case, as in "g" of g6.
pub fn shaft(letter: &str, grade: u32, size: f64) -> Option<Limits> {
    let r = range(size)?;
    if !GRADES.contains(&grade) {
        return None;
    }
    let t = it(grade, size)?;
    let upper_from = |es: f64| Limits {
        upper: es,
        lower: es - t,
    };
    let lower_from = |ei: f64| Limits {
        upper: ei + t,
        lower: ei,
    };
    Some(match letter {
        "e" => upper_from(E[r]),
        "f" => upper_from(F[r]),
        "g" => upper_from(G[r]),
        "h" => upper_from(0.0),
        "js" => Limits {
            upper: t / 2.0,
            lower: -t / 2.0,
        },
        "k" => lower_from(if grade <= 7 { K[r] } else { 0.0 }),
        "m" => lower_from(M[r]),
        "n" => lower_from(N[r]),
        "p" => lower_from(P[r]),
        _ => return None,
    })
}

/// A hole's limits: `letter` upper case, as in "H" of H7. E to H mirror the
/// shafts' (EI = −es); K to P take the shafts' ei turned over, plus, at the
/// finer grades, Δ — the step from the grade below — so that a hole and a
/// shaft a grade apart make the same fit as the shaft-basis pair.
pub fn hole(letter: &str, grade: u32, size: f64) -> Option<Limits> {
    if !GRADES.contains(&grade) {
        return None;
    }
    let t = it(grade, size)?;
    let r = range(size)?;
    let delta = t - it(grade - 1, size)?;
    let lower_from = |ei: f64| Limits {
        upper: ei + t,
        lower: ei,
    };
    let upper_from = |es: f64| Limits {
        upper: es,
        lower: es - t,
    };
    // Up to 3 mm the tables give K, M, N without Δ, at every grade.
    let small = r == 0;
    Some(match letter {
        "E" => lower_from(-E[r]),
        "F" => lower_from(-F[r]),
        "G" => lower_from(-G[r]),
        "H" => lower_from(0.0),
        "JS" => Limits {
            upper: t / 2.0,
            lower: -t / 2.0,
        },
        "K" if small => upper_from(0.0),
        "K" if grade <= 8 => upper_from(-K[r] + delta),
        "K" => upper_from(0.0),
        "M" if small => upper_from(-M[r]),
        "M" if grade <= 8 => upper_from(-M[r] + delta),
        "M" => upper_from(-M[r]),
        "N" if small => upper_from(-N[r]),
        "N" if grade <= 8 => upper_from(-N[r] + delta),
        "N" => upper_from(0.0),
        "P" if grade <= 7 && !small => upper_from(-P[r] + delta),
        "P" => upper_from(-P[r]),
        _ => return None,
    })
}

/// A hole and a shaft at `size` mm, and what they make.
pub fn fit(
    size: f64,
    hole_letter: &str,
    hole_grade: u32,
    shaft_letter: &str,
    shaft_grade: u32,
) -> Option<Fit> {
    let h = hole(hole_letter, hole_grade, size)?;
    let s = shaft(shaft_letter, shaft_grade, size)?;
    let max_clearance = h.upper - s.lower;
    let min_clearance = h.lower - s.upper;
    let kind = if min_clearance >= 0.0 {
        FitKind::Clearance
    } else if max_clearance <= 0.0 {
        FitKind::Interference
    } else {
        FitKind::Transition
    };
    Some(Fit {
        hole: h,
        shaft: s,
        max_clearance,
        min_clearance,
        kind,
    })
}

/// A fit written as the drawing writes it: "H7/g6".
pub fn parse(text: &str) -> Option<(String, u32, String, u32)> {
    let (h, s) = text.trim().split_once('/')?;
    let split = |t: &str| -> Option<(String, u32)> {
        let at = t.find(|c: char| c.is_ascii_digit())?;
        Some((t[..at].to_string(), t[at..].parse().ok()?))
    };
    let (hl, hg) = split(h.trim())?;
    let (sl, sg) = split(s.trim())?;
    Some((hl, hg, sl, sg))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(l: Option<Limits>) -> (f64, f64) {
        let l = l.expect("in the tables");
        (l.upper, l.lower)
    }

    #[test]
    fn the_tables_give_what_the_standard_prints() {
        // ISO 286-2, at 25 mm.
        assert_eq!(limits(hole("H", 7, 25.0)), (21.0, 0.0));
        assert_eq!(limits(shaft("g", 6, 25.0)), (-7.0, -20.0));
        assert_eq!(limits(shaft("f", 7, 25.0)), (-20.0, -41.0));
        assert_eq!(limits(shaft("k", 6, 25.0)), (15.0, 2.0));
        assert_eq!(limits(shaft("p", 6, 25.0)), (35.0, 22.0));
        assert_eq!(limits(shaft("js", 6, 25.0)), (6.5, -6.5));
        // The holes past H, Δ and all.
        assert_eq!(limits(hole("K", 7, 25.0)), (6.0, -15.0));
        assert_eq!(limits(hole("M", 7, 25.0)), (0.0, -21.0));
        assert_eq!(limits(hole("N", 7, 25.0)), (-7.0, -28.0));
        assert_eq!(limits(hole("P", 7, 25.0)), (-14.0, -35.0));
        assert_eq!(limits(hole("F", 8, 25.0)), (53.0, 20.0));
        // At 50 mm, the top of its range, and just over.
        assert_eq!(limits(hole("H", 7, 50.0)), (25.0, 0.0));
        assert_eq!(limits(hole("H", 7, 50.01)), (30.0, 0.0));
        assert_eq!(limits(shaft("h", 9, 100.0)), (0.0, -87.0));
        // Out of the tables.
        assert!(hole("H", 7, 600.0).is_none());
        assert!(shaft("z", 6, 25.0).is_none());
        assert!(shaft("g", 12, 25.0).is_none());
    }

    #[test]
    fn a_fit_says_what_it_makes() {
        let f = fit(25.0, "H", 7, "g", 6).unwrap();
        assert_eq!(
            (f.max_clearance, f.min_clearance, f.kind),
            (41.0, 7.0, FitKind::Clearance)
        );
        let f = fit(25.0, "H", 7, "k", 6).unwrap();
        assert_eq!(
            (f.max_clearance, f.min_clearance, f.kind),
            (19.0, -15.0, FitKind::Transition)
        );
        let f = fit(25.0, "H", 7, "p", 6).unwrap();
        assert_eq!(
            (f.max_clearance, f.min_clearance, f.kind),
            (-1.0, -35.0, FitKind::Interference)
        );
        assert_eq!(parse(" H7 / g6 "), Some(("H".into(), 7, "g".into(), 6)));
        assert_eq!(parse("JS8/h7"), Some(("JS".into(), 8, "h".into(), 7)));
        assert_eq!(parse("H7"), None);
    }
}
