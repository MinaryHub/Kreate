//! IES LM-63 photometric files: how a real luminaire spreads its light.
//!
//! A downlight throws most of its light straight down and almost none
//! sideways; a wall-washer throws it off to one side; a bare bulb evenly.
//! Lighting manufacturers publish that shape, measured, as an IES file, and
//! an architect's render that ignores it lights a room the way no fitting
//! does. This reads one into what a light needs: its peak intensity and the
//! shape relative to it, by angle from the direction the light points.
//!
//! **Rotationally symmetric.** The file may give the intensity at several
//! horizontal angles round the light; they are averaged into one vertical
//! profile. That is exact for the round downlights and spots most files
//! describe, and the mean of an asymmetric one — which a lamp in the
//! renderer, carrying a profile in its own record, can hold.

/// How many samples a profile has, evenly from straight along the light's
/// axis (0°) to straight back (180°).
pub const PROFILE_SAMPLES: usize = 32;

/// A luminaire's light, read from its IES file.
#[derive(Clone, Debug, PartialEq)]
pub struct Photometry {
    /// The brightest the luminaire is in any direction, in candela.
    pub peak_candela: f64,
    /// The intensity by angle from the axis, relative to the peak: the first
    /// sample along the axis, the last straight back, the rest even between.
    pub profile: [f32; PROFILE_SAMPLES],
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum IesError {
    #[error("not an IES file: no TILT line")]
    NoTilt,
    #[error("the file ends before its numbers do")]
    Truncated,
    #[error("a number in the file does not read: {0}")]
    BadNumber(String),
    #[error("the file gives no light")]
    Dark,
}

/// Reads an IES LM-63 file (any year's: 1986, 1991, 1995, 2002, 2019).
pub fn read(text: &str) -> Result<Photometry, IesError> {
    // Keywords and the TILT line, then the numbers, whitespace or commas apart.
    let mut lines = text.lines();
    let mut tilt = None;
    for line in lines.by_ref() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("TILT=") {
            tilt = Some(rest.trim().to_string());
            break;
        }
    }
    let tilt = tilt.ok_or(IesError::NoTilt)?;
    let rest: String = lines.collect::<Vec<_>>().join(" ");
    let mut numbers = rest
        .split(|c: char| c.is_whitespace() || c == ',')
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<f64>()
                .map_err(|_| IesError::BadNumber(s.to_string()))
        });
    let mut next = || numbers.next().ok_or(IesError::Truncated)?;
    // A tilt given in the file: its angles and factors come first, and are
    // for a lamp that is tilted in its fitting — not what is drawn here.
    if tilt == "INCLUDE" {
        let _geometry = next()?;
        let pairs = next()? as usize;
        for _ in 0..pairs.min(1 << 16) * 2 {
            next()?;
        }
    }
    let _lamps = next()?;
    let _lumens = next()?;
    let multiplier = next()?;
    let vertical = next()? as usize;
    let horizontal = next()? as usize;
    let _photometric_type = next()?;
    let _units = next()?;
    for _ in 0..3 {
        next()?; // width, length, height
    }
    let ballast = next()?;
    let _future = next()?;
    let _watts = next()?;
    if vertical == 0 || horizontal == 0 || vertical > 4096 || horizontal > 4096 {
        return Err(IesError::Truncated);
    }
    let v_angles: Vec<f64> = (0..vertical).map(|_| next()).collect::<Result<_, _>>()?;
    for _ in 0..horizontal {
        next()?;
    }
    let mut mean = vec![0.0f64; vertical];
    for _ in 0..horizontal {
        for m in mean.iter_mut() {
            *m += next()?;
        }
    }
    let scale = multiplier * if ballast > 0.0 { ballast } else { 1.0 } / horizontal as f64;
    for m in mean.iter_mut() {
        *m = (*m * scale).max(0.0);
    }
    let peak = mean.iter().copied().fold(0.0f64, f64::max);
    if peak.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
        return Err(IesError::Dark);
    }
    // Onto even steps from 0° to 180°, linearly between the file's angles;
    // outside the angles it covers (an uplight from 90° up, a downlight to
    // 90° down) there is no light.
    let mut profile = [0.0f32; PROFILE_SAMPLES];
    for (i, p) in profile.iter_mut().enumerate() {
        let a = i as f64 * 180.0 / (PROFILE_SAMPLES - 1) as f64;
        let first = v_angles[0];
        let last = v_angles[vertical - 1];
        *p = if a < first - 1e-6 || a > last + 1e-6 {
            0.0
        } else {
            let k = v_angles.partition_point(|&x| x < a).min(vertical - 1);
            let value = if k == 0 {
                mean[0]
            } else {
                let (a0, a1) = (v_angles[k - 1], v_angles[k]);
                let t = if a1 > a0 { (a - a0) / (a1 - a0) } else { 0.0 };
                mean[k - 1] + (mean[k] - mean[k - 1]) * t
            };
            (value / peak) as f32
        };
    }
    Ok(Photometry {
        peak_candela: peak,
        profile,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A downlight: bright straight down, falling to nothing at 90°.
    const DOWNLIGHT: &str = "IESNA:LM-63-2002
[TEST] a downlight
[MANUFAC] nobody
TILT=NONE
1 1000 1.0 7 1 1 2 0.1 0.1 0.0
1.0 1.0 20
0 15 30 45 60 75 90
0
800 760 600 400 200 50 0
";

    #[test]
    fn a_downlight_is_bright_along_its_axis_and_dark_to_the_side() {
        let p = read(DOWNLIGHT).expect("reads");
        assert_eq!(p.peak_candela, 800.0);
        assert_eq!(p.profile[0], 1.0);
        // 30° is about sample five: 600 of 800.
        let at = |deg: f64| p.profile[(deg / 180.0 * 31.0).round() as usize];
        assert!((at(30.0) - 0.75).abs() < 0.05, "{}", at(30.0));
        assert_eq!(at(120.0), 0.0, "nothing above the horizon");
    }

    #[test]
    fn horizontal_planes_are_averaged_and_the_multiplier_applies() {
        let text = "IESNA:LM-63-1995
TILT=NONE
1 -1 2.0 3 2 1 2 0 0 0
1.0 1.0 10
0 90 180
0 90
100 50 0
300 150 0
";
        let p = read(text).expect("reads");
        // (100 + 300) / 2 × 2.
        assert_eq!(p.peak_candela, 400.0);
        assert!((p.profile[(90.0f64 / 180.0 * 31.0).round() as usize] - 0.5).abs() < 0.05);
    }

    #[test]
    fn what_is_not_an_ies_file_is_refused() {
        assert_eq!(read("hello"), Err(IesError::NoTilt));
        assert_eq!(read("TILT=NONE\n1 2 3"), Err(IesError::Truncated));
        assert!(matches!(
            read("TILT=NONE\n1 2 x"),
            Err(IesError::BadNumber(_))
        ));
    }
}
