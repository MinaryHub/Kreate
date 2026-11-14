//! Fingers on a touch screen as the mouse and the view's navigation.
//!
//! The shell listens to a mouse: its tools take a left button pressed,
//! dragged and released. Android sends fingers alone, and [`Touches`]
//! turns them into what the shell understands:
//!
//! - **One finger** is the left button. A tap clicks where it lands; a
//!   finger that moves past a few pixels presses there and drags. Until it
//!   has moved, nothing is pressed — so a second finger arriving in that
//!   moment starts a gesture rather than a click.
//! - **Two fingers** turn the view as they move together and zoom as they
//!   pinch, about their middle. Two fingers tapped together and lifted
//!   without moving are a right click, where the first one touched.
//! - **Three fingers** (or more) move the view.
//!
//! Once a one-finger drag has started, more fingers are let go: the drag is
//! the tool's, and turning the view under it would move what it is
//! dragging against.

/// How far a finger moves, in pixels, before it is a drag and not a tap.
pub const SLOP: f64 = 12.0;

/// A finger, as the window reports it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Phase {
    Down,
    Move,
    Up,
}

/// What the fingers stand for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Out {
    /// The pointer is here.
    Pointer(f64, f64),
    /// The left button goes down, or up, where the pointer is.
    Press,
    Release,
    /// A right click where the pointer is.
    RightClick,
    /// Turn the view by this much, in pixels as a mouse drag would.
    Orbit(f64, f64),
    /// Move the view by this much, in pixels.
    Pan(f64, f64),
    /// Zoom by this natural log of the fingers' spread's change: above
    /// zero they spread apart, in.
    Zoom(f64),
}

#[derive(Clone, Debug, Default)]
enum Mode {
    #[default]
    Idle,
    /// One finger, which has (or has not yet) pressed the left button.
    One {
        id: u64,
        start: (f64, f64),
        pressed: bool,
    },
    /// Two fingers or more: a gesture. `moved` once any went past the slop;
    /// `most` the most fingers down at once; `first` where the first one
    /// touched.
    Gesture {
        moved: bool,
        most: usize,
        first: (f64, f64),
        travel: f64,
    },
}

/// The fingers down, and what they are doing.
#[derive(Clone, Debug, Default)]
pub struct Touches {
    fingers: Vec<(u64, (f64, f64))>,
    mode: Mode,
}

fn centre(points: &[(f64, f64)]) -> (f64, f64) {
    let n = points.len().max(1) as f64;
    let (x, y) = points
        .iter()
        .fold((0.0, 0.0), |(x, y), p| (x + p.0, y + p.1));
    (x / n, y / n)
}

fn spread(points: &[(f64, f64)]) -> f64 {
    let c = centre(points);
    points
        .iter()
        .map(|p| ((p.0 - c.0).powi(2) + (p.1 - c.1).powi(2)).sqrt())
        .sum::<f64>()
        / points.len().max(1) as f64
}

impl Touches {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fingers down now.
    pub fn count(&self) -> usize {
        self.fingers.len()
    }

    /// One finger's event, and what it stands for.
    pub fn touch(&mut self, id: u64, phase: Phase, at: (f64, f64)) -> Vec<Out> {
        match phase {
            Phase::Down => self.down(id, at),
            Phase::Move => self.moved(id, at),
            Phase::Up => self.up(id, at),
        }
    }

    fn down(&mut self, id: u64, at: (f64, f64)) -> Vec<Out> {
        self.fingers.retain(|f| f.0 != id);
        self.fingers.push((id, at));
        match self.mode.clone() {
            Mode::Idle => {
                self.mode = Mode::One {
                    id,
                    start: at,
                    pressed: false,
                };
                vec![Out::Pointer(at.0, at.1)]
            }
            // A drag under way keeps the button; the new finger is let go.
            Mode::One { pressed: true, .. } => Vec::new(),
            Mode::One { start, .. } => {
                self.mode = Mode::Gesture {
                    moved: false,
                    most: self.fingers.len(),
                    first: start,
                    travel: 0.0,
                };
                Vec::new()
            }
            Mode::Gesture {
                moved,
                most,
                first,
                travel,
            } => {
                self.mode = Mode::Gesture {
                    moved,
                    most: most.max(self.fingers.len()),
                    first,
                    travel,
                };
                Vec::new()
            }
        }
    }

    fn moved(&mut self, id: u64, at: (f64, f64)) -> Vec<Out> {
        let Some(i) = self.fingers.iter().position(|f| f.0 == id) else {
            return Vec::new();
        };
        let before: Vec<(f64, f64)> = self.fingers.iter().map(|f| f.1).collect();
        self.fingers[i].1 = at;
        let after: Vec<(f64, f64)> = self.fingers.iter().map(|f| f.1).collect();
        match &mut self.mode {
            Mode::One {
                id: one,
                start,
                pressed,
            } if *one == id => {
                let far = ((at.0 - start.0).powi(2) + (at.1 - start.1).powi(2)).sqrt() > SLOP;
                if *pressed {
                    vec![Out::Pointer(at.0, at.1)]
                } else if far {
                    // Pressed where it touched, then dragged to here.
                    *pressed = true;
                    vec![
                        Out::Pointer(start.0, start.1),
                        Out::Press,
                        Out::Pointer(at.0, at.1),
                    ]
                } else {
                    Vec::new()
                }
            }
            Mode::Gesture { moved, travel, .. } => {
                let (c0, c1) = (centre(&before), centre(&after));
                let shift = (c1.0 - c0.0, c1.1 - c0.1);
                let (s0, s1) = (spread(&before), spread(&after));
                *travel += shift.0.abs() + shift.1.abs() + (s1 - s0).abs();
                if *travel > SLOP {
                    *moved = true;
                }
                if !*moved {
                    return Vec::new();
                }
                let mut out = Vec::new();
                if self.fingers.len() >= 3 {
                    out.push(Out::Pan(shift.0, shift.1));
                } else {
                    out.push(Out::Orbit(shift.0, shift.1));
                    if s0 > 1.0 && s1 > 1.0 {
                        out.push(Out::Zoom((s1 / s0).ln()));
                    }
                }
                out
            }
            _ => Vec::new(),
        }
    }

    fn up(&mut self, id: u64, at: (f64, f64)) -> Vec<Out> {
        if !self.fingers.iter().any(|f| f.0 == id) {
            return Vec::new();
        }
        self.fingers.retain(|f| f.0 != id);
        match self.mode.clone() {
            Mode::One {
                id: one,
                start,
                pressed,
            } if one == id => {
                self.mode = Mode::Idle;
                if pressed {
                    vec![Out::Pointer(at.0, at.1), Out::Release]
                } else {
                    // A tap: a click where it landed.
                    vec![Out::Pointer(start.0, start.1), Out::Press, Out::Release]
                }
            }
            Mode::Gesture {
                moved, most, first, ..
            } if self.fingers.is_empty() => {
                self.mode = Mode::Idle;
                if !moved && most == 2 {
                    vec![Out::Pointer(first.0, first.1), Out::RightClick]
                } else {
                    Vec::new()
                }
            }
            _ => {
                if self.fingers.is_empty() {
                    self.mode = Mode::Idle;
                }
                Vec::new()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tap_clicks_where_it_landed() {
        let mut t = Touches::new();
        assert_eq!(
            t.touch(1, Phase::Down, (10.0, 20.0)),
            vec![Out::Pointer(10.0, 20.0)]
        );
        // A tremble within the slop is still a tap.
        assert!(t.touch(1, Phase::Move, (13.0, 22.0)).is_empty());
        assert_eq!(
            t.touch(1, Phase::Up, (13.0, 22.0)),
            vec![Out::Pointer(10.0, 20.0), Out::Press, Out::Release]
        );
        assert_eq!(t.count(), 0);
    }

    #[test]
    fn a_finger_that_moves_drags_from_where_it_touched() {
        let mut t = Touches::new();
        t.touch(1, Phase::Down, (0.0, 0.0));
        assert_eq!(
            t.touch(1, Phase::Move, (30.0, 0.0)),
            vec![Out::Pointer(0.0, 0.0), Out::Press, Out::Pointer(30.0, 0.0)]
        );
        assert_eq!(
            t.touch(1, Phase::Move, (40.0, 5.0)),
            vec![Out::Pointer(40.0, 5.0)]
        );
        // A second finger during the drag is let go.
        assert!(t.touch(2, Phase::Down, (100.0, 100.0)).is_empty());
        assert!(t.touch(2, Phase::Move, (150.0, 100.0)).is_empty());
        assert!(t.touch(2, Phase::Up, (150.0, 100.0)).is_empty());
        assert_eq!(
            t.touch(1, Phase::Up, (40.0, 5.0)),
            vec![Out::Pointer(40.0, 5.0), Out::Release]
        );
    }

    #[test]
    fn two_fingers_turn_and_pinch_and_three_move_the_view() {
        let mut t = Touches::new();
        t.touch(1, Phase::Down, (0.0, 0.0));
        t.touch(2, Phase::Down, (100.0, 0.0));
        // Both along by 20, one after the other: the middle moves 10 each.
        let mut orbit = (0.0, 0.0);
        for (id, x) in [(1, 20.0), (2, 120.0)] {
            for o in t.touch(id, Phase::Move, (x, 0.0)) {
                if let Out::Orbit(dx, dy) = o {
                    orbit.0 += dx;
                    orbit.1 += dy;
                }
            }
        }
        assert!((orbit.0 - 20.0).abs() < 1e-9 && orbit.1 == 0.0, "{orbit:?}");
        // Spreading apart zooms in by the spread's ratio.
        let out = t.touch(2, Phase::Move, (220.0, 0.0));
        let zoom: f64 = out
            .iter()
            .filter_map(|o| if let Out::Zoom(z) = o { Some(*z) } else { None })
            .sum();
        assert!((zoom - 2f64.ln()).abs() < 1e-9, "{out:?}");
        // A third finger: the fingers' middle moving pans.
        t.touch(3, Phase::Down, (100.0, 100.0));
        let out = t.touch(3, Phase::Move, (100.0, 130.0));
        assert!(
            matches!(out.as_slice(), [Out::Pan(x, y)] if x.abs() < 1e-9 && (y - 10.0).abs() < 1e-9),
            "{out:?}"
        );
        for id in [1, 2, 3] {
            assert!(t.touch(id, Phase::Up, (0.0, 0.0)).is_empty());
        }
        // And the next finger is a fresh one.
        assert_eq!(
            t.touch(4, Phase::Down, (5.0, 5.0)),
            vec![Out::Pointer(5.0, 5.0)]
        );
    }

    #[test]
    fn two_fingers_tapped_are_a_right_click() {
        let mut t = Touches::new();
        t.touch(1, Phase::Down, (10.0, 10.0));
        t.touch(2, Phase::Down, (60.0, 10.0));
        assert!(t.touch(2, Phase::Up, (60.0, 10.0)).is_empty());
        assert_eq!(
            t.touch(1, Phase::Up, (10.0, 10.0)),
            vec![Out::Pointer(10.0, 10.0), Out::RightClick]
        );
    }
}
