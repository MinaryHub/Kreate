//! Arithmetic and units in a numeric field.
//!
//! §2.2 asks for `2*3.5`, `1/8`, `2m` and `30deg` where a number is expected.
//! This is why: the numbers a modeller wants are usually the answer to a small
//! sum. A hole eight of the way along a 1.4 metre panel is `1.4/8`, and the
//! alternative to typing that is a calculator and a transcription error.
//!
//! # What it is, and firmly is not
//!
//! Four operators, unary minus, parentheses, and a unit suffix. No variables, no
//! functions, no assignment. A field that evaluates a *language* is a field that
//! has to answer questions about scope and side effects, and none of that helps
//! anyone place a vertex. §6.2 puts the real expression language — drivers and
//! constraints — behind Rhai, in its own tier, for exactly that reason.
//!
//! # Units
//!
//! A suffix converts into the field's own unit rather than being carried along.
//! Kreate's length unit is the metre, so `2cm` is `0.02`; its angles are radians
//! at the operator and degrees at the keyboard, so `30deg` and plain `30` in an
//! angle field mean the same thing.
//!
//! A unit that does not match the field is refused rather than ignored. Typing
//! `30deg` into a distance is a mistake, and silently treating it as thirty
//! metres would move something a very long way.

/// What a field is asking for, so a suffix can be checked against it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    /// Metres. The bare number is already metres.
    Length,
    /// Radians at the operator; a bare number is read as degrees, because that
    /// is what someone typing into an angle field means.
    Angle,
    /// A ratio. No suffix makes sense.
    Factor,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ExprError {
    /// Nothing to evaluate yet. Not an error the user should see: it is what a
    /// half-typed `2*` looks like, and the pointer keeps driving until it parses.
    Incomplete,
    /// A suffix the field cannot use.
    WrongUnit,
    /// Malformed.
    Bad,
}

/// Evaluates an expression in the given unit.
///
/// `Err(Incomplete)` for anything still being typed, which the caller treats as
/// "the pointer is still in charge" rather than as a failure.
///
/// A bare number in a length field is in the scene's unit — `5` with the
/// scene in millimetres is five millimetres — as on a drawing, where the
/// unit is written once in the corner rather than after every dimension.
pub fn eval(text: &str, unit: Unit) -> Result<f64, ExprError> {
    eval_in(text, unit, crate::display_length_unit())
}

/// [`eval`], with the unit a bare length is in given rather than the
/// scene's.
pub fn eval_in(text: &str, unit: Unit, bare_length: crate::LengthUnit) -> Result<f64, ExprError> {
    let text = text.trim();
    if text.is_empty() {
        return Err(ExprError::Incomplete);
    }

    let (body, suffix) = split_suffix(text);
    let scale = match (suffix, unit) {
        (None, Unit::Angle) => std::f64::consts::PI / 180.0,
        (None, Unit::Length) => bare_length.metres(),
        (None, _) => 1.0,
        (Some("m"), Unit::Length) => 1.0,
        (Some("cm"), Unit::Length) => 0.01,
        (Some("mm"), Unit::Length) => 0.001,
        (Some("km"), Unit::Length) => 1000.0,
        (Some("in"), Unit::Length) => 0.0254,
        (Some("ft"), Unit::Length) => 0.3048,
        (Some("deg" | "d" | "°"), Unit::Angle) => std::f64::consts::PI / 180.0,
        (Some("rad" | "r"), Unit::Angle) => 1.0,
        // A suffix the field cannot use. Refused rather than ignored: `30deg`
        // in a distance is a mistake, and reading it as thirty metres would move
        // something a very long way.
        (Some(_), _) => return Err(ExprError::WrongUnit),
    };

    let mut parser = Parser {
        bytes: body.as_bytes(),
        at: 0,
    };
    let value = parser.expression()?;
    parser.skip_space();
    if parser.at != parser.bytes.len() {
        return Err(ExprError::Bad);
    }
    if !value.is_finite() {
        // A division by zero, or an overflow. Refused rather than passed on: an
        // infinite translation is not a thing to apply and then undo.
        return Err(ExprError::Bad);
    }
    Ok(value * scale)
}

/// Splits a trailing alphabetic unit off the end.
///
/// Only *trailing* letters, which is what keeps an exponent intact: `2e3` ends
/// in a digit, so nothing is taken from it and the number parser sees the whole
/// thing. A guard against `e` specifically was written here first and was dead
/// code — it could only ever fire on `2e`, which is malformed either way.
fn split_suffix(text: &str) -> (&str, Option<&str>) {
    let boundary = text
        .char_indices()
        .rev()
        .take_while(|(_, c)| c.is_alphabetic() || *c == '°')
        .last()
        .map(|(i, _)| i);
    let Some(i) = boundary else {
        return (text, None);
    };
    (text[..i].trim_end(), Some(&text[i..]))
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn skip_space(&mut self) {
        while self.bytes.get(self.at) == Some(&b' ') {
            self.at += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_space();
        self.bytes.get(self.at).copied()
    }

    /// `term (('+' | '-') term)*`
    fn expression(&mut self) -> Result<f64, ExprError> {
        let mut value = self.term()?;
        while let Some(op) = self.peek() {
            if op != b'+' && op != b'-' {
                break;
            }
            self.at += 1;
            let rhs = self.term()?;
            value = if op == b'+' { value + rhs } else { value - rhs };
        }
        Ok(value)
    }

    /// `unary (('*' | '/') unary)*`
    fn term(&mut self) -> Result<f64, ExprError> {
        let mut value = self.unary()?;
        while let Some(op) = self.peek() {
            if op != b'*' && op != b'/' {
                break;
            }
            self.at += 1;
            let rhs = self.unary()?;
            value = if op == b'*' { value * rhs } else { value / rhs };
        }
        Ok(value)
    }

    fn unary(&mut self) -> Result<f64, ExprError> {
        match self.peek() {
            Some(b'-') => {
                self.at += 1;
                Ok(-self.unary()?)
            }
            Some(b'+') => {
                self.at += 1;
                self.unary()
            }
            _ => self.atom(),
        }
    }

    fn atom(&mut self) -> Result<f64, ExprError> {
        match self.peek() {
            None => Err(ExprError::Incomplete),
            Some(b'(') => {
                self.at += 1;
                let value = self.expression()?;
                match self.peek() {
                    Some(b')') => {
                        self.at += 1;
                        Ok(value)
                    }
                    // An unclosed bracket is something still being typed, not a
                    // mistake. Reported as incomplete so the field keeps taking
                    // input instead of flashing an error at every keystroke.
                    None => Err(ExprError::Incomplete),
                    Some(_) => Err(ExprError::Bad),
                }
            }
            Some(c) if c.is_ascii_digit() || c == b'.' => self.number(),
            Some(_) => Err(ExprError::Bad),
        }
    }

    fn number(&mut self) -> Result<f64, ExprError> {
        let start = self.at;
        while let Some(c) = self.bytes.get(self.at) {
            let c = *c;
            if c.is_ascii_digit() || c == b'.' {
                self.at += 1;
            } else if (c == b'e' || c == b'E')
                && self
                    .bytes
                    .get(self.at + 1)
                    .is_some_and(|n| n.is_ascii_digit() || *n == b'-' || *n == b'+')
            {
                self.at += 2;
            } else {
                break;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.at]).map_err(|_| ExprError::Bad)?;
        // A lone `.` is a number someone has started typing.
        if text == "." {
            return Err(ExprError::Incomplete);
        }
        text.parse().map_err(|_| ExprError::Bad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn length(text: &str) -> Result<f64, ExprError> {
        eval_in(text, Unit::Length, crate::LengthUnit::Metres)
    }

    #[test]
    fn a_plain_number_is_itself() {
        assert_eq!(length("2"), Ok(2.0));
        assert_eq!(length("2.5"), Ok(2.5));
        assert_eq!(length("-3"), Ok(-3.0));
        assert_eq!(length(".5"), Ok(0.5));
    }

    #[test]
    fn the_four_operators_work_and_bind_correctly() {
        assert_eq!(length("2*3.5"), Ok(7.0));
        assert_eq!(length("1/8"), Ok(0.125));
        assert_eq!(length("1+2*3"), Ok(7.0), "multiplication binds tighter");
        assert_eq!(length("(1+2)*3"), Ok(9.0), "brackets override it");
        assert_eq!(length("10-2-3"), Ok(5.0), "subtraction is left to right");
        assert_eq!(length("100/10/2"), Ok(5.0), "and so is division");
    }

    #[test]
    fn unary_minus_works_where_it_appears() {
        assert_eq!(length("-2*3"), Ok(-6.0));
        assert_eq!(length("2*-3"), Ok(-6.0));
        assert_eq!(length("--2"), Ok(2.0));
        assert_eq!(length("-(1+1)"), Ok(-2.0));
    }

    #[test]
    fn spaces_are_ignored() {
        assert_eq!(length(" 2 * ( 3 + 1 ) "), Ok(8.0));
    }

    #[test]
    fn length_units_convert_to_metres() {
        assert_eq!(length("2m"), Ok(2.0));
        assert_eq!(length("50cm"), Ok(0.5));
        assert_eq!(length("2.5km"), Ok(2500.0));
        assert!((length("7mm").unwrap() - 0.007).abs() < 1e-12);
    }

    #[test]
    fn a_bare_length_is_in_the_scenes_unit_and_a_suffix_overrides_it() {
        use crate::LengthUnit;
        let mm = |t: &str| eval_in(t, Unit::Length, LengthUnit::Millimetres).unwrap();
        assert!((mm("5") - 0.005).abs() < 1e-12);
        assert!((mm("2m") - 2.0).abs() < 1e-12);
        assert!((mm("1in") - 0.0254).abs() < 1e-12);
        // Angles are not lengths: a bare one stays degrees.
        let deg = eval_in("90", Unit::Angle, LengthUnit::Millimetres).unwrap();
        assert!((deg - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn a_unit_applies_to_the_whole_expression() {
        // `10+5cm` is fifteen centimetres, not ten metres and five centimetres.
        // Stated because both readings are defensible and only one can be true:
        // the suffix names the unit the *field* is being given, which is how a
        // dimension is written on a drawing.
        assert!((length("10+5cm").unwrap() - 0.15).abs() < 1e-12);
    }

    #[test]
    fn an_angle_field_reads_a_bare_number_as_degrees() {
        // Nobody types radians. The operator takes radians, the keyboard gives
        // degrees, and the conversion belongs here rather than in every caller.
        let ninety = eval("90", Unit::Angle).unwrap();
        assert!((ninety - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        let same = eval("90deg", Unit::Angle).unwrap();
        assert!((same - ninety).abs() < 1e-12);
        let radians = eval("1rad", Unit::Angle).unwrap();
        assert!((radians - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_degree_sign_is_a_degree() {
        let a = eval("45°", Unit::Angle).unwrap();
        assert!((a - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
    }

    #[test]
    fn a_unit_the_field_cannot_use_is_refused() {
        // Refused rather than ignored. `30deg` in a distance field is a mistake,
        // and reading it as thirty metres would move something a very long way
        // for a reason the user could not see.
        assert_eq!(length("30deg"), Err(ExprError::WrongUnit));
        assert_eq!(eval("2m", Unit::Angle), Err(ExprError::WrongUnit));
        assert_eq!(eval("2m", Unit::Factor), Err(ExprError::WrongUnit));
        assert_eq!(eval("30deg", Unit::Factor), Err(ExprError::WrongUnit));
    }

    #[test]
    fn a_factor_field_takes_a_bare_number_unchanged() {
        assert_eq!(eval("2", Unit::Factor), Ok(2.0));
        assert_eq!(eval("1/4", Unit::Factor), Ok(0.25));
    }

    #[test]
    fn half_typed_input_is_incomplete_rather_than_wrong() {
        // The distinction that lets the pointer keep driving while someone types.
        // Reported as an error, every keystroke of `2*3` would flash a complaint
        // and the field would be unusable.
        assert_eq!(length(""), Err(ExprError::Incomplete));
        assert_eq!(length("2*"), Err(ExprError::Incomplete));
        assert_eq!(length("-"), Err(ExprError::Incomplete));
        assert_eq!(length("("), Err(ExprError::Incomplete));
        assert_eq!(length("(1+2"), Err(ExprError::Incomplete));
        assert_eq!(length("."), Err(ExprError::Incomplete));
    }

    #[test]
    fn nonsense_is_refused() {
        assert_eq!(length("2)"), Err(ExprError::Bad));
        assert_eq!(length("2 3"), Err(ExprError::Bad));
        assert_eq!(length("banana"), Err(ExprError::WrongUnit));
        assert_eq!(length("2%"), Err(ExprError::Bad));
    }

    #[test]
    fn a_division_by_zero_is_refused_rather_than_infinite() {
        // An infinite translation is not something to apply and then undo.
        assert_eq!(length("1/0"), Err(ExprError::Bad));
        assert_eq!(length("-1/0"), Err(ExprError::Bad));
        assert_eq!(length("0/0"), Err(ExprError::Bad));
    }

    #[test]
    fn an_exponent_is_not_mistaken_for_a_unit() {
        // `2e3` ends in a *digit*, so the suffix split takes nothing from it and
        // the number parser sees it whole. Worth pinning: scientific notation in
        // a coordinate field is how someone types a very small offset, and a
        // suffix rule that reached into the middle of a number would break it.
        assert_eq!(length("2e3"), Ok(2000.0));
        assert_eq!(length("1.5e-2"), Ok(0.015));
    }

    #[test]
    fn a_long_expression_does_not_run_away() {
        // Deeply nested brackets recurse. A field is user input, so the depth is
        // whatever someone leant on the bracket key for; this checks the parser
        // returns rather than misbehaving.
        let deep = format!("{}1{}", "(".repeat(64), ")".repeat(64));
        assert_eq!(length(&deep), Ok(1.0));
    }
}
