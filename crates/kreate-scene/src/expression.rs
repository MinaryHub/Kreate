//! The little language a driver's expression is written in.
//!
//! Numbers, names, the arithmetic a person expects — `+ - * / %`, `**` for a
//! power, unary minus, parentheses — comparisons that answer one or zero, and
//! a fixed set of functions. Python's spelling where the two differ, because
//! that is what Blender's drivers are written in and what a person arriving
//! from there will type: `sin(frame / 10) * 2` means the same thing in both.
//!
//! # What it is, and firmly is not
//!
//! An expression and not a program. No assignment, no loops, no strings, and
//! nothing that reaches outside the numbers it is handed: a scene opened from
//! somewhere else evaluates its drivers the moment it loads, and a driver that
//! could do anything but arithmetic would be a file that could do anything at
//! all. That is the reason Blender asks before running a file's drivers, and
//! the reason this one does not need to.
//!
//! §6.2 names Rhai for this tier. It is not in the build — the registry here is
//! offline — and on inspection the tier needs a calculator with names more than
//! it needs a scripting language. When scripts arrive (Phase 9) they register
//! operators; an expression stays an expression.

use std::fmt;

/// What went wrong, for the driver's panel to say.
#[derive(Clone, Debug, PartialEq)]
pub enum ExprError {
    /// Nothing to evaluate.
    Empty,
    /// Malformed at this byte.
    Syntax { at: usize },
    /// A name that is neither a variable nor a constant.
    UnknownName(String),
    /// A function this language does not have.
    UnknownFunction(String),
    /// A function given the wrong number of arguments.
    Arguments { function: String, expected: usize },
    /// Came out as infinity or not a number — a division by zero, a square
    /// root of something negative. Refused rather than written into a
    /// transform, where it would put the object nowhere.
    NotFinite,
}

impl fmt::Display for ExprError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExprError::Empty => write!(f, "empty"),
            ExprError::Syntax { at } => write!(f, "syntax error at {at}"),
            ExprError::UnknownName(n) => write!(f, "unknown name '{n}'"),
            ExprError::UnknownFunction(n) => write!(f, "unknown function '{n}'"),
            ExprError::Arguments { function, expected } => {
                write!(f, "{function} takes {expected}")
            }
            ExprError::NotFinite => write!(f, "not a finite number"),
        }
    }
}

/// A parsed expression.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Number(f64),
    Name(String),
    Negate(Box<Expr>),
    Binary(Op, Box<Expr>, Box<Expr>),
    Call(String, Vec<Expr>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    Less,
    LessEq,
    Greater,
    GreaterEq,
    Equal,
    NotEqual,
}

/// The functions, with how many arguments each takes.
///
/// Radians throughout, as Python's `math` is; `radians` and `degrees` are
/// there for the rest.
pub const FUNCTIONS: &[(&str, usize)] = &[
    ("sin", 1),
    ("cos", 1),
    ("tan", 1),
    ("asin", 1),
    ("acos", 1),
    ("atan", 1),
    ("atan2", 2),
    ("sqrt", 1),
    ("abs", 1),
    ("floor", 1),
    ("ceil", 1),
    ("round", 1),
    ("exp", 1),
    ("log", 1),
    ("pow", 2),
    ("min", 2),
    ("max", 2),
    ("clamp", 3),
    ("lerp", 3),
    ("radians", 1),
    ("degrees", 1),
];

/// Parses `text`.
pub fn parse(text: &str) -> Result<Expr, ExprError> {
    let mut p = Parser {
        bytes: text.as_bytes(),
        at: 0,
        depth: 0,
    };
    p.skip_space();
    if p.at >= p.bytes.len() {
        return Err(ExprError::Empty);
    }
    let e = p.comparison()?;
    p.skip_space();
    if p.at < p.bytes.len() {
        return Err(ExprError::Syntax { at: p.at });
    }
    Ok(e)
}

/// Parses and evaluates `text`, looking names up with `name`.
///
/// `pi`, `tau` and `e` are answered here unless `name` answers them first — a
/// variable called `e` is the author's, and should win.
pub fn evaluate(text: &str, name: &dyn Fn(&str) -> Option<f64>) -> Result<f64, ExprError> {
    parse(text)?.evaluate(name)
}

impl Expr {
    pub fn evaluate(&self, name: &dyn Fn(&str) -> Option<f64>) -> Result<f64, ExprError> {
        let v = self.value(name)?;
        if v.is_finite() {
            Ok(v)
        } else {
            Err(ExprError::NotFinite)
        }
    }

    /// Every name the expression reads, in order, once each.
    pub fn names(&self) -> Vec<&str> {
        let mut out = Vec::new();
        self.collect(&mut out);
        out
    }

    fn collect<'a>(&'a self, out: &mut Vec<&'a str>) {
        match self {
            Expr::Number(_) => {}
            Expr::Name(n) => {
                if !out.contains(&n.as_str()) {
                    out.push(n);
                }
            }
            Expr::Negate(e) => e.collect(out),
            Expr::Binary(_, a, b) => {
                a.collect(out);
                b.collect(out);
            }
            Expr::Call(_, args) => args.iter().for_each(|a| a.collect(out)),
        }
    }

    fn value(&self, name: &dyn Fn(&str) -> Option<f64>) -> Result<f64, ExprError> {
        Ok(match self {
            Expr::Number(v) => *v,
            Expr::Name(n) => match name(n) {
                Some(v) => v,
                None => match n.as_str() {
                    "pi" => std::f64::consts::PI,
                    "tau" => std::f64::consts::TAU,
                    "e" => std::f64::consts::E,
                    _ => return Err(ExprError::UnknownName(n.clone())),
                },
            },
            Expr::Negate(e) => -e.value(name)?,
            Expr::Binary(op, a, b) => {
                let (a, b) = (a.value(name)?, b.value(name)?);
                let truth = |t: bool| if t { 1.0 } else { 0.0 };
                match op {
                    Op::Add => a + b,
                    Op::Sub => a - b,
                    Op::Mul => a * b,
                    Op::Div => a / b,
                    // Python's: the sign of the divisor, so `-1 % 24` is 23 —
                    // which is what a looping frame counter wants.
                    Op::Rem => a - b * (a / b).floor(),
                    Op::Pow => a.powf(b),
                    Op::Less => truth(a < b),
                    Op::LessEq => truth(a <= b),
                    Op::Greater => truth(a > b),
                    Op::GreaterEq => truth(a >= b),
                    Op::Equal => truth(a == b),
                    Op::NotEqual => truth(a != b),
                }
            }
            Expr::Call(f, args) => {
                let Some((_, expected)) = FUNCTIONS.iter().find(|(n, _)| n == f) else {
                    return Err(ExprError::UnknownFunction(f.clone()));
                };
                if args.len() != *expected {
                    return Err(ExprError::Arguments {
                        function: f.clone(),
                        expected: *expected,
                    });
                }
                let v: Vec<f64> = args
                    .iter()
                    .map(|a| a.value(name))
                    .collect::<Result<_, _>>()?;
                match f.as_str() {
                    "sin" => v[0].sin(),
                    "cos" => v[0].cos(),
                    "tan" => v[0].tan(),
                    "asin" => v[0].asin(),
                    "acos" => v[0].acos(),
                    "atan" => v[0].atan(),
                    "atan2" => v[0].atan2(v[1]),
                    "sqrt" => v[0].sqrt(),
                    "abs" => v[0].abs(),
                    "floor" => v[0].floor(),
                    "ceil" => v[0].ceil(),
                    "round" => v[0].round(),
                    "exp" => v[0].exp(),
                    "log" => v[0].ln(),
                    "pow" => v[0].powf(v[1]),
                    "min" => v[0].min(v[1]),
                    "max" => v[0].max(v[1]),
                    "clamp" => v[0].max(v[1]).min(v[2]),
                    "lerp" => v[0] + (v[1] - v[0]) * v[2],
                    "radians" => v[0].to_radians(),
                    "degrees" => v[0].to_degrees(),
                    _ => unreachable!("every entry of FUNCTIONS has an arm"),
                }
            }
        })
    }
}

/// How deep parentheses and calls may nest: far past anything a person types,
/// and short of a stack overflow on a file that nests ten thousand deep.
const MAX_DEPTH: usize = 64;

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
    depth: usize,
}

impl Parser<'_> {
    fn skip_space(&mut self) {
        while self
            .bytes
            .get(self.at)
            .is_some_and(|b| b.is_ascii_whitespace())
        {
            self.at += 1;
        }
    }

    fn peek(&mut self) -> Option<u8> {
        self.skip_space();
        self.bytes.get(self.at).copied()
    }

    fn eat(&mut self, token: &str) -> bool {
        self.skip_space();
        if self.bytes[self.at..].starts_with(token.as_bytes()) {
            self.at += token.len();
            true
        } else {
            false
        }
    }

    fn comparison(&mut self) -> Result<Expr, ExprError> {
        let left = self.sum()?;
        // Longest first, so `<=` is not read as `<` and a stray `=`.
        for (token, op) in [
            ("<=", Op::LessEq),
            (">=", Op::GreaterEq),
            ("==", Op::Equal),
            ("!=", Op::NotEqual),
            ("<", Op::Less),
            (">", Op::Greater),
        ] {
            if self.eat(token) {
                let right = self.sum()?;
                return Ok(Expr::Binary(op, Box::new(left), Box::new(right)));
            }
        }
        Ok(left)
    }

    fn sum(&mut self) -> Result<Expr, ExprError> {
        let mut left = self.product()?;
        loop {
            let op = match self.peek() {
                Some(b'+') => Op::Add,
                Some(b'-') => Op::Sub,
                _ => return Ok(left),
            };
            self.at += 1;
            let right = self.product()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
    }

    fn product(&mut self) -> Result<Expr, ExprError> {
        let mut left = self.unary()?;
        loop {
            let op = match self.peek() {
                // `**` is a power, not two multiplications.
                Some(b'*') if !self.bytes[self.at..].starts_with(b"**") => Op::Mul,
                Some(b'/') => Op::Div,
                Some(b'%') => Op::Rem,
                _ => return Ok(left),
            };
            self.at += 1;
            let right = self.unary()?;
            left = Expr::Binary(op, Box::new(left), Box::new(right));
        }
    }

    fn unary(&mut self) -> Result<Expr, ExprError> {
        if self.peek() == Some(b'-') {
            self.at += 1;
            self.deeper()?;
            let e = self.unary()?;
            self.depth -= 1;
            return Ok(Expr::Negate(Box::new(e)));
        }
        if self.peek() == Some(b'+') {
            self.at += 1;
            self.deeper()?;
            let e = self.unary()?;
            self.depth -= 1;
            return Ok(e);
        }
        self.power()
    }

    /// Right-associative, and binding tighter than unary minus on its left as
    /// Python's does: `-2 ** 2` is -4 and `2 ** -1` is a half.
    fn power(&mut self) -> Result<Expr, ExprError> {
        let base = self.atom()?;
        if self.eat("**") {
            self.deeper()?;
            let exponent = self.unary()?;
            self.depth -= 1;
            return Ok(Expr::Binary(Op::Pow, Box::new(base), Box::new(exponent)));
        }
        Ok(base)
    }

    fn deeper(&mut self) -> Result<(), ExprError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            Err(ExprError::Syntax { at: self.at })
        } else {
            Ok(())
        }
    }

    fn atom(&mut self) -> Result<Expr, ExprError> {
        match self.peek() {
            Some(b'(') => {
                self.at += 1;
                self.deeper()?;
                let e = self.comparison()?;
                self.depth -= 1;
                if !self.eat(")") {
                    return Err(ExprError::Syntax { at: self.at });
                }
                Ok(e)
            }
            Some(b) if b.is_ascii_digit() || b == b'.' => self.number(),
            Some(b) if b.is_ascii_alphabetic() || b == b'_' => {
                let start = self.at;
                while self
                    .bytes
                    .get(self.at)
                    .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_')
                {
                    self.at += 1;
                }
                let name = String::from_utf8_lossy(&self.bytes[start..self.at]).into_owned();
                if self.eat("(") {
                    self.deeper()?;
                    let mut args = Vec::new();
                    if !self.eat(")") {
                        loop {
                            args.push(self.comparison()?);
                            if self.eat(")") {
                                break;
                            }
                            if !self.eat(",") {
                                return Err(ExprError::Syntax { at: self.at });
                            }
                        }
                    }
                    self.depth -= 1;
                    return Ok(Expr::Call(name, args));
                }
                Ok(Expr::Name(name))
            }
            _ => Err(ExprError::Syntax { at: self.at }),
        }
    }

    fn number(&mut self) -> Result<Expr, ExprError> {
        let start = self.at;
        while self
            .bytes
            .get(self.at)
            .is_some_and(|b| b.is_ascii_digit() || *b == b'.')
        {
            self.at += 1;
        }
        // An exponent: `1e-3`. Only when a digit follows, so `2e` stays a
        // syntax error rather than a silent two.
        if matches!(self.bytes.get(self.at), Some(b'e' | b'E')) {
            let mut look = self.at + 1;
            if matches!(self.bytes.get(look), Some(b'+' | b'-')) {
                look += 1;
            }
            if self.bytes.get(look).is_some_and(u8::is_ascii_digit) {
                self.at = look;
                while self.bytes.get(self.at).is_some_and(u8::is_ascii_digit) {
                    self.at += 1;
                }
            }
        }
        std::str::from_utf8(&self.bytes[start..self.at])
            .ok()
            .and_then(|t| t.parse().ok())
            .map(Expr::Number)
            .ok_or(ExprError::Syntax { at: start })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eval(text: &str) -> Result<f64, ExprError> {
        evaluate(text, &|n| match n {
            "frame" => Some(12.0),
            "var" => Some(0.5),
            _ => None,
        })
    }

    #[test]
    fn arithmetic_binds_the_way_python_does() {
        assert_eq!(eval("1 + 2 * 3"), Ok(7.0));
        assert_eq!(eval("(1 + 2) * 3"), Ok(9.0));
        assert_eq!(eval("2 ** 3 ** 2"), Ok(512.0), "power is right-associative");
        assert_eq!(
            eval("-2 ** 2"),
            Ok(-4.0),
            "power binds tighter than a leading minus"
        );
        assert_eq!(eval("2 ** -1"), Ok(0.5));
        assert_eq!(
            eval("10 - 4 - 3"),
            Ok(3.0),
            "subtraction is left-associative"
        );
        assert_eq!(
            eval("-1 % 24"),
            Ok(23.0),
            "remainder takes the divisor's sign"
        );
        assert_eq!(eval("1e-3 * 1000"), Ok(1.0));
    }

    #[test]
    fn names_functions_and_constants() {
        assert_eq!(eval("frame / 4 + var"), Ok(3.5));
        assert!((eval("sin(pi / 2)").unwrap() - 1.0).abs() < 1e-12);
        assert_eq!(eval("clamp(frame, 0, 10)"), Ok(10.0));
        assert_eq!(eval("max(var, min(3, 2))"), Ok(2.0));
        assert_eq!(eval("lerp(0, 10, var)"), Ok(5.0));
        assert_eq!(eval("frame > 10"), Ok(1.0));
        assert_eq!(eval("(frame <= 10) * 5"), Ok(0.0));
        assert!((eval("degrees(radians(30))").unwrap() - 30.0).abs() < 1e-9);
    }

    #[test]
    fn a_variable_called_e_is_the_authors() {
        assert_eq!(evaluate("e", &|n| (n == "e").then_some(2.0)), Ok(2.0));
        assert_eq!(evaluate("e", &|_| None), Ok(std::f64::consts::E));
    }

    #[test]
    fn mistakes_are_named() {
        assert_eq!(eval(""), Err(ExprError::Empty));
        assert_eq!(eval("1 +"), Err(ExprError::Syntax { at: 3 }));
        assert_eq!(eval("(1"), Err(ExprError::Syntax { at: 2 }));
        assert_eq!(eval("1 2"), Err(ExprError::Syntax { at: 2 }));
        assert_eq!(
            eval("height * 2"),
            Err(ExprError::UnknownName("height".into()))
        );
        assert_eq!(
            eval("sinh(1)"),
            Err(ExprError::UnknownFunction("sinh".into()))
        );
        assert_eq!(
            eval("min(1)"),
            Err(ExprError::Arguments {
                function: "min".into(),
                expected: 2
            })
        );
        assert_eq!(eval("1 / 0"), Err(ExprError::NotFinite));
        assert_eq!(eval("sqrt(-1)"), Err(ExprError::NotFinite));
        assert_eq!(eval("2e"), Err(ExprError::Syntax { at: 1 }));
    }

    #[test]
    fn deep_nesting_is_refused_rather_than_overflowing() {
        let deep = format!("{}1{}", "(".repeat(10_000), ")".repeat(10_000));
        assert!(matches!(eval(&deep), Err(ExprError::Syntax { .. })));
        for sign in ["-", "+"] {
            let signs = format!("{}1", sign.repeat(10_000));
            assert!(matches!(eval(&signs), Err(ExprError::Syntax { .. })));
        }
        let fine = format!("{}1{}", "(".repeat(30), ")".repeat(30));
        assert_eq!(eval(&fine), Ok(1.0));
    }

    #[test]
    fn the_names_an_expression_reads() {
        let e = parse("sin(frame) * var + frame").unwrap();
        assert_eq!(e.names(), vec!["frame", "var"]);
    }
}
