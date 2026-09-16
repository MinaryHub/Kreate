//! Reflected operator parameters.
//!
//! A leaf crate with no dependencies. Deliberately separate from `kreate-ops`
//! so the UI can read an operator's schema and build a parameter map without
//! being able to reach the operator implementations or the document: it emits
//! `(op_id, params)` and the application dispatches.
//!
//! Every operator declares its parameters as a plain Rust struct with
//! `#[derive(OpParams)]`. That single declaration then drives, with no further
//! work per operator:
//!
//! - the adjust-last-operation panel,
//! - the properties editor,
//! - the scripting binding and "copy as script" in the info log,
//! - the keymap payload (a binding can override any parameter),
//! - the undo record,
//! - the i18n keys for names and tooltips,
//! - the golden-test harness.
//!
//! The cost is that parameters must be runtime-typed and serialisable rather
//! than "just take a `&mut Mesh` and a `Vec3`". That overhead is real, and it is
//! worth paying precisely because it is *derived*: an operator author writes an
//! ordinary struct and gets all of the above. Retrofitting reflection onto two
//! hundred hand-written operators is the kind of task that kills a project's
//! second year.
//!
//! # Property paths
//!
//! The other half of reflection, and a different question: not "what parameters
//! does this operator take" but "which property of which datablock does this
//! curve drive". [`PropertyPath`] is that, and it lives here for the same
//! reason the schema does — the interface has to be able to hold one without
//! being able to reach the document. See [`path`] for why the target is an id
//! and the property is a name.

mod path;

pub use path::{AnimatableDesc, DatablockKind, PathError, PropertyPath, Uid};

use std::fmt::Write as _;

/// A parameter's runtime value.
#[derive(Clone, Debug, PartialEq)]
pub enum PropertyValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Vec3([f64; 3]),
    Text(String),
    /// Index into the corresponding [`PropertyKind::Enum`] variant list.
    Enum(u32),
    /// A list of element indices.
    ///
    /// The parameter a resolved selection is: box select, lasso select, a script
    /// picking elements by name, a restored selection from a file. All of those
    /// are "these specific elements", and the alternative — passing the *screen
    /// rectangle* and re-resolving it — is worse in every direction. It makes the
    /// operator depend on a camera, it cannot be replayed headlessly, and
    /// adjusting it after the view has moved silently selects something else.
    Ints(Vec<u32>),
}

impl PropertyValue {
    pub fn kind_name(&self) -> &'static str {
        match self {
            PropertyValue::Bool(_) => "bool",
            PropertyValue::Int(_) => "int",
            PropertyValue::Float(_) => "float",
            PropertyValue::Vec3(_) => "vec3",
            PropertyValue::Text(_) => "text",
            PropertyValue::Enum(_) => "enum",
            PropertyValue::Ints(_) => "ints",
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            PropertyValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_int(&self) -> Option<i64> {
        match self {
            PropertyValue::Int(v) => Some(*v),
            PropertyValue::Enum(v) => Some(*v as i64),
            _ => None,
        }
    }

    pub fn as_float(&self) -> Option<f64> {
        match self {
            PropertyValue::Float(v) => Some(*v),
            // An integer literal is a perfectly good float parameter value; a
            // script writing `1` where `1.0` was expected should not fail.
            PropertyValue::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    pub fn as_vec3(&self) -> Option<[f64; 3]> {
        match self {
            PropertyValue::Vec3(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            PropertyValue::Text(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_ints(&self) -> Option<&[u32]> {
        match self {
            PropertyValue::Ints(v) => Some(v),
            _ => None,
        }
    }
}

/// What kind of value a parameter holds, and what the UI should offer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PropertyKind {
    Bool,
    Int,
    Float,
    Vec3,
    /// Three numbers that are a colour rather than a direction.
    ///
    /// Carried as a [`PropertyValue::Vec3`], so nothing about storage,
    /// scripting, undo or any file already written changes. What changes is
    /// the widget: a swatch that opens a picker, instead of three spinners
    /// labelled X, Y and Z. The labels were not merely unhelpful — a base
    /// colour has no axes, and the axis colours the editor paints down the
    /// side of each field meant something about direction.
    Color,
    Text,
    Enum {
        variants: &'static [&'static str],
    },
    /// A list of element indices, with no widget of its own.
    ///
    /// The property editor shows a count rather than the values. A thousand
    /// spinners is not a way to edit a selection, and the thing that *does* edit
    /// one is the viewport.
    Ints,
}

/// The physical quantity a numeric parameter represents.
///
/// Drives display (`0.05 m` versus `5 cm`), the drag-scrub step, and what the
/// numeric-input parser accepts: typing `30deg` into a rotation field or `2m`
/// into a length field must work, and must mean different things.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Unit {
    /// A bare number.
    #[default]
    None,
    /// Metres.
    Length,
    /// Radians internally, shown in degrees.
    Angle,
    /// A 0..1 proportion, shown as a percentage.
    Factor,
    /// A count.
    Count,
}

/// Everything the UI, the scripting layer and the i18n system need to know about
/// one parameter.
#[derive(Clone, Copy, Debug)]
pub struct PropertyDesc {
    pub name: &'static str,
    pub kind: PropertyKind,
    /// Hard limits. Values outside are clamped, including from scripts.
    pub min: Option<f64>,
    pub max: Option<f64>,
    /// The range a drag-scrub covers. Typing a value outside it is still allowed
    /// as long as the hard limits permit — a slider that cannot be exceeded is a
    /// common and infuriating limitation in modelling tools.
    pub soft_min: Option<f64>,
    pub soft_max: Option<f64>,
    pub step: Option<f64>,
    pub unit: Unit,
    /// Doc comment on the field, used as the fallback tooltip when no
    /// translation is loaded.
    pub doc: &'static str,
}

impl PropertyDesc {
    /// Clamps a value to the hard limits.
    pub fn clamp(&self, value: PropertyValue) -> PropertyValue {
        let lo = self.min.unwrap_or(f64::NEG_INFINITY);
        let hi = self.max.unwrap_or(f64::INFINITY);
        match value {
            PropertyValue::Float(v) => PropertyValue::Float(v.clamp(lo, hi)),
            PropertyValue::Int(v) => PropertyValue::Int((v as f64).clamp(lo, hi) as i64),
            PropertyValue::Vec3(v) => {
                PropertyValue::Vec3([v[0].clamp(lo, hi), v[1].clamp(lo, hi), v[2].clamp(lo, hi)])
            }
            PropertyValue::Enum(v) => {
                // An out-of-range enum from a stale keymap or an old file must
                // not index past the variant list.
                if let PropertyKind::Enum { variants } = self.kind {
                    PropertyValue::Enum(v.min(variants.len().saturating_sub(1) as u32))
                } else {
                    PropertyValue::Enum(v)
                }
            }
            other => other,
        }
    }

    /// i18n key for this parameter's label, under a given operator.
    ///
    /// Composed rather than stored, so a params struct shared by two operators
    /// does not have to pick one of them, and so the key can never drift out of
    /// step with the registered id.
    pub fn name_key(&self, op_id: &str) -> String {
        i18n_key(&["op", op_id, "param", self.name, "name"])
    }

    pub fn desc_key(&self, op_id: &str) -> String {
        i18n_key(&["op", op_id, "param", self.name, "desc"])
    }
}

/// Builds an i18n key from parts, in a form a Fluent identifier accepts.
///
/// Fluent identifiers are `[A-Za-z][A-Za-z0-9_-]*` — **no dots**. Operator ids
/// are dotted (`mesh.bevel`), and the obvious key spelling
/// `op.mesh.bevel.param.width.name` does not parse as Fluent at all, so the
/// separators are normalised here rather than at every call site. Doing it in one
/// place is also what keeps a key stable: it is derived from the operator id, so
/// it cannot drift out of step with what was registered.
pub fn i18n_key(parts: &[&str]) -> String {
    let mut key = String::with_capacity(parts.iter().map(|p| p.len() + 1).sum());
    for (i, part) in parts.iter().enumerate() {
        if i > 0 {
            key.push('-');
        }
        for c in part.chars() {
            key.push(match c {
                'A'..='Z' | 'a'..='z' | '0'..='9' | '_' => c,
                // Dots, spaces and anything else become the separator, so a key
                // is always a valid identifier whatever an operator is called.
                _ => '-',
            });
        }
    }
    key
}

/// Whether `key` is a valid Fluent message identifier.
pub fn is_valid_i18n_key(key: &str) -> bool {
    let mut chars = key.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

/// A set of parameter values, in schema declaration order.
///
/// Declaration order rather than sorted or hashed: it is what a user sees in the
/// panel and in a generated script line, and it must not vary between runs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PropertyMap {
    entries: Vec<(&'static str, PropertyValue)>,
}

impl PropertyMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, name: &str) -> Option<&PropertyValue> {
        self.entries
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| v)
    }

    /// Inserts or replaces, preserving first-insertion order.
    pub fn set(&mut self, name: &'static str, value: PropertyValue) {
        match self.entries.iter_mut().find(|(n, _)| *n == name) {
            Some(slot) => slot.1 = value,
            None => self.entries.push((name, value)),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (&'static str, &PropertyValue)> {
        self.entries.iter().map(|(n, v)| (*n, v))
    }

    /// Applies another map's values on top of this one.
    ///
    /// How a keymap binding overrides parameters: `mesh.select_all` bound with
    /// `{ action: Toggle }` is the same operator with one value replaced.
    pub fn overlay(&mut self, other: &PropertyMap) {
        for (name, value) in other.iter() {
            self.set(name, value.clone());
        }
    }

    /// Only the entries that differ from the schema defaults.
    ///
    /// What gets written into a history entry and a generated script line: a
    /// call listing thirty defaulted arguments is unreadable and, worse, freezes
    /// today's defaults into a script that should follow them.
    pub fn non_defaults(&self, schema: &[PropertyDesc], defaults: &PropertyMap) -> PropertyMap {
        let mut out = PropertyMap::new();
        for desc in schema {
            let (Some(value), Some(default)) = (self.get(desc.name), defaults.get(desc.name))
            else {
                continue;
            };
            if value != default {
                out.set(desc.name, value.clone());
            }
        }
        out
    }

    /// Renders a script call for the info log's "copy as script".
    ///
    /// This is the single best discoverability feature a tool of this kind can
    /// have — every action a user performs shows them the line that reproduces
    /// it — and it costs nothing because the history already stores exactly
    /// `(op_id, params)`.
    pub fn to_script_call(&self, op_id: &str) -> String {
        let mut s = String::new();
        let _ = write!(s, "kreate.ops.{op_id}(");
        for (i, (name, value)) in self.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            let _ = write!(s, "{name}={}", format_value(value));
        }
        s.push(')');
        s
    }
}

fn format_value(v: &PropertyValue) -> String {
    match v {
        PropertyValue::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        PropertyValue::Int(i) => i.to_string(),
        PropertyValue::Float(f) => format_float(*f),
        PropertyValue::Vec3(v) => format!(
            "({}, {}, {})",
            format_float(v[0]),
            format_float(v[1]),
            format_float(v[2])
        ),
        PropertyValue::Text(t) => format!("{t:?}"),
        PropertyValue::Enum(e) => e.to_string(),
        // A Python list, so a generated line pastes back and runs. Written out
        // in full however long it is: a truncated selection would produce a
        // script that runs and does something different, which is worse than a
        // long line.
        PropertyValue::Ints(v) => {
            let mut out = String::from("[");
            for (i, n) in v.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let _ = write!(out, "{n}");
            }
            out.push(']');
            out
        }
    }
}

/// Formats a float so a generated script round-trips exactly but stays readable.
fn format_float(f: f64) -> String {
    if f == f.trunc() && f.abs() < 1e15 {
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
}

/// A struct that can describe itself as a set of reflected parameters.
///
/// Implemented by `#[derive(OpParams)]`; implementing it by hand is possible but
/// means keeping the schema and the fields in step manually, which is exactly
/// the drift the derive exists to prevent.
pub trait OpParams: Sized + Clone + Default {
    fn schema() -> &'static [PropertyDesc];

    fn to_map(&self) -> PropertyMap;

    /// Rebuilds from a map. Missing entries take the field's default, and values
    /// are clamped to the schema's hard limits.
    ///
    /// Both matter for durability: a keymap or a saved file written against an
    /// older version will be missing parameters that have since been added, and
    /// must still load.
    fn from_map(map: &PropertyMap) -> Self;

    fn defaults_map() -> PropertyMap {
        Self::default().to_map()
    }

    /// The parameters that differ from the defaults.
    fn diff_from_defaults(&self) -> PropertyMap {
        self.to_map()
            .non_defaults(Self::schema(), &Self::defaults_map())
    }

    /// A script line reproducing this invocation, listing only what was changed.
    fn to_script_call(&self, op_id: &str) -> String {
        self.diff_from_defaults().to_script_call(op_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCHEMA: &[PropertyDesc] = &[
        PropertyDesc {
            name: "width",
            kind: PropertyKind::Float,
            min: Some(0.0),
            max: Some(10.0),
            soft_min: Some(0.0),
            soft_max: Some(1.0),
            step: Some(0.01),
            unit: Unit::Length,
            doc: "Width of the bevel.",
        },
        PropertyDesc {
            name: "segments",
            kind: PropertyKind::Int,
            min: Some(1.0),
            max: Some(64.0),
            soft_min: None,
            soft_max: None,
            step: Some(1.0),
            unit: Unit::Count,
            doc: "",
        },
    ];

    #[test]
    fn hard_limits_clamp_but_soft_limits_do_not() {
        // A drag should stop at the soft limit, but typing a larger number must
        // still work. Sliders that cannot be exceeded are a standing complaint
        // about modelling tools.
        let width = &SCHEMA[0];
        assert_eq!(
            width.clamp(PropertyValue::Float(5.0)),
            PropertyValue::Float(5.0)
        );
        assert_eq!(
            width.clamp(PropertyValue::Float(-1.0)),
            PropertyValue::Float(0.0)
        );
        assert_eq!(
            width.clamp(PropertyValue::Float(99.0)),
            PropertyValue::Float(10.0)
        );
        assert!(width.soft_max.unwrap() < width.max.unwrap());
    }

    #[test]
    fn out_of_range_enums_are_clamped_into_the_variant_list() {
        // A keymap or a file written against a version with more variants must
        // not index past the end.
        let desc = PropertyDesc {
            name: "mode",
            kind: PropertyKind::Enum {
                variants: &["a", "b"],
            },
            min: None,
            max: None,
            soft_min: None,
            soft_max: None,
            step: None,
            unit: Unit::None,
            doc: "",
        };
        assert_eq!(desc.clamp(PropertyValue::Enum(9)), PropertyValue::Enum(1));
    }

    #[test]
    fn map_preserves_insertion_order() {
        let mut m = PropertyMap::new();
        m.set("zeta", PropertyValue::Int(1));
        m.set("alpha", PropertyValue::Int(2));
        m.set("zeta", PropertyValue::Int(3));
        assert_eq!(
            m.iter().map(|(n, _)| n).collect::<Vec<_>>(),
            vec!["zeta", "alpha"],
            "replacing a value must not reorder the map"
        );
        assert_eq!(m.get("zeta"), Some(&PropertyValue::Int(3)));
    }

    #[test]
    fn overlay_applies_a_keymap_style_override() {
        let mut base = PropertyMap::new();
        base.set("action", PropertyValue::Enum(0));
        base.set("width", PropertyValue::Float(1.0));

        let mut override_map = PropertyMap::new();
        override_map.set("action", PropertyValue::Enum(2));
        base.overlay(&override_map);

        assert_eq!(base.get("action"), Some(&PropertyValue::Enum(2)));
        assert_eq!(base.get("width"), Some(&PropertyValue::Float(1.0)));
    }

    #[test]
    fn non_defaults_lists_only_what_changed() {
        let mut defaults = PropertyMap::new();
        defaults.set("width", PropertyValue::Float(0.1));
        defaults.set("segments", PropertyValue::Int(1));

        let mut actual = defaults.clone();
        actual.set("segments", PropertyValue::Int(4));

        let diff = actual.non_defaults(SCHEMA, &defaults);
        assert_eq!(diff.len(), 1);
        assert_eq!(diff.get("segments"), Some(&PropertyValue::Int(4)));
    }

    #[test]
    fn script_call_is_readable_and_lists_only_changes() {
        let mut m = PropertyMap::new();
        m.set("width", PropertyValue::Float(0.25));
        m.set("segments", PropertyValue::Int(4));
        assert_eq!(
            m.to_script_call("mesh.bevel"),
            "kreate.ops.mesh.bevel(width=0.25, segments=4)"
        );
    }

    #[test]
    fn script_call_formats_each_type() {
        let mut m = PropertyMap::new();
        m.set("flag", PropertyValue::Bool(true));
        m.set("offset", PropertyValue::Vec3([0.0, 0.0, 1.5]));
        m.set("name", PropertyValue::Text("Cube".into()));
        m.set("whole", PropertyValue::Float(2.0));
        assert_eq!(
            m.to_script_call("object.add"),
            r#"kreate.ops.object.add(flag=True, offset=(0.0, 0.0, 1.5), name="Cube", whole=2.0)"#
        );
    }

    #[test]
    fn an_empty_map_still_renders_a_valid_call() {
        assert_eq!(
            PropertyMap::new().to_script_call("mesh.select_all"),
            "kreate.ops.mesh.select_all()"
        );
    }

    #[test]
    fn integers_are_accepted_where_a_float_is_expected() {
        // A script writing `width=1` rather than `width=1.0` must work.
        assert_eq!(PropertyValue::Int(1).as_float(), Some(1.0));
        assert_eq!(PropertyValue::Float(1.5).as_int(), None);
    }

    #[test]
    fn i18n_keys_are_composed_from_the_operator_id() {
        // Composed rather than stored on the field, so a shared params struct
        // does not have to pick an owner and the key cannot drift.
        let d = &SCHEMA[0];
        assert_eq!(d.name_key("mesh.bevel"), "op-mesh-bevel-param-width-name");
        assert_eq!(d.desc_key("mesh.inset"), "op-mesh-inset-param-width-desc");
    }

    #[test]
    fn keys_are_valid_fluent_identifiers() {
        // Fluent ids permit no dots, so the obvious dotted spelling of a key does
        // not parse at all. Every generated key must survive the catalog.
        for op_id in ["mesh.bevel", "object.add_primitive", "weird op/name"] {
            for d in SCHEMA {
                for key in [d.name_key(op_id), d.desc_key(op_id)] {
                    assert!(is_valid_i18n_key(&key), "{key:?} is not a valid Fluent id");
                }
            }
        }
    }

    #[test]
    fn key_building_normalises_separators() {
        assert_eq!(
            i18n_key(&["op", "mesh.bevel", "name"]),
            "op-mesh-bevel-name"
        );
        assert_eq!(i18n_key(&["op", "a b", "c"]), "op-a-b-c");
        assert_eq!(i18n_key(&["keeps_underscores"]), "keeps_underscores");
    }
}
