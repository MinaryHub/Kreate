//! End-to-end tests for `#[derive(OpParams)]`.
//!
//! An integration test rather than a unit test, because the derive generates
//! code that refers to `::kreate_ops::...` — exactly as a downstream operator
//! crate will. Testing it from inside the crate would exercise a path no real
//! caller takes.

use kreate_ops::property::OpParams as _;
use kreate_ops::{OpParams, PropertyKind, PropertyMap, PropertyValue, Unit};

/// A representative operator's parameters, using every supported form.
#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct BevelParams {
    /// Width of the bevel.
    #[param(min = 0.0, max = 100.0, soft_max = 1.0, step = 0.01, unit = "length")]
    pub width: f64,
    /// Number of segments across the bevel.
    #[param(min = 1, max = 64, unit = "count")]
    pub segments: i32,
    /// Clamp the width so adjacent bevels do not overlap.
    pub clamp_overlap: bool,
    /// How the corners are joined.
    #[param(variants("sharp", "patch", "arc"))]
    pub miter: u32,
    /// Direction of the offset.
    pub offset: [f64; 3],
    /// Name given to the resulting selection.
    pub tag: String,
}

impl Default for BevelParams {
    fn default() -> Self {
        Self {
            width: 0.1,
            segments: 1,
            clamp_overlap: true,
            miter: 0,
            offset: [0.0, 0.0, 0.0],
            tag: String::new(),
        }
    }
}

/// An operator that genuinely has no parameters.
#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SelectAllParams {}

#[test]
fn schema_lists_every_field_in_declaration_order() {
    let names: Vec<&str> = BevelParams::schema().iter().map(|d| d.name).collect();
    assert_eq!(
        names,
        vec![
            "width",
            "segments",
            "clamp_overlap",
            "miter",
            "offset",
            "tag"
        ]
    );
}

#[test]
fn rust_types_map_to_reflected_kinds() {
    let by_name = |n: &str| {
        *BevelParams::schema()
            .iter()
            .find(|d| d.name == n)
            .expect("field")
    };
    assert_eq!(by_name("width").kind, PropertyKind::Float);
    assert_eq!(by_name("segments").kind, PropertyKind::Int);
    assert_eq!(by_name("clamp_overlap").kind, PropertyKind::Bool);
    assert_eq!(by_name("offset").kind, PropertyKind::Vec3);
    assert_eq!(by_name("tag").kind, PropertyKind::Text);
    assert!(matches!(
        by_name("miter").kind,
        PropertyKind::Enum { variants } if variants == ["sharp", "patch", "arc"]
    ));
}

#[test]
fn ranges_units_and_steps_survive_the_attribute() {
    let width = BevelParams::schema()[0];
    assert_eq!(width.min, Some(0.0));
    assert_eq!(width.max, Some(100.0));
    assert_eq!(width.soft_max, Some(1.0));
    assert_eq!(width.step, Some(0.01));
    assert_eq!(width.unit, Unit::Length);
    // A field with no attribute gets sensible neutral values rather than
    // accidentally inheriting the previous field's.
    let clamp = BevelParams::schema()[2];
    assert_eq!(clamp.min, None);
    assert_eq!(clamp.unit, Unit::None);
}

#[test]
fn doc_comments_become_the_fallback_tooltip() {
    // Documentation and UI text cannot drift apart if they are the same text.
    assert_eq!(BevelParams::schema()[0].doc, "Width of the bevel.");
    assert_eq!(
        BevelParams::schema()[2].doc,
        "Clamp the width so adjacent bevels do not overlap."
    );
}

#[test]
fn round_trips_through_a_property_map() {
    let p = BevelParams {
        width: 0.25,
        segments: 4,
        clamp_overlap: false,
        miter: 2,
        offset: [1.0, -2.0, 0.5],
        tag: "edge".into(),
    };
    assert_eq!(BevelParams::from_map(&p.to_map()), p);
}

#[test]
fn missing_entries_fall_back_to_defaults() {
    // The durability property: a keymap or a file written against an older
    // version is missing parameters added since, and must still load.
    let mut partial = PropertyMap::new();
    partial.set("segments", PropertyValue::Int(8));

    let p = BevelParams::from_map(&partial);
    assert_eq!(p.segments, 8);
    assert_eq!(p.width, BevelParams::default().width);
    assert_eq!(p.clamp_overlap, BevelParams::default().clamp_overlap);
}

#[test]
fn unknown_entries_are_ignored_rather_than_failing() {
    // A parameter removed in a later version must not make an old file unloadable.
    let mut map = BevelParams::default().to_map();
    map.set("legacy_option", PropertyValue::Int(3));
    assert_eq!(BevelParams::from_map(&map), BevelParams::default());
}

#[test]
fn values_from_a_map_are_clamped_to_the_hard_limits() {
    // A script or a hand-edited keymap can supply anything; the operator must
    // not have to defend itself.
    let mut map = PropertyMap::new();
    map.set("width", PropertyValue::Float(-5.0));
    map.set("segments", PropertyValue::Int(9999));
    let p = BevelParams::from_map(&map);
    assert_eq!(p.width, 0.0);
    assert_eq!(p.segments, 64);
}

#[test]
fn a_wrongly_typed_value_leaves_the_default_in_place() {
    let mut map = PropertyMap::new();
    map.set("clamp_overlap", PropertyValue::Text("yes".into()));
    let p = BevelParams::from_map(&map);
    assert_eq!(p.clamp_overlap, BevelParams::default().clamp_overlap);
}

#[test]
fn integers_are_accepted_for_float_parameters() {
    // `kreate.ops.mesh.bevel(width=1)` is a reasonable thing for someone to type.
    let mut map = PropertyMap::new();
    map.set("width", PropertyValue::Int(1));
    assert_eq!(BevelParams::from_map(&map).width, 1.0);
}

#[test]
fn an_out_of_range_enum_is_clamped_into_the_variant_list() {
    let mut map = PropertyMap::new();
    map.set("miter", PropertyValue::Enum(99));
    assert_eq!(
        BevelParams::from_map(&map).miter,
        2,
        "3 variants, so index 2"
    );
}

#[test]
fn a_script_call_lists_only_what_changed() {
    // This is the payoff: the history entry already holds `(op_id, params)`, so
    // "copy as script" is free, and it shows a line the user can actually read
    // rather than one restating thirty defaults.
    let p = BevelParams {
        segments: 4,
        ..BevelParams::default()
    };
    assert_eq!(
        p.to_script_call("mesh.bevel"),
        "kreate.ops.mesh.bevel(segments=4)"
    );

    let p = BevelParams {
        width: 0.5,
        miter: 1,
        ..BevelParams::default()
    };
    assert_eq!(
        p.to_script_call("mesh.bevel"),
        "kreate.ops.mesh.bevel(width=0.5, miter=1)"
    );
}

#[test]
fn an_unchanged_operator_produces_a_bare_call() {
    assert_eq!(
        BevelParams::default().to_script_call("mesh.bevel"),
        "kreate.ops.mesh.bevel()"
    );
}

#[test]
fn a_parameterless_operator_works() {
    assert!(SelectAllParams::schema().is_empty());
    assert!(SelectAllParams::default().to_map().is_empty());
    assert_eq!(
        SelectAllParams::default().to_script_call("mesh.select_all"),
        "kreate.ops.mesh.select_all()"
    );
    assert_eq!(
        SelectAllParams::from_map(&PropertyMap::new()),
        SelectAllParams::default()
    );
}

#[test]
fn i18n_keys_are_derivable_for_every_parameter() {
    // The gate that keeps UI text translatable: every parameter has a key, and
    // it is composed from the operator id at use time rather than stored.
    for d in BevelParams::schema() {
        let key = d.name_key("mesh.bevel");
        assert!(key.starts_with("op-mesh-bevel-param-"), "{key}");
        assert!(key.ends_with("-name"), "{key}");
        // Fluent identifiers permit no dots, so a key that is not a valid
        // identifier simply cannot be looked up.
        assert!(
            kreate_ops::property::is_valid_i18n_key(&key),
            "{key} is not a valid Fluent identifier"
        );
    }
}

#[test]
fn a_keymap_style_override_applies_on_top_of_defaults() {
    // How a binding carries parameters: `mesh.bevel` bound with `{segments: 3}`
    // is the same operator with one value replaced.
    let mut params = BevelParams::default().to_map();
    let mut binding = PropertyMap::new();
    binding.set("segments", PropertyValue::Int(3));
    params.overlay(&binding);

    let p = BevelParams::from_map(&params);
    assert_eq!(p.segments, 3);
    assert_eq!(p.width, BevelParams::default().width);
}
