//! Widgets generated from an operator's reflected schema.
//!
//! This is the payoff of `#[derive(OpParams)]`. Nothing here knows what
//! `mesh.bevel` is: it reads a `&[PropertyDesc]`, draws the right widget for each
//! entry with the right range, step, unit and tooltip, and hands back an edited
//! `PropertyMap`. Adding an operator adds its panel.
//!
//! Two behaviours are worth stating because they are frequent complaints about
//! modelling tools:
//!
//! - **A drag stops at the soft limit; a typed value is only bound by the hard
//!   one.** A slider that cannot be exceeded is infuriating when the useful range
//!   turns out to be wider than the designer guessed.
//! - **Angles are edited in degrees and stored in radians.** Nobody types
//!   `0.2617993877`.

use egui::Ui;
use kreate_i18n::{Catalog, TextKind};
use kreate_reflect::{PropertyDesc, PropertyKind, PropertyMap, PropertyValue, Unit};

/// How a value is shown and parsed, derived from its unit.
pub struct UnitDisplay {
    /// Multiplier from stored value to displayed value.
    pub to_display: f64,
    pub suffix: &'static str,
    /// Drag increment in *display* units.
    pub default_step: f64,
    pub decimals: usize,
}

impl UnitDisplay {
    pub fn of(unit: Unit) -> Self {
        match unit {
            Unit::None => Self {
                to_display: 1.0,
                suffix: "",
                default_step: 0.01,
                decimals: 3,
            },
            Unit::Length => Self {
                to_display: 1.0,
                suffix: " m",
                default_step: 0.01,
                decimals: 4,
            },
            // Stored in radians, edited in degrees.
            Unit::Angle => Self {
                to_display: 180.0 / std::f64::consts::PI,
                suffix: "\u{00B0}",
                default_step: 1.0,
                decimals: 2,
            },
            Unit::Factor => Self {
                to_display: 100.0,
                suffix: " %",
                default_step: 1.0,
                decimals: 1,
            },
            Unit::Count => Self {
                to_display: 1.0,
                suffix: "",
                default_step: 1.0,
                decimals: 0,
            },
        }
    }

    pub fn from_display(&self, shown: f64) -> f64 {
        shown / self.to_display
    }
}

/// Labels for one parameter, translated where a translation exists.
fn labels(catalog: &Catalog, op_id: &str, desc: &PropertyDesc) -> (String, String) {
    let name = catalog.format(&desc.name_key(op_id), TextKind::Ui, &[]);
    // The key coming back means there is no translation; fall back to the field
    // name rather than showing `op-mesh-bevel-param-width-name` in the panel.
    let name = if name == desc.name_key(op_id) {
        prettify(desc.name)
    } else {
        name
    };

    let tip = catalog.format(&desc.desc_key(op_id), TextKind::Tooltip, &[]);
    // The doc comment on the field is the fallback tooltip, which is why
    // documentation and UI text cannot drift apart.
    let tip = if tip == desc.desc_key(op_id) {
        desc.doc.to_string()
    } else {
        tip
    };
    (name, tip)
}

/// `soft_max` -> `Soft Max`, for a field with no translation yet.
fn prettify(field: &str) -> String {
    field
        .split('_')
        .map(|word| {
            let mut c = word.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Draws every parameter in `schema`, editing `values` in place.
///
/// Returns `true` if anything changed, which the caller turns into an
/// `AdjustLast` command — the adjust-last-operation panel needs no more than
/// that, because re-running with edited parameters is the same path redo takes.
pub fn property_editor(
    ui: &mut Ui,
    catalog: &Catalog,
    op_id: &str,
    schema: &[PropertyDesc],
    values: &mut PropertyMap,
) -> bool {
    property_editor_where(ui, catalog, op_id, schema, values, |_| true)
}

/// The same, drawing only the parameters `keep` accepts.
///
/// **For a parameter block that covers several shapes of the same thing.** A
/// light's cone angle, angular diameter and source radius all ride in one
/// block, because changing the *kind* is one of the edits and splitting them
/// per kind would leave the panel switching operators mid-gesture. That is
/// right for the operator and wrong for the panel: a point light was offering
/// a cone angle that did nothing, which is a control that lies.
///
/// The predicate takes the parameter's name rather than the panel filtering
/// the schema itself, so that what a kind *has* stays written down once, next
/// to the operator that reads it.
pub fn property_editor_where(
    ui: &mut Ui,
    catalog: &Catalog,
    op_id: &str,
    schema: &[PropertyDesc],
    values: &mut PropertyMap,
    keep: impl Fn(&str) -> bool,
) -> bool {
    property_editor_keyed(ui, catalog, op_id, schema, values, keep, |_| None, "").0
}

/// Whether a value is animated, as the diamond beside it shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyMark {
    /// Not animated: a hollow diamond, to key it.
    None,
    /// Animated, with no key on this frame.
    Animated,
    /// Keyed on this frame.
    Keyed,
}

/// The same, with a keyframe diamond beside every parameter `mark` answers for.
///
/// Blender's decorator column: a hollow diamond keys the value where it
/// stands, a lit one says it is keyed on this frame and takes that key away.
/// Returns what changed and the name of the parameter whose diamond was
/// clicked, for the caller to key or unkey — this crate knows no documents.
#[allow(clippy::too_many_arguments)]
pub fn property_editor_keyed(
    ui: &mut Ui,
    catalog: &Catalog,
    op_id: &str,
    schema: &[PropertyDesc],
    values: &mut PropertyMap,
    keep: impl Fn(&str) -> bool,
    mark: impl Fn(&str) -> Option<KeyMark>,
    key_tip: &str,
) -> (bool, Option<&'static str>) {
    let out = property_editor_full(
        ui,
        catalog,
        op_id,
        schema,
        values,
        keep,
        mark,
        key_tip,
        |_| None,
        &FieldMenuText::default(),
    );
    (out.changed, out.keyed)
}

/// What a value's right-click menu can offer: the caller says, per field,
/// since only it knows what is animatable or driven.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FieldMenu {
    /// It can be keyed: Insert and Delete Keyframe.
    pub animatable: bool,
    /// It has a key on this frame, so Delete Keyframe does something.
    pub keyed_here: bool,
    /// Whether each component is driven, one per component; empty for a
    /// value drivers cannot reach.
    pub driven: Vec<bool>,
    /// Reset to Default is offered.
    pub resettable: bool,
}

/// What was chosen from a value's right-click menu.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldAction {
    InsertKey,
    DeleteKey,
    AddDriver(u8),
    RemoveDriver(u8),
    Reset,
}

/// The menu's words, and each component's name (X, Y, Z…).
#[derive(Clone, Debug, Default)]
pub struct FieldMenuText {
    pub insert_key: String,
    pub delete_key: String,
    pub add_driver: String,
    pub remove_driver: String,
    pub reset: String,
    pub components: [String; 4],
}

/// What a panel drawn by [`property_editor_full`] asked for.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EditorOutcome {
    pub changed: bool,
    /// A diamond was clicked on this field.
    pub keyed: Option<&'static str>,
    /// A right-click menu entry was chosen on this field.
    pub action: Option<(&'static str, FieldAction)>,
}

/// The purple Blender marks a driven value with.
pub const DRIVEN: egui::Color32 = egui::Color32::from_rgb(0xB5, 0x86, 0xF0);

/// The whole decorated editor: diamonds, and a right-click menu on each
/// field's name and diamond — Blender's, where keying, driving and resetting
/// a value all start. A driven value's name is purple.
#[allow(clippy::too_many_arguments)]
pub fn property_editor_full(
    ui: &mut Ui,
    catalog: &Catalog,
    op_id: &str,
    schema: &[PropertyDesc],
    values: &mut PropertyMap,
    keep: impl Fn(&str) -> bool,
    mark: impl Fn(&str) -> Option<KeyMark>,
    key_tip: &str,
    menu: impl Fn(&str) -> Option<FieldMenu>,
    text: &FieldMenuText,
) -> EditorOutcome {
    let mut out = EditorOutcome::default();
    let decorated = schema.iter().any(|d| mark(d.name).is_some());
    egui::Grid::new(("props", op_id))
        .num_columns(if decorated { 3 } else { 2 })
        .spacing([8.0, 4.0])
        .show(ui, |ui| {
            // Hidden parameters carry context, not choices: recorded and
            // replayed, never drawn. See `PropertyDesc::hidden`.
            for desc in schema.iter().filter(|d| !d.hidden && keep(d.name)) {
                let (name, tip) = labels(catalog, op_id, desc);
                let field_menu = menu(desc.name);
                let driven = field_menu.as_ref().is_some_and(|m| m.driven.iter().any(|d| *d));
                let label = if driven {
                    ui.label(egui::RichText::new(name).color(DRIVEN))
                } else {
                    ui.label(name)
                };
                let label = if !tip.is_empty() {
                    // Wrapped here rather than by the layout engine: a tooltip
                    // is prose, and `epaint` finds no break opportunity in a
                    // Korean sentence — see `crate::text`.
                    label.on_hover_text(crate::text::wrap_tooltip(ui, &tip))
                } else {
                    label
                };
                if let Some(m) = &field_menu {
                    let label = label.interact(egui::Sense::click());
                    if let Some(a) = field_context_menu(&label, m, text) {
                        out.action = Some((desc.name, a));
                    }
                }
                out.changed |= property_widget(ui, desc, values);
                if decorated {
                    match mark(desc.name) {
                        Some(m) => {
                            let response = key_diamond(ui, m);
                            let response = if key_tip.is_empty() { response } else { response.on_hover_text(key_tip) };
                            if response.clicked() {
                                out.keyed = Some(desc.name);
                            }
                            if let Some(fm) = &field_menu {
                                if let Some(a) = field_context_menu(&response, fm, text) {
                                    out.action = Some((desc.name, a));
                                }
                            }
                        }
                        None => {
                            ui.label("");
                        }
                    }
                }
                ui.end_row();
            }
        });
    out
}

/// The right-click menu of one field.
fn field_context_menu(response: &egui::Response, m: &FieldMenu, text: &FieldMenuText) -> Option<FieldAction> {
    let mut chosen = None;
    response.context_menu(|ui| {
        if m.animatable {
            if ui.button(&text.insert_key).clicked() {
                chosen = Some(FieldAction::InsertKey);
                ui.close();
            }
            if ui.add_enabled(m.keyed_here, egui::Button::new(&text.delete_key)).clicked() {
                chosen = Some(FieldAction::DeleteKey);
                ui.close();
            }
        }
        if !m.driven.is_empty() {
            ui.separator();
            let one = m.driven.len() == 1;
            let name = |i: usize, base: &str| {
                // i18n-exempt: a translated verb and a component's letter
                if one { base.to_string() } else { format!("{base} ({})", text.components[i.min(3)]) }
            };
            for (i, driven) in m.driven.iter().enumerate() {
                let (label, action) = if *driven {
                    (name(i, &text.remove_driver), FieldAction::RemoveDriver(i as u8))
                } else {
                    (name(i, &text.add_driver), FieldAction::AddDriver(i as u8))
                };
                if ui.button(label).clicked() {
                    chosen = Some(action);
                    ui.close();
                }
            }
        }
        if m.resettable {
            ui.separator();
            if ui.button(&text.reset).clicked() {
                chosen = Some(FieldAction::Reset);
                ui.close();
            }
        }
    });
    chosen
}

/// A keyframe diamond, painted: hollow, dim or lit.
fn key_diamond(ui: &mut Ui, mark: KeyMark) -> egui::Response {
    let size = egui::Vec2::splat(ui.spacing().interact_size.y);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    let theme = crate::theme::from_ui(ui);
    let c = rect.center();
    let r = size.y * 0.26;
    let points = vec![
        egui::pos2(c.x, c.y - r),
        egui::pos2(c.x + r, c.y),
        egui::pos2(c.x, c.y + r),
        egui::pos2(c.x - r, c.y),
    ];
    let hover = response.hovered();
    let (fill, stroke) = match mark {
        KeyMark::Keyed => (theme.accent, theme.accent),
        KeyMark::Animated => (theme.text_dim, theme.text_dim),
        KeyMark::None => (egui::Color32::TRANSPARENT, if hover { theme.text } else { theme.text_faint }),
    };
    ui.painter().add(egui::Shape::convex_polygon(points, fill, egui::Stroke::new(1.2, stroke)));
    response
}

/// Draws the widget for one parameter.
pub fn property_widget(ui: &mut Ui, desc: &PropertyDesc, values: &mut PropertyMap) -> bool {
    let current = values.get(desc.name).cloned();
    let mut changed = false;

    match (&desc.kind, current) {
        (PropertyKind::Bool, Some(PropertyValue::Bool(mut v))) => {
            if ui.checkbox(&mut v, "").changed() {
                values.set(desc.name, PropertyValue::Bool(v));
                changed = true;
            }
        }

        (PropertyKind::Int, Some(PropertyValue::Int(v))) => {
            let mut n = v;
            if drag_int(ui, desc, &mut n) {
                values.set(desc.name, desc.clamp(PropertyValue::Int(n)));
                changed = true;
            }
        }

        (PropertyKind::Float, Some(PropertyValue::Float(v))) => {
            let mut f = v;
            if drag_float(ui, desc, &mut f) {
                values.set(desc.name, desc.clamp(PropertyValue::Float(f)));
                changed = true;
            }
        }

        (PropertyKind::Vec3, Some(PropertyValue::Vec3(v))) => {
            // **Stacked, not in a row.** Three labelled fields want about 240
            // points and the properties column is nearer 180, so the row ran
            // off the right edge and Z was simply not on screen — a location
            // you cannot read in the panel whose job is to show it. Blender
            // stacks these for the same reason, in both its properties editor
            // and its sidebar, and the axis markers make the column scan as
            // fast as a row did.
            let mut out = v;
            ui.vertical(|ui| {
                for (i, axis) in ["X", "Y", "Z"].iter().enumerate() {
                    ui.horizontal(|ui| {
                        // i18n-exempt: axis letters are conventionally untranslated
                        ui.label(*axis);
                        let mut component = out[i];
                        let field = ui.scope(|ui| drag_float(ui, desc, &mut component));
                        axis_marker(ui, field.response.rect, i);
                        if field.inner {
                            out[i] = component;
                            changed = true;
                        }
                    });
                }
            });
            if changed {
                values.set(desc.name, desc.clamp(PropertyValue::Vec3(out)));
            }
        }

        (PropertyKind::Color, Some(PropertyValue::Vec3(v))) => {
            // A bar that opens a picker, and nothing else on the row.
            //
            // The three numbers *are* worth having — a base colour is often a
            // measured reflectance somebody needs to type exactly — but they
            // belong in the picker, which already carries R/G/B, H/S/V and a
            // hex field. Beside the swatch they did not fit: the properties
            // column is a column, and three float fields ran off the right
            // edge of it with the last one cut in half. A value you cannot
            // read is worse than one behind a click.
            //
            // The values are display-referred already: `as_swatch` encodes,
            // which is why a default base colour of 0.8 scene-linear reads
            // 0.906. So they go to the picker as they are. Handing it
            // scene-linear numbers would put the wheel's midpoint where the
            // eye finds nothing.
            let mut rgb = [v[0] as f32, v[1] as f32, v[2] as f32];
            let response = ui
                .scope(|ui| {
                    // A wide bar rather than a small square, so that the
                    // colour is legible as a colour.
                    let width = ui.available_width().clamp(48.0, 160.0);
                    ui.spacing_mut().interact_size.x = width;
                    egui::widgets::color_picker::color_edit_button_rgb(ui, &mut rgb)
                })
                .inner;
            if response.changed() {
                let out = [rgb[0] as f64, rgb[1] as f64, rgb[2] as f64];
                values.set(desc.name, desc.clamp(PropertyValue::Vec3(out)));
                changed = true;
            }
        }

        (PropertyKind::Text, Some(PropertyValue::Text(v))) => {
            let mut text = v;
            if ui.text_edit_singleline(&mut text).changed() {
                // Normalise on the way in: Hangul arriving as decomposed jamo
                // from a paste or an IME renders as separate marks and compares
                // unequal to the same word typed normally.
                values.set(
                    desc.name,
                    PropertyValue::Text(kreate_i18n::korean::normalize(&text)),
                );
                changed = true;
            }
        }

        (PropertyKind::Enum { variants }, Some(PropertyValue::Enum(v))) => {
            let selected = variants.get(v as usize).copied().unwrap_or("?");
            let mut choice = v;
            egui::ComboBox::from_id_salt((desc.name, "enum"))
                .selected_text(prettify(selected))
                .show_ui(ui, |ui| {
                    for (i, variant) in variants.iter().enumerate() {
                        if ui
                            .selectable_value(&mut choice, i as u32, prettify(variant))
                            .clicked()
                        {
                            changed = true;
                        }
                    }
                });
            if changed {
                values.set(desc.name, desc.clamp(PropertyValue::Enum(choice)));
            }
        }

        // A resolved element set: shown as a count, not as values. A thousand
        // spinners is not a way to edit a selection, and the thing that does
        // edit one is the viewport.
        //
        // A bare number rather than a phrase, so there is no prose here to need
        // translating — the row's own label already says what is being counted.
        (PropertyKind::Ints, Some(PropertyValue::Ints(v))) => {
            ui.weak(v.len().to_string());
        }

        // The value is missing or has the wrong type — from an older file, or a
        // schema that changed. Show it as unavailable rather than guessing, and
        // never panic on a document the application can otherwise open.
        (_, other) => {
            let what = other
                .map(|v| v.kind_name().to_string())
                .unwrap_or_else(|| "missing".into());
            ui.weak(format!("<{what}>"));
        }
    }
    changed
}

/// A drag-scrub float field, in display units.
/// A coloured strip down the left edge of an axis field.
///
/// The letter says which axis; the colour is what lets the eye find the right
/// box without reading. It is the same three colours as the grid and the
/// navigation gizmo — see [`crate::theme::Theme::axis`] — which is the point:
/// after a week the user reaches for "the red one" without deciding to.
///
/// Drawn rather than styled because egui has no per-widget border, and drawn
/// *after* the field so it is not painted over.
fn axis_marker(ui: &Ui, rect: egui::Rect, axis: usize) {
    const WIDTH: f32 = 2.0;
    if rect.width() <= WIDTH {
        return;
    }
    let strip = egui::Rect::from_min_max(rect.min, egui::pos2(rect.min.x + WIDTH, rect.max.y));
    ui.painter().rect_filled(
        strip,
        egui::CornerRadius {
            nw: ui.visuals().widgets.inactive.corner_radius.nw,
            sw: ui.visuals().widgets.inactive.corner_radius.sw,
            ne: 0,
            se: 0,
        },
        crate::theme::from_ui(ui).axis(axis),
    );
}

fn drag_float(ui: &mut Ui, desc: &PropertyDesc, value: &mut f64) -> bool {
    let display = UnitDisplay::of(desc.unit);
    let mut shown = *value * display.to_display;

    let step = desc.step.unwrap_or(display.default_step) * display.to_display;
    let mut drag = egui::DragValue::new(&mut shown)
        .speed(step)
        .suffix(display.suffix)
        .max_decimals(display.decimals);

    // The soft range bounds the *drag*; typing is bound only by the hard range.
    // A slider that cannot be exceeded is a standing complaint about modelling
    // tools, and the schema distinguishes the two for exactly this reason.
    let soft_lo = desc.soft_min.or(desc.min);
    let soft_hi = desc.soft_max.or(desc.max);
    if let (Some(lo), Some(hi)) = (soft_lo, soft_hi) {
        drag = drag.range((lo * display.to_display)..=(hi * display.to_display));
    }

    if ui.add(drag).changed() {
        *value = display.from_display(shown);
        true
    } else {
        false
    }
}

fn drag_int(ui: &mut Ui, desc: &PropertyDesc, value: &mut i64) -> bool {
    let mut n = *value;
    let mut drag = egui::DragValue::new(&mut n).speed(desc.step.unwrap_or(1.0));
    if let (Some(lo), Some(hi)) = (desc.min.or(desc.soft_min), desc.max.or(desc.soft_max)) {
        drag = drag.range((lo as i64)..=(hi as i64));
    }
    if ui.add(drag).changed() {
        *value = n;
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(name: &'static str, kind: PropertyKind, unit: Unit) -> PropertyDesc {
        PropertyDesc {
            name,
            kind,
            min: None,
            max: None,
            soft_min: None,
            soft_max: None,
            step: None,
            unit,
            doc: "",
            hidden: false,
        }
    }

    #[test]
    fn angles_are_edited_in_degrees_and_stored_in_radians() {
        // Nobody types 0.2617993877.
        let d = UnitDisplay::of(Unit::Angle);
        let radians = 15f64.to_radians();
        assert!((radians * d.to_display - 15.0).abs() < 1e-9);
        assert!((d.from_display(15.0) - radians).abs() < 1e-12);
        assert_eq!(d.suffix, "\u{00B0}");
    }

    #[test]
    fn factors_are_edited_as_percentages() {
        let d = UnitDisplay::of(Unit::Factor);
        assert!((0.25 * d.to_display - 25.0).abs() < 1e-9);
        assert!((d.from_display(25.0) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn lengths_are_shown_in_metres_without_conversion() {
        let d = UnitDisplay::of(Unit::Length);
        assert_eq!(d.to_display, 1.0);
        assert_eq!(d.suffix, " m");
        // Four decimals, so a millimetre is editable.
        assert!(d.decimals >= 4);
    }

    #[test]
    fn counts_drag_in_whole_steps() {
        let d = UnitDisplay::of(Unit::Count);
        assert_eq!(d.default_step, 1.0);
        assert_eq!(d.decimals, 0);
    }

    #[test]
    fn unit_conversions_round_trip() {
        for unit in [
            Unit::None,
            Unit::Length,
            Unit::Angle,
            Unit::Factor,
            Unit::Count,
        ] {
            let d = UnitDisplay::of(unit);
            let stored = 0.375;
            assert!(
                (d.from_display(stored * d.to_display) - stored).abs() < 1e-12,
                "{unit:?} does not round-trip"
            );
        }
    }

    #[test]
    fn field_names_become_readable_labels_without_a_translation() {
        // A new parameter must not show `soft_max` or a raw i18n key in the panel.
        assert_eq!(prettify("soft_max"), "Soft Max");
        assert_eq!(prettify("width"), "Width");
        assert_eq!(prettify("clamp_overlap"), "Clamp Overlap");
        assert_eq!(prettify(""), "");
    }

    #[test]
    fn labels_prefer_a_translation_and_fall_back_to_the_field() {
        let mut catalog = Catalog::new();
        catalog
            .add(
                "ko-KR",
                "op-mesh-bevel-param-width-name = 폭\nop-mesh-bevel-param-width-desc = 베벨의 폭입니다.\n",
            )
            .unwrap();

        let mut d = desc("width", PropertyKind::Float, Unit::Length);
        d.doc = "Width of the bevel.";
        let (name, tip) = labels(&catalog, "mesh.bevel", &d);
        assert_eq!(name, "폭");
        assert_eq!(tip, "베벨의 폭입니다.");

        // An untranslated parameter falls back to the field name and its doc
        // comment, so documentation and UI text cannot drift apart.
        let mut other = desc("clamp_overlap", PropertyKind::Bool, Unit::None);
        other.doc = "Clamp the width.";
        let (name, tip) = labels(&catalog, "mesh.bevel", &other);
        assert_eq!(name, "Clamp Overlap");
        assert_eq!(tip, "Clamp the width.");
    }

    #[test]
    fn the_tooltip_toggle_suppresses_translated_tooltips_only() {
        // English menus with Korean tooltips, and the reverse, are both things
        // Korean professionals ask for.
        let mut catalog = Catalog::new();
        catalog
            .add(
                "ko-KR",
                "op-x-param-w-name = 폭\nop-x-param-w-desc = 설명\n",
            )
            .unwrap();
        catalog.options.tooltips = false;

        let mut d = desc("w", PropertyKind::Float, Unit::None);
        d.doc = "fallback doc";
        let (name, tip) = labels(&catalog, "x", &d);
        assert_eq!(name, "폭", "the label is still translated");
        assert_eq!(
            tip, "fallback doc",
            "the tooltip falls back to the doc comment"
        );
    }

    #[test]
    fn a_soft_range_is_narrower_than_the_hard_one_the_widget_enforces() {
        // The distinction the drag widget relies on: it clamps to the soft range,
        // while a typed value is bound only by the hard range.
        let d = PropertyDesc {
            name: "width",
            kind: PropertyKind::Float,
            min: Some(0.0),
            max: Some(100.0),
            soft_min: Some(0.0),
            soft_max: Some(1.0),
            step: Some(0.01),
            unit: Unit::Length,
            doc: "",
            hidden: false,
        };
        assert!(d.soft_max.unwrap() < d.max.unwrap());
        // And the hard range is what actually constrains a stored value.
        assert_eq!(
            d.clamp(PropertyValue::Float(50.0)),
            PropertyValue::Float(50.0)
        );
        assert_eq!(
            d.clamp(PropertyValue::Float(500.0)),
            PropertyValue::Float(100.0)
        );
    }
}
