//! Keymap presets.
//!
//! Four, because a modeller arrives with muscle memory from somewhere and the
//! plan counts that as part of interoperability: a tool that makes an
//! experienced user relearn navigation is a tool they evaluate for ten minutes.
//!
//! The presets differ almost entirely in **navigation**, which is why navigation
//! had to be bindable. Beyond that they share the modelling keys, because those
//! are the ones this application actually defines.

use kreate_reflect::PropertyValue;

use crate::keymap::{Action, Binding, Context, EventKind, Input, Keymap, Modifiers, MouseButton};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Preset {
    #[default]
    Kreate,
    Blender,
    Maya,
    Max,
}

pub const ALL: &[Preset] = &[Preset::Kreate, Preset::Blender, Preset::Maya, Preset::Max];

impl Preset {
    pub fn name(self) -> &'static str {
        match self {
            Preset::Kreate => "kreate",
            Preset::Blender => "blender",
            Preset::Maya => "maya",
            Preset::Max => "max",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        ALL.iter().copied().find(|p| p.name() == name)
    }
}

/// The navigation half, which is what actually differs between presets.
fn navigation(preset: Preset) -> Vec<Binding> {
    let orbit = |id: &str, button: MouseButton, modifiers: Modifiers| {
        Binding::new(
            id,
            Context::Viewport,
            Input::Mouse(button),
            Action::new("view.orbit"),
        )
        .with_modifiers(modifiers)
        .on(EventKind::Drag)
    };
    let pan = |id: &str, button: MouseButton, modifiers: Modifiers| {
        Binding::new(
            id,
            Context::Viewport,
            Input::Mouse(button),
            Action::new("view.pan"),
        )
        .with_modifiers(modifiers)
        .on(EventKind::Drag)
    };
    // Zoom is on the wheel in every preset. The drag-zoom variants differ, but
    // the wheel is universal and nobody's muscle memory disagrees about it.
    let wheel = Binding::new(
        "view.dolly.wheel",
        Context::Viewport,
        Input::Wheel,
        Action::new("view.dolly"),
    );
    // Shift with the wheel scrolls sideways nearly everywhere, and it is the
    // only horizontal pan available on a mouse with no tilt wheel and no
    // trackpad.
    let shift_wheel = Binding::new(
        "view.pan.shift_wheel",
        Context::Viewport,
        Input::Wheel,
        Action::new("view.pan"),
    )
    .with_modifiers(Modifiers::shift());

    let dolly = |id: &str, button: MouseButton, modifiers: Modifiers| {
        Binding::new(
            id,
            Context::Viewport,
            Input::Mouse(button),
            Action::new("view.dolly"),
        )
        .with_modifiers(modifiers)
        .on(EventKind::Drag)
    };

    // Navigation that needs no middle button.
    //
    // Not every mouse has a usable one — a scroll wheel that only clicks with a
    // deliberate push, a trackpad, a tablet stylus — and a modeller whose view
    // cannot be turned is not usable at all. These are Maya's combinations, so
    // they cost nothing to learn and are already in a lot of people's hands, and
    // they sit alongside the middle-button ones rather than replacing them.
    //
    // No conflict with selection: a plain left drag stays a rubber band, and
    // `Alt` is a strictly heavier match, so the keymap resolves it first.
    let alt_fallback = || {
        vec![
            orbit("view.orbit.alt_lmb", MouseButton::Left, Modifiers::alt()),
            pan("view.pan.alt_mmb", MouseButton::Middle, Modifiers::alt()),
            dolly("view.dolly.alt_rmb", MouseButton::Right, Modifiers::alt()),
        ]
    };

    match preset {
        // Blender's, and ours: middle button orbits, shift-middle pans.
        Preset::Kreate | Preset::Blender => {
            let mut bindings = vec![
                orbit("view.orbit.mmb", MouseButton::Middle, Modifiers::NONE),
                pan(
                    "view.pan.shift_mmb",
                    MouseButton::Middle,
                    Modifiers::shift(),
                ),
                dolly(
                    "view.dolly.ctrl_mmb",
                    MouseButton::Middle,
                    Modifiers::ctrl(),
                ),
                wheel,
                shift_wheel,
            ];
            bindings.extend(alt_fallback());
            bindings
        }
        // Maya: Alt with the three buttons. The reason navigation is bindable
        // at all — this cannot be expressed by a branch in an event handler.
        Preset::Maya => {
            let mut bindings = alt_fallback();
            bindings.push(wheel);
            bindings.push(shift_wheel);
            bindings
        }
        // 3ds Max: middle button pans, Alt-middle orbits — the two swapped
        // relative to Blender, which is precisely the sort of difference that
        // makes a user reach for a preset.
        Preset::Max => vec![
            pan("view.pan.mmb", MouseButton::Middle, Modifiers::NONE),
            orbit("view.orbit.alt_mmb", MouseButton::Middle, Modifiers::alt()),
            // Alt+LMB orbits here too, for the same reason it does in the
            // default preset: a mouse without a middle button must still be able
            // to turn the view.
            orbit("view.orbit.alt_lmb", MouseButton::Left, Modifiers::alt()),
            dolly("view.dolly.alt_rmb", MouseButton::Right, Modifiers::alt()),
            wheel,
            shift_wheel,
        ],
    }
}

/// Bindings every preset shares: the modelling keys this application defines.
fn common() -> Vec<Binding> {
    let key = |id: &str, name: &str, context: Context, action: Action| {
        Binding::new(id, context, Input::key(name), action)
    };
    let enum_param = |action: Action, name: &'static str, value: u32| {
        action.with(name, PropertyValue::Enum(value))
    };

    vec![
        // --- global -------------------------------------------------------
        // File commands live in the keymap rather than being hard-coded, for the
        // same reason navigation does: a user coming from another application
        // expects their own shortcuts, and one whose muscle memory is denied
        // notices immediately.
        key("file.new", "N", Context::Global, Action::new("file.new"))
            .with_modifiers(Modifiers::ctrl()),
        key("file.open", "O", Context::Global, Action::new("file.open"))
            .with_modifiers(Modifiers::ctrl()),
        key("file.save", "S", Context::Global, Action::new("file.save"))
            .with_modifiers(Modifiers::ctrl()),
        key(
            "file.save_as",
            "S",
            Context::Global,
            Action::new("file.save_as"),
        )
        .with_modifiers(Modifiers::ctrl_shift()),
        key("edit.undo", "Z", Context::Global, Action::new("edit.undo"))
            .with_modifiers(Modifiers::ctrl()),
        key("edit.redo", "Y", Context::Global, Action::new("edit.redo"))
            .with_modifiers(Modifiers::ctrl()),
        key(
            "view.frame_all",
            "F",
            Context::Viewport,
            Action::new("view.frame_all"),
        ),
        key(
            "view.toggle_mode",
            "Tab",
            Context::Viewport,
            Action::new("view.toggle_edit_mode"),
        ),
        // Display toggles. In the keymap rather than hard-coded in the
        // application for the same reason navigation is: they are things a user
        // rebinds, and a table that holds only *some* of the keys is a table
        // whose remapping screen has exceptions to explain.
        key(
            "view.shading",
            "Z",
            Context::Viewport,
            Action::new("view.cycle_shading"),
        ),
        key(
            "view.projection",
            "O",
            Context::Viewport,
            Action::new("view.toggle_projection"),
        ),
        // On `Shift+G` rather than bare `G`, which Move now takes in both
        // modes. Not a stylistic move: `ObjectMode` and `EditMesh` are searched
        // before `Viewport`, the application is always in one of the two, and a
        // bare `G` here would therefore be unreachable rather than merely
        // shadowed — see `Keymap::conflicts`, which now says so.
        key(
            "view.grid",
            "G",
            Context::Viewport,
            Action::new("view.toggle_grid"),
        )
        .with_modifiers(Modifiers::shift()),
        key(
            "view.wireframe",
            "W",
            Context::Viewport,
            Action::new("view.toggle_wireframe"),
        ),
        key(
            "view.auto_depth",
            "Y",
            Context::Viewport,
            Action::new("view.toggle_auto_depth"),
        ),
        // The two standing transform settings. Toggles rather than holds: they
        // are how the user works, not something wanted for one drag — `Ctrl`
        // covers the one-drag case by inverting the snap setting.
        key(
            "view.snap",
            "Backquote",
            Context::Viewport,
            Action::new("view.toggle_snap"),
        ),
        key(
            "view.proportional",
            "O",
            Context::EditMesh,
            Action::new("view.toggle_proportional"),
        ),
        key(
            "view.snap_mode",
            "Backquote",
            Context::Viewport,
            Action::new("view.cycle_snap_mode"),
        )
        .with_modifiers(Modifiers::shift()),
        key(
            "edit.adjust_smaller",
            "BracketLeft",
            Context::Viewport,
            Action::new("edit.adjust_last").with("factor", PropertyValue::Float(0.5)),
        ),
        key(
            "edit.adjust_larger",
            "BracketRight",
            Context::Viewport,
            Action::new("edit.adjust_last").with("factor", PropertyValue::Float(2.0)),
        ),
        key(
            "app.print_script",
            "P",
            Context::Viewport,
            Action::new("app.print_script"),
        ),
        key(
            "mesh.select_mode_vert",
            "Digit1",
            Context::EditMesh,
            enum_param(Action::new("view.select_mode"), "mode", 0),
        ),
        key(
            "mesh.select_mode_edge",
            "Digit2",
            Context::EditMesh,
            enum_param(Action::new("view.select_mode"), "mode", 1),
        ),
        key(
            "mesh.select_mode_face",
            "Digit3",
            Context::EditMesh,
            enum_param(Action::new("view.select_mode"), "mode", 2),
        ),
        key(
            "object.add_next",
            "A",
            Context::ObjectMode,
            Action::new("object.add_next_primitive"),
        )
        .with_modifiers(Modifiers::shift()),
        // Move, Rotate, Scale on G, R, S — the same three keys in object mode
        // and in edit mode, so the gesture does not change with the mode. `G`
        // for grab rather than `M` for move: it is what every modeller's hand
        // already knows, and this preset is Blender's deliberately.
        //
        // The binding ids stay as they are — `object.translate_up` is still
        // called that on `G`: a released id is part of a saved keymap, and
        // renaming one would silently drop a user's remapping of it.
        //
        // Still fixed-step operators rather than interactive modal tools. The
        // full `TransformOp` — axis constraints, numeric entry, snapping,
        // proportional editing — is Phase 2's, and these are what the keys do
        // until it lands. The ids and the parameter names do not change when it
        // does, so a keymap written today keeps working.
        key(
            "object.translate_up",
            "G",
            Context::ObjectMode,
            Action::new("object.translate")
                .with("translation", PropertyValue::Vec3([0.0, 0.0, 1.0])),
        ),
        key(
            "object.rotate",
            "R",
            Context::ObjectMode,
            Action::new("object.rotate")
                .with(
                    "angle",
                    PropertyValue::Float(std::f64::consts::FRAC_PI_6 / 2.0),
                )
                .with("axis", PropertyValue::Enum(2)),
        ),
        key(
            "object.scale",
            "S",
            Context::ObjectMode,
            Action::new("object.scale").with("scale", PropertyValue::Vec3([1.2, 1.2, 1.2])),
        ),
        // --- edit mode: the same three keys, acting on the selection --------
        //
        // One set of operators for vertices, edges and faces. An edge is its two
        // vertices and a face is its ring, so there is nothing mode-specific to
        // do: `mesh.translate` and friends move whatever the selection touches.
        key(
            "mesh.translate",
            "G",
            Context::EditMesh,
            Action::new("mesh.translate")
                .with("translation", PropertyValue::Vec3([0.0, 0.0, 0.25])),
        ),
        key(
            "mesh.rotate",
            "R",
            Context::EditMesh,
            Action::new("mesh.rotate")
                .with(
                    "angle",
                    PropertyValue::Float(std::f64::consts::FRAC_PI_6 / 2.0),
                )
                .with("axis", PropertyValue::Enum(2)),
        ),
        key(
            "mesh.scale",
            "S",
            Context::EditMesh,
            Action::new("mesh.scale").with("scale", PropertyValue::Vec3([1.2, 1.2, 1.2])),
        ),
        // --- object mode ---------------------------------------------------
        key(
            "object.add_cube",
            "A",
            Context::ObjectMode,
            enum_param(Action::new("object.add_primitive"), "shape", 1),
        ),
        key(
            "object.select_all",
            "A",
            Context::ObjectMode,
            enum_param(Action::new("object.select_all"), "action", 2),
        )
        .with_modifiers(Modifiers::ctrl()),
        key(
            "object.duplicate",
            "D",
            Context::ObjectMode,
            Action::new("object.duplicate").with("linked", PropertyValue::Bool(true)),
        ),
        key(
            "object.delete",
            "X",
            Context::ObjectMode,
            Action::new("object.delete"),
        ),
        // --- edit mode -----------------------------------------------------
        key(
            "mesh.extrude",
            "E",
            Context::EditMesh,
            Action::new("mesh.extrude_region").with("offset", PropertyValue::Float(1.0)),
        ),
        key(
            "mesh.inset",
            "I",
            Context::EditMesh,
            Action::new("mesh.inset").with("thickness", PropertyValue::Float(0.1)),
        ),
        key(
            "mesh.bridge",
            "B",
            Context::EditMesh,
            Action::new("mesh.bridge"),
        ),
        key(
            "mesh.subdivide",
            "S",
            Context::EditMesh,
            Action::new("mesh.subdivide").with("iterations", PropertyValue::Int(1)),
        )
        // Moved off the bare `S` when Scale took it. `Shift+S` rather than
        // somewhere else entirely: the muscle memory for subdivide is already on
        // this key, and a modifier is a smaller thing to relearn than a new
        // letter.
        .with_modifiers(Modifiers::shift()),
        key(
            "mesh.loop_cut",
            "R",
            Context::EditMesh,
            Action::new("mesh.loop_cut").with("cuts", PropertyValue::Int(1)),
        )
        .with_modifiers(Modifiers::ctrl()),
        key(
            "mesh.delete",
            "X",
            Context::EditMesh,
            Action::new("mesh.delete"),
        ),
        key(
            "mesh.dissolve",
            "X",
            Context::EditMesh,
            Action::new("mesh.dissolve"),
        )
        .with_modifiers(Modifiers::ctrl()),
        // Back on `M` and `Shift+M`, where they were before Move took the bare
        // key — and where Blender has them. The `Ctrl` they wore in between
        // existed only for that collision, and `G` has now ended it.
        key(
            "mesh.merge",
            "M",
            Context::EditMesh,
            Action::new("mesh.merge"),
        ),
        key(
            "mesh.weld",
            "M",
            Context::EditMesh,
            Action::new("mesh.merge").with("distance", PropertyValue::Float(1e-4)),
        )
        .with_modifiers(Modifiers::shift()),
        key(
            "mesh.bevel",
            "B",
            Context::EditMesh,
            Action::new("mesh.bevel").with("width", PropertyValue::Float(0.1)),
        )
        .with_modifiers(Modifiers::ctrl()),
        key(
            "mesh.split_faces",
            "Y",
            Context::EditMesh,
            enum_param(Action::new("mesh.split"), "mode", 0),
        ),
        key(
            "mesh.rip_edges",
            "V",
            Context::EditMesh,
            enum_param(Action::new("mesh.split"), "mode", 1),
        ),
        key(
            "mesh.fill",
            "F",
            Context::EditMesh,
            Action::new("mesh.fill"),
        ),
        key(
            "mesh.grid_fill",
            "F",
            Context::EditMesh,
            Action::new("mesh.fill").with("grid", PropertyValue::Bool(true)),
        )
        .with_modifiers(Modifiers::shift()),
        key(
            "mesh.normals_recalculate",
            "N",
            Context::EditMesh,
            enum_param(Action::new("mesh.normals"), "action", 0),
        )
        .with_modifiers(Modifiers::ctrl()),
        key(
            "mesh.normals_flip",
            "N",
            Context::EditMesh,
            enum_param(Action::new("mesh.normals"), "action", 2),
        )
        .with_modifiers(Modifiers::shift()),
        key(
            "mesh.shade_smooth",
            "N",
            Context::EditMesh,
            Action::new("mesh.shade").with("smooth", PropertyValue::Bool(true)),
        ),
        key(
            "mesh.mark_sharp",
            "K",
            Context::EditMesh,
            Action::new("mesh.mark_sharp").with("sharp", PropertyValue::Bool(true)),
        ),
        key(
            "mesh.clear_sharp",
            "K",
            Context::EditMesh,
            Action::new("mesh.mark_sharp").with("sharp", PropertyValue::Bool(false)),
        )
        .with_modifiers(Modifiers::alt()),
        key(
            "mesh.auto_smooth",
            "K",
            Context::EditMesh,
            Action::new("mesh.mark_sharp")
                .with("angle", PropertyValue::Float(std::f64::consts::FRAC_PI_6)),
        )
        .with_modifiers(Modifiers::shift()),
        key(
            "mesh.select_loop",
            "L",
            Context::EditMesh,
            enum_param(Action::new("mesh.select"), "action", 3),
        ),
        key(
            "mesh.select_ring",
            "L",
            Context::EditMesh,
            enum_param(Action::new("mesh.select"), "action", 4),
        )
        .with_modifiers(Modifiers::shift()),
        key(
            "mesh.select_all",
            "A",
            Context::EditMesh,
            enum_param(Action::new("mesh.select"), "action", 0),
        )
        .with_modifiers(Modifiers::ctrl()),
        key(
            "mesh.select_none",
            "A",
            Context::EditMesh,
            enum_param(Action::new("mesh.select"), "action", 1),
        )
        .with_modifiers(Modifiers::alt()),
        key(
            "mesh.select_invert",
            "I",
            Context::EditMesh,
            enum_param(Action::new("mesh.select"), "action", 2),
        )
        .with_modifiers(Modifiers::ctrl()),
    ]
}

/// Builds a preset's keymap.
pub fn load(preset: Preset) -> Keymap {
    let mut bindings = navigation(preset);
    bindings.extend(common());
    Keymap::new(bindings)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_names_round_trip() {
        // The name is what a preferences file stores, so it is a compatibility
        // surface rather than a label.
        for p in ALL {
            assert_eq!(Preset::from_name(p.name()), Some(*p));
        }
        assert_eq!(Preset::from_name("emacs"), None);
    }

    #[test]
    fn every_preset_binds_the_modelling_keys_identically() {
        // Presets exist to match navigation habits. Changing what `E` does
        // between them would be a different application, not a different keymap.
        let reference: Vec<(String, String)> = load(Preset::Kreate)
            .bindings()
            .filter(|b| !b.action.is_view())
            .map(|b| (b.id.0.clone(), b.action.op_id.clone()))
            .collect();
        for p in ALL {
            let theirs: Vec<(String, String)> = load(*p)
                .bindings()
                .filter(|b| !b.action.is_view())
                .map(|b| (b.id.0.clone(), b.action.op_id.clone()))
                .collect();
            assert_eq!(theirs, reference, "{p:?} changed a modelling key");
        }
    }

    #[test]
    fn the_presets_really_navigate_differently() {
        // Otherwise the whole mechanism is ceremony. Compared as the set of
        // triggers each one uses for navigation.
        let triggers = |p: Preset| -> Vec<(Input, Modifiers, String)> {
            let mut v: Vec<_> = load(p)
                .bindings()
                .filter(|b| b.action.is_view() && b.action.op_id != "view.frame_all")
                .map(|b| (b.input.clone(), b.modifiers, b.action.op_id.clone()))
                .collect();
            v.sort();
            v
        };
        assert_ne!(triggers(Preset::Kreate), triggers(Preset::Maya));
        assert_ne!(triggers(Preset::Kreate), triggers(Preset::Max));
        assert_ne!(triggers(Preset::Maya), triggers(Preset::Max));
        assert_eq!(
            triggers(Preset::Kreate),
            triggers(Preset::Blender),
            "ours is Blender's, deliberately"
        );
    }

    #[test]
    fn parameters_ride_along_with_the_binding() {
        // A keymap entry carries a parameter override, so two keys can reach the
        // same operator with different arguments. Without it, `M` and `Shift+M`
        // would need two operators for one idea.
        let k = load(Preset::Kreate);
        let merge = k
            .resolve(
                Context::EditMesh,
                &Input::key("M"),
                Modifiers::default(),
                EventKind::Press,
            )
            .unwrap();
        let weld = k
            .resolve(
                Context::EditMesh,
                &Input::key("M"),
                Modifiers::shift(),
                EventKind::Press,
            )
            .unwrap();
        assert_eq!(merge.op_id, weld.op_id);
        assert!(merge.params.get("distance").is_none());
        assert!(weld.params.get("distance").is_some());
    }

    #[test]
    fn g_moves_in_both_modes_and_the_grid_survived_it() {
        // `G` for grab, in every preset, in both modes — the transform keys are
        // in `common()`, so a preset that lost one would have lost it silently.
        for preset in ALL.iter().copied() {
            let k = load(preset);
            for (context, op) in [
                (Context::ObjectMode, "object.translate"),
                (Context::EditMesh, "mesh.translate"),
            ] {
                assert_eq!(
                    k.resolve(context, &Input::key("G"), Modifiers::NONE, EventKind::Press)
                        .map(|a| a.op_id),
                    Some(op.to_string()),
                    "{preset:?}: G does not move in {context:?}"
                );
            }

            // And the grid toggle is still reachable, from both modes. It used
            // to be on bare `G`, where taking that key for Move would have
            // killed it outright rather than merely moved it: the inner
            // contexts are searched first and the application is always in one
            // of them.
            for context in [Context::ObjectMode, Context::EditMesh] {
                assert_eq!(
                    k.resolve(
                        context,
                        &Input::key("G"),
                        Modifiers::shift(),
                        EventKind::Press
                    )
                    .map(|a| a.op_id),
                    Some("view.toggle_grid".to_string()),
                    "{preset:?}: the grid toggle is unreachable from {context:?}"
                );
            }
        }
    }

    #[test]
    fn merge_and_weld_hold_m_without_a_modifier_between_them() {
        // They wore a `Ctrl` only while Move held the bare key. `M` is theirs
        // again, and the pair still reads as one idea at two tolerances.
        let k = load(Preset::Kreate);
        let plain = k
            .resolve(
                Context::EditMesh,
                &Input::key("M"),
                Modifiers::NONE,
                EventKind::Press,
            )
            .expect("M merges");
        assert_eq!(plain.op_id, "mesh.merge");
        assert!(plain.params.get("distance").is_none());
        // And nothing answers `M` in object mode, where merge has no meaning —
        // rather than object mode quietly inheriting edit mode's merge.
        assert!(k
            .resolve(
                Context::ObjectMode,
                &Input::key("M"),
                Modifiers::NONE,
                EventKind::Press
            )
            .is_none());
    }

    #[test]
    fn x_means_different_things_in_the_two_modes() {
        let k = load(Preset::Kreate);
        assert_eq!(
            k.resolve(
                Context::ObjectMode,
                &Input::key("X"),
                Modifiers::NONE,
                EventKind::Press
            )
            .unwrap()
            .op_id,
            "object.delete"
        );
        assert_eq!(
            k.resolve(
                Context::EditMesh,
                &Input::key("X"),
                Modifiers::NONE,
                EventKind::Press
            )
            .unwrap()
            .op_id,
            "mesh.delete"
        );
    }
}
