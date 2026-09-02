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

    match preset {
        // Blender's, and ours: middle button orbits, shift-middle pans.
        Preset::Kreate | Preset::Blender => vec![
            orbit("view.orbit.mmb", MouseButton::Middle, Modifiers::NONE),
            pan(
                "view.pan.shift_mmb",
                MouseButton::Middle,
                Modifiers::shift(),
            ),
            Binding::new(
                "view.dolly.ctrl_mmb",
                Context::Viewport,
                Input::Mouse(MouseButton::Middle),
                Action::new("view.dolly"),
            )
            .with_modifiers(Modifiers::ctrl())
            .on(EventKind::Drag),
            wheel,
        ],
        // Maya: Alt with the three buttons. The reason navigation is bindable
        // at all — this cannot be expressed by a branch in an event handler.
        Preset::Maya => vec![
            orbit("view.orbit.alt_lmb", MouseButton::Left, Modifiers::alt()),
            pan("view.pan.alt_mmb", MouseButton::Middle, Modifiers::alt()),
            Binding::new(
                "view.dolly.alt_rmb",
                Context::Viewport,
                Input::Mouse(MouseButton::Right),
                Action::new("view.dolly"),
            )
            .with_modifiers(Modifiers::alt())
            .on(EventKind::Drag),
            wheel,
        ],
        // 3ds Max: middle button pans, Alt-middle orbits — the two swapped
        // relative to Blender, which is precisely the sort of difference that
        // makes a user reach for a preset.
        Preset::Max => vec![
            pan("view.pan.mmb", MouseButton::Middle, Modifiers::NONE),
            orbit("view.orbit.alt_mmb", MouseButton::Middle, Modifiers::alt()),
            wheel,
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
        key(
            "view.grid",
            "G",
            Context::Viewport,
            Action::new("view.toggle_grid"),
        ),
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
        key(
            "object.translate_up",
            "T",
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
        ),
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
                Modifiers::NONE,
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
