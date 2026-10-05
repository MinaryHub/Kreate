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
        // The clock, on Blender's keys and everywhere — an animator plays from
        // whichever editor the pointer is over. Space plays and stops; the
        // arrows step a frame, or with Shift jump to either end of the range.
        key(
            "time.play",
            "Space",
            Context::Global,
            Action::new("time.play"),
        ),
        key(
            "time.next",
            "ArrowRight",
            Context::Global,
            Action::new("time.step").with("delta", PropertyValue::Int(1)),
        ),
        key(
            "time.previous",
            "ArrowLeft",
            Context::Global,
            Action::new("time.step").with("delta", PropertyValue::Int(-1)),
        ),
        key(
            "time.end",
            "ArrowRight",
            Context::Global,
            Action::new("time.jump").with("to_end", PropertyValue::Bool(true)),
        )
        .with_modifiers(Modifiers::shift()),
        key(
            "time.start",
            "ArrowLeft",
            Context::Global,
            Action::new("time.jump"),
        )
        .with_modifiers(Modifiers::shift()),
        // Blender's menu search, on Blender's key, everywhere: a command whose
        // menu you cannot find is one you can still name.
        key(
            "app.search",
            "F3",
            Context::Global,
            Action::new("app.search"),
        ),
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
        // Blender's three pies, on Blender's keys: shading on `Z`, the view
        // directions on the key under Escape, modes on `Ctrl+Tab`. `Tab`
        // alone still toggles edit mode, which is the gesture used a hundred
        // times for every once the pie is.
        key(
            "view.shading",
            "Z",
            Context::Viewport,
            Action::new("app.pie").with("menu", PropertyValue::Text("shading".into())),
        ),
        key(
            "view.pie",
            "Backquote",
            Context::Viewport,
            Action::new("app.pie").with("menu", PropertyValue::Text("view".into())),
        ),
        key(
            "view.mode_pie",
            "Tab",
            Context::Viewport,
            Action::new("app.pie").with("menu", PropertyValue::Text("mode".into())),
        )
        .with_modifiers(Modifiers::ctrl()),
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
            "view.sidebar",
            "N",
            Context::Viewport,
            Action::new("view.toggle_sidebar"),
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
        // The two standing transform settings. Toggles rather than holds: they
        // are how the user works, not something wanted for one drag — `Ctrl`
        // covers the one-drag case by inverting the snap setting.
        // `Shift+Tab`, Blender's snapping toggle. It sat on the key under
        // Escape, which the application never learnt the name of — so it was
        // a binding no keystroke could reach. That key is the view pie's now,
        // as it is in Blender.
        key(
            "view.snap",
            "Tab",
            Context::Viewport,
            Action::new("view.toggle_snap"),
        )
        .with_modifiers(Modifiers::shift()),
        key(
            "view.proportional",
            "O",
            Context::EditMesh,
            Action::new("view.toggle_proportional"),
        ),
        key(
            "view.snap_mode",
            "Tab",
            Context::Viewport,
            Action::new("view.cycle_snap_mode"),
        )
        .with_modifiers(Modifiers::ctrl_shift()),
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
        // Blender's numpad views: `/` local view (on the main row too), `0`
        // through the scene camera, `.` framing the selection.
        key(
            "view.local_view",
            "NumpadDivide",
            Context::Viewport,
            Action::new("view.local_view"),
        ),
        key(
            "view.local_view_main",
            "Slash",
            Context::Viewport,
            Action::new("view.local_view"),
        ),
        key(
            "view.camera",
            "Numpad0",
            Context::Viewport,
            Action::new("view.camera"),
        ),
        // Blender's Alt+Z: see through the surfaces, and select through them.
        key(
            "view.xray",
            "Z",
            Context::Viewport,
            Action::new("view.toggle_xray"),
        )
        .with_modifiers(Modifiers::alt()),
        key(
            "view.frame_selected",
            "NumpadDecimal",
            Context::Viewport,
            Action::new("view.frame_selected"),
        ),
        // Blender's Shift+S: the snap pie — the 3D cursor to the selection or
        // the world's origin, or the selection to the cursor.
        key(
            "view.snap_pie",
            "S",
            Context::Viewport,
            Action::new("app.pie").with("menu", PropertyValue::Text("snap".into())),
        )
        .with_modifiers(Modifiers::shift()),
        // Shift+Right-click puts the 3D cursor on what is under the pointer.
        Binding::new(
            "view.cursor_place",
            Context::Viewport,
            Input::Mouse(MouseButton::Right),
            Action::new("view.cursor_place"),
        )
        .with_modifiers(Modifiers::shift()),
        // Blender's Shift+A: the Add menu, under the pointer.
        key(
            "object.add_menu",
            "A",
            Context::ObjectMode,
            Action::new("app.menu").with("menu", PropertyValue::Text("add".into())),
        )
        .with_modifiers(Modifiers::shift()),
        // Blender's M: Move to Collection, a menu of them under the pointer.
        key(
            "object.move_to_collection",
            "M",
            Context::ObjectMode,
            Action::new("app.menu").with("menu", PropertyValue::Text("collection".into())),
        ),
        // Blender's Ctrl+A: the Apply menu.
        key(
            "object.apply_menu",
            "A",
            Context::ObjectMode,
            Action::new("app.menu").with("menu", PropertyValue::Text("apply".into())),
        )
        .with_modifiers(Modifiers::ctrl()),
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
        // --- the UV editor: the same three keys again ------------------------
        //
        // `G`, `R`, `S` mean move, turn and scale wherever the pointer is, and
        // the operator behind them is what changes. That is the whole reason
        // the UV editor is its own context rather than part of the viewport's:
        // one key, one meaning, resolved by where the pointer is.
        key(
            "uv.translate",
            "G",
            Context::UvEditor,
            enum_param(Action::new("uv.transform"), "mode", 0),
        ),
        key(
            "uv.rotate",
            "R",
            Context::UvEditor,
            enum_param(Action::new("uv.transform"), "mode", 1),
        ),
        key(
            "uv.scale",
            "S",
            Context::UvEditor,
            enum_param(Action::new("uv.transform"), "mode", 2),
        ),
        key(
            "uv.smart_project",
            "U",
            Context::UvEditor,
            Action::new("uv.smart_project"),
        ),
        key("uv.pack", "P", Context::UvEditor, Action::new("uv.pack")),
        key(
            "uv.select_all",
            "A",
            Context::UvEditor,
            enum_param(Action::new("uv.select"), "action", 0),
        ),
        key(
            "uv.select_none",
            "A",
            Context::UvEditor,
            enum_param(Action::new("uv.select"), "action", 1),
        )
        .with_modifiers(Modifiers::alt()),
        key(
            "uv.select_invert",
            "I",
            Context::UvEditor,
            enum_param(Action::new("uv.select"), "action", 2),
        )
        .with_modifiers(Modifiers::ctrl()),
        key(
            "uv.select_linked",
            "L",
            Context::UvEditor,
            enum_param(Action::new("uv.select"), "action", 3),
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
        // --- pose mode: the same three keys, on bones ------------------------
        key(
            "pose.translate",
            "G",
            Context::Pose,
            Action::new("pose.translate"),
        ),
        key(
            "pose.rotate",
            "R",
            Context::Pose,
            Action::new("pose.rotate"),
        ),
        key("pose.scale", "S", Context::Pose, Action::new("pose.scale")),
        // Blender's: `A` all, `Alt+A` none, `Ctrl+I` the others; and
        // `Alt+G/R/S` put what was moved, turned or scaled back at rest.
        // Keys on the same keys as in object mode; the operators know which
        // mode they are in and key the bones.
        key(
            "pose.keyframe_insert",
            "I",
            Context::Pose,
            Action::new("anim.keyframe_insert"),
        ),
        key(
            "pose.keyframe_delete",
            "I",
            Context::Pose,
            Action::new("anim.keyframe_delete"),
        )
        .with_modifiers(Modifiers::alt()),
        key(
            "pose.select_all",
            "A",
            Context::Pose,
            enum_param(Action::new("pose.select_all"), "action", 0),
        ),
        key(
            "pose.select_none",
            "A",
            Context::Pose,
            enum_param(Action::new("pose.select_all"), "action", 1),
        )
        .with_modifiers(Modifiers::alt()),
        // Blender's sculpt brushes on their letters.
        key(
            "sculpt.draw",
            "X",
            Context::Sculpt,
            Action::new("sculpt.brush").with("brush", PropertyValue::Enum(0)),
        ),
        key(
            "sculpt.smooth",
            "S",
            Context::Sculpt,
            Action::new("sculpt.brush").with("brush", PropertyValue::Enum(1)),
        ),
        key(
            "sculpt.grab",
            "G",
            Context::Sculpt,
            Action::new("sculpt.brush").with("brush", PropertyValue::Enum(2)),
        ),
        key(
            "sculpt.inflate",
            "I",
            Context::Sculpt,
            Action::new("sculpt.brush").with("brush", PropertyValue::Enum(3)),
        ),
        // Blender's Shift+I: IK on the active bone, to the other selected one.
        key(
            "pose.ik_add",
            "I",
            Context::Pose,
            Action::new("pose.ik_add"),
        )
        .with_modifiers(Modifiers::shift()),
        key(
            "pose.select_invert",
            "I",
            Context::Pose,
            enum_param(Action::new("pose.select_all"), "action", 3),
        )
        .with_modifiers(Modifiers::ctrl()),
        key(
            "pose.clear_location",
            "G",
            Context::Pose,
            enum_param(Action::new("pose.clear_transform"), "what", 1),
        )
        .with_modifiers(Modifiers::alt()),
        key(
            "pose.clear_rotation",
            "R",
            Context::Pose,
            enum_param(Action::new("pose.clear_transform"), "what", 2),
        )
        .with_modifiers(Modifiers::alt()),
        key(
            "pose.clear_scale",
            "S",
            Context::Pose,
            enum_param(Action::new("pose.clear_transform"), "what", 3),
        )
        .with_modifiers(Modifiers::alt()),
        // --- armature edit mode: building the skeleton -------------------------
        key(
            "armature.translate",
            "G",
            Context::EditArmature,
            Action::new("armature.translate"),
        ),
        key(
            "armature.rotate",
            "R",
            Context::EditArmature,
            Action::new("armature.rotate"),
        ),
        key(
            "armature.scale",
            "S",
            Context::EditArmature,
            Action::new("armature.scale"),
        ),
        key(
            "armature.extrude",
            "E",
            Context::EditArmature,
            Action::new("armature.extrude"),
        ),
        key(
            "armature.delete",
            "X",
            Context::EditArmature,
            Action::new("armature.delete"),
        ),
        key(
            "armature.bone_add",
            "A",
            Context::EditArmature,
            Action::new("armature.bone_add"),
        )
        .with_modifiers(Modifiers::shift()),
        key(
            "armature.select_all",
            "A",
            Context::EditArmature,
            enum_param(Action::new("armature.select_all"), "action", 0),
        ),
        key(
            "armature.select_none",
            "A",
            Context::EditArmature,
            enum_param(Action::new("armature.select_all"), "action", 1),
        )
        .with_modifiers(Modifiers::alt()),
        key(
            "armature.select_invert",
            "I",
            Context::EditArmature,
            enum_param(Action::new("armature.select_all"), "action", 3),
        )
        .with_modifiers(Modifiers::ctrl()),
        // --- object mode ---------------------------------------------------
        // Selection as in Blender: A all, Alt+A none, Ctrl+I the rest.
        key(
            "object.select_all",
            "A",
            Context::ObjectMode,
            enum_param(Action::new("object.select_all"), "action", 0),
        ),
        key(
            "object.select_none",
            "A",
            Context::ObjectMode,
            enum_param(Action::new("object.select_all"), "action", 1),
        )
        .with_modifiers(Modifiers::alt()),
        key(
            "object.select_invert",
            "I",
            Context::ObjectMode,
            enum_param(Action::new("object.select_all"), "action", 3),
        )
        .with_modifiers(Modifiers::ctrl()),
        key(
            "object.duplicate",
            "D",
            Context::ObjectMode,
            Action::new("object.duplicate").with("linked", PropertyValue::Bool(true)),
        ),
        // `Ctrl+J` to join and `P` to separate, where every modeller's hand
        // expects them. `P` is free in edit mode; in the UV editor it packs,
        // which is a different context and therefore not a conflict.
        // Blender's Ctrl+P, its most-used choice: bind the selected meshes to
        // the active armature with automatic weights. The menu has the rest.
        // Blender's Ctrl+P and Alt+P: a menu of what to parent to, and of how
        // to let go.
        key(
            "object.parent_menu",
            "P",
            Context::ObjectMode,
            Action::new("app.menu").with("menu", PropertyValue::Text("parent".into())),
        )
        .with_modifiers(Modifiers::ctrl()),
        key(
            "object.parent_clear_menu",
            "P",
            Context::ObjectMode,
            Action::new("app.menu").with("menu", PropertyValue::Text("parent_clear".into())),
        )
        .with_modifiers(Modifiers::alt()),
        // Blender's H, Shift+H and Alt+H, in object mode as in edit mode.
        key(
            "object.hide",
            "H",
            Context::ObjectMode,
            Action::new("object.hide"),
        ),
        key(
            "object.hide_unselected",
            "H",
            Context::ObjectMode,
            Action::new("object.hide").with("unselected", PropertyValue::Bool(true)),
        )
        .with_modifiers(Modifiers::shift()),
        key(
            "object.reveal",
            "H",
            Context::ObjectMode,
            Action::new("object.reveal"),
        )
        .with_modifiers(Modifiers::alt()),
        // Blender's F2: rename the active object.
        key(
            "object.rename",
            "F2",
            Context::ObjectMode,
            Action::new("app.rename"),
        ),
        key(
            "object.join",
            "J",
            Context::ObjectMode,
            Action::new("object.join"),
        )
        .with_modifiers(Modifiers::ctrl()),
        // Blender's Bool Tool: select the cutters, then the object to cut,
        // and Ctrl with the numpad's minus, plus or star. The main row's minus
        // and equals (plus without Shift) for a keyboard with no numpad, and
        // Ctrl+Shift+8, the main row's star.
        key(
            "object.boolean_difference",
            "NumpadSubtract",
            Context::ObjectMode,
            enum_param(Action::new("object.boolean"), "operation", 0),
        )
        .with_modifiers(Modifiers::ctrl()),
        key(
            "object.boolean_union",
            "NumpadAdd",
            Context::ObjectMode,
            enum_param(Action::new("object.boolean"), "operation", 1),
        )
        .with_modifiers(Modifiers::ctrl()),
        key(
            "object.boolean_intersect",
            "NumpadMultiply",
            Context::ObjectMode,
            enum_param(Action::new("object.boolean"), "operation", 2),
        )
        .with_modifiers(Modifiers::ctrl()),
        key(
            "object.boolean_difference_main",
            "Minus",
            Context::ObjectMode,
            enum_param(Action::new("object.boolean"), "operation", 0),
        )
        .with_modifiers(Modifiers::ctrl()),
        key(
            "object.boolean_union_main",
            "Equal",
            Context::ObjectMode,
            enum_param(Action::new("object.boolean"), "operation", 1),
        )
        .with_modifiers(Modifiers::ctrl()),
        key(
            "object.boolean_intersect_main",
            "Digit8",
            Context::ObjectMode,
            enum_param(Action::new("object.boolean"), "operation", 2),
        )
        .with_modifiers(Modifiers::ctrl_shift()),
        key(
            "mesh.separate",
            "P",
            Context::EditMesh,
            Action::new("mesh.separate"),
        ),
        key(
            "object.delete",
            "X",
            Context::ObjectMode,
            Action::new("object.delete"),
        ),
        // Copy and paste on the keys every application puts them on. Both are
        // the application's rather than operators — `edit.`, like undo: copy
        // edits nothing, and paste has to be given the clipboard's text before
        // the `object.paste` operator can run.
        // Keyframes, on Blender's keys: `I` keys the selection's transform at
        // the current frame, `Alt+I` takes the keys there away. Object mode
        // only — `I` in edit mode is Inset.
        key(
            "anim.keyframe_insert",
            "I",
            Context::ObjectMode,
            Action::new("anim.keyframe_insert"),
        ),
        key(
            "anim.keyframe_delete",
            "I",
            Context::ObjectMode,
            Action::new("anim.keyframe_delete"),
        )
        .with_modifiers(Modifiers::alt()),
        key(
            "edit.copy",
            "C",
            Context::ObjectMode,
            Action::new("edit.copy"),
        )
        .with_modifiers(Modifiers::ctrl()),
        // Blender's C: circle select, a brush that selects what it passes over.
        key(
            "object.circle_select",
            "C",
            Context::ObjectMode,
            Action::new("view.circle_select"),
        ),
        key(
            "mesh.circle_select",
            "C",
            Context::EditMesh,
            Action::new("view.circle_select"),
        ),
        key(
            "edit.paste",
            "V",
            Context::ObjectMode,
            Action::new("edit.paste"),
        )
        .with_modifiers(Modifiers::ctrl()),
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
        // Shade smooth has no key. It had bare `N`, and `N` is the
        // viewport sidebar in every tool of this kind — which cannot share
        // it: `EditMesh` is searched before `Viewport`, so a sidebar
        // binding here would be *unreachable* in edit mode rather than
        // merely shadowed, and `Keymap::conflicts` says so. Shade smooth
        // stays on the toolbar and the right-click menu, which is also
        // where every other tool leaves it.
        // Box projection on the bare key, because it is the one projection
        // that needs nothing from the view: a binding is data and cannot carry
        // a camera. The view-dependent variant lives on the right-click menu,
        // which is the only place that has a view to pass. `U` for unwrap, as
        // every modeller's hand expects.
        // Blender's U ▸ Unwrap, its first entry and the one reached for; the
        // projections are on the right-click menu and the UV menu.
        key(
            "uv.unwrap",
            "U",
            Context::EditMesh,
            Action::new("uv.unwrap"),
        ),
        // Blender's K is the knife; marking sharp is in the Edge menu.
        key(
            "mesh.knife",
            "K",
            Context::EditMesh,
            Action::new("mesh.knife"),
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
        // Blender's L and Shift+L: the piece under the pointer, in or out.
        // Loops and rings are Alt+click and Ctrl+Alt+click, as in Blender.
        key(
            "mesh.select_linked_pick",
            "L",
            Context::EditMesh,
            Action::new("mesh.select_linked_pick"),
        ),
        key(
            "mesh.deselect_linked_pick",
            "L",
            Context::EditMesh,
            Action::new("mesh.select_linked_pick").with("deselect", PropertyValue::Bool(true)),
        )
        .with_modifiers(Modifiers::shift()),
        // Blender's grow, shrink and linked: Ctrl with the numpad's plus and
        // minus, and the main row's for a keyboard without one.
        key(
            "mesh.select_more",
            "NumpadAdd",
            Context::EditMesh,
            enum_param(Action::new("mesh.select"), "action", 6),
        )
        .with_modifiers(Modifiers::ctrl()),
        key(
            "mesh.select_more_main",
            "Equal",
            Context::EditMesh,
            enum_param(Action::new("mesh.select"), "action", 6),
        )
        .with_modifiers(Modifiers::ctrl()),
        key(
            "mesh.select_less",
            "NumpadSubtract",
            Context::EditMesh,
            enum_param(Action::new("mesh.select"), "action", 7),
        )
        .with_modifiers(Modifiers::ctrl()),
        key(
            "mesh.select_less_main",
            "Minus",
            Context::EditMesh,
            enum_param(Action::new("mesh.select"), "action", 7),
        )
        .with_modifiers(Modifiers::ctrl()),
        key(
            "mesh.select_linked",
            "L",
            Context::EditMesh,
            enum_param(Action::new("mesh.select"), "action", 8),
        )
        .with_modifiers(Modifiers::ctrl()),
        // H hides the selection, Shift+H everything else, Alt+H brings it back.
        key(
            "mesh.hide",
            "H",
            Context::EditMesh,
            Action::new("mesh.hide"),
        ),
        // Blender's Shift+G: Select Similar, a menu of what to compare.
        key(
            "mesh.select_similar",
            "G",
            Context::EditMesh,
            Action::new("app.menu").with("menu", PropertyValue::Text("similar".into())),
        )
        .with_modifiers(Modifiers::shift()),
        key(
            "mesh.hide_unselected",
            "H",
            Context::EditMesh,
            Action::new("mesh.hide").with("unselected", PropertyValue::Bool(true)),
        )
        .with_modifiers(Modifiers::shift()),
        key(
            "mesh.reveal",
            "H",
            Context::EditMesh,
            Action::new("mesh.reveal"),
        )
        .with_modifiers(Modifiers::alt()),
        key(
            "mesh.select_all",
            "A",
            Context::EditMesh,
            enum_param(Action::new("mesh.select"), "action", 0),
        ),
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

            // And the grid toggle is still reachable from object mode. It used
            // to be on bare `G`, where taking that key for Move would have
            // killed it outright rather than merely moved it: the inner
            // contexts are searched first and the application is always in one
            // of them. In edit mode Shift+G is Blender's Select Similar, and
            // the grid is in the Overlays menu.
            assert_eq!(
                k.resolve(
                    Context::ObjectMode,
                    &Input::key("G"),
                    Modifiers::shift(),
                    EventKind::Press
                )
                .map(|a| a.op_id),
                Some("view.toggle_grid".to_string()),
                "{preset:?}: the grid toggle is unreachable from object mode"
            );
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
        // And `M` in object mode is its own — Move to Collection, as in
        // Blender — rather than object mode quietly inheriting edit mode's merge.
        let object = k
            .resolve(
                Context::ObjectMode,
                &Input::key("M"),
                Modifiers::NONE,
                EventKind::Press,
            )
            .expect("M is bound in object mode");
        assert_eq!(object.op_id, "app.menu");
    }

    #[test]
    fn the_three_uv_keys_reach_one_operator_with_three_different_modes() {
        // One operator serves `G`, `R` and `S` in the UV editor, so the *mode*
        // is the only thing telling them apart. A falsification sweep changed
        // `R`'s mode to Move and nothing failed: all three keys would have
        // started a move, and the panel would have looked like rotate and
        // scale simply did nothing.
        let k = load(Preset::Kreate);
        let mode_of = |key: &str| -> Option<i64> {
            let action = k.resolve(
                Context::UvEditor,
                &Input::key(key),
                Modifiers::NONE,
                EventKind::Press,
            )?;
            assert_eq!(
                action.op_id, "uv.transform",
                "{key} reached the wrong operator"
            );
            action.params.get("mode").and_then(|v| v.as_int())
        };
        assert_eq!(mode_of("G"), Some(0), "G must move");
        assert_eq!(mode_of("R"), Some(1), "R must rotate");
        assert_eq!(mode_of("S"), Some(2), "S must scale");
    }

    #[test]
    fn the_four_uv_selection_keys_reach_four_different_actions() {
        // The same failure from the other side: `uv.select` takes an `action`,
        // and four bindings that agreed on it would look like three of the
        // keys did nothing.
        let k = load(Preset::Kreate);
        let action_of = |key: &str, mods: Modifiers| -> Option<i64> {
            let action = k.resolve(Context::UvEditor, &Input::key(key), mods, EventKind::Press)?;
            assert_eq!(action.op_id, "uv.select");
            action.params.get("action").and_then(|v| v.as_int())
        };
        assert_eq!(action_of("A", Modifiers::NONE), Some(0));
        assert_eq!(action_of("A", Modifiers::alt()), Some(1));
        assert_eq!(action_of("I", Modifiers::ctrl()), Some(2));
        assert_eq!(action_of("L", Modifiers::NONE), Some(3));
    }

    #[test]
    fn a_selects_all_and_the_add_and_apply_menus_are_where_blender_keeps_them() {
        let k = load(Preset::Kreate);
        let resolve =
            |context, key: &str, mods| k.resolve(context, &Input::key(key), mods, EventKind::Press);
        for (context, op) in [
            (Context::ObjectMode, "object.select_all"),
            (Context::EditMesh, "mesh.select"),
        ] {
            let all = resolve(context, "A", Modifiers::NONE).expect("A is bound");
            assert_eq!(
                (
                    all.op_id.as_str(),
                    all.params.get("action").and_then(|v| v.as_int())
                ),
                (op, Some(0))
            );
            let none = resolve(context, "A", Modifiers::alt()).expect("Alt+A is bound");
            assert_eq!(none.params.get("action").and_then(|v| v.as_int()), Some(1));
        }
        let menu = |key, mods| {
            let a = resolve(Context::ObjectMode, key, mods).expect("bound");
            assert_eq!(a.op_id, "app.menu");
            a.params.get("menu").cloned()
        };
        assert_eq!(
            menu("A", Modifiers::shift()),
            Some(PropertyValue::Text("add".into()))
        );
        assert_eq!(
            menu("A", Modifiers::ctrl()),
            Some(PropertyValue::Text("apply".into()))
        );
    }

    #[test]
    fn join_and_separate_are_reachable_from_the_keyboard() {
        // An operator nothing can reach is an operator nobody uses. Nothing
        // checked these two until a falsification sweep moved `Ctrl+J` to a key
        // nobody presses and every test still passed.
        let k = load(Preset::Kreate);
        assert_eq!(
            k.resolve(
                Context::ObjectMode,
                &Input::key("J"),
                Modifiers::ctrl(),
                EventKind::Press
            )
            .map(|a| a.op_id),
            Some("object.join".to_string())
        );
        assert_eq!(
            k.resolve(
                Context::EditMesh,
                &Input::key("P"),
                Modifiers::NONE,
                EventKind::Press
            )
            .map(|a| a.op_id),
            Some("mesh.separate".to_string())
        );
        // `P` means something else in the UV editor, which is a different
        // context and therefore not a conflict — worth pinning, because the
        // two *would* conflict if either moved into the other's scope.
        assert_eq!(
            k.resolve(
                Context::UvEditor,
                &Input::key("P"),
                Modifiers::NONE,
                EventKind::Press
            )
            .map(|a| a.op_id),
            Some("uv.pack".to_string())
        );
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
