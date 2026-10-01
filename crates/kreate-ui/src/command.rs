//! What the UI is allowed to ask for.
//!
//! The insulation clause: this crate *reads* a view of the world and *emits*
//! commands. It cannot mutate a document, cannot reach an operator
//! implementation, and cannot see the registry. The application drains the
//! queue and dispatches.
//!
//! That is not ceremony. It is what keeps undo, macro recording and scripting
//! working from a single mechanism, and it means the day egui turns out to be
//! the wrong toolkit, one crate is replaced rather than the application being
//! rewritten.

use kreate_reflect::PropertyMap;

/// Something the user asked for.
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    /// Select a bone of an armature object, by the object's outliner key and
    /// the bone's name: the outliner's bone rows. The application turns it
    /// into the pick of whatever mode it is in.
    SelectBone { object: u32, bone: String, extend: bool },
    /// Run an operator. The only way the UI causes a document change.
    RunOperator {
        op_id: String,
        params: PropertyMap,
    },
    /// Re-run the last operation with edited parameters — the adjust panel.
    AdjustLast {
        params: PropertyMap,
    },
    Undo,
    Redo,
    /// Undo or redo until this many steps are applied — Edit ▸ Undo History.
    UndoTo(usize),
    /// Replace the object selection.
    SetSelection {
        objects: Vec<u32>,
        active: Option<u32>,
    },
    /// Enters or leaves mesh edit mode.
    ///
    /// Not an operator: the mode lives in the application, not in the document,
    /// so there is nothing for an undo step to restore. Making it one would put
    /// a history entry between the user and the edit they actually want back.
    ToggleEditMode,
    /// Switches to a named mode.
    ///
    /// Carries the mode's *tag* rather than the mode, because the interface
    /// cannot see `kreate_ops::Mode` — it does not depend on the operator crate,
    /// on purpose, and this is the same reason it dispatches operators by
    /// `op_id`. An unknown tag is ignored by the application, which is the
    /// behaviour a script or a stale keymap should get.
    ///
    /// `ToggleEditMode` stays beside it as the gesture `Tab` performs: a toggle
    /// is what the user means by that key, and it is not expressible as "set
    /// this mode".
    SetMode(String),
    /// Chooses which element type edit-mode selection acts on.
    SetSelectMode(u32),
    /// Picks the sculpt brush: draw, smooth, grab, inflate.
    SetSculptBrush(u32),
    /// Choose how the viewport shades.
    ///
    /// Carries the mode rather than a tag, unlike [`Command::SetMode`]: the
    /// interface can see `kreate_render` — it draws a navigation gizmo out of a
    /// camera — and what it cannot see is the operator crate. The two look alike
    /// and the reason they differ is which dependency exists.
    SetShading(kreate_render::ShadingMode),
    /// Snap the view to look along a world axis.
    ///
    /// Carries the axis rather than a camera, so the application decides how to
    /// get there — which is what keeps the animation policy in one place instead
    /// of every caller choosing a duration.
    SnapToAxis(kreate_render::ViewAxis),
    /// Orbit by a screen-space drag on the navigation gizmo.
    OrbitBy {
        dx: f32,
        dy: f32,
    },
    /// Focus the viewport on the current selection.
    FrameSelection,
    /// Discard the document and start an empty one.
    ///
    /// Not an operator, for the same reason edit mode is not: there is no
    /// sensible undo step for "the document you were editing is gone", and
    /// putting one in the history would place a trap between the user and the
    /// work they wanted back.
    NewDocument,
    /// Ask for a file and open it.
    OpenDocument,
    /// Save to the document's own path, asking for one if it has none.
    SaveDocument,
    /// Always ask for a path.
    SaveDocumentAs,
    /// Ask for an image file and make it the world's background.
    ///
    /// A command rather than an operator because the *loading* is a file read,
    /// and an operator that touched the filesystem could not be replayed. The
    /// application opens the file and adds the datablock; the link is made by
    /// `world.set_image`, which is where the undo step belongs.
    PickWorldImage,
    /// Put the world's background back to a colour.
    ClearWorldImage,
    /// Ask for an image file and put it in one of the active material's
    /// texture slots.
    ///
    /// A command for the same reason `PickWorldImage` is one: the file read
    /// cannot live in an operator. `material.set_image` makes the link. `map`
    /// is the operator's parameter of the same name — which slot — and it
    /// also decides how the file is read: a base colour as sRGB, a roughness
    /// or normal map as the numbers it holds.
    PickMaterialImage { map: u32 },
    /// Take one of the active material's textures off.
    ClearMaterialImage { map: u32 },
    /// Ask for an image file and put it in an Image Texture node of the
    /// material in `slot`. A command for the reason `PickMaterialImage` is:
    /// `node.set_image` makes the link, the file read cannot be an operator.
    PickNodeImage { slot: u32, node: u32 },
    /// Give the active object a material, so that there is one to edit.
    ///
    /// A command rather than a menu entry wired straight to the operator
    /// because the button that runs it lives in the Material tab, which is the
    /// only place somebody looks when they want one.
    NewMaterial,
    /// Show or hide one object, named by the outliner's opaque key.
    ///
    /// From the eye on an outliner row, which acts on *that row* whether or
    /// not it is selected — a user hides the thing in the way, which is by
    /// definition not the thing they are working on. The application resolves
    /// the key to an identity and runs `object.set_visible`, so the edit is
    /// one undo step like any other.
    SetVisible {
        object: u32,
        visible: bool,
    },
    /// Run a script: operator calls, one per line, each its own undo step.
    ///
    /// From the console's input line. The application runs it through the
    /// history and hands the console back what happened to each line.
    RunScript {
        source: String,
    },
    /// Run an action the way a key press would.
    ///
    /// **The other door into the application.** `RunOperator` goes through
    /// the history, which is right for anything that edits the document and
    /// wrong for everything else: `view.frame_all` is not an undo step, and
    /// `object.translate` from a menu should start the same drag the G key
    /// starts rather than moving by a fixed amount. The keymap already routes
    /// both correctly; this hands a menu entry the same route, so a header
    /// menu and a key are two ways of saying one thing.
    Dispatch {
        op_id: String,
        params: PropertyMap,
    },
    /// Ask for a path and write the scene to an interchange format.
    ///
    /// Separate from saving, because it is not saving: an export goes through
    /// the IR and comes out the other side as much of the document as the
    /// format can hold, and the document keeps no path from it. Calling it
    /// "save" would invite a user to treat a `.glb` as their working file and
    /// find out later what it dropped.
    ExportScene,
    /// Open a recovered autosave, and forget it either way.
    RecoverDocument {
        path: std::path::PathBuf,
    },
    /// Discard a recovered autosave without opening it.
    DiscardRecovery {
        path: std::path::PathBuf,
    },
}

/// A queue the UI appends to and the application drains.
#[derive(Debug, Default)]
pub struct CommandQueue {
    commands: Vec<Command>,
}

impl CommandQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, command: Command) {
        self.commands.push(command);
    }

    pub fn run(&mut self, op_id: impl Into<String>, params: PropertyMap) {
        self.push(Command::RunOperator {
            op_id: op_id.into(),
            params,
        });
    }

    /// Queues an action to be dispatched the way a key press is. See
    /// [`Command::Dispatch`].
    pub fn dispatch(&mut self, op_id: impl Into<String>, params: PropertyMap) {
        self.push(Command::Dispatch {
            op_id: op_id.into(),
            params,
        });
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Takes everything queued, leaving the queue empty.
    ///
    /// Drained once per frame in the order the user caused it, so a
    /// duplicate-then-move pair from one click stays in that order.
    pub fn drain(&mut self) -> Vec<Command> {
        std::mem::take(&mut self.commands)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_reflect::PropertyValue;

    #[test]
    fn commands_drain_in_the_order_they_were_queued() {
        let mut q = CommandQueue::new();
        q.run("object.duplicate", PropertyMap::new());
        q.push(Command::Undo);
        q.run("object.translate", PropertyMap::new());

        let drained = q.drain();
        assert_eq!(drained.len(), 3);
        assert!(
            matches!(&drained[0], Command::RunOperator { op_id, .. } if op_id == "object.duplicate")
        );
        assert_eq!(drained[1], Command::Undo);
        assert!(q.is_empty(), "draining must empty the queue");
    }

    #[test]
    fn a_queued_operator_carries_its_parameters() {
        let mut q = CommandQueue::new();
        let mut p = PropertyMap::new();
        p.set("size", PropertyValue::Float(2.0));
        q.run("object.add_primitive", p.clone());

        match &q.drain()[0] {
            Command::RunOperator { params, .. } => {
                assert_eq!(params.get("size"), Some(&PropertyValue::Float(2.0)))
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn an_empty_queue_drains_to_nothing() {
        assert!(CommandQueue::new().drain().is_empty());
    }
}
