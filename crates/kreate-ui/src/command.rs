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
    /// Chooses which element type edit-mode selection acts on.
    SetSelectMode(u32),
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
