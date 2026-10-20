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
    /// The outliner's drag and drop onto an object: `child` parented to
    /// `parent`, both outliner keys. The selection is left alone.
    ParentObject {
        child: u32,
        parent: u32,
    },
    /// The outliner's drag and drop onto a collection's heading; an empty
    /// name is the scene itself.
    MoveObjectToCollection {
        object: u32,
        collection: String,
    },
    /// File ▸ Open Recent.
    OpenPath(std::path::PathBuf),
    /// Select a bone of an armature object, by the object's outliner key and
    /// the bone's name: the outliner's bone rows. The application turns it
    /// into the pick of whatever mode it is in.
    SelectBone {
        object: u32,
        bone: String,
        extend: bool,
    },
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
    /// Draw in a sketch again, by its object's name: select it and enter
    /// sketch mode — the feature list's double click.
    EditSketch(String),
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
    PickMaterialImage {
        map: u32,
    },
    /// Take one of the active material's textures off.
    ClearMaterialImage {
        map: u32,
    },
    /// Ask for an image file and put it in an Image Texture node of the
    /// material in `slot`. A command for the reason `PickMaterialImage` is:
    /// `node.set_image` makes the link, the file read cannot be an operator.
    PickNodeImage {
        slot: u32,
        node: u32,
    },
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
    /// From the Claude panel: say this to Claude, in the conversation so far.
    ChatSend {
        text: String,
    },
    /// From the chat panel: pick files to attach to the next message.
    ChatAttach,
    /// From the chat panel: take an attachment off the next message — an
    /// index into the list the panel was given.
    ChatDetach {
        index: usize,
    },
    /// From the Claude panel: stop the answer in progress.
    ChatStop,
    /// From the Claude panel: forget the conversation and start another.
    ChatNew,
    /// Opens the folder the conversations are kept in.
    ChatHistory,
    /// Saves the conversation as a Markdown file, where the user says.
    ChatExport,
    /// Opens the folder the application's logs are kept in.
    OpenLogFolder,
    /// Help ▸ Connect to AI Apps: add Kreate's connector to this assistant,
    /// by its place in `kreate::connect::App::ALL`.
    ConnectApp {
        index: usize,
    },
    /// From the chat panel: undo everything one answer did to the document.
    ChatRevertTurn {
        turn: usize,
    },
    /// From the chat panel: talk to another assistant — an index into the
    /// list the panel was given.
    ChatProvider {
        index: usize,
    },
    /// From the chat panel: answer with another of the assistant's models —
    /// an index into the list the panel was given.
    ChatModel {
        index: usize,
    },
    /// From the Claude panel: sign Claude Code in, in the browser.
    ChatSignIn,
    /// From the Claude panel: the code the sign-in page showed.
    ChatSignInCode {
        code: String,
    },
    /// From the Claude panel: give up on signing in.
    ChatSignInCancel,
    /// Signs the chat's assistant out of its account.
    ChatSignOut,
    /// File ▸ Shared Session.
    OpenCollab,
    /// From the session window: host a session on this machine.
    CollabHost {
        name: String,
        port: u16,
        key: String,
    },
    /// From the session window: join one at `address`.
    CollabJoin {
        name: String,
        address: String,
        key: String,
    },
    /// Leave the session; the scene stays as it was.
    CollabLeave,
    /// Help ▸ Contact.
    OpenContact,
    /// Help ▸ About Kreate.
    OpenAbout,
    /// Help ▸ Support Kreate: the sponsors' page, in the browser.
    Donate,
    /// From the contact form: send this to the developers. `kind` is one of
    /// `contact::KINDS`.
    ContactSend {
        kind: String,
        name: String,
        email: String,
        message: String,
        include_details: bool,
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
    /// Ask for a scene file and add what it holds to the document, as one
    /// undo step — beside what is there, where Open would replace it.
    ImportScene,
    /// The same, for one format: an index into the application's import
    /// formats, the file dialog showing only those files.
    ImportFormat {
        index: usize,
    },
    /// Export to one format: an index into the application's export formats.
    ExportFormat {
        index: usize,
    },
    /// From the asset browser: list this source (by its place in the menu).
    AssetSource {
        index: usize,
    },
    /// From the asset browser: list this kind — models, materials, HDRIs.
    AssetKind {
        index: usize,
    },
    /// From the asset browser: fetch at this size.
    AssetResolution {
        index: usize,
    },
    /// From the asset browser: fetch and add this asset.
    AssetAdd {
        index: usize,
    },
    /// From the asset browser: link this `.kreate`'s objects rather than copy
    /// them.
    AssetLink {
        index: usize,
    },
    /// From the asset browser: this asset is the one looked at.
    AssetSelect {
        index: usize,
    },
    /// From the asset browser: read the list again.
    AssetRefresh,
    /// From the asset browser: ask a source that searches itself for these
    /// words — Enter in the search box.
    AssetSearch {
        text: String,
    },
    /// From the asset browser: list only what moves, or everything.
    AssetAnimatedOnly {
        on: bool,
    },
    /// From the asset browser: make a model with AI, from these words — or,
    /// with none, from a picture the user is asked for.
    AssetGenerate {
        text: Option<String>,
    },
    /// From the asset browser: keep this Hugging Face token (empty to forget
    /// it).
    AssetHfToken {
        token: String,
    },
    /// From the asset browser: keep this key for the source that wants one
    /// (empty to forget it).
    AssetKey {
        key: String,
    },
    /// From the asset browser: ask for a folder to read.
    AssetAddFolder,
    /// From the asset browser: stop reading this folder.
    AssetRemoveFolder {
        index: usize,
    },
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
