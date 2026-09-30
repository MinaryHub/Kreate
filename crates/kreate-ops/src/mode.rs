//! Modes, and everything that changes when the mode does.
//!
//! §2.6 and §3's fourteenth "decide now": a mode is not a flag the interface
//! branches on, it is the axis that swaps four things at once — the keymap,
//! the header, which operators are available, and which overlays are drawn.
//! Scatter `if mode == EditMesh` through the application and it holds up for
//! two modes and falls over at seven, because every branch has to be found
//! again for every mode added.
//!
//! So each mode has one [`ModeInfo`] — a card that answers every question the
//! application asks of a mode — and the application asks the card. Adding a
//! mode is a variant and a card; the compiler points at the one `match` that
//! has to learn it, and nothing else in the program names a mode by variant.
//! `tests/mode_gate.rs` holds that last claim: it reads the source and fails
//! on a `Mode::EditMesh` anywhere but here.
//!
//! # What the card says
//!
//! Not "is this edit mode" but the questions that phrase stood in for. Sculpt
//! and weight paint will edit a mesh too, and will not want vertex dots or
//! element picking; pose mode acts on bones and needs an armature. Each of
//! those is a different answer to a question the card already asks, rather
//! than a new special case at every call site.

/// Which mode the application is in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Mode {
    #[default]
    Object,
    EditMesh,
    EditArmature,
    Pose,
}

/// What a mode needs to be entered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Needs {
    Nothing,
    /// An active mesh object.
    Mesh,
    /// An active armature object.
    Armature,
}

/// What there is to enter a mode with: the answers to every [`Needs`].
///
/// A struct rather than one flag per call, so a mode that needs something new
/// is a field here and not a new argument threaded through every caller.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Has {
    /// A mesh the edit tools can act on.
    pub mesh: bool,
    /// An active armature.
    pub armature: bool,
}

impl Has {
    pub const NOTHING: Has = Has { mesh: false, armature: false };
    pub const EVERYTHING: Has = Has { mesh: true, armature: true };

    /// What `doc` has, as the header and the application both ask it.
    pub fn of(doc: &kreate_scene::Document) -> Self {
        Self {
            mesh: crate::ops::mesh::target_mesh(doc).is_some(),
            armature: doc.selection.active.and_then(|a| doc.armature_of(a)).is_some(),
        }
    }
}

/// What a click, a drag and a transform act on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActsOn {
    /// Whole objects: picking selects them, `G` moves them.
    Objects,
    /// The vertices, edges and faces of the meshes being edited.
    MeshElements,
    /// The bones of the active armature, posed.
    Bones,
    /// The heads and tails of the active armature's bones, at rest.
    BoneJoints,
}

/// A header menu a mode offers after Add.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Menu {
    Object,
    /// Mesh, Vertex, Edge and Face — the four edit-mode menus.
    Mesh,
    /// Pose: selecting bones and clearing what was done to them.
    Pose,
    /// Armature: adding, extruding and deleting bones.
    Armature,
}

/// Everything the application asks of a mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModeInfo {
    pub mode: Mode,
    /// A stable name for keymaps, scripts and files. Not the discriminant:
    /// serialising that would make a saved keymap depend on declaration order.
    pub tag: &'static str,
    /// The catalogue key for the mode's name.
    pub label_key: &'static str,
    pub needs: Needs,
    pub acts_on: ActsOn,
    /// The keymap context keys resolve in, by its tag in `kreate-input`.
    ///
    /// A tag rather than the type, because this crate does not see the input
    /// crate and should not: an operator has no business knowing about keys.
    /// A test there holds every mode's tag to a real context.
    pub keymap: &'static str,
    /// The header menus after Add, in order.
    pub menus: &'static [Menu],
    /// Whether the vertex / edge / face buttons are in the header.
    pub element_buttons: bool,
    /// Whether the toolbar offers the edit tools (extrude, inset, bevel, loop
    /// cut) after the common ones.
    pub edit_tools: bool,
    /// Whether a mesh's elements are drawn as selectable dots and lines.
    pub shows_elements: bool,
    /// The operators `G`, `R` and `S` run, in that order.
    pub transform: [&'static str; 3],
    /// Where `Tab` goes from here: into what is being edited, or back out.
    pub tab: Mode,
}

/// The cards, in the order the mode menu lists them.
pub const MODES: [ModeInfo; 4] = [
    ModeInfo {
        mode: Mode::Object,
        tag: "object",
        label_key: "mode-object",
        needs: Needs::Nothing,
        acts_on: ActsOn::Objects,
        keymap: "object_mode",
        menus: &[Menu::Object],
        element_buttons: false,
        edit_tools: false,
        shows_elements: false,
        transform: ["object.translate", "object.rotate", "object.scale"],
        tab: Mode::EditMesh,
    },
    ModeInfo {
        mode: Mode::EditMesh,
        tag: "edit_mesh",
        label_key: "mode-edit",
        needs: Needs::Mesh,
        acts_on: ActsOn::MeshElements,
        keymap: "edit_mesh",
        menus: &[Menu::Mesh],
        element_buttons: true,
        edit_tools: true,
        shows_elements: true,
        transform: ["mesh.translate", "mesh.rotate", "mesh.scale"],
        tab: Mode::Object,
    },
    ModeInfo {
        mode: Mode::EditArmature,
        tag: "edit_armature",
        label_key: "mode-edit-armature",
        needs: Needs::Armature,
        acts_on: ActsOn::BoneJoints,
        keymap: "edit_armature",
        menus: &[Menu::Armature],
        element_buttons: false,
        edit_tools: false,
        shows_elements: false,
        transform: ["armature.translate", "armature.rotate", "armature.scale"],
        tab: Mode::Object,
    },
    ModeInfo {
        mode: Mode::Pose,
        tag: "pose",
        label_key: "mode-pose",
        needs: Needs::Armature,
        acts_on: ActsOn::Bones,
        keymap: "pose",
        menus: &[Menu::Pose],
        element_buttons: false,
        edit_tools: false,
        shows_elements: false,
        transform: ["pose.translate", "pose.rotate", "pose.scale"],
        // Blender's: Tab from pose mode goes to the skeleton's edit mode.
        tab: Mode::EditArmature,
    },
];

impl Mode {
    /// Every mode, in the order the header should offer them.
    pub const ALL: [Mode; 4] = [Mode::Object, Mode::EditMesh, Mode::EditArmature, Mode::Pose];

    /// This mode's card.
    ///
    /// The one `match` on the variants that the rest of the program relies
    /// on; a new mode is a compile error here until it has a card.
    pub fn info(self) -> &'static ModeInfo {
        match self {
            Mode::Object => &MODES[0],
            Mode::EditMesh => &MODES[1],
            Mode::EditArmature => &MODES[2],
            Mode::Pose => &MODES[3],
        }
    }

    pub fn tag(self) -> &'static str {
        self.info().tag
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        MODES.iter().find(|m| m.tag == tag).map(|m| m.mode)
    }

    pub fn label_key(self) -> &'static str {
        self.info().label_key
    }

    /// What clicks, drags and transforms act on in this mode.
    pub fn acts_on(self) -> ActsOn {
        self.info().acts_on
    }

    /// Whether the mode acts on the elements of the meshes being edited — the
    /// question most call sites that once said `== EditMesh` were asking.
    pub fn edits_elements(self) -> bool {
        self.acts_on() == ActsOn::MeshElements
    }

    /// Whether entering this mode requires an active mesh object.
    pub fn needs_mesh(self) -> bool {
        self.info().needs == Needs::Mesh
    }

    /// Whether this mode can be entered, given what is active.
    ///
    /// One expression, in one place: the header that greys out the entry and
    /// the application that refuses the change must not disagree.
    pub fn can_enter(self, has: Has) -> bool {
        match self.info().needs {
            Needs::Nothing => true,
            Needs::Mesh => has.mesh,
            Needs::Armature => has.armature,
        }
    }

    /// Whether a mode list shows this mode at all, given what there is.
    ///
    /// Greyed rather than hidden is the rule — a menu whose entries come and
    /// go cannot be learnt — except for another kind of object's modes. An
    /// armature's Edit Mode and a mesh's are both called Edit Mode, and a
    /// greyed twin beside the one you can enter is a riddle; Blender lists an
    /// object's own modes and no one else's.
    pub fn offered(self, has: Has) -> bool {
        match self.info().needs {
            Needs::Nothing => true,
            Needs::Mesh => !has.armature,
            Needs::Armature => has.armature,
        }
    }

    /// Where `Tab` goes from here, given what there is.
    ///
    /// The card's answer when it can be entered. Otherwise the first mode
    /// whose own Tab comes back here and that can be — so Tab on an armature
    /// in object mode edits the skeleton, as it edits a mesh's vertices on a
    /// mesh, without object mode's card having to know about either.
    pub fn toggled(self, has: Has) -> Mode {
        let card = self.info().tab;
        if card.can_enter(has) {
            return card;
        }
        Mode::ALL
            .iter()
            .copied()
            .find(|m| *m != self && m.info().tab == self && m.can_enter(has))
            .unwrap_or(card)
    }

    /// Whether a mesh's elements are on screen as selectable things.
    pub fn shows_elements(self) -> bool {
        self.info().shows_elements
    }

    /// The mode that runs when nothing has been asked for: a new document, a
    /// loaded one, the state after leaving any other.
    pub const RESTING: Mode = Mode::Object;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_mode_has_its_own_card_and_the_cards_are_in_order() {
        for (index, mode) in Mode::ALL.iter().enumerate() {
            assert_eq!(mode.info().mode, *mode, "{mode:?} reads another mode's card");
            assert_eq!(MODES[index].mode, *mode);
            assert_eq!(Mode::from_tag(mode.tag()), Some(*mode));
        }
        let mut tags: Vec<&str> = MODES.iter().map(|m| m.tag).collect();
        tags.sort_unstable();
        tags.dedup();
        assert_eq!(tags.len(), MODES.len(), "two modes share a tag");
        assert_eq!(Mode::from_tag("sculpt_from_the_future"), None);
    }

    #[test]
    fn tab_leads_somewhere_and_back() {
        // Every mode's Tab is another mode, and pressing it twice from the
        // resting mode comes home — the toggle a person means by that key.
        for mode in Mode::ALL {
            assert_ne!(mode.info().tab, mode, "{mode:?}: Tab goes nowhere");
        }
        let there = Mode::RESTING.info().tab;
        assert_eq!(there.info().tab, Mode::RESTING);
    }

    #[test]
    fn a_mode_that_acts_on_elements_is_the_one_that_shows_them() {
        // True of the two modes there are; the day sculpt arrives it acts on a
        // mesh without dots, and this test is where that decision is made on
        // purpose rather than by accident.
        for mode in Mode::ALL {
            let info = mode.info();
            assert_eq!(info.shows_elements, info.acts_on == ActsOn::MeshElements, "{mode:?}");
            assert_eq!(info.element_buttons, info.shows_elements, "{mode:?}");
            // Whatever it acts on, it needs what that lives on.
            if info.acts_on == ActsOn::MeshElements {
                assert_eq!(info.needs, Needs::Mesh, "{mode:?} edits a mesh it may not have");
            }
            if matches!(info.acts_on, ActsOn::Bones | ActsOn::BoneJoints) {
                assert_eq!(info.needs, Needs::Armature, "{mode:?} poses an armature it may not have");
            }
        }
    }

    #[test]
    fn g_r_and_s_are_operators_of_the_kind_the_mode_acts_on() {
        for mode in Mode::ALL {
            let prefix = match mode.acts_on() {
                ActsOn::Objects => "object.",
                ActsOn::MeshElements => "mesh.",
                ActsOn::Bones => "pose.",
                ActsOn::BoneJoints => "armature.",
            };
            for op in mode.info().transform {
                assert!(op.starts_with(prefix), "{mode:?}: {op}");
            }
        }
    }

    #[test]
    fn tab_edits_whatever_is_active() {
        let mesh = Has { mesh: true, armature: false };
        let armature = Has { mesh: false, armature: true };
        assert_eq!(Mode::Object.toggled(mesh), Mode::EditMesh);
        assert_eq!(Mode::Object.toggled(armature), Mode::EditArmature);
        assert_eq!(Mode::EditArmature.toggled(armature), Mode::Object);
        assert_eq!(Mode::Pose.toggled(armature), Mode::EditArmature);
    }
}
