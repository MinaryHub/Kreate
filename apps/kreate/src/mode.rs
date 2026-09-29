//! What changing mode entails.
//!
//! The decision, separated from the doing. `App::set_mode` needs a window, a
//! GPU and an event loop, so nothing that lives inside it can be tested — and a
//! falsification sweep proved that outright: removing the guard that stops the
//! application entering edit mode with no mesh broke no test at all, and neither
//! did swapping the selection sweep that runs on the way in for the one that
//! runs on the way out.
//!
//! So the parts that are a decision are here, where they are a pure function of
//! three values. This is the same shape `gpu_sync` takes, for the same stated
//! reason: "did it do more than it had to" is invisible at a glance, and the
//! only place it can be pinned down is a function that takes what it knows and
//! returns what it decided.
//!
//! What is left in the application is the application of this: four statements,
//! each of which the compiler checks.

use kreate_ops::operator::Mode;

/// What a mode change requires of the application.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Change {
    /// The mode to end up in.
    pub to: Mode,
    /// Whether the mesh's elements should all be selected, or all deselected.
    ///
    /// Everything on the way into edit mode, because entering with nothing
    /// selected makes every tool a no-op and the user's first keystroke appear
    /// to do nothing at all. Nothing on the way out, so the object-mode outline
    /// is not competing with a stale element highlight.
    pub select_all: bool,
    /// Whether an edit session should be open afterwards.
    ///
    /// Only for a mode that edits a mesh's elements. Holding a `BMesh` for a
    /// mesh nobody is editing costs about four times what the document form
    /// does, and the plan is explicit that it lives no longer than edit mode.
    pub open_session: bool,
}

/// Decides whether a mode can be entered, and what entering it entails.
///
/// `None` means the change does not happen: either it is already the current
/// mode, or the mode needs something that is not there. Refusing rather than
/// entering a mode with nothing to act on — a mode where every tool declines is
/// worse than not having switched, because the user has no way to tell which
/// happened.
pub fn plan(from: Mode, to: Mode, has_mesh: bool) -> Option<Change> {
    if from == to || !to.can_enter(has_mesh) {
        return None;
    }
    Some(Change {
        to,
        select_all: to.edits_elements(),
        open_session: to.edits_elements() && has_mesh,
    })
}

/// The mode `Tab` goes to from here.
///
/// A toggle, which is what the user means by that key and which is not
/// expressible as "set this mode". It stays a two-way gesture as modes are
/// added: out of whatever is being edited, or back into it.
pub fn toggled(from: Mode) -> Mode {
    from.info().tab
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entering_edit_mode_needs_a_mesh() {
        // The guard whose removal broke nothing. Entering a mode with nothing to
        // act on leaves every tool declining, and the user cannot tell that from
        // the tool being broken.
        assert_eq!(plan(Mode::Object, Mode::EditMesh, false), None);
        assert!(plan(Mode::Object, Mode::EditMesh, true).is_some());

        // Leaving does not: an object-mode user with nothing selected is the
        // ordinary case, and refusing to leave would trap them.
        assert!(plan(Mode::EditMesh, Mode::Object, false).is_some());
    }

    #[test]
    fn the_selection_sweep_runs_the_right_way_round() {
        // Swapping these two is a bug with no visible cause: entering edit mode
        // would clear the selection the user had, and the tool they reach for
        // next would decline. A sweep that swapped them broke nothing before
        // this test existed.
        let entering = plan(Mode::Object, Mode::EditMesh, true).expect("can enter");
        assert!(entering.select_all, "entering must select everything");

        let leaving = plan(Mode::EditMesh, Mode::Object, true).expect("can leave");
        assert!(!leaving.select_all, "leaving must clear the selection");
    }

    #[test]
    fn the_edit_session_lives_exactly_as_long_as_the_mode_that_needs_it() {
        assert!(
            plan(Mode::Object, Mode::EditMesh, true)
                .expect("can enter")
                .open_session
        );
        assert!(
            !plan(Mode::EditMesh, Mode::Object, true)
                .expect("can leave")
                .open_session
        );
    }

    #[test]
    fn changing_to_the_mode_already_in_use_is_not_a_change() {
        // Otherwise `Tab` pressed twice quickly, or a script setting the mode it
        // is already in, would sweep the selection and rebuild the edit session
        // for nothing — which at a million vertices is a quarter of a second of
        // the user's time to arrive where they already were.
        for mode in Mode::ALL {
            assert_eq!(plan(mode, mode, true), None, "{mode:?}");
            assert_eq!(plan(mode, mode, false), None, "{mode:?}");
        }
    }

    #[test]
    fn every_mode_can_be_planned_into_and_out_of() {
        // From `Mode::ALL` rather than a list written here: a mode added
        // without an answer to "what does entering it entail" is exactly what
        // this is for, and a hardcoded pair would not see it.
        for to in Mode::ALL {
            let entered = Mode::ALL
                .into_iter()
                .filter(|from| *from != to)
                .any(|from| plan(from, to, true).is_some());
            assert!(entered, "no mode reaches {to:?}");
        }
    }

    #[test]
    fn tab_is_a_two_way_gesture() {
        assert_eq!(toggled(Mode::Object), Mode::EditMesh);
        assert_eq!(toggled(Mode::EditMesh), Mode::Object);
        // And it always leads somewhere else, from every mode there is.
        for mode in Mode::ALL {
            assert_ne!(toggled(mode), mode, "{mode:?} toggles to itself");
        }
    }
}
