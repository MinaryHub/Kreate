//! Every mode's card points at things that exist.
//!
//! The card names a keymap context by tag and operators by id, because the
//! crates it lives in cannot see the keymap or the registry. That is a
//! promise nothing checks at compile time, so it is checked here, where all
//! three are in reach: a tag that names no context would put a mode's keys in
//! object mode's, and an id that names no operator would make `G` do nothing.

use kreate_input::Context;
use kreate_ops::mode::{ActsOn, Menu};
use kreate_ops::{Mode, OperatorRegistry};

#[test]
fn every_modes_keys_resolve_in_a_context_keystrokes_arrive_in() {
    let leaves: Vec<Context> = Context::leaves().collect();
    for mode in Mode::ALL {
        let context = Context::from_tag(mode.info().keymap).unwrap_or_else(|| {
            panic!(
                "{mode:?} names keymap `{}`, which is no context",
                mode.info().keymap
            )
        });
        assert!(
            leaves.contains(&context),
            "{mode:?}'s context {context:?} is never dispatched in"
        );
    }
    for context in Context::ALL {
        assert_eq!(Context::from_tag(context.tag()), Some(*context));
    }
}

#[test]
fn every_modes_g_r_and_s_are_real_operators() {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    for mode in Mode::ALL {
        for op in mode.info().transform {
            assert!(
                registry.get(op).is_some(),
                "{mode:?}: `{op}` is not an operator"
            );
        }
    }
}

#[test]
fn the_header_offers_what_the_mode_acts_on() {
    // Object menus where objects are acted on, mesh menus where elements are —
    // a mode whose header offered the other would be a header whose every
    // entry declined.
    for mode in Mode::ALL {
        let info = mode.info();
        let want = match info.acts_on {
            ActsOn::Objects => Menu::Object,
            ActsOn::MeshElements => Menu::Mesh,
            ActsOn::Bones => Menu::Pose,
            ActsOn::BoneJoints => Menu::Armature,
            ActsOn::Weights => Menu::Weights,
            ActsOn::Sculpt => Menu::Sculpt,
            ActsOn::Texture => Menu::Texture,
            ActsOn::Sketch => Menu::Sketch,
        };
        assert!(
            info.menus.contains(&want),
            "{mode:?} does not offer {want:?}"
        );
    }
}
