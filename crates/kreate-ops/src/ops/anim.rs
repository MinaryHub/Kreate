//! Keyframes: putting the scene's values into its curves.
//!
//! Blender's `I` and `Alt+I`. Inserting a key reads what the selected objects
//! are now and writes it into their actions at a frame; deleting one takes
//! the keys at a frame away. Both are edits — they undo, they replay — and
//! both carry the frame as a parameter, because the frame the timeline is on
//! is not something a replay can know: a script that says "key at frame 12"
//! has to mean frame 12 whatever the clock says when it runs.

use kreate_ops_derive::OpParams;
use kreate_scene::animation::{self, Action, Interpolation};
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<InsertKeyframe>();
    registry.register::<DeleteKeyframe>();
}

/// Which properties a key is inserted for.
///
/// Blender's keying sets, the four a person reaches for: everything, or one of
/// the three. The order is the `what` parameter's.
pub const KEYING_SETS: [&[&str]; 4] = [
    &["location", "rotation_euler", "scale"],
    &["location"],
    &["rotation_euler"],
    &["scale"],
];

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct InsertKeyframeParams {
    /// Which frame. The application fills in the one the timeline is on.
    pub frame: f64,
    /// What to key.
    #[param(variants("loc_rot_scale", "location", "rotation", "scale"))]
    pub what: u32,
}

impl Default for InsertKeyframeParams {
    fn default() -> Self {
        Self { frame: 1.0, what: 0 }
    }
}

/// Keys the selected objects' transforms at a frame.
///
/// An object with no action gets one, named after it, which is what Blender
/// does and what makes the first `I` on a fresh scene work.
pub struct InsertKeyframe;

impl Operator for InsertKeyframe {
    type Params = InsertKeyframeParams;
    fn id() -> &'static str {
        "anim.keyframe_insert"
    }
    fn label() -> &'static str {
        "Insert Keyframe"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        !doc.selection.is_empty()
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        if !params.frame.is_finite() {
            return Ok(OpResult::Cancelled);
        }
        let properties = KEYING_SETS
            .get(params.what as usize)
            .copied()
            .unwrap_or(KEYING_SETS[0]);
        let targets = doc.document().selection.objects.clone();
        let d = doc.document_mut();
        let mut keyed = false;
        for id in targets {
            let Some(object) = d.object(id) else { continue };
            let transform = object.transform;
            let action = match object.action.filter(|a| d.action(*a).is_some()) {
                Some(action) => action,
                None => {
                    let name = format!("{}Action", object.name);
                    let action = d.add_action(Action::new(name));
                    if let Some(o) = d.object_mut(id) {
                        o.action = Some(action);
                    }
                    action
                }
            };
            let Some(action) = d.action_mut(action) else { continue };
            for property in properties {
                let Some(value) = animation::read_object(&transform, property) else {
                    continue;
                };
                for index in 0..3u8 {
                    action.curve_mut(property, index).insert(
                        params.frame,
                        value[index as usize],
                        Interpolation::default(),
                    );
                }
            }
            keyed = true;
        }
        Ok(if keyed {
            OpResult::Finished
        } else {
            OpResult::Cancelled
        })
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct DeleteKeyframeParams {
    pub frame: f64,
}

impl Default for DeleteKeyframeParams {
    fn default() -> Self {
        Self { frame: 1.0 }
    }
}

/// Takes away every key the selected objects have at a frame.
///
/// A curve left with no keys is removed too, so the property it drove stops
/// being animated rather than being held at a value nobody keyed.
pub struct DeleteKeyframe;

impl Operator for DeleteKeyframe {
    type Params = DeleteKeyframeParams;
    fn id() -> &'static str {
        "anim.keyframe_delete"
    }
    fn label() -> &'static str {
        "Delete Keyframe"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.selection
            .objects
            .iter()
            .any(|id| doc.object(*id).is_some_and(|o| o.action.is_some()))
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let actions: Vec<_> = {
            let d = doc.document();
            // Two selected objects sharing one action delete once.
            let mut seen = std::collections::HashSet::new();
            d.selection
                .objects
                .iter()
                .filter_map(|id| d.object(*id).and_then(|o| o.action))
                .filter(|a| seen.insert(*a))
                .collect()
        };
        // Asked before anything is borrowed mutably, so a frame with nothing
        // on it is a cancel with no step to undo.
        let any = actions.iter().any(|a| {
            doc.document()
                .action(*a)
                .is_some_and(|a| a.curves.iter().any(|c| c.keys.iter().any(|k| (k.frame - params.frame).abs() < animation::SAME_FRAME)))
        });
        if !any {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        for action in actions {
            if let Some(action) = d.action_mut(action) {
                for curve in &mut action.curves {
                    curve.remove(params.frame);
                }
                action.curves.retain(|c| !c.keys.is_empty());
            }
        }
        Ok(OpResult::Finished)
    }
}
