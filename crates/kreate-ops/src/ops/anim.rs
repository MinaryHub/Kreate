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
    registry.register::<SelectKeysBox>();
    registry.register::<SelectAllKeys>();
    registry.register::<MoveKeys>();
    registry.register::<DeleteKeys>();
    registry.register::<SetKeyInterpolation>();
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

// --- the dope sheet's edits -------------------------------------------------
//
// These act on "the keys the dope sheet shows": the actions of the selected
// objects, which is what Blender's dope sheet shows with Only Show Selected —
// its default. So they need no list of which actions, and a replay acts on the
// same keys whenever the same objects are selected.

/// The actions the dope sheet shows, once each, in the selection's order.
pub fn shown_actions(doc: &Document) -> Vec<kreate_scene::ActionId> {
    let mut seen = std::collections::HashSet::new();
    doc.selection
        .objects
        .iter()
        .filter_map(|id| doc.object(*id).and_then(|o| o.action))
        .filter(|a| doc.action(*a).is_some() && seen.insert(*a))
        .collect()
}

fn any_shown(doc: &Document) -> bool {
    !shown_actions(doc).is_empty()
}

/// One dope-sheet row, as text: an action's identity, a slash, and the
/// property the row is for — empty for the row that stands for them all.
pub fn row_target(action_uid: u64, property: Option<&str>) -> String {
    format!("{action_uid}/{}", property.unwrap_or(""))
}

fn parse_targets(text: &str) -> Vec<(u64, Option<String>)> {
    text.split_whitespace()
        .filter_map(|t| {
            let (uid, property) = t.split_once('/')?;
            let uid = uid.parse().ok()?;
            Some((uid, (!property.is_empty()).then(|| property.to_string())))
        })
        .collect()
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SelectKeysBoxParams {
    /// Which rows, as `row_target` writes them, separated by spaces.
    ///
    /// Text rather than a list of numbers because a row is an action *and* a
    /// property, and because a script line should say what it selects: `12/location`.
    pub targets: String,
    pub frame_min: f64,
    pub frame_max: f64,
    /// Add to the selection rather than replacing it.
    pub extend: bool,
}

/// Selects the keys in some rows between two frames: a click is a box one
/// frame wide on one row.
pub struct SelectKeysBox;

impl Operator for SelectKeysBox {
    type Params = SelectKeysBoxParams;
    fn id() -> &'static str {
        "anim.keys_select_box"
    }
    fn label() -> &'static str {
        "Select Keys"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_shown(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let shown = shown_actions(doc.document());
        let targets = parse_targets(&params.targets);
        let d = doc.document_mut();
        let mut changed = false;
        if !params.extend {
            for id in &shown {
                if let Some(a) = d.action_mut(*id) {
                    changed |= a.select_all_with(|_| false);
                }
            }
        }
        for (uid, property) in targets {
            // Only the actions on show: a target naming another is a stale
            // script line, and selecting keys nobody can see is a surprise.
            let Some(id) = d.action_by_uid(kreate_scene::Uid::from_raw(uid)).filter(|a| shown.contains(a)) else {
                continue;
            };
            if let Some(a) = d.action_mut(id) {
                changed |= a.select_range(property.as_deref(), params.frame_min, params.frame_max, true);
            }
        }
        Ok(if changed { OpResult::Finished } else { OpResult::Cancelled })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SelectAllKeysParams {
    #[param(variants("select", "deselect", "invert"))]
    pub action: u32,
}

/// Selects, deselects or inverts every key the dope sheet shows.
pub struct SelectAllKeys;

impl Operator for SelectAllKeys {
    type Params = SelectAllKeysParams;
    fn id() -> &'static str {
        "anim.keys_select_all"
    }
    fn label() -> &'static str {
        "Select All Keys"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_shown(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let shown = shown_actions(doc.document());
        let d = doc.document_mut();
        let mut changed = false;
        for id in shown {
            if let Some(a) = d.action_mut(id) {
                changed |= match params.action {
                    0 => a.select_all_with(|_| true),
                    1 => a.select_all_with(|_| false),
                    _ => a.select_all_with(|s| !s),
                };
            }
        }
        Ok(if changed { OpResult::Finished } else { OpResult::Cancelled })
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct MoveKeysParams {
    /// How many frames. Whole in practice — the dope sheet rounds — but a
    /// script may say otherwise.
    pub offset: f64,
}

/// Moves the selected keys along time.
pub struct MoveKeys;

impl Operator for MoveKeys {
    type Params = MoveKeysParams;
    fn id() -> &'static str {
        "anim.keys_move"
    }
    fn label() -> &'static str {
        "Move Keys"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_shown(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        if params.offset == 0.0 || !params.offset.is_finite() {
            return Ok(OpResult::Cancelled);
        }
        let shown = shown_actions(doc.document());
        if !shown.iter().any(|a| doc.document().action(*a).is_some_and(|a| a.any_selected())) {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        for id in shown {
            if let Some(a) = d.action_mut(id) {
                a.move_selected(params.offset);
            }
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct DeleteKeysParams {}

/// Deletes the selected keys.
pub struct DeleteKeys;

impl Operator for DeleteKeys {
    type Params = DeleteKeysParams;
    fn id() -> &'static str {
        "anim.keys_delete"
    }
    fn label() -> &'static str {
        "Delete Keys"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_shown(doc)
    }
    fn execute(
        _params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let shown = shown_actions(doc.document());
        if !shown.iter().any(|a| doc.document().action(*a).is_some_and(|a| a.any_selected())) {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        for id in shown {
            if let Some(a) = d.action_mut(id) {
                a.delete_selected();
            }
        }
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, Default, PartialEq, OpParams)]
pub struct SetKeyInterpolationParams {
    #[param(variants("constant", "linear", "bezier"))]
    pub interpolation: u32,
}

/// Sets how the selected keys travel to the next: Blender's `T`.
pub struct SetKeyInterpolation;

impl Operator for SetKeyInterpolation {
    type Params = SetKeyInterpolationParams;
    fn id() -> &'static str {
        "anim.keys_interpolation"
    }
    fn label() -> &'static str {
        "Set Interpolation"
    }
    fn poll(_ctx: &Context, doc: &Document) -> bool {
        any_shown(doc)
    }
    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(kind) = Interpolation::ALL.get(params.interpolation as usize).copied() else {
            return Ok(OpResult::Cancelled);
        };
        let shown = shown_actions(doc.document());
        let any = shown.iter().any(|a| {
            doc.document().action(*a).is_some_and(|a| {
                a.curves.iter().any(|c| c.keys.iter().any(|k| k.selected && k.interpolation != kind))
            })
        });
        if !any {
            return Ok(OpResult::Cancelled);
        }
        let d = doc.document_mut();
        for id in shown {
            if let Some(a) = d.action_mut(id) {
                a.set_selected_interpolation(kind);
            }
        }
        Ok(OpResult::Finished)
    }
}

