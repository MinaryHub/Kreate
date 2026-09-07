//! What a command looks like to a user: a name, a sentence, and its shortcut.
//!
//! # Why this is one type rather than three lookups at each call site
//!
//! A menu entry, a toolbar button's tooltip and a context-menu row all need the
//! same three facts, resolved the same way. Written out at each place they are
//! needed, the three would drift: one would forget the shortcut, another would
//! fall back to the operator id when a translation is missing, a third would
//! show the *binding* id instead of the operator's name. They are assembled
//! once, here.
//!
//! # Where each fact comes from
//!
//! - **Name** — the catalogue, keyed `op-<id>` with dots turned to dashes, so
//!   `mesh.extrude_region` is `op-mesh-extrude-region`. Falls back to the
//!   operator's built-in English label, which is why a missing translation
//!   shows readable text rather than a key.
//! - **Description** — the same key with `-desc`. Optional: a command whose name
//!   says everything needs no sentence, and inventing one for it would be noise
//!   in every tooltip.
//! - **Shortcut** — resolved through the *keymap*, never a hardcoded table. A
//!   user who rebound a key has to see the key they chose; §2.5 makes remapping
//!   a first-class feature and a tooltip that lied about it would undo that.
//!
//! # Why the description is a separate translation toggle
//!
//! §3.5 ships "translate the interface" and "translate tooltips" as independent
//! switches, because many Korean professionals want an English interface with
//! Korean explanations: the menu names are what they read in tutorials and
//! remember, and the explanations are what they actually want in their own
//! language. That distinction lives in the catalogue, and this type just asks
//! for each in its own category.

use kreate_i18n::Catalog;

/// A command as a user sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandInfo {
    pub name: String,
    /// One sentence, or empty when the name says enough.
    pub description: String,
    /// The key that runs it, or empty when nothing does.
    pub shortcut: String,
}

impl CommandInfo {
    /// Whether there is anything worth showing in a tooltip.
    ///
    /// A tooltip with only the name repeats the label the pointer is already
    /// over, and a box that appears to say nothing is worse than no box.
    pub fn has_tooltip(&self) -> bool {
        !self.description.is_empty() || !self.shortcut.is_empty()
    }

    /// The tooltip text: the sentence, then the shortcut.
    pub fn tooltip(&self) -> String {
        match (self.description.is_empty(), self.shortcut.is_empty()) {
            (true, true) => String::new(),
            (false, true) => self.description.clone(),
            (true, false) => self.shortcut.clone(),
            (false, false) => format!("{}\n{}", self.description, self.shortcut),
        }
    }
}

/// The catalogue key for an operator's name.
///
/// Dots to dashes, because Fluent identifiers cannot contain a dot: `mesh.
/// extrude_region` becomes `op-mesh-extrude-region`. Underscores go too, so the
/// key reads the way every other key in the catalogue does.
pub fn name_key(op_id: &str) -> String {
    format!("op-{}", op_id.replace(['.', '_'], "-"))
}

/// The catalogue key for a *command*, which may be one variant of an operator.
///
/// Several menu rows can reach one operator with different arguments —
/// `mesh.shade` smooths or facets, `mesh.normals` recalculates or flips — and
/// they are different commands to the user with different names. The
/// distinguishing parameters are appended, so `mesh.shade` with `smooth=true`
/// keys as `op-mesh-shade-smooth-true`.
///
/// Only the parameters the caller passed, and in the map's own sorted order, so
/// the key is stable: a key that depended on insertion order would resolve
/// differently between two callers asking the same question.
pub fn variant_id(op_id: &str, params: &kreate_reflect::PropertyMap) -> String {
    let mut id = op_id.to_string();
    for (name, value) in params.iter() {
        // Decided *before* anything is appended. Written the other way round —
        // push the name, then skip the value — it left a trailing `offset.` on
        // the key for every command carrying a setting.
        let Some(selector) = variant_value(value) else {
            continue;
        };
        id.push('.');
        id.push_str(name);
        id.push('.');
        id.push_str(&selector);
    }
    id
}

/// A parameter's contribution to a variant key, if it selects a variant at all.
///
/// Only the discrete kinds. An offset or a width is a *setting* of one command
/// rather than a different command, and putting it in the key would give every
/// distinct offset its own name — none of which would resolve.
fn variant_value(value: &kreate_reflect::PropertyValue) -> Option<String> {
    match value {
        kreate_reflect::PropertyValue::Bool(b) => Some(b.to_string()),
        kreate_reflect::PropertyValue::Enum(e) => Some(e.to_string()),
        kreate_reflect::PropertyValue::Int(i) => Some(i.to_string()),
        _ => None,
    }
}

/// The catalogue key for an operator's one-sentence description.
pub fn description_key(op_id: &str) -> String {
    format!("{}-desc", name_key(op_id))
}

/// Assembles what is known about a command.
///
/// `fallback_name` is the operator's built-in English label, used when the
/// catalogue has no entry — a partly translated build stays usable and a missing
/// key shows as readable text rather than as `op-mesh-extrude-region`.
pub fn describe(
    catalog: &Catalog,
    op_id: &str,
    fallback_name: &str,
    shortcut: Option<String>,
) -> CommandInfo {
    let name_key = name_key(op_id);
    let name = if catalog.has(&name_key) {
        catalog.get(&name_key)
    } else {
        fallback_name.to_string()
    };
    let description_key = description_key(op_id);
    let description = if catalog.has(&description_key) {
        catalog.format(&description_key, kreate_i18n::TextKind::Tooltip, &[])
    } else {
        String::new()
    };
    CommandInfo {
        name,
        description,
        shortcut: shortcut.unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_fluent_identifiers() {
        // Fluent identifiers cannot contain a dot, and an operator id is full of
        // them. Underscores go the same way so the key reads like every other
        // key in the catalogue.
        assert_eq!(name_key("mesh.extrude_region"), "op-mesh-extrude-region");
        assert_eq!(
            description_key("mesh.extrude_region"),
            "op-mesh-extrude-region-desc"
        );
        assert_eq!(name_key("object.add_primitive"), "op-object-add-primitive");
    }

    #[test]
    fn a_tooltip_with_nothing_in_it_is_not_shown() {
        let bare = CommandInfo {
            name: "Extrude".into(),
            description: String::new(),
            shortcut: String::new(),
        };
        assert!(!bare.has_tooltip());
        assert!(bare.tooltip().is_empty());
    }

    #[test]
    fn a_shortcut_alone_is_worth_a_tooltip() {
        // The commonest case in a young build: no prose written yet, but the key
        // is the thing a user most wants to learn from hovering.
        let info = CommandInfo {
            name: "Extrude".into(),
            description: String::new(),
            shortcut: "E".into(),
        };
        assert!(info.has_tooltip());
        assert_eq!(info.tooltip(), "E");
    }

    #[test]
    fn a_tooltip_puts_the_sentence_above_the_shortcut() {
        let info = CommandInfo {
            name: "Extrude".into(),
            description: "Pulls the selected faces out along their normal.".into(),
            shortcut: "E".into(),
        };
        assert_eq!(
            info.tooltip(),
            "Pulls the selected faces out along their normal.\nE"
        );
    }
}
