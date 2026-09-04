//! Reading files older than this build.
//!
//! The rule the plan sets, and the reason it is a module rather than a habit:
//! **every migration is kept forever, and each one is tested against a committed
//! file that was actually written by the version it migrates from.** A migration
//! that is only tested against a file synthesised by today's code tests today's
//! idea of what that version wrote, which is exactly the thing that drifts.
//!
//! # Why this exists before there is anything to migrate
//!
//! The chain is empty today: version 1.0 is the only version. Building the
//! mechanism now is not speculation about future changes — it is about where the
//! *first* one will go. A migration bolted on later tends to arrive as an `if`
//! in the reader, then a second one beside it, and within a year the reader
//! cannot say which versions it supports. A chain can: it is a list, and the list
//! is the answer.
//!
//! # What a migration may and may not do
//!
//! It rewrites the *file's* representation, never the document. By the time a
//! document exists the migration has already happened, so nothing downstream has
//! to know which version a file came from — which is the property that stops
//! version checks leaking into the rest of the program.
//!
//! Migrations run on the parsed JSON rather than on typed records, because a
//! record type describes the *current* shape and an old file does not have it.
//! Deserialising first and migrating after would mean every old field had to
//! survive in today's structs, which is how a format ends up carrying a decade
//! of dead fields.

use serde_json::Value;

/// A single step from one minor version to the next.
pub struct Migration {
    /// The version this reads.
    pub from: (u32, u32),
    /// The version it produces.
    pub to: (u32, u32),
    /// What it is for, in a few words. Shown when a file is upgraded.
    pub description: &'static str,
    /// Rewrites the manifest and the scene in place.
    pub apply: fn(manifest: &mut Value, scene: &mut Value),
}

#[derive(Debug, thiserror::Error)]
pub enum MigrationError {
    #[error("no way to read version {major}.{minor}: {reason}")]
    Unsupported {
        major: u32,
        minor: u32,
        reason: &'static str,
    },
}

/// Every migration this build knows, in order.
///
/// Append only. Removing one makes a file that used to open stop opening, and
/// the person it stops opening for will be someone who has not upgraded in two
/// years — exactly the person least able to work around it.
pub fn chain() -> &'static [Migration] {
    &[]
}

/// Brings a parsed file up to the current version.
///
/// Returns the descriptions of the steps applied, which is worth surfacing: a
/// user whose file was silently rewritten on load deserves to know, and so does
/// a bug report.
pub fn migrate(
    manifest: &mut Value,
    scene: &mut Value,
    from: (u32, u32),
    to: (u32, u32),
) -> Result<Vec<&'static str>, MigrationError> {
    if from.0 != to.0 {
        return Err(MigrationError::Unsupported {
            major: from.0,
            minor: from.1,
            reason: "major versions are not migrated, only refused or read directly",
        });
    }
    if from.1 > to.1 {
        // A newer minor version. Additive by definition, so it is read as-is —
        // the unknown fields ride along in the `extra` maps and go back out
        // untouched. This is not a migration, it is the absence of one.
        return Ok(Vec::new());
    }

    let mut at = from;
    let mut applied = Vec::new();
    while at != to {
        let step = chain().iter().find(|m| m.from == at).ok_or({
            MigrationError::Unsupported {
                major: at.0,
                minor: at.1,
                reason: "no migration leads out of this version",
            }
        })?;
        (step.apply)(manifest, scene);
        applied.push(step.description);
        at = step.to;
    }
    Ok(applied)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic chain, so the mechanism is exercised while the real chain is
    /// empty.
    ///
    /// Without this the migration code would be entirely untested until the day
    /// it first mattered — which is the day a user's file is at stake.
    fn synthetic() -> Vec<Migration> {
        vec![
            Migration {
                from: (1, 0),
                to: (1, 1),
                description: "rename a field",
                apply: |manifest, _scene| {
                    if let Some(old) = manifest.get("old_name").cloned() {
                        manifest["new_name"] = old;
                        manifest.as_object_mut().unwrap().remove("old_name");
                    }
                },
            },
            Migration {
                from: (1, 1),
                to: (1, 2),
                description: "add a default",
                apply: |_manifest, scene| {
                    scene["added"] = serde_json::json!(42);
                },
            },
        ]
    }

    /// The same walk as [`migrate`], against a chain the test supplies.
    fn run(
        steps: &[Migration],
        manifest: &mut Value,
        scene: &mut Value,
        from: (u32, u32),
        to: (u32, u32),
    ) -> Vec<&'static str> {
        let mut at = from;
        let mut applied = Vec::new();
        while at != to {
            let step = steps.iter().find(|m| m.from == at).expect("a step");
            (step.apply)(manifest, scene);
            applied.push(step.description);
            at = step.to;
        }
        applied
    }

    #[test]
    fn a_chain_of_migrations_runs_in_order() {
        let steps = synthetic();
        let mut manifest = serde_json::json!({ "old_name": "value" });
        let mut scene = serde_json::json!({});
        let applied = run(&steps, &mut manifest, &mut scene, (1, 0), (1, 2));

        assert_eq!(applied, vec!["rename a field", "add a default"]);
        assert_eq!(manifest["new_name"], "value");
        assert!(manifest.get("old_name").is_none());
        assert_eq!(scene["added"], 42);
    }

    #[test]
    fn a_partial_walk_stops_where_it_is_told() {
        // A file two versions old being read by a build one version old. The
        // chain has to stop at the reader, not run to the end of the list.
        let steps = synthetic();
        let mut manifest = serde_json::json!({ "old_name": "value" });
        let mut scene = serde_json::json!({});
        let applied = run(&steps, &mut manifest, &mut scene, (1, 0), (1, 1));
        assert_eq!(applied, vec!["rename a field"]);
        assert!(scene.get("added").is_none(), "it ran a step too far");
    }

    #[test]
    fn a_current_file_is_not_migrated() {
        let mut manifest = serde_json::json!({});
        let mut scene = serde_json::json!({});
        let applied = migrate(&mut manifest, &mut scene, (1, 0), (1, 0)).expect("no work");
        assert!(applied.is_empty());
    }

    #[test]
    fn a_newer_minor_version_is_read_rather_than_migrated() {
        // Minor bumps are additive, so a file from a newer build needs no
        // migration — its extra fields ride along in the `extra` maps. Migrating
        // *backwards* would mean inventing a way to remove things, which is how
        // a format loses data on a round trip through an older version.
        let mut manifest = serde_json::json!({});
        let mut scene = serde_json::json!({});
        let applied = migrate(&mut manifest, &mut scene, (1, 9), (1, 0)).expect("read as-is");
        assert!(applied.is_empty());
    }

    #[test]
    fn a_gap_in_the_chain_is_an_error_rather_than_a_silent_skip() {
        // The failure this guards: a migration deleted or misnumbered, leaving a
        // version nothing leads out of. Reading such a file as though it were
        // current would interpret old data with new rules, which is worse than
        // refusing it.
        let mut manifest = serde_json::json!({});
        let mut scene = serde_json::json!({});
        let err = migrate(&mut manifest, &mut scene, (1, 0), (1, 5)).unwrap_err();
        assert!(matches!(err, MigrationError::Unsupported { .. }));
    }

    #[test]
    fn a_different_major_version_is_not_migrated() {
        let mut manifest = serde_json::json!({});
        let mut scene = serde_json::json!({});
        assert!(migrate(&mut manifest, &mut scene, (2, 0), (1, 0)).is_err());
    }

    #[test]
    fn the_chain_is_contiguous_and_ordered() {
        // An invariant over the real chain, so a migration added out of order or
        // with a wrong `from` fails here rather than on someone's file. Vacuous
        // while the chain is empty, and exactly the test that stops being vacuous
        // the moment it matters.
        for pair in chain().windows(2) {
            assert_eq!(
                pair[0].to, pair[1].from,
                "the chain jumps from {:?} to {:?}",
                pair[0].to, pair[1].from
            );
        }
        for step in chain() {
            assert_eq!(step.from.0, step.to.0, "a step crosses a major version");
            assert!(step.to.1 > step.from.1, "a step does not move forward");
        }
    }
}
