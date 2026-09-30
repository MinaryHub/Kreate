//! Named layouts, and switching between them.
//!
//! A workspace is a [`Screen`] with a name. That is the whole idea, and it is
//! only worth having because the screen is data: laying the window out as a tree
//! of areas is what makes "save this arrangement and give it a name" a list
//! rather than a feature.
//!
//! # Why a shipped set rather than an empty one
//!
//! A tool that starts with one layout and a "new workspace" button teaches
//! nobody what workspaces are for. The three here are the arrangements the
//! editors that exist can actually express, and each answers a question the user
//! already has: where do I model, where do I see the scene, where do I edit UVs.
//! More arrive with the editors that need them — a shading workspace wants a
//! shader editor — and adding empty ones now would be furniture for rooms that
//! have not been built.
//!
//! # Why an index rather than a name
//!
//! Workspaces are renamed by the user, so a name is not a handle. The index is,
//! for as long as the list does not change under it — which is why switching
//! goes through the same deferred queue the layout edits use, rather than
//! happening halfway through a frame that is drawing tabs from the list.

use crate::area::{Axis, EditorId, Screen};

/// One named layout.
#[derive(Clone, Debug, PartialEq)]
pub struct Workspace {
    pub name: String,
    pub screen: Screen,
}

/// The set of layouts, and which one is in use.
///
/// Never empty and the active index is always valid: a window with no layout has
/// nothing to draw and no way back, which is the same argument [`Screen`] makes
/// about its last area.
#[derive(Clone, Debug, PartialEq)]
pub struct Workspaces {
    spaces: Vec<Workspace>,
    active: usize,
}

impl Default for Workspaces {
    fn default() -> Self {
        Self {
            spaces: vec![
                Workspace {
                    name: "Layout".into(),
                    screen: layout_screen(),
                },
                Workspace {
                    name: "Modeling".into(),
                    screen: modeling_screen(),
                },
                Workspace {
                    name: "UV Editing".into(),
                    screen: uv_screen(),
                },
                Workspace {
                    name: "Shading".into(),
                    screen: shading_screen(),
                },
                Workspace {
                    name: "Animation".into(),
                    screen: animation_screen(),
                },
                Workspace {
                    name: "Scripting".into(),
                    screen: scripting_screen(),
                },
            ],
            active: 0,
        }
    }
}

/// The viewport with the scene beside it: what a new session opens onto.
fn layout_screen() -> Screen {
    let mut screen = Screen::single(EditorId::VIEWPORT);
    let viewport = screen.areas()[0].0;
    let outliner = screen
        .split(viewport, Axis::Vertical, 0.78, EditorId::OUTLINER)
        .expect("the screen has one area to split");
    screen
        .split(outliner, Axis::Horizontal, 0.4, EditorId::PROPERTIES)
        .expect("the outliner was just created");
    // The timeline along the bottom of the viewport, where Blender's Layout
    // keeps it: the clock is always in reach, and short enough not to cost
    // the viewport anything a modeller would miss.
    screen
        .split(viewport, Axis::Horizontal, 0.84, EditorId::TIMELINE)
        .expect("the viewport is still there to split");
    screen
}

/// The viewport with only what a modelling tool needs beside it.
///
/// No outliner: the properties editor is where the last operation's parameters
/// and the modifier stack are, and those are what a hand reaches for while
/// shaping something. The scene list is a different question.
fn modeling_screen() -> Screen {
    // Blender's Modeling workspace keeps a small outliner above the
    // properties: a modeller switching between parts of a model needs to name
    // the part, and the tree is where names are.
    let mut screen = Screen::single(EditorId::VIEWPORT);
    let viewport = screen.areas()[0].0;
    let side = screen
        .split(viewport, Axis::Vertical, 0.82, EditorId::OUTLINER)
        .expect("the screen has one area to split");
    screen
        .split(side, Axis::Horizontal, 0.3, EditorId::PROPERTIES)
        .expect("the side was just created");
    screen
}

/// The scene and the console side by side: type a call, watch it happen.
///
/// Blender's Scripting workspace. The console takes the larger share because
/// a script line is long and a viewport is legible small.
fn scripting_screen() -> Screen {
    let mut screen = Screen::single(EditorId::VIEWPORT);
    let viewport = screen.areas()[0].0;
    screen
        .split(viewport, Axis::Vertical, 0.45, EditorId::CONSOLE)
        .expect("the screen has one area to split");
    screen
}

/// Blender's Shading workspace: the lit scene above, the material's graph
/// below, and the properties at the side for the material's slots.
fn shading_screen() -> Screen {
    let mut screen = Screen::single(EditorId::VIEWPORT);
    let viewport = screen.areas()[0].0;
    screen
        .split(viewport, Axis::Horizontal, 0.55, EditorId::SHADER)
        .expect("the screen has one area to split");
    screen
        .split(viewport, Axis::Vertical, 0.78, EditorId::PROPERTIES)
        .expect("the viewport is still there to split");
    screen
}

/// Blender's Animation workspace: the scene above, every key below with the
/// clock under it, and the outliner and properties at the side — the tree to
/// pick what is animated and the panel with its diamonds.
fn animation_screen() -> Screen {
    let mut screen = Screen::single(EditorId::VIEWPORT);
    let viewport = screen.areas()[0].0;
    let outliner = screen
        .split(viewport, Axis::Vertical, 0.8, EditorId::OUTLINER)
        .expect("the screen has one area to split");
    screen
        .split(outliner, Axis::Horizontal, 0.35, EditorId::PROPERTIES)
        .expect("the outliner was just created");
    let sheet = screen
        .split(viewport, Axis::Horizontal, 0.55, EditorId::DOPE_SHEET)
        .expect("the viewport is still there to split");
    screen
        .split(sheet, Axis::Horizontal, 0.66, EditorId::TIMELINE)
        .expect("the dope sheet was just created");
    screen
}

/// The mesh and its UVs side by side, which is the only way to unwrap anything.
fn uv_screen() -> Screen {
    let mut screen = Screen::single(EditorId::UV_EDITOR);
    let uv = screen.areas()[0].0;
    screen
        .split(uv, Axis::Vertical, 0.5, EditorId::VIEWPORT)
        .expect("the screen has one area to split");
    screen
}

impl Workspaces {
    /// One workspace, for a caller that wants to build its own set.
    pub fn single(name: impl Into<String>, screen: Screen) -> Self {
        Self {
            spaces: vec![Workspace {
                name: name.into(),
                screen,
            }],
            active: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.spaces.len()
    }

    /// Never true. Present because clippy asks for it beside `len`, and because
    /// saying so is clearer than the reader working it out.
    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn active_index(&self) -> usize {
        self.active
    }

    pub fn active(&self) -> &Workspace {
        // The index is an invariant, not an assumption: every path that could
        // move it clamps. Falling back rather than indexing blind, because a
        // panic here would take the window with it.
        self.spaces
            .get(self.active)
            .unwrap_or_else(|| &self.spaces[0])
    }

    pub fn screen(&self) -> &Screen {
        &self.active().screen
    }

    pub fn screen_mut(&mut self) -> &mut Screen {
        let index = self.active.min(self.spaces.len() - 1);
        &mut self.spaces[index].screen
    }

    pub fn iter(&self) -> impl Iterator<Item = &Workspace> {
        self.spaces.iter()
    }

    /// Switches to a workspace. Out-of-range is ignored.
    pub fn switch_to(&mut self, index: usize) -> bool {
        if index >= self.spaces.len() {
            return false;
        }
        self.active = index;
        true
    }

    /// Adds a workspace and makes it active.
    ///
    /// Active immediately, because the only reason to add one is to arrange it.
    pub fn add(&mut self, name: impl Into<String>, screen: Screen) -> usize {
        self.spaces.push(Workspace {
            name: name.into(),
            screen,
        });
        self.active = self.spaces.len() - 1;
        self.active
    }

    /// Copies the active workspace, names the copy, and switches to it.
    ///
    /// How a user makes their own: start from one that nearly works. Starting
    /// from an empty screen means rebuilding the arrangement they already had.
    pub fn duplicate_active(&mut self, name: impl Into<String>) -> usize {
        let screen = self.active().screen.clone();
        self.add(name, screen)
    }

    /// Removes a workspace, unless it is the last one.
    ///
    /// The active index follows: removing the active workspace leaves the one
    /// that took its place active, which is what closing a tab does everywhere
    /// else.
    pub fn remove(&mut self, index: usize) -> bool {
        if self.spaces.len() <= 1 || index >= self.spaces.len() {
            return false;
        }
        self.spaces.remove(index);
        if self.active > index {
            self.active -= 1;
        }
        self.active = self.active.min(self.spaces.len() - 1);
        true
    }

    /// Renames a workspace. An empty name is refused: a tab with no label is a
    /// tab the user cannot tell from its neighbour.
    pub fn rename(&mut self, index: usize, name: impl Into<String>) -> bool {
        let name = name.into();
        if name.trim().is_empty() {
            return false;
        }
        match self.spaces.get_mut(index) {
            Some(space) => {
                space.name = name;
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shipped_set_is_six_distinct_arrangements() {
        // Each has to answer a different question, or it is a tab that costs
        // the user a decision and gives nothing back. Layout, Modeling, UV
        // Editing, Shading — with the node graph — Animation — with every
        // key — and Scripting.
        let w = Workspaces::default();
        assert_eq!(w.len(), 6);
        assert_eq!(w.active_index(), 0);
        assert_eq!(w.active().name, "Layout");

        let names: Vec<&str> = w.iter().map(|s| s.name.as_str()).collect();
        let mut unique = names.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), names.len(), "two workspaces share a name");

        // And the layouts themselves differ, not just their names.
        let shapes: Vec<Vec<_>> = w.iter().map(|s| s.screen.to_nodes()).collect();
        for (i, a) in shapes.iter().enumerate() {
            for b in &shapes[i + 1..] {
                assert_ne!(a, b, "two workspaces have the same layout");
            }
        }

        // Every one of them can show the scene, which is the point of the tool.
        for space in w.iter() {
            assert!(
                space
                    .screen
                    .areas()
                    .iter()
                    .any(|(_, kind)| *kind == EditorId::VIEWPORT),
                "{} has no viewport",
                space.name
            );
        }
    }

    #[test]
    fn switching_changes_which_screen_is_edited() {
        // The whole point: an edit lands in the workspace that is showing, and
        // the others are left as they were. A `screen_mut` that returned the
        // first workspace's screen would round-trip every test above and lose
        // the user's arrangement the moment they switched.
        let mut w = Workspaces::default();
        let before: Vec<_> = w.iter().map(|s| s.screen.to_nodes()).collect();

        assert!(w.switch_to(2));
        assert_eq!(w.active_index(), 2);
        let area = w.screen().areas()[0].0;
        w.screen_mut()
            .split(area, Axis::Horizontal, 0.5, EditorId::PREFERENCES)
            .expect("split the active screen");

        let after: Vec<_> = w.iter().map(|s| s.screen.to_nodes()).collect();
        assert_eq!(after[0], before[0], "workspace 0 was edited");
        assert_eq!(after[1], before[1], "workspace 1 was edited");
        assert_ne!(after[2], before[2], "workspace 2 was not edited");
    }

    #[test]
    fn switching_out_of_range_is_ignored() {
        let mut w = Workspaces::default();
        assert!(!w.switch_to(99));
        assert_eq!(w.active_index(), 0, "the active workspace moved anyway");
    }

    #[test]
    fn a_new_workspace_starts_from_the_one_in_use() {
        // How a user makes their own: from one that nearly works. Starting from
        // an empty screen means rebuilding the arrangement they already had.
        let mut w = Workspaces::default();
        w.switch_to(2);
        let source = w.screen().to_nodes();

        let index = w.duplicate_active("Mine");
        assert_eq!(index, 6);
        assert_eq!(w.active_index(), 6, "the copy is not active");
        assert_eq!(w.active().name, "Mine");
        assert_eq!(w.screen().to_nodes(), source, "the copy is not a copy");

        // And it is a copy, not a share: editing one must not move the other.
        let area = w.screen().areas()[0].0;
        w.screen_mut()
            .split(area, Axis::Vertical, 0.5, EditorId::OUTLINER)
            .expect("split");
        assert_ne!(
            w.iter().nth(2).expect("the original").screen.to_nodes(),
            w.screen().to_nodes()
        );
    }

    #[test]
    fn removing_a_workspace_keeps_the_active_index_pointing_somewhere() {
        // Every position relative to the active one, because each moves the
        // index differently and an index past the end is a panic in the middle
        // of a frame.
        for active in 0..6 {
            for removed in 0..6 {
                let mut w = Workspaces::default();
                assert!(w.switch_to(active));
                let was_active = w.active().name.clone();
                assert!(w.remove(removed), "active={active} removed={removed}");

                assert_eq!(w.len(), 5);
                assert!(
                    w.active_index() < w.len(),
                    "active={active} removed={removed} left index {} of {}",
                    w.active_index(),
                    w.len()
                );
                // Removing something else must not move the user.
                if removed != active {
                    assert_eq!(
                        w.active().name,
                        was_active,
                        "active={active} removed={removed} switched workspace"
                    );
                }
            }
        }
    }

    #[test]
    fn the_last_workspace_cannot_be_removed() {
        // Same argument as the last area: a window with no layout has nothing
        // to draw and no way back.
        let mut w = Workspaces::single("Only", layout_screen());
        assert!(!w.remove(0));
        assert_eq!(w.len(), 1);
        assert!(!w.remove(99));
    }

    #[test]
    fn a_workspace_cannot_be_renamed_to_nothing() {
        // A tab with no label is a tab the user cannot tell from its neighbour.
        let mut w = Workspaces::default();
        assert!(w.rename(1, "Shaping"));
        assert_eq!(w.iter().nth(1).expect("there").name, "Shaping");

        assert!(!w.rename(1, ""));
        assert!(!w.rename(1, "   "), "whitespace is not a name");
        assert_eq!(w.iter().nth(1).expect("there").name, "Shaping");
        assert!(!w.rename(99, "Nowhere"));
    }
}
