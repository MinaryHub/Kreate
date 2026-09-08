//! The screen: a tree of areas, each showing one editor.
//!
//! # Why the layout is data rather than code
//!
//! The shell used to name its panels at the call site — `Panel::left("outliner")`,
//! `Panel::right("properties")` — which reads well and stops working at about
//! the sixth panel. The role was baked into the code, so the user could not move
//! anything, and every new editor added a line to one function.
//!
//! The plan puts nine more editors on the schedule: a shader editor and an image
//! editor with materials, a console and an info log with scripting, a timeline, a
//! dope sheet, a graph editor and a drivers editor with animation, a render
//! result with rendering. Nine more hardcoded panels is an application whose
//! layout nobody can change, and converting *then* means rewriting all nine.
//!
//! So the window is a tree: every node is either a split or an area, and an area
//! shows whichever editor it is set to. Splitting, joining, resizing and changing
//! an area's editor are all edits to this structure, and the structure
//! serialises — which is what makes named workspaces a saved layout rather than a
//! special case.
//!
//! # Why ids rather than paths
//!
//! An area's state — a UV editor's zoom, an outliner's scroll, a graph editor's
//! view range — has to survive its neighbours being split and closed. A path
//! into the tree does not: splitting anything above an area changes its path,
//! and the state would follow the wrong area or be lost. Ids are stable for the
//! life of an area, so state keyed on them stays put.
//!
//! Splits carry ids for the same reason, one level down: a drag on a boundary has
//! to name the split it is moving, and it has to keep naming the same one while
//! the pointer is down.
//!
//! # What is not here
//!
//! The top bar, the status bar and the recovery notice are not areas. They belong
//! to the window rather than to the layout, the user cannot move them, and every
//! application in this category treats them the same way.

use egui::{Pos2, Rect, Vec2};

/// A stable handle for an area, so per-area state survives edits elsewhere.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AreaId(pub u32);

/// A stable handle for a boundary between two areas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SplitId(pub u32);

/// Which way a split divides its space.
///
/// `Vertical` is a vertical *line* — the two children sit side by side. Named
/// after the boundary rather than after the arrangement because that is what the
/// user grabs, and because "horizontal split" is ambiguous in every direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    /// A vertical boundary: children are left and right.
    Vertical,
    /// A horizontal boundary: children are top and bottom.
    Horizontal,
}

/// What an area is showing.
///
/// A closed set for now, matched in one place in the shell. It becomes a
/// registry of trait objects when a script can register an editor of its own —
/// the tree above does not change when that happens, which is the point of
/// keeping the two apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EditorKind {
    Viewport,
    Outliner,
    Properties,
    UvEditor,
    Preferences,
}

impl EditorKind {
    /// Every kind, in the order a menu should offer them.
    pub const ALL: [EditorKind; 5] = [
        EditorKind::Viewport,
        EditorKind::Outliner,
        EditorKind::Properties,
        EditorKind::UvEditor,
        EditorKind::Preferences,
    ];

    /// The catalogue key for this editor's name.
    ///
    /// A key rather than a string: the editor-type dropdown is UI prose and goes
    /// through Fluent like the rest of it.
    pub fn name_key(self) -> &'static str {
        match self {
            EditorKind::Viewport => "editor-viewport",
            EditorKind::Outliner => "editor-outliner",
            EditorKind::Properties => "editor-properties",
            EditorKind::UvEditor => "editor-uv",
            EditorKind::Preferences => "editor-preferences",
        }
    }

    /// A stable name for the file format, independent of the enum's order.
    ///
    /// Serialising the discriminant would make a saved layout depend on the
    /// order variants happen to be declared in, so inserting an editor in the
    /// middle would silently reinterpret everyone's saved workspaces.
    pub fn tag(self) -> &'static str {
        match self {
            EditorKind::Viewport => "viewport",
            EditorKind::Outliner => "outliner",
            EditorKind::Properties => "properties",
            EditorKind::UvEditor => "uv",
            EditorKind::Preferences => "preferences",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.tag() == tag)
    }
}

/// One node of a screen, in the order a pre-order walk visits them.
///
/// The serialised form of the tree. A pre-order sequence of a full binary tree
/// is unambiguous — read a node, and if it is a split, read its two children
/// the same way — which means the file needs no nesting, no brackets and no
/// indentation to be read back exactly.
///
/// **Area ids are not part of it.** They are session-local handles for
/// session-local state, and inventing them at load time is both simpler and
/// more honest than writing numbers that mean nothing to the next run. What is
/// lost is a UV editor's zoom across a restart, which is worth less than a
/// format that cannot be got wrong.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum NodeSpec {
    Split { axis: Axis, ratio: f32 },
    Area { editor: EditorKind },
}

impl Axis {
    /// A stable name for the file format.
    pub fn tag(self) -> &'static str {
        match self {
            Axis::Vertical => "v",
            Axis::Horizontal => "h",
        }
    }

    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "v" => Some(Axis::Vertical),
            "h" => Some(Axis::Horizontal),
            _ => None,
        }
    }
}

/// How thin an area may become, in interface points.
///
/// Applied when the user drags a boundary. Not applied when the *window*
/// shrinks: squeezing areas is what a small window does, and refusing to lay out
/// at all would be worse than laying out small.
pub const MIN_AREA: f32 = 48.0;

/// How wide the draggable strip between two areas is.
pub const HANDLE: f32 = 4.0;

#[derive(Clone, Debug, PartialEq)]
enum Node {
    Split {
        id: SplitId,
        axis: Axis,
        /// The first child's share, in `0.0..=1.0`.
        ratio: f32,
        first: Box<Node>,
        second: Box<Node>,
    },
    Leaf {
        id: AreaId,
        editor: EditorKind,
    },
}

/// One workspace's layout.
#[derive(Clone, Debug, PartialEq)]
pub struct Screen {
    root: Node,
    /// One counter for both kinds of id, so an area and a split can never be
    /// confused for one another in a saved file or a log line.
    next: u32,
}

/// Where an area ended up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AreaRect {
    pub id: AreaId,
    pub editor: EditorKind,
    pub rect: Rect,
}

/// A draggable boundary.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Handle {
    pub id: SplitId,
    pub axis: Axis,
    /// The strip the pointer grabs.
    pub rect: Rect,
    /// The space this boundary divides.
    ///
    /// Not the screen. A nested split's ratio is a share of its *parent's*
    /// extent, so a drag that measured against the window would set the ratio
    /// from the wrong denominator and the boundary would not follow the
    /// pointer. The caller cannot work this out from the handle's own rect,
    /// which spans only the parent in one axis.
    pub parent: Rect,
}

/// Everything a frame needs to draw the screen.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Layout {
    pub areas: Vec<AreaRect>,
    pub handles: Vec<Handle>,
}

impl Screen {
    /// A screen of one area.
    pub fn single(editor: EditorKind) -> Self {
        Self {
            root: Node::Leaf {
                id: AreaId(0),
                editor,
            },
            next: 1,
        }
    }

    fn take_id(&mut self) -> u32 {
        let id = self.next;
        self.next += 1;
        id
    }

    /// Every area, in tree order.
    pub fn areas(&self) -> Vec<(AreaId, EditorKind)> {
        let mut out = Vec::new();
        collect(&self.root, &mut out);
        out
    }

    pub fn area_count(&self) -> usize {
        self.areas().len()
    }

    /// What `area` is showing.
    pub fn editor(&self, area: AreaId) -> Option<EditorKind> {
        self.areas()
            .into_iter()
            .find(|(id, _)| *id == area)
            .map(|(_, editor)| editor)
    }

    /// Points `area` at another editor. `false` if there is no such area.
    pub fn set_editor(&mut self, area: AreaId, kind: EditorKind) -> bool {
        match find_leaf_mut(&mut self.root, area) {
            Some(editor) => {
                *editor = kind;
                true
            }
            None => false,
        }
    }

    /// Splits `area` in two, returning the id of the new one.
    ///
    /// The existing area keeps its id and becomes the first child, so its state
    /// stays with it. The new area is the second child — below or to the right,
    /// which is where a user expects the new one to appear.
    pub fn split(
        &mut self,
        area: AreaId,
        axis: Axis,
        ratio: f32,
        editor: EditorKind,
    ) -> Option<AreaId> {
        // The ids are taken before the tree is touched, because taking them
        // needs `&mut self` and so does the walk.
        let split_id = SplitId(self.take_id());
        let new_area = AreaId(self.take_id());
        let ratio = sane_ratio(ratio);

        let Some(node) = find_node_mut(&mut self.root, area) else {
            // Nothing was split, so the ids are spent. Harmless: they are a
            // counter, not a slot, and a gap in them means nothing.
            return None;
        };
        // The old subtree is *moved* into the new split rather than rebuilt, so
        // an area really is the same area afterwards and keeps whatever state is
        // keyed on its id. The leaf left behind is a placeholder for one
        // statement and is overwritten on the next line.
        let new_leaf = Node::Leaf {
            id: new_area,
            editor,
        };
        let existing = std::mem::replace(node, new_leaf.clone());
        *node = Node::Split {
            id: split_id,
            axis,
            ratio,
            first: Box::new(existing),
            second: Box::new(new_leaf),
        };
        Some(new_area)
    }

    /// Closes `area`; its sibling takes the space they shared.
    ///
    /// `false` when there is no such area, or when it is the last one — a screen
    /// with no areas has nothing to show and no way back.
    pub fn close(&mut self, area: AreaId) -> bool {
        if matches!(self.root, Node::Leaf { .. }) {
            return false;
        }
        close_in(&mut self.root, area)
    }

    /// Moves a boundary. The ratio is clamped so neither side goes below
    /// [`MIN_AREA`] within `extent` points of space.
    pub fn resize(&mut self, split: SplitId, ratio: f32, extent: f32) -> bool {
        let Some(r) = find_split_mut(&mut self.root, split) else {
            return false;
        };
        // A space too small for two minimum areas cannot honour the clamp, and
        // pinning the ratio to 0.5 there at least keeps the drag continuous
        // rather than snapping to one end.
        *r = if extent < MIN_AREA * 2.0 || !ratio.is_finite() {
            0.5
        } else {
            let low = MIN_AREA / extent;
            ratio.clamp(low, 1.0 - low)
        };
        true
    }

    /// The tree as a flat pre-order sequence.
    pub fn to_nodes(&self) -> Vec<NodeSpec> {
        let mut out = Vec::new();
        flatten(&self.root, &mut out);
        out
    }

    /// Rebuilds a screen from a pre-order sequence, with fresh area ids.
    ///
    /// `None` when the sequence is not exactly one tree: empty, truncated — a
    /// split whose children are missing — or with nodes left over. A caller
    /// reading a file should fall back to the default layout, which is what the
    /// rest of the preferences layer does with anything it cannot make sense of:
    /// a settings file must never be a reason the application will not start.
    pub fn from_nodes(nodes: &[NodeSpec]) -> Option<Self> {
        let mut next = 0u32;
        let mut rest = nodes;
        let root = build(&mut rest, &mut next)?;
        // Trailing nodes mean the file described more than one tree, which is
        // not a layout. Accepting the first and discarding the rest would give
        // the user a screen they did not save.
        rest.is_empty().then_some(Self { root, next })
    }

    /// Where everything goes, given the space the screen has.
    ///
    /// Both lists come out in the tree's own traversal order, which for
    /// `handles` means **innermost first**: the boundary at the root of the tree
    /// is the last one, not the first. Nothing depends on the order, and a
    /// caller that wants a particular boundary should say which by its axis or
    /// its id rather than by its position.
    pub fn layout(&self, rect: Rect) -> Layout {
        let mut out = Layout::default();
        place(&self.root, rect, &mut out);
        out
    }
}

fn flatten(node: &Node, out: &mut Vec<NodeSpec>) {
    match node {
        Node::Leaf { editor, .. } => out.push(NodeSpec::Area { editor: *editor }),
        Node::Split {
            axis,
            ratio,
            first,
            second,
            ..
        } => {
            out.push(NodeSpec::Split {
                axis: *axis,
                ratio: *ratio,
            });
            flatten(first, out);
            flatten(second, out);
        }
    }
}

/// Consumes one node and its children from the front of `nodes`.
fn build(nodes: &mut &[NodeSpec], next: &mut u32) -> Option<Node> {
    let (head, tail) = nodes.split_first()?;
    *nodes = tail;
    match *head {
        NodeSpec::Area { editor } => {
            let id = AreaId(*next);
            *next += 1;
            Some(Node::Leaf { id, editor })
        }
        NodeSpec::Split { axis, ratio } => {
            let id = SplitId(*next);
            *next += 1;
            // Both children, in order, or nothing: a split with one child is
            // not a smaller tree, it is a broken one.
            let first = Box::new(build(nodes, next)?);
            let second = Box::new(build(nodes, next)?);
            Some(Node::Split {
                id,
                axis,
                ratio: sane_ratio(ratio),
                first,
                second,
            })
        }
    }
}

fn collect(node: &Node, out: &mut Vec<(AreaId, EditorKind)>) {
    match node {
        Node::Leaf { id, editor } => out.push((*id, *editor)),
        Node::Split { first, second, .. } => {
            collect(first, out);
            collect(second, out);
        }
    }
}

fn find_leaf_mut(node: &mut Node, area: AreaId) -> Option<&mut EditorKind> {
    match node {
        Node::Leaf { id, editor } => (*id == area).then_some(editor),
        Node::Split { first, second, .. } => {
            find_leaf_mut(first, area).or_else(|| find_leaf_mut(second, area))
        }
    }
}

fn find_node_mut(node: &mut Node, area: AreaId) -> Option<&mut Node> {
    match node {
        Node::Leaf { id, .. } => (*id == area).then_some(node),
        Node::Split { first, second, .. } => {
            // Written with a raw match rather than `or_else` because both arms
            // borrow `node` mutably and the borrow checker needs them apart.
            if let Some(found) = find_node_mut(first, area) {
                return Some(found);
            }
            find_node_mut(second, area)
        }
    }
}

fn find_split_mut(node: &mut Node, split: SplitId) -> Option<&mut f32> {
    match node {
        Node::Leaf { .. } => None,
        Node::Split {
            id,
            ratio,
            first,
            second,
            ..
        } => {
            if *id == split {
                return Some(ratio);
            }
            if let Some(found) = find_split_mut(first, split) {
                return Some(found);
            }
            find_split_mut(second, split)
        }
    }
}

/// Replaces the split containing `area` with `area`'s sibling.
fn close_in(node: &mut Node, area: AreaId) -> bool {
    let Node::Split { first, second, .. } = node else {
        return false;
    };
    let keep = match (first.as_ref(), second.as_ref()) {
        (Node::Leaf { id, .. }, _) if *id == area => Some(false),
        (_, Node::Leaf { id, .. }) if *id == area => Some(true),
        _ => None,
    };
    if let Some(keep_first) = keep {
        // Moved out whole so a surviving *subtree* keeps all of its ids and all
        // of the state keyed on them. The leaf left behind is a placeholder for
        // one statement.
        let Node::Split { first, second, .. } = std::mem::replace(
            node,
            Node::Leaf {
                id: AreaId(u32::MAX),
                editor: EditorKind::Viewport,
            },
        ) else {
            unreachable!("matched a split on the line above");
        };
        *node = if keep_first { *first } else { *second };
        return true;
    }
    let Node::Split { first, second, .. } = node else {
        unreachable!("matched a split at the top of this function");
    };
    close_in(first, area) || close_in(second, area)
}

/// A ratio that can be used, whatever arrived.
///
/// The file is text a user or another build can have written, and `clamp`
/// answers NaN with NaN — so clamping alone leaves a NaN in the tree, and every
/// coordinate derived from it is NaN too. An area at NaN is not drawn and cannot
/// be clicked, and nothing says why.
fn sane_ratio(ratio: f32) -> f32 {
    if ratio.is_finite() {
        ratio.clamp(0.0, 1.0)
    } else {
        0.5
    }
}

/// Where a boundary sits inside the space it divides.
///
/// Held half a handle away from both edges, because that is what stops a child
/// coming out with a negative extent. A ratio of exactly zero or one is *legal*
/// — `split` clamps into that range and so does the file format — and it puts
/// the line on an edge, half a handle past the child on that side.
/// `Rect::from_min_max` builds the inverted result without complaint, and an
/// inverted rect is not something egui reports: the area is simply not there.
///
/// When the space is narrower than a handle there is no such position and the
/// midpoint is the least bad answer: the areas come out with nothing in them,
/// which is what a window dragged narrower than its contents should do.
fn boundary(min: f32, extent: f32, ratio: f32) -> f32 {
    let half = HANDLE * 0.5;
    let (lo, hi) = (min + half, min + extent - half);
    if lo > hi {
        return min + extent * 0.5;
    }
    (min + extent * sane_ratio(ratio)).clamp(lo, hi)
}

fn place(node: &Node, rect: Rect, out: &mut Layout) {
    match node {
        Node::Leaf { id, editor } => out.areas.push(AreaRect {
            id: *id,
            editor: *editor,
            rect,
        }),
        Node::Split {
            id,
            axis,
            ratio,
            first,
            second,
        } => {
            let half = HANDLE * 0.5;
            match axis {
                Axis::Vertical => {
                    let x = boundary(rect.min.x, rect.width(), *ratio);
                    place(
                        first,
                        Rect::from_min_max(rect.min, Pos2::new(x - half, rect.max.y)),
                        out,
                    );
                    place(
                        second,
                        Rect::from_min_max(Pos2::new(x + half, rect.min.y), rect.max),
                        out,
                    );
                    out.handles.push(Handle {
                        id: *id,
                        axis: *axis,
                        rect: Rect::from_center_size(
                            Pos2::new(x, rect.center().y),
                            Vec2::new(HANDLE, rect.height()),
                        ),
                        parent: rect,
                    });
                }
                Axis::Horizontal => {
                    let y = boundary(rect.min.y, rect.height(), *ratio);
                    place(
                        first,
                        Rect::from_min_max(rect.min, Pos2::new(rect.max.x, y - half)),
                        out,
                    );
                    place(
                        second,
                        Rect::from_min_max(Pos2::new(rect.min.x, y + half), rect.max),
                        out,
                    );
                    out.handles.push(Handle {
                        id: *id,
                        axis: *axis,
                        rect: Rect::from_center_size(
                            Pos2::new(rect.center().x, y),
                            Vec2::new(rect.width(), HANDLE),
                        ),
                        parent: rect,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deliberately not square and not centred: a square rect lets an axis
    /// mix-up pass, and one at the origin hides an offset that is added instead
    /// of multiplied.
    fn space() -> Rect {
        Rect::from_min_size(Pos2::new(30.0, 70.0), Vec2::new(1200.0, 800.0))
    }

    /// A screen of four areas over three splits of mixed axis, with ratios that
    /// are not halves.
    ///
    /// ```text
    /// ┌──────────────┬────────┐
    /// │              │  out   │
    /// │   viewport   ├────────┤
    /// │              │ props  │
    /// ├──────────────┴────────┤
    /// │          uv           │
    /// └───────────────────────┘
    /// ```
    fn nested() -> (Screen, [AreaId; 4]) {
        let mut screen = Screen::single(EditorKind::Viewport);
        let viewport = screen.areas()[0].0;
        let uv = screen
            .split(viewport, Axis::Horizontal, 0.7, EditorKind::UvEditor)
            .expect("split the only area");
        let outliner = screen
            .split(viewport, Axis::Vertical, 0.75, EditorKind::Outliner)
            .expect("split the viewport");
        let properties = screen
            .split(outliner, Axis::Horizontal, 0.4, EditorKind::Properties)
            .expect("split the outliner");
        (screen, [viewport, outliner, properties, uv])
    }

    #[test]
    fn a_screen_survives_being_flattened_and_rebuilt() {
        // The claim the file format rests on. Compared by *layout* rather than
        // by `==`, because the ids are deliberately not preserved: what has to
        // come back is the arrangement, not the handles.
        let (screen, _) = nested();
        let rebuilt = Screen::from_nodes(&screen.to_nodes()).expect("its own output");

        let before = screen.layout(space());
        let after = rebuilt.layout(space());
        assert_eq!(before.areas.len(), after.areas.len());
        for (a, b) in before.areas.iter().zip(&after.areas) {
            assert_eq!(a.editor, b.editor, "editor moved");
            assert_eq!(a.rect, b.rect, "area moved");
        }
        assert_eq!(before.handles.len(), after.handles.len());
        for (a, b) in before.handles.iter().zip(&after.handles) {
            assert_eq!(a.axis, b.axis);
            assert_eq!(a.rect, b.rect, "boundary moved");
        }
    }

    #[test]
    fn a_rebuilt_screen_can_still_be_edited() {
        // Fresh ids have to be usable and distinct. A rebuild that handed every
        // area the same id would round-trip perfectly and then break the first
        // time the user split anything.
        let (screen, _) = nested();
        let mut rebuilt = Screen::from_nodes(&screen.to_nodes()).expect("rebuilds");

        let mut ids: Vec<_> = rebuilt.areas().iter().map(|(id, _)| *id).collect();
        let count = ids.len();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), count, "an id was handed out twice");

        let victim = rebuilt.areas()[1].0;
        assert!(rebuilt
            .split(victim, Axis::Vertical, 0.5, EditorKind::Preferences)
            .is_some());
        assert_eq!(rebuilt.area_count(), count + 1);
        // And the new id collides with nothing, which is what `next` is for.
        let mut ids: Vec<_> = rebuilt.areas().iter().map(|(id, _)| *id).collect();
        let count = ids.len();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), count);
    }

    #[test]
    fn a_single_area_flattens_to_one_node() {
        let screen = Screen::single(EditorKind::UvEditor);
        assert_eq!(
            screen.to_nodes(),
            vec![NodeSpec::Area {
                editor: EditorKind::UvEditor
            }]
        );
    }

    #[test]
    fn a_broken_sequence_is_refused_rather_than_half_read() {
        // Every way a file can be wrong. A layout is one tree exactly; the
        // alternative to refusing is handing the user a screen they never
        // saved, which they then have to work out how to undo.
        assert!(Screen::from_nodes(&[]).is_none(), "empty");

        // A split with no children at all, and one with only the first.
        let split = NodeSpec::Split {
            axis: Axis::Vertical,
            ratio: 0.5,
        };
        let area = NodeSpec::Area {
            editor: EditorKind::Viewport,
        };
        assert!(Screen::from_nodes(&[split]).is_none(), "no children");
        assert!(Screen::from_nodes(&[split, area]).is_none(), "one child");

        // Two trees, one after the other.
        assert!(
            Screen::from_nodes(&[area, area]).is_none(),
            "trailing nodes"
        );
        assert!(
            Screen::from_nodes(&[split, area, area, area]).is_none(),
            "a whole extra area"
        );

        // And the well-formed one in the middle of all that really does work,
        // so this is not a test that everything is refused.
        assert!(Screen::from_nodes(&[split, area, area]).is_some());
    }

    #[test]
    fn no_ratio_can_invert_an_area() {
        // Found by this test. A ratio of exactly zero or one is *legal* —
        // `split` clamps into that range and so does the file format — and it
        // puts the boundary on an edge, half a handle past the child on that
        // side. `Rect::from_min_max` builds the inverted result without
        // complaint, and an inverted rect is not something egui reports: the
        // area simply is not there, with no indication why.
        //
        // The out-of-range values are here because the file is text a user or
        // another build can have written; the in-range ones are here because
        // they were the actual defect.
        for bad in [-1.0, 0.0, 1.0, 2.0, f32::INFINITY, f32::NAN] {
            let nodes = [
                NodeSpec::Split {
                    axis: Axis::Vertical,
                    ratio: bad,
                },
                NodeSpec::Area {
                    editor: EditorKind::Viewport,
                },
                NodeSpec::Area {
                    editor: EditorKind::Outliner,
                },
            ];
            let screen = Screen::from_nodes(&nodes).expect("well formed");
            let layout = screen.layout(space());
            assert_eq!(layout.areas.len(), 2, "ratio {bad} lost an area");
            for area in &layout.areas {
                assert!(
                    area.rect.width() >= 0.0 && area.rect.height() >= 0.0,
                    "ratio {bad} inverted {:?}",
                    area.rect
                );
                assert!(
                    space().expand(1.0).contains_rect(area.rect),
                    "ratio {bad} put {:?} outside the screen",
                    area.rect
                );
            }
            // And the boundary stays somewhere the pointer can reach it, or the
            // user has no way to undo whatever put it there.
            for handle in &layout.handles {
                assert!(
                    space().expand(1.0).contains_rect(handle.rect),
                    "ratio {bad} put the boundary at {:?}",
                    handle.rect
                );
            }
        }
    }

    #[test]
    fn every_axis_has_a_distinct_tag_and_survives_a_round_trip() {
        for axis in [Axis::Vertical, Axis::Horizontal] {
            assert_eq!(Axis::from_tag(axis.tag()), Some(axis));
        }
        assert_ne!(Axis::Vertical.tag(), Axis::Horizontal.tag());
        assert_eq!(Axis::from_tag("nonesuch"), None);
    }

    #[test]
    fn one_area_fills_the_space_exactly() {
        let screen = Screen::single(EditorKind::Viewport);
        let layout = screen.layout(space());
        assert_eq!(layout.areas.len(), 1);
        assert_eq!(layout.areas[0].rect, space());
        assert!(layout.handles.is_empty(), "nothing to drag");
    }

    #[test]
    fn a_vertical_split_divides_the_width_at_the_ratio() {
        // Which axis is which, checked against the arithmetic rather than
        // against a symmetric picture. `Vertical` names the boundary, so the
        // *width* is what gets divided.
        let mut screen = Screen::single(EditorKind::Viewport);
        let left = screen.areas()[0].0;
        screen
            .split(left, Axis::Vertical, 0.25, EditorKind::Outliner)
            .expect("split");

        let layout = screen.layout(space());
        let first = layout.areas.iter().find(|a| a.id == left).expect("first");
        let second = layout
            .areas
            .iter()
            .find(|a| a.id != left)
            .expect("the new one");

        assert!((first.rect.width() - (300.0 - HANDLE * 0.5)).abs() < 1e-3);
        assert!((second.rect.width() - (900.0 - HANDLE * 0.5)).abs() < 1e-3);
        // The height is untouched by a vertical split, which is the half an
        // axis mix-up would get wrong.
        assert_eq!(first.rect.height(), space().height());
        assert_eq!(second.rect.height(), space().height());
        assert!(first.rect.max.x < second.rect.min.x, "left of, not above");
    }

    #[test]
    fn a_horizontal_split_divides_the_height() {
        let mut screen = Screen::single(EditorKind::Viewport);
        let top = screen.areas()[0].0;
        screen
            .split(top, Axis::Horizontal, 0.25, EditorKind::UvEditor)
            .expect("split");

        let layout = screen.layout(space());
        let first = layout.areas.iter().find(|a| a.id == top).expect("first");
        let second = layout.areas.iter().find(|a| a.id != top).expect("second");

        assert!((first.rect.height() - (200.0 - HANDLE * 0.5)).abs() < 1e-3);
        assert_eq!(first.rect.width(), space().width());
        assert!(first.rect.max.y < second.rect.min.y, "above, not left of");
    }

    #[test]
    fn areas_never_overlap_and_leave_only_the_handles_uncovered() {
        // The property that makes the layout usable at all, over a tree deep
        // enough that a mistake in the recursion has somewhere to hide.
        let (screen, _) = nested();
        let layout = screen.layout(space());
        assert_eq!(layout.areas.len(), 4);
        assert_eq!(layout.handles.len(), 3);

        for (i, a) in layout.areas.iter().enumerate() {
            assert!(a.rect.width() > 0.0 && a.rect.height() > 0.0, "{a:?}");
            assert!(space().contains_rect(a.rect), "{a:?} escaped the screen");
            for b in &layout.areas[i + 1..] {
                let overlap = a.rect.intersect(b.rect);
                assert!(
                    overlap.width() <= 0.0 || overlap.height() <= 0.0,
                    "{:?} overlaps {:?}",
                    a.id,
                    b.id
                );
            }
        }

        // What is left over is the handles and nothing else. Checked as area,
        // because a recursion that dropped a subtree would still pass every
        // overlap test above.
        let covered: f32 = layout
            .areas
            .iter()
            .map(|a| a.rect.width() * a.rect.height())
            .sum();
        let total = space().width() * space().height();
        let handles: f32 = layout
            .handles
            .iter()
            .map(|h| h.rect.width() * h.rect.height())
            .sum();
        // The handles overlap each other where they cross, so the gap is at most
        // their total area and at least most of it.
        let gap = total - covered;
        assert!(
            gap > handles * 0.5 && gap < handles * 1.5,
            "uncovered {gap} against {handles} of handle"
        );
    }

    #[test]
    fn an_area_keeps_its_id_when_something_else_is_split() {
        // The whole reason ids exist. A UV editor's zoom is keyed on its area,
        // and splitting the viewport must not move it to another editor or lose
        // it.
        let (mut screen, [viewport, outliner, properties, uv]) = nested();
        let before: Vec<_> = screen.areas();

        screen
            .split(viewport, Axis::Vertical, 0.5, EditorKind::Preferences)
            .expect("split");

        for id in [outliner, properties, uv] {
            assert_eq!(
                screen.editor(id),
                before
                    .iter()
                    .find(|(b, _)| *b == id)
                    .map(|(_, editor)| *editor),
                "area {id:?} changed editor"
            );
        }
        assert_eq!(screen.area_count(), 5);
    }

    #[test]
    fn closing_an_area_gives_its_space_to_its_sibling() {
        // Exactly its sibling's, not redistributed: the sibling has to end up
        // occupying the rect the pair occupied together, or closing an area
        // shuffles the whole screen.
        let (mut screen, [_, outliner, properties, _]) = nested();
        let pair = {
            let layout = screen.layout(space());
            let a = layout
                .areas
                .iter()
                .find(|a| a.id == outliner)
                .expect("outliner")
                .rect;
            let b = layout
                .areas
                .iter()
                .find(|a| a.id == properties)
                .expect("properties")
                .rect;
            a.union(b)
        };

        assert!(screen.close(properties));
        assert_eq!(screen.area_count(), 3);
        assert!(screen.editor(properties).is_none());

        let after = screen.layout(space());
        let grown = after
            .areas
            .iter()
            .find(|a| a.id == outliner)
            .expect("outliner survived")
            .rect;
        assert!(
            (grown.min.x - pair.min.x).abs() < 1e-3
                && (grown.min.y - pair.min.y).abs() < 1e-3
                && (grown.max.x - pair.max.x).abs() < 1e-3
                && (grown.max.y - pair.max.y).abs() < 1e-3,
            "sibling got {grown:?}, the pair had {pair:?}"
        );
        assert_eq!(after.handles.len(), 2, "the split went with it");
    }

    #[test]
    fn closing_a_subtree_s_neighbour_keeps_the_subtree_whole() {
        // The case a shallow tree cannot show: the survivor is a *split*, not a
        // leaf, and moving it must carry its children and their ids.
        let (mut screen, [viewport, outliner, properties, uv]) = nested();
        assert!(screen.close(uv));
        assert_eq!(screen.area_count(), 3);
        for id in [viewport, outliner, properties] {
            assert!(screen.editor(id).is_some(), "{id:?} was lost");
        }
        let layout = screen.layout(space());
        assert_eq!(layout.areas.len(), 3);
        // And the survivors now fill the whole screen.
        let covered: f32 = layout
            .areas
            .iter()
            .map(|a| a.rect.width() * a.rect.height())
            .sum();
        let total = space().width() * space().height();
        assert!(covered > total * 0.98, "{covered} of {total}");
    }

    #[test]
    fn the_last_area_cannot_be_closed() {
        // A screen with no areas shows nothing and offers no way back.
        let mut screen = Screen::single(EditorKind::Viewport);
        let only = screen.areas()[0].0;
        assert!(!screen.close(only));
        assert_eq!(screen.area_count(), 1);
    }

    #[test]
    fn closing_an_area_that_is_not_there_changes_nothing() {
        let (mut screen, _) = nested();
        let before = screen.clone();
        assert!(!screen.close(AreaId(999)));
        assert_eq!(screen, before);
    }

    #[test]
    fn changing_an_area_s_editor_changes_only_that_one() {
        let (mut screen, [viewport, outliner, ..]) = nested();
        assert!(screen.set_editor(viewport, EditorKind::Preferences));
        assert_eq!(screen.editor(viewport), Some(EditorKind::Preferences));
        assert_eq!(screen.editor(outliner), Some(EditorKind::Outliner));
        assert!(!screen.set_editor(AreaId(999), EditorKind::Viewport));
    }

    #[test]
    fn a_boundary_drag_cannot_squeeze_an_area_to_nothing() {
        // Without the clamp a boundary dragged to the edge leaves an area a few
        // pixels wide, which the user then cannot grab to undo.
        let mut screen = Screen::single(EditorKind::Viewport);
        let left = screen.areas()[0].0;
        screen
            .split(left, Axis::Vertical, 0.5, EditorKind::Outliner)
            .expect("split");
        let split = screen.layout(space()).handles[0].id;

        assert!(screen.resize(split, 0.0, 1200.0));
        let layout = screen.layout(space());
        for area in &layout.areas {
            assert!(
                area.rect.width() >= MIN_AREA - HANDLE,
                "{:?} is {} wide",
                area.id,
                area.rect.width()
            );
        }

        assert!(screen.resize(split, 1.0, 1200.0));
        for area in &screen.layout(space()).areas {
            assert!(area.rect.width() >= MIN_AREA - HANDLE);
        }
    }

    #[test]
    fn a_space_too_small_for_two_areas_still_lays_out() {
        // A window dragged narrower than two minimum areas. Refusing to lay out
        // would be worse than laying out small, and the clamp has no valid
        // answer to give.
        let mut screen = Screen::single(EditorKind::Viewport);
        let left = screen.areas()[0].0;
        screen
            .split(left, Axis::Vertical, 0.5, EditorKind::Outliner)
            .expect("split");
        let split = screen.layout(space()).handles[0].id;
        assert!(screen.resize(split, 0.9, 60.0));

        let tiny = Rect::from_min_size(Pos2::ZERO, Vec2::new(60.0, 400.0));
        let layout = screen.layout(tiny);
        assert_eq!(layout.areas.len(), 2);
        for area in &layout.areas {
            assert!(area.rect.width() > 0.0, "{area:?}");
        }
    }

    #[test]
    fn resizing_a_boundary_that_is_not_there_changes_nothing() {
        let (mut screen, _) = nested();
        let before = screen.clone();
        assert!(!screen.resize(SplitId(999), 0.5, 1000.0));
        assert_eq!(screen, before);
    }

    #[test]
    fn a_nested_boundary_reports_the_space_it_divides_not_the_screen() {
        // What a drag needs in order to follow the pointer. A nested split's
        // ratio is a share of its parent, so a caller measuring against the
        // window would divide by the wrong number — and on a boundary two
        // levels down the boundary would move at a fraction of the pointer's
        // speed. The handle's own rect cannot supply it: it spans the parent in
        // one axis and is four points wide in the other.
        let (screen, [_, outliner, ..]) = nested();
        let layout = screen.layout(space());
        let column = layout
            .areas
            .iter()
            .find(|a| a.id == outliner)
            .expect("outliner")
            .rect;

        // The split between the outliner and the properties below it. Its
        // parent is the right-hand column, which is much narrower than the
        // screen — which is the whole point.
        let nested_handle = layout
            .handles
            .iter()
            .find(|h| h.axis == Axis::Horizontal && h.rect.width() < space().width() * 0.5)
            .expect("the split inside the column");
        assert!(
            (nested_handle.parent.width() - column.width()).abs() < HANDLE + 1e-3,
            "parent is {} wide, the column is {}",
            nested_handle.parent.width(),
            column.width()
        );
        assert!(
            nested_handle.parent.width() < space().width() * 0.5,
            "the parent is the screen, so a drag would use the wrong denominator"
        );
        assert!(space().contains_rect(nested_handle.parent));

        // And the outermost boundary really does divide the whole screen, so
        // this test is not simply asserting that every parent is small.
        let outer = layout
            .handles
            .iter()
            .find(|h| h.rect.width() >= space().width() - 1e-3)
            .expect("the outermost split");
        assert_eq!(outer.parent, space());
    }

    #[test]
    fn a_handle_sits_between_the_areas_it_divides() {
        // Where the pointer has to be for the drag to make sense. A handle
        // placed by the parent's centre rather than the split line would still
        // be inside the screen and would move the wrong boundary.
        let (screen, _) = nested();
        let layout = screen.layout(space());
        for handle in &layout.handles {
            assert!(space().contains_rect(handle.rect), "{handle:?}");
            let touching: Vec<&AreaRect> = layout
                .areas
                .iter()
                .filter(|a| {
                    let gap = a.rect.intersect(handle.rect.expand(1.0));
                    gap.width() > 0.0 && gap.height() > 0.0
                })
                .collect();
            assert!(
                touching.len() >= 2,
                "{:?} divides {} area(s)",
                handle.id,
                touching.len()
            );
        }
    }

    #[test]
    fn splitting_an_area_that_is_not_there_changes_the_tree_not_at_all() {
        let (mut screen, _) = nested();
        let before = screen.areas();
        assert!(screen
            .split(AreaId(999), Axis::Vertical, 0.5, EditorKind::Viewport)
            .is_none());
        assert_eq!(screen.areas(), before);
    }

    #[test]
    fn every_editor_kind_has_a_distinct_tag_and_survives_a_round_trip() {
        // The tags are the file format. Serialising the enum's discriminant
        // would make a saved layout depend on declaration order, so inserting an
        // editor in the middle would reinterpret everybody's workspaces.
        let mut seen: Vec<&str> = Vec::new();
        for kind in EditorKind::ALL {
            let tag = kind.tag();
            assert!(!seen.contains(&tag), "duplicate tag {tag}");
            seen.push(tag);
            assert_eq!(EditorKind::from_tag(tag), Some(kind));
            assert!(
                kind.name_key().starts_with("editor-"),
                "{tag} has key {}",
                kind.name_key()
            );
        }
        assert_eq!(EditorKind::from_tag("nonesuch"), None);
    }
}
