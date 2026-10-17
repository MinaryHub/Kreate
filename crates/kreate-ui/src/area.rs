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

/// Which editor an area is showing.
///
/// A string and not an enum, and that is the whole of §2.6's registry
/// decision. An enum is a closed set: nothing outside this crate can add a
/// variant, so a script cannot register an editor (Phase 9) and every new one
/// costs a variant plus an arm in each of several matches. Four new pillars
/// want nine more editor types.
///
/// The id *is* the serialised form, which removes a whole class of mistake as
/// well. The enum needed a separate `tag`, because writing a discriminant would
/// have made a saved layout depend on the order the variants happened to be
/// declared in — insert an editor in the middle and everyone's workspaces
/// silently reinterpret. With the id as the name there is nothing to keep in
/// step.
///
/// # An id this build does not know is kept, not replaced
///
/// A layout from a later version, or one saved while an add-on was installed,
/// names editors that are not here. Such an area keeps its id and is drawn as
/// missing. What it must not do is quietly become a viewport: the user's
/// carefully arranged layout would come back subtly wrong, they would fix it,
/// and installing the add-on again would not bring it back.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EditorId(std::borrow::Cow<'static, str>);

impl EditorId {
    pub const VIEWPORT: Self = Self::known("viewport");
    pub const OUTLINER: Self = Self::known("outliner");
    pub const PROPERTIES: Self = Self::known("properties");
    pub const UV_EDITOR: Self = Self::known("uv");
    pub const PREFERENCES: Self = Self::known("preferences");
    /// The history as a script, and a line to type one. Phase 9's first editor.
    pub const CONSOLE: Self = Self::known("console");
    /// A conversation with Claude, who edits the document through its operators.
    pub const CLAUDE: Self = Self::known("claude");
    /// A material's node graph.
    pub const SHADER: Self = Self::known("shader");
    /// The active object's geometry node graph.
    pub const GEOMETRY: Self = Self::known("geometry_nodes");
    /// The scene's clock and a ruler to move it along.
    pub const TIMELINE: Self = Self::known("timeline");
    /// Every key of the selected objects, to select and retime.
    pub const DOPE_SHEET: Self = Self::known("dope_sheet");
    /// Each curve as value over time, with its keys and handles.
    pub const GRAPH: Self = Self::known("graph");
    /// The drivers of the selected objects: expressions and their variables.
    pub const DRIVERS: Self = Self::known("drivers");
    /// Each object's action strips, laid over time.
    pub const NLA: Self = Self::known("nla");
    /// Models, materials and environments to add, local or online.
    pub const ASSETS: Self = Self::known("assets");

    /// An id spelled out in this build.
    pub const fn known(id: &'static str) -> Self {
        Self(std::borrow::Cow::Borrowed(id))
    }

    /// An id read from a file, which may name nothing this build has.
    pub fn parse(id: &str) -> Self {
        Self(std::borrow::Cow::Owned(id.to_string()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for EditorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// What the shell needs to know about an editor without drawing it.
///
/// The drawing itself is still a match in the shell, and deliberately so for
/// now: moving it behind a trait means a context object carrying everything any
/// editor could want, which is the same pile of arguments in a different shape
/// until there is a second family of editors to make the abstraction pay. What
/// could not wait is the *registration*, because the closed enum is what makes
/// an add-on's editor impossible and makes an unknown id unrepresentable.
#[derive(Clone, Debug, PartialEq)]
pub struct EditorInfo {
    pub id: EditorId,
    /// The catalogue key for this editor's name.
    ///
    /// A key rather than a string: the editor-type dropdown is UI prose and
    /// goes through Fluent like the rest of it.
    pub name_key: &'static str,
    /// Whether the body reaches the edges of the area.
    ///
    /// True for the viewport, whose rendered image is meant to touch the edges;
    /// everything else takes the panel inset. A property of the editor rather
    /// than a special case in the shell, which is what it used to be.
    pub full_bleed: bool,
}

/// Every editor this build can show, in the order a menu offers them.
#[derive(Clone, Debug, Default)]
pub struct EditorRegistry {
    entries: Vec<EditorInfo>,
}

impl EditorRegistry {
    /// The editors this build ships.
    pub fn with_builtins() -> Self {
        let mut registry = Self::default();
        for (id, name_key, full_bleed) in [
            (EditorId::VIEWPORT, "editor-viewport", true),
            (EditorId::OUTLINER, "editor-outliner", false),
            (EditorId::PROPERTIES, "editor-properties", false),
            (EditorId::UV_EDITOR, "editor-uv", false),
            (EditorId::PREFERENCES, "editor-preferences", false),
            (EditorId::CONSOLE, "editor-console", false),
            (EditorId::CLAUDE, "editor-claude", false),
            (EditorId::ASSETS, "editor-assets", false),
            (EditorId::SHADER, "editor-shader", true),
            (EditorId::GEOMETRY, "editor-geometry-nodes", true),
            (EditorId::TIMELINE, "editor-timeline", true),
            (EditorId::DOPE_SHEET, "editor-dope-sheet", true),
            (EditorId::GRAPH, "editor-graph", true),
            (EditorId::DRIVERS, "editor-drivers", true),
            (EditorId::NLA, "editor-nla", true),
        ] {
            registry.register(EditorInfo {
                id,
                name_key,
                full_bleed,
            });
        }
        registry
    }

    /// Adds an editor, or replaces one already registered under the same id.
    ///
    /// Replacing rather than refusing, so that an add-on can override a
    /// built-in — and so that registering twice is not a hard error in a
    /// reload, which is the commonest thing a script does.
    pub fn register(&mut self, info: EditorInfo) {
        match self.entries.iter_mut().find(|e| e.id == info.id) {
            Some(slot) => *slot = info,
            None => self.entries.push(info),
        }
    }

    pub fn get(&self, id: &EditorId) -> Option<&EditorInfo> {
        self.entries.iter().find(|e| &e.id == id)
    }

    pub fn is_known(&self, id: &EditorId) -> bool {
        self.get(id).is_some()
    }

    /// Every registered editor, in registration order.
    pub fn iter(&self) -> impl Iterator<Item = &EditorInfo> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Whether an area's body reaches the edges.
    ///
    /// An id this build does not know takes the inset, because what is drawn
    /// there is a message and a message wants a margin.
    pub fn full_bleed(&self, id: &EditorId) -> bool {
        self.get(id).is_some_and(|e| e.full_bleed)
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
#[derive(Clone, Debug, PartialEq)]
pub enum NodeSpec {
    Split { axis: Axis, ratio: f32 },
    Area { editor: EditorId },
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
        editor: EditorId,
    },
}

/// Which side of an area another is docked on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
    Top,
    Bottom,
}

/// One workspace's layout.
#[derive(Clone, Debug, PartialEq)]
pub struct Screen {
    root: Node,
    /// One counter for both kinds of id, so an area and a split can never be
    /// confused for one another in a saved file or a log line.
    next: u32,
    /// Areas taken out of the tree to float in windows of their own. Part of
    /// the layout, so they are saved with it and come back on the next start
    /// — a panel left floating at quit was otherwise simply gone.
    floating: Vec<Floating>,
}

/// An area floating in a window of its own.
#[derive(Clone, Debug, PartialEq)]
pub struct Floating {
    /// Its id, kept while it floats so it docks back as itself.
    pub id: AreaId,
    pub editor: EditorId,
    /// Where its window is.
    pub rect: Rect,
}

/// Where an area ended up.
#[derive(Clone, Debug, PartialEq)]
pub struct AreaRect {
    pub id: AreaId,
    pub editor: EditorId,
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
    pub fn single(editor: EditorId) -> Self {
        Self {
            root: Node::Leaf {
                id: AreaId(0),
                editor,
            },
            next: 1,
            floating: Vec::new(),
        }
    }

    /// The floating areas.
    pub fn floating(&self) -> &[Floating] {
        &self.floating
    }

    /// Takes `area` out of the tree to float at `rect`. `false` when it is
    /// not in the tree or is the last area.
    pub fn float(&mut self, area: AreaId, rect: Rect) -> bool {
        let Some(editor) = self.editor(area) else {
            return false;
        };
        if !self.close(area) {
            return false;
        }
        self.floating.push(Floating {
            id: area,
            editor,
            rect,
        });
        true
    }

    /// A new floating area showing `editor`: how a saved one comes back.
    pub fn add_floating(&mut self, editor: EditorId, rect: Rect) -> AreaId {
        let id = AreaId(self.take_id());
        self.floating.push(Floating { id, editor, rect });
        id
    }

    /// Docks a floating area beside `beside`, on `side`. `false`, and it
    /// still floats, when it is not floating or `beside` is not there.
    pub fn dock_floating(&mut self, area: AreaId, beside: AreaId, side: Side, share: f32) -> bool {
        let Some(at) = self.floating.iter().position(|f| f.id == area) else {
            return false;
        };
        let editor = self.floating[at].editor.clone();
        if !self.dock(beside, side, share, area, editor) {
            return false;
        }
        self.floating.remove(at);
        true
    }

    /// Moves a floating area's window.
    pub fn set_floating_rect(&mut self, area: AreaId, rect: Rect) -> bool {
        match self.floating.iter_mut().find(|f| f.id == area) {
            Some(f) if f.rect != rect => {
                f.rect = rect;
                true
            }
            _ => false,
        }
    }

    /// Closes a floating area, as its window's close button does.
    pub fn remove_floating(&mut self, area: AreaId) -> bool {
        let before = self.floating.len();
        self.floating.retain(|f| f.id != area);
        self.floating.len() != before
    }

    fn take_id(&mut self) -> u32 {
        let id = self.next;
        self.next += 1;
        id
    }

    /// Every area, in tree order.
    pub fn areas(&self) -> Vec<(AreaId, EditorId)> {
        let mut out = Vec::new();
        collect(&self.root, &mut out);
        out
    }

    pub fn area_count(&self) -> usize {
        self.areas().len()
    }

    /// What `area` is showing.
    pub fn editor(&self, area: AreaId) -> Option<EditorId> {
        self.areas()
            .into_iter()
            .find(|(id, _)| *id == area)
            .map(|(_, editor)| editor)
    }

    /// Points `area` at another editor. `false` if there is no such area.
    pub fn set_editor(&mut self, area: AreaId, kind: EditorId) -> bool {
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
        editor: EditorId,
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

    /// Puts an area back into the screen beside `beside`, on `side`, taking
    /// `share` of the space they then divide — keeping the id it had, so what
    /// was kept for it while it floated (a properties tab, a scroll) is still
    /// its own.
    ///
    /// `false` when `beside` is not on the screen or `id` already is.
    pub fn dock(
        &mut self,
        beside: AreaId,
        side: Side,
        share: f32,
        id: AreaId,
        editor: EditorId,
    ) -> bool {
        if self.areas().iter().any(|(a, _)| *a == id) {
            return false;
        }
        // An id from before is below the counter already; one that is not
        // moves it on, so nothing made later can take it.
        self.next = self.next.max(id.0 + 1);
        let split_id = SplitId(self.take_id());
        let share = sane_ratio(share);
        let Some(node) = find_node_mut(&mut self.root, beside) else {
            return false;
        };
        let leaf = Node::Leaf { id, editor };
        let existing = std::mem::replace(node, leaf.clone());
        let (axis, before) = match side {
            Side::Left => (Axis::Vertical, true),
            Side::Right => (Axis::Vertical, false),
            Side::Top => (Axis::Horizontal, true),
            Side::Bottom => (Axis::Horizontal, false),
        };
        *node = if before {
            Node::Split {
                id: split_id,
                axis,
                ratio: share,
                first: Box::new(leaf),
                second: Box::new(existing),
            }
        } else {
            Node::Split {
                id: split_id,
                axis,
                ratio: sane_ratio(1.0 - share),
                first: Box::new(existing),
                second: Box::new(leaf),
            }
        };
        true
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
        rest.is_empty().then_some(Self {
            root,
            next,
            floating: Vec::new(),
        })
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
        Node::Leaf { editor, .. } => out.push(NodeSpec::Area {
            editor: editor.clone(),
        }),
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
    match head {
        NodeSpec::Area { editor } => {
            let id = AreaId(*next);
            *next += 1;
            Some(Node::Leaf {
                id,
                editor: editor.clone(),
            })
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
                axis: *axis,
                ratio: sane_ratio(*ratio),
                first,
                second,
            })
        }
    }
}

fn collect(node: &Node, out: &mut Vec<(AreaId, EditorId)>) {
    match node {
        Node::Leaf { id, editor } => out.push((*id, editor.clone())),
        Node::Split { first, second, .. } => {
            collect(first, out);
            collect(second, out);
        }
    }
}

fn find_leaf_mut(node: &mut Node, area: AreaId) -> Option<&mut EditorId> {
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
                editor: EditorId::VIEWPORT,
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
            editor: editor.clone(),
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
        let mut screen = Screen::single(EditorId::VIEWPORT);
        let viewport = screen.areas()[0].0;
        let uv = screen
            .split(viewport, Axis::Horizontal, 0.7, EditorId::UV_EDITOR)
            .expect("split the only area");
        let outliner = screen
            .split(viewport, Axis::Vertical, 0.75, EditorId::OUTLINER)
            .expect("split the viewport");
        let properties = screen
            .split(outliner, Axis::Horizontal, 0.4, EditorId::PROPERTIES)
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
            .split(victim, Axis::Vertical, 0.5, EditorId::PREFERENCES)
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
        let screen = Screen::single(EditorId::UV_EDITOR);
        assert_eq!(
            screen.to_nodes(),
            vec![NodeSpec::Area {
                editor: EditorId::UV_EDITOR
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
            editor: EditorId::VIEWPORT,
        };
        assert!(
            Screen::from_nodes(std::slice::from_ref(&split)).is_none(),
            "no children"
        );
        assert!(
            Screen::from_nodes(&[split.clone(), area.clone()]).is_none(),
            "one child"
        );

        // Two trees, one after the other.
        assert!(
            Screen::from_nodes(&[area.clone(), area.clone()]).is_none(),
            "trailing nodes"
        );
        assert!(
            Screen::from_nodes(&[split.clone(), area.clone(), area.clone(), area.clone()])
                .is_none(),
            "a whole extra area"
        );

        // And the well-formed one in the middle of all that really does work,
        // so this is not a test that everything is refused.
        assert!(Screen::from_nodes(&[split.clone(), area.clone(), area.clone()]).is_some());
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
                    editor: EditorId::VIEWPORT,
                },
                NodeSpec::Area {
                    editor: EditorId::OUTLINER,
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
        let screen = Screen::single(EditorId::VIEWPORT);
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
        let mut screen = Screen::single(EditorId::VIEWPORT);
        let left = screen.areas()[0].0;
        screen
            .split(left, Axis::Vertical, 0.25, EditorId::OUTLINER)
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
        let mut screen = Screen::single(EditorId::VIEWPORT);
        let top = screen.areas()[0].0;
        screen
            .split(top, Axis::Horizontal, 0.25, EditorId::UV_EDITOR)
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
            .split(viewport, Axis::Vertical, 0.5, EditorId::PREFERENCES)
            .expect("split");

        for id in [outliner, properties, uv] {
            assert_eq!(
                screen.editor(id),
                before
                    .iter()
                    .find(|(b, _)| *b == id)
                    .map(|(_, editor)| editor.clone()),
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
        let mut screen = Screen::single(EditorId::VIEWPORT);
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
        assert!(screen.set_editor(viewport, EditorId::PREFERENCES));
        assert_eq!(screen.editor(viewport), Some(EditorId::PREFERENCES));
        assert_eq!(screen.editor(outliner), Some(EditorId::OUTLINER));
        assert!(!screen.set_editor(AreaId(999), EditorId::VIEWPORT));
    }

    #[test]
    fn a_boundary_drag_cannot_squeeze_an_area_to_nothing() {
        // Without the clamp a boundary dragged to the edge leaves an area a few
        // pixels wide, which the user then cannot grab to undo.
        let mut screen = Screen::single(EditorId::VIEWPORT);
        let left = screen.areas()[0].0;
        screen
            .split(left, Axis::Vertical, 0.5, EditorId::OUTLINER)
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
        let mut screen = Screen::single(EditorId::VIEWPORT);
        let left = screen.areas()[0].0;
        screen
            .split(left, Axis::Vertical, 0.5, EditorId::OUTLINER)
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
            .split(AreaId(999), Axis::Vertical, 0.5, EditorId::VIEWPORT)
            .is_none());
        assert_eq!(screen.areas(), before);
    }

    #[test]
    fn every_registered_editor_has_a_distinct_id_and_a_translation_key() {
        // The id *is* the file format. It used to be a separate `tag`, because
        // writing the enum's discriminant would have made a saved layout depend
        // on declaration order — insert an editor in the middle and everybody's
        // workspaces reinterpret. With the id as the name there is nothing to
        // keep in step, which is one whole class of mistake gone.
        let registry = EditorRegistry::with_builtins();
        assert!(registry.len() >= 5, "the builtins are not registered");

        let mut seen: Vec<&str> = Vec::new();
        for info in registry.iter() {
            let id = info.id.as_str();
            assert!(!seen.contains(&id), "duplicate id {id}");
            seen.push(id);
            assert!(
                info.name_key.starts_with("editor-"),
                "{id} has key {}",
                info.name_key
            );
            assert_eq!(
                registry.get(&info.id).map(|e| e.id.clone()),
                Some(info.id.clone())
            );
        }
    }

    #[test]
    fn an_editor_can_be_registered_without_this_crate_knowing_about_it() {
        // §2.6's decision, and the thing a closed enum made impossible. Four
        // new pillars want nine more editor types and Phase 9 wants a script to
        // be able to add one; neither can happen if the set is a `match` in
        // here.
        let mut registry = EditorRegistry::with_builtins();
        let before = registry.len();
        let mine = EditorId::parse("com.example.graph");
        assert!(!registry.is_known(&mine));

        registry.register(EditorInfo {
            id: mine.clone(),
            name_key: "editor-example-graph",
            full_bleed: false,
        });
        assert_eq!(registry.len(), before + 1);
        assert!(registry.is_known(&mine));
        // And it appears where a menu would look for it.
        assert!(registry.iter().any(|e| e.id == mine));
    }

    #[test]
    fn registering_twice_replaces_rather_than_duplicating() {
        // A script reloading is the commonest thing that happens, and it must
        // not leave two entries with the same id — which would put the editor
        // in the menu twice and make "which one is it" undecidable.
        let mut registry = EditorRegistry::with_builtins();
        let before = registry.len();
        registry.register(EditorInfo {
            id: EditorId::VIEWPORT,
            name_key: "editor-viewport-replaced",
            full_bleed: false,
        });
        assert_eq!(registry.len(), before, "a duplicate entry was added");
        assert_eq!(
            registry.get(&EditorId::VIEWPORT).map(|e| e.name_key),
            Some("editor-viewport-replaced")
        );
        assert!(!registry.full_bleed(&EditorId::VIEWPORT));
    }

    #[test]
    fn an_id_this_build_does_not_know_is_a_representable_thing() {
        // The other half of why the enum had to go. An area naming an editor
        // that is not here — a layout from a later version, or one saved while
        // an add-on was installed — has to be *expressible*, so that it can be
        // kept and shown as missing instead of quietly becoming a viewport.
        let registry = EditorRegistry::with_builtins();
        let stranger = EditorId::parse("com.example.compositor");
        assert!(!registry.is_known(&stranger));
        assert_eq!(registry.get(&stranger), None);
        // And it takes the panel inset, because what is drawn there is a
        // message and a message wants a margin.
        assert!(!registry.full_bleed(&stranger));

        // A screen can hold one, which is the point: the tree does not care
        // whether the editor exists.
        let screen = Screen::single(stranger.clone());
        assert_eq!(screen.areas().len(), 1);
        assert_eq!(screen.areas()[0].1, stranger);
    }

    #[test]
    fn a_known_id_and_a_parsed_one_are_the_same_value() {
        // So that a layout read from a file compares equal to one built in
        // code. Two representations of the same id that did not compare equal
        // would make every loaded workspace look like it named unknown
        // editors.
        assert_eq!(EditorId::parse("viewport"), EditorId::VIEWPORT);
        assert_eq!(EditorId::VIEWPORT.as_str(), "viewport");
        assert!(EditorRegistry::with_builtins().is_known(&EditorId::parse("viewport")));
    }

    #[test]
    fn the_viewport_is_the_only_builtin_that_reaches_the_edges() {
        // Which used to be a comparison against one id in the shell. As a
        // property of the registered editor, an editor that wants the same
        // treatment says so where it is registered.
        let registry = EditorRegistry::with_builtins();
        assert!(registry.full_bleed(&EditorId::VIEWPORT));
        for id in [
            EditorId::OUTLINER,
            EditorId::PROPERTIES,
            EditorId::UV_EDITOR,
            EditorId::PREFERENCES,
        ] {
            assert!(!registry.full_bleed(&id), "{id} claims the whole area");
        }
    }

    /// Undocked and docked again: the area comes back with its own id, on
    /// the side asked, taking the share asked.
    #[test]
    fn an_area_taken_out_docks_back_with_its_id_where_it_is_put() {
        let mut screen = Screen::single(EditorId::VIEWPORT);
        let viewport = screen.areas()[0].0;
        let outliner = screen
            .split(viewport, Axis::Vertical, 0.75, EditorId::OUTLINER)
            .unwrap();
        // Taken out to float.
        assert!(screen.close(outliner));
        assert_eq!(screen.area_count(), 1);
        // Docked on the viewport's left, a quarter of the width.
        assert!(screen.dock(viewport, Side::Left, 0.25, outliner, EditorId::OUTLINER));
        let space = Rect::from_min_size(Pos2::ZERO, Vec2::new(1000.0, 500.0));
        let layout = screen.layout(space);
        let rect = |id: AreaId| layout.areas.iter().find(|a| a.id == id).unwrap().rect;
        assert!(
            rect(outliner).center().x < rect(viewport).center().x,
            "on the left"
        );
        assert!(
            (rect(outliner).width() - 250.0).abs() < 10.0,
            "{:?}",
            rect(outliner)
        );
        // Not twice, and not beside an area that is not there.
        assert!(!screen.dock(viewport, Side::Right, 0.25, outliner, EditorId::OUTLINER));
        assert!(!screen.dock(
            AreaId(999),
            Side::Right,
            0.25,
            AreaId(1000),
            EditorId::CONSOLE
        ));
        // Below: a horizontal split, the new area second.
        let console = AreaId(1000);
        assert!(screen.dock(viewport, Side::Bottom, 0.3, console, EditorId::CONSOLE));
        let layout = screen.layout(space);
        let rect = |id: AreaId| layout.areas.iter().find(|a| a.id == id).unwrap().rect;
        assert!(
            rect(console).center().y > rect(viewport).center().y,
            "below"
        );
        // What it makes next does not take an id handed back.
        let next = screen
            .split(viewport, Axis::Vertical, 0.5, EditorId::PROPERTIES)
            .unwrap();
        assert!(next.0 > console.0);
    }
}
