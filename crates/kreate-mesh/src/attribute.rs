//! Named, typed attribute layers over the four mesh domains.
//!
//! Everything that is not topology lives here: positions, UVs, creases, material
//! indices, vertex groups, selection state, and any custom layer a plugin adds.
//! There are deliberately no hardcoded attribute fields on `MeshData` other than
//! topology — retrofitting a generic attribute system onto hardcoded fields is
//! the single most painful refactor in this domain, and every importer's
//! semantics depend on the domain model existing first.
//!
//! Each layer's array sits behind its own [`Arc`], which is what makes
//! structural-sharing undo cheap: moving vertices copies the position array and
//! shares everything else. Mutation therefore goes through `make_mut_*`, which
//! performs the copy-on-write.

use std::sync::Arc;

use glam::{Vec2, Vec3, Vec4};

/// Which mesh element a layer's values are indexed by.
///
/// These map 1:1 onto USD's `vertex` / `faceVarying` / `uniform` primvar
/// interpolations, which is why glTF split vertices, FBX `ByPolygonVertex`
/// layers, and OBJ's separate `v`/`vt`/`vn` index streams all reduce to the same
/// problem: put it in the Corner domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Domain {
    /// One value per vertex.
    Point,
    /// One value per edge.
    Edge,
    /// One value per face-corner (a face's use of a vertex). USD `faceVarying`.
    Corner,
    /// One value per face. USD `uniform`.
    Face,
}

impl Domain {
    pub const ALL: [Domain; 4] = [Domain::Point, Domain::Edge, Domain::Corner, Domain::Face];

    pub fn name(self) -> &'static str {
        match self {
            Domain::Point => "point",
            Domain::Edge => "edge",
            Domain::Corner => "corner",
            Domain::Face => "face",
        }
    }
}

/// The value type of a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttrType {
    F32,
    F32x2,
    F32x3,
    F32x4,
    I32,
    Bool,
    /// Packed 8-bit RGBA, for vertex colours that do not need float precision.
    U8x4,
}

impl AttrType {
    /// Bytes per element. Used by the memory-budget tests; the numbers are the
    /// point of choosing packed types (`U8x4` colours, `f16` UVs) in the first
    /// place, so they are worth being able to add up.
    pub fn size(self) -> usize {
        match self {
            AttrType::F32 | AttrType::I32 | AttrType::U8x4 => 4,
            AttrType::Bool => 1,
            AttrType::F32x2 => 8,
            AttrType::F32x3 => 12,
            AttrType::F32x4 => 16,
        }
    }
}

/// A layer's values. One `Arc` per array — see the module docs.
#[derive(Clone, Debug)]
pub enum AttrData {
    F32(Arc<Vec<f32>>),
    F32x2(Arc<Vec<Vec2>>),
    F32x3(Arc<Vec<Vec3>>),
    F32x4(Arc<Vec<Vec4>>),
    I32(Arc<Vec<i32>>),
    Bool(Arc<Vec<bool>>),
    U8x4(Arc<Vec<[u8; 4]>>),
}

impl AttrData {
    pub fn ty(&self) -> AttrType {
        match self {
            AttrData::F32(_) => AttrType::F32,
            AttrData::F32x2(_) => AttrType::F32x2,
            AttrData::F32x3(_) => AttrType::F32x3,
            AttrData::F32x4(_) => AttrType::F32x4,
            AttrData::I32(_) => AttrType::I32,
            AttrData::Bool(_) => AttrType::Bool,
            AttrData::U8x4(_) => AttrType::U8x4,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            AttrData::F32(v) => v.len(),
            AttrData::F32x2(v) => v.len(),
            AttrData::F32x3(v) => v.len(),
            AttrData::F32x4(v) => v.len(),
            AttrData::I32(v) => v.len(),
            AttrData::Bool(v) => v.len(),
            AttrData::U8x4(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Builds a new array by blending several source values per output element.
    ///
    /// `offsets` is a CSR index: output element `i` blends the sources named by
    /// `indices[offsets[i]..offsets[i + 1]]`. A flat pair rather than a list of
    /// lists, because a refining modifier produces one of these per domain and
    /// per level, and a `Vec` per element would allocate once per vertex of the
    /// result.
    ///
    /// What "blend" means depends on the type, and the differences are not
    /// arbitrary:
    ///
    /// - **Floats average.** A UV or a weight halfway between two corners is the
    ///   mean of them.
    /// - **Booleans require all sources.** These are selection and marker layers.
    ///   A new vertex between one selected vertex and one unselected one is not
    ///   selected — taking *any* source instead makes a selection bleed outward
    ///   a ring on every subdivision.
    /// - **Integers take the first source.** They are indices — a material, a
    ///   group — and the mean of two indices is a third thing that means nothing.
    /// - **Byte quadruples average**, as colours. This is the one type whose
    ///   right answer depends on use: averaged as a colour, meaningless as a
    ///   packed id. Colour is the only thing this codebase puts in one.
    pub fn interpolate(&self, offsets: &[u32], indices: &[u32]) -> AttrData {
        let count = offsets.len().saturating_sub(1);
        // The sources of output element `i`, as a slice.
        let sources = |i: usize| -> &[u32] {
            let start = offsets[i] as usize;
            let end = offsets[i + 1] as usize;
            &indices[start.min(indices.len())..end.min(indices.len())]
        };

        macro_rules! mean {
            ($values:expr, $zero:expr) => {{
                let v = $values;
                let mut out = Vec::with_capacity(count);
                for i in 0..count {
                    let src = sources(i);
                    if src.is_empty() {
                        out.push($zero);
                        continue;
                    }
                    let mut sum = $zero;
                    let mut n = 0.0f32;
                    for index in src {
                        if let Some(value) = v.get(*index as usize) {
                            sum += *value;
                            n += 1.0;
                        }
                    }
                    out.push(if n > 0.0 { sum / n } else { $zero });
                }
                out
            }};
        }

        match self {
            AttrData::F32(v) => AttrData::F32(Arc::new(mean!(v, 0.0f32))),
            AttrData::F32x2(v) => AttrData::F32x2(Arc::new(mean!(v, Vec2::ZERO))),
            AttrData::F32x3(v) => AttrData::F32x3(Arc::new(mean!(v, Vec3::ZERO))),
            AttrData::F32x4(v) => AttrData::F32x4(Arc::new(mean!(v, Vec4::ZERO))),
            AttrData::Bool(v) => AttrData::Bool(Arc::new(
                (0..count)
                    .map(|i| {
                        let src = sources(i);
                        !src.is_empty()
                            && src
                                .iter()
                                .all(|j| v.get(*j as usize).copied().unwrap_or(false))
                    })
                    .collect(),
            )),
            AttrData::I32(v) => AttrData::I32(Arc::new(
                (0..count)
                    .map(|i| {
                        sources(i)
                            .first()
                            .and_then(|j| v.get(*j as usize).copied())
                            .unwrap_or(0)
                    })
                    .collect(),
            )),
            AttrData::U8x4(v) => AttrData::U8x4(Arc::new(
                (0..count)
                    .map(|i| {
                        let src = sources(i);
                        let mut sum = [0u32; 4];
                        let mut n = 0u32;
                        for j in src {
                            if let Some(value) = v.get(*j as usize) {
                                for (s, c) in sum.iter_mut().zip(value) {
                                    *s += u32::from(*c);
                                }
                                n += 1;
                            }
                        }
                        if n == 0 {
                            return [0; 4];
                        }
                        let mut out = [0u8; 4];
                        for (o, s) in out.iter_mut().zip(sum) {
                            *o = (s / n) as u8;
                        }
                        out
                    })
                    .collect(),
            )),
        }
    }

    /// Heap bytes this layer's array occupies.
    pub fn memory_bytes(&self) -> usize {
        self.len() * self.ty().size()
    }

    /// Allocates `len` default values of the given type.
    pub fn zeroed(ty: AttrType, len: usize) -> Self {
        match ty {
            AttrType::F32 => AttrData::F32(Arc::new(vec![0.0; len])),
            AttrType::F32x2 => AttrData::F32x2(Arc::new(vec![Vec2::ZERO; len])),
            AttrType::F32x3 => AttrData::F32x3(Arc::new(vec![Vec3::ZERO; len])),
            AttrType::F32x4 => AttrData::F32x4(Arc::new(vec![Vec4::ZERO; len])),
            AttrType::I32 => AttrData::I32(Arc::new(vec![0; len])),
            AttrType::Bool => AttrData::Bool(Arc::new(vec![false; len])),
            AttrType::U8x4 => AttrData::U8x4(Arc::new(vec![[0; 4]; len])),
        }
    }

    /// Grows or shrinks to `len`, filling new entries with the type's default.
    /// Copies only if the `Arc` is shared, so a no-op resize is free.
    fn resize(&mut self, len: usize) {
        if self.len() == len {
            return;
        }
        match self {
            AttrData::F32(v) => Arc::make_mut(v).resize(len, 0.0),
            AttrData::F32x2(v) => Arc::make_mut(v).resize(len, Vec2::ZERO),
            AttrData::F32x3(v) => Arc::make_mut(v).resize(len, Vec3::ZERO),
            AttrData::F32x4(v) => Arc::make_mut(v).resize(len, Vec4::ZERO),
            AttrData::I32(v) => Arc::make_mut(v).resize(len, 0),
            AttrData::Bool(v) => Arc::make_mut(v).resize(len, false),
            AttrData::U8x4(v) => Arc::make_mut(v).resize(len, [0; 4]),
        }
    }

    /// Rewrites entry `i` with the type's default value.
    ///
    /// Needed because arenas reuse freed slots and attribute arrays are indexed
    /// by slot: without this a new element silently inherits the UVs, crease and
    /// selection state of whatever was deleted in its place, which only shows up
    /// after a delete and reads as random corruption.
    fn set_default_at(&mut self, i: usize) {
        if i >= self.len() {
            return;
        }
        match self {
            AttrData::F32(v) => Arc::make_mut(v)[i] = 0.0,
            AttrData::F32x2(v) => Arc::make_mut(v)[i] = Vec2::ZERO,
            AttrData::F32x3(v) => Arc::make_mut(v)[i] = Vec3::ZERO,
            AttrData::F32x4(v) => Arc::make_mut(v)[i] = Vec4::ZERO,
            AttrData::I32(v) => Arc::make_mut(v)[i] = 0,
            AttrData::Bool(v) => Arc::make_mut(v)[i] = false,
            AttrData::U8x4(v) => Arc::make_mut(v)[i] = [0; 4],
        }
    }

    /// Builds a new array by picking entries in the given order.
    ///
    /// The operation every renumbering needs: compaction, and converting between
    /// a slot-indexed edit structure and a densely indexed storage one. Indices
    /// past the end take the type's default rather than panicking, so a caller
    /// with a partial map gets a defined result instead of a crash inside a
    /// conversion.
    pub fn gather(&self, indices: &[u32]) -> Self {
        fn pick<T: Copy + Default>(src: &[T], indices: &[u32]) -> Vec<T> {
            indices
                .iter()
                .map(|&i| src.get(i as usize).copied().unwrap_or_default())
                .collect()
        }
        match self {
            AttrData::F32(v) => AttrData::F32(Arc::new(pick(v, indices))),
            AttrData::F32x2(v) => AttrData::F32x2(Arc::new(pick(v, indices))),
            AttrData::F32x3(v) => AttrData::F32x3(Arc::new(pick(v, indices))),
            AttrData::F32x4(v) => AttrData::F32x4(Arc::new(pick(v, indices))),
            AttrData::I32(v) => AttrData::I32(Arc::new(pick(v, indices))),
            AttrData::Bool(v) => AttrData::Bool(Arc::new(pick(v, indices))),
            AttrData::U8x4(v) => AttrData::U8x4(Arc::new(pick(v, indices))),
        }
    }

    /// Whether two layers share the same allocation.
    ///
    /// This is the free, exact "did this change?" test that the evaluation cache
    /// and the undo system rely on — no manual invalidation discipline needed.
    pub fn ptr_eq(&self, other: &Self) -> bool {
        match (self, other) {
            (AttrData::F32(a), AttrData::F32(b)) => Arc::ptr_eq(a, b),
            (AttrData::F32x2(a), AttrData::F32x2(b)) => Arc::ptr_eq(a, b),
            (AttrData::F32x3(a), AttrData::F32x3(b)) => Arc::ptr_eq(a, b),
            (AttrData::F32x4(a), AttrData::F32x4(b)) => Arc::ptr_eq(a, b),
            (AttrData::I32(a), AttrData::I32(b)) => Arc::ptr_eq(a, b),
            (AttrData::Bool(a), AttrData::Bool(b)) => Arc::ptr_eq(a, b),
            (AttrData::U8x4(a), AttrData::U8x4(b)) => Arc::ptr_eq(a, b),
            // Two different types are never the same array, whatever they hold.
            _ => false,
        }
    }
}

/// One named layer.
#[derive(Clone, Debug)]
pub struct Layer {
    pub name: Box<str>,
    pub domain: Domain,
    pub data: AttrData,
}

/// Well-known layer names.
///
/// Layers whose name starts with `.` are internal: they round-trip through edit
/// mode and undo like any other layer, but are not shown to the user and are not
/// written to interchange formats.
pub mod names {
    /// Point / `F32x3`. Every mesh has this.
    pub const POSITION: &str = "position";
    /// Face / `I32`.
    pub const MATERIAL_INDEX: &str = "material_index";
    /// Face / `Bool`.
    pub const SHADE_SMOOTH: &str = "shade_smooth";
    /// Edge / `F32`, in `0..=1`.
    pub const CREASE: &str = "crease";
    /// Edge / `Bool`.
    pub const SHARP_EDGE: &str = "sharp_edge";
    /// Edge / `Bool`. UV seam.
    pub const SEAM: &str = "seam";
    /// Corner / `F32x3`. Present only when custom split normals are authored;
    /// its presence overrides computed normals.
    pub const CUSTOM_NORMAL: &str = "custom_normal";

    /// Selection lives in attribute layers rather than a side table, so it
    /// survives `MeshData` <-> `BMesh` round trips and shares structurally under
    /// undo without any extra machinery.
    pub const SELECT_VERT: &str = ".select_vert";
    pub const SELECT_EDGE: &str = ".select_edge";
    pub const SELECT_FACE: &str = ".select_face";

    /// `true` for layers that are internal bookkeeping rather than user data.
    pub fn is_internal(name: &str) -> bool {
        name.starts_with('.')
    }

    /// `true` for the three selection layers.
    ///
    /// Named rather than inferred from the leading dot: selection is the one
    /// internal layer a renderer can apply without touching geometry, and a
    /// future internal layer that happens to start with a dot must not be
    /// swept into that category by accident.
    pub fn is_selection(name: &str) -> bool {
        matches!(name, SELECT_VERT | SELECT_EDGE | SELECT_FACE)
    }
}

/// A set of named layers, kept sorted by name.
///
/// A sorted `Vec` rather than a hash map, because iteration order is part of the
/// contract: operator output must never depend on hash iteration order, or
/// golden tests go flaky and file writes stop being reproducible. Layer counts
/// are small (tens), so binary search beats hashing anyway.
#[derive(Clone, Debug, Default)]
pub struct AttributeStore {
    layers: Vec<Layer>,
}

impl AttributeStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn index_of(&self, name: &str) -> Option<usize> {
        self.layers.binary_search_by(|l| (*l.name).cmp(name)).ok()
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Heap bytes every layer's array occupies.
    ///
    /// Names are not counted: a layer name is a handful of bytes and this
    /// number exists to be compared against a per-vertex budget, where the
    /// arrays are the whole story.
    pub fn memory_bytes(&self) -> usize {
        self.layers.iter().map(|l| l.data.memory_bytes()).sum()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.index_of(name).is_some()
    }

    pub fn get(&self, name: &str) -> Option<&Layer> {
        self.index_of(name).map(|i| &self.layers[i])
    }

    /// Iterates layers in name order.
    pub fn iter(&self) -> impl Iterator<Item = &Layer> {
        self.layers.iter()
    }

    pub fn iter_domain(&self, domain: Domain) -> impl Iterator<Item = &Layer> {
        self.layers.iter().filter(move |l| l.domain == domain)
    }

    /// Inserts or replaces a layer, returning the previous one if any.
    pub fn insert(&mut self, name: &str, domain: Domain, data: AttrData) -> Option<Layer> {
        let layer = Layer {
            name: name.into(),
            domain,
            data,
        };
        match self.layers.binary_search_by(|l| (*l.name).cmp(name)) {
            Ok(i) => Some(std::mem::replace(&mut self.layers[i], layer)),
            Err(i) => {
                self.layers.insert(i, layer);
                None
            }
        }
    }

    /// Adds a zero-filled layer if absent. Returns whether it was added.
    pub fn ensure(&mut self, name: &str, domain: Domain, ty: AttrType, len: usize) -> bool {
        if self.contains(name) {
            return false;
        }
        self.insert(name, domain, AttrData::zeroed(ty, len));
        true
    }

    pub fn remove(&mut self, name: &str) -> Option<Layer> {
        self.index_of(name).map(|i| self.layers.remove(i))
    }

    /// Resizes every layer in `domain` to `len`.
    ///
    /// Called when topology changes. Layers in other domains keep their `Arc`
    /// untouched and so stay structurally shared.
    pub fn resize_domain(&mut self, domain: Domain, len: usize) {
        for l in self.layers.iter_mut().filter(|l| l.domain == domain) {
            l.data.resize(len);
        }
    }

    /// Resets entry `index` of every layer in a domain to its default.
    ///
    /// Called when an arena hands back a recycled slot.
    pub fn reset_at(&mut self, domain: Domain, index: usize) {
        for layer in self.layers.iter_mut() {
            if layer.domain == domain {
                layer.data.set_default_at(index);
            }
        }
    }

    /// Rebuilds every layer of a domain by picking entries in the given order.
    ///
    /// The bulk renumbering hook: one call reindexes every layer in lockstep, so
    /// a caller cannot reindex three of four and leave the fourth silently
    /// misaligned.
    pub fn gather_domain(&mut self, domain: Domain, indices: &[u32]) {
        for layer in self.layers.iter_mut() {
            if layer.domain == domain {
                layer.data = layer.data.gather(indices);
            }
        }
    }
}

/// Generates the typed accessor pairs: a read accessor, and a copy-on-write
/// `make_mut_*` which is the only way to mutate a layer.
macro_rules! typed_accessors {
    ($( $variant:ident, $elem:ty, $get:ident, $make_mut:ident );* $(;)?) => {
        impl AttributeStore {
            $(
                /// Reads the named layer if it exists and has this type.
                pub fn $get(&self, name: &str) -> Option<&[$elem]> {
                    match &self.get(name)?.data {
                        AttrData::$variant(v) => Some(v.as_slice()),
                        _ => None,
                    }
                }

                /// Mutable access, cloning the array only if it is shared with a
                /// snapshot (copy-on-write).
                pub fn $make_mut(&mut self, name: &str) -> Option<&mut Vec<$elem>> {
                    let i = self.index_of(name)?;
                    match &mut self.layers[i].data {
                        AttrData::$variant(v) => Some(Arc::make_mut(v)),
                        _ => None,
                    }
                }
            )*
        }
    };
}

typed_accessors! {
    F32,   f32,     get_f32,   make_mut_f32;
    F32x2, Vec2,    get_f32x2, make_mut_f32x2;
    F32x3, Vec3,    get_f32x3, make_mut_f32x3;
    F32x4, Vec4,    get_f32x4, make_mut_f32x4;
    I32,   i32,     get_i32,   make_mut_i32;
    Bool,  bool,    get_bool,  make_mut_bool;
    U8x4,  [u8; 4], get_u8x4,  make_mut_u8x4;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store_with_positions(n: usize) -> AttributeStore {
        let mut s = AttributeStore::new();
        s.insert(
            names::POSITION,
            Domain::Point,
            AttrData::F32x3(Arc::new(vec![Vec3::ZERO; n])),
        );
        s
    }

    #[test]
    fn iteration_is_deterministic_regardless_of_insertion_order() {
        let collect = |order: [&str; 3]| {
            let mut s = AttributeStore::new();
            for n in order {
                s.insert(n, Domain::Point, AttrData::zeroed(AttrType::F32, 1));
            }
            s.iter().map(|l| l.name.to_string()).collect::<Vec<_>>()
        };
        assert_eq!(
            collect(["zeta", "alpha", "mid"]),
            collect(["mid", "zeta", "alpha"])
        );
        assert_eq!(collect(["zeta", "alpha", "mid"]), ["alpha", "mid", "zeta"]);
    }

    #[test]
    fn typed_read_rejects_wrong_type() {
        let s = store_with_positions(3);
        assert!(s.get_f32x3(names::POSITION).is_some());
        assert!(s.get_f32(names::POSITION).is_none());
        assert!(s.get_f32x3("nope").is_none());
    }

    #[test]
    fn make_mut_is_copy_on_write() {
        let mut a = store_with_positions(2);
        // A snapshot is a clone of the store: Arcs are shared, nothing is copied.
        let snapshot = a.clone();
        let before = snapshot.get_f32x3(names::POSITION).unwrap().to_vec();

        a.make_mut_f32x3(names::POSITION).unwrap()[0] = Vec3::X;

        assert_eq!(a.get_f32x3(names::POSITION).unwrap()[0], Vec3::X);
        assert_eq!(
            snapshot.get_f32x3(names::POSITION).unwrap(),
            before.as_slice(),
            "mutating the live store must not disturb the snapshot"
        );
    }

    #[test]
    fn untouched_layers_stay_shared_after_a_write() {
        let mut a = store_with_positions(2);
        a.insert(
            names::CREASE,
            Domain::Edge,
            AttrData::zeroed(AttrType::F32, 4),
        );
        let snapshot = a.clone();

        a.make_mut_f32x3(names::POSITION).unwrap()[0] = Vec3::X;

        // The property the whole undo design rests on: a position edit must not
        // deep-copy unrelated layers.
        assert!(
            a.get(names::CREASE)
                .unwrap()
                .data
                .ptr_eq(&snapshot.get(names::CREASE).unwrap().data),
            "crease layer should still be shared"
        );
        assert!(
            !a.get(names::POSITION)
                .unwrap()
                .data
                .ptr_eq(&snapshot.get(names::POSITION).unwrap().data),
            "position layer should have diverged"
        );
    }

    #[test]
    fn resize_domain_only_touches_that_domain() {
        let mut s = store_with_positions(2);
        s.insert(
            names::MATERIAL_INDEX,
            Domain::Face,
            AttrData::zeroed(AttrType::I32, 1),
        );
        s.resize_domain(Domain::Point, 5);
        assert_eq!(s.get_f32x3(names::POSITION).unwrap().len(), 5);
        assert_eq!(s.get_i32(names::MATERIAL_INDEX).unwrap().len(), 1);
    }

    #[test]
    fn ensure_does_not_clobber_existing_data() {
        let mut s = store_with_positions(2);
        s.make_mut_f32x3(names::POSITION).unwrap()[1] = Vec3::Y;
        assert!(!s.ensure(names::POSITION, Domain::Point, AttrType::F32x3, 2));
        assert_eq!(s.get_f32x3(names::POSITION).unwrap()[1], Vec3::Y);
    }

    #[test]
    fn selection_layers_are_internal() {
        assert!(names::is_internal(names::SELECT_VERT));
        assert!(!names::is_internal(names::POSITION));
    }
}
