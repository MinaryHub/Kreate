//! The modifier stack: an object's geometry as a function of its datablock.
//!
//! Three modifiers, applied in order, and that is deliberate. The point of this
//! module is not the features — mirror, subdivision and array are each a few
//! hundred lines in `kreate-mesh` — it is to force the separation between the
//! mesh a user *edits* and the mesh that gets *drawn* to exist while the
//! codebase is still small enough to change. Without it every operator, every
//! renderer path and every picking query quietly assumes "the mesh on screen is
//! the mesh in the document", and by the time a modifier makes that false there
//! are hundreds of call sites saying so.
//!
//! # The cache does not live in the document
//!
//! It would be natural to hang the evaluated mesh off `Document`. It is wrong:
//! `Document` is cloned for every undo snapshot and *restored* from one, so a
//! cache inside it travels backwards in time along with the geometry. The cache
//! therefore lives here, is owned by the application, and takes the document as
//! an argument.
//!
//! # An evaluated scene is a function of the document *and the time*
//!
//! §2.4's decision, made here while there is nothing animated to break. The
//! evaluated mesh for an object is not a function of the document alone: as
//! soon as one modifier reads the clock — a wave, a build, anything driven — it
//! is a function of `(document, time)`, and the immutable snapshot the renderer
//! draws is that pair's result.
//!
//! Adding that later is not a field. It is every evaluation path, every cache,
//! and every operator context that had assumed the signature — which is why
//! §5's fourth risk is the only one it calls fatal, and why [`EvalCtx`] exists
//! before anything reads it.
//!
//! **The memory trap, decided with it.** Caching a result per frame is a copy
//! of the mesh per frame: scrubbing a hundred frames of a subdivided model is a
//! hundred subdivided models. So the cache holds *one* time. A stack that does
//! not read the clock keeps its entry across a time change — which is every
//! stack today, so scrubbing a scene with no animation in it costs nothing —
//! and anything wanting several frames at once asks for them explicitly, which
//! is what a bake is.
//!
//! # Keyed on the input handle and the stack, not on the object
//!
//! An entry is looked up by the identity of its input `Arc` plus the stack that
//! was applied to it. Two linked duplicates with the same stack therefore share
//! one evaluated mesh — one evaluation, one GPU upload — exactly as two linked
//! duplicates with *no* stack share the datablock itself. Keying on `ObjectId`
//! would evaluate the same thing once per object and turn a thousand instances
//! back into a thousand meshes.
//!
//! The stack is compared for equality rather than hashed. A parameter hash is
//! smaller, but a collision shows the user the wrong geometry for no visible
//! reason, and a stack is a handful of numbers — there is nothing to save.

use std::sync::Arc;

use kreate_math::SceneTime;
use kreate_mesh::modifiers::{
    array, bevel, boolean, mirror, solidify, subdivide, ArrayParams, BevelParams, BooleanOp,
    MirrorParams, SolidifyParams, SubdivideParams,
};
use kreate_mesh::MeshData;
use kreate_reflect::Uid;

use crate::document::{Document, ObjectId};

/// Everything an evaluation depends on besides its input geometry.
///
/// One field today, and the whole reason the struct exists rather than a bare
/// parameter: the next thing to go in here — a dependency graph's view of the
/// scene, a random seed, a resolution the viewport wants and the renderer does
/// not — goes in without touching a caller. A `time: f64` argument threaded
/// through by hand would need every one of them changed again.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EvalCtx {
    /// When the scene is.
    ///
    /// See [`SceneTime`] for why this is not an `f64`: a bare number is either a
    /// frame or a quantity of seconds, and a modifier reading it as one while a
    /// curve wrote the other is off by the frame rate.
    pub time: SceneTime,
}

impl EvalCtx {
    /// The context for a still scene at the first frame.
    ///
    /// What a script, a test, an importer or an exporter has. Named rather than
    /// spelled `Default::default()` at fifty call sites, so that the day
    /// something else joins `time` it is obvious which of those fifty meant
    /// "the beginning" and which meant "I had nothing to pass".
    pub fn still() -> Self {
        Self::default()
    }

    pub fn at(time: SceneTime) -> Self {
        Self { time }
    }
}

/// One modifier and its parameters.
///
/// An enum rather than a boxed trait: the set is closed, small, and needs to be
/// serialisable, comparable and copyable. A trait object would need each of
/// those bolted back on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ModifierKind {
    Mirror(MirrorParams),
    Subdivide(SubdivideParams),
    Array(ArrayParams),
    Solidify(SolidifyParams),
    Bevel(BevelParams),
    /// Combined with another object's mesh — the one thing in the stack that
    /// reads beyond its own input, so the cache keys it on that object too.
    Boolean(BooleanModifier),
    /// A modifier that moves the mesh according to the clock.
    ///
    /// Exists so that the cache's time-keyed path can be driven, and only in a
    /// test build. Nothing in the shipped set reads the clock yet, which leaves
    /// the two lines §2.4's memory decision lives in — record the time, and
    /// replace the previous instant's entry rather than joining it —
    /// unreachable. A test that could only inspect the *shape* of the cache key
    /// would report success while the thing that stops a mesh accumulating per
    /// frame went unexercised.
    ///
    /// It does what the first real one will: translates by an amount that
    /// depends on the time, so "did this re-evaluate" is answerable by looking
    /// at where the vertices are.
    #[cfg(test)]
    Drift(f32),
}

impl ModifierKind {
    /// The stable identifier, for keymaps, scripting and the file format.
    pub fn id(&self) -> &'static str {
        match self {
            ModifierKind::Mirror(_) => "mirror",
            ModifierKind::Subdivide(_) => "subdivide",
            ModifierKind::Array(_) => "array",
            ModifierKind::Solidify(_) => "solidify",
            ModifierKind::Bevel(_) => "bevel",
            ModifierKind::Boolean(_) => "boolean",
            #[cfg(test)]
            ModifierKind::Drift(_) => "drift",
        }
    }

    /// The translation key for the user-visible name.
    ///
    /// Never the name itself: a bare string here is exactly what the CI gate
    /// forbids, and a modifier's name is on screen whenever its panel is.
    pub fn name_key(&self) -> &'static str {
        match self {
            ModifierKind::Mirror(_) => "modifier-mirror",
            ModifierKind::Subdivide(_) => "modifier-subdivide",
            ModifierKind::Array(_) => "modifier-array",
            ModifierKind::Solidify(_) => "modifier-solidify",
            ModifierKind::Bevel(_) => "modifier-bevel",
            ModifierKind::Boolean(_) => "modifier-boolean",
            #[cfg(test)]
            ModifierKind::Drift(_) => "modifier-drift",
        }
    }

    /// Whether this modifier's result changes when the clock does.
    ///
    /// `false` for all three of them, and that is not a placeholder: a mirror,
    /// a subdivision and an array are functions of their input and their
    /// parameters and nothing else. Asking the question is what lets the cache
    /// keep their results across a time change, so scrubbing a scene with no
    /// animation in it re-evaluates nothing.
    ///
    /// The day a modifier reads the clock, this returns `true` for it and the
    /// cache does the right thing without being told again.
    pub fn depends_on_time(&self) -> bool {
        match self {
            ModifierKind::Mirror(_)
            | ModifierKind::Subdivide(_)
            | ModifierKind::Array(_)
            | ModifierKind::Solidify(_)
            | ModifierKind::Bevel(_)
            | ModifierKind::Boolean(_) => false,
            #[cfg(test)]
            ModifierKind::Drift(_) => true,
        }
    }

    fn apply(&self, input: &MeshData, ctx: &EvalCtx, operands: &[Operand]) -> MeshData {
        // Taken and not read, by every modifier that ships so far. The
        // parameter is the point: it is the one place a time-dependent modifier
        // needs it, and threading it through afterwards is the change §5 calls
        // fatal.
        let _ = ctx;
        match self {
            ModifierKind::Mirror(p) => mirror(input, p),
            ModifierKind::Subdivide(p) => subdivide(input, p),
            ModifierKind::Array(p) => array(input, p),
            ModifierKind::Solidify(p) => solidify(input, p),
            ModifierKind::Bevel(p) => bevel(input, p),
            ModifierKind::Boolean(b) => match operands.iter().find(|o| o.uid == b.operand) {
                Some(operand) => {
                    let mut cutter = (*operand.mesh).clone();
                    for p in cutter.positions_mut() {
                        *p = operand.to_local.transform_point3(p.as_dvec3()).as_vec3();
                    }
                    boolean(input, &cutter, b.op)
                }
                // Nothing to combine with — no object picked yet, or it is
                // gone: the input as it was, not nothing.
                None => input.clone(),
            },
            #[cfg(test)]
            ModifierKind::Drift(per_second) => {
                let mut moved = input.clone();
                let offset = *per_second * ctx.time.seconds() as f32;
                for p in moved.positions_mut() {
                    p.x += offset;
                }
                moved
            }
        }
    }
}

/// A boolean modifier's settings: the other object, by its identity so a
/// rename does not break it, and what to do with it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BooleanModifier {
    pub operand: Uid,
    pub op: BooleanOp,
}

/// Another object's mesh, as a boolean reads it: its datablock, and the
/// transform from its space into the space of the object being evaluated.
#[derive(Clone, Debug)]
pub struct Operand {
    pub uid: Uid,
    pub mesh: Arc<MeshData>,
    pub to_local: kreate_math::DAffine3,
}

impl PartialEq for Operand {
    fn eq(&self, other: &Self) -> bool {
        self.uid == other.uid
            && Arc::ptr_eq(&self.mesh, &other.mesh)
            && self.to_local == other.to_local
    }
}

/// A modifier in a stack: what it is, and whether it is on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModifierInstance {
    pub kind: ModifierKind,
    /// Disabled modifiers stay in the stack. Toggling one off to see the cage
    /// underneath is the commonest thing a user does with the panel, and it must
    /// not lose the parameters.
    pub enabled: bool,
}

impl ModifierInstance {
    pub fn new(kind: ModifierKind) -> Self {
        Self {
            kind,
            enabled: true,
        }
    }
}

/// Applies a stack to a mesh.
///
/// Linear and in order, which is not a simplification to be apologised for:
/// mirror-then-subdivide and subdivide-then-mirror are different results, and a
/// stack whose order did not matter would be hiding that from the user.
pub fn apply_stack(input: &MeshData, stack: &[ModifierInstance], ctx: &EvalCtx) -> MeshData {
    apply_stack_with(input, stack, ctx, &[])
}

/// The same, with the other objects a boolean in the stack reads.
pub fn apply_stack_with(
    input: &MeshData,
    stack: &[ModifierInstance],
    ctx: &EvalCtx,
    operands: &[Operand],
) -> MeshData {
    let mut current: Option<MeshData> = None;
    for modifier in stack.iter().filter(|m| m.enabled) {
        let next = modifier
            .kind
            .apply(current.as_ref().unwrap_or(input), ctx, operands);
        current = Some(next);
    }
    current.unwrap_or_else(|| input.clone())
}

/// The objects some shown object's enabled boolean cuts with.
///
/// The viewport draws these as wireframes, as Blender's Bool Tool sets them
/// to: a cutter drawn solid sits exactly over the hole it makes, so the one
/// thing a boolean does is the one thing that cannot be seen. Derived rather
/// than stored, so there is no display setting to save or to forget to undo —
/// switching the modifier off, or removing it, gives the cutter its surface
/// back.
pub fn boolean_cutters(doc: &Document) -> Vec<ObjectId> {
    let mut out: Vec<ObjectId> = Vec::new();
    for (id, object) in doc.objects() {
        if !doc.is_shown(id) {
            continue;
        }
        for m in object.modifiers.iter().filter(|m| m.enabled) {
            let ModifierKind::Boolean(b) = m.kind else {
                continue;
            };
            if let Some(cutter) = doc.object_by_uid(b.operand).filter(|c| *c != id) {
                if !out.contains(&cutter) {
                    out.push(cutter);
                }
            }
        }
    }
    out
}

/// The mesh `object`'s modifier at `index` would make of its datablock on its
/// own — what Apply bakes. Blender's rule: the one modifier, on the mesh as
/// stored, whatever sits below it in the stack. `None` for no such modifier.
pub fn apply_one(doc: &Document, object: ObjectId, index: usize) -> Option<MeshData> {
    let o = doc.object(object)?;
    let modifier = *o.modifiers.get(index)?;
    let input = doc.mesh(o.mesh()?)?;
    let one = [ModifierInstance {
        enabled: true,
        ..modifier
    }];
    let operands = operands_of(doc, object, &one);
    Some(apply_stack_with(input, &one, &EvalCtx::still(), &operands))
}

/// The other objects `id`'s booleans read, resolved now: each one's mesh and
/// where it is relative to `id`. An operand that is `id` itself, or not a mesh,
/// is left out, and its boolean passes its input through.
fn operands_of(doc: &Document, id: ObjectId, stack: &[ModifierInstance]) -> Vec<Operand> {
    let mut out: Vec<Operand> = Vec::new();
    let to_local = doc.world_transform(id).inverse();
    for m in stack.iter().filter(|m| m.enabled) {
        let ModifierKind::Boolean(b) = m.kind else {
            continue;
        };
        if out.iter().any(|o| o.uid == b.operand) {
            continue;
        }
        let Some(other) = doc.object_by_uid(b.operand).filter(|o| *o != id) else {
            continue;
        };
        let Some(mesh) = doc
            .object(other)
            .and_then(|o| o.mesh())
            .and_then(|m| doc.mesh_arc(m))
        else {
            continue;
        };
        out.push(Operand {
            uid: b.operand,
            mesh: mesh.clone(),
            to_local: to_local * doc.world_transform(other),
        });
    }
    out
}

/// Whether any enabled modifier in a stack reads the clock.
///
/// What lets the cache keep a stack's result across a time change. False for
/// every stack that can be built today, which is why scrubbing a scene with no
/// animation in it re-evaluates nothing.
fn depends_on_time(stack: &[ModifierInstance]) -> bool {
    stack
        .iter()
        .filter(|m| m.enabled)
        .any(|m| m.kind.depends_on_time())
}

/// Whether a stack would change its input at all.
///
/// An empty stack — or one with everything switched off — must return the input
/// handle itself rather than a copy of it, so that an object with no modifiers
/// is indistinguishable from one before modifiers existed. That is what keeps
/// linked duplicates sharing a single GPU mesh.
fn is_identity(stack: &[ModifierInstance]) -> bool {
    !stack.iter().any(|m| m.enabled)
}

/// A mesh with its shape keys mixed in, and what it was mixed from.
#[derive(Clone, Debug)]
struct ShapeEntry {
    input: Arc<MeshData>,
    keys: Vec<crate::shape::ShapeKey>,
    output: Arc<MeshData>,
    used: u64,
}

/// A geometry graph's result, and the input and graph it was made from.
#[derive(Clone, Debug)]
struct GeoEntry {
    input: Arc<MeshData>,
    graph: crate::geonodes::GeoGraph,
    output: Arc<MeshData>,
    used: u64,
}

/// A mesh with its simulation laid on: what it was made from and of.
#[derive(Clone, Debug)]
struct SimEntry {
    object: kreate_reflect::Uid,
    input: Arc<MeshData>,
    /// The bake it read, by address.
    cache: usize,
    /// The frame it is of, when the bake has anything for it; `None` for
    /// hair at rest, which is the same at every frame.
    frame: Option<i64>,
    physics: crate::physics::ObjectPhysics,
    /// The instance objects' meshes, by address: a changed one is new
    /// particles.
    instances: Vec<usize>,
    output: Arc<MeshData>,
    used: u64,
}

/// A skinned mesh, and the input and bone matrices it was made from.
#[derive(Clone, Debug)]
struct SkinEntry {
    input: Arc<MeshData>,
    matrices: Vec<kreate_math::DAffine3>,
    preserve_volume: bool,
    output: Arc<MeshData>,
    used: u64,
}

struct Entry {
    input: Arc<MeshData>,
    stack: Vec<ModifierInstance>,
    /// The other objects a boolean read, and where they were: a cutter that
    /// moves is a different result from the same stack.
    operands: Vec<Operand>,
    output: Arc<MeshData>,
    /// The time this entry was evaluated at, when its stack reads the clock.
    ///
    /// `None` for a stack that does not — which is every stack today — and that
    /// is the whole point of storing it this way rather than storing the time
    /// unconditionally. An entry with no time is valid at every time, so
    /// scrubbing a scene with no animation in it re-evaluates nothing; an entry
    /// *with* one is valid only at that instant, so scrubbing an animated scene
    /// does not accumulate a mesh per frame.
    at: Option<SceneTime>,
    /// The sweep generation this entry was last asked for.
    used: u64,
}

/// Evaluated meshes, kept between frames.
///
/// Owned by the application, not by the document — see the module docs.
#[derive(Default)]
pub struct ModifierCache {
    entries: Vec<Entry>,
    /// Skinned meshes, one per bound input: see `ModifierCache::skinned`.
    skins: Vec<SkinEntry>,
    /// Shape-key mixes, one per input and set of values.
    shapes: Vec<ShapeEntry>,
    /// Geometry node results, one per input.
    geo: Vec<GeoEntry>,
    /// Simulation results laid on, one per object.
    sims: Vec<SimEntry>,
    generation: u64,
    /// Counters, so "did it re-evaluate more than it had to" is answerable. It
    /// is invisible at a glance and shows up only as a stutter on a scene big
    /// enough that it is already too late to notice.
    hits: u64,
    misses: u64,
}

impl ModifierCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// The evaluated mesh for one object, evaluating it if necessary.
    ///
    /// `None` when the object has no mesh, or references one that is not in the
    /// document.
    pub fn evaluated(
        &mut self,
        doc: &Document,
        id: ObjectId,
        ctx: &EvalCtx,
    ) -> Option<Arc<MeshData>> {
        let base = self.evaluated_base(doc, id, ctx)?;
        Some(self.simulated(doc, id, base))
    }

    /// The evaluated mesh before any simulation is laid on: what a bake
    /// starts from.
    pub fn evaluated_base(
        &mut self,
        doc: &Document,
        id: ObjectId,
        ctx: &EvalCtx,
    ) -> Option<Arc<MeshData>> {
        let object = doc.object(id)?;
        let stored = doc.mesh_arc(object.mesh()?)?;
        // Shape keys first, as Blender's are — under the armature and the
        // stack — and not while edit mode has a key traded into the mesh.
        let shaped =
            if object.editing_shape.is_none() && object.shape_keys.iter().any(|k| k.value != 0.0) {
                self.shaped(stored, &object.shape_keys)
            } else {
                stored.clone()
            };
        let input = &shaped;
        // A mesh bound to an armature is skinned first, as Blender's Armature
        // modifier usually sits at the top of the stack, and the stack runs on
        // what the bones made of it.
        let skinned = match doc.skin_matrices(id) {
            Some((armature, matrices)) => {
                self.skinned(doc, input, armature, matrices, object.preserve_volume)
            }
            None => input.clone(),
        };
        let operands = operands_of(doc, id, &object.modifiers);
        let stacked = self.evaluate_with(&skinned, &object.modifiers, ctx, &operands);
        // Geometry nodes, after the stack, as Blender's modifier sits last.
        Some(match &object.geometry_nodes {
            Some(graph) => self.noded(&stacked, graph),
            None => stacked,
        })
    }

    /// `input` with the object's baked cloth vertices, particles and hair laid
    /// on, at the document's frame. See [`crate::physics`].
    fn simulated(&mut self, doc: &Document, id: ObjectId, input: Arc<MeshData>) -> Arc<MeshData> {
        let Some(object) = doc.object(id) else {
            return input;
        };
        let physics = &object.physics;
        // A baked liquid domain is its surface, not its box.
        if let Some(crate::physics::Fluid::Domain(_)) = physics.fluid {
            if let Some(surface) = doc.physics_cache.fluid_at(object.uid, doc.playback.frame) {
                return surface.clone();
            }
        }
        if physics.soft.is_none() && physics.particles.is_empty() {
            return input;
        }
        let cache = &doc.physics_cache;
        let uid = object.uid;
        let baked = cache.soft_of(uid).is_some() || cache.particles.iter().any(|t| t.object == uid);
        let frame = baked.then(|| doc.playback.frame.round() as i64);
        let instance_of = |s: &crate::physics::ParticleSettings| {
            let o = doc.object(doc.object_by_uid(s.instance?)?)?;
            Some((doc.mesh_arc(o.mesh()?)?.clone(), o.transform))
        };
        let instances: Vec<Option<(Arc<MeshData>, kreate_math::DTransform)>> =
            physics.particles.iter().map(instance_of).collect();
        let instance_keys: Vec<usize> = instances
            .iter()
            .map(|i| i.as_ref().map_or(0, |(m, _)| Arc::as_ptr(m) as usize))
            .collect();
        let cache_key = Arc::as_ptr(cache) as usize;
        if let Some(entry) = self.sims.iter_mut().find(|e| {
            e.object == uid
                && Arc::ptr_eq(&e.input, &input)
                && e.cache == cache_key
                && e.frame == frame
                && e.physics == *physics
                && e.instances == instance_keys
        }) {
            entry.used = self.generation;
            return entry.output.clone();
        }
        let at = doc.playback.frame;
        let mut mesh = (*input).clone();
        if let Some(v) = cache.soft_at(uid, at) {
            if v.len() == mesh.num_verts() as usize {
                mesh.positions_mut().copy_from_slice(v);
            }
        }
        let mut parts: Vec<MeshData> = Vec::new();
        for (i, (s, inst)) in physics.particles.iter().zip(&instances).enumerate() {
            // The instance as it stands — turned and sized — but not moved:
            // every particle is its own place.
            let shaped = inst.as_ref().map(|(m, t)| {
                let mut m = (**m).clone();
                for p in m.positions_mut() {
                    *p = (t.rotation * (p.as_dvec3() * t.scale)).as_vec3();
                }
                m
            });
            let particles = cache.particles_at(uid, i, at).map(|p| &p[..]);
            if let Some(m) =
                crate::physics::particles::system_mesh(&input, s, particles, shaped.as_ref())
            {
                parts.push(m);
            }
        }
        let show_emitter = physics.particles.iter().all(|s| s.show_emitter);
        let output = if parts.is_empty() {
            mesh
        } else {
            let mut list: Vec<kreate_mesh::edit::join::Part<'_>> = Vec::new();
            if show_emitter {
                list.push(kreate_mesh::edit::join::Part {
                    mesh: &mesh,
                    transform: kreate_math::DAffine3::IDENTITY,
                });
            }
            for m in &parts {
                list.push(kreate_mesh::edit::join::Part {
                    mesh: m,
                    transform: kreate_math::DAffine3::IDENTITY,
                });
            }
            kreate_mesh::edit::join::join(&list).0
        };
        let output = Arc::new(output);
        self.sims.retain(|e| e.object != uid);
        self.sims.push(SimEntry {
            object: uid,
            input,
            cache: cache_key,
            frame,
            physics: physics.clone(),
            instances: instance_keys,
            output: output.clone(),
            used: self.generation,
        });
        output
    }

    /// The graph's result for `input`, from the cache while both are the same.
    fn noded(&mut self, input: &Arc<MeshData>, graph: &crate::geonodes::GeoGraph) -> Arc<MeshData> {
        if let Some(entry) = self
            .geo
            .iter_mut()
            .find(|e| Arc::ptr_eq(&e.input, input) && e.graph == *graph)
        {
            entry.used = self.generation;
            return entry.output.clone();
        }
        let output = graph
            .evaluate(input)
            .map(Arc::new)
            .unwrap_or_else(|| Arc::new(MeshData::empty()));
        self.geo.retain(|e| !Arc::ptr_eq(&e.input, input));
        self.geo.push(GeoEntry {
            input: input.clone(),
            graph: graph.clone(),
            output: output.clone(),
            used: self.generation,
        });
        output
    }

    /// The mesh with its keys mixed in, from the cache while the input and the
    /// values are the same. A new mix of one input replaces the old.
    fn shaped(&mut self, input: &Arc<MeshData>, keys: &[crate::shape::ShapeKey]) -> Arc<MeshData> {
        if let Some(entry) = self
            .shapes
            .iter_mut()
            .find(|e| Arc::ptr_eq(&e.input, input) && e.keys == keys)
        {
            entry.used = self.generation;
            return entry.output.clone();
        }
        let output = crate::shape::mix(input, keys)
            .map(Arc::new)
            .unwrap_or_else(|| input.clone());
        self.shapes.retain(|e| !Arc::ptr_eq(&e.input, input));
        self.shapes.push(ShapeEntry {
            input: input.clone(),
            keys: keys.to_vec(),
            output: output.clone(),
            used: self.generation,
        });
        output
    }

    /// The skinned mesh, from the cache while the input and the bones' matrices
    /// are the same ones. One entry per input: a new pose replaces the old,
    /// for the memory reason a time-dependent stack gives.
    fn skinned(
        &mut self,
        doc: &Document,
        input: &Arc<MeshData>,
        armature: crate::document::ArmatureId,
        matrices: Vec<kreate_math::DAffine3>,
        preserve_volume: bool,
    ) -> Arc<MeshData> {
        if let Some(entry) = self.skins.iter_mut().find(|e| {
            Arc::ptr_eq(&e.input, input)
                && e.matrices == matrices
                && e.preserve_volume == preserve_volume
        }) {
            entry.used = self.generation;
            return entry.output.clone();
        }
        let output = doc
            .armature(armature)
            .and_then(|a| crate::skin::skin_with(input, a, &matrices, preserve_volume))
            .map(Arc::new)
            .unwrap_or_else(|| input.clone());
        self.skins.retain(|e| !Arc::ptr_eq(&e.input, input));
        self.skins.push(SkinEntry {
            input: input.clone(),
            matrices,
            preserve_volume,
            output: output.clone(),
            used: self.generation,
        });
        output
    }

    /// What `mesh` was skinned from and by, when it is exactly a skinned
    /// result — no modifier ran after the bones. For a renderer that skins on
    /// the GPU: it keeps the rest shape uploaded once and needs only the
    /// matrices to draw this.
    pub fn skin_source(
        &self,
        mesh: &Arc<MeshData>,
    ) -> Option<(&Arc<MeshData>, &[kreate_math::DAffine3], bool)> {
        self.skins
            .iter()
            .find(|e| Arc::ptr_eq(&e.output, mesh) && !Arc::ptr_eq(&e.input, &e.output))
            .map(|e| (&e.input, e.matrices.as_slice(), e.preserve_volume))
    }

    /// The evaluated result of applying `stack` to `input` at `ctx`'s time.
    pub fn evaluate(
        &mut self,
        input: &Arc<MeshData>,
        stack: &[ModifierInstance],
        ctx: &EvalCtx,
    ) -> Arc<MeshData> {
        self.evaluate_with(input, stack, ctx, &[])
    }

    /// The same, with the other objects its booleans read.
    pub fn evaluate_with(
        &mut self,
        input: &Arc<MeshData>,
        stack: &[ModifierInstance],
        ctx: &EvalCtx,
        operands: &[Operand],
    ) -> Arc<MeshData> {
        if is_identity(stack) {
            return input.clone();
        }
        // Only recorded for a stack that reads the clock. An entry with no time
        // is valid at every time, which is what stops a still scene
        // re-evaluating when the frame changes; see [`Entry::at`].
        let at = depends_on_time(stack).then_some(ctx.time);
        // The whole invalidation mechanism, in one comparison. It is exact and
        // free, and it works only because the entry *holds* the input handle:
        // `Arc::make_mut` copies on write only while a handle is shared, so an
        // entry storing an id would watch the document mutate in place behind an
        // unchanged pointer and go on serving stale geometry.
        if let Some(entry) = self.entries.iter_mut().find(|e| {
            Arc::ptr_eq(&e.input, input) && e.stack == stack && e.at == at && e.operands == operands
        }) {
            entry.used = self.generation;
            self.hits += 1;
            return entry.output.clone();
        }

        let output = Arc::new(apply_stack_with(input, stack, ctx, operands));
        self.misses += 1;
        // A time-dependent entry replaces the one for the previous instant
        // rather than joining it. This is §2.4's memory decision: a result per
        // frame is a copy of the mesh per frame, so scrubbing a hundred frames
        // of a subdivided model would be a hundred subdivided models. Anything
        // that genuinely wants several frames at once — motion blur wants two,
        // a geometry cache wants all of them — asks explicitly, and that is what
        // a bake is.
        if at.is_some() {
            self.entries
                .retain(|e| !(Arc::ptr_eq(&e.input, input) && e.stack == stack));
        }
        self.entries.push(Entry {
            input: input.clone(),
            stack: stack.to_vec(),
            operands: operands.to_vec(),
            output: output.clone(),
            at,
            used: self.generation,
        });
        output
    }

    /// Every visible mesh object's evaluated geometry, in document order.
    ///
    /// The renderer's and the picker's view of the scene: neither of them sees a
    /// `MeshId` any more, which is the separation this module exists to create.
    pub fn evaluate_scene(
        &mut self,
        doc: &Document,
        ctx: &EvalCtx,
    ) -> Vec<(ObjectId, Arc<MeshData>)> {
        self.evaluate_scene_with_cage(doc, None, ctx)
    }

    /// The same, with one object left unevaluated.
    ///
    /// Edit mode shows and edits the **cage**: the mesh as it is in the document,
    /// not the result of its modifier stack. That is not a simplification — it is
    /// what makes element selection mean anything. An element index taken from
    /// the id buffer is an index into whatever was drawn, so drawing a mirrored
    /// copy would have a click on the reflection name a face on the other side,
    /// and a click on a subdivided surface name a face that does not exist in the
    /// document at all.
    ///
    /// Everything else in the scene still shows its modifier result, so the
    /// object being edited stays in context.
    pub fn evaluate_scene_with_cage(
        &mut self,
        doc: &Document,
        cage: Option<ObjectId>,
        ctx: &EvalCtx,
    ) -> Vec<(ObjectId, Arc<MeshData>)> {
        let ids: Vec<ObjectId> = doc
            .objects()
            .filter(|(id, o)| doc.is_shown(*id) && o.mesh().is_some())
            .map(|(id, _)| id)
            .collect();
        ids.into_iter()
            .filter_map(|id| {
                if Some(id) == cage {
                    let mesh = doc.object(id)?.mesh()?;
                    return doc.mesh_arc(mesh).map(|m| (id, m.clone()));
                }
                self.evaluated(doc, id, ctx).map(|m| (id, m))
            })
            .collect()
    }

    /// Drops entries that were not asked for since the last sweep.
    ///
    /// Called once a frame, after evaluation. Without it the cache keeps one
    /// entry per parameter value the user has ever dragged through, which on a
    /// subdivision modifier is a copy of the mesh per frame of the drag.
    pub fn sweep(&mut self) {
        let generation = self.generation;
        self.entries.retain(|e| e.used == generation);
        self.skins.retain(|e| e.used == generation);
        self.shapes.retain(|e| e.used == generation);
        self.geo.retain(|e| e.used == generation);
        self.sims.retain(|e| e.used == generation);
        self.generation += 1;
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Cache hits and misses since the cache was created.
    pub fn stats(&self) -> (u64, u64) {
        (self.hits, self.misses)
    }

    /// Roughly how much the evaluated meshes occupy.
    ///
    /// Only the outputs: the inputs are the document's and are counted there.
    pub fn memory_bytes(&self) -> usize {
        self.entries
            .iter()
            .map(|e| e.output.memory_bytes())
            .sum::<usize>()
            + self
                .skins
                .iter()
                .map(|e| e.output.memory_bytes())
                .sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::{Object, ObjectData};
    use kreate_mesh::primitives;

    fn doc_with_cube() -> (Document, ObjectId) {
        let mut doc = Document::new();
        let id = doc.add_mesh_object("Cube", primitives::cube(2.0));
        (doc, id)
    }

    fn mirror_x() -> ModifierInstance {
        ModifierInstance::new(ModifierKind::Mirror(MirrorParams::default()))
    }

    #[test]
    fn an_object_with_no_modifiers_evaluates_to_its_own_datablock() {
        // Not merely to an equal mesh: to the *same handle*. An object with no
        // modifiers has to be indistinguishable from one before modifiers
        // existed, or every linked duplicate becomes its own GPU upload.
        let (doc, id) = doc_with_cube();
        let mut cache = ModifierCache::new();
        let evaluated = cache.evaluated(&doc, id, &EvalCtx::still()).unwrap();
        let datablock = doc
            .mesh_arc(doc.object(id).unwrap().mesh().unwrap())
            .unwrap();
        assert!(Arc::ptr_eq(&evaluated, datablock));
        assert!(cache.is_empty(), "the identity stack needs no entry");
    }

    #[test]
    fn a_stack_changes_the_evaluated_mesh_and_leaves_the_datablock_alone() {
        // The separation, stated as a test: the document still holds the cage the
        // user edits, and the thing on screen is something else.
        let (mut doc, id) = doc_with_cube();
        doc.object_mut(id)
            .unwrap()
            .modifiers
            .push(ModifierInstance::new(ModifierKind::Subdivide(
                SubdivideParams {
                    levels: 1,
                    ..Default::default()
                },
            )));

        let mut cache = ModifierCache::new();
        let evaluated = cache.evaluated(&doc, id, &EvalCtx::still()).unwrap();
        assert_eq!(evaluated.num_faces(), 24);
        let mesh_id = doc.object(id).unwrap().mesh().unwrap();
        assert_eq!(doc.mesh(mesh_id).unwrap().num_faces(), 6, "the cage moved");
    }

    #[test]
    fn a_disabled_modifier_keeps_its_parameters_and_does_nothing() {
        let (mut doc, id) = doc_with_cube();
        let mut instance = mirror_x();
        instance.enabled = false;
        doc.object_mut(id).unwrap().modifiers.push(instance);

        let mut cache = ModifierCache::new();
        let evaluated = cache.evaluated(&doc, id, &EvalCtx::still()).unwrap();
        assert_eq!(evaluated.num_faces(), 6);
        assert_eq!(
            doc.object(id).unwrap().modifiers[0].kind,
            ModifierKind::Mirror(MirrorParams::default()),
            "toggling a modifier off must not lose what it was set to"
        );
    }

    #[test]
    fn the_order_of_the_stack_matters() {
        // Arraying a mirror and mirroring an array are different meshes: the
        // array's offset is a fraction of the bounding box, and mirroring first
        // doubles that box. A stack whose order did not matter would be hiding
        // that from the user.
        //
        // Compared by bounds rather than by face count, because the counts agree
        // — both orders make four copies of the input — which is exactly why a
        // count is not enough to tell these two apart.
        let cube = primitives::cube(2.0);
        let array = ModifierInstance::new(ModifierKind::Array(ArrayParams::default()));
        let mirror = ModifierInstance::new(ModifierKind::Mirror(MirrorParams::default()));
        let a = apply_stack(&cube, &[mirror, array], &EvalCtx::still());
        let b = apply_stack(&cube, &[array, mirror], &EvalCtx::still());

        assert_eq!(a.num_faces(), b.num_faces());
        assert_ne!(
            a.bounds().min.x,
            b.bounds().min.x,
            "the order made no difference: {:?} against {:?}",
            a.bounds(),
            b.bounds()
        );
    }

    #[test]
    fn an_unchanged_document_is_a_cache_hit() {
        let (mut doc, id) = doc_with_cube();
        doc.object_mut(id).unwrap().modifiers.push(mirror_x());

        let mut cache = ModifierCache::new();
        let first = cache.evaluated(&doc, id, &EvalCtx::still()).unwrap();
        let second = cache.evaluated(&doc, id, &EvalCtx::still()).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(cache.stats(), (1, 1));
    }

    #[test]
    fn editing_the_input_mesh_invalidates_the_entry() {
        let (mut doc, id) = doc_with_cube();
        doc.object_mut(id).unwrap().modifiers.push(mirror_x());
        let mut cache = ModifierCache::new();
        let before = cache.evaluated(&doc, id, &EvalCtx::still()).unwrap();

        let mesh_id = doc.object(id).unwrap().mesh().unwrap();
        doc.mesh_mut(mesh_id).unwrap().positions_mut()[0].y += 1.0;

        let after = cache.evaluated(&doc, id, &EvalCtx::still()).unwrap();
        assert!(!Arc::ptr_eq(&before, &after), "stale geometry survived");
        assert_eq!(cache.stats().1, 2, "the second call re-evaluated");
    }

    #[test]
    fn changing_a_parameter_invalidates_the_entry() {
        let (mut doc, id) = doc_with_cube();
        doc.object_mut(id)
            .unwrap()
            .modifiers
            .push(ModifierInstance::new(ModifierKind::Array(ArrayParams {
                count: 2,
                ..Default::default()
            })));
        let mut cache = ModifierCache::new();
        let before = cache.evaluated(&doc, id, &EvalCtx::still()).unwrap();

        doc.object_mut(id).unwrap().modifiers[0].kind = ModifierKind::Array(ArrayParams {
            count: 3,
            ..Default::default()
        });
        let after = cache.evaluated(&doc, id, &EvalCtx::still()).unwrap();
        assert_eq!(before.num_verts(), 16);
        assert_eq!(after.num_verts(), 24);
    }

    #[test]
    fn two_linked_duplicates_with_the_same_stack_share_one_evaluated_mesh() {
        // The reason for keying on the input handle rather than on the object:
        // per-object keying evaluates the same geometry once per instance and
        // turns a thousand linked duplicates back into a thousand meshes.
        let (mut doc, first) = doc_with_cube();
        doc.object_mut(first).unwrap().modifiers.push(mirror_x());
        let second = doc.duplicate_linked(first).unwrap();

        let mut cache = ModifierCache::new();
        let a = cache.evaluated(&doc, first, &EvalCtx::still()).unwrap();
        let b = cache.evaluated(&doc, second, &EvalCtx::still()).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(cache.len(), 1, "one entry for both instances");
    }

    #[test]
    fn two_linked_duplicates_with_different_stacks_do_not() {
        let (mut doc, first) = doc_with_cube();
        doc.object_mut(first).unwrap().modifiers.push(mirror_x());
        let second = doc.duplicate_linked(first).unwrap();
        doc.object_mut(second).unwrap().modifiers[0].kind = ModifierKind::Mirror(MirrorParams {
            x: false,
            y: true,
            ..Default::default()
        });

        let mut cache = ModifierCache::new();
        let a = cache.evaluated(&doc, first, &EvalCtx::still()).unwrap();
        let b = cache.evaluated(&doc, second, &EvalCtx::still()).unwrap();
        assert!(!Arc::ptr_eq(&a, &b));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn the_sweep_drops_what_nobody_asked_for() {
        // Otherwise the cache keeps one entry per parameter value the user has
        // ever dragged through — on a subdivision modifier, a copy of the mesh
        // for every frame of the drag.
        let (mut doc, id) = doc_with_cube();
        let mut cache = ModifierCache::new();
        for count in 2..8u32 {
            doc.object_mut(id).unwrap().modifiers =
                vec![ModifierInstance::new(ModifierKind::Array(ArrayParams {
                    count,
                    ..Default::default()
                }))];
            cache.evaluated(&doc, id, &EvalCtx::still()).unwrap();
        }
        assert_eq!(cache.len(), 6, "every parameter value left an entry");

        cache.sweep();
        cache.evaluated(&doc, id, &EvalCtx::still()).unwrap();
        cache.sweep();
        assert_eq!(cache.len(), 1, "only the live one survives");
    }

    #[test]
    fn a_swept_entry_is_re_evaluated_rather_than_lost() {
        let (mut doc, id) = doc_with_cube();
        doc.object_mut(id).unwrap().modifiers.push(mirror_x());
        let mut cache = ModifierCache::new();
        let before = cache.evaluated(&doc, id, &EvalCtx::still()).unwrap();
        cache.sweep();
        cache.sweep();
        assert!(cache.is_empty());

        let after = cache.evaluated(&doc, id, &EvalCtx::still()).unwrap();
        assert_eq!(before.num_faces(), after.num_faces());
    }

    #[test]
    fn evaluating_the_scene_skips_hidden_objects() {
        let (mut doc, id) = doc_with_cube();
        let hidden = doc.duplicate_linked(id).unwrap();
        doc.object_mut(hidden).unwrap().visible = false;
        doc.add_object(Object::new("Empty", ObjectData::Empty));

        let mut cache = ModifierCache::new();
        let scene = cache.evaluate_scene(&doc, &EvalCtx::still());
        assert_eq!(scene.len(), 1);
        assert_eq!(scene[0].0, id);
    }

    #[test]
    fn the_edited_object_shows_its_cage_while_the_rest_show_their_result() {
        // What makes element selection mean anything: an element index from the
        // id buffer indexes whatever was drawn, so a click on a mirrored copy
        // would otherwise name a face on the other side of the model.
        let (mut doc, edited) = doc_with_cube();
        doc.object_mut(edited)
            .unwrap()
            .modifiers
            .push(ModifierInstance::new(ModifierKind::Subdivide(
                SubdivideParams {
                    levels: 1,
                    ..Default::default()
                },
            )));
        let other = doc.add_mesh_object("Other", primitives::cube(2.0));
        doc.object_mut(other)
            .unwrap()
            .modifiers
            .push(ModifierInstance::new(ModifierKind::Subdivide(
                SubdivideParams {
                    levels: 1,
                    ..Default::default()
                },
            )));

        let mut cache = ModifierCache::new();
        let scene = cache.evaluate_scene_with_cage(&doc, Some(edited), &EvalCtx::still());

        let of = |id: ObjectId| {
            scene
                .iter()
                .find(|(o, _)| *o == id)
                .map(|(_, m)| m.num_faces())
                .unwrap()
        };
        assert_eq!(of(edited), 6, "the edited object shows its cage");
        assert_eq!(of(other), 24, "everything else shows its result");
    }

    #[test]
    fn the_cage_is_the_datablock_handle_itself() {
        // Not a copy of it: the GPU cache and the picker key on handle identity,
        // and a fresh clone every frame would re-upload the mesh on every frame
        // of edit mode.
        let (mut doc, id) = doc_with_cube();
        doc.object_mut(id).unwrap().modifiers.push(mirror_x());
        let mut cache = ModifierCache::new();

        let scene = cache.evaluate_scene_with_cage(&doc, Some(id), &EvalCtx::still());
        let datablock = doc
            .mesh_arc(doc.object(id).unwrap().mesh().unwrap())
            .unwrap();
        assert!(Arc::ptr_eq(&scene[0].1, datablock));

        let again = cache.evaluate_scene_with_cage(&doc, Some(id), &EvalCtx::still());
        assert!(
            Arc::ptr_eq(&scene[0].1, &again[0].1),
            "a new handle every frame"
        );
    }

    #[test]
    fn an_object_with_no_mesh_evaluates_to_nothing() {
        let mut doc = Document::new();
        let id = doc.add_object(Object::new("Empty", ObjectData::Empty));
        let mut cache = ModifierCache::new();
        assert!(cache.evaluated(&doc, id, &EvalCtx::still()).is_none());
    }
}

// --- time as an input -------------------------------------------------------

#[cfg(test)]
mod time_tests {
    use super::*;
    use crate::document::{Object, ObjectData};
    use kreate_mesh::primitives;

    fn at(frame: f64) -> EvalCtx {
        EvalCtx::at(SceneTime::new(frame, 24.0).expect("a valid time"))
    }

    fn cube_with_a_stack() -> (Arc<MeshData>, Vec<ModifierInstance>) {
        let cube = Arc::new(primitives::cube(1.0));
        let stack = vec![ModifierInstance::new(ModifierKind::Subdivide(
            SubdivideParams::default(),
        ))];
        (cube, stack)
    }

    /// A stack that reads the clock: see [`ModifierKind::Drift`].
    fn drifting() -> (Arc<MeshData>, Vec<ModifierInstance>) {
        let cube = Arc::new(primitives::cube(1.0));
        let stack = vec![ModifierInstance::new(ModifierKind::Drift(1.0))];
        (cube, stack)
    }

    /// Where a mesh's first vertex is, for telling one frame from another.
    fn first_x(mesh: &MeshData) -> f32 {
        mesh.positions()[0].x
    }

    #[test]
    fn nothing_that_exists_yet_reads_the_clock() {
        // Stated so that the tests below are read correctly: the *reason*
        // scrubbing is free today is that no modifier is time-dependent, and
        // the day one is, this test is what says so.
        for kind in [
            ModifierKind::Mirror(MirrorParams::default()),
            ModifierKind::Subdivide(SubdivideParams::default()),
            ModifierKind::Array(ArrayParams::default()),
        ] {
            assert!(
                !kind.depends_on_time(),
                "{} claims to depend on time; the cache tests below assume \
                 otherwise",
                kind.id()
            );
        }
        let (_, stack) = cube_with_a_stack();
        assert!(!depends_on_time(&stack));
        // And the test-only one does, or every test below is vacuous.
        let (_, drifting) = drifting();
        assert!(depends_on_time(&drifting));
    }

    #[test]
    fn a_still_scene_does_not_re_evaluate_when_the_frame_changes() {
        // The half of §2.4's memory decision that costs nothing and saves
        // everything: a stack that does not read the clock is valid at every
        // time, so scrubbing a model with no animation in it is free.
        //
        // Without this, moving the playhead one frame would re-subdivide every
        // object in the scene — and it would look like the timeline being slow
        // rather than like a cache key being too coarse.
        let (cube, stack) = cube_with_a_stack();
        let mut cache = ModifierCache::new();

        let first = cache.evaluate(&cube, &stack, &at(1.0));
        let (hits, misses) = cache.stats();
        assert_eq!((hits, misses), (0, 1));

        for frame in 2..=100 {
            let again = cache.evaluate(&cube, &stack, &at(frame as f64));
            assert!(
                Arc::ptr_eq(&again, &first),
                "frame {frame} got a different mesh"
            );
        }
        let (hits, misses) = cache.stats();
        assert_eq!(misses, 1, "the scene was re-evaluated {misses} times");
        assert_eq!(hits, 99);
        assert_eq!(cache.len(), 1, "one entry, not one per frame");
    }

    #[test]
    fn the_time_is_part_of_the_key_when_the_stack_reads_it() {
        // The other half: a stack that reads the clock gets a different answer
        // at a different instant, and the cache does not hand back the previous
        // frame's.
        let (cube, stack) = drifting();
        let mut cache = ModifierCache::new();

        let one = cache.evaluate(&cube, &stack, &at(24.0));
        let two = cache.evaluate(&cube, &stack, &at(48.0));
        assert!(
            (first_x(&two) - first_x(&one) - 1.0).abs() < 1e-5,
            "a second later should be one unit along: {} then {}",
            first_x(&one),
            first_x(&two)
        );

        // And asking for the first instant again is a fresh evaluation rather
        // than a hit, because its entry was replaced. Which is the trade: the
        // cache holds one frame, and scrubbing back costs what scrubbing
        // forward did.
        let (_, misses_before) = cache.stats();
        let again = cache.evaluate(&cube, &stack, &at(24.0));
        let (_, misses_after) = cache.stats();
        assert_eq!(misses_after, misses_before + 1);
        assert!((first_x(&again) - first_x(&one)).abs() < 1e-6);
    }

    #[test]
    fn scrubbing_an_animated_scene_does_not_accumulate_a_mesh_per_frame() {
        // §2.4's memory decision, and the reason it is a decision rather than
        // an oversight: a cache keyed on time with nothing dropping the old
        // entries holds a copy of the mesh per frame, so scrubbing a hundred
        // frames of a subdivided model is a hundred subdivided models.
        let (cube, stack) = drifting();
        let mut cache = ModifierCache::new();

        for frame in 1..=100 {
            cache.evaluate(&cube, &stack, &at(frame as f64));
            assert_eq!(
                cache.len(),
                1,
                "after frame {frame} the cache holds {} entries",
                cache.len()
            );
        }
        let (hits, misses) = cache.stats();
        assert_eq!(misses, 100, "every frame is a fresh evaluation");
        assert_eq!(hits, 0);
    }

    #[test]
    fn two_frames_of_the_same_instant_still_hit() {
        // The cache is not useless for an animated scene — it holds *one*
        // instant, and a frame is drawn more than once. `evaluate_scene` is
        // called twice a frame on purpose (the auto-depth pivot needs the
        // scene before the frame is encoded), so a time-keyed entry that
        // missed on the second call would double every animated frame's cost.
        let (cube, stack) = drifting();
        let mut cache = ModifierCache::new();
        let first = cache.evaluate(&cube, &stack, &at(12.0));
        let second = cache.evaluate(&cube, &stack, &at(12.0));
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(cache.stats(), (1, 1));
    }

    #[test]
    fn a_still_stack_and_an_animated_one_coexist() {
        // Two objects, one animated. Moving the clock must re-evaluate the
        // animated one and leave the other's entry alone — a cache that cleared
        // itself on a time change would be correct and would re-subdivide the
        // static half of the scene every frame.
        let (cube, still) = cube_with_a_stack();
        let (_, animated) = drifting();
        let mut cache = ModifierCache::new();

        let held = cache.evaluate(&cube, &still, &at(1.0));
        cache.evaluate(&cube, &animated, &at(1.0));
        assert_eq!(cache.len(), 2);

        cache.evaluate(&cube, &animated, &at(2.0));
        let again = cache.evaluate(&cube, &still, &at(2.0));
        assert!(
            Arc::ptr_eq(&again, &held),
            "the still stack was re-evaluated when the clock moved"
        );
        assert_eq!(cache.len(), 2, "one entry each, still");
    }

    #[test]
    fn a_disabled_modifier_cannot_make_a_stack_read_the_clock() {
        // It is not applied, so it cannot contribute — the same rule
        // `is_identity` follows. A stack whose only time-dependent modifier is
        // switched off would otherwise be re-evaluated every frame for nothing.
        let (cube, stack) = drifting();
        let mut off = stack.clone();
        off[0].enabled = false;
        assert!(!depends_on_time(&off));
        assert!(is_identity(&off));

        let mut cache = ModifierCache::new();
        let one = cache.evaluate(&cube, &off, &at(1.0));
        let two = cache.evaluate(&cube, &off, &at(500.0));
        assert!(Arc::ptr_eq(&one, &two));
        assert!(
            Arc::ptr_eq(&one, &cube),
            "a switched-off stack is the input"
        );
    }

    #[test]
    fn the_rate_is_part_of_the_instant() {
        // Frame 24 at 24 fps and frame 24 at 48 fps are different times, and a
        // modifier reading seconds gets different answers. A key comparing only
        // the frame number would serve one for the other, and the symptom would
        // be an animation that stops responding to a rate change.
        let (cube, stack) = drifting();
        let mut cache = ModifierCache::new();
        let slow = cache.evaluate(
            &cube,
            &stack,
            &EvalCtx::at(SceneTime::new(24.0, 24.0).expect("valid")),
        );
        let slow_x = first_x(&slow);
        let fast = cache.evaluate(
            &cube,
            &stack,
            &EvalCtx::at(SceneTime::new(24.0, 48.0).expect("valid")),
        );
        assert!(
            (first_x(&fast) - slow_x).abs() > 0.4,
            "the same frame at another rate gave the same answer: {slow_x} and {}",
            first_x(&fast)
        );
    }

    #[test]
    fn the_context_is_a_struct_so_that_the_next_field_costs_nobody_anything() {
        // Why `EvalCtx` and not `time: SceneTime` as a parameter. §5's fourth
        // risk is about the *signature*: the day evaluation also needs a
        // dependency graph's view, or a seed, or a resolution the viewport
        // wants and the renderer does not, it goes in here and no caller
        // changes.
        let ctx = EvalCtx::still();
        assert_eq!(ctx.time, SceneTime::default());
        assert_eq!(EvalCtx::at(SceneTime::default()), ctx);

        let moved = EvalCtx::at(SceneTime::new(48.0, 24.0).expect("valid"));
        assert_ne!(moved, ctx);
        assert!((moved.time.seconds() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn evaluating_a_scene_passes_the_time_through() {
        // End to end, so the parameter is not merely present but arriving.
        let mut doc = Document::new();
        let mesh = doc.add_mesh(primitives::cube(1.0));
        let id = doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));
        doc.object_mut(id)
            .expect("object")
            .modifiers
            .push(ModifierInstance::new(ModifierKind::Subdivide(
                SubdivideParams::default(),
            )));

        let mut cache = ModifierCache::new();
        let early = cache.evaluate_scene(&doc, &at(1.0));
        let late = cache.evaluate_scene(&doc, &at(240.0));
        assert_eq!(early.len(), 1);
        assert_eq!(late.len(), 1);
        // The same mesh, because nothing in the stack reads the clock.
        assert!(Arc::ptr_eq(&early[0].1, &late[0].1));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn a_boolean_follows_its_cutter_and_is_cached_while_it_stays() {
        use kreate_mesh::modifiers::BooleanOp;
        let mut doc = Document::new();
        let target = doc.add_mesh_object("Target", kreate_mesh::primitives::cube(2.0));
        let cutter = doc.add_mesh_object("Cutter", kreate_mesh::primitives::cube(1.0));
        doc.object_mut(cutter).unwrap().transform.translation =
            kreate_math::DVec3::new(1.0, 0.0, 0.0);
        let operand = doc.object_uid(cutter).unwrap();
        doc.object_mut(target)
            .unwrap()
            .modifiers
            .push(ModifierInstance::new(ModifierKind::Boolean(
                BooleanModifier {
                    operand,
                    op: BooleanOp::Difference,
                },
            )));
        let mut cache = ModifierCache::new();
        let ctx = EvalCtx::still();
        let a = cache.evaluated(&doc, target, &ctx).unwrap();
        assert!(a.num_faces() > 6, "nothing was cut");
        let b = cache.evaluated(&doc, target, &ctx).unwrap();
        assert!(Arc::ptr_eq(&a, &b), "an unchanged cutter evaluated again");
        // The cutter moves away: a whole cube again.
        doc.object_mut(cutter).unwrap().transform.translation =
            kreate_math::DVec3::new(9.0, 0.0, 0.0);
        let c = cache.evaluated(&doc, target, &ctx).unwrap();
        assert!(!Arc::ptr_eq(&a, &c), "the moved cutter was not noticed");
        assert!((c.bounds().max.x - 1.0).abs() < 1e-5);
        // No cutter picked: the input as it was.
        doc.object_mut(target).unwrap().modifiers[0].kind =
            ModifierKind::Boolean(BooleanModifier {
                operand: kreate_reflect::Uid::NONE,
                op: BooleanOp::Difference,
            });
        assert_eq!(cache.evaluated(&doc, target, &ctx).unwrap().num_faces(), 6);
    }

    #[test]
    fn a_cutter_is_named_while_an_enabled_boolean_uses_it() {
        use kreate_mesh::modifiers::BooleanOp;
        let mut doc = Document::new();
        let target = doc.add_mesh_object("Target", kreate_mesh::primitives::cube(2.0));
        let cutter = doc.add_mesh_object("Cutter", kreate_mesh::primitives::cube(1.0));
        assert!(boolean_cutters(&doc).is_empty());
        let operand = doc.object_uid(cutter).unwrap();
        doc.object_mut(target)
            .unwrap()
            .modifiers
            .push(ModifierInstance::new(ModifierKind::Boolean(
                BooleanModifier {
                    operand,
                    op: BooleanOp::Difference,
                },
            )));
        assert_eq!(boolean_cutters(&doc), vec![cutter]);
        // Off: the cutter is an ordinary object again.
        doc.object_mut(target).unwrap().modifiers[0].enabled = false;
        assert!(boolean_cutters(&doc).is_empty());
        // On, but on a hidden target: nothing is being cut on screen.
        doc.object_mut(target).unwrap().modifiers[0].enabled = true;
        doc.object_mut(target).unwrap().visible = false;
        assert!(boolean_cutters(&doc).is_empty());
    }
}
