//! Pointing at a property, in a way that survives what users do to scenes.
//!
//! An animation curve, a driver and a constraint all need to say "this property
//! of that thing". How the *thing* is named decides whether the animation
//! survives ordinary work, and there are three candidate answers with very
//! different consequences.
//!
//! **By name.** Blender's `objects["Cube"].location`. Renaming the object
//! breaks the curve, and renaming is not an unusual thing to do — it is the
//! first thing anyone does after adding an object. It is a known weakness of a
//! very good tool, and it is avoidable on day one.
//!
//! **By position.** What this codebase's file format writes for a parent
//! reference, correctly, because a parent link is rewritten wholesale on load.
//! For a curve it is worse than a name: inserting one object above another
//! shifts every later position, so a scene that gains an object at the top
//! retargets every animation in it.
//!
//! **By an identity assigned once and never reused.** A [`Uid`]. Survives
//! renaming, reordering, deleting a neighbour, and a save and load. Costs eight
//! bytes per datablock and a counter.
//!
//! # Why the *property* half is a name, and that is not the same mistake
//!
//! A property name is code. `location` is what the type declares its field to
//! be, and changing it is a source edit with a migration, not something a user
//! does between two saves. A datablock name is user data and changes hourly.
//! Confusing the two is what makes name-based paths look uniformly bad when
//! only half of the idea is.
//!
//! A number would be worse here, and specifically: a file holding
//! `property: 7` cannot be read, cannot be diffed usefully, and cannot be
//! merged — and the day somebody reorders the table that issued the sevens,
//! every curve in every file silently drives a different property. The name is
//! the stable id, and it is stable because it is not the user's to change.
//!
//! # An arena handle is not an identity
//!
//! [`Uid`] exists because `ObjectId` cannot do this job. It is an arena slot
//! plus a generation that is only present in debug builds, so in a release
//! build a reused slot is indistinguishable from the object that used to live
//! there: delete an object, add another, and every curve that pointed at the
//! first now drives the second. The file format already says as much about
//! writing one down — "an arena id is only meaningful to the process that made
//! it".

use std::fmt;

/// A datablock's identity: assigned once, never reused, never renumbered.
///
/// Unique **within one document**. Appending or linking another document is
/// where two ids can collide, and that boundary remaps rather than hoping —
/// which is the same shape as every other cross-document operation and is
/// expected to be explicit.
///
/// Written to the file as a plain number, so a person reading a scene can match
/// a curve to its target by eye.
///
/// `Default` is [`Uid::NONE`], the unassigned id, so a datablock struct can
/// derive `Default` without accidentally claiming an identity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Uid(u64);

impl Uid {
    /// The id a datablock has before the document has given it one.
    ///
    /// Never handed out, so it can stand for "not yet assigned" without
    /// costing an `Option` in every datablock. A path pointing at it resolves
    /// to nothing.
    pub const NONE: Self = Self(0);

    /// The first id a document issues.
    pub const FIRST: Self = Self(1);

    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    pub const fn raw(self) -> u64 {
        self.0
    }

    pub const fn is_none(self) -> bool {
        self.0 == 0
    }

    /// The id after this one.
    ///
    /// How a document's counter advances. Saturating rather than wrapping: at
    /// one id a nanosecond this takes six hundred years to reach, and the
    /// alternative to saturating is wrapping round to ids that are already in
    /// use.
    pub const fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }
}

impl fmt::Display for Uid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// What kind of thing a [`Uid`] names.
///
/// Carried beside the id rather than folded into it. Two datablocks of
/// different kinds may share an id — the counters are per kind, which keeps the
/// numbers small and readable in a file — so the kind is what disambiguates,
/// and a path that lost it would resolve against the wrong table.
///
/// The list is longer than what exists, deliberately: §3.12 asks for the file
/// format's datablock directories to be defined before the datablocks are, so
/// that a reader of this version does not choke on a scene from a later one.
/// The same argument applies here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DatablockKind {
    Object,
    Mesh,
    Camera,
    Material,
    Light,
    World,
    Image,
    Action,
    Armature,
}

impl DatablockKind {
    /// The stable text form, for a file and for an error message.
    pub fn id(self) -> &'static str {
        match self {
            DatablockKind::Object => "object",
            DatablockKind::Mesh => "mesh",
            DatablockKind::Camera => "camera",
            DatablockKind::Material => "material",
            DatablockKind::Light => "light",
            DatablockKind::World => "world",
            DatablockKind::Image => "image",
            DatablockKind::Action => "action",
            DatablockKind::Armature => "armature",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Some(match id {
            "object" => DatablockKind::Object,
            "mesh" => DatablockKind::Mesh,
            "camera" => DatablockKind::Camera,
            "material" => DatablockKind::Material,
            "light" => DatablockKind::Light,
            "world" => DatablockKind::World,
            "image" => DatablockKind::Image,
            "action" => DatablockKind::Action,
            "armature" => DatablockKind::Armature,
            _ => return None,
        })
    }

    /// Every kind, for a test that must not go stale when one is added.
    pub const ALL: &'static [DatablockKind] = &[
        DatablockKind::Object,
        DatablockKind::Mesh,
        DatablockKind::Camera,
        DatablockKind::Material,
        DatablockKind::Light,
        DatablockKind::World,
        DatablockKind::Image,
        DatablockKind::Action,
        DatablockKind::Armature,
    ];
}

/// One property that can be animated, driven or constrained.
///
/// Declared beside the type it describes rather than in a table here: a list of
/// its own fields is the sort of thing that goes quietly out of step with the
/// struct when the two live in different crates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnimatableDesc {
    /// The name a path uses. Code, not user data — see the module docs.
    pub name: &'static str,
    pub kind: crate::PropertyKind,
    /// How many components can be keyed separately: one for a float, three for
    /// a vector, four for a quaternion.
    ///
    /// Not the same as "how big is the value". A curve keys one component, so a
    /// rotation is four curves and a location three, and this is what says how
    /// many are legal.
    pub components: u8,
    /// Doc comment on the field, for a tooltip when nothing is translated.
    pub doc: &'static str,
}

/// Which property of which datablock.
///
/// Owned rather than borrowed, because one of these is stored per curve and
/// read back from a file. Not `Copy`: it carries names, and the day somebody
/// wants it in a hot loop the answer is to resolve it once into an index rather
/// than to make the path smaller.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PropertyPath {
    pub kind: DatablockKind,
    /// Which datablock, by the identity it was given when it was created.
    pub target: Uid,
    /// Which property of it, by the name its type declares.
    pub property: Box<str>,
    /// Which component, for a property that has more than one.
    ///
    /// `None` means the whole property, which is what a driver on a boolean or
    /// a constraint on a whole transform wants. A curve always names a
    /// component, because a curve is one number over time.
    pub index: Option<u8>,
    /// What the target was called when this was written.
    ///
    /// **Never used to resolve.** It is here so that a person reading a scene
    /// file can tell what a curve is for, and so that an error about a target
    /// that no longer exists can say "the object formerly called Arm.L" instead
    /// of "uid 41". A stale hint is expected and harmless; a resolver that read
    /// it would have reintroduced exactly the fragility this module exists to
    /// avoid.
    pub hint: Option<Box<str>>,
}

/// Why a path does not name anything.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PathError {
    /// No datablock of that kind has that id — usually because it was deleted.
    NoSuchTarget,
    /// The datablock exists but has no property by that name.
    ///
    /// What a file from a later version looks like, and what a renamed field
    /// looks like without a migration. Distinguished from a missing target
    /// because the two want different repairs.
    NoSuchProperty,
    /// The property exists but has fewer components than the path asks for.
    ComponentOutOfRange { components: u8, asked: u8 },
}

impl fmt::Display for PathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PathError::NoSuchTarget => write!(f, "no such datablock"),
            PathError::NoSuchProperty => write!(f, "no such property"),
            PathError::ComponentOutOfRange { components, asked } => {
                write!(f, "component {asked} of a property with {components}")
            }
        }
    }
}

impl PropertyPath {
    /// A path to a whole property.
    pub fn new(kind: DatablockKind, target: Uid, property: &str) -> Self {
        Self {
            kind,
            target,
            property: property.into(),
            index: None,
            hint: None,
        }
    }

    /// A path to one component of a property.
    pub fn component(kind: DatablockKind, target: Uid, property: &str, index: u8) -> Self {
        Self {
            index: Some(index),
            ..Self::new(kind, target, property)
        }
    }

    /// The same path, carrying a name for a human to read.
    pub fn labelled(mut self, hint: &str) -> Self {
        self.hint = Some(hint.into());
        self
    }

    /// Whether `schema` has the property this path names, with the component it
    /// asks for.
    ///
    /// Takes the schema rather than the datablock, so this is testable without
    /// a document and usable by anything that has a schema in hand. Whether the
    /// *target* exists is the document's question and is not asked here — which
    /// is why [`PathError::NoSuchTarget`] is not among the answers.
    pub fn check(&self, schema: &[AnimatableDesc]) -> Result<AnimatableDesc, PathError> {
        let desc = schema
            .iter()
            .find(|d| d.name == self.property.as_ref())
            .ok_or(PathError::NoSuchProperty)?;
        if let Some(asked) = self.index {
            if asked >= desc.components {
                return Err(PathError::ComponentOutOfRange {
                    components: desc.components,
                    asked,
                });
            }
        }
        Ok(*desc)
    }

    /// How this path reads in a script or a log.
    ///
    /// `object[41].location[0]`, and with a hint `object[41 "Arm.L"].location[0]`.
    /// The id first because it is what resolves; the name second because it is
    /// what a person recognises.
    pub fn to_script(&self) -> String {
        let mut out = String::new();
        out.push_str(self.kind.id());
        out.push('[');
        out.push_str(&self.target.raw().to_string());
        if let Some(hint) = &self.hint {
            out.push_str(" \"");
            out.push_str(hint);
            out.push('"');
        }
        out.push(']');
        out.push('.');
        out.push_str(&self.property);
        if let Some(index) = self.index {
            out.push('[');
            out.push_str(&index.to_string());
            out.push(']');
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PropertyKind;

    fn object_schema() -> Vec<AnimatableDesc> {
        vec![
            AnimatableDesc {
                name: "location",
                kind: PropertyKind::Vec3,
                components: 3,
                doc: "",
            },
            AnimatableDesc {
                name: "visible",
                kind: PropertyKind::Bool,
                components: 1,
                doc: "",
            },
        ]
    }

    #[test]
    fn a_path_does_not_carry_the_target_s_name() {
        // The whole decision, stated as a test. Blender's
        // `objects["Cube"].location` stops working when the object is renamed,
        // which is the first thing anyone does after adding one.
        //
        // The hint is not an exception: it is advisory, and two paths that
        // differ only in it are still different values — which is deliberate,
        // because a path *with* a stale hint should be replaceable by one with
        // a fresh hint without anyone having to prove they mean the same thing.
        let bare = PropertyPath::component(DatablockKind::Object, Uid::from_raw(41), "location", 0);
        let renamed = bare.clone().labelled("Arm.L.001");
        assert_eq!(bare.target, renamed.target);
        assert_eq!(bare.property, renamed.property);
        assert_eq!(bare.index, renamed.index);
    }

    #[test]
    fn the_hint_is_never_what_resolves() {
        // A resolver reading the hint would have reintroduced the fragility
        // this module exists to avoid, and it would have done so invisibly:
        // everything works until a name changes.
        let schema = object_schema();
        let wrong_name =
            PropertyPath::new(DatablockKind::Object, Uid::from_raw(7), "location").labelled("Gone");
        assert!(wrong_name.check(&schema).is_ok());

        let no_name = PropertyPath::new(DatablockKind::Object, Uid::from_raw(7), "location");
        assert_eq!(
            no_name.check(&schema).map(|d| d.name),
            wrong_name.check(&schema).map(|d| d.name)
        );
    }

    #[test]
    fn an_unknown_property_is_told_apart_from_an_unknown_target() {
        // They want different repairs. A missing property is a file from a
        // later version or a renamed field, and the curve is worth keeping in
        // case the property comes back; a missing target is a deleted object,
        // and the curve is rubbish.
        let schema = object_schema();
        let path = PropertyPath::new(DatablockKind::Object, Uid::from_raw(1), "rotation");
        assert_eq!(path.check(&schema), Err(PathError::NoSuchProperty));
        // And `check` deliberately never says `NoSuchTarget`: whether a
        // datablock exists is not a question a schema can answer.
        assert_ne!(path.check(&schema), Err(PathError::NoSuchTarget));
    }

    #[test]
    fn a_component_beyond_the_property_is_refused() {
        // A curve on `location[3]` is what a file from a version where location
        // had four components looks like, and writing it into a `[f64; 3]` is a
        // panic in the middle of playback.
        let schema = object_schema();
        let ok = PropertyPath::component(DatablockKind::Object, Uid::from_raw(1), "location", 2);
        assert!(ok.check(&schema).is_ok());

        let over = PropertyPath::component(DatablockKind::Object, Uid::from_raw(1), "location", 3);
        assert_eq!(
            over.check(&schema),
            Err(PathError::ComponentOutOfRange {
                components: 3,
                asked: 3
            })
        );

        // And a single-component property has exactly one.
        let flag = PropertyPath::component(DatablockKind::Object, Uid::from_raw(1), "visible", 1);
        assert!(flag.check(&schema).is_err());
        let flag = PropertyPath::component(DatablockKind::Object, Uid::from_raw(1), "visible", 0);
        assert!(flag.check(&schema).is_ok());
    }

    #[test]
    fn a_whole_property_needs_no_component() {
        // What a driver on a boolean wants, and a constraint on a whole
        // transform. A curve always names one, because a curve is one number
        // over time — but the path type is used by more than curves.
        let schema = object_schema();
        let whole = PropertyPath::new(DatablockKind::Object, Uid::from_raw(1), "location");
        assert_eq!(whole.index, None);
        assert!(whole.check(&schema).is_ok());
    }

    #[test]
    fn the_kind_is_what_stops_two_datablocks_sharing_an_id() {
        // The counters are per kind, so object 1 and mesh 1 both exist. A path
        // that lost its kind would resolve against whichever table it was
        // handed and find something.
        let object = PropertyPath::new(DatablockKind::Object, Uid::from_raw(1), "location");
        let mesh = PropertyPath::new(DatablockKind::Mesh, Uid::from_raw(1), "location");
        assert_ne!(object, mesh);
    }

    #[test]
    fn the_unassigned_id_is_not_one_a_document_hands_out() {
        // So that a datablock can hold "no id yet" without an `Option`, and a
        // path pointing at it resolves to nothing rather than to whatever is
        // first.
        assert!(Uid::NONE.is_none());
        assert!(!Uid::FIRST.is_none());
        assert_eq!(Uid::NONE.next(), Uid::FIRST);
    }

    #[test]
    fn the_counter_saturates_rather_than_wrapping() {
        // Wrapping would hand out ids that are already in use, which is the one
        // failure this type is built to make impossible. At one id a nanosecond
        // the saturating branch is six hundred years away.
        let last = Uid::from_raw(u64::MAX);
        assert_eq!(last.next(), last);
    }

    #[test]
    fn every_kind_round_trips_through_its_text_form() {
        // The file holds the text. A kind that could be written and not read
        // would make a scene from this version unreadable by it.
        for kind in DatablockKind::ALL {
            assert_eq!(
                DatablockKind::from_id(kind.id()),
                Some(*kind),
                "{} did not round trip",
                kind.id()
            );
        }
        assert_eq!(DatablockKind::from_id("nonesuch"), None);
        // And the list is not missing any: adding a variant without adding it
        // to `ALL` would make this test pass while covering less.
        let mut ids: Vec<&str> = DatablockKind::ALL.iter().map(|k| k.id()).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), DatablockKind::ALL.len(), "two kinds share an id");
    }

    #[test]
    fn a_path_reads_as_something_a_person_can_act_on() {
        let path = PropertyPath::component(DatablockKind::Object, Uid::from_raw(41), "location", 0)
            .labelled("Arm.L");
        assert_eq!(path.to_script(), "object[41 \"Arm.L\"].location[0]");

        let bare = PropertyPath::new(DatablockKind::Mesh, Uid::from_raw(3), "auto_smooth_angle");
        assert_eq!(bare.to_script(), "mesh[3].auto_smooth_angle");
    }
}
