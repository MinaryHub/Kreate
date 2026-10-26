//! A drawing: what an object draws in its own plane — curves, points,
//! text and dimensions, on layers, held by constraints. The 2D drafting
//! mode's document, and what a sketch becomes. See design/2d-drafting.md.
//!
//! # Where it lives
//!
//! On its object, beside the sketch and the features it replaces, rather
//! than as a datablock of its own: the plane is the object's local XY, so a
//! drawing on a part's face moves with the part, and a drawing for paper is
//! one object in an otherwise empty scene. What drawings share — blocks,
//! linetypes, text and dimension styles — is the document's.
//!
//! # Identity
//!
//! Every entity has an [`EntityId`] that never changes and is never reused
//! in its drawing. Constraints, dimensions, a feature's axis, a parameter's
//! binding and the connector all say which entity they mean by it. A
//! sketch named its points by their place in a list, which shifted under
//! every deletion; an id does not.

use kreate_geom2d::{Curve2D, DVec2};

/// An entity's identity within its drawing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct EntityId(pub u32);

/// A layer's identity within its drawing. Layer zero, "0", always exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct LayerId(pub u32);

/// A colour as drafting gives one: the layer's, the block's, an AutoCAD
/// colour index, or red-green-blue.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ColorRef {
    #[default]
    ByLayer,
    ByBlock,
    Index(u8),
    Rgb([u8; 3]),
}

/// A linetype: the layer's, the block's, or one of the document's by name.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum LinetypeRef {
    #[default]
    ByLayer,
    ByBlock,
    Named(String),
}

/// A line's weight: the layer's, the block's, or millimetres on paper.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Lineweight {
    #[default]
    ByLayer,
    ByBlock,
    Mm(f32),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub id: LayerId,
    pub name: String,
    pub color: ColorRef,
    /// The layer's linetype by name: `Continuous` for solid lines.
    pub linetype: String,
    pub lineweight: Lineweight,
    /// Shown. Off, its entities are not drawn but still regenerate.
    pub on: bool,
    /// Frozen: not drawn and not regenerated.
    pub frozen: bool,
    /// Locked: drawn, and snapped to, but not selectable or editable.
    pub locked: bool,
    /// Printed.
    pub plot: bool,
}

impl Layer {
    pub fn new(id: LayerId, name: &str) -> Layer {
        Layer {
            id,
            name: name.to_string(),
            color: ColorRef::Index(7),
            linetype: "Continuous".to_string(),
            lineweight: Lineweight::Mm(0.25),
            on: true,
            frozen: false,
            locked: false,
            plot: true,
        }
    }
}

/// How a line of text sits on its insertion point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TextAlign {
    #[default]
    Left,
    Center,
    Right,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Text {
    pub at: DVec2,
    /// Cap height, in the drawing's unit (metres).
    pub height: f64,
    /// Radians anticlockwise from the drawing's X.
    pub rotation: f64,
    pub content: String,
    pub align: TextAlign,
    /// A font by its name, or empty for the default.
    pub font: String,
}

/// Which part of an entity a constraint or dimension takes hold of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Part {
    /// The entity as a whole: a line as a line, a circle as a circle.
    Whole,
    Start,
    End,
    /// A circle's, an arc's or an ellipse's centre.
    Center,
    /// Half way along.
    Mid,
    /// A polyline's vertex, or a spline's fit point.
    Vertex(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EntityRef {
    pub entity: EntityId,
    pub part: Part,
}

impl EntityRef {
    pub fn whole(entity: EntityId) -> Self {
        Self {
            entity,
            part: Part::Whole,
        }
    }

    pub fn at(entity: EntityId, part: Part) -> Self {
        Self { entity, part }
    }
}

/// A geometric relation the solver keeps. Dimensions are entities, not
/// constraints: see [`Dimension`].
#[derive(Clone, Debug, PartialEq)]
pub enum Constraint {
    /// Two points one.
    Coincident(EntityRef, EntityRef),
    /// A line level, or two points level with each other.
    Horizontal(EntityRef, Option<EntityRef>),
    /// A line upright, or two points one above the other.
    Vertical(EntityRef, Option<EntityRef>),
    Parallel(EntityId, EntityId),
    Perpendicular(EntityId, EntityId),
    /// Two curves touching without crossing at the point they share.
    Tangent(EntityId, EntityId),
    /// Two lines of one length, or two circles or arcs of one radius.
    Equal(EntityId, EntityId),
    Concentric(EntityId, EntityId),
    /// A point on a curve (or on its extension, for a line).
    PointOn(EntityRef, EntityId),
    /// Two points mirrored across a line.
    Symmetric(EntityRef, EntityRef, EntityId),
    /// A point at a line's middle.
    Midpoint(EntityRef, EntityId),
    /// A point held where it is.
    Fixed(EntityRef, DVec2),
    /// Two curves meeting with one curvature: smooth to the eye.
    Smooth(EntityId, EntityId),
}

impl Constraint {
    /// Every entity it names.
    pub fn entities(&self) -> Vec<EntityId> {
        use Constraint::*;
        match self {
            Coincident(a, b) => vec![a.entity, b.entity],
            Horizontal(a, b) | Vertical(a, b) => {
                let mut v = vec![a.entity];
                v.extend(b.map(|b| b.entity));
                v
            }
            Parallel(a, b)
            | Perpendicular(a, b)
            | Tangent(a, b)
            | Equal(a, b)
            | Concentric(a, b)
            | Smooth(a, b) => vec![*a, *b],
            PointOn(p, c) | Midpoint(p, c) => vec![p.entity, *c],
            Symmetric(a, b, l) => vec![a.entity, b.entity, *l],
            Fixed(a, _) => vec![a.entity],
        }
    }
}

/// What a dimension measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DimKind {
    /// Along X between two points.
    Horizontal,
    /// Along Y between two points.
    Vertical,
    /// Straight between two points, or a line's length.
    Aligned,
    /// Between two lines, at their crossing.
    Angular,
    Radius,
    Diameter,
    ArcLength,
    /// A point's X or Y from an origin point.
    OrdinateX,
    OrdinateY,
}

/// A tolerance written after a dimension's value.
#[derive(Clone, Debug, PartialEq)]
pub enum Tolerance {
    /// ± the same either way.
    Symmetric(f64),
    /// Above and below, each its own.
    Deviation { upper: f64, lower: f64 },
    /// The largest and smallest allowed, written instead of the value.
    Limits { upper: f64, lower: f64 },
    /// An ISO fit: `H7`, `g6`.
    Fit(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Dimension {
    pub kind: DimKind,
    /// What it measures: two points, a line, a circle, two lines.
    pub refs: Vec<EntityRef>,
    /// Where its text was put.
    pub text_at: DVec2,
    /// A style of the document's by name; empty for the default.
    pub style: String,
    /// Driving: the value holds the geometry to it. Driven: the value is
    /// measured from the geometry and shown.
    pub driving: bool,
    /// For a driving dimension what the geometry is held to; for a driven
    /// one what it last measured. Metres, or radians for an angle.
    pub value: f64,
    /// Text in place of the measured value, `<>` standing for it:
    /// `<> TYP`, `4× Ø<>`.
    pub text: Option<String>,
    pub tolerance: Option<Tolerance>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum EntityKind {
    Point(DVec2),
    Curve(Curve2D),
    Text(Text),
    Dimension(Dimension),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Entity {
    pub id: EntityId,
    pub kind: EntityKind,
    pub layer: LayerId,
    pub color: ColorRef,
    pub linetype: LinetypeRef,
    pub lineweight: Lineweight,
    /// A construction entity guides the drawing — a centre line, a
    /// reference circle — and bounds no region and is not printed.
    pub construction: bool,
}

/// What an object draws in its own plane. See the module notes.
#[derive(Clone, Debug, PartialEq)]
pub struct Drawing {
    /// In the order they are drawn: later ones over earlier.
    pub entities: Vec<Entity>,
    pub layers: Vec<Layer>,
    pub constraints: Vec<Constraint>,
    /// The layer new entities go on.
    pub current_layer: LayerId,
    next_entity: u32,
    next_layer: u32,
}

impl Default for Drawing {
    fn default() -> Self {
        Drawing {
            entities: Vec::new(),
            layers: vec![Layer::new(LayerId(0), "0")],
            constraints: Vec::new(),
            current_layer: LayerId(0),
            next_entity: 1,
            next_layer: 1,
        }
    }
}

impl Drawing {
    pub fn new() -> Self {
        Self::default()
    }

    /// The counters as a file holds them, for saving and loading: the next
    /// entity id and the next layer id to be issued.
    pub fn counters(&self) -> (u32, u32) {
        (self.next_entity, self.next_layer)
    }

    /// Puts back what [`Drawing::counters`] gave, never below what the
    /// drawing already holds — a file whose counter is behind its own ids
    /// would otherwise hand an id out twice.
    pub fn set_counters(&mut self, entity: u32, layer: u32) {
        let top_entity = self.entities.iter().map(|e| e.id.0 + 1).max().unwrap_or(1);
        let top_layer = self.layers.iter().map(|l| l.id.0 + 1).max().unwrap_or(1);
        self.next_entity = entity.max(top_entity);
        self.next_layer = layer.max(top_layer);
    }

    /// Adds an entity on the current layer, its colour, linetype and weight
    /// the layer's. Its id.
    pub fn add(&mut self, kind: EntityKind) -> EntityId {
        let id = EntityId(self.next_entity);
        self.next_entity += 1;
        self.entities.push(Entity {
            id,
            kind,
            layer: self.current_layer,
            color: ColorRef::ByLayer,
            linetype: LinetypeRef::ByLayer,
            lineweight: Lineweight::ByLayer,
            construction: false,
        });
        id
    }

    pub fn add_curve(&mut self, curve: Curve2D) -> EntityId {
        self.add(EntityKind::Curve(curve))
    }

    pub fn add_layer(&mut self, name: &str) -> LayerId {
        let id = LayerId(self.next_layer);
        self.next_layer += 1;
        self.layers.push(Layer::new(id, name));
        id
    }

    pub fn layer(&self, id: LayerId) -> Option<&Layer> {
        self.layers.iter().find(|l| l.id == id)
    }

    pub fn entity(&self, id: EntityId) -> Option<&Entity> {
        self.entities.iter().find(|e| e.id == id)
    }

    pub fn entity_mut(&mut self, id: EntityId) -> Option<&mut Entity> {
        self.entities.iter_mut().find(|e| e.id == id)
    }

    pub fn curve(&self, id: EntityId) -> Option<&Curve2D> {
        match &self.entity(id)?.kind {
            EntityKind::Curve(c) => Some(c),
            _ => None,
        }
    }

    /// Removes entities, and every constraint and dimension that named
    /// one of them: a relation with nothing at one end means nothing.
    pub fn remove(&mut self, ids: &[EntityId]) {
        self.entities.retain(|e| !ids.contains(&e.id));
        self.constraints
            .retain(|c| c.entities().iter().all(|e| !ids.contains(e)));
        loop {
            let gone: Vec<EntityId> = self
                .entities
                .iter()
                .filter(|e| match &e.kind {
                    EntityKind::Dimension(d) => {
                        d.refs.iter().any(|r| self.entity(r.entity).is_none())
                    }
                    _ => false,
                })
                .map(|e| e.id)
                .collect();
            if gone.is_empty() {
                break;
            }
            self.entities.retain(|e| !gone.contains(&e.id));
        }
    }

    /// Where an entity's part is, if it has that part.
    pub fn point_of(&self, r: EntityRef) -> Option<DVec2> {
        let e = self.entity(r.entity)?;
        match (&e.kind, r.part) {
            (EntityKind::Point(p), Part::Whole | Part::Start | Part::End | Part::Center) => {
                Some(*p)
            }
            (EntityKind::Text(t), Part::Whole | Part::Start) => Some(t.at),
            (EntityKind::Curve(c), part) => match part {
                Part::Start => Some(c.start()),
                Part::End => Some(c.end()),
                Part::Mid => {
                    let (lo, hi) = c.domain();
                    Some(c.point((lo + hi) / 2.0))
                }
                Part::Center => match c {
                    Curve2D::Circle { c, .. }
                    | Curve2D::Arc { c, .. }
                    | Curve2D::Ellipse { c, .. } => Some(*c),
                    _ => None,
                },
                Part::Vertex(i) => match c {
                    Curve2D::Polyline { verts, .. } => verts.get(i as usize).map(|v| v.0),
                    Curve2D::Spline(s) => s.fit.get(i as usize).copied(),
                    _ => None,
                },
                Part::Whole => None,
            },
            _ => None,
        }
    }

    /// The curves that bound regions: every curve not for construction.
    /// With each, the entity it is.
    pub fn profile_curves(&self) -> Vec<(EntityId, &Curve2D)> {
        self.entities
            .iter()
            .filter(|e| !e.construction)
            .filter_map(|e| match &e.kind {
                EntityKind::Curve(c) => Some((e.id, c)),
                _ => None,
            })
            .collect()
    }

    /// The closed regions its curves make. A boundary piece's `source` is
    /// the place in [`Drawing::profile_curves`] it came from.
    pub fn regions(&self, tol: f64) -> Vec<kreate_geom2d::Region> {
        let curves: Vec<Curve2D> = self
            .profile_curves()
            .into_iter()
            .map(|(_, c)| c.clone())
            .collect();
        kreate_geom2d::regions(&curves, tol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64, y: f64) -> DVec2 {
        DVec2::new(x, y)
    }

    #[test]
    fn ids_are_never_reused() {
        let mut d = Drawing::new();
        let a = d.add_curve(Curve2D::Line {
            a: v(0.0, 0.0),
            b: v(1.0, 0.0),
        });
        let b = d.add(EntityKind::Point(v(2.0, 2.0)));
        d.remove(&[b]);
        let c = d.add(EntityKind::Point(v(3.0, 3.0)));
        assert_ne!(b, c, "a removed id is not handed out again");
        assert!(c > a);
        // Nor after a round trip through the counters.
        let (ne, nl) = d.counters();
        let mut back = d.clone();
        back.set_counters(0, 0);
        assert_eq!(back.counters(), (ne, nl), "never below what is there");
    }

    /// Removing an entity takes the relations on it, and the dimensions
    /// that measured it.
    #[test]
    fn what_named_a_removed_entity_goes_with_it() {
        let mut d = Drawing::new();
        let a = d.add_curve(Curve2D::Line {
            a: v(0.0, 0.0),
            b: v(1.0, 0.0),
        });
        let b = d.add_curve(Curve2D::Line {
            a: v(1.0, 0.0),
            b: v(1.0, 1.0),
        });
        d.constraints.push(Constraint::Perpendicular(a, b));
        d.constraints
            .push(Constraint::Horizontal(EntityRef::whole(a), None));
        let dim = d.add(EntityKind::Dimension(Dimension {
            kind: DimKind::Aligned,
            refs: vec![EntityRef::whole(b)],
            text_at: v(1.2, 0.5),
            style: String::new(),
            driving: true,
            value: 1.0,
            text: None,
            tolerance: None,
        }));
        d.remove(&[b]);
        assert_eq!(
            d.constraints,
            vec![Constraint::Horizontal(EntityRef::whole(a), None)]
        );
        assert!(d.entity(dim).is_none(), "its dimension went too");
    }

    #[test]
    fn parts_are_where_they_should_be() {
        let mut d = Drawing::new();
        let arc = d.add_curve(Curve2D::Arc {
            c: v(1.0, 1.0),
            r: 1.0,
            start: 0.0,
            sweep: std::f64::consts::PI,
        });
        let at = |p| d.point_of(EntityRef::at(arc, p)).unwrap();
        assert!(at(Part::Center).distance(v(1.0, 1.0)) < 1e-12);
        assert!(at(Part::Start).distance(v(2.0, 1.0)) < 1e-12);
        assert!(at(Part::End).distance(v(0.0, 1.0)) < 1e-12);
        assert!(at(Part::Mid).distance(v(1.0, 2.0)) < 1e-12);
    }

    /// Construction lines bound nothing.
    #[test]
    fn construction_curves_make_no_regions() {
        let mut d = Drawing::new();
        d.add_curve(Curve2D::Circle {
            c: v(0.0, 0.0),
            r: 1.0,
        });
        let centre_line = d.add_curve(Curve2D::Line {
            a: v(-2.0, 0.0),
            b: v(2.0, 0.0),
        });
        assert_eq!(d.regions(1e-6).len(), 2, "the line cuts the disc in two");
        d.entity_mut(centre_line).unwrap().construction = true;
        assert_eq!(d.regions(1e-6).len(), 1, "a construction line does not");
    }
}
