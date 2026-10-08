//! The built-in operators.
//!
//! Each module registers its own operators, and [`register_builtin`] is the one
//! call an application makes. An add-on registers into the same registry with the
//! same trait, so a third-party operator is indistinguishable from a built-in one
//! — which is the point of the design rather than a nice side effect.

pub mod anim;
pub mod armature;
pub mod collection;
pub mod design;
pub mod driver;
pub mod geo;
pub mod mesh;
pub mod modifier;
pub mod node;
pub mod object;
pub mod origin;
pub mod paint;
pub mod physics;
pub mod pose;
pub mod render;
pub mod sculpt;
pub mod shape;
pub mod sketch;
pub mod texture;
pub mod uv;

/// Registers every built-in operator.
pub fn register_builtin(registry: &mut crate::operator::OperatorRegistry) {
    anim::register(registry);
    armature::register(registry);
    collection::register(registry);
    driver::register(registry);
    geo::register(registry);
    mesh::register(registry);
    modifier::register(registry);
    node::register(registry);
    object::register(registry);
    origin::register(registry);
    design::register(registry);
    sketch::register(registry);
    render::register(registry);
    paint::register(registry);
    physics::register(registry);
    pose::register(registry);
    sculpt::register(registry);
    shape::register(registry);
    texture::register(registry);
    uv::register(registry);
}
