//! The built-in operators.
//!
//! Each module registers its own operators, and [`register_builtin`] is the one
//! call an application makes. An add-on registers into the same registry with the
//! same trait, so a third-party operator is indistinguishable from a built-in one
//! — which is the point of the design rather than a nice side effect.

pub mod object;

/// Registers every built-in operator.
pub fn register_builtin(registry: &mut crate::operator::OperatorRegistry) {
    object::register(registry);
}
