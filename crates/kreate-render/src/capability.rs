//! What the GPU can be relied on to do.
//!
//! The plan requires this decided before any large-scene work starts, and the
//! reason is a specific failure: prototyping GPU-driven rendering on one machine
//! with one backend, then discovering a hundred call sites assume a feature
//! another platform does not have. `MULTI_DRAW_INDIRECT_COUNT` is absent on
//! Metal; bindless is a backend-conditional native feature; mesh shaders are
//! experimental at best. Any of them, adopted casually, makes macOS a rewrite.
//!
//! So capability is a **tier**, not a bag of feature flags, and the tier the
//! architecture targets is fixed in advance:
//!
//! | Tier | Needs | Used for |
//! |------|-------|----------|
//! | [`Tier::Portable`] | WebGPU core | The floor. Everything must work here. |
//! | [`Tier::ComputeDriven`] | Compute + `draw_indirect` + storage in the vertex stage | **The target.** |
//! | [`Tier::Native`] | Vulkan/DX12 extras — timestamps, pipeline caches | Optional accelerators. |
//! | [`Tier::Experimental`] | Mesh shaders and the like | Not used. |
//!
//! # Why compute-driven is the target and not something higher
//!
//! Everything the large-scene design needs is in that tier and available on
//! Vulkan, Metal, DX12 *and* WebGPU: a compute pass to cull, one
//! `draw_indirect` per material batch, and storage buffers readable from a
//! vertex shader. Culling a whole scene into a single indirect draw is
//! expressible with exactly that much — see the cluster design in the plan — and
//! it gets most of what a mesh-shader path would, with no portability risk.
//!
//! Anything above the target is an accelerator: allowed to make things faster
//! where present, never allowed to be the only way something works. That rule is
//! what this module exists to make checkable.

/// What a device can be relied on to do.
///
/// Ordered, so `tier >= Tier::ComputeDriven` is the question a caller usually
/// wants to ask.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    /// WebGPU core only. The floor every path must work at.
    Portable,
    /// Compute shaders, `draw_indirect`, and storage buffers in the vertex
    /// stage. **What the architecture targets on every platform.**
    ComputeDriven,
    /// Native extras: GPU timestamps, on-disk pipeline caches. Accelerators and
    /// diagnostics, never the only route.
    Native,
    /// Mesh shaders and friends. Deliberately unused — see the module docs.
    Experimental,
}

impl Tier {
    /// The tier the architecture is designed against.
    ///
    /// Named rather than inferred so that "we require this" is a statement in
    /// the code and not a pattern someone has to notice across twenty files.
    pub const TARGET: Tier = Tier::ComputeDriven;

    pub fn name(self) -> &'static str {
        match self {
            Tier::Portable => "portable",
            Tier::ComputeDriven => "compute-driven",
            Tier::Native => "native",
            Tier::Experimental => "experimental",
        }
    }
}

/// What one adapter actually offers, and the tier that adds up to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Capabilities {
    pub tier: Tier,
    /// Storage buffers readable from a vertex shader.
    ///
    /// Load-bearing, not a nicety: the wireframe, the vertex points and the id
    /// pass all build their primitives from storage rather than from vertex
    /// buffers, which is what makes a mesh's whole wireframe one draw call.
    pub storage_in_vertex_stage: bool,
    /// `draw_indirect`, which is WebGPU core.
    pub indirect_draw: bool,
    /// GPU timestamp queries, for per-pass timing.
    pub timestamps: bool,
    /// An on-disk pipeline cache, so shader compilation is paid once per install
    /// rather than once per launch.
    pub pipeline_cache: bool,
}

impl Capabilities {
    /// Reads what an adapter offers.
    pub fn of(adapter: &wgpu::Adapter) -> Self {
        let features = adapter.features();
        let limits = adapter.limits();

        // Zero storage buffers in the vertex stage is the one core limit that can
        // legitimately be zero — WebGPU's own default minimum is zero — and it is
        // exactly what the overlays need. Worth asking about by name.
        let storage_in_vertex_stage = limits.max_storage_buffers_per_shader_stage >= 4;
        let indirect_draw = limits.max_compute_workgroup_size_x > 0;

        let timestamps = features.contains(wgpu::Features::TIMESTAMP_QUERY);
        let pipeline_cache = features.contains(wgpu::Features::PIPELINE_CACHE);

        let tier = if !storage_in_vertex_stage || !indirect_draw {
            Tier::Portable
        } else if timestamps || pipeline_cache {
            Tier::Native
        } else {
            Tier::ComputeDriven
        };

        Self {
            tier,
            storage_in_vertex_stage,
            indirect_draw,
            timestamps,
            pipeline_cache,
        }
    }

    /// Whether this device reaches the tier the architecture targets.
    pub fn meets_target(&self) -> bool {
        self.tier >= Tier::TARGET
    }

    /// A one-line summary, for the log and the crash report.
    ///
    /// Worth having: "the model does not draw" arrives without the one fact that
    /// explains it, and a tier in the first line of a report saves a round trip.
    pub fn summary(&self) -> String {
        let mut extras: Vec<&str> = Vec::new();
        if self.timestamps {
            extras.push("timestamps");
        }
        if self.pipeline_cache {
            extras.push("pipeline-cache");
        }
        if extras.is_empty() {
            return format!("tier {}", self.tier.name());
        }
        format!("tier {} (+{})", self.tier.name(), extras.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_tiers_are_ordered_so_a_caller_can_ask_for_at_least() {
        assert!(Tier::Portable < Tier::ComputeDriven);
        assert!(Tier::ComputeDriven < Tier::Native);
        assert!(Tier::Native < Tier::Experimental);
    }

    #[test]
    fn the_target_is_compute_driven_on_every_platform() {
        // The decision this module exists to record. Everything the large-scene
        // design needs is in this tier and present on Vulkan, Metal, DX12 and
        // WebGPU; anything above it is an accelerator.
        assert_eq!(Tier::TARGET, Tier::ComputeDriven);
    }

    #[test]
    fn the_target_is_not_the_top_tier() {
        // If the target were the highest tier, the tiers would be doing no work:
        // the point is that mesh shaders and the rest sit *above* what anything
        // is allowed to require.
        assert!(Tier::TARGET < Tier::Experimental);
    }

    /// What a device that only meets the floor looks like.
    fn floor() -> Capabilities {
        Capabilities {
            tier: Tier::Portable,
            storage_in_vertex_stage: false,
            indirect_draw: true,
            timestamps: false,
            pipeline_cache: false,
        }
    }

    #[test]
    fn a_device_without_vertex_stage_storage_does_not_meet_the_target() {
        // The overlays build their primitives from storage buffers in the vertex
        // stage — the wireframe, the vertex points, the id pass. Without it they
        // have no route at all, so it is not an accelerator.
        assert!(!floor().meets_target());
    }

    #[test]
    fn native_extras_are_above_the_target_rather_than_required() {
        let native = Capabilities {
            tier: Tier::Native,
            storage_in_vertex_stage: true,
            indirect_draw: true,
            timestamps: true,
            pipeline_cache: true,
        };
        assert!(native.meets_target());

        let plain = Capabilities {
            tier: Tier::ComputeDriven,
            storage_in_vertex_stage: true,
            indirect_draw: true,
            timestamps: false,
            pipeline_cache: false,
        };
        assert!(
            plain.meets_target(),
            "a device with no native extras still has to be fully supported"
        );
    }

    #[test]
    fn the_summary_names_the_tier_and_what_is_extra() {
        assert_eq!(floor().summary(), "tier portable");

        let native = Capabilities {
            tier: Tier::Native,
            storage_in_vertex_stage: true,
            indirect_draw: true,
            timestamps: true,
            pipeline_cache: false,
        };
        assert_eq!(native.summary(), "tier native (+timestamps)");
    }
}
