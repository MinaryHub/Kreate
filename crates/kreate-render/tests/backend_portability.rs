//! Every shader translates to Metal, Direct3D and Vulkan.
//!
//! The plan's portability rule is that the renderer must need only compute plus
//! a single indirect draw, so that macOS is a first-class target rather than a
//! rewrite discovered in year two. That rule is easy to state and easy to break
//! by accident, and the usual way it gets broken is a shader construct that
//! happens to work on one backend.
//!
//! Nobody here has a Mac. What can still be checked, offline and in CI, is
//! whether the shaders **translate**: naga is the same compiler wgpu uses at run
//! time, so putting every shader through its Metal, HLSL and SPIR-V back ends
//! catches the whole class of "this construct has no Metal equivalent" before it
//! becomes a bug report from someone else's laptop.
//!
//! # What this proves and what it does not
//!
//! It proves the shaders are expressible on all three backends and that naga
//! will not refuse them. It does not prove the Metal driver produces the same
//! pixels, that the pipelines build, or that performance holds. Those need real
//! hardware, and the plan's Phase 4 gate says so.
//!
//! It is still the difference between "we believe it is portable" and "the
//! compiler agrees it is".

use naga::back::{hlsl, msl, spv};
use naga::valid::{Capabilities, ValidationFlags, Validator};

/// The oldest Metal Shading Language the renderer requires.
///
/// Stated here rather than taken from `msl::Options::default()`, which is MSL
/// 1.0 — a version no Mac has shipped with for over a decade, and which lacks
/// `instance_id`. The first run of this file used the default and reported the
/// cluster path's `@builtin(instance_index)` as un-translatable to Metal, which
/// was alarming and wrong.
///
/// 2.0 is macOS 10.13, which is also wgpu's own floor, so requiring it costs
/// nothing. Naming a floor is worth more than picking a high version: anything
/// needing MSL 2.2 or 2.3 — `primitive_id`, barycentric coordinates — now fails
/// here, and those are exactly the constructs the plan avoids because the
/// wireframe and picking designs must not depend on them.
const MSL_FLOOR: (u8, u8) = (2, 0);

/// Every shader the renderer ships, by the name its file has.
fn shaders() -> Vec<(&'static str, &'static str)> {
    vec![
        ("solid", include_str!("../src/shaders/solid.wgsl")),
        ("grid", include_str!("../src/shaders/grid.wgsl")),
        ("id", include_str!("../src/shaders/id.wgsl")),
        (
            "overlay_edge",
            include_str!("../src/shaders/overlay_edge.wgsl"),
        ),
        (
            "overlay_element",
            include_str!("../src/shaders/overlay_element.wgsl"),
        ),
        (
            "select_region",
            include_str!("../src/shaders/select_region.wgsl"),
        ),
        (
            "cluster_cull",
            include_str!("../src/shaders/cluster_cull.wgsl"),
        ),
        ("hiz_build", include_str!("../src/shaders/hiz_build.wgsl")),
        ("deform", include_str!("../src/shaders/deform.wgsl")),
    ]
}

/// Parses and validates, returning what the back ends need.
fn front(name: &str, source: &str) -> (naga::Module, naga::valid::ModuleInfo) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(source)));
    // The default capability set, deliberately: validating with
    // `Capabilities::all()` would accept constructs that only some backends
    // support, which is the opposite of what this file is for.
    let info = Validator::new(ValidationFlags::all(), Capabilities::default())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name} failed validation: {e:?}"));
    (module, info)
}

#[test]
fn every_shader_translates_to_metal() {
    // The one that matters. Metal is the backend nobody here can run, and the
    // one the plan repeatedly warns about: no `MULTI_DRAW_INDIRECT_COUNT`, no
    // `POLYGON_MODE_LINE`, mesh shaders on a separate path. The architecture
    // avoids all three by design; this checks the shaders followed suit.
    for (name, source) in shaders() {
        let (module, info) = front(name, source);
        let options = msl::Options {
            lang_version: MSL_FLOOR,
            ..msl::Options::default()
        };
        let pipeline = msl::PipelineOptions::default();
        msl::write_string(&module, &info, &options, &pipeline)
            .unwrap_or_else(|e| panic!("{name} does not translate to Metal: {e:?}"));
    }
}

#[test]
fn every_shader_translates_to_hlsl() {
    for (name, source) in shaders() {
        let (module, info) = front(name, source);
        let mut out = String::new();
        let options = hlsl::Options::default();
        let pipeline = hlsl::PipelineOptions::default();
        let mut writer = hlsl::Writer::new(&mut out, &options, &pipeline);
        writer
            .write(&module, &info, None)
            .unwrap_or_else(|e| panic!("{name} does not translate to HLSL: {e:?}"));
    }
}

#[test]
fn every_shader_translates_to_spirv() {
    for (name, source) in shaders() {
        let (module, info) = front(name, source);
        let options = spv::Options::default();
        spv::write_vec(&module, &info, &options, None)
            .unwrap_or_else(|e| panic!("{name} does not translate to SPIR-V: {e:?}"));
    }
}

#[test]
fn no_shader_needs_a_capability_beyond_the_webgpu_core() {
    // Validated against the *default* capabilities, which is the WebGPU core
    // set. A shader that needs more would compile here and fail on a device
    // that does not offer the feature — and since `required_features` is
    // deliberately empty, no device offers any.
    //
    // Separate from the translation tests because the failure is different in
    // kind: those say "this backend cannot express it", this says "this needs a
    // feature we promised not to require".
    for (name, source) in shaders() {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(source)));
        Validator::new(ValidationFlags::all(), Capabilities::default())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name} needs a capability outside the WebGPU core: {e:?}"));
    }
}

#[test]
fn the_shader_list_covers_every_shader_in_the_directory() {
    // A shader added without being listed here would be tested on nothing. That
    // is exactly how the portability rule erodes: not by someone overriding it,
    // but by a new file quietly escaping the check.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/shaders");
    let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
        .expect("shader directory")
        .filter_map(|e| {
            let path = e.ok()?.path();
            (path.extension()? == "wgsl")
                .then(|| path.file_stem()?.to_str().map(str::to_owned))
                .flatten()
        })
        .collect();
    on_disk.sort();

    let mut listed: Vec<String> = shaders().iter().map(|(n, _)| n.to_string()).collect();
    listed.sort();

    assert_eq!(
        on_disk, listed,
        "the shader list in this file is out of step with src/shaders"
    );
}
