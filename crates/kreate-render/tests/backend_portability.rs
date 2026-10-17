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
///
/// The source is a `Cow` because not every file is a shader on its own.
/// `shading.wgsl` is a *fragment*: `Surface`, `Light`, `brdf_shade` and
/// `illuminance_at` all come from the two files concatenated ahead of it, so
/// parsing it alone fails. What a device receives is the composition, so that
/// is what is checked here — and dropping the file from this list instead
/// would have left the fragment tested by nothing at all.
fn shaders() -> Vec<(&'static str, std::borrow::Cow<'static, str>)> {
    vec![
        ("solid", include_str!("../src/shaders/solid.wgsl").into()),
        ("grid", include_str!("../src/shaders/grid.wgsl").into()),
        (
            "ground_shadow",
            include_str!("../src/shaders/ground_shadow.wgsl").into(),
        ),
        ("id", include_str!("../src/shaders/id.wgsl").into()),
        (
            "overlay_edge",
            include_str!("../src/shaders/overlay_edge.wgsl").into(),
        ),
        (
            "overlay_element",
            include_str!("../src/shaders/overlay_element.wgsl").into(),
        ),
        (
            "select_region",
            include_str!("../src/shaders/select_region.wgsl").into(),
        ),
        (
            "cluster_cull",
            include_str!("../src/shaders/cluster_cull.wgsl").into(),
        ),
        (
            "hiz_build",
            include_str!("../src/shaders/hiz_build.wgsl").into(),
        ),
        ("deform", include_str!("../src/shaders/deform.wgsl").into()),
        ("skin", include_str!("../src/shaders/skin.wgsl").into()),
        ("agx", include_str!("../src/shaders/agx.wgsl").into()),
        ("fxaa", kreate_render::fxaa::source().into()),
        ("ao", kreate_render::ao::source().into()),
        ("outline", kreate_render::outline::source().into()),
        ("volume", kreate_render::volume::source().into()),
        ("accum", kreate_render::accum::source().into()),
        ("temporal", kreate_render::accum::temporal_source().into()),
        ("bloom", kreate_render::bloom::source().into()),
        ("ssr", kreate_render::ssr::source().into()),
        (
            "pathtrace",
            kreate_render::pathtrace::shader_source().into(),
        ),
        (
            "pathtrace_resolve",
            kreate_render::pathtrace::resolve_source().into(),
        ),
        ("denoise", kreate_render::denoise::source().into()),
        (
            "contact_depth",
            include_str!("../src/shaders/contact_depth.wgsl").into(),
        ),
        ("lights", include_str!("../src/shaders/lights.wgsl").into()),
        ("brdf", include_str!("../src/shaders/brdf.wgsl").into()),
        ("shadow", include_str!("../src/shaders/shadow.wgsl").into()),
        ("tiles", include_str!("../src/shaders/tiles.wgsl").into()),
        (
            "markers",
            include_str!("../src/shaders/markers.wgsl").into(),
        ),
        (
            "environment",
            include_str!("../src/shaders/environment.wgsl").into(),
        ),
        // The composition, not the fragment. See the note above.
        ("shading", kreate_render::shading::probe_source().into()),
        // Also a fragment, and also checked as what a device receives — which
        // means with a material on the end. The library half does not compile
        // alone: its fragment stage calls `material_sample`, and that comes
        // from a material's own compiled code.
        (
            "rendered",
            kreate_render::shading::rendered_source_with(&a_material()).into(),
        ),
        // A fragment too, and checked as what a device receives: it reads the
        // shading uniform and the environment cube, so the composition is the
        // thing that has to translate.
        (
            "background",
            kreate_render::shading::background_source().into(),
        ),
    ]
}

/// A material reading every node kind there is, each wired where the
/// surface takes its type: what the Metal translation of a graph built in
/// the node editor has to survive.
fn every_node_material() -> String {
    use kreate_scene::shader::{Link, Node, NodeKind, Socket};
    let mut g = kreate_scene::shader::ShaderGraph::standard(
        "every",
        kreate_math::SceneRgb::new(0.8, 0.8, 0.8),
        0.0,
        0.5,
    );
    let wire = |g: &mut kreate_scene::shader::ShaderGraph, from, socket, to, at| {
        let _ = g.link(Link {
            from_node: from,
            from_socket: socket,
            to_node: to,
            to_socket: at,
        });
    };
    // Into a chain of mixes, so every colour, number and point is read.
    for tag in NodeKind::ADDABLE {
        let Some(kind) = NodeKind::new_of(tag) else {
            continue;
        };
        let outputs = kind.outputs().to_vec();
        let inputs = kind.inputs().to_vec();
        let Some(n) = g.add_node(Node::new(tag, kind)) else {
            continue;
        };
        if inputs.first() == Some(&Socket::Coordinate) {
            if let Some(t) = g.add_node(Node::new("T", NodeKind::TexCoord)) {
                wire(&mut g, t, 0, n, 0);
            }
        }
        let mix = g
            .add_node(Node::new("M", NodeKind::MixColor { factor: 0.5 }))
            .expect("a mix");
        match outputs.first() {
            Some(Socket::Color) => wire(&mut g, n, 0, mix, 1),
            Some(Socket::Scalar) => wire(&mut g, n, 0, mix, 0),
            Some(Socket::Coordinate) => {
                if let Some(noise) = g.add_node(Node::new("N", NodeKind::new_of("noise").unwrap()))
                {
                    wire(&mut g, n, 0, noise, 4);
                    wire(&mut g, noise, 0, mix, 0);
                }
            }
            Some(Socket::Vector) => wire(&mut g, n, 0, 0, 3),
            None => {}
        }
        // The chain: this mix into the surface's colour, the last one into
        // this one's second colour.
        if let Some(previous) = g.incoming(0, 0).map(|l| l.from_node) {
            wire(&mut g, previous, 0, mix, 2);
        }
        wire(&mut g, mix, 0, 0, 0);
    }
    kreate_shader::compile(&g)
        .expect("every node compiles")
        .wgsl
}

/// A plain grey material, compiled.
///
/// Through the real compiler, so that what is translated to HLSL, Metal and
/// SPIR-V is what a device is actually handed.
fn a_material() -> String {
    let graph = kreate_scene::shader::ShaderGraph::standard(
        "test",
        kreate_math::SceneRgb::new(0.8, 0.8, 0.8),
        0.0,
        0.5,
    );
    kreate_shader::compile(&graph)
        .expect("a standard surface compiles")
        .wgsl
}

/// A name a Metal function declares twice, which its compiler refuses
/// though naga writes it: a value read beside an early return once came out
/// as two `const auto _tmp`s in the ground shadow, and the pipeline failed
/// on the user's machine and nowhere here.
fn declared_twice(msl: &str) -> Option<String> {
    // One set of names per open brace: a name again in an inner block is
    // legal C++, the same block twice is not.
    let mut scopes: Vec<std::collections::HashSet<String>> = vec![Default::default()];
    for line in msl.lines() {
        if let Some(rest) = line.trim_start().strip_prefix("const auto ") {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if let Some(scope) = scopes.last_mut() {
                if !name.is_empty() && !scope.insert(name.clone()) {
                    return Some(name);
                }
            }
        }
        for c in line.chars() {
            match c {
                '{' => scopes.push(Default::default()),
                '}' if scopes.len() > 1 => {
                    scopes.pop();
                }
                _ => {}
            }
        }
    }
    None
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
        let (module, info) = front(name, &source);
        let options = msl::Options {
            lang_version: MSL_FLOOR,
            ..msl::Options::default()
        };
        let pipeline = msl::PipelineOptions::default();
        let (text, _) = msl::write_string(&module, &info, &options, &pipeline)
            .unwrap_or_else(|e| panic!("{name} does not translate to Metal: {e:?}"));
        if let Some(twice) = declared_twice(&text) {
            panic!("{name}: the Metal text declares `{twice}` twice in one function");
        }
    }
}

#[test]
fn every_shader_translates_to_hlsl() {
    for (name, source) in shaders() {
        let (module, info) = front(name, &source);
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
        let (module, info) = front(name, &source);
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
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&source)));
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

#[test]
fn a_material_of_every_node_translates_to_metal() {
    // Not in the list above, which is the shader directory's files: this is
    // one the node editor builds, and the place the Metal compiler once met
    // a name declared twice — which naga writes without complaint.
    let source = kreate_render::shading::rendered_source_with(&every_node_material());
    let (module, info) = front("rendered, every node", &source);
    let options = msl::Options {
        lang_version: MSL_FLOOR,
        ..msl::Options::default()
    };
    let (text, _) = msl::write_string(&module, &info, &options, &msl::PipelineOptions::default())
        .expect("translates to Metal");
    assert_eq!(declared_twice(&text), None);
}

#[test]
fn a_name_declared_twice_in_one_block_is_caught_and_in_two_is_not() {
    let twice = "void f() {\n    const auto _tmp = a;\n    const auto _tmp = b;\n}\n";
    assert_eq!(declared_twice(twice), Some("_tmp".to_string()));
    let apart = "void f() {\n    if (x) {\n        const auto _tmp = a;\n    }\n    const auto _tmp = b;\n}\n";
    assert_eq!(declared_twice(apart), None);
}
