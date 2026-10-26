//! The sketches of format 1.4, kept: a file of every kind of sketch this
//! build makes, and what this build makes of it.
//!
//! The 2D drafting mode (design/2d-drafting.md) replaces how a sketch is
//! held — points and lines with circles as constraints on rings of points —
//! with curves, and a migration takes old files across. The rule in
//! `kreate_io::migrate` is that a migration is tested against a file the
//! old version actually wrote, so this one is written now, before anything
//! changes, and committed. **Never regenerated.**
//!
//! Beside it, `golden-v1.4-sketches.expected.txt`: what this build reads
//! out of it — each sketch's closed regions and each body's volume and
//! bounds. That is the promise the migration keeps: the same regions, the
//! same solids.
//!
//! `KREATE_WRITE_SKETCH_GOLDEN=1 cargo test -p kreate-ops --test sketch_golden`
//! writes both. The test that always runs reads the committed file back
//! and checks it against the committed expectation.

use kreate_ops::history::History;
use kreate_ops::operator::{Context, Mode, OperatorRegistry};
use kreate_scene::document::Document;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

fn assets() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../kreate-io/tests/assets")
}

const FILE: &str = "golden-v1.4-sketches.kreate";
const EXPECTED: &str = "golden-v1.4-sketches.expected.txt";

/// Every kind of sketch, as a person would make it: script lines, run
/// through the operators in order.
const SCRIPT: &[&str] = &[
    // A plate: a dimensioned rectangle held to a parameter, a hole, 10 mm
    // up, its edges rounded.
    "scene.units_set(length=\"millimetres\")",
    "design.parameter_set(name=\"width\", expression=\"40\")",
    "sketch.new()",
    "sketch.rectangle(a=(0, 0, 0), b=(0.035, 0.018, 0), join=1)",
    "sketch.constrain(kind=\"distance\", points=[0, 1], value=0.04)",
    "sketch.constrain(kind=\"distance\", points=[1, 2], value=0.02)",
    "sketch.circle(centre=(0.02, 0.01, 0), radius=0.005)",
    "design.bind(kind=\"sketch\", object=\"Sketch\", index=5, expression=\"width\")",
    "feature.extrude(seeds=[0.002, 0.002], distance=0.01)",
    // A pocket in its top, from a sketch on that face; then rounded.
    "sketch.new(plane=\"face\", object=\"Body\", at=(0.03, 0.01, 0.01))",
    "sketch.circle(centre=(0, 0, 0), radius=0.003)",
    "feature.extrude(seeds=[0, 0], distance=-0.004, operation=\"cut\", target=\"Body\")",
    "feature.round(body=\"Body\", radius=0.001)",
    // A turned part: an open profile about the sketch's own Y.
    "sketch.new(plane=\"xz\")",
    "sketch.line(points=[0.06, 0, 0.09, 0, 0.09, 0.02, 0.075, 0.035, 0.06, 0.035])",
    "feature.revolve(seeds=[0.075, 0.015], about=\"sketch_y\", angle=6.283185307179586)",
    // Every curve, and every constraint between them.
    "sketch.new()",
    "sketch.line(points=[0, 0.05, 0.03, 0.05, 0.03, 0.08], joins=[], closed=false)",
    "sketch.arc(start=(0.04, 0.05, 0), through=(0.05, 0.06, 0), end=(0.06, 0.05, 0))",
    "sketch.spline(points=[0, 0.1, 0.01, 0.11, 0.02, 0.1, 0.03, 0.11])",
    "sketch.spline(points=[0.05, 0.1, 0.06, 0.11, 0.07, 0.1, 0.06, 0.09], closed=true)",
    "sketch.polygon(centre=(0.1, 0.05, 0), radius=0.01, sides=6)",
    "sketch.slot(a=(0.1, 0.08, 0), b=(0.13, 0.08, 0), width=0.008)",
    "sketch.point(at=(0.15, 0.15, 0))",
    "sketch.rectangle(a=(0.15, 0.05, 0), b=(0.18, 0.07, 0))",
    "sketch.circle(centre=(0.2, 0.06, 0), radius=0.006)",
    // The editing tools and the constraints between lines.
    "sketch.new()",
    "sketch.line(points=[0.25, 0, 0.29, 0.01])",
    "sketch.line(points=[0.25, 0.02, 0.28, 0.04])",
    "sketch.constrain(kind=\"parallel\", points=[1, 2, 3, 4])",
    "sketch.constrain(kind=\"equal\", points=[1, 2, 3, 4])",
    "sketch.line(points=[0.3, 0, 0.33, 0, 0.33, 0.03, 0.3, 0.03], closed=true)",
    "sketch.constrain(kind=\"angle\", points=[5, 6, 6, 7], value=1.5707963267948966)",
    "sketch.fillet(point=6, radius=0.005)",
    "sketch.offset(lines=[0], distance=0.003)",
];

fn run_script(doc: &mut Document) {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut history = History::new();
    for line in SCRIPT {
        let call = kreate_ops::script::parse_call(&registry, line)
            .unwrap_or_else(|e| panic!("{line}: {e}"));
        history
            .dispatch(
                &registry,
                doc,
                &Context { mode: Mode::Object },
                call.op_id,
                &call.params,
            )
            .unwrap_or_else(|e| panic!("{line}: {e}"));
    }
}

/// What a document's sketches and bodies come to, as text: sketches by
/// name, with their regions' areas; bodies by name, with volume and bounds.
/// Rounded so that a migration's floating point is not a difference.
fn reading(doc: &Document) -> String {
    let mut out = String::new();
    let mut objects: Vec<_> = doc.objects().collect();
    objects.sort_by(|a, b| a.1.name.cmp(&b.1.name));
    for (id, object) in &objects {
        if !object.sketch.is_empty() || !object.sketch_lines.is_empty() {
            let mut areas: Vec<f64> = kreate_scene::feature::sketch_regions(doc, *id)
                .iter()
                .map(|r| r.area() as f64)
                .collect();
            areas.sort_by(f64::total_cmp);
            let areas: Vec<String> = areas.iter().map(|a| format!("{:.4e}", a)).collect();
            let _ = writeln!(out, "sketch {} regions [{}]", object.name, areas.join(", "));
            let constraints: Vec<String> = object
                .sketch
                .iter()
                .map(|c| {
                    let kind = kreate_scene::sketch::KINDS[c.kind()];
                    match c.value() {
                        Some(v) => format!("{kind}={v:.4}"),
                        None => kind.to_string(),
                    }
                })
                .collect();
            let _ = writeln!(
                out,
                "  {} lines, constraints [{}]",
                object.sketch_lines.len(),
                constraints.join(", ")
            );
        }
        if !object.features.is_empty() {
            let mut cache = kreate_scene::modifier::ModifierCache::new();
            let Some(mesh) = cache.evaluated(doc, *id, &kreate_scene::modifier::EvalCtx::still())
            else {
                let _ = writeln!(out, "body {} (nothing)", object.name);
                continue;
            };
            let volume = kreate_mesh::measure::mass_properties(&mesh, |p| p).volume;
            let (lo, hi) = mesh
                .positions()
                .iter()
                .fold((glam::Vec3::MAX, glam::Vec3::MIN), |(lo, hi), p| {
                    (lo.min(*p), hi.max(*p))
                });
            let _ = writeln!(
                out,
                "body {} features {} volume {:.4e} bounds [{:.4}, {:.4}, {:.4}] [{:.4}, {:.4}, {:.4}]",
                object.name,
                object.features.len(),
                volume,
                lo.x,
                lo.y,
                lo.z,
                hi.x,
                hi.y,
                hi.z
            );
        }
    }
    out
}

#[test]
fn the_sketches_of_format_one_four_still_read_as_they_did() {
    let file = assets().join(FILE);
    let expected = assets().join(EXPECTED);
    if std::env::var_os("KREATE_WRITE_SKETCH_GOLDEN").is_some() {
        let mut doc = Document::new();
        run_script(&mut doc);
        let bytes =
            kreate_io::container::write(&doc, &kreate_io::container::Manifest::default(), &[])
                .expect("writes");
        std::fs::write(&file, bytes).expect("the file");
        std::fs::write(&expected, reading(&doc)).expect("the expectation");
        eprintln!("wrote {} and {}", file.display(), expected.display());
        return;
    }
    let bytes = std::fs::read(&file).expect("the committed file");
    let doc = kreate_io::container::read(std::io::Cursor::new(bytes))
        .expect("reads")
        .document;
    let want = std::fs::read_to_string(&expected).expect("the committed expectation");
    assert_eq!(reading(&doc), want);
}
