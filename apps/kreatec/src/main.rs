//! `kreatec` — Kreate's command-line companion.
//!
//! This exists from day one for three reasons: it makes round-trip fidelity
//! testable in CI without a GPU, it lets users self-diagnose their own files
//! instead of filing "it's broken", and it is a genuine product feature for
//! pipeline work.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use kreate_io::container::{self, Manifest, UnknownEntry};
use kreate_io::gltf_export::{write_gltf, GltfExportOptions};
use kreate_io::gltf_import::{parse_gltf, GltfOptions};
use kreate_io::obj::{parse_obj, write_obj, ObjOptions};
use kreate_io::stl::{parse_stl, write_stl, StlOptions};
use kreate_mesh::MeshData;
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::Document;

const USAGE: &str = "\
kreatec — Kreate command-line tools

USAGE:
    kreatec inspect  <file>                  Summarise a file's contents
    kreatec validate <file>                  Check structural invariants
    kreatec convert  <in> <out>              Read and rewrite through the IR
    kreatec gen      <shape> <out>           Write a synthetic test mesh
    kreatec diff     <a> <b>                 Semantic diff of two files
    kreatec textconv <file>                  Stable text rendering, for git

FORMATS:
    .kreate, .obj, .stl, .gltf, .glb -- read and written, chosen by extension.

    .kreate is the native format and the only one that carries a scene:
    objects, hierarchy, transforms and shared mesh datablocks. The others
    carry meshes, so reading one produces a document of unparented objects.

    STL is triangles only, so a convert to it triangulates and says so.
    glTF is too, but records the original faces in a KREATE_ngons extension,
    so a Kreate mesh survives the round trip with its n-gons intact.

USING DIFF WITH GIT:
    A .kreate file is a zip, so git cannot diff it usefully on its own.
    Register `textconv` and git will diff the readable rendering instead:

        git config diff.kreate.textconv \"kreatec textconv\"
        echo \"*.kreate diff=kreate\" >> .gitattributes

    `kreatec diff` is the standalone form, and is what to reach for when the
    question is \"what changed\" rather than \"show me a patch\".

SHAPES:
    plane, grid[:N], cube, sphere[:SEG,RINGS], icosphere[:SUBDIV],
    cylinder[:SEG], cone[:SEG], torus[:MAJ,MIN]

OPTIONS:
    --z-up            Treat the source as Z-up (default for OBJ is Y-up)
    --scale <metres>  Metres per source unit (default 1.0)
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &[String]) -> Result<(), String> {
    let mut opts = ObjOptions::default();
    let mut positional: Vec<&str> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--z-up" => opts.axis = kreate_math::AxisConvention::KREATE,
            "--scale" => {
                i += 1;
                let v = args.get(i).ok_or("--scale needs a value")?;
                opts.unit_scale = v.parse().map_err(|_| format!("bad --scale '{v}'"))?;
            }
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(());
            }
            other => positional.push(other),
        }
        i += 1;
    }

    match positional.split_first() {
        Some((&"inspect", [path])) => inspect(Path::new(path), opts),
        Some((&"validate", [path])) => validate(Path::new(path), opts),
        Some((&"convert", [src, dst])) => convert(Path::new(src), &PathBuf::from(dst), opts),
        Some((&"gen", [shape, dst])) => gen(shape, &PathBuf::from(dst), opts),
        Some((&"diff", [a, b])) => diff(Path::new(a), Path::new(b), opts),
        Some((&"textconv", [path])) => textconv(Path::new(path), opts),
        _ => {
            print!("{USAGE}");
            Err("no command given".into())
        }
    }
}

/// Writes a document to `.kreate`, carrying unrecognised entries through.
///
/// `unknown` is not optional bookkeeping. §5.2 of the plan requires that a
/// document round-tripped through an older build keeps the data that build did
/// not understand, and a converter that dropped it would be the exact tool that
/// destroys a colleague's file by opening and saving it.
fn save_document(
    path: &Path,
    document: &Document,
    manifest: &Manifest,
    unknown: &[UnknownEntry],
) -> Result<(), String> {
    let bytes = container::write(document, manifest, unknown)
        .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    std::fs::write(path, bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

/// Reads a file through the IR, printing anything that had to be assumed or
/// dropped along the way.
fn load(path: &Path, opts: ObjOptions) -> Result<Vec<(String, MeshData)>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let scene = if is_gltf(path) {
        // glTF declares its own units and orientation, so the command line
        // flags do not apply: overriding what the file states would be
        // second-guessing the one format that bothers to say.
        parse_gltf(&bytes, &GltfOptions::default())
            .map_err(|e| format!("{}: {e}", path.display()))?
    } else if is_stl(path) {
        // STL declares neither units nor orientation, so the command line's
        // `--scale` and `--z-up` are the only source of truth. Passed through
        // rather than left at the format's default, or the flags would silently
        // do nothing for the one format that most needs them.
        let stl = StlOptions {
            axis: opts.axis,
            unit_scale: opts.unit_scale,
            ..StlOptions::default()
        };
        parse_stl(&bytes, &stl).map_err(|e| format!("{}: {e}", path.display()))?
    } else {
        let text = String::from_utf8(bytes)
            .map_err(|e| format!("{} is not utf-8: {e}", path.display()))?;
        parse_obj(&text, opts).map_err(|e| format!("{}: {e}", path.display()))?
    };
    let (meshes, source) = scene.into_mesh_data();
    for note in &source.notes {
        println!("  note: {note:?}");
    }
    Ok(meshes)
}

fn is_kreate(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("kreate"))
}

/// Everything a command needs about a file it was handed.
struct Opened {
    document: Document,
    /// Present only for `.kreate`; the interchange formats do not carry one.
    manifest: Option<Manifest>,
    unknown: Vec<UnknownEntry>,
}

/// Reads any supported file as a document.
///
/// The interchange formats have no scene, so their meshes become unparented
/// objects named after the mesh. That is a real widening rather than a
/// convenience: it means `inspect`, `validate` and `diff` are written once
/// against a document instead of three times against whatever each format
/// happens to produce.
fn open(path: &Path, opts: ObjOptions) -> Result<Opened, String> {
    if is_kreate(path) {
        let file = std::fs::File::open(path)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let loaded = container::read(std::io::BufReader::new(file))
            .map_err(|e| format!("{}: {e}", path.display()))?;
        // Links refreshed here too, so `inspect` and `validate` report the
        // geometry a user would actually see rather than whatever was cached in
        // the file — and so `diff` between two shots does not report a
        // difference that is really an asset having moved on.
        let mut document = loaded.document;
        let directory = path.parent().unwrap_or(Path::new("."));
        for problem in kreate_io::library::refresh_links(&mut document, directory) {
            println!("  note: {problem}");
        }
        for entry in &loaded.unknown {
            // Not a warning. An unknown entry is how a newer build's data
            // survives a round trip through this one, and saying so is how the
            // user learns their file has more in it than this build understands.
            println!("  note: kept unrecognised entry {}", entry.name);
        }
        return Ok(Opened {
            document,
            manifest: Some(loaded.manifest),
            unknown: loaded.unknown,
        });
    }
    let meshes = load(path, opts)?;
    let mut document = Document::new();
    for (name, mesh) in meshes {
        let id = document.add_mesh(mesh);
        document.add_object(Object::new(name, ObjectData::Mesh(id)));
    }
    Ok(Opened {
        document,
        manifest: None,
        unknown: Vec::new(),
    })
}

/// The meshes of a document, in document order, paired with an object's name.
///
/// A mesh datablock shared by several objects appears once, under the first
/// object that uses it — which is the point of the indirection and what a
/// summary should show.
fn named_meshes(document: &Document) -> Vec<(String, &MeshData)> {
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for (_, object) in document.objects() {
        let Some(id) = object.mesh() else { continue };
        if !seen.insert(id) {
            continue;
        }
        if let Some(mesh) = document.mesh(id) {
            out.push((object.name.clone(), mesh));
        }
    }
    out
}

fn is_gltf(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("gltf") || e.eq_ignore_ascii_case("glb"))
}

fn is_stl(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("stl"))
}

/// Writes one mesh in whichever format the destination names.
///
/// Notes are printed rather than swallowed: exporting to STL drops n-gons, UVs
/// and names, and a user who is not told will find out from the next program to
/// open the file.
fn save(path: &Path, meshes: &[(String, &MeshData)], opts: ObjOptions) -> Result<(), String> {
    if is_kreate(path) {
        // A mesh list is not a scene, so this builds the minimal document that
        // holds it. `save_document` is the path that keeps a hierarchy.
        let mut document = Document::new();
        for (name, mesh) in meshes {
            let id = document.add_mesh((*mesh).clone());
            document.add_object(Object::new(name.clone(), ObjectData::Mesh(id)));
        }
        return save_document(path, &document, &Manifest::default(), &[]);
    }
    if is_gltf(path) {
        let gltf = GltfExportOptions {
            // `.gltf` gets the JSON form with an embedded buffer, `.glb` the
            // binary one. Chosen from the extension rather than a flag, because
            // a `.glb` full of JSON is a file no viewer will open.
            binary: path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("glb")),
            ..GltfExportOptions::default()
        };
        // No material table: `into_mesh_data` keeps the per-face material
        // indices but not the materials themselves, so the exporter writes
        // placeholders and the partition survives. Worth saying, since the
        // colours will not come back.
        let (bytes, notes) = write_gltf(meshes, &[], &gltf);
        for note in &notes {
            println!("  note: {note:?}");
        }
        return std::fs::write(path, bytes)
            .map_err(|e| format!("cannot write {}: {e}", path.display()));
    }
    if is_stl(path) {
        let stl = StlOptions {
            axis: opts.axis,
            unit_scale: opts.unit_scale,
            ..StlOptions::default()
        };
        // STL holds a single unnamed solid, so several meshes cannot travel in
        // one file. Said plainly rather than writing the first and dropping the
        // rest.
        if meshes.len() > 1 {
            return Err(format!(
                "{} meshes cannot be written to one STL; it holds a single unnamed solid",
                meshes.len()
            ));
        }
        let empty = MeshData::empty();
        let mesh = meshes.first().map(|(_, m)| *m).unwrap_or(&empty);
        let (bytes, notes) = write_stl(mesh, &stl);
        for note in &notes {
            println!("  note: {note:?}");
        }
        std::fs::write(path, bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))
    } else {
        let text = write_obj(meshes, opts);
        std::fs::write(path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))
    }
}

fn inspect(path: &Path, opts: ObjOptions) -> Result<(), String> {
    let opened = open(path, opts)?;
    if let Some(manifest) = &opened.manifest {
        println!(
            "{}: {} v{}.{} by {} ({} up, {} m per unit)",
            path.display(),
            manifest.format,
            manifest.version_major,
            manifest.version_minor,
            manifest.generator,
            manifest.up_axis,
            manifest.unit_scale,
        );
        if manifest.min_reader_major > container::VERSION_MAJOR {
            println!(
                "  warning: this file wants a reader of major version {} or newer",
                manifest.min_reader_major
            );
        }
        for (id, depth) in opened.document.hierarchy() {
            let Some(object) = opened.document.object(id) else {
                continue;
            };
            let t = &object.transform;
            let kind = match object.data {
                ObjectData::Mesh(_) => "mesh",
                ObjectData::Empty => "empty",
                ObjectData::Camera(_) => "camera",
                ObjectData::Light(_) => "light",
                ObjectData::Armature(_) => "armature",
            };
            println!(
                "  {:indent$}{} [{kind}] at {:.4?}{}{}",
                "",
                object.name,
                t.translation.to_array(),
                if object.visible { "" } else { " hidden" },
                match &object.linkage {
                    kreate_scene::document::Linkage::Local => String::new(),
                    kreate_scene::document::Linkage::Linked { library, name } =>
                        format!(" linked from {library}:{name}"),
                },
                indent = depth * 2,
            );
        }
    }
    let meshes = named_meshes(&opened.document);
    println!("{}: {} mesh(es)", path.display(), meshes.len());
    for (name, m) in &meshes {
        let m = *m;
        let b = m.bounds();
        let tri = m.triangulate();
        let ngons = (0..m.num_faces()).filter(|&f| m.face_len(f) > 4).count();
        let quads = (0..m.num_faces()).filter(|&f| m.face_len(f) == 4).count();
        println!(
            "  {name}: {} verts, {} edges, {} faces ({} quads, {} n-gons), {} tris",
            m.num_verts(),
            m.num_edges(),
            m.num_faces(),
            quads,
            ngons,
            tri.num_triangles()
        );
        println!(
            "    bounds: {:.4?} .. {:.4?}  (V-E+F = {})",
            b.min.to_array(),
            b.max.to_array(),
            m.num_verts() as i64 - m.num_edges() as i64 + m.num_faces() as i64
        );
        let layers: Vec<String> = m
            .attributes()
            .iter()
            .map(|l| format!("{}:{}", l.name, l.domain.name()))
            .collect();
        println!("    layers: {}", layers.join(", "));
    }
    Ok(())
}

fn validate(path: &Path, opts: ObjOptions) -> Result<(), String> {
    let opened = open(path, opts)?;
    let meshes = named_meshes(&opened.document);
    let mut total = 0;
    for (name, m) in &meshes {
        let violations = m.validate();
        total += violations.len();
        if violations.is_empty() {
            println!("  {name}: ok");
        } else {
            println!("  {name}: {} violation(s)", violations.len());
            for v in &violations {
                println!("    {v:?}");
            }
        }
    }
    // The document's own invariants, not just each mesh's. A hierarchy with a
    // cycle or an object pointing at a mesh that is not there is a file that
    // opens and then misbehaves, and it is exactly what a user asking "is my
    // file broken" needs told.
    for (id, object) in opened.document.objects() {
        if let ObjectData::Mesh(mesh) = object.data {
            if opened.document.mesh(mesh).is_none() {
                println!(
                    "  {}: refers to a mesh that is not in the file",
                    object.name
                );
                total += 1;
            }
        }
        if let Some(parent) = object.parent {
            if opened.document.object(parent).is_none() {
                println!(
                    "  {}: parented to an object that is not in the file",
                    object.name
                );
                total += 1;
            } else if opened.document.is_ancestor(id, parent) {
                println!("  {}: is its own ancestor", object.name);
                total += 1;
            }
        }
    }

    if total == 0 {
        Ok(())
    } else {
        Err(format!("{total} violation(s)"))
    }
}

/// A stable, line-oriented rendering of a file.
///
/// This is the `textconv` half of git integration: git converts each side of a
/// diff through this and diffs the text, so the patch a reviewer sees is about
/// objects and counts rather than about compressed bytes.
///
/// Two properties matter and neither is optional. It has to be **stable** — the
/// same document must render identically every time, or every diff is noise —
/// and it has to be **local**, so that changing one object changes a few lines
/// rather than shifting the whole file. Hence one object per line in document
/// order, and no running totals.
fn textconv(path: &Path, opts: ObjOptions) -> Result<(), String> {
    let opened = open(path, opts)?;
    let document = &opened.document;
    if let Some(m) = &opened.manifest {
        // The generator is deliberately omitted: it carries a version number, so
        // including it would make every file rebuilt by a newer build show a
        // change on its first line whether or not anything in it moved.
        println!(
            "format {} v{}.{}",
            m.format, m.version_major, m.version_minor
        );
        println!("units {} {}", m.up_axis, m.unit_scale);
    }
    for (id, depth) in document.hierarchy() {
        let Some(object) = document.object(id) else {
            continue;
        };
        let t = &object.transform;
        // Fixed precision, not `{}`. A float's shortest round-trip form changes
        // length with its value, so a coordinate drifting in its last bit would
        // reflow the line and read as a bigger change than it is.
        print!(
            "{:indent$}object {} translation {:.6} {:.6} {:.6} rotation {:.6} {:.6} {:.6} {:.6} scale {:.6} {:.6} {:.6}",
            "",
            object.name,
            t.translation.x, t.translation.y, t.translation.z,
            t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w,
            t.scale.x, t.scale.y, t.scale.z,
            indent = depth * 2,
        );
        match object.data {
            ObjectData::Mesh(mesh) => match document.mesh(mesh) {
                Some(m) => print!(
                    " mesh verts {} edges {} faces {}",
                    m.num_verts(),
                    m.num_edges(),
                    m.num_faces()
                ),
                None => print!(" mesh missing"),
            },
            ObjectData::Empty => print!(" empty"),
            ObjectData::Camera(_) => print!(" camera"),
            ObjectData::Light(_) => print!(" light"),
            ObjectData::Armature(armature) => match document.armature(armature) {
                Some(a) => print!(" armature bones {}", a.bones.len()),
                None => print!(" armature missing"),
            },
        }
        if !object.visible {
            print!(" hidden");
        }
        if let kreate_scene::document::Linkage::Linked { library, name } = &object.linkage {
            print!(" linked {library}:{name}");
        }
        println!();
    }
    Ok(())
}

/// What changed between two files.
///
/// Objects are matched by name, because that is what a person uses to refer to
/// them and what survives a save. Not by index — inserting an object at the top
/// would then report every object after it as changed — and not by arena id,
/// which is not written to the file at all.
///
/// The consequence is stated rather than hidden: renaming an object reads as a
/// delete plus an add. That is the honest answer without a rename heuristic, and
/// a heuristic that guesses wrong about a rename is worse than one that does not
/// guess.
fn diff(a: &Path, b: &Path, opts: ObjOptions) -> Result<(), String> {
    let left = open(a, opts)?;
    let right = open(b, opts)?;

    let index = |document: &Document| -> std::collections::BTreeMap<String, ObjectSummary> {
        document
            .objects()
            .map(|(_, object)| {
                let mesh = object.mesh().and_then(|id| document.mesh(id));
                (
                    object.name.clone(),
                    ObjectSummary {
                        verts: mesh.map(|m| m.num_verts()).unwrap_or(0),
                        edges: mesh.map(|m| m.num_edges()).unwrap_or(0),
                        faces: mesh.map(|m| m.num_faces()).unwrap_or(0),
                        translation: object.transform.translation.to_array(),
                        rotation: [
                            object.transform.rotation.x,
                            object.transform.rotation.y,
                            object.transform.rotation.z,
                            object.transform.rotation.w,
                        ],
                        scale: object.transform.scale.to_array(),
                        parent: object
                            .parent
                            .and_then(|p| document.object(p))
                            .map(|p| p.name.clone()),
                        visible: object.visible,
                    },
                )
            })
            .collect()
    };
    let left_objects = index(&left.document);
    let right_objects = index(&right.document);
    let mut changes = 0;

    for name in left_objects.keys() {
        if !right_objects.contains_key(name) {
            println!("- {name}");
            changes += 1;
        }
    }
    for (name, after) in &right_objects {
        let Some(before) = left_objects.get(name) else {
            println!("+ {name}");
            changes += 1;
            continue;
        };
        for line in before.differences(after) {
            println!("~ {name}: {line}");
            changes += 1;
        }
    }

    if changes == 0 {
        println!("no semantic difference");
    }
    Ok(())
}

/// One object reduced to what a diff can talk about.
struct ObjectSummary {
    verts: u32,
    edges: u32,
    faces: u32,
    translation: [f64; 3],
    rotation: [f64; 4],
    scale: [f64; 3],
    parent: Option<String>,
    visible: bool,
}

impl ObjectSummary {
    fn differences(&self, other: &ObjectSummary) -> Vec<String> {
        // Compared exactly, with no tolerance.
        //
        // A tolerance was the obvious first instinct — "two saves of the same
        // document should not diff" — and it was wrong twice over. It was not
        // needed: `serde_json` is built here with `float_roundtrip`, so an `f64`
        // written to a `.kreate` and read back is the same `f64`, bit for bit,
        // and the interchange formats carry no transforms at all for a tolerance
        // to protect. And it was harmful: a tool whose job is to answer "what
        // changed" should not have an opinion about which real changes are too
        // small to mention. A nanometre epsilon suppresses a move the user made.
        //
        // Both facts are asserted in `tests/cli.rs` rather than assumed, because
        // if the round trip ever stops being exact this comparison is the first
        // thing that will start lying.
        let moved = |a: &[f64], b: &[f64]| a != b;

        let mut out = Vec::new();
        if self.verts != other.verts || self.edges != other.edges || self.faces != other.faces {
            out.push(format!(
                "geometry {} -> {} verts, {} -> {} edges, {} -> {} faces",
                self.verts, other.verts, self.edges, other.edges, self.faces, other.faces
            ));
        }
        if moved(&self.translation, &other.translation) {
            out.push(format!(
                "moved {:.6?} -> {:.6?}",
                self.translation, other.translation
            ));
        }
        if moved(&self.rotation, &other.rotation) {
            out.push(format!(
                "rotated {:.6?} -> {:.6?}",
                self.rotation, other.rotation
            ));
        }
        if moved(&self.scale, &other.scale) {
            out.push(format!("scaled {:.6?} -> {:.6?}", self.scale, other.scale));
        }
        if self.parent != other.parent {
            let show = |p: &Option<String>| p.clone().unwrap_or_else(|| "(root)".to_string());
            out.push(format!(
                "reparented {} -> {}",
                show(&self.parent),
                show(&other.parent)
            ));
        }
        if self.visible != other.visible {
            out.push(if other.visible { "shown" } else { "hidden" }.to_string());
        }
        out
    }
}

fn convert(src: &Path, dst: &Path, opts: ObjOptions) -> Result<(), String> {
    let opened = open(src, opts)?;

    // Native to native keeps the whole document — hierarchy, shared datablocks,
    // unrecognised entries and the source manifest. Going out through the mesh
    // list would silently flatten a scene into a pile of unparented objects,
    // which is a conversion nobody asked for.
    if is_kreate(dst) {
        let manifest = opened.manifest.clone().unwrap_or_default();
        save_document(dst, &opened.document, &manifest, &opened.unknown)?;
        println!(
            "wrote {} ({} object(s), {} mesh(es))",
            dst.display(),
            opened.document.object_count(),
            named_meshes(&opened.document).len()
        );
        return Ok(());
    }

    let meshes = named_meshes(&opened.document);
    if opened.manifest.is_some() && opened.document.object_count() > meshes.len() {
        // Said rather than left to be discovered: the interchange formats have
        // no scene, so everything that is not geometry has nowhere to go.
        println!(
            "  note: {} object(s) carry no mesh and will not appear in {}",
            opened.document.object_count() - meshes.len(),
            dst.display()
        );
    }
    save(dst, &meshes, opts)?;
    println!("wrote {} ({} mesh(es))", dst.display(), meshes.len());
    Ok(())
}

/// Builds a synthetic mesh by name.
///
/// The synthetic scene generator exists so that performance work and round-trip
/// tests have reproducible targets from day one, rather than depending on
/// whatever asset happened to be lying around.
fn build_shape(spec: &str) -> Result<MeshData, String> {
    use kreate_mesh::primitives as p;
    let (name, args) = spec.split_once(':').unwrap_or((spec, ""));
    let nums: Vec<u32> = args
        .split(',')
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().unwrap_or(0))
        .collect();
    let n = |i: usize, default: u32| nums.get(i).copied().filter(|&v| v > 0).unwrap_or(default);

    Ok(match name {
        "plane" => p::plane(2.0),
        "grid" => p::grid(2.0, n(0, 8), n(1, n(0, 8))),
        "cube" => p::cube(2.0),
        "sphere" => p::uv_sphere(1.0, n(0, 32), n(1, 16)),
        "icosphere" => p::ico_sphere(1.0, n(0, 3)),
        "cylinder" => p::cylinder(1.0, 2.0, n(0, 32)),
        "cone" => p::cone(1.0, 2.0, n(0, 32)),
        "torus" => p::torus(1.0, 0.3, n(0, 32), n(1, 16)),
        other => return Err(format!("unknown shape '{other}'")),
    })
}

fn gen(shape: &str, dst: &Path, opts: ObjOptions) -> Result<(), String> {
    let mesh = build_shape(shape)?;
    let violations = mesh.validate();
    if !violations.is_empty() {
        return Err(format!("generated mesh is invalid: {violations:?}"));
    }
    save(dst, &[(shape.to_string(), &mesh)], opts)?;
    println!(
        "wrote {} ({} verts, {} faces)",
        dst.display(),
        mesh.num_verts(),
        mesh.num_faces()
    );
    Ok(())
}
