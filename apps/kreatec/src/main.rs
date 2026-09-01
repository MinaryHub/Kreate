//! `kreatec` — Kreate's command-line companion.
//!
//! This exists from day one for three reasons: it makes round-trip fidelity
//! testable in CI without a GPU, it lets users self-diagnose their own files
//! instead of filing "it's broken", and it is a genuine product feature for
//! pipeline work.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use kreate_io::obj::{parse_obj, write_obj, ObjOptions};
use kreate_mesh::MeshData;

const USAGE: &str = "\
kreatec — Kreate command-line tools

USAGE:
    kreatec inspect  <file.obj>              Summarise a file's contents
    kreatec validate <file.obj>              Check structural invariants
    kreatec convert  <in.obj> <out.obj>      Read and rewrite through the IR
    kreatec gen      <shape> <out.obj>       Write a synthetic test mesh

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
        _ => {
            print!("{USAGE}");
            Err("no command given".into())
        }
    }
}

/// Reads a file through the IR, printing anything that had to be assumed or
/// dropped along the way.
fn load(path: &Path, opts: ObjOptions) -> Result<Vec<(String, MeshData)>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let scene = parse_obj(&text, opts).map_err(|e| format!("{}: {e}", path.display()))?;
    let (meshes, source) = scene.into_mesh_data();
    for note in &source.notes {
        println!("  note: {note:?}");
    }
    Ok(meshes)
}

fn inspect(path: &Path, opts: ObjOptions) -> Result<(), String> {
    let meshes = load(path, opts)?;
    println!("{}: {} mesh(es)", path.display(), meshes.len());
    for (name, m) in &meshes {
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
    let meshes = load(path, opts)?;
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
    if total == 0 {
        Ok(())
    } else {
        Err(format!("{total} violation(s)"))
    }
}

fn convert(src: &Path, dst: &Path, opts: ObjOptions) -> Result<(), String> {
    let meshes = load(src, opts)?;
    let refs: Vec<(String, &MeshData)> = meshes.iter().map(|(n, m)| (n.clone(), m)).collect();
    let text = write_obj(&refs, opts);
    std::fs::write(dst, text).map_err(|e| format!("cannot write {}: {e}", dst.display()))?;
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
    let text = write_obj(&[(shape.to_string(), &mesh)], opts);
    std::fs::write(dst, text).map_err(|e| format!("cannot write {}: {e}", dst.display()))?;
    println!(
        "wrote {} ({} verts, {} faces)",
        dst.display(),
        mesh.num_verts(),
        mesh.num_faces()
    );
    Ok(())
}
