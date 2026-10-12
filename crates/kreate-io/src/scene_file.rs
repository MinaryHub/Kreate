//! A scene file, whatever its format, read into a document of its own.
//!
//! What Open, Import, a drop and the asset browser all read through, and what
//! `object.import` adds: one place that knows which extension is which reader,
//! so no caller has its own list to fall behind.
//!
//! # Prepared off the main thread
//!
//! Reading a large model takes long enough to stop the interface. The
//! application reads it on a thread of its own and leaves the result here,
//! with [`prepare`]; the operator that adds it finds it with [`take`] and does
//! only the adding. A file not prepared — a script replaying the step — is
//! read where it is asked for.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use kreate_scene::Document;

use crate::build::BuildNote;

/// The extensions read as scenes, beside Kreate's own `.kreate`.
pub const SCENE_FORMATS: &[&str] = &[
    "gltf", "glb", "fbx", "obj", "stl", "ply", "3mf", "off", "usd", "usda", "usdc", "usdz", "dae",
    "urdf", "step", "stp",
];

/// Whether `path` names a file this module reads.
pub fn is_scene_file(path: &Path) -> bool {
    path.extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .is_some_and(|e| e == "kreate" || SCENE_FORMATS.contains(&e.as_str()))
}

/// Reads a scene file into a fresh document, through the IR — or, for a
/// `.kreate`, the document it holds, its links refreshed.
///
/// Every format lands here, which is the point: what a file said and what the
/// document holds is one conversion in one place, and §5.7's fidelity contract
/// is written against it.
pub fn read_scene_file(path: &Path) -> Result<(Document, Vec<BuildNote>), String> {
    read_scene_file_with(path, &ReadProgress::default())
}

/// What a read is doing, for a progress bar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadStage {
    /// The file's bytes coming off the disk.
    Reading,
    /// The file's records being read: a CAD file's entities.
    Parsing,
    /// Exact surfaces being cut into triangles, face by face.
    Shaping,
    /// What was read becoming objects.
    Building,
}

/// How far a read has got, shared between the thread reading and the window
/// showing it — and the window's way to ask it to stop.
///
/// A large CAD file takes minutes, and a read that says nothing for minutes
/// is indistinguishable from one that has stopped.
#[derive(Debug, Default)]
pub struct ReadProgress {
    stage: AtomicU8,
    done: AtomicU64,
    total: AtomicU64,
    cancel: AtomicBool,
}

impl ReadProgress {
    /// Starts a stage, `total` steps long (zero for one that cannot say).
    pub fn start(&self, stage: ReadStage, total: u64) {
        self.done.store(0, Ordering::Relaxed);
        self.total.store(total, Ordering::Relaxed);
        self.stage.store(stage as u8, Ordering::Relaxed);
    }
    pub fn advance(&self, done: u64) {
        self.done.store(done, Ordering::Relaxed);
    }
    /// The stage, and how far through it: `None` when it cannot say.
    pub fn get(&self) -> (ReadStage, Option<f32>) {
        let stage = match self.stage.load(Ordering::Relaxed) {
            0 => ReadStage::Reading,
            1 => ReadStage::Parsing,
            2 => ReadStage::Shaping,
            _ => ReadStage::Building,
        };
        let total = self.total.load(Ordering::Relaxed);
        let done = self.done.load(Ordering::Relaxed);
        let fraction = (total > 0).then(|| (done as f64 / total as f64).clamp(0.0, 1.0) as f32);
        (stage, fraction)
    }
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

/// What a cancelled read answers.
pub const CANCELLED: &str = "cancelled";

/// [`read_scene_file`], saying how far it has got in `progress` and stopping
/// when it is asked to.
pub fn read_scene_file_with(
    path: &Path,
    progress: &ReadProgress,
) -> Result<(Document, Vec<BuildNote>), String> {
    progress.start(ReadStage::Reading, 0);
    let lower = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if lower == "kreate" {
        let file = std::fs::File::open(path).map_err(|e| format!("cannot open: {e}"))?;
        let mut document = crate::container::read(std::io::BufReader::new(file))
            .map_err(|e| e.to_string())?
            .document;
        let directory = path.parent().unwrap_or(Path::new("."));
        let _ = crate::library::refresh_links(&mut document, directory);
        return Ok((document, Vec::new()));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read: {e}"))?;
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "Import".to_string());
    let text = |bytes: Vec<u8>| String::from_utf8(bytes).map_err(|e| format!("not utf-8: {e}"));
    let scene = match lower.as_str() {
        // A texture an FBX names but does not carry is looked for beside it:
        // where it says, then by its file name alone — what a model copied
        // from another machine still has.
        "fbx" => {
            let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
            crate::parse_fbx(&bytes, &|named: &str| {
                let named = named.replace('\\', "/");
                let bare = named.rsplit('/').next().map(|n| dir.join(n));
                std::fs::read(dir.join(&named))
                    .ok()
                    .or_else(|| bare.and_then(|b| std::fs::read(b).ok()))
            })
            .map_err(|e| e.to_string())?
        }
        "3mf" => crate::parse_3mf(&bytes).map_err(|e| e.to_string())?,
        "ply" => crate::parse_ply(&bytes, &name).map_err(|e| e.to_string())?,
        "off" => crate::parse_off(&text(bytes)?, &name).map_err(|e| e.to_string())?,
        // glTF declares its own units and orientation, so there is nothing
        // for an option to override: second-guessing the one format that
        // bothers to say would be the wrong kind of helpful.
        "gltf" | "glb" => crate::gltf_import::parse_gltf_file(path, &Default::default())
            .map_err(|e| e.to_string())?,
        "stl" => crate::stl::parse_stl(&bytes, &Default::default()).map_err(|e| e.to_string())?,
        "obj" => crate::parse_obj(&text(bytes)?, Default::default()).map_err(|e| e.to_string())?,
        "dae" => crate::dae::parse_dae_file(path).map_err(|e| e.to_string())?,
        // Read lossily: a CAD program's names are often in its own code page.
        "step" | "stp" => {
            crate::step_read::parse_step_with(&String::from_utf8_lossy(&bytes), &name, progress)
                .map_err(|e| match e {
                    crate::step_read::StepError::Cancelled => CANCELLED.to_string(),
                    e => e.to_string(),
                })?
        }
        // USD finds what a stage references from the stage's own folder, so
        // it reads by path.
        "usd" | "usda" | "usdc" | "usdz" => {
            crate::usd_read::parse_usd_file(path).map_err(|e| e.to_string())?
        }
        "urdf" => crate::urdf::parse_urdf(
            &text(bytes)?,
            path.parent().unwrap_or(Path::new(".")),
            &read_part,
        )
        .map_err(|e| e.to_string())?,
        other => return Err(format!("`.{other}` is not a scene file Kreate reads")),
    };
    if progress.cancelled() {
        return Err(CANCELLED.to_string());
    }
    progress.start(ReadStage::Building, 0);
    let mut doc = Document::new();
    let (made, notes) = crate::build::build_into(&scene, &mut doc);
    if let Some(first) = made.first() {
        doc.selection.add(*first);
    }
    Ok((doc, notes))
}

/// A mesh file a robot description names, in Kreate's axes and metres. ROS
/// takes an STL or an OBJ as metres and Z up, whatever a printer or another
/// program would, so they are read that way here; a format that says its own
/// axes and units (glTF, COLLADA) is read by what it says.
pub fn read_part(path: &Path) -> Result<crate::ir::ImportScene, String> {
    let lower = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    let bytes = || std::fs::read(path).map_err(|e| format!("cannot read: {e}"));
    let name = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "part".to_string());
    let scene = match lower.as_str() {
        "stl" => crate::stl::parse_stl(
            &bytes()?,
            &crate::stl::StlOptions {
                unit_scale: 1.0,
                ..Default::default()
            },
        )
        .map_err(|e| e.to_string())?,
        "obj" => crate::parse_obj(
            &String::from_utf8_lossy(&bytes()?),
            crate::obj::ObjOptions {
                axis: kreate_math::AxisConvention::KREATE,
                unit_scale: 1.0,
            },
        )
        .map_err(|e| e.to_string())?,
        "ply" => crate::parse_ply(&bytes()?, &name).map_err(|e| e.to_string())?,
        "dae" => crate::dae::parse_dae_file(path).map_err(|e| e.to_string())?,
        "gltf" | "glb" => crate::gltf_import::parse_gltf_file(path, &Default::default())
            .map_err(|e| e.to_string())?,
        "usd" | "usda" | "usdc" | "usdz" => {
            crate::usd_read::parse_usd_file(path).map_err(|e| e.to_string())?
        }
        other => return Err(format!("`.{other}` meshes are not read")),
    };
    let mut scene = scene.baked_to_kreate();
    // A part's one mesh by the file's name: an STL's own name is "stl".
    let single = scene.meshes.len() == 1;
    for m in &mut scene.meshes {
        if single || m.name.is_empty() || m.name == "Import" {
            m.name = name.clone();
        }
    }
    Ok(scene)
}

/// Documents read ahead, by path and the file's modification time.
type Prepared = HashMap<PathBuf, (Option<SystemTime>, Arc<Document>)>;

fn prepared() -> &'static Mutex<Prepared> {
    static PREPARED: OnceLock<Mutex<Prepared>> = OnceLock::new();
    PREPARED.get_or_init(|| Mutex::new(HashMap::new()))
}

fn modified(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

/// Leaves a document read from `path` for [`take`] to find.
pub fn prepare(path: &Path, document: Document) {
    if let Ok(mut map) = prepared().lock() {
        map.insert(path.to_path_buf(), (modified(path), Arc::new(document)));
    }
}

/// The document read ahead from `path`, if it is there and the file has not
/// changed since; or the file, read now.
pub fn take(path: &Path) -> Result<Arc<Document>, String> {
    let ready = prepared().lock().ok().and_then(|mut map| map.remove(path));
    match ready {
        Some((when, document)) if when == modified(path) => Ok(document),
        _ => read_scene_file(path).map(|(document, _)| Arc::new(document)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn a_cube_file(dir: &Path) -> PathBuf {
        std::fs::create_dir_all(dir).expect("made");
        let mesh = kreate_mesh::primitives::cube(1.0);
        let path = dir.join("cube.obj");
        std::fs::write(
            &path,
            crate::write_obj(&[("Cube".to_string(), &mesh)], Default::default()),
        )
        .expect("written");
        path
    }

    #[test]
    fn a_file_is_read_by_its_extension_and_a_stranger_is_refused() {
        let dir = std::env::temp_dir().join(format!("kreate-scenefile-{}", std::process::id()));
        let path = a_cube_file(&dir);
        let (doc, _) = read_scene_file(&path).expect("read");
        assert_eq!(doc.objects().count(), 1);
        assert!(is_scene_file(&path) && !is_scene_file(Path::new("notes.txt")));
        let other = dir.join("notes.txt");
        std::fs::write(&other, "hello").expect("written");
        assert!(read_scene_file(&other).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_prepared_document_is_taken_once_and_a_changed_file_is_read_again() {
        let dir = std::env::temp_dir().join(format!("kreate-prepared-{}", std::process::id()));
        let path = a_cube_file(&dir);
        // What was prepared is what is taken — here, a document of two
        // objects standing for the file's one.
        let mut two = Document::new();
        two.add_mesh_object("A", kreate_mesh::primitives::cube(1.0));
        two.add_mesh_object("B", kreate_mesh::primitives::cube(1.0));
        prepare(&path, two);
        assert_eq!(take(&path).expect("taken").objects().count(), 2);
        // Taken once: the second time reads the file.
        assert_eq!(take(&path).expect("read").objects().count(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[cfg(test)]
mod robots_and_stages {
    use super::*;

    #[test]
    fn a_robot_and_a_stage_open_through_the_one_reader() {
        let dir = std::env::temp_dir().join(format!("kreate-robot-{}", std::process::id()));
        std::fs::create_dir_all(dir.join("meshes")).unwrap();
        // A link's mesh as ROS has it: metres, and a capital extension.
        std::fs::write(
            dir.join("meshes").join("arm.STL"),
            "solid arm\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 0.5 0 0\nvertex 0 0.5 0\nendloop\nendfacet\nendsolid arm\n",
        )
        .unwrap();
        std::fs::write(
            dir.join("bot.urdf"),
            r#"<robot name="bot">
                <link name="base"/>
                <link name="arm"><visual><geometry><mesh filename="meshes/arm.stl"/></geometry></visual></link>
                <joint name="hinge" type="continuous">
                  <parent link="base"/><child link="arm"/>
                  <origin xyz="0 0 1"/><axis xyz="0 0 1"/>
                </joint>
            </robot>"#,
        )
        .unwrap();
        std::fs::write(
            dir.join("stage.usda"),
            "#usda 1.0\n(\n    upAxis = \"Z\"\n    metersPerUnit = 1\n)\ndef Cube \"Box\" { double size = 1 }\n",
        )
        .unwrap();
        let (robot, _) = read_scene_file(&dir.join("bot.urdf")).unwrap();
        let (stage, _) = read_scene_file(&dir.join("stage.usda")).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let names: Vec<&str> = robot.objects().map(|(_, o)| o.name.as_str()).collect();
        for want in ["base", "hinge", "arm"] {
            assert!(names.contains(&want), "{want} in {names:?}");
        }
        // The STL in metres, a metre up: its far corner half a metre out.
        let (id, mesh) = robot
            .objects()
            .find_map(|(id, o)| match o.data {
                kreate_scene::document::ObjectData::Mesh(m) => Some((id, robot.mesh(m)?)),
                _ => None,
            })
            .unwrap();
        let w = robot.world_transform(id);
        let far = mesh
            .positions()
            .iter()
            .map(|p| w.transform_point3(p.as_dvec3()))
            .fold(0.0f64, |a, p| a.max(p.x));
        assert!((far - 0.5).abs() < 1e-6, "{far}");
        assert!(stage.objects().any(|(_, o)| o.name == "Box"));
        assert!(is_scene_file(Path::new("x.usdz")) && is_scene_file(Path::new("x.URDF")));
    }
}
