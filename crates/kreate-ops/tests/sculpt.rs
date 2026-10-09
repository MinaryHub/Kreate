//! Sculpting through the operators: the new brushes and settings, the mask,
//! face sets, voxel remesh and dynamic topology.

use kreate_ops::operator::Context;
use kreate_ops::{History, OperatorRegistry, PropertyMap, PropertyValue};
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::{Document, MeshId};

fn wired(mesh: kreate_mesh::MeshData) -> (OperatorRegistry, History, Document, MeshId) {
    let mut registry = OperatorRegistry::new();
    kreate_ops::register_builtin(&mut registry);
    let mut doc = Document::new();
    let m = doc.add_mesh(mesh);
    let id = doc.add_object(Object::new("Clay", ObjectData::Mesh(m)));
    doc.selection.add(id);
    doc.selection.active = Some(id);
    (registry, History::new(), doc, m)
}

fn run(
    registry: &OperatorRegistry,
    history: &mut History,
    doc: &mut Document,
    op: &str,
    params: &[(&'static str, PropertyValue)],
) -> Result<(), kreate_ops::OpError> {
    let mut p = PropertyMap::new();
    for (k, v) in params {
        p.set(k, v.clone());
    }
    history.dispatch(registry, doc, &Context::default(), op, &p).map(|_| ())
}

fn top(doc: &Document, mesh: MeshId) -> f32 {
    doc.mesh(mesh).unwrap().positions().iter().map(|p| p.z).fold(f32::MIN, f32::max)
}

fn dab(x: f64, y: f64, r: f64) -> PropertyValue {
    PropertyValue::Floats(vec![x, y, 0.0, r])
}

#[test]
fn every_brush_runs_from_a_script_line() {
    for brush in 0..14u32 {
        let (registry, mut history, mut doc, _) = wired(kreate_mesh::primitives::grid(2.0, 8, 8));
        run(&registry, &mut history, &mut doc, "sculpt.stroke", &[
            ("brush", PropertyValue::Enum(brush)),
            ("dabs", dab(0.0, 0.0, 0.6)),
            ("strength", PropertyValue::Float(1.0)),
        ])
        .unwrap_or_else(|e| panic!("brush {brush}: {e}"));
    }
}

#[test]
fn the_mask_brush_and_mask_operator_protect_and_release() {
    let (registry, mut history, mut doc, mesh) = wired(kreate_mesh::primitives::grid(2.0, 8, 8));
    run(&registry, &mut history, &mut doc, "sculpt.mask", &[("action", PropertyValue::Enum(2))]).unwrap();
    run(&registry, &mut history, &mut doc, "sculpt.stroke", &[("dabs", dab(0.0, 0.0, 1.0)), ("strength", PropertyValue::Float(1.0))]).unwrap();
    assert_eq!(top(&doc, mesh), 0.0, "all masked: nothing moves");
    run(&registry, &mut history, &mut doc, "sculpt.mask", &[("action", PropertyValue::Enum(1))]).unwrap();
    run(&registry, &mut history, &mut doc, "sculpt.stroke", &[("dabs", dab(0.0, 0.0, 1.0)), ("strength", PropertyValue::Float(1.0))]).unwrap();
    assert!(top(&doc, mesh) > 0.0, "inverted: free again");
}

#[test]
fn a_face_set_hides_isolates_and_comes_back() {
    let (registry, mut history, mut doc, mesh) = wired(kreate_mesh::primitives::grid(2.0, 8, 8));
    run(&registry, &mut history, &mut doc, "sculpt.stroke", &[
        ("brush", PropertyValue::Enum(13)),
        ("dabs", dab(0.5, 0.5, 0.4)),
        ("face_set", PropertyValue::Int(4)),
    ])
    .unwrap();
    let sets = kreate_mesh::sculpt::face_sets_of(doc.mesh(mesh).unwrap());
    let painted = sets.iter().position(|s| *s == 4).expect("painted");
    let plain = sets.iter().position(|s| *s == 0).unwrap();
    let hidden = |doc: &Document| {
        kreate_mesh::selection::hidden_flags(doc.mesh(mesh).unwrap(), kreate_mesh::Domain::Face)
    };
    run(&registry, &mut history, &mut doc, "sculpt.face_set_visibility", &[
        ("action", PropertyValue::Enum(0)),
        ("face", PropertyValue::Int(painted as i64)),
    ])
    .unwrap();
    assert!(hidden(&doc)[painted] && !hidden(&doc)[plain]);
    // A brush leaves what is hidden alone.
    run(&registry, &mut history, &mut doc, "sculpt.stroke", &[("dabs", dab(0.5, 0.5, 0.6)), ("strength", PropertyValue::Float(1.0))]).unwrap();
    let m = doc.mesh(mesh).unwrap();
    for v in m.face_verts(painted as u32) {
        let shown_elsewhere = (0..m.num_faces())
            .any(|f| !hidden(&doc)[f as usize] && m.face_verts(f).contains(v));
        if !shown_elsewhere {
            assert_eq!(m.positions()[*v as usize].z, 0.0);
        }
    }
    run(&registry, &mut history, &mut doc, "sculpt.face_set_visibility", &[("action", PropertyValue::Enum(2))]).unwrap();
    assert!(hidden(&doc).iter().all(|h| !h));
    run(&registry, &mut history, &mut doc, "sculpt.face_set_visibility", &[
        ("action", PropertyValue::Enum(1)),
        ("face", PropertyValue::Int(painted as i64)),
    ])
    .unwrap();
    assert!(!hidden(&doc)[painted] && hidden(&doc)[plain], "isolated");
}

#[test]
fn face_set_from_mask_and_nothing_masked_is_said() {
    let (registry, mut history, mut doc, mesh) = wired(kreate_mesh::primitives::grid(2.0, 4, 4));
    assert!(run(&registry, &mut history, &mut doc, "sculpt.face_set_from_mask", &[]).is_err());
    run(&registry, &mut history, &mut doc, "sculpt.mask", &[("action", PropertyValue::Enum(2))]).unwrap();
    run(&registry, &mut history, &mut doc, "sculpt.face_set_from_mask", &[]).unwrap();
    assert!(kreate_mesh::sculpt::face_sets_of(doc.mesh(mesh).unwrap()).iter().all(|s| *s == 1));
}

#[test]
fn voxel_remesh_rebuilds_and_is_one_undo_step() {
    let (registry, mut history, mut doc, mesh) = wired(kreate_mesh::primitives::cube(2.0));
    let before = doc.mesh(mesh).unwrap().num_faces();
    run(&registry, &mut history, &mut doc, "sculpt.voxel_remesh", &[("voxel_size", PropertyValue::Float(0.2))]).unwrap();
    assert!(doc.mesh(mesh).unwrap().num_faces() > before * 10);
    assert!(history.undo(&mut doc));
    assert_eq!(doc.mesh(mesh).unwrap().num_faces(), before);
    let err = run(&registry, &mut history, &mut doc, "sculpt.voxel_remesh", &[("voxel_size", PropertyValue::Float(0.0001))]);
    assert!(err.unwrap_err().to_string().contains("voxel size"));
}

#[test]
fn an_alpha_by_name_shapes_the_stroke_and_an_unknown_one_is_said() {
    let stroke = |alpha: &str| {
        let (registry, mut history, mut doc, mesh) = wired(kreate_mesh::primitives::grid(2.0, 32, 32));
        let r = run(&registry, &mut history, &mut doc, "sculpt.stroke", &[
            ("dabs", dab(0.0, 0.0, 0.8)),
            ("strength", PropertyValue::Float(1.0)),
            ("view", PropertyValue::Vec3([0.0, 0.0, -1.0])),
            ("alpha", PropertyValue::Text(alpha.into())),
        ]);
        r.map(|_| doc.mesh(mesh).unwrap().positions().iter().map(|p| p.z).collect::<Vec<f32>>())
    };
    let plain = stroke("").unwrap();
    let stripes = stroke("stripes").unwrap();
    assert_ne!(plain, stripes);
    assert!(stroke("plaid").unwrap_err().to_string().contains("no alpha"));
}

#[test]
fn dynamic_topology_adds_detail_where_the_brush_went() {
    let (registry, mut history, mut doc, mesh) = wired(kreate_mesh::primitives::grid(2.0, 4, 4));
    let before = doc.mesh(mesh).unwrap().num_faces();
    run(&registry, &mut history, &mut doc, "sculpt.stroke", &[
        ("dabs", dab(0.0, 0.0, 0.6)),
        ("dyntopo", PropertyValue::Bool(true)),
        ("detail", PropertyValue::Float(0.2)),
    ])
    .unwrap();
    assert!(doc.mesh(mesh).unwrap().num_faces() > before * 3);
    assert!(history.undo(&mut doc));
    assert_eq!(doc.mesh(mesh).unwrap().num_faces(), before);
}

#[test]
fn pen_pressure_softens_a_dab() {
    let height = |pressure: f64| {
        let (registry, mut history, mut doc, mesh) = wired(kreate_mesh::primitives::grid(2.0, 8, 8));
        run(&registry, &mut history, &mut doc, "sculpt.stroke", &[
            ("dabs", dab(0.0, 0.0, 0.8)),
            ("strength", PropertyValue::Float(1.0)),
            ("pressures", PropertyValue::Floats(vec![pressure])),
        ])
        .unwrap();
        top(&doc, mesh)
    };
    assert!(height(0.2) < height(1.0) * 0.5);
}
