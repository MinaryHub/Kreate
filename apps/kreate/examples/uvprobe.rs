fn main() {
    let p = "/Users/smic/Library/Caches/Kreate/assets/polyhaven/models/ArmChair_01/1k/ArmChair_01_1k.gltf";
    let (gdoc, buffers, _) = gltf::import(p).unwrap();
    // Every file corner: position in Kreate space (Y up -> Z up) and UV in Kreate's origin.
    let mut corners: Vec<([f32; 3], [f32; 2])> = Vec::new();
    for mesh in gdoc.meshes() {
        for prim in mesh.primitives() {
            let r = prim.reader(|b| Some(&buffers[b.index()]));
            let pos: Vec<[f32; 3]> = r.read_positions().unwrap().collect();
            let uv: Vec<[f32; 2]> = r.read_tex_coords(0).unwrap().into_f32().collect();
            for (p, u) in pos.iter().zip(&uv) {
                corners.push(([p[0], -p[2], p[1]], [u[0], 1.0 - u[1]]));
            }
        }
    }
    let s = kreate_io::gltf_import::parse_gltf_file(std::path::Path::new(p), &Default::default())
        .unwrap();
    let check = |label: &str, doc: &kreate_scene::Document| {
        for (_, o) in doc.objects() {
            let kreate_scene::document::ObjectData::Mesh(id) = o.data else {
                continue;
            };
            let mesh = doc.mesh(id).unwrap();
            let layer = mesh
                .attributes()
                .get(kreate_mesh::attribute::names::UV)
                .expect("uv");
            let kreate_mesh::attribute::AttrData::F32x2(uv) = &layer.data else {
                panic!()
            };
            let pos = mesh.positions();
            let (mut bad, mut total, mut at) = (0, 0, 0usize);
            for face in mesh.faces() {
                for (k, v) in face.iter().enumerate() {
                    total += 1;
                    let p = pos[*v as usize];
                    let u = uv[at + k];
                    let ok = corners.iter().any(|(cp, cu)| {
                        (cp[0] - p.x).abs() < 1e-4
                            && (cp[1] - p.y).abs() < 1e-4
                            && (cp[2] - p.z).abs() < 1e-4
                            && (cu[0] - u.x).abs() < 1e-4
                            && (cu[1] - u.y).abs() < 1e-4
                    });
                    if !ok {
                        bad += 1;
                    }
                }
                at += face.len();
            }
            println!(
                "{label}: {total} corners, {bad} without a matching file corner; domain {:?}",
                layer.domain
            );
        }
    };
    let mut doc = kreate_scene::Document::new();
    kreate_io::build_into(&s, &mut doc);
    check("built", &doc);
    let text = kreate_io::clipboard::encode(&doc).unwrap();
    let back = kreate_io::clipboard::decode(&text).unwrap();
    check("pasted", &back);
}
