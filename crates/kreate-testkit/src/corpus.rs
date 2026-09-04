//! The adversarial mesh corpus.
//!
//! Every specimen is here because something once went wrong on a mesh shaped
//! like it, or because the plan names it as a hazard. Each carries a `hazard`
//! string saying what it is for, so a failure report names the problem rather
//! than a filename.
//!
//! This is deliberately small. A corpus of two hundred meshes is a corpus nobody
//! reads and whose failures nobody diagnoses; each of these earns its place by
//! breaking something different.

use glam::{Vec2, Vec3};
use kreate_mesh::attribute::{names, AttrData, Domain};
use kreate_mesh::MeshData;

pub struct Specimen {
    /// Used in test names and golden filenames, so it is stable and ASCII even
    /// when the mesh's own object name is not.
    pub name: &'static str,
    /// What this specimen is expected to break.
    pub hazard: &'static str,
    pub mesh: MeshData,
}

/// The full corpus, in a fixed order.
pub fn corpus() -> Vec<Specimen> {
    vec![
        Specimen {
            name: "cube_quads",
            hazard: "the baseline: quads must not come back as triangles",
            mesh: cube(),
        },
        Specimen {
            name: "concave_ngon",
            hazard: "an L-shaped face; a fan triangulation of it leaves the polygon",
            mesh: concave_ngon(),
        },
        Specimen {
            name: "wide_ngon",
            hazard: "a 24-gon; formats with a per-face vertex limit truncate it",
            mesh: wide_ngon(24),
        },
        Specimen {
            name: "mixed_winding",
            hazard: "adjacent faces wound oppositely; a normal-based fixer 'corrects' one",
            mesh: mixed_winding(),
        },
        Specimen {
            name: "five_uv_sets",
            hazard: "five UV layers; every format keeps one and most lose the names",
            mesh: five_uv_sets(),
        },
        Specimen {
            name: "smoothing_groups",
            hazard: "three shallow marked ribs on a smooth wall; lost flags shade them wrong",
            mesh: smoothing_groups(),
        },
        Specimen {
            name: "korean_names",
            hazard: "non-ASCII object and layer names through text formats and zip entries",
            mesh: korean_names(),
        },
        Specimen {
            name: "extreme_scale",
            hazard: "coordinates at 1e-4 and 1e5 in one mesh; f32 text formats lose the small ones",
            mesh: extreme_scale(),
        },
        Specimen {
            name: "many_materials",
            hazard: "eight materials over twelve faces; glTF splits into eight primitives",
            mesh: many_materials(),
        },
    ]
}

/// A unit cube as six quads over eight vertices, wound outwards.
pub fn cube() -> MeshData {
    let p = Vec3::new;
    MeshData::from_faces(
        vec![
            p(-1.0, -1.0, -1.0),
            p(1.0, -1.0, -1.0),
            p(1.0, 1.0, -1.0),
            p(-1.0, 1.0, -1.0),
            p(-1.0, -1.0, 1.0),
            p(1.0, -1.0, 1.0),
            p(1.0, 1.0, 1.0),
            p(-1.0, 1.0, 1.0),
        ],
        &[
            vec![0, 3, 2, 1],
            vec![4, 5, 6, 7],
            vec![0, 1, 5, 4],
            vec![1, 2, 6, 5],
            vec![2, 3, 7, 6],
            vec![3, 0, 4, 7],
        ],
    )
}

/// One L-shaped hexagonal face.
///
/// A fan from vertex 0 leaves the polygon at the reflex corner, so a
/// triangulator that fans without checking produces a face that is visibly wrong
/// and a normal that may point the other way.
fn concave_ngon() -> MeshData {
    let p = |x: f32, y: f32| Vec3::new(x, y, 0.0);
    MeshData::from_faces(
        vec![
            p(0.0, 0.0),
            p(2.0, 0.0),
            p(2.0, 1.0),
            p(1.0, 1.0),
            p(1.0, 2.0),
            p(0.0, 2.0),
        ],
        &[vec![0, 1, 2, 3, 4, 5]],
    )
}

/// A single regular n-gon.
fn wide_ngon(sides: u32) -> MeshData {
    let positions: Vec<Vec3> = (0..sides)
        .map(|i| {
            let a = i as f32 / sides as f32 * std::f32::consts::TAU;
            Vec3::new(a.cos(), a.sin(), 0.0)
        })
        .collect();
    let face: Vec<u32> = (0..sides).collect();
    MeshData::from_faces(positions, &[face])
}

/// Two triangles sharing an edge, wound in opposite directions.
///
/// Legal, authored, and not something an importer may quietly repair: a mesh
/// whose windings a tool "fixes" on load is a mesh whose author's intent was
/// overruled without being asked.
fn mixed_winding() -> MeshData {
    MeshData::from_faces(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        // The second face is the first one's reverse ordering: [0, 2, 3] would
        // face the same way, [3, 2, 0] faces the other.
        &[vec![0, 1, 2], vec![3, 2, 0]],
    )
}

/// A quad carrying five differently named UV layers.
///
/// Five because one is the easy case and two is the case everyone tests. The
/// layer names carry as much as the coordinates: a pipeline addresses a UV set
/// by name, and a format that renumbers them to `uv0..uv4` has broken the
/// material bindings even though every coordinate survived.
fn five_uv_sets() -> MeshData {
    let mut mesh = MeshData::from_faces(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ],
        &[vec![0, 1, 2, 3]],
    );
    for (i, name) in ["uv_map", "uv_lightmap", "uv_detail", "uv_decal", "uv_wear"]
        .into_iter()
        .enumerate()
    {
        // Each set offset differently, so a mix-up between them is visible
        // rather than being hidden by five identical layers.
        let d = i as f32 * 0.1;
        mesh.attributes_mut().insert(
            name,
            Domain::Corner,
            AttrData::F32x2(
                vec![
                    Vec2::new(d, d),
                    Vec2::new(1.0 - d, d),
                    Vec2::new(1.0 - d, 1.0 - d),
                    Vec2::new(d, 1.0 - d),
                ]
                .into(),
            ),
        );
    }
    mesh
}

/// A ribbed barrel: a twenty-four sided wall in four rows, with a marked crease
/// around every ring — three ribs plus both cap rims.
///
/// The shading hazard: the creases must stay creased while the rest of the wall
/// stays smooth. A format that stores one flag per face instead of per edge
/// cannot express this, and the usual failure is a cylinder that goes entirely
/// faceted or entirely smooth.
///
/// # Why a barrel and not a cylinder
///
/// Two earlier versions of this specimen tested nothing, and both passed.
///
/// A plain cylinder's only sharp edges are its cap rims, which meet the wall at
/// ninety degrees. A normal generator marks those sharp from the geometry alone,
/// so the flags there are decoration: a renderer that ignored sharp edges
/// entirely still reproduced this specimen's golden image exactly.
///
/// Adding a single vertical seam edge fixed the principle and not the test. One
/// edge touches two vertices, so removing sharp-edge handling changed three
/// corner normals out of a hundred and twenty — a few pixels, well inside the
/// comparison's tolerance. It passed too.
///
/// So: the wall alternates between two radii, and every ring is marked sharp.
/// Each rib runs the whole way round, and there are three of them, so losing the
/// flags changes three bands across every view.
///
/// # Why the rib angle is what it is
///
/// The angle at a rib has to sit in a band, and both ends of the band are real:
///
/// - **below** the auto-smooth angle, or the crease is produced by the geometry
///   and the flag is decoration again;
/// - **far enough above zero** that losing the crease moves pixels. An attempt
///   with a seventeen degree waist rotated the shaded normal by about eight
///   degrees and changed the image by less than the visual harness's per-channel
///   tolerance. It was a crease a person could see in the golden and the
///   comparison could not, so that version passed too.
///
/// About twenty-four degrees — four fifths of the thirty degree threshold — is a
/// shading step the comparison resolves. The corpus test asserts both bounds so
/// this cannot drift back to either failure.
///
/// # Why three ribs and not one
///
/// One rib at twenty-four degrees was caught, but by two images out of
/// forty-five and by a margin of 1.76% against a 1.5% budget. A margin that thin
/// is a test that a different driver's shading can switch off without anyone
/// noticing. Three ribs put the same signal across three bands of every view,
/// which turns a coin-flip into a result.
fn smoothing_groups() -> MeshData {
    const SIDES: u32 = 24;
    // Radius and height per ring. The radius alternates, which is what gives
    // consecutive wall rows a dihedral angle to disagree about; a straight
    // cylinder's rows are coplanar, and splitting normals across a crease between
    // two coplanar faces produces the same normals and so no visible crease.
    //
    // The offsets are tuned together: a row spans 0.5 in z and 0.106 in radius,
    // so it leans about twelve degrees off the axis and consecutive rows meet at
    // about twenty-four. Changing one without the other moves the rib angle out
    // of the band the corpus test asserts.
    const RINGS: [(f32, f32); 5] = [
        (1.0, -1.0),
        (1.106, -0.5),
        (1.0, 0.0),
        (1.106, 0.5),
        (1.0, 1.0),
    ];

    let mut positions = Vec::new();
    for (radius, z) in RINGS {
        for i in 0..SIDES {
            let a = i as f32 / SIDES as f32 * std::f32::consts::TAU;
            positions.push(Vec3::new(a.cos() * radius, a.sin() * radius, z));
        }
    }
    let mut faces = Vec::new();
    for row in 0..RINGS.len() as u32 - 1 {
        let lower = row * SIDES;
        let upper = (row + 1) * SIDES;
        for i in 0..SIDES {
            let j = (i + 1) % SIDES;
            faces.push(vec![lower + i, lower + j, upper + j, upper + i]);
        }
    }
    let top = RINGS.len() as u32 - 1;
    faces.push((0..SIDES).rev().collect());
    faces.push((top * SIDES..(top + 1) * SIDES).collect());

    let mut mesh = MeshData::from_faces(positions, &faces);
    mesh.rebuild_edges();

    // Sharp wherever both endpoints sit on the same ring: the two cap rims and
    // the three ribs. The rims are sharp by geometry anyway; the ribs are the
    // ones only the flag can make sharp, and they are why this rule is worth
    // having.
    let ring_of = |v: u32| v / SIDES;
    let sharp: Vec<bool> = mesh
        .edges()
        .iter()
        .map(|e| ring_of(e[0]) == ring_of(e[1]))
        .collect();
    mesh.attributes_mut().insert(
        names::SHARP_EDGE,
        Domain::Edge,
        AttrData::Bool(sharp.into()),
    );
    let smooth = vec![true; mesh.num_faces() as usize];
    mesh.attributes_mut().insert(
        names::SHADE_SMOOTH,
        Domain::Face,
        AttrData::Bool(smooth.into()),
    );
    mesh
}

/// A cube whose object name and UV layer name are Korean.
///
/// Text formats write the name into the file, zip containers write it into an
/// entry, and both have historically mangled anything outside ASCII. The name is
/// also the thing a user recognises their object by, so losing it is not
/// cosmetic.
fn korean_names() -> MeshData {
    let mut mesh = cube();
    let corners = mesh.num_corners() as usize;
    mesh.attributes_mut().insert(
        "uv_전개도",
        Domain::Corner,
        AttrData::F32x2(vec![Vec2::ZERO; corners].into()),
    );
    mesh
}

/// The name to give [`korean_names`] when a format stores one.
pub const KOREAN_OBJECT_NAME: &str = "한글 오브젝트 이름";

/// One mesh spanning nine orders of magnitude.
///
/// The hazard is not the large coordinate, it is the small one sitting next to
/// it: a text format that writes six significant figures represents 1e5 fine and
/// collapses 1e-4 to zero, and the two faces end up sharing a vertex.
fn extreme_scale() -> MeshData {
    MeshData::from_faces(
        vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1e-4, 0.0, 0.0),
            Vec3::new(1e-4, 1e-4, 0.0),
            Vec3::new(1e5, 0.0, 0.0),
            Vec3::new(1e5, 1e5, 0.0),
        ],
        &[vec![0, 1, 2], vec![0, 3, 4]],
    )
}

/// Twelve faces over eight material slots.
fn many_materials() -> MeshData {
    let mut mesh = MeshData::from_faces(
        cube().positions().to_vec(),
        &[
            vec![0, 3, 2],
            vec![0, 2, 1],
            vec![4, 5, 6],
            vec![4, 6, 7],
            vec![0, 1, 5],
            vec![0, 5, 4],
            vec![1, 2, 6],
            vec![1, 6, 5],
            vec![2, 3, 7],
            vec![2, 7, 6],
            vec![3, 0, 4],
            vec![3, 4, 7],
        ],
    );
    let materials: Vec<i32> = (0..12).map(|i| i % 8).collect();
    mesh.attributes_mut().insert(
        names::MATERIAL_INDEX,
        Domain::Face,
        AttrData::I32(materials.into()),
    );
    mesh
}

/// A flat grid of `cells * cells` quads, for the size-dependent benchmarks.
///
/// Not part of [`corpus`]: it is parameterised, and a million-quad specimen has
/// no business in a suite that runs on every commit.
pub fn dense_grid(cells: u32) -> MeshData {
    let side = cells + 1;
    let mut positions = Vec::with_capacity((side * side) as usize);
    for y in 0..side {
        for x in 0..side {
            positions.push(Vec3::new(x as f32, y as f32, 0.0));
        }
    }
    let mut faces = Vec::with_capacity((cells * cells) as usize);
    for y in 0..cells {
        for x in 0..cells {
            let i = y * side + x;
            faces.push(vec![i, i + 1, i + side + 1, i + side]);
        }
    }
    MeshData::from_faces(positions, &faces)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_specimen_is_structurally_valid() {
        // A corpus that is itself invalid tests nothing: every failure downstream
        // becomes ambiguous between "the format broke it" and "it arrived
        // broken".
        for specimen in corpus() {
            let violations = specimen.mesh.validate();
            assert!(
                violations.is_empty(),
                "{} ({}) is not valid: {violations:?}",
                specimen.name,
                specimen.hazard
            );
        }
    }

    #[test]
    fn every_specimen_has_a_distinct_name() {
        let mut names: Vec<&str> = corpus().iter().map(|s| s.name).collect();
        names.sort_unstable();
        let count = names.len();
        names.dedup();
        assert_eq!(names.len(), count);
    }

    #[test]
    fn the_hazards_are_actually_present() {
        // Each specimen claims to carry a hazard. A specimen that lost its
        // hazard in an edit still passes every round-trip test and quietly stops
        // testing anything, so the claims are checked rather than trusted.
        let all = corpus();
        let get = |name: &str| {
            all.iter()
                .find(|s| s.name == name)
                .unwrap_or_else(|| panic!("{name} is missing from the corpus"))
        };

        assert!(get("cube_quads").mesh.faces().all(|f| f.len() == 4));
        assert_eq!(get("wide_ngon").mesh.face_len(0), 24);
        assert!(
            get("concave_ngon").mesh.face_len(0) > 4,
            "a concave n-gon needs more than four corners to be concave"
        );

        // Mixed winding: the two faces must disagree about which way is out.
        let mixed = &get("mixed_winding").mesh;
        let normals = mixed.compute_face_normals();
        assert!(
            normals[0].normalize().dot(normals[1].normalize()) < -0.9,
            "the two faces are wound the same way: {normals:?}"
        );

        let uvs = get("five_uv_sets")
            .mesh
            .attributes()
            .iter()
            .filter(|l| l.name.starts_with("uv"))
            .count();
        assert_eq!(uvs, 5);

        // Sharp edges: some but not all, or the specimen is uniformly shaded and
        // proves nothing.
        let prism = &get("smoothing_groups").mesh;
        let sharp = prism
            .attributes()
            .get_bool(names::SHARP_EDGE)
            .expect("sharp edges");
        assert!(sharp.iter().any(|&s| s) && sharp.iter().any(|&s| !s));

        // And the wall has to be smooth *by angle*, not merely by not being
        // marked sharp. A normal generator averages faces within an auto-smooth
        // angle, so a wall whose facets are further apart than that angle comes
        // out faceted whatever the sharp flags say — which is what a twelve-sided
        // prism did, sitting exactly on the renderer's thirty-degree default.
        //
        // The angle is checked here rather than imported, because `kreate-mesh`
        // may not depend on the renderer. The coupling is real, so it is written
        // down and asserted with margin instead of being left implicit.
        const RENDERER_AUTO_SMOOTH_DEGREES: f32 = 30.0;
        // One row of the wall, so neighbours are compared within a row rather
        // than across a rib — the ribs are deliberately steeper.
        const SIDES_IN_CORPUS_PRISM: usize = 24;
        let normals = prism.compute_face_normals();
        // The wall faces only; the two caps are last and are meant to be sharp.
        let wall = &normals[..normals.len() - 2];
        // Neighbours *within* a row. Comparing across a row boundary would pick
        // up a rib, which is deliberately steeper.
        let widest = wall
            .chunks(SIDES_IN_CORPUS_PRISM)
            .flat_map(|row| {
                row.windows(2).map(|w| {
                    w[0].normalize()
                        .dot(w[1].normalize())
                        .clamp(-1.0, 1.0)
                        .acos()
                })
            })
            .fold(0.0f32, f32::max)
            .to_degrees();
        assert!(
            widest < RENDERER_AUTO_SMOOTH_DEGREES * 0.75,
            "adjacent wall faces are {widest:.1} degrees apart, too close to the \
             {RENDERER_AUTO_SMOOTH_DEGREES} degree auto-smooth threshold for the \
             wall to be reliably smooth"
        );

        // And at least one sharp edge has to be *shallow* — sharp because it is
        // marked, not because its faces disagree. Without one, the sharp flags
        // are redundant with the angle test and the specimen tests nothing about
        // them: a renderer that ignored the flags outright still matched this
        // specimen's golden image exactly.
        let mut shallow_sharp = 0;
        let mut shallow_angles: Vec<f32> = Vec::new();
        for (edge, &is_sharp) in prism.edges().iter().zip(sharp) {
            if !is_sharp {
                continue;
            }
            let touching: Vec<usize> = (0..prism.num_faces() as usize)
                .filter(|&f| {
                    let verts = prism.face_verts(f as u32);
                    verts.contains(&edge[0]) && verts.contains(&edge[1])
                })
                .collect();
            if touching.len() != 2 {
                continue;
            }
            let a = normals[touching[0]].normalize();
            let b = normals[touching[1]].normalize();
            let angle = a.dot(b).clamp(-1.0, 1.0).acos().to_degrees();
            if angle < RENDERER_AUTO_SMOOTH_DEGREES * 0.9 {
                shallow_sharp += 1;
                shallow_angles.push(angle);
            }
        }
        assert!(
            shallow_sharp > 0,
            "every sharp edge in the prism is also sharp by angle, so the sharp \
             flags are not being tested by it"
        );

        // And the crease has to be steep enough to see. The waist is the widest
        // angle among the sharp edges that are shallow; too small and the shading
        // step it creates is finer than the visual harness's tolerance, which is
        // a crease that is real, visible to a person, and untested.
        let waist = shallow_angles.iter().fold(0.0f32, |a, &b| a.max(b));
        assert!(
            waist > RENDERER_AUTO_SMOOTH_DEGREES * 0.5,
            "the marked ribs are only {waist:.1} degrees; too shallow for losing \
             them to change the rendered image measurably"
        );
        assert!(
            waist < RENDERER_AUTO_SMOOTH_DEGREES * 0.85,
            "the marked ribs are {waist:.1} degrees, close enough to the \
             {RENDERER_AUTO_SMOOTH_DEGREES} degree threshold that the geometry \
             may create them without the flag"
        );

        assert!(get("korean_names")
            .mesh
            .attributes()
            .iter()
            .any(|l| !l.name.is_ascii()));

        let bounds = get("extreme_scale").mesh.bounds();
        assert!(bounds.max.x > 1e4);

        let materials = get("many_materials")
            .mesh
            .attributes()
            .get_i32(names::MATERIAL_INDEX)
            .expect("materials");
        let mut distinct = materials.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), 8);
    }

    #[test]
    fn the_dense_grid_is_the_size_it_says() {
        let grid = dense_grid(10);
        assert_eq!(grid.num_faces(), 100);
        assert_eq!(grid.num_verts(), 121);
        assert!(grid.validate().is_empty());
    }
}
