//! Every reader, fed files that have gone wrong: a sound file of each
//! format, damaged a few hundred ways, must come back as an error or a
//! scene — never as a panic.
//!
//! # Why a reader must not panic
//!
//! A file from outside is the one input the program does not control. The
//! readers are built to refuse what they cannot read, and the round-trip
//! tests prove they read what the writers write; what neither proves is
//! what happens to a file that is *nearly* right — a count that says a
//! million where there are ten, an index past the end, a chunk cut short,
//! a number that is `nan`. A panic there takes the document down with the
//! file, and a crash on open is the one bug every user meets.
//!
//! # What this is, and is not
//!
//! A mutation test: each seed is damaged by a fixed, seeded sequence of
//! bit flips, truncations, deletions, duplications, insertions and — for
//! text — numbers swapped for the awkward ones. The sequence is the same
//! every run, so a failure here is a failure next time too, and the bytes
//! that caused it are written out to be looked at. It is not coverage-guided
//! fuzzing, which finds what this cannot: that wants `cargo fuzz` and a
//! nightly compiler, and runs for hours rather than seconds. This is the
//! part of it that fits in a test.

use kreate_math::DAffine3;
use kreate_mesh::primitives;
use kreate_scene::document::{Object, ObjectData};
use kreate_scene::Document;
use std::io::Cursor;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// How many damaged copies of each seed are read. A few hundred keeps the
/// whole file under a few seconds; `MUTATION_ROUNDS=100000` makes a longer
/// hunt of it by hand, which is what a bug the first few hundred miss
/// takes.
fn rounds() -> u32 {
    std::env::var("MUTATION_ROUNDS")
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(300)
}

/// A reader, as the harness calls it: the bytes, nothing back. What it
/// returns is its own business; that it returns is the test.
type Reader = Box<dyn Fn(&[u8])>;

/// A sound file of one format and the reader for it.
struct Seed {
    name: &'static str,
    bytes: Vec<u8>,
    /// Text formats also get their numbers swapped; binary ones would only
    /// be corrupted by that, which the byte mutations do already.
    text: bool,
    read: Reader,
}

/// xorshift64*: enough randomness for damage, with no dependency and the
/// same sequence on every machine.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next() % n as u64) as usize
        }
    }
}

/// The awkward numbers: past every bound, negative where a count is
/// expected, not a number at all.
const NUMBERS: &[&str] = &[
    "0",
    "-1",
    "4294967295",
    "4294967296",
    "18446744073709551615",
    "-9223372036854775808",
    "1e308",
    "1e-308",
    "nan",
    "inf",
    "-inf",
    "999999999999999999999",
    "0.000000000000000000001",
    "2147483647",
    "65535",
];

/// One damaged copy of `seed`.
fn damage(seed: &[u8], text: bool, rng: &mut Rng) -> Vec<u8> {
    let mut b = seed.to_vec();
    let ops = 1 + rng.below(3);
    for _ in 0..ops {
        if b.is_empty() {
            b.push(rng.next() as u8);
            continue;
        }
        let kind = rng.below(if text { 8 } else { 6 });
        let at = rng.below(b.len());
        match kind {
            // Bits flipped, one to eight of them.
            0 => {
                for _ in 0..1 + rng.below(8) {
                    let i = rng.below(b.len());
                    b[i] ^= 1 << rng.below(8);
                }
            }
            // A byte set to anything.
            1 => b[at] = rng.next() as u8,
            // Cut short.
            2 => b.truncate(at),
            // A run taken out.
            3 => {
                let end = (at + 1 + rng.below(64)).min(b.len());
                b.drain(at..end);
            }
            // A run said twice.
            4 => {
                let end = (at + 1 + rng.below(64)).min(b.len());
                let run = b[at..end].to_vec();
                b.splice(end..end, run);
            }
            // Noise put in.
            5 => {
                let noise: Vec<u8> = (0..1 + rng.below(32)).map(|_| rng.next() as u8).collect();
                b.splice(at..at, noise);
            }
            // A number swapped for an awkward one.
            6 | 7 => {
                let digits: Vec<(usize, usize)> = number_runs(&b);
                if let Some(&(start, end)) = digits.get(rng.below(digits.len().max(1))) {
                    if end <= b.len() {
                        let n = NUMBERS[rng.below(NUMBERS.len())].as_bytes().to_vec();
                        b.splice(start..end, n);
                    }
                }
            }
            _ => unreachable!(),
        }
    }
    b
}

/// Where the numbers are in a text: each run of digits, with a sign and a
/// point if they are there.
fn number_runs(b: &[u8]) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i].is_ascii_digit() {
            let start = if i > 0 && b[i - 1] == b'-' { i - 1 } else { i };
            let mut end = i;
            while end < b.len() && (b[end].is_ascii_digit() || b[end] == b'.') {
                end += 1;
            }
            runs.push((start, end));
            i = end;
        } else {
            i += 1;
        }
    }
    runs
}

fn lossy(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// The seeds: one sound file of each format Kreate reads, made by its own
/// writer where there is one, written here where there is not.
fn seeds() -> Vec<Seed> {
    let cube = primitives::cube(1.0);
    let sphere = primitives::uv_sphere(0.5, 8, 4);
    let parts = [
        ("cube".to_string(), &cube, DAffine3::IDENTITY),
        (
            "sphere".to_string(),
            &sphere,
            DAffine3::from_translation(kreate_math::DVec3::new(2.0, 0.0, 0.0)),
        ),
    ];
    let meshes = [("cube".to_string(), &cube), ("sphere".to_string(), &sphere)];
    let mut seeds: Vec<Seed> = Vec::new();

    // glTF, text and binary.
    for (name, binary) in [("gltf", false), ("glb", true)] {
        let options = kreate_io::gltf_export::GltfExportOptions {
            binary,
            ..Default::default()
        };
        let (bytes, _) = kreate_io::gltf_export::write_gltf(&meshes, &[], &options);
        seeds.push(Seed {
            name,
            bytes,
            text: !binary,
            read: Box::new(|b| {
                let _ = kreate_io::gltf_import::parse_gltf(
                    b,
                    &kreate_io::gltf_import::GltfOptions::default(),
                );
            }),
        });
    }
    seeds.push(Seed {
        name: "obj",
        bytes: kreate_io::obj::write_obj(&meshes, kreate_io::obj::ObjOptions::default())
            .into_bytes(),
        text: true,
        read: Box::new(|b| {
            let _ = kreate_io::obj::parse_obj(&lossy(b), kreate_io::obj::ObjOptions::default());
        }),
    });
    let (stl, _) = kreate_io::stl::write_stl(&cube, &kreate_io::stl::StlOptions::default());
    seeds.push(Seed {
        name: "stl",
        bytes: stl,
        text: false,
        read: Box::new(|b| {
            let _ = kreate_io::stl::parse_stl(b, &kreate_io::stl::StlOptions::default());
        }),
    });
    seeds.push(Seed {
        name: "step",
        bytes: kreate_io::step::write_step("part", &parts, 1e-3).into_bytes(),
        text: true,
        read: Box::new(|b| {
            let _ = kreate_io::step_read::parse_step(&lossy(b), "part");
        }),
    });
    seeds.push(Seed {
        name: "3mf",
        bytes: kreate_io::threemf::write_3mf(&parts, 1e-3),
        text: false,
        read: Box::new(|b| {
            let _ = kreate_io::threemf::parse_3mf(b);
        }),
    });
    seeds.push(Seed {
        name: "usda",
        bytes: USDA.as_bytes().to_vec(),
        text: true,
        read: Box::new(|b| {
            let _ = kreate_io::usd_read::parse_usda_text(&lossy(b));
        }),
    });
    let layer = kreate_io::usda::parse_usda(USDA).expect("the sample stage parses");
    seeds.push(Seed {
        name: "usdc",
        bytes: kreate_io::usdc_write::write_usdc(&layer),
        text: false,
        read: Box::new(|b| {
            let _ = kreate_io::usdc::parse_usdc(b);
        }),
    });
    seeds.push(Seed {
        name: "fbx",
        bytes: FBX.as_bytes().to_vec(),
        text: true,
        read: Box::new(|b| {
            let _ = kreate_io::fbx::parse_fbx(b, &|_| None);
        }),
    });
    seeds.push(Seed {
        name: "ply",
        bytes: PLY.as_bytes().to_vec(),
        text: true,
        read: Box::new(|b| {
            let _ = kreate_io::ply::parse_ply(b, "ply");
        }),
    });
    seeds.push(Seed {
        name: "off",
        bytes: OFF.as_bytes().to_vec(),
        text: true,
        read: Box::new(|b| {
            let _ = kreate_io::off::parse_off(&lossy(b), "off");
        }),
    });
    seeds.push(Seed {
        name: "dae",
        bytes: DAE.as_bytes().to_vec(),
        text: true,
        read: Box::new(|b| {
            let _ = kreate_io::dae::parse_dae(&lossy(b), &|_| None);
        }),
    });
    seeds.push(Seed {
        name: "kmesh",
        bytes: kreate_io::kmesh::write(&cube, &[]),
        text: false,
        read: Box::new(|b| {
            let _ = kreate_io::kmesh::read(b);
        }),
    });
    let mut doc = Document::new();
    let mesh = doc.add_mesh(primitives::cube(1.0));
    doc.add_object(Object::new("cube", ObjectData::Mesh(mesh)));
    let kreate = kreate_io::container::write(&doc, &kreate_io::container::Manifest::default(), &[])
        .expect("writes");
    seeds.push(Seed {
        name: "kreate",
        bytes: kreate,
        text: false,
        read: Box::new(|b| {
            let _ = kreate_io::container::read(Cursor::new(b.to_vec()));
        }),
    });
    // Older files too: what a reader meets most is not what its own writer
    // made yesterday.
    for (name, file) in [
        ("kreate-v1", "golden-v1.kreate"),
        ("kmesh-v1", "golden-v1.kmesh"),
    ] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/assets")
            .join(file);
        let bytes = std::fs::read(&path).expect("a golden file");
        let read: Reader = if file.ends_with(".kmesh") {
            Box::new(|b| {
                let _ = kreate_io::kmesh::read(b);
            })
        } else {
            Box::new(|b| {
                let _ = kreate_io::container::read(Cursor::new(b.to_vec()));
            })
        };
        seeds.push(Seed {
            name,
            bytes,
            text: false,
            read,
        });
    }
    seeds
}

/// Reads every damaged copy of every seed; a panic is a failure, with the
/// bytes that caused it saved beside the build to be looked at.
#[test]
fn no_damaged_file_panics_a_reader() {
    let seeds = seeds();
    // Each seed is read sound first, so a seed this harness got wrong is
    // told apart from a reader that fails on damage.
    for seed in &seeds {
        let sound = catch_unwind(AssertUnwindSafe(|| (seed.read)(&seed.bytes)));
        assert!(sound.is_ok(), "{}: the sound seed itself panics", seed.name);
    }
    // The panic hook prints every panic; several hundred reads of broken
    // files would print the stack of each. The failure is reported below,
    // once, with the bytes.
    // `MUTATION_LOUD=1` keeps the hook, to see a panic as it happens.
    let hook = std::panic::take_hook();
    if std::env::var_os("MUTATION_LOUD").is_none() {
        std::panic::set_hook(Box::new(|_| {}));
    }
    let mut failures = Vec::new();
    let trace = std::env::var_os("MUTATION_TRACE").is_some();
    let current =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/mutation-current.bin");
    for (i, seed) in seeds.iter().enumerate() {
        let mut rng =
            Rng(0x9E37_79B9_7F4A_7C15 ^ (i as u64 + 1).wrapping_mul(0x1234_5678_9ABC_DEF1));
        for round in 0..rounds() {
            let bytes = damage(&seed.bytes, seed.text, &mut rng);
            // `MUTATION_TRACE=1` says what is about to be read and keeps the
            // bytes on disk first: for a read the process does not survive —
            // killed for the memory it took — the last line and that file
            // are what is left to work from.
            if trace {
                eprintln!("{} round {round}", seed.name);
                let _ = std::fs::write(&current, &bytes);
            }
            let result = catch_unwind(AssertUnwindSafe(|| (seed.read)(&bytes)));
            if let Err(panic) = result {
                let message = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "(no message)".to_string());
                let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../target/mutation-failures");
                let _ = std::fs::create_dir_all(&dir);
                let file = dir.join(format!("{}-{round}.bin", seed.name));
                let _ = std::fs::write(&file, &bytes);
                failures.push(format!(
                    "{} round {round}: {message}\n    the file: {}",
                    seed.name,
                    file.display()
                ));
                // One per seed is enough to work from.
                break;
            }
        }
    }
    std::panic::set_hook(hook);
    assert!(
        failures.is_empty(),
        "readers that panic on a damaged file:\n  {}",
        failures.join("\n  ")
    );
}

/// The damage is real damage: most copies differ from the seed, and the
/// harness itself never panics making them.
#[test]
fn the_damage_changes_the_file() {
    let seed = b"solid x\n  facet normal 0 0 1\n    outer loop\n      vertex 0 0 0\n      vertex 1 0 0\n      vertex 0 1 0\n    endloop\n  endfacet\nendsolid x\n";
    let mut rng = Rng(7);
    let changed = (0..100)
        .filter(|_| damage(seed, true, &mut rng) != seed.to_vec())
        .count();
    assert!(changed > 90, "{changed} of 100 copies were damaged");
    let mut rng = Rng(8);
    for _ in 0..200 {
        let _ = damage(b"", false, &mut rng);
        let _ = damage(b"x", true, &mut rng);
    }
}

const USDA: &str = r#"#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 0.01
    upAxis = "Y"
)

def Xform "World"
{
    double3 xformOp:translate = (1, 2, 3)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Mesh "Tri"
    {
        point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)]
        int[] faceVertexCounts = [3, 3]
        int[] faceVertexIndices = [0, 1, 2, 1, 3, 2]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (0, 1), (1, 1)] (
            interpolation = "vertex"
        )
    }
}
"#;

const FBX: &str = r#"; FBX 7.4.0 project file
FBXHeaderExtension:  {
	FBXHeaderVersion: 1003
	Creator: "Test"
}
GlobalSettings:  {
	Version: 1000
	Properties70:  {
		P: "UpAxis", "int", "Integer", "",1
		P: "UpAxisSign", "int", "Integer", "",1
		P: "UnitScaleFactor", "double", "Number", "",100
	}
}
Objects:  {
	Geometry: 100, "Geometry::Shape", "Mesh" {
		Vertices: *15 {
			a: 0,0,0,1,0,0,1,1,0,0,1,0,0.5,2,0
		}
		PolygonVertexIndex: *7 {
			a: 0,1,2,-4,3,2,-5
		}
		LayerElementNormal: 0 {
			MappingInformationType: "ByVertice"
			ReferenceInformationType: "Direct"
			Normals: *15 {
				a: 0,0,1,0,0,1,0,0,1,0,0,1,0,0,1
			}
		}
		LayerElementUV: 0 {
			MappingInformationType: "ByPolygonVertex"
			ReferenceInformationType: "IndexToDirect"
			UV: *4 {
				a: 0,0,1,1
			}
			UVIndex: *7 {
				a: 0,1,1,0,0,1,1
			}
		}
		LayerElementMaterial: 0 {
			MappingInformationType: "ByPolygon"
			ReferenceInformationType: "IndexToDirect"
			Materials: *2 {
				a: 1,0
			}
		}
	}
	Model: 200, "Model::Root", "Null" {
		Properties70:  {
			P: "Lcl Translation", "Lcl Translation", "", "A",1,2,3
		}
	}
	Model: 300, "Model::Shape", "Mesh" {
		Properties70:  {
			P: "Lcl Rotation", "Lcl Rotation", "", "A",0,0,90
			P: "Lcl Scaling", "Lcl Scaling", "", "A",2,2,2
		}
	}
	Material: 400, "Material::Red", "" {
		Properties70:  {
			P: "DiffuseColor", "Color", "", "A",1,0,0
		}
	}
	Material: 500, "Material::Blue", "" {
		Properties70:  {
			P: "DiffuseColor", "Color", "", "A",0,0,1
		}
	}
}
Connections:  {
	C: "OO",300,200
	C: "OO",200,0
	C: "OO",100,300
	C: "OO",400,300
	C: "OO",500,300
}
"#;

const PLY: &str = "ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nelement face 2\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n3 0 1 2\n3 1 3 2\n";

const OFF: &str = "OFF\n4 2 0\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n3 0 1 2\n3 1 3 2\n";

const DAE: &str = r##"<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
  <library_geometries>
    <geometry id="quad" name="quad">
      <mesh>
        <source id="quad-positions">
          <float_array id="quad-positions-array" count="12">0 0 0 1 0 0 0 1 0 1 1 0</float_array>
          <technique_common>
            <accessor source="#quad-positions-array" count="4" stride="3">
              <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="quad-vertices"><input semantic="POSITION" source="#quad-positions"/></vertices>
        <triangles count="2">
          <input semantic="VERTEX" source="#quad-vertices" offset="0"/>
          <p>0 1 2 1 3 2</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="Scene" name="Scene">
      <node id="Quad" name="Quad" type="NODE">
        <matrix sid="transform">1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1</matrix>
        <instance_geometry url="#quad"/>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>
"##;
