//! The `.kmesh` format's guarantees, each stated as a test.
//!
//! Four of them cannot be added later, which is why they are pinned now rather
//! than when something depends on them:
//!
//! - **Round trip.** What goes in comes out, attributes and all.
//! - **Determinism.** The same mesh writes the same bytes, so version control
//!   stores one blob for an unchanged file and a golden test means something.
//! - **Forward compatibility.** A file from a newer build, opened and saved by
//!   an older one, keeps what the older one could not read. Add this rule after
//!   files exist and every file written before it is already lossy.
//! - **Safety against a corrupt file.** The format's whole premise is that
//!   loading is a cast, which is exactly the shape that turns a bad file into an
//!   out-of-bounds read.

use kreate_io::kmesh::{self, UnknownBuffer};
use kreate_mesh::attribute::{names, AttrData, AttrType, Domain};
use kreate_mesh::MeshData;
use std::sync::Arc;

/// A mesh with something in every attribute domain and every element type.
///
/// Deliberately not a bare cube: a round-trip test on positions alone would pass
/// with the entire attribute system broken, and the attribute system is what
/// makes the format worth having.
fn rich_mesh() -> MeshData {
    let mut mesh = kreate_mesh::primitives::cube(2.0);
    let verts = mesh.num_verts() as usize;
    let corners = mesh.corner_verts().len();
    let faces = mesh.num_faces() as usize;
    let edges = mesh.num_edges() as usize;

    mesh.attributes_mut().insert(
        "uv0",
        Domain::Corner,
        AttrData::F32x2(Arc::new(
            (0..corners)
                .map(|i| glam::Vec2::new(i as f32 * 0.25, i as f32 * -0.5))
                .collect(),
        )),
    );
    mesh.attributes_mut().insert(
        "material_index",
        Domain::Face,
        AttrData::I32(Arc::new((0..faces).map(|i| i as i32 % 3).collect())),
    );
    mesh.attributes_mut().insert(
        "crease",
        Domain::Edge,
        AttrData::F32(Arc::new((0..edges).map(|i| i as f32 * 0.1).collect())),
    );
    mesh.attributes_mut().insert(
        names::SELECT_VERT,
        Domain::Point,
        AttrData::Bool(Arc::new((0..verts).map(|i| i % 3 == 0).collect())),
    );
    mesh.attributes_mut().insert(
        "color",
        Domain::Point,
        AttrData::U8x4(Arc::new(
            (0..verts).map(|i| [i as u8, 7, 200, 255]).collect(),
        )),
    );
    mesh.attributes_mut().insert(
        "weight",
        Domain::Point,
        AttrData::F32x4(Arc::new(
            (0..verts)
                .map(|i| glam::Vec4::splat(i as f32 * 0.125))
                .collect(),
        )),
    );
    mesh
}

fn assert_same(a: &MeshData, b: &MeshData) {
    assert_eq!(a.num_verts(), b.num_verts(), "vertex count");
    assert_eq!(a.face_offsets(), b.face_offsets(), "face offsets");
    assert_eq!(a.corner_verts(), b.corner_verts(), "corner verts");
    assert_eq!(a.edges(), b.edges(), "edges");

    let mut left: Vec<_> = a
        .attributes()
        .iter()
        .map(|l| (l.name.to_string(), l.domain, l.data.clone()))
        .collect();
    let mut right: Vec<_> = b
        .attributes()
        .iter()
        .map(|l| (l.name.to_string(), l.domain, l.data.clone()))
        .collect();
    left.sort_by(|x, y| x.0.cmp(&y.0));
    right.sort_by(|x, y| x.0.cmp(&y.0));
    assert_eq!(left.len(), right.len(), "attribute count");
    for (l, r) in left.iter().zip(&right) {
        assert_eq!(l.0, r.0, "attribute name");
        assert_eq!(l.1, r.1, "domain of {}", l.0);
        assert_eq!(l.2.ty(), r.2.ty(), "type of {}", l.0);
        assert_eq!(l.2.len(), r.2.len(), "length of {}", l.0);
        // Compared as bytes: `AttrData` holds an `Arc<Vec<_>>` per variant and
        // has no equality, and writing one by hand for a test would be a second
        // definition of what "the same attribute" means.
        assert_eq!(
            format!("{:?}", l.2),
            format!("{:?}", r.2),
            "values of {}",
            l.0
        );
    }
}

#[test]
fn a_mesh_survives_a_round_trip_with_every_attribute() {
    let mesh = rich_mesh();
    let bytes = kmesh::write(&mesh, &[]);
    let loaded = kmesh::read(&bytes).expect("reads back");
    assert_same(&mesh, &loaded.mesh);
    assert!(loaded.unknown.is_empty());
}

#[test]
fn writing_the_same_mesh_twice_produces_the_same_bytes() {
    // Determinism, which is what lets version control store one blob for an
    // unchanged file. Two separately built copies of the same mesh, not the same
    // object written twice — the latter would pass even if the writer depended
    // on the store's insertion order, which differs between two documents that
    // arrived at the same mesh by different routes.
    let a = kmesh::write(&rich_mesh(), &[]);
    let b = kmesh::write(&rich_mesh(), &[]);
    assert_eq!(a, b, "two writes of the same mesh differ");
}

#[test]
fn attribute_order_in_the_store_does_not_change_the_file() {
    // A file whose bytes depended on the order attributes happened to be
    // inserted would show a diff on every save even when nothing changed.
    //
    // Worth stating even though `AttributeStore` is what guarantees it — it
    // keeps its layers sorted by name, so the writer inherits a canonical order
    // rather than establishing one. The guarantee belongs to the format, so it
    // is asserted at the format; if the store ever stopped being sorted, this is
    // where that would surface.
    let mut first = kreate_mesh::primitives::cube(2.0);
    let verts = first.num_verts() as usize;
    let mut second = first.clone();

    let a = AttrData::F32(Arc::new(vec![1.0; verts]));
    let b = AttrData::F32(Arc::new(vec![2.0; verts]));
    first
        .attributes_mut()
        .insert("alpha", Domain::Point, a.clone());
    first
        .attributes_mut()
        .insert("beta", Domain::Point, b.clone());
    second.attributes_mut().insert("beta", Domain::Point, b);
    second.attributes_mut().insert("alpha", Domain::Point, a);

    assert_eq!(kmesh::write(&first, &[]), kmesh::write(&second, &[]));
}

#[test]
fn every_buffer_is_aligned_so_it_can_be_cast_in_place() {
    // The premise of the whole format. If a buffer landed at an odd offset, a
    // memory-mapped reader could not cast it and would have to copy — which is
    // the cost this layout exists to avoid.
    //
    // Checked by reading the file back through the parser, which rejects a
    // misaligned buffer; a file that round-trips has aligned buffers by
    // construction. Also checked directly against the raw table, so a parser
    // that stopped checking would not take this test with it.
    let bytes = kmesh::write(&rich_mesh(), &[]);
    for offset in buffer_offsets(&bytes) {
        assert_eq!(offset % 16, 0, "buffer at {offset} is not 16-byte aligned");
    }
    assert!(kmesh::read(&bytes).is_ok());
}

/// Pulls the buffer lengths straight out of the table, without the parser.
fn buffer_lengths(bytes: &[u8]) -> Vec<u64> {
    const HEADER: usize = 64;
    const ENTRY: usize = 48;
    let count = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
    (0..count)
        .map(|i| {
            let at = HEADER + i * ENTRY + 24;
            u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
        })
        .collect()
}

/// Pulls the buffer offsets straight out of the table, without the parser.
fn buffer_offsets(bytes: &[u8]) -> Vec<u64> {
    const HEADER: usize = 64;
    const ENTRY: usize = 48;
    let count = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
    (0..count)
        .map(|i| {
            let at = HEADER + i * ENTRY + 16;
            u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap())
        })
        .collect()
}

#[test]
fn the_padding_between_buffers_is_zeroed() {
    // Alignment leaves gaps, and what is in them matters twice over: padding
    // that varied would make two writes of one mesh differ, and padding taken
    // from whatever the allocator last held would put arbitrary process memory
    // into a file the user may send to someone else.
    //
    // Checked directly rather than inferred from the determinism test, which a
    // deterministic *pattern* would also satisfy.
    let bytes = kmesh::write(&rich_mesh(), &[]);
    let mut covered = vec![false; bytes.len()];
    // The header, table and name blob are all meaningful.
    let names_offset = u32::from_le_bytes(bytes[40..44].try_into().unwrap()) as usize;
    let names_length = u32::from_le_bytes(bytes[44..48].try_into().unwrap()) as usize;
    for byte in covered.iter_mut().take(names_offset + names_length) {
        *byte = true;
    }
    for (offset, length) in buffer_offsets(&bytes).iter().zip(buffer_lengths(&bytes)) {
        for byte in covered
            .iter_mut()
            .skip(*offset as usize)
            .take(length as usize)
        {
            *byte = true;
        }
    }

    let mut gaps = 0;
    for (i, byte) in bytes.iter().enumerate() {
        if !covered[i] {
            gaps += 1;
            assert_eq!(*byte, 0, "padding byte at {i} is {byte}, not zero");
        }
    }
    assert!(gaps > 0, "the file has no padding, so this proved nothing");
}

#[test]
fn a_buffer_from_a_newer_writer_survives_being_saved_by_this_one() {
    // The forward-compatibility rule, in the exact shape that matters: a file
    // written by a future build, opened here and saved again, must still contain
    // what this build could not interpret.
    //
    // Without it, one save in an older version is silent data loss — the user
    // sees no error, and whatever the newer build stored is simply gone.
    let mesh = rich_mesh();
    let future = UnknownBuffer {
        name: "$something_from_the_future".to_string(),
        // A domain and type this build has no idea about.
        domain: 99,
        kind: 4242,
        count: 5,
        bytes: vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
    };

    let written = kmesh::write(&mesh, std::slice::from_ref(&future));
    let loaded = kmesh::read(&written).expect("reads");
    assert_eq!(loaded.unknown, vec![future.clone()], "the buffer was lost");

    // And saving it again keeps it, byte for byte.
    let again = kmesh::write(&loaded.mesh, &loaded.unknown);
    let reloaded = kmesh::read(&again).expect("reads again");
    assert_eq!(reloaded.unknown, vec![future]);
    assert_same(&mesh, &reloaded.mesh);
}

#[test]
fn a_file_that_demands_a_newer_reader_is_refused() {
    // `min_reader_major` is the file's own statement about who may read it. A
    // reader that ignored it would not fail — it would succeed and produce
    // something wrong, which is worse.
    let mut bytes = kmesh::write(&rich_mesh(), &[]);
    // The field sits after magic, major and minor.
    bytes[16..20].copy_from_slice(&99u32.to_le_bytes());
    match kmesh::read(&bytes) {
        Err(kmesh::KmeshError::TooNew { needed }) => assert_eq!(needed, 99),
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn a_newer_minor_version_is_still_readable() {
    // The other half of the versioning rule: minor bumps are additive and must
    // not lock an older reader out. A reader that refused them would turn every
    // additive change into a compatibility break.
    let mut bytes = kmesh::write(&rich_mesh(), &[]);
    bytes[12..16].copy_from_slice(&7u32.to_le_bytes());
    let loaded = kmesh::read(&bytes).expect("a newer minor version must still open");
    assert_eq!(loaded.version_minor, 7);
}

#[test]
fn a_file_without_magic_is_refused() {
    let mut bytes = kmesh::write(&rich_mesh(), &[]);
    bytes[0] = b'X';
    assert!(matches!(
        kmesh::read(&bytes),
        Err(kmesh::KmeshError::BadMagic)
    ));
}

#[test]
fn every_truncation_of_a_valid_file_is_an_error_and_never_a_panic() {
    // The safety property for a format whose loading step is a cast. A
    // truncated file is the commonest kind of corruption — an interrupted save,
    // a full disk, a partial download — and every prefix has to come back as an
    // error rather than as an out-of-bounds read.
    //
    // Every prefix, not a sample: the interesting ones are exactly at the
    // boundaries between header, table, names and payload, and a sample would
    // step over them.
    let bytes = kmesh::write(&rich_mesh(), &[]);
    // The file ends with padding to a 16-byte boundary, and that padding carries
    // nothing — a prefix that keeps every buffer but drops the trailing zeroes
    // is a complete file, and accepting it is correct rather than lax. The first
    // version of this test demanded that *every* prefix fail and caught the
    // writer being tidy rather than the reader being unsafe.
    let last_byte_that_matters = buffer_offsets(&bytes)
        .iter()
        .zip(buffer_lengths(&bytes))
        .map(|(offset, length)| offset + length)
        .max()
        .expect("the file has buffers") as usize;

    for len in 0..last_byte_that_matters {
        let result = kmesh::read(&bytes[..len]);
        assert!(
            result.is_err(),
            "a {len}-byte prefix of a {}-byte file was accepted, but its last              buffer does not end until {last_byte_that_matters}",
            bytes.len()
        );
    }
    assert!(kmesh::read(&bytes).is_ok());
}

#[test]
fn corrupting_any_single_byte_of_the_header_or_table_never_panics() {
    // Fuzzing in miniature, over the region where a bad value does damage: the
    // header and the table hold every offset and length the reader trusts.
    //
    // Some corruptions produce a valid file with different contents, so the
    // assertion is not "this must fail" but "this must not crash" — which is
    // the property that matters when the input came from somewhere else.
    let original = kmesh::write(&rich_mesh(), &[]);
    let table_end = 64 + 48 * 9;
    for i in 0..table_end.min(original.len()) {
        for bit in [0x01u8, 0x80] {
            let mut bytes = original.clone();
            bytes[i] ^= bit;
            // The result is irrelevant; not unwinding is the whole test.
            let _ = kmesh::read(&bytes);
        }
    }
}

#[test]
fn a_corrupt_vertex_count_is_refused_rather_than_allocated() {
    // The specific failure the byte-corruption sweep found, pinned so it cannot
    // come back quietly.
    //
    // `num_verts` was the one number in the file that nothing else bounded: a
    // bare `u32` that the mesh constructor turned straight into an allocation.
    // One flipped bit asked for 25 GB — and an allocation failure *aborts* the
    // process, so there was no error to report, no unwinding, and no diagnostic.
    // Worse than a panic, and reached from any malformed file.
    let mut bytes = kmesh::write(&rich_mesh(), &[]);
    // The field sits after magic, the two versions, min_reader_major and the
    // buffer count.
    bytes[24..28].copy_from_slice(&2_147_483_647u32.to_le_bytes());
    match kmesh::read(&bytes) {
        Err(kmesh::KmeshError::VertexCountMismatch { claimed, .. }) => {
            assert_eq!(claimed, 2_147_483_647);
        }
        other => panic!("expected a refusal, got {other:?}"),
    }
}

#[test]
fn a_mesh_without_positions_is_refused() {
    // Position is mandatory, and it is also what bounds the vertex count. A file
    // without it would otherwise be accepted as a mesh whose every vertex sits at
    // the origin — geometry silently replaced with nothing.
    let mut mesh = kreate_mesh::primitives::cube(2.0);
    mesh.attributes_mut().remove(names::POSITION);
    let bytes = kmesh::write(&mesh, &[]);
    assert!(matches!(
        kmesh::read(&bytes),
        Err(kmesh::KmeshError::MissingBuffer(_))
    ));
}

#[test]
fn an_attribute_of_the_wrong_length_is_refused() {
    // A file can declare a point attribute with the wrong number of values.
    // Accepting it would put a short array behind an index the rest of the
    // system takes from the vertex count.
    let mut mesh = kreate_mesh::primitives::cube(2.0);
    mesh.attributes_mut()
        .insert("bad", Domain::Point, AttrData::F32(Arc::new(vec![1.0; 3])));
    let bytes = kmesh::write(&mesh, &[]);
    assert!(matches!(
        kmesh::read(&bytes),
        Err(kmesh::KmeshError::AttributeLengthMismatch { .. })
    ));
}

#[test]
fn an_empty_mesh_round_trips() {
    let mesh = MeshData::empty();
    let bytes = kmesh::write(&mesh, &[]);
    let loaded = kmesh::read(&bytes).expect("an empty mesh is still a mesh");
    assert_eq!(loaded.mesh.num_verts(), 0);
    assert_eq!(loaded.mesh.num_faces(), 0);
}

#[test]
fn a_large_mesh_round_trips_without_changing_size_class() {
    // Guards the claim that the format is a layout and not an encoding: the file
    // should be about the size of the arrays it holds, not a multiple of it. A
    // writer that accidentally stored values as text, or duplicated a buffer,
    // shows up here rather than as a slow load on someone's machine.
    let mesh = kreate_mesh::primitives::uv_sphere(1.0, 64, 32);
    let bytes = kmesh::write(&mesh, &[]);
    let payload = mesh.memory_bytes();
    assert!(
        bytes.len() < payload * 2,
        "{} bytes on disk for {payload} bytes of mesh",
        bytes.len()
    );
    let loaded = kmesh::read(&bytes).expect("reads");
    assert_same(&mesh, &loaded.mesh);
}

#[test]
fn the_type_codes_never_move() {
    // The codes are written into files. Renumbering them would silently
    // reinterpret every attribute in every file already on disk — a float array
    // read as integers, a selection read as colour. Pinned so that adding a type
    // means appending, never inserting.
    let mesh_for = |ty: AttrType, len: usize| {
        let mut mesh = kreate_mesh::primitives::cube(2.0);
        let n = mesh.num_verts() as usize;
        let _ = len;
        mesh.attributes_mut()
            .insert("probe", Domain::Point, AttrData::zeroed(ty, n));
        kmesh::write(&mesh, &[])
    };
    // Read the probe attribute's type code straight out of the table.
    let code_of = |ty: AttrType| -> u32 {
        let bytes = mesh_for(ty, 0);
        let count = u32::from_le_bytes(bytes[20..24].try_into().unwrap()) as usize;
        let names_offset = u32::from_le_bytes(bytes[40..44].try_into().unwrap()) as usize;
        for i in 0..count {
            let at = 64 + i * 48;
            let name_offset = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
            let name_length =
                u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap()) as usize;
            let name = &bytes[names_offset + name_offset..names_offset + name_offset + name_length];
            if name == b"probe" {
                return u32::from_le_bytes(bytes[at + 12..at + 16].try_into().unwrap());
            }
        }
        panic!("probe attribute not found in the table");
    };

    assert_eq!(code_of(AttrType::F32), 0);
    assert_eq!(code_of(AttrType::F32x2), 1);
    assert_eq!(code_of(AttrType::F32x3), 2);
    assert_eq!(code_of(AttrType::F32x4), 3);
    assert_eq!(code_of(AttrType::I32), 4);
    assert_eq!(code_of(AttrType::Bool), 5);
    assert_eq!(code_of(AttrType::U8x4), 6);
}
