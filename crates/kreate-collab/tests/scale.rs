//! How the session's bookkeeping holds up at factory scale: tens of
//! thousands of parts, four in five of them sharing a mesh with others.
//!
//! Ignored; run in release:
//! `cargo test -p kreate-collab --release --test scale -- --ignored --nocapture`

use std::time::Instant;

use kreate_collab::replica::Replica;
use kreate_math::DVec3;
use kreate_scene::{Document, Object, ObjectData, Uid};

/// `n` parts, a fifth of them with a mesh of their own (about 200
/// triangles) and the rest sharing those.
fn factory(n: usize) -> Document {
    let mut doc = Document::new();
    let unique = (n / 5).max(1);
    let meshes: Vec<_> = (0..unique)
        .map(|_| doc.add_mesh(kreate_mesh::primitives::uv_sphere(0.1, 12, 9)))
        .collect();
    for i in 0..n {
        let mut o = Object::new(format!("Part{i}"), ObjectData::Mesh(meshes[i % unique]));
        o.transform.translation = DVec3::new((i % 250) as f64, (i / 250) as f64, 0.0);
        // Restored rather than added: adding names each part uniquely by
        // looking at every other, which is not what is being measured.
        doc.restore_object(o, Uid::from_raw(i as u64 + 1));
    }
    doc
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

/// Best of a few runs: the floor, which is what the code costs rather than
/// what the machine was doing.
fn best(runs: usize, mut f: impl FnMut() -> f64) -> f64 {
    (0..runs).map(|_| f()).fold(f64::INFINITY, f64::min)
}

#[test]
#[ignore]
fn session_bookkeeping_at_factory_scale() {
    for n in [5_000usize, 10_000, 20_000, 50_000] {
        let mut doc = factory(n);
        let triangles: usize = doc
            .meshes()
            .map(|(_, m)| {
                (0..m.num_faces())
                    .map(|f| m.face_len(f) as usize - 2)
                    .sum::<usize>()
            })
            .sum::<usize>()
            / (n / 5).max(1);

        let base = doc.clone();
        let idle = best(5, || {
            let t = Instant::now();
            let c = doc.changes_since(&base);
            assert!(c.is_empty());
            ms(t)
        });

        let snapshot = best(3, || {
            let t = Instant::now();
            let bytes = kreate_io::sync::encode_all(&doc).unwrap();
            assert!(!bytes.is_empty());
            ms(t)
        });

        // One part moved: what goes out, and what the other side does with it.
        let mut ra = Replica::new(&mut doc, 1);
        let seed = ra.seed(&doc).unwrap();
        ra.ack();
        let mut other = Document::new();
        let mut rb = Replica::new(&mut other, 2);
        let t = Instant::now();
        rb.adopt_snapshot(&mut other, &seed).unwrap();
        let join = ms(t);
        let part = doc.objects().next().unwrap().0;
        let (mut send, mut take) = (f64::INFINITY, f64::INFINITY);
        for k in 0..5 {
            doc.object_mut(part).unwrap().transform.translation.z = k as f64 + 1.0;
            let t = Instant::now();
            let patch = ra.outgoing(&doc).expect("the move");
            send = send.min(ms(t));
            ra.ack();
            let t = Instant::now();
            let changed = rb
                .incoming(&mut other, 1, &patch)
                .unwrap()
                .expect("arrives");
            take = take.min(ms(t));
            assert_eq!(changed.changed.len(), 1);
        }
        eprintln!(
            "{n:>6} objects (~{triangles} tris/mesh): idle changes_since {idle:8.2} ms | \
             move: send {send:8.2} ms, receive {take:8.2} ms | encode_all {snapshot:8.2} ms | \
             join {join:8.2} ms"
        );
    }
}
