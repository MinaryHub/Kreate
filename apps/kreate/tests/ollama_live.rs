//! Ollama, for real: a model on this computer answers through Kreate's own
//! tool loop. Needs Ollama running with a tool-using model, a Kreate open,
//! and `kreate-mcp` built — so it is run by hand:
//! `cargo test -p kreate --test ollama_live -- --ignored --nocapture`.

use std::sync::{mpsc, Arc, Mutex};

#[test]
#[ignore]
fn a_local_model_looks_at_the_scene_through_the_tools() {
    let models = kreate::ollama::models().expect("Ollama answers");
    let model = models
        .iter()
        .find(|m| m.tools)
        .cloned()
        .expect("a model that can use tools");
    let connector = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("kreate-mcp");
    let (tx, rx) = mpsc::channel();
    kreate::ollama::turn(
        model,
        Arc::new(Mutex::new(Vec::new())),
        std::env::var("KREATE_OLLAMA_PROMPT").unwrap_or_else(|_| {
            "Call get_scene, then tell me in one sentence how many objects the scene has.".into()
        }),
        Vec::new(),
        connector,
        "You are inside Kreate, a 3D modelling application.",
        Arc::default(),
        tx,
        Arc::new(|| {}),
    );
    let events: Vec<_> = rx.try_iter().collect();
    let mut said = String::new();
    for e in &events {
        match e {
            kreate::claude::Event::TextPart(t) => said.push_str(t),
            other => eprintln!("{other:?}"),
        }
    }
    eprintln!("said: {said}");
    assert!(events
        .iter()
        .any(|e| matches!(e, kreate::claude::Event::Tool { .. })));
    assert!(matches!(
        events.last(),
        Some(kreate::claude::Event::Done(None))
    ));
    assert!(!said.trim().is_empty());
}
