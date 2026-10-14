//! What `kreate-mcp` asks of the open document, answered on the main
//! thread: the bridge's requests, served by method — the scene and a mesh
//! as the connector sees them, scripts, captures, the clock, assets, the
//! checks — and the JSON each comes back as. See `kreate::bridge` for the
//! wire, and `kreate-mcp` for the tools these methods stand behind.

use super::*;

impl App {
    /// Answers every request waiting since the last wake.
    pub(super) fn serve_bridge(&mut self) {
        let Some(bridge) = self.bridge.as_ref() else {
            return;
        };
        for request in bridge.pending() {
            // A search may wait on a slow source and on pictures: answered
            // from a thread of its own, the window going on meanwhile.
            // A height map takes Z-Image some seconds: answered from a
            // thread, with the file's path.
            if request.method == "height_map" {
                std::thread::spawn(move || {
                    let words = request
                        .params
                        .get("prompt")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    let result = if words.is_empty() {
                        Err("`prompt` says what the surface is: skin, scales, bark…".to_string())
                    } else {
                        kreate::generate::height_map(&words, &|_| {}).map(|path| {
                            serde_json::json!({
                                "path": path.display().to_string(),
                                "next": "kreate.ops.image.open(path=..., color=False), then sculpt.displace_image(image=<uid>) on a mesh fine enough to carry it",
                            })
                        })
                    };
                    request.answer(result);
                });
                continue;
            }
            if request.method == "assets" {
                let folders = self.ui.asset_folders.clone();
                std::thread::spawn(move || {
                    let result = assets_search(&request.params, &folders);
                    request.answer(result);
                });
                continue;
            }
            let result = self.bridge_call(&request.method, &request.params);
            // An asset asked for with `wait` is answered when it has landed.
            let waiting = request.method == "add_asset"
                && request.params.get("wait").and_then(|v| v.as_bool()) == Some(true);
            match (&result, waiting) {
                (Ok(value), true) => {
                    let id = value["id"].as_str().unwrap_or_default().to_string();
                    match self.asset_wishes.get_mut(&id) {
                        Some(wish) => wish.reply = Some(request),
                        None => request.answer(result),
                    }
                }
                _ => request.answer(result),
            }
        }
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    pub(super) fn bridge_call(
        &mut self,
        method: &str,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        use serde_json::json;
        let text = |name: &str| {
            params
                .get(name)
                .and_then(|v| v.as_str())
                .map(str::to_string)
        };
        match method {
            "scene" => Ok(self.scene_json()),
            // The clock and what moves: read it, move it, play or stop it,
            // and switch a take on or off — a strip changed with its other
            // settings kept, which a script line naming only `mute` would
            // reset.
            "timeline" => {
                if let (Some(object), Some(strip)) =
                    (text("object"), params.get("strip").and_then(|v| v.as_u64()))
                {
                    let mute = params
                        .get("mute")
                        .and_then(|v| v.as_bool())
                        .ok_or("`mute` says whether the strip is switched off")?;
                    let id = self
                        .doc
                        .objects()
                        .find(|(_, o)| o.name == object)
                        .map(|(id, _)| id)
                        .ok_or_else(|| format!("no object called `{object}`"))?;
                    let current = self
                        .doc
                        .object(id)
                        .and_then(|o| o.nla.get(strip as usize))
                        .map(|s| kreate_ops::ops::anim::nla_strip_params(strip as usize, s))
                        .ok_or_else(|| format!("`{object}` has no strip {strip}"))?;
                    let mut changed = current;
                    changed.mute = mute;
                    self.doc.selection.clear();
                    self.doc.selection.add(id);
                    self.doc.selection.active = Some(id);
                    let before = self.history.cursor();
                    self.run_op(
                        "anim.nla_strip_set",
                        kreate_reflect::OpParams::to_map(&changed),
                    );
                    if self.history.cursor() == before {
                        return Err("the strip did not change".into());
                    }
                }
                if let Some(frame) = params.get("frame").and_then(|v| v.as_f64()) {
                    self.set_frame(frame);
                }
                if let Some(play) = params.get("play").and_then(|v| v.as_bool()) {
                    self.playing = play.then(std::time::Instant::now);
                    self.redraw.set(RedrawReason::PLAYING);
                }
                Ok(self.timeline_json())
            }
            "mesh" => {
                let name = text("object").unwrap_or_default();
                let world = text("space").as_deref() != Some("local");
                let selected = params
                    .get("selected")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let faces = params
                    .get("faces")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let limit = params
                    .get("limit")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(2000)
                    .clamp(1, 100_000) as usize;
                self.mesh_json(&name, world, selected, faces, limit)
            }
            "render" => {
                let path = text("path").ok_or("`path` is required")?;
                let number = |key: &str, default: u64| {
                    params.get(key).and_then(|v| v.as_u64()).unwrap_or(default)
                };
                let size = (number("width", 1920) as u32, number("height", 1080) as u32);
                let samples = number("samples", 128).clamp(1, 512) as u32;
                let limit = std::time::Duration::from_secs(number("time_limit", 110).clamp(1, 110));
                self.render_still(
                    text("camera").as_deref(),
                    size,
                    samples,
                    std::path::Path::new(&path),
                    limit,
                )
            }
            "overlaps" => {
                let tolerance = params
                    .get("tolerance")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.001)
                    .max(0.0);
                let exact = params
                    .get("exact")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(true);
                Ok(self.overlaps_json(tolerance, exact))
            }
            "operators" => Ok(self.operators_json(text("filter").as_deref().unwrap_or(""))),
            "run" => {
                let script = text("script").ok_or("`script` is required")?;
                // Said, so a person reading the console knows whose lines
                // these are.
                self.ui.console_log.push(kreate_ui::console::ConsoleLine {
                    text: self.ui.catalog.get("console-from-assistant"),
                    kind: kreate_ui::console::LineKind::Note,
                });
                let lines: Vec<_> = self
                    .run_script_reporting(&script)
                    .into_iter()
                    .map(|line| {
                        let mut v = match line.result {
                            Ok(r) => json!({ "line": line.text, "ok": true, "result": r }),
                            Err(e) => json!({ "line": line.text, "ok": false, "error": e }),
                        };
                        let listed = |made: &[(String, u64)]| -> serde_json::Value {
                            made.iter()
                                .map(|(name, uid)| json!({ "name": name, "uid": uid }))
                                .collect()
                        };
                        if !line.created.is_empty() {
                            v["created"] = listed(&line.created);
                        }
                        if !line.materials.is_empty() {
                            v["materials_created"] = listed(&line.materials);
                        }
                        if !line.images.is_empty() {
                            v["images_created"] = listed(&line.images);
                        }
                        v
                    })
                    .collect();
                let active = self
                    .doc
                    .selection
                    .active
                    .and_then(|id| self.doc.object(id))
                    .map(|o| o.name.clone());
                Ok(json!({ "lines": lines, "mode": self.op_ctx.mode.tag(), "active": active }))
            }
            // An application action by its keymap id — `render.image`,
            // `render.animation`, `view.frame_all` — as if its key were pressed.
            // For what has no operator: a render runs frame by frame, so this
            // starts it and `render_status` follows it.
            "action" => {
                let id = text("id").ok_or("`id` is required")?;
                self.dispatch(&Action {
                    op_id: id,
                    params: PropertyMap::new(),
                });
                Ok(json!({ "message": self.ui.status.clone().unwrap_or_default() }))
            }
            "render_status" => Ok(match &self.render_job {
                Some(job) => json!({
                    "rendering": true,
                    "frame": job.frames.map(|(f, _)| f),
                    "last_frame": job.frames.map(|(_, e)| e),
                    "samples": self.gfx.as_ref().map(|g| g.viewport.path_trace_samples()).unwrap_or(0),
                    "of": job.settings.samples,
                    "written": job.written.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
                }),
                None => json!({
                    "rendering": false,
                    "last": self.last_render.as_ref().map(|(_, _, _, m)| m.clone()),
                }),
            }),
            "undo" | "redo" => {
                let steps = params
                    .get("steps")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1)
                    .clamp(1, 1000);
                for _ in 0..steps {
                    if method == "undo" {
                        self.undo();
                    } else {
                        self.redo();
                    }
                }
                self.redraw.set(RedrawReason::SCENE_CHANGED);
                Ok(json!({ "steps": steps, "history_position": self.history.cursor() }))
            }
            // From several sides at once, or of some objects alone: what an
            // assistant needs to judge a shape it cannot turn round itself.
            "capture" if params.get("views").is_some() || params.get("only").is_some() => {
                self.capture_views(params)
            }
            "context" => Ok(self.context_json()),
            // The model's outline against a picture's, and where they differ.
            "silhouette" => {
                use kreate::silhouette::{compare, outline, picture_outline, Side};
                let path = text("reference").ok_or("`reference` is the picture's path")?;
                let bytes = std::fs::read(&path).map_err(|e| format!("cannot read {path}: {e}"))?;
                let view = text("view").unwrap_or_else(|| "front".into());
                let side = Side::from_name(&view)
                    .ok_or("`view` is front, back, left, right (or side) or top")?;
                let names: Option<Vec<String>> =
                    params.get("objects").and_then(|v| v.as_array()).map(|a| {
                        a.iter()
                            .filter_map(|n| n.as_str().map(str::to_string))
                            .collect()
                    });
                self.evaluate_scene();
                let mut triangles = Vec::new();
                for (id, mesh) in &self.evaluated {
                    let Some(o) = self.doc.object(*id) else {
                        continue;
                    };
                    if names.as_ref().is_some_and(|n| !n.contains(&o.name)) {
                        continue;
                    }
                    let world = self.doc.world_transform(*id);
                    let p: Vec<DVec3> = mesh
                        .positions()
                        .iter()
                        .map(|q| world.transform_point3(q.as_dvec3()))
                        .collect();
                    for f in 0..mesh.num_faces() {
                        let ring = mesh.face_verts(f);
                        for i in 1..ring.len().saturating_sub(1) {
                            triangles.push([
                                p[ring[0] as usize],
                                p[ring[i] as usize],
                                p[ring[i + 1] as usize],
                            ]);
                        }
                    }
                }
                if triangles.is_empty() {
                    return Err("no mesh to outline".into());
                }
                let (model, plane) = outline(&triangles, side).ok_or("the model has no extent")?;
                // A difference is put at the model's middle depth.
                let (lo, hi) = triangles.iter().flatten().fold(
                    (DVec3::splat(f64::MAX), DVec3::splat(f64::MIN)),
                    |(lo, hi), p| (lo.min(*p), hi.max(*p)),
                );
                let depth = ((lo + hi) * 0.5).dot(side.depth_axis());
                let c = compare(&model, plane, &picture_outline(&bytes)?, side, depth)?;
                let differences: Vec<serde_json::Value> = c
                    .differences
                    .iter()
                    .map(|d| {
                        json!({
                            "model_has": if d.model_has_more { "more" } else { "less" },
                            "at": d.at.map(|v| (v * 1000.0).round() / 1000.0),
                            "share": (d.share * 1000.0).round() / 1000.0,
                        })
                    })
                    .collect();
                Ok(json!({
                    "view": view,
                    "agreement": (c.agreement * 1000.0).round() / 1000.0,
                    "model_width_over_height": (c.model_aspect * 1000.0).round() / 1000.0,
                    "picture_width_over_height": (c.picture_aspect * 1000.0).round() / 1000.0,
                    "differences": differences,
                    "png_base64": kreate::bridge::base64(&c.png),
                }))
            }
            // A standing figure against a realistic adult's proportions.
            "proportions" => {
                let names: Option<Vec<String>> =
                    params.get("objects").and_then(|v| v.as_array()).map(|a| {
                        a.iter()
                            .filter_map(|n| n.as_str().map(str::to_string))
                            .collect()
                    });
                self.evaluate_scene();
                let mut points = Vec::new();
                for (id, mesh) in &self.evaluated {
                    let Some(o) = self.doc.object(*id) else {
                        continue;
                    };
                    if names.as_ref().is_some_and(|n| !n.contains(&o.name)) {
                        continue;
                    }
                    let world = self.doc.world_transform(*id);
                    points.extend(
                        mesh.positions()
                            .iter()
                            .map(|p| world.transform_point3(p.as_dvec3())),
                    );
                }
                let measures = kreate::proportions::measure(&points);
                if measures.is_empty() {
                    return Err("no figure to measure".into());
                }
                let round = |v: f64| (v * 1000.0).round() / 1000.0;
                Ok(json!({
                    "standard": "a realistic adult; a stylised figure departs on purpose",
                    "measures": measures
                        .iter()
                        .map(|m| json!({
                            "name": m.name,
                            "value": round(m.value),
                            "usual": [m.usual.0, m.usual.1],
                            "verdict": m.verdict(),
                            "how": m.how,
                        }))
                        .collect::<Vec<_>>(),
                }))
            }
            // A mesh read for what a sculptor looks for, each with where.
            "analyze_mesh" => {
                let id = match text("object") {
                    Some(name) => self
                        .doc
                        .objects()
                        .find(|(_, o)| o.name == name)
                        .map(|(id, _)| id)
                        .ok_or_else(|| format!("no object called `{name}`"))?,
                    None => self.doc.selection.active.ok_or("nothing is active")?,
                };
                let object = self.doc.object(id).ok_or("no such object")?;
                let mesh = object
                    .mesh()
                    .and_then(|m| self.doc.mesh(m))
                    .ok_or_else(|| format!("`{}` is not a mesh", object.name))?;
                let r = kreate::meshcheck::analyze(mesh, &self.doc.world_transform(id));
                let spots = |list: &[kreate::meshcheck::Spot]| -> Vec<serde_json::Value> {
                    list.iter()
                        .map(|s| {
                            json!({
                                "at": s.at,
                                "count": s.count,
                                "worst": (s.worst * 100.0).round() / 100.0,
                            })
                        })
                        .collect()
                };
                Ok(json!({
                    "object": object.name,
                    "vertices": r.vertices,
                    "faces": { "triangles": r.triangles, "quads": r.quads, "ngons": r.ngons },
                    "typical_edge_m": r.edge_length,
                    "open_edges": spots(&r.open_edges),
                    "non_manifold_edges": spots(&r.non_manifold_edges),
                    "degenerate_faces": r.degenerate_faces,
                    "sliver_share": (r.sliver_share * 1000.0).round() / 1000.0,
                    "bumpy_places": spots(&r.bumpy),
                    "asymmetry": r.asymmetry.as_ref().map(|(mean, worst, places)| json!({
                        "mean_edges": (mean * 100.0).round() / 100.0,
                        "worst_edges": (worst * 100.0).round() / 100.0,
                        "places": spots(places),
                    })),
                }))
            }
            // Every bone bent in turn and the skin measured: what is torn,
            // where, and what no bone holds.
            "rig_check" => {
                let rig = match text("rig") {
                    Some(name) => self
                        .doc
                        .objects()
                        .find(|(_, o)| o.name == name)
                        .map(|(id, _)| id)
                        .ok_or_else(|| format!("no object called `{name}`"))?,
                    None => self
                        .doc
                        .selection
                        .active
                        .filter(|id| {
                            matches!(
                                self.doc.object(*id).map(|o| &o.data),
                                Some(kreate_scene::document::ObjectData::Armature(_))
                            )
                        })
                        .or_else(|| {
                            self.doc
                                .objects()
                                .find(|(_, o)| {
                                    matches!(
                                        o.data,
                                        kreate_scene::document::ObjectData::Armature(_)
                                    )
                                })
                                .map(|(id, _)| id)
                        })
                        .ok_or("the scene has no armature")?,
                };
                let degrees = params.get("angle").and_then(|v| v.as_f64()).unwrap_or(45.0);
                let report = kreate::rigcheck::check(&self.doc, rig, degrees)?;
                let bones: Vec<serde_json::Value> = report
                    .bones
                    .iter()
                    .map(|b| {
                        json!({
                            "bone": b.bone,
                            "moved_vertices": b.moved,
                            "worst_edge_ratio": (b.worst_ratio * 1000.0).round() / 1000.0,
                            "torn_edges": b.torn,
                            "pinched_edges": b.pinched,
                            "worst_at": b.worst_at,
                        })
                    })
                    .collect();
                let idle: Vec<&str> = report
                    .bones
                    .iter()
                    .filter(|b| b.moved == 0)
                    .map(|b| b.bone.as_str())
                    .collect();
                Ok(json!({
                    "rig": report.rig,
                    "meshes": report.meshes,
                    "bent_by_degrees": degrees,
                    "bones_worst_first": bones,
                    "bones_moving_nothing": idle,
                    "unweighted_vertices": report.unweighted,
                    "unweighted_sample": report.unweighted_sample,
                }))
            }
            // A model made by AI, added when it arrives — as an asset is.
            "generate" => {
                let request = match (text("prompt"), text("image")) {
                    (Some(words), _) if !words.trim().is_empty() => {
                        kreate::generate::Request::Text(words)
                    }
                    (_, Some(path)) => {
                        let path = std::path::PathBuf::from(path);
                        if !path.is_file() {
                            return Err(format!("no picture at {}", path.display()));
                        }
                        kreate::generate::Request::Image(path)
                    }
                    _ => return Err("give `prompt` (words) or `image` (a picture's path)".into()),
                };
                if self.ui.asset_generating {
                    return Err("a model is being made already; one at a time".into());
                }
                let library = self
                    .assets
                    .as_ref()
                    .ok_or("the asset library is not running")?;
                library.generate(request);
                self.ui.asset_generating = true;
                Ok(json!({
                    "message": "making it on Hugging Face: a minute or two; it is added to the scene when it arrives — check with get_scene"
                }))
            }
            "capture" => {
                let camera = match params.get("camera") {
                    Some(serde_json::Value::String(name)) if !name.is_empty() => {
                        Some(Some(name.clone()))
                    }
                    Some(serde_json::Value::Bool(true)) => Some(None),
                    _ => None,
                };
                let (w, h, png) = match camera {
                    None => self.capture_png()?,
                    Some(name) => {
                        let object = self.scene_camera(name.as_deref()).ok_or(match &name {
                            Some(n) => format!("no camera called {n:?}"),
                            None => "the scene has no camera".to_string(),
                        })?;
                        // Through the camera for this one picture, then back
                        // to where the user was looking — and with the
                        // camera's lens, as its own view would be.
                        let saved = (
                            self.camera,
                            self.motion,
                            self.camera_view,
                            self.camera_view_object,
                        );
                        self.motion.interrupt(&self.camera);
                        self.camera = self.view_through(object);
                        self.motion.target = self.camera;
                        self.camera_view = Some(saved.0);
                        self.camera_view_object = Some(object);
                        let shot = self.capture_png();
                        (
                            self.camera,
                            self.motion,
                            self.camera_view,
                            self.camera_view_object,
                        ) = saved;
                        self.redraw.set(RedrawReason::CAMERA_MOVED);
                        shot?
                    }
                };
                let mut shot =
                    json!({ "width": w, "height": h, "png_base64": kreate::bridge::base64(&png) });
                if let Some(path) = text("path") {
                    std::fs::write(&path, &png).map_err(|e| format!("cannot write {path}: {e}"))?;
                    shot["path"] = json!(path);
                }
                Ok(shot)
            }
            "open" => {
                let path = text("path").ok_or("`path` is required")?;
                self.load_from(std::path::Path::new(&path));
                let said = self.ui.status.clone().unwrap_or_default();
                if said.starts_with("cannot") {
                    Err(said)
                } else {
                    Ok(json!({ "message": said }))
                }
            }
            "assets" => assets_search(params, &self.ui.asset_folders),
            "add_asset" => {
                use kreate::assets::{Kind, Source};
                let kind = text("kind")
                    .and_then(|k| Kind::from_tag(&k))
                    .ok_or("`kind` is model, material or hdri")?;
                let source = Source::from_tag(&text("source").unwrap_or("polyhaven".into()))
                    .ok_or(
                    "`source` is polyhaven, ambientcg, objaverse, sketchfab, polypizza or local",
                )?;
                let id = text("id").ok_or("`id` is required")?;
                let resolution = match text("resolution").as_deref() {
                    Some("2k") => "2k",
                    Some("4k") => "4k",
                    _ => "1k",
                };
                let entry = kreate::assets::find_now(source, kind, &self.ui.asset_folders, &id)?;
                let library = self
                    .assets
                    .as_ref()
                    .ok_or("the asset library is not running")?;
                let point = |key: &str| -> Option<DVec3> {
                    let a = params.get(key)?.as_array()?;
                    Some(DVec3::new(
                        a.first()?.as_f64()?,
                        a.get(1)?.as_f64()?,
                        a.get(2)?.as_f64()?,
                    ))
                };
                self.asset_wishes.insert(
                    entry.id.clone(),
                    ImportWish {
                        location: point("location"),
                        height: params
                            .get("height")
                            .and_then(|v| v.as_f64())
                            .filter(|h| *h > 0.0),
                        reply: None,
                    },
                );
                if self.ui.asset_fetching.insert(entry.id.clone()) {
                    library.fetch(entry.clone(), resolution);
                }
                Ok(json!({
                    "id": entry.id,
                    "message": format!("fetching {}; it is added when it arrives", entry.name),
                }))
            }
            "import" => {
                let path = text("path").ok_or("`path` is required")?;
                // Here and now, not on a thread: the connector is told what
                // happened when it returns.
                let mut params = PropertyMap::new();
                params.set("path", PropertyValue::Text(path.clone()));
                self.run_op("object.import", params);
                if self
                    .history
                    .last()
                    .is_some_and(|e| e.op_id == "object.import")
                {
                    self.status(&format!("added {path}"));
                } else {
                    self.status(&format!("cannot import {path}"));
                }
                let said = self.ui.status.clone().unwrap_or_default();
                if said.starts_with("cannot") {
                    Err(said)
                } else {
                    Ok(json!({ "message": said }))
                }
            }
            "save" => {
                let path = match text("path") {
                    Some(p) => std::path::PathBuf::from(p),
                    None => self
                        .file
                        .path()
                        .map(|p| p.to_path_buf())
                        .ok_or("the document has never been saved; give a path")?,
                };
                self.save_to(path).map(|m| json!({ "message": m }))
            }
            "export" => {
                let path = text("path").ok_or("`path` is required")?;
                self.export_to(std::path::Path::new(&path))
                    .map(|m| json!({ "message": m }))
            }
            other => Err(format!("unknown method `{other}`")),
        }
    }

    /// Pictures from named sides — front, back, left, right, top, bottom,
    /// or the user's own view — framed on the scene or on the objects named
    /// in `only`, which are all that shows. The user's view and what they
    /// had hidden come back afterwards.
    pub(super) fn capture_views(
        &mut self,
        params: &serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        use kreate_render::ViewAxis;
        use serde_json::json;
        let names: Vec<String> = match params.get("views") {
            Some(serde_json::Value::Array(v)) => v
                .iter()
                .filter_map(|n| n.as_str().map(str::to_lowercase))
                .collect(),
            _ => vec!["user".to_string()],
        };
        if names.is_empty() || names.len() > 6 {
            return Err("`views` names one to six sides".into());
        }
        let axis_of = |name: &str| -> Result<Option<ViewAxis>, String> {
            Ok(match name {
                "front" => Some(ViewAxis::NegY),
                "back" => Some(ViewAxis::PosY),
                "right" => Some(ViewAxis::PosX),
                "left" => Some(ViewAxis::NegX),
                "top" => Some(ViewAxis::PosZ),
                "bottom" => Some(ViewAxis::NegZ),
                "user" => None,
                other => {
                    return Err(format!(
                        "`{other}` is not a side: front, back, left, right, top, bottom or user"
                    ))
                }
            })
        };
        let axes = names
            .iter()
            .map(|n| axis_of(n))
            .collect::<Result<Vec<_>, _>>()?;
        // The others hidden for these pictures only.
        let only: Option<Vec<String>> = params.get("only").and_then(|v| v.as_array()).map(|a| {
            a.iter()
                .filter_map(|n| n.as_str().map(str::to_string))
                .collect()
        });
        if let Some(only) = &only {
            for name in only {
                if !self.doc.objects().any(|(_, o)| &o.name == name) {
                    return Err(format!("no object called `{name}`"));
                }
            }
        }
        let was_saved = !self.file.is_modified(&self.doc);
        let hidden: Vec<kreate_scene::ObjectId> = match &only {
            Some(only) => self
                .doc
                .objects()
                .filter(|(_, o)| o.visible && !only.contains(&o.name))
                .map(|(id, _)| id)
                .collect(),
            None => Vec::new(),
        };
        for id in &hidden {
            if let Some(o) = self.doc.object_mut(*id) {
                o.visible = false;
            }
        }
        let saved = (
            self.camera,
            self.motion,
            self.camera_view,
            self.camera_view_object,
        );
        self.motion.interrupt(&self.camera);
        self.camera_view = None;
        self.camera_view_object = None;
        // Framed on the meshes that show: a lamp or a camera far off is
        // part of the scene, not of the shape being looked at.
        self.evaluate_scene();
        let boxes = self.mesh_boxes();
        let (center, radius) = match boxes
            .iter()
            .copied()
            .reduce(|a, b| (a.0.min(b.0), a.1.max(b.1)))
        {
            Some((lo, hi)) => ((lo + hi) * 0.5, ((hi - lo).length() * 0.5).max(1e-3)),
            None => {
                let b = self.doc.world_bounds();
                let c = b.center();
                (
                    DVec3::new(c.x as f64, c.y as f64, c.z as f64),
                    b.bounding_radius().max(1e-3) as f64,
                )
            }
        };
        let aspect = self
            .gfx
            .as_ref()
            .map(|g| g.viewport.aspect())
            .unwrap_or(16.0 / 9.0);
        let mut shots = Vec::new();
        let mut failed = None;
        for (name, axis) in names.iter().zip(&axes) {
            let mut view = match axis {
                Some(axis) => {
                    let mut c = saved.0.looking_along(*axis);
                    c.projection = kreate_render::Projection::Orthographic;
                    c
                }
                None => saved.0,
            };
            // Framed on what shows when a side was asked for or the scene was
            // narrowed; the user's view as it is otherwise.
            if axis.is_some() || only.is_some() {
                view.frame_sphere(center, radius, aspect);
            }
            self.camera = view;
            self.motion.target = view;
            self.redraw.set(RedrawReason::CAMERA_MOVED);
            match self.capture_png() {
                Ok((w, h, png)) => shots.push(json!({
                    "view": name,
                    "width": w,
                    "height": h,
                    "png_base64": kreate::bridge::base64(&png),
                })),
                Err(e) => {
                    failed = Some(e);
                    break;
                }
            }
        }
        (
            self.camera,
            self.motion,
            self.camera_view,
            self.camera_view_object,
        ) = saved;
        for id in &hidden {
            if let Some(o) = self.doc.object_mut(*id) {
                o.visible = true;
            }
        }
        if was_saved && !hidden.is_empty() {
            self.file.still_saved(&self.doc);
        }
        self.redraw.set(RedrawReason::SCENE_CHANGED);
        match failed {
            Some(e) => Err(e),
            None => Ok(json!({ "shots": shots })),
        }
    }

    /// What the user is working on, in a breath: the mode, the selection
    /// (and in edit mode how many vertices, edges and faces of it), the 3D
    /// cursor, where the view looks from, the shading and the frame — what
    /// "this", "here" and "make it rounder" refer to.
    pub(super) fn context_json(&self) -> serde_json::Value {
        use kreate_mesh::attribute::names;
        use serde_json::json;
        let doc = &self.doc;
        let name = |id: kreate_scene::ObjectId| doc.object(id).map(|o| o.name.clone());
        let selected: Vec<String> = doc
            .selection
            .objects
            .iter()
            .filter_map(|id| name(*id))
            .collect();
        let elements = self
            .op_ctx
            .mode
            .edits_elements()
            .then(|| kreate_ops::ops::mesh::target_mesh(doc))
            .flatten()
            .and_then(|m| doc.mesh(m))
            .map(|mesh| {
                let count = |layer: &str| {
                    mesh.attributes()
                        .get_bool(layer)
                        .map_or(0, |f| f.iter().filter(|b| **b).count())
                };
                json!({
                    "vertices": count(names::SELECT_VERT),
                    "edges": count(names::SELECT_EDGE),
                    "faces": count(names::SELECT_FACE),
                    "of_vertices": mesh.num_verts(),
                    "of_faces": mesh.num_faces(),
                })
            });
        let eye = self.camera.eye();
        let looking = (self.camera.pivot - eye).normalize_or_zero();
        json!({
            "mode": self.op_ctx.mode.tag(),
            "active": doc.selection.active.and_then(name),
            "selected": selected,
            "selected_elements": elements,
            "cursor": doc.cursor.to_array(),
            "view": {
                "eye": eye.to_array(),
                "looking": looking.to_array(),
                "pivot": self.camera.pivot.to_array(),
                "orthographic": self.camera.projection == kreate_render::Projection::Orthographic,
                "through_camera": self.camera_view_object.and_then(name),
                "shading": format!("{:?}", self.ui.view_status.shading),
            },
            "frame": doc.playback.frame,
            "last_operation": self
                .history
                .last()
                .map(|e| e.to_script_preview(kreate::shell::SCRIPT_PREVIEW)),
        })
    }

    /// The clock, and every object that moves with what it plays.
    pub(super) fn timeline_json(&self) -> serde_json::Value {
        use serde_json::json;
        let p = self.doc.playback;
        let animated: Vec<serde_json::Value> = self
            .doc
            .objects()
            .filter(|(_, o)| o.action.is_some() || !o.nla.is_empty())
            .map(|(_, o)| {
                let action = o.action.and_then(|a| self.doc.action(a)).map(|a| {
                    let frames = a.key_frames();
                    json!({
                        "name": a.name,
                        "first_key": frames.first(),
                        "last_key": frames.last(),
                    })
                });
                let strips: Vec<serde_json::Value> = o
                    .nla
                    .iter()
                    .enumerate()
                    .map(|(i, s)| {
                        json!({
                            "index": i,
                            "name": s.name,
                            "start": s.start,
                            "frames": s.action_end - s.action_start,
                            "muted": s.mute,
                        })
                    })
                    .collect();
                json!({ "object": o.name, "action": action, "strips": strips })
            })
            .collect();
        json!({
            "frame": p.frame,
            "start": p.start,
            "end": p.end,
            "fps": p.fps,
            "playing": self.playing.is_some(),
            "animated": animated,
        })
    }

    /// The document as the connector sees it: what is in it and where.
    /// One mesh's vertices, and its faces if asked: what an edit by position
    /// needs to know where things are. The active object when `name` is empty.
    ///
    /// Indices are the mesh's own, so they are the ones `mesh.select_set`
    /// takes. Capped at `limit` vertices, with the count of what was left out,
    /// so a scanned model does not flood the answer.
    pub(super) fn mesh_json(
        &self,
        name: &str,
        world: bool,
        selected: bool,
        faces: bool,
        limit: usize,
    ) -> Result<serde_json::Value, String> {
        use kreate_mesh::attribute::Domain;
        use serde_json::json;
        let doc = &self.doc;
        let id = if name.is_empty() {
            doc.selection
                .active
                .ok_or("no object is active; name one")?
        } else {
            doc.objects()
                .find(|(_, o)| o.name == name)
                .map(|(id, _)| id)
                .ok_or(format!("no object called {name:?}"))?
        };
        let object = doc.object(id).ok_or("the object is gone")?;
        let mesh = object
            .mesh()
            .and_then(|m| doc.mesh(m))
            .ok_or(format!("{:?} is not a mesh", object.name))?;
        let to_world = doc.world_transform(id);
        let at = |p: kreate_math::Vec3| {
            let p = p.as_dvec3();
            if world {
                to_world.transform_point3(p).to_array()
            } else {
                p.to_array()
            }
        };
        let wanted = |domain: Domain, i: u32| {
            !selected || kreate_mesh::selection::is_selected(mesh, domain, i)
        };
        let mut vertices = Vec::new();
        let mut matching = 0usize;
        for (i, p) in mesh.positions().iter().enumerate() {
            if !wanted(Domain::Point, i as u32) {
                continue;
            }
            matching += 1;
            if vertices.len() < limit {
                vertices.push(json!({ "index": i, "position": at(*p) }));
            }
        }
        let mut out = json!({
            "object": object.name,
            "space": if world { "world" } else { "local" },
            "units": { "length": "m" },
            "vertex_count": mesh.num_verts(),
            "edge_count": mesh.num_edges(),
            "face_count": mesh.num_faces(),
            "vertices": vertices,
            "omitted": matching.saturating_sub(limit),
        });
        if faces {
            let normals = mesh.compute_face_normals();
            let linear = to_world.matrix3.inverse().transpose();
            let list: Vec<_> = (0..mesh.num_faces())
                .filter(|f| wanted(Domain::Face, *f))
                .take(limit)
                .map(|f| {
                    let n = normals[f as usize].as_dvec3();
                    let n = if world {
                        (linear * n).normalize_or_zero()
                    } else {
                        n
                    };
                    // The middle, in the same space as the vertices: what
                    // a selection by position tests, and where to look.
                    let ring = mesh.face_verts(f);
                    let middle = ring
                        .iter()
                        .map(|v| mesh.positions()[*v as usize])
                        .sum::<kreate_math::Vec3>()
                        / ring.len().max(1) as f32;
                    json!({
                        "index": f,
                        "vertices": ring,
                        "normal": n.to_array(),
                        "center": at(middle),
                    })
                })
                .collect();
            out["faces"] = json!(list);
        }
        Ok(out)
    }

    /// Which visible mesh objects run into each other, by their world-space
    /// boxes: two whose boxes overlap by more than `tolerance` on every axis.
    /// Touching — a vase standing on a table — is not running into.
    ///
    /// With `exact`, each pair whose boxes overlap is then tested on its
    /// triangles (see `kreate_mesh::intersect`): kept only if the surfaces
    /// cross or one is inside the other, and said which. Without it, boxes
    /// alone — quick, and a pair can be reported that only comes close.
    pub(super) fn overlaps_json(&self, tolerance: f64, exact: bool) -> serde_json::Value {
        use serde_json::json;
        let doc = &self.doc;
        let boxes: Vec<_> = doc
            .objects()
            .filter(|(id, _)| doc.is_shown(*id))
            .filter_map(|(id, o)| Some((id, o.name.clone(), doc.object_world_bounds(id)?)))
            .collect();
        let related = |a, b| {
            let mut up = doc.object(a).and_then(|o| o.parent);
            while let Some(p) = up {
                if p == b {
                    return true;
                }
                up = doc.object(p).and_then(|o| o.parent);
            }
            false
        };
        let mut pairs = Vec::new();
        let mut touching = Vec::new();
        for (i, (a, an, (alo, ahi))) in boxes.iter().enumerate() {
            for (b, bn, (blo, bhi)) in &boxes[i + 1..] {
                let depth = ahi.min(*bhi) - alo.max(*blo);
                if depth.min_element() > tolerance && !related(*a, *b) && !related(*b, *a) {
                    if !exact {
                        pairs.push(json!({ "a": an, "b": bn, "overlap": depth.to_array() }));
                        continue;
                    }
                    let mesh = |id| {
                        doc.object(id)
                            .and_then(|o| o.mesh())
                            .and_then(|m| doc.mesh(m))
                    };
                    let (Some(ma), Some(mb)) = (mesh(*a), mesh(*b)) else {
                        continue;
                    };
                    let (wa, wb) = (doc.world_transform(*a), doc.world_transform(*b));
                    let box_overlap = depth.to_array();
                    // Closed parts: by the volume they share. Surfaces that
                    // cross may only scrape — pressed together by a hair, an
                    // import's rounding — and faces that lie in one plane fool
                    // a surface test into calling a plain overlap one part
                    // "inside" the other. Under a tenth of the tolerance deep
                    // is touching; all of the smaller one shared is inside.
                    if let Some((volume, deep)) =
                        kreate_mesh::intersect::shared_volume(ma, wa, mb, wb)
                    {
                        if deep < tolerance * 0.1 {
                            if volume > 0.0 {
                                touching.push(json!({ "a": an, "b": bn, "depth": deep }));
                            }
                            continue;
                        }
                        let own = |m, w: kreate_math::DAffine3| {
                            let det = w.matrix3.determinant().abs();
                            kreate_mesh::measure::mass_properties(m, |p| p).volume.abs() * det
                        };
                        let (va, vb) = (own(ma, wa), own(mb, wb));
                        let how = if volume >= va * 0.999 {
                            format!("{an} inside {bn}")
                        } else if volume >= vb * 0.999 {
                            format!("{bn} inside {an}")
                        } else {
                            "crossing".to_string()
                        };
                        pairs.push(json!({
                            "a": an, "b": bn, "contact": how,
                            "volume": volume, "depth": deep, "box_overlap": box_overlap,
                        }));
                        continue;
                    }
                    // Open, or too big to measure: by the surfaces alone.
                    let (ta, tb) = (
                        kreate_mesh::intersect::world_triangles(ma, wa),
                        kreate_mesh::intersect::world_triangles(mb, wb),
                    );
                    use kreate_mesh::intersect::Contact;
                    let how = match kreate_mesh::intersect::contact(&ta, &tb) {
                        Contact::Clear => continue,
                        Contact::Crossing => "crossing".to_string(),
                        Contact::Inside {
                            first_in_second: true,
                        } => format!("{an} inside {bn}"),
                        Contact::Inside {
                            first_in_second: false,
                        } => format!("{bn} inside {an}"),
                    };
                    pairs.push(
                        json!({ "a": an, "b": bn, "contact": how, "box_overlap": box_overlap }),
                    );
                }
            }
        }
        json!({ "checked": boxes.len(), "overlapping": pairs, "touching": touching })
    }

    pub(super) fn scene_json(&self) -> serde_json::Value {
        use kreate_scene::document::ObjectData;
        use serde_json::json;
        let doc = &self.doc;
        // Colours as the set_* operators take them — sRGB swatches, as a
        // colour picker holds them — so a value read here can be written back
        // unchanged. The linear numbers the renderer uses are not what a
        // script types.
        let rgb = |c: kreate_math::SceneRgb| {
            let s = c.as_swatch();
            [s.r, s.g, s.b]
        };
        let objects: Vec<_> = doc
            .objects()
            .map(|(id, o)| {
                let t = &o.transform;
                let r = kreate_scene::animation::to_euler(t.rotation);
                let kind = match o.data {
                    ObjectData::Empty => "empty",
                    ObjectData::Mesh(_) => "mesh",
                    ObjectData::Camera(_) => "camera",
                    ObjectData::Light(_) => "light",
                    ObjectData::Armature(_) => "armature",
                };
                let mut v = json!({
                    "name": o.name,
                    "type": kind,
                    "location": t.translation.to_array(),
                    // In radians, as every operator takes it, so a value read
                    // here goes back unchanged; and in degrees, to read.
                    "rotation": [r.x, r.y, r.z],
                    "rotation_degrees": [r.x.to_degrees(), r.y.to_degrees(), r.z.to_degrees()],
                    "scale": t.scale.to_array(),
                    "visible": doc.is_shown(id),
                    "selected": doc.selection.contains(id),
                    "active": doc.selection.active == Some(id),
                    "parent": o.parent.and_then(|p| doc.object(p)).map(|p| p.name.clone()),
                    "uid": doc.object_uid(id).map(|u| u.raw()),
                });
                match o.data {
                    ObjectData::Mesh(mesh) => {
                        if let Some(m) = doc.mesh(mesh) {
                            v["vertices"] = json!(m.num_verts());
                            v["faces"] = json!(m.num_faces());
                            // Blender's Dimensions: the mesh's own extent times
                            // the scale. And where it is in the world, which is
                            // what placing one thing on another needs.
                            let local = m.bounds();
                            if local.min.x <= local.max.x {
                                let size = (local.max - local.min).as_dvec3() * t.scale;
                                v["dimensions"] = json!(size.abs().to_array());
                                if let Some((lo, hi)) = doc.object_world_bounds(id) {
                                    v["bounds"] = json!({ "min": lo.to_array(), "max": hi.to_array() });
                                }
                            }
                        }
                        let slots: Vec<_> = (0..32)
                            .map_while(|slot| doc.material_id_at(id, slot))
                            .filter_map(|m| doc.material(m).map(|mat| json!({ "name": mat.name, "uid": doc.material_uid(m).map(|u| u.raw()) })))
                            .collect();
                        v["materials"] = json!(slots);
                    }
                    ObjectData::Light(light) => {
                        if let Some(l) = doc.light(light) {
                            v["light"] = json!({
                                "kind": format!("{:?}", l.kind),
                                "color": rgb(l.color),
                                "intensity": l.intensity,
                            });
                        }
                    }
                    ObjectData::Camera(camera) => {
                        if let Some(c) = doc.camera(camera) {
                            v["camera"] = json!({
                                "focal_length_mm": c.focal_length_mm,
                                "auto_exposure": c.auto_exposure,
                                "f_stop": c.f_stop,
                                "iso": c.iso,
                                "exposure_compensation": c.exposure_compensation,
                            });
                        }
                    }
                    _ => {}
                }
                v
            })
            .collect();
        let materials: Vec<_> = doc
            .materials()
            .map(|(mid, m)| {
                let users = doc
                    .objects()
                    .filter(|(oid, _)| {
                        (0..32)
                            .map_while(|slot| doc.material_id_at(*oid, slot))
                            .any(|x| x == mid)
                    })
                    .count();
                json!({
                    "name": m.name,
                    "uid": doc.material_uid(mid).map(|u| u.raw()),
                    "users": users,
                    "base_color": rgb(m.base_color),
                    "metallic": m.metallic,
                    "roughness": m.roughness,
                    "alpha": m.alpha,
                    "emission": rgb(m.emission),
                    "emission_strength_nits": m.emission_strength,
                    "transmission": m.transmission,
                    "ior": m.ior,
                    "uses_nodes": m.graph.is_some(),
                })
            })
            .collect();
        json!({
            // What every number here and every operator's parameters are in,
            // and the unit the user sees lengths in on screen.
            "units": {
                "length": "m",
                "angle": "rad",
                "display_length": doc.design.length_unit.tag(),
            },
            "file": self.file.path().map(|p| p.display().to_string()),
            "mode": self.op_ctx.mode.tag(),
            "frame": doc.playback.frame,
            "cursor": doc.cursor.to_array(),
            "objects": objects,
            "materials": materials,
            "images": doc
                .images()
                .map(|(iid, i)| json!({ "name": i.name, "uid": doc.image_uid(iid).map(|u| u.raw()), "width": i.width(), "height": i.height() }))
                .collect::<Vec<_>>(),
            // The exposure the picture is being made at, and whether the meter
            // chose it or the scene camera's dials did.
            "view": {
                "shading": format!("{:?}", self.ui.view_status.shading),
                "exposure_ev": self.ui.view_status.exposure,
                "metered": self.ui.view_status.metered,
                "through_camera": self.camera_view_object.and_then(|id| doc.object(id)).map(|o| o.name.clone()),
            },
            "last_operation": self
                .history
                .last()
                .map(|e| e.to_script_preview(kreate::shell::SCRIPT_PREVIEW)),
        })
    }

    /// The operators a script can call, with their parameters: the grammar
    /// `run` speaks.
    pub(super) fn operators_json(&self, filter: &str) -> serde_json::Value {
        use kreate_reflect::PropertyKind;
        use serde_json::json;
        let ops: Vec<_> = self
            .registry
            .iter()
            .filter(|info| info.id.starts_with(filter) || info.id.contains(filter))
            .map(|info| {
                let params: Vec<_> = info
                    .schema
                    .iter()
                    .filter(|d| !d.hidden)
                    .map(|d| {
                        let (kind, variants): (&str, &[&str]) = match d.kind {
                            PropertyKind::Bool => ("bool", &[]),
                            PropertyKind::Int => ("int", &[]),
                            PropertyKind::Float => ("float", &[]),
                            PropertyKind::Vec3 => ("vec3", &[]),
                            PropertyKind::Color => ("color", &[]),
                            PropertyKind::Text => ("text", &[]),
                            PropertyKind::Enum { variants } => ("enum", variants),
                            PropertyKind::Ints => ("ints", &[]),
                            PropertyKind::Floats => ("floats", &[]),
                        };
                        let mut p = json!({ "name": d.name, "kind": kind });
                        if !variants.is_empty() {
                            p["variants"] = json!(variants);
                        }
                        if let Some(min) = d.min {
                            p["min"] = json!(min);
                        }
                        if let Some(max) = d.max {
                            p["max"] = json!(max);
                        }
                        if !d.doc.is_empty() {
                            p["doc"] = json!(d.doc);
                        }
                        p
                    })
                    .collect();
                json!({
                    "id": info.id,
                    "label": info.label,
                    "available_now": info.poll(&self.op_ctx, &self.doc),
                    "params": params,
                })
            })
            .collect::<Vec<_>>();
        // The lines the application answers itself, listed with the operators
        // because a script writes them the same way.
        let mut ops = ops;
        for (id, doc, params) in kreate::appcall::LISTING {
            if id.starts_with(filter) || id.contains(filter) {
                ops.push(json!({
                    "id": id,
                    "label": doc,
                    "available_now": true,
                    "params": params.iter().map(|(name, doc)| json!({ "name": name, "doc": doc })).collect::<Vec<_>>(),
                }));
            }
        }
        if ops.len() <= FULL_LISTING {
            return json!({ "operators": ops });
        }
        // Many: one line each, names and kinds without the descriptions — a
        // whole family in full ran past what a reply holds, and was set aside
        // in a file a session with no file tools cannot read.
        let lines: Vec<String> = ops.iter().map(compact_operator).collect();
        json!({
            "operators": lines,
            "note": "Short form: a filter matching a few operators (an exact id) lists limits and descriptions. * marks one that cannot run right now.",
        })
    }
}

/// How many operators a listing gives in full before it gives one line each.
const FULL_LISTING: usize = 12;

/// One operator as a line: `id(name: kind, choice: a|b|c, ...)`.
fn compact_operator(op: &serde_json::Value) -> String {
    let params: Vec<String> = op["params"]
        .as_array()
        .map(|ps| {
            ps.iter()
                .map(|p| {
                    let name = p["name"].as_str().unwrap_or("");
                    match p["variants"].as_array() {
                        Some(v) if v.len() <= 12 => {
                            let names: Vec<&str> = v.iter().filter_map(|x| x.as_str()).collect();
                            format!("{name}: {}", names.join("|"))
                        }
                        Some(v) => format!("{name}: enum of {}", v.len()),
                        None => match p["kind"].as_str() {
                            Some(kind) => format!("{name}: {kind}"),
                            None => name.to_string(),
                        },
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    let unavailable = if op["available_now"].as_bool() == Some(false) {
        "*"
    } else {
        ""
    };
    format!(
        "{}{unavailable}({})",
        op["id"].as_str().unwrap_or(""),
        params.join(", ")
    )
}

#[cfg(test)]
mod listing_tests {
    use super::*;

    #[test]
    fn a_long_listing_is_one_line_an_operator() {
        let op = serde_json::json!({
            "id": "object.add_gear",
            "available_now": false,
            "params": [
                { "name": "kind", "kind": "enum", "variants": ["spur", "rack"], "doc": "long words" },
                { "name": "teeth", "kind": "int", "doc": "more words" },
            ],
        });
        assert_eq!(
            compact_operator(&op),
            "object.add_gear*(kind: spur|rack, teeth: int)"
        );
    }
}
