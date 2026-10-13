//! File ▸ Shared Session, on the application's side: hosting and joining,
//! the session's turn each frame, the others drawn in the viewport, and the
//! locks that keep two people out of one mesh's inside.

use super::*;

use kreate_collab::{Presence, Session, SessionEvent};
use kreate_render::OverlayLine;

/// The modes that edit the inside of an object, which one person at a time
/// may do.
fn edits_inside(mode: OpMode) -> bool {
    mode.acts_on() != kreate_ops::mode::ActsOn::Objects
}

impl App {
    pub(super) fn collab_host(&mut self, name: String, port: u16, key: String) {
        self.collab_leave_quietly();
        save_name(&name);
        match Session::host(port, &name, &key) {
            Ok(session) => {
                let message = self
                    .ui
                    .catalog
                    .get_args("collab-started", &[("address", &session.address)]);
                self.session = Some(session);
                self.status(&message);
            }
            Err(e) => self.collab_failed(&e.to_string()),
        }
    }

    pub(super) fn collab_join(&mut self, name: String, address: String, key: String) {
        self.collab_leave_quietly();
        save_name(&name);
        match Session::join(&address, &name, &key) {
            Ok(session) => self.session = Some(session),
            Err(e) => self.collab_failed(&e.to_string()),
        }
    }

    fn collab_failed(&mut self, reason: &str) {
        let message = self
            .ui
            .catalog
            .get_args("collab-failed", &[("reason", reason)]);
        self.status(&message);
    }

    pub(super) fn collab_leave(&mut self) {
        if self.session.is_some() {
            self.collab_leave_quietly();
            let message = self.ui.catalog.get("collab-left");
            self.status(&message);
        }
    }

    pub(super) fn collab_leave_quietly(&mut self) {
        if let Some(mut session) = self.session.take() {
            session.flush(&self.doc);
        }
        self.collab_locks.clear();
        self.doc.set_uid_site(0);
        self.ui.collab_view = None;
    }

    pub(super) fn open_collab(&mut self) {
        if self.ui.collab.name.is_empty() {
            self.ui.collab.name = load_name();
        }
        self.ui.collab.open = true;
    }

    /// The session's turn: once each time the event loop wakes.
    pub(super) fn collab_tick(&mut self) {
        if self.session.is_none() {
            return;
        }
        let aspect = self.viewport_aspect_of();
        let look = Some(presence(
            &self.doc,
            &self.camera,
            aspect,
            &self.collab_locks,
        ));
        let Some(session) = &mut self.session else {
            return;
        };
        let events = session.tick(&mut self.doc, look);
        let live = session.is_live();
        let closed = session.is_closed();
        for event in events {
            match event {
                SessionEvent::Connected { .. } => {
                    if !self.session.as_ref().is_some_and(|s| s.is_hosting()) {
                        let address = self.session.as_ref().map(|s| s.address.clone());
                        let message = self.ui.catalog.get_args(
                            "collab-joined",
                            &[("address", address.as_deref().unwrap_or(""))],
                        );
                        self.status(&message);
                    }
                }
                SessionEvent::SceneReplaced => {
                    // History was of the scene that is gone.
                    let doc = std::mem::take(&mut self.doc);
                    self.adopt(doc);
                    self.file = DocumentFile::new();
                    let message = self.ui.catalog.get("collab-replaced");
                    self.status(&message);
                }
                SessionEvent::Changed => {
                    self.redraw.set(RedrawReason::SCENE_CHANGED);
                }
                SessionEvent::Joined(peer) => {
                    let message = self
                        .ui
                        .catalog
                        .get_args("collab-peer-joined", &[("name", &peer.name)]);
                    self.status(&message);
                }
                SessionEvent::Left(peer) => {
                    let message = self
                        .ui
                        .catalog
                        .get_args("collab-peer-left", &[("name", &peer.name)]);
                    self.status(&message);
                }
                SessionEvent::Closed(reason) => {
                    let message = self
                        .ui
                        .catalog
                        .get_args("collab-closed", &[("reason", &reason)]);
                    self.status(&message);
                }
            }
        }
        if closed {
            self.collab_leave_quietly();
            return;
        }
        let Some(session) = &self.session else {
            return;
        };
        let name_of = |uid: u64| {
            self.doc
                .object_by_uid(kreate_scene::Uid::from_raw(uid))
                .and_then(|id| self.doc.object(id))
                .map(|o| o.name.clone())
        };
        let view = kreate_ui::collab::CollabView {
            live,
            hosting: session.is_hosting(),
            address: session.address.clone(),
            key: String::new(),
            color: session.color,
            peers: session
                .peers
                .values()
                .map(|p| {
                    let editing = session
                        .looks
                        .get(&p.site)
                        .and_then(|l| l.editing)
                        .and_then(name_of);
                    (p.name.clone(), p.color, editing)
                })
                .collect(),
            waiting: session.waiting(),
        };
        if self.ui.collab_view.as_ref() != Some(&view) {
            self.ui.collab_view = Some(view);
            self.redraw.set(RedrawReason::UI);
        }
        // Others' views move without any input here.
        if !session.looks.is_empty() {
            self.redraw.set(RedrawReason::UI);
        }
    }

    fn viewport_aspect_of(&self) -> f64 {
        self.viewport_px
            .map(|(_, _, w, h)| (w / h.max(1.0)) as f64)
            .unwrap_or(1.0)
    }

    /// After an undo or redo: others' edits stay.
    pub(super) fn collab_after_undo(&mut self) {
        if let Some(session) = &mut self.session {
            session.after_undo(&mut self.doc);
        }
    }

    /// Whether the session lets this participant into `mode`; takes the locks
    /// when it does and lets go of them on the way out.
    ///
    /// The objects whose inside a mode edits are the active one and, in edit
    /// mode, the selected meshes with it.
    pub(super) fn collab_may_enter(&mut self, mode: OpMode) -> bool {
        let Some(session) = &mut self.session else {
            return true;
        };
        if !edits_inside(mode) {
            for uid in self.collab_locks.drain(..) {
                session.unlock(uid);
            }
            return true;
        }
        let mut wanted: Vec<ObjectId> = self.doc.selection.active.into_iter().collect();
        if mode.acts_on() == kreate_ops::mode::ActsOn::MeshElements {
            wanted.extend(self.doc.selection.objects.iter().copied());
        }
        wanted.dedup();
        let uids: Vec<u64> = wanted
            .iter()
            .filter_map(|id| self.doc.object_uid(*id))
            .map(|u| u.raw())
            .collect();
        for (id, uid) in wanted.iter().zip(&uids) {
            if let Some(peer) = session.held_by_other(*uid) {
                let object = self
                    .doc
                    .object(*id)
                    .map(|o| o.name.clone())
                    .unwrap_or_default();
                let message = self.ui.catalog.get_args(
                    "collab-locked",
                    &[("name", &peer.name), ("object", &object)],
                );
                self.status(&message);
                return false;
            }
        }
        for uid in self.collab_locks.drain(..) {
            if !uids.contains(&uid) {
                session.unlock(uid);
            }
        }
        for uid in &uids {
            session.lock(*uid);
        }
        self.collab_locks = uids;
        true
    }
}

/// Where this participant is looking and what they hold.
fn presence(
    doc: &Document,
    camera: &kreate_render::Camera,
    aspect: f64,
    editing: &[u64],
) -> Presence {
    let v = |d: DVec3| [d.x, d.y, d.z];
    Presence {
        eye: v(camera.eye()),
        forward: v(camera.forward()),
        up: v(camera.up()),
        fov: camera.fov_y,
        aspect,
        selection: doc
            .selection
            .objects
            .iter()
            .filter_map(|id| doc.object_uid(*id))
            .map(|u| u.raw())
            .collect(),
        editing: editing.first().copied(),
    }
}

/// The others, for the viewport: each one's view as a small frustum where
/// they stand, and a box round what they have selected, in their colour.
pub(super) fn peer_lines(session: Option<&Session>, doc: &Document) -> Vec<OverlayLine> {
    let mut lines = Vec::new();
    let Some(session) = session else {
        return lines;
    };
    for (site, look) in &session.looks {
        let Some(peer) = session.peers.get(site) else {
            continue;
        };
        let c = peer.color;
        let color = [
            f32::from(c[0]) / 255.0,
            f32::from(c[1]) / 255.0,
            f32::from(c[2]) / 255.0,
            1.0,
        ];
        let line = |a: DVec3, b: DVec3| OverlayLine {
            a,
            b,
            color,
            width_px: 2.0,
            dashed: false,
            on_top: false,
        };
        let d = |a: [f64; 3]| DVec3::new(a[0], a[1], a[2]);
        let (eye, forward, up) = (d(look.eye), d(look.forward), d(look.up));
        if forward.length_squared() > 0.0 && up.length_squared() > 0.0 {
            let right = forward.cross(up).normalize_or_zero();
            // A frustum half a metre deep: a marker of where they are, not of
            // how far they can see.
            let depth = 0.5;
            let half_h = (look.fov.clamp(0.05, 3.0) * 0.5).tan() * depth;
            let half_w = half_h * look.aspect.clamp(0.2, 5.0);
            let centre = eye + forward * depth;
            let corners = [
                centre + right * half_w + up * half_h,
                centre - right * half_w + up * half_h,
                centre - right * half_w - up * half_h,
                centre + right * half_w - up * half_h,
            ];
            for i in 0..4 {
                lines.push(line(eye, corners[i]));
                lines.push(line(corners[i], corners[(i + 1) % 4]));
            }
            // Which way is up for them: a small roof over the top edge.
            let top = centre + up * (half_h * 1.6);
            lines.push(line(corners[0], top));
            lines.push(line(corners[1], top));
        }
        for uid in &look.selection {
            let Some(id) = doc.object_by_uid(kreate_scene::Uid::from_raw(*uid)) else {
                continue;
            };
            let Some((lo, hi)) = doc.object_world_bounds(id) else {
                continue;
            };
            let p = |x: bool, y: bool, z: bool| {
                DVec3::new(
                    if x { hi.x } else { lo.x },
                    if y { hi.y } else { lo.y },
                    if z { hi.z } else { lo.z },
                )
            };
            for (a, b) in [
                ((false, false, false), (true, false, false)),
                ((false, true, false), (true, true, false)),
                ((false, false, true), (true, false, true)),
                ((false, true, true), (true, true, true)),
                ((false, false, false), (false, true, false)),
                ((true, false, false), (true, true, false)),
                ((false, false, true), (false, true, true)),
                ((true, false, true), (true, true, true)),
                ((false, false, false), (false, false, true)),
                ((true, false, false), (true, false, true)),
                ((false, true, false), (false, true, true)),
                ((true, true, false), (true, true, true)),
            ] {
                let mut l = line(p(a.0, a.1, a.2), p(b.0, b.1, b.2));
                l.dashed = true;
                l.on_top = true;
                lines.push(l);
            }
        }
    }
    lines
}

fn name_path() -> Option<std::path::PathBuf> {
    kreate::prefs::config_dir().map(|d| d.join("collab-name"))
}

fn load_name() -> String {
    name_path()
        .and_then(|p| std::fs::read_to_string(p).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

fn save_name(name: &str) {
    if let Some(path) = name_path() {
        let _ = kreate::prefs::write_atomic(&path, name);
    }
}
