//! Renders the user asks for: F12's still and Ctrl+F12's animation, run a
//! frame at a time by the frame loop and written out, and the picture
//! read back for the connector's captures.

use super::*;

impl App {
    /// F12 and Ctrl+F12: a render through the scene camera, at the scene's
    /// render settings, run a frame at a time by the frame loop.
    pub(super) fn start_render_job(&mut self, animation: bool) {
        if self.render_job.is_some() || self.rendering_still || self.gfx.is_none() {
            return;
        }
        let Some(camera) = self.scene_camera(None) else {
            self.status("there is no camera to render through: add one with Add > Camera");
            return;
        };
        let settings = self.doc.render.clone();
        let folder = self.render_folder();
        let gfx = self.gfx.as_mut().expect("checked");
        let v = &gfx.viewport;
        let saved = SavedView {
            camera: self.camera,
            motion: self.motion,
            camera_view: self.camera_view,
            camera_view_object: self.camera_view_object,
            wanted_viewport: self.wanted_viewport,
            look: (
                v.shading,
                v.path_trace,
                v.show_grid,
                v.show_wireframe,
                v.show_outline,
                v.xray,
                v.face_orientation,
                v.sample_cap,
                v.path_trace_denoise,
            ),
            frame: self.doc.playback.frame,
        };
        // The rendered look goes on a frame later (`apply_render_look`):
        // the frame that first draws it builds the renderer's pipelines and
        // can take seconds, and a frame drawn before it puts up the
        // progress window, so the wait is shown rather than a frozen window.
        let _ = gfx;
        self.ui.render_progress = Some((self.ui.catalog.get("render-preparing"), f32::NAN));
        self.rendering_still = true;
        self.capturing = true;
        self.ui.render_result = None;
        let frames = animation.then_some((self.doc.playback.start, self.doc.playback.end));
        if let Some((first, _)) = frames {
            self.set_frame(first as f64);
        }
        // An animation starts through the camera its first frame is cut to.
        let camera = self.scene_camera(None).unwrap_or(camera);
        self.wanted_viewport = (settings.width, settings.height);
        self.render_job = Some(RenderJob {
            camera,
            settings,
            frames,
            folder,
            saved,
            started: Instant::now(),
            sampling: None,
            look: false,
            drawn: 0,
            ids: None,
            blur: None,
            written: Vec::new(),
            video: None,
        });
        // Motion blur: the shutter, as a fraction of a frame, from the
        // camera; a closed one has nothing to smear.
        let shutter = self.shutter_of(camera);
        if self.doc.render.motion_blur && shutter > 0.0 {
            let base = frames.map_or(self.doc.playback.frame, |(first, _)| first as f64);
            let samples = self.doc.render.samples;
            if let Some(gfx) = self.gfx.as_mut() {
                gfx.viewport.sample_cap = Some(Blur::samples_each(samples));
            }
            if let Some(job) = self.render_job.as_mut() {
                job.blur = Some(Blur {
                    step: 0,
                    steps: Blur::steps_for(samples),
                    shutter,
                    base,
                    sum: Vec::new(),
                });
            }
            self.set_frame(Blur::time(base, shutter, 0, Blur::steps_for(samples)));
        }
        self.aim_render_camera();
        self.redraw.set(RedrawReason::SCENE_CHANGED);
    }

    /// Looks through the render's camera, at the shot's aspect.
    pub(super) fn aim_render_camera(&mut self) {
        let Some(job) = &self.render_job else { return };
        let (object, before) = (job.camera, job.saved.camera);
        self.motion.interrupt(&self.camera);
        self.camera = self.view_through(object);
        self.motion.target = self.camera;
        if self.camera_view.is_none() {
            self.camera_view = Some(before);
        }
        self.camera_view_object = Some(object);
    }

    /// Where renders go when the settings name no place: beside the
    /// document, or in the pictures folder.
    pub(super) fn render_folder(&self) -> std::path::PathBuf {
        if let Some(dir) = self.file.path().and_then(|p| p.parent()) {
            return dir.to_path_buf();
        }
        let home = std::env::var_os("USERPROFILE")
            .or_else(|| std::env::var_os("HOME"))
            .map(std::path::PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        home.join("Pictures").join("Kreate")
    }

    /// One step of the render under way, after each drawn frame: counts the
    /// samples, and when the picture is done writes it and moves on.
    /// The render's look on the viewport: the path tracer at the settings'
    /// samples, denoising and bounces, and nothing drawn over the picture.
    fn apply_render_look(&mut self) {
        let Some(job) = self.render_job.as_mut() else {
            return;
        };
        job.look = true;
        job.drawn = 0;
        let settings = job.settings.clone();
        let blurred = job.blur.is_some();
        let Some(gfx) = self.gfx.as_mut() else {
            return;
        };
        let v = &mut gfx.viewport;
        v.shading = kreate_render::ShadingMode::Rendered;
        v.path_trace = true;
        v.show_grid = false;
        v.show_wireframe = false;
        v.show_outline = false;
        v.xray = false;
        v.face_orientation = false;
        v.sample_cap = Some(if blurred {
            Blur::samples_each(settings.samples)
        } else {
            settings.samples
        });
        v.path_trace_denoise = settings.denoise;
        v.path_trace_bounces = settings.bounces;
        self.redraw.set(RedrawReason::SCENE_CHANGED);
    }

    pub(super) fn advance_render_job(&mut self) {
        let Some(job) = self.render_job.as_mut() else {
            return;
        };
        job.drawn += 1;
        // The frame just drawn showed "preparing"; the next builds the
        // renderer.
        if !job.look {
            self.apply_render_look();
            return;
        }
        let (drawn, size, started) = (
            job.drawn,
            (job.settings.width, job.settings.height),
            job.started,
        );
        let frames = job.frames;
        let total = job.settings.samples;
        // A motion-blurred picture is drawn moment by moment across the
        // shutter, each with its share of the samples.
        let moment = job.blur.as_ref().map(|b| (b.step, b.steps));
        // The first frames of a picture still carry the last one's state —
        // the old size, the old frame — so they are drawn and not judged.
        if drawn <= 3 {
            self.wanted_viewport = size;
            self.aim_render_camera();
        }
        self.redraw.set(RedrawReason::ACCUMULATE);
        let Some(v) = self.gfx.as_ref().map(|g| &g.viewport) else {
            return;
        };
        let samples = v.path_trace_samples();
        let traced = v.path_tracing();
        let settled = drawn > 3
            && !v.accumulating()
            && ((traced && samples > 0) || (!traced && started.elapsed().as_secs() >= 5));
        if !settled {
            let detail = match moment {
                Some((m, steps)) => format!("shutter {}/{steps}", m + 1),
                None => format!("{samples}/{total} samples"),
            };
            let progress = match frames {
                Some((f, end)) => format!("rendering frame {f} of {end}: {detail}"),
                None => format!("rendering: {detail}"),
            };
            // How far through, for the bar: the frames done and this one's
            // share of its samples, or of its shutter.
            let share = match moment {
                Some((m, steps)) => m as f32 / steps.max(1) as f32,
                None => (samples as f32 / total.max(1) as f32).min(1.0),
            };
            let fraction = match frames {
                Some((f, end)) => {
                    let start = self.doc.playback.start;
                    let count = (end - start + 1).max(1) as f32;
                    ((f - start) as f32 + share) / count
                }
                None => share,
            };
            // Time left: what the samples so far took, for those to come.
            // From the first sample, not from the start, so the renderer's
            // preparing is not taken for slow sampling.
            let job = self.render_job.as_mut().expect("checked");
            if job.sampling.is_none() && fraction > 0.0 {
                job.sampling = Some((Instant::now(), fraction));
            }
            let left = job.sampling.and_then(|(since, at)| {
                let done = fraction - at;
                (done > 0.02).then(|| since.elapsed().as_secs_f32() * (1.0 - fraction) / done)
            });
            let left = match left {
                Some(s) => format!(
                    ", {}",
                    self.ui.catalog.format(
                        "render-time-left",
                        kreate_i18n::TextKind::Ui,
                        &[("seconds", &format!("{:.0}", s.max(0.0).ceil()))],
                    )
                ),
                None => String::new(),
            };
            self.ui.render_progress = Some((
                format!("{progress}, {:.0} s{left}", started.elapsed().as_secs_f32()),
                fraction,
            ));
            self.status(&format!(
                "{progress}, {:.0} s (Esc stops)",
                started.elapsed().as_secs_f32()
            ));
            return;
        }
        // Motion blur: this was one moment of the shutter. Added to the
        // mean, and the next moment drawn, until the shutter has closed.
        if self.render_job.as_ref().is_some_and(|j| j.blur.is_some()) {
            let forward = (self.camera.pivot - self.camera.eye())
                .normalize_or_zero()
                .as_vec3();
            // The exposure the picture is shown at, whatever set it: the
            // scene's meter, or the studio's own when it lights the scene.
            // Metered here, a scene with no lamp read as an unlit one and
            // was shown stops too bright.
            let exposure = self
                .gfx
                .as_ref()
                .map_or(0.0, |g| g.viewport.display().exposure);
            let passes = self
                .gfx
                .as_ref()
                .ok_or("the window closed".to_string())
                .and_then(|g| {
                    g.viewport
                        .read_passes(&g.gpu.device, &g.gpu.queue, exposure, forward)
                });
            let Ok(passes) = passes else {
                self.finish_render_job("motion blur: the picture could not be read back");
                return;
            };
            let job = self.render_job.as_mut().expect("checked");
            let blur = job.blur.as_mut().expect("checked");
            if blur.sum.len() != passes.combined.len() {
                blur.sum = vec![[0.0; 4]; passes.combined.len()];
            }
            for (s, c) in blur.sum.iter_mut().zip(&passes.combined) {
                for i in 0..4 {
                    s[i] += c[i];
                }
            }
            blur.step += 1;
            if blur.step < blur.steps {
                let t = Blur::time(blur.base, blur.shutter, blur.step, blur.steps);
                job.drawn = 0;
                self.set_frame(t);
                return;
            }
        }
        let job = self.render_job.as_ref().expect("checked");
        // An animation as a video: this frame into the one file.
        if frames.is_some() && job.settings.video() {
            let path = job.settings.video_path(&job.folder);
            if let Err(e) = self.write_video_frame(&path) {
                let message = self.ui.catalog.format(
                    "render-failed",
                    kreate_i18n::TextKind::Ui,
                    &[("why", &e)],
                );
                self.finish_render_job(&message);
                return;
            }
            if let Some((f, end)) = frames {
                if f < end {
                    self.next_render_frame(f, end);
                    return;
                }
            }
            // The last frame: the file finished in `finish_render_job`.
            let message = self.ui.catalog.format(
                "render-saved",
                kreate_i18n::TextKind::Ui,
                &[("path", &path.display().to_string())],
            );
            self.finish_render_job(&message);
            return;
        }
        let path = match frames {
            Some((f, _)) => job.settings.frame_path(&job.folder, f),
            None => job.settings.still_path(&job.folder),
        };
        let format = job.settings.format;
        match self.write_render(&path, format) {
            Ok(()) => {
                if let Some(job) = self.render_job.as_mut() {
                    job.written.push(path.clone());
                }
            }
            Err(e) => {
                let message = self.ui.catalog.format(
                    "render-failed",
                    kreate_i18n::TextKind::Ui,
                    &[("why", &e)],
                );
                self.finish_render_job(&message);
                return;
            }
        }
        if let Some((f, end)) = frames {
            if f < end {
                self.next_render_frame(f, end);
                return;
            }
        }
        let shown = match frames {
            Some(_) => path
                .parent()
                .map(|p| p.display().to_string())
                .unwrap_or_default(),
            None => path.display().to_string(),
        };
        let message = self.ui.catalog.format(
            "render-saved",
            kreate_i18n::TextKind::Ui,
            &[("path", &shown)],
        );
        self.finish_render_job(&message);
    }

    /// How long a camera's shutter is open, as a fraction of a frame.
    fn shutter_of(&self, camera: kreate_scene::document::ObjectId) -> f64 {
        self.doc
            .object(camera)
            .and_then(|o| match o.data {
                kreate_scene::document::ObjectData::Camera(c) => self.doc.camera(c),
                _ => None,
            })
            .map_or(0.0, |c| c.shutter)
    }

    /// Moves an animation's render on from frame `f`, of `end`.
    fn next_render_frame(&mut self, f: i32, end: i32) {
        self.set_frame((f + 1) as f64);
        // A marker cuts to another camera on this frame: the render cuts
        // with it. Chosen by the whole frame, not the moments a blurred one
        // is drawn at, which reach back into the frame before the cut.
        let cut = self
            .doc
            .marker_camera((f + 1) as f64)
            .map(|c| (c, self.shutter_of(c)));
        if let Some(job) = self.render_job.as_mut() {
            if let Some((camera, shutter)) = cut {
                job.camera = camera;
                if let Some(blur) = job.blur.as_mut() {
                    // A closed shutter on the new camera is the briefest
                    // open one: the render's blur was set up at the start.
                    blur.shutter = shutter.max(1e-3);
                }
            }
            job.frames = Some((f + 1, end));
            job.drawn = 0;
            job.ids = None;
            if let Some(blur) = job.blur.as_mut() {
                blur.step = 0;
                blur.base = (f + 1) as f64;
                blur.sum.clear();
                let t = Blur::time(blur.base, blur.shutter, 0, blur.steps);
                self.set_frame(t);
            }
        }
    }

    /// Adds the picture on screen to the animation's video, opening it at
    /// the first frame.
    fn write_video_frame(&mut self, path: &std::path::Path) -> Result<(), String> {
        let (w, h, png, _) = self.rendered_picture()?;
        let rgba = image::load_from_memory(&png)
            .map_err(|e| e.to_string())?
            .to_rgba8()
            .into_raw();
        let fps = self.doc.playback.fps;
        let job = self.render_job.as_mut().ok_or("the render stopped")?;
        if job.video.is_none() {
            let quality = job.settings.video_quality;
            job.video = Some(kreate::video::Video::new(path, w, h, fps, quality)?);
            job.written.push(path.to_path_buf());
        }
        job.video
            .as_mut()
            .expect("opened above")
            .push(&rgba, w, h)?;
        self.last_render = Some((w, h, rgba, path.display().to_string()));
        Ok(())
    }

    /// The picture on screen as a PNG, its size, and — for a motion-blurred
    /// one — the scene-linear mean it was made from.
    #[allow(clippy::type_complexity)]
    fn rendered_picture(&self) -> Result<(u32, u32, Vec<u8>, Option<Vec<[f32; 4]>>), String> {
        // A motion-blurred picture is the mean of the shutter's moments,
        // shown through the same view transform on the CPU; any other is the
        // one on screen.
        let blurred: Option<Vec<[f32; 4]>> = self
            .render_job
            .as_ref()
            .and_then(|j| j.blur.as_ref())
            .map(|b| {
                let n = b.steps.max(1) as f32;
                b.sum
                    .iter()
                    .map(|c| [c[0] / n, c[1] / n, c[2] / n, 1.0])
                    .collect()
            });

        let (w, h, png) = match &blurred {
            Some(mean) => {
                let (w, h) = self
                    .gfx
                    .as_ref()
                    .map(|g| g.viewport.render_size())
                    .ok_or("the window closed")?;
                // The look the picture is shown in; its exposure is in the
                // mean already.
                let view = kreate_math::DisplayTransform {
                    view: self
                        .gfx
                        .as_ref()
                        .map_or(kreate_math::ViewTransform::AgX, |g| {
                            g.viewport.display().view
                        }),
                    exposure: 0.0,
                };
                let mut rgba = Vec::with_capacity(mean.len() * 4);
                for c in mean {
                    let s = view
                        .apply(kreate_math::SceneRgb::new(c[0], c[1], c[2]))
                        .encode();
                    rgba.extend([s.r, s.g, s.b].map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8));
                    rgba.push(255);
                }
                let image = image::RgbaImage::from_raw(w, h, rgba)
                    .ok_or("the blurred picture has the wrong size")?;
                let mut png = Vec::new();
                image
                    .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
                    .map_err(|e| e.to_string())?;
                (w, h, png)
            }
            None => self.read_target_png()?,
        };
        Ok((w, h, png, blurred))
    }

    /// Writes the picture on screen to `path`: the display's PNG, or an EXR
    /// of the scene-linear picture with its depth and normal passes.
    pub(super) fn write_render(
        &mut self,
        path: &std::path::Path,
        format: kreate_scene::render::OutputFormat,
    ) -> Result<(), String> {
        let (w, h, png, blurred) = self.rendered_picture()?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        let bytes = match format {
            // A still with the video format set is written as its picture.
            kreate_scene::render::OutputFormat::Png | kreate_scene::render::OutputFormat::Mp4 => {
                png.clone()
            }
            kreate_scene::render::OutputFormat::Exr => {
                let exposure = scenelights::shot(&self.doc, self.time.rate(), self.camera.pivot)
                    .exposure as f32;
                let forward = (self.camera.pivot - self.camera.eye())
                    .normalize_or_zero()
                    .as_vec3();
                let gfx = self.gfx.as_ref().ok_or("the window closed")?;
                let mut passes =
                    gfx.viewport
                        .read_passes(&gfx.gpu.device, &gfx.gpu.queue, exposure, forward)?;
                if let Some(mean) = blurred {
                    passes.combined = mean;
                }
                let crypto = self
                    .render_job
                    .as_ref()
                    .and_then(|j| j.ids.as_ref())
                    .map(|ids| {
                        let names: Vec<Option<String>> = ids
                            .iter()
                            .map(|&key| {
                                key.checked_sub(1)
                                    .and_then(|k| kreate::shell::object_from_key(&self.doc, k))
                                    .and_then(|id| self.doc.object(id))
                                    .map(|o| o.name.clone())
                            })
                            .collect();
                        names
                    });
                exr_of(&passes, crypto.as_deref())
            }
        };
        std::fs::write(path, &bytes).map_err(|e| format!("{}: {e}", path.display()))?;
        let rgba = image::load_from_memory(&png)
            .map_err(|e| e.to_string())?
            .to_rgba8()
            .into_raw();
        self.last_render = Some((w, h, rgba, path.display().to_string()));
        Ok(())
    }

    /// Ends the render, done or stopped: the view goes back as it was and the
    /// last picture is shown.
    pub(super) fn finish_render_job(&mut self, message: &str) {
        let Some(mut job) = self.render_job.take() else {
            return;
        };
        self.ui.render_progress = None;
        // A video's index, which is what makes it play: done, or stopped
        // part way, it holds the frames it was given.
        let mut message = message.to_string();
        if let Some(video) = job.video.take() {
            let path = video.path().display().to_string();
            if let Err(e) = video.finish() {
                message = self.ui.catalog.format(
                    "render-failed",
                    kreate_i18n::TextKind::Ui,
                    &[("why", &e)],
                );
            } else if let Some((_, _, _, shown)) = &mut self.last_render {
                *shown = path;
            }
        }
        let message = message.as_str();
        let saved = job.saved;
        self.rendering_still = false;
        self.capturing = false;
        if let Some(gfx) = self.gfx.as_mut() {
            let v = &mut gfx.viewport;
            (
                v.shading,
                v.path_trace,
                v.show_grid,
                v.show_wireframe,
                v.show_outline,
                v.xray,
                v.face_orientation,
                v.sample_cap,
                v.path_trace_denoise,
            ) = saved.look;
            v.path_trace_bounces = 4;
        }
        self.camera = saved.camera;
        self.motion = saved.motion;
        self.camera_view = saved.camera_view;
        self.camera_view_object = saved.camera_view_object;
        self.wanted_viewport = saved.wanted_viewport;
        // An animation or a blurred still moved the clock; it goes back.
        if job.frames.is_some() || job.blur.is_some() {
            self.set_frame(saved.frame);
        }
        if let Some((_, _, _, path)) = &mut self.last_render {
            if !job.written.is_empty() {
                *path = message.to_string();
            }
        }
        // What the render wrote, for the result window to open and play.
        if !job.written.is_empty() {
            self.last_render_files = std::mem::take(&mut job.written);
        }
        self.show_last_render();
        if job.written.is_empty() {
            if let Some(view) = &mut self.ui.render_result {
                view.message = message.to_string();
            }
        }
        self.status(message);
        self.redraw.set(RedrawReason::CAMERA_MOVED);
    }

    /// View Render: the last render's picture, in its window.
    pub(super) fn show_last_render(&mut self) {
        let Some((w, h, rgba, message)) = &self.last_render else {
            self.status("nothing has been rendered yet: Render > Render Image (F12)");
            return;
        };
        let image = egui::ColorImage::from_rgba_unmultiplied([*w as usize, *h as usize], rgba);
        let texture =
            self.egui_ctx
                .load_texture("render-result", image, egui::TextureOptions::LINEAR);
        self.ui.render_result = Some(kreate::shell::RenderResultView {
            texture,
            size: (*w, *h),
            message: message.clone(),
            files: self.last_render_files.clone(),
            fps: self.doc.playback.fps,
            player: None,
        });
        self.redraw.set(RedrawReason::UI);
    }

    /// Renders the scene through a camera at a size of its own: the rendered
    /// shading, path traced to `samples`, with no overlays — and writes it to
    /// `path` as a PNG. The window's own view is put back afterwards.
    ///
    /// The viewport is the renderer, so for the length of the render it draws
    /// the shot at the shot's size; the panel shows it stretched meanwhile.
    pub(super) fn render_still(
        &mut self,
        camera: Option<&str>,
        (width, height): (u32, u32),
        samples: u32,
        path: &std::path::Path,
        limit: std::time::Duration,
    ) -> Result<serde_json::Value, String> {
        if self.gfx.is_none() {
            return Err("the window is not open yet".into());
        }
        let object = self.scene_camera(camera).ok_or(match camera {
            Some(n) => format!("no camera called {n:?}"),
            None => "the scene has no camera; add one with object.add_camera".to_string(),
        })?;
        let size = (width.clamp(16, 8192), height.clamp(16, 8192));

        // Everything this changes, to put back.
        let view = (
            self.camera,
            self.motion,
            self.camera_view,
            self.camera_view_object,
            self.wanted_viewport,
        );
        let gfx = self.gfx.as_mut().expect("checked");
        let look = {
            let v = &gfx.viewport;
            (
                v.shading,
                v.path_trace,
                v.show_grid,
                v.show_wireframe,
                v.show_outline,
                v.xray,
                v.face_orientation,
                v.sample_cap,
            )
        };
        {
            let v = &mut gfx.viewport;
            v.shading = kreate_render::ShadingMode::Rendered;
            v.path_trace = true;
            v.show_grid = false;
            v.show_wireframe = false;
            v.show_outline = false;
            v.xray = false;
            v.face_orientation = false;
            v.sample_cap = Some(samples);
        }
        self.rendering_still = true;
        self.capturing = true;

        // One frame at the shot's size first: the lens is fitted to the
        // aspect the target has, which is the new one only once it is there.
        self.wanted_viewport = size;
        self.render();
        self.motion.interrupt(&self.camera);
        self.camera = self.view_through(object);
        self.motion.target = self.camera;
        self.camera_view = Some(view.0);
        self.camera_view_object = Some(object);

        // Until the path tracer has summed the samples. It has to have the
        // scene first, which takes a frame or a few to build; until then the
        // view is not path traced and "not accumulating" means nothing. If it
        // never gets one — a device without the capability — the raster
        // picture is what there is, after a grace period.
        let started = Instant::now();
        let grace = std::time::Duration::from_secs(5).min(limit);
        let mut finished = false;
        while started.elapsed() < limit {
            self.wanted_viewport = size;
            self.restart_mean_if_changed();
            self.render();
            self.redraw.clear();
            let Some(v) = self.gfx.as_ref().map(|g| &g.viewport) else {
                break;
            };
            let traced = v.path_tracing();
            if (traced && v.path_trace_samples() > 0 && !v.accumulating())
                || (!traced && started.elapsed() >= grace && !v.accumulating())
            {
                finished = true;
                break;
            }
        }
        let summed = self
            .gfx
            .as_ref()
            .map(|g| g.viewport.path_trace_samples())
            .unwrap_or(0);
        let shot = self.read_target_png();

        // Back as it was, whatever happened.
        self.rendering_still = false;
        self.capturing = false;
        if let Some(gfx) = self.gfx.as_mut() {
            let v = &mut gfx.viewport;
            (
                v.shading,
                v.path_trace,
                v.show_grid,
                v.show_wireframe,
                v.show_outline,
                v.xray,
                v.face_orientation,
                v.sample_cap,
            ) = look;
        }
        (
            self.camera,
            self.motion,
            self.camera_view,
            self.camera_view_object,
            self.wanted_viewport,
        ) = view;
        self.redraw.set(RedrawReason::CAMERA_MOVED);

        let (w, h, png) = shot?;
        std::fs::write(path, &png).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        // A small copy to look at: the file is the render, this is for checking it.
        let preview = image::load_from_memory(&png)
            .map(|i| i.thumbnail(800, 800))
            .map_err(|e| e.to_string())
            .and_then(|i| {
                let mut out = Vec::new();
                i.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
                    .map_err(|e| e.to_string())
                    .map(|_| out)
            })?;
        Ok(serde_json::json!({
            "path": path.display().to_string(),
            "width": w,
            "height": h,
            "samples": summed,
            "finished": finished,
            "png_base64": kreate::bridge::base64(&preview),
        }))
    }

    /// The viewport as a PNG, once the still view has settled.
    pub(super) fn capture_png(&mut self) -> Result<(u32, u32, Vec<u8>), String> {
        if self.gfx.is_none() {
            return Err("the window is not open yet".into());
        }
        // Draw until the picture stops changing: the still view's mean, or
        // the path tracer's, filled. Bounded, so a view that never settles
        // still answers.
        let started = Instant::now();
        self.capturing = true;
        self.restart_mean_if_changed();
        self.render();
        self.redraw.clear();
        while self.gfx.as_ref().is_some_and(|g| g.viewport.accumulating())
            && started.elapsed() < std::time::Duration::from_secs(30)
        {
            self.render();
            self.redraw.clear();
        }
        self.capturing = false;
        self.read_target_png()
    }

    /// The viewport's colour target as it is, as a PNG.
    pub(super) fn read_target_png(&self) -> Result<(u32, u32, Vec<u8>), String> {
        let gfx = self.gfx.as_ref().ok_or("the window closed")?;
        let (w, h) = gfx.viewport.render_size();
        let row = (w * 4).div_ceil(256) * 256;
        let buffer = gfx.gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("capture"),
            size: (row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = gfx
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("capture"),
            });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &gfx.target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        gfx.gpu.queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        gfx.gpu
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("the GPU did not finish: {e:?}"))?;
        let mut pixels = Vec::with_capacity((w * h * 4) as usize);
        {
            let data = buffer.slice(..).get_mapped_range();
            for y in 0..h as usize {
                let start = y * row as usize;
                pixels.extend_from_slice(&data[start..start + (w * 4) as usize]);
            }
        }
        buffer.unmap();
        // Opaque: the target's alpha is a compositing detail, not the picture.
        for p in pixels.as_chunks_mut::<4>().0 {
            p[3] = 255;
        }
        let image =
            image::RgbaImage::from_raw(w, h, pixels).ok_or("the capture has the wrong size")?;
        let mut png = Vec::new();
        image
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .map_err(|e| format!("cannot encode the capture: {e}"))?;
        Ok((w, h, png))
    }
}
