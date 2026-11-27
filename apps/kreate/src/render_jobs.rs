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
        let Some(camera) = self
            .scene_camera(None)
            .or_else(|| self.align_or_create_camera_to_view())
        else {
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
        let shutter = self.shutter_of(camera);
        let motion_blur_active = settings.motion_blur && shutter > 0.0;
        let tiles = if !motion_blur_active
            && (settings.width > TileGrid::TILE_SIZE || settings.height > TileGrid::TILE_SIZE)
        {
            Some(TileGrid::new(settings.width, settings.height))
        } else {
            None
        };
        self.wanted_viewport = tiles
            .as_ref()
            .and_then(|t| t.current_rect().map(|[_, _, tw, th]| (tw, th)))
            .unwrap_or((settings.width, settings.height));
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
            tiles,
            written: Vec::new(),
            video: None,
            denoised: false,
        });
        // Motion blur: the shutter, as a fraction of a frame, from the
        // camera; a closed one has nothing to smear.
        if motion_blur_active {
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
                    albedo_sum: Vec::new(),
                    normal_sum: Vec::new(),
                    denoised_mean: None,
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
        let tile_uniform = job.tiles.as_ref().and_then(|t| t.current_uniform());
        let tiled = job.tiles.is_some();
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
        // Tiled renders denoise once on the stitched full frame (no tile seams).
        v.path_trace_denoise = settings.denoise && !tiled;
        v.path_trace_bounces = settings.bounces;
        v.transparent_background = settings.transparent_background;
        v.path_trace_tile = tile_uniform;
        v.path_trace_noise_threshold = quiet_enough(settings.samples);
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
        let tile_info = job.tiles.as_ref().and_then(|t| {
            t.current_rect()
                .map(|r| (t.index, t.rects.len(), r, [r[0], r[1], t.full_w, t.full_h]))
        });
        let (drawn, size, started) = (
            job.drawn,
            tile_info
                .map(|(_, _, [_, _, tw, th], _)| (tw, th))
                .unwrap_or((job.settings.width, job.settings.height)),
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
            if let (Some(gfx), Some((_, _, _, uniform))) = (self.gfx.as_mut(), tile_info) {
                gfx.viewport.path_trace_tile = Some(uniform);
            }
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
            let detail = match (moment, tile_info) {
                (Some((m, steps)), _) => format!("shutter {}/{steps}", m + 1),
                (None, Some((ti, tn, _, _))) => {
                    format!("tile {}/{tn}: {samples}/{total} samples", ti + 1)
                }
                (None, None) => format!("{samples}/{total} samples"),
            };
            let progress = match frames {
                Some((f, end)) => format!("rendering frame {f} of {end}: {detail}"),
                None => format!("rendering: {detail}"),
            };
            // How far through, for the bar: the frames done and this one's
            // share of its samples, tiles, or shutter.
            let share = match (moment, tile_info) {
                (Some((m, steps)), _) => m as f32 / steps.max(1) as f32,
                (None, Some((ti, tn, _, _))) => {
                    let s = (samples as f32 / total.max(1) as f32).min(1.0);
                    (ti as f32 + s) / tn.max(1) as f32
                }
                (None, None) => (samples as f32 / total.max(1) as f32).min(1.0),
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
        // Tiled 4K/8K render: copy this finished tile into the full-frame
        // stitched buffers and move to the next tile until all tiles are done.
        if let Some((ti, tn, [tx, ty, tw, th], _)) = tile_info {
            let forward = (self.camera.pivot - self.camera.eye())
                .normalize_or_zero()
                .as_vec3();
            let exposure = self
                .gfx
                .as_ref()
                .map_or(0.0, |g| g.viewport.display().exposure);
            let read_res = self
                .gfx
                .as_ref()
                .ok_or("the window closed".to_string())
                .and_then(|g| {
                    let passes =
                        g.viewport
                            .read_passes(&g.gpu.device, &g.gpu.queue, exposure, forward)?;
                    let traced = g.viewport.read_traced(&g.gpu.device, &g.gpu.queue).ok();
                    Ok((passes, traced))
                });
            let Ok((passes, traced)) = read_res else {
                self.finish_render_job("tiled render: tile could not be read back");
                return;
            };
            let job = self.render_job.as_mut().expect("checked");
            let grid = job.tiles.as_mut().expect("checked");
            grid.exposure = exposure;
            let fw = grid.full_w as usize;
            let cw = passes.width.min(tw) as usize;
            let ch = passes.height.min(th) as usize;
            for row in 0..ch {
                let dst_y = ty as usize + row;
                let dst_base = dst_y * fw + tx as usize;
                let src_base = row * passes.width as usize;
                grid.combined[dst_base..dst_base + cw]
                    .copy_from_slice(&passes.combined[src_base..src_base + cw]);
                grid.normal_pass[dst_base..dst_base + cw]
                    .copy_from_slice(&passes.normal[src_base..src_base + cw]);
                grid.depth[dst_base..dst_base + cw]
                    .copy_from_slice(&passes.depth[src_base..src_base + cw]);
                if let Some(ref g) = traced {
                    if g.width == passes.width && g.height == passes.height {
                        grid.albedo[dst_base..dst_base + cw]
                            .copy_from_slice(&g.albedo[src_base..src_base + cw]);
                        grid.normal_oidn[dst_base..dst_base + cw]
                            .copy_from_slice(&g.normal[src_base..src_base + cw]);
                    }
                }
            }
            grid.index += 1;
            if ti + 1 < tn {
                let next_rect = grid.current_rect().expect("more tiles");
                let next_uniform = grid.current_uniform();
                job.drawn = 0;
                self.wanted_viewport = (next_rect[2], next_rect[3]);
                if let Some(gfx) = self.gfx.as_mut() {
                    gfx.viewport.path_trace_tile = next_uniform;
                    gfx.viewport.restart_accumulation();
                }
                self.redraw.set(RedrawReason::SCENE_CHANGED);
                return;
            }
            grid.done = true;
            // Run Open Image Denoise once on the full stitched frame (zero tile seams).
            if job.settings.denoise && kreate_render::oidn::available() {
                let exposure_scale = 2.0f32.powf(grid.exposure);
                let inv_scale = if exposure_scale > 1e-8 {
                    1.0 / exposure_scale
                } else {
                    1.0
                };
                let color: Vec<[f32; 3]> = grid
                    .combined
                    .iter()
                    .map(|c| [c[0] * inv_scale, c[1] * inv_scale, c[2] * inv_scale])
                    .collect();
                let started_oidn = Instant::now();
                match kreate_render::oidn::denoise(&kreate_render::oidn::Noisy {
                    width: grid.full_w as usize,
                    height: grid.full_h as usize,
                    color: &color,
                    albedo: &grid.albedo,
                    normal: &grid.normal_oidn,
                }) {
                    Ok(clean) => {
                        for (dst, rgb) in grid.combined.iter_mut().zip(&clean) {
                            dst[0] = rgb[0] * exposure_scale;
                            dst[1] = rgb[1] * exposure_scale;
                            dst[2] = rgb[2] * exposure_scale;
                        }
                        tracing::info!(
                            "denoised tiled {}x{} render with OIDN in {:.2} s",
                            grid.full_w,
                            grid.full_h,
                            started_oidn.elapsed().as_secs_f32()
                        );
                    }
                    Err(e) => {
                        tracing::warn!("Open Image Denoise (tiled): {e}");
                    }
                }
            }
        }
        // The samples are in. With Open Image Denoise installed, the picture
        // is filtered by it rather than by the view's own blur: read back,
        // denoised on the CPU, put in place of the mean, and shown for a
        // frame before it is written.
        if self.denoise_with_oidn() {
            return;
        }
        // Motion blur: this was one moment of the shutter. Added to the
        // mean, and the next moment drawn, until the shutter has closed.
        if self.render_job.as_ref().is_some_and(|j| j.blur.is_some()) {
            let forward = (self.camera.pivot - self.camera.eye())
                .normalize_or_zero()
                .as_vec3();
            let exposure = self
                .gfx
                .as_ref()
                .map_or(0.0, |g| g.viewport.display().exposure);
            let read_res = self
                .gfx
                .as_ref()
                .ok_or("the window closed".to_string())
                .and_then(|g| {
                    let passes =
                        g.viewport
                            .read_passes(&g.gpu.device, &g.gpu.queue, exposure, forward)?;
                    let traced = g.viewport.read_traced(&g.gpu.device, &g.gpu.queue).ok();
                    Ok((passes, traced))
                });
            let Ok((passes, traced)) = read_res else {
                self.finish_render_job("motion blur: the picture could not be read back");
                return;
            };
            let job = self.render_job.as_mut().expect("checked");
            let denoise_on = job.settings.denoise;
            let blur = job.blur.as_mut().expect("checked");
            if blur.sum.len() != passes.combined.len() {
                blur.sum = vec![[0.0; 4]; passes.combined.len()];
            }
            for (s, c) in blur.sum.iter_mut().zip(&passes.combined) {
                for i in 0..4 {
                    s[i] += c[i];
                }
            }
            if let Some(ref g) = traced {
                if blur.albedo_sum.len() != g.albedo.len() {
                    blur.albedo_sum = vec![[0.0; 3]; g.albedo.len()];
                    blur.normal_sum = vec![[0.0; 3]; g.normal.len()];
                }
                for (a, b) in blur.albedo_sum.iter_mut().zip(&g.albedo) {
                    a[0] += b[0];
                    a[1] += b[1];
                    a[2] += b[2];
                }
                for (a, b) in blur.normal_sum.iter_mut().zip(&g.normal) {
                    a[0] += b[0];
                    a[1] += b[1];
                    a[2] += b[2];
                }
            }
            blur.step += 1;
            if blur.step < blur.steps {
                let t = Blur::time(blur.base, blur.shutter, blur.step, blur.steps);
                job.drawn = 0;
                self.set_frame(t);
                return;
            }
            // Run OIDN on the final motion-blur averaged linear buffer.
            if denoise_on && kreate_render::oidn::available() {
                let n = blur.steps.max(1) as f32;
                let exposure_scale = 2.0f32.powf(exposure);
                let inv_scale = if exposure_scale > 1e-8 {
                    1.0 / exposure_scale
                } else {
                    1.0
                };
                let color: Vec<[f32; 3]> = blur
                    .sum
                    .iter()
                    .map(|c| {
                        [
                            (c[0] / n) * inv_scale,
                            (c[1] / n) * inv_scale,
                            (c[2] / n) * inv_scale,
                        ]
                    })
                    .collect();
                let albedo: Vec<[f32; 3]> = if blur.albedo_sum.len() == color.len() {
                    blur.albedo_sum
                        .iter()
                        .map(|v| [v[0] / n, v[1] / n, v[2] / n])
                        .collect()
                } else {
                    vec![[1.0; 3]; color.len()]
                };
                let normal: Vec<[f32; 3]> = if blur.normal_sum.len() == color.len() {
                    blur.normal_sum
                        .iter()
                        .map(|v| [v[0] / n, v[1] / n, v[2] / n])
                        .collect()
                } else {
                    vec![[0.0; 3]; color.len()]
                };
                match kreate_render::oidn::denoise(&kreate_render::oidn::Noisy {
                    width: passes.width as usize,
                    height: passes.height as usize,
                    color: &color,
                    albedo: &albedo,
                    normal: &normal,
                }) {
                    Ok(clean) => {
                        let mean: Vec<[f32; 4]> = blur
                            .sum
                            .iter()
                            .zip(&clean)
                            .map(|(orig, rgb)| {
                                [
                                    rgb[0] * exposure_scale,
                                    rgb[1] * exposure_scale,
                                    rgb[2] * exposure_scale,
                                    (orig[3] / n).clamp(0.0, 1.0),
                                ]
                            })
                            .collect();
                        blur.denoised_mean = Some(mean);
                    }
                    Err(e) => {
                        tracing::warn!("Open Image Denoise (motion blur): {e}");
                    }
                }
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
    /// Runs Open Image Denoise over the finished picture, once, when the
    /// render asks for denoising and the library is installed. `true` when
    /// it has just done so, and the picture wants a frame to be shown before
    /// it is written.
    fn denoise_with_oidn(&mut self) -> bool {
        let Some(job) = self.render_job.as_mut() else {
            return false;
        };
        // Tiled renders and motion-blur renders run OIDN on their stitched /
        // averaged linear buffer directly.
        if job.denoised || !job.settings.denoise || job.blur.is_some() || job.tiles.is_some() {
            return false;
        }
        job.denoised = true;
        if !kreate_render::oidn::available() {
            return false;
        }
        let Some(gfx) = self.gfx.as_mut() else {
            return false;
        };
        let started = Instant::now();
        let result = gfx
            .viewport
            .read_traced(&gfx.gpu.device, &gfx.gpu.queue)
            .and_then(|gathered| {
                let clean = kreate_render::oidn::denoise(&kreate_render::oidn::Noisy {
                    width: gathered.width as usize,
                    height: gathered.height as usize,
                    color: &gathered.color,
                    albedo: &gathered.albedo,
                    normal: &gathered.normal,
                })?;
                Ok((gathered, clean))
            });
        match result {
            Ok((gathered, clean)) => {
                gfx.viewport.path_trace_denoise = false;
                gfx.viewport
                    .show_traced_mean(&gfx.gpu.queue, &gathered, &clean);
                tracing::info!(
                    "denoised with Open Image Denoise in {:.2} s",
                    started.elapsed().as_secs_f32()
                );
                self.redraw.set(RedrawReason::ACCUMULATE);
                true
            }
            Err(e) => {
                // The view's own filter has the picture as it was.
                tracing::warn!("Open Image Denoise: {e}; the view's own filter is kept");
                false
            }
        }
    }

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
            job.denoised = false;
            if let Some(grid) = job.tiles.as_mut() {
                grid.reset();
                if let Some(rect) = grid.current_rect() {
                    self.wanted_viewport = (rect[2], rect[3]);
                }
            }
            // The next frame is filtered as the settings say until its own
            // samples are in: the last one's denoising turned the view's off.
            let denoise = job.settings.denoise && job.tiles.is_none();
            let tile_uniform = job.tiles.as_ref().and_then(|t| t.current_uniform());
            if let Some(gfx) = self.gfx.as_mut() {
                gfx.viewport.path_trace_denoise = denoise;
                gfx.viewport.path_trace_tile = tile_uniform;
            }
            if let Some(blur) = job.blur.as_mut() {
                blur.step = 0;
                blur.base = (f + 1) as f64;
                blur.sum.clear();
                blur.albedo_sum.clear();
                blur.normal_sum.clear();
                blur.denoised_mean = None;
                let t = Blur::time(blur.base, blur.shutter, 0, blur.steps);
                self.set_frame(t);
            }
        }
    }

    /// Adds the picture on screen to the animation's video, opening it at
    /// the first frame.
    fn write_video_frame(&mut self, path: &std::path::Path) -> Result<(), String> {
        let (w, h, rgba, _) = self.rendered_rgba()?;
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

    /// The picture on screen as raw RGBA8 rows, its size, and — for a
    /// motion-blurred or tiled one — the scene-linear mean it was made from.
    #[allow(clippy::type_complexity)]
    fn rendered_rgba(&self) -> Result<(u32, u32, Vec<u8>, Option<Vec<[f32; 4]>>), String> {
        let transparent = self
            .render_job
            .as_ref()
            .is_some_and(|j| j.settings.transparent_background);
        let view = kreate_math::DisplayTransform {
            view: self
                .gfx
                .as_ref()
                .map_or(kreate_math::ViewTransform::AgX, |g| {
                    g.viewport.display().view
                }),
            exposure: 0.0,
        };
        // Tiled 4K/8K render: tonemap the stitched full-frame linear buffer.
        if let Some(grid) = self
            .render_job
            .as_ref()
            .and_then(|j| j.tiles.as_ref())
            .filter(|g| g.done)
        {
            let mut rgba = Vec::with_capacity(grid.combined.len() * 4);
            for c in &grid.combined {
                let s = view
                    .apply(kreate_math::SceneRgb::new(c[0], c[1], c[2]))
                    .encode();
                rgba.extend([s.r, s.g, s.b].map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8));
                let a = if transparent {
                    (c[3].clamp(0.0, 1.0) * 255.0).round() as u8
                } else {
                    255
                };
                rgba.push(a);
            }
            return Ok((grid.full_w, grid.full_h, rgba, Some(grid.combined.clone())));
        }
        // A motion-blurred picture is the mean of the shutter's moments,
        // shown through the same view transform on the CPU; any other is the
        // one on screen.
        let blurred: Option<Vec<[f32; 4]>> = self
            .render_job
            .as_ref()
            .and_then(|j| j.blur.as_ref())
            .map(|b| {
                if let Some(ref clean) = b.denoised_mean {
                    return clean.clone();
                }
                let n = b.steps.max(1) as f32;
                b.sum
                    .iter()
                    .map(|c| {
                        [
                            c[0] / n,
                            c[1] / n,
                            c[2] / n,
                            if transparent {
                                (c[3] / n).clamp(0.0, 1.0)
                            } else {
                                1.0
                            },
                        ]
                    })
                    .collect()
            });

        let (w, h, rgba) = match &blurred {
            Some(mean) => {
                let (w, h) = self
                    .gfx
                    .as_ref()
                    .map(|g| g.viewport.render_size())
                    .ok_or("the window closed")?;
                let mut rgba = Vec::with_capacity(mean.len() * 4);
                for c in mean {
                    let s = view
                        .apply(kreate_math::SceneRgb::new(c[0], c[1], c[2]))
                        .encode();
                    rgba.extend([s.r, s.g, s.b].map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8));
                    let a = if transparent {
                        (c[3].clamp(0.0, 1.0) * 255.0).round() as u8
                    } else {
                        255
                    };
                    rgba.push(a);
                }
                if rgba.len() != (w as usize) * (h as usize) * 4 {
                    return Err("the blurred picture has the wrong size".to_string());
                }
                (w, h, rgba)
            }
            None => self.read_target_rgba()?,
        };
        Ok((w, h, rgba, blurred))
    }

    /// The picture on screen as a PNG, its size, and — for a motion-blurred
    /// one — the scene-linear mean it was made from.
    #[allow(clippy::type_complexity)]
    fn rendered_picture(&self) -> Result<(u32, u32, Vec<u8>, Option<Vec<[f32; 4]>>), String> {
        let (w, h, rgba, blurred) = self.rendered_rgba()?;
        let image =
            image::RgbaImage::from_raw(w, h, rgba).ok_or("the capture has the wrong size")?;
        let mut png = Vec::new();
        image
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .map_err(|e| format!("cannot encode the capture: {e}"))?;
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
                let passes = if let Some(grid) = self
                    .render_job
                    .as_ref()
                    .and_then(|j| j.tiles.as_ref())
                    .filter(|g| g.done)
                {
                    kreate_render::viewport::RenderPasses {
                        width: grid.full_w,
                        height: grid.full_h,
                        combined: grid.combined.clone(),
                        normal: grid.normal_pass.clone(),
                        depth: grid.depth.clone(),
                    }
                } else {
                    let forward = (self.camera.pivot - self.camera.eye())
                        .normalize_or_zero()
                        .as_vec3();
                    let gfx = self.gfx.as_ref().ok_or("the window closed")?;
                    let exposure = gfx.viewport.display().exposure;
                    let mut passes = gfx.viewport.read_passes(
                        &gfx.gpu.device,
                        &gfx.gpu.queue,
                        exposure,
                        forward,
                    )?;
                    if let Some(mean) = blurred {
                        passes.combined = mean;
                    }
                    passes
                };
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
            v.transparent_background = false;
            v.path_trace_tile = None;
            v.path_trace_noise_threshold = quiet_enough(0);
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
        let object = self
            .scene_camera(camera)
            .or_else(|| {
                if camera.is_none() {
                    self.align_or_create_camera_to_view()
                } else {
                    None
                }
            })
            .ok_or(match camera {
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
            v.path_trace_noise_threshold = quiet_enough(samples);
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
        // The summed picture through Open Image Denoise, where it is
        // installed, as a render from the Render menu goes: the view's own
        // filter leaves most of the grain in a picture of a few hundred
        // samples, and this is a picture someone keeps.
        let own_filter = self
            .gfx
            .as_ref()
            .is_some_and(|g| g.viewport.path_trace_denoise);
        if summed > 0 && own_filter && kreate_render::oidn::available() {
            let cleaned = self.gfx.as_mut().is_some_and(|gfx| {
                let clean = gfx
                    .viewport
                    .read_traced(&gfx.gpu.device, &gfx.gpu.queue)
                    .and_then(|gathered| {
                        let clean = kreate_render::oidn::denoise(&kreate_render::oidn::Noisy {
                            width: gathered.width as usize,
                            height: gathered.height as usize,
                            color: &gathered.color,
                            albedo: &gathered.albedo,
                            normal: &gathered.normal,
                        })?;
                        Ok((gathered, clean))
                    });
                match clean {
                    Ok((gathered, clean)) => {
                        gfx.viewport.path_trace_denoise = false;
                        gfx.viewport
                            .show_traced_mean(&gfx.gpu.queue, &gathered, &clean);
                        true
                    }
                    Err(e) => {
                        tracing::warn!("Open Image Denoise: {e}; the view's own filter is kept");
                        false
                    }
                }
            });
            if cleaned {
                // A frame for the cleaned picture to be shown in.
                self.wanted_viewport = size;
                self.render();
                self.redraw.clear();
            }
        }
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
            v.path_trace_denoise = own_filter;
            v.path_trace_noise_threshold = quiet_enough(0);
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

    /// The viewport's colour target as raw opaque RGBA8 rows.
    pub(super) fn read_target_rgba(&self) -> Result<(u32, u32, Vec<u8>), String> {
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
        // Keep coverage alpha when Film Transparent is enabled; otherwise mark
        // opaque because the target's alpha is a compositing detail.
        if !self.doc.render.transparent_background {
            for p in pixels.as_chunks_mut::<4>().0 {
                p[3] = 255;
            }
        }
        Ok((w, h, pixels))
    }

    /// The viewport's colour target as it is, as a PNG.
    pub(super) fn read_target_png(&self) -> Result<(u32, u32, Vec<u8>), String> {
        let (w, h, pixels) = self.read_target_rgba()?;
        let image =
            image::RgbaImage::from_raw(w, h, pixels).ok_or("the capture has the wrong size")?;
        let mut png = Vec::new();
        image
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .map_err(|e| format!("cannot encode the capture: {e}"))?;
        Ok((w, h, png))
    }
}

/// How quiet a pixel has to be for the path tracer to stop adding to it —
/// its mean's standard error as a share of the mean — by how many samples a
/// render asked for: a hundredth for the view and a quick render, less for
/// one asked for more, or the samples it asked for were never taken where
/// the picture was nearly, and not quite, clean.
pub(crate) fn quiet_enough(samples: u32) -> f32 {
    if samples >= 1024 {
        0.002
    } else if samples >= 256 {
        0.004
    } else {
        0.01
    }
}
