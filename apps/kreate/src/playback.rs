//! Playing a rendered image sequence in the render window: the frames read
//! from their files on a thread of their own, at a size for the screen, and
//! shown at the scene's frame rate.
//!
//! A video plays in the system's player, which plays it better than this
//! could; a sequence of pictures has no player of its own, so it gets this.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// The widest a played frame is kept: enough for the window, and a sequence
/// of hundreds held in memory at that size is hundreds of megabytes at most.
const PLAY_WIDTH: u32 = 960;

/// A sequence being played.
pub struct Player {
    /// What the thread has read so far, in order; `None` for a frame it could
    /// not read.
    loaded: Arc<Mutex<Vec<Option<egui::ColorImage>>>>,
    textures: Vec<Option<egui::TextureHandle>>,
    count: usize,
    fps: f64,
    started: Instant,
    /// The frame shown when paused, or where playing began.
    at: usize,
    pub playing: bool,
}

impl Player {
    /// Starts reading `frames` and playing them at `fps`.
    pub fn new(frames: Vec<PathBuf>, fps: f64, wake: impl Fn() + Send + 'static) -> Self {
        let loaded = Arc::new(Mutex::new(Vec::with_capacity(frames.len())));
        let into = Arc::clone(&loaded);
        let count = frames.len();
        std::thread::spawn(move || {
            for path in frames {
                let image = image::open(&path).ok().map(|i| {
                    let i = if i.width() > PLAY_WIDTH {
                        i.resize(
                            PLAY_WIDTH,
                            (i.height() as u64 * PLAY_WIDTH as u64 / i.width() as u64) as u32,
                            image::imageops::FilterType::Triangle,
                        )
                    } else {
                        i
                    };
                    let rgba = i.to_rgba8();
                    egui::ColorImage::from_rgba_unmultiplied(
                        [rgba.width() as usize, rgba.height() as usize],
                        rgba.as_raw(),
                    )
                });
                match into.lock() {
                    Ok(mut v) => v.push(image),
                    Err(_) => return,
                }
                wake();
            }
        });
        Self {
            loaded,
            textures: vec![None; count],
            count,
            fps: if fps.is_finite() && fps > 0.0 {
                fps
            } else {
                24.0
            },
            started: Instant::now(),
            at: 0,
            playing: true,
        }
    }

    /// How many frames there are, and how many are read.
    pub fn progress(&self) -> (usize, usize) {
        let read = self.loaded.lock().map_or(0, |v| v.len());
        (read, self.count)
    }

    /// Which frame is shown now.
    pub fn frame(&self) -> usize {
        if !self.playing || self.count == 0 {
            return self.at;
        }
        let read = self.progress().0.max(1);
        let elapsed = self.started.elapsed().as_secs_f64() * self.fps;
        // Loops over what has been read; a sequence still loading plays the
        // part it has.
        (self.at + elapsed as usize) % read.min(self.count)
    }

    /// Plays from where it is shown, or holds there.
    pub fn toggle(&mut self) {
        self.at = self.frame();
        self.started = Instant::now();
        self.playing = !self.playing;
    }

    /// Moves to frame `i` and holds there.
    pub fn seek(&mut self, i: usize) {
        self.at = i.min(self.count.saturating_sub(1));
        self.playing = false;
    }

    /// The texture for the frame shown now, made from what was read.
    pub fn texture(&mut self, ctx: &egui::Context) -> Option<egui::TextureHandle> {
        let i = self.frame();
        if self.textures.get(i)?.is_none() {
            let image = self.loaded.lock().ok()?.get(i)?.clone()?;
            self.textures[i] = Some(ctx.load_texture(
                format!("render-frame-{i}"),
                image,
                egui::TextureOptions::LINEAR,
            ));
        }
        self.textures[i].clone()
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}
