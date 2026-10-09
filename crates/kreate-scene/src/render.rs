//! How a finished picture is made: its size, how long the path tracer runs,
//! and where it is written.
//!
//! The scene's, as Blender keeps its render settings in the scene: saved with
//! the file, so the shot a document was set up for renders the same on the
//! next machine.

/// The file a render is written as.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OutputFormat {
    /// Eight bits a channel, the display's picture: what is looked at.
    #[default]
    Png,
    /// Half floats, scene-linear before the view transform, with the render
    /// passes beside the picture: what is composited.
    Exr,
}

impl OutputFormat {
    pub const ALL: [OutputFormat; 2] = [OutputFormat::Png, OutputFormat::Exr];

    /// The file extension, without the dot.
    pub fn extension(self) -> &'static str {
        match self {
            OutputFormat::Png => "png",
            OutputFormat::Exr => "exr",
        }
    }
}

/// The scene's render settings.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderSettings {
    /// The picture's size in pixels.
    pub width: u32,
    pub height: u32,
    /// How many path-traced samples each pixel is the mean of.
    pub samples: u32,
    pub format: OutputFormat,
    /// Where a render is written: a file for a still, a folder — or a file
    /// name that gets the frame number — for an animation. Empty is the
    /// user's pictures folder.
    pub output: String,
    /// Whether the path tracer's picture is filtered for noise.
    pub denoise: bool,
    /// Motion blur: whether moving objects smear across the shutter.
    pub motion_blur: bool,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            samples: 128,
            format: OutputFormat::Png,
            output: String::new(),
            denoise: true,
            motion_blur: false,
        }
    }
}

impl RenderSettings {
    /// The file one frame of an animation goes to: `output` with the frame
    /// number, four digits, before the extension — `shot_0001.png` — or
    /// `<output>/0001.png` when `output` names a folder.
    pub fn frame_path(&self, folder: &std::path::Path, frame: i32) -> std::path::PathBuf {
        let ext = self.format.extension();
        let number = format!("{frame:04}");
        let output = std::path::Path::new(&self.output);
        if self.output.is_empty() {
            return folder.join(format!("{number}.{ext}"));
        }
        let looks_like_folder =
            self.output.ends_with(['/', '\\']) || output.is_dir() || output.extension().is_none();
        if looks_like_folder {
            output.join(format!("{number}.{ext}"))
        } else {
            let stem = output
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            output.with_file_name(format!("{stem}_{number}.{ext}"))
        }
    }

    /// The file a still goes to: `output` if it names a file, with the
    /// format's extension, or `render.<ext>` in it if it names a folder.
    pub fn still_path(&self, folder: &std::path::Path) -> std::path::PathBuf {
        let ext = self.format.extension();
        if self.output.is_empty() {
            return folder.join(format!("render.{ext}"));
        }
        let output = std::path::Path::new(&self.output);
        if self.output.ends_with(['/', '\\']) || output.is_dir() {
            output.join(format!("render.{ext}"))
        } else {
            output.with_extension(ext)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn frames_are_numbered_beside_a_file_name_or_inside_a_folder() {
        let mut s = RenderSettings {
            output: "C:/out/shot.png".into(),
            ..Default::default()
        };
        assert_eq!(
            s.frame_path(Path::new("x"), 7),
            Path::new("C:/out/shot_0007.png")
        );
        s.output = "C:/out/frames/".into();
        s.format = OutputFormat::Exr;
        assert_eq!(
            s.frame_path(Path::new("x"), 12),
            Path::new("C:/out/frames/").join("0012.exr")
        );
        s.output.clear();
        assert_eq!(
            s.frame_path(Path::new("pics"), 3),
            Path::new("pics").join("0003.exr")
        );
    }

    #[test]
    fn a_still_takes_the_formats_extension() {
        let s = RenderSettings {
            output: "C:/out/shot.png".into(),
            format: OutputFormat::Exr,
            ..Default::default()
        };
        assert_eq!(s.still_path(Path::new("x")), Path::new("C:/out/shot.exr"));
        let s = RenderSettings::default();
        assert_eq!(
            s.still_path(Path::new("pics")),
            Path::new("pics").join("render.png")
        );
    }
}
