//! The scene's render settings, as an operator: the Render tab edits them,
//! a script sets them, and a change is one undo step.

use kreate_scene::render::OutputFormat;
use kreate_scene::{DocMut, Document};

use crate::operator::{Context, OpError, OpResult, Operator};
use crate::OpParams;

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<SetRenderSettings>();
    registry.register::<SetFrameRange>();
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SetFrameRangeParams {
    /// The first frame played and rendered.
    pub start: i64,
    /// The last frame played and rendered.
    pub end: i64,
    /// Frames a second.
    #[param(min = 1.0, max = 1000.0)]
    pub fps: f64,
}

impl Default for SetFrameRangeParams {
    fn default() -> Self {
        let p = kreate_scene::animation::Playback::default();
        Self {
            start: p.start as i64,
            end: p.end as i64,
            fps: p.fps,
        }
    }
}

/// Sets the scene's frame range and rate: what plays, and what an animation
/// render writes.
pub struct SetFrameRange;

impl Operator for SetFrameRange {
    type Params = SetFrameRangeParams;

    fn id() -> &'static str {
        "scene.set_frame_range"
    }
    fn label() -> &'static str {
        "Frame Range"
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let start = params.start.clamp(i32::MIN as i64 / 2, i32::MAX as i64 / 2) as i32;
        let end = (params.end.clamp(i32::MIN as i64 / 2, i32::MAX as i64 / 2) as i32).max(start);
        let fps = if params.fps.is_finite() {
            params.fps.clamp(1.0, 1000.0)
        } else {
            24.0
        };
        let p = doc.document().playback;
        if (p.start, p.end, p.fps) == (start, end, fps) {
            return Ok(OpResult::Cancelled);
        }
        let playback = &mut doc.document_mut().playback;
        playback.start = start;
        playback.end = end;
        playback.fps = fps;
        playback.frame = playback.frame.clamp(start as f64, end as f64);
        Ok(OpResult::Finished)
    }
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct SetRenderSettingsParams {
    /// The picture's width in pixels.
    #[param(min = 16.0, max = 16384.0)]
    pub width: i64,
    /// The picture's height in pixels.
    #[param(min = 16.0, max = 16384.0)]
    pub height: i64,
    /// How many path-traced samples each pixel is the mean of.
    #[param(min = 1.0, max = 65536.0, soft_max = 4096.0)]
    pub samples: i64,
    /// PNG for a picture to look at, EXR for scene-linear values and render
    /// passes to composite.
    #[param(variants("png", "exr"))]
    pub format: u32,
    /// Where renders go: a file, or a folder for an animation's frames.
    /// Empty is the pictures folder.
    pub output: String,
    /// Filter the path tracer's noise.
    pub denoise: bool,
    /// Smear what moves across the camera's shutter.
    pub motion_blur: bool,
}

impl Default for SetRenderSettingsParams {
    fn default() -> Self {
        render_params(&Document::new())
    }
}

/// The parameters that leave the settings as they are.
pub fn render_params(doc: &Document) -> SetRenderSettingsParams {
    let r = &doc.render;
    SetRenderSettingsParams {
        width: r.width as i64,
        height: r.height as i64,
        samples: r.samples as i64,
        format: OutputFormat::ALL
            .iter()
            .position(|f| *f == r.format)
            .unwrap_or(0) as u32,
        output: r.output.clone(),
        denoise: r.denoise,
        motion_blur: r.motion_blur,
    }
}

/// Sets the scene's render settings.
pub struct SetRenderSettings;

impl Operator for SetRenderSettings {
    type Params = SetRenderSettingsParams;

    fn id() -> &'static str {
        "render.set_settings"
    }
    fn label() -> &'static str {
        "Render Settings"
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let next = kreate_scene::render::RenderSettings {
            width: params.width.clamp(16, 16384) as u32,
            height: params.height.clamp(16, 16384) as u32,
            samples: params.samples.clamp(1, 65536) as u32,
            format: OutputFormat::ALL
                .get(params.format as usize)
                .copied()
                .unwrap_or_default(),
            output: params.output.trim().to_string(),
            denoise: params.denoise,
            motion_blur: params.motion_blur,
        };
        if doc.document().render == next {
            return Ok(OpResult::Cancelled);
        }
        doc.document_mut().render = next;
        Ok(OpResult::Finished)
    }
}

#[cfg(test)]
mod range_tests {
    use super::*;
    use crate::operator::OperatorRegistry;
    use kreate_reflect::{PropertyMap, PropertyValue};

    #[test]
    fn a_range_ends_no_earlier_than_it_starts_and_keeps_the_frame_inside() {
        let mut registry = OperatorRegistry::new();
        register(&mut registry);
        let mut history = crate::history::History::new();
        let mut doc = Document::new();
        doc.playback.frame = 100.0;
        let mut params = PropertyMap::new();
        params.set("start", PropertyValue::Int(10));
        params.set("end", PropertyValue::Int(5));
        params.set("fps", PropertyValue::Float(30.0));
        history
            .dispatch(
                &registry,
                &mut doc,
                &Context::default(),
                "scene.set_frame_range",
                &params,
            )
            .expect("runs");
        assert_eq!(
            (doc.playback.start, doc.playback.end, doc.playback.fps),
            (10, 10, 30.0)
        );
        assert_eq!(doc.playback.frame, 10.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::OperatorRegistry;
    use kreate_reflect::{PropertyMap, PropertyValue};

    #[test]
    fn settings_are_set_and_undone_as_one_step() {
        let mut registry = OperatorRegistry::new();
        register(&mut registry);
        let mut history = crate::history::History::new();
        let mut doc = Document::new();
        let mut params = PropertyMap::new();
        params.set("width", PropertyValue::Int(800));
        params.set("format", PropertyValue::Enum(1));
        params.set("output", PropertyValue::Text("  C:/renders/shot  ".into()));
        history
            .dispatch(
                &registry,
                &mut doc,
                &Context::default(),
                "render.set_settings",
                &params,
            )
            .expect("runs");
        assert_eq!(doc.render.width, 800);
        assert_eq!(
            doc.render.height, 1080,
            "a parameter left out keeps its value"
        );
        assert_eq!(doc.render.format, OutputFormat::Exr);
        assert_eq!(doc.render.output, "C:/renders/shot");
        assert!(history.undo(&mut doc));
        assert_eq!(doc.render, kreate_scene::render::RenderSettings::default());
    }
}
