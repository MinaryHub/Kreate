//! An animation written as one MP4 file: each rendered frame encoded as
//! H.264 by Cisco's OpenH264, built from its source, and the stream put in an
//! MP4 container — what every player and every site takes.
//!
//! The frames arrive as the display's RGBA, one at a time, as the render job
//! finishes each; the file is complete when [`Video::finish`] writes its
//! index. A render stopped part way still finishes what it has.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use openh264::encoder::{
    BitRate, Encoder, EncoderConfig, FrameRate, FrameType, IntraFramePeriod, RateControlMode,
};
use openh264::formats::{RgbaSliceU8, YUVBuffer};
use openh264::OpenH264API;

/// The units the file counts time in: a common multiple of the usual rates,
/// so 24, 25, 30 and 60 frames a second are whole ticks.
const TIMESCALE: u32 = 6000;

/// An MP4 being written.
pub struct Video {
    path: PathBuf,
    encoder: Encoder,
    /// The container, opened once the first frame gives the stream's
    /// parameter sets.
    writer: Option<mp4::Mp4Writer<BufWriter<File>>>,
    /// The size frames are encoded at: the picture's, made even, as 4:2:0
    /// chroma needs.
    width: u32,
    height: u32,
    /// Ticks a frame lasts.
    duration: u32,
    frames: u64,
}

impl Video {
    /// Starts a video of `width` by `height` frames at `fps`, to be written
    /// at `path`.
    pub fn new(path: &Path, width: u32, height: u32, fps: f64) -> Result<Self, String> {
        let (width, height) = (width & !1, height & !1);
        if width < 16 || height < 16 {
            return Err(format!("{width} by {height} is too small for a video"));
        }
        let fps = if fps.is_finite() && fps > 0.0 {
            fps
        } else {
            24.0
        };
        // Enough bits that a render's detail survives: about a sixth of a
        // bit per pixel per frame, which is a high-quality H.264 rate.
        let bits = (width as f64 * height as f64 * fps * 0.18).clamp(2.0e6, 80.0e6) as u32;
        let config = EncoderConfig::new()
            .rate_control_mode(RateControlMode::Quality)
            .bitrate(BitRate::from_bps(bits))
            .max_frame_rate(FrameRate::from_hz(fps as f32))
            // A key frame each two seconds, so a player can seek.
            .intra_frame_period(IntraFramePeriod::from_num_frames((fps * 2.0).round() as u32))
            .skip_frames(false);
        let encoder = Encoder::with_api_config(OpenH264API::from_source(), config)
            .map_err(|e| format!("the H.264 encoder would not start: {e}"))?;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
        }
        Ok(Self {
            path: path.to_path_buf(),
            encoder,
            writer: None,
            width,
            height,
            duration: (TIMESCALE as f64 / fps).round().max(1.0) as u32,
            frames: 0,
        })
    }

    /// Where the video is written.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// How many frames are in it so far.
    pub fn frames(&self) -> u64 {
        self.frames
    }

    /// Adds one frame, `rgba` of `width` by `height` pixels: cropped to the
    /// video's even size if it is a pixel larger.
    pub fn push(&mut self, rgba: &[u8], width: u32, height: u32) -> Result<(), String> {
        if width < self.width || height < self.height || rgba.len() < (width * height * 4) as usize
        {
            return Err(format!(
                "a frame of {width} by {height} does not fill a {} by {} video",
                self.width, self.height
            ));
        }
        let cropped;
        let pixels = if width == self.width && height == self.height {
            rgba
        } else {
            let row = (width * 4) as usize;
            let keep = (self.width * 4) as usize;
            cropped = (0..self.height as usize)
                .flat_map(|y| rgba[y * row..y * row + keep].iter().copied())
                .collect::<Vec<u8>>();
            &cropped
        };
        let yuv = YUVBuffer::from_rgb_source(RgbaSliceU8::new(
            pixels,
            (self.width as usize, self.height as usize),
        ));
        let stream = self
            .encoder
            .encode(&yuv)
            .map_err(|e| format!("a frame would not encode: {e}"))?;
        let key = matches!(stream.frame_type(), FrameType::IDR | FrameType::I);
        // The NAL units, split from their start codes: the parameter sets go
        // in the container's header, the rest in the frame's sample with a
        // length before each, as MP4 keeps them.
        let mut sps = None;
        let mut pps = None;
        let mut sample = Vec::new();
        for l in 0..stream.num_layers() {
            let Some(layer) = stream.layer(l) else {
                continue;
            };
            for n in 0..layer.nal_count() {
                let Some(unit) = layer.nal_unit(n) else {
                    continue;
                };
                let body = strip_start_code(unit);
                let Some(&head) = body.first() else {
                    continue;
                };
                match head & 0x1f {
                    7 => sps = Some(body.to_vec()),
                    8 => pps = Some(body.to_vec()),
                    _ => {
                        sample.extend_from_slice(&(body.len() as u32).to_be_bytes());
                        sample.extend_from_slice(body);
                    }
                }
            }
        }
        if self.writer.is_none() {
            let (Some(sps), Some(pps)) = (sps, pps) else {
                return Err("the encoder's first frame had no parameter sets".into());
            };
            self.writer = Some(self.open(sps, pps)?);
        }
        if sample.is_empty() {
            // A skipped frame: nothing to write, and the time it would have
            // held goes to the frame before.
            return Ok(());
        }
        let writer = self.writer.as_mut().expect("opened above");
        writer
            .write_sample(
                1,
                &mp4::Mp4Sample {
                    start_time: self.frames * self.duration as u64,
                    duration: self.duration,
                    rendering_offset: 0,
                    is_sync: key,
                    bytes: bytes::Bytes::from(sample),
                },
            )
            .map_err(|e| format!("{}: {e}", self.path.display()))?;
        self.frames += 1;
        Ok(())
    }

    fn open(&self, sps: Vec<u8>, pps: Vec<u8>) -> Result<mp4::Mp4Writer<BufWriter<File>>, String> {
        let file = File::create(&self.path).map_err(|e| format!("{}: {e}", self.path.display()))?;
        let config = mp4::Mp4Config {
            major_brand: str::parse("isom").expect("a four-character code"),
            minor_version: 512,
            compatible_brands: ["isom", "iso2", "avc1", "mp41"]
                .iter()
                .map(|b| str::parse(b).expect("a four-character code"))
                .collect(),
            timescale: TIMESCALE,
        };
        let mut writer = mp4::Mp4Writer::write_start(BufWriter::new(file), &config)
            .map_err(|e| format!("{}: {e}", self.path.display()))?;
        writer
            .add_track(&mp4::TrackConfig {
                track_type: mp4::TrackType::Video,
                timescale: TIMESCALE,
                language: "und".to_string(),
                media_conf: mp4::MediaConfig::AvcConfig(mp4::AvcConfig {
                    width: self.width as u16,
                    height: self.height as u16,
                    seq_param_set: sps,
                    pic_param_set: pps,
                }),
            })
            .map_err(|e| format!("{}: {e}", self.path.display()))?;
        Ok(writer)
    }

    /// Writes the file's index, which is what makes it playable. Returns how
    /// many frames it holds.
    pub fn finish(mut self) -> Result<u64, String> {
        let Some(mut writer) = self.writer.take() else {
            return Ok(0);
        };
        writer
            .write_end()
            .map_err(|e| format!("{}: {e}", self.path.display()))?;
        Ok(self.frames)
    }
}

/// A NAL unit without the Annex B start code the encoder puts before it.
fn strip_start_code(unit: &[u8]) -> &[u8] {
    if unit.starts_with(&[0, 0, 0, 1]) {
        &unit[4..]
    } else if unit.starts_with(&[0, 0, 1]) {
        &unit[3..]
    } else {
        unit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_few_frames_make_a_playable_mp4() {
        let dir = std::env::temp_dir().join(format!("kreate-video-{}", std::process::id()));
        let path = dir.join("test.mp4");
        let (w, h) = (64u32, 48u32);
        let mut video = Video::new(&path, w, h, 24.0).expect("starts");
        for f in 0..12u32 {
            let rgba: Vec<u8> = (0..w * h)
                .flat_map(|i| {
                    let x = i % w;
                    [((x * 4 + f * 8) % 256) as u8, 80, 160, 255]
                })
                .collect();
            video.push(&rgba, w, h).expect("encodes");
        }
        assert_eq!(video.finish().expect("finishes"), 12);
        // Read back: one video track of twelve samples, half a second long.
        let file = File::open(&path).unwrap();
        let size = file.metadata().unwrap().len();
        let reader = mp4::Mp4Reader::read_header(std::io::BufReader::new(file), size).unwrap();
        let track = reader.tracks().values().next().expect("a track");
        assert_eq!(track.sample_count(), 12);
        assert_eq!(track.width(), w as u16);
        assert!((reader.duration().as_secs_f64() - 0.5).abs() < 0.05);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_odd_size_is_cropped_to_even() {
        let dir = std::env::temp_dir().join(format!("kreate-video-odd-{}", std::process::id()));
        let path = dir.join("odd.mp4");
        let mut video = Video::new(&path, 65, 49, 30.0).expect("starts");
        let rgba = vec![128u8; 65 * 49 * 4];
        video.push(&rgba, 65, 49).expect("encodes");
        assert_eq!(video.finish().expect("finishes"), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
