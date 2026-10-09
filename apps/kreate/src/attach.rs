//! What the chat panel hands an assistant besides words: pictures, PDFs and
//! text files.
//!
//! Each arrives as bytes — from a file the user picked, a file copied in the
//! system's file manager, or a picture on the clipboard — and is made into
//! something every turn can carry: a picture small enough for the model to
//! take whole, a PDF as it is, a text file as its text.
//!
//! # Pictures are made to fit
//!
//! A model reads a picture at about 1568 pixels on its long side and refuses
//! one over five megabytes. A photo straight off a phone is both, so a
//! picture over either is scaled down here, on the way in — what the panel
//! shows as attached is then what is sent, and nothing fails at the far end
//! for a reason the user cannot see.

use std::path::Path;

/// The long side a picture is scaled down to, when it is larger.
pub const MAX_SIDE: u32 = 1568;
/// The most a picture may weigh when it is sent.
pub const MAX_IMAGE_BYTES: usize = 5 * 1024 * 1024;
/// The most text a file may hold to be read into the message.
pub const MAX_TEXT_BYTES: usize = 256 * 1024;
/// The most a PDF may weigh.
pub const MAX_PDF_BYTES: usize = 32 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `data` is the encoded picture. The size is unknown (zero) for a format
    /// this build cannot decode, which is sent as it is.
    Image {
        media_type: &'static str,
        width: u32,
        height: u32,
    },
    /// `data` is the file.
    Pdf,
    /// `data` is UTF-8.
    Text,
}

/// One thing attached to the next message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Attachment {
    pub name: String,
    pub kind: Kind,
    pub data: Vec<u8>,
}

impl Attachment {
    /// Reads a file the user picked or copied.
    pub fn from_path(path: &Path) -> Result<Self, String> {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| "file".into());
        let size = std::fs::metadata(path)
            .map_err(|e| format!("{name}: {e}"))?
            .len() as usize;
        if size > MAX_PDF_BYTES.max(64 * 1024 * 1024) {
            return Err(format!("{name} is too large to attach"));
        }
        let bytes = std::fs::read(path).map_err(|e| format!("{name}: {e}"))?;
        Self::from_bytes(name, bytes)
    }

    /// Makes an attachment of a file's bytes, by what they are.
    pub fn from_bytes(name: String, bytes: Vec<u8>) -> Result<Self, String> {
        if let Some(media_type) = image_type(&bytes) {
            return Self::from_encoded_image(name, bytes, media_type);
        }
        if bytes.starts_with(b"%PDF") {
            if bytes.len() > MAX_PDF_BYTES {
                return Err(format!("{name} is larger than 32 MB"));
            }
            return Ok(Self {
                name,
                kind: Kind::Pdf,
                data: bytes,
            });
        }
        match std::str::from_utf8(&bytes) {
            Ok(text) if !text.contains('\0') => {
                if bytes.len() > MAX_TEXT_BYTES {
                    return Err(format!("{name} is more text than can be sent (256 KB)"));
                }
                Ok(Self {
                    name,
                    kind: Kind::Text,
                    data: bytes,
                })
            }
            _ => Err(format!(
                "{name}: only pictures (PNG, JPEG, GIF, WebP), PDFs and text files can be attached"
            )),
        }
    }

    /// A picture from the clipboard: four bytes a pixel, rows top to bottom.
    pub fn from_rgba(name: String, width: u32, height: u32, rgba: Vec<u8>) -> Result<Self, String> {
        let image = image::RgbaImage::from_raw(width, height, rgba)
            .ok_or("the clipboard's picture is not the size it says")?;
        Self::fitted(name, image::DynamicImage::ImageRgba8(image))
    }

    fn from_encoded_image(
        name: String,
        bytes: Vec<u8>,
        media_type: &'static str,
    ) -> Result<Self, String> {
        let decodable = matches!(media_type, "image/png" | "image/jpeg");
        if !decodable {
            // GIF and WebP: sent as they are, which the model takes, when
            // they are small enough; this build cannot scale them.
            if bytes.len() > MAX_IMAGE_BYTES {
                return Err(format!("{name} is larger than 5 MB"));
            }
            return Ok(Self {
                name,
                kind: Kind::Image {
                    media_type,
                    width: 0,
                    height: 0,
                },
                data: bytes,
            });
        }
        let image = image::load_from_memory(&bytes).map_err(|e| format!("{name}: {e}"))?;
        let (width, height) = (image.width(), image.height());
        if width.max(height) <= MAX_SIDE && bytes.len() <= MAX_IMAGE_BYTES {
            return Ok(Self {
                name,
                kind: Kind::Image {
                    media_type,
                    width,
                    height,
                },
                data: bytes,
            });
        }
        Self::fitted(name, image)
    }

    /// Scales a picture to fit, and encodes it: PNG, or JPEG when the PNG
    /// would be too heavy — a photo, as a rule.
    fn fitted(name: String, image: image::DynamicImage) -> Result<Self, String> {
        let long = image.width().max(image.height());
        let image = if long > MAX_SIDE {
            let scale = MAX_SIDE as f64 / long as f64;
            let w = ((image.width() as f64 * scale).round() as u32).max(1);
            let h = ((image.height() as f64 * scale).round() as u32).max(1);
            image.resize_exact(w, h, image::imageops::FilterType::Triangle)
        } else {
            image
        };
        let (width, height) = (image.width(), image.height());
        let mut png = Vec::new();
        image
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .map_err(|e| format!("{name}: {e}"))?;
        if png.len() <= MAX_IMAGE_BYTES {
            return Ok(Self {
                name: with_extension(&name, "png"),
                kind: Kind::Image {
                    media_type: "image/png",
                    width,
                    height,
                },
                data: png,
            });
        }
        let mut jpeg = Vec::new();
        image::DynamicImage::ImageRgb8(image.to_rgb8())
            .write_to(
                &mut std::io::Cursor::new(&mut jpeg),
                image::ImageFormat::Jpeg,
            )
            .map_err(|e| format!("{name}: {e}"))?;
        Ok(Self {
            name: with_extension(&name, "jpg"),
            kind: Kind::Image {
                media_type: "image/jpeg",
                width,
                height,
            },
            data: jpeg,
        })
    }

    /// What the panel says about it, after its name: "1200 × 800", "PDF,
    /// 1.2 MB", "text, 3 KB".
    pub fn describe(&self) -> String {
        // i18n-exempt: sizes and format names
        match &self.kind {
            Kind::Image { width, height, .. } if *width > 0 => format!("{width} × {height}"),
            Kind::Image { media_type, .. } => {
                media_type.trim_start_matches("image/").to_uppercase()
            }
            Kind::Pdf => format!("PDF, {}", size(self.data.len())),
            Kind::Text => format!("text, {}", size(self.data.len())),
        }
    }

    pub fn is_image(&self) -> bool {
        matches!(self.kind, Kind::Image { .. })
    }
}

/// The picture format the bytes are, by their first bytes.
fn image_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.len() >= 12 && &bytes[0..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
        Some("image/webp")
    } else {
        None
    }
}

fn with_extension(name: &str, ext: &str) -> String {
    let stem = Path::new(name)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| name.to_string());
    format!("{stem}.{ext}")
}

fn size(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{} KB", bytes.div_ceil(1024))
    }
}

/// The message's words with the text files read into it, each fenced and
/// named, after what the user typed.
pub fn prompt_with_text(text: &str, attachments: &[Attachment]) -> String {
    let mut out = text.to_string();
    for a in attachments.iter().filter(|a| a.kind == Kind::Text) {
        let body = String::from_utf8_lossy(&a.data);
        out.push_str(&format!(
            "\n\n<file name=\"{}\">\n{}\n</file>",
            a.name, body
        ));
    }
    out
}

/// One user turn as Claude Code's `--input-format stream-json` reads it: the
/// pictures and PDFs first, as the model reads best, then the words.
pub fn claude_user_message(text: &str, attachments: &[Attachment]) -> serde_json::Value {
    let mut content = Vec::new();
    for a in attachments {
        let data = base64::encode(&a.data);
        match &a.kind {
            Kind::Image { media_type, .. } => content.push(serde_json::json!({
                "type": "image",
                "source": { "type": "base64", "media_type": media_type, "data": data },
            })),
            Kind::Pdf => content.push(serde_json::json!({
                "type": "document",
                "source": { "type": "base64", "media_type": "application/pdf", "data": data },
                "title": a.name,
            })),
            Kind::Text => {}
        }
    }
    content.push(serde_json::json!({
        "type": "text",
        "text": prompt_with_text(text, attachments),
    }));
    serde_json::json!({
        "type": "user",
        "message": { "role": "user", "content": content },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut out = Vec::new();
        image::DynamicImage::new_rgba8(width, height)
            .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
            .unwrap();
        out
    }

    #[test]
    fn a_small_picture_is_sent_as_it_is() {
        let bytes = png(40, 30);
        let a = Attachment::from_bytes("shot.png".into(), bytes.clone()).unwrap();
        assert_eq!(a.data, bytes);
        assert_eq!(
            a.kind,
            Kind::Image {
                media_type: "image/png",
                width: 40,
                height: 30
            }
        );
        assert_eq!(a.describe(), "40 × 30");
    }

    #[test]
    fn a_large_picture_is_scaled_to_fit() {
        let a = Attachment::from_bytes("big.png".into(), png(4000, 1000)).unwrap();
        let Kind::Image { width, height, .. } = a.kind else {
            panic!("{:?}", a.kind)
        };
        assert_eq!((width, height), (MAX_SIDE, 392));
        let back = image::load_from_memory(&a.data).unwrap();
        assert_eq!((back.width(), back.height()), (width, height));
    }

    #[test]
    fn a_clipboard_picture_becomes_a_png() {
        let a = Attachment::from_rgba("pasted".into(), 2, 2, vec![255; 16]).unwrap();
        assert_eq!(a.name, "pasted.png");
        assert!(a.data.starts_with(b"\x89PNG"));
        assert!(Attachment::from_rgba("bad".into(), 3, 3, vec![0; 4]).is_err());
    }

    #[test]
    fn pdfs_and_text_are_told_apart_from_other_files() {
        let pdf = Attachment::from_bytes("a.pdf".into(), b"%PDF-1.7 ...".to_vec()).unwrap();
        assert_eq!(pdf.kind, Kind::Pdf);
        let text =
            Attachment::from_bytes("notes.md".into(), "# 메모\nhi".as_bytes().to_vec()).unwrap();
        assert_eq!(text.kind, Kind::Text);
        assert!(Attachment::from_bytes("a.bin".into(), vec![0, 159, 146, 150]).is_err());
    }

    #[test]
    fn text_files_are_read_into_the_words_and_pictures_go_first() {
        let pic = Attachment::from_bytes("p.png".into(), png(4, 4)).unwrap();
        let note = Attachment::from_bytes("n.txt".into(), b"hello".to_vec()).unwrap();
        let message = claude_user_message("make this", &[note, pic]);
        let content = message["message"]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2, "the text file is not a block of its own");
        assert_eq!(content[0]["type"], "image");
        assert_eq!(content[0]["source"]["media_type"], "image/png");
        assert_eq!(content[1]["type"], "text");
        let words = content[1]["text"].as_str().unwrap();
        assert!(words.starts_with("make this"));
        assert!(words.contains("<file name=\"n.txt\">\nhello\n</file>"));
        assert_eq!(message["type"], "user");
    }
}
