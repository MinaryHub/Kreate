//! OpenEXR, written by hand: what a render's scene-linear picture and its
//! passes are saved as.
//!
//! The format a compositor reads. Written here rather than through a crate,
//! because the part a renderer needs is small — one part, scanlines, no
//! compression — and the crate is not available offline. What is written is
//! a file every EXR reader opens: uncompressed is the format's simplest case
//! and the one all of them support.
//!
//! Channels are half floats unless a channel asks for full ones — an object
//! id hashed for Cryptomatte has to survive to the bit, and a half cannot hold
//! it. A reader of these same files is here too, for the tests and for
//! reading a render back; it refuses anything compressed.

/// One channel of the picture: a name such as `R` or `depth.Z`, and a value
/// per pixel, rows from the top.
pub struct Channel<'a> {
    pub name: &'a str,
    pub values: &'a [f32],
    /// Thirty-two bits rather than sixteen.
    pub full_float: bool,
}

const MAGIC: [u8; 4] = [0x76, 0x2f, 0x31, 0x01];
const HALF: i32 = 1;
const FLOAT: i32 = 2;

fn attribute(out: &mut Vec<u8>, name: &str, kind: &str, value: &[u8]) {
    out.extend_from_slice(name.as_bytes());
    out.push(0);
    out.extend_from_slice(kind.as_bytes());
    out.push(0);
    out.extend_from_slice(&(value.len() as i32).to_le_bytes());
    out.extend_from_slice(value);
}

/// An EXR of `width` by `height` with these channels and, beside the
/// standard header, these string attributes (Cryptomatte's manifest goes
/// there).
///
/// Panics if a channel does not have a value for every pixel: that is a
/// caller's bug, not a file's.
pub fn write(
    width: u32,
    height: u32,
    channels: &[Channel<'_>],
    strings: &[(&str, &str)],
) -> Vec<u8> {
    let pixels = width as usize * height as usize;
    for c in channels {
        assert_eq!(
            c.values.len(),
            pixels,
            "channel {} has the wrong number of values",
            c.name
        );
    }
    // The format wants the channels in name order, in the header and in every
    // scanline alike.
    let mut order: Vec<&Channel<'_>> = channels.iter().collect();
    order.sort_by(|a, b| a.name.cmp(b.name));

    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&2u32.to_le_bytes());

    let mut list = Vec::new();
    for c in &order {
        list.extend_from_slice(c.name.as_bytes());
        list.push(0);
        list.extend_from_slice(&(if c.full_float { FLOAT } else { HALF }).to_le_bytes());
        list.extend_from_slice(&[0, 0, 0, 0]); // pLinear and three reserved bytes
        list.extend_from_slice(&1i32.to_le_bytes());
        list.extend_from_slice(&1i32.to_le_bytes());
    }
    list.push(0);
    attribute(&mut out, "channels", "chlist", &list);
    attribute(&mut out, "compression", "compression", &[0]);
    let mut window = Vec::new();
    for v in [0i32, 0, width as i32 - 1, height as i32 - 1] {
        window.extend_from_slice(&v.to_le_bytes());
    }
    attribute(&mut out, "dataWindow", "box2i", &window);
    attribute(&mut out, "displayWindow", "box2i", &window);
    attribute(&mut out, "lineOrder", "lineOrder", &[0]);
    attribute(&mut out, "pixelAspectRatio", "float", &1.0f32.to_le_bytes());
    let mut centre = Vec::new();
    centre.extend_from_slice(&0.0f32.to_le_bytes());
    centre.extend_from_slice(&0.0f32.to_le_bytes());
    attribute(&mut out, "screenWindowCenter", "v2f", &centre);
    attribute(
        &mut out,
        "screenWindowWidth",
        "float",
        &1.0f32.to_le_bytes(),
    );
    for (name, value) in strings {
        attribute(&mut out, name, "string", value.as_bytes());
    }
    out.push(0);

    // The offset table, then one block per scanline.
    let line_bytes: usize = order
        .iter()
        .map(|c| if c.full_float { 4 } else { 2 })
        .sum::<usize>()
        * width as usize;
    let table_at = out.len();
    let first_block = table_at + 8 * height as usize;
    for y in 0..height as usize {
        let offset = (first_block + y * (8 + line_bytes)) as u64;
        out.extend_from_slice(&offset.to_le_bytes());
    }
    for y in 0..height as usize {
        out.extend_from_slice(&(y as i32).to_le_bytes());
        out.extend_from_slice(&(line_bytes as i32).to_le_bytes());
        for c in &order {
            let row = &c.values[y * width as usize..(y + 1) * width as usize];
            if c.full_float {
                for v in row {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            } else {
                for v in row {
                    out.extend_from_slice(&half::f16::from_f32(*v).to_le_bytes());
                }
            }
        }
    }
    out
}

/// MurmurHash3, 32-bit, seed zero: what Cryptomatte hashes names with.
pub fn murmur3_32(bytes: &[u8]) -> u32 {
    const C1: u32 = 0xcc9e_2d51;
    const C2: u32 = 0x1b87_3593;
    let mut h: u32 = 0;
    let (blocks, tail) = bytes.as_chunks::<4>();
    for block in blocks {
        let mut k = u32::from_le_bytes(*block);
        k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h ^= k;
        h = h.rotate_left(13).wrapping_mul(5).wrapping_add(0xe654_6b64);
    }
    let mut k = 0u32;
    for (i, b) in tail.iter().enumerate() {
        k |= (*b as u32) << (8 * i);
    }
    if !tail.is_empty() {
        k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h ^= k;
    }
    h ^= bytes.len() as u32;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h
}

/// A name's Cryptomatte id: its hash, nudged off the exponents that make a
/// float infinite, NaN or denormal, and read as a float — the
/// `uint32_to_float32` conversion the specification names. The bits are what
/// the manifest lists.
pub fn cryptomatte_id(name: &str) -> f32 {
    let mut h = murmur3_32(name.as_bytes());
    let exponent = (h >> 23) & 0xff;
    if exponent == 0 || exponent == 0xff {
        h ^= 1 << 23;
    }
    f32::from_bits(h)
}

/// The Cryptomatte header attributes for a layer called `layer` holding
/// `names`: its name, hash, conversion and manifest, under the key the
/// specification derives from the layer's name.
pub fn cryptomatte_attributes(layer: &str, names: &[&str]) -> Vec<(String, String)> {
    let key = format!("{:08x}", murmur3_32(layer.as_bytes()));
    let key = &key[..7];
    let mut manifest = String::from("{");
    for (i, name) in names.iter().enumerate() {
        if i > 0 {
            manifest.push(',');
        }
        let escaped: String = name
            .chars()
            .flat_map(|c| match c {
                '"' => vec!['\\', '"'],
                '\\' => vec!['\\', '\\'],
                c => vec![c],
            })
            .collect();
        manifest.push_str(&format!(
            "\"{escaped}\":\"{:08x}\"",
            cryptomatte_id(name).to_bits()
        ));
    }
    manifest.push('}');
    vec![
        (format!("cryptomatte/{key}/name"), layer.to_string()),
        (
            format!("cryptomatte/{key}/hash"),
            "MurmurHash3_32".to_string(),
        ),
        (
            format!("cryptomatte/{key}/conversion"),
            "uint32_to_float32".to_string(),
        ),
        (format!("cryptomatte/{key}/manifest"), manifest),
    ]
}

/// What [`read`] finds.
#[derive(Debug)]
pub struct Read {
    pub width: u32,
    pub height: u32,
    /// Each channel's name and values, in the file's (name) order.
    pub channels: Vec<(String, Vec<f32>)>,
    /// The header's string attributes.
    pub strings: Vec<(String, String)>,
}

/// Reads an uncompressed, single-part, scanline EXR — what [`write`] makes.
pub fn read(bytes: &[u8]) -> Result<Read, String> {
    let mut at = 0usize;
    let take = |at: &mut usize, n: usize| -> Result<&[u8], String> {
        let slice = bytes.get(*at..*at + n).ok_or("the file ends early")?;
        *at += n;
        Ok(slice)
    };
    let text = |at: &mut usize| -> Result<String, String> {
        let end = bytes[*at..]
            .iter()
            .position(|b| *b == 0)
            .ok_or("an unterminated name")?
            + *at;
        let s = String::from_utf8_lossy(&bytes[*at..end]).to_string();
        *at = end + 1;
        Ok(s)
    };
    if take(&mut at, 4)? != MAGIC {
        return Err("not an OpenEXR file".into());
    }
    let version = u32::from_le_bytes(take(&mut at, 4)?.try_into().unwrap());
    if version & 0xff != 2 || version & !0xff != 0 {
        return Err("only single-part scanline files are read".into());
    }
    let mut list: Vec<(String, i32)> = Vec::new();
    let mut window = None;
    let mut strings = Vec::new();
    loop {
        let name = text(&mut at)?;
        if name.is_empty() {
            break;
        }
        let kind = text(&mut at)?;
        let size = i32::from_le_bytes(take(&mut at, 4)?.try_into().unwrap()) as usize;
        let value = take(&mut at, size)?;
        match (name.as_str(), kind.as_str()) {
            ("channels", "chlist") => {
                let mut i = 0;
                while i < value.len() && value[i] != 0 {
                    let end = value[i..]
                        .iter()
                        .position(|b| *b == 0)
                        .ok_or("a bad channel list")?
                        + i;
                    let channel = String::from_utf8_lossy(&value[i..end]).to_string();
                    let kind = i32::from_le_bytes(
                        value[end + 1..end + 5]
                            .try_into()
                            .map_err(|_| "a bad channel list")?,
                    );
                    list.push((channel, kind));
                    i = end + 1 + 16;
                }
            }
            ("compression", _) => {
                if value.first() != Some(&0) {
                    return Err("only uncompressed files are read".into());
                }
            }
            ("dataWindow", "box2i") => {
                let v: Vec<i32> = value
                    .chunks(4)
                    .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                window = Some((v[0], v[1], v[2], v[3]));
            }
            (_, "string") => {
                strings.push((name.clone(), String::from_utf8_lossy(value).to_string()))
            }
            _ => {}
        }
    }
    let (x0, y0, x1, y1) = window.ok_or("no data window")?;
    let (width, height) = ((x1 - x0 + 1) as u32, (y1 - y0 + 1) as u32);
    at += 8 * height as usize;
    let mut channels: Vec<(String, Vec<f32>)> = list
        .iter()
        .map(|(n, _)| (n.clone(), Vec::with_capacity((width * height) as usize)))
        .collect();
    for _ in 0..height {
        take(&mut at, 8)?;
        for (i, (_, kind)) in list.iter().enumerate() {
            for _ in 0..width {
                let v = if *kind == FLOAT {
                    f32::from_le_bytes(take(&mut at, 4)?.try_into().unwrap())
                } else {
                    half::f16::from_le_bytes(take(&mut at, 2)?.try_into().unwrap()).to_f32()
                };
                channels[i].1.push(v);
            }
        }
    }
    Ok(Read {
        width,
        height,
        channels,
        strings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_is_written_reads_back_channel_for_channel() {
        let (w, h) = (3, 2);
        let r = [0.0, 1.0, 2.5, 100.0, 0.18, 65000.0];
        let g = [0.5; 6];
        let id = [1.234_567_9e-12, 3.0, 4.0, 5.0, 6.0, 7.0];
        let bytes = write(
            w,
            h,
            &[
                Channel {
                    name: "R",
                    values: &r,
                    full_float: false,
                },
                Channel {
                    name: "G",
                    values: &g,
                    full_float: false,
                },
                Channel {
                    name: "CryptoObject00.R",
                    values: &id,
                    full_float: true,
                },
            ],
            &[("cryptomatte/abc/name", "CryptoObject")],
        );
        let back = read(&bytes).expect("reads");
        assert_eq!((back.width, back.height), (w, h));
        // Name order, as the format requires.
        let names: Vec<_> = back.channels.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["CryptoObject00.R", "G", "R"]);
        assert_eq!(back.channels[0].1, id, "a full float survives to the bit");
        for (got, want) in back.channels[2].1.iter().zip(r) {
            assert!(
                (got - want).abs() <= want.abs() * 1e-3 + 1e-3,
                "{got} against {want}"
            );
        }
        assert_eq!(
            back.strings,
            [(
                "cryptomatte/abc/name".to_string(),
                "CryptoObject".to_string()
            )]
        );
    }

    #[test]
    fn murmur3_matches_the_reference_vectors() {
        // From the reference implementation's own tests, seed zero.
        assert_eq!(murmur3_32(b""), 0);
        assert_eq!(murmur3_32(b"hello"), 0x248b_fa47);
        assert_eq!(murmur3_32(b"Hello, world!"), 0xc036_3e43);
        assert_eq!(
            murmur3_32(b"The quick brown fox jumps over the lazy dog"),
            0x2e4f_f723
        );
    }

    #[test]
    fn a_cryptomatte_id_is_always_an_ordinary_float_and_listed_by_its_bits() {
        for name in ["Cube", "Lamp", "", "a much longer object name.001"] {
            let id = cryptomatte_id(name);
            assert!(
                id.is_finite() && (id == 0.0 || id.is_normal()),
                "{name}: {id}"
            );
        }
        let attrs = cryptomatte_attributes("CryptoObject", &["Cube"]);
        let manifest = &attrs
            .iter()
            .find(|(k, _)| k.ends_with("/manifest"))
            .unwrap()
            .1;
        assert_eq!(
            manifest,
            &format!("{{\"Cube\":\"{:08x}\"}}", cryptomatte_id("Cube").to_bits())
        );
        // The key is the layer name's hash, seven hex digits.
        assert!(
            attrs[0].0.starts_with("cryptomatte/")
                && attrs[0].0.len() == "cryptomatte/".len() + 7 + "/name".len()
        );
    }

    #[test]
    fn the_header_is_what_the_format_says() {
        let bytes = write(
            1,
            1,
            &[Channel {
                name: "Y",
                values: &[1.0],
                full_float: false,
            }],
            &[],
        );
        assert_eq!(&bytes[..4], &MAGIC);
        assert_eq!(&bytes[4..8], &[2, 0, 0, 0]);
        // One offset, pointing at the one scanline, which starts with its y.
        let table = bytes.len() - (8 + 8 + 2);
        let offset = u64::from_le_bytes(bytes[table..table + 8].try_into().unwrap()) as usize;
        assert_eq!(offset, table + 8);
        assert_eq!(&bytes[offset..offset + 4], &0i32.to_le_bytes());
    }
}
