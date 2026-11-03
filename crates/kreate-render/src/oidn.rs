//! Intel Open Image Denoise, for a finished render.
//!
//! The path tracer's own filter (`denoise`) is an edge-avoiding blur: right
//! for a view that refines while it is looked at, and visibly a blur on a
//! picture someone keeps — fine texture and soft shading run together. OIDN
//! is a network trained on path-traced pictures: it takes the noisy light,
//! the first surface's colour and its normal, and gives back the picture the
//! samples were converging to. It is what every production renderer ships.
//!
//! # Loaded when it is there, not linked
//!
//! OIDN is a native library of some tens of megabytes, with a device module
//! per kind of hardware. Linking it would make it a build dependency on every
//! platform and every developer's machine; loading it makes it an install
//! option. So it is looked for — beside the executable first, then under
//! `KREATE_OIDN` (the library file, or the folder holding it), then wherever
//! the platform looks — and when it is not found a render is filtered as it
//! always was. [`available`] says which.
//!
//! Only the C API's stable core is used (`oidnNewDevice`, buffers, the "RT"
//! filter), which has not changed since OIDN 2.0.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::PathBuf;
use std::sync::OnceLock;

type Device = *mut c_void;
type Filter = *mut c_void;
type Buffer = *mut c_void;

const DEVICE_TYPE_DEFAULT: c_int = 0;
const DEVICE_TYPE_CPU: c_int = 1;
const FORMAT_FLOAT3: c_int = 3;
const QUALITY_HIGH: c_int = 6;

/// The functions this uses, resolved once.
struct Api {
    // Kept alive for as long as the functions below may be called: the
    // library, and what was loaded ahead of it (see `load`).
    _library: libloading::Library,
    _ahead: Vec<libloading::Library>,
    new_device: unsafe extern "C" fn(c_int) -> Device,
    commit_device: unsafe extern "C" fn(Device),
    release_device: unsafe extern "C" fn(Device),
    device_error: unsafe extern "C" fn(Device, *mut *const c_char) -> c_int,
    new_buffer: unsafe extern "C" fn(Device, usize) -> Buffer,
    write_buffer: unsafe extern "C" fn(Buffer, usize, usize, *const c_void),
    read_buffer: unsafe extern "C" fn(Buffer, usize, usize, *mut c_void),
    release_buffer: unsafe extern "C" fn(Buffer),
    new_filter: unsafe extern "C" fn(Device, *const c_char) -> Filter,
    set_image: unsafe extern "C" fn(
        Filter,
        *const c_char,
        Buffer,
        c_int,
        usize,
        usize,
        usize,
        usize,
        usize,
    ),
    set_bool: unsafe extern "C" fn(Filter, *const c_char, bool),
    set_int: unsafe extern "C" fn(Filter, *const c_char, c_int),
    commit_filter: unsafe extern "C" fn(Filter),
    execute_filter: unsafe extern "C" fn(Filter),
    release_filter: unsafe extern "C" fn(Filter),
}

// The functions are plain C entry points; the library handle is only read.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

/// The file names the library goes by, newest ABI first.
fn names() -> &'static [&'static str] {
    if cfg!(windows) {
        &["OpenImageDenoise.dll"]
    } else if cfg!(target_os = "macos") {
        &["libOpenImageDenoise.2.dylib", "libOpenImageDenoise.dylib"]
    } else {
        &["libOpenImageDenoise.so.2", "libOpenImageDenoise.so"]
    }
}

/// Where to look, in order: beside the executable, `KREATE_OIDN`, then the
/// bare name for the platform's own search.
fn candidates() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|p| p.to_path_buf()))
    {
        for name in names() {
            out.push(dir.join(name));
            // A macOS bundle keeps its libraries beside the executable's
            // folder, in Frameworks.
            out.push(dir.join("../Frameworks").join(name));
        }
    }
    if let Some(given) = std::env::var_os("KREATE_OIDN").map(PathBuf::from) {
        if given.is_dir() {
            for name in names() {
                out.push(given.join(name));
            }
        } else {
            out.push(given);
        }
    }
    out.extend(names().iter().map(PathBuf::from));
    out
}

/// OIDN's own libraries beside `main` that it needs loaded first: its core
/// and Threading Building Blocks.
///
/// Its release builds look for them in `../lib` from their own folder, the
/// layout of its archive — not beside themselves, where a program ships
/// them. Loaded by full path first, they are already there by name when the
/// main library asks for them. (The release workflow also points them at
/// their own folder; this is for a copy made by hand.)
fn ahead_of(main: &std::path::Path) -> Vec<libloading::Library> {
    let Some(dir) = main.parent().filter(|d| !d.as_os_str().is_empty()) else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<std::path::PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            let name = p
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            let lib = name.ends_with(".so.12")
                || name.ends_with(".dylib")
                || name.ends_with(".dll")
                || name.contains(".so.");
            lib && (name.starts_with("libtbb.")
                || name.starts_with("tbb12")
                || name.starts_with("libOpenImageDenoise_core")
                || name.starts_with("OpenImageDenoise_core"))
        })
        .collect();
    // TBB first: the core does not need it, the CPU device does.
    names.sort_by_key(|p| !p.to_string_lossy().contains("tbb"));
    names
        .iter()
        // SAFETY: as for the main library below.
        .filter_map(|p| unsafe { libloading::Library::new(p) }.ok())
        .collect()
}

fn load() -> Option<Api> {
    for path in candidates() {
        // A relative bare name is the platform's search; a path that does not
        // exist is not worth asking it about.
        if path.components().count() > 1 && !path.exists() {
            continue;
        }
        let ahead = ahead_of(&path);
        // SAFETY: loading runs the library's initialisers; OIDN's are its own
        // registration of device modules.
        let Ok(library) = (unsafe { libloading::Library::new(&path) }) else {
            continue;
        };
        if let Some(api) = resolve(library, ahead) {
            tracing::info!("denoising finished renders with {}", path.display());
            return Some(api);
        }
    }
    None
}

fn resolve(library: libloading::Library, ahead: Vec<libloading::Library>) -> Option<Api> {
    // SAFETY: each symbol is looked up by its C name and given the signature
    // `oidn.h` declares for it.
    unsafe {
        macro_rules! sym {
            ($name:literal) => {
                *library.get($name).ok()?
            };
        }
        let api = Api {
            new_device: sym!(b"oidnNewDevice\0"),
            commit_device: sym!(b"oidnCommitDevice\0"),
            release_device: sym!(b"oidnReleaseDevice\0"),
            device_error: sym!(b"oidnGetDeviceError\0"),
            new_buffer: sym!(b"oidnNewBuffer\0"),
            write_buffer: sym!(b"oidnWriteBuffer\0"),
            read_buffer: sym!(b"oidnReadBuffer\0"),
            release_buffer: sym!(b"oidnReleaseBuffer\0"),
            new_filter: sym!(b"oidnNewFilter\0"),
            set_image: sym!(b"oidnSetFilterImage\0"),
            set_bool: sym!(b"oidnSetFilterBool\0"),
            set_int: sym!(b"oidnSetFilterInt\0"),
            commit_filter: sym!(b"oidnCommitFilter\0"),
            execute_filter: sym!(b"oidnExecuteFilter\0"),
            release_filter: sym!(b"oidnReleaseFilter\0"),
            _library: library,
            _ahead: ahead,
        };
        Some(api)
    }
}

fn api() -> Option<&'static Api> {
    static API: OnceLock<Option<Api>> = OnceLock::new();
    API.get_or_init(load).as_ref()
}

/// Whether OIDN was found, and a finished render can be filtered with it.
pub fn available() -> bool {
    api().is_some()
}

/// A path-traced picture to filter: scene-linear light, the first surface's
/// colour (white where a path met nothing) and its normal, one per pixel, in
/// rows from the top.
pub struct Noisy<'a> {
    pub width: usize,
    pub height: usize,
    pub color: &'a [[f32; 3]],
    pub albedo: &'a [[f32; 3]],
    pub normal: &'a [[f32; 3]],
}

/// Owns OIDN objects for the length of one filtering, and releases them
/// whichever way it ends.
struct Held<'a> {
    api: &'a Api,
    device: Device,
    buffers: Vec<Buffer>,
    filter: Filter,
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        // SAFETY: each handle came from this API and is released once.
        unsafe {
            if !self.filter.is_null() {
                (self.api.release_filter)(self.filter);
            }
            for b in &self.buffers {
                (self.api.release_buffer)(*b);
            }
            if !self.device.is_null() {
                (self.api.release_device)(self.device);
            }
        }
    }
}

impl Held<'_> {
    /// The device's error, if it has one.
    fn error(&self) -> Option<String> {
        let mut message: *const c_char = std::ptr::null();
        // SAFETY: `device` is live; the message, when set, is a C string the
        // device owns until its next call.
        let code = unsafe { (self.api.device_error)(self.device, &mut message) };
        (code != 0).then(|| {
            if message.is_null() {
                format!("OIDN error {code}")
            } else {
                // SAFETY: as above.
                unsafe { CStr::from_ptr(message) }
                    .to_string_lossy()
                    .into_owned()
            }
        })
    }
}

/// The picture with its noise taken away. `Err` says why not — OIDN not
/// found, the images not the size they say, or the device's own complaint.
pub fn denoise(noisy: &Noisy<'_>) -> Result<Vec<[f32; 3]>, String> {
    let api = api().ok_or("Open Image Denoise was not found")?;
    let n = noisy.width * noisy.height;
    if n == 0 || noisy.color.len() != n || noisy.albedo.len() != n || noisy.normal.len() != n {
        return Err("the images are not the size they say".into());
    }
    let bytes = n * std::mem::size_of::<[f32; 3]>();
    // SAFETY: every call below is OIDN's C API used as `oidn.h` documents
    // it, on handles created here and released by `Held`.
    unsafe {
        let mut held = Held {
            api,
            device: (api.new_device)(DEVICE_TYPE_DEFAULT),
            buffers: Vec::new(),
            filter: std::ptr::null_mut(),
        };
        if held.device.is_null() {
            held.device = (api.new_device)(DEVICE_TYPE_CPU);
        }
        if held.device.is_null() {
            return Err("Open Image Denoise has no device here".into());
        }
        (api.commit_device)(held.device);
        if let Some(e) = held.error() {
            return Err(e);
        }
        let mut image = |data: Option<&[[f32; 3]]>| -> Result<Buffer, String> {
            let buffer = (api.new_buffer)(held.device, bytes);
            if buffer.is_null() {
                return Err(held.error().unwrap_or_else(|| "no buffer".into()));
            }
            held.buffers.push(buffer);
            if let Some(data) = data {
                (api.write_buffer)(buffer, 0, bytes, data.as_ptr().cast());
            }
            Ok(buffer)
        };
        let color = image(Some(noisy.color))?;
        let albedo = image(Some(noisy.albedo))?;
        let normal = image(Some(noisy.normal))?;
        let output = image(None)?;
        let rt = CString::new("RT").expect("no nul");
        held.filter = (api.new_filter)(held.device, rt.as_ptr());
        if held.filter.is_null() {
            return Err(held.error().unwrap_or_else(|| "no filter".into()));
        }
        let stride = std::mem::size_of::<[f32; 3]>();
        for (name, buffer) in [
            ("color", color),
            ("albedo", albedo),
            ("normal", normal),
            ("output", output),
        ] {
            let name = CString::new(name).expect("no nul");
            (api.set_image)(
                held.filter,
                name.as_ptr(),
                buffer,
                FORMAT_FLOAT3,
                noisy.width,
                noisy.height,
                0,
                stride,
                stride * noisy.width,
            );
        }
        let hdr = CString::new("hdr").expect("no nul");
        (api.set_bool)(held.filter, hdr.as_ptr(), true);
        let quality = CString::new("quality").expect("no nul");
        (api.set_int)(held.filter, quality.as_ptr(), QUALITY_HIGH);
        (api.commit_filter)(held.filter);
        (api.execute_filter)(held.filter);
        if let Some(e) = held.error() {
            return Err(e);
        }
        let mut out = vec![[0.0f32; 3]; n];
        (api.read_buffer)(output, 0, bytes, out.as_mut_ptr().cast());
        if let Some(e) = held.error() {
            return Err(e);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat grey picture with noise on it comes back flatter, and as
    /// bright. Skipped where OIDN is not installed.
    #[test]
    fn noise_on_a_flat_picture_is_taken_away() {
        if !available() {
            eprintln!("Open Image Denoise not found; skipping");
            return;
        }
        let (w, h) = (64, 64);
        let mut state = 12345u32;
        let mut random = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as f32 / u32::MAX as f32
        };
        let color: Vec<[f32; 3]> = (0..w * h)
            .map(|_| {
                let v = 0.5 * (0.5 + random());
                [v, v, v]
            })
            .collect();
        let albedo = vec![[0.8f32; 3]; w * h];
        let normal = vec![[0.0f32, 0.0, 1.0]; w * h];
        let out = denoise(&Noisy {
            width: w,
            height: h,
            color: &color,
            albedo: &albedo,
            normal: &normal,
        })
        .expect("denoises");
        let stats = |p: &[[f32; 3]]| {
            let m = p.iter().map(|c| c[0]).sum::<f32>() / p.len() as f32;
            let s = (p.iter().map(|c| (c[0] - m).powi(2)).sum::<f32>() / p.len() as f32).sqrt();
            (m, s)
        };
        let ((m0, s0), (m1, s1)) = (stats(&color), stats(&out));
        assert!(
            (m1 - m0).abs() < 0.05 * m0,
            "brightness moved: {m0} to {m1}"
        );
        assert!(s1 < s0 * 0.3, "noise {s0} only came down to {s1}");
    }

    #[test]
    fn a_picture_of_the_wrong_size_is_refused() {
        let one = [[0.0f32; 3]; 1];
        let r = denoise(&Noisy {
            width: 2,
            height: 2,
            color: &one,
            albedo: &one,
            normal: &one,
        });
        assert!(r.is_err());
    }
}
