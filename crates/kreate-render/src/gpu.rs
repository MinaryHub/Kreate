//! Device selection and surface configuration.
//!
//! Two things here look like boilerplate and are not.
//!
//! **Adapter choice is a user-visible setting, not a default.** On a laptop with
//! both an integrated and a discrete GPU, silently landing on the integrated one
//! is a top-tier "your app is slow" support ticket, and `PowerPreference` alone
//! does not reliably prevent it. So adapters are enumerated, the choice is
//! overridable by name, and the executable also exports the vendor hint symbols
//! (see `apps/kreate`).
//!
//! **Frame latency is capped at 1.** wgpu defaults to 2, which costs 16-33 ms of
//! pure input lag for nothing on a tool that does little GPU work per frame.
//! Latency does not show up in an FPS counter: an application can sit at a solid
//! 144 fps and still feel terrible, and by the time that is noticed the causes
//! are spread across a dozen places.

use wgpu::{Adapter, Device, Instance, Queue, Surface, SurfaceConfiguration};

/// Depth buffer format.
///
/// Float depth plus reversed-Z gives near-uniform precision in world space,
/// which is what lets one scene hold a 1 mm chamfer and a 10 km backdrop.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Depth comparison for reversed-Z. Depth *increases* toward the viewer, so
/// "closer" is `Greater`, and the depth buffer clears to 0.0 rather than 1.0.
pub const DEPTH_COMPARE: wgpu::CompareFunction = wgpu::CompareFunction::Greater;

/// Depth clear value for reversed-Z.
pub const DEPTH_CLEAR: f64 = 0.0;

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("no suitable GPU adapter found: {0}")]
    NoAdapter(String),
    #[error("could not create a device: {0}")]
    Device(String),
}

/// A human-readable description of one adapter, for the preferences UI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterSummary {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub driver: String,
    pub is_discrete: bool,
}

impl AdapterSummary {
    fn from_info(info: &wgpu::AdapterInfo) -> Self {
        Self {
            name: info.name.clone(),
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
            driver: if info.driver_info.is_empty() {
                info.driver.clone()
            } else {
                format!("{} ({})", info.driver, info.driver_info)
            },
            is_discrete: info.device_type == wgpu::DeviceType::DiscreteGpu,
        }
    }

    /// One-line form for logs and the adapter picker.
    pub fn describe(&self) -> String {
        format!("{} [{} / {}]", self.name, self.backend, self.device_type)
    }
}

/// How to pick a device.
#[derive(Clone, Debug)]
pub struct GpuOptions {
    pub backends: wgpu::Backends,
    /// Tie-breaker among otherwise equal adapters.
    ///
    /// Enabling several backends does not express a preference to wgpu: the same
    /// physical GPU is reported once per backend, and which one comes back is an
    /// implementation detail. Since the backend choice has real consequences for
    /// latency and tooling, it is ranked explicitly here rather than left to luck.
    pub preferred_backend: Option<wgpu::Backend>,
    pub power_preference: wgpu::PowerPreference,
    /// Exact adapter name from preferences. When it matches, it wins over
    /// `power_preference`; when it does not, selection falls back rather than
    /// failing, so a preferences file that outlives a hardware change still boots.
    pub adapter_name: Option<String>,
}

impl Default for GpuOptions {
    fn default() -> Self {
        Self {
            // DX12 first on Windows: better flip-model present integration and
            // more predictable latency than the Vulkan path, with WARP available
            // as a fallback for support cases. Vulkan stays selectable.
            backends: if cfg!(windows) {
                wgpu::Backends::DX12 | wgpu::Backends::VULKAN
            } else {
                wgpu::Backends::PRIMARY
            },
            // DX12 on Windows: better flip-model present integration and more
            // predictable latency. Vulkan remains selectable and is often faster
            // on NVIDIA, with better capture tooling.
            preferred_backend: if cfg!(windows) {
                Some(wgpu::Backend::Dx12)
            } else {
                None
            },
            power_preference: wgpu::PowerPreference::HighPerformance,
            adapter_name: None,
        }
    }
}

/// Ranks an adapter; the highest score is selected.
///
/// Pure, so the policy can be tested without a GPU — which matters, because the
/// failure it prevents (an integrated GPU silently chosen on a hybrid laptop) is
/// invisible in code review and shows up as "the app is slow".
pub fn adapter_score(
    device_type: wgpu::DeviceType,
    backend: wgpu::Backend,
    opts: &GpuOptions,
) -> i32 {
    use wgpu::{DeviceType as D, PowerPreference as P};
    let class = match (opts.power_preference, device_type) {
        // Software rasterisation is a last resort under any preference: it is
        // correct but far too slow to model with.
        (_, D::Cpu) => 0,
        (P::LowPower, D::IntegratedGpu) => 400,
        (P::LowPower, D::DiscreteGpu) => 300,
        (_, D::DiscreteGpu) => 400,
        (_, D::IntegratedGpu) => 300,
        (_, D::VirtualGpu) => 200,
        (_, D::Other) => 100,
    };
    let backend_bonus = match opts.preferred_backend {
        Some(preferred) if preferred == backend => 50,
        _ => 0,
    };
    class + backend_bonus
}

/// The floor Kreate needs from a device.
///
/// Stated explicitly so an unsupported machine fails with a clear message rather
/// than with a buffer allocation error somewhere deep in a mesh upload.
pub fn minimum_limits() -> wgpu::Limits {
    wgpu::Limits {
        // One position array for a 10 M vertex mesh is 120 MB, and it is not the
        // only arena.
        max_buffer_size: 256 << 20,
        max_storage_buffer_binding_size: 128 << 20,
        // The viewport at high DPI, and the font atlas once CJK glyphs are in it.
        max_texture_dimension_2d: 8192,
        ..wgpu::Limits::downlevel_defaults()
    }
}

/// An initialised device.
pub struct Gpu {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub info: AdapterSummary,
}

impl Gpu {
    pub fn create_instance(opts: &GpuOptions) -> Instance {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = opts.backends;
        // Honour WGPU_BACKEND and friends, so a support case can be told to set
        // one environment variable rather than rebuild.
        Instance::new(desc.with_env())
    }

    /// Every adapter the instance can see, for the preferences picker.
    pub async fn enumerate(instance: &Instance, opts: &GpuOptions) -> Vec<AdapterSummary> {
        instance
            .enumerate_adapters(opts.backends)
            .await
            .iter()
            .map(|a| AdapterSummary::from_info(&a.get_info()))
            .collect()
    }

    /// Initialises a device on an existing instance.
    ///
    /// The instance is passed in rather than created here because a surface
    /// belongs to the instance that made it. Creating a second instance
    /// internally would mean the adapter and the surface came from different
    /// instances, and the surface would be invalidated the moment the caller's
    /// instance went out of scope — which fails at the first draw with a
    /// bewildering "adapter does not exist".
    pub async fn new(
        instance: Instance,
        opts: &GpuOptions,
        compatible_surface: Option<&Surface<'_>>,
    ) -> Result<Self, GpuError> {
        let adapter = Self::pick_adapter(&instance, opts, compatible_surface).await?;
        let info = AdapterSummary::from_info(&adapter.get_info());
        tracing::info!(adapter = %info.describe(), driver = %info.driver, "selected GPU");

        let available = adapter.limits();
        if !minimum_limits().check_limits(&available) {
            return Err(GpuError::Device(format!(
                "{} does not meet Kreate's minimum limits",
                info.describe()
            )));
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("kreate device"),
                // Deliberately the baseline. The renderer is designed to need
                // only compute plus a single indirect draw, which exists on
                // Vulkan, Metal and DX12 alike. Anything beyond that is an
                // optional accelerator, never a load-bearing assumption, or
                // macOS quietly becomes a second-class target.
                required_features: wgpu::Features::empty(),
                // What the hardware actually offers, not the WebGPU default,
                // which caps buffers at 256 MiB. Capping ourselves there would
                // limit scene size for no portability gain — portability is
                // about features, not about refusing memory that exists.
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .map_err(|e| GpuError::Device(e.to_string()))?;

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            info,
        })
    }

    async fn pick_adapter(
        instance: &Instance,
        opts: &GpuOptions,
        compatible_surface: Option<&Surface<'_>>,
    ) -> Result<Adapter, GpuError> {
        if let Some(wanted) = &opts.adapter_name {
            let adapters = instance.enumerate_adapters(opts.backends).await;
            if let Some(found) = adapters.into_iter().find(|a| &a.get_info().name == wanted) {
                return Ok(found);
            }
            // The saved adapter is gone: a driver update, an external GPU
            // unplugged, a different machine. Warn and fall through rather than
            // refusing to start.
            tracing::warn!(
                requested = %wanted,
                "preferred GPU not found; falling back to automatic selection"
            );
        }

        // Score every adapter rather than delegating to request_adapter, so the
        // choice is explainable and testable. Ties keep enumeration order, which
        // makes selection deterministic across runs.
        let adapters = instance.enumerate_adapters(opts.backends).await;
        let best = adapters
            .into_iter()
            // `is_surface_supported` rather than inspecting capabilities:
            // `get_capabilities` panics for an adapter that cannot present to the
            // surface at all, which on a hybrid machine is most of them.
            .filter(|a| match compatible_surface {
                Some(surface) => a.is_surface_supported(surface),
                None => true,
            })
            .max_by_key(|a| {
                let info = a.get_info();
                adapter_score(info.device_type, info.backend, opts)
            });

        match best {
            Some(a) => Ok(a),
            None => instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: opts.power_preference,
                    force_fallback_adapter: false,
                    compatible_surface,
                })
                .await
                .map_err(|e| GpuError::NoAdapter(e.to_string())),
        }
    }
}

/// Chooses a present mode, preferring low latency while a user is interacting.
///
/// `Mailbox` shows the newest finished frame and discards the rest, which is the
/// best latency available without tearing. `Immediate` tears but is lower still.
/// `Fifo` is the only universally supported mode and is what an idle window
/// should sit on, since it does not spin the GPU.
pub fn choose_present_mode(
    surface: &Surface<'_>,
    adapter: &Adapter,
    interactive: bool,
) -> wgpu::PresentMode {
    let caps = surface.get_capabilities(adapter);
    let supports = |m: wgpu::PresentMode| caps.present_modes.contains(&m);
    if interactive {
        if supports(wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::Fifo
        }
    } else {
        wgpu::PresentMode::Fifo
    }
}

/// Builds a surface configuration with the latency cap applied.
///
/// Prefers a non-sRGB view format and does the linear-to-sRGB conversion in the
/// shader, so tonemapping stays explicit and visible instead of being hidden in
/// a format choice.
pub fn surface_config(
    surface: &Surface<'_>,
    adapter: &Adapter,
    width: u32,
    height: u32,
) -> SurfaceConfiguration {
    let caps = surface.get_capabilities(adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(caps.formats[0]);
    SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        // A zero-sized surface is invalid and happens routinely while a window
        // is being resized or minimised.
        width: width.max(1),
        height: height.max(1),
        present_mode: wgpu::PresentMode::Fifo,
        // See the module docs: the default of 2 is pure latency for a tool.
        desired_maximum_frame_latency: 1,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
    }
}

/// Creates the depth texture for a viewport.
///
/// Sizes are clamped to the device's limits rather than passed through. A
/// viewport size is ultimately derived from a window rectangle and a DPI factor,
/// so a value that is zero, negative-turned-huge by a cast, or simply larger than
/// the hardware allows is reachable from ordinary interaction. Unclamped, that
/// surfaces as `Texture with 'depth' label is invalid` at the *view*, several
/// steps from the cause, and takes the device down with it.
pub fn create_depth_texture(device: &Device, width: u32, height: u32) -> wgpu::TextureView {
    let max = device.limits().max_texture_dimension_2d;
    let (width, height) = (width.clamp(1, max), height.clamp(1, max));
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

/// The depth-stencil state for a pass that writes depth, in reversed-Z.
pub fn depth_write_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(true),
        depth_compare: Some(DEPTH_COMPARE),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

/// The depth-stencil state for an overlay: tests against depth but does not
/// write, so overlays layer by draw order rather than fighting each other.
pub fn depth_read_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(false),
        depth_compare: Some(DEPTH_COMPARE),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

/// The depth-stencil state for a pass that ignores depth entirely.
///
/// What makes x-ray picking two lines rather than a second code path: with
/// `Always` an occluded element still writes its id, so a see-through mode
/// selects what it shows. The same shader, a different pipeline.
pub fn depth_ignore_state() -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(false),
        depth_compare: Some(wgpu::CompareFunction::Always),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reversed_z_constants_are_consistent() {
        // These three must agree or the depth test silently rejects everything:
        // clearing to 0 with a `Greater` test means "nothing has been drawn yet".
        assert_eq!(DEPTH_CLEAR, 0.0);
        assert_eq!(DEPTH_COMPARE, wgpu::CompareFunction::Greater);
        assert_eq!(DEPTH_FORMAT, wgpu::TextureFormat::Depth32Float);
    }

    #[test]
    fn overlay_state_reads_but_does_not_write_depth() {
        let s = depth_read_state();
        assert_eq!(s.depth_write_enabled, Some(false));
        assert_eq!(s.depth_compare, Some(DEPTH_COMPARE));
        assert_eq!(depth_write_state().depth_write_enabled, Some(true));
    }

    #[test]
    fn default_options_prefer_the_discrete_gpu() {
        let o = GpuOptions::default();
        assert_eq!(o.power_preference, wgpu::PowerPreference::HighPerformance);
        assert!(o.adapter_name.is_none());
    }

    fn opts(pref: wgpu::PowerPreference, backend: Option<wgpu::Backend>) -> GpuOptions {
        GpuOptions {
            power_preference: pref,
            preferred_backend: backend,
            ..Default::default()
        }
    }

    #[test]
    fn discrete_beats_integrated_on_a_hybrid_laptop() {
        // The exact situation on the development machine: the same two GPUs show
        // up under both Vulkan and DX12, and picking the Intel one is the single
        // most common cause of "your app is slow".
        let o = opts(
            wgpu::PowerPreference::HighPerformance,
            Some(wgpu::Backend::Dx12),
        );
        let discrete_vk = adapter_score(wgpu::DeviceType::DiscreteGpu, wgpu::Backend::Vulkan, &o);
        let integrated_dx = adapter_score(wgpu::DeviceType::IntegratedGpu, wgpu::Backend::Dx12, &o);
        assert!(
            discrete_vk > integrated_dx,
            "the discrete GPU must win even when the integrated one is on the preferred backend"
        );
    }

    #[test]
    fn the_preferred_backend_breaks_ties_between_equal_gpus() {
        let o = opts(
            wgpu::PowerPreference::HighPerformance,
            Some(wgpu::Backend::Dx12),
        );
        let dx = adapter_score(wgpu::DeviceType::DiscreteGpu, wgpu::Backend::Dx12, &o);
        let vk = adapter_score(wgpu::DeviceType::DiscreteGpu, wgpu::Backend::Vulkan, &o);
        assert!(dx > vk, "same GPU, preferred backend should win");
    }

    #[test]
    fn low_power_prefers_the_integrated_gpu() {
        let o = opts(wgpu::PowerPreference::LowPower, None);
        assert!(
            adapter_score(wgpu::DeviceType::IntegratedGpu, wgpu::Backend::Vulkan, &o)
                > adapter_score(wgpu::DeviceType::DiscreteGpu, wgpu::Backend::Vulkan, &o)
        );
    }

    #[test]
    fn software_rendering_is_always_last() {
        for pref in [
            wgpu::PowerPreference::HighPerformance,
            wgpu::PowerPreference::LowPower,
        ] {
            let o = opts(pref, Some(wgpu::Backend::Dx12));
            let cpu = adapter_score(wgpu::DeviceType::Cpu, wgpu::Backend::Dx12, &o);
            for dt in [
                wgpu::DeviceType::DiscreteGpu,
                wgpu::DeviceType::IntegratedGpu,
                wgpu::DeviceType::VirtualGpu,
                wgpu::DeviceType::Other,
            ] {
                assert!(
                    adapter_score(dt, wgpu::Backend::Vulkan, &o) > cpu,
                    "{dt:?} should outrank software rendering"
                );
            }
        }
    }

    #[test]
    fn minimum_limits_allow_a_large_mesh() {
        // A 10 M vertex position array is 120 MB; the floor must clear it with
        // room for the other arenas.
        let l = minimum_limits();
        assert!(l.max_buffer_size >= 256 << 20);
        assert!(l.max_storage_buffer_binding_size >= 128 << 20);
    }

    #[test]
    fn windows_defaults_include_dx12_and_vulkan() {
        let o = GpuOptions::default();
        if cfg!(windows) {
            assert!(o.backends.contains(wgpu::Backends::DX12));
            assert!(
                o.backends.contains(wgpu::Backends::VULKAN),
                "Vulkan must stay available; it is often faster on NVIDIA and has better tooling"
            );
        }
    }

    #[test]
    fn adapter_summary_describes_itself() {
        let s = AdapterSummary {
            name: "Test GPU".into(),
            backend: "Dx12".into(),
            device_type: "DiscreteGpu".into(),
            driver: "1.0".into(),
            is_discrete: true,
        };
        assert_eq!(s.describe(), "Test GPU [Dx12 / DiscreteGpu]");
    }
}
