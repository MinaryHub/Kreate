//! The material tab's preview: one material on a ball, lit as the material
//! preview lights the viewport.
//!
//! A viewport of its own, and a small one. Showing a material on the model
//! means switching the whole scene to the preview's shading; this answers
//! "what is this made of" for one slot, or for a physical material before
//! it is chosen, without touching the view the user is working in.
//!
//! # Drawn when it changed, and until it settles
//!
//! The picture is kept: a frame whose material compiled to the same thing,
//! with the same images, in the same display, draws nothing. A new one
//! starts the viewport's running mean again and is drawn each frame until
//! the mean is full, which is the same sixteen frames a still view takes.

use kreate_render::viewport::{Draw, MaterialSource, Viewport};
use kreate_render::{Camera, GpuMesh, RenderMeshCpu, ShadingMode};
use kreate_scene::document::MaterialId;
use kreate_scene::Document;

/// The picture's side, in pixels: twice the panel's, for a sharp ball on a
/// high-density screen.
pub const SIZE: u32 = 256;

/// The ball's radius, in metres. A finish is laid out in metres too, so this
/// is what says how coarse a grain looks: ten centimetres across shows a few
/// growth rings of oak, and a speckle of granite as stones.
const RADIUS: f32 = 0.05;

/// What a preview is of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreviewOf {
    /// A material of the document.
    Material(MaterialId),
    /// A physical material from the library, by its id, before it is chosen.
    Physical(&'static str),
}

/// What the picture was made of, to tell whether it needs making again.
#[derive(Clone, PartialEq)]
struct Drawn {
    of: PreviewOf,
    compiled: kreate_shader::Compiled,
    revisions: Vec<u64>,
    display: kreate_render::display::Display,
}

pub struct MaterialPreview {
    viewport: Viewport,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    /// The picture, as the interface draws it.
    pub id: egui::TextureId,
    ball: GpuMesh,
    camera: Camera,
    drawn: Option<Drawn>,
}

impl MaterialPreview {
    /// Rendered into as sRGB and sampled as plain bytes, as the viewport's
    /// own picture is: see `ViewportTarget::SAMPLED_AS` in the application.
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
    const SAMPLED_AS: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    pub fn new(device: &wgpu::Device, renderer: &mut egui_wgpu::Renderer) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("material preview"),
            size: wgpu::Extent3d {
                width: SIZE,
                height: SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[Self::SAMPLED_AS],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampled = texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(Self::SAMPLED_AS),
            ..Default::default()
        });
        let id = renderer.register_native_texture(device, &sampled, wgpu::FilterMode::Linear);
        let mut viewport = Viewport::new(device, Self::FORMAT, SIZE, SIZE);
        viewport.shading = ShadingMode::Material;
        viewport.show_grid = false;
        viewport.show_ground_shadow = false;
        viewport.show_wireframe = false;
        let ball = GpuMesh::upload(
            device,
            &viewport.layouts,
            &RenderMeshCpu::extract(&kreate_mesh::primitives::uv_sphere(RADIUS, 64, 32)),
            glam::Vec4::ONE,
        );
        // The default view's three-quarter turn, close enough that the ball
        // fills most of the square.
        let camera = Camera {
            distance: 0.15,
            ..Camera::default()
        };
        Self {
            viewport,
            texture,
            view,
            id,
            ball,
            camera,
            drawn: None,
        }
    }

    /// The picture, `SIZE` square, for copying out.
    pub fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }

    /// Whether the picture still has frames to add.
    pub fn accumulating(&self) -> bool {
        self.viewport.accumulating()
    }

    /// Draws `of` into the picture, if it is not already there. Returns
    /// whether it drew, so the caller knows to ask for another frame.
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        doc: &Document,
        of: PreviewOf,
        punchy: bool,
    ) -> bool {
        let compiled = match of {
            PreviewOf::Material(id) => match doc.material(id) {
                Some(material) => crate::materials::compile_material(doc, material),
                None => return false,
            },
            PreviewOf::Physical(id) => match kreate_scene::material_library::find(id) {
                Some(entry) => crate::materials::compile_material(doc, &entry.material(id)),
                None => return false,
            },
        };
        let revisions = compiled
            .images
            .iter()
            .map(|id| doc.image(*id).map_or(0, |i| i.revision()))
            .collect();
        let display = kreate_render::display::Display {
            view: if punchy {
                kreate_math::ViewTransform::AgXPunchy
            } else {
                kreate_math::ViewTransform::AgX
            },
            exposure: crate::scenelights::STUDIO_EXPOSURE,
        };
        let now = Drawn {
            of,
            compiled,
            revisions,
            display,
        };
        let changed = self.drawn.as_ref() != Some(&now);
        if !changed && !self.viewport.accumulating() {
            return false;
        }
        if changed {
            let same_code = self.drawn.as_ref().is_some_and(|d| {
                d.compiled.wgsl == now.compiled.wgsl && d.revisions == now.revisions
            });
            let c = &now.compiled;
            if same_code {
                self.viewport
                    .set_material_constants(queue, 0, &c.constants, &c.image_slots);
            } else {
                let pixels = crate::materials::pixels_for(doc, c);
                let images: Vec<_> = pixels.iter().map(|i| i.source()).collect();
                self.viewport.set_materials(
                    device,
                    queue,
                    &[MaterialSource {
                        wgsl: &c.wgsl,
                        constants: &c.constants,
                        image_slots: &c.image_slots,
                        images: &images,
                    }],
                );
            }
            self.viewport.set_transmissive(&[c.transmission()]);
            self.viewport.set_display(queue, now.display);
            if self.drawn.is_none() {
                self.viewport.set_environment(
                    device,
                    queue,
                    kreate_render::studio::WIDTH,
                    kreate_render::studio::HEIGHT,
                    &kreate_render::studio::equirect(),
                );
            }
            self.viewport.restart_accumulation();
            self.drawn = Some(now);
        }
        // The studio as the material preview turns it for this view.
        let studio = crate::scenelights::material_preview(self.camera.rotation);
        self.viewport.set_world(studio.world);
        self.viewport.set_world_shape(queue, studio.yaw, 1.0);
        self.viewport
            .set_lights(device, queue, self.camera.eye(), &studio.lights);
        let mut draws = vec![Draw::new(&self.ball, glam::DAffine3::IDENTITY)];
        self.viewport
            .prepare(device, queue, &self.camera, &mut draws);
        self.viewport.render(encoder, &self.view, &draws);
        true
    }
}
