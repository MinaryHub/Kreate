//! High-poly to low-poly texture baking operator (`mesh.bake_textures`).
//!
//! Bakes Normal, Ambient Occlusion, Base Color, Metallic-Roughness (ORM),
//! Height, or Curvature maps from selected high-poly source objects onto the
//! active low-poly object's UV layout. When only one object is selected, bakes
//! from the object onto itself (useful for AO and Curvature).

use kreate_mesh::bake::{bake_textures, BakeConfig, BakeImage, BakeKind, BakeMaterial, BakeSource};
use kreate_ops_derive::OpParams;
use kreate_scene::image::{ColorSpace, Image, PixelFormat, Source, Tile, TILE};
use kreate_scene::modifier::{EvalCtx, ModifierCache};
use kreate_scene::{DocMut, Document, Material, MaterialMap, ObjectId};

use crate::operator::{Context, OpError, OpResult, Operator};

pub fn register(registry: &mut crate::operator::OperatorRegistry) {
    registry.register::<BakeTextures>();
}

#[derive(Clone, Debug, PartialEq, OpParams)]
pub struct BakeTexturesParams {
    /// Map to bake.
    #[param(variants(
        "normal",
        "ao",
        "base_color",
        "metallic_roughness",
        "height",
        "curvature"
    ))]
    pub kind: u32,
    /// Output resolution (width and height in pixels).
    #[param(min = 64, max = 4096, unit = "count")]
    pub size: u32,
    /// Outward cage offset along the low-poly normal before firing inward rays.
    #[param(min = 0.0001, soft_max = 1.0, unit = "length")]
    pub cage_extrusion: f64,
    /// Maximum ray search distance from the extruded cage.
    #[param(min = 0.001, soft_max = 10.0, unit = "length")]
    pub max_ray_distance: f64,
    /// UV island padding margin in pixels.
    #[param(min = 0, max = 64, unit = "count")]
    pub margin: u32,
    /// Hemisphere samples per texel for Ambient Occlusion.
    #[param(min = 1, max = 128, unit = "count")]
    pub ao_samples: u32,
    /// Occlusion ray reach for Ambient Occlusion.
    #[param(min = 0.01, soft_max = 10.0, unit = "length")]
    pub ao_distance: f64,
    /// Invert green channel for DirectX normal maps (off = OpenGL/glTF +Y).
    pub flip_normal_y: bool,
    /// Assign the baked image to the active object's matching material slot.
    pub assign: bool,
    /// Optional image name; empty uses `<Object>_<Map>`.
    pub name: String,
}

impl Default for BakeTexturesParams {
    fn default() -> Self {
        Self {
            kind: 0,
            size: 1024,
            cage_extrusion: 0.05,
            max_ray_distance: 0.5,
            margin: 16,
            ao_samples: 16,
            ao_distance: 1.0,
            flip_normal_y: false,
            assign: true,
            name: String::new(),
        }
    }
}

/// Bakes high-poly detail (or self-AO/curvature) onto the active mesh's UVs.
pub struct BakeTextures;

impl Operator for BakeTextures {
    type Params = BakeTexturesParams;

    fn id() -> &'static str {
        "mesh.bake_textures"
    }

    fn label() -> &'static str {
        "Bake Textures"
    }

    fn poll(_ctx: &Context, doc: &Document) -> bool {
        doc.selection
            .active
            .and_then(|id| doc.object(id))
            .is_some_and(|o| o.mesh().is_some())
    }

    fn execute(
        params: &Self::Params,
        doc: &mut DocMut<'_>,
        _ctx: &Context,
    ) -> Result<OpResult, OpError> {
        let Some(target_id) = doc.document().selection.active else {
            return Ok(OpResult::Cancelled);
        };
        let mut cache = ModifierCache::new();
        let eval_ctx = EvalCtx::still();
        let Some(target_mesh) = cache.evaluated(doc.document(), target_id, &eval_ctx) else {
            return Err(OpError::Failed("active object has no mesh".into()));
        };
        if target_mesh
            .attributes()
            .get_f32x2(kreate_mesh::attribute::names::UV)
            .is_none()
        {
            return Err(OpError::Failed(
                "active target mesh has no UV coordinates — unwrap it first".into(),
            ));
        }

        // Other selected meshes act as high-poly sources; if only the target is
        // selected, self-bake (e.g. AO or curvature).
        let mut source_ids: Vec<ObjectId> = doc
            .document()
            .selection
            .objects
            .iter()
            .copied()
            .filter(|&id| {
                id != target_id
                    && doc
                        .document()
                        .object(id)
                        .is_some_and(|o| o.mesh().is_some())
            })
            .collect();
        if source_ids.is_empty() {
            source_ids.push(target_id);
        }

        // Evaluate source meshes and extract any RGBA8 texture bytes needed for
        // BaseColor / MetallicRoughness transfer.
        type BakedTex = (u32, u32, Vec<u8>);
        type SlotData = ([f32; 3], f32, f32, Option<BakedTex>, Option<BakedTex>);
        struct OwnedSource {
            mesh: std::sync::Arc<kreate_mesh::MeshData>,
            world: glam::DAffine3,
            slot_colors: Vec<SlotData>,
        }

        let mut owned_sources = Vec::with_capacity(source_ids.len());
        for sid in source_ids {
            let Some(mesh) = cache.evaluated(doc.document(), sid, &eval_ctx) else {
                continue;
            };
            let world = doc.document().world_transform(sid);
            let slot_count = doc
                .document()
                .object(sid)
                .and_then(|o| o.mesh())
                .map(|m| doc.document().mesh_slots(m).len())
                .unwrap_or(0)
                .max(1);
            let mut slot_colors = Vec::with_capacity(slot_count);
            for slot in 0..slot_count {
                if let Some(mat) = doc.document().material_at(sid, slot) {
                    let bc_img = mat.base_color_image.and_then(|id| {
                        let img = doc.document().image(id)?;
                        Some((
                            img.width(),
                            img.height(),
                            kreate_io::image_import::rgba8_of(img, ColorSpace::Srgb),
                        ))
                    });
                    let mr_img = mat.metallic_roughness_image.and_then(|id| {
                        let img = doc.document().image(id)?;
                        Some((
                            img.width(),
                            img.height(),
                            kreate_io::image_import::rgba8_of(img, ColorSpace::Raw),
                        ))
                    });
                    slot_colors.push((
                        [mat.base_color.r, mat.base_color.g, mat.base_color.b],
                        mat.metallic,
                        mat.roughness,
                        bc_img,
                        mr_img,
                    ));
                } else {
                    slot_colors.push(([0.8, 0.8, 0.8], 0.0, 0.5, None, None));
                }
            }
            owned_sources.push(OwnedSource {
                mesh,
                world,
                slot_colors,
            });
        }

        let sources: Vec<BakeSource<'_>> = owned_sources
            .iter()
            .map(|s| BakeSource {
                mesh: &s.mesh,
                world: s.world,
                materials: s
                    .slot_colors
                    .iter()
                    .map(|(bc, m, r, bc_img, mr_img)| BakeMaterial {
                        base_color: *bc,
                        metallic: *m,
                        roughness: *r,
                        base_color_image: bc_img.as_ref().map(|(w, h, px)| BakeImage {
                            width: *w,
                            height: *h,
                            rgba8: px.as_slice(),
                        }),
                        metallic_roughness_image: mr_img.as_ref().map(|(w, h, px)| BakeImage {
                            width: *w,
                            height: *h,
                            rgba8: px.as_slice(),
                        }),
                    })
                    .collect(),
            })
            .collect();

        let kind = BakeKind::from_index(params.kind);
        let size = params.size.clamp(64, 4096);
        let config = BakeConfig {
            width: size,
            height: size,
            kind,
            cage_extrusion: params.cage_extrusion.max(1e-4) as f32,
            max_ray_distance: params.max_ray_distance.max(1e-3) as f32,
            margin: params.margin.min(64),
            ao_samples: params.ao_samples.clamp(1, 128),
            ao_distance: params.ao_distance.max(1e-3) as f32,
            flip_normal_y: params.flip_normal_y,
            ..BakeConfig::default()
        };

        let target_world = doc.document().world_transform(target_id);
        let baked = bake_textures(&target_mesh, target_world, &sources, &config)
            .ok_or_else(|| OpError::Failed("baking produced no output".into()))?;

        let target_name = doc
            .document()
            .object(target_id)
            .map(|o| o.name.clone())
            .unwrap_or_else(|| "Mesh".into());
        let image_name = if params.name.trim().is_empty() {
            format!("{target_name}_{}", kind.suffix())
        } else {
            params.name.trim().to_string()
        };

        let image = image_from_rgba8(&image_name, baked.width, baked.height, &baked.rgba8);
        let d = doc.document_mut();
        let img_id = d.add_image(image);

        if params.assign {
            if let Some(map_slot) = match kind {
                BakeKind::TangentNormal => Some(MaterialMap::Normal),
                BakeKind::AmbientOcclusion => Some(MaterialMap::Occlusion),
                BakeKind::BaseColor => Some(MaterialMap::BaseColor),
                BakeKind::MetallicRoughness => Some(MaterialMap::MetallicRoughness),
                BakeKind::Height | BakeKind::Curvature => None,
            } {
                if let Some(mesh_id) = d.object(target_id).and_then(|o| o.mesh()) {
                    let mat_id = match d.mesh_slots(mesh_id).first().copied().flatten() {
                        Some(id) => id,
                        None => {
                            let m = d.add_material(Material {
                                name: format!("{target_name} Material"),
                                ..Material::default()
                            });
                            d.set_mesh_slots(mesh_id, vec![Some(m)]);
                            m
                        }
                    };
                    if let Some(mat) = d.material_mut(mat_id) {
                        *mat.map_mut(map_slot) = Some(img_id);
                    }
                }
            }
        }

        Ok(OpResult::Finished)
    }
}

/// Packs flat RGBA8 rows into a tiled `Image` datablock (`Source::Generated`).
fn image_from_rgba8(name: &str, w: u32, h: u32, rgba: &[u8]) -> Image {
    let mut image = Image::new(name, w, h, PixelFormat::Rgba8);
    image.source = Source::Generated;
    let tx_count = w.div_ceil(TILE);
    let ty_count = h.div_ceil(TILE);
    for ty in 0..ty_count {
        for tx in 0..tx_count {
            let mut tile = Tile::blank(PixelFormat::Rgba8);
            let dst = tile.bytes_mut();
            let x0 = tx * TILE;
            let y0 = ty * TILE;
            let copy_w = (w - x0).min(TILE) as usize;
            let copy_h = (h - y0).min(TILE) as usize;
            for row in 0..copy_h {
                let src_off = (((y0 + row as u32) * w + x0) * 4) as usize;
                let dst_off = row * (TILE as usize) * 4;
                let len = copy_w * 4;
                dst[dst_off..dst_off + len].copy_from_slice(&rgba[src_off..src_off + len]);
            }
            image.set_tile(tx, ty, Some(std::sync::Arc::new(tile)));
        }
    }
    image
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_mesh::primitives;

    #[test]
    fn bake_textures_operator_creates_and_assigns_normal_map() {
        let mut doc = Document::new();
        let high = doc.add_mesh_object("High", primitives::uv_sphere(1.05, 20, 14));
        let low = doc.add_mesh_object("Low", primitives::cube(2.0));
        doc.selection.clear();
        doc.selection.add(high);
        doc.selection.add(low);
        doc.selection.active = Some(low);

        let mut edit = DocMut::new(&mut doc);
        let params = BakeTexturesParams {
            kind: 0, // normal
            size: 64,
            cage_extrusion: 0.4,
            max_ray_distance: 2.0,
            margin: 4,
            assign: true,
            ..Default::default()
        };
        BakeTextures::execute(&params, &mut edit, &Context::default()).expect("bakes");
        let mat = doc.material_at(low, 0).expect("material created");
        let img_id = mat.normal_image.expect("normal map assigned");
        let img = doc.image(img_id).expect("image stored");
        assert_eq!((img.width(), img.height()), (64, 64));
        assert_eq!(img.name, "Low_Normal");
    }
}
