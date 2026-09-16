//! Compiling the document's materials for the viewport.
//!
//! The joining §0 leaves to the application. `kreate-render` cannot see a
//! `ShaderGraph` any more than it can see a `Document`, so a material reaches
//! it as text, numbers and pixels — and this is where the document turns into
//! those.
//!
//! # Recompiled every frame, re-uploaded almost never
//!
//! Compiling is string building and costs microseconds; uploading rebuilds a
//! texture atlas and a pipeline. So every frame compiles, and what is compared
//! is the *code*: if it is the same, only the numbers are written, which is
//! sixteen bytes a material. That is what keeps dragging a colour slider from
//! rebuilding a scene's worth of textures, and it works because the compiler
//! keeps constants out of the text on purpose.
//!
//! # Index zero is the material an object has not got
//!
//! A fresh cube has no material and must still be shaded. So the list always
//! starts with a plain grey, and an object with nothing assigned points at it.
//! Not an error and not invisible — the same judgement the renderer used to
//! make with its default `Surface`.

use kreate_scene::document::MaterialId;
use kreate_scene::shader::ShaderGraph;
use kreate_scene::Document;

/// The document's materials, compiled.
pub struct Materials {
    /// Parallel: what each material compiled to.
    pub compiled: Vec<kreate_shader::Compiled>,
    /// Where each of the document's materials landed in `compiled`.
    index: std::collections::HashMap<MaterialId, u32>,
}

impl Materials {
    /// Compiles every material in the document, plus the default at zero.
    pub fn compile(doc: &Document) -> Self {
        // The default first, so its index is zero whatever else happens.
        let mut compiled = vec![compile_one(&ShaderGraph::standard(
            "Default",
            kreate_math::SceneRgb::new(0.8, 0.8, 0.8),
            0.0,
            0.5,
        ))];
        let mut index = std::collections::HashMap::new();
        for (id, material) in doc.materials() {
            index.insert(id, compiled.len() as u32);
            // The flat fields as the graph they are. A material will carry its
            // own graph when there is an editor to build one in; until then
            // this is the same model reached by a shorter route, which is what
            // `ShaderGraph::standard` exists to make checkable.
            compiled.push(compile_one(&ShaderGraph::standard(
                material.name.clone(),
                material.base_color,
                material.metallic,
                material.roughness,
            )));
        }
        Self { compiled, index }
    }

    /// Which material an object is drawn with.
    ///
    /// Slot zero, which is every object that has not been given per-face
    /// materials — per-face slots need the draw split by slot, which is a
    /// change to how a mesh is uploaded rather than to what is handed over
    /// here.
    pub fn index_of(&self, doc: &Document, object: kreate_scene::ObjectId) -> u32 {
        doc.material_id_at(object, 0)
            .and_then(|id| self.index.get(&id).copied())
            .unwrap_or(0)
    }

    /// Whether these compiled to the same code as `other`.
    ///
    /// The whole list, because a material inserted in the middle moves every
    /// index after it — and an index that moved without the viewport being told
    /// is an object wearing somebody else's material.
    pub fn same_code_as(&self, other: &Materials) -> bool {
        self.compiled.len() == other.compiled.len()
            && self
                .compiled
                .iter()
                .zip(&other.compiled)
                .all(|(a, b)| a.wgsl == b.wgsl)
    }
}

fn compile_one(graph: &ShaderGraph) -> kreate_shader::Compiled {
    // A graph built here cannot fail to compile: it has a surface and no wires.
    // A graph from a *file* can, and when materials carry their own the error
    // becomes a note beside the material rather than an `expect`.
    kreate_shader::compile(graph).expect("a standard surface always compiles")
}

#[cfg(test)]
mod tests {
    use super::*;
    use kreate_scene::document::{Material, Object, ObjectData};

    fn doc_with_materials(n: usize) -> Document {
        let mut doc = Document::new();
        let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
        let mut slots = Vec::new();
        for i in 0..n {
            slots.push(Some(doc.add_material(Material {
                name: format!("M{i}"),
                base_color: kreate_math::SceneRgb::new(i as f32 / 10.0, 0.5, 0.5),
                ..Default::default()
            })));
        }
        doc.set_mesh_slots(mesh, slots);
        doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));
        doc
    }

    #[test]
    fn an_object_with_no_material_takes_the_one_at_zero() {
        // A fresh cube has no material and must still be shaded. The index it
        // gets has to be a real one, or the draw is skipped and the object
        // vanishes.
        let mut doc = Document::new();
        let mesh = doc.add_mesh(kreate_mesh::primitives::cube(1.0));
        let cube = doc.add_object(Object::new("Cube", ObjectData::Mesh(mesh)));
        let materials = Materials::compile(&doc);
        assert_eq!(materials.index_of(&doc, cube), 0);
        assert_eq!(materials.compiled.len(), 1, "only the default");
    }

    #[test]
    fn every_material_gets_its_own_numbers_and_they_share_one_shader() {
        // The property the pipeline cache is built on, checked where the
        // document meets it: ten materials, ten sets of constants, one piece of
        // code. If the code differed, a scene of tinted plastics would compile
        // a shader apiece.
        let doc = doc_with_materials(10);
        let materials = Materials::compile(&doc);
        assert_eq!(materials.compiled.len(), 11, "ten and the default");
        let first = &materials.compiled[0].wgsl;
        assert!(materials.compiled.iter().all(|c| &c.wgsl == first));
        // And the numbers really do differ, or the claim above is about
        // nothing.
        assert_ne!(materials.compiled[1].constants, materials.compiled[2].constants);
    }

    #[test]
    fn an_objects_material_is_the_one_in_its_first_slot() {
        let doc = doc_with_materials(3);
        let (object, _) = doc.objects().next().expect("the cube");
        let materials = Materials::compile(&doc);
        let index = materials.index_of(&doc, object);
        assert_ne!(index, 0, "the cube took the default");
        // And it carries that material's colour rather than another's.
        let (_, first) = doc.materials().next().expect("a material");
        let constants = &materials.compiled[index as usize].constants[0];
        assert_eq!(constants[0], first.base_color.r);
    }

    #[test]
    fn changing_a_colour_does_not_change_the_code() {
        // What decides between writing sixteen bytes and rebuilding a scene's
        // textures. If this ever stops holding, dragging a colour slider
        // rebuilds every material's atlas once a frame.
        let mut doc = doc_with_materials(3);
        let before = Materials::compile(&doc);
        let (id, _) = doc.materials().next().expect("a material");
        doc.material_mut(id).expect("it is there").base_color =
            kreate_math::SceneRgb::new(1.0, 0.0, 0.0);
        let after = Materials::compile(&doc);
        assert!(before.same_code_as(&after));
        assert_ne!(before.compiled[1].constants, after.compiled[1].constants);
    }

    #[test]
    fn adding_a_material_does_change_the_list() {
        // The other half: a material added in the middle moves every index
        // after it, and an index that moved without the viewport being told is
        // an object wearing somebody else's material.
        let doc = doc_with_materials(3);
        let before = Materials::compile(&doc);
        let doc = doc_with_materials(4);
        let after = Materials::compile(&doc);
        assert!(!before.same_code_as(&after));
    }
}
