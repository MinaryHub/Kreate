//! The Model Context Protocol, for Kreate: what Claude talks to.
//!
//! Claude Code and Claude Desktop start `kreate-mcp` and speak JSON-RPC 2.0 to
//! it on its standard input and output, one message per line. Each tool call
//! becomes a request to the Kreate that is running on this machine, through
//! the socket it opens at start-up (see `kreate::bridge` in the application),
//! and comes back as the tool's result.
//!
//! The protocol is written out here rather than taken from an SDK: what a
//! tools-only server needs is four methods, and the parts it can get wrong are
//! few enough to test one by one.

use serde_json::{json, Value};

/// The newest protocol revision this server speaks. A client asking for an
/// older one it also knows is answered in that one.
pub const PROTOCOL_VERSION: &str = "2025-06-18";
const KNOWN_VERSIONS: [&str; 3] = ["2025-06-18", "2025-03-26", "2024-11-05"];

/// The other end: a running Kreate, or a stand-in in a test.
pub trait App {
    /// Sends one request and waits for its answer.
    fn call(&mut self, method: &str, params: Value) -> Result<Value, String>;
}

/// What a model reads before it uses the tools.
const INSTRUCTIONS: &str = "Kreate is a 3D modelling application, open on the user's screen. \
These tools act on the document it has open: every edit goes through the same operators as \
its menus and lands in its undo history. Start with get_scene to see what is there. Edits are \
script lines run by run_script; list_operators gives every operator and its parameters. \
Units: lengths in metres and angles in radians, in what every tool takes and gives (get_scene also gives rotations in degrees, to read); speak to the user in the length unit get_scene names as display_length. Z is up. Use capture_viewport to look at the result, or camera=true to see \
the shot through the scene camera; render_image writes a finished, path-traced picture \
through a camera to a file.\n\
Workflow: object.* operators work in object mode on the selection and the active object. \
mesh.* and uv.* operators work in edit mode on the selected elements: enter it with \
kreate.ops.object.mode_set(mode=\"edit\") and leave it with mode=\"object\". \
Selecting faces: by direction, kreate.ops.mesh.select_where(mode=\"face\", normal=(0,0,1)) \
takes every face facing up (angle= widens it, min= and max= limit it to a world box that \
holds the faces' centers — make the box generous, not a sliver around one guess); by number, \
get_mesh(faces=true) gives each face's index, center and normal, and \
kreate.ops.mesh.select_set(mode=\"face\", elements=[...]) selects exactly those. Prefer \
these to many small boxes: a box that misses selects nothing, and the next line acts on \
nothing. \
run_script stops at the first line that fails and says why and how to fix it; read that \
error, fix the cause (often: select the object, or enter edit mode) and send the rest again \
rather than repeating the same lines. \
run_script returns, for each line, the objects it created (name and uid), so name and \
reuse them rather than guessing. Object names are unique: a second \"Cube\" becomes \"Cube.001\". \
get_scene gives each mesh's dimensions and world bounds, for sizing and placing, and \
get_mesh its vertex positions and faces; \
object.set_dimensions sizes an object in metres. Reuse a material with \
material.link(slot=..., uid=...) using the uid get_scene gives; name materials with \
material.rename. Load textures with kreate.ops.image.open(path=...), which returns the \
image's uid, or make one with image.generate (noise, voronoi, wood, marble, checker, bricks; \
output=\"color\", \"height\" for a roughness map or \"normal\" for bumps): every pattern \
tiles. run_script reports the uids of images and materials a line made. Put an image on a \
material with material.set_image(slot=..., map=..., uid=...); unwrap or project UVs first \
(uv.project(mode=\"box\") in edit mode needs no seams).\n\
Light is physical: a sun is tens of thousands of lux, and a world background's strength is \
in nits, so beside a sun a daylight sky needs strength in the thousands (3000 to 10000) or \
it renders black; indoors with lamps only, a few nits is right.\n\
Shapes: object.add_lathe turns a radius, height profile into a vase, cup or bottle \
(resolution=6 curves it through the points); \
object.add_sweep carries a circle or a profile of your own along a path of points — a \
handle, a spout (scale_end narrows it), a pipe, a cable, a frame (closed=true). \
Round furniture edges with mesh.bevel(width=..., segments=3 or more) in edit mode, or \
non-destructively with a bevel modifier (object.modifier_set_bevel(segments=...)). Under a \
subdivide modifier, mesh.crease(value=...) keeps selected edges sharp (1) or softly \
rounded (0.2 to 0.8), and edges marked with mesh.mark_sharp stay sharp. object.shade_smooth(angle=...) smooths a whole \
object. Placement: object.place_on(target=...) stands the active object on another; \
object.align lines up sides of bounding boxes; check_overlaps finds parts that run into \
each other. mesh.select_where selects elements by a world-space box, a face direction or a \
material slot instead of by index. data.purge removes unused meshes, materials and images.\n\
Looking: get_context says what the user has selected and where they look — read it when they \
say \"this\" or \"here\". capture_viewport(views=[\"front\",\"right\",\"top\"]) shows a shape \
from three sides at once; only=[...] shows some objects alone.\n\
Assets before modelling by hand: search_assets finds models, materials and HDRIs (Poly Haven, \
ambientCG, Poly Pizza for low-poly and animated models, the user's folders) and add_asset adds \
one beside what is there; get_scene shows it when it has arrived. When nothing fits, \
generate_model makes one from words or a picture (a minute or two).\n\
Organic shapes — a character, a creature, a face: work in a loop and measure each turn. \
compare_silhouette(reference=..., view=\"front\" then \"right\") against the user's picture says \
where the model has too much or too little; check_proportions says how a figure departs from \
an adult's proportions; analyze_mesh says where the surface is bumpy, open or lopsided. Fix \
with sculpt.deform_region (push, inflate, move, scale, flatten or smooth a ball- or egg-shaped \
region by metres; mirror_x=true for both sides) and sculpt.crease_path (grooves and ridges \
along points), then measure again; stop when the agreement stops rising. Fine detail — pores, \
scales, bark — comes last: generate_height_map, image.open(color=False), then \
sculpt.displace_image on a mesh dense enough to carry it.\n\
Rigging a standing figure: select the mesh and run object.add_humanoid_rig (it fits and binds \
a skeleton). Then check_rig: fix the worst bone's torn edges with \
object.vertex_weights_set(group=bone, from=..., to=..., exclusive=true) and \
object.vertex_weights_smooth, and check again until no bone tears. Build one side and copy it \
with armature.symmetrize and object.vertex_weights_mirror.\n\
Animation: timeline reads and moves the clock and switches a file's takes (NLA strips) on and \
off; key poses with anim.keyframe_insert in pose mode, then timeline(frame=...) and \
capture_viewport to look at a pose. A chat answer's changes can be undone together by the \
user, so make each request one coherent set of changes.";

/// Ways of working, as `prompts/list` offers them: a name, what it is for,
/// the one thing it asks, and the request it makes, `{}` where that goes.
const PROMPTS: &[(&str, &str, &str, &str)] = &[
    (
        "rig_character",
        "Rig a standing character so it bends cleanly",
        "character",
        "Rig {} in Kreate. Select its mesh and run object.add_humanoid_rig. Then loop: \
         check_rig; for the bone that tears most, fix its weights with \
         object.vertex_weights_set (a box round the wrong vertices, exclusive=true) and \
         object.vertex_weights_smooth; check_rig again. Stop when no bone tears. Finally pose \
         an arm and a leg and capture_viewport from the front to show me.",
    ),
    (
        "sculpt_to_reference",
        "Shape a model toward a reference picture",
        "picture",
        "Shape the selected model in Kreate toward the picture at {}. Loop: \
         compare_silhouette with it (front, and right if it shows the side); for the biggest \
         difference, use sculpt.deform_region (mirror_x for a symmetric figure) to add or \
         take away there; analyze_mesh and smooth what got bumpy; compare again. Report the \
         agreement each round and stop when it stops rising.",
    ),
    (
        "dress_scene",
        "Build a scene from free assets",
        "scene",
        "Build {} in Kreate from assets: search_assets for each thing it needs (models, then \
         a floor material, then an HDRI for light), add_asset each, wait for get_scene to show \
         them, size and place them with object.set_dimensions and object.place_on, \
         check_overlaps, and capture_viewport to show me.",
    ),
    (
        "make_model",
        "Make a model from a description, with AI",
        "description",
        "Make {} in Kreate. First search_assets for something close; if nothing fits, \
         generate_model from the description. When it arrives, size it sensibly, \
         analyze_mesh it, and capture_viewport from three sides to show me.",
    ),
];

/// The tools, as `tools/list` describes them.
pub fn tools() -> Value {
    json!([
        {
            "name": "get_scene",
            "description": "What the open document holds: the units (lengths in metres, angles in radians, and the unit the user's screen shows lengths in); every object (name, uid, type, location, rotation as XYZ Euler in radians — as operators take it — and in degrees, scale, visibility, selection, parent; for meshes vertex and face counts, dimensions in metres, world-space bounds and the materials in its slots with their uids; kind, colour and intensity for lights; lens and exposure settings for cameras), every material (uid, how many objects use it, base colour, metallic, roughness, emission, transmission, IOR; colours are sRGB, as material.set_properties takes them), every image (name, uid, size), the editing mode, the view's exposure in stops and whether the meter set it, the 3D cursor, the current frame and the document's file.",
            "inputSchema": { "type": "object", "properties": {} },
            "annotations": { "readOnlyHint": true }
        },
        {
            "name": "list_operators",
            "description": "The operators run_script can call, with their parameters: name, kind (bool, int, float, vec3, color, text, enum with its variants, ints, floats), limits and a description, and whether the operator can run right now. Filter by id, e.g. \"object.\", \"mesh.\", \"material.\", \"light.\". More than 12 matches come back one line each (name: kind, enum choices a|b), without limits and descriptions; filter by an exact id for those.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "filter": { "type": "string", "description": "Only operators whose id contains this." }
                }
            },
            "annotations": { "readOnlyHint": true }
        },
        {
            "name": "run_script",
            "description": "Runs script lines in Kreate, one operator call per line, each its own undo step. The syntax is the one the history records: kreate.ops.<operator id>(<name>=<value>, ...). Values: numbers, True/False, \"text\", (x, y, z) for vectors and colours, an enum by its variant name in quotes or by index, [1, 2, 3] for lists. Parameters left out take their defaults. Examples:\nkreate.ops.object.add_primitive(shape=\"uv_sphere\", size=1.0, location=(0.0, 0.0, 2.0))\nkreate.ops.object.translate(translation=(1.0, 0.0, 0.0))\nkreate.ops.material.set_properties(color=(0.8, 0.1, 0.1), roughness=0.3)\nOperators act on the selection and the active object, as the menus do. A parameter left out of a set_* operator (object.set_transform, material.set_properties, light.set_properties, camera.set_properties, world.set_properties) keeps its current value; elsewhere it takes the default.\nTwo lines are not operators but are written like them: kreate.ops.object.mode_set(mode=\"edit\") (object, edit, sculpt, weight_paint, pose) switches mode for the lines after it, and kreate.ops.image.open(path=\"/abs/file.png\", color=True) reads an image and returns its uid (color=False for normal and roughness maps).\nReturns, for each line, what it became in the history or why it failed, and the objects, materials and images it created (name, uid); then the mode and the active object.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "script": { "type": "string", "description": "One operator call per line." }
                },
                "required": ["script"]
            }
        },
        {
            "name": "render_image",
            "description": "Renders the scene through a scene camera to a PNG file: rendered shading, path traced to the given number of samples, no grid, outlines or other overlays, at the size asked for. Returns a small preview to check it by. Takes as long as the samples do, up to time_limit seconds (at most 110); finished=false means it stopped at the limit with fewer samples. The user's view is put back afterwards.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Where to write the PNG: an absolute path." },
                    "camera": { "type": "string", "description": "A camera object's name. Left out: the active object if it is a camera, else the first camera." },
                    "width": { "type": "integer", "minimum": 16, "maximum": 8192, "default": 1920 },
                    "height": { "type": "integer", "minimum": 16, "maximum": 8192, "default": 1080 },
                    "samples": { "type": "integer", "minimum": 1, "maximum": 512, "default": 128 },
                    "time_limit": { "type": "integer", "minimum": 1, "maximum": 110, "default": 110 }
                },
                "required": ["path"]
            }
        },
        {
            "name": "get_mesh",
            "description": "One mesh object's vertices: index and position, in world space (default) or the object's own; with faces=true also each face's index, vertex indices, normal and center. The indices are the ones mesh.select_set takes. Capped at limit vertices; omitted says how many more there were. selected=true gives only what is selected in edit mode.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "object": { "type": "string", "description": "The object's name. Empty or left out: the active object." },
                    "space": { "type": "string", "enum": ["world", "local"], "default": "world" },
                    "faces": { "type": "boolean", "default": false },
                    "selected": { "type": "boolean", "default": false },
                    "limit": { "type": "integer", "minimum": 1, "default": 2000 }
                }
            },
            "annotations": { "readOnlyHint": true }
        },
        {
            "name": "check_overlaps",
            "description": "Which visible mesh objects run into each other. Pairs whose bounding boxes overlap by more than the tolerance are tested on their actual surfaces: reported with contact \"crossing\" when they share volume — with that volume in cubic metres and its average depth in metres — or \"X inside Y\" when one is wholly within the other, and the box overlap in metres. Parts whose surfaces only meet or scrape (sharing less than a tenth of the tolerance in depth) are listed apart under \"touching\", not as overlaps; a shape and its own parent or child are not reported. exact=false skips the surface test and reports every box overlap, which is quicker and cruder.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tolerance": { "type": "number", "minimum": 0, "default": 0.001, "description": "Metres of box overlap to ignore." },
                    "exact": { "type": "boolean", "default": true }
                }
            },
            "annotations": { "readOnlyHint": true }
        },
        {
            "name": "capture_viewport",
            "description": "A PNG of Kreate's 3D viewport, after the picture has finished refining. By default what the user sees: the shading mode, overlays and view they have chosen. With camera, the view through a scene camera, with its lens and exposure, for checking a composition; the user's view is left as it was.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "camera": {
                        "description": "true for the active camera object or else the first camera, or a camera object's name.",
                        "type": ["boolean", "string"]
                    },
                    "path": { "type": "string", "description": "Also write the PNG here: an absolute path." },
                    "views": {
                        "type": "array",
                        "items": { "type": "string", "enum": ["front", "back", "left", "right", "top", "bottom", "user"] },
                        "description": "Several pictures at once, from these sides (orthographic, framed on what shows) or the user's view. Use front, right and top together to judge a shape in three dimensions."
                    },
                    "only": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Objects' names: show only these, framed, for these pictures. What the user had hidden or shown is left as it was."
                    }
                }
            },
            "annotations": { "readOnlyHint": true }
        },
        {
            "name": "get_context",
            "description": "What the user is working on now, briefly: the editing mode, the active and selected objects (and in edit mode how many vertices, edges and faces are selected), the 3D cursor, where the view looks from and which way, the shading, the frame and the last operation. Read it first when the user says \"this\", \"here\" or \"the selected one\".",
            "inputSchema": { "type": "object", "properties": {} },
            "annotations": { "readOnlyHint": true }
        },
        {
            "name": "check_proportions",
            "description": "Measures a figure standing up Z (every visible mesh, or `objects`) against a realistic adult's proportions: how many heads tall (the chin found where the neck stops being thin), the legs' share of the height (to the crotch), the arm span over the height when the arms are out, and the chest's and hips' width over the height — each with the usual range and whether it is below, within or above. A stylised character departs on purpose; use it to make the departure a choice, and with analyze_mesh's asymmetry for left against right.",
            "inputSchema": {
                "type": "object",
                "properties": { "objects": { "type": "array", "items": { "type": "string" }, "description": "Only these objects; default every visible mesh." } }
            },
            "annotations": { "readOnlyHint": true }
        },
        {
            "name": "compare_silhouette",
            "description": "Compares the model's outline from one side with a reference picture's — concept art, a photo, a turnaround sheet's view — both scaled to the same height and set where they overlap most, so framing does not matter but proportions do. Returns the agreement (shared over either, 0 to 1), each one's width over height, the biggest differences (where the model has more than the picture or less, in the world on a plane through the model, and how big), and a picture of the two: shared grey, the model's extra red, its shortfall blue. The picture should show one figure on a plain background, or with a transparent one. Use it after each change to steer the shape toward the reference.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "reference": { "type": "string", "description": "An absolute path to the picture (PNG, JPEG or WebP)." },
                    "view": { "type": "string", "enum": ["front", "back", "left", "right", "side", "top"], "description": "Which side the picture shows. Default front." },
                    "objects": { "type": "array", "items": { "type": "string" }, "description": "Only these objects; default every visible mesh." }
                },
                "required": ["reference"]
            },
            "annotations": { "readOnlyHint": true }
        },
        {
            "name": "analyze_mesh",
            "description": "Reads a mesh (the active object, or `object`) for what a sculptor looks for, each finding with where it is in the world: face kinds and the typical edge length (the unit the other measures use); open edges (holes) and non-manifold edges; degenerate faces and the share of sliver triangles; bumpy places — vertices that stand off their neighbours differently from how the neighbours stand off theirs, in edge lengths, worst first; and, for a mesh across X = 0, how far each vertex's mirror is from the surface (mean, worst, and where). Use it to find where to smooth, fill or even out, and again after.",
            "inputSchema": {
                "type": "object",
                "properties": { "object": { "type": "string", "description": "The mesh object's name; default the active one." } }
            },
            "annotations": { "readOnlyHint": true }
        },
        {
            "name": "generate_height_map",
            "description": "Draws a seamless greyscale height map of a surface — \"pored skin\", \"reptile scales\", \"oak bark\", \"wrinkled leather\" — with Z-Image on Hugging Face's free GPUs (about ten seconds), white high and black low, and returns its file's path. Then open it as data with kreate.ops.image.open(path=..., color=False) and raise the surface with sculpt.displace_image(image=<uid>, strength=metres, mapping=\"box\" or \"uv\", center/radius for one place). The mesh must have the vertices to carry the detail: subdivide or voxel-remesh first.",
            "inputSchema": {
                "type": "object",
                "properties": { "prompt": { "type": "string", "description": "The surface, in a few words." } },
                "required": ["prompt"]
            }
        },
        {
            "name": "check_rig",
            "description": "Tries a rig: bends each bone of an armature in turn (by `angle` degrees about its own X, default 45) and measures the meshes bound to it, without changing the document. Returns the bones worst first, each with how many vertices it moved, its worst edge's stretch as a ratio of its rest length (1 is perfect; past 1.5 counts as torn — a vertex left behind by a misplaced weight), how many edges tore and where the worst one is (world, at rest), and how many were pinched under a third (the inside of a bend folding: normal in a few, a too-hard joint if many); bones that move nothing; and vertices no bone holds, with a few of their positions. Use it after binding and after each weight fix: fix the worst bone with object.vertex_weights_set (exclusive, by box) and object.vertex_weights_smooth, then check again.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "rig": { "type": "string", "description": "The armature object's name; default the active armature, or the first." },
                    "angle": { "type": "number", "description": "How far to bend each bone, in degrees. Default 45." }
                }
            },
            "annotations": { "readOnlyHint": true }
        },
        {
            "name": "generate_model",
            "description": "Makes a 3D model with AI on Hugging Face's free GPUs and adds it to the scene beside what is there: from `prompt`, words describing one object (they become a picture with Z-Image Turbo, then a model with TRELLIS), or from `image`, the path of a picture of one object on a plain background. Takes a minute or two and runs in the background; check get_scene for it. Without a Hugging Face token in the Assets panel the free GPU time may not be enough, and the panel says so. Kept under Assets ▸ Made by AI.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "prompt": { "type": "string", "description": "One object, e.g. \"a red fire hydrant\"." },
                    "image": { "type": "string", "description": "An absolute path to a PNG or JPEG of one object." }
                }
            }
        },
        {
            "name": "timeline",
            "description": "The clock and what moves. With no arguments, reads it: the current frame, the range, frames per second, whether it is playing, and every animated object with the action it plays (its first and last key) and its NLA strips — a file's other takes, each with an index, a name, where it starts and whether it is muted. `frame` moves the clock there (nothing to undo, as in Blender); `play` starts or stops playback. `object` with `strip` and `mute` switches one strip on (mute false) or off, keeping its other settings — one undo step. Use it with capture_viewport to look at a pose at a given frame.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "frame": { "type": "number", "description": "Move the clock to this frame." },
                    "play": { "type": "boolean", "description": "Start (true) or stop (false) playback." },
                    "object": { "type": "string", "description": "An object's name, to change one of its strips." },
                    "strip": { "type": "integer", "minimum": 0, "description": "Which strip, by its index." },
                    "mute": { "type": "boolean", "description": "Switch the strip off (true) or on (false)." }
                }
            }
        },
        {
            "name": "undo",
            "description": "Undoes the last steps of the document's history.",
            "inputSchema": {
                "type": "object",
                "properties": { "steps": { "type": "integer", "minimum": 1, "default": 1 } }
            }
        },
        {
            "name": "redo",
            "description": "Redoes steps that were undone.",
            "inputSchema": {
                "type": "object",
                "properties": { "steps": { "type": "integer", "minimum": 1, "default": 1 } }
            }
        },
        {
            "name": "open_file",
            "description": "Opens a file in Kreate in place of the current document: a .kreate document, or a scene to import — glTF (.gltf/.glb), FBX, OBJ, STL, PLY, 3MF, OFF, USD (.usd/.usda/.usdc/.usdz), COLLADA (.dae), or a robot's URDF (its links as empties, each joint an empty whose local Z is the joint's axis). Unsaved changes in the current document are lost. To add a scene to the open document instead, use import_file.",
            "inputSchema": {
                "type": "object",
                "properties": { "path": { "type": "string", "description": "An absolute path." } },
                "required": ["path"]
            },
            "annotations": { "destructiveHint": true }
        },
        {
            "name": "import_file",
            "description": "Adds a scene file's contents to the open document, beside what is there, as one undo step: glTF (.gltf/.glb), FBX, OBJ, STL, PLY, 3MF, OFF, USD (.usd/.usda/.usdc/.usdz), COLLADA (.dae) or URDF. What it adds is selected.",
            "inputSchema": {
                "type": "object",
                "properties": { "path": { "type": "string", "description": "An absolute path." } },
                "required": ["path"]
            }
        },
        {
            "name": "search_assets",
            "description": "Searches free asset libraries for models, materials (texture sets) or HDRIs (environment lights): Poly Haven and ambientCG, both CC0; Objaverse, Sketchfab's Creative Commons models (people, characters, props of every kind; searched by Sketchfab, no account needed, only licences allowing changes and commercial use; CC Attribution asks a credit); Poly Pizza, low-poly models, many animated, CC0 or CC-BY (needs the user's own key, entered in the Assets panel); Sketchfab itself (needs the user's API token, entered in the Assets panel; an empty search lists the user's own models and likes, words search every downloadable model, each with its own licence); or the user's own asset folders. Returns up to `limit` matches, each with the id add_asset takes, its licence and whether it is animated. ambientCG has no models; Poly Pizza has only models.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Words to match against names, categories and tags, e.g. \"wooden floor\"." },
                    "kind": { "type": "string", "enum": ["model", "material", "hdri"] },
                    "source": { "type": "string", "enum": ["polyhaven", "ambientcg", "objaverse", "sketchfab", "polypizza", "local"], "description": "Default polyhaven." },
                    "animated": { "type": "boolean", "description": "Only models that move. Default false." },
                    "thumbnails": { "type": "boolean", "description": "Also the pictures of the first six, to choose by looking. Default false." },
                    "limit": { "type": "integer", "description": "Default 20." }
                },
                "required": ["kind"]
            }
        },
        {
            "name": "add_asset",
            "description": "Downloads an asset found by search_assets and adds it: a model into the scene beside what is there, a material onto the active object (made from its texture maps), an HDRI as the world's light. Downloads run in the background; the asset appears when it has arrived — check with get_scene.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string" },
                    "kind": { "type": "string", "enum": ["model", "material", "hdri"] },
                    "source": { "type": "string", "enum": ["polyhaven", "ambientcg", "objaverse", "sketchfab", "polypizza", "local"], "description": "Default polyhaven." },
                    "resolution": { "type": "string", "enum": ["1k", "2k", "4k"], "description": "Texture size. Default 1k." },
                    "location": { "type": "array", "items": { "type": "number" }, "minItems": 3, "maxItems": 3, "description": "A model: where the middle of its base goes, in metres. Default beside what is there." },
                    "height": { "type": "number", "description": "A model: how tall to make it, in metres. Default the file's size." },
                    "wait": { "type": "boolean", "description": "Answer when it has arrived (up to two minutes), with what was added — names, where each starts and its size. Default false: answer at once." }
                },
                "required": ["id", "kind"]
            }
        },
        {
            "name": "save_file",
            "description": "Saves the document as .kreate. Without a path it saves over the file it came from; a path saves it there and makes that its file.",
            "inputSchema": {
                "type": "object",
                "properties": { "path": { "type": "string", "description": "An absolute path ending in .kreate." } }
            }
        },
        {
            "name": "export_file",
            "description": "Writes the scene to another format, chosen by the extension: .glb or .gltf (meshes, materials, lights, cameras, hierarchy, animation), .usda (with its pictures beside it) or .usdz (one file), .urdf (a robot: links, joints with their limits, inertia, visual and collision meshes written beside it as STL), .3mf, .step, .obj or .stl (geometry only). The document keeps its own file.",
            "inputSchema": {
                "type": "object",
                "properties": { "path": { "type": "string", "description": "An absolute path." } },
                "required": ["path"]
            }
        }
    ])
}

/// Answers one message, or nothing for a notification.
pub fn handle(message: &Value, app: &mut dyn App) -> Option<Value> {
    let id = message.get("id").cloned();
    let method = message.get("method").and_then(Value::as_str).unwrap_or("");
    // A notification has no id and gets no answer, whatever it says.
    let id = id?;
    let result = match method {
        "initialize" => {
            let asked = message
                .pointer("/params/protocolVersion")
                .and_then(Value::as_str)
                .unwrap_or(PROTOCOL_VERSION);
            let version = if KNOWN_VERSIONS.contains(&asked) {
                asked
            } else {
                PROTOCOL_VERSION
            };
            Ok(json!({
                "protocolVersion": version,
                "capabilities": {
                    "tools": { "listChanged": false },
                    "prompts": { "listChanged": false }
                },
                "serverInfo": { "name": "kreate", "version": env!("CARGO_PKG_VERSION") },
                "instructions": INSTRUCTIONS,
            }))
        }
        "ping" => Ok(json!({})),
        "prompts/list" => Ok(json!({
            "prompts": PROMPTS
                .iter()
                .map(|(name, description, argument, _)| json!({
                    "name": name,
                    "description": description,
                    "arguments": [{ "name": argument, "required": true }],
                }))
                .collect::<Vec<_>>()
        })),
        "prompts/get" => {
            let name = message
                .pointer("/params/name")
                .and_then(Value::as_str)
                .unwrap_or("");
            match PROMPTS.iter().find(|p| p.0 == name) {
                None => Err((-32602, format!("no prompt called `{name}`"))),
                Some((_, description, argument, text)) => {
                    let given = message
                        .pointer(&format!("/params/arguments/{argument}"))
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    Ok(json!({
                        "description": description,
                        "messages": [{
                            "role": "user",
                            "content": { "type": "text", "text": text.replace("{}", given) }
                        }]
                    }))
                }
            }
        }
        "tools/list" => Ok(json!({ "tools": tools() })),
        "tools/call" => {
            let name = message
                .pointer("/params/name")
                .and_then(Value::as_str)
                .unwrap_or("");
            let args = message
                .pointer("/params/arguments")
                .cloned()
                .unwrap_or_else(|| json!({}));
            call_tool(name, &args, app)
        }
        other => Err((-32601, format!("method not found: {other}"))),
    };
    Some(match result {
        Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
        Err((code, message)) => {
            json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
        }
    })
}

/// A tool call's result. A tool that ran and failed is a result with
/// `isError`, so the model reads why; only a tool that does not exist is a
/// protocol error.
fn call_tool(name: &str, args: &Value, app: &mut dyn App) -> Result<Value, (i64, String)> {
    let pick = |names: &[&str]| -> Value {
        let mut out = serde_json::Map::new();
        for n in names {
            if let Some(v) = args.get(*n) {
                out.insert((*n).to_string(), v.clone());
            }
        }
        Value::Object(out)
    };
    let (method, params) = match name {
        "get_scene" => ("scene", json!({})),
        "list_operators" => ("operators", pick(&["filter"])),
        "run_script" => ("run", pick(&["script"])),
        "capture_viewport" => ("capture", pick(&["camera", "path", "views", "only"])),
        "get_context" => ("context", json!({})),
        "check_overlaps" => ("overlaps", pick(&["tolerance", "exact"])),
        "get_mesh" => (
            "mesh",
            pick(&["object", "space", "faces", "selected", "limit"]),
        ),
        "render_image" => (
            "render",
            pick(&["path", "camera", "width", "height", "samples", "time_limit"]),
        ),
        "timeline" => (
            "timeline",
            pick(&["frame", "play", "object", "strip", "mute"]),
        ),
        "undo" => ("undo", pick(&["steps"])),
        "redo" => ("redo", pick(&["steps"])),
        "open_file" => ("open", pick(&["path"])),
        "import_file" => ("import", pick(&["path"])),
        "search_assets" => (
            "assets",
            pick(&["query", "kind", "source", "animated", "thumbnails", "limit"]),
        ),
        "add_asset" => (
            "add_asset",
            pick(&[
                "id",
                "kind",
                "source",
                "resolution",
                "location",
                "height",
                "wait",
            ]),
        ),
        "save_file" => ("save", pick(&["path"])),
        "export_file" => ("export", pick(&["path"])),
        other => return Err((-32602, format!("no tool called `{other}`"))),
    };
    let answer = app.call(method, params);
    Ok(match answer {
        Err(why) => json!({ "content": [{ "type": "text", "text": why }], "isError": true }),
        // A search with pictures: each found one's picture before the list.
        Ok(mut value) if method == "assets" && value.to_string().contains("png_base64") => {
            let mut content = Vec::new();
            if let Some(found) = value.get_mut("found").and_then(Value::as_array_mut) {
                for item in found.iter_mut() {
                    if let Some(png) = item.as_object_mut().and_then(|o| o.remove("png_base64")) {
                        content
                            .push(json!({ "type": "image", "data": png, "mimeType": "image/png" }));
                        content.push(json!({
                            "type": "text",
                            "text": format!("above: {}", item.get("id").and_then(Value::as_str).unwrap_or("?")),
                        }));
                    }
                }
            }
            content.push(json!({ "type": "text", "text": serde_json::to_string_pretty(&value).unwrap_or_default() }));
            json!({ "content": content })
        }
        // The outlines' picture, and what was measured.
        Ok(mut value) if method == "silhouette" => {
            let data = value
                .get("png_base64")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            if let Some(map) = value.as_object_mut() {
                map.remove("png_base64");
            }
            json!({
                "content": [
                    { "type": "image", "data": data, "mimeType": "image/png" },
                    { "type": "text", "text": serde_json::to_string_pretty(&value).unwrap_or_default() }
                ]
            })
        }
        // Several pictures: each, then what it is of.
        Ok(value) if method == "capture" && value.get("shots").is_some() => {
            let mut content = Vec::new();
            for shot in value["shots"].as_array().into_iter().flatten() {
                content.push(json!({
                    "type": "image",
                    "data": shot.get("png_base64").and_then(Value::as_str).unwrap_or(""),
                    "mimeType": "image/png",
                }));
                content.push(json!({
                    "type": "text",
                    "text": format!(
                        "{} view, {} x {}",
                        shot.get("view").and_then(Value::as_str).unwrap_or("?"),
                        shot.get("width").unwrap_or(&json!(0)),
                        shot.get("height").unwrap_or(&json!(0)),
                    ),
                }));
            }
            json!({ "content": content })
        }
        Ok(value) if method == "capture" || method == "render" => {
            let data = value
                .get("png_base64")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let size = if method == "render" {
                format!(
                    "rendered {} x {} at {} samples{} to {}",
                    value.get("width").unwrap_or(&json!(0)),
                    value.get("height").unwrap_or(&json!(0)),
                    value.get("samples").unwrap_or(&json!(0)),
                    if value.get("finished") == Some(&json!(false)) {
                        " (stopped at the time limit)"
                    } else {
                        ""
                    },
                    value.get("path").and_then(Value::as_str).unwrap_or("?"),
                )
            } else {
                format!(
                    "viewport {} x {}",
                    value.get("width").unwrap_or(&json!(0)),
                    value.get("height").unwrap_or(&json!(0))
                )
            };
            json!({
                "content": [
                    { "type": "image", "data": data, "mimeType": "image/png" },
                    { "type": "text", "text": size }
                ]
            })
        }
        Ok(value) => {
            // A script whose lines failed ran, and says so: the model needs
            // to see which line and why, as an error it will not miss.
            let failed = method == "run"
                && value
                    .get("lines")
                    .and_then(Value::as_array)
                    .is_some_and(|lines| lines.iter().any(|l| l.get("ok") == Some(&json!(false))));
            let text = serde_json::to_string_pretty(&value).unwrap_or_default();
            json!({ "content": [{ "type": "text", "text": text }], "isError": failed })
        }
    })
}

/// The running Kreate, reached through the port and token it wrote down.
pub struct Socket {
    stream: Option<(
        std::io::BufReader<std::net::TcpStream>,
        std::net::TcpStream,
        String,
    )>,
}

impl Default for Socket {
    fn default() -> Self {
        Self::new()
    }
}

impl Socket {
    pub fn new() -> Self {
        Self { stream: None }
    }

    /// Where Kreate writes its port and token: `bridge.json` in its settings
    /// folder, which is `%APPDATA%\Kreate` on Windows and
    /// `$XDG_CONFIG_HOME/kreate` or `~/.config/kreate` elsewhere — the same
    /// rule as the application's `prefs::config_dir`.
    pub fn info_path() -> Option<std::path::PathBuf> {
        use std::path::Path;
        let dir = if cfg!(windows) {
            std::env::var_os("APPDATA").map(|base| Path::new(&base).join("Kreate"))
        } else {
            std::env::var_os("XDG_CONFIG_HOME")
                .map(|base| Path::new(&base).join("kreate"))
                .or_else(|| {
                    std::env::var_os("HOME").map(|home| Path::new(&home).join(".config/kreate"))
                })
        };
        dir.map(|d| d.join("bridge.json"))
    }

    fn connect(&mut self) -> Result<(), String> {
        const NOT_RUNNING: &str = "Kreate is not running, or was started before the connector existed. Start Kreate and try again.";
        let path = Self::info_path().ok_or("cannot find Kreate's settings folder")?;
        let text = std::fs::read_to_string(&path).map_err(|_| NOT_RUNNING.to_string())?;
        let info: Value =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let port = info
            .get("port")
            .and_then(Value::as_u64)
            .ok_or("no port in bridge.json")? as u16;
        let token = info
            .get("token")
            .and_then(Value::as_str)
            .ok_or("no token in bridge.json")?
            .to_string();
        let stream = std::net::TcpStream::connect(("127.0.0.1", port))
            .map_err(|_| NOT_RUNNING.to_string())?;
        let reader = std::io::BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
        self.stream = Some((reader, stream, token));
        Ok(())
    }

    fn exchange(&mut self, method: &str, params: &Value) -> Result<Value, String> {
        use std::io::{BufRead, Write};
        let (reader, writer, token) = self.stream.as_mut().ok_or("not connected")?;
        let request = json!({ "token": token, "method": method, "params": params });
        writeln!(writer, "{request}").map_err(|e| e.to_string())?;
        writer.flush().map_err(|e| e.to_string())?;
        let mut line = String::new();
        if reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            return Err("Kreate closed the connection".into());
        }
        serde_json::from_str(&line)
            .map_err(|e| format!("Kreate answered something unreadable: {e}"))
    }
}

impl App for Socket {
    fn call(&mut self, method: &str, params: Value) -> Result<Value, String> {
        // Connected on first use and again whenever the connection broke:
        // Kreate may have been restarted, on a new port, since the last call.
        let mut answer = Err(String::new());
        for _ in 0..2 {
            if self.stream.is_none() {
                self.connect()?;
            }
            answer = self.exchange(method, &params);
            if answer.is_ok() {
                break;
            }
            self.stream = None;
        }
        let answer = answer?;
        match (answer.get("ok"), answer.get("error")) {
            (Some(ok), _) => Ok(ok.clone()),
            (_, Some(err)) => Err(err
                .as_str()
                .unwrap_or("Kreate reported an error")
                .to_string()),
            _ => Err("Kreate's answer had neither a result nor an error".into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Records what it was asked and answers from a script.
    struct Fake {
        asked: Vec<(String, Value)>,
        answer: Result<Value, String>,
    }

    impl App for Fake {
        fn call(&mut self, method: &str, params: Value) -> Result<Value, String> {
            self.asked.push((method.to_string(), params));
            self.answer.clone()
        }
    }

    fn fake(answer: Result<Value, String>) -> Fake {
        Fake {
            asked: Vec::new(),
            answer,
        }
    }

    #[test]
    fn initialize_answers_in_the_clients_version_when_it_knows_it() {
        let mut app = fake(Ok(json!({})));
        let reply = handle(
            &json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26"}}),
            &mut app,
        )
        .expect("a request is answered");
        assert_eq!(reply["result"]["protocolVersion"], "2025-03-26");
        assert_eq!(reply["result"]["serverInfo"]["name"], "kreate");
        assert!(reply["result"]["capabilities"]["tools"].is_object());
        let newer = handle(
            &json!({"jsonrpc":"2.0","id":2,"method":"initialize","params":{"protocolVersion":"2099-01-01"}}),
            &mut app,
        )
        .unwrap();
        assert_eq!(newer["result"]["protocolVersion"], PROTOCOL_VERSION);
    }

    #[test]
    fn a_notification_gets_no_answer() {
        let mut app = fake(Ok(json!({})));
        assert!(handle(
            &json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
            &mut app
        )
        .is_none());
    }

    #[test]
    fn every_tool_has_a_name_a_description_and_an_object_schema() {
        let tools = tools();
        let tools = tools.as_array().expect("a list");
        assert!(tools.len() >= 9);
        for t in tools {
            assert!(t["name"].is_string());
            assert!(t["description"].as_str().is_some_and(|d| !d.is_empty()));
            assert_eq!(t["inputSchema"]["type"], "object");
        }
    }

    #[test]
    fn a_tool_call_reaches_the_app_with_its_arguments_and_returns_text() {
        let mut app = fake(Ok(json!({"lines":[{"line":"x","ok":true}]})));
        let reply = handle(
            &json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"run_script","arguments":{"script":"kreate.ops.object.delete()","stray":1}}}),
            &mut app,
        )
        .unwrap();
        assert_eq!(app.asked[0].0, "run");
        assert_eq!(
            app.asked[0].1,
            json!({"script":"kreate.ops.object.delete()"})
        );
        assert_eq!(reply["result"]["content"][0]["type"], "text");
        assert_eq!(reply["result"]["isError"], false);
    }

    #[test]
    fn a_failed_script_line_is_an_error_result_the_model_reads() {
        let mut app = fake(Ok(
            json!({"lines":[{"line":"x","ok":false,"error":"no operator"}]}),
        ));
        let reply = handle(
            &json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"run_script","arguments":{"script":"x"}}}),
            &mut app,
        )
        .unwrap();
        assert_eq!(reply["result"]["isError"], true);
    }

    #[test]
    fn kreate_not_running_is_a_tool_error_and_not_a_protocol_error() {
        let mut app = fake(Err("Kreate is not running".into()));
        let reply = handle(
            &json!({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"get_scene"}}),
            &mut app,
        )
        .unwrap();
        assert!(reply.get("error").is_none());
        assert_eq!(reply["result"]["isError"], true);
        assert!(reply["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("not running"));
    }

    #[test]
    fn a_capture_comes_back_as_an_image() {
        let mut app = fake(Ok(json!({"png_base64":"iVBORw0=","width":4,"height":3})));
        let reply = handle(
            &json!({"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"capture_viewport","arguments":{}}}),
            &mut app,
        )
        .unwrap();
        let first = &reply["result"]["content"][0];
        assert_eq!(first["type"], "image");
        assert_eq!(first["mimeType"], "image/png");
        assert_eq!(first["data"], "iVBORw0=");
    }

    #[test]
    fn an_unknown_tool_and_an_unknown_method_are_protocol_errors() {
        let mut app = fake(Ok(json!({})));
        let tool = handle(
            &json!({"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"nope"}}),
            &mut app,
        )
        .unwrap();
        assert_eq!(tool["error"]["code"], -32602);
        let method = handle(
            &json!({"jsonrpc":"2.0","id":8,"method":"resources/list"}),
            &mut app,
        )
        .unwrap();
        assert_eq!(method["error"]["code"], -32601);
        assert!(app.asked.is_empty());
    }

    #[test]
    fn workflows_are_offered_as_prompts_with_the_argument_filled_in() {
        let mut app = fake(Ok(json!({})));
        let list = handle(
            &json!({"jsonrpc":"2.0","id":1,"method":"prompts/list"}),
            &mut app,
        )
        .unwrap();
        let names: Vec<&str> = list["result"]["prompts"]
            .as_array()
            .expect("prompts")
            .iter()
            .filter_map(|p| p["name"].as_str())
            .collect();
        assert!(names.contains(&"rig_character") && names.contains(&"sculpt_to_reference"));
        let got = handle(
            &json!({"jsonrpc":"2.0","id":2,"method":"prompts/get",
                    "params":{"name":"rig_character","arguments":{"character":"the knight"}}}),
            &mut app,
        )
        .unwrap();
        let text = got["result"]["messages"][0]["content"]["text"]
            .as_str()
            .expect("text");
        assert!(text.starts_with("Rig the knight"), "{text}");
        assert!(app.asked.is_empty(), "nothing asked of Kreate");
    }
}
