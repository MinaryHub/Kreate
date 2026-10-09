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
Units are metres, Z is up. Use capture_viewport to look at the result, or camera=true to see \
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
material slot instead of by index. data.purge removes unused meshes, materials and images.";

/// The tools, as `tools/list` describes them.
pub fn tools() -> Value {
    json!([
        {
            "name": "get_scene",
            "description": "What the open document holds: every object (name, uid, type, location, rotation in degrees as XYZ Euler, scale, visibility, selection, parent; for meshes vertex and face counts, dimensions in metres, world-space bounds and the materials in its slots with their uids; kind, colour and intensity for lights; lens and exposure settings for cameras), every material (uid, how many objects use it, base colour, metallic, roughness, emission, transmission, IOR; colours are sRGB, as material.set_properties takes them), every image (name, uid, size), the editing mode, the view's exposure in stops and whether the meter set it, the 3D cursor, the current frame and the document's file.",
            "inputSchema": { "type": "object", "properties": {} },
            "annotations": { "readOnlyHint": true }
        },
        {
            "name": "list_operators",
            "description": "The operators run_script can call, with their parameters: name, kind (bool, int, float, vec3, color, text, enum with its variants, ints, floats), limits and a description, and whether the operator can run right now. Filter by id, e.g. \"object.\", \"mesh.\", \"material.\", \"light.\".",
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
            "description": "Which visible mesh objects run into each other. Pairs whose bounding boxes overlap by more than the tolerance are tested on their actual surfaces: reported with contact \"crossing\" when the surfaces pass through each other, or \"X inside Y\" when one is wholly within the other, and the box overlap in metres. Touching (a vase standing on a table, a chair against a wall) is not reported, nor are a shape and its own parent or child. exact=false skips the surface test and reports every box overlap, which is quicker and cruder.",
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
                    "path": { "type": "string", "description": "Also write the PNG here: an absolute path." }
                }
            },
            "annotations": { "readOnlyHint": true }
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
            "description": "Opens a file in Kreate in place of the current document: a .kreate document, or a scene to import — glTF (.gltf/.glb), FBX, OBJ, STL, PLY, 3MF or OFF. Unsaved changes in the current document are lost. To add a scene to the open document instead, use import_file.",
            "inputSchema": {
                "type": "object",
                "properties": { "path": { "type": "string", "description": "An absolute path." } },
                "required": ["path"]
            },
            "annotations": { "destructiveHint": true }
        },
        {
            "name": "import_file",
            "description": "Adds a scene file's contents to the open document, beside what is there, as one undo step: glTF (.gltf/.glb), FBX, OBJ, STL, PLY, 3MF or OFF. What it adds is selected.",
            "inputSchema": {
                "type": "object",
                "properties": { "path": { "type": "string", "description": "An absolute path." } },
                "required": ["path"]
            }
        },
        {
            "name": "search_assets",
            "description": "Searches free asset libraries for models, materials (texture sets) or HDRIs (environment lights): Poly Haven and ambientCG, both CC0, or the user's own asset folders. Returns up to `limit` matches, each with the id add_asset takes. ambientCG has no models.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Words to match against names, categories and tags, e.g. \"wooden floor\"." },
                    "kind": { "type": "string", "enum": ["model", "material", "hdri"] },
                    "source": { "type": "string", "enum": ["polyhaven", "ambientcg", "local"], "description": "Default polyhaven." },
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
                    "source": { "type": "string", "enum": ["polyhaven", "ambientcg", "local"], "description": "Default polyhaven." },
                    "resolution": { "type": "string", "enum": ["1k", "2k", "4k"], "description": "Texture size. Default 1k." }
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
            "description": "Writes the scene to another format, chosen by the extension: .glb or .gltf (meshes, materials, lights, cameras, hierarchy, animation), .obj or .stl (geometry only). The document keeps its own file.",
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
                "capabilities": { "tools": { "listChanged": false } },
                "serverInfo": { "name": "kreate", "version": env!("CARGO_PKG_VERSION") },
                "instructions": INSTRUCTIONS,
            }))
        }
        "ping" => Ok(json!({})),
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
        "capture_viewport" => ("capture", pick(&["camera", "path"])),
        "check_overlaps" => ("overlaps", pick(&["tolerance", "exact"])),
        "get_mesh" => (
            "mesh",
            pick(&["object", "space", "faces", "selected", "limit"]),
        ),
        "render_image" => (
            "render",
            pick(&["path", "camera", "width", "height", "samples", "time_limit"]),
        ),
        "undo" => ("undo", pick(&["steps"])),
        "redo" => ("redo", pick(&["steps"])),
        "open_file" => ("open", pick(&["path"])),
        "import_file" => ("import", pick(&["path"])),
        "search_assets" => ("assets", pick(&["query", "kind", "source", "limit"])),
        "add_asset" => ("add_asset", pick(&["id", "kind", "source", "resolution"])),
        "save_file" => ("save", pick(&["path"])),
        "export_file" => ("export", pick(&["path"])),
        other => return Err((-32602, format!("no tool called `{other}`"))),
    };
    let answer = app.call(method, params);
    Ok(match answer {
        Err(why) => json!({ "content": [{ "type": "text", "text": why }], "isError": true }),
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
}
