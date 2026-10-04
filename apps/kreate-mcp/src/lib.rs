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
Units are metres, Z is up. Use capture_viewport to look at the result.";

/// The tools, as `tools/list` describes them.
pub fn tools() -> Value {
    json!([
        {
            "name": "get_scene",
            "description": "What the open document holds: every object (name, type, location, rotation in degrees as XYZ Euler, scale, visibility, selection, materials; vertex and face counts for meshes; kind, colour and intensity for lights), every material (base colour, metallic, roughness, emission, transmission, IOR), the editing mode, the 3D cursor, the current frame and the document's file.",
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
            "description": "Runs script lines in Kreate, one operator call per line, each its own undo step. The syntax is the one the history records: kreate.ops.<operator id>(<name>=<value>, ...). Values: numbers, True/False, \"text\", (x, y, z) for vectors and colours, an enum by its variant name in quotes or by index, [1, 2, 3] for lists. Parameters left out take their defaults. Examples:\nkreate.ops.object.add_primitive(shape=\"uv_sphere\", size=1.0, location=(0.0, 0.0, 2.0))\nkreate.ops.object.translate(translation=(1.0, 0.0, 0.0))\nkreate.ops.material.set_properties(color=(0.8, 0.1, 0.1), roughness=0.3)\nOperators act on the selection and the active object, as the menus do. Returns, for each line, what it became in the history or why it failed.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "script": { "type": "string", "description": "One operator call per line." }
                },
                "required": ["script"]
            }
        },
        {
            "name": "capture_viewport",
            "description": "A PNG of Kreate's 3D viewport as it is now, after the picture has finished refining. Shows what the user sees: the shading mode, overlays and camera they have chosen.",
            "inputSchema": { "type": "object", "properties": {} },
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
            "description": "Opens a file in Kreate in place of the current document: a .kreate document, or a glTF (.gltf/.glb), OBJ or STL scene to import. Unsaved changes in the current document are lost.",
            "inputSchema": {
                "type": "object",
                "properties": { "path": { "type": "string", "description": "An absolute path." } },
                "required": ["path"]
            },
            "annotations": { "destructiveHint": true }
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
            let version = if KNOWN_VERSIONS.contains(&asked) { asked } else { PROTOCOL_VERSION };
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
            let name = message.pointer("/params/name").and_then(Value::as_str).unwrap_or("");
            let args = message.pointer("/params/arguments").cloned().unwrap_or_else(|| json!({}));
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
        "capture_viewport" => ("capture", json!({})),
        "undo" => ("undo", pick(&["steps"])),
        "redo" => ("redo", pick(&["steps"])),
        "open_file" => ("open", pick(&["path"])),
        "save_file" => ("save", pick(&["path"])),
        "export_file" => ("export", pick(&["path"])),
        other => return Err((-32602, format!("no tool called `{other}`"))),
    };
    let answer = app.call(method, params);
    Ok(match answer {
        Err(why) => json!({ "content": [{ "type": "text", "text": why }], "isError": true }),
        Ok(value) if method == "capture" => {
            let data = value.get("png_base64").and_then(Value::as_str).unwrap_or("").to_string();
            let size = format!(
                "viewport {} x {}",
                value.get("width").unwrap_or(&json!(0)),
                value.get("height").unwrap_or(&json!(0))
            );
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
    stream: Option<(std::io::BufReader<std::net::TcpStream>, std::net::TcpStream, String)>,
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
                .or_else(|| std::env::var_os("HOME").map(|home| Path::new(&home).join(".config/kreate")))
        };
        dir.map(|d| d.join("bridge.json"))
    }

    fn connect(&mut self) -> Result<(), String> {
        const NOT_RUNNING: &str = "Kreate is not running, or was started before the connector existed. Start Kreate and try again.";
        let path = Self::info_path().ok_or("cannot find Kreate's settings folder")?;
        let text = std::fs::read_to_string(&path).map_err(|_| NOT_RUNNING.to_string())?;
        let info: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let port = info.get("port").and_then(Value::as_u64).ok_or("no port in bridge.json")? as u16;
        let token = info.get("token").and_then(Value::as_str).ok_or("no token in bridge.json")?.to_string();
        let stream = std::net::TcpStream::connect(("127.0.0.1", port)).map_err(|_| NOT_RUNNING.to_string())?;
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
        serde_json::from_str(&line).map_err(|e| format!("Kreate answered something unreadable: {e}"))
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
            (_, Some(err)) => Err(err.as_str().unwrap_or("Kreate reported an error").to_string()),
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
        Fake { asked: Vec::new(), answer }
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
        assert!(handle(&json!({"jsonrpc":"2.0","method":"notifications/initialized"}), &mut app).is_none());
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
        assert_eq!(app.asked[0].1, json!({"script":"kreate.ops.object.delete()"}));
        assert_eq!(reply["result"]["content"][0]["type"], "text");
        assert_eq!(reply["result"]["isError"], false);
    }

    #[test]
    fn a_failed_script_line_is_an_error_result_the_model_reads() {
        let mut app = fake(Ok(json!({"lines":[{"line":"x","ok":false,"error":"no operator"}]})));
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
        assert!(reply["result"]["content"][0]["text"].as_str().unwrap().contains("not running"));
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
        let method = handle(&json!({"jsonrpc":"2.0","id":8,"method":"resources/list"}), &mut app).unwrap();
        assert_eq!(method["error"]["code"], -32601);
        assert!(app.asked.is_empty());
    }
}
