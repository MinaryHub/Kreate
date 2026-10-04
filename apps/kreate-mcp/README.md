# kreate-mcp

An MCP server that lets Claude work in the Kreate you have open: read the
scene, run operators, look at the viewport, and open, save or export files.
Every edit goes through Kreate's own operators and can be undone in Kreate.

## Build

```
cargo build --release -p kreate -p kreate-mcp
```

This gives `target/release/kreate.exe` and `target/release/kreate-mcp.exe`.

## Connect

Start Kreate first. While it runs, it accepts connections on the loopback
address only. It writes the port and a random token to `bridge.json` in its
settings folder (`%APPDATA%\Kreate` on Windows), and the connector reads them
from there.

**Claude Code**

```
claude mcp add kreate -- C:\path\to\Kreate\target\release\kreate-mcp.exe
```

**Claude Desktop**: add this to `claude_desktop_config.json`, then restart
Claude Desktop:

```json
{
  "mcpServers": {
    "kreate": {
      "command": "C:\\path\\to\\Kreate\\target\\release\\kreate-mcp.exe"
    }
  }
}
```

## Tools

| Tool | What it does |
|---|---|
| `get_scene` | Objects, materials, lights, mode, 3D cursor, frame, file |
| `list_operators` | Every operator a script can call, with its parameters |
| `run_script` | Runs `kreate.ops.<operator>(name=value, ...)` lines, one undo step each |
| `capture_viewport` | The viewport as a PNG image |
| `undo` / `redo` | Steps through the history |
| `open_file` | Opens a .kreate file, or imports .gltf, .glb, .obj or .stl |
| `save_file` | Saves as .kreate |
| `export_file` | Writes .glb, .gltf, .obj or .stl |

If Kreate is not running, every tool returns an error that says so. Start
Kreate and call the tool again; the connector reconnects by itself.
