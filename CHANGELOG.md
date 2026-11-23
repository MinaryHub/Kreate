# Changelog

Every release of Kreate, newest first. The format is
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the numbers follow
[Semantic Versioning](https://semver.org/) — see [RELEASING.md](RELEASING.md)
for what each part of a number promises and how a release is cut.

Write what changed under **Unreleased** as it lands, in the words a user would
use; `scripts/release.sh` turns that section into the next release.

## [Unreleased]

## [0.15.1] - 2026-11-23

### Fixed

- The Android build: the shell's STEP export needs kreate-occt, which the
  Android library did not depend on, so 0.15.0's release stopped there and
  published nothing.

## [0.15.0] - 2026-11-23

### Added

- Assistants change a design as a designer would. get_scene now gives each
  body's features (kind, amount and what it means, operation, whether it
  is bound to a parameter, and the feature.set line to change it), each
  drawing's degrees of freedom, each part's mate, and the design's
  parameters, bindings and measurements — so "make the hole 2 mm deeper"
  edits the feature instead of building the part again.
- run_script answers with what changed — objects added and removed, and
  for each one changed its location, size, faces or features from what to
  what — and takes atomic (a failed line undoes the lines before it) and
  dry_run (run, report, and undo everything).
- Scripts take names, loops and arithmetic — `d = 6mm`, `for i in
  range(6):`, `location=(i * d, r * cos(a), 0)` — unrolled into one undo
  step a line, and `$last` names the object made last.
- list_operators finds operators by words in English or Korean ("fillet",
  "구멍", "mirror"), best first, each with an example line; a misspelt
  operator or parameter is answered with the near ones and what it takes.
- The chat can ask before the assistant changes anything: each change is
  tried and shown with what it would add, remove or change, and made only
  when you press Apply.
- The assistant hears what you did yourself since it last looked — the
  steps and what they changed — with its next answer, and each chat
  message carries the mode, the selection and the file.
- mesh.select_named picks faces and edges by what they are: the top,
  bottom, front, back, left or right faces, the largest face, the open,
  sharp or longest edges.
- On-canvas 3D Move, Rotate and Scale gizmos in the viewport, and
  multi-object relative transform editing in the Transform panel (hold Alt
  or check Apply to all selected).
- Korean initial-consonant (초성) search and 2-beolsik Korean↔English key
  matching in the F3 Command Palette and Keymap Editor, Korean IME parity
  for single-key shortcuts across the Node Editor, Dope Sheet and Graph
  Editor, and automatic OS Korean locale detection with Preferences
  language and translation-scope selectors.
- A CAD command bar (`/` or `:`) in 2D sketch and drawing modes with
  drafting aliases (`REC`, `CHA`, `TR`, `EX`, `PL`, `EL`, `POL`, `F`, `O`,
  `MI`, `DIM`), dynamic `@dx,dy`, `#x,y` and `distance<angle` coordinate
  entry, and sweep-and-prune intersection snapping on large drawings.
- A **3D Preview** toggle on pending assistant changes (preserving your
  redo history), a one-click `@Selection` context chip in chat, script
  conditionals (`if`/`elif`/`else`), `f"..."` formatting, tuple indexing,
  live scene queries (`bbox`, `size`, `pos`, `param`), and a multi-line
  console (`Shift+Enter`).
- **Save Selection as Asset…** in the Asset Browser, workspace Duplicate,
  Rename and Delete in the workspace switcher, `kreatec` conversion and
  Git `diff`/`textconv` across all 18 supported 3D/CAD/2D formats
  (including STEP, 3MF and DXF drawings), and a built-in native HTTPS
  client with no external `curl` dependency.

### Changed

- Help ▸ Report an Issue and Help ▸ Ask a Question replace the contact
  form's e-mail: what you write opens on GitHub, filled in — a bug report
  issue, or a discussion in Q&A or Ideas — with the version, platform and
  graphics card if you like, and you post it there signed in. Answers come
  back to you on GitHub. Nothing is sent from Kreate itself.
- Easier to read and to find your way: text is larger (14 pt, nothing under
  11), and Preferences ▸ Interface Size scales the whole interface from 90%
  to 175%. The mode tabs are full-size, the open one filled in its mode's
  colour, and outside object mode the viewport is framed in that colour.
  Fields and dialog buttons are outlined at 3:1 so a value field no longer
  looks like a label; menus and toolbars stay flat.
- Korean names for every operator parameter (위치, 회전, 크기…) where the
  panels showed the English field name.
- The toolbar's Add and drawing tools are drawn in the text colour: orange
  is only for what is selected or in hand, so the sketch tool held shows.
- Toolbar groups that do not fit go behind » instead of being cut off, and a
  header row that is clipped shows ‹ › to scroll it.
- The status bar says what the mouse does in the mode (click, Shift+click,
  drag); a sketch tool's instruction is in its banner at the top of the
  view rather than also in the corner; the properties panel names its tab
  and shows a scroll bar and a fade when there is more below; the history
  strip says what it is for while it is empty.
- Faster `.kreate` opening and saving via memory-mapped zero-copy slicing
  and multicore mesh encoding/decoding, multicore normal calculation on
  large meshes, longest-prefix caching in the modifier stack, and shared
  BVH + parallel hidden-line removal for 4-view drawing sheets.
- Lower viewport VRAM (~65% less in Studio mode) with lazy render targets
  and immediate Path Tracer BVH release on leaving Rendered mode,
  1,024-vertex dirty-chunk GPU uploads during sculpting, in-place material
  uniform updates in the path tracer without rebuilding the BVH, base-mesh
  reuse during modal topology drags, and compressed sparse vertex deltas in
  live collaboration.

### Fixed

- Ribbon tool buttons activate immediately on click, and the Adjust Last
  Operation panel stays open while orbiting (`MMB`/`Alt`) and reopens with
  `F9`.
- `File ▸ Link` and link refresh now carry and update mesh materials,
  shader-graph textures and material overrides alongside geometry.
- Undo snapshots count uniquely held mesh bytes accurately so the 2 GiB
  history budget holds on sculpted million-face scenes, and recovered crash
  backups remain on disk until saving succeeds.
- Analytic OCCT B-rep feature bodies write exact NURBS surfaces to STEP
  when the exact kernel is enabled, and Android S-Pen stylus pressure
  reaches sculpt and paint strokes.

## [0.14.0] - 2026-11-17

### Added

- A drawing sheet can have a **section**, A-A, in the isometric view's
  place: the parts cut through their middle front to back, the cut
  hatched at 45° with its holes left clear, the cutting line marked A–A
  on the top view (File ▸ Drawing Sheet ▸ Section A-A).
- The path tracer reads textures at the detail they have — to 2048
  pixels a side, within a memory budget — filtered and with mips, where
  it shrank every one to 512 pixels, nearest texel: labels, wood grain
  and normal maps render as sharp as the viewport shows them.
- The rendered view, path traced to all its samples and still, is put
  through Open Image Denoise once where it is installed, as a render is.
- Patterns as Fusion's: a rectangular pattern in two directions — a grid
  (`feature.pattern(count2=…, step2=…)`) — and a circular one spread over
  an angle, end to end, about a construction axis
  (`feature.circular(angle=…, around="Axis")`).
- **Holes at sketch points**, as Fusion's Hole (Mechanical ▸ Holes ▸
  Holes at Sketch Points, with the sketch selected): a hole at every
  point of the sketch — at every circle's centre where it has no points
  — plain, counterbored, countersunk or tapped, blind or through all, as
  one step on the timeline that follows the points (`feature.hole`).
- **Construction planes and axes** (Mechanical ▸ Construct), as Fusion's
  Construct menu has them: an offset plane off a face (its offset and
  angle set in the Adjust panel after), a midplane between two faces, a
  plane through three points, an axis through two, and an offset plane
  from the ground. They are drawn in the view, and the sketch plane menu
  offers each plane to draw on (`datum.plane`, `datum.plane_points`,
  `datum.axis`, `draft.new(plane="object")`).
- **Fillet and Chamfer on the edges you pick**, as Fusion and SolidWorks
  do (Mechanical ▸ Fillet / Chamfer): click edges of a body — a click on
  one takes the edges it runs smoothly into, a round edge whole — type
  the radius, Enter. Click a picked edge again to let it go. The fillet
  is a step on the timeline and stays on its edges through edits that
  leave them where they were (`feature.round(edges=…)`).
- The timeline marks a feature that makes nothing — its sketch gone, its
  region or its edges gone, nothing before a pattern to repeat — in red
  with a ⚠, and says why on hover.
- A **rollback marker**: right-click a feature ▸ Roll Back to Here builds
  the timeline only that far; a feature made meanwhile goes in at the
  marker; Roll to End puts the rest back.
- Features can be renamed: right-click ▸ Rename.

## [0.13.0] - 2026-11-17

### Added

- The rest of Fusion's assembly, under Mechanical ▸ Assembly Joints:
  **Motion Link** (one joint follows another by a ratio — gears, a
  screw and its slide); **Contact** on a joint (what it moves stops at
  touching the part in its way, along the whole of its travel);
  **Exploded View** (the parts moved out from their middle, undone with
  Undo); and **New Component** (the selected parts gathered under one
  object, moved and hidden as one).
- **Assembly joints as Fusion makes them** (Mechanical ▸ Assembly Joints ▸
  Joint: …): click where on the moving part the joint is, then where on
  the other part it joins — a face is caught at its middle, or at a
  hole's centre when the click is by the hole — and the part is brought
  there, faces together, and joined. Seven motions: Rigid, Revolute,
  Slider, and the new Cylindrical (turns and slides along the axis),
  Pin-Slot (turns, and slides across), Planar (slides in the plane and
  turns) and Ball (turns every way) — `assembly.joint` for scripts.
- The joints panel has a slider for every way a joint moves, and a ▶
  that plays a joint through its travel and puts it back, as Fusion's
  Animate Joint does. `robot.set_joint` takes `index` for a joint's
  second or third way of moving.
- Extrude has what Fusion's and SolidWorks's have under the distance:
  **Through all** of the part in its way, however thick it is or becomes;
  a **second side** with its own distance; a **start offset** from the
  sketch's plane; a **taper** for a mould's draft; and **Intersect** beside
  Join and Cut, which keeps only what the part and the extrude share
  (`feature.extrude(through=…, second=…, offset=…, taper=…,
  operation="intersect")`; a revolve can intersect too).
- The chat panel has an **Effort** list beside the model: how hard the
  assistant thinks before it answers — Low to Max for Claude, Low to High
  for ChatGPT — kept for each assistant. The assistant picked is the one
  it starts with next time.
- The chat says how long an answer has been coming and which tool it is
  waiting on, and under each answer how long it took and the tokens it
  read and wrote — with their cost, when an API key paid for them.
- Up in the chat's empty box brings back what was last said; an empty
  conversation offers three things to ask, a click each.
- A dimension takes a length with its unit — `20mm`, `1.5 in`, `2 cm` —
  in whatever unit the scene is in. Tab goes to the next number while one
  is being typed in a sketch.
- A sketch says whether it is fully defined, by its prompt: or how many of
  its curves can still move.
- Two parallel sides picked with the Dimension tool are dimensioned as the
  distance between them (level or upright), not as an angle of nought.
- The physical library has paint: car paint under a clear coat in five
  colours, powder coat in two; and polished and anodised aluminium, and
  gloss black and white plastic.
- A render through the connector (`render_image`) is cleaned by Open Image
  Denoise where it is installed, as a render from the Render menu is.
- The material preview looks like SolidWorks: a part's edges — where its
  faces turn, not the facets of a curve — drawn in dark ink, and a pale
  steel-blue sweep to white behind it in the default studio.
- The wireframe overlay starts checked: every edge of a mesh over the
  preview, until it is unchecked in Overlays.
- Drawing or dragging in a sketch catches the sketch's own lines and
  curves, as it catches its points, with snapping off: a circle's centre
  dragged to another circle's rim lands on the rim.
- Faces pushed into a body (E, inward) cut what they sweep through when
  they meet something on the way — the body's far side, a hole, another
  pocket: pushed deeper than the body is thick, a pocket becomes a hole
  through it. A push that meets nothing is as before.
- The Extrude tool shows what it will make: the regions picked as a
  solid, outlined at the sketch and at the distance — in red when it will
  cut a part, in the accent when it joins one or stands alone — and turns
  the view a three-quarter way off the sketch to show its depth; the view
  can be orbited while it is open.
- A sketch's first point can be typed from a point: with a drawing tool
  in hand, over a face's corner or any point of the sketch, type how far
  across and up ("20, 10") and Enter — the line, rectangle or circle
  starts there. Over nothing, it is measured from the sketch's origin.
- G in a sketch takes up what is selected: it follows the pointer, caught
  by what drawing is caught by, until a click sets it down; Escape or the
  right button puts it back.
- Moving objects with snapping on, sketches are caught and catch: a
  sketch's ends, centres and lines are there for anything to land on, and
  a sketch being moved lands by its own nearest end.
- A sketch offset shows what it will make as the pointer moves, with the
  distance written by it; a distance typed and Enter makes it there, on
  the pointer's side, without a click.
- Pattern along a path in a sketch (Pattern ▸ Along Path): pick the
  curve — or curves joined end to end — and the copies follow it, spread
  over the whole of it or a spacing apart, turned with the path or not.
  `draft.pattern(kind="path", path=[…])` for scripts.

### Changed

- Renders and the rendered view look as a product shot does. The default
  studio is a white cove under bright lamps — pale below, darker above, the
  softboxes several times either — so chrome and brass show shape and not a
  flat grey; the part stands on light paper. The path tracer reflects the
  environment at its own sharpness (it read a 128-pixel copy), aims at the
  lamps and the environment at every surface (it took one of them by
  turns), and spreads each pixel's samples as a bell, not a box. A render
  is made at High quality by default (256 samples, 6 bounces), gets the
  samples it asks for past 512, and goes onto an eight-bit screen with a
  step of noise, so a backdrop's sweep no longer bands.
- A sketch on a face holds the face's outline as a line for each straight
  side: a line of the sketch can be made parallel to, square to or
  dimensioned from one side of the face.
- A new body carries the sketch it was made from: moving the body moves
  it whole.
- A pattern along a path is on the path: what is repeated is put with
  its middle on the path's start and the copies go on from there, where
  they had traced the path's shape beside it.
- The Sketch tab asks where the sketch goes every time — a plane for a
  new one, or one of the sketches there are — where, with a sketch
  selected, it had gone straight back into that one. F2, Tab on a sketch
  and a double-click in the timeline still open the sketch itself.

### Fixed

- An assistant reading the scene sees a body made from sketches as it is:
  its size in get_scene, its volume in measure, its faces in get_mesh,
  check_overlaps and analyze_mesh. It saw an empty, open surface.
- Boolean, Hole and Join read a body made from sketches as what it shows:
  cut with one, they cut nothing; a hole in one "missed the part"; joined,
  it was removed. Applying a modifier to a body is refused with what to do
  instead, where it used to take the modifier off and lose its work.
- A hole stays with its part when the part is moved; so does the live cut
  of an extrude into a mesh.
- A pattern, a ring or a mirror whose feature is suppressed is left out
  with it, where it repeated the feature before that one instead.
- A face pushed in exactly as deep as its wall is thick makes a hole
  through, not a floor of no thickness.
- A sweep along a curve follows the curve when its points are moved.
- An interference check says which pair it could not finish, where it
  showed the whole of one part as the overlap.
- In a sketch: a line or spline under way is kept when another tool is
  taken up or the sketch left; a double click ends a line where it is and
  leaves no line of no length; a control-point spline can be finished
  (Enter, right click); what was selected stays selected through a drag;
  a click a few pixels off a point no longer moves it; a trimmed curve
  keeps its layer and its being a construction line, and its untouched end
  stays joined; an angle dimension holds its side; a conic's fullness is
  read as a number in a millimetre scene; a pattern's path is not carried
  from one sketch into another; a circle with its centre fixed is no
  longer drawn as reference.
- A sketch on a face no longer shows the face's outline as a rectangle
  you drew: it is reference, drawn thin and purple inside the sketch and
  not drawn again outside it, where the face's own edges already are.
- A trim in a sketch reaching a curve's end — a rectangle's side cut at
  its corner — left a point with no line to it there.
- Extruding one of several boxes drawn on a face extruded the others
  instead: the Extrude tool picked every profile at once, and the click on
  the box wanted put that one back. It now starts with what is selected in
  the sketch, or the one profile when there is just one, or none; a click
  picks the region under the pointer itself, not the edge it was caught by.
- After drawing on a part's side or a tilted plane, orbiting the view
  turned it about one axis only: the view was left rolled, as squared to
  the sketch. Orbiting levels it first.

## [0.12.0] - 2026-11-15

### Added

- Help ▸ Check for Updates looks for a newer release now, rather than only
  when Kreate starts, and says what it found: the latest is running; a
  newer one, downloaded and used at the next start (or already waiting);
  or why this copy does not update itself — a build folder's, the
  Microsoft Store's, or turned off.
- A tablet keeps the desktop's layout, at a finger's size: once a finger
  touches the screen, controls grow to a finger's height, the seams
  between areas take a finger's width, and picking vertices and edges
  reaches as far again and a half; a mouse moved brings the desktop's
  sizes back. A finger held still for half a second and lifted is a right
  click. Hide Panels, beside undo and redo, puts every area but the
  viewport away until it is pressed again — for a tablet held upright, or
  whenever the scene wants the room.

### Fixed

- The chat panel's buttons (New Chat, Export, History, API keys) were
  drawn over the model's name when the panel was narrow; they go under
  the pickers when they do not fit beside them.
- Signing Gemini in from the chat panel ended "Authentication cancelled by
  user": the Gemini CLI signs in with Google only where it can ask, so Sign
  In now opens it in Terminal to choose Login with Google, and the panel
  takes up the account once its credentials are saved. Off a Mac it says
  to do the same in a terminal, or to save a Gemini API key.
- A Gemini CLI given an API key at its own /auth (kept in the keychain)
  was sent to a Google sign-in from Kreate anyway, and the panel waited
  for one that would never come. The CLI's own choice of how to sign in
  now stands, and the panel names it signed in; signing in in Terminal
  ends when the CLI is given a key there too, and says so instead of
  asking for a code to paste.

## [0.11.0] - 2026-11-14

### Added

- A phone's layout, on Android below 600 points across (or anywhere with
  `KREATE_LAYOUT=phone`): the viewport whole; a bar with the mode, undo,
  redo, search and every menu; along the bottom Add, Move, Rotate, Scale
  and Delete, and tabs that raise the scene, the properties, the material
  or the assistant in a sheet. With Move, Rotate or Scale in hand, a
  finger dragged across the view transforms what is selected, and lifting
  it confirms; a tap still selects. Controls are a finger's size.

### Fixed

- Gemini (and Codex installed by npm) in the chat panel said "env: node:
  No such file or directory" when Kreate was started from Finder or the
  Dock: the assistant's program now starts with its own folder, and where
  package managers install, on its PATH.

## [0.10.0] - 2026-11-14

### Added

- API keys for the chat panel's assistants (API keys, beside History): an
  Anthropic key for Claude, a Google one for Gemini, an OpenAI one for
  ChatGPT. Each is kept only in Kreate's settings folder, readable by the
  user alone, shown again only by its last four characters, and handed to
  the assistant's program when it starts — ahead of any account it is
  signed in to. A key saved for Gemini works too, where one put in the
  Gemini CLI's own settings had been passed over for a Google sign-in.

## [0.9.0] - 2026-11-14

### Added

- An Android build, to try on a device: the Android workflow makes an APK
  (64-bit ARM, Android 8 and up, debug-signed). It is the desktop program
  with a mouse and keyboard in mind; one finger is the left button, files
  are saved to and opened from the app's Documents folder, and finger
  gestures, a file picker and the on-screen keyboard are still to come.
  See apps/kreate-android/README.md.
- On Android: two fingers turn and pinch-zoom the view, three move it, and
  a two-finger tap is a right click; one finger taps and drags. Open
  brings up the system's file picker, Save offers a copy where you like,
  and the keyboard on screen comes up for a text field. Coming back from
  another app no longer loses the picture.
- 3D paths (Add ▸ Curve ▸ 3D Path), Fusion's 3D sketch: click points in
  space — they catch on corners, edges and faces — and Enter makes a curve
  object through them, straight or smooth, its corners bent to a radius
  as a pipe's are. A sweep follows it; any curve object is now a path a
  sweep can take.
- Conic curves in a sketch (exact: an ellipse's arc, a parabola or a
  hyperbola's by rho), and Text on Path: letters along a curve as outlines
  to extrude or cut, round a closed curve outside it and centred on top.

## [0.8.0] - 2026-11-11

### Added

- More ways to draw in a sketch, as Fusion has them: a 3-point (turned)
  rectangle, a polygon around a circle or by one edge, an ellipse, an arc
  slot, a circle tangent to two lines (at a radius) or three, and a
  control-point spline. A Collinear constraint keeps two lines on one line.

- Project and Intersect in a sketch (Sketch menu): click a part and its
  outline and edges seen square to the drawing — or where the drawing's
  plane cuts it — come in held, on a PROJECTED layer, to draw against and
  close profiles with; they follow the part when the drawing is taken up
  again (or Update Projections). Slice cuts away what is in front of the
  drawing's plane while drawing.

- Sketch tools as Fusion has them: Extend (a line or an arc on to what it
  meets), Break (split where crossed), Chamfer (a straight cut between two
  lines), and Move, Turn or Scale (about a point, or as a copy; a scale
  takes the entities' own dimensions with it).
- The sketch line tool infers as Fusion's does: within two degrees a
  segment turns level, upright, along or square to a line, or on along an
  arc it leaves — shown by the pointer, held as a constraint (Ctrl draws
  free); an end landing on a curve is held on it, at a line's middle to
  it. A while drawing lines makes the next segment a tangent arc.
- Typed values stay as dimensions: a line's typed length, a rectangle's
  width and height, and a circle's diameter. A line takes a second typed
  value as its angle, in degrees.
- Sketch tools: Center Rectangle, Tangent Arc on its own (from the end of a
  line or arc), and the Midpoint and Symmetry constraints.
- Offset takes a chain whole: curves joined end to end are offset to one
  side together and meet at their corners again, lines where they cross
  and curves round the corner. With nothing selected, the first click takes
  the chain under it.
- Selecting in a sketch: a double-click takes the chain a curve is joined
  into; Ctrl-click (Cmd-click on a Mac) takes something back out; a box
  drawn right to left is a crossing box, dashed, taking what it touches.
- The AI chat picks its model: beside the assistant, a list of the models
  its program takes (Claude's Opus, Sonnet and Haiku; Gemini's and
  ChatGPT's), its own default first, and a box at the bottom for any
  other model name. Remembered for each assistant.

### Changed

- A constraint that says what is already held is refused, and a driving
  dimension of what is already held goes in as a driven one, instead of
  over-constraining the sketch.
- In a sketch, Delete (and a Mac's delete key) removes what is selected,
  and X makes it construction, as in Fusion; X had deleted. Delete also
  removes objects and mesh elements, beside X.
- A sketch no longer marks every corner where two ends meet.
- Ctrl+Shift+Z redoes, beside Ctrl+Y.
- Add Modifier lists every modifier: Triangulate, Weld, Displace, Simple
  Deform, Screw, Decimate and Shrinkwrap were missing from the panel.
- An operation that fails says why in the status bar; it had said so only
  to the console.
- The Save, Undo and Redo buttons show their keys.
- In the Extrude, Revolve, Pattern and Edit Feature dialogs Enter is OK
  (and Escape cancels Edit Feature); a greyed OK says what it is waiting
  for.
- A child dragged onto the outliner's Scene Collection row comes off its
  parent.
- The console brings back the lines run before, with Up and Down.
- A feature can be suppressed and put back, and moved earlier or later
  among its body's, from its right-click menu in the timeline — with
  Delete there too. `feature.suppress` and `feature.move` for scripts.
- A part mated to another (Align Faces) keeps its mate: the Object tab's
  Mate panel changes the gap between the two faces and turns the part
  about their normal, the faces staying together, or takes the mate off.
  `object.mate_set` for scripts. Saved with the document.
- Drawing sheets mark what is round and seen end on — holes, bosses,
  shafts — with a centre mark, write its diameter once for all of one
  size ("4x Ø5"), and say where each is from the view's corner.
- An HDRI of your own as the environment (Render ▸ Environment ▸ Add
  HDRI…): it lights the part, is what its metal reflects and is what is
  behind it, in the preview and in renders, and stays in the list.
- In edit mode Subdivide is Alt+S and Recalculate Normals is Alt+N: on
  Shift+S and Ctrl+N they hid the snap pie and New File there.
- NLA strips play from rest each frame, as in Blender: a strip that adds,
  with nothing under it, adds to no movement rather than to the last
  frame's.

### Fixed

- The Delete key did nothing anywhere: no binding on it could be reached,
  the timeline's included.
- Importing a DXF whose text held Hangul, or any letters from a code-page
  file, stopped the program. A DXF with an absurd block array no longer
  hangs the import, a large one reads in a fraction of the time, and a
  turned block inside a mirrored one lands where it should.
- DXF export writes text an R12 reader can read: degrees, diameter and
  plus-minus as DXF's codes, Hangul as `\U+` escapes, a per cent sign as
  one; a dimension's Ø, ° and ± were written doubled and came back as
  `%c`.
- With four lamps or more, some went dark across parts of the view, and a
  lamp with an IES profile hid the two after it: the light cull read the
  lamps at the wrong stride.
- The path tracer: an empty scene could hang the GPU; a face lying on the
  studio floor flickered against it; a dolly or zoom threw the carried
  picture away, leaving a moving view noisy; part-subsurface materials
  came out too bright.
- A sheet-metal flange away from the origin did not weld to its plate,
  leaving the body open; a bend angle from a script or a file is kept
  between flat and folded back.
- The Triangulate modifier dropped sharp edges, creases and seams. A loop
  cut of three or more slid to an end stacked its loops on one spot. A
  voxel remesh with a tiny voxel size could stop the program.
- Sketch tools: a quarter turn (Move/Turn/Scale) collapsed level and
  upright lines; Break left dimensions on the wrong piece; Extend, Chamfer
  and Fillet were pulled back by the line's own length dimension; a
  fillet's lines did not follow when its radius was changed; a copy of
  text was not turned or scaled.
- A projected outline was thrown away and remade each time a sketch was
  opened, taking the constraints and dimensions on it.
- `draft.dimension_set` from a script made the dimension driven, so
  nothing moved; `feature.set` naming only an amount turned a cut into a
  join. A slide joint's default travel is 10 cm each way, not 3.14 m.
- Pasting or importing objects left a boolean's cutter, a path, a
  feature's sketch and the like pointing at the originals.
- Renaming a bone lost its animation in NLA strips. A particle instance
  turned or scaled did not update its particles.
- Typing in the chat or any field while a sketch tool, the knife or
  Measure was active also drove the tool.
- In a sketch, a drag interrupted by a tool key or Delete could undo
  everything drawn since on the next drag; after an undo, Delete could
  remove the curve next to the one selected; a dragged point could snap to
  the wrong curve.
- Recover asked nothing before replacing the open document, and deleted
  an autosave it had failed to read.
- A request or an import arriving in the middle of a move or a drag was
  silently undone by it.
- A number typed into a field was an undo step for each keystroke.
- The Adjust panel kept the first run's values when the same operation
  ran twice. Clicking a pie menu's entry also clicked the view behind it.
  The graph editor's keys stopped working after a middle-button pan.
- The chat's box for another model's name could not be clicked into.
- NLA: an Add, Subtract or Multiply strip built on the last frame's
  result, so the object flew off as it played, and a strip of part
  influence crept; the same frame asked for twice gave two poses.
- Where there is only Windows' software renderer (a virtual machine, a
  remote desktop without a GPU) asking for path tracing stopped the
  program; the rendered view stays rasterised there.
- A long text file or video transcript attached to a Claude chat message
  kept the assistant from starting.
- What was saved in a crash was never offered back at the next start.
- The keymap editor could not take a key with Ctrl, Shift or Alt; a key
  captured onto a mouse binding killed it; a binding left listening when
  its panel closed took the next key pressed anywhere. Escape stops it
  listening.
- The Solidify modifier dropped UVs, materials and smooth shading.
- A DXF's layer linetypes and frozen layers were lost on import.
- In a render, the studio floor and backdrop were missing behind glass
  and cut-out materials: a glass on the floor stood on the bare
  environment with no shadow under it.
- The node editors kept one graph's selection when another was shown, so
  X removed nodes of a material nobody had touched. In Geometry Nodes,
  Shift+D, Ctrl+G, Alt+G and copy and paste emptied the selection and did
  nothing; they are off there until those exist.
- A drawing entity written by a newer Kreate stayed unread for good once
  an older one had saved the file.
- A small part's measured inertia was written to URDF as zero, and a
  mirrored part left out of its link's mass.
- A document holding a number that is not finite saved as a file that
  would not open; the save is refused instead and says why. Parenting to
  an object scaled to nothing no longer makes one.
- Dragging a modifier's value was an undo step for every frame. Escape in
  a driver's or a parameter's expression kept what was typed. X over the
  graph editor's channel list deleted the object.

## [0.7.1] - 2026-11-07

### Fixed

- The Wireframe overlay turns the wireframe on and off in the material
  preview, the view Kreate works in; it had done so only in a shading
  mode the program no longer offers, so the switch did nothing. The
  preview starts with it off, as it always looked.

## [0.7.0] - 2026-11-06

### Added

- Outdoor's sky is a physical daylight model (Preetham's), its sun the
  colour the air leaves it; and a Sunset room, the sun low and orange.
- A point light or a spot can take a luminaire's measured shape from its
  IES file (Light tab, Load IES…), in the viewport and the path tracer.
- Materials can scatter light below their surface — skin, marble, wax,
  milk — with a colour and a radius: the path tracer walks it through the
  solid, so a face out of the light glows with what came through; the
  viewport shows its colour.
- Glass can take colour out of what passes through it, by an absorption
  colour and distance: a wine bottle, sea water. glTF's
  `KHR_materials_volume` carries it.
- Materials have a clear coat (car paint, varnish), a fabric sheen, a
  brushed (anisotropic) highlight and a specular weight, in the Material
  tab, the path tracer and the viewport alike; glTF files carry them as
  `KHR_materials_clearcoat`, `_sheen`, `_anisotropy` and `_specular`.
- A non-metal's reflection follows its index of refraction: water reflects
  less than glass, a gem far more.
- A finished render (F12, Ctrl+F12) is denoised by Intel Open Image
  Denoise when it is installed beside Kreate — the release builds carry
  it — keeping fine texture and soft shading that the view's own filter
  blurs. Without it a render is filtered as before.

- Sheet metal from a sketch (Mechanical ▸ Sheet Metal ▸ From Sketch): any
  outline, holes and all, made a plate with flanges bent up from its
  straight edges (every one that can take one, or those named), as a body
  that follows the sketch; File ▸ Export Drawing ▸ Sheet Metal Flat
  Pattern writes its blank as DXF or PDF, with the bend lines and each
  bend's direction, angle and radius.
- Assembly joints (Mechanical ▸ Assembly Joints): a hinge, a slider or a
  fixed joint at the 3D cursor joins the active part to another; the part
  then turns or slides within its limits from the Robot panel's slider,
  carrying what is joined to it. They are the same joints a URDF robot
  has, so they pose, mimic, rig and export the same way. Check Joint
  Motion takes a joint through its travel and says where its parts run
  into the others.
- The Physical Properties panel shows a part's moments of inertia about
  its centre of mass, by its material's density; a URDF export gives each
  link with none of its own the mass, centre and inertia measured from its
  meshes, so a simulator has bodies to move.

- A 2D drawing — a sketch or a drafting drawing — can be exported as DXF
  (File ▸ Export Drawing ▸ Active 2D Drawing, File ▸ Export, or the
  connector's `export_file` with `.dxf`): R12, which every drafting program
  and cutting machine reads, with its layers, colours and linetypes, arcs
  as arcs, ellipses and splines as fine polylines, and dimensions drawn out
  with their tolerances.
- Drawing sheets (File ▸ Export Drawing ▸ Drawing Sheet…): the shown parts
  on A4–A0 paper at a standard scale, in third-angle (KS B 0001) or
  first-angle projection, with an isometric view, the overall sizes, a
  title block and a parts list, written as a vector PDF (Korean text
  included) or a DXF. Made from the parts as they are each time.
- Sweep and Loft are features (Mechanical ▸ Shapes): the body keeps them
  on its timeline and follows when the profile, the path or a section is
  edited, and they can join or cut an existing body. Sweep with no profile
  sketch still makes a round pipe as a plain part.

### Changed

- The path tracer uses the GPU's own ray tracing where it has it (Vulkan
  ray queries); `KREATE_RAY_QUERY=0` turns that off. Elsewhere it walks its
  own trees as before.
- Linked copies of a mesh are stored once in the path tracer and placed
  where each one stands, so a scene of many copies costs the memory and the
  build time of one.
- The path tracer's samples are a scrambled Sobol sequence rather than
  independent random numbers: the same picture in about half the samples.
- Among several lamps, the path tracer aims at each by how much it lights
  the point, so a scene with many dim lamps and one bright one is far less
  grainy.

### Fixed

- Editing a pattern feature showed its count divided by the length unit
  (3 copies read as 3000 in millimetres).
- A file saved by a newer Kreate keeps the features this one does not know
  through an open and a save; opening it says how many were kept unbuilt.

## [0.5.3] - 2026-10-28

### Added

- The Claude connector has an `environment` tool that reads or picks the
  studio (Studio, Bright Studio, Grey Room, Outdoor), as the Render tab
  does; the context reports it too.

### Changed

- The connector's file tools say they open DXF drawings and STEP parts,
  and its render tool says what a render stands on.

## [0.5.2] - 2026-10-27

### Added

- Drawing in a sketch catches the parts already modelled — their corners,
  edges' middles, faces' centres and edges, laid onto the sketch's plane — whether snapping
  is on or not, as it catches the other sketches.
- The outliner gathers the sketches under a Sketches heading, a sketch on
  a part's face among them.
- The sketch-plane menu lists the sketches there are, to open one.
- An extrude's join or cut can go into any part, picked in its dialog.
- Offset and mirror take a distance: an offset's, typed first or changed
  in the last-operation panel (negative for the other side), and a
  mirror's, which moves its line and so the gap to the copy.
- Environment presets in the Render tab — Studio, Bright Studio, Grey Room
  and Outdoor — each its own lights, backdrop and exposure, remembered
  between sessions.
- Final renders and the rendered view stand the parts on an invisible
  floor that catches their shadows and a faint reflection, in front of a
  light studio backdrop, as a product shot has.

### Changed

- The studio light has an overhead strip and two side strips beside its
  key, fill and rim, for longer highlights on curved parts.

- F2 with no sketch selected asks where a new one goes, rather than
  opening the first sketch in the scene.
- A join or cut into a part that is not a sketch body leaves only the
  result: no body of the extrude's own beside it.
- An offset of several curves is one step.

### Fixed

- Choosing what to mirror, offset or pattern and then the tool lost the
  selection.

## [0.5.1] - 2026-10-27

### Added

- DXF import: File ▸ Import, a drop on the window or the connector brings a
  2D drawing in as a drawing — lines, circles, arcs, ellipses, polylines
  with their bulges, splines, points, TEXT and MTEXT, on their layers with
  their colours, blocks drawn out where they are inserted, in the unit the
  file declares (millimetres when it says none). One step to undo.
- A drawing's text is shown in 2D, at its size and turned as it is.
- F2 into a drawing frames what is drawn in it.
- A sketch fillet's radius can be typed before its corner is clicked, and
  it stays a radius dimension that can be changed afterwards.
- More ways to draw a circle and an arc: a 2-point circle (the ends of a
  diameter), a 3-point circle, and a centre-point arc (centre, start,
  end), beside the centre-and-rim circle and the 3-point arc.
- The sketch ribbon has a Modify group — fillet, trim, offset, mirror,
  pattern — apart from the tools that draw.

### Changed

- New objects are added at 500 mm, a cube's edge.

### Fixed

- In the material preview, a sun's shadow fell as a hard straight band
  across lit faces when the DirectX shader compiler was installed.
- Changing a trim's numbers in the last-operation panel broke the line
  apart; it no longer offers them.
- A 3-point arc in a drawing away from the origin came out on the wrong
  circle.

## [0.5.0] - 2026-10-27

### Added

- A 2D drafting mode, the first part of it: drawings of exact lines,
  circles, arcs, ellipses, polylines and splines, held by constraints
  (coincident, horizontal, vertical, parallel, perpendicular, tangent,
  equal, concentric, on a curve, symmetric, midpoint, fixed) and by
  dimensions that drive the drawing or, switched to driven, only measure
  it, shown in brackets. Every sketch tool draws in one — line, rectangle,
  circle, arc, spline, polygon, slot, point, text, trim (showing the piece
  first), offset, fillet, mirror, pattern — and what is drawn from a point
  already there is joined to it. A drawing's closed regions extrude and
  revolve, and its dimensions bind to parameters. Its panel lists its
  dimensions and constraints. The connector reads one with `get_drawing`
  and draws with the `draft.*` operators.
- F2 goes between 2D and 3D: into the selected drawing (or the scene's
  first, or a new one) and back out. In 2D the view stays square to the
  drawing; what would turn it pans it.
- A welcome card over the empty document at the first start: what Kreate
  does, in four lines each saying how to begin — modelling, designing a
  part, shading and rendering, working with an assistant — and a button
  that opens a sample scene. Closed once, it stays closed; Help ▸ Welcome
  brings it back.
- The Material tab shows the slot's material on a ball under the material
  preview's light, and pointing at a physical material in its menu shows
  that one — what a part would be made of, before choosing it and without
  switching the view.
- Physical materials have a finish: wood grain with its rings, marble
  veins, granite speckle, concrete blotches and pores, leather grain, a
  brushed hairline on steel and titanium, a cast surface on iron and MDF,
  a moulded texture on plastics and rubber, frost on frosted glass, and a
  satin unevenness on the other metals. Laid out in metres, so a grain is
  the same few millimetres on a cup and on a table.
- A sketch's open profile revolves into a solid about the sketch's own
  horizontal or vertical axis, as well as about one of its lines.
- X-ray selection in edit mode reaches every vertex, edge and face behind
  the front ones, not only those a pixel happened to show.

### Changed

- New sketches are drawings, and a file's sketches open as drawings: the
  same objects, building the same solids, their revolve axes and the
  parameters bound to their dimensions carried across. The file format is
  1.5; Kreate 0.3 keeps a drawing in a file it saves but does not show it.
- Rename is Ctrl+F2 (or a double-click in the outliner); F2 is the 2D/3D
  switch.
- The zip reader is `zip` 9: three advisories on 6's extraction, one on
  permissions kept through symbolic links, no longer apply. File dialogs
  are `rfd` 0.17, which drops the unmaintained D-Bus crates 0.15 carried.
- CI: Linux runs the renderer's tests on Mesa's lavapipe, which draws what
  a GPU draws; Windows, whose runner has only WARP, no longer runs them —
  WARP draws no shadows and traces nothing, and the job had been red on
  that for every run on record.

### Fixed

- Pushing a box's end or middle band in to make a notch left a spike on
  the face beside it in the released program (not in a development build).
- A damaged file no longer takes the program down with it. Every reader
  is now tested against thousands of broken copies of a sound file — bits
  flipped, cut short, numbers swapped for impossible ones — and what that
  found is fixed: a PLY naming more faces than there are atoms made that
  many empty faces until the system killed Kreate; a vertex at infinity
  overflowed the welder; counts and offsets at the end of the integers
  overflowed the USD, FBX and COLLADA readers; and the glTF library
  asserts on a file whose accessors disagree, which is now an error.
- Updating on macOS left Kreate.app's seal broken and its version the old
  one; the bundle is now signed again and its Info.plist set, after the
  update and again once the files put aside are deleted.
- The status bar's "update downloaded" message is translated.
- The font list no longer offers the faces macOS keeps for its own
  interface (".SF NS", ".ADT Slab Numeric Light"), which no font menu
  shows and some of which hold numerals only.
- The fonts — the two bundled and the computer's — are their own crate,
  `kreate-fonts`, read by the mesh kernel and by the font picker; the
  interface had been reaching into the mesh kernel for them, which the
  architecture tests forbid and which CI had never reached to say so.

## [0.3.1] - 2026-10-22

### Added

- A robot read from a URDF can be posed with an armature: Object ▸ Parent ▸
  Make Armature from Robot (or Make Armature in the Robot panel) puts a bone
  at every joint that moves, standing on the joint's axis and held to its
  limits, with an effector and an IK target at the end of each arm. Turn the
  bones in pose mode, or switch the IK on and drag the target, and the robot
  follows; key the bones and it plays, and writes to glTF moving. The Robot
  panel's sliders and the bones stay in step either way, mimic joints
  included, and the URDF written out is the robot as before.
- IK honours a bone's Limit Rotation: a bone held to turning about its own Z
  turns only that way, within its range, while the chain reaches.
- A Limit Location bone constraint, for a slider's travel.
- Objects can hang from a bone (Ctrl+P ▸ Bone, Blender's Bone parent): they
  move with it rigidly, as they are, and keep doing so through saving, undo,
  copy and paste, and glTF export.
- Reading a large file shows what it is doing: a card over the viewport with
  the stage, how far through it, how long it has taken, and Cancel. Opening a
  scene file no longer freezes the window while it reads.

- STEP files (`.step`, `.stp`) open and import: solids and surfaces from any
  CAD program — planes, cylinders, cones, spheres, tori and splines — cut
  into triangles that follow the true shape, watertight where the part is.
  An assembly comes in as its parts, each placed where the assembly puts it
  and shared between its copies, with the file's colours as materials and
  its own unit (millimetres, inches) honoured. STEP export was already there.

- An Assets editor: models, materials and HDRIs from your own folders or
  from Poly Haven (CC0), searchable, with pictures. A model is added to the
  scene, a material is made of its maps and put on the active object, an
  HDRI lights the world. Downloads are kept in the cache for next time.
- FBX characters come in rigged and moving: a skin's bones and weights
  become an armature, and each take — Mixamo's dance, a walk — an action,
  the first playing and the rest as muted strips. Meshes sharing one
  skeleton share one armature, and the bones nothing is weighted to (a
  fingertip, an eye) are bones too.
- The connector's asset tools see and place: `search_assets` can return the
  first six's pictures, to choose by looking, and searches no longer hold
  up the window; `add_asset` takes where a model's base goes and how tall
  it should be, and with `wait` answers once it has landed with what was
  added, where and how big.
- The connector teaches its ways of working: its instructions now say when
  to reach for assets, AI generation, the sculpting loop (compare, shape,
  measure again), the rigging loop and the timeline; and it offers them as
  prompts — rig a character, sculpt to a reference, dress a scene from
  assets, make a model — for a client to start from.
- The connector's `check_proportions` measures a standing figure against a
  realistic adult — heads tall, leg length, arm span, chest and hip width —
  each with the usual range and whether it is below, within or above.
- The connector's `compare_silhouette` sets the model's outline from a side
  against a reference picture's — concept art, a photo — at one height and
  where they overlap most, and says how much they agree, their proportions,
  where the model has too much or too little, and shows it in red and blue.
- Detail from a picture: `sculpt.displace_image` raises and lowers the
  surface by a height map, through its UVs or projected from three sides,
  so many millimetres, round one place or all over; and the connector's
  `generate_height_map` draws one — skin, scales, bark — with AI, made
  level and seamless by Kreate.
- Sculpting by number: `sculpt.deform_region` pushes, swells, moves,
  scales, flattens or smooths a ball- or egg-shaped region by so many
  millimetres, fading to its rim, mirrored across X if asked and held by the
  mask; `sculpt.crease_path` carves a groove or raises a ridge along a path
  of points. What an assistant sculpts with.
- The connector's `analyze_mesh` reads a mesh as a sculptor would — holes,
  broken surface, slivers, bumpy places, and where its halves differ —
  each with where it is, so an assistant knows where to work next.
- `object.add_humanoid_rig(fingers=True)` adds a thumb and four fingers a
  hand, three bones each, spread across the hand as measured. Off by
  default: on a Mixamo hand the template's fingers sit between the real
  ones and tear the middle finger's first joint, so it suits simple open
  hands, and `check_rig` says when it does not.
- Automatic weights can look before they warm: with `visibility`, a vertex
  takes heat only from bones it can see, not through skin — so a body's
  side stays off an arm hanging beside it. The humanoid rig turns it on when
  the arms hang down; held out, a figure tears less without it.
- Binding With Automatic Weights spreads each bone's weight by heat along
  the surface, as Blender does, rather than by distance through the air: a
  thigh no longer pulls on the other leg, nor an upper arm on the ribs. On a
  Mixamo body the edges a 45° bend tore went from hundreds to a handful.
  The old weights by distance stay, as `weights="envelope"`.
- `object.add_humanoid_rig` fits a standard skeleton — hips to head, and
  shoulder, arm, hand, thigh, shin and foot a side, `.L` and `.R` — to the
  selected figure standing up Z, arms out or down, measuring its height, its
  legs, its reach and its feet, and binds it in the same step. Object ▸
  Parent ▸ Add Humanoid Rig.
- Symmetry for rigs: `armature.symmetrize` copies every left bone (`.L`,
  `_l`, `Left…`) to the right across X, parents and roll mirrored, and
  `object.vertex_weights_mirror` copies one half's weights to the other, a
  left bone's to the right one's — half a character rigged, the other half
  made from it.
- The connector's `check_rig` bends every bone of a rig in turn and
  measures the skin: which bones tear it and where, which pinch, which move
  nothing, and which vertices no bone holds — so an assistant can fix the
  worst and check again. A 67-bone character takes a tenth of a second.
- Vertex weights set exactly — `object.vertex_weights_set`, by vertex
  numbers, by a box, or the selection; replace, add or subtract; alone, so
  those vertices leave every other bone — and smoothed along the edges with
  `object.vertex_weights_smooth`. What an assistant rigging a model needs
  where a brush is guesswork.
- Assets ▸ Made by AI makes a model from words or a picture, on Hugging
  Face's free GPUs: words become a picture with Z-Image Turbo, the picture a
  model with TRELLIS, and the model lands in the scene and stays in the list
  to add again. A free Hugging Face token, kept in the settings folder,
  raises how much GPU time Hugging Face lends. The connector has it too:
  `generate_model`.
- Help ▸ Connect to AI Apps adds Kreate's tools to Claude Desktop, Claude
  Code, Cursor, VS Code, the Gemini CLI or Codex in one click — their other
  settings kept, a copy of the file as it was left beside it, a file that
  does not parse left alone — and shows which are already connected.
- The connector's capture takes `views` — front, back, left, right, top,
  bottom or the user's — and `only`, some objects alone, framed; and
  `get_context` says in a breath what is selected (and how many vertices,
  edges and faces in edit mode), where the cursor is and where the view
  looks from: what "this" and "here" mean to an assistant.
- The connector has a clock: `timeline` reads the frame, the range and
  every animated object with its takes, moves to a frame, plays or stops,
  and switches a take on or off with its timing kept — so an assistant can
  look at a pose at frame 30, or play the walk instead of the run.
- Under each answer in the chat panel, what it did — how many steps, how
  many objects added and removed — and a button that undoes all of it at
  once, as long as you have not changed the document since.
- Your own model files have pictures in the Assets editor: each is read and
  drawn once, from the front right and a little above, and the picture kept
  for next time.
- Objaverse in the Assets editor: Sketchfab's Creative Commons models —
  people, characters, creatures, props — found with Sketchfab's search and
  fetched from the Objaverse dataset on Hugging Face, with no account.
  Only licences that allow changing and selling are listed; a CC
  Attribution model says the credit it asks for. Poly Pizza, which needs a
  key, moves after it.
- Poly Pizza in the Assets editor: thousands of low-poly models, many of
  them animated characters, CC0 or CC-BY. Enter searches it; "Animated
  only" lists what moves. It wants a free API key of your own, pasted into
  the panel once and kept in Kreate's settings folder. A CC-BY model says
  the credit it asks for when it is added.
- Tiles show "▶ Animated" for a model that moves — Poly Pizza's, and your
  own glTF and FBX files — and "CC-BY" where a credit is owed.
- ambientCG in the Assets editor: its two thousand CC0 materials and its
  HDRIs, beside Poly Haven's.
- Compressed OpenEXR images open — as a world's light, a texture or an
  asset — and a folder's HDR and EXR files show their pictures in the
  Assets editor.
- Link a `.kreate` from the Assets editor rather than copy it: its objects
  are read-only here and read again from the file each time the document
  opens, so fixing a prop in its library fixes it in every scene.
- An Emission Map on a material: a picture the glow is multiplied by — a
  screen, a lit window. glTF's emissive texture imports and exports as one.
- FBX cameras (lens and film back) and lights (point, sun, spot) are
  imported; FBX gives lights no unit, so the brightness taken is noted.
- The AI chat can find and add assets: `search_assets` looks through
  Poly Haven, ambientCG or your folders, and `add_asset` brings one in.
- A saved document reopens seen from where it was looked at when saved:
  the view's angle, distance, lens and projection go in the file.
- Cut-out materials: a surface is cut away where its base colour image is
  clear, below the material's Alpha Clip — leaves, labels, grilles — in the
  studio view, the rendered view and the path-traced render. glTF's `MASK`
  imports as one and exports as one; `BLEND` (a clock's front, a lamp's
  shade) imports as glass.
- A cut-out casts the shadow of what is left of it — a leaf's, not its
  card's — in the rendered view and the path-traced render.

### Changed

- Updates now come from the releases of github.com/MinaryHub/Kreate,
  where the source is too, instead of a separate releases-only repository.
- A new look, "Graphite Workshop": the chrome is three graphite steps, darker
  than the viewport, so the model is the brightest thing on screen; orange
  marks only what is selected or active. Numbers are set in IBM Plex Mono
  (bundled, SIL Open Font License) so columns of values line up; rows and
  fields are 24 px. The workspaces are one segmented group, the mode is named
  in its own quiet tint in the viewport header and the status bar, the status
  bar's shortcuts sit on key caps, and the outliner marks the active object
  with an accent edge.

- A new document is empty: no cube, no lamp, no camera. The world is still
  there, so what is added is lit in the rendered view.
- An import frames what arrived, so a part a few centimetres across is not a
  dot on the grid.

- Animated glTF characters from Sketchfab (and so Objaverse) move as they
  should. Their skins hang under nodes scaled to a hundredth or a fifth,
  which glTF does not count and Kreate did — the body walked off from its
  bones, or stood still beside them; joints scaled as they were bound grew
  their children; and a mesh hung from a joint (a robot's plate, a
  skeleton's bone, a sword in a hand) stayed behind. Each now matches what
  glTF says, frame by frame. A rig whose matrices are in centimetres under
  joints scaled to metres stands its bones at the body's size, not forty
  times it; a node scaled to nothing (a particle before it appears) no
  longer turns everything to NaN; and files using the older
  specular-glossiness, unlit or texture-transform extensions open.
- Tests keep to a temporary folder: a test that saved, failed or remembered
  a recent file no longer writes into your settings, caches or autosaves,
  where a hundred stale test sessions had piled up as recovery offers.
- New, Open and a dropped document ask Save, Don't Save or Cancel when the
  open one has unsaved changes; Save saves first (asking where, if it has
  never been saved) and a save dialog closed leaves you where you were.
- The first save of a document opens in the folder you last kept one in,
  or Documents, titled to say you are choosing the folder and the name.
- `B` during a move picks the snap base, as in Blender: the selection holds
  still, the pointer finds a vertex, edge or face of what is moving, a
  click takes it, and from then on that point lands on whatever the
  pointer snaps to — this corner onto that one. Rotate and scale keep `B`
  as the snapping toggle.
- A glTF character with several takes — idle, walk, run — keeps them all.
  The first plays; each other is a muted strip on the rig, to switch on.
  They were read as one, the last take over the rest.
- Hiding an object in the outliner hides its children with it. Each child
  keeps its own eye, drawn faint while its parent is hidden, and comes back
  when the parent does.
- An imported model or asset lands beside what is already there instead of
  inside it. Every asset is made at the origin, so a small one added after
  a large one was hidden inside it — in the outliner and nowhere to be seen.
- Pictures show in the Flat view and not the Studio one. Studio is lit to
  read shape and shows each material's colour; Flat shows its texture. A
  cut-out leaf is still cut in both.
- Importing a model — File ▸ Import, a dropped file, an asset — reads it in
  the background, so a large one no longer stops the window, and records
  the step as the file's path rather than the whole model: undo history
  and autosaves stay small.

### Fixed

- Walk navigation went smoothly only while the keyboard repeated, and drifted
  back to where it started: the camera's smoothing pulled it to its old
  target every frame. It now walks every frame, at a pace set by how far off
  the view is looking.
- In the wireframe shading mode, dense meshes faded away as the view drew
  back — their wires are all there is of them, and now they stay.

- Cameras and lights from glTF files look and shine where the file says:
  each was tipped a quarter turn, a level camera looking at the floor and a
  sun shining sideways. Written out again, they stay the right way round.
- A `.gltf` whose buffers and textures are files beside it opens and
  imports; it needed everything embedded. References that leave the file's
  folder are refused.
- glTF import no longer reports that every texture went unread.
- Textures are the right way up. V runs up the picture, as in Blender and
  OBJ; every image was drawn upside down, which on a model whose texture
  holds all its parts — most from Poly Haven — put the wrong part on every
  face. In the studio view as well as the rendered one.

## [0.3.0] - 2026-10-09

### Added

- Import FBX (binary and ASCII: meshes, hierarchy, materials and diffuse
  textures), 3MF (including slicer packages that keep each object in its own
  part), PLY (text and binary) and OFF, beside glTF, OBJ and STL.
- File ▸ Import adds a scene file to the open document as one undo step;
  Open still replaces it. The AI connector gains `import_file` to match.
- Sculpting: ten more brushes (Clay Strips, Crease, Flatten, Scrape, Fill,
  Pinch, Layer, Snake Hook, Mask, Face Set); a mask that every brush respects;
  face sets that can be hidden, isolated and shown again; falloff shapes,
  spacing, a stabiliser, front-faces-only, auto-smooth, Y/Z and radial
  symmetry, alpha stamps and pen pressure; Voxel Remesh (Ctrl+R) and dynamic
  topology.
- Shader nodes: Noise, Voronoi, Checker and Wave textures (bands or rings,
  for wood and marble). They read the surface's position in its object, so
  they need no UV unwrap and move with the object.
- Image Texture node: Box (triplanar) projection, which lays an image on
  from the object's three axes with a blend between them — a tiling texture
  on a mesh with no UVs, without stretching.
- Bump node: turns a height — a noise, a wave, an image's channel — into a
  tilt of the surface, and can sit on top of a normal map.
- Color Ramp node (up to eight stops; linear, constant or eased), and
  Texture Coordinate and Mapping nodes: a texture can be read by UV, by the
  object's position or its normal, and moved, turned and scaled.
- The path-traced render runs each material's node graph — the same code
  the viewport draws with — so procedural textures, projected images, normal
  maps and bumps appear in the final render, not only in the viewport.
- The path-traced render shows each face's own material on an object whose
  faces wear several; it used the first slot's for the whole object.
- Rigid bodies: a gravity factor of their own (Physics tab), so one body can
  float, fall slowly or rise while the rest of the scene falls.
- Bevel: with an even number of segments, the corner where three bevelled
  edges meet is filled with quads rather than a fan of triangles.

### Fixed

- The Physics tab changes the active object only. It used to apply all of its
  settings to every selected object, so adjusting it with a rigid body
  selected beside an object without one turned the rigid body off.
- Bake to Keyframes says on hover that it turns the baked rigid bodies into
  keys and stops them being rigid bodies.
- Orthographic view (Numpad 5) and the viewport lens stay as set; both went
  straight back on the next frame.
- The camera view (Numpad 0) follows its camera as it moves or plays an
  animation, and lets go when you move the view yourself.
- Cmd+V pastes a picture or files into the AI chat on a Mac.
- A sculpt stroke after changing the mask, hidden parts or face sets, with
  nothing else changed, used the old ones.

## [0.2.0] - 2026-10-09

### Added

- Physics: rigid bodies that fall, collide and stack; cloth that drapes and
  can be pinned by a vertex group; soft bodies that wobble back to shape;
  particles; and hair, still or swinging. Set it up in the Physics tab or
  Object ▸ Physics, press Bake, and play the timeline. Rigid bodies can be
  baked to keyframes.
- In sculpt, weight paint and texture paint, F resizes the brush and Shift+F
  sets its strength by moving the pointer.
- Outliner search: find an object by name.
- Numpad 0 goes through the cameras when one is already the view.
- Design tools: a scene length unit, measuring and dimensions, exact
  coordinates, a work plane, mass and volume, a mesh check, a section view,
  3MF, STEP and drawing (DXF/SVG) export, a parts list, sketches with
  constraints, and aligning one part's face to another's.
- The Claude panel takes attached files and pasted images.
- Updates itself: at start Kreate looks for a newer release on GitHub,
  downloads it in the background, and uses it the next time it starts.
  `KREATE_NO_UPDATE=1` turns this off.
- Help ▸ Contact: write to the developers from inside Kreate. The message goes
  through a relay, so neither the address it reaches nor any key is in the
  application.
- Help ▸ About Kreate, with the version and the commit it was built from.
- A Claude panel: a chat with Claude Code that edits the open document through
  its operators, one undo step per edit. Shows the account it is signed in as
  and signs in from the panel.
- Script lines can switch mode (`object.mode_set`) and read an image file
  (`image.open`); each line reports the objects, materials and images it made.
- Shapes for scripts: `object.add_lathe` (a turned profile), `object.add_sweep`
  (a profile along a path), `mesh.bevel` and the bevel modifier with rounded
  segments, partial creases (`mesh.crease`) honoured by subdivision.
- `mesh.select_where` (by box, direction or material), `object.set_dimensions`,
  `object.shade_smooth`, `object.place_on`, `object.align`, `material.rename`,
  `data.purge`, and `image.generate` for tiling noise, voronoi, wood, marble,
  checker and brick textures.
- For the Claude connector: `get_mesh`, `check_overlaps` on real surfaces,
  `render_image` to a file through a camera, and a camera view for
  `capture_viewport`.

### Changed

- Sculpting keeps up with the pointer: a stroke is worked once a frame, and
  starting one no longer pauses on a dense mesh.
- Flat and studio shading show each face's material colour.
- A `set_*` line that leaves a parameter out keeps its current value instead of
  resetting it to the default.
- New objects never take a name in use: a second `Cube` is `Cube.001`.
- `get_scene` reports colours in sRGB, as the operators take them, and gives
  uids, dimensions, bounds, images and the view's exposure.

### Fixed

- The outliner's scroll bar no longer covers the visibility eyes.
- A duplicate keeps its materials.
- A viewport capture taken while the window was hidden came back empty.
- `material.link` works on an object with no material slots yet.

## [0.1.0]

The first version: the editor, its operators, the renderer, files and the
Claude connector, before releases were numbered.

[Unreleased]: https://github.com/MinaryHub/Kreate/compare/v0.15.1...HEAD
[0.15.1]: https://github.com/MinaryHub/Kreate/releases/tag/v0.15.1
[0.15.0]: https://github.com/MinaryHub/Kreate/releases/tag/v0.15.0
[0.14.0]: https://github.com/MinaryHub/Kreate/releases/tag/v0.14.0
[0.13.0]: https://github.com/MinaryHub/Kreate/releases/tag/v0.13.0
[0.12.0]: https://github.com/MinaryHub/Kreate/releases/tag/v0.12.0
[0.11.0]: https://github.com/MinaryHub/Kreate/releases/tag/v0.11.0
[0.10.0]: https://github.com/MinaryHub/Kreate/releases/tag/v0.10.0
[0.9.0]: https://github.com/MinaryHub/Kreate/releases/tag/v0.9.0
[0.8.0]: https://github.com/MinaryHub/Kreate/releases/tag/v0.8.0
[0.7.1]: https://github.com/MinaryHub/Kreate/releases/tag/v0.7.1
[0.7.0]: https://github.com/MinaryHub/Kreate/releases/tag/v0.7.0
[0.5.3]: https://github.com/MinaryHub/Kreate/releases/tag/v0.5.3
[0.5.2]: https://github.com/MinaryHub/Kreate/releases/tag/v0.5.2
[0.5.1]: https://github.com/MinaryHub/Kreate/releases/tag/v0.5.1
[0.5.0]: https://github.com/MinaryHub/Kreate/releases/tag/v0.5.0
[0.3.1]: https://github.com/MinaryHub/Kreate/releases/tag/v0.3.1
[0.3.0]: https://github.com/MinaryHub/Kreate/releases/tag/v0.3.0
[0.2.0]: https://github.com/MinaryHub/Kreate/releases/tag/v0.2.0
[0.1.0]: https://github.com/MinaryHub/Kreate/commits/main
