# Changelog

Every release of Kreate, newest first. The format is
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the numbers follow
[Semantic Versioning](https://semver.org/) — see [RELEASING.md](RELEASING.md)
for what each part of a number promises and how a release is cut.

Write what changed under **Unreleased** as it lands, in the words a user would
use; `scripts/release.sh` turns that section into the next release.

## [Unreleased]

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

[Unreleased]: https://github.com/MinaryHub/Kreate/compare/v0.5.2...HEAD
[0.5.2]: https://github.com/MinaryHub/Kreate/releases/tag/v0.5.2
[0.5.1]: https://github.com/MinaryHub/Kreate/releases/tag/v0.5.1
[0.5.0]: https://github.com/MinaryHub/Kreate/releases/tag/v0.5.0
[0.3.1]: https://github.com/MinaryHub/Kreate/releases/tag/v0.3.1
[0.3.0]: https://github.com/MinaryHub/Kreate/releases/tag/v0.3.0
[0.2.0]: https://github.com/MinaryHub/Kreate/releases/tag/v0.2.0
[0.1.0]: https://github.com/MinaryHub/Kreate/commits/main
