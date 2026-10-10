# Changelog

Every release of Kreate, newest first. The format is
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the numbers follow
[Semantic Versioning](https://semver.org/) — see [RELEASING.md](RELEASING.md)
for what each part of a number promises and how a release is cut.

Write what changed under **Unreleased** as it lands, in the words a user would
use; `scripts/release.sh` turns that section into the next release.

## [Unreleased]

### Added

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

[Unreleased]: https://github.com/knoxxr/Kreate/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/knoxxr/Kreate/releases/tag/v0.3.0
[0.2.0]: https://github.com/knoxxr/Kreate/releases/tag/v0.2.0
[0.1.0]: https://github.com/knoxxr/Kreate/commits/main
