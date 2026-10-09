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
- ambientCG in the Assets editor: its two thousand CC0 materials and its
  HDRIs, beside Poly Haven's.
- Compressed OpenEXR images open — as a world's light, a texture or an
  asset — and a folder's HDR and EXR files show their pictures in the
  Assets editor.
- Link a `.kreate` from the Assets editor rather than copy it: its objects
  are read-only here and read again from the file each time the document
  opens, so fixing a prop in its library fixes it in every scene.
- The AI chat can find and add assets: `search_assets` looks through
  Poly Haven, ambientCG or your folders, and `add_asset` brings one in.
- A saved document reopens seen from where it was looked at when saved:
  the view's angle, distance, lens and projection go in the file.
- Cut-out materials: a surface is cut away where its base colour image is
  clear, below the material's Alpha Clip — leaves, labels, grilles — in the
  studio view, the rendered view and the path-traced render. glTF's `MASK`
  imports as one and exports as one; `BLEND` (a clock's front, a lamp's
  shade) imports as glass.

### Changed

- Importing a model — File ▸ Import, a dropped file, an asset — reads it in
  the background, so a large one no longer stops the window, and records
  the step as the file's path rather than the whole model: undo history
  and autosaves stay small.

### Fixed

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
