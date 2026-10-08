# Changelog

Every release of Kreate, newest first. The format is
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the numbers follow
[Semantic Versioning](https://semver.org/) — see [RELEASING.md](RELEASING.md)
for what each part of a number promises and how a release is cut.

Write what changed under **Unreleased** as it lands, in the words a user would
use; `scripts/release.sh` turns that section into the next release.

## [Unreleased]

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

[Unreleased]: https://github.com/knoxxr/Kreate/commits/main
[0.1.0]: https://github.com/knoxxr/Kreate/commits/main
