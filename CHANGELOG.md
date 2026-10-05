# Changelog

Every release of Kreate, newest first. The format is
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the numbers follow
[Semantic Versioning](https://semver.org/) — see [RELEASING.md](RELEASING.md)
for what each part of a number promises and how a release is cut.

Write what changed under **Unreleased** as it lands, in the words a user would
use; `scripts/release.sh` turns that section into the next release.

## [Unreleased]

### Added

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

- A `set_*` line that leaves a parameter out keeps its current value instead of
  resetting it to the default.
- New objects never take a name in use: a second `Cube` is `Cube.001`.
- `get_scene` reports colours in sRGB, as the operators take them, and gives
  uids, dimensions, bounds, images and the view's exposure.

### Fixed

- A duplicate keeps its materials.
- A viewport capture taken while the window was hidden came back empty.
- `material.link` works on an object with no material slots yet.

## [0.1.0]

The first version: the editor, its operators, the renderer, files and the
Claude connector, before releases were numbered.

[Unreleased]: https://github.com/knoxxr/Kreate/commits/main
[0.1.0]: https://github.com/knoxxr/Kreate/commits/main
