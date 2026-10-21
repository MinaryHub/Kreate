# Releasing Kreate

## The number

Kreate's version is `MAJOR.MINOR.PATCH`, written once, in the workspace
`Cargo.toml` (`[workspace.package] version`). Every crate inherits it, and the
application reads it at build time — the About box, crash reports, messages
to support and the Claude connector all say the same number.

While the major number is `0`:

- **MINOR** goes up for anything a user would call new, and for anything that
  breaks what they had: a script line that no longer means the same thing, an
  operator renamed, a default changed.
- **PATCH** goes up for fixes that change nothing a user relied on.

From `1.0.0` on, the usual promise: MAJOR for breaking changes, MINOR for
additions, PATCH for fixes. Breaking means a document, a script, a keymap or
a workspace file written by the previous release stops meaning the same thing.

The `.kreate` file format has a version of its own, inside the file, so a file
can be read by releases that never heard of the one that wrote it. A release
that changes the format says so under **Changed** in the changelog.

The build also records the commit it came from (`KREATE_COMMIT`, with `-dirty`
if the tree had changes), so a report from a development build can be traced.

## Writing it down

Every change a user would notice gets a line under `## [Unreleased]` in
[CHANGELOG.md](CHANGELOG.md) in the same commit, under **Added**, **Changed**,
**Fixed** or **Removed**. Written for the user, not the reviewer.

## Cutting a release

From an up-to-date `main` with a clean tree:

```sh
scripts/release.sh minor        # or: patch, major, or an exact 0.3.0
git push origin main --follow-tags
```

The script checks the tree, raises the version in `Cargo.toml` and
`Cargo.lock`, turns **Unreleased** into the new version's section with today's
date, commits `Release X.Y.Z` and makes the annotated tag `vX.Y.Z`. It pushes
nothing; the push above does.

The tag starts the **Build binaries** workflow, which checks that the tag and
`Cargo.toml` agree, builds Kreate for Linux, macOS (Apple silicon and Intel)
and Windows, and publishes a GitHub release with the binaries and that
version's changelog section as its notes.

## Updates

Each platform's build is also published as `kreate-<platform>.zip` with
`kreate` and `kreate-mcp` at its top level. An installed Kreate asks GitHub
for the latest release when it starts, downloads that zip quietly when it is
newer, and puts it in place the next time it starts (`apps/kreate/src/update.rs`).

- The request is made without signing in; this repository,
  **MinaryHub/Kreate**, is public, so installed copies read its releases
  directly. `KREATE_UPDATE_REPO=owner/name` points a copy at another
  repository.
- Drafts and prereleases are never offered.
- A build run from a `target` folder never updates itself (set `KREATE_UPDATE=1`
  to try it); `KREATE_NO_UPDATE=1` turns updates off.

## When a release goes wrong

A tag that has been pushed is never moved. Fix forward: the fix goes under
**Unreleased** and becomes the next patch release.
