#!/usr/bin/env bash
# Cuts a release: raises the version, dates the changelog, commits and tags.
#
#   scripts/release.sh patch|minor|major|X.Y.Z
#
# Pushes nothing. Afterwards: git push origin main --follow-tags
# See RELEASING.md.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

die() { echo "release: $*" >&2; exit 1; }

[[ $# -eq 1 ]] || die "usage: scripts/release.sh patch|minor|major|X.Y.Z"

# --- the tree it is cut from ---------------------------------------------
[[ "$(git branch --show-current)" == "main" ]] || die "release from main, not $(git branch --show-current)"
[[ -z "$(git status --porcelain)" ]] || die "the tree has changes; commit or stash them first"

# --- the numbers ------------------------------------------------------------
current=$(awk '/^\[workspace.package\]/{p=1; next} /^\[/{p=0} p && /^version *=/{gsub(/[" ]/, "", $3); print $3; exit}' Cargo.toml)
[[ "$current" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]] || die "cannot read the version in Cargo.toml (found '$current')"
major=${BASH_REMATCH[1]} minor=${BASH_REMATCH[2]} patch=${BASH_REMATCH[3]}

case "$1" in
  patch) next="$major.$minor.$((patch + 1))" ;;
  minor) next="$major.$((minor + 1)).0" ;;
  major) next="$((major + 1)).0.0" ;;
  *)
    [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || die "not a version: $1"
    next="$1"
    ;;
esac

# Strictly later, compared field by field.
newer=$(printf '%s\n%s\n' "$current" "$next" | sort -t. -k1,1n -k2,2n -k3,3n | tail -1)
[[ "$next" != "$current" && "$newer" == "$next" ]] || die "$next is not later than $current"
git rev-parse -q --verify "refs/tags/v$next" >/dev/null && die "the tag v$next already exists"

# --- the changelog ----------------------------------------------------------
# Something must be written under Unreleased: a release with no notes is a
# release nobody can tell apart from the last.
notes=$(awk '/^## \[Unreleased\]/{p=1; next} /^## \[/{p=0} p' CHANGELOG.md | sed '/^[[:space:]]*$/d')
[[ -n "$notes" ]] || die "nothing under ## [Unreleased] in CHANGELOG.md"

today=$(date +%Y-%m-%d)
repo="https://github.com/knoxxr/Kreate"
tmp=$(mktemp)
awk -v ver="$next" -v today="$today" -v repo="$repo" '
  /^## \[Unreleased\]$/ { print; print ""; print "## [" ver "] - " today; next }
  /^\[Unreleased\]: / { print "[Unreleased]: " repo "/compare/v" ver "...HEAD"; print "[" ver "]: " repo "/releases/tag/v" ver; next }
  { print }
' CHANGELOG.md > "$tmp"
mv "$tmp" CHANGELOG.md

# --- the version ------------------------------------------------------------
tmp=$(mktemp)
awk -v ver="$next" '
  /^\[workspace.package\]/ { p = 1 }
  /^\[/ && !/^\[workspace.package\]/ { p = 0 }
  p && /^version *=/ && !done { print "version = \"" ver "\""; done = 1; next }
  { print }
' Cargo.toml > "$tmp"
mv "$tmp" Cargo.toml
# The lock file carries every workspace crate's version; refresh only those.
cargo update --workspace --quiet

# --- commit and tag ---------------------------------------------------------
git add Cargo.toml Cargo.lock CHANGELOG.md
git commit --quiet -m "Release $next"
git tag -a "v$next" -m "Kreate $next" -m "$notes"

echo "Kreate $current -> $next, committed and tagged v$next."
echo "To publish: git push origin main --follow-tags"
