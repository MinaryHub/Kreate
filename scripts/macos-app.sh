#!/usr/bin/env bash
# Builds Kreate.app from a release build: the two programs side by side in
# Contents/MacOS (where the updater and the connector look for each other),
# and the icon from assets/kreate.ico, which the Dock and Finder read from the
# bundle — on macOS the window's own icon is not used.
#
#   scripts/macos-app.sh [destination folder, default /Applications]
#
# Signs it ad hoc: enough for this Mac, not for handing to another.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

die() { echo "macos-app: $*" >&2; exit 1; }

[[ "$(uname)" == "Darwin" ]] || die "builds a macOS bundle; run it on a Mac"
dest=${1:-/Applications}
app="$dest/Kreate.app"

version=$(awk '/^\[workspace.package\]/{p=1; next} /^\[/{p=0} p && /^version *=/{gsub(/[" ]/, "", $3); print $3; exit}' Cargo.toml)
[[ -n "$version" ]] || die "cannot read the version in Cargo.toml"

cargo build --release -q -p kreate -p kreate-mcp

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

# --- the icon: the .ico holds PNGs from 16 to 256 pixels ---------------------
python3 - "apps/kreate/assets/kreate.ico" "$work" <<'PY'
import struct, sys
data = open(sys.argv[1], "rb").read()
for i in range(struct.unpack("<H", data[4:6])[0]):
    w, _, _, _, _, _, size, off = struct.unpack("<BBBBHHII", data[6 + 16 * i:22 + 16 * i])
    png = data[off:off + size]
    if png[:4] == b"\x89PNG":
        open(f"{sys.argv[2]}/{w or 256}.png", "wb").write(png)
PY
set_dir="$work/kreate.iconset"
mkdir "$set_dir"
for size in 16 32 128 256 512; do
  for scale in 1 2; do
    px=$((size * scale))
    name="icon_${size}x${size}"
    [[ $scale == 2 ]] && name="${name}@2x"
    # The nearest size the .ico has at or above this one, or its largest.
    src=""
    for have in 16 24 32 48 64 128 256; do
      if [[ -f "$work/$have.png" && $have -ge $px ]]; then src="$work/$have.png"; break; fi
    done
    [[ -n "$src" ]] || src="$work/256.png"
    sips -s format png -z "$px" "$px" "$src" --out "$set_dir/$name.png" >/dev/null
  done
done
iconutil -c icns "$set_dir" -o "$work/kreate.icns"

# --- the bundle -----------------------------------------------------------
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
# Removed first, not written over: a running copy keeps its own file, where
# writing into it would get the running process killed for a bad signature.
rm -f "$app/Contents/MacOS/kreate" "$app/Contents/MacOS/kreate-mcp"
cp target/release/kreate target/release/kreate-mcp "$app/Contents/MacOS/"
cp "$work/kreate.icns" "$app/Contents/Resources/kreate.icns"
cat > "$app/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Kreate</string>
  <key>CFBundleDisplayName</key><string>Kreate</string>
  <key>CFBundleIdentifier</key><string>kr.smic.kreate</string>
  <key>CFBundleExecutable</key><string>kreate</string>
  <key>CFBundleIconFile</key><string>kreate</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$version</string>
  <key>CFBundleVersion</key><string>$version</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>LSApplicationCategoryType</key><string>public.app-category.graphics-design</string>
</dict>
</plist>
EOF
codesign --force --deep -s - "$app" 2>/dev/null
# Finder and the Dock keep icons they have seen; tell them this one changed.
touch "$app"
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$app"

echo "Kreate $version -> $app"
