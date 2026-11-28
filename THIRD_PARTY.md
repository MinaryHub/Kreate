# Third-Party Components & Licenses

Kreate itself is dual-licensed under **MIT OR Apache-2.0**. When built with
optional native accelerators, the following dynamically linked third-party
libraries may accompany release distributions:

## OpenCASCADE Technology (OCCT)
- **Feature flag**: `--features occt` (`crates/kreate-occt`)
- **License**: GNU Lesser General Public License v2.1 (LGPL-2.1) with the
  Open CASCADE Exception for dynamic linking.
- **Linkage**: Linked **dynamically** as separate shared libraries
  (`libTK*.so` / `libTK*.dylib` / `TK*.dll`) beside the Kreate executable so
  users may inspect or replace them in accordance with LGPL-2.1.
- **Source**: https://dev.opencascade.org/

## Intel Open Image Denoise (OIDN) & oneTBB
- **Loaded at runtime**: `crates/kreate-render/src/oidn.rs`
- **License**: Apache-2.0
- **Source**: https://www.openimagedenoise.org/
