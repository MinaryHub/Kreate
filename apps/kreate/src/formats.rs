//! The file formats File ▸ Import and File ▸ Export offer one by one, each
//! with the extensions it goes by.
//!
//! One table for the menus and the file dialogs, so a format the readers
//! learn is offered in both places or in neither.

/// A format: the name people know it by, and its extensions (the first is
/// the one a new file gets).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Format {
    pub name: &'static str,
    pub extensions: &'static [&'static str],
}

impl Format {
    /// "glTF (.gltf, .glb)": for a menu.
    pub fn label(&self) -> String {
        let exts: Vec<String> = self.extensions.iter().map(|e| format!(".{e}")).collect();
        format!("{} ({})", self.name, exts.join(", "))
    }
}

/// What Import reads, in the order the menu offers it.
pub const IMPORT: &[Format] = &[
    Format {
        name: "glTF",
        extensions: &["glb", "gltf"],
    },
    Format {
        name: "FBX",
        extensions: &["fbx"],
    },
    Format {
        name: "Wavefront OBJ",
        extensions: &["obj"],
    },
    Format {
        name: "STL",
        extensions: &["stl"],
    },
    Format {
        name: "PLY",
        extensions: &["ply"],
    },
    Format {
        name: "3MF",
        extensions: &["3mf"],
    },
    Format {
        name: "OFF",
        extensions: &["off"],
    },
    Format {
        name: "USD",
        extensions: &["usd", "usda", "usdc", "usdz"],
    },
    Format {
        name: "COLLADA",
        extensions: &["dae"],
    },
    Format {
        name: "URDF (robot)",
        extensions: &["urdf"],
    },
    Format {
        name: "STEP (CAD)",
        extensions: &["step", "stp"],
    },
    Format {
        name: "DXF (2D drawing)",
        extensions: &["dxf"],
    },
];

/// What Export writes, in the order the menu offers it.
pub const EXPORT: &[Format] = &[
    Format {
        name: "glTF binary",
        extensions: &["glb"],
    },
    Format {
        name: "glTF",
        extensions: &["gltf"],
    },
    Format {
        name: "Wavefront OBJ",
        extensions: &["obj"],
    },
    Format {
        name: "STL",
        extensions: &["stl"],
    },
    Format {
        name: "3MF (3D printing)",
        extensions: &["3mf"],
    },
    Format {
        name: "STEP (CAD, faceted)",
        extensions: &["step", "stp"],
    },
    Format {
        name: "Parts list (CSV)",
        extensions: &["csv"],
    },
    Format {
        name: "USD (text)",
        extensions: &["usda"],
    },
    Format {
        name: "USD (binary)",
        extensions: &["usdc", "usd"],
    },
    Format {
        name: "USDZ (one file, for AR)",
        extensions: &["usdz"],
    },
    Format {
        name: "URDF (robot)",
        extensions: &["urdf"],
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn import_offers_exactly_what_the_readers_read() {
        let mut offered: Vec<&str> = IMPORT
            .iter()
            .flat_map(|f| f.extensions.iter().copied())
            .collect();
        let mut read: Vec<&str> = kreate_io::scene_file::SCENE_FORMATS.to_vec();
        offered.sort_unstable();
        read.sort_unstable();
        assert_eq!(offered, read);
    }

    #[test]
    fn a_label_names_its_extensions() {
        assert_eq!(IMPORT[0].label(), "glTF (.glb, .gltf)");
    }
}
