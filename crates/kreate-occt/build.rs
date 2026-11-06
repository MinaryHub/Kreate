//! With the `occt` feature, compiles the C shim against an installed
//! OpenCASCADE and links its libraries dynamically (LGPL-2.1: they stay
//! replaceable files beside the program). Where OCCT is: `OCCT_INCLUDE` and
//! `OCCT_LIB`, or the usual places on Linux and macOS.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shim/occt_shim.cpp");
    println!("cargo:rerun-if-env-changed=OCCT_INCLUDE");
    println!("cargo:rerun-if-env-changed=OCCT_LIB");
    #[cfg(feature = "occt")]
    occt();
}

#[cfg(feature = "occt")]
fn occt() {
    use std::path::PathBuf;
    let first = |env: &str, guesses: &[&str]| -> Option<PathBuf> {
        std::env::var_os(env)
            .map(PathBuf::from)
            .or_else(|| guesses.iter().map(PathBuf::from).find(|p| p.exists()))
    };
    let include = first(
        "OCCT_INCLUDE",
        &[
            "/usr/include/opencascade",
            "/usr/local/include/opencascade",
            "/opt/homebrew/opt/opencascade/include/opencascade",
            "/usr/local/opt/opencascade/include/opencascade",
        ],
    )
    .expect("OpenCASCADE's headers not found: set OCCT_INCLUDE");
    let lib = first(
        "OCCT_LIB",
        &[
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/aarch64-linux-gnu",
            "/usr/local/lib",
            "/opt/homebrew/opt/opencascade/lib",
            "/usr/local/opt/opencascade/lib",
        ],
    )
    .expect("OpenCASCADE's libraries not found: set OCCT_LIB");
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .include(&include)
        .file("shim/occt_shim.cpp")
        .warnings(false)
        .compile("kreate_occt_shim");
    println!("cargo:rustc-link-search=native={}", lib.display());
    for name in [
        "TKernel",
        "TKMath",
        "TKG2d",
        "TKG3d",
        "TKGeomBase",
        "TKBRep",
        "TKGeomAlgo",
        "TKTopAlgo",
        "TKPrim",
        "TKBO",
        "TKBool",
        "TKFillet",
        "TKOffset",
        "TKShHealing",
        "TKMesh",
        "TKHLR",
        "TKSTEP",
        "TKSTEPBase",
        "TKSTEPAttr",
        "TKSTEP209",
        "TKXSBase",
    ] {
        println!("cargo:rustc-link-lib=dylib={name}");
    }
}
