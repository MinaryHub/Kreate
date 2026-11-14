//! Kreate on Android: the desktop shell, started by a NativeActivity.
//!
//! The shell is the same source as the desktop application's — its
//! `main.rs` is this library's `app` module — so Android runs what the
//! desktop runs. What differs is only the start: winit is handed the
//! activity, and the directories a Linux desktop keeps a user's settings,
//! caches and files in are pointed at the ones Android gives the app.
//!
//! Android has no file dialog a program can call and wait on, and no
//! clipboard the desktop crates reach; `kreate::android` stands in for
//! both (see there).
#![cfg(target_os = "android")]

#[path = "../../kreate/src/main.rs"]
#[allow(dead_code, unused_imports)]
mod app;

use winit::event_loop::EventLoop;
use winit::platform::android::activity::AndroidApp;
use winit::platform::android::EventLoopBuilderExtAndroid;

/// Where NativeActivity starts the program.
#[no_mangle]
fn android_main(android: AndroidApp) {
    // The app's own storage as the home the shell's paths are made from:
    // settings in `$XDG_CONFIG_HOME/kreate`, autosave and caches under
    // `$XDG_CACHE_HOME`, files the person saves under `Documents`. The
    // external directory is the one a computer sees over USB and the Files
    // app shows, so a saved model can be taken off the device.
    let internal = android.internal_data_path();
    let external = android.external_data_path();
    if let Some(home) = external.clone().or_else(|| internal.clone()) {
        std::env::set_var("HOME", &home);
        std::env::set_var("KREATE_DOCUMENTS", home.join("Documents"));
        let _ = std::fs::create_dir_all(home.join("Documents"));
    }
    if let Some(dir) = internal.or(external) {
        std::env::set_var("XDG_CONFIG_HOME", dir.join("config"));
        std::env::set_var("XDG_CACHE_HOME", dir.join("cache"));
        std::env::set_var("XDG_DATA_HOME", dir.join("data"));
    }
    kreate::logs::init();
    app::install_panic_hook();
    let event_loop = EventLoop::builder()
        .with_android_app(android)
        .build()
        .expect("create event loop");
    app::run(event_loop, Vec::new());
}
