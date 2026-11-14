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
//! both (see there). The system's file picker it opens is the activity's
//! own Java (`KreateActivity`, in the Gradle project), reached here through
//! JNI; the file chosen comes back through `nativeFileArrived`.
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
    let picker = android.clone();
    kreate::android::set_picker(move |request| picker::open(&picker, request));
    let keys = android.clone();
    kreate::android::set_keyboard(move |show| {
        // Forced, not implicit: an implicit request is ignored with no
        // keyboard attached, which is when one is wanted.
        if show {
            keys.show_soft_input(false);
        } else {
            keys.hide_soft_input(false);
        }
    });
    let event_loop = EventLoop::builder()
        .with_android_app(android)
        .build()
        .expect("create event loop");
    app::run(event_loop, Vec::new());
}

/// The activity's file picker, through JNI.
mod picker {
    use std::ffi::{c_void, CStr, CString};
    use std::path::PathBuf;
    use std::ptr::null_mut;

    use jni_sys::{jclass, jobject, jstring, jvalue, JNIEnv, JavaVM, JNI_OK, JNI_VERSION_1_6};
    use kreate::android::Request;
    use winit::platform::android::activity::AndroidApp;

    /// This thread's JNI environment, attaching it to the VM if it is not.
    unsafe fn env(vm: *mut JavaVM) -> Option<*mut JNIEnv> {
        let mut env: *mut c_void = null_mut();
        if ((**vm).GetEnv?)(vm, &mut env, JNI_VERSION_1_6) == JNI_OK {
            return Some(env.cast());
        }
        if ((**vm).AttachCurrentThread?)(vm, &mut env, null_mut()) == JNI_OK {
            return Some(env.cast());
        }
        None
    }

    unsafe fn string(env: *mut JNIEnv, s: &str) -> jstring {
        let c = CString::new(s.replace('\0', "")).unwrap_or_default();
        ((**env).NewStringUTF.expect("NewStringUTF"))(env, c.as_ptr())
    }

    /// `KreateActivity.<name>(<args>)`, its exception (if any) cleared.
    unsafe fn call(
        app: &AndroidApp,
        name: &CStr,
        sig: &CStr,
        args: impl FnOnce(*mut JNIEnv) -> Vec<jvalue>,
    ) {
        let vm = app.vm_as_ptr() as *mut JavaVM;
        let Some(env) = env(vm) else {
            return;
        };
        let f = &**env;
        let activity = app.activity_as_ptr() as jobject;
        let class = (f.GetObjectClass.expect("GetObjectClass"))(env, activity);
        let method = (f.GetMethodID.expect("GetMethodID"))(env, class, name.as_ptr(), sig.as_ptr());
        if method.is_null() {
            (f.ExceptionClear.expect("ExceptionClear"))(env);
        } else {
            let args = args(env);
            (f.CallVoidMethodA.expect("CallVoidMethodA"))(env, activity, method, args.as_ptr());
            for a in &args {
                (f.DeleteLocalRef.expect("DeleteLocalRef"))(env, a.l);
            }
        }
        if (f.ExceptionCheck.expect("ExceptionCheck"))(env) != 0 {
            (f.ExceptionDescribe.expect("ExceptionDescribe"))(env);
            (f.ExceptionClear.expect("ExceptionClear"))(env);
        }
        (f.DeleteLocalRef.expect("DeleteLocalRef"))(env, class);
    }

    /// The system's picker opened for `request`. It answers later, on the
    /// activity's thread: see `nativeFileArrived`.
    pub fn open(app: &AndroidApp, request: Request) {
        unsafe {
            match request {
                Request::Open { .. } => call(app, c"openDocument", c"()V", |_| Vec::new()),
                Request::Save { path, name } => call(
                    app,
                    c"createDocument",
                    c"(Ljava/lang/String;Ljava/lang/String;)V",
                    |env| {
                        vec![
                            jvalue {
                                l: string(env, &path.to_string_lossy()),
                            },
                            jvalue {
                                l: string(env, &name),
                            },
                        ]
                    },
                ),
            }
        }
    }

    /// `KreateActivity.nativeFileArrived(path)`: a file the picker chose,
    /// copied into the app's Documents.
    #[no_mangle]
    pub unsafe extern "system" fn Java_com_minaryhub_kreate_KreateActivity_nativeFileArrived(
        env: *mut JNIEnv,
        _class: jclass,
        path: jstring,
    ) {
        let f = &**env;
        let chars = (f.GetStringUTFChars.expect("GetStringUTFChars"))(env, path, null_mut());
        if chars.is_null() {
            return;
        }
        let text = CStr::from_ptr(chars).to_string_lossy().into_owned();
        (f.ReleaseStringUTFChars.expect("ReleaseStringUTFChars"))(env, path, chars);
        kreate::android::file_arrived(PathBuf::from(text));
    }
}
