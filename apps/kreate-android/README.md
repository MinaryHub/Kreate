# Kreate for Android

The desktop shell, built as `libkreate_android.so` and started by a
NativeActivity. `src/lib.rs` takes in `apps/kreate/src/main.rs` as a module;
only the start differs.

## Getting the APK

The **Android** workflow (`.github/workflows/android.yml`) builds it on every
push that touches the code, and by hand from the Actions tab. The APK is the
run's artifact, `kreate-android-arm64`. It is signed with a debug key: install
it by hand (allow installs from unknown sources), not through a store.

Building it yourself needs the Android SDK and NDK, and cargo-ndk:

```sh
rustup target add aarch64-linux-android
cargo install cargo-ndk
cargo ndk -t arm64-v8a --platform 26 --link-libcxx-shared \
  -o apps/kreate-android/gradle/app/src/main/jniLibs \
  build --release -p kreate-android
cd apps/kreate-android/gradle && gradle assembleRelease
```

## What works, and what does not yet

- It draws with Vulkan (Android 8 and up, 64-bit ARM).
- A mouse and keyboard work as on the desktop: a tablet with them attached,
  or Samsung DeX, is the way to use it now.
- One finger is the left mouse button. Orbiting, panning and zooming with
  fingers, and a keyboard on screen, are not there yet.
- There is no file dialog. Files are saved to the app's `Documents` folder
  (`Android/data/com.minaryhub.kreate/files/Documents`, visible over USB
  and in the Files app), and Open opens the newest file of the kind asked
  for there — put a file in that folder to open it. File ▸ Open Recent
  works as usual.
- A question that would be a dialog (save before closing?) is answered yes.
- The clipboard is empty, and the exact (OCCT) kernel is not built in.
