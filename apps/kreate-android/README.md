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

## Using it

- It draws with Vulkan (Android 8 and up, 64-bit ARM).
- **Fingers**:
  - one finger is the left mouse button: a tap clicks, a drag drags;
  - two fingers turn the view as they move and zoom as they pinch;
  - two fingers tapped together are a right click;
  - three fingers move the view.
- A mouse and keyboard work as on the desktop.
- **Files**:
  - Open brings up the system's file picker. The file chosen is copied
    into the app's `Documents` folder and opened.
  - Save writes into `Documents`, then the system's save screen offers to
    put a copy anywhere (Downloads, a cloud drive…).
  - File ▸ Open Recent works as usual.
- **Typing**: the keyboard on screen comes up when a text field is
  focused.

## Not yet

- Typing in Korean (or with any input method that composes letters):
  NativeActivity hands the program keys, not composed text.
- The clipboard (it is empty), and the exact (OCCT) kernel.
- A layout made for a phone's screen: it is the desktop's.
