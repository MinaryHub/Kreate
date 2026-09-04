//! Tier two crash reporting: the dump writer that survives the crash.
//!
//! # Why a second process at all
//!
//! The panic hook in `main.rs` covers a Rust panic, which is the common case and
//! the one where the process is still healthy enough to write a file. It does not
//! cover the crashes §3.4 singles out:
//!
//! - a **GPU fault** — a Windows TDR resets the device and the process usually
//!   dies with an access violation, not a panic;
//! - a **stack overflow** — there is no stack left to run a hook on;
//! - an **abort** — `panic = "abort"`, a double panic, or an allocator failure
//!   skips unwinding entirely.
//!
//! In all three the crashing process cannot be trusted to write anything: its
//! heap may be corrupt, its stack may be gone, and running arbitrary code from a
//! signal or exception handler is how a crash report becomes a second crash. So
//! the dump is written by a **different process**, which is alive and healthy and
//! merely reads the crashed one's memory.
//!
//! # Why the same executable
//!
//! The reporter is this same binary re-invoked with [`REPORTER_FLAG`]. A separate
//! `kreate-crash-reporter.exe` would be a second artifact to build, sign, ship,
//! install, and keep in step with the version of the application it is reporting
//! for — and the failure mode of getting that wrong is a crash reporter that does
//! not run, discovered only when a crash report is needed.
//!
//! # What is deliberately not done
//!
//! Nothing is uploaded anywhere. The dump is written next to the autosaves and
//! the user is told where. Sending a memory dump of somebody's work to a server
//! is a decision for them to make explicitly, not something a modelling tool does
//! on their behalf the first time it falls over.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

/// The argument that turns this executable into the dump writer.
pub const REPORTER_FLAG: &str = "--crash-reporter";

/// A handle that keeps the reporter alive.
///
/// Held for the life of the process. Dropping it detaches the platform handler
/// and lets the reporter exit, which is the right thing on a clean shutdown and
/// which is why this is returned rather than leaked.
pub struct Reporter {
    _handler: crash_handler::CrashHandler,
    child: std::process::Child,
}

impl Drop for Reporter {
    fn drop(&mut self) {
        // The handler is detached by its own `Drop` before this runs. Killing
        // the reporter rather than waiting for it: it is blocked in a message
        // loop that only exits on a dump or a disconnect, and on a clean
        // shutdown there is nothing for it to report.
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Where dumps and reports go. Beside the autosaves, for the same reasons.
fn report_dir() -> Option<PathBuf> {
    crate::docfile::autosave_root()
}

/// Starts the reporter process and installs the platform crash handler.
///
/// Returns `None` when either half could not be set up. That is not fatal and
/// must not be: an application that refuses to start because its crash reporter
/// would not start has turned a diagnostic into an outage.
pub fn install() -> Option<Reporter> {
    let dir = report_dir()?;
    std::fs::create_dir_all(&dir).ok()?;

    // Named for this process, so two running instances do not fight over one
    // socket. On Windows the name is still a path; the crate takes care of the
    // platform difference.
    let socket = dir.join(format!("crash-{}.sock", std::process::id()));
    let _ = std::fs::remove_file(&socket);

    let executable = std::env::current_exe().ok()?;
    let child = std::process::Command::new(executable)
        .arg(REPORTER_FLAG)
        .arg(&socket)
        .spawn()
        .ok()?;

    // The server needs to be listening before the client can connect, and there
    // is no handshake to wait on. Retried rather than slept once: a fixed sleep
    // is either too short on a loaded machine or wasted on an idle one.
    let mut client = None;
    for _ in 0..50 {
        if let Ok(c) = minidumper::Client::with_name(minidumper::SocketName::Path(&socket)) {
            client = Some(c);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let client = client?;
    let _ = client.send_message(MESSAGE_VERSION, env!("CARGO_PKG_VERSION"));

    // SAFETY: the closure runs from a crash handler — a vectored exception
    // handler on Windows, a signal handler elsewhere — so it must do as little
    // as possible and must not allocate or take locks. `request_dump` writes to
    // an already-connected socket and blocks until the reporter is finished,
    // which is the whole contract this crate exists to provide.
    let handler = crash_handler::CrashHandler::attach(unsafe {
        crash_handler::make_crash_event(move |context: &crash_handler::CrashContext| {
            crash_handler::CrashEventResult::Handled(client.request_dump(context).is_ok())
        })
    })
    .ok()?;

    Some(Reporter {
        _handler: handler,
        child,
    })
}

/// Message kinds the client sends. Only one so far.
const MESSAGE_VERSION: u32 = 1;

/// Runs the dump writer. Called when this executable is invoked with the flag.
///
/// Never returns to the caller's normal path: this process exists only to wait
/// for its parent to crash.
pub fn run_reporter(socket: &Path) -> ! {
    let mut server = match minidumper::Server::with_name(minidumper::SocketName::Path(socket)) {
        Ok(server) => server,
        Err(e) => {
            eprintln!(
                "crash reporter could not listen on {}: {e}",
                socket.display()
            );
            std::process::exit(1);
        }
    };
    let dir = socket
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    // The dump is named after the socket, which `install` names after the
    // process being watched. The reporter cannot use its *own* process id — an
    // early version did, and produced files named after the wrong process, which
    // is worse than a timestamp because it looks like an answer.
    let subject = socket
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "crash".to_string());
    let shutdown = AtomicBool::new(false);
    let handler = Handler { dir, subject };
    // A stale timeout, so a reporter whose parent vanished without disconnecting
    // — killed from a task manager, say — exits instead of living forever.
    let result = server.run(
        Box::new(handler),
        &shutdown,
        Some(std::time::Duration::from_secs(5)),
    );
    if let Err(e) = result {
        eprintln!("crash reporter stopped: {e}");
    }
    let _ = std::fs::remove_file(socket);
    std::process::exit(0);
}

struct Handler {
    dir: PathBuf,
    /// The watched process's socket stem, used to name its dump.
    subject: String,
}

impl minidumper::ServerHandler for Handler {
    fn create_minidump_file(&self) -> Result<(std::fs::File, PathBuf), std::io::Error> {
        std::fs::create_dir_all(&self.dir)?;
        // Named after the crashing process, not by a timestamp: a user reporting
        // a problem can say which one, and two crashes on one machine do not
        // overwrite each other the way `crash.dmp` would.
        let path = self.dir.join(format!("{}.dmp", self.subject));
        let file = std::fs::File::create(&path)?;
        Ok((file, path))
    }

    fn on_minidump_created(
        &self,
        result: Result<minidumper::MinidumpBinary, minidumper::Error>,
    ) -> minidumper::LoopAction {
        match result {
            Ok(binary) => {
                // Flushed explicitly. The parent is about to be gone and nothing
                // else will close this file; a dump that is still in the page
                // cache when the machine goes down is not a dump.
                let mut file = binary.file;
                let _ = std::io::Write::flush(&mut file);
                let _ = file.sync_all();
                eprintln!("minidump written to {}", binary.path.display());
            }
            Err(e) => eprintln!("minidump failed: {e}"),
        }
        // Exit after one. The process being reported on is dead, so there will
        // not be a second request, and staying alive would leave a stray process
        // behind after every crash.
        minidumper::LoopAction::Exit
    }

    fn on_message(&self, kind: u32, buffer: Vec<u8>) {
        if kind == MESSAGE_VERSION {
            // Recorded beside the dump rather than only printed: a dump is
            // useless without knowing which build it came from, and the reporter
            // cannot ask the crashed process afterwards.
            let text = String::from_utf8_lossy(&buffer).to_string();
            let _ = std::fs::create_dir_all(&self.dir);
            let _ = std::fs::write(self.dir.join("crash-build.txt"), text);
        }
    }
}

/// Deliberately crashes, to verify that the reporter works.
///
/// Debug builds only. A crash reporter that has never been made to fire is a
/// crash reporter nobody knows the state of, and the failure mode — no dump when
/// one is finally needed — is discovered at the worst possible time. So there is
/// a way to fire it on purpose.
///
/// Behind `debug_assertions` so it cannot exist in a shipped build. A release
/// binary with a documented "crash now" switch is a liability, and the thing
/// being verified is the plumbing, which is the same in both.
#[cfg(debug_assertions)]
pub fn crash_on_purpose() -> ! {
    eprintln!("crashing on purpose to exercise the crash reporter");
    // A null read, which raises an access violation on Windows and SIGSEGV
    // elsewhere. Chosen over `abort()` because it is the shape of the crashes
    // this tier exists for: a fault the process cannot handle, not a controlled
    // exit.
    //
    // SAFETY: none. Reading address zero is the entire point.
    unsafe {
        let nowhere: *const u32 = std::ptr::null();
        std::hint::black_box(std::ptr::read_volatile(nowhere));
    }
    // Unreachable in practice; the handler does not return.
    std::process::abort();
}

/// Parses the reporter flag out of the command line.
///
/// Returns the socket path when this process should be the reporter.
pub fn reporter_socket(args: &[String]) -> Option<PathBuf> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == REPORTER_FLAG {
            return iter.next().map(PathBuf::from);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_flag_is_recognised_with_its_socket() {
        let args = vec![REPORTER_FLAG.to_string(), "/tmp/crash.sock".to_string()];
        assert_eq!(
            reporter_socket(&args),
            Some(PathBuf::from("/tmp/crash.sock"))
        );
    }

    #[test]
    fn an_ordinary_launch_is_not_the_reporter() {
        // The consequence of getting this wrong is that a normal launch becomes a
        // process that silently waits for a socket and never shows a window.
        assert!(reporter_socket(&[]).is_none());
        assert!(reporter_socket(&["model.kreate".to_string()]).is_none());
        assert!(reporter_socket(&["--lang".to_string(), "ko".to_string()]).is_none());
    }

    #[test]
    fn the_flag_without_a_socket_is_not_the_reporter() {
        // Rather than defaulting to some path: a reporter listening somewhere
        // nobody is connecting to is a process that never exits.
        assert!(reporter_socket(&[REPORTER_FLAG.to_string()]).is_none());
    }

    #[test]
    fn the_flag_is_found_after_other_arguments() {
        let args = vec![
            "--lang".to_string(),
            "ko".to_string(),
            REPORTER_FLAG.to_string(),
            "sock".to_string(),
        ];
        assert_eq!(reporter_socket(&args), Some(PathBuf::from("sock")));
    }
}
