//! The application's log, kept on disk: what was done and what went wrong,
//! for working out afterwards why the window stopped answering or closed.
//!
//! One file a day, `logs/kreate-YYYY-MM-DD.log` in the settings folder,
//! appended to by every run that day; files older than [`KEEP_DAYS`] are
//! removed when the application starts. What goes in: every operator run
//! and how long it took, anything slower than a second said as such, and
//! every warning and error — the same lines the console shows.

use std::path::{Path, PathBuf};

/// How many days of logs are kept.
pub const KEEP_DAYS: u64 = 7;

/// The folder the logs are written to.
pub fn log_dir() -> Option<PathBuf> {
    crate::prefs::config_dir().map(|d| d.join("logs"))
}

/// Days since 1970-01-01 as a calendar date (UTC): the proleptic Gregorian
/// conversion, Howard Hinnant's, so no date library is needed for a file name.
pub fn civil_date(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d)
}

/// Today, as days since 1970, UTC.
pub fn today() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| (d.as_secs() / 86_400) as i64)
        .unwrap_or(0)
}

/// The day's file name.
pub fn file_name(days: i64) -> String {
    let (y, m, d) = civil_date(days);
    format!("kreate-{y:04}-{m:02}-{d:02}.log")
}

/// The day a log file's name says, as days since 1970.
fn day_of(name: &str) -> Option<i64> {
    let date = name.strip_prefix("kreate-")?.strip_suffix(".log")?;
    let mut parts = date.split('-').map(|p| p.parse::<i64>().ok());
    let (y, m, d) = (parts.next()??, parts.next()??, parts.next()??);
    // Back from a date to days: by search near the estimate, which is plenty
    // for a handful of files a start.
    let guess = (y - 1970) * 365 + (y - 1969) / 4 + (m - 1) * 30 + d;
    (guess - 40..guess + 40).find(|g| civil_date(*g) == (y, m as u32, d as u32))
}

/// Removes the logs in `dir` from before the last `keep` days counting
/// `today`; what it removed.
pub fn prune(dir: &Path, today: i64, keep: u64) -> Vec<PathBuf> {
    let mut gone = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return gone;
    };
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if let Some(day) = day_of(&name) {
            if today - day >= keep as i64 && std::fs::remove_file(e.path()).is_ok() {
                gone.push(e.path());
            }
        }
    }
    gone
}

/// Starts logging to the console and to today's file. The file is best
/// effort: without a settings folder the console still has it all.
pub fn init() {
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    use tracing_subscriber::Layer;
    let console_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "kreate=info,kreate_render=info".into());
    let console = tracing_subscriber::fmt::layer().with_filter(console_filter);
    let file = log_dir().and_then(|dir| {
        std::fs::create_dir_all(&dir).ok()?;
        prune(&dir, today(), KEEP_DAYS);
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join(file_name(today())))
            .ok()
    });
    let file_layer = file.map(|f| {
        tracing_subscriber::fmt::layer()
            .with_ansi(false)
            .with_writer(std::sync::Mutex::new(f))
            .with_filter(tracing_subscriber::EnvFilter::new(
                "warn,kreate=info,kreate_ops=info,kreate_render=info",
            ))
    });
    tracing_subscriber::registry()
        .with(console)
        .with(file_layer)
        .init();
}

/// Opens `path` in the system's file browser.
pub fn reveal(path: &Path) {
    let _ = std::fs::create_dir_all(path);
    let program = if cfg!(windows) {
        "explorer.exe"
    } else if cfg!(target_os = "macos") {
        "open"
    } else {
        "xdg-open"
    };
    let _ = crate::claude::quiet_command(program).arg(path).spawn();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_round_trip() {
        assert_eq!(civil_date(0), (1970, 1, 1));
        assert_eq!(civil_date(20_376), (2025, 10, 15));
        assert_eq!(civil_date(11_016), (2000, 2, 29));
        assert_eq!(file_name(20_376), "kreate-2025-10-15.log");
        for day in [0, 59, 60, 11_016, 20_376, 30_000] {
            assert_eq!(day_of(&file_name(day)), Some(day));
        }
    }

    /// A week kept: the files from before it go, today's and the rest stay,
    /// and what is not a log is left alone — and what was written is there
    /// to read by the next start.
    #[test]
    fn old_logs_are_removed_and_recent_ones_kept() {
        let dir = std::env::temp_dir().join(format!("kreate-log-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let today = 20_376;
        for back in [0, 3, 6, 7, 30] {
            std::fs::write(dir.join(file_name(today - back)), format!("day -{back}\n")).unwrap();
        }
        std::fs::write(dir.join("notes.txt"), "mine").unwrap();
        let gone = prune(&dir, today, KEEP_DAYS);
        assert_eq!(gone.len(), 2, "{gone:?}");
        let mut left: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        left.sort();
        assert_eq!(
            left,
            vec![
                file_name(today - 6),
                file_name(today - 3),
                file_name(today),
                "notes.txt".to_string()
            ]
        );
        assert_eq!(
            std::fs::read_to_string(dir.join(file_name(today))).unwrap(),
            "day -0\n"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
