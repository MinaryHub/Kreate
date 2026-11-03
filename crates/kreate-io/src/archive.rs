//! Reading entries out of a zip without letting the zip decide how much
//! memory that takes.
//!
//! Deflate compresses runs of zeros about a thousand to one, so a file of a
//! few megabytes can inflate to tens of gigabytes — and reading every entry
//! to its end, as a loader does, is then the whole of the attack. Each entry
//! is read only as far as it says it is long, and what all of them say is
//! counted against a budget before any is read.

use std::io::{self, Read};

/// What a whole archive may inflate to.
///
/// Far above any scene this application writes — a ten-million-vertex mesh
/// is a few hundred megabytes — and far below what would take a desktop
/// machine's memory with it.
pub const DEFAULT_BUDGET: u64 = 8 << 30;

/// A running total of what an archive's entries may still inflate to.
#[derive(Debug)]
pub struct Budget {
    left: u64,
}

impl Budget {
    pub fn new(bytes: u64) -> Self {
        Self { left: bytes }
    }

    /// Reads one entry whole, refusing one that says it is larger than what
    /// is left, and one that turns out larger than it says.
    pub fn read<R: Read + ?Sized>(
        &mut self,
        entry: &mut zip::read::ZipFile<'_, R>,
    ) -> io::Result<Vec<u8>> {
        let declared = entry.size();
        let name = entry.name().map(|n| n.into_owned()).unwrap_or_default();
        if declared > self.left {
            return Err(too_large(&name));
        }
        let mut bytes = Vec::new();
        entry.take(declared + 1).read_to_end(&mut bytes)?;
        if bytes.len() as u64 > declared {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{name} inflates past the size it gives"),
            ));
        }
        self.left -= bytes.len() as u64;
        Ok(bytes)
    }

    /// The same, as text.
    pub fn read_string<R: Read + ?Sized>(
        &mut self,
        entry: &mut zip::read::ZipFile<'_, R>,
    ) -> io::Result<String> {
        let bytes = self.read(entry)?;
        String::from_utf8(bytes).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

impl Default for Budget {
    fn default() -> Self {
        Self::new(DEFAULT_BUDGET)
    }
}

fn too_large(name: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("{name} would inflate past what an archive is allowed"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Write};

    fn zip_of(entries: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let mut out = Cursor::new(Vec::new());
        let mut w = zip::ZipWriter::new(&mut out);
        for (name, bytes) in entries {
            w.start_file(*name, zip::write::SimpleFileOptions::default())
                .unwrap();
            w.write_all(bytes).unwrap();
        }
        w.finish().unwrap();
        out.into_inner()
    }

    #[test]
    fn entries_within_the_budget_read_whole() {
        let bytes = zip_of(&[("a", vec![1; 100]), ("b", vec![2; 50])]);
        let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).unwrap();
        let mut budget = Budget::new(150);
        assert_eq!(
            budget.read(&mut zip.by_name("a").unwrap()).unwrap().len(),
            100
        );
        assert_eq!(
            budget.read(&mut zip.by_name("b").unwrap()).unwrap().len(),
            50
        );
    }

    #[test]
    fn an_archive_that_inflates_past_the_budget_is_refused() {
        // A megabyte of zeros compresses to a kilobyte or so: a small file
        // that would be a large read.
        let bytes = zip_of(&[("zeros", vec![0; 1 << 20])]);
        assert!(bytes.len() < 1 << 14);
        let mut zip = zip::ZipArchive::new(Cursor::new(bytes)).unwrap();
        let mut budget = Budget::new(1 << 16);
        assert!(budget.read(&mut zip.by_name("zeros").unwrap()).is_err());
    }
}
