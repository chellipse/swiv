use std::{
    fs,
    path::{Path, PathBuf},
    thread,
    time::{Duration, SystemTime},
};

use anyhow::{Result, anyhow};
use image::DynamicImage;

use crate::cache::LADDER;

/// Bump to invalidate all cached thumbnails whenever the decode pipeline
/// changes in a way that alters output pixels (filters, ICC handling, new
/// backends). The janitor removes directories of other versions.
/// v2: EXIF orientation is now applied
const CACHE_VERSION: u32 = 2;

const DISK_BUDGET_BYTES: u64 = 1024 * 1024 * 1024;

/// On-disk thumbnail cache, used only by the loader worker threads.
///
/// Layout: `$XDG_CACHE_HOME/swiv/thumbs/v<version>/<bucket>/<name>.qoi`
/// where `<name>` encodes the canonical source path (hashed), its mtime and
/// its size — so a modified source simply misses and the stale entry ages
/// out via the janitor. Entries hold post-ICC sRGB pixels, so hits skip the
/// color transform too. QOI keeps encode/decode near-memcpy speed.
#[derive(Debug, Clone)]
pub struct ThumbDiskCache {
    root: PathBuf,
}

impl ThumbDiskCache {
    pub fn new() -> Result<Self> {
        let cache_home = std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .filter(|p| p.is_absolute())
            .or_else(|| {
                std::env::var_os("HOME")
                    .map(|h| PathBuf::from(h).join(".cache"))
            })
            .ok_or(anyhow!("neither XDG_CACHE_HOME nor HOME is set"))?;

        let root = cache_home
            .join(env!("CARGO_PKG_NAME"))
            .join("thumbs")
            .join(format!("v{CACHE_VERSION}"));

        for bucket in LADDER {
            fs::create_dir_all(root.join(bucket.to_string()))?;
        }

        Ok(Self { root })
    }

    /// Stable filename identifying (source file, content identity)
    fn entry_name(src: &Path) -> Option<String> {
        let canonical = src.canonicalize().ok()?;
        let meta = fs::metadata(&canonical).ok()?;
        let mtime = meta
            .modified()
            .ok()?
            .duration_since(SystemTime::UNIX_EPOCH)
            .ok()?
            .as_secs();

        Some(format!(
            "{:016x}-{mtime}-{}.qoi",
            fnv1a64(canonical.as_os_str().as_encoded_bytes()),
            meta.len(),
        ))
    }

    fn entry_path(&self, src: &Path, bucket: u32) -> Option<PathBuf> {
        Some(
            self.root
                .join(bucket.to_string())
                .join(Self::entry_name(src)?),
        )
    }

    pub fn load(&self, src: &Path, bucket: u32) -> Option<DynamicImage> {
        let path = self.entry_path(src, bucket)?;
        let img = image::open(&path).ok()?;

        // touch so the janitor's lru ordering reflects usage, not creation
        let _ = fs::File::options()
            .write(true)
            .open(&path)
            .and_then(|f| f.set_modified(SystemTime::now()));

        Some(img)
    }

    /// The smallest cached thumb of `src` larger than `bucket`, for building
    /// new thumbs without re-decoding the original
    pub fn load_larger(&self, src: &Path, bucket: u32) -> Option<DynamicImage> {
        LADDER
            .into_iter()
            .filter(|b| *b > bucket)
            .find_map(|b| self.load(src, b))
    }

    /// Best effort: failures are logged and swallowed, the thumb will just
    /// be regenerated next run
    pub fn store(
        &self,
        src: &Path,
        bucket: u32,
        width: u32,
        height: u32,
        bytes: &[u8],
    ) {
        let Some(path) = self.entry_path(src, bucket) else {
            return;
        };

        // write-then-rename so concurrent readers (and crashes) never see a
        // partial file; pid-suffixed so concurrent swiv instances don't
        // clobber each other's writes
        let tmp = path.with_extension(format!("{}.tmp", std::process::id()));
        let result = image::save_buffer_with_format(
            &tmp,
            bytes,
            width,
            height,
            image::ExtendedColorType::Rgba8,
            image::ImageFormat::Qoi,
        )
        .map_err(anyhow::Error::from)
        .and_then(|()| Ok(fs::rename(&tmp, &path)?));

        if let Err(e) = result {
            tracing::warn!("Failed to store thumb for {src:?}: {e:?}");
            let _ = fs::remove_file(&tmp);
        }
    }

    /// One pass at startup: drops other cache versions, cleans up crashed
    /// writes, and evicts least-recently-used entries down to the budget
    pub fn spawn_janitor(&self) {
        let root = self.root.clone();
        thread::spawn(move || {
            if let Err(e) = janitor(&root) {
                tracing::warn!("Thumb cache janitor failed: {e:?}");
            }
        });
    }
}

fn janitor(root: &Path) -> Result<()> {
    // remove caches of other versions entirely
    if let Some(parent) = root.parent() {
        for entry in fs::read_dir(parent)?.flatten() {
            if Some(entry.file_name().as_os_str()) != root.file_name() {
                tracing::debug!(
                    "Removing stale cache version: {:?}",
                    entry.path()
                );
                let _ = fs::remove_dir_all(entry.path());
            }
        }
    }

    let mut files = Vec::new();
    let mut total: u64 = 0;
    for bucket in LADDER {
        let Ok(entries) = fs::read_dir(root.join(bucket.to_string())) else {
            continue;
        };

        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(meta) = entry.metadata() else { continue };
            let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);

            // leftover partial writes from crashed runs; the age guard
            // avoids racing a concurrently running instance
            if path.extension().is_some_and(|e| e == "tmp") {
                if mtime.elapsed().unwrap_or_default()
                    > Duration::from_secs(24 * 60 * 60)
                {
                    let _ = fs::remove_file(&path);
                }
                continue;
            }

            total += meta.len();
            files.push((path, meta.len(), mtime));
        }
    }

    tracing::debug!(
        "Thumb cache: {} entries, {} MB",
        files.len(),
        total / 1_000_000
    );

    if total <= DISK_BUDGET_BYTES {
        return Ok(());
    }

    // evict least-recently-used first; loads touch mtime
    files.sort_by_key(|(_, _, mtime)| *mtime);
    for (path, len, _) in files {
        if total <= DISK_BUDGET_BYTES {
            break;
        }
        if fs::remove_file(&path).is_ok() {
            total -= len;
        }
    }

    Ok(())
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
