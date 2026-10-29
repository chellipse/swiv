use std::{
    cmp::Reverse,
    fmt::Debug,
    fs::{self, DirEntry},
    path::{Path, PathBuf},
    time::SystemTime,
};

use anyhow::{Result, anyhow};
use clap::Parser;
use itertools::Itertools;
use winit::event_loop::{ControlFlow, EventLoop};

use swiv::{app::App, font, raw::RawMode, tracing_format::CustomFormat};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Gallery,
    SingleImage,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum)]
pub enum SortOrder {
    /// Sort by modification time, newest first
    #[default]
    Mtime,
    /// Sort by status change time (Unix) or creation time (Windows), newest first
    Ctime,
    /// Sort by access time, newest first
    Atime,
    /// Sort by creation/birth time, newest first (filesystem-dependent)
    Btime,
    /// Sort lexically by filename (natural sort)
    Name,
    /// No sorting, filesystem order
    None,
}

#[derive(Debug, Clone, Parser)]
struct Cli {
    /// Can be a file or directory
    target: PathBuf,

    #[arg(short, long)]
    recursive: bool,

    /// How many levels of nesting to traverse if recursion is enabled
    #[arg(long, default_value_t = 16)]
    recursion_limit: u8,

    /// Sort order for files
    #[arg(long, value_enum, default_value_t = SortOrder::default())]
    sort_by: SortOrder,

    /// Ignore file extension during path filtering
    #[arg(long)]
    ignore_ext: bool,

    /// Do not ignore hidden files
    #[arg(long)]
    no_ignore: bool,

    /// Sample jemalloc's allocated bytes every 10ms and append them to this
    /// file as "wallclock-millisecond-ts,allocated-in-bytes" lines.
    /// The file is truncated at startup
    #[arg(long)]
    mem_usage_csv_path: Option<PathBuf>,

    /// Maximum memory in MiB a single image decode may allocate; images
    /// needing more fail to load. 0 disables the limit
    #[arg(long, default_value_t = 8192)]
    max_decode_mem: u64,

    /// Maximum GPU memory in MiB used for cached textures. 0 derives a
    /// default from total system memory (25%, clamped to 512..=4096)
    #[arg(long, default_value_t = 0)]
    max_gpu_mem: u64,

    /// Demosaic + develop the actual sensor data when viewing camera raws
    /// in single image mode, instead of showing the camera's embedded
    /// preview jpeg (thumbnails always use the embedded previews)
    #[arg(long)]
    raw_mode: bool,

    /// Font for the single image path bar: a fontconfig family name or a
    /// path to a font file
    #[arg(long, value_name = "FAMILY|PATH", default_value = font::DEFAULT_FAMILY)]
    bar_font: String,
}

impl Cli {
    fn get_paths(&self) -> Result<(Vec<PathBuf>, Option<usize>)> {
        let recursion =
            self.recursive.then_some(self.recursion_limit).unwrap_or(0);

        let (target, cursor_entry) = match self.target.metadata()?.is_dir() {
            true => (self.target.clone(), None),
            false => {
                // parent() of a bare filename ("foo.jpg") is Some(""),
                // which read_dir rejects — map it to "."
                let parent = match self.target.parent() {
                    Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
                    _ => PathBuf::from("."),
                };
                // rebuilt as parent/name so it compares equal to open_dir
                // output however the target was spelled ("foo.jpg" alone
                // would never match the "./foo.jpg" the walk yields)
                let name = self
                    .target
                    .file_name()
                    .ok_or(anyhow!("no file name in {:?}", self.target))?;
                let entry = parent.join(name);
                (parent, Some(entry))
            }
        };

        let mut iter: Box<dyn Iterator<Item = _>> = Box::new(
            Self::open_dir(&target, recursion, self.no_ignore)?
                .map(|entry| entry.path()),
        );

        if self.ignore_ext {
            #[rustfmt::skip]
            const EXTENSIONS: &[&str] = &[
                // formats image-rs supports
                "png", "jpg", "jpeg", "gif", "webp", "bmp", "ico", "tiff",
                "tif", "tga", "dds", "farbfeld", "ff", "pnm", "pbm", "pgm",
                "pam", "ppm", "hdr", "exr", "qoi", "avif",
                // via the psd backend
                "psd", "psb",
                // camera raws via the rawler backend
                "ari", "arw", "cr2", "cr3", "crm", "crw", "dcr", "dcs", "dng",
                "erf", "iiq", "kdc", "mef", "mos", "mrw", "nef", "nrw", "orf",
                "ori", "pef", "raf", "raw", "rw2", "rwl", "srw", "3fr", "fff",
                "x3f", "qtk",
            ];
            iter = Box::new(iter.filter(|path| {
                path.extension()
                    .and_then(|x| x.to_str())
                    .map(|x| x.to_lowercase()) // ignore case
                    .map(|x| EXTENSIONS.contains(&x.as_str()))
                    .unwrap_or(false)
            }));
        };

        iter = match self.sort_by {
            SortOrder::None => iter,
            SortOrder::Name => Box::new(iter.sorted_by(|a, b| {
                lexical_sort::natural_lexical_cmp(
                    &a.to_string_lossy(),
                    &b.to_string_lossy(),
                )
            })),
            time_sort => {
                let get_time = move |path: &PathBuf| -> Option<SystemTime> {
                    let meta = std::fs::metadata(path).ok()?;
                    match time_sort {
                        SortOrder::Mtime => meta.modified().ok(),
                        SortOrder::Atime => meta.accessed().ok(),
                        SortOrder::Ctime => {
                            #[cfg(unix)]
                            {
                                use std::os::unix::fs::MetadataExt;
                                Some(
                                    SystemTime::UNIX_EPOCH
                                        + std::time::Duration::from_secs(
                                            meta.ctime() as u64,
                                        ),
                                )
                            }
                            #[cfg(not(unix))]
                            {
                                meta.created().ok()
                            }
                        }
                        SortOrder::Btime => meta.created().ok(),
                        _ => unreachable!(),
                    }
                };
                Box::new(iter.sorted_by_key(|path| Reverse(get_time(path))))
            }
        };

        let paths: Vec<_> = iter.collect();

        let cursor_idx = cursor_entry
            .and_then(|x| paths.iter().enumerate().find(|(_, y)| x == **y))
            .map(|(idx, _)| idx);

        Ok((paths, cursor_idx))
    }

    fn open_dir(
        dir: impl AsRef<Path>,
        recursion: u8,
        no_ignore: bool,
    ) -> Result<Box<dyn Iterator<Item = DirEntry>>> {
        Ok(Box::new(
            fs::read_dir(dir.as_ref())?
                .flatten()
                .flat_map(move |entry| {
                    let path = entry.path();

                    // so we traverse symbolic links
                    let Ok(meta) = std::fs::metadata(&path) else {
                        // NOTE since std::fs::metadata chases symlinks, this will
                        // err on dead links
                        tracing::warn!("Failed to get metadata from: {path:?}");
                        return None;
                    };

                    match (
                        meta.file_type(),
                        recursion,
                        path.file_name()
                            .and_then(|x| {
                                x.as_encoded_bytes().get(0).map(|n| n == &b'.')
                            })
                            .unwrap_or(false)
                            <= no_ignore,
                    ) {
                        (ft, _, _) if ft.is_file() => {
                            let iter: Box<dyn Iterator<Item = DirEntry>> =
                                Box::new(std::iter::once(entry));
                            Some(iter)
                        }
                        (ft, r, true) if ft.is_dir() && r > 0 => {
                            Self::open_dir(
                                path,
                                recursion.saturating_sub(1),
                                no_ignore,
                            )
                            .ok()
                        }
                        // TODO make this traverse a set number of symlinks
                        _ => None,
                    }
                })
                .flatten(),
        ))
    }
}

/// Samples jemalloc's `stats.allocated` every 10ms into a csv of
/// "wallclock-millisecond-ts,allocated-in-bytes" lines.
///
/// Writes are deliberately unbuffered: the app terminates via
/// [`std::process::exit`], which would discard anything sitting in a
/// [`std::io::BufWriter`].
fn spawn_mem_logger(path: PathBuf) {
    use std::io::Write;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    std::thread::spawn(move || {
        let mut file = match std::fs::File::create(&path) {
            Ok(file) => file,
            Err(e) => {
                tracing::error!(
                    "Failed to create mem usage csv at {path:?}: {e}"
                );
                return;
            }
        };

        let (Ok(epoch), Ok(allocated)) = (
            tikv_jemalloc_ctl::epoch::mib(),
            tikv_jemalloc_ctl::stats::allocated::mib(),
        ) else {
            tracing::error!("Failed to look up jemalloc stat mibs");
            return;
        };

        loop {
            // jemalloc stats are cached and only refresh when the epoch is
            // advanced
            let _ = epoch.advance();

            let Ok(bytes) = allocated.read() else {
                tracing::error!("Failed to read jemalloc stats.allocated");
                return;
            };

            let ts = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis();

            if let Err(e) = writeln!(file, "{ts},{bytes}") {
                tracing::error!("Failed to write to mem usage csv: {e}");
                return;
            }

            std::thread::sleep(Duration::from_millis(10));
        }
    });
}

fn main() {
    let (stdout, guard) = tracing_appender::non_blocking(std::io::stdout());

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| {
                    tracing_subscriber::EnvFilter::new("error")
                }),
        )
        .with_writer(stdout)
        .event_format(CustomFormat::new())
        .init();

    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Wait);

    let cli = Cli::parse();

    if let Some(path) = cli.mem_usage_csv_path.clone() {
        spawn_mem_logger(path);
    }

    let (paths, cursor_idx) = match cli.get_paths() {
        Ok(x) => x,
        Err(e) => {
            tracing::error!("{e} from {:?}", cli.target);
            // Drop the guard to flush queued log messages before exiting;
            // process::exit skips destructors.
            drop(guard);
            std::process::exit(1);
        }
    };

    // the font loads off-thread, so a typo'd path is caught here rather
    // than quietly falling back
    if font::looks_like_path(&cli.bar_font)
        && !Path::new(&cli.bar_font).is_file()
    {
        tracing::error!("No font file at {:?}", cli.bar_font);
        drop(guard);
        std::process::exit(1);
    }

    let max_decode_alloc =
        (cli.max_decode_mem > 0).then(|| cli.max_decode_mem * 1024 * 1024);
    let gpu_budget =
        (cli.max_gpu_mem > 0).then(|| cli.max_gpu_mem * 1024 * 1024);
    let raw_mode = match cli.raw_mode {
        true => RawMode::Develop,
        false => RawMode::Preview,
    };
    let mut app = App::new(
        paths,
        cursor_idx,
        max_decode_alloc,
        gpu_budget,
        raw_mode,
        cli.bar_font,
    );
    event_loop.run_app(&mut app).unwrap();
}

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
