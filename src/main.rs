use std::{
    fmt::Debug,
    fs::{self, DirEntry},
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Result};
use clap::Parser;
use itertools::Itertools;
use winit::event_loop::{ControlFlow, EventLoop};

use swiv::{app::App, tracing_format::CustomFormat};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Gallery,
    SingleImage,
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

    /// Do not perform a lexical sort over all paths
    #[arg(long)]
    no_sort: bool,

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
}

impl Cli {
    fn get_paths(&self) -> Result<(Vec<PathBuf>, Option<usize>)> {
        let recursion = self.recursive.then_some(self.recursion_limit).unwrap_or(0);

        let (target, cursor_entry) = match self.target.metadata()?.is_dir() {
            true => (self.target.clone(), None),
            false => (
                self.target.parent().ok_or(anyhow!("None"))?.to_path_buf(),
                Some(self.target.clone()),
            ),
        };

        let mut iter: Box<dyn Iterator<Item = _>> =
            Box::new(Self::open_dir(&target, recursion, self.no_ignore)?.map(|entry| entry.path()));

        if self.ignore_ext {
            // currently we only support formats image-rs supports
            const EXTENSIONS: &[&str] = &[
                "png", "jpg", "jpeg", "gif", "webp", "bmp", "ico", "tiff", "tif", "tga", "dds",
                "farbfeld", "ff", "pnm", "pbm", "pgm", "pam", "ppm", "hdr", "exr", "qoi", "avif",
            ];
            iter = Box::new(iter.filter(|path| {
                path.extension()
                    .and_then(|x| x.to_str())
                    .map(|x| x.to_lowercase()) // ignore case
                    .map(|x| EXTENSIONS.contains(&x.as_str()))
                    .unwrap_or(false)
            }));
        };

        if !self.no_sort {
            iter = Box::new(iter.sorted_by(|a, b| {
                lexical_sort::natural_lexical_cmp(&a.to_string_lossy(), &b.to_string_lossy())
            }));
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
                            .and_then(|x| x.as_encoded_bytes().get(0).map(|n| n == &b'.'))
                            .unwrap_or(false)
                            <= no_ignore,
                    ) {
                        (ft, _, _) if ft.is_file() => {
                            let iter: Box<dyn Iterator<Item = DirEntry>> =
                                Box::new(std::iter::once(entry));
                            Some(iter)
                        }
                        (ft, r, true) if ft.is_dir() && r > 0 => {
                            Self::open_dir(path, recursion.saturating_sub(1), no_ignore).ok()
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
                tracing::error!("Failed to create mem usage csv at {path:?}: {e}");
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
    let (stdout, _guard) = tracing_appender::non_blocking(std::io::stdout());

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("error")),
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
            std::process::exit(1);
        }
    };

    let max_decode_alloc = (cli.max_decode_mem > 0).then(|| cli.max_decode_mem * 1024 * 1024);
    let mut app = App::new(paths, cursor_idx, max_decode_alloc);
    event_loop.run_app(&mut app).unwrap();
}

#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;
