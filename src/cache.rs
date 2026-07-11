use std::{
    collections::{BTreeMap, HashMap},
    path::{Path, PathBuf},
};

use anyhow::Result;
use crossbeam_channel::{unbounded, Receiver, Sender};

use crate::lazy::{GenericImage, ImageRequest, RenderableImage};

/// The resolution an image is decoded at.
///
/// The derived ordering (smaller buckets < larger buckets < Full) is what
/// makes the fallback range scans in [`ImageCache::get_renderable`] work.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LoadClass {
    /// Aspect-fit into a square bounding box of this size in physical px,
    /// always one of [`LADDER`]
    Thumb(u32),
    /// Native resolution (still capped for now, see [`GenericImage::decode`])
    Full,
}

impl LoadClass {
    /// Collapses classes that cannot differ for an image with the given
    /// native dimensions: a thumb bucket that fully contains the image
    /// decodes to exactly the pixels Full would (the decoder never
    /// upscales), so they should share one entry instead of duplicating
    /// decodes and textures
    fn canonicalize(self, native: Option<(u32, u32)>) -> Self {
        match (self, native) {
            (Self::Thumb(bucket), Some((w, h))) if w <= bucket && h <= bucket => Self::Full,
            _ => self,
        }
    }
}

/// The fixed set of thumbnail bucket sizes. Rounding the needed cell size
/// *up* to a bucket guarantees a cell never displays more pixels than its
/// thumbnail holds (at worst ~2x oversampled), while bounding how many
/// variants of an image can ever exist no matter how many window size and
/// row count combinations are passed through.
pub const LADDER: [u32; 5] = [128, 256, 512, 1024, 2048];

/// The smallest bucket that fully covers a gallery cell of side `cell_px`.
pub fn bucket_for(cell_px: u32) -> u32 {
    LADDER
        .into_iter()
        .find(|&b| b >= cell_px)
        .unwrap_or(LADDER[LADDER.len() - 1])
}

/// Unified per-(path, class) image state: the CPU-side load state machine
/// and the GPU-side renderable live in one entry, so residency decisions can
/// be made in one place.
#[derive(Debug)]
pub struct ImageCache {
    entries: HashMap<PathBuf, PathEntry>,
    req_sender: Sender<ImageRequest>,
    /// Bumped once per rendered frame; entries touched in the current frame
    /// are the working set and are never evicted
    frame: u64,
    gpu_bytes_total: u64,
    gpu_budget: u64,
}

#[derive(Debug, Default)]
struct PathEntry {
    /// Native dimensions from a header-only probe at first sight;
    /// None when probing failed
    native: Option<(u32, u32)>,
    classes: BTreeMap<LoadClass, CacheEntry>,
}

#[derive(Debug, Default)]
struct CacheEntry {
    cpu: CpuState,
    gpu: Option<RenderableImage>,
    gpu_bytes: u64,
    last_used: u64,
}

#[derive(Debug, Default)]
enum CpuState {
    #[default]
    NotRequested,
    Requested(Receiver<Result<GenericImage>>),
    /// Pixels live on the GPU only; the decoded bytes were dropped at upload
    Uploaded,
    Failed,
}

impl ImageCache {
    /// At most this many *large* Full entries stay resident: they can be far
    /// larger than any thumb, and plain byte-budget lru would let one burst
    /// of single-image navigation evict every thumbnail
    const MAX_RESIDENT_FULL: usize = 3;

    /// Full entries below this don't count toward [`Self::MAX_RESIDENT_FULL`]:
    /// images smaller than the largest thumb bucket canonicalize to Full
    /// (see [`LoadClass::canonicalize`]) and would otherwise thrash in a
    /// gallery of small images
    const LARGE_FULL_BYTES: u64 = 2048 * 2048 * 4;

    pub fn new(req_sender: Sender<ImageRequest>, gpu_budget: u64) -> Self {
        tracing::info!("GPU texture budget: {} MB", gpu_budget / 1_000_000);
        Self {
            entries: HashMap::new(),
            req_sender,
            frame: 0,
            gpu_bytes_total: 0,
            gpu_budget,
        }
    }

    /// Marks the start of a new frame for working-set tracking
    pub fn advance_frame(&mut self) {
        self.frame += 1;
    }

    /// Read-only lookup of the best resident renderable for `class`: no
    /// requests are issued and no in-flight decodes are polled.
    ///
    /// The render path must use this rather than [`Self::get_renderable`]:
    /// state only advances via the resize paths, which lay out vertices
    /// immediately after an upload, so a renderable with an uninitialized
    /// vertex buffer (which would skip its draw and flicker for a frame) can
    /// never reach a render pass.
    pub fn get_resident(&mut self, path: &Path, class: LoadClass) -> Option<&mut RenderableImage> {
        let entry = self.entries.get_mut(path)?;
        let class = class.canonicalize(entry.native);
        let best = Self::best_resident(&entry.classes, class)?;
        let entry = entry.classes.get_mut(&best).unwrap();
        entry.last_used = self.frame;
        entry.gpu.as_mut()
    }

    /// Advances the load state for `path` and returns the best renderable
    /// for `class`: requests a load on first sight, polls in-flight requests,
    /// and uploads via `upload` once bytes are available.
    ///
    /// While the wanted class isn't resident, the nearest resident class is
    /// returned instead, so cells show a re-scaled stand-in rather than
    /// nothing during bucket transitions.
    ///
    /// Callers must lay out vertices on the returned renderable; see
    /// [`Self::get_resident`].
    pub fn get_renderable(
        &mut self,
        path: &Path,
        class: LoadClass,
        mut upload: impl FnMut(&GenericImage, LoadClass) -> RenderableImage,
    ) -> Option<&mut RenderableImage> {
        // NOTE: the entry api would require cloning path even when the entry
        // already exists, which is hot path. so we eat a second lookup on the
        // cold path instead
        if !self.entries.contains_key(path) {
            // header-only read, cheap enough for the ui thread; a failed
            // probe is not fatal (the decode is authoritative for errors)
            let native = GenericImage::probe_dimensions(path)
                .inspect_err(|e| tracing::debug!("Probe failed: {e:?} from {path:?}"))
                .ok();
            self.entries.insert(
                path.to_path_buf(),
                PathEntry {
                    native,
                    ..Default::default()
                },
            );
        }
        let path_entry = self.entries.get_mut(path).unwrap();
        let class = class.canonicalize(path_entry.native);

        // request the wanted class on first sight
        let entry = path_entry.classes.entry(class).or_default();
        if matches!(entry.cpu, CpuState::NotRequested) {
            let (response_sender, response_receiver) = unbounded();
            self.req_sender
                .send(ImageRequest::new(
                    response_sender,
                    path.to_path_buf(),
                    class,
                ))
                .unwrap();
            entry.cpu = CpuState::Requested(response_receiver);
        }

        // poll every in-flight request for this path, not just the wanted
        // class: a decode completing for a stale bucket would otherwise sit
        // in its channel holding the decoded bytes indefinitely
        for (entry_class, entry) in path_entry.classes.iter_mut() {
            let CpuState::Requested(receiver) = &entry.cpu else {
                continue;
            };
            let Ok(resp) = receiver.try_recv() else {
                continue;
            };
            match resp {
                Ok(img) => {
                    let renderable = upload(&img, *entry_class);
                    entry.gpu_bytes = renderable.gpu_bytes();
                    self.gpu_bytes_total += entry.gpu_bytes;
                    entry.last_used = self.frame;
                    entry.gpu = Some(renderable);
                    entry.cpu = CpuState::Uploaded;
                    // img dropped here: the CPU copy only lives until the
                    // pixels are on the GPU
                }
                Err(e) => {
                    // warn: this is the only place the user can learn why
                    // an image was dropped from the collection
                    tracing::warn!("Load failed: {e:?} from {path:?}");
                    entry.cpu = CpuState::Failed;
                }
            }
        }

        let best = Self::best_resident(&path_entry.classes, class)?;
        let entry = path_entry.classes.get_mut(&best).unwrap();
        entry.last_used = self.frame;
        entry.gpu.as_mut()
    }

    /// Evicts least-recently-used textures until the byte budget and the
    /// resident-Full cap hold, never touching the current frame's working
    /// set. Returns whether anything was evicted — the caller must then
    /// re-lay out the visible set before rendering, since eviction can
    /// promote a stale-or-never-laid-out sibling to "best resident".
    pub fn enforce_budget(&mut self) -> bool {
        let resident_large_fulls = self
            .entries
            .values()
            .filter(|entry| {
                entry.classes.get(&LoadClass::Full).is_some_and(|full| {
                    full.gpu.is_some() && full.gpu_bytes > Self::LARGE_FULL_BYTES
                })
            })
            .count();

        if self.gpu_bytes_total <= self.gpu_budget
            && resident_large_fulls <= Self::MAX_RESIDENT_FULL
        {
            return false;
        }

        // every unpinned resident entry, least recently used first
        let mut candidates: Vec<(PathBuf, LoadClass, u64, u64)> = self
            .entries
            .iter()
            .flat_map(|(path, path_entry)| {
                path_entry
                    .classes
                    .iter()
                    .filter(|(_, entry)| entry.gpu.is_some() && entry.last_used < self.frame)
                    .map(|(class, entry)| (path.clone(), *class, entry.last_used, entry.gpu_bytes))
            })
            .collect();
        candidates.sort_by_key(|(_, _, last_used, _)| *last_used);

        let mut evicted = false;

        let mut large_fulls = resident_large_fulls;
        for (path, class, _, _) in candidates.iter().filter(|(_, class, _, gpu_bytes)| {
            *class == LoadClass::Full && *gpu_bytes > Self::LARGE_FULL_BYTES
        }) {
            if large_fulls <= Self::MAX_RESIDENT_FULL {
                break;
            }
            if self.evict(path, *class) {
                large_fulls -= 1;
                evicted = true;
            }
        }

        for (path, class, _, _) in &candidates {
            if self.gpu_bytes_total <= self.gpu_budget {
                break;
            }
            evicted |= self.evict(path, *class);
        }

        if evicted {
            tracing::debug!(
                "GPU cache: {} MB resident after eviction",
                self.gpu_bytes_total / 1_000_000
            );
        }
        evicted
    }

    /// Idempotent; the entry re-materializes through the disk cache on the
    /// next advancing lookup
    fn evict(&mut self, path: &Path, class: LoadClass) -> bool {
        let Some(entry) = self
            .entries
            .get_mut(path)
            .and_then(|path_entry| path_entry.classes.get_mut(&class))
        else {
            return false;
        };

        if entry.gpu.is_none() {
            return false;
        }

        tracing::trace!("Evicting {class:?} of {path:?}");
        entry.gpu = None;
        entry.cpu = CpuState::NotRequested;
        self.gpu_bytes_total -= entry.gpu_bytes;
        entry.gpu_bytes = 0;
        true
    }

    /// The wanted class when resident, else the nearest resident fallback,
    /// preferring sharper-than-wanted over blurrier-than-wanted
    fn best_resident(
        classes: &BTreeMap<LoadClass, CacheEntry>,
        class: LoadClass,
    ) -> Option<LoadClass> {
        classes
            .range(class..)
            .chain(classes.range(..class).rev())
            .find(|(_, entry)| entry.gpu.is_some())
            .map(|(class, _)| *class)
    }
}