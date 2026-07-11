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
    pub fn new(req_sender: Sender<ImageRequest>) -> Self {
        Self {
            entries: HashMap::new(),
            req_sender,
        }
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
        entry.classes.get_mut(&best).unwrap().gpu.as_mut()
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
        mut upload: impl FnMut(&GenericImage) -> RenderableImage,
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
        for entry in path_entry.classes.values_mut() {
            let CpuState::Requested(receiver) = &entry.cpu else {
                continue;
            };
            let Ok(resp) = receiver.try_recv() else {
                continue;
            };
            match resp {
                Ok(img) => {
                    entry.gpu = Some(upload(&img));
                    entry.cpu = CpuState::Uploaded;
                    // img dropped here: the CPU copy only lives until the
                    // pixels are on the GPU
                }
                Err(e) => {
                    tracing::debug!("Load failed: {e:?} from {path:?}");
                    entry.cpu = CpuState::Failed;
                }
            }
        }

        let best = Self::best_resident(&path_entry.classes, class)?;
        path_entry.classes.get_mut(&best).unwrap().gpu.as_mut()
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