use std::{collections::HashMap, path::Path, path::PathBuf};

use anyhow::Result;
use crossbeam_channel::{unbounded, Receiver, Sender};

use crate::lazy::{GenericImage, ImageRequest, RenderableImage};

/// Unified per-path image state: the CPU-side load state machine and the
/// GPU-side renderable live in one entry, so residency decisions can be
/// made in one place.
#[derive(Debug)]
pub struct ImageCache {
    entries: HashMap<PathBuf, CacheEntry>,
    req_sender: Sender<ImageRequest>,
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

    /// Advances the load state for `path`: requests a load on first sight,
    /// polls an in-flight request, and uploads via `upload` once bytes are
    /// available. Returns the renderable while one exists.
    pub fn get_renderable(
        &mut self,
        path: &Path,
        upload: impl FnOnce(&GenericImage) -> RenderableImage,
    ) -> Option<&mut RenderableImage> {
        // NOTE: the entry api would require cloning path even when the entry
        // already exists, which is hot path. so we eat a second lookup on the
        // cold path instead
        if !self.entries.contains_key(path) {
            self.entries
                .insert(path.to_path_buf(), CacheEntry::default());
        }
        let entry = self.entries.get_mut(path).unwrap();

        match &entry.cpu {
            CpuState::NotRequested => {
                let (response_sender, response_receiver) = unbounded();
                self.req_sender
                    .send(ImageRequest::new(response_sender, path.to_path_buf()))
                    .unwrap();
                entry.cpu = CpuState::Requested(response_receiver);
            }
            CpuState::Requested(receiver) => {
                if let Ok(resp) = receiver.try_recv() {
                    match resp {
                        Ok(img) => {
                            entry.gpu = Some(upload(&img));
                            entry.cpu = CpuState::Uploaded;
                            // img dropped here: the CPU copy only lives until
                            // the pixels are on the GPU
                        }
                        Err(e) => {
                            tracing::debug!("Load failed: {e:?} from {path:?}");
                            entry.cpu = CpuState::Failed;
                        }
                    }
                }
            }
            _ => {}
        }

        entry.gpu.as_mut()
    }
}
