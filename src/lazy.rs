use std::{
    cell::Cell,
    fs::File,
    io::BufReader,
    ops::{Div, Rem},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
};

use anyhow::{anyhow, Result};
use crossbeam_channel::{unbounded, Receiver, Sender};
use image::{DynamicImage, EncodableLayout, ImageDecoder, ImageReader, RgbaImage};
use wgpu::{
    util::DeviceExt, BindGroup, BindGroupLayout, Buffer, Device, Queue, RenderPass, Sampler,
    TextureFormat, TextureUsages,
};

use crate::{cache::LoadClass, disk_cache::ThumbDiskCache};

/// A path in the image collection plus its current gallery slot.
/// Load state lives in [`crate::cache::ImageCache`], keyed by the path.
#[derive(Debug)]
pub struct LazyImage {
    grid_pos: Cell<usize>,
    path: PathBuf,
}

impl LazyImage {
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            grid_pos: Cell::new(0),
        }
    }

    pub fn set_pos(&self, val: usize) {
        self.grid_pos.set(val);
    }

    pub fn get_pos(&self) -> usize {
        self.grid_pos.get()
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}

/// Configuration shared by all decode workers
#[derive(Debug, Clone)]
pub struct DecodeContext {
    pub disk_cache: Option<ThumbDiskCache>,
    /// Decodes predicted to allocate more than this fail; None = unlimited
    pub max_alloc: Option<u64>,
}

pub struct ImageRequest {
    pub response_channel: Sender<Result<GenericImage>>,
    pub path: PathBuf,
    pub class: LoadClass,
}

impl ImageRequest {
    pub fn new(
        response_channel: Sender<Result<GenericImage>>,
        path: PathBuf,
        class: LoadClass,
    ) -> Self {
        Self {
            response_channel,
            path,
            class,
        }
    }

    /// Returns whether the result was an error
    pub fn eval(self, ctx: &DecodeContext) -> bool {
        let result = GenericImage::decode(&self.path, self.class, ctx);
        let is_err = result.is_err();
        self.response_channel.send(result).unwrap();
        is_err
    }
}

#[derive(Debug)]
pub struct ImageLoaderService {
    sender: Sender<ImageRequest>,
    /// Bool represents if the completion resulted in Err
    completion_receiver: Receiver<(PathBuf, bool)>,
    #[allow(dead_code)]
    handles: Vec<JoinHandle<()>>,
}

impl ImageLoaderService {
    const MIN_PAR: usize = 1;
    const MAX_PAR: usize = 4;

    pub fn new(parallelism: usize, max_alloc: Option<u64>) -> Self {
        let parallelism = match (parallelism, thread::available_parallelism()) {
            (0, Ok(b)) => b.get(),
            (0, Err(_)) => Self::MIN_PAR,
            (a, _) => a,
        };
        let parallelism = parallelism.clamp(Self::MIN_PAR, Self::MAX_PAR);
        tracing::info!("ImageLoaderService parallelism: {parallelism}");

        let disk_cache = match ThumbDiskCache::new() {
            Ok(cache) => {
                cache.spawn_janitor();
                Some(cache)
            }
            Err(e) => {
                tracing::warn!("Thumbnail disk cache disabled: {e:?}");
                None
            }
        };

        let ctx = DecodeContext {
            disk_cache,
            max_alloc,
        };

        let (completion_sender, completion_receiver) = unbounded();
        let (sender, receiver) = unbounded::<ImageRequest>();
        let mut handles = Vec::new();
        for id in 0..parallelism {
            let receiver = receiver.clone();
            let completion_sender = completion_sender.clone();
            let ctx = ctx.clone();

            let handle = thread::spawn(move || loop {
                match receiver.recv() {
                    Ok(req) => {
                        let path = req.path.clone();
                        tracing::trace!("Request on {id}:{:?}", path);
                        let was_err = req.eval(&ctx);
                        let _ = completion_sender.send((path, was_err));
                    }
                    Err(_) => break,
                };
            });
            handles.push(handle);
        }

        Self {
            sender,
            completion_receiver,
            handles,
        }
    }

    pub fn clone_sender(&self) -> Sender<ImageRequest> {
        self.sender.clone()
    }

    pub fn completed(&self) -> Vec<(PathBuf, bool)> {
        let mut completed = Vec::new();
        while let Ok(result) = self.completion_receiver.try_recv() {
            completed.push(result)
        }
        completed
    }
}

#[derive(Debug)]
pub struct GenericImage {
    pub width: u32,
    pub height: u32,
    pub bytes: Vec<u8>,
    pub format: TextureFormat,
    pub pixel_width: u32,
}

impl GenericImage {
    fn is_psd(path: &Path) -> bool {
        path.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("psd") || e.eq_ignore_ascii_case("psb"))
    }

    /// Header-only read of the native dimensions; no pixel decode
    pub fn probe_dimensions(path: impl AsRef<Path>) -> Result<(u32, u32)> {
        let path = path.as_ref();
        if Self::is_psd(path) {
            return crate::psd::probe_dimensions(path);
        }

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(ImageReader::new(reader)
            .with_guessed_format()?
            .into_dimensions()?)
    }

    /// Decode entry point, wrapping the backends with the thumbnail disk
    /// cache
    pub fn decode(path: impl AsRef<Path>, class: LoadClass, ctx: &DecodeContext) -> Result<Self> {
        let path = path.as_ref();
        let (LoadClass::Thumb(bucket), Some(cache)) = (class, ctx.disk_cache.as_ref()) else {
            return Self::decode_backend(path, class, ctx.max_alloc);
        };

        if let Some(img) = cache.load(path, bucket) {
            tracing::trace!("Thumb cache hit: {path:?} @{bucket}");
            return Ok(Self::from_dynamic_srgb(img));
        }

        // build from a larger cached thumb when possible: skips re-decoding
        // the original, and the quality cost of a second lanczos pass at
        // these sizes is negligible
        if let Some(larger) = cache.load_larger(path, bucket) {
            tracing::trace!("Thumb cache promotion: {path:?} @{bucket}");
            let this = Self::from_dynamic_srgb(Self::downscale_to_bucket(larger, bucket));
            cache.store(path, bucket, this.width, this.height, &this.bytes);
            return Ok(this);
        }

        let this = Self::decode_backend(path, class, ctx.max_alloc)?;
        cache.store(path, bucket, this.width, this.height, &this.bytes);
        Ok(this)
    }

    /// Format dispatch across decode backends (camera raws via rawler will
    /// slot in here too)
    fn decode_backend(path: &Path, class: LoadClass, max_alloc: Option<u64>) -> Result<Self> {
        match Self::is_psd(path) {
            true => Self::decode_psd(path, class, max_alloc),
            false => Self::decode_image_rs(path, class, max_alloc),
        }
    }

    fn decode_psd(path: &Path, class: LoadClass, max_alloc: Option<u64>) -> Result<Self> {
        tracing::trace!("Loading {path:?} as {class:?} via psd backend");

        let composite = crate::psd::decode(path, max_alloc)?;
        let img = RgbaImage::from_raw(composite.width, composite.height, composite.rgba)
            .ok_or_else(|| anyhow!("psd composite buffer size mismatch"))?;

        Self::finish_decode(DynamicImage::ImageRgba8(img), composite.icc, class, path)
    }

    /// For pixels that are already sRGB (i.e. out of the disk cache); no
    /// ICC transform is applied
    fn from_dynamic_srgb(img: DynamicImage) -> Self {
        Self {
            width: img.width(),
            height: img.height(),
            bytes: img.to_rgba8().as_bytes().to_vec(),
            format: TextureFormat::Rgba8UnormSrgb,
            pixel_width: 4,
        }
    }

    /// Aspect-fit into the bucket box, but never upscale
    fn downscale_to_bucket(mut img: DynamicImage, bucket: u32) -> DynamicImage {
        if img.width() <= bucket && img.height() <= bucket {
            return img;
        }

        // fast integer pre-shrink to <=2x the target, then a single quality
        // pass; far cheaper than lanczos over the full image
        if img.width() > bucket * 2 || img.height() > bucket * 2 {
            img = img.thumbnail(bucket * 2, bucket * 2);
        }
        img.resize(bucket, bucket, image::imageops::FilterType::Lanczos3)
    }

    fn decode_image_rs(
        path: impl AsRef<Path>,
        class: LoadClass,
        max_alloc: Option<u64>,
    ) -> Result<Self> {
        tracing::trace!("Loading {:?} as {class:?}", path.as_ref());

        let file = File::open(&path)?;
        let reader = BufReader::new(file);

        let mut reader = ImageReader::new(reader).with_guessed_format()?;
        // replace the default 512MiB output cap, which silently rejects
        // legitimate giant images (gigapixel mosaics), with the user's
        // configured decode budget
        let mut limits = image::Limits::no_limits();
        limits.max_alloc = max_alloc;
        reader.limits(limits);
        let mut decoder = reader.into_decoder()?;
        let icc_profile = decoder.icc_profile()?;
        let img = DynamicImage::from_decoder(decoder)?;

        Self::finish_decode(img, icc_profile, class, path.as_ref())
    }

    /// Shared tail of every decode backend: class-dependent sizing, RGBA8
    /// conversion, and the ICC transform to sRGB
    fn finish_decode(
        mut img: DynamicImage,
        icc_profile: Option<Vec<u8>>,
        class: LoadClass,
        path: &Path,
    ) -> Result<Self> {
        match class {
            LoadClass::Full => {
                // TODO drop this cap once GPU residency is budgeted (it makes
                // single image mode lossy above 4k)
                const MAX_WIDTH: u32 = 3840;
                const MAX_HEIGHT: u32 = 2160;
                if img.width() > MAX_WIDTH || img.height() > MAX_HEIGHT {
                    tracing::trace!("Resizing {}x{} from {path:?}", img.width(), img.height());
                    img = img.resize(MAX_WIDTH, MAX_HEIGHT, image::imageops::FilterType::Triangle);
                }
            }
            LoadClass::Thumb(bucket) => {
                img = Self::downscale_to_bucket(img, bucket);
            }
        }

        let width = img.width();
        let height = img.height();

        let mut bytes = img.to_rgba8().as_bytes().to_vec();
        let format = TextureFormat::Rgba8UnormSrgb;
        let pixel_width = 4;

        if let Some(data) = icc_profile {
            let profile = lcms2::Profile::new_icc(&data)?;
            let t = lcms2::Transform::new(
                &profile,
                lcms2::PixelFormat::RGBA_8,
                &lcms2::Profile::new_srgb(),
                lcms2::PixelFormat::RGBA_8,
                lcms2::Intent::Perceptual,
            )?;

            tracing::trace!("Transforming {path:?} via ICC profile");
            t.transform_in_place(&mut bytes);
        }

        tracing::trace!("Finished: {path:?}");
        Ok(Self {
            width,
            height,
            bytes,
            format,
            pixel_width,
        })
    }
}

#[derive(Debug)]
pub struct RenderableImage {
    bind_group: BindGroup,
    vertex_buffer: LazyVertexBuffer,
    width: u32,
    height: u32,
}

impl RenderableImage {
    /// Creates a new RenderableImage from a GenericImage, uploading it to the GPU.
    /// This packages all GPU state needed to render the image.
    /// The vertex buffer is lazily initialized on the first resize.
    pub fn new(
        img: &GenericImage,
        device: &Device,
        queue: &Queue,
        bind_group_layout: &BindGroupLayout,
        sampler: &Sampler,
    ) -> Self {
        // Create GPU texture from the image data
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RenderableImage texture"),
            size: wgpu::Extent3d {
                width: img.width,
                height: img.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: img.format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create bind group that associates the texture with the shader
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
            label: Some("RenderableImage bind group"),
        });

        // Upload image data to GPU
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &img.bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(img.width * img.pixel_width),
                rows_per_image: Some(img.height),
            },
            wgpu::Extent3d {
                width: img.width,
                height: img.height,
                depth_or_array_layers: 1,
            },
        );

        Self {
            bind_group,
            vertex_buffer: LazyVertexBuffer::new(device),
            width: img.width,
            height: img.height,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn render(&mut self, renderpass: &mut RenderPass) {
        if let Some(vertex_buffer) = &self.vertex_buffer.get() {
            renderpass.set_bind_group(0, &self.bind_group, &[]);
            renderpass.set_vertex_buffer(0, vertex_buffer.slice(..));
            renderpass.draw(0..6, 0..1);
        }
    }

    pub fn resize(&mut self, vertices: Vertices) {
        self.vertex_buffer.resize(vertices);
    }
}

#[derive(Debug)]
pub struct LazyVertexBuffer {
    device: Device,
    vertex_buffer: Option<Buffer>,
    mapped: Arc<AtomicBool>,
    waiting_to_resize: Option<Vertices>,
}

impl LazyVertexBuffer {
    pub fn new(device: &Device) -> Self {
        Self {
            vertex_buffer: None,
            device: device.clone(),
            mapped: Arc::new(AtomicBool::new(false)),
            waiting_to_resize: None,
        }
    }

    pub fn get(&mut self) -> Option<&Buffer> {
        if let Some(vertices) = std::mem::replace(&mut self.waiting_to_resize, None) {
            self.resize(vertices);
            // NOTE: if we don't return here than it can cause an invalid state
            // since between the read of self.mapped in self.resize and the read
            // of self.mapped in this function, the blocking callback could complete
            // meaning we would return a dirty vertex buffer
            return None;
        }

        if let Some(buffer) = &self.vertex_buffer
            && !self.mapped.load(Ordering::Acquire)
        {
            Some(buffer)
        } else {
            None
        }
    }

    pub fn resize(&mut self, vertices: Vertices) {
        // Initialize vertex buffer on first resize if not already created
        if self.vertex_buffer.is_none() {
            self.vertex_buffer = Some(self.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("RenderableImage vertex buffer"),
                    contents: bytemuck::cast_slice(&vertices.finish()),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::MAP_WRITE,
                },
            ));
            return;
        }

        if self.mapped.load(Ordering::Acquire) {
            self.waiting_to_resize = Some(vertices);
            return;
        } else {
            self.waiting_to_resize = None;
        }

        let vertex_buffer = self.vertex_buffer.as_ref().unwrap();
        let capturable = vertex_buffer.clone();
        self.mapped.store(true, Ordering::Release);
        let is_mapped = self.mapped.clone();
        vertex_buffer.map_async(wgpu::MapMode::Write, .., move |result| {
            if result.is_ok() {
                let mut view = capturable.get_mapped_range_mut(..);
                let floats: &mut [f32] = bytemuck::cast_slice_mut(&mut view);
                floats.copy_from_slice(&vertices.finish());
                drop(view);
                capturable.unmap();
                is_mapped.store(false, Ordering::Release);
            }
        });
    }
}

#[derive(Debug, Clone)]
pub struct Vertices {
    pub tlx: f32,  // top-left x
    pub tly: f32,  // top-left y
    pub trx: f32,  // top-right x
    pub try_: f32, // top-right y (try is a keyword)
    pub blx: f32,  // bottom-left x
    pub bly: f32,  // bottom-left y
    pub brx: f32,  // bottom-right x
    pub bry: f32,  // bottom-right y
}

impl Vertices {
    pub fn new() -> Self {
        Self {
            tlx: -1.0,
            tly: 1.0,
            trx: 1.0,
            try_: 1.0,
            blx: -1.0,
            bly: -1.0,
            brx: 1.0,
            bry: -1.0,
        }
    }

    pub fn finish(&self) -> [f32; 24] {
        [
            // Position x-y Texture x-y
            self.blx, self.bly, 0.0, 1.0, // Bottom left
            self.brx, self.bry, 1.0, 1.0, // Bottom right
            self.tlx, self.tly, 0.0, 0.0, // Top left
            self.tlx, self.tly, 0.0, 0.0, // Top left
            self.brx, self.bry, 1.0, 1.0, // Bottom right
            self.trx, self.try_, 1.0, 0.0, // Top right
        ]
    }

    pub fn aspect(mut self, aspect: f32) -> Self {
        match aspect {
            x if x > 1.0 => {
                // Letterbox vertically - adjust y coordinates
                self.bly /= aspect;
                self.bry /= aspect;
                self.tly /= aspect;
                self.try_ /= aspect;
            }
            x if x < 1.0 => {
                // Pillarbox horizontally - adjust x coordinates
                let inv_aspect = 1.0 / aspect;
                self.blx /= inv_aspect;
                self.brx /= inv_aspect;
                self.tlx /= inv_aspect;
                self.trx /= inv_aspect;
            }
            _ => {}
        }

        self
    }

    pub fn scale(mut self, scale: f32) -> Self {
        self.tlx *= scale;
        self.tly *= scale;
        self.trx *= scale;
        self.try_ *= scale;
        self.blx *= scale;
        self.bly *= scale;
        self.brx *= scale;
        self.bry *= scale;

        self
    }

    pub fn transpose(mut self, x: f32, y: f32) -> Self {
        self.tlx += x;
        self.tly += y;
        self.trx += x;
        self.try_ += y;
        self.blx += x;
        self.bly += y;
        self.brx += x;
        self.bry += y;

        self
    }

    pub fn top(mut self, value: f32) -> Self {
        self.tly = value;
        self.try_ = value;
        self
    }

    pub fn bot(mut self, value: f32) -> Self {
        self.bly = value;
        self.bry = value;
        self
    }

    pub fn left(mut self, value: f32) -> Self {
        self.tlx = value;
        self.blx = value;
        self
    }

    pub fn right(mut self, value: f32) -> Self {
        self.trx = value;
        self.brx = value;
        self
    }

    pub fn gallery(
        self,
        ww: f32,
        wh: f32,
        iw: f32,
        ih: f32,
        row_no: f32,
        col_no: f32,
        pos: f32,
    ) -> Self {
        let margin = ww.rem(wh.div(row_no).trunc()) / ww;

        let col_idx = pos % col_no;
        let row_idx = (pos - col_idx) / col_no;

        let col_offset = col_idx - (col_no - 1.0) / 2.0;
        let row_offset = row_idx - (row_no - 1.0) / 2.0;

        let cell_height = 2.0 / row_no;
        let cell_width = (2.0 - margin * 2.0) / col_no;

        let unit_height = 1.0 / row_no;
        let unit_width = (1.0 - margin) / col_no;

        self.top(unit_height)
            .bot(-unit_height)
            .left(-unit_width)
            .right(unit_width)
            .aspect(iw.div(ih))
            .transpose(col_offset * cell_width, -row_offset * cell_height)
    }

    pub fn gallery_cursor(
        self,
        ww: f32,
        wh: f32,
        row_no: f32,
        col_no: f32,
        cursor_idx: f32,
    ) -> Self {
        let margin = ww.rem(wh.div(row_no).trunc()) / ww;

        let col_idx = cursor_idx % col_no;
        let row_idx = (cursor_idx - col_idx) / col_no;

        let col_offset = col_idx - (col_no - 1.0) / 2.0;
        let row_offset = row_idx - (row_no - 1.0) / 2.0;

        let cell_height = 2.0 / row_no;
        let cell_width = (2.0 - margin * 2.0) / col_no;

        let unit_height = 1.0 / row_no;
        let unit_width = (1.0 - margin) / col_no;

        self.top(unit_height)
            .bot(unit_height - (unit_height / 2.5))
            .left(-unit_width)
            .right(-unit_width + (unit_width / 2.5))
            .transpose(col_offset * cell_width, -row_offset * cell_height)
    }

    pub fn single_image(
        self,
        ww: f32,
        wh: f32,
        iw: f32,
        ih: f32,
        scale_factor: f32,
        offset_x: f32,
        offset_y: f32,
    ) -> Self {
        self.top(1.0)
            .bot(-1.0)
            .left(-1.0)
            .right(1.0)
            .aspect(iw.div(ih) / ww.div(wh))
            .scale(scale_factor)
            .transpose(offset_x, offset_y)
    }
}
