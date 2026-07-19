use std::{
    cell::Cell,
    fs::File,
    io::BufReader,
    ops::{Div, Rem},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
};

use anyhow::{anyhow, bail, ensure, Result};
use crossbeam_channel::{unbounded, Receiver, Sender};
use image::{
    metadata::Orientation, DynamicImage, EncodableLayout, ImageDecoder, ImageReader, RgbaImage,
};
use wgpu::{
    util::DeviceExt, BindGroup, BindGroupLayout, Buffer, Device, Queue, RenderPass, Sampler,
    TextureFormat, TextureUsages,
};

use crate::{cache::LoadClass, disk_cache::ThumbDiskCache, raw::RawMode};

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
    /// The device's 2D texture size limit; Full decodes clamp to it. Holds
    /// a conservative default until the render state stores the real value
    /// once the GPU device exists (always before the first request fires,
    /// since requests originate from the render state's cache)
    pub max_texture_dim: Arc<AtomicU32>,
    /// Serializes decodes with giant intermediate buffers, see
    /// [`GenericImage::decode_backend`]
    pub huge_decode_lock: Arc<Mutex<()>>,
    /// How Full-resolution raws are rendered
    pub raw_mode: RawMode,
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

    pub fn new(
        parallelism: usize,
        max_alloc: Option<u64>,
        max_texture_dim: Arc<AtomicU32>,
        raw_mode: RawMode,
    ) -> Self {
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
            max_texture_dim,
            huge_decode_lock: Arc::new(Mutex::new(())),
            raw_mode,
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
        if crate::raw::is_raw(path) {
            // rawler mmaps with populate(), far too heavy for the ui-thread
            // probe. skipping costs nothing: sensors are all larger than the
            // biggest thumb bucket, so canonicalization never applies, and
            // raw preview decodes never cross the huge-decode threshold
            bail!("raw files are not probed");
        }

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        Ok(ImageReader::new(reader)
            .with_guessed_format()?
            .into_dimensions()?)
    }

    /// Decode entry point, validating the result before it can reach a
    /// texture upload
    pub fn decode(path: impl AsRef<Path>, class: LoadClass, ctx: &DecodeContext) -> Result<Self> {
        let this = Self::decode_cached(path.as_ref(), class, ctx)?;
        // wgpu panics on zero-dimension textures; real files decode to this
        // (gifs with a 0x0 logical screen descriptor). validated here rather
        // than per-backend so disk-cache hits are covered too
        ensure!(
            this.width > 0 && this.height > 0,
            "decoded to a zero-sized image ({}x{})",
            this.width,
            this.height
        );
        Ok(this)
    }

    /// Wraps the backends with the thumbnail disk cache
    fn decode_cached(path: &Path, class: LoadClass, ctx: &DecodeContext) -> Result<Self> {
        let (LoadClass::Thumb(bucket), Some(cache)) = (class, ctx.disk_cache.as_ref()) else {
            return Self::decode_backend(path, class, ctx);
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

        let this = Self::decode_backend(path, class, ctx)?;
        cache.store(path, bucket, this.width, this.height, &this.bytes);
        Ok(this)
    }

    /// Decodes predicted to hold a buffer bigger than this run one at a time
    const HUGE_DECODE_BYTES: u64 = 512 * 1024 * 1024;

    /// Format dispatch across decode backends (camera raws via rawler will
    /// slot in here too)
    fn decode_backend(path: &Path, class: LoadClass, ctx: &DecodeContext) -> Result<Self> {
        // decoding materializes the native-size image no matter the class,
        // so serialize the giant ones: concurrent workers would otherwise
        // multiply a multi-GB buffer by the pool size
        let _guard = match Self::probe_dimensions(path) {
            Ok((w, h)) if w as u64 * h as u64 * 4 > Self::HUGE_DECODE_BYTES => {
                tracing::debug!("Serializing huge decode: {path:?} ({w}x{h})");
                Some(ctx.huge_decode_lock.lock().unwrap())
            }
            _ => None,
        };

        match path {
            p if Self::is_psd(p) => Self::decode_psd(p, class, ctx),
            p if crate::raw::is_raw(p) => Self::decode_raw(p, class, ctx),
            p => Self::decode_image_rs(p, class, ctx),
        }
    }

    fn decode_psd(path: &Path, class: LoadClass, ctx: &DecodeContext) -> Result<Self> {
        tracing::trace!("Loading {path:?} as {class:?} via psd backend");

        let composite = crate::psd::decode(path, ctx.max_alloc)?;
        let img = RgbaImage::from_raw(composite.width, composite.height, composite.rgba)
            .ok_or_else(|| anyhow!("psd composite buffer size mismatch"))?;

        Self::finish_decode(
            DynamicImage::ImageRgba8(img),
            composite.icc,
            None,
            class,
            path,
            ctx,
        )
    }

    fn decode_raw(path: &Path, class: LoadClass, ctx: &DecodeContext) -> Result<Self> {
        tracing::trace!("Loading {path:?} as {class:?} via raw backend");

        let preview = match (class, ctx.raw_mode) {
            // thumbs always come from embedded previews: developing sensor
            // data for a gallery cell would be absurd
            (LoadClass::Thumb(bucket), _) => crate::raw::decode_preview(path, Some(bucket))?,
            (LoadClass::Full, RawMode::Preview) => crate::raw::decode_preview(path, None)?,
            (LoadClass::Full, RawMode::Develop) => {
                // develop buffers are always big enough to count as huge
                // (the extension-based probe skip bypasses the usual check)
                let _guard = ctx.huge_decode_lock.lock().unwrap();
                crate::raw::develop(path).or_else(|e| {
                    tracing::warn!("Raw develop failed for {path:?} ({e:?}); using the preview");
                    crate::raw::decode_preview(path, None)
                })?
            }
        };

        Self::finish_decode(
            preview.image,
            preview.icc,
            preview.orientation,
            class,
            path,
            ctx,
        )
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

    fn decode_image_rs(path: impl AsRef<Path>, class: LoadClass, ctx: &DecodeContext) -> Result<Self> {
        tracing::trace!("Loading {:?} as {class:?}", path.as_ref());

        let file = File::open(&path)?;
        let reader = BufReader::new(file);

        let mut reader = ImageReader::new(reader).with_guessed_format()?;
        // replace the default 512MiB output cap, which silently rejects
        // legitimate giant images (gigapixel mosaics), with the user's
        // configured decode budget
        let mut limits = image::Limits::no_limits();
        limits.max_alloc = ctx.max_alloc;
        reader.limits(limits);
        let mut decoder = reader.into_decoder()?;
        let orientation = decoder.orientation().ok();
        let icc_profile = decoder.icc_profile()?;
        let img = DynamicImage::from_decoder(decoder)?;

        Self::finish_decode(img, icc_profile, orientation, class, path.as_ref(), ctx)
    }

    /// Shared tail of every decode backend: class-dependent sizing, EXIF
    /// orientation, RGBA8 conversion, and the ICC transform to sRGB
    fn finish_decode(
        mut img: DynamicImage,
        icc_profile: Option<Vec<u8>>,
        orientation: Option<Orientation>,
        class: LoadClass,
        path: &Path,
        ctx: &DecodeContext,
    ) -> Result<Self> {
        match class {
            LoadClass::Full => {
                // lossless up to what the device can sample; beyond that
                // (gigapixel mosaics) would need texture tiling
                let max_dim = ctx.max_texture_dim.load(Ordering::Relaxed);
                if img.width() > max_dim || img.height() > max_dim {
                    tracing::debug!(
                        "Downscaling {}x{} from {path:?} to the {max_dim}px device texture limit",
                        img.width(),
                        img.height(),
                    );
                }
                img = Self::downscale_to_bucket(img, max_dim);
            }
            LoadClass::Thumb(bucket) => {
                img = Self::downscale_to_bucket(img, bucket);
            }
        }

        // after sizing on purpose: both sizing boxes are square, so the
        // result is identical and the rotation touches far fewer pixels
        if let Some(orientation) = orientation {
            img.apply_orientation(orientation);
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

/// Blit pipeline that fills a texture's mip chain by successively
/// downsampling each level from the one above, in linear space (the sRGB
/// format decodes/encodes at sample/write time)
#[derive(Debug)]
pub struct MipGenerator {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: BindGroupLayout,
    sampler: Sampler,
}

impl MipGenerator {
    pub fn new(device: &Device, format: TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mip blit shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("blit.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mip blit bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mip blit pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Mip blit pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(format.into())],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self {
            pipeline,
            bind_group_layout,
            sampler,
        }
    }

    fn generate(&self, device: &Device, queue: &Queue, texture: &wgpu::Texture) {
        let mut encoder = device.create_command_encoder(&Default::default());

        let views: Vec<_> = (0..texture.mip_level_count())
            .map(|level| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        for level in 1..views.len() {
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Mip blit bind group"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&views[level - 1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.sampler),
                    },
                ],
            });

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Mip blit pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &views[level],
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        queue.submit([encoder.finish()]);
    }
}

#[derive(Debug)]
pub struct RenderableImage {
    bind_group: BindGroup,
    vertex_buffer: LazyVertexBuffer,
    width: u32,
    height: u32,
    mip_level_count: u32,
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
        mips: Option<&MipGenerator>,
    ) -> Self {
        // full chain when mipped: minification of huge textures aliases
        // badly without it
        let mip_level_count = match mips {
            Some(_) => u32::BITS - img.width.max(img.height).leading_zeros(),
            None => 1,
        };

        let usage = match mips {
            Some(_) => {
                TextureUsages::TEXTURE_BINDING
                    | TextureUsages::COPY_DST
                    | TextureUsages::RENDER_ATTACHMENT
            }
            None => TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        };

        // Create GPU texture from the image data
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RenderableImage texture"),
            size: wgpu::Extent3d {
                width: img.width,
                height: img.height,
                depth_or_array_layers: 1,
            },
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: img.format,
            usage,
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

        if let Some(generator) = mips
            && mip_level_count > 1
        {
            generator.generate(device, queue, &texture);
        }

        Self {
            bind_group,
            vertex_buffer: LazyVertexBuffer::new(device),
            width: img.width,
            height: img.height,
            mip_level_count,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Approximate GPU memory of the texture (a full mip chain adds ~1/3)
    pub fn gpu_bytes(&self) -> u64 {
        let base = self.width as u64 * self.height as u64 * 4;
        match self.mip_level_count > 1 {
            true => base * 4 / 3,
            false => base,
        }
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
                match capturable.get_mapped_range_mut(..) {
                    Ok(mut view) => {
                        // BufferViewMut is write-only in wgpu 30 (mapped
                        // memory may be write-combining)
                        view.copy_from_slice(bytemuck::cast_slice(&vertices.finish()));
                        drop(view);
                        capturable.unmap();
                        is_mapped.store(false, Ordering::Release);
                    }
                    Err(e) => {
                        // can't happen after a successful map; recover the
                        // buffer for future resizes anyway
                        tracing::warn!("Mapped range unavailable: {e:?}");
                        capturable.unmap();
                        is_mapped.store(false, Ordering::Release);
                    }
                }
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
