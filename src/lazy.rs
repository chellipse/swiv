use std::{
    cell::{Cell, RefCell},
    marker::PhantomData,
    ops::{Div, Rem},
    path::{Path, PathBuf},
    rc::Rc,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
};

use anyhow::{anyhow, Result};
use crossbeam_channel::{unbounded, Receiver, Sender};
use image::{DynamicImage, EncodableLayout, ImageDecoder};
use wgpu::{
    util::DeviceExt, BindGroup, BindGroupLayout, Buffer, Device, Queue, RenderPass, Sampler,
    TextureFormat, TextureUsages,
};

#[derive(Debug)]
pub struct LazyImage {
    state: RefCell<LazyImageState>,
    grid_pos: Cell<usize>,
    path: PathBuf,
    _not_send: PhantomData<*const ()>, // Makes the type !Send
}

#[derive(Debug)]
enum LazyImageState {
    Uninitialized(Sender<ImageRequest>),
    Requested(Receiver<Result<GenericImage>>),
    Initialized(Rc<GenericImage>),
    Error(anyhow::Error),
}

impl LazyImage {
    pub fn new(path: PathBuf, req_sender: Sender<ImageRequest>) -> Self {
        Self {
            path,
            grid_pos: Cell::new(0),
            state: RefCell::new(LazyImageState::Uninitialized(req_sender)),
            _not_send: PhantomData,
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

    pub fn get(&self) -> Option<Result<Rc<GenericImage>>> {
        // Transition state if needed
        {
            let mut state = self.state.borrow_mut();
            match &*state {
                LazyImageState::Uninitialized(sender) => {
                    let (response_channel, receiver) = unbounded();
                    sender
                        .send(ImageRequest::new(response_channel, self.path.clone()))
                        .unwrap();
                    *state = LazyImageState::Requested(receiver);
                }
                LazyImageState::Requested(receiver) => {
                    if let Ok(resp) = receiver.try_recv() {
                        match resp {
                            Ok(resp) => {
                                *state = LazyImageState::Initialized(Rc::new(resp));
                            }
                            Err(e) => {
                                *state = LazyImageState::Error(e);
                            }
                        };
                    }
                }
                _ => {}
            }
        }

        // Return result
        let state = self.state.borrow();
        match &*state {
            LazyImageState::Initialized(result) => Some(Ok(Rc::clone(result))),
            LazyImageState::Error(e) => Some(Err(anyhow!("{e:?} from {:?}", &self.path))),
            _ => None,
        }
    }
}

pub struct ImageRequest {
    pub response_channel: Sender<Result<GenericImage>>,
    pub path: PathBuf,
}

impl ImageRequest {
    pub fn new(response_channel: Sender<Result<GenericImage>>, path: PathBuf) -> Self {
        Self {
            response_channel,
            path,
        }
    }

    pub fn eval(self) {
        self.response_channel
            .send(GenericImage::new(&self.path))
            .unwrap();
    }
}

#[derive(Debug)]
pub struct ImageLoaderService {
    sender: Sender<ImageRequest>,
    completion_receiver: Receiver<PathBuf>,
    #[allow(dead_code)]
    handles: Vec<JoinHandle<()>>,
}

impl ImageLoaderService {
    const MIN_PAR: usize = 2;
    const MAX_PAR: usize = 16;

    pub fn new(parallelism: usize) -> Self {
        let parallelism = match (parallelism, thread::available_parallelism()) {
            (0, Ok(b)) => b.get(),
            (0, Err(_)) => Self::MIN_PAR,
            (a, _) => a,
        };
        let parallelism = parallelism.min(Self::MIN_PAR).max(Self::MAX_PAR);
        tracing::info!("ImageLoaderService parallelism: {parallelism}");

        let (completion_sender, completion_receiver) = unbounded::<PathBuf>();
        let (sender, receiver) = unbounded::<ImageRequest>();
        let mut handles = Vec::new();
        for id in 0..parallelism {
            let receiver = receiver.clone();
            let completion_sender = completion_sender.clone();

            let handle = thread::spawn(move || loop {
                match receiver.recv() {
                    Ok(req) => {
                        let path = req.path.clone();
                        tracing::trace!("Request on {id}:{:?}", path);
                        req.eval();
                        let _ = completion_sender.send(path);
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

    pub fn completed(&self) -> Vec<PathBuf> {
        let mut completed = Vec::new();
        while let Ok(path) = self.completion_receiver.try_recv() {
            completed.push(path)
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
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        tracing::trace!("Loading {:?}", path.as_ref());

        let mut decoder = image::ImageReader::open(&path)?.into_decoder()?;
        let icc_profile = decoder.icc_profile()?;
        let mut img = DynamicImage::from_decoder(decoder)?;

        const MAX_WIDTH: u32 = 3840;
        const MAX_HEIGHT: u32 = 2160;
        if img.width() > MAX_WIDTH || img.height() > MAX_HEIGHT {
            tracing::trace!(
                "Resizing {}x{} from {:?}",
                img.width(),
                img.height(),
                path.as_ref(),
            );
            img = img.resize(MAX_WIDTH, MAX_HEIGHT, image::imageops::FilterType::Triangle);
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

            tracing::trace!("Transforming {:?} via ICC profile", path.as_ref());
            t.transform_in_place(&mut bytes);
        }

        tracing::trace!("Finished: {:?}", path.as_ref());
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
    vertex_buffer: Option<Buffer>,
    device: Device,
    width: u32,
    height: u32,
    mapped: Arc<AtomicBool>,
    waiting_to_resize: Option<Vertices>,
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
            vertex_buffer: None,
            device: device.clone(),
            width: img.width,
            height: img.height,
            mapped: Arc::new(AtomicBool::new(false)),
            waiting_to_resize: None,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn render(&mut self, renderpass: &mut RenderPass) {
        if let Some(func) = std::mem::replace(&mut self.waiting_to_resize, None) {
            self.resize(func);
            return;
        }

        // Only render if vertex buffer is initialized
        if let Some(vertex_buffer) = &self.vertex_buffer {
            if !self.mapped.load(Ordering::Acquire) {
                renderpass.set_bind_group(0, &self.bind_group, &[]);
                renderpass.set_vertex_buffer(0, vertex_buffer.slice(..));
                renderpass.draw(0..6, 0..1);
            }
        }
    }

    pub fn resize(&mut self, vertices: Vertices) {
        // tracing::debug!("{vertices:?}");
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
}
