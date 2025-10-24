use std::{
    ops::Add,
    path::{Path, PathBuf},
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
    pub state: LazyImageState,
    pub path: PathBuf,
    pub size: Option<ImageResizeSpec>,
    pub visible: bool,
    pub selected: bool,
    selection_pos: Option<(f32, f32, f32, f32, f32)>, // (x, y, col_unit, row_unit, col_margin)
}

impl LazyImage {
    pub fn new(
        path: PathBuf,
        size: Option<ImageResizeSpec>,
        req_sender: Sender<ImageRequest>,
    ) -> Self {
        Self {
            path,
            size,
            state: LazyImageState::Uninitialized(req_sender),
            visible: false,
            selected: false,
            selection_pos: None,
        }
    }

    pub fn resize(&mut self, size: ImageResizeSpec) -> Result<()> {
        self.poll()?;
        self.visible = size.visible;
        self.size = Some(size);

        // Store position for selection indicator regardless of state
        if size.visible {
            self.selection_pos = Some((
                size.pos_x,
                size.pos_y,
                size.col_unit,
                size.row_unit,
                size.col_margin,
            ));
        } else {
            self.selection_pos = None;
        }

        match &mut self.state {
            LazyImageState::Uninitialized(_) => {}
            LazyImageState::Requested(_) => {}
            LazyImageState::Initialized(img) => {
                img.renderable_image.resize(self.size.as_ref().unwrap());
            }
            LazyImageState::Error(_) => {}
        }

        Ok(())
    }

    pub fn render(&mut self, renderpass: &mut RenderPass) -> Result<()> {
        self.poll()?;

        match &mut self.state {
            LazyImageState::Uninitialized(sender) => {
                let (response_channel, receiver) = unbounded();
                sender
                    .send(ImageRequest::new(response_channel, self.path.clone()))
                    .unwrap();
                self.state = LazyImageState::Requested(receiver);
            }
            LazyImageState::Requested(_) => {}
            LazyImageState::Initialized(resp) => {
                resp.renderable_image.render(renderpass);
            }
            LazyImageState::Error(_) => unreachable!(),
        }

        Ok(())
    }

    /// Get selection indicator vertices if this image is selected
    /// Returns vertices regardless of initialization state
    pub fn get_selection_indicator_vertices(&self) -> Option<[f32; 24]> {
        if !self.selected {
            return None;
        }

        self.selection_pos
            .map(|(pos_x, pos_y, col_unit, row_unit, col_margin)| {
                let indicator_size = 0.1; // Size in grid units
                let margin_offset = col_margin / 2.0;
                #[rustfmt::skip]
                let indicator_vertices: [f32; 24] = [
                    // Position x-y, Texture x-y (texture coords don't matter for inversion)
                    -1.0 + margin_offset + (pos_x * col_unit), 1.0 - (pos_y * row_unit), 0.5, 0.5, // Top left
                    -1.0 + margin_offset + (pos_x * col_unit) + (indicator_size * col_unit), 1.0 - (pos_y * row_unit), 0.5, 0.5, // Top right
                    -1.0 + margin_offset + (pos_x * col_unit), 1.0 - (pos_y * row_unit) - (indicator_size * row_unit), 0.5, 0.5, // Bottom left
                    -1.0 + margin_offset + (pos_x * col_unit), 1.0 - (pos_y * row_unit) - (indicator_size * row_unit), 0.5, 0.5, // Bottom left
                    -1.0 + margin_offset + (pos_x * col_unit) + (indicator_size * col_unit), 1.0 - (pos_y * row_unit), 0.5, 0.5, // Top right
                    -1.0 + margin_offset + (pos_x * col_unit) + (indicator_size * col_unit), 1.0 - (pos_y * row_unit) - (indicator_size * row_unit), 0.5, 0.5, // Bottom right
                ];
                indicator_vertices
            })
    }

    fn poll(&mut self) -> Result<()> {
        match &self.state {
            LazyImageState::Uninitialized(_) => {}
            LazyImageState::Requested(receiver) => {
                if let Ok(resp) = receiver.try_recv() {
                    let mut resp = match resp {
                        Ok(resp) => resp,
                        Err(e) => {
                            let err = Err(anyhow!("{e:?}"));
                            self.state = LazyImageState::Error(e);
                            return err;
                        }
                    };

                    if let Some(size) = &self.size {
                        resp.renderable_image.resize(size);
                    }
                    self.state = LazyImageState::Initialized(resp);
                }
            }
            LazyImageState::Initialized(_) => {}
            LazyImageState::Error(e) => {
                return Err(anyhow!("{e:?}"));
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
pub enum LazyImageState {
    Uninitialized(Sender<ImageRequest>),
    Requested(Receiver<Result<ImageResponse>>),
    Initialized(ImageResponse),
    Error(anyhow::Error),
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct ImageResizeSpec {
    pub vp_width: u32,
    pub vp_height: u32,
    pub pos: u32,
    pub pos_x: f32,
    pub pos_y: f32,
    pub rows: u32,
    pub cols: u32,
    pub row_unit: f32,
    pub col_unit: f32,
    pub col_space: f32,
    pub col_margin: f32,
    pub visible: bool,
}

impl ImageResizeSpec {
    pub fn new(vp_width: u32, vp_height: u32, mut pos: u32, rows: u32, offset: u32) -> Self {
        let row_unit = 2.0 / rows as f32;

        let col_space = vp_width as f32 / (vp_height as f32 / rows as f32);
        let col_unit = 2.0 / col_space;
        let cols = col_space.trunc() as u32;
        let col_margin = (col_space % 1.0) * col_unit;

        let min_pos = cols * offset;
        let max_pos = (cols * offset) + (cols * rows);

        let visible = if pos >= min_pos && pos < max_pos {
            true
        } else {
            false
        };

        pos -= min_pos;

        // TODO rework sizing so it works on windows where height > width
        // HACK make these all use checked div/mod so we don't panic on zero
        let pos_x = pos.checked_rem(cols).unwrap_or(0) as f32;
        let pos_y = (pos - pos_x as u32).checked_div(cols).unwrap_or(0) as f32;

        Self {
            vp_width,
            vp_height,
            pos,
            rows,
            row_unit,
            col_margin,
            col_space,
            col_unit,
            cols,
            visible,
            pos_x,
            pos_y,
        }
    }
}

pub struct ImageRequest {
    pub response_channel: Sender<Result<ImageResponse>>,
    pub path: PathBuf,
}

impl ImageRequest {
    pub fn new(response_channel: Sender<Result<ImageResponse>>, path: PathBuf) -> Self {
        Self {
            response_channel,
            path,
        }
    }

    pub fn eval(
        self,
        device: &Device,
        queue: &Queue,
        bind_group_layout: &BindGroupLayout,
        sampler: &Sampler,
    ) {
        self.response_channel
            .send(self.eval_inner(device, queue, bind_group_layout, sampler))
            .unwrap();
    }

    pub fn eval_inner(
        &self,
        device: &Device,
        queue: &Queue,
        bind_group_layout: &BindGroupLayout,
        sampler: &Sampler,
    ) -> Result<ImageResponse> {
        let img = GenericImage::new(&self.path)?;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("eval_inner texture"),
            size: wgpu::Extent3d {
                width: img.width,
                height: img.height,
                depth_or_array_layers: 1,
            },
            // TODO add/try mipmapping
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: img.format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("eval_inner bind group"),
        });

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
                bytes_per_row: Some(img.width * img.pixel_width), // Assuming RGBA
                rows_per_image: Some(img.height),
            },
            wgpu::Extent3d {
                width: img.width,
                height: img.height,
                depth_or_array_layers: 1,
            },
        );

        let renderable_image = RenderableImage::new(&device, bind_group, img.width, img.height);

        Ok(ImageResponse { renderable_image })
    }
}

#[derive(Debug)]
pub struct ImageResponse {
    renderable_image: RenderableImage,
}

pub struct ImageLoaderServiceHandle {
    sender: Sender<ImageRequest>,
    #[allow(dead_code)]
    handles: Vec<JoinHandle<()>>,
}

impl ImageLoaderServiceHandle {
    pub fn new(
        device: &Device,
        queue: &Queue,
        layout: &BindGroupLayout,
        sampler: &Sampler,
        parallelism: usize,
    ) -> Self {
        let parallelism = match (parallelism, thread::available_parallelism()) {
            (0, Ok(b)) => b.get().min(4),
            (a, Ok(b)) => a.min(b.get()).min(4),
            (0, Err(_)) => 1,
            (a, Err(_)) => a.min(4),
        };
        log::info!("ImageLoaderService parallelism: {parallelism}");

        let (sender, receiver) = unbounded::<ImageRequest>();
        let mut handles = Vec::new();
        for id in 0..parallelism {
            let receiver = receiver.clone();
            let device = device.clone();
            let queue = queue.clone();
            let layout = layout.clone();
            let sampler = sampler.clone();

            let handle = thread::spawn(move || loop {
                match receiver.recv() {
                    Ok(req) => {
                        log::debug!("{id}:{:?}", req.path);
                        req.eval(&device, &queue, &layout, &sampler);
                    }
                    Err(_) => break,
                };
            });
            handles.push(handle);
        }

        Self { sender, handles }
    }

    pub fn clone_sender(&self) -> Sender<ImageRequest> {
        self.sender.clone()
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
        log::debug!("Path: {:?}", path.as_ref());

        let mut decoder = image::ImageReader::open(&path)?.into_decoder()?;
        let icc_profile = decoder.icc_profile()?;
        let mut img = DynamicImage::from_decoder(decoder)?;

        const MAX_WIDTH: u32 = 2u32.pow(11);
        const MAX_HEIGHT: u32 = 2u32.pow(10);
        if img.width() > MAX_WIDTH || img.height() > MAX_HEIGHT {
            log::debug!(
                "Resize: {}x{} Len {} from {:?}",
                img.width(),
                img.height(),
                img.as_bytes().len(),
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

            log::debug!("Transforming {:?} via ICC profile.", path.as_ref());
            t.transform_in_place(&mut bytes);
        }

        log::debug!("Finished: {:?}", path.as_ref());
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
    pub bind_group: BindGroup,
    pub vertex_buffer: Buffer,
    pub width: u32,
    pub height: u32,
    mapped: Arc<AtomicBool>,
    waiting_to_resize: Option<ImageResizeSpec>,
}

impl RenderableImage {
    pub fn new(device: &Device, bind_group: BindGroup, width: u32, height: u32) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RenderableImage vertex buffer"),
            contents: bytemuck::cast_slice(&[0.0f32; 6 * 4]),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::MAP_WRITE,
        });

        Self {
            bind_group,
            vertex_buffer,
            width,
            height,
            mapped: Arc::new(AtomicBool::new(false)),
            waiting_to_resize: None,
        }
    }

    pub fn render(&mut self, renderpass: &mut RenderPass) {
        if let Some(size) = self.waiting_to_resize {
            self.resize(&size);
            return;
        }

        if !self.mapped.load(Ordering::Acquire) {
            renderpass.set_bind_group(0, &self.bind_group, &[]);
            renderpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            renderpass.draw(0..6, 0..1);
        }
    }

    pub fn resize(&mut self, size: &ImageResizeSpec) {
        if self.mapped.load(Ordering::Acquire) {
            self.waiting_to_resize = Some(*size);
            return;
        } else {
            self.waiting_to_resize = None;
        }

        if !size.visible {
            return;
        }

        let pos_x = size.pos_x;
        let pos_y = size.pos_y;
        let col_unit = size.col_unit;
        let row_unit = size.row_unit;
        let col_margin = size.col_margin;

        let width = self.width;
        let height = self.height;

        let capturable = self.vertex_buffer.clone();
        self.mapped.store(true, Ordering::Release);
        let is_mapped = self.mapped.clone();
        self.vertex_buffer
            .map_async(wgpu::MapMode::Write, .., move |result| {
                if result.is_ok() {
                    #[rustfmt::skip]
                    let mut vertices: [f32; 24] = [
                        // Position x-y Texture x-y
                        -1.0 + (pos_x * col_unit), 1.0 - (pos_y.add(1.0) * row_unit), 0.0, 1.0, // Bottom left
                        -1.0 + (pos_x.add(1.0) * col_unit), 1.0 - (pos_y.add(1.0) * row_unit), 1.0, 1.0, // Bottom right
                        -1.0 + (pos_x * col_unit), 1.0 - (pos_y * row_unit), 0.0, 0.0, // Top left
                        -1.0 + (pos_x * col_unit), 1.0 - (pos_y * row_unit), 0.0, 0.0, // Top left
                        -1.0 + (pos_x.add(1.0) * col_unit), 1.0 - (pos_y.add(1.0) * row_unit), 1.0, 1.0, // Bottom right
                        -1.0 + (pos_x.add(1.0) * col_unit), 1.0 - (pos_y * row_unit), 1.0, 0.0, // Top right
                    ];

                    vertices.chunks_mut(4).for_each(|slice| {
                        slice[0] = slice[0] + (col_margin / 2.0);
                    });

                    let aspect = width as f32 / height as f32;
                    match aspect {
                        x if x > 1.0 => {
                            let error = 1.0 / aspect - 1.0;
                            let half_abs_err = error.abs() / 2.0;
                            let unit_offset = half_abs_err * row_unit;
                            vertices[1] += unit_offset; // Bottom left
                            vertices[5] += unit_offset; // Bottom right
                            vertices[9] -= unit_offset; // Top left
                            vertices[13] -= unit_offset; // Top left
                            vertices[17] += unit_offset; // Bottom right
                            vertices[21] -= unit_offset; // Top right
                        }
                        x if x < 1.0 => {
                            let error = aspect - 1.0;
                            let half_abs_err = error.abs() / 2.0;
                            let unit_offset = half_abs_err * col_unit;
                            vertices[0] += unit_offset; // Bottom left
                            vertices[4] -= unit_offset; // Bottom right
                            vertices[8] += unit_offset; // Top left
                            vertices[12] += unit_offset; // Top left
                            vertices[16] -= unit_offset; // Bottom right
                            vertices[20] -= unit_offset; // Top right
                        }
                        _ => {}
                    }

                    let mut view = capturable.get_mapped_range_mut(..);
                    let floats: &mut [f32] = bytemuck::cast_slice_mut(&mut view);
                    floats.copy_from_slice(&vertices[..]);
                    drop(view);
                    capturable.unmap();
                    is_mapped.store(false, Ordering::Release);
                }
            });
    }
}
