use std::{
    fmt::Debug,
    fs::{self, DirEntry},
    ops::Mul,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{anyhow, Result};
use clap::Parser;
use itertools::Itertools;
use wgpu::{Features, TextureFormat, TextureUsages};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, MouseButton, MouseScrollDelta, TouchPhase, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

mod lazy;

use lazy::{ImageLoaderServiceHandle, ImageResizeSpec, LazyImage, SingleImageResizeSpec};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
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
}

impl Cli {
    fn get_paths(&self) -> Result<Vec<PathBuf>> {
        let recursion = self.recursive.then_some(self.recursion_limit).unwrap_or(0);

        let target = match self.target.metadata()?.is_dir() {
            true => self.target.clone(),
            false => self.target.parent().ok_or(anyhow!("None"))?.to_path_buf(),
        };

        let mut iter: Box<dyn Iterator<Item = _>> =
            Box::new(Self::open_dir(&target, recursion)?.map(|entry| entry.path()));

        if !self.ignore_ext {
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

        Ok(iter.collect())
    }

    fn open_dir(
        dir: impl AsRef<Path>,
        recursion: u8,
    ) -> Result<Box<dyn Iterator<Item = DirEntry>>> {
        Ok(Box::new(
            fs::read_dir(dir.as_ref())?
                .flatten()
                .flat_map(move |entry| {
                    let path = entry.path();

                    // so we traverse symbolic links
                    let ft = std::fs::metadata(&path)
                        .expect("Failed to get metadata")
                        .file_type();

                    if ft.is_file() {
                        let iter: Box<dyn Iterator<Item = DirEntry>> =
                            Box::new(std::iter::once(entry));
                        return Some(iter);
                    }

                    if recursion > 0 && ft.is_dir() {
                        return Self::open_dir(path, recursion.saturating_sub(1)).ok();
                    }

                    // TODO write a recursive func which will traverse a set
                    // number of symbolic links
                    if ft.is_symlink() {
                        return None;
                    }

                    // filter other stuff, such are sockets
                    None
                })
                .flatten(),
        ))
    }
}

struct State {
    window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface<'static>,
    surface_format: TextureFormat,
    render_pipeline: wgpu::RenderPipeline,
    invert_pipeline: wgpu::RenderPipeline,
    selection_buffer: wgpu::Buffer,
    selection_bind_group: wgpu::BindGroup,
    images: Vec<LazyImage>,
    rows: u32,
    selected_idx: usize,
    offset: u64,
    /// used to ensure things are resized after an image removal
    needs_resize: bool,
    // Display mode
    mode: Mode,
    // Single image mode state
    single_image_zoom: f32,
    single_image_pan_x: f32,
    single_image_pan_y: f32,
    // Mouse drag state
    mouse_current_pos: Option<(f64, f64)>,
    // When dragging, this stores the point on the image (in normalized -1 to 1 coords) that was clicked
    drag_anchor_image_pos: Option<(f32, f32)>,
}

impl State {
    async fn new(window: Arc<Window>, cli: &Cli) -> State {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: Features::MAPPABLE_PRIMARY_BUFFERS
                    | Features::TEXTURE_FORMAT_16BIT_NORM,
                ..Default::default()
            })
            .await
            .unwrap();

        let size = window.inner_size();

        let surface = instance.create_surface(window.clone()).unwrap();
        let cap = surface.get_capabilities(&adapter);
        let surface_format = cap.formats[0];

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            label: None,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        // Position
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        // Texture coordinates
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format.add_srgb_suffix(),
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Create invert pipeline with subtractive blending for color inversion
        let invert_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Invert Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float32x2,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format.add_srgb_suffix(),
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::OneMinusDst,
                            dst_factor: wgpu::BlendFactor::Zero,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: 4, // Might not be needed with mipmapping
            ..Default::default()
        });

        let selection_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Selection indicator buffer"),
            size: std::mem::size_of::<[f32; 24]>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create a dummy 1x1 white texture for the selection indicator
        let dummy_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dummy selection texture"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &dummy_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8, 255u8, 255u8, 255u8], // White pixel
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let dummy_texture_view = dummy_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let selection_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&dummy_texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
            label: Some("Selection indicator bind group"),
        });

        let image_loader_service_handle =
            ImageLoaderServiceHandle::new(&device, &queue, &bind_group_layout, &sampler, 0);

        let paths = cli.get_paths().unwrap();
        tracing::info!("Path count: {}", paths.len());
        const NO_FIRST_DISPLAYED: usize = 10;
        tracing::debug!(
            "First {NO_FIRST_DISPLAYED} paths: {:?}",
            &paths[..paths.len().min(NO_FIRST_DISPLAYED)]
        );

        let images: Vec<_> = paths
            .into_iter()
            .map(|path| LazyImage::new(path, None, image_loader_service_handle.clone_sender()))
            .collect();

        let state = Self {
            window,
            device,
            queue,
            size,
            surface,
            surface_format,
            render_pipeline,
            invert_pipeline,
            selection_buffer,
            selection_bind_group,
            images,
            rows: 3,
            selected_idx: 0,
            offset: 0,
            needs_resize: false,
            mode: Mode::Gallery,
            single_image_zoom: 1.0,
            single_image_pan_x: 0.0,
            single_image_pan_y: 0.0,
            mouse_current_pos: None,
            drag_anchor_image_pos: None,
        };

        state.configure_surface();

        state
    }

    fn configure_surface(&self) {
        let surface_config = wgpu::SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: self.surface_format,
            view_formats: vec![self.surface_format.add_srgb_suffix()],
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            width: self.size.width,
            height: self.size.height,
            desired_maximum_frame_latency: 2,
            present_mode: wgpu::PresentMode::AutoVsync,
        };
        self.surface.configure(&self.device, &surface_config);
    }

    /// Calculate the number of columns in the current grid
    fn get_cols(&self) -> u64 {
        ImageResizeSpec::new(self.size.width, self.size.height, 0, self.rows, 0)
            .cols
            .max(1) as u64
    }

    /// Calculate the row offset needed to ensure selected_idx is visible
    fn calculate_offset(&self, current_offset: u64) -> u64 {
        let cols = self.get_cols();
        let selected_row = (self.selected_idx as u64) / cols;

        // Calculate the current visible row range
        let first_visible_row = current_offset;
        let last_visible_row = current_offset + (self.rows as u64).saturating_sub(1);

        // Only scroll if selected row is outside the visible range
        if selected_row < first_visible_row {
            // Selected is above visible area, scroll up to show it at the top
            selected_row
        } else if selected_row > last_visible_row {
            // Selected is below visible area, scroll down to show it at the bottom
            selected_row.saturating_sub((self.rows as u64).saturating_sub(1))
        } else {
            // Selected is already visible, don't change offset
            current_offset
        }
    }

    /// Scroll the gallery view by a number of rows
    fn gallery_scroll(&mut self, row_delta: i64) {
        let cols = self.get_cols();

        // Calculate max offset (last row that could be at the top)
        let total_rows = (self.images.len() as u64 + cols - 1) / cols; // Ceiling division
        let max_offset = total_rows.saturating_sub(self.rows as u64);

        // Apply scroll delta
        if row_delta > 0 {
            self.offset = (self.offset + row_delta as u64).min(max_offset);
        } else if row_delta < 0 {
            self.offset = self.offset.saturating_sub((-row_delta) as u64);
        }

        // Ensure selected_idx is within visible range
        let selected_row = (self.selected_idx as u64) / cols;
        let first_visible_row = self.offset;
        let last_visible_row = self.offset + (self.rows as u64).saturating_sub(1);

        if selected_row < first_visible_row {
            // Selected is above visible area, move it to the first visible row
            self.selected_idx = (first_visible_row * cols) as usize;
        } else if selected_row > last_visible_row {
            // Selected is below visible area, move it to the last visible row
            self.selected_idx = ((last_visible_row * cols) as usize).min(self.images.len() - 1);
        }

        self.resize(None);
    }

    /// Calculate the range of visible image indices
    fn get_visible_range(&self) -> (usize, usize) {
        let cols = self.get_cols() as usize;
        let rows = self.rows as usize;
        let visible_start = (self.offset as usize) * cols;
        let visible_end = (visible_start + (cols * rows)).min(self.images.len());
        (visible_start, visible_end)
    }

    /// Remove images at the specified indices (in reverse order to maintain correct indices)
    fn remove_images_at(&mut self, mut indices: Vec<usize>) {
        if indices.len() > 0 {
            self.needs_resize = true;
        }

        // Sort in reverse order
        indices.sort_unstable_by(|a, b| b.cmp(a));
        for idx in indices {
            let img = self.images.remove(idx);
            tracing::debug!("Removed: {:?}", img.path);
        }
    }

    /// Calculate pan offsets to keep the drag anchor point under the cursor
    fn calculate_pan_for_drag(&self) -> (f32, f32) {
        if let (Some((cursor_x, cursor_y)), Some((anchor_x, anchor_y))) =
            (self.mouse_current_pos, self.drag_anchor_image_pos)
        {
            // Convert cursor position from window coords to normalized device coords (-1 to 1)
            let cursor_ndc_x = (cursor_x / self.size.width as f64) * 2.0 - 1.0;
            let cursor_ndc_y = 1.0 - (cursor_y / self.size.height as f64) * 2.0;

            // The pan offset should position the image such that anchor_x/y appears at cursor_ndc_x/y
            // anchor point in image space + pan = cursor position in NDC
            // Therefore: pan = cursor_ndc - anchor
            let pan_x = cursor_ndc_x as f32 - anchor_x;
            let pan_y = cursor_ndc_y as f32 - anchor_y;

            (pan_x, pan_y)
        } else {
            (self.single_image_pan_x, self.single_image_pan_y)
        }
    }

    fn resize(&mut self, new_size: Option<winit::dpi::PhysicalSize<u32>>) {
        if let Some(new_size) = new_size {
            self.size = new_size;
        }

        let mut errors_to_remove = Vec::new();

        match self.mode {
            Mode::Gallery => {
                // Calculate offset to keep selected_idx visible (only scroll if needed)
                self.offset = self.calculate_offset(self.offset);

                // Calculate visible range
                let (visible_start, visible_end) = self.get_visible_range();

                // Only resize visible images
                for (idx, img) in self.images.iter_mut().enumerate() {
                    if idx >= visible_start && idx < visible_end {
                        let relative_slot = (idx - visible_start) as u32;
                        tracing::trace!("Resizing: {:?}", img.path);

                        let spec = ImageResizeSpec::new(
                            self.size.width,
                            self.size.height,
                            relative_slot,
                            self.rows,
                            0, // offset is 0 because we already calculated the visible start
                        );

                        // Mark as selected if this is the selected index
                        img.selected = idx == self.selected_idx;

                        match img.resize(spec) {
                            Ok(()) => {}
                            Err(e) => {
                                tracing::warn!("{e:?}");
                                errors_to_remove.push(idx);
                            }
                        }
                    } else {
                        // Mark as selected even if not visible (for when scrolling)
                        img.selected = idx == self.selected_idx;
                    }
                }
            }
            Mode::SingleImage => {
                // Calculate pan based on drag state
                let (pan_x, pan_y) = self.calculate_pan_for_drag();

                // Create spec first to avoid borrow checker issues
                let spec = SingleImageResizeSpec::new(
                    self.size.width,
                    self.size.height,
                    self.single_image_zoom,
                    pan_x,
                    pan_y,
                );

                // Resize only the selected image for single image mode
                if let Some(img) = self.images.get_mut(self.selected_idx) {
                    tracing::trace!("Resizing (SingleImage): {:?}", img.path);

                    img.selected = true;

                    match img.resize_single_image(spec) {
                        Ok(()) => {}
                        Err(e) => {
                            tracing::warn!("{e:?}");
                            errors_to_remove.push(self.selected_idx);
                        }
                    }
                }

                // Mark all other images as not selected
                for (idx, img) in self.images.iter_mut().enumerate() {
                    if idx != self.selected_idx {
                        img.selected = false;
                    }
                }
            }
        }

        // Remove images that had errors
        self.remove_images_at(errors_to_remove);

        self.configure_surface();
    }

    fn set_needs_resize(&mut self) {
        self.needs_resize = true;
    }

    fn resize_if_needed(&mut self) {
        if self.needs_resize {
            self.resize(None);
            self.needs_resize = false;
        }
    }

    fn render(&mut self) -> bool {
        let surface_texture = self
            .surface
            .get_current_texture()
            .expect("failed to acquire next swapchain texture");
        let texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor {
                format: Some(self.surface_format.add_srgb_suffix()),
                ..Default::default()
            });

        let mut encoder = self.device.create_command_encoder(&Default::default());
        let mut renderpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        renderpass.set_pipeline(&self.render_pipeline);

        let mut errors_to_remove = Vec::new();

        match self.mode {
            Mode::Gallery => {
                // Calculate visible range
                let (visible_start, visible_end) = self.get_visible_range();

                // Render only visible images
                for (idx, img) in self.images.iter_mut().enumerate() {
                    if idx >= visible_start && idx < visible_end {
                        match img.render(&mut renderpass) {
                            Ok(()) => {}
                            Err(e) => {
                                tracing::trace!("{e:?}");
                                errors_to_remove.push(idx);
                            }
                        }
                    }
                }

                // Render selection indicator after all images
                // Find the selected image and render its indicator
                for img in &self.images {
                    if let Some(vertices) = img.get_selection_indicator_vertices() {
                        self.queue.write_buffer(
                            &self.selection_buffer,
                            0,
                            bytemuck::cast_slice(&vertices),
                        );
                        renderpass.set_pipeline(&self.invert_pipeline);
                        renderpass.set_bind_group(0, &self.selection_bind_group, &[]);
                        renderpass.set_vertex_buffer(0, self.selection_buffer.slice(..));
                        renderpass.draw(0..6, 0..1);
                        break; // Only one image should be selected
                    }
                }
            }
            Mode::SingleImage => {
                // Render only the selected image
                if let Some(img) = self.images.get_mut(self.selected_idx) {
                    match img.render(&mut renderpass) {
                        Ok(()) => {}
                        Err(e) => {
                            tracing::trace!("{e:?}");
                            errors_to_remove.push(self.selected_idx);
                        }
                    }
                }
            }
        }

        // Remove images that had errors
        self.remove_images_at(errors_to_remove);

        // Check if we have no more images to display
        if self.images.is_empty() {
            eprintln!(
                "{}: no more files to display, aborting",
                env!("CARGO_PKG_NAME")
            );
            return false;
        }

        drop(renderpass);

        self.queue.submit([encoder.finish()]);
        self.window.pre_present_notify();
        surface_texture.present();

        true
    }

    fn move_left(&mut self) {
        self.reset_single_image_pos();
        if self.selected_idx > 0 {
            self.selected_idx -= 1;
            self.resize(None);
        }
    }

    fn move_right(&mut self) {
        self.reset_single_image_pos();
        if self.selected_idx + 1 < self.images.len() {
            self.selected_idx += 1;
            self.resize(None);
        }
    }

    fn move_up(&mut self) {
        self.reset_single_image_pos();
        let cols = self.get_cols() as usize;
        if self.selected_idx >= cols {
            self.selected_idx -= cols;
            self.resize(None);
        }
    }

    fn move_down(&mut self) {
        self.reset_single_image_pos();
        let cols = self.get_cols() as usize;
        if self.selected_idx + cols < self.images.len() {
            self.selected_idx += cols;
        } else if self.selected_idx < self.images.len() - 1 {
            // Can't move down a full row, so go to the last image
            self.selected_idx = self.images.len() - 1;
        }
        self.resize(None);
    }

    fn gallery_zoom(&mut self, delta: i32) {
        match delta {
            0 => self.rows = 3,
            1..=i32::MAX => self.rows = self.rows.saturating_sub(delta as u32).max(2),
            i32::MIN..=-1 => self.rows = self.rows.saturating_add((-delta) as u32).min(32),
        }
        self.resize(None);
    }

    fn single_image_zoom(&mut self, factor: f32) {
        let old_zoom = self.single_image_zoom;
        let new_zoom = if factor == 0.0 {
            // factor == 0.0 means reset
            1.0
        } else {
            (old_zoom * factor).clamp(0.1, 10.0)
        };

        // If not dragging, adjust pan to keep center point fixed
        if self.drag_anchor_image_pos.is_none() {
            if factor == 0.0 {
                // Reset pan as well
                self.single_image_pan_x = 0.0;
                self.single_image_pan_y = 0.0;
            } else {
                let zoom_ratio = new_zoom / old_zoom;
                self.single_image_pan_x *= zoom_ratio;
                self.single_image_pan_y *= zoom_ratio;
            }
        }
        // If dragging, the drag anchor will be preserved automatically

        self.single_image_zoom = new_zoom;
        self.resize(None);
    }

    fn reset_single_image_pos(&mut self) {
        self.single_image_zoom = 1.0;
        self.single_image_pan_x = 0.0;
        self.single_image_pan_y = 0.0;
    }

    fn toggle_mode(&mut self) {
        self.mode = match self.mode {
            Mode::Gallery => {
                // Reset zoom and pan when entering single image mode
                self.reset_single_image_pos();
                Mode::SingleImage
            }
            Mode::SingleImage => Mode::Gallery,
        };
        self.resize(None);
    }

    /// Find the image index under the given cursor position in gallery mode
    /// Returns None if the cursor is not over any image
    fn get_image_at_cursor(&self, cursor_x: f64, cursor_y: f64) -> Option<usize> {
        if self.mode != Mode::Gallery {
            return None;
        }

        // Convert cursor position to normalized device coordinates (-1 to 1)
        let ndc_x = (cursor_x / self.size.width as f64) * 2.0 - 1.0;
        let ndc_y = 1.0 - (cursor_y / self.size.height as f64) * 2.0;

        // Get the grid layout parameters
        let cols = self.get_cols();
        let spec = ImageResizeSpec::new(self.size.width, self.size.height, 0, self.rows, 0);
        let col_unit = spec.col_unit as f64;
        let row_unit = spec.row_unit as f64;
        let col_margin = spec.col_margin as f64;

        // Calculate which column and row the cursor is in
        // Account for the centered margin
        let adjusted_x = ndc_x - (col_margin / 2.0);

        // Check if cursor is within the grid bounds
        if adjusted_x < -1.0 || adjusted_x >= -1.0 + (cols as f64 * col_unit) {
            return None;
        }
        if ndc_y > 1.0 || ndc_y < -1.0 {
            return None;
        }

        // Calculate grid position
        let col = ((adjusted_x + 1.0) / col_unit).floor() as i64;
        let row = ((1.0 - ndc_y) / row_unit).floor() as i64;

        // Convert to absolute grid position (accounting for offset)
        let abs_row = row + self.offset as i64;

        if col < 0 || row < 0 || abs_row < 0 {
            return None;
        }

        // Calculate the image index
        let idx = (abs_row as u64 * cols + col as u64) as usize;

        // Check if this index is valid
        if idx < self.images.len() {
            Some(idx)
        } else {
            None
        }
    }
}

struct App {
    state: Option<State>,
    shifted: bool,
    cli: Cli,
}

impl App {
    fn new(cli: Cli) -> Self {
        Self {
            state: None,
            shifted: false,
            cli,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create window object
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );

        let state = pollster::block_on(State::new(window.clone(), &self.cli));
        self.state = Some(state);

        window.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let state = self.state.as_mut().unwrap();
        match event {
            WindowEvent::CloseRequested => {
                tracing::debug!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // will trigger a resize if any images were removed during the render call
                state.resize_if_needed();

                if !state.render() {
                    // No more images to display
                    event_loop.exit();
                    return;
                }
                // Emits a new redraw requested event.
                state.window.request_redraw();
            }
            WindowEvent::Resized(size) => {
                tracing::debug!("Window Resize: {size:?}");
                // Reconfigures the size of the surface. We do not re-render
                // here as this event is always followed up by redraw request.
                state.resize(Some(size));
            }
            WindowEvent::CursorMoved { position, .. } => {
                state.mouse_current_pos = Some((position.x, position.y));

                if state.mode == Mode::SingleImage && state.drag_anchor_image_pos.is_some() {
                    // Recalculate position to keep anchor under cursor
                    state.set_needs_resize();
                }
            }
            WindowEvent::MouseInput {
                state: button_state,
                button,
                ..
            } => {
                match button {
                    MouseButton::Left => {
                        if button_state == ElementState::Pressed {
                            match state.mode {
                                Mode::Gallery => {
                                    // In gallery mode, select the image under cursor and switch to single image mode
                                    if let Some((cursor_x, cursor_y)) = state.mouse_current_pos {
                                        if let Some(idx) =
                                            state.get_image_at_cursor(cursor_x, cursor_y)
                                        {
                                            state.selected_idx = idx;
                                        }
                                    }
                                    state.toggle_mode();
                                }
                                Mode::SingleImage => {
                                    // In single image mode, switch back to gallery
                                    state.toggle_mode();
                                }
                            }
                        }
                    }
                    MouseButton::Right => {
                        if state.mode == Mode::SingleImage {
                            match button_state {
                                ElementState::Pressed => {
                                    // Calculate which point on the image we clicked
                                    if let Some((cursor_x, cursor_y)) = state.mouse_current_pos {
                                        // Convert cursor to NDC
                                        let cursor_ndc_x =
                                            (cursor_x / state.size.width as f64) * 2.0 - 1.0;
                                        let cursor_ndc_y =
                                            1.0 - (cursor_y / state.size.height as f64) * 2.0;

                                        // The image point that appears at cursor_ndc is: cursor_ndc - pan
                                        let image_x =
                                            cursor_ndc_x as f32 - state.single_image_pan_x;
                                        let image_y =
                                            cursor_ndc_y as f32 - state.single_image_pan_y;

                                        state.drag_anchor_image_pos = Some((image_x, image_y));
                                    }
                                }
                                ElementState::Released => {
                                    // Update the stored pan to the current calculated pan
                                    if state.drag_anchor_image_pos.is_some() {
                                        let (pan_x, pan_y) = state.calculate_pan_for_drag();
                                        state.single_image_pan_x = pan_x;
                                        state.single_image_pan_y = pan_y;
                                    }
                                    state.drag_anchor_image_pos = None;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::MouseWheel {
                delta,
                phase: TouchPhase::Moved,
                ..
            } => {
                // tracing::debug!("{delta:?}");
                match (delta, state.mode) {
                    (MouseScrollDelta::LineDelta(_, y), Mode::Gallery) => {
                        // Negative y means scroll down (increase offset)
                        // Positive y means scroll up (decrease offset)
                        let scroll_delta = -y as i64;
                        if scroll_delta != 0 {
                            state.gallery_scroll(scroll_delta);
                        }
                    }
                    (MouseScrollDelta::PixelDelta(pos), Mode::Gallery) => {
                        // Convert pixel delta to rows (assuming ~20 pixels per row)
                        let scroll_delta = (-pos.y / 20.0).round() as i64;
                        if scroll_delta != 0 {
                            state.gallery_scroll(scroll_delta);
                        }
                    }
                    (MouseScrollDelta::LineDelta(_, y), Mode::SingleImage) => {
                        state.single_image_zoom(1.0 + y.mul(0.2));
                    }
                    (MouseScrollDelta::PixelDelta(_), Mode::SingleImage) => {}
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                match event.physical_key {
                    PhysicalKey::Code(KeyCode::ShiftLeft | KeyCode::ShiftRight) => {
                        self.shifted = match event.state {
                            ElementState::Pressed => true,
                            ElementState::Released => false,
                        };
                    }
                    _ => {}
                }

                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::KeyQ | KeyCode::Escape) => {
                            event_loop.exit();
                        }
                        PhysicalKey::Code(KeyCode::Enter) => {
                            state.toggle_mode();
                        }
                        PhysicalKey::Code(KeyCode::KeyH | KeyCode::ArrowLeft) => {
                            state.move_left();
                        }
                        PhysicalKey::Code(KeyCode::KeyL | KeyCode::ArrowRight) => {
                            state.move_right();
                        }
                        PhysicalKey::Code(KeyCode::KeyJ | KeyCode::ArrowDown) => {
                            state.move_down();
                        }
                        PhysicalKey::Code(KeyCode::KeyK | KeyCode::ArrowUp) => {
                            state.move_up();
                        }
                        PhysicalKey::Code(KeyCode::Equal) => {
                            if self.shifted {
                                match state.mode {
                                    Mode::Gallery => state.gallery_zoom(1),
                                    Mode::SingleImage => state.single_image_zoom(1.2),
                                }
                            } else {
                                match state.mode {
                                    Mode::Gallery => state.gallery_zoom(0),
                                    Mode::SingleImage => state.single_image_zoom(0.0),
                                }
                            }
                        }
                        PhysicalKey::Code(KeyCode::Minus) => match state.mode {
                            Mode::Gallery => state.gallery_zoom(-1),
                            Mode::SingleImage => state.single_image_zoom(1.0 / 1.2),
                        },
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let (stdout, _guard) = tracing_appender::non_blocking(std::io::stdout());

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("error")),
        )
        .with_writer(stdout)
        .with_target(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::new(Cli::parse());
    event_loop.run_app(&mut app).unwrap();
}
