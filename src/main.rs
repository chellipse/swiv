use std::{
    fmt::Debug,
    fs::{self, DirEntry},
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{anyhow, Result};
use clap::Parser;
use itertools::Itertools;
use wgpu::{Features, TextureFormat, TextureUsages};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

mod lazy;

use lazy::{ImageLoaderServiceHandle, ImageResizeSpec, LazyImage};

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
                    let ft = entry
                        .metadata()
                        .expect("Failed to get metadata")
                        .file_type();

                    if ft.is_file() {
                        let iter: Box<dyn Iterator<Item = DirEntry>> =
                            Box::new(std::iter::once(entry));
                        return Some(iter);
                    }

                    if recursion > 0 && ft.is_dir() {
                        // if let Some(name) = path.file_name() {
                        //     if name.as_encoded_bytes().get(0) != Some(&b'.') {
                        //         return Self::open_dir(path, recursion.saturating_sub(1)).ok();
                        //     }
                        // }
                        return Self::open_dir(entry.path(), recursion.saturating_sub(1)).ok();
                    }

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
        log::info!("Path count: {}", paths.len());

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
        // Sort in reverse order
        indices.sort_unstable_by(|a, b| b.cmp(a));
        for idx in indices {
            self.images.remove(idx);
        }
    }

    fn resize(&mut self, new_size: Option<winit::dpi::PhysicalSize<u32>>) {
        if let Some(new_size) = new_size {
            self.size = new_size;
        }

        // Calculate offset to keep selected_idx visible (only scroll if needed)
        self.offset = self.calculate_offset(self.offset);

        // Calculate visible range
        let (visible_start, visible_end) = self.get_visible_range();

        let mut errors_to_remove = Vec::new();

        // Only resize visible images
        for (idx, img) in self.images.iter_mut().enumerate() {
            if idx >= visible_start && idx < visible_end {
                let relative_slot = (idx - visible_start) as u32;
                log::debug!("Resizing: {:?}", img.path);

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
                        log::warn!("{e:?}");
                        errors_to_remove.push(idx);
                    }
                }
            } else {
                // Mark as selected even if not visible (for when scrolling)
                img.selected = idx == self.selected_idx;
            }
        }

        // Remove images that had errors
        self.remove_images_at(errors_to_remove);

        self.configure_surface();
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

        // Calculate visible range
        let (visible_start, visible_end) = self.get_visible_range();

        let mut errors_to_remove = Vec::new();

        // Render only visible images
        for (idx, img) in self.images.iter_mut().enumerate() {
            if idx >= visible_start && idx < visible_end {
                match img.render(&mut renderpass) {
                    Ok(()) => {}
                    Err(e) => {
                        log::warn!("{e:?}");
                        errors_to_remove.push(idx);
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

        // Render selection indicator after all images
        // Find the selected image and render its indicator
        for img in &self.images {
            if let Some(vertices) = img.get_selection_indicator_vertices() {
                self.queue
                    .write_buffer(&self.selection_buffer, 0, bytemuck::cast_slice(&vertices));
                renderpass.set_pipeline(&self.invert_pipeline);
                renderpass.set_bind_group(0, &self.selection_bind_group, &[]);
                renderpass.set_vertex_buffer(0, self.selection_buffer.slice(..));
                renderpass.draw(0..6, 0..1);
                break; // Only one image should be selected
            }
        }

        drop(renderpass);

        self.queue.submit([encoder.finish()]);
        self.window.pre_present_notify();
        surface_texture.present();

        true
    }

    fn move_left(&mut self) {
        if self.selected_idx > 0 {
            self.selected_idx -= 1;
            self.resize(None);
        }
    }

    fn move_right(&mut self) {
        if self.selected_idx + 1 < self.images.len() {
            self.selected_idx += 1;
            self.resize(None);
        }
    }

    fn move_up(&mut self) {
        let cols = self.get_cols() as usize;
        if self.selected_idx >= cols {
            self.selected_idx -= cols;
            self.resize(None);
        }
    }

    fn move_down(&mut self) {
        let cols = self.get_cols() as usize;
        if self.selected_idx + cols < self.images.len() {
            self.selected_idx += cols;
        } else if self.selected_idx < self.images.len() - 1 {
            // Can't move down a full row, so go to the last image
            self.selected_idx = self.images.len() - 1;
        }
        self.resize(None);
    }

    fn zoom_in(&mut self) {
        self.rows = self.rows.saturating_sub(1).max(1);
        self.resize(None);
    }

    fn zoom_out(&mut self) {
        self.rows = self.rows.saturating_add(1).min(32);
        self.resize(None);
    }

    fn reset_zoom(&mut self) {
        if self.rows != 3 {
            self.rows = 3;
            self.resize(None);
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
                log::debug!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                // log::debug!("Redraw");
                if !state.render() {
                    // No more images to display
                    event_loop.exit();
                    return;
                }
                // Emits a new redraw requested event.
                state.window.request_redraw();
            }
            WindowEvent::Resized(size) => {
                log::debug!("RESIZED: {size:?}");
                // Reconfigures the size of the surface. We do not re-render
                // here as this event is always followed up by redraw request.
                state.resize(Some(size));
            }
            WindowEvent::CursorMoved { .. } => {}
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
                                state.zoom_in();
                            } else {
                                state.reset_zoom();
                            }
                        }
                        PhysicalKey::Code(KeyCode::Minus) => {
                            state.zoom_out();
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::Builder::from_default_env()
        .format_timestamp_millis()
        .init();

    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::new(Cli::parse());
    event_loop.run_app(&mut app).unwrap();
}
