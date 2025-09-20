use std::{
    fmt::Debug,
    fs::{self, DirEntry},
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::Result;
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
}

impl Cli {
    fn get_paths(&self) -> Result<Vec<PathBuf>> {
        let recursion = self.recursive.then_some(self.recursion_limit).unwrap_or(0);

        let target = match self.target.metadata()?.is_dir() {
            true => self.target.clone(),
            false => self.target.parent().unwrap().to_path_buf(),
        };

        Ok(Self::open_dir(&target, recursion)?
            .map(|entry| entry.path())
            .sorted_by(|a, b| {
                lexical_sort::natural_lexical_cmp(&a.to_string_lossy(), &b.to_string_lossy())
            })
            .collect())
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
    images: Vec<LazyImage>,
    rows: u32,
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
            images,
            rows: 3,
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

    fn resize(&mut self, new_size: Option<winit::dpi::PhysicalSize<u32>>) {
        if let Some(new_size) = new_size {
            self.size = new_size;
        }

        let cols = ImageResizeSpec::new(
            self.size.width,
            self.size.height,
            0,
            self.rows,
            self.offset as u32,
        )
        .cols
        .max(1) as u64; // HACK avoid div by zero

        let max_offset = (self.images.len() as u64)
            .div_ceil(cols)
            .saturating_sub(self.rows as u64);
        if self.offset > max_offset {
            self.offset = max_offset;
        }

        let mut slot: u32 = 0;
        self.images.retain_mut(|img| {
            log::debug!("Resizing: {:?}", img.path);
            let spec = ImageResizeSpec::new(
                self.size.width,
                self.size.height,
                slot,
                self.rows,
                self.offset as u32,
            );
            if spec.visible {
                match img.resize(spec) {
                    Ok(()) => {
                        slot += 1;
                        true
                    }
                    Err(e) => {
                        log::warn!("{e:?}");
                        false
                    }
                }
            } else {
                slot += 1;
                img.visible = false;
                true
            }
        });

        self.configure_surface();
    }

    fn render(&mut self) {
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

        let mut was_err = false;
        self.images.retain_mut(|img| {
            if img.visible {
                match img.render(&mut renderpass) {
                    Ok(()) => true,
                    Err(e) => {
                        log::warn!("{e:?}");
                        was_err = true;
                        false
                    }
                }
            } else {
                true
            }
        });

        drop(renderpass);

        self.queue.submit([encoder.finish()]);
        self.window.pre_present_notify();
        surface_texture.present();

        if was_err {
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
                state.render();
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
                        PhysicalKey::Code(KeyCode::KeyJ | KeyCode::ArrowDown) => {
                            state.offset = state.offset.saturating_add(1);
                            state.resize(None);
                        }
                        PhysicalKey::Code(KeyCode::KeyK | KeyCode::ArrowUp) => {
                            if state.offset != 0 {
                                state.offset = state.offset.saturating_sub(1);
                                state.resize(None);
                            }
                        }
                        PhysicalKey::Code(KeyCode::Equal) => {
                            if self.shifted {
                                state.rows = state.rows.saturating_sub(1).max(1);
                                state.resize(None);
                            } else {
                                if state.rows != 3 {
                                    state.rows = 3;
                                    state.resize(None);
                                }
                            }
                        }
                        PhysicalKey::Code(KeyCode::Minus) => {
                            state.rows = state.rows.saturating_add(1).min(32);
                            state.resize(None);
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
