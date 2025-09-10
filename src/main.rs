use std::{
    ops::Add,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::Result;
use clap::Parser;
use image::{DynamicImage, EncodableLayout, ImageDecoder};
use wgpu::{
    util::DeviceExt, BindGroup, Buffer, Device, Features, RenderPass, TextureFormat, TextureUsages,
};
use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowId},
};

#[derive(Debug, Clone, Parser)]
struct Cli {
    image: PathBuf,
}

#[derive(Debug)]
struct GenericImage {
    width: u32,
    height: u32,
    bytes: Vec<u8>,
    format: TextureFormat,
    pixel_width: u32,
}

impl GenericImage {
    fn new(path: impl AsRef<Path>) -> Result<Self> {
        log::debug!("Path: {:?}", path.as_ref());

        let mut decoder = image::ImageReader::open(&path)?.into_decoder()?;
        let icc_profile = decoder.icc_profile()?;
        let mut img = DynamicImage::from_decoder(decoder)?;

        const MAX_WIDTH: u32 = 2u32.pow(12);
        const MAX_HEIGHT: u32 = 2u32.pow(12);
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

        // let (bytes, format, pixel_width) = match img {
        //     DynamicImage::ImageRgb8(img) => {
        //         log::debug!("Processing ImageRgb8 format, converting to Rgba8UnormSrgb");
        //         (
        //             img.as_bytes()
        //                 .chunks(3)
        //                 .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], u8::MAX])
        //                 .collect(),
        //             TextureFormat::Rgba8Unorm,
        //             4,
        //         )
        //     }
        //     DynamicImage::ImageRgba8(img) => {
        //         log::debug!("Processing ImageRgba8 format, using Rgba8Unorm");
        //         (img.as_bytes().to_vec(), TextureFormat::Rgba8Unorm, 4)
        //     }
        //     DynamicImage::ImageRgb16(img) => {
        //         log::debug!(
        //             "Processing ImageRgb16 format, converting to Rgba8UnormSrgb with bit shifting"
        //         );
        //         (
        //             img.as_bytes()
        //                 .chunks(6)
        //                 .flat_map(|slice| {
        //                     let r = u16::from_ne_bytes([slice[0], slice[1]]) >> 8;
        //                     let g = u16::from_ne_bytes([slice[2], slice[3]]) >> 8;
        //                     let b = u16::from_ne_bytes([slice[4], slice[5]]) >> 8;
        //                     [r as u8, g as u8, b as u8, u8::MAX]
        //                 })
        //                 .collect(),
        //             TextureFormat::Rgba8UnormSrgb,
        //             4,
        //         )
        //     }
        //     DynamicImage::ImageRgba16(img) => {
        //         log::debug!(
        //             "Processing ImageRgba16 format, converting to Rgba8UnormSrgb with bit shifting"
        //         );
        //         (
        //             img.as_bytes()
        //                 .chunks(8)
        //                 .flat_map(|slice| {
        //                     let r = u16::from_ne_bytes([slice[0], slice[1]]) >> 8;
        //                     let g = u16::from_ne_bytes([slice[2], slice[3]]) >> 8;
        //                     let b = u16::from_ne_bytes([slice[4], slice[5]]) >> 8;
        //                     let a = u16::from_ne_bytes([slice[6], slice[7]]) >> 8;
        //                     [r as u8, g as u8, b as u8, a as u8]
        //                 })
        //                 .collect(),
        //             TextureFormat::Rgba8UnormSrgb,
        //             4,
        //         )
        //     }
        //     DynamicImage::ImageRgb32F(_) => {
        //         return Err(anyhow!(
        //             "Unhandled DynamicImage type: Rgb32F from {:?}",
        //             path.as_ref()
        //         ));
        //     }
        //     DynamicImage::ImageRgba32F(_) => {
        //         return Err(anyhow!(
        //             "Unhandled DynamicImage type: Rgba32F from {:?}",
        //             path.as_ref()
        //         ));
        //     }
        //     DynamicImage::ImageLuma8(_) => {
        //         log::debug!("Processing ImageLuma8 format, converting to Rgba8UnormSrgb via to_rgba8() from {:?}", path.as_ref());
        //         // This means it'll take up more space than technically needed on the gpu.
        //         // Could possilby handle this in the shader to avoid the conversion
        //         (
        //             img.to_rgba8().as_bytes().to_vec(),
        //             TextureFormat::Rgba8UnormSrgb,
        //             4,
        //         )
        //     }
        //     DynamicImage::ImageLumaA8(_) => {
        //         return Err(anyhow!(
        //             "Unhandled DynamicImage type: LumaA8 from {:?}",
        //             path.as_ref()
        //         ));
        //     }
        //     DynamicImage::ImageLuma16(_) => {
        //         return Err(anyhow!(
        //             "Unhandled DynamicImage type: Luma16 from {:?}",
        //             path.as_ref()
        //         ));
        //     }
        //     DynamicImage::ImageLumaA16(_) => {
        //         return Err(anyhow!(
        //             "Unhandled DynamicImage type: LumaA16 from {:?}",
        //             path.as_ref()
        //         ));
        //     }
        //     _ => return Err(anyhow!("Unhandled DynamicImage type: ")),
        // };
        // log::info!("Final Len: {}", bytes.len());

        // ig we'll see if this is too expensive
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

        Ok(Self {
            width,
            height,
            bytes,
            format,
            pixel_width,
        })
    }
}

struct RenderableImage {
    bind_group: BindGroup,
    vertex_buffer: Buffer,
    width: u32,
    height: u32,
    visible: bool,
}

impl RenderableImage {
    fn new(device: &Device, bind_group: BindGroup, width: u32, height: u32) -> Self {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[0.0f32; 6 * 4]),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::MAP_WRITE,
        });

        Self {
            bind_group,
            vertex_buffer,
            width,
            height,
            visible: false,
        }
    }

    fn render(&self, renderpass: &mut RenderPass) {
        if self.visible {
            renderpass.set_bind_group(0, &self.bind_group, &[]);
            renderpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            renderpass.draw(0..6, 0..1);
        }
    }

    fn resize(&mut self, vp_width: u32, vp_height: u32, pos: u32, rows: u32) {
        let row_unit = 2.0 / rows as f32;

        let col_space = vp_width as f32 / (vp_height as f32 / rows as f32);
        let col_unit = 2.0 / col_space;
        let cols = col_space.trunc() as u32;
        let col_margin = (col_space % 1.0) * col_unit;

        if pos >= (rows * cols) || cols == 0 {
            self.visible = false;
            return;
        } else {
            self.visible = true;
        }

        let pos_x = (pos % cols) as f32;
        let pos_y = ((pos - pos_x as u32) / cols) as f32;

        let width = self.width;
        let height = self.height;

        let capturable = self.vertex_buffer.clone();
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
                }
            });
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
    images: Vec<RenderableImage>,
    rows: u32,
}

impl State {
    async fn new(window: Arc<Window>) -> State {
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

        let limits = adapter.limits();
        log::info!("LIMITS: {:#?}", limits);

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

        let imgs: Vec<_> = std::env::args()
            .nth(1)
            .and_then(|dir_path| std::fs::read_dir(dir_path).ok())
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let path = entry.path();
                if path.is_file() {
                    path.to_str().and_then(|p| {
                        let img = GenericImage::new(p);
                        if img.is_err() {
                            log::warn!("{:?} from {p:?}", img.unwrap_err());
                            None
                        } else {
                            img.ok()
                        }
                    })
                } else {
                    None
                }
            })
            .collect();

        let images = imgs
            .iter()
            .map(|img| {
                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
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
                    label: Some("texture_bind_group"),
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

                RenderableImage::new(&device, bind_group, img.width, img.height)
            })
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
        for (i, img) in self.images.iter_mut().enumerate() {
            img.resize(self.size.width, self.size.height, i as u32, self.rows);
        }
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

        self.images
            .iter()
            .for_each(|img| img.render(&mut renderpass));

        drop(renderpass);

        self.queue.submit([encoder.finish()]);
        self.window.pre_present_notify();
        surface_texture.present();
    }
}

#[derive(Default)]
struct App {
    state: Option<State>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // Create window object
        let window = Arc::new(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );

        let state = pollster::block_on(State::new(window.clone()));
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
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::KeyQ) | PhysicalKey::Code(KeyCode::Escape) => {
                            event_loop.exit();
                        }
                        _ => {}
                    }

                    match event.logical_key.to_text() {
                        Some("+") => {
                            state.rows = state.rows.saturating_sub(1).max(1);
                            state.resize(None);
                        }
                        Some("-") => {
                            state.rows = state.rows.saturating_add(1).min(64);
                            state.resize(None);
                        }
                        _ => {}
                    }
                }
                // log::info!("EVENT: {event:?}");
            }
            _ => {
                // log::info!("EVENT: {event:?}");
            }
        }
    }
}

fn main() {
    env_logger::Builder::from_default_env()
        .format_timestamp_millis()
        .init();

    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
