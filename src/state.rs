use std::{
    cell::RefCell,
    sync::{Arc, atomic::AtomicU32},
};

use crossbeam_channel::Sender;
use wgpu::{Features, TextureFormat, TextureUsages, wgt::PollType};
use winit::window::{Fullscreen, Window};

use crate::{
    bar::{self, BarMetrics},
    cache::{ImageCache, LoadClass},
    font::BarFonts,
    lazy::{
        GenericImage, ImageRequest, LazyImage, LazyVertexBuffer, MipGenerator,
        RenderableImage, Vertices,
    },
};

/// Everything the path bar's pixels depend on
#[derive(Debug, PartialEq)]
struct PathBarKey {
    path: String,
    position: String,
    width: u32,
    metrics: BarMetrics,
    /// faces only get appended, so this changes whenever they do
    face_count: usize,
}

#[derive(Debug, Default)]
struct PathBar {
    key: Option<PathBarKey>,
    renderable: Option<RenderableImage>,
}

#[derive(Debug)]
pub struct State {
    window: Arc<Window>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface<'static>,
    surface_format: TextureFormat,
    render_pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    image_cache: RefCell<ImageCache>,
    mip_generator: MipGenerator,
    // gallery mode cursor rendering
    invert_pipeline: wgpu::RenderPipeline,
    cursor_buffer: RefCell<LazyVertexBuffer>,
    cursor_bind_group: wgpu::BindGroup,
    // single image mode path bar
    path_bar: RefCell<PathBar>,
}

impl State {
    pub async fn new(
        window: Arc<Window>,
        req_sender: Sender<ImageRequest>,
        gpu_budget: u64,
        max_texture_dim: Arc<AtomicU32>,
    ) -> State {
        let instance = wgpu::Instance::new(
            wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
        );
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: Features::MAPPABLE_PRIMARY_BUFFERS
                    | Features::TEXTURE_FORMAT_16BIT_NORM,
                // the wgpu defaults stop at 8192 regardless of hardware;
                // native-resolution (Full) textures want the real limit
                required_limits: wgpu::Limits {
                    max_texture_dimension_2d: adapter
                        .limits()
                        .max_texture_dimension_2d,
                    ..Default::default()
                },
                ..Default::default()
            })
            .await
            .unwrap();

        // publish the real limit to the decode workers, which clamp Full
        // decodes to it
        max_texture_dim.store(
            device.limits().max_texture_dimension_2d,
            std::sync::atomic::Ordering::Relaxed,
        );
        tracing::info!(
            "Device texture size limit: {}",
            device.limits().max_texture_dimension_2d
        );

        let size = window.inner_size();

        let surface = instance.create_surface(window.clone()).unwrap();
        let cap = surface.get_capabilities(&adapter);
        let surface_format = cap.formats[0];

        let shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("shader.wgsl").into(),
                ),
            });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float {
                                filterable: true,
                            },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(
                            wgpu::SamplerBindingType::Filtering,
                        ),
                        count: None,
                    },
                ],
                label: None,
            });

        let pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let render_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Render Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<[f32; 4]>()
                            as wgpu::BufferAddress,
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
                                offset: std::mem::size_of::<[f32; 2]>()
                                    as wgpu::BufferAddress,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                        ],
                    })],
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
                multiview_mask: None,
                cache: None,
            });

        // Create invert pipeline with subtractive blending for color inversion
        let invert_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Invert Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<[f32; 4]>()
                            as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                offset: 0,
                                shader_location: 0,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                            wgpu::VertexAttribute {
                                offset: std::mem::size_of::<[f32; 2]>()
                                    as wgpu::BufferAddress,
                                shader_location: 1,
                                format: wgpu::VertexFormat::Float32x2,
                            },
                        ],
                    })],
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
                multiview_mask: None,
                cache: None,
            });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 4, // Might not be needed with mipmapping
            ..Default::default()
        });

        let cursor_buffer = RefCell::new(LazyVertexBuffer::new(&device));

        // Create a dummy 1x1 white texture for the cursor indicator
        let dummy_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Dummy cursor texture"),
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
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );

        let dummy_texture_view =
            dummy_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let cursor_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &dummy_texture_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                ],
                label: Some("Cursor indicator bind group"),
            });

        let mip_generator =
            MipGenerator::new(&device, TextureFormat::Rgba8UnormSrgb);

        let state = Self {
            window,
            device,
            queue,
            size,
            surface,
            surface_format,
            render_pipeline,
            invert_pipeline,
            cursor_buffer,
            cursor_bind_group,
            bind_group_layout,
            sampler,
            image_cache: RefCell::new(ImageCache::new(req_sender, gpu_budget)),
            mip_generator,
            path_bar: RefCell::default(),
        };

        state.configure_surface();

        state
    }

    pub fn toggle_fullscreen(&self) {
        let fullscreen = if self.window.fullscreen().is_some() {
            None
        } else {
            Some(Fullscreen::Borderless(None))
        };
        self.window.set_fullscreen(fullscreen);
    }

    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }

    pub fn resize_surface(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        self.size = new_size;
        self.configure_surface();
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
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        self.surface.configure(&self.device, &surface_config);
    }

    pub fn set_title(&self, title: &str) {
        self.window.set_title(title);
    }

    pub fn scale_factor(&self) -> f64 {
        self.window.scale_factor()
    }

    /// Re-rasterizes the path bar when anything it shows changed, so calling
    /// it every frame is cheap. The texture is exactly the bar's size and
    /// [`Self::render`] draws it under a viewport covering exactly the bar,
    /// so texels land 1:1 on pixels
    pub fn set_path_bar(
        &self,
        fonts: &BarFonts,
        metrics: BarMetrics,
        path: &str,
        position: &str,
    ) {
        let faces = fonts.faces();
        let key = PathBarKey {
            path: path.to_owned(),
            position: position.to_owned(),
            width: self.size.width,
            metrics,
            face_count: faces.len(),
        };

        let mut path_bar = self.path_bar.borrow_mut();
        if path_bar.key.as_ref() == Some(&key) {
            return;
        }

        // arriving fallbacks change the face count, which redraws the bar
        for text in [path, position] {
            fonts.request_missing(&faces, text);
        }

        path_bar.renderable =
            (key.width > 0 && metrics.height > 0).then(|| {
                let img =
                    bar::rasterize(&faces, metrics, key.width, path, position);
                let mut renderable = RenderableImage::new(
                    &img,
                    &self.device,
                    &self.queue,
                    &self.bind_group_layout,
                    &self.sampler,
                    None,
                );
                renderable.resize(Vertices::new());
                renderable
            });
        path_bar.key = Some(key);
    }

    /// See [`ImageCache::enforce_budget`]; true means the caller must
    /// re-lay out the visible set before the next render
    pub fn enforce_gpu_budget(&self) -> bool {
        self.image_cache.borrow_mut().enforce_budget()
    }

    fn upload(&self, img: &GenericImage, class: LoadClass) -> RenderableImage {
        // only Full gets a mip chain: thumbs are at most ~2x oversampled by
        // construction, while Full can be minified by orders of magnitude
        let mips = (class == LoadClass::Full).then_some(&self.mip_generator);

        RenderableImage::new(
            img,
            &self.device,
            &self.queue,
            &self.bind_group_layout,
            &self.sampler,
            mips,
        )
    }

    /// `advance` decides whether the cache may request/poll/upload during the
    /// lookup. Rendering must pass false: uploads create renderables with
    /// uninitialized vertex buffers, which would skip their draw for a frame
    /// (a visible flicker when replacing an already-displayed fallback). The
    /// resize paths pass true, as they lay out vertices right after upload.
    fn apply_op_to_image_states<'a>(
        &self,
        images: impl Iterator<Item = &'a LazyImage>,
        class: LoadClass,
        advance: bool,
        mut func: impl FnMut(&mut RenderableImage, usize),
    ) {
        let mut cache = self.image_cache.borrow_mut();
        for img in images {
            let renderable = match advance {
                true => cache.get_renderable(img.path(), class, |g, c| {
                    self.upload(g, c)
                }),
                false => cache.get_resident(img.path(), class),
            };

            if let Some(renderable) = renderable {
                func(renderable, img.get_pos());
            }
        }
    }

    pub fn resize<'a>(
        &self,
        images: impl Iterator<Item = &'a LazyImage>,
        class: LoadClass,
        ww: f32,
        wh: f32,
        row_no: f32,
        col_no: f32,
        cursor_idx: f32,
    ) {
        self.apply_op_to_image_states(
            images,
            class,
            true,
            |renderable, pos| {
                let iw = renderable.width() as f32;
                let ih = renderable.height() as f32;
                renderable.resize(
                    Vertices::new()
                        .gallery(ww, wh, iw, ih, row_no, col_no, pos as f32),
                );
            },
        );

        self.cursor_buffer.borrow_mut().resize(
            Vertices::new().gallery_cursor(ww, wh, row_no, col_no, cursor_idx),
        );
        let _ = self.device.poll(PollType::wait_indefinitely());
    }

    pub fn resize_single_image<'a>(
        &self,
        image: &'a LazyImage,
        ww: f32,
        wh: f32,
        scale_factor: f32,
        offset_x: f32,
        offset_y: f32,
    ) {
        let mut cache = self.image_cache.borrow_mut();
        if let Some(renderable) =
            cache.get_renderable(image.path(), LoadClass::Full, |g, c| {
                self.upload(g, c)
            })
        {
            let iw = renderable.width() as f32;
            let ih = renderable.height() as f32;
            renderable.resize(Vertices::new().single_image(
                ww,
                wh,
                iw,
                ih,
                scale_factor,
                offset_x,
                offset_y,
            ));

            let _ = self.device.poll(PollType::wait_indefinitely());
        }
    }

    /// `show_bar` must be the same decision the images were laid out with
    /// (their height is what's left above the bar), and the bar must have
    /// been set for the current frame via [`Self::set_path_bar`]
    pub fn render<'a>(
        &self,
        images: impl Iterator<Item = &'a LazyImage>,
        class: LoadClass,
        show_cursor: bool,
        show_bar: bool,
    ) {
        self.image_cache.borrow_mut().advance_frame();

        let mut path_bar = self.path_bar.borrow_mut();
        let (ww, wh) = (self.size.width, self.size.height);
        let bar = match (show_bar, &path_bar.key) {
            (true, Some(key)) => Some(key.metrics.height),
            _ => None,
        };
        debug_assert!(bar.is_none_or(|bar_h| bar_h < wh));

        let surface_texture = match self.surface.get_current_texture() {
            // suboptimal: the Resized event path reconfigures the surface,
            // so just render this frame as-is
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            other => {
                // the sleep stops the continuous-redraw loop from spinning
                // at full speed while e.g. occluded
                tracing::debug!("Skipping frame: {other:?}");
                std::thread::sleep(std::time::Duration::from_millis(16));
                return;
            }
        };
        let texture_view =
            surface_texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor {
                    format: Some(self.surface_format.add_srgb_suffix()),
                    ..Default::default()
                });

        let mut encoder =
            self.device.create_command_encoder(&Default::default());
        let mut renderpass =
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
                multiview_mask: None,
            });

        renderpass.set_pipeline(&self.render_pipeline);

        if let Some(bar_h) = bar {
            let content_h = (wh - bar_h) as f32;
            renderpass.set_viewport(0.0, 0.0, ww as f32, content_h, 0.0, 1.0);
        }

        self.apply_op_to_image_states(images, class, false, |renderable, _| {
            renderable.render(&mut renderpass);
        });

        // laid out in the same content area as the images
        if show_cursor {
            if let Some(buffer) = self.cursor_buffer.borrow_mut().get() {
                renderpass.set_pipeline(&self.invert_pipeline);
                renderpass.set_bind_group(0, &self.cursor_bind_group, &[]);
                renderpass.set_vertex_buffer(0, buffer.slice(..));
                renderpass.draw(0..6, 0..1);
            } else {
                tracing::warn!("No cursor buffer");
            }
        }

        // drawn last and opaque, so even a zoomed image can't show over it
        if let Some(bar_h) = bar
            && let Some(renderable) = &mut path_bar.renderable
        {
            let content_h = (wh - bar_h) as f32;
            renderpass.set_viewport(
                0.0,
                content_h,
                ww as f32,
                bar_h as f32,
                0.0,
                1.0,
            );
            renderpass.set_pipeline(&self.render_pipeline);
            renderable.render(&mut renderpass);
        }

        drop(renderpass);

        self.queue.submit([encoder.finish()]);
        self.window.pre_present_notify();
        self.queue.present(surface_texture);
    }
}
