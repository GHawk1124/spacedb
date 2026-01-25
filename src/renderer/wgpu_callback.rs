//! egui_wgpu integration for 3D scene rendering
//!
//! Uses offscreen rendering with depth buffer, then blits to egui's render pass.

use anyhow::Result;
use glam::{Mat4, Vec3};
use parking_lot::RwLock;
use std::path::Path;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use super::earth::{generate_earth_sphere, EarthVertex};
use super::{Camera, CameraUniform, OrbitVertex, SatelliteInstance};

/// Per-frame data passed to the callback
#[derive(Clone)]
pub struct SceneRenderData {
    pub camera: Camera,
    pub aspect_ratio: f32,
    pub sun_direction: Vec3,
    pub time: f32,
    pub satellites: Arc<Vec<SatelliteInstance>>,
    pub orbit_track: Arc<Vec<OrbitVertex>>,
}

impl Default for SceneRenderData {
    fn default() -> Self {
        Self {
            camera: Camera::default(),
            aspect_ratio: 16.0 / 9.0,
            sun_direction: Vec3::new(1.0, 0.3, 0.5).normalize(),
            time: 0.0,
            satellites: Arc::new(Vec::new()),
            orbit_track: Arc::new(Vec::new()),
        }
    }
}

/// GPU resources for 3D scene rendering, stored in callback_resources
pub struct SceneRenderResources {
    // Offscreen render target
    offscreen_texture: wgpu::Texture,
    offscreen_view: wgpu::TextureView,
    offscreen_size: (u32, u32),
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    target_format: wgpu::TextureFormat,

    // Camera
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,

    // Earth
    earth_vertex_buffer: wgpu::Buffer,
    earth_index_buffer: wgpu::Buffer,
    earth_index_count: u32,
    earth_pipeline: wgpu::RenderPipeline,
    earth_bind_group: wgpu::BindGroup,
    earth_uniform_buffer: wgpu::Buffer,

    // Skybox
    skybox_pipeline: wgpu::RenderPipeline,
    skybox_bind_group: wgpu::BindGroup,

    // Satellites (instanced)
    satellite_pipeline: wgpu::RenderPipeline,
    satellite_buffer: wgpu::Buffer,
    satellite_capacity: u32,

    // Orbit track
    orbit_pipeline: wgpu::RenderPipeline,
    orbit_buffer: wgpu::Buffer,
    orbit_capacity: u32,

    // Blit pipeline (for drawing offscreen texture to egui)
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group_layout: wgpu::BindGroupLayout,
    blit_bind_group: wgpu::BindGroup,
    blit_sampler: wgpu::Sampler,

    // Shared render data (updated each frame)
    render_data: RwLock<SceneRenderData>,
}

/// Uniforms for Earth rendering
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct EarthUniforms {
    model: [[f32; 4]; 4],
    sun_direction: [f32; 4],
    time: f32,
    _padding: [f32; 3],
}

impl SceneRenderResources {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        assets_path: &Path,
    ) -> Result<Self> {
        log::info!("Initializing SceneRenderResources ({}x{})", width, height);

        // Create offscreen render target
        let (offscreen_texture, offscreen_view) =
            Self::create_offscreen_texture(device, width, height, target_format);
        let (depth_texture, depth_view) = Self::create_depth_texture(device, width, height);

        // Create camera uniform buffer
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Camera bind group layout
        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Camera Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera Bind Group"),
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // Load textures
        let (earth_day, earth_night, earth_clouds, skybox) =
            Self::load_textures(device, queue, assets_path)?;

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Texture Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Earth mesh
        let (earth_vertices, earth_indices) = generate_earth_sphere(64, 32);

        let earth_vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Earth Vertex Buffer"),
            contents: bytemuck::cast_slice(&earth_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let earth_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Earth Index Buffer"),
            contents: bytemuck::cast_slice(&earth_indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        // Earth uniforms
        let earth_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Earth Uniform Buffer"),
            size: std::mem::size_of::<EarthUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Earth bind group layout
        let earth_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Earth Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let earth_day_view = earth_day.create_view(&wgpu::TextureViewDescriptor::default());
        let earth_night_view = earth_night.create_view(&wgpu::TextureViewDescriptor::default());
        let earth_clouds_view = earth_clouds.create_view(&wgpu::TextureViewDescriptor::default());
        let skybox_view = skybox.create_view(&wgpu::TextureViewDescriptor::default());

        let earth_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Earth Bind Group"),
            layout: &earth_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: earth_uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&earth_day_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&earth_night_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&earth_clouds_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Earth pipeline
        let earth_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Earth Shader"),
            source: wgpu::ShaderSource::Wgsl(EARTH_SHADER.into()),
        });

        let earth_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Earth Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &earth_bind_group_layout],
                push_constant_ranges: &[],
            });

        let earth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Earth Pipeline"),
            layout: Some(&earth_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &earth_shader,
                entry_point: Some("vs_main"),
                buffers: &[EarthVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &earth_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Skybox pipeline
        let skybox_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Skybox Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
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

        let skybox_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Skybox Bind Group"),
            layout: &skybox_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&skybox_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let skybox_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Skybox Shader"),
            source: wgpu::ShaderSource::Wgsl(SKYBOX_SHADER.into()),
        });

        let skybox_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Skybox Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &skybox_bind_group_layout],
                push_constant_ranges: &[],
            });

        let skybox_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Skybox Pipeline"),
            layout: Some(&skybox_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &skybox_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &skybox_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Satellite pipeline (instanced points)
        let satellite_capacity = 50000u32;
        let satellite_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Satellite Instance Buffer"),
            size: (satellite_capacity as usize * std::mem::size_of::<SatelliteInstance>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let satellite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Satellite Shader"),
            source: wgpu::ShaderSource::Wgsl(SATELLITE_SHADER.into()),
        });

        let satellite_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Satellite Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let satellite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Satellite Pipeline"),
            layout: Some(&satellite_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &satellite_shader,
                entry_point: Some("vs_main"),
                buffers: &[SatelliteInstance::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &satellite_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Orbit track pipeline (line strip)
        let orbit_capacity = 1024u32;
        let orbit_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Orbit Track Buffer"),
            size: (orbit_capacity as usize * std::mem::size_of::<OrbitVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let orbit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Orbit Shader"),
            source: wgpu::ShaderSource::Wgsl(ORBIT_SHADER.into()),
        });

        let orbit_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Orbit Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let orbit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Orbit Pipeline"),
            layout: Some(&orbit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &orbit_shader,
                entry_point: Some("vs_main"),
                buffers: &[OrbitVertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &orbit_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineStrip,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: false,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Blit pipeline (to draw offscreen texture to egui's render pass)
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blit Shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });

        let blit_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Blit Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
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

        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Blit Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blit Bind Group"),
            layout: &blit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&offscreen_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&blit_sampler),
                },
            ],
        });

        let blit_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blit Pipeline Layout"),
            bind_group_layouts: &[&blit_bind_group_layout],
            push_constant_ranges: &[],
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit Pipeline"),
            layout: Some(&blit_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None, // No depth for blit
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Ok(Self {
            offscreen_texture,
            offscreen_view,
            offscreen_size: (width, height),
            depth_texture,
            depth_view,
            target_format,
            camera_buffer,
            camera_bind_group,
            earth_vertex_buffer,
            earth_index_buffer,
            earth_index_count: earth_indices.len() as u32,
            earth_pipeline,
            earth_bind_group,
            earth_uniform_buffer,
            skybox_pipeline,
            skybox_bind_group,
            satellite_pipeline,
            satellite_buffer,
            satellite_capacity,
            orbit_pipeline,
            orbit_buffer,
            orbit_capacity,
            blit_pipeline,
            blit_bind_group_layout,
            blit_bind_group,
            blit_sampler,
            render_data: RwLock::new(SceneRenderData::default()),
        })
    }

    fn create_offscreen_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Offscreen Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn create_depth_texture(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn load_textures(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        assets_path: &Path,
    ) -> Result<(wgpu::Texture, wgpu::Texture, wgpu::Texture, wgpu::Texture)> {
        let load_texture = |name: &str| -> Result<wgpu::Texture> {
            let path = assets_path.join(name);
            log::info!("Loading texture: {:?}", path);

            let img = image::open(&path)
                .map_err(|e| anyhow::anyhow!("Failed to load texture {:?}: {}", path, e))?;
            let rgba = img.to_rgba8();
            let (width, height) = rgba.dimensions();

            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(name),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * width),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );

            Ok(texture)
        };

        let load_hdri_texture = |name: &str| -> Result<wgpu::Texture> {
            let path = assets_path.join(name);
            log::info!("Loading HDRI texture: {:?}", path);

            let img = image::open(&path)
                .map_err(|e| anyhow::anyhow!("Failed to load HDRI {:?}: {}", path, e))?;
            let rgb = img.to_rgb32f();
            let (width, height) = rgb.dimensions();
            let data = rgb.into_raw();

            // Tone-map HDR to LDR (Reinhard + gamma) for display
            let mut rgba = Vec::with_capacity((width * height * 4) as usize);
            for chunk in data.chunks_exact(3) {
                let mapped_r = chunk[0] / (1.0 + chunk[0]);
                let mapped_g = chunk[1] / (1.0 + chunk[1]);
                let mapped_b = chunk[2] / (1.0 + chunk[2]);
                let gamma = |c: f32| c.clamp(0.0, 1.0).powf(1.0 / 2.2);
                rgba.push((gamma(mapped_r) * 255.0) as u8);
                rgba.push((gamma(mapped_g) * 255.0) as u8);
                rgba.push((gamma(mapped_b) * 255.0) as u8);
                rgba.push(255);
            }

            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(name),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * width),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );

            Ok(texture)
        };

        let earth_day = load_texture("2k_earth_daymap.jpg")?;
        let earth_night = load_texture("2k_earth_nightmap.jpg")?;
        let earth_clouds = load_texture("2k_earth_clouds.jpg")?;

        // Load HDRI skybox (4K)
        let skybox = load_hdri_texture("Starfield_Free/StudioHDR_2_StarField_01_4K.hdr")
            .or_else(|_| load_hdri_texture("Starfield_Free/StudioHDR_2_StarField_01_4K.exr"))
            .unwrap_or_else(|_| {
                // Create a simple dark texture as fallback
                let size = 64u32;
                let data: Vec<u8> = (0..size * size * 4)
                    .map(|i| if i % 4 == 3 { 255 } else { 5 })
                    .collect();

                let texture = device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Skybox Fallback"),
                    size: wgpu::Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8UnormSrgb,

                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                });

                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * size),
                        rows_per_image: Some(size),
                    },
                    wgpu::Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: 1,
                    },
                );

                texture
            });

        Ok((earth_day, earth_night, earth_clouds, skybox))
    }

    /// Update render data (called from app each frame)
    pub fn set_render_data(&self, data: SceneRenderData) {
        *self.render_data.write() = data;
    }

    /// Resize offscreen buffers if needed
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.offscreen_size != (width, height) && width > 0 && height > 0 {
            let (offscreen_texture, offscreen_view) =
                Self::create_offscreen_texture(device, width, height, self.target_format);
            let (depth_texture, depth_view) = Self::create_depth_texture(device, width, height);

            // Recreate blit bind group with new texture view
            self.blit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Blit Bind Group"),
                layout: &self.blit_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&offscreen_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&self.blit_sampler),
                    },
                ],
            });

            self.offscreen_texture = offscreen_texture;
            self.offscreen_view = offscreen_view;
            self.depth_texture = depth_texture;
            self.depth_view = depth_view;
            self.offscreen_size = (width, height);
        }
    }

    /// Render the 3D scene to offscreen buffer
    pub fn render_offscreen(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let data = self.render_data.read();

        // Update camera uniform
        let camera_uniform = CameraUniform::from_camera(&data.camera, data.aspect_ratio);
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&camera_uniform));

        // Update earth uniforms
        let earth_uniforms = EarthUniforms {
            model: Mat4::IDENTITY.to_cols_array_2d(),
            sun_direction: [
                data.sun_direction.x,
                data.sun_direction.y,
                data.sun_direction.z,
                0.0,
            ],
            time: data.time,
            _padding: [0.0; 3],
        };
        queue.write_buffer(
            &self.earth_uniform_buffer,
            0,
            bytemuck::bytes_of(&earth_uniforms),
        );

        // Update satellites
        let satellite_count = data.satellites.len().min(self.satellite_capacity as usize);
        if satellite_count > 0 {
            queue.write_buffer(
                &self.satellite_buffer,
                0,
                bytemuck::cast_slice(&data.satellites[..satellite_count]),
            );
        }

        // Update orbit track
        let orbit_count = data.orbit_track.len().min(self.orbit_capacity as usize);
        if orbit_count > 0 {
            queue.write_buffer(
                &self.orbit_buffer,
                0,
                bytemuck::cast_slice(&data.orbit_track[..orbit_count]),
            );
        }

        // Begin render pass to offscreen texture
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Scene Offscreen Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.offscreen_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.05,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            // Draw skybox first (at infinity)
            render_pass.set_pipeline(&self.skybox_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.skybox_bind_group, &[]);
            render_pass.draw(0..3, 0..1);

            // Draw Earth
            render_pass.set_pipeline(&self.earth_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            render_pass.set_bind_group(1, &self.earth_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.earth_vertex_buffer.slice(..));
            render_pass
                .set_index_buffer(self.earth_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..self.earth_index_count, 0, 0..1);

            // Draw orbit track (if any)
            if orbit_count > 1 {
                render_pass.set_pipeline(&self.orbit_pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.orbit_buffer.slice(..));
                render_pass.draw(0..orbit_count as u32, 0..1);
            }

            // Draw satellites (instanced)
            if satellite_count > 0 {
                render_pass.set_pipeline(&self.satellite_pipeline);
                render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
                render_pass.set_vertex_buffer(0, self.satellite_buffer.slice(..));
                // Draw 6 vertices (2 triangles for a quad) per instance
                render_pass.draw(0..6, 0..satellite_count as u32);
            }
        }
    }

    /// Blit the offscreen texture to egui's render pass
    pub fn blit(&self, render_pass: &mut wgpu::RenderPass<'static>) {
        render_pass.set_pipeline(&self.blit_pipeline);
        render_pass.set_bind_group(0, &self.blit_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

/// The callback that egui_wgpu will invoke
pub struct SceneCallback {
    pub viewport_size: (u32, u32),
}

impl egui_wgpu::CallbackTrait for SceneCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        egui_encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        if let Some(resources) = callback_resources.get_mut::<SceneRenderResources>() {
            // Resize if needed
            resources.resize(device, self.viewport_size.0, self.viewport_size.1);

            // Render to offscreen buffer
            resources.render_offscreen(device, queue, egui_encoder);
        }
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        if let Some(resources) = callback_resources.get::<SceneRenderResources>() {
            resources.blit(render_pass);
        }
    }
}

// Shader sources
const EARTH_SHADER: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};

struct EarthUniforms {
    model: mat4x4<f32>,
    sun_direction: vec4<f32>,
    time: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> earth: EarthUniforms;
@group(1) @binding(1) var day_texture: texture_2d<f32>;
@group(1) @binding(2) var night_texture: texture_2d<f32>;
@group(1) @binding(3) var clouds_texture: texture_2d<f32>;
@group(1) @binding(4) var tex_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = (earth.model * vec4<f32>(in.position, 1.0)).xyz;
    out.clip_position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.normal = normalize((earth.model * vec4<f32>(in.normal, 0.0)).xyz);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let sun_dir = normalize(earth.sun_direction.xyz);
    let normal = normalize(in.normal);
    
    // Day/night blending based on sun angle
    let sun_dot = dot(normal, sun_dir);
    let day_factor = smoothstep(-0.1, 0.3, sun_dot);
    
    // Sample textures
    let day_color = textureSample(day_texture, tex_sampler, in.uv).rgb;
    let night_color = textureSample(night_texture, tex_sampler, in.uv).rgb * 2.0;
    let clouds = textureSample(clouds_texture, tex_sampler, in.uv).r;
    
    // Mix day and night
    var color = mix(night_color, day_color, day_factor);
    
    // Add clouds (brighter on day side)
    let cloud_brightness = mix(0.3, 1.0, day_factor);
    color = mix(color, vec3<f32>(cloud_brightness), clouds * 0.5 * day_factor);
    
    // Atmospheric rim lighting
    let view_dir = normalize(camera.camera_pos.xyz - in.world_pos);
    let rim = 1.0 - max(dot(view_dir, normal), 0.0);
    let atmosphere = vec3<f32>(0.3, 0.5, 1.0) * pow(rim, 4.0) * 0.6;
    color += atmosphere;
    
    return vec4<f32>(color, 1.0);
}
"#;

const SKYBOX_SHADER: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var skybox_texture: texture_2d<f32>;
@group(1) @binding(1) var skybox_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) direction: vec3<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Fullscreen triangle
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    
    var out: VertexOutput;
    let pos = positions[vertex_index];
    out.clip_position = vec4<f32>(pos, 0.9999, 1.0);
    
    // Compute view direction from clip space
    let inv_proj_00 = 1.0 / camera.proj[0][0];
    let inv_proj_11 = 1.0 / camera.proj[1][1];
    let view_dir = normalize(vec3<f32>(pos.x * inv_proj_00, pos.y * inv_proj_11, -1.0));
    
    // Transform to world space (rotation only)
    let inv_view_rot = transpose(mat3x3<f32>(
        camera.view[0].xyz,
        camera.view[1].xyz,
        camera.view[2].xyz
    ));
    out.direction = inv_view_rot * view_dir;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dir = normalize(in.direction);
    let u = atan2(dir.z, dir.x) / (2.0 * 3.14159265) + 0.5;
    let v = asin(clamp(dir.y, -1.0, 1.0)) / 3.14159265 + 0.5;
    let color = textureSample(skybox_texture, skybox_sampler, vec2<f32>(u, v)).rgb;
    return vec4<f32>(color, 1.0);
}
"#;

const SATELLITE_SHADER: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct InstanceInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
    @location(2) size: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    instance: InstanceInput,
) -> VertexOutput {
    // Billboard quad vertices
    var offsets = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0)
    );
    
    let offset = offsets[vertex_index];
    
    // Calculate billboard size based on distance
    let dist = length(camera.camera_pos.xyz - instance.position);
    let base_size = instance.size * 0.008;
    let screen_size = clamp(base_size / dist, 0.0008, 0.02);
    
    // Get camera right and up vectors from view matrix
    let right = vec3<f32>(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
    let up = vec3<f32>(camera.view[0][1], camera.view[1][1], camera.view[2][1]);
    
    // Billboard position
    let billboard_pos = instance.position + 
        right * offset.x * screen_size * dist +
        up * offset.y * screen_size * dist;
    
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(billboard_pos, 1.0);
    out.color = instance.color;
    out.uv = offset * 0.5 + 0.5;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Circular point with soft edge
    let dist = length(in.uv - vec2<f32>(0.5));
    let alpha = 1.0 - smoothstep(0.35, 0.5, dist);
    
    if (alpha < 0.01) {
        discard;
    }
    
    return vec4<f32>(in.color.rgb, in.color.a * alpha);
}
"#;

const ORBIT_SHADER: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

const BLIT_SHADER: &str = r#"
@group(0) @binding(0) var blit_texture: texture_2d<f32>;
@group(0) @binding(1) var blit_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Fullscreen triangle
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    
    var out: VertexOutput;
    let pos = positions[vertex_index];
    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = pos * 0.5 + 0.5;
    out.uv.y = 1.0 - out.uv.y; // Flip Y for correct orientation
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(blit_texture, blit_sampler, in.uv);
}
"#;
