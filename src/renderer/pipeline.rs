//! Main rendering pipeline and resources

use std::sync::Arc;
use wgpu::util::DeviceExt;

use super::{Camera, CameraUniform, EarthMesh, EarthUniforms, OrbitTrack, SatelliteBuffer};

/// All GPU resources for rendering
pub struct RenderResources {
    // Camera
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub camera_bind_group_layout: wgpu::BindGroupLayout,

    // Earth
    pub earth_mesh: EarthMesh,
    pub earth_uniforms_buffer: wgpu::Buffer,
    pub earth_pipeline: wgpu::RenderPipeline,
    pub earth_bind_group: wgpu::BindGroup,

    // Textures
    pub earth_day_texture: wgpu::Texture,
    pub earth_night_texture: wgpu::Texture,
    pub earth_clouds_texture: wgpu::Texture,
    pub skybox_texture: wgpu::Texture,
    pub texture_sampler: wgpu::Sampler,

    // Skybox
    pub skybox_pipeline: wgpu::RenderPipeline,
    pub skybox_bind_group: wgpu::BindGroup,

    // Satellites
    pub satellite_buffer: SatelliteBuffer,
    pub satellite_pipeline: wgpu::RenderPipeline,

    // Orbit track
    pub orbit_track: OrbitTrack,
    pub orbit_pipeline: wgpu::RenderPipeline,

    // Depth buffer
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
}

/// Shader sources
pub mod shaders {
    pub const EARTH_SHADER: &str = r#"
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
    let day_factor = smoothstep(-0.2, 0.2, sun_dot);
    
    // Sample textures
    let day_color = textureSample(day_texture, tex_sampler, in.uv).rgb;
    let night_color = textureSample(night_texture, tex_sampler, in.uv).rgb * 0.5;
    let clouds = textureSample(clouds_texture, tex_sampler, in.uv).r;
    
    // Mix day and night
    var color = mix(night_color, day_color, day_factor);
    
    // Add clouds (brighter on day side)
    let cloud_color = vec3<f32>(1.0, 1.0, 1.0) * clouds;
    color = mix(color, cloud_color, clouds * day_factor * 0.6);
    
    // Simple atmospheric rim
    let view_dir = normalize(camera.camera_pos.xyz - in.world_pos);
    let rim = 1.0 - max(dot(view_dir, normal), 0.0);
    let atmosphere = vec3<f32>(0.3, 0.5, 1.0) * pow(rim, 3.0) * 0.5;
    color += atmosphere;
    
    return vec4<f32>(color, 1.0);
}
"#;

    pub const SKYBOX_SHADER: &str = r#"
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

// Fullscreen triangle
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0)
    );
    
    var out: VertexOutput;
    let pos = positions[vertex_index];
    out.clip_position = vec4<f32>(pos, 0.9999, 1.0);
    
    // Compute view direction
    let inv_proj = mat4x4<f32>(
        camera.proj[0],
        camera.proj[1],
        camera.proj[2],
        camera.proj[3]
    );
    // Simplified inverse - just get view ray
    let clip = vec4<f32>(pos, 1.0, 1.0);
    let view_dir = normalize(vec3<f32>(pos.x / camera.proj[0][0], pos.y / camera.proj[1][1], -1.0));
    
    // Transform to world space (rotation only from view matrix)
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
    
    // Equirectangular mapping
    let u = atan2(dir.z, dir.x) / (2.0 * 3.14159265359) + 0.5;
    let v = asin(clamp(dir.y, -1.0, 1.0)) / 3.14159265359 + 0.5;
    
    let color = textureSample(skybox_texture, skybox_sampler, vec2<f32>(u, 1.0 - v));
    
    // Tone mapping for HDR
    let mapped = color.rgb / (color.rgb + vec3<f32>(1.0));
    
    return vec4<f32>(mapped, 1.0);
}
"#;

    pub const SATELLITE_SHADER: &str = r#"
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
    @location(2) size: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @builtin(point_size) point_size: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    
    // Point size based on distance
    let dist = length(camera.camera_pos.xyz - in.position);
    out.point_size = clamp(in.size * 50.0 / dist, 2.0, 20.0);
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

    pub const ORBIT_SHADER: &str = r#"
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
}

impl RenderResources {
    pub fn create_depth_texture(
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
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        (texture, view)
    }
}
