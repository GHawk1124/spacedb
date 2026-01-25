//! Earth rendering - sphere mesh with textures

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// Vertex for Earth mesh
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct EarthVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl EarthVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<EarthVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        }
    }
}

/// Generate a UV sphere mesh for Earth
/// Returns (vertices, indices)
pub fn generate_earth_sphere(segments: u32, rings: u32) -> (Vec<EarthVertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    // Generate vertices
    for ring in 0..=rings {
        let phi = std::f32::consts::PI * ring as f32 / rings as f32;
        let y = phi.cos();
        let ring_radius = phi.sin();

        for seg in 0..=segments {
            let theta = 2.0 * std::f32::consts::PI * seg as f32 / segments as f32;
            let x = ring_radius * theta.cos();
            let z = ring_radius * theta.sin();

            let position = Vec3::new(x, y, z);
            let normal = position.normalize();

            // UV mapping (equirectangular)
            let u = seg as f32 / segments as f32;
            let v = ring as f32 / rings as f32;

            vertices.push(EarthVertex {
                position: position.to_array(),
                normal: normal.to_array(),
                uv: [u, v],
            });
        }
    }

    // Generate indices
    for ring in 0..rings {
        for seg in 0..segments {
            let current = ring * (segments + 1) + seg;
            let next = current + segments + 1;

            // First triangle
            indices.push(current);
            indices.push(next);
            indices.push(current + 1);

            // Second triangle
            indices.push(current + 1);
            indices.push(next);
            indices.push(next + 1);
        }
    }

    (vertices, indices)
}
