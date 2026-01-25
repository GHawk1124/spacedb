//! Satellite rendering - instanced points and shapes

use bytemuck::{Pod, Zeroable};

/// Instance data for each satellite point
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SatelliteInstance {
    /// Position in world space (Earth radii)
    pub position: [f32; 3],
    /// Color based on altitude (RGBA)
    pub color: [f32; 4],
    /// Size multiplier
    pub size: f32,
}

impl SatelliteInstance {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SatelliteInstance>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // Position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Color
                wgpu::VertexAttribute {
                    offset: 12,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Size
                wgpu::VertexAttribute {
                    offset: 28,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

/// Get color for a satellite based on altitude (in km)
pub fn altitude_to_color(altitude_km: f64) -> [f32; 4] {
    // Color gradient:
    // LEO (< 2000 km): Blue to Cyan
    // MEO (2000-35000 km): Cyan to Green
    // GEO (~35786 km): Yellow
    // HEO (> 40000 km): Orange to Red

    let alt = altitude_km as f32;

    if alt < 500.0 {
        // Very low LEO: Deep blue
        [0.2, 0.4, 1.0, 1.0]
    } else if alt < 2000.0 {
        // LEO: Blue to cyan
        let t = (alt - 500.0) / 1500.0;
        [0.2, 0.4 + 0.6 * t, 1.0, 1.0]
    } else if alt < 20000.0 {
        // MEO lower: Cyan to green
        let t = (alt - 2000.0) / 18000.0;
        [0.2 * (1.0 - t), 1.0, 1.0 - t, 1.0]
    } else if alt < 35000.0 {
        // MEO upper: Green to yellow
        let t = (alt - 20000.0) / 15000.0;
        [t, 1.0, 0.0, 1.0]
    } else if alt < 40000.0 {
        // GEO region: Yellow
        [1.0, 1.0, 0.0, 1.0]
    } else {
        // HEO: Orange to red
        let t = ((alt - 40000.0) / 50000.0).min(1.0);
        [1.0, 1.0 - 0.5 * t, 0.0, 1.0]
    }
}

/// Orbit track line data
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct OrbitVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

impl OrbitVertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<OrbitVertex>() as wgpu::BufferAddress,
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
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}
