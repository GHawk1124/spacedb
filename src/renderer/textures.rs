//! Texture loading utilities

use anyhow::{Context, Result};
use image::{DynamicImage, GenericImageView};
use std::path::Path;

/// Load a texture from a file path
pub fn load_texture(path: impl AsRef<Path>) -> Result<TextureData> {
    let path = path.as_ref();
    log::info!("Loading texture: {:?}", path);

    let img = image::open(path).with_context(|| format!("Failed to load texture: {:?}", path))?;

    Ok(TextureData::from_image(img))
}

/// Raw texture data ready for GPU upload
pub struct TextureData {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    pub format: wgpu::TextureFormat,
}

impl TextureData {
    pub fn from_image(img: DynamicImage) -> Self {
        let (width, height) = img.dimensions();
        let rgba = img.to_rgba8();

        Self {
            width,
            height,
            data: rgba.into_raw(),
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
        }
    }

    /// Create GPU texture from this data
    pub fn create_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &str,
    ) -> wgpu::Texture {
        let size = wgpu::Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
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
            &self.data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * self.width),
                rows_per_image: Some(self.height),
            },
            size,
        );

        texture
    }
}

/// Load HDR environment map for skybox
pub fn load_hdr(path: impl AsRef<Path>) -> Result<HdrData> {
    use image::ImageDecoder;

    let path = path.as_ref();
    log::info!("Loading HDR: {:?}", path);

    let file =
        std::fs::File::open(path).with_context(|| format!("Failed to open HDR: {:?}", path))?;
    let reader = std::io::BufReader::new(file);

    let decoder =
        image::codecs::hdr::HdrDecoder::new(reader).with_context(|| "Failed to decode HDR")?;

    let (width, height) = decoder.dimensions();

    // Read into buffer and convert
    let mut buf = vec![0u8; decoder.total_bytes() as usize];
    decoder
        .read_image(&mut buf)
        .with_context(|| "Failed to read HDR pixels")?;

    // HDR decoder outputs Rgb32F, convert to RGBA f32
    // Each pixel is 3 f32 values (12 bytes)
    let pixels: &[f32] = bytemuck::cast_slice(&buf);
    let data: Vec<f32> = pixels
        .chunks(3)
        .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], 1.0])
        .collect();

    Ok(HdrData {
        width,
        height,
        data,
    })
}

/// HDR texture data
pub struct HdrData {
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>,
}

impl HdrData {
    /// Create GPU texture from HDR data
    pub fn create_texture(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        label: &str,
    ) -> wgpu::Texture {
        let size = wgpu::Extent3d {
            width: self.width,
            height: self.height,
            depth_or_array_layers: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
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
            bytemuck::cast_slice(&self.data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16 * self.width), // 4 floats * 4 bytes
                rows_per_image: Some(self.height),
            },
            size,
        );

        texture
    }
}

/// Create a sampler for textures
pub fn create_sampler(device: &wgpu::Device, label: &str) -> wgpu::Sampler {
    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(label),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    })
}
