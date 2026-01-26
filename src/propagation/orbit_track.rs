//! Orbit track generation

use crate::renderer::OrbitVertex;
use glam::Vec3;

use super::EARTH_RADIUS_KM;
use satkit::sgp4::sgp4;

/// Generate orbit track directly from TLE and time
pub fn generate_orbit_track_from_tle(
    tle: &satkit::TLE,
    center_time: &satkit::Instant,
    num_points: u32,
    color: [f32; 4],
) -> Vec<OrbitVertex> {
    let mut vertices = Vec::with_capacity((num_points as usize + 1) * 2);
    let mut positions: Vec<Vec3> = Vec::with_capacity(num_points as usize + 1);

    // Get current state to estimate orbital period
    let mut tle = tle.clone();
    let center_time = center_time.clone();
    let result = match sgp4(&mut tle, &[center_time]) {
        Ok(state) => state,
        Err(_) => return vertices,
    };
    let pos = result.pos.column(0);

    // Calculate orbital period
    let mu = 398600.4418_f64;
    let radius_km = (pos[0].powi(2) + pos[1].powi(2) + pos[2].powi(2)).sqrt() / 1000.0;
    let period_seconds = 2.0 * std::f64::consts::PI * (radius_km.powi(3) / mu).sqrt();

    // Propagate for one full orbit
    let start_time = center_time - satkit::Duration::from_seconds(period_seconds / 2.0);
    let step = period_seconds / num_points as f64;

    for i in 0..=num_points {
        let prop_time = start_time + satkit::Duration::from_seconds(step * i as f64);

        let mut tle = tle.clone();
        if let Ok(result) = sgp4(&mut tle, &[prop_time]) {
            // TEME uses Z-up (polar axis), but rendering uses Y-up
            // Convert: TEME X -> Render X, TEME Z -> Render Y, TEME Y -> Render Z
            let pos = result.pos.column(0);
            let pos_er = Vec3::new(
                pos[0] as f32 / 1000.0 / EARTH_RADIUS_KM as f32,
                pos[2] as f32 / 1000.0 / EARTH_RADIUS_KM as f32, // TEME Z -> Render Y
                -pos[1] as f32 / 1000.0 / EARTH_RADIUS_KM as f32, // TEME Y -> Render -Z
            );

            positions.push(pos_er);
        }
    }

    let thickness = 0.014_f32; // Earth radii
    let count = positions.len();

    for i in 0..count {
        let pos = positions[i];
        let prev = if i == 0 {
            positions[i]
        } else {
            positions[i - 1]
        };
        let next = if i + 1 >= count {
            positions[i]
        } else {
            positions[i + 1]
        };

        let mut tangent = next - prev;
        let tangent_len = tangent.length();
        if tangent_len > 1.0e-6 {
            tangent /= tangent_len;
        } else {
            tangent = Vec3::X;
        }

        let mut radial = pos;
        let radial_len = radial.length();
        if radial_len > 1.0e-6 {
            radial /= radial_len;
        } else {
            radial = Vec3::Y;
        }

        let mut plane_normal = tangent.cross(radial);
        let plane_len = plane_normal.length();
        if plane_len > 1.0e-6 {
            plane_normal /= plane_len;
        } else {
            plane_normal = radial.cross(Vec3::Y);
            let nlen = plane_normal.length();
            if nlen > 1.0e-6 {
                plane_normal /= nlen;
            } else {
                plane_normal = Vec3::Z;
            }
        }

        let mut offset_dir = plane_normal.cross(tangent);
        let offset_len = offset_dir.length();
        if offset_len > 1.0e-6 {
            offset_dir /= offset_len;
        } else {
            offset_dir = Vec3::X;
        }

        let offset = offset_dir * thickness;
        let left = pos + offset;
        let right = pos - offset;

        // Color gradient: past (dim) -> present (bright) -> future (dim)
        let t = i as f32 / (count - 1).max(1) as f32;
        let brightness = 0.6 + 0.4 * (1.0 - 2.0 * (t - 0.5).abs());
        let alpha = (color[3] * brightness).min(1.0);
        let final_color = [color[0], color[1], color[2], alpha];

        vertices.push(OrbitVertex {
            position: left.to_array(),
            color: final_color,
            side: -1.0,
        });
        vertices.push(OrbitVertex {
            position: right.to_array(),
            color: final_color,
            side: 1.0,
        });
    }

    vertices
}
