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
    let mut vertices = Vec::with_capacity(num_points as usize + 1);

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
    let radius_km = (pos[0].powi(2) + pos[1].powi(2) + pos[2].powi(2)).sqrt();
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
                pos[2] as f32 / 1000.0 / EARTH_RADIUS_KM as f32, // TEME Z (polar) -> Render Y (up)
                pos[1] as f32 / 1000.0 / EARTH_RADIUS_KM as f32, // TEME Y -> Render Z
            );

            // Color gradient: past (dim) -> present (bright) -> future (dim)
            let t = i as f32 / num_points as f32;
            let brightness = 1.0 - 2.0 * (t - 0.5).abs();
            let alpha = 0.3 + 0.7 * brightness;

            vertices.push(OrbitVertex {
                position: pos_er.to_array(),
                color: [color[0], color[1], color[2], color[3] * alpha],
            });
        }
    }

    vertices
}
