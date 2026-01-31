//! Harris-Priester atmospheric density model
//!
//! Harris-Priester is a simple analytical atmosphere model that provides
//! a good balance between accuracy and computational efficiency.
//!
//! Reference: Harris, I. and Priester, W., "Time-Dependent Structure of the
//! Upper Atmosphere", Journal of the Atmospheric Sciences, Vol. 19, 1962.

use super::{AtmosphereDensity, AtmosphereModel};
use crate::propagation::hifi::state::EARTH_RADIUS_M;
use nalgebra::Vector3;
use satkit::Instant;

/// Harris-Priester atmospheric model
///
/// This is a diurnal atmosphere model that accounts for the
/// day-night density variation (atmospheric bulge toward the Sun).
///
/// # Algorithm
///
/// The model interpolates between minimum (night) and maximum (day)
/// density values based on tabulated data from 100-1000 km altitude.
/// Uses the cosine of the half-angle between the satellite position
/// and the Sun-Earth line (apex of the diurnal bulge).
pub struct HarrisPriester {
    /// Exponent for day/night interpolation (typically 2-6)
    /// Higher values create sharper transition between day/night
    n_prm: f64,
}

impl Default for HarrisPriester {
    fn default() -> Self {
        Self::new()
    }
}

impl HarrisPriester {
    /// Create a new Harris-Priester model with default parameters
    pub fn new() -> Self {
        Self { n_prm: 2.0 }
    }

    /// Create with custom bulge exponent
    ///
    /// # Arguments
    /// * `n` - Exponent for day/night interpolation (typically 2-6)
    ///         n=2 for LEO, n=6 for higher orbits
    pub fn with_exponent(n: f64) -> Self {
        Self { n_prm: n }
    }

    /// Harris-Priester density table
    /// Format: (altitude_km, rho_min_kg_m3, rho_max_kg_m3)
    ///
    /// Source: Vallado, "Fundamentals of Astrodynamics and Applications"
    /// Updated coefficients for moderate solar activity (F10.7 ~ 150)
    const DENSITY_TABLE: [(f64, f64, f64); 50] = [
        // Alt(km), Min density, Max density (kg/m³)
        (100.0, 4.974e-07, 4.974e-07),
        (120.0, 2.490e-08, 2.490e-08),
        (130.0, 8.377e-09, 8.710e-09),
        (140.0, 3.899e-09, 4.059e-09),
        (150.0, 2.122e-09, 2.215e-09),
        (160.0, 1.263e-09, 1.344e-09),
        (170.0, 8.008e-10, 8.758e-10),
        (180.0, 5.283e-10, 6.010e-10),
        (190.0, 3.617e-10, 4.297e-10),
        (200.0, 2.557e-10, 3.162e-10),
        (210.0, 1.839e-10, 2.396e-10),
        (220.0, 1.341e-10, 1.853e-10),
        (230.0, 9.949e-11, 1.455e-10),
        (240.0, 7.488e-11, 1.157e-10),
        (250.0, 5.709e-11, 9.308e-11),
        (260.0, 4.403e-11, 7.555e-11),
        (270.0, 3.430e-11, 6.182e-11),
        (280.0, 2.697e-11, 5.095e-11),
        (290.0, 2.139e-11, 4.226e-11),
        (300.0, 1.708e-11, 3.526e-11),
        (320.0, 1.099e-11, 2.511e-11),
        (340.0, 7.214e-12, 1.819e-11),
        (360.0, 4.824e-12, 1.337e-11),
        (380.0, 3.274e-12, 9.955e-12),
        (400.0, 2.249e-12, 7.492e-12),
        (420.0, 1.558e-12, 5.684e-12),
        (440.0, 1.091e-12, 4.355e-12),
        (460.0, 7.701e-13, 3.362e-12),
        (480.0, 5.474e-13, 2.612e-12),
        (500.0, 3.916e-13, 2.042e-12),
        (520.0, 2.819e-13, 1.605e-12),
        (540.0, 2.042e-13, 1.267e-12),
        (560.0, 1.488e-13, 1.005e-12),
        (580.0, 1.092e-13, 7.997e-13),
        (600.0, 8.070e-14, 6.390e-13),
        (620.0, 6.012e-14, 5.123e-13),
        (640.0, 4.519e-14, 4.121e-13),
        (660.0, 3.430e-14, 3.325e-13),
        (680.0, 2.632e-14, 2.691e-13),
        (700.0, 2.043e-14, 2.185e-13),
        (720.0, 1.607e-14, 1.779e-13),
        (740.0, 1.281e-14, 1.452e-13),
        (760.0, 1.036e-14, 1.190e-13),
        (780.0, 8.496e-15, 9.776e-14),
        (800.0, 7.069e-15, 8.059e-14),
        (850.0, 4.680e-15, 5.500e-14),
        (900.0, 3.200e-15, 3.800e-14),
        (950.0, 2.210e-15, 2.640e-14),
        (1000.0, 1.560e-15, 1.870e-14),
        (1100.0, 8.000e-16, 9.600e-15),
    ];

    /// Interpolate density from the table
    fn interpolate_density(&self, altitude_km: f64) -> (f64, f64) {
        let table = &Self::DENSITY_TABLE;

        // Clamp to table bounds
        if altitude_km <= table[0].0 {
            return (table[0].1, table[0].2);
        }
        if altitude_km >= table[table.len() - 1].0 {
            return (table[table.len() - 1].1, table[table.len() - 1].2);
        }

        // Find bracketing entries
        let mut i = 0;
        while i < table.len() - 1 && table[i + 1].0 < altitude_km {
            i += 1;
        }

        let (h0, rho_min0, rho_max0) = table[i];
        let (h1, rho_min1, rho_max1) = table[i + 1];

        // Logarithmic interpolation (density varies exponentially with altitude)
        let t = (altitude_km - h0) / (h1 - h0);

        let ln_rho_min = (1.0 - t) * rho_min0.ln() + t * rho_min1.ln();
        let ln_rho_max = (1.0 - t) * rho_max0.ln() + t * rho_max1.ln();

        (ln_rho_min.exp(), ln_rho_max.exp())
    }
}

impl AtmosphereModel for HarrisPriester {
    fn density(&self, position: &Vector3<f64>, epoch: &Instant) -> AtmosphereDensity {
        let altitude_km = (position.norm() - EARTH_RADIUS_M) / 1000.0;

        if altitude_km < 100.0 {
            // Below Karman line - use exponential approximation
            let rho0 = 1.225; // Sea level density kg/m³
            let h0 = 8.5; // Scale height km
            let rho = rho0 * (-altitude_km / h0).exp();
            return AtmosphereDensity::new(rho);
        }

        if altitude_km > 1100.0 {
            return AtmosphereDensity::zero();
        }

        // Get min/max density from table
        let (rho_min, rho_max) = self.interpolate_density(altitude_km);

        // Compute Sun position to determine day/night factor
        // For simplicity, we use the satellite's position relative to an approximate Sun direction
        // Full implementation would compute actual Sun position from epoch

        // Approximate Sun position in GCRF (simplified - assumes Sun roughly along +X at vernal equinox)
        // A full implementation would use satkit's Sun ephemeris
        let sun_dir = Self::approximate_sun_direction(epoch);

        // Compute the half-angle cosine between satellite and apex (Sun direction)
        // The apex is 30° ahead of the Sun in local time
        let apex_lag_rad = 30.0_f64.to_radians();
        let apex_dir = rotate_z(&sun_dir, apex_lag_rad);

        // Satellite direction from Earth center
        let sat_dir = position.normalize();

        // Cosine of angle between satellite and apex
        let cos_psi = sat_dir.dot(&apex_dir).max(-1.0).min(1.0);

        // Day/night interpolation factor
        // psi_half = half-angle, ranges from 0 (at apex) to π (at antipode)
        let psi_half = ((1.0 + cos_psi) / 2.0).sqrt().max(0.0).min(1.0);

        // Interpolate using the n_prm exponent
        // rho = rho_min + (rho_max - rho_min) * cos^n(psi/2)
        let factor = psi_half.powf(self.n_prm);
        let rho = rho_min + (rho_max - rho_min) * factor;

        AtmosphereDensity::new(rho)
    }

    fn name(&self) -> &'static str {
        "Harris-Priester"
    }

    fn description(&self) -> &'static str {
        "Harris-Priester diurnal atmosphere model"
    }

    fn requires_space_weather(&self) -> bool {
        false
    }
}

impl HarrisPriester {
    /// Approximate Sun direction in GCRF coordinates
    ///
    /// This is a simplified calculation. For high accuracy, use
    /// satkit's Sun ephemeris functions.
    fn approximate_sun_direction(epoch: &Instant) -> Vector3<f64> {
        // Days since J2000.0
        let jd = epoch.as_jd();
        let t = jd - 2451545.0;

        // Mean longitude of Sun (degrees)
        let l0 = 280.46 + 0.9856474 * t;

        // Mean anomaly (degrees)
        let m = 357.528 + 0.9856003 * t;
        let m_rad = m.to_radians();

        // Ecliptic longitude (degrees)
        let lambda = l0 + 1.915 * m_rad.sin() + 0.020 * (2.0 * m_rad).sin();
        let lambda_rad = lambda.to_radians();

        // Obliquity of ecliptic (degrees)
        let epsilon_rad = 23.439_f64.to_radians();

        // Unit vector to Sun in GCRF (equatorial coordinates)
        let x = lambda_rad.cos();
        let y = lambda_rad.sin() * epsilon_rad.cos();
        let z = lambda_rad.sin() * epsilon_rad.sin();

        Vector3::new(x, y, z).normalize()
    }
}

/// Rotate a vector around the Z-axis
fn rotate_z(v: &Vector3<f64>, angle_rad: f64) -> Vector3<f64> {
    let c = angle_rad.cos();
    let s = angle_rad.sin();
    Vector3::new(c * v.x - s * v.y, s * v.x + c * v.y, v.z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harris_priester_basic() {
        let model = HarrisPriester::new();
        let epoch = Instant::from_datetime(2026, 1, 29, 12, 0, 0.0).unwrap();

        // Position at ~400 km altitude over equator
        let r = EARTH_RADIUS_M + 400_000.0;
        let position = Vector3::new(r, 0.0, 0.0);

        let density = model.density(&position, &epoch);

        // At 400 km, density should be roughly 2e-12 to 8e-12 kg/m³
        println!("Density at 400 km: {:.3e} kg/m³", density.rho);
        assert!(density.rho > 1e-13, "Density too low: {}", density.rho);
        assert!(density.rho < 1e-10, "Density too high: {}", density.rho);
    }

    #[test]
    fn test_density_table_interpolation() {
        let model = HarrisPriester::new();

        // Test at table points
        let (rho_min, rho_max) = model.interpolate_density(400.0);
        println!("At 400 km: min={:.3e}, max={:.3e}", rho_min, rho_max);
        assert!((rho_min - 2.249e-12).abs() < 1e-14);
        assert!((rho_max - 7.492e-12).abs() < 1e-14);

        // Test interpolation between points
        let (rho_min, rho_max) = model.interpolate_density(410.0);
        println!("At 410 km: min={:.3e}, max={:.3e}", rho_min, rho_max);
        assert!(rho_min > 1.5e-12 && rho_min < 2.3e-12);
        assert!(rho_max > 5.5e-12 && rho_max < 7.5e-12);
    }

    #[test]
    fn test_above_atmosphere() {
        let model = HarrisPriester::new();
        let epoch = Instant::from_datetime(2026, 1, 29, 12, 0, 0.0).unwrap();

        // Position at 2000 km altitude (above atmosphere)
        let r = EARTH_RADIUS_M + 2_000_000.0;
        let position = Vector3::new(r, 0.0, 0.0);

        let density = model.density(&position, &epoch);
        assert_eq!(density.rho, 0.0);
    }

    #[test]
    fn test_density_decreases_with_altitude() {
        let model = HarrisPriester::new();
        let epoch = Instant::from_datetime(2026, 1, 29, 12, 0, 0.0).unwrap();

        let mut last_density = f64::MAX;
        for alt_km in [200, 300, 400, 500, 600, 700, 800] {
            let r = EARTH_RADIUS_M + alt_km as f64 * 1000.0;
            let position = Vector3::new(r, 0.0, 0.0);
            let density = model.density(&position, &epoch);

            println!("Alt {} km: {:.3e} kg/m³", alt_km, density.rho);
            assert!(
                density.rho < last_density,
                "Density should decrease with altitude: {} at {} km vs {} at lower alt",
                density.rho,
                alt_km,
                last_density
            );
            last_density = density.rho;
        }
    }
}
