//! Exponential atmospheric density model
//!
//! The simplest atmosphere model, using an exponential density decay
//! with a single scale height. Fast but not very accurate.

use super::{AtmosphereDensity, AtmosphereModel};
use crate::propagation::hifi::state::EARTH_RADIUS_M;
use nalgebra::Vector3;
use satkit::Instant;

/// Exponential atmosphere model
///
/// Uses a simple exponential decay: ρ(h) = ρ₀ × exp(-h / H)
///
/// This is the fastest model but least accurate, as it doesn't account for:
/// - Temperature variations
/// - Day/night differences
/// - Solar activity effects
/// - Composition changes with altitude
#[derive(Debug, Clone)]
pub struct Exponential {
    /// Reference density at sea level (kg/m³)
    pub rho0: f64,

    /// Scale height (meters)
    pub scale_height: f64,

    /// Maximum altitude for non-zero density (meters)
    pub max_altitude: f64,
}

impl Default for Exponential {
    fn default() -> Self {
        Self::standard()
    }
}

impl Exponential {
    /// Standard Earth atmosphere parameters
    pub fn standard() -> Self {
        Self {
            rho0: 1.225,               // kg/m³ at sea level
            scale_height: 8500.0,      // ~8.5 km
            max_altitude: 1_000_000.0, // 1000 km
        }
    }

    /// Create with custom parameters
    pub fn new(rho0: f64, scale_height: f64, max_altitude: f64) -> Self {
        Self {
            rho0,
            scale_height,
            max_altitude,
        }
    }

    /// Create a model tuned for a specific altitude regime
    ///
    /// Uses different scale heights for different altitudes to better
    /// approximate real atmosphere behavior.
    pub fn altitude_tuned(reference_altitude_km: f64) -> Self {
        // Scale height increases with altitude
        let scale_height = if reference_altitude_km < 100.0 {
            8_500.0 // Troposphere/stratosphere
        } else if reference_altitude_km < 200.0 {
            27_000.0 // Lower thermosphere
        } else if reference_altitude_km < 400.0 {
            50_000.0 // Mid thermosphere
        } else {
            75_000.0 // Upper thermosphere
        };

        Self {
            rho0: 1.225,
            scale_height,
            max_altitude: 1_000_000.0,
        }
    }
}

impl AtmosphereModel for Exponential {
    fn density(&self, position: &Vector3<f64>, _epoch: &Instant) -> AtmosphereDensity {
        let altitude = position.norm() - EARTH_RADIUS_M;

        if altitude < 0.0 {
            // Below surface
            return AtmosphereDensity::new(self.rho0);
        }

        if altitude > self.max_altitude {
            return AtmosphereDensity::zero();
        }

        let rho = self.rho0 * (-altitude / self.scale_height).exp();

        AtmosphereDensity::new(rho)
    }

    fn name(&self) -> &'static str {
        "Exponential"
    }

    fn description(&self) -> &'static str {
        "Simple exponential density decay"
    }

    fn requires_space_weather(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_sea_level() {
        let model = Exponential::standard();
        let epoch = Instant::from_datetime(2026, 1, 29, 12, 0, 0.0).unwrap();
        let position = Vector3::new(EARTH_RADIUS_M, 0.0, 0.0);

        let density = model.density(&position, &epoch);
        assert!((density.rho - 1.225).abs() < 0.001);
    }

    #[test]
    fn test_exponential_one_scale_height() {
        let model = Exponential::standard();
        let epoch = Instant::from_datetime(2026, 1, 29, 12, 0, 0.0).unwrap();

        // At one scale height, density should be ~37% of surface (1/e)
        let position = Vector3::new(EARTH_RADIUS_M + 8500.0, 0.0, 0.0);
        let density = model.density(&position, &epoch);

        let expected = 1.225 * (-1.0_f64).exp();
        assert!((density.rho - expected).abs() < 0.001);
    }
}
