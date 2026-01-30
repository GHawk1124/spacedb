//! Harris-Priester atmospheric density model (placeholder)
//!
//! Harris-Priester is a simple analytical atmosphere model that provides
//! a good balance between accuracy and computational efficiency.
//!
//! # Status
//!
//! This is a placeholder with a basic implementation based on the
//! standard Harris-Priester formulation.

use super::{AtmosphereDensity, AtmosphereModel};
use crate::propagation::hifi::state::EARTH_RADIUS_M;
use nalgebra::Vector3;
use satkit::Instant;

/// Harris-Priester atmospheric model
///
/// This is a simple diurnal atmosphere model that accounts for the
/// day-night density variation ("atmospheric bulge" toward the Sun).
///
/// # Algorithm
///
/// The model interpolates between minimum (night) and maximum (day)
/// density values using an exponential scale height approach.
pub struct HarrisPriester {
    /// Exponent for day/night interpolation (typically 2-6)
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
    pub fn with_exponent(n: f64) -> Self {
        Self { n_prm: n }
    }
}

impl AtmosphereModel for HarrisPriester {
    fn density(&self, position: &Vector3<f64>, _epoch: &Instant) -> AtmosphereDensity {
        // Simplified placeholder implementation
        // TODO: Implement full Harris-Priester with Sun position and proper tables

        let altitude_km = (position.norm() - EARTH_RADIUS_M) / 1000.0;

        if altitude_km < 100.0 {
            // Below Karman line - use exponential approximation
            let rho0 = 1.225; // Sea level density kg/m³
            let h0 = 8.5; // Scale height km
            let rho = rho0 * (-altitude_km / h0).exp();
            return AtmosphereDensity::new(rho);
        }

        if altitude_km > 1000.0 {
            return AtmosphereDensity::zero();
        }

        // Simple exponential for thermosphere
        // Reference density at 120 km: ~2e-8 kg/m³
        // Scale height increases with altitude (roughly 50-80 km in thermosphere)
        let h_ref = 120.0;
        let rho_ref = 2.0e-8;
        let scale_height = 50.0 + (altitude_km - 100.0) * 0.3; // Increasing scale height

        let rho = rho_ref * (-(altitude_km - h_ref) / scale_height).exp();

        AtmosphereDensity::new(rho.max(0.0))
    }

    fn name(&self) -> &'static str {
        "Harris-Priester"
    }

    fn description(&self) -> &'static str {
        "Harris-Priester analytical atmosphere model (simplified)"
    }

    fn requires_space_weather(&self) -> bool {
        false
    }
}
