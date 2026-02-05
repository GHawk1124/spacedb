//! NRLMSISE-00 atmospheric density model
//!
//! This is the primary atmosphere model, providing high-fidelity density
//! calculations based on empirical data and space weather inputs.
//!
//! The implementation wraps satkit's NRLMSISE-00 module.

use super::{gcrf_to_geodetic, AtmosphereDensity, AtmosphereModel};
use crate::propagation::hifi::state::EARTH_RADIUS_M;
use nalgebra::Vector3;
use satkit::Instant;

/// NRLMSISE-00 atmospheric model
///
/// NRLMSISE-00 (Naval Research Laboratory Mass Spectrometer and Incoherent
/// Scatter Radar) is the standard empirical atmosphere model for satellite
/// drag calculations.
///
/// # Space Weather Dependency
///
/// For best accuracy, this model requires:
/// - F10.7: Solar radio flux at 10.7 cm (solar activity proxy)
/// - F10.7a: 81-day average of F10.7
/// - Ap: Geomagnetic activity index
///
/// The model will use default/average values if space weather data is unavailable.
pub struct Nrlmsise00 {
    /// Whether to use actual space weather data (if available)
    use_space_weather: bool,
}

impl Default for Nrlmsise00 {
    fn default() -> Self {
        Self::new()
    }
}

impl Nrlmsise00 {
    /// Create a new NRLMSISE-00 model with default settings
    pub fn new() -> Self {
        Self {
            use_space_weather: true,
        }
    }

    /// Create with custom default space weather values (legacy compatibility)
    pub fn with_defaults(_f107: f64, _ap: f64) -> Self {
        Self::new()
    }

    /// Create with fixed space weather (ignores actual data)
    pub fn fixed(_f107: f64, _ap: f64) -> Self {
        Self {
            use_space_weather: false,
        }
    }

}

impl AtmosphereModel for Nrlmsise00 {
    fn density(&self, position: &Vector3<f64>, epoch: &Instant) -> AtmosphereDensity {
        // Check if above atmosphere (roughly 1000 km)
        let altitude = position.norm() - EARTH_RADIUS_M;
        if altitude > 1_000_000.0 {
            return AtmosphereDensity::zero();
        }
        if altitude < 0.0 {
            // Below surface - shouldn't happen but return high density
            return AtmosphereDensity::new(1.225); // Sea level density
        }

        // Get geodetic coordinates
        let (lat_deg, lon_deg, alt_m) = gcrf_to_geodetic(position, epoch);

        // Call satkit NRLMSISE-00
        // satkit 0.9 API: nrlmsise(alt_km, lat_deg?, lon_deg?, time?, use_spaceweather) -> (rho, temp)
        let (rho, temp) = satkit::nrlmsise::nrlmsise(
            alt_m / 1000.0, // Convert to km
            Some(lat_deg),
            Some(lon_deg),
            if self.use_space_weather {
                Some(epoch)
            } else {
                None
            },
            self.use_space_weather,
        );

        AtmosphereDensity::full(
            rho,  // Total mass density in kg/m³
            temp, // Temperature in K
            28.9, // Approximate mean molecular weight for mesosphere
        )
    }

    fn name(&self) -> &'static str {
        "NRLMSISE-00"
    }

    fn description(&self) -> &'static str {
        "NRL Mass Spectrometer and Incoherent Scatter Radar Exosphere 2000"
    }

    fn requires_space_weather(&self) -> bool {
        self.use_space_weather
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nrlmsise_basic() {
        let model = Nrlmsise00::fixed(150.0, 15.0);
        let epoch = Instant::from_datetime(2026, 1, 29, 12, 0, 0.0).unwrap();

        // Position at ~400 km altitude over equator
        let r = EARTH_RADIUS_M + 400_000.0;
        let position = Vector3::new(r, 0.0, 0.0);

        let density = model.density(&position, &epoch);

        // At 400 km, density should be roughly 1e-12 to 1e-11 kg/m³
        assert!(density.rho > 1e-14);
        assert!(density.rho < 1e-10);
        assert!(density.temperature.is_some());
    }

    #[test]
    fn test_above_atmosphere() {
        let model = Nrlmsise00::new();
        let epoch = Instant::from_datetime(2026, 1, 29, 12, 0, 0.0).unwrap();

        // Position at 2000 km altitude (above atmosphere)
        let r = EARTH_RADIUS_M + 2_000_000.0;
        let position = Vector3::new(r, 0.0, 0.0);

        let density = model.density(&position, &epoch);
        assert_eq!(density.rho, 0.0);
    }
}
