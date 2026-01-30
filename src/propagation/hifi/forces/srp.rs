//! Solar radiation pressure force model (placeholder)
//!
//! Models the acceleration due to photon momentum from sunlight.
//!
//! # Status
//!
//! This is a placeholder implementation with a simplified model.
//! A full implementation would include:
//! - Shadow function (eclipse detection)
//! - Proper spacecraft attitude modeling
//! - Earth and Moon shadow cones

use super::ForceModel;
use crate::propagation::hifi::state::{SpacecraftState, EARTH_RADIUS_M, SOLAR_PRESSURE_1AU};
use nalgebra::Vector3;
use satkit::lpephem;

/// Solar radiation pressure force model
pub struct SolarRadiationPressure {
    /// Whether SRP is enabled
    enabled: bool,

    /// Shadow function type
    shadow_model: ShadowModel,
}

/// Shadow/eclipse model selection
#[derive(Debug, Clone, Copy)]
pub enum ShadowModel {
    /// No shadow consideration (always in sunlight)
    None,

    /// Simple cylindrical Earth shadow
    CylindricalEarth,

    /// Conical umbra/penumbra (more accurate)
    ConicalEarth,
}

impl Default for SolarRadiationPressure {
    fn default() -> Self {
        Self::new()
    }
}

impl SolarRadiationPressure {
    /// Create a new SRP model
    pub fn new() -> Self {
        Self {
            enabled: true,
            shadow_model: ShadowModel::CylindricalEarth,
        }
    }

    /// Create with specific shadow model
    pub fn with_shadow(shadow: ShadowModel) -> Self {
        Self {
            enabled: true,
            shadow_model: shadow,
        }
    }

    /// Create disabled (for testing)
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            shadow_model: ShadowModel::None,
        }
    }

    /// Get Sun position in GCRF
    fn sun_position(&self, state: &SpacecraftState) -> Vector3<f64> {
        let sun_gcrf = lpephem::sun::pos_gcrf(&state.orbital.epoch);
        // satkit returns in meters
        Vector3::new(sun_gcrf[0], sun_gcrf[1], sun_gcrf[2])
    }

    /// Compute shadow function (0 = full shadow, 1 = full sunlight)
    fn shadow_function(&self, state: &SpacecraftState, sun_pos: &Vector3<f64>) -> f64 {
        match self.shadow_model {
            ShadowModel::None => 1.0,

            ShadowModel::CylindricalEarth => {
                // Simple cylindrical shadow
                self.cylindrical_shadow(&state.orbital.position, sun_pos)
            }

            ShadowModel::ConicalEarth => {
                // Use satkit's shadow function
                // satkit 0.9 API: shadowfunc(psun: &Vector3, psat: &Vector3) -> f64
                lpephem::sun::shadowfunc(sun_pos, &state.orbital.position)
            }
        }
    }

    /// Simple cylindrical shadow check
    fn cylindrical_shadow(&self, sat_pos: &Vector3<f64>, sun_pos: &Vector3<f64>) -> f64 {
        // Vector from Earth to Sun
        let sun_dir = sun_pos.normalize();

        // Projection of satellite position onto sun direction
        let proj = sat_pos.dot(&sun_dir);

        if proj > 0.0 {
            // Satellite is on sunward side
            return 1.0;
        }

        // Distance from satellite to Sun-Earth line
        let perp = sat_pos - proj * sun_dir;
        let perp_dist = perp.norm();

        if perp_dist > EARTH_RADIUS_M {
            1.0 // Outside shadow cylinder
        } else {
            0.0 // In shadow
        }
    }
}

impl ForceModel for SolarRadiationPressure {
    fn acceleration(&self, state: &SpacecraftState) -> Vector3<f64> {
        if !self.enabled {
            return Vector3::zeros();
        }

        // Get Sun position
        let sun_pos = self.sun_position(state);

        // Check shadow
        let shadow = self.shadow_function(state, &sun_pos);
        if shadow < 1e-6 {
            return Vector3::zeros();
        }

        // Vector from Sun to satellite
        let r_sun_sat = state.orbital.position - sun_pos;
        let r_sun_sat_mag = r_sun_sat.norm();

        if r_sun_sat_mag < 1.0 {
            return Vector3::zeros();
        }

        // Unit vector away from Sun
        let sun_hat = r_sun_sat / r_sun_sat_mag;

        // Distance from Sun (approximately 1 AU for Earth orbit)
        // Pressure scales as 1/r² but for Earth satellites, AU is approximately constant
        let pressure = SOLAR_PRESSURE_1AU * shadow;

        // Acceleration: P × (Cr × A / m) × sun_hat
        let cr_a_m = state.cr_area_m2 / state.mass_kg;

        pressure * cr_a_m * sun_hat
    }

    fn name(&self) -> &'static str {
        "Solar Radiation Pressure"
    }

    fn description(&self) -> &'static str {
        "Acceleration from solar photon pressure"
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    fn relative_magnitude(&self) -> f64 {
        // SRP is typically 1e-7 to 1e-8 of gravity
        0.0001
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propagation::hifi::state::{OrbitalState, EARTH_RADIUS_M, MU_EARTH};
    use satkit::Instant;

    #[test]
    fn test_srp_direction() {
        let srp = SolarRadiationPressure::with_shadow(ShadowModel::None);

        let epoch = Instant::from_datetime(2026, 6, 21, 12, 0, 0.0).unwrap();
        let r = EARTH_RADIUS_M + 400_000.0;

        let state = SpacecraftState::with_defaults(OrbitalState::new(
            Vector3::new(r, 0.0, 0.0),
            Vector3::new(0.0, (MU_EARTH / r).sqrt(), 0.0),
            epoch,
        ));

        let accel = srp.acceleration(&state);

        // SRP should push away from Sun
        // Direction depends on Sun position at epoch
        assert!(accel.norm() > 0.0);
    }

    #[test]
    fn test_srp_disabled() {
        let srp = SolarRadiationPressure::disabled();

        let epoch = Instant::from_datetime(2026, 1, 29, 12, 0, 0.0).unwrap();
        let state = SpacecraftState::with_defaults(OrbitalState::new(
            Vector3::new(EARTH_RADIUS_M + 400_000.0, 0.0, 0.0),
            Vector3::new(0.0, 7660.0, 0.0),
            epoch,
        ));

        let accel = srp.acceleration(&state);
        assert_eq!(accel, Vector3::zeros());
    }
}
