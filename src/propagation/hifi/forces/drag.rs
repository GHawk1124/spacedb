//! Atmospheric drag force model
//!
//! Computes acceleration due to atmospheric drag using the formula:
//!
//! a = -½ ρ |v_rel|² (Cd × A / m) v̂_rel
//!
//! where:
//! - ρ is atmospheric density from the configured atmosphere model
//! - v_rel is velocity relative to the rotating atmosphere
//! - Cd × A is the drag coefficient times cross-sectional area
//! - m is spacecraft mass

use super::ForceModel;
use crate::propagation::hifi::atmosphere::AtmosphereModel;
use crate::propagation::hifi::state::{SpacecraftState, OMEGA_EARTH};
use nalgebra::Vector3;

/// Atmospheric drag force model
///
/// This is a generic type parameterized by the atmosphere model,
/// allowing different density models to be used interchangeably.
pub struct AtmosphericDrag<A: AtmosphereModel> {
    /// Atmosphere density model
    atmosphere: A,

    /// Whether drag is currently enabled
    enabled: bool,

    /// Minimum altitude for drag calculation (meters)
    /// Below this, we assume the satellite has re-entered
    min_altitude: f64,

    /// Maximum altitude for drag calculation (meters)
    /// Above this, atmospheric density is negligible
    max_altitude: f64,
}

impl<A: AtmosphereModel> AtmosphericDrag<A> {
    /// Create a new drag model with the given atmosphere
    pub fn new(atmosphere: A) -> Self {
        Self {
            atmosphere,
            enabled: true,
            min_altitude: 100_000.0,   // 100 km
            max_altitude: 1_000_000.0, // 1000 km
        }
    }

    /// Create with custom altitude limits
    pub fn with_limits(atmosphere: A, min_alt: f64, max_alt: f64) -> Self {
        Self {
            atmosphere,
            enabled: true,
            min_altitude: min_alt,
            max_altitude: max_alt,
        }
    }

    /// Enable or disable drag
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get reference to the atmosphere model
    pub fn atmosphere(&self) -> &A {
        &self.atmosphere
    }

    /// Compute velocity relative to rotating atmosphere
    ///
    /// The atmosphere co-rotates with Earth, so we need to subtract
    /// the rotational velocity at the spacecraft's position.
    fn relative_velocity(&self, state: &SpacecraftState) -> Vector3<f64> {
        // Earth's angular velocity vector (pointing along +Z in GCRF)
        // Note: This is approximate since GCRF Z is not exactly aligned with Earth's pole
        let omega = Vector3::new(0.0, 0.0, OMEGA_EARTH);

        // Atmospheric velocity at spacecraft position: ω × r
        let v_atm = omega.cross(&state.orbital.position);

        // Relative velocity: v_spacecraft - v_atmosphere
        state.orbital.velocity - v_atm
    }
}

impl<A: AtmosphereModel> ForceModel for AtmosphericDrag<A> {
    fn acceleration(&self, state: &SpacecraftState) -> Vector3<f64> {
        if !self.enabled {
            return Vector3::zeros();
        }

        let altitude = state.orbital.altitude();

        // Skip if outside atmosphere
        if altitude < self.min_altitude || altitude > self.max_altitude {
            return Vector3::zeros();
        }

        // Get atmospheric density
        let density = self
            .atmosphere
            .density(&state.orbital.position, &state.orbital.epoch);

        if density.rho <= 0.0 {
            return Vector3::zeros();
        }

        // Compute relative velocity
        let v_rel = self.relative_velocity(state);
        let v_rel_mag = v_rel.norm();

        if v_rel_mag < 1e-6 {
            return Vector3::zeros();
        }

        // Unit vector in direction of relative velocity
        let v_hat = v_rel / v_rel_mag;

        // Ballistic coefficient term: Cd × A / m
        let cd_a_m = state.cd_area_m2 / state.mass_kg;

        // Drag acceleration: a = -½ ρ v² (Cd A / m) v̂
        let accel_mag = -0.5 * density.rho * v_rel_mag * v_rel_mag * cd_a_m;

        accel_mag * v_hat
    }

    fn name(&self) -> &'static str {
        "Atmospheric Drag"
    }

    fn description(&self) -> &'static str {
        "Aerodynamic drag from atmospheric density"
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    fn relative_magnitude(&self) -> f64 {
        // Drag is typically 1e-5 to 1e-7 of gravity at LEO altitudes
        0.001
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propagation::hifi::atmosphere::Exponential;
    use crate::propagation::hifi::state::{OrbitalState, EARTH_RADIUS_M, MU_EARTH};
    use satkit::Instant;

    #[test]
    fn test_drag_direction() {
        let atmosphere = Exponential::standard();
        let drag = AtmosphericDrag::new(atmosphere);

        let epoch = Instant::from_datetime(2026, 1, 29, 12, 0, 0.0).unwrap();
        let r = EARTH_RADIUS_M + 400_000.0;
        let v = (MU_EARTH / r).sqrt();

        // Circular orbit in xy plane
        let state = SpacecraftState::with_defaults(OrbitalState::new(
            Vector3::new(r, 0.0, 0.0),
            Vector3::new(0.0, v, 0.0),
            epoch,
        ));

        let accel = drag.acceleration(&state);

        // Drag should oppose velocity (negative y component)
        // Note: relative velocity is slightly different due to atmosphere rotation
        assert!(accel.y < 0.0);
        assert!(accel.norm() > 0.0);
    }

    #[test]
    fn test_drag_above_atmosphere() {
        let atmosphere = Exponential::new(1.225, 8500.0, 500_000.0);
        let drag = AtmosphericDrag::new(atmosphere);

        let epoch = Instant::from_datetime(2026, 1, 29, 12, 0, 0.0).unwrap();
        let r = EARTH_RADIUS_M + 2_000_000.0; // 2000 km - above max altitude

        let state = SpacecraftState::with_defaults(OrbitalState::new(
            Vector3::new(r, 0.0, 0.0),
            Vector3::new(0.0, 7000.0, 0.0),
            epoch,
        ));

        let accel = drag.acceleration(&state);
        assert_eq!(accel, Vector3::zeros());
    }

    #[test]
    fn test_drag_scales_with_density() {
        let atm_dense = Exponential::new(1.225, 8500.0, 1_000_000.0);
        let atm_thin = Exponential::new(1.225, 20000.0, 1_000_000.0); // Larger scale height = less dense

        let drag_dense = AtmosphericDrag::new(atm_dense);
        let drag_thin = AtmosphericDrag::new(atm_thin);

        let epoch = Instant::from_datetime(2026, 1, 29, 12, 0, 0.0).unwrap();
        let r = EARTH_RADIUS_M + 400_000.0;

        let state = SpacecraftState::with_defaults(OrbitalState::new(
            Vector3::new(r, 0.0, 0.0),
            Vector3::new(0.0, 7660.0, 0.0),
            epoch,
        ));

        let accel_dense = drag_dense.acceleration(&state);
        let accel_thin = drag_thin.acceleration(&state);

        // Dense atmosphere should produce more drag
        assert!(accel_dense.norm() > accel_thin.norm());
    }
}
