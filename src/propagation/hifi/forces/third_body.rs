//! Third-body gravitational perturbations (placeholder)
//!
//! Models gravitational effects from the Sun and Moon.
//!
//! # Status
//!
//! This is a placeholder with a basic implementation using
//! satkit's low-precision ephemeris.

use super::ForceModel;
use crate::propagation::hifi::state::SpacecraftState;
use nalgebra::Vector3;
use satkit::lpephem;

/// Third-body gravitational perturbation model
pub struct ThirdBody {
    /// Include Sun
    include_sun: bool,

    /// Include Moon
    include_moon: bool,

    /// Whether enabled
    enabled: bool,
}

impl Default for ThirdBody {
    fn default() -> Self {
        Self::sun_and_moon()
    }
}

impl ThirdBody {
    /// Create a model with both Sun and Moon
    pub fn sun_and_moon() -> Self {
        Self {
            include_sun: true,
            include_moon: true,
            enabled: true,
        }
    }

    /// Create a Sun-only model
    pub fn sun_only() -> Self {
        Self {
            include_sun: true,
            include_moon: false,
            enabled: true,
        }
    }

    /// Create a Moon-only model
    pub fn moon_only() -> Self {
        Self {
            include_sun: false,
            include_moon: true,
            enabled: true,
        }
    }

    /// Create disabled
    pub fn disabled() -> Self {
        Self {
            include_sun: false,
            include_moon: false,
            enabled: false,
        }
    }

    /// Compute third-body acceleration
    ///
    /// Uses the standard formula:
    /// a = μ_body × (r_sat_body/|r_sat_body|³ - r_earth_body/|r_earth_body|³)
    fn third_body_accel(
        &self,
        sat_pos: &Vector3<f64>,
        body_pos: &Vector3<f64>,
        mu_body: f64,
    ) -> Vector3<f64> {
        // Vector from satellite to body
        let r_sat_body = body_pos - sat_pos;
        let r_sat_body_mag = r_sat_body.norm();

        if r_sat_body_mag < 1.0 {
            return Vector3::zeros();
        }

        // Vector from Earth center to body (body_pos in GCRF is already this)
        let r_earth_body = body_pos;
        let r_earth_body_mag = r_earth_body.norm();

        if r_earth_body_mag < 1.0 {
            return Vector3::zeros();
        }

        // Acceleration components
        let term1 = r_sat_body / (r_sat_body_mag.powi(3));
        let term2 = r_earth_body / (r_earth_body_mag.powi(3));

        mu_body * (term1 - term2)
    }
}

impl ForceModel for ThirdBody {
    fn acceleration(&self, state: &SpacecraftState) -> Vector3<f64> {
        if !self.enabled {
            return Vector3::zeros();
        }

        let mut accel = Vector3::zeros();

        // Sun perturbation
        if self.include_sun {
            let sun_gcrf = lpephem::sun::pos_gcrf(&state.orbital.epoch);
            let sun_pos = Vector3::new(sun_gcrf[0], sun_gcrf[1], sun_gcrf[2]);

            // Sun's gravitational parameter (m³/s²)
            const MU_SUN: f64 = 1.32712440018e20;

            accel += self.third_body_accel(&state.orbital.position, &sun_pos, MU_SUN);
        }

        // Moon perturbation
        if self.include_moon {
            let moon_gcrf = lpephem::moon::pos_gcrf(&state.orbital.epoch);
            let moon_pos = Vector3::new(moon_gcrf[0], moon_gcrf[1], moon_gcrf[2]);

            // Moon's gravitational parameter (m³/s²)
            const MU_MOON: f64 = 4.902800066e12;

            accel += self.third_body_accel(&state.orbital.position, &moon_pos, MU_MOON);
        }

        accel
    }

    fn name(&self) -> &'static str {
        match (self.include_sun, self.include_moon) {
            (true, true) => "Third-Body (Sun+Moon)",
            (true, false) => "Third-Body (Sun)",
            (false, true) => "Third-Body (Moon)",
            (false, false) => "Third-Body (disabled)",
        }
    }

    fn description(&self) -> &'static str {
        "Gravitational perturbations from Sun and Moon"
    }

    fn enabled(&self) -> bool {
        self.enabled && (self.include_sun || self.include_moon)
    }

    fn relative_magnitude(&self) -> f64 {
        // Third-body effects are typically 1e-7 to 1e-9 of central gravity
        0.0001
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propagation::hifi::state::{OrbitalState, EARTH_RADIUS_M, MU_EARTH};
    use satkit::Instant;

    #[test]
    fn test_third_body_nonzero() {
        let third_body = ThirdBody::sun_and_moon();

        let epoch = Instant::from_datetime(2026, 1, 29, 12, 0, 0.0).unwrap();
        let r = EARTH_RADIUS_M + 400_000.0;

        let state = SpacecraftState::with_defaults(OrbitalState::new(
            Vector3::new(r, 0.0, 0.0),
            Vector3::new(0.0, (MU_EARTH / r).sqrt(), 0.0),
            epoch,
        ));

        let accel = third_body.acceleration(&state);

        // Should be non-zero but small
        assert!(accel.norm() > 0.0);
        assert!(accel.norm() < 1e-4); // Much less than gravity
    }

    #[test]
    fn test_third_body_disabled() {
        let third_body = ThirdBody::disabled();

        let epoch = Instant::from_datetime(2026, 1, 29, 12, 0, 0.0).unwrap();
        let state = SpacecraftState::with_defaults(OrbitalState::new(
            Vector3::new(EARTH_RADIUS_M + 400_000.0, 0.0, 0.0),
            Vector3::new(0.0, 7660.0, 0.0),
            epoch,
        ));

        let accel = third_body.acceleration(&state);
        assert_eq!(accel, Vector3::zeros());
    }
}
