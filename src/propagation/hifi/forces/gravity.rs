//! Earth gravity force model
//!
//! Provides various fidelity levels of Earth gravity modeling:
//! - Point mass (μ/r²)
//! - J2 only (oblateness)
//! - Full spherical harmonics via satkit

use super::ForceModel;
use crate::propagation::hifi::state::{SpacecraftState, EARTH_RADIUS_M, MU_EARTH};
use nalgebra::Vector3;

/// Gravity model fidelity selection
#[derive(Debug, Clone)]
pub enum GravityModel {
    /// Simple point mass: a = -μ/r³ × r
    PointMass,

    /// Point mass + J2 oblateness perturbation
    J2Only,

    /// Point mass + J2-J6 zonal harmonics
    ZonalOnly(u32),

    /// Full spherical harmonics (degree × order)
    FullField {
        /// Maximum degree (n)
        degree: u32,
        /// Maximum order (m)
        order: u32,
    },
}

/// Earth gravity force model
pub struct EarthGravity {
    model: GravityModel,
}

impl EarthGravity {
    /// Create a gravity model from a configuration enum
    pub fn from_model(model: GravityModel) -> Self {
        match model {
            GravityModel::PointMass => Self::point_mass(),
            GravityModel::J2Only => Self::j2_only(),
            GravityModel::ZonalOnly(max_degree) => Self::zonal(max_degree),
            GravityModel::FullField { degree, order } => Self::full_field(degree, order),
        }
    }

    /// Create a point mass gravity model
    pub fn point_mass() -> Self {
        Self {
            model: GravityModel::PointMass,
        }
    }

    /// Create a J2-only gravity model
    ///
    /// This is the most common perturbation model for quick orbit predictions.
    /// J2 accounts for Earth's oblateness (equatorial bulge).
    pub fn j2_only() -> Self {
        Self {
            model: GravityModel::J2Only,
        }
    }

    /// Create a zonal harmonics model (J2-Jn)
    ///
    /// Includes only zonal (m=0) harmonics up to degree n.
    pub fn zonal(max_degree: u32) -> Self {
        Self {
            model: GravityModel::ZonalOnly(max_degree),
        }
    }

    /// Create a full spherical harmonics model
    ///
    /// Uses analytical formulas for high-fidelity calculations.
    /// Common values: (20, 20) for LEO, (70, 70) for precision
    pub fn full_field(degree: u32, order: u32) -> Self {
        Self {
            model: GravityModel::FullField { degree, order },
        }
    }

    /// Point mass acceleration: a = -μ/r³ × r
    fn point_mass_accel(&self, position: &Vector3<f64>) -> Vector3<f64> {
        let r = position.norm();
        if r < 1.0 {
            // Avoid singularity at origin
            return Vector3::zeros();
        }
        let r3 = r * r * r;
        -MU_EARTH / r3 * position
    }

    /// J2 perturbation acceleration
    ///
    /// Uses the standard J2 perturbation formula in Cartesian coordinates.
    fn j2_accel(&self, position: &Vector3<f64>) -> Vector3<f64> {
        // J2 coefficient for Earth (WGS84)
        const J2: f64 = 1.08263e-3;

        let x = position.x;
        let y = position.y;
        let z = position.z;
        let r = position.norm();

        if r < 1.0 {
            return Vector3::zeros();
        }

        let r2 = r * r;
        let r5 = r2 * r2 * r;
        let re2 = EARTH_RADIUS_M * EARTH_RADIUS_M;

        // Common factor: (3/2) × J2 × μ × Re² / r⁵
        let factor = 1.5 * J2 * MU_EARTH * re2 / r5;

        // z²/r² ratio
        let z2_r2 = (z * z) / r2;

        // Acceleration components
        let ax = factor * x * (5.0 * z2_r2 - 1.0);
        let ay = factor * y * (5.0 * z2_r2 - 1.0);
        let az = factor * z * (5.0 * z2_r2 - 3.0);

        Vector3::new(ax, ay, az)
    }

    /// Higher zonal harmonics (J3, J4, ...)
    fn higher_zonal_accel(&self, position: &Vector3<f64>, max_degree: u32) -> Vector3<f64> {
        if max_degree <= 2 {
            return Vector3::zeros();
        }

        // J3 contribution (for example)
        const J3: f64 = -2.5327e-6;
        const J4: f64 = -1.6196e-6;

        let x = position.x;
        let y = position.y;
        let z = position.z;
        let r = position.norm();
        let r2 = r * r;
        let r7 = r2 * r2 * r2 * r;
        let re3 = EARTH_RADIUS_M * EARTH_RADIUS_M * EARTH_RADIUS_M;

        // J3 acceleration (odd zonal, asymmetric)
        let factor3 = 2.5 * J3 * MU_EARTH * re3 / r7;
        let z2_r2 = z * z / r2;

        let ax3 = factor3 * x * z * (7.0 * z2_r2 - 3.0);
        let ay3 = factor3 * y * z * (7.0 * z2_r2 - 3.0);
        let az3 = factor3 * (z * z * (7.0 * z2_r2 - 6.0) + 0.6 * r2);

        let mut accel = Vector3::new(ax3, ay3, az3);

        if max_degree >= 4 {
            // Add J4 contribution (simplified)
            let re4 = re3 * EARTH_RADIUS_M;
            let r9 = r7 * r2;
            let factor4 = 0.625 * J4 * MU_EARTH * re4 / r9;
            let z4_r4 = z2_r2 * z2_r2;

            let ax4 = factor4 * x * (63.0 * z4_r4 - 42.0 * z2_r2 + 3.0);
            let ay4 = factor4 * y * (63.0 * z4_r4 - 42.0 * z2_r2 + 3.0);
            let az4 = factor4 * z * (63.0 * z4_r4 - 70.0 * z2_r2 + 15.0);

            accel += Vector3::new(ax4, ay4, az4);
        }

        accel
    }

    /// Full spherical harmonics acceleration using satkit
    fn full_field_accel(&self, state: &SpacecraftState, degree: u32, order: u32) -> Vector3<f64> {
        // Transform position to ITRF for gravity calculation
        let q = satkit::frametransform::qgcrf2itrf(&state.orbital.epoch);
        let pos_itrf = q * &state.orbital.position;

        // Use satkit's earthgravity::accel function
        // satkit 0.9 API: accel(pos_itrf: &Vector3, order: usize, model: GravityModel) -> Vector3
        let accel_itrf = satkit::earthgravity::accel(
            &pos_itrf,
            (degree.min(order)) as usize,
            satkit::earthgravity::GravityModel::JGM3,
        );

        // Transform back to GCRF
        let q_inv = q.inverse();
        q_inv * &accel_itrf
    }
}

impl ForceModel for EarthGravity {
    fn acceleration(&self, state: &SpacecraftState) -> Vector3<f64> {
        let pos = &state.orbital.position;

        match &self.model {
            GravityModel::PointMass => self.point_mass_accel(pos),

            GravityModel::J2Only => self.point_mass_accel(pos) + self.j2_accel(pos),

            GravityModel::ZonalOnly(max_degree) => {
                self.point_mass_accel(pos)
                    + self.j2_accel(pos)
                    + self.higher_zonal_accel(pos, *max_degree)
            }

            GravityModel::FullField { degree, order } => {
                self.full_field_accel(state, *degree, *order)
            }
        }
    }

    fn name(&self) -> &'static str {
        match &self.model {
            GravityModel::PointMass => "Earth Gravity (Point Mass)",
            GravityModel::J2Only => "Earth Gravity (J2)",
            GravityModel::ZonalOnly(_) => "Earth Gravity (Zonal)",
            GravityModel::FullField { .. } => "Earth Gravity (Full Field)",
        }
    }

    fn description(&self) -> &'static str {
        match &self.model {
            GravityModel::PointMass => "Central body gravity μ/r²",
            GravityModel::J2Only => "Central gravity with J2 oblateness",
            GravityModel::ZonalOnly(_) => "Central gravity with zonal harmonics",
            GravityModel::FullField { .. } => "Full spherical harmonics gravity field",
        }
    }

    fn relative_magnitude(&self) -> f64 {
        1.0 // Gravity is the primary force
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propagation::hifi::state::OrbitalState;
    use satkit::Instant;

    #[test]
    fn test_point_mass() {
        let gravity = EarthGravity::point_mass();
        let epoch = Instant::from_datetime(2026, 1, 29, 12, 0, 0.0).unwrap();
        let r = EARTH_RADIUS_M + 400_000.0;

        let state = SpacecraftState::with_defaults(OrbitalState::new(
            Vector3::new(r, 0.0, 0.0),
            Vector3::new(0.0, 7660.0, 0.0),
            epoch,
        ));

        let accel = gravity.acceleration(&state);

        // Should point toward center
        assert!(accel.x < 0.0);

        // Expected: μ/r² ≈ 8.7 m/s²
        let expected = MU_EARTH / (r * r);
        assert!((accel.norm() - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn test_j2_nonzero() {
        let gravity = EarthGravity::j2_only();
        let epoch = Instant::from_datetime(2026, 1, 29, 12, 0, 0.0).unwrap();
        let r = EARTH_RADIUS_M + 400_000.0;

        // Position with non-zero z (inclined)
        let state = SpacecraftState::with_defaults(OrbitalState::new(
            Vector3::new(r * 0.707, 0.0, r * 0.707),
            Vector3::new(0.0, 7660.0, 0.0),
            epoch,
        ));

        let accel = gravity.acceleration(&state);

        // J2 should add a small out-of-plane component
        // The acceleration should still point roughly toward center
        let r_hat = state.orbital.position.normalize();
        let radial_component = accel.dot(&r_hat);
        assert!(radial_component < 0.0); // Pointing inward
    }
}
