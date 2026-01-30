//! Force models for orbital mechanics
//!
//! This module provides composable force models that can be combined
//! to create a complete dynamics model for orbit propagation.
//!
//! # Architecture
//!
//! Each force model implements the `ForceModel` trait, which computes
//! the acceleration contribution at a given spacecraft state.
//!
//! Multiple models are combined using `CompositeForce`, which sums
//! all enabled force contributions.
//!
//! # Available Models
//!
//! - **EarthGravity**: Central body gravity with optional J2-Jn perturbations
//! - **AtmosphericDrag**: Drag using configurable atmosphere models
//! - **SolarRadiationPressure**: SRP (placeholder)
//! - **ThirdBody**: Sun/Moon gravitational perturbations (placeholder)

mod drag;
mod gravity;
mod srp;
mod third_body;

pub use drag::AtmosphericDrag;
pub use gravity::{EarthGravity, GravityModel};
pub use srp::SolarRadiationPressure;
pub use third_body::ThirdBody;

use crate::propagation::hifi::state::SpacecraftState;
use nalgebra::Vector3;

/// Trait for force model contributions
///
/// Each force model computes its acceleration contribution at a given
/// spacecraft state. Models should be thread-safe for parallel propagation.
pub trait ForceModel: Send + Sync {
    /// Compute acceleration contribution at given state
    ///
    /// # Arguments
    ///
    /// * `state` - Current spacecraft state (position, velocity, epoch, physical properties)
    ///
    /// # Returns
    ///
    /// Acceleration vector in the same frame as state.position (GCRF), in m/s²
    fn acceleration(&self, state: &SpacecraftState) -> Vector3<f64>;

    /// Force model name for debugging and logging
    fn name(&self) -> &'static str;

    /// Brief description of the model
    fn description(&self) -> &'static str {
        self.name()
    }

    /// Whether this force model is currently enabled
    ///
    /// Disabled models are skipped during acceleration computation.
    fn enabled(&self) -> bool {
        true
    }

    /// Estimated relative contribution to total acceleration
    ///
    /// Used for adaptive step sizing. Returns a value 0-1 where:
    /// - 1.0 = primary force (central gravity)
    /// - 0.1 = significant perturbation (J2)
    /// - 0.01 = minor perturbation (drag, SRP)
    fn relative_magnitude(&self) -> f64 {
        0.1
    }
}

/// Composite force model that aggregates multiple force contributions
///
/// This is the primary way to combine multiple force models into
/// a complete dynamics model.
///
/// # Example
///
/// ```ignore
/// let mut forces = CompositeForce::new();
/// forces.add(Box::new(EarthGravity::j2_only()));
/// forces.add(Box::new(AtmosphericDrag::new(atmosphere)));
/// forces.add(Box::new(SolarRadiationPressure::new()));
///
/// let total_accel = forces.total_acceleration(&state);
/// ```
pub struct CompositeForce {
    forces: Vec<Box<dyn ForceModel>>,
}

impl Default for CompositeForce {
    fn default() -> Self {
        Self::new()
    }
}

impl CompositeForce {
    /// Create an empty composite force model
    pub fn new() -> Self {
        Self { forces: Vec::new() }
    }

    /// Add a force model to the composite
    pub fn add(&mut self, force: Box<dyn ForceModel>) {
        log::debug!("Adding force model: {}", force.name());
        self.forces.push(force);
    }

    /// Create a builder for convenient force model construction
    pub fn builder() -> CompositeForceBuilder {
        CompositeForceBuilder::new()
    }

    /// Get the number of force models
    pub fn len(&self) -> usize {
        self.forces.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.forces.is_empty()
    }

    /// List all force model names
    pub fn model_names(&self) -> Vec<&'static str> {
        self.forces.iter().map(|f| f.name()).collect()
    }

    /// Compute total acceleration from all enabled forces
    pub fn total_acceleration(&self, state: &SpacecraftState) -> Vector3<f64> {
        self.forces
            .iter()
            .filter(|f| f.enabled())
            .map(|f| f.acceleration(state))
            .fold(Vector3::zeros(), |acc, a| acc + a)
    }

    /// Compute acceleration with individual contributions for debugging
    pub fn acceleration_breakdown(
        &self,
        state: &SpacecraftState,
    ) -> Vec<(&'static str, Vector3<f64>)> {
        self.forces
            .iter()
            .filter(|f| f.enabled())
            .map(|f| (f.name(), f.acceleration(state)))
            .collect()
    }
}

/// Builder for CompositeForce
pub struct CompositeForceBuilder {
    forces: Vec<Box<dyn ForceModel>>,
}

impl CompositeForceBuilder {
    fn new() -> Self {
        Self { forces: Vec::new() }
    }

    /// Add a force model
    pub fn with(mut self, force: Box<dyn ForceModel>) -> Self {
        self.forces.push(force);
        self
    }

    /// Add central body gravity
    pub fn with_gravity(self, model: EarthGravity) -> Self {
        self.with(Box::new(model))
    }

    /// Add atmospheric drag
    pub fn with_drag<A: crate::propagation::hifi::atmosphere::AtmosphereModel + 'static>(
        self,
        drag: AtmosphericDrag<A>,
    ) -> Self {
        self.with(Box::new(drag))
    }

    /// Add solar radiation pressure
    pub fn with_srp(self, srp: SolarRadiationPressure) -> Self {
        self.with(Box::new(srp))
    }

    /// Build the composite force model
    pub fn build(self) -> CompositeForce {
        CompositeForce {
            forces: self.forces,
        }
    }
}

/// Standard force model configurations
impl CompositeForce {
    /// Minimal configuration: point mass gravity only
    pub fn point_mass_only() -> Self {
        let mut forces = Self::new();
        forces.add(Box::new(EarthGravity::point_mass()));
        forces
    }

    /// Basic LEO configuration: J2 gravity + drag
    pub fn leo_basic() -> Self {
        use crate::propagation::hifi::atmosphere::Nrlmsise00;

        let mut forces = Self::new();
        forces.add(Box::new(EarthGravity::j2_only()));
        forces.add(Box::new(AtmosphericDrag::new(Nrlmsise00::new())));
        forces
    }

    /// High-fidelity LEO configuration: full gravity + drag + SRP
    pub fn leo_high_fidelity() -> Self {
        use crate::propagation::hifi::atmosphere::Nrlmsise00;

        let mut forces = Self::new();
        forces.add(Box::new(EarthGravity::full_field(20, 20)));
        forces.add(Box::new(AtmosphericDrag::new(Nrlmsise00::new())));
        forces.add(Box::new(SolarRadiationPressure::new()));
        forces.add(Box::new(ThirdBody::sun_and_moon()));
        forces
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propagation::hifi::state::{OrbitalState, EARTH_RADIUS_M, MU_EARTH};
    use satkit::Instant;

    #[test]
    fn test_composite_force_empty() {
        let forces = CompositeForce::new();
        assert!(forces.is_empty());

        let epoch = Instant::from_datetime(2026, 1, 29, 12, 0, 0.0).unwrap();
        let state = SpacecraftState::with_defaults(OrbitalState::new(
            Vector3::new(EARTH_RADIUS_M + 400_000.0, 0.0, 0.0),
            Vector3::new(0.0, (MU_EARTH / (EARTH_RADIUS_M + 400_000.0)).sqrt(), 0.0),
            epoch,
        ));

        let accel = forces.total_acceleration(&state);
        assert_eq!(accel, Vector3::zeros());
    }

    #[test]
    fn test_composite_force_gravity() {
        let mut forces = CompositeForce::new();
        forces.add(Box::new(EarthGravity::point_mass()));

        let epoch = Instant::from_datetime(2026, 1, 29, 12, 0, 0.0).unwrap();
        let r = EARTH_RADIUS_M + 400_000.0;
        let state = SpacecraftState::with_defaults(OrbitalState::new(
            Vector3::new(r, 0.0, 0.0),
            Vector3::new(0.0, (MU_EARTH / r).sqrt(), 0.0),
            epoch,
        ));

        let accel = forces.total_acceleration(&state);

        // Should be pointing toward Earth center (negative x)
        assert!(accel.x < 0.0);
        assert!(accel.y.abs() < 1e-10);
        assert!(accel.z.abs() < 1e-10);

        // Magnitude should be μ/r² ≈ 8.7 m/s² at 400 km
        let expected_mag = MU_EARTH / (r * r);
        assert!((accel.norm() - expected_mag).abs() / expected_mag < 1e-10);
    }
}
