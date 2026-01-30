//! High-fidelity orbit propagation module
//!
//! This module provides accurate numerical integration for orbital mechanics,
//! designed for orbital decay prediction and precise position forecasting.
//!
//! # Architecture
//!
//! The module is organized around composable, hot-swappable components:
//!
//! - **Integrator**: Numerical integration methods (RK 9(8) via satkit, future CUDA)
//! - **AtmosphereModel**: Atmospheric density models (NRLMSISE-00, JB2008, etc.)
//! - **ForceModel**: Individual force contributions (gravity, drag, SRP, third-body)
//! - **HiFiPropagator**: Orchestrates integration with configurable forces
//!
//! # Example
//!
//! ```ignore
//! use spacedb::propagation::hifi::*;
//!
//! // Create atmosphere model
//! let atmosphere = Nrlmsise00::new();
//!
//! // Build composite force model
//! let mut forces = CompositeForce::new();
//! forces.add(Box::new(EarthGravity::j2_only()));
//! forces.add(Box::new(AtmosphericDrag::new(atmosphere)));
//!
//! // Create propagator with satkit RK 9(8)
//! let integrator = SatkitRK98::new();
//! let propagator = HiFiPropagator::new(integrator, forces);
//!
//! // Propagate spacecraft state
//! let result = propagator.propagate(initial_state, target_epoch)?;
//! ```

pub mod atmosphere;
pub mod forces;
pub mod integrator;
pub mod state;

mod hifi_propagator;
mod settings;

// Re-export main types
pub use atmosphere::{AtmosphereDensity, AtmosphereModel, AtmosphereModelType};
pub use forces::{CompositeForce, ForceModel};
pub use hifi_propagator::{HiFiPropagator, PropagationError, PropagationResult, PropagatorConfig};
pub use integrator::{Integrator, NativeRK4, SatkitRK98};
pub use settings::{GravityModelChoice, HiFiSettings, IntegratorType, PropagatorType};
pub use state::{OrbitalState, SpacecraftState};

/// Build an orbital state in GCRF from a TLE at a given epoch
pub fn orbital_state_from_tle(tle: &satkit::TLE, epoch: &satkit::Instant) -> Option<OrbitalState> {
    let mut tle = tle.clone();
    let result = satkit::sgp4::sgp4(&mut tle, &[epoch.clone()]).ok()?;

    let pos = result.pos.column(0);
    let vel = result.vel.column(0);

    let pos_teme = nalgebra::Vector3::new(pos[0], pos[1], pos[2]);
    let vel_teme = nalgebra::Vector3::new(vel[0], vel[1], vel[2]);

    let q_teme_to_gcrf = satkit::frametransform::qteme2gcrf(epoch);
    let pos_gcrf = q_teme_to_gcrf.transform_vector(&pos_teme);
    let vel_gcrf = q_teme_to_gcrf.transform_vector(&vel_teme);

    Some(OrbitalState::new(pos_gcrf, vel_gcrf, *epoch))
}
