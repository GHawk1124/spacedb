//! Orbital propagation module
//!
//! This module provides two propagation approaches:
//!
//! ## SGP4 Propagation (Fast)
//!
//! The `propagator` and `orbit_track` submodules use SGP4 via satkit for
//! fast, TLE-based propagation suitable for real-time visualization of
//! thousands of satellites.
//!
//! ## High-Fidelity Propagation (Accurate)
//!
//! The `hifi` submodule provides numerical integration with configurable
//! force models (gravity, drag, SRP, third-body) for accurate orbital
//! decay prediction and precise ephemeris generation.
//!
//! # Example
//!
//! ```ignore
//! use spacedb::propagation::hifi::*;
//!
//! // Create propagator with J2 gravity and NRLMSISE-00 atmosphere
//! let forces = CompositeForce::leo_basic();
//! let integrator = SatkitRK98::new();
//! let propagator = HiFiPropagator::new(integrator, forces);
//!
//! // Propagate spacecraft state
//! let result = propagator.propagate(initial_state, target_epoch)?;
//! ```

pub mod hifi;
mod orbit_track;
mod propagator;

pub use orbit_track::*;
pub use propagator::*;
