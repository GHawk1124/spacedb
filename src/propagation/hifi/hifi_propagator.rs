//! High-fidelity orbit propagator
//!
//! Orchestrates the numerical integration with configurable force models
//! to produce accurate orbital predictions.

use super::forces::CompositeForce;
use super::integrator::{Integrator, StepResult};
use super::state::{OrbitalState, SpacecraftState};
use satkit::{Duration, Instant};

/// Result of orbit propagation
#[derive(Debug, Clone)]
pub struct PropagationResult {
    /// Final spacecraft state
    pub final_state: SpacecraftState,

    /// State history (if requested)
    pub history: Vec<SpacecraftState>,

    /// Total number of integration steps taken
    pub steps_taken: usize,

    /// Total propagation time in seconds
    pub propagation_time: f64,

    /// Whether propagation completed successfully
    pub success: bool,

    /// Error message if propagation failed
    pub error_message: Option<String>,
}

/// Propagation error types
#[derive(Debug, Clone)]
pub enum PropagationError {
    /// Satellite re-entered atmosphere
    Reentry { altitude_km: f64, epoch: Instant },

    /// Satellite escaped Earth's gravity
    Escape { epoch: Instant },

    /// Integration failed to converge
    IntegrationFailure { message: String },

    /// Invalid initial state
    InvalidState { message: String },

    /// Target epoch is before initial epoch (backward propagation not supported yet)
    BackwardPropagation,
}

impl std::fmt::Display for PropagationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reentry { altitude_km, epoch } => {
                write!(
                    f,
                    "Satellite re-entered at altitude {:.1} km at {:?}",
                    altitude_km, epoch
                )
            }
            Self::Escape { epoch } => {
                write!(f, "Satellite escaped Earth's gravity at {:?}", epoch)
            }
            Self::IntegrationFailure { message } => {
                write!(f, "Integration failed: {}", message)
            }
            Self::InvalidState { message } => {
                write!(f, "Invalid state: {}", message)
            }
            Self::BackwardPropagation => {
                write!(f, "Backward propagation not yet supported")
            }
        }
    }
}

impl std::error::Error for PropagationError {}

/// High-fidelity orbit propagator configuration
#[derive(Clone, Debug)]
pub struct PropagatorConfig {
    /// Nominal step size in seconds
    pub step_size: f64,

    /// Error tolerance for adaptive stepping
    pub tolerance: f64,

    /// Maximum number of steps before giving up
    pub max_steps: usize,

    /// Whether to store state history
    pub store_history: bool,

    /// History output interval (seconds, 0 = every step)
    pub history_interval: f64,

    /// Minimum altitude before declaring reentry (meters)
    pub reentry_altitude: f64,

    /// Maximum altitude before declaring escape (meters)
    pub escape_altitude: f64,
}

impl Default for PropagatorConfig {
    fn default() -> Self {
        Self {
            step_size: 60.0,      // 1 minute
            tolerance: 1e-10,     // High precision
            max_steps: 1_000_000, // ~1 million steps max
            store_history: false,
            history_interval: 0.0,
            reentry_altitude: 100_000.0,      // 100 km
            escape_altitude: 1_000_000_000.0, // 1 million km
        }
    }
}

impl PropagatorConfig {
    /// Quick propagation settings (lower accuracy, faster)
    pub fn fast() -> Self {
        Self {
            step_size: 120.0,
            tolerance: 1e-8,
            max_steps: 100_000,
            ..Default::default()
        }
    }

    /// High-precision settings
    pub fn high_precision() -> Self {
        Self {
            step_size: 30.0,
            tolerance: 1e-12,
            max_steps: 10_000_000,
            ..Default::default()
        }
    }

    /// With history storage enabled
    pub fn with_history(mut self, interval: f64) -> Self {
        self.store_history = true;
        self.history_interval = interval;
        self
    }
}

/// High-fidelity orbit propagator
///
/// Combines a numerical integrator with force models to propagate
/// spacecraft states forward in time.
///
pub struct HiFiPropagator {
    /// Numerical integrator (swappable at runtime)
    integrator: Box<dyn Integrator>,

    /// Composite force model
    forces: CompositeForce,

    /// Propagation configuration
    config: PropagatorConfig,
}

impl HiFiPropagator {
    /// Create a new propagator with the given integrator and forces
    pub fn new(integrator: impl Integrator + 'static, forces: CompositeForce) -> Self {
        Self {
            integrator: Box::new(integrator),
            forces,
            config: PropagatorConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        integrator: Box<dyn Integrator>,
        forces: CompositeForce,
        config: PropagatorConfig,
    ) -> Self {
        Self {
            integrator,
            forces,
            config,
        }
    }

    /// Get configuration reference
    pub fn config(&self) -> &PropagatorConfig {
        &self.config
    }

    /// Get mutable configuration reference
    pub fn config_mut(&mut self) -> &mut PropagatorConfig {
        &mut self.config
    }

    /// Propagate from initial state to target epoch
    pub fn propagate(
        &self,
        initial: SpacecraftState,
        target_epoch: Instant,
    ) -> Result<PropagationResult, PropagationError> {
        // Validate initial state
        if initial.orbital.position.norm() < 1.0 {
            return Err(PropagationError::InvalidState {
                message: "Position magnitude too small".to_string(),
            });
        }

        // Check propagation direction
        let total_duration = (target_epoch - initial.orbital.epoch).as_seconds();
        if total_duration < 0.0 {
            return Err(PropagationError::BackwardPropagation);
        }

        if total_duration == 0.0 {
            return Ok(PropagationResult {
                final_state: initial,
                history: vec![],
                steps_taken: 0,
                propagation_time: 0.0,
                success: true,
                error_message: None,
            });
        }

        let mut state = initial;
        let mut history = Vec::new();
        let mut steps = 0;
        let mut last_history_time = state.orbital.epoch;

        // Main propagation loop
        while state.orbital.epoch < target_epoch && steps < self.config.max_steps {
            // Compute remaining time
            let remaining = (target_epoch - state.orbital.epoch).as_seconds();
            let dt = remaining.min(self.config.step_size);

            // Define derivatives function for the integrator
            let forces = &self.forces;
            let derivatives = |orbital_state: &OrbitalState| {
                // Create temporary spacecraft state for force calculation
                let temp_state = SpacecraftState {
                    orbital: orbital_state.clone(),
                    mass_kg: state.mass_kg,
                    cd_area_m2: state.cd_area_m2,
                    cr_area_m2: state.cr_area_m2,
                };

                let accel = forces.total_acceleration(&temp_state);
                (orbital_state.velocity, accel)
            };

            // Take integration step
            let result: StepResult = self.integrator.adaptive_step(
                &state.orbital,
                dt,
                self.config.tolerance,
                &derivatives,
            );

            if !result.success {
                return Err(PropagationError::IntegrationFailure {
                    message: "Integrator rejected step".to_string(),
                });
            }

            // Update state
            state.orbital = result.state;
            steps += 1;

            // Check for reentry
            let altitude = state.orbital.altitude();
            if altitude < self.config.reentry_altitude {
                return Err(PropagationError::Reentry {
                    altitude_km: altitude / 1000.0,
                    epoch: state.orbital.epoch,
                });
            }

            // Check for escape
            if altitude > self.config.escape_altitude {
                return Err(PropagationError::Escape {
                    epoch: state.orbital.epoch,
                });
            }

            // Store history if requested
            if self.config.store_history {
                let time_since_last = (state.orbital.epoch - last_history_time).as_seconds();
                if self.config.history_interval <= 0.0
                    || time_since_last >= self.config.history_interval
                {
                    history.push(state.clone());
                    last_history_time = state.orbital.epoch;
                }
            }
        }

        // Check if we completed
        let success = steps < self.config.max_steps;

        Ok(PropagationResult {
            final_state: state,
            history,
            steps_taken: steps,
            propagation_time: total_duration,
            success,
            error_message: if success {
                None
            } else {
                Some("Maximum steps exceeded".to_string())
            },
        })
    }

    /// Propagate for a duration
    pub fn propagate_duration(
        &self,
        initial: SpacecraftState,
        duration: Duration,
    ) -> Result<PropagationResult, PropagationError> {
        let target_epoch = initial.orbital.epoch + duration;
        self.propagate(initial, target_epoch)
    }

    /// Propagate for a number of orbital periods
    pub fn propagate_periods(
        &self,
        initial: SpacecraftState,
        periods: f64,
    ) -> Result<PropagationResult, PropagationError> {
        let period = initial.orbital.period().unwrap_or(5400.0); // Default ~90 min
        let duration = Duration::from_seconds(period * periods);
        self.propagate_duration(initial, duration)
    }

    /// Get a state at a specific epoch (convenience method)
    pub fn state_at(
        &self,
        initial: SpacecraftState,
        target_epoch: Instant,
    ) -> Result<SpacecraftState, PropagationError> {
        let result = self.propagate(initial, target_epoch)?;
        Ok(result.final_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propagation::hifi::forces::EarthGravity;
    use crate::propagation::hifi::integrator::NativeRK4;
    use crate::propagation::hifi::state::{EARTH_RADIUS_M, MU_EARTH};

    #[test]
    fn test_propagate_one_orbit() {
        // Set up propagator with point mass gravity
        let mut forces = CompositeForce::new();
        forces.add(Box::new(EarthGravity::point_mass()));

        let integrator = NativeRK4::new();
        let propagator = HiFiPropagator::new(integrator, forces);

        // Circular orbit at 400 km
        let r = EARTH_RADIUS_M + 400_000.0;
        let v = (MU_EARTH / r).sqrt();
        let epoch = Instant::from_datetime(2026, 1, 29, 12, 0, 0.0).unwrap();

        let initial = SpacecraftState::with_defaults(OrbitalState::new(
            Vector3::new(r, 0.0, 0.0),
            Vector3::new(0.0, v, 0.0),
            epoch,
        ));

        // Propagate one orbit
        let period = initial.orbital.period().unwrap();
        let result = propagator
            .propagate_duration(initial.clone(), Duration::from_seconds(period))
            .unwrap();

        // After one orbit, should return to approximately the same position
        let final_pos = result.final_state.orbital.position;
        let initial_pos = initial.orbital.position;

        let pos_error = (final_pos - initial_pos).norm() / initial_pos.norm();
        assert!(pos_error < 1e-4, "Position error too large: {}", pos_error);

        assert!(result.success);
        assert!(result.steps_taken > 0);
    }

    #[test]
    fn test_propagate_with_j2() {
        // Set up propagator with J2 gravity
        let mut forces = CompositeForce::new();
        forces.add(Box::new(EarthGravity::j2_only()));

        let integrator = NativeRK4::new();
        let propagator = HiFiPropagator::new(integrator, forces);

        // Inclined orbit at 400 km
        let r = EARTH_RADIUS_M + 400_000.0;
        let v = (MU_EARTH / r).sqrt();
        let epoch = Instant::from_datetime(2026, 1, 29, 12, 0, 0.0).unwrap();

        // 45-degree inclination
        let initial = SpacecraftState::with_defaults(OrbitalState::new(
            Vector3::new(r, 0.0, 0.0),
            Vector3::new(0.0, v * 0.707, v * 0.707),
            epoch,
        ));

        // Propagate 10 minutes
        let result = propagator
            .propagate_duration(initial, Duration::from_seconds(600.0))
            .unwrap();

        assert!(result.success);

        // J2 should cause nodal precession (RAAN drift)
        // For a 400 km, 45° inclined orbit, nodal precession is ~6°/day
        // In 10 minutes, this is ~0.04°, which is small but measurable
    }

    #[test]
    fn test_reentry_detection() {
        let mut forces = CompositeForce::new();
        forces.add(Box::new(EarthGravity::point_mass()));

        let integrator = NativeRK4::new();
        let propagator = HiFiPropagator::new(integrator, forces);

        // Very low orbit that will reenter
        let r = EARTH_RADIUS_M + 50_000.0; // 50 km - below reentry threshold
        let v = (MU_EARTH / r).sqrt();
        let epoch = Instant::from_datetime(2026, 1, 29, 12, 0, 0.0).unwrap();

        let initial = SpacecraftState::with_defaults(OrbitalState::new(
            Vector3::new(r, 0.0, 0.0),
            Vector3::new(0.0, v, 0.0),
            epoch,
        ));

        let result = propagator.propagate_duration(initial, Duration::from_seconds(60.0));

        // Should immediately trigger reentry
        assert!(matches!(result, Err(PropagationError::Reentry { .. })));
    }
}
