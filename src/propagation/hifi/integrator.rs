//! Numerical integrators for orbit propagation
//!
//! This module provides a trait-based abstraction for numerical integration,
//! designed to allow hot-swapping between different implementations including
//! a future CUDA-based integrator.
//!
//! # Available Integrators
//!
//!
//! # Available Integrators
//!
//!
//! # Available Integrators
//!
//! - **NativeRK4**: Native Rust implementation of Runge-Kutta 4 with adaptive stepping (default)
//! - **SatkitRK98**: Wrapper around satkit's high-precision propagator (RK 9(8))
//!
//! # Future Integrators
//!
//! - **CudaVernerRK98**: GPU-accelerated Verner RK 9(8) for batch propagation

use crate::propagation::hifi::state::OrbitalState;
use nalgebra::Vector3;
use satkit::Duration;

/// Result of a single integration step
#[derive(Debug, Clone)]
pub struct StepResult {
    /// New state after the step
    pub state: OrbitalState,

    /// Actual step size used (for adaptive methods)
    pub dt_used: f64,

    /// Estimated local truncation error (if available)
    pub error_estimate: Option<f64>,

    /// Whether step was successful
    pub success: bool,
}

/// Trait for numerical integrators
///
/// This trait abstracts the numerical integration method, allowing different
/// implementations to be swapped at runtime. The trait is designed to be
/// compatible with future CUDA implementations.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` to allow parallel propagation
/// of multiple satellites.
pub trait Integrator: Send + Sync {
    /// Take a single integration step
    fn step(
        &self,
        state: &OrbitalState,
        dt: f64,
        derivatives: &dyn Fn(&OrbitalState) -> (Vector3<f64>, Vector3<f64>),
    ) -> OrbitalState;

    /// Take an adaptive step with error control
    fn adaptive_step(
        &self,
        state: &OrbitalState,
        dt_suggested: f64,
        tolerance: f64,
        derivatives: &dyn Fn(&OrbitalState) -> (Vector3<f64>, Vector3<f64>),
    ) -> StepResult;

    /// Integrator name
    fn name(&self) -> &'static str;

    /// Integrator order (for error estimation)
    fn order(&self) -> u8;

    /// Number of function evaluations per step
    fn stages(&self) -> usize;
}

/// Runge-Kutta 4 integrator with adaptive stepping via step doubling
///
/// This is a classic RK4 integrator with embedded error estimation
/// via step doubling for adaptive step size control.
pub struct NativeRK4 {
    /// Minimum allowed step size (seconds)
    pub min_step: f64,

    /// Maximum allowed step size (seconds)
    pub max_step: f64,

    /// Safety factor for step size adjustment
    pub safety: f64,

    /// Maximum step growth factor
    pub max_growth: f64,

    /// Maximum step shrink factor
    pub max_shrink: f64,

    /// Fixed-step tolerance for non-adaptive calls
    pub fixed_tolerance: f64,
}

impl Default for NativeRK4 {
    fn default() -> Self {
        Self::new()
    }
}

impl NativeRK4 {
    /// Create with default settings
    pub fn new() -> Self {
        Self {
            min_step: 0.1,   // 0.1 seconds
            max_step: 300.0, // 5 minutes
            safety: 0.9,     // 90% of optimal step
            max_growth: 5.0, // Max 5x step increase
            max_shrink: 0.2, // Min 1/5 step decrease
            fixed_tolerance: 1e-10,
        }
    }

    /// Create with custom settings
    pub fn with_settings(min_step: f64, max_step: f64) -> Self {
        Self {
            min_step,
            max_step,
            ..Self::default()
        }
    }

    /// Create for high-precision work
    pub fn high_precision() -> Self {
        Self {
            min_step: 0.01, // 10 ms
            max_step: 60.0, // 1 minute
            safety: 0.95,
            max_growth: 2.0,
            max_shrink: 0.1,
            fixed_tolerance: 1e-12,
        }
    }

    /// Create for fast propagation (lower accuracy)
    pub fn fast() -> Self {
        Self {
            min_step: 1.0,   // 1 second
            max_step: 600.0, // 10 minutes
            safety: 0.8,
            max_growth: 10.0,
            max_shrink: 0.5,
            fixed_tolerance: 1e-8,
        }
    }

    /// RK4 step
    fn rk4_step(
        &self,
        state: &OrbitalState,
        dt: f64,
        derivatives: &dyn Fn(&OrbitalState) -> (Vector3<f64>, Vector3<f64>),
    ) -> OrbitalState {
        let (v1, a1) = derivatives(state);

        let s2 = OrbitalState::new(
            state.position + v1 * (dt / 2.0),
            state.velocity + a1 * (dt / 2.0),
            state.epoch + Duration::from_seconds(dt / 2.0),
        );
        let (v2, a2) = derivatives(&s2);

        let s3 = OrbitalState::new(
            state.position + v2 * (dt / 2.0),
            state.velocity + a2 * (dt / 2.0),
            state.epoch + Duration::from_seconds(dt / 2.0),
        );
        let (v3, a3) = derivatives(&s3);

        let s4 = OrbitalState::new(
            state.position + v3 * dt,
            state.velocity + a3 * dt,
            state.epoch + Duration::from_seconds(dt),
        );
        let (v4, a4) = derivatives(&s4);

        let new_pos = state.position + (v1 + 2.0 * v2 + 2.0 * v3 + v4) * (dt / 6.0);
        let new_vel = state.velocity + (a1 + 2.0 * a2 + 2.0 * a3 + a4) * (dt / 6.0);
        let new_epoch = state.epoch + Duration::from_seconds(dt);

        OrbitalState::new(new_pos, new_vel, new_epoch)
    }
}

impl Integrator for NativeRK4 {
    fn step(
        &self,
        state: &OrbitalState,
        dt: f64,
        derivatives: &dyn Fn(&OrbitalState) -> (Vector3<f64>, Vector3<f64>),
    ) -> OrbitalState {
        self.rk4_step(state, dt, derivatives)
    }

    fn adaptive_step(
        &self,
        state: &OrbitalState,
        dt_suggested: f64,
        tolerance: f64,
        derivatives: &dyn Fn(&OrbitalState) -> (Vector3<f64>, Vector3<f64>),
    ) -> StepResult {
        // Clamp initial step size
        let mut h = dt_suggested.clamp(self.min_step, self.max_step);

        loop {
            // Take one full step
            let y_full = self.rk4_step(state, h, derivatives);

            // Take two half steps
            let y_half1 = self.rk4_step(state, h / 2.0, derivatives);
            let y_half2 = self.rk4_step(&y_half1, h / 2.0, derivatives);

            // Estimate error (difference between methods)
            let error_pos = (y_full.position - y_half2.position).norm();
            let error_vel = (y_full.velocity - y_half2.velocity).norm();
            let error = error_pos.max(error_vel);

            if error < tolerance || h <= self.min_step {
                // Accept step, use Richardson extrapolation for better accuracy
                let pos = (16.0 * y_half2.position - y_full.position) / 15.0;
                let vel = (16.0 * y_half2.velocity - y_full.velocity) / 15.0;

                return StepResult {
                    state: OrbitalState::new(pos, vel, y_half2.epoch),
                    dt_used: h,
                    error_estimate: Some(error),
                    success: true,
                };
            }

            // Reduce step size
            let factor = self.safety * (tolerance / error).powf(0.2);
            h = (h * factor.clamp(self.max_shrink, 1.0)).max(self.min_step);
        }
    }

    fn name(&self) -> &'static str {
        "Native RK4 (adaptive)"
    }

    fn order(&self) -> u8 {
        4
    }

    fn stages(&self) -> usize {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::propagation::hifi::state::{EARTH_RADIUS_M, MU_EARTH};
    use satkit::Instant;

    #[test]
    fn test_rk_circular_orbit() {
        let integrator = NativeRK4::new();

        // Circular orbit at 400 km
        let r = EARTH_RADIUS_M + 400_000.0;
        let v = (MU_EARTH / r).sqrt();
        let epoch = Instant::from_datetime(2026, 1, 29, 12, 0, 0.0).unwrap();

        let state = OrbitalState::new(Vector3::new(r, 0.0, 0.0), Vector3::new(0.0, v, 0.0), epoch);

        // Simple derivatives: point mass gravity
        let derivatives = |s: &OrbitalState| {
            let r = s.position.norm();
            let accel = -MU_EARTH / (r * r * r) * s.position;
            (s.velocity, accel)
        };

        // Take one step
        let new_state = integrator.step(&state, 60.0, &derivatives);

        // Radius should be approximately preserved (circular orbit)
        let new_r = new_state.position.norm();
        assert!((new_r - r).abs() / r < 1e-6);

        // Speed should be approximately preserved
        let new_v = new_state.velocity.norm();
        assert!((new_v - v).abs() / v < 1e-6);
    }
}

use satkit::orbitprop::{PropSettings, SatPropertiesStatic};

/// Wrapper around satkit's RK9(8) propagator
///
/// This implementation ignores the force model provided via the `derivatives`
/// closure and instead relies on `satkit`'s internal force models, configured
/// to match the `HiFiSettings` as closely as possible.
pub struct SatkitRK98 {
    settings: PropSettings,
    satprops: Option<SatPropertiesStatic>,
}

impl SatkitRK98 {
    pub fn new(settings: PropSettings, satprops: Option<SatPropertiesStatic>) -> Self {
        Self { settings, satprops }
    }
}

impl Integrator for SatkitRK98 {
    fn step(
        &self,
        state: &OrbitalState,
        dt: f64,
        _derivatives: &dyn Fn(&OrbitalState) -> (Vector3<f64>, Vector3<f64>),
    ) -> OrbitalState {
        // Simple step falls back to adaptive step
        let res = self.adaptive_step(state, dt, 1e-9, _derivatives);
        res.state
    }

    fn adaptive_step(
        &self,
        state: &OrbitalState,
        dt_suggested: f64,
        _tolerance: f64, // Ignored, using settings tolerance
        _derivatives: &dyn Fn(&OrbitalState) -> (Vector3<f64>, Vector3<f64>),
    ) -> StepResult {
        // Convert input state to satkit SimpleState
        let mut simple_state = satkit::orbitprop::SimpleState::zeros();
        // Position (meters)
        simple_state[0] = state.position[0];
        simple_state[1] = state.position[1];
        simple_state[2] = state.position[2];
        // Velocity (m/s)
        simple_state[3] = state.velocity[0];
        simple_state[4] = state.velocity[1];
        simple_state[5] = state.velocity[2];

        let start_time = state.epoch;
        let end_time = start_time + satkit::Duration::from_seconds(dt_suggested);

        // Propagate
        let props_ref = self
            .satprops
            .as_ref()
            .map(|p| p as &dyn satkit::orbitprop::SatProperties);

        match satkit::orbitprop::propagate(
            &simple_state,
            &start_time,
            &end_time,
            &self.settings,
            props_ref,
        ) {
            Ok(result) => {
                let final_state_vec = result.state_end;
                let new_pos =
                    Vector3::new(final_state_vec[0], final_state_vec[1], final_state_vec[2]);
                let new_vel =
                    Vector3::new(final_state_vec[3], final_state_vec[4], final_state_vec[5]);

                StepResult {
                    state: OrbitalState::new(new_pos, new_vel, result.time_end),
                    dt_used: (result.time_end - start_time).as_seconds(),
                    error_estimate: None, // satkit internal error control
                    success: true,
                }
            }
            Err(e) => {
                log::error!("Satkit propagation failed: {}", e);
                // Return original state on failure (step size 0)
                StepResult {
                    state: state.clone(),
                    dt_used: 0.0,
                    error_estimate: None,
                    success: false,
                }
            }
        }
    }

    fn name(&self) -> &'static str {
        "Satkit RK9(8) (native)"
    }

    fn order(&self) -> u8 {
        8
    }

    fn stages(&self) -> usize {
        16
    }
}
