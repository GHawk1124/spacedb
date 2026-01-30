//! JB2008 atmospheric density model (placeholder)
//!
//! Jacchia-Bowman 2008 is a more recent empirical atmosphere model that
//! provides improved accuracy in the thermosphere compared to NRLMSISE-00.
//!
//! # Status
//!
//! This is a placeholder implementation. When invoked, it will fall back
//! to NRLMSISE-00 with a warning.

use super::{AtmosphereDensity, AtmosphereModel, Nrlmsise00};
use nalgebra::Vector3;
use satkit::Instant;

/// JB2008 atmospheric model (placeholder)
///
/// # Note
///
/// This model is not yet implemented. It currently falls back to NRLMSISE-00.
/// A full implementation would require:
/// - JB2008 algorithm implementation
/// - Additional space weather inputs (S10.7, M10.7, Y10.7)
/// - Dst geomagnetic storm index
pub struct Jb2008;

impl AtmosphereModel for Jb2008 {
    fn density(&self, position: &Vector3<f64>, epoch: &Instant) -> AtmosphereDensity {
        log::warn!("JB2008 not implemented, falling back to NRLMSISE-00");
        let fallback = Nrlmsise00::new();
        fallback.density(position, epoch)
    }

    fn name(&self) -> &'static str {
        "JB2008 (placeholder)"
    }

    fn description(&self) -> &'static str {
        "Jacchia-Bowman 2008 - NOT YET IMPLEMENTED"
    }

    fn requires_space_weather(&self) -> bool {
        true
    }
}
