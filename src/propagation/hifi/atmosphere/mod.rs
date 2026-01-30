//! Atmospheric density models for drag calculations
//!
//! This module provides a trait-based abstraction for different atmosphere models,
//! allowing hot-swappable implementations for different accuracy/performance tradeoffs.
//!
//! # Implemented Models
//!
//! - **NRLMSISE-00**: Standard empirical model using space weather data (via satkit)
//!
//! # Placeholder Models (to be implemented)
//!
//! - **JB2008**: Jacchia-Bowman 2008, more accurate for thermosphere
//! - **Harris-Priester**: Simple analytical model, computationally efficient
//! - **Exponential**: Basic exponential decay, fastest but least accurate

mod exponential;
mod harris_priester;
mod jb2008;
mod nrlmsise00;

pub use exponential::Exponential;
pub use harris_priester::HarrisPriester;
pub use jb2008::Jb2008;
pub use nrlmsise00::Nrlmsise00;

use nalgebra::Vector3;
use satkit::Instant;

/// Output from an atmosphere model
#[derive(Debug, Clone, Copy)]
pub struct AtmosphereDensity {
    /// Total atmospheric mass density in kg/m³
    pub rho: f64,

    /// Exospheric temperature in Kelvin (if available)
    pub temperature: Option<f64>,

    /// Mean molecular weight in g/mol (if available)
    pub molecular_weight: Option<f64>,
}

impl AtmosphereDensity {
    /// Create a density-only result
    pub fn new(rho: f64) -> Self {
        Self {
            rho,
            temperature: None,
            molecular_weight: None,
        }
    }

    /// Create with full output
    pub fn full(rho: f64, temperature: f64, molecular_weight: f64) -> Self {
        Self {
            rho,
            temperature: Some(temperature),
            molecular_weight: Some(molecular_weight),
        }
    }

    /// Zero density (for altitudes above atmosphere)
    pub fn zero() -> Self {
        Self {
            rho: 0.0,
            temperature: None,
            molecular_weight: None,
        }
    }
}

/// Trait for atmospheric density models
///
/// Implementations must be thread-safe (Send + Sync) to allow
/// parallel propagation of multiple satellites.
pub trait AtmosphereModel: Send + Sync {
    /// Compute atmospheric density at given position and time
    ///
    /// # Arguments
    ///
    /// * `position` - Position in GCRF frame, meters
    /// * `epoch` - Time of density calculation
    ///
    /// # Returns
    ///
    /// Atmospheric density and optional auxiliary data
    fn density(&self, position: &Vector3<f64>, epoch: &Instant) -> AtmosphereDensity;

    /// Model name for logging and display
    fn name(&self) -> &'static str;

    /// Brief description of the model
    fn description(&self) -> &'static str {
        "Atmospheric density model"
    }

    /// Whether this model requires space weather data
    fn requires_space_weather(&self) -> bool {
        false
    }
}

/// Enum for runtime model selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtmosphereModelType {
    /// NRLMSISE-00 empirical model (default, most accurate)
    Nrlmsise00,

    /// Jacchia-Bowman 2008 (placeholder)
    Jb2008,

    /// Harris-Priester analytical model (placeholder)
    HarrisPriester,

    /// Simple exponential model (fastest)
    Exponential,
}

impl AtmosphereModelType {
    /// Get the model name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Nrlmsise00 => "NRLMSISE-00",
            Self::Jb2008 => "JB2008",
            Self::HarrisPriester => "Harris-Priester",
            Self::Exponential => "Exponential",
        }
    }

    /// Create a boxed model instance
    pub fn create(&self) -> Box<dyn AtmosphereModel> {
        match self {
            Self::Nrlmsise00 => Box::new(Nrlmsise00::new()),
            Self::Jb2008 => Box::new(Jb2008),
            Self::HarrisPriester => Box::new(HarrisPriester::new()),
            Self::Exponential => Box::new(Exponential::default()),
        }
    }
}

impl Default for AtmosphereModelType {
    fn default() -> Self {
        Self::Nrlmsise00
    }
}

impl AtmosphereModel for Box<dyn AtmosphereModel> {
    fn density(&self, position: &Vector3<f64>, epoch: &Instant) -> AtmosphereDensity {
        self.as_ref().density(position, epoch)
    }

    fn name(&self) -> &'static str {
        self.as_ref().name()
    }

    fn description(&self) -> &'static str {
        self.as_ref().description()
    }

    fn requires_space_weather(&self) -> bool {
        self.as_ref().requires_space_weather()
    }
}
