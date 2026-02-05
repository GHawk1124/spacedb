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
mod lookup;
mod nrlmsise00;

pub use exponential::Exponential;
pub use harris_priester::HarrisPriester;
pub use jb2008::Jb2008;
pub use lookup::{LookupAccuracy, LookupAtmosphere, LookupConfig};
pub use nrlmsise00::Nrlmsise00;

use crate::propagation::hifi::state::EARTH_RADIUS_M;
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
    /// NRLMSISE-00 with lookup-table acceleration
    Nrlmsise00Lookup,

    /// Jacchia-Bowman 2008 (placeholder)
    Jb2008,
    /// JB2008 with lookup-table acceleration
    Jb2008Lookup,

    /// Harris-Priester analytical model (placeholder)
    HarrisPriester,
    /// Harris-Priester with lookup-table acceleration
    HarrisPriesterLookup,

    /// Simple exponential model (fastest)
    Exponential,
    /// Exponential model with lookup-table acceleration
    ExponentialLookup,
}

impl AtmosphereModelType {
    /// Get the model name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Nrlmsise00 => "NRLMSISE-00",
            Self::Nrlmsise00Lookup => "NRLMSISE-00 (Lookup)",
            Self::Jb2008 => "JB2008",
            Self::Jb2008Lookup => "JB2008 (Lookup)",
            Self::HarrisPriester => "Harris-Priester",
            Self::HarrisPriesterLookup => "Harris-Priester (Lookup)",
            Self::Exponential => "Exponential",
            Self::ExponentialLookup => "Exponential (Lookup)",
        }
    }

    pub fn is_lookup(&self) -> bool {
        matches!(
            self,
            Self::Nrlmsise00Lookup
                | Self::Jb2008Lookup
                | Self::HarrisPriesterLookup
                | Self::ExponentialLookup
        )
    }

    /// Create a boxed model instance
    pub fn create(&self) -> Box<dyn AtmosphereModel> {
        self.create_with_accuracy(LookupAccuracy::Medium)
    }

    /// Create a boxed model instance with a lookup-table accuracy (if applicable)
    pub fn create_with_accuracy(&self, accuracy: LookupAccuracy) -> Box<dyn AtmosphereModel> {
        match self {
            Self::Nrlmsise00 => Box::new(Nrlmsise00::new()),
            Self::Nrlmsise00Lookup => Box::new(LookupAtmosphere::new(
                Box::new(Nrlmsise00::new()),
                LookupConfig::for_accuracy(accuracy),
            )),
            Self::Jb2008 => Box::new(Jb2008),
            Self::Jb2008Lookup => Box::new(LookupAtmosphere::new(
                Box::new(Jb2008),
                LookupConfig::for_accuracy(accuracy),
            )),
            Self::HarrisPriester => Box::new(HarrisPriester::new()),
            Self::HarrisPriesterLookup => Box::new(LookupAtmosphere::new(
                Box::new(HarrisPriester::new()),
                LookupConfig::for_accuracy(accuracy),
            )),
            Self::Exponential => Box::new(Exponential::default()),
            Self::ExponentialLookup => Box::new(LookupAtmosphere::new(
                Box::new(Exponential::default()),
                LookupConfig::for_accuracy(accuracy),
            )),
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

/// Convert GCRF position to geodetic coordinates (lat, lon, altitude)
pub(crate) fn gcrf_to_geodetic(position: &Vector3<f64>, epoch: &Instant) -> (f64, f64, f64) {
    let q_gcrf_to_itrf = satkit::frametransform::qgcrf2itrf(epoch);
    let pos_itrf = q_gcrf_to_itrf * position;

    let coord = satkit::itrfcoord::ITRFCoord::from_slice(&[pos_itrf.x, pos_itrf.y, pos_itrf.z])
        .unwrap_or_else(|_| {
            let r = pos_itrf.norm();
            satkit::itrfcoord::ITRFCoord::from_geodetic_deg(
                (pos_itrf.z / r).asin().to_degrees(),
                pos_itrf.y.atan2(pos_itrf.x).to_degrees(),
                r - EARTH_RADIUS_M,
            )
        });

    let mut lon = coord.longitude_deg();
    if lon > 180.0 {
        lon -= 360.0;
    } else if lon < -180.0 {
        lon += 360.0;
    }

    (coord.latitude_deg(), lon, coord.hae())
}
