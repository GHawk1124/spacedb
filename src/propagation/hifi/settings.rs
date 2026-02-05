//! Configuration helpers for high-fidelity propagation

use super::atmosphere::{AtmosphereModelType, LookupAccuracy};
use super::forces::{
    AtmosphericDrag, CompositeForce, EarthGravity, EphemerisType, GravityModel,
    SolarRadiationPressure, ThirdBody,
};
use super::hifi_propagator::PropagatorConfig;
use super::integrator::{Integrator, NativeRK4, SatkitRK98};
use super::HiFiPropagator;

/// Propagator type selection for real-time and high-fidelity propagation
///
/// This enum unifies the selection between fast SGP4 propagation (for real-time
/// visualization) and high-fidelity numerical integrators (for accurate predictions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropagatorType {
    /// SGP4 analytic propagator (fast, TLE-based)
    /// Best for real-time visualization of many satellites
    Sgp4,
    /// Native RK4 integrator with adaptive stepping
    /// Good balance of speed and accuracy with configurable force models
    NativeRk4,
    /// Satkit RK 9(8) high-precision integrator
    /// Best accuracy, uses satkit's internal force models
    SatkitRk98,
}

impl PropagatorType {
    /// Display name for the propagator
    pub fn name(&self) -> &'static str {
        match self {
            Self::Sgp4 => "SGP4 (Fast)",
            Self::NativeRk4 => "Native RK4 (Adaptive)",
            Self::SatkitRk98 => "Satkit RK 9(8) (High Precision)",
        }
    }

    /// Short description of when to use this propagator
    pub fn description(&self) -> &'static str {
        match self {
            Self::Sgp4 => "Fast analytic propagation using TLEs. Best for real-time visualization.",
            Self::NativeRk4 => "Numerical integration with configurable force models. Good for accurate predictions with custom physics.",
            Self::SatkitRk98 => "High-precision RK 9(8) integrator. Best accuracy for orbital decay prediction.",
        }
    }

    /// All available propagator types
    pub fn all() -> &'static [PropagatorType] {
        &[
            PropagatorType::Sgp4,
            PropagatorType::NativeRk4,
            PropagatorType::SatkitRk98,
        ]
    }

    /// Check if this is a high-fidelity propagator (requires numerical integration)
    pub fn is_high_fidelity(&self) -> bool {
        matches!(self, Self::NativeRk4 | Self::SatkitRk98)
    }

    /// Check if this is SGP4 (fast analytic)
    pub fn is_sgp4(&self) -> bool {
        matches!(self, Self::Sgp4)
    }

    /// Create an integrator if this is a high-fidelity type
    /// Returns None for SGP4
    pub fn create_integrator(&self, settings: &HiFiSettings) -> Option<Box<dyn Integrator>> {
        match self {
            Self::Sgp4 => None,
            Self::NativeRk4 => Some(Box::new(NativeRK4::new())),
            Self::SatkitRk98 => {
                let mut props = satkit::orbitprop::PropSettings::default();
                // Map settings
                props.abs_error = settings.config.tolerance;
                props.rel_error = settings.config.tolerance * 1e-3;

                // Map gravity model
                match settings.gravity {
                    GravityModelChoice::PointMass => props.gravity_order = 0,
                    GravityModelChoice::J2 => props.gravity_order = 2,
                    GravityModelChoice::FullField20 => props.gravity_order = 20,
                }

                props.use_spaceweather = false; // Default off for now

                // Spacecraft properties
                let cd_val = 2.2 * 1.0 / 1000.0; // Cd * A / m
                let cr_val = if settings.include_srp {
                    1.8 * 1.0 / 1000.0 // Cr * A / m
                } else {
                    0.0
                };

                let satprops = Some(satkit::orbitprop::SatPropertiesStatic::new(cd_val, cr_val));

                Some(Box::new(SatkitRK98::new(props, satprops)))
            }
        }
    }
}

/// Legacy integrator type - kept for backward compatibility
/// Use `PropagatorType` for new code
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegratorType {
    /// Native RK4 with adaptive stepping
    NativeRk4,
    /// Satkit RK 9(8)
    SatkitRk98,
}

impl IntegratorType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::NativeRk4 => "Native RK4 (adaptive)",
            Self::SatkitRk98 => "Satkit RK 9(8) (native)",
        }
    }

    pub fn all() -> &'static [IntegratorType] {
        &[IntegratorType::NativeRk4, IntegratorType::SatkitRk98]
    }

    pub fn create(&self, settings: &HiFiSettings) -> Box<dyn Integrator> {
        match self {
            Self::NativeRk4 => Box::new(NativeRK4::new()),
            Self::SatkitRk98 => {
                let mut props = satkit::orbitprop::PropSettings::default();
                // Map settings
                props.abs_error = 1e-9;
                props.rel_error = 1e-12;
                props.gravity_order = 4;

                // Map gravity model
                match settings.gravity {
                    GravityModelChoice::PointMass => props.gravity_order = 0,
                    GravityModelChoice::J2 => props.gravity_order = 2,
                    GravityModelChoice::FullField20 => props.gravity_order = 20,
                }

                props.use_spaceweather = false;

                // Spacecraft properties
                let cd_val = 2.2 * 1.0 / 1000.0;
                let cr_val = if settings.include_srp {
                    1.8 * 1.0 / 1000.0
                } else {
                    0.0
                };

                let satprops = Some(satkit::orbitprop::SatPropertiesStatic::new(cd_val, cr_val));

                Box::new(SatkitRK98::new(props, satprops))
            }
        }
    }
}

impl From<PropagatorType> for IntegratorType {
    fn from(prop_type: PropagatorType) -> Self {
        match prop_type {
            PropagatorType::Sgp4 => IntegratorType::NativeRk4, // Default fallback
            PropagatorType::NativeRk4 => IntegratorType::NativeRk4,
            PropagatorType::SatkitRk98 => IntegratorType::SatkitRk98,
        }
    }
}

/// Runtime-selectable gravity configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GravityModelChoice {
    PointMass,
    J2,
    FullField20,
}

impl GravityModelChoice {
    pub fn name(&self) -> &'static str {
        match self {
            Self::PointMass => "Point Mass",
            Self::J2 => "J2",
            Self::FullField20 => "Full Field (20x20)",
        }
    }

    pub fn all() -> &'static [GravityModelChoice] {
        &[
            GravityModelChoice::PointMass,
            GravityModelChoice::J2,
            GravityModelChoice::FullField20,
        ]
    }

    pub fn to_model(&self) -> GravityModel {
        match self {
            Self::PointMass => GravityModel::PointMass,
            Self::J2 => GravityModel::J2Only,
            Self::FullField20 => GravityModel::FullField {
                degree: 20,
                order: 20,
            },
        }
    }
}

/// High-fidelity propagation settings (swappable models)
#[derive(Debug, Clone)]
pub struct HiFiSettings {
    /// Selected propagator type (unified selection)
    pub propagator: PropagatorType,
    /// Legacy integrator field - kept for backward compatibility
    /// Prefer using `propagator` field for new code
    pub integrator: IntegratorType,
    pub atmosphere: AtmosphereModelType,
    pub lookup_accuracy: LookupAccuracy,
    pub gravity: GravityModelChoice,
    pub include_srp: bool,
    pub include_third_body: bool,
    /// Ephemeris type for third-body calculations
    pub ephemeris: EphemerisType,
    pub config: PropagatorConfig,
    pub decay_horizon_days: f64,
}

impl Default for HiFiSettings {
    fn default() -> Self {
        Self {
            propagator: PropagatorType::SatkitRk98,
            integrator: IntegratorType::SatkitRk98,
            atmosphere: AtmosphereModelType::Nrlmsise00,
            lookup_accuracy: LookupAccuracy::Medium,
            gravity: GravityModelChoice::FullField20,
            include_srp: true,
            include_third_body: true,
            ephemeris: EphemerisType::LowPrecision,
            config: PropagatorConfig::high_precision(),
            decay_horizon_days: 3650.0,
        }
    }
}

impl HiFiSettings {
    pub fn build_forces(&self) -> CompositeForce {
        let mut forces = CompositeForce::new();
        forces.add(Box::new(EarthGravity::from_model(self.gravity.to_model())));
        forces.add(Box::new(AtmosphericDrag::new(
            self.atmosphere.create_with_accuracy(self.lookup_accuracy),
        )));

        if self.include_srp {
            forces.add(Box::new(SolarRadiationPressure::new()));
        }

        if self.include_third_body {
            forces.add(Box::new(ThirdBody::sun_and_moon_with_ephemeris(
                self.ephemeris,
            )));
        }

        forces
    }

    pub fn build_propagator(&self) -> Option<HiFiPropagator> {
        let integrator = self.propagator.create_integrator(self)?;
        let forces = self.build_forces();
        Some(HiFiPropagator::with_config(
            integrator,
            forces,
            self.config.clone(),
        ))
    }

    /// Sync the legacy integrator field with the new propagator field
    pub fn sync_integrator(&mut self) {
        if self.propagator.is_high_fidelity() {
            self.integrator = self.propagator.into();
        }
    }
}
