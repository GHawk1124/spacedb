//! Orbital and spacecraft state representations
//!
//! Provides the core state vectors used for numerical integration.

use nalgebra::Vector3;
use satkit::Instant;

/// Full orbital state vector for numerical integration
///
/// Position and velocity are in the Geocentric Celestial Reference Frame (GCRF),
/// which is an Earth-centered inertial frame suitable for orbit propagation.
#[derive(Debug, Clone)]
pub struct OrbitalState {
    /// Position in GCRF frame (meters)
    pub position: Vector3<f64>,

    /// Velocity in GCRF frame (m/s)
    pub velocity: Vector3<f64>,

    /// Epoch (time) of this state
    pub epoch: Instant,
}

impl OrbitalState {
    /// Create a new orbital state
    pub fn new(position: Vector3<f64>, velocity: Vector3<f64>, epoch: Instant) -> Self {
        Self {
            position,
            velocity,
            epoch,
        }
    }

    /// Create from position/velocity in kilometers and km/s
    pub fn from_km(pos_km: Vector3<f64>, vel_km_s: Vector3<f64>, epoch: Instant) -> Self {
        Self {
            position: pos_km * 1000.0,
            velocity: vel_km_s * 1000.0,
            epoch,
        }
    }

    /// Get position in kilometers
    pub fn position_km(&self) -> Vector3<f64> {
        self.position / 1000.0
    }

    /// Get velocity in km/s
    pub fn velocity_km_s(&self) -> Vector3<f64> {
        self.velocity / 1000.0
    }

    /// Compute orbital radius (distance from Earth center) in meters
    pub fn radius(&self) -> f64 {
        self.position.norm()
    }

    /// Compute altitude above Earth surface in meters
    pub fn altitude(&self) -> f64 {
        self.radius() - EARTH_RADIUS_M
    }

    /// Compute altitude above Earth surface in kilometers
    pub fn altitude_km(&self) -> f64 {
        self.altitude() / 1000.0
    }

    /// Compute orbital speed in m/s
    pub fn speed(&self) -> f64 {
        self.velocity.norm()
    }

    /// Compute specific orbital energy (vis-viva) in J/kg
    pub fn specific_energy(&self) -> f64 {
        let v2 = self.velocity.norm_squared();
        let r = self.position.norm();
        0.5 * v2 - MU_EARTH / r
    }

    /// Compute semi-major axis in meters (negative for hyperbolic)
    pub fn semi_major_axis(&self) -> f64 {
        -MU_EARTH / (2.0 * self.specific_energy())
    }

    /// Compute orbital period in seconds (only valid for elliptical orbits)
    pub fn period(&self) -> Option<f64> {
        let a = self.semi_major_axis();
        if a > 0.0 {
            Some(2.0 * std::f64::consts::PI * (a.powi(3) / MU_EARTH).sqrt())
        } else {
            None // Hyperbolic or parabolic
        }
    }
}

/// Extended state with physical properties for drag and SRP calculations
#[derive(Debug, Clone)]
pub struct SpacecraftState {
    /// Orbital state (position, velocity, epoch)
    pub orbital: OrbitalState,

    /// Spacecraft dry mass in kilograms
    pub mass_kg: f64,

    /// Drag coefficient times cross-sectional area (Cd * A) in m²
    ///
    /// Typical values:
    /// - Cd ≈ 2.0-2.5 for most satellites
    /// - A depends on spacecraft geometry and attitude
    pub cd_area_m2: f64,

    /// Reflectivity coefficient times area (Cr * A) in m² for solar radiation pressure
    ///
    /// Typical values:
    /// - Cr = 1.0 for perfect absorber
    /// - Cr = 2.0 for perfect reflector
    /// - Cr ≈ 1.2-1.5 for typical satellites
    pub cr_area_m2: f64,
}

impl SpacecraftState {
    /// Create a new spacecraft state
    pub fn new(orbital: OrbitalState, mass_kg: f64, cd_area_m2: f64, cr_area_m2: f64) -> Self {
        Self {
            orbital,
            mass_kg,
            cd_area_m2,
            cr_area_m2,
        }
    }

    /// Create a spacecraft state with default physical properties
    ///
    /// Uses typical values for a small satellite:
    /// - Mass: 100 kg
    /// - Cd*A: 2.2 * 1.0 m² = 2.2 m²
    /// - Cr*A: 1.5 * 1.0 m² = 1.5 m²
    pub fn with_defaults(orbital: OrbitalState) -> Self {
        Self {
            orbital,
            mass_kg: 100.0,
            cd_area_m2: 2.2,
            cr_area_m2: 1.5,
        }
    }

    /// Create from DISCOS physical data
    ///
    /// Uses cross-sectional area from DISCOS and assumes:
    /// - Cd = 2.2 (typical drag coefficient)
    /// - Cr = 1.5 (typical reflectivity)
    pub fn from_discos(
        orbital: OrbitalState,
        mass_kg: Option<f64>,
        cross_section_m2: Option<f64>,
    ) -> Self {
        let mass = mass_kg.unwrap_or(100.0);
        let area = cross_section_m2.unwrap_or(1.0);

        Self {
            orbital,
            mass_kg: mass,
            cd_area_m2: 2.2 * area,
            cr_area_m2: 1.5 * area,
        }
    }

    /// Get ballistic coefficient (m / (Cd * A)) in kg/m²
    ///
    /// Lower values mean more drag, faster decay.
    /// Typical range: 10-200 kg/m²
    pub fn ballistic_coefficient(&self) -> f64 {
        if self.cd_area_m2 > 0.0 {
            self.mass_kg / self.cd_area_m2
        } else {
            f64::INFINITY
        }
    }

    /// Get area-to-mass ratio for SRP (Cr * A / m) in m²/kg
    pub fn srp_area_to_mass(&self) -> f64 {
        if self.mass_kg > 0.0 {
            self.cr_area_m2 / self.mass_kg
        } else {
            0.0
        }
    }
}

// Physical constants
/// Earth's gravitational parameter (GM) in m³/s²
pub const MU_EARTH: f64 = 3.986004418e14;

/// Earth's mean equatorial radius in meters
pub const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// Earth's rotation rate in rad/s
pub const OMEGA_EARTH: f64 = 7.2921150e-5;

/// Speed of light in m/s
pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

/// Solar radiation pressure at 1 AU in N/m² (W/m² / c)
pub const SOLAR_PRESSURE_1AU: f64 = 4.56e-6;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orbital_state_iss() {
        // Approximate ISS orbit: 420 km altitude, ~7.66 km/s
        let epoch = Instant::from_datetime(2026, 1, 29, 12, 0, 0.0).unwrap();
        let r = EARTH_RADIUS_M + 420_000.0; // 420 km altitude
        let v = (MU_EARTH / r).sqrt(); // Circular orbit velocity

        let state = OrbitalState::new(Vector3::new(r, 0.0, 0.0), Vector3::new(0.0, v, 0.0), epoch);

        assert!((state.altitude_km() - 420.0).abs() < 1.0);
        assert!((state.speed() / 1000.0 - 7.66).abs() < 0.1);

        let period = state.period().unwrap();
        assert!((period / 60.0 - 92.0).abs() < 2.0); // ~92 minutes
    }
}
