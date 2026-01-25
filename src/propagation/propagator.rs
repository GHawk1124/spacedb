//! SGP4 propagation using satkit

use crate::data::{SpaceObject, TleData};
use glam::Vec3;
use satkit::sgp4::sgp4;
use std::collections::HashMap;

/// Propagation result for a single satellite
#[derive(Debug, Clone, Copy)]
pub struct SatelliteState {
    /// Position in Earth-Centered Inertial frame (kilometers, normalized to Earth radii)
    pub position: Vec3,
    /// Velocity in ECI frame (km/s, optional for display)
    pub velocity: Vec3,
    /// Altitude above Earth surface (km)
    pub altitude_km: f64,
    /// Age of TLE in days (for error estimation)
    pub tle_age_days: f64,
}

/// Earth radius in kilometers
pub const EARTH_RADIUS_KM: f64 = 6371.0;

/// Manages SGP4 propagation for all satellites
pub struct Propagator {
    /// Cached TLE data indexed by NORAD ID
    tles: HashMap<u32, satkit::TLE>,
    /// Current simulation time
    current_time: satkit::Instant,
}

impl Propagator {
    pub fn new() -> Self {
        // Start at current UTC time
        let now = chrono::Utc::now();
        let current_time = satkit::Instant::from_datetime(
            now.year() as i32,
            now.month() as i32,
            now.day() as i32,
            now.hour() as i32,
            now.minute() as i32,
            now.second() as f64,
        )
        .unwrap_or_else(|_| satkit::Instant::from_datetime(2026, 1, 24, 12, 0, 0.0).unwrap());

        Self {
            tles: HashMap::new(),
            current_time,
        }
    }

    /// Load TLEs from space objects
    pub fn load_tles(&mut self, objects: &HashMap<String, SpaceObject>) {
        self.tles.clear();

        for (norad_str, obj) in objects {
            if let (Ok(norad), Some(tle_data)) = (norad_str.parse::<u32>(), &obj.tle) {
                if let Some(tle) = parse_tle(tle_data) {
                    self.tles.insert(norad, tle);
                }
            }
        }

        log::info!("Loaded {} TLEs for propagation", self.tles.len());
    }

    /// Get current simulation time
    pub fn current_time(&self) -> &satkit::Instant {
        &self.current_time
    }

    /// Advance time by delta seconds
    pub fn advance_time(&mut self, delta_seconds: f64) {
        self.current_time = self.current_time + satkit::Duration::from_seconds(delta_seconds);
    }

    /// Format current time as string
    pub fn format_time(&self) -> String {
        let (year, month, day, hour, min, sec) = self.current_time.as_datetime();
        format!(
            "{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC",
            year, month, day, hour, min, sec as u32
        )
    }

    /// Propagate a single satellite
    pub fn propagate(&self, norad_id: u32) -> Option<SatelliteState> {
        let tle = self.tles.get(&norad_id)?;

        // Propagate using SGP4
        let mut tle = tle.clone();
        match sgp4(&mut tle, &[self.current_time]) {
            Ok(result) => {
                // pos and vel are in TEME frame, in meters and m/s
                // TEME uses Z-up (polar axis), but rendering uses Y-up
                // Convert while preserving right-handedness:
                // TEME X -> Render X, TEME Z -> Render Y, TEME Y -> Render -Z (negated!)
                let pos = result.pos.column(0);
                let vel = result.vel.column(0);
                let pos_km = Vec3::new(
                    pos[0] as f32 / 1000.0,
                    pos[2] as f32 / 1000.0,  // TEME Z (polar) -> Render Y (up)
                    -pos[1] as f32 / 1000.0, // TEME Y -> Render -Z (negated for right-handedness)
                );
                let vel_kms = Vec3::new(
                    vel[0] as f32 / 1000.0,
                    vel[2] as f32 / 1000.0,  // TEME Z -> Render Y
                    -vel[1] as f32 / 1000.0, // TEME Y -> Render -Z
                );
                let vel_kms = Vec3::new(
                    vel[0] as f32 / 1000.0,
                    vel[2] as f32 / 1000.0, // TEME Z -> Render Y
                    vel[1] as f32 / 1000.0, // TEME Y -> Render Z
                );

                // Normalize to Earth radii for rendering
                let pos_er = pos_km / EARTH_RADIUS_KM as f32;

                // Calculate altitude
                let altitude_km = (pos_km.length() as f64) - EARTH_RADIUS_KM;

                // Calculate TLE age
                let tle_age_days = (self.current_time - tle.epoch).as_seconds() / 86400.0;

                Some(SatelliteState {
                    position: pos_er,
                    velocity: vel_kms,
                    altitude_km,
                    tle_age_days: tle_age_days.abs(),
                })
            }
            Err(_) => None,
        }
    }

    /// Propagate all satellites and return positions
    pub fn propagate_all(&self) -> HashMap<u32, SatelliteState> {
        let mut results = HashMap::with_capacity(self.tles.len());

        for &norad_id in self.tles.keys() {
            if let Some(state) = self.propagate(norad_id) {
                results.insert(norad_id, state);
            }
        }

        results
    }

    /// Get TLE count
    pub fn tle_count(&self) -> usize {
        self.tles.len()
    }

    /// Get a reference to TLE for a satellite
    pub fn get_tle(&self, norad_id: u32) -> Option<&satkit::TLE> {
        self.tles.get(&norad_id)
    }
}

/// Parse TLE data into satkit TLE
fn parse_tle(tle_data: &TleData) -> Option<satkit::TLE> {
    match satkit::TLE::load_2line(&tle_data.line1, &tle_data.line2) {
        Ok(tle) => Some(tle),
        Err(e) => {
            log::trace!("Failed to parse TLE: {}", e);
            None
        }
    }
}

use chrono::Datelike;
use chrono::Timelike;
