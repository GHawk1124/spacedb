//! Lookup-table wrapper for atmosphere models
//!
//! This provides a configurable, quantized cache over any AtmosphereModel.
//! It trades accuracy for speed by reusing densities for nearby positions
//! and times based on a fixed grid.

use super::{gcrf_to_geodetic, AtmosphereDensity, AtmosphereModel};
use chrono::NaiveDate;
use nalgebra::Vector3;
use parking_lot::RwLock;
use satkit::Instant;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LookupAccuracy {
    Low,
    Medium,
    High,
}

impl LookupAccuracy {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Low => "Fastest, coarsest lookup grid",
            Self::Medium => "Balanced lookup grid",
            Self::High => "Slowest, finest lookup grid",
        }
    }

    pub fn all() -> &'static [LookupAccuracy] {
        &[Self::Low, Self::Medium, Self::High]
    }
}

impl std::str::FromStr for LookupAccuracy {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_lowercase().as_str() {
            "low" => Ok(Self::Low),
            "medium" | "med" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LookupConfig {
    pub altitude_step_km: f64,
    pub latitude_step_deg: f64,
    pub longitude_step_deg: f64,
    pub time_step_seconds: i64,
}

impl LookupConfig {
    pub fn for_accuracy(accuracy: LookupAccuracy) -> Self {
        match accuracy {
            LookupAccuracy::Low => Self {
                altitude_step_km: 25.0,
                latitude_step_deg: 15.0,
                longitude_step_deg: 15.0,
                time_step_seconds: 6 * 3600,
            },
            LookupAccuracy::Medium => Self {
                altitude_step_km: 10.0,
                latitude_step_deg: 5.0,
                longitude_step_deg: 5.0,
                time_step_seconds: 2 * 3600,
            },
            LookupAccuracy::High => Self {
                altitude_step_km: 2.0,
                latitude_step_deg: 2.0,
                longitude_step_deg: 2.0,
                time_step_seconds: 30 * 60,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct LookupKey {
    alt_idx: i32,
    lat_idx: i32,
    lon_idx: i32,
    time_idx: i64,
}

pub struct LookupAtmosphere {
    model: Box<dyn AtmosphereModel>,
    config: LookupConfig,
    cache: RwLock<HashMap<LookupKey, AtmosphereDensity>>,
}

impl LookupAtmosphere {
    pub fn new(model: Box<dyn AtmosphereModel>, config: LookupConfig) -> Self {
        Self {
            model,
            config,
            cache: RwLock::new(HashMap::new()),
        }
    }

    fn make_key(&self, position: &Vector3<f64>, epoch: &Instant) -> LookupKey {
        let (lat_deg, lon_deg, alt_m) = gcrf_to_geodetic(position, epoch);
        let alt_km = alt_m / 1000.0;

        LookupKey {
            alt_idx: quantize(alt_km, self.config.altitude_step_km),
            lat_idx: quantize(lat_deg, self.config.latitude_step_deg),
            lon_idx: quantize(lon_deg, self.config.longitude_step_deg),
            time_idx: quantize_time(epoch, self.config.time_step_seconds),
        }
    }
}

impl AtmosphereModel for LookupAtmosphere {
    fn density(&self, position: &Vector3<f64>, epoch: &Instant) -> AtmosphereDensity {
        let key = self.make_key(position, epoch);

        if let Some(density) = self.cache.read().get(&key).copied() {
            return density;
        }

        let density = self.model.density(position, epoch);
        self.cache.write().insert(key, density);
        density
    }

    fn name(&self) -> &'static str {
        self.model.name()
    }

    fn description(&self) -> &'static str {
        "Lookup-table wrapped atmosphere model"
    }

    fn requires_space_weather(&self) -> bool {
        self.model.requires_space_weather()
    }
}

fn quantize(value: f64, step: f64) -> i32 {
    if step <= 0.0 {
        return 0;
    }
    (value / step).round() as i32
}

fn quantize_time(epoch: &Instant, step_seconds: i64) -> i64 {
    if step_seconds <= 0 {
        return 0;
    }
    epoch_seconds(epoch) / step_seconds
}

fn epoch_seconds(epoch: &Instant) -> i64 {
    let (year, month, day, hour, min, sec) = epoch.as_datetime();
    let sec_floor = sec.floor().max(0.0) as u32;

    let date = NaiveDate::from_ymd_opt(year, month as u32, day as u32)
        .unwrap_or_else(|| NaiveDate::from_ymd_opt(1970, 1, 1).unwrap());
    let datetime = date
        .and_hms_opt(hour as u32, min as u32, sec_floor)
        .unwrap_or_else(|| NaiveDate::from_ymd_opt(1970, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap());

    datetime.and_utc().timestamp()
}
