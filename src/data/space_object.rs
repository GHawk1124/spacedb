//! Space object data structures matching the Python-generated JSON schema

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Root structure of the space_objects.json file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpaceObjectDatabase {
    pub generated_at: String,
    pub objects: HashMap<String, SpaceObject>,
}

/// A single space object (satellite, debris, rocket body, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpaceObject {
    pub norad_cat_id: u32,
    pub cospar_id: Option<String>,
    pub name: Option<String>,
    pub object_type: Option<String>,
    pub country: Option<String>,
    pub launch_date: Option<String>,
    pub decay_date: Option<String>,
    pub analyst_kind: Option<String>,
    pub in_satcat: bool,
    pub tle: Option<TleData>,
    pub sources: SourceFlags,

    /// DISCOS data (merged during load if available)
    #[serde(skip)]
    pub discos: Option<DiscosData>,
}

/// Two-Line Element set data for orbit propagation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TleData {
    pub epoch: String,
    pub line1: String,
    pub line2: String,
}

/// Flags indicating which data sources contributed to this object
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceFlags {
    pub spacetrack_satcat: bool,
    pub spacetrack_gp: bool,
    pub gcat: bool,
    pub discos: bool,
}

/// DISCOS data for physical characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscosData {
    #[serde(rename = "cosparId")]
    pub cospar_id: Option<String>,
    pub satno: Option<u32>,
    pub name: Option<String>,
    #[serde(rename = "objectClass")]
    pub object_class: Option<String>,
    pub mass: Option<f64>,
    pub shape: Option<String>,
    pub width: Option<f64>,
    pub height: Option<f64>,
    pub depth: Option<f64>,
    pub diameter: Option<f64>,
    pub span: Option<f64>,
    #[serde(rename = "xSectMax")]
    pub x_sect_max: Option<f64>,
    #[serde(rename = "xSectMin")]
    pub x_sect_min: Option<f64>,
    #[serde(rename = "xSectAvg")]
    pub x_sect_avg: Option<f64>,
}

impl DiscosData {
    /// Get approximate dimensions for the shape
    pub fn dimensions(&self) -> (f64, f64, f64) {
        let h = self.height.unwrap_or(1.0);
        let w = self.width.unwrap_or_else(|| self.diameter.unwrap_or(1.0));
        let d = self.depth.unwrap_or_else(|| self.diameter.unwrap_or(w));
        (w, h, d)
    }

    /// Estimate a representative cross-sectional area in m²
    pub fn cross_section_m2(&self) -> Option<f64> {
        if let Some(area) = self.x_sect_avg {
            return Some(area);
        }
        if let Some(area) = self.x_sect_max {
            return Some(area);
        }
        if let Some(area) = self.x_sect_min {
            return Some(area);
        }

        let width = self.width.or(self.diameter);
        let height = self.height.or(self.diameter);
        match (width, height) {
            (Some(w), Some(h)) => Some(w * h),
            _ => None,
        }
    }
}

impl SpaceObject {
    /// Get display name (falls back to NORAD ID if no name)
    pub fn display_name(&self) -> String {
        self.name
            .clone()
            .unwrap_or_else(|| format!("NORAD {}", self.norad_cat_id))
    }

    /// Check if this object has a valid TLE for propagation
    pub fn has_valid_tle(&self) -> bool {
        self.tle.is_some()
    }

    /// Check if this object has decayed
    pub fn is_decayed(&self) -> bool {
        self.decay_date.is_some()
    }
}
