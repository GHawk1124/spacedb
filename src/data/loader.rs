//! Data loading and caching from JSON files

use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result};
use flate2::read::GzDecoder;

use super::{DiscosData, SpaceObjectDatabase};

/// Load the main space objects database from JSON
pub fn load_space_objects(path: impl AsRef<Path>) -> Result<SpaceObjectDatabase> {
    let path = path.as_ref();
    log::info!("Loading space objects from {:?}", path);

    let file = File::open(path)
        .with_context(|| format!("Failed to open space objects file: {:?}", path))?;

    let reader = BufReader::new(file);
    let db: SpaceObjectDatabase =
        serde_json::from_reader(reader).with_context(|| "Failed to parse space objects JSON")?;

    log::info!(
        "Loaded {} space objects (generated at {})",
        db.objects.len(),
        db.generated_at
    );

    Ok(db)
}

/// Load DISCOS data from gzipped JSON cache
pub fn load_discos_cache(path: impl AsRef<Path>) -> Result<HashMap<u32, DiscosData>> {
    let path = path.as_ref();
    log::info!("Loading DISCOS cache from {:?}", path);

    let file =
        File::open(path).with_context(|| format!("Failed to open DISCOS cache: {:?}", path))?;

    let reader = BufReader::new(file);
    let gz = GzDecoder::new(reader);

    // DISCOS cache uses string keys
    let raw: HashMap<String, DiscosData> =
        serde_json::from_reader(gz).with_context(|| "Failed to parse DISCOS cache")?;

    // Convert to u32 keys
    let mut result = HashMap::with_capacity(raw.len());
    for (key, value) in raw {
        if let Ok(norad) = key.parse::<u32>() {
            result.insert(norad, value);
        }
    }

    log::info!("Loaded {} DISCOS entries", result.len());
    Ok(result)
}

/// Merge DISCOS data into space objects
pub fn merge_discos_data(db: &mut SpaceObjectDatabase, discos: HashMap<u32, DiscosData>) {
    let mut merged = 0;
    for (norad_str, obj) in db.objects.iter_mut() {
        if let Ok(norad) = norad_str.parse::<u32>() {
            if let Some(discos_data) = discos.get(&norad) {
                obj.discos = Some(discos_data.clone());
                merged += 1;
            }
        }
    }
    log::info!("Merged DISCOS data for {} objects", merged);
}

/// Load the complete database with DISCOS data merged
pub fn load_complete_database(
    space_objects_path: impl AsRef<Path>,
    discos_cache_path: impl AsRef<Path>,
) -> Result<SpaceObjectDatabase> {
    let mut db = load_space_objects(space_objects_path)?;

    // Try to load DISCOS data, but don't fail if not available
    match load_discos_cache(discos_cache_path) {
        Ok(discos) => merge_discos_data(&mut db, discos),
        Err(e) => log::warn!("Could not load DISCOS cache: {}", e),
    }

    Ok(db)
}

/// Statistics about the loaded database
#[derive(Debug, Default)]
pub struct DatabaseStats {
    pub total_objects: usize,
    pub objects_with_tle: usize,
    pub decayed_objects: usize,
    pub payloads: usize,
    pub rocket_bodies: usize,
    pub debris: usize,
    pub objects_with_discos: usize,
}

impl DatabaseStats {
    pub fn from_database(db: &SpaceObjectDatabase) -> Self {
        let mut stats = Self::default();
        stats.total_objects = db.objects.len();

        for obj in db.objects.values() {
            if obj.has_valid_tle() {
                stats.objects_with_tle += 1;
            }
            if obj.is_decayed() {
                stats.decayed_objects += 1;
            }
            if obj.discos.is_some() {
                stats.objects_with_discos += 1;
            }

            match obj.object_type.as_deref() {
                Some("PAYLOAD") => stats.payloads += 1,
                Some("ROCKET BODY") => stats.rocket_bodies += 1,
                Some("DEBRIS") => stats.debris += 1,
                _ => {}
            }
        }

        stats
    }
}
