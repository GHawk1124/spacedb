//! Search and filtering functionality for space objects

use std::collections::HashMap;

use super::{SpaceObject, SpaceObjectDatabase};

/// Search index for fast lookups
pub struct SearchIndex {
    /// Name (lowercased) -> list of NORAD IDs
    name_index: HashMap<String, Vec<u32>>,
    /// All objects sorted by name for browsing
    sorted_by_name: Vec<u32>,
    /// NORAD ID -> index in sorted_by_name
    norad_to_idx: HashMap<u32, usize>,
}

impl SearchIndex {
    /// Build search index from database
    pub fn build(db: &SpaceObjectDatabase) -> Self {
        let mut name_index: HashMap<String, Vec<u32>> = HashMap::new();
        let mut items: Vec<(String, u32)> = Vec::with_capacity(db.objects.len());

        for (norad_str, obj) in &db.objects {
            let norad = match norad_str.parse::<u32>() {
                Ok(n) => n,
                Err(_) => continue,
            };

            let name = obj.display_name();
            let name_lower = name.to_lowercase();

            // Index by words in name
            for word in name_lower.split_whitespace() {
                name_index.entry(word.to_string()).or_default().push(norad);
            }

            // Also index full name
            name_index
                .entry(name_lower.clone())
                .or_default()
                .push(norad);

            items.push((name, norad));
        }

        // Sort by name
        items.sort_by(|a, b| a.0.cmp(&b.0));

        let sorted_by_name: Vec<u32> = items.iter().map(|(_, n)| *n).collect();
        let norad_to_idx: HashMap<u32, usize> = sorted_by_name
            .iter()
            .enumerate()
            .map(|(i, &n)| (n, i))
            .collect();

        log::info!("Built search index with {} name entries", name_index.len());

        Self {
            name_index,
            sorted_by_name,
            norad_to_idx,
        }
    }

    /// Search for objects matching a query string
    pub fn search(&self, query: &str, limit: usize) -> Vec<u32> {
        let query_lower = query.to_lowercase().trim().to_string();

        if query_lower.is_empty() {
            return self.sorted_by_name.iter().take(limit).copied().collect();
        }

        // Try exact match first
        if let Some(matches) = self.name_index.get(&query_lower) {
            return matches.iter().take(limit).copied().collect();
        }

        // Try NORAD ID
        if let Ok(norad) = query_lower.parse::<u32>() {
            if self.norad_to_idx.contains_key(&norad) {
                return vec![norad];
            }
        }

        // Prefix matching on words
        let mut results: Vec<u32> = Vec::new();
        let mut seen = std::collections::HashSet::new();

        for (key, norads) in &self.name_index {
            if key.starts_with(&query_lower) || key.contains(&query_lower) {
                for &norad in norads {
                    if seen.insert(norad) {
                        results.push(norad);
                        if results.len() >= limit {
                            return results;
                        }
                    }
                }
            }
        }

        results
    }

    /// Get all objects sorted by name (for browsing)
    pub fn all_sorted(&self) -> &[u32] {
        &self.sorted_by_name
    }
}

/// Filter criteria for object browser
#[derive(Debug, Clone)]
pub struct ObjectFilter {
    pub object_types: Vec<String>,
    pub countries: Vec<String>,
    pub has_tle_only: bool,
    pub exclude_decayed: bool,
    pub size_filter_enabled: bool,
    pub size_min_m: f64,
    pub size_max_m: f64,
    pub include_unknown_size: bool,
}

impl Default for ObjectFilter {
    fn default() -> Self {
        Self {
            object_types: Vec::new(),
            countries: Vec::new(),
            has_tle_only: false,
            exclude_decayed: false,
            size_filter_enabled: false,
            size_min_m: 0.0,
            size_max_m: 0.0,
            include_unknown_size: true,
        }
    }
}

impl ObjectFilter {
    /// Check if an object matches this filter
    pub fn matches(&self, obj: &SpaceObject) -> bool {
        // Object type filter
        if !self.object_types.is_empty() {
            let obj_type = obj.object_type.as_deref().unwrap_or("");
            if !self.object_types.iter().any(|t| t == obj_type) {
                return false;
            }
        }

        // Country filter
        if !self.countries.is_empty() {
            let country = obj.country.as_deref().unwrap_or("");
            if !self.countries.iter().any(|c| c == country) {
                return false;
            }
        }

        // TLE filter
        if self.has_tle_only && !obj.has_valid_tle() {
            return false;
        }

        // Decayed filter
        if self.exclude_decayed && obj.is_decayed() {
            return false;
        }

        // Size filter (requires DISCOS data)
        if self.size_filter_enabled {
            if let Some(discos) = &obj.discos {
                let (w, h, d) = discos.dimensions();
                let size_m = w.max(h).max(d);
                if self.size_min_m > 0.0 && size_m < self.size_min_m {
                    return false;
                }
                if self.size_max_m > 0.0 && size_m > self.size_max_m {
                    return false;
                }
            } else if !self.include_unknown_size {
                return false;
            }
        }

        // TODO: Altitude filtering requires propagation

        true
    }
}
