//! Search and filtering functionality for space objects

use std::collections::HashMap;
use std::sync::Arc;

use nucleo::pattern::{CaseMatching, Normalization};
use nucleo::{Config, Nucleo, Utf32String};

use super::{SpaceObject, SpaceObjectDatabase};

/// Search index for fast lookups
pub struct SearchIndex {
    /// Name (lowercased) -> list of NORAD IDs
    name_index: HashMap<String, Vec<u32>>,
    /// All objects sorted by name for browsing
    sorted_by_name: Vec<u32>,
    /// NORAD ID -> index in sorted_by_name
    norad_to_idx: HashMap<u32, usize>,
    /// Fuzzy matcher
    matcher: Nucleo<SearchItem>,
    /// Last query used for append optimization
    last_query: String,
    /// Whether matcher is still running
    matcher_running: bool,
}

struct SearchItem {
    norad: u32,
    haystack: String,
}

impl SearchIndex {
    /// Build search index from database
    pub fn build(db: &SpaceObjectDatabase) -> Self {
        let mut name_index: HashMap<String, Vec<u32>> = HashMap::new();
        let mut items: Vec<(String, u32)> = Vec::with_capacity(db.objects.len());
        let matcher = Nucleo::new(Config::DEFAULT, Arc::new(|| {}), None, 1);
        let injector = matcher.injector();

        for (norad_str, obj) in &db.objects {
            let norad = match norad_str.parse::<u32>() {
                Ok(n) => n,
                Err(_) => continue,
            };

            let name = obj.display_name();
            let name_lower = name.to_lowercase();

            // Also index full name
            name_index
                .entry(name_lower.clone())
                .or_default()
                .push(norad);

            let haystack = format!("{} {}", name_lower, norad);
            let item = SearchItem { norad, haystack };
            injector.push(item, |data, cols| {
                cols[0] = Utf32String::from(data.haystack.as_str());
            });

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
            matcher,
            last_query: String::new(),
            matcher_running: false,
        }
    }

    /// Search for objects matching a query string
    pub fn search(&mut self, query: &str, limit: usize) -> Vec<u32> {
        let query_lower = query.to_lowercase().trim().to_string();

        if query_lower.is_empty() {
            self.last_query.clear();
            self.matcher_running = false;
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

        let pattern_changed = query_lower != self.last_query;
        if pattern_changed {
            let append = query_lower.starts_with(&self.last_query)
                && query_lower.len() > self.last_query.len();
            self.matcher.pattern.reparse(
                0,
                &query_lower,
                CaseMatching::Respect,
                Normalization::Smart,
                append,
            );
            self.last_query = query_lower;
        }

        let status = self.matcher.tick(10);
        self.matcher_running = status.running;
        let snapshot = self.matcher.snapshot();

        let mut results = Vec::new();
        let take = limit.min(snapshot.matched_item_count() as usize) as u32;
        for item in snapshot.matched_items(0..take) {
            results.push(item.data.norad);
        }

        results
    }

    pub fn matcher_is_running(&self) -> bool {
        self.matcher_running
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
