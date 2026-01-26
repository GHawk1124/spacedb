use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{anyhow, Result};
use clap::Args;
use glam::Vec3;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;

use crate::data::{load_complete_database, ObjectFilter, SpaceObjectDatabase};
use crate::propagation::{Propagator, EARTH_RADIUS_KM};

#[derive(Args, Debug, Clone)]
pub struct CollisionArgs {
    /// Output JSON file path
    #[arg(long, default_value = "out/collision_events.json")]
    pub output: PathBuf,
    /// Time horizon in hours
    #[arg(long, default_value_t = 72.0)]
    pub hours: f64,
    /// Propagation step in seconds
    #[arg(long, default_value_t = 60)]
    pub step_seconds: u64,
    /// Distance threshold in kilometers
    #[arg(long, default_value_t = 10.0)]
    pub distance_km: f64,
    /// Max number of events to keep in output
    #[arg(long, default_value_t = 50000)]
    pub max_events: usize,
}

#[derive(Debug, Serialize, Clone)]
struct CollisionEvent {
    norad_a: u32,
    norad_b: u32,
    time_utc: String,
    distance_km: f64,
    relative_speed_kms: f64,
    score: f64,
}

#[derive(Debug, Serialize)]
struct CollisionDatabase {
    generated_at: String,
    start_time_utc: String,
    hours: f64,
    step_seconds: u64,
    distance_km: f64,
    total_objects: usize,
    total_pairs: usize,
    events: Vec<CollisionEvent>,
}

#[derive(Clone, Copy)]
struct ObjectState {
    norad: u32,
    pos_km: Vec3,
    vel_kms: Vec3,
}

pub fn run_collision_scan(args: CollisionArgs) -> Result<()> {
    if args.step_seconds == 0 {
        return Err(anyhow!("step-seconds must be > 0"));
    }
    if args.distance_km <= 0.0 {
        return Err(anyhow!("distance-km must be > 0"));
    }

    log::info!("Loading database...");
    let db_path = PathBuf::from("out/space_objects.json");
    let discos_path = PathBuf::from("data/cache/discos_objects_by_satno.json.gz");
    let database = load_complete_database(&db_path, &discos_path)?;

    let mut filter = ObjectFilter::default();
    filter.has_tle_only = true;
    filter.exclude_decayed = true;

    let ids = collect_ids(&database, &filter);
    if ids.is_empty() {
        return Err(anyhow!("no objects matched filters"));
    }

    log::info!("Initializing propagator...");
    let mut propagator = Propagator::new();
    propagator.load_tles(&database.objects);

    let steps = ((args.hours * 3600.0) / args.step_seconds as f64).ceil() as u64;
    let total_steps = steps + 1;
    let start_time = *propagator.current_time();
    let threshold_km = args.distance_km as f32;
    let threshold_sq = threshold_km * threshold_km;
    let cell_size = threshold_km.max(1.0);
    let inv_cell = 1.0 / cell_size;

    let mut events: HashMap<(u32, u32), CollisionEvent> = HashMap::new();

    log::info!(
        "Scanning {} objects for {} hours ({} steps, {} samples)...",
        ids.len(),
        args.hours,
        steps,
        total_steps
    );

    let progress = ProgressBar::new(total_steps);
    progress.set_style(
        ProgressStyle::with_template(
            "{elapsed_precise} {bar:40.cyan/blue} {pos}/{len} {percent}% ETA {eta_precise}",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    for step in 0..=steps {
        let time =
            start_time + satkit::Duration::from_seconds(step as f64 * args.step_seconds as f64);
        propagator.set_time(time);
        let states = propagator.propagate_subset(&ids);
        if states.is_empty() {
            progress.inc(1);
            continue;
        }

        let mut objects: Vec<ObjectState> = Vec::with_capacity(states.len());
        for (norad, state) in states.iter() {
            let pos_km = state.position * EARTH_RADIUS_KM as f32;
            objects.push(ObjectState {
                norad: *norad,
                pos_km,
                vel_kms: state.velocity,
            });
        }

        let mut cells: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for (idx, obj) in objects.iter().enumerate() {
            let key = cell_key(obj.pos_km, inv_cell);
            cells.entry(key).or_default().push(idx);
        }

        let cell_keys: Vec<(i32, i32, i32)> = cells.keys().copied().collect();
        for key in cell_keys {
            let list_a = match cells.get(&key) {
                Some(list) => list,
                None => continue,
            };

            for dz in -1..=1 {
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let neighbor = (key.0 + dx, key.1 + dy, key.2 + dz);
                        if neighbor < key {
                            continue;
                        }
                        let list_b = match cells.get(&neighbor) {
                            Some(list) => list,
                            None => continue,
                        };

                        for &i in list_a {
                            let start_j = if neighbor == key { i + 1 } else { 0 };
                            for &j in list_b.iter().skip(start_j) {
                                let a = objects[i];
                                let b = objects[j];
                                let delta = a.pos_km - b.pos_km;
                                let dist_sq = delta.length_squared();
                                if dist_sq > threshold_sq {
                                    continue;
                                }

                                let dist_km = dist_sq.sqrt() as f64;
                                let rel_speed = (a.vel_kms - b.vel_kms).length() as f64;
                                let score = rel_speed / dist_km.max(0.001);

                                let (norad_a, norad_b) = if a.norad < b.norad {
                                    (a.norad, b.norad)
                                } else {
                                    (b.norad, a.norad)
                                };
                                let key_pair = (norad_a, norad_b);

                                let entry = events.entry(key_pair).or_insert(CollisionEvent {
                                    norad_a,
                                    norad_b,
                                    time_utc: time.to_string(),
                                    distance_km: dist_km,
                                    relative_speed_kms: rel_speed,
                                    score,
                                });

                                if dist_km < entry.distance_km {
                                    *entry = CollisionEvent {
                                        norad_a,
                                        norad_b,
                                        time_utc: time.to_string(),
                                        distance_km: dist_km,
                                        relative_speed_kms: rel_speed,
                                        score,
                                    };
                                }
                            }
                        }
                    }
                }
            }
        }

        progress.inc(1);
    }

    progress.finish_and_clear();

    let mut events_vec: Vec<CollisionEvent> = events.into_values().collect();
    events_vec.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if events_vec.len() > args.max_events {
        events_vec.truncate(args.max_events);
    }

    let db = CollisionDatabase {
        generated_at: chrono::Utc::now().to_rfc3339(),
        start_time_utc: start_time.to_string(),
        hours: args.hours,
        step_seconds: args.step_seconds,
        distance_km: args.distance_km,
        total_objects: ids.len(),
        total_pairs: events_vec.len(),
        events: events_vec,
    };

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(&args.output)?;
    serde_json::to_writer_pretty(file, &db)?;

    log::info!("Wrote collision database to {:?}", args.output);
    Ok(())
}

fn collect_ids(database: &SpaceObjectDatabase, filter: &ObjectFilter) -> Vec<u32> {
    let mut ids = Vec::new();
    for (norad_str, obj) in &database.objects {
        if let Ok(norad) = norad_str.parse::<u32>() {
            if filter.matches(obj) {
                ids.push(norad);
            }
        }
    }
    ids
}

fn cell_key(pos_km: Vec3, inv_cell: f32) -> (i32, i32, i32) {
    (
        (pos_km.x * inv_cell).floor() as i32,
        (pos_km.y * inv_cell).floor() as i32,
        (pos_km.z * inv_cell).floor() as i32,
    )
}
