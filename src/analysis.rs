use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{anyhow, Result};
use clap::Args;
use glam::Vec3;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Serialize;

use crate::data::{load_complete_database, ObjectFilter, SpaceObjectDatabase};
use crate::propagation::hifi::{
    orbital_state_from_tle, AtmosphereModelType, GravityModelChoice, HiFiSettings, IntegratorType,
    PropagatorConfig, PropagatorType, SpacecraftState,
};
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

#[derive(Args, Debug, Clone)]
pub struct DecayArgs {
    /// Output JSON file path
    #[arg(long, default_value = "out/decay_predictions.json")]
    pub output: PathBuf,
    /// Time horizon in days for decay prediction
    #[arg(long, default_value_t = 3650.0)]
    pub days: f64,
    /// Propagator to use (native-rk4 or satkit-rk98)
    #[arg(long, value_enum, default_value = "native-rk4")]
    pub propagator: DecayPropagatorType,
    /// Atmosphere model
    #[arg(long, value_enum, default_value = "nrlmsise00")]
    pub atmosphere: DecayAtmosphereType,
    /// Gravity model
    #[arg(long, value_enum, default_value = "j2")]
    pub gravity: DecayGravityType,
    /// Include solar radiation pressure
    #[arg(long, default_value_t = true)]
    pub srp: bool,
    /// Include third-body perturbations (Sun/Moon) - requires ~100MB JPL ephemeris download
    #[arg(long, default_value_t = false)]
    pub third_body: bool,
    /// Integration step size in seconds
    #[arg(long, default_value_t = 60.0)]
    pub step_seconds: f64,
    /// Error tolerance for adaptive stepping
    #[arg(long, default_value_t = 1e-9)]
    pub tolerance: f64,
    /// Reentry altitude threshold in km (fragmentation/decay)
    #[arg(long, default_value_t = 70.0)]
    pub reentry_altitude_km: f64,
    /// Max number of satellites to process (0 = all)
    #[arg(long, default_value_t = 0)]
    pub limit: usize,
    /// NORAD IDs to process (comma-separated, e.g., "25544,20580")
    #[arg(long, value_delimiter = ',')]
    pub norad_ids: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum DecayPropagatorType {
    NativeRk4,
    SatkitRk98,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum DecayAtmosphereType {
    Nrlmsise00,
    Jb2008,
    HarrisPriester,
    Exponential,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum DecayGravityType {
    PointMass,
    J2,
    FullField20,
}

impl DecayPropagatorType {
    fn to_propagator_type(self) -> PropagatorType {
        match self {
            Self::NativeRk4 => PropagatorType::NativeRk4,
            Self::SatkitRk98 => PropagatorType::SatkitRk98,
        }
    }
}

impl DecayAtmosphereType {
    fn to_atmosphere_type(self) -> AtmosphereModelType {
        match self {
            Self::Nrlmsise00 => AtmosphereModelType::Nrlmsise00,
            Self::Jb2008 => AtmosphereModelType::Jb2008,
            Self::HarrisPriester => AtmosphereModelType::HarrisPriester,
            Self::Exponential => AtmosphereModelType::Exponential,
        }
    }
}

impl DecayGravityType {
    fn to_gravity_choice(self) -> GravityModelChoice {
        match self {
            Self::PointMass => GravityModelChoice::PointMass,
            Self::J2 => GravityModelChoice::J2,
            Self::FullField20 => GravityModelChoice::FullField20,
        }
    }
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

/// Result of a single decay prediction
#[derive(Debug, Serialize, Clone)]
pub struct DecayPrediction {
    /// NORAD catalog ID
    pub norad_id: u32,
    /// Object name (if available)
    pub name: Option<String>,
    /// Object type (if available)
    pub object_type: Option<String>,
    /// Country (if available)
    pub country: Option<String>,
    /// Launch date (if available)
    pub launch_date: Option<String>,
    /// Initial epoch of the TLE used
    pub tle_epoch: String,
    /// Age of TLE at prediction time (days)
    pub tle_age_days: f64,
    /// Initial altitude (km)
    pub initial_altitude_km: f64,
    /// Predicted decay/reentry epoch (if occurred within horizon)
    pub decay_epoch: Option<String>,
    /// Altitude at which decay was detected (km)
    pub decay_altitude_km: Option<f64>,
    /// Days until decay from prediction start (if decayed)
    pub days_to_decay: Option<f64>,
    /// Status: "decayed", "survived", "error", "no_tle"
    pub status: String,
    /// Error message (if status is "error")
    pub error_message: Option<String>,
    /// Propagation time (seconds)
    pub propagation_seconds: f64,
}

/// Decay prediction database output
#[derive(Debug, Serialize)]
pub struct DecayDatabase {
    pub generated_at: String,
    pub start_time_utc: String,
    pub horizon_days: f64,
    pub reentry_altitude_km: f64,
    pub propagator: String,
    pub atmosphere: String,
    pub gravity: String,
    pub total_objects: usize,
    pub decayed_count: usize,
    pub survived_count: usize,
    pub error_count: usize,
    pub predictions: Vec<DecayPrediction>,
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

/// Run decay prediction for satellites
/// Initialize satkit data files (downloads if missing)
fn ensure_satkit_data() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    
    INIT.call_once(|| {
        log::info!("Checking satkit data files...");
        let data_dir = satkit::utils::datadir();
        log::info!("Satkit data directory: {:?}", data_dir);
        
        // Download missing files, don't force update
        match satkit::utils::update_datafiles(None, false) {
            Ok(_) => log::info!("Satkit data files ready"),
            Err(e) => log::warn!("Could not update satkit data files: {}", e),
        }
        
        // Update Earth Orientation Parameters for long-term predictions
        log::info!("Updating Earth Orientation Parameters...");
        match satkit::earth_orientation_params::update() {
            Ok(_) => log::info!("EOP data updated"),
            Err(e) => log::warn!("Could not update EOP data: {}", e),
        }
        
        // Disable the EOP time warning for long-term predictions
        // The warning is expected when predicting far into the future
        satkit::earth_orientation_params::disable_eop_time_warning();
    });
}

pub fn run_decay_prediction(args: DecayArgs) -> Result<()> {
    if args.days <= 0.0 {
        return Err(anyhow!("days must be > 0"));
    }
    if args.step_seconds <= 0.0 {
        return Err(anyhow!("step-seconds must be > 0"));
    }

    // Ensure satkit data files are available before starting
    ensure_satkit_data();

    log::info!("Loading database...");
    let db_path = PathBuf::from("out/space_objects.json");
    let discos_path = PathBuf::from("data/cache/discos_objects_by_satno.json.gz");
    let database = load_complete_database(&db_path, &discos_path)?;

    // Collect NORAD IDs to process
    let ids: Vec<u32> = if !args.norad_ids.is_empty() {
        args.norad_ids.clone()
    } else {
        let mut filter = ObjectFilter::default();
        filter.has_tle_only = true;
        filter.exclude_decayed = true;
        collect_ids(&database, &filter)
    };

    if ids.is_empty() {
        return Err(anyhow!("no objects matched filters"));
    }

    // Apply limit if specified
    let ids = if args.limit > 0 && args.limit < ids.len() {
        ids[..args.limit].to_vec()
    } else {
        ids
    };

    log::info!(
        "Processing {} satellites for decay prediction (horizon: {:.0} days)...",
        ids.len(),
        args.days
    );

    // Build HiFi settings from arguments
    let mut config = PropagatorConfig::high_precision();
    config.step_size = args.step_seconds;
    config.tolerance = args.tolerance;
    config.reentry_altitude = args.reentry_altitude_km * 1000.0; // Convert km to meters

    let settings = HiFiSettings {
        propagator: args.propagator.to_propagator_type(),
        integrator: match args.propagator {
            DecayPropagatorType::NativeRk4 => IntegratorType::NativeRk4,
            DecayPropagatorType::SatkitRk98 => IntegratorType::SatkitRk98,
        },
        atmosphere: args.atmosphere.to_atmosphere_type(),
        gravity: args.gravity.to_gravity_choice(),
        include_srp: args.srp,
        include_third_body: args.third_body,
        config,
        decay_horizon_days: args.days,
    };

    let start_time = satkit::Instant::now();
    let horizon_duration = satkit::Duration::from_days(args.days);
    let target_epoch = start_time + horizon_duration;

    log::info!(
        "Using propagator: {}, Atmosphere: {}, Gravity: {}",
        settings.propagator.name(),
        settings.atmosphere.name(),
        settings.gravity.name()
    );

    // Build propagator (this will fail if SGP4 is selected)
    let propagator = match settings.build_propagator() {
        Some(p) => p,
        None => {
            return Err(anyhow!(
                "SGP4 cannot be used for decay prediction. Use --propagator native-rk4 or satkit-rk98"
            ));
        }
    };

    let progress = ProgressBar::new(ids.len() as u64);
    progress.set_style(
        ProgressStyle::with_template(
            "{elapsed_precise} {bar:40.cyan/blue} {pos}/{len} {percent}% ETA {eta_precise} {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    let mut predictions: Vec<DecayPrediction> = Vec::with_capacity(ids.len());
    let mut decayed_count = 0;
    let mut survived_count = 0;
    let mut error_count = 0;

    for norad_id in &ids {
        let norad_str = norad_id.to_string();
        let obj = database.objects.get(&norad_str);

        progress.set_message(format!("NORAD {}", norad_id));

        // Get object metadata
        let name = obj.map(|o| o.display_name());
        let object_type = obj.and_then(|o| o.object_type.clone());
        let country = obj.and_then(|o| o.country.clone());
        let launch_date = obj.and_then(|o| o.launch_date.clone());

        // Get TLE
        let tle = match obj.and_then(|o| o.tle.as_ref()) {
            Some(tle) => tle,
            None => {
                predictions.push(DecayPrediction {
                    norad_id: *norad_id,
                    name: name.clone(),
                    object_type: object_type.clone(),
                    country: country.clone(),
                    launch_date: launch_date.clone(),
                    tle_epoch: "N/A".to_string(),
                    tle_age_days: 0.0,
                    initial_altitude_km: 0.0,
                    decay_epoch: None,
                    decay_altitude_km: None,
                    days_to_decay: None,
                    status: "no_tle".to_string(),
                    error_message: Some("No TLE available".to_string()),
                    propagation_seconds: 0.0,
                });
                error_count += 1;
                progress.inc(1);
                continue;
            }
        };

        // Parse TLE
        let satkit_tle = match satkit::TLE::load_2line(&tle.line1, &tle.line2) {
            Ok(t) => t,
            Err(e) => {
                predictions.push(DecayPrediction {
                    norad_id: *norad_id,
                    name: name.clone(),
                    object_type: object_type.clone(),
                    country: country.clone(),
                    launch_date: launch_date.clone(),
                    tle_epoch: tle.epoch.clone(),
                    tle_age_days: 0.0,
                    initial_altitude_km: 0.0,
                    decay_epoch: None,
                    decay_altitude_km: None,
                    days_to_decay: None,
                    status: "error".to_string(),
                    error_message: Some(format!("Failed to parse TLE: {}", e)),
                    propagation_seconds: 0.0,
                });
                error_count += 1;
                progress.inc(1);
                continue;
            }
        };

        // Build initial state
        let orbital_state = match orbital_state_from_tle(&satkit_tle, &start_time) {
            Some(s) => s,
            None => {
                predictions.push(DecayPrediction {
                    norad_id: *norad_id,
                    name: name.clone(),
                    object_type: object_type.clone(),
                    country: country.clone(),
                    launch_date: launch_date.clone(),
                    tle_epoch: tle.epoch.clone(),
                    tle_age_days: 0.0,
                    initial_altitude_km: 0.0,
                    decay_epoch: None,
                    decay_altitude_km: None,
                    days_to_decay: None,
                    status: "error".to_string(),
                    error_message: Some("Failed to convert TLE to orbital state".to_string()),
                    propagation_seconds: 0.0,
                });
                error_count += 1;
                progress.inc(1);
                continue;
            }
        };

        // Get DISCOS data for spacecraft properties
        let (mass, area) = obj
            .and_then(|o| o.discos.as_ref())
            .map(|d| (d.mass, d.x_sect_avg))
            .unwrap_or((None, None));

        let initial_state = SpacecraftState::from_discos(orbital_state, mass, area);
        let initial_altitude_km = initial_state.orbital.altitude() / 1000.0;

        // Calculate TLE age
        let tle_age_days = (start_time - satkit_tle.epoch).as_seconds() / 86400.0;

        // Run propagation
        let prop_start = std::time::Instant::now();
        let result = propagator.propagate(initial_state, target_epoch);
        let propagation_seconds = prop_start.elapsed().as_secs_f64();

        match result {
            Ok(_) => {
                // Satellite survived the entire horizon without decay
                predictions.push(DecayPrediction {
                    norad_id: *norad_id,
                    name,
                    object_type,
                    country,
                    launch_date,
                    tle_epoch: tle.epoch.clone(),
                    tle_age_days: tle_age_days.abs(),
                    initial_altitude_km,
                    decay_epoch: None,
                    decay_altitude_km: None,
                    days_to_decay: None,
                    status: "survived".to_string(),
                    error_message: None,
                    propagation_seconds,
                });
                survived_count += 1;
            }
            Err(crate::propagation::hifi::PropagationError::Reentry { altitude_km, epoch }) => {
                // Satellite decayed!
                let days_to_decay = (epoch - start_time).as_seconds() / 86400.0;
                predictions.push(DecayPrediction {
                    norad_id: *norad_id,
                    name,
                    object_type,
                    country,
                    launch_date,
                    tle_epoch: tle.epoch.clone(),
                    tle_age_days: tle_age_days.abs(),
                    initial_altitude_km,
                    decay_epoch: Some(epoch.to_string()),
                    decay_altitude_km: Some(altitude_km),
                    days_to_decay: Some(days_to_decay),
                    status: "decayed".to_string(),
                    error_message: None,
                    propagation_seconds,
                });
                decayed_count += 1;
            }
            Err(e) => {
                // Other error (escape, integration failure, etc.)
                predictions.push(DecayPrediction {
                    norad_id: *norad_id,
                    name,
                    object_type,
                    country,
                    launch_date,
                    tle_epoch: tle.epoch.clone(),
                    tle_age_days: tle_age_days.abs(),
                    initial_altitude_km,
                    decay_epoch: None,
                    decay_altitude_km: None,
                    days_to_decay: None,
                    status: "error".to_string(),
                    error_message: Some(e.to_string()),
                    propagation_seconds,
                });
                error_count += 1;
            }
        }

        progress.inc(1);
    }

    progress.finish_and_clear();

    // Sort predictions: decayed first (by days_to_decay), then survived, then errors
    predictions.sort_by(|a, b| match (a.status.as_str(), b.status.as_str()) {
        ("decayed", "decayed") => a
            .days_to_decay
            .partial_cmp(&b.days_to_decay)
            .unwrap_or(std::cmp::Ordering::Equal),
        ("decayed", _) => std::cmp::Ordering::Less,
        (_, "decayed") => std::cmp::Ordering::Greater,
        ("survived", "error") => std::cmp::Ordering::Less,
        ("error", "survived") => std::cmp::Ordering::Greater,
        _ => std::cmp::Ordering::Equal,
    });

    let db = DecayDatabase {
        generated_at: chrono::Utc::now().to_rfc3339(),
        start_time_utc: start_time.to_string(),
        horizon_days: args.days,
        reentry_altitude_km: args.reentry_altitude_km,
        propagator: settings.propagator.name().to_string(),
        atmosphere: settings.atmosphere.name().to_string(),
        gravity: settings.gravity.name().to_string(),
        total_objects: ids.len(),
        decayed_count,
        survived_count,
        error_count,
        predictions,
    };

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(&args.output)?;
    serde_json::to_writer_pretty(file, &db)?;

    log::info!("Wrote decay predictions to {:?}", args.output);
    log::info!(
        "Results: {} decayed, {} survived, {} errors",
        decayed_count,
        survived_count,
        error_count
    );

    Ok(())
}
