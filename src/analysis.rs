use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use clap::Args;
use glam::Vec3;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::data::{load_complete_database, ObjectFilter, SpaceObject, SpaceObjectDatabase};
use crate::propagation::hifi::forces::EphemerisType;
use crate::propagation::hifi::{
    orbital_state_from_tle, AtmosphereModelType, GravityModelChoice, HiFiSettings, IntegratorType,
    LookupAccuracy, PropagatorConfig, PropagatorType, SpacecraftState,
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
    /// Propagator to use (sgp4 for speed, native-rk4 or satkit-rk98 for accuracy)
    #[arg(long, value_enum, default_value = "satkit-rk98")]
    pub propagator: CollisionPropagatorType,
    /// Atmosphere model (only used with native-rk4 or satkit-rk98)
    #[arg(long, value_enum, default_value = "harris-priester")]
    pub atmosphere: DecayAtmosphereType,
    /// Lookup-table accuracy (only used with *-lookup atmospheres)
    #[arg(long, value_enum, default_value = "medium")]
    pub lookup_accuracy: AtmosphereLookupAccuracy,
    /// Gravity model (only used with native-rk4 or satkit-rk98)
    #[arg(long, value_enum, default_value = "j2")]
    pub gravity: DecayGravityType,
    /// Include solar radiation pressure (only used with HiFi propagators)
    #[arg(long, default_value_t = false)]
    pub srp: bool,
    /// Include third-body perturbations (Sun/Moon) (only used with HiFi propagators)
    #[arg(long, default_value_t = false)]
    pub third_body: bool,
    /// Ephemeris for third-body (low-precision or high-precision)
    #[arg(long, value_enum, default_value = "low-precision")]
    pub ephemeris: DecayEphemerisType,
    /// Max number of satellites to process (0 = all)
    #[arg(long, default_value_t = 0)]
    pub limit: usize,
    /// NORAD IDs to process (comma-separated, e.g., "25544,20580")
    #[arg(long, value_delimiter = ',')]
    pub norad_ids: Vec<u32>,
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
    #[arg(long, value_enum, default_value = "satkit-rk98")]
    pub propagator: DecayPropagatorType,
    /// Atmosphere model
    #[arg(long, value_enum, default_value = "nrlmsise00")]
    pub atmosphere: DecayAtmosphereType,
    /// Lookup-table accuracy (only used with *-lookup atmospheres)
    #[arg(long, value_enum, default_value = "medium")]
    pub lookup_accuracy: AtmosphereLookupAccuracy,
    /// Gravity model
    #[arg(long, value_enum, default_value = "j2")]
    pub gravity: DecayGravityType,
    /// Include solar radiation pressure
    #[arg(long, default_value_t = true)]
    pub srp: bool,
    /// Include third-body perturbations (Sun/Moon)
    #[arg(long, default_value_t = false)]
    pub third_body: bool,
    /// Ephemeris for third-body (low-precision or high-precision)
    /// High-precision requires ~100MB JPL DE440 ephemeris download
    #[arg(long, value_enum, default_value = "low-precision")]
    pub ephemeris: DecayEphemerisType,
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

#[derive(Args, Debug, Clone)]
pub struct ValueArgs {
    /// Path to TOML configuration file
    #[arg(long, short = 'c')]
    pub config: PathBuf,
    /// Output JSON file path
    #[arg(long, default_value = "out/value_scores.json")]
    pub output: PathBuf,
    /// NORAD IDs to process (overrides config filter)
    #[arg(long, value_delimiter = ',')]
    pub norad_ids: Vec<u32>,
    /// Pre-computed decay predictions JSON file
    #[arg(long)]
    pub decay_input: Option<PathBuf>,
    /// Pre-computed collision events JSON file
    #[arg(long)]
    pub collision_input: Option<PathBuf>,
    /// Max number of objects to process (0 = all)
    #[arg(long, default_value_t = 0)]
    pub limit: usize,
    /// Verbose output
    #[arg(long, short = 'v')]
    pub verbose: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum CollisionPropagatorType {
    /// SGP4/SDP4 analytical propagator (fast, good for short-term)
    Sgp4,
    /// Native RK4 numerical integrator (slower, more accurate)
    NativeRk4,
    /// Satkit RK9(8) numerical integrator (slowest, highest accuracy)
    SatkitRk98,
}

impl CollisionPropagatorType {
    fn is_hifi(self) -> bool {
        matches!(self, Self::NativeRk4 | Self::SatkitRk98)
    }

    fn to_propagator_type(self) -> Option<PropagatorType> {
        match self {
            Self::Sgp4 => None, // SGP4 uses different propagator
            Self::NativeRk4 => Some(PropagatorType::NativeRk4),
            Self::SatkitRk98 => Some(PropagatorType::SatkitRk98),
        }
    }
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
    Nrlmsise00Lookup,
    Jb2008Lookup,
    HarrisPriesterLookup,
    ExponentialLookup,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum AtmosphereLookupAccuracy {
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum DecayGravityType {
    PointMass,
    J2,
    FullField20,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum DecayEphemerisType {
    /// Fast analytical approximation (no external data)
    LowPrecision,
    /// JPL DE440 ephemeris (~100MB download, sub-arcsecond accuracy)
    HighPrecision,
}

impl DecayEphemerisType {
    fn to_ephemeris_type(self) -> EphemerisType {
        match self {
            Self::LowPrecision => EphemerisType::LowPrecision,
            Self::HighPrecision => EphemerisType::HighPrecision,
        }
    }
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
            Self::Nrlmsise00Lookup => AtmosphereModelType::Nrlmsise00Lookup,
            Self::Jb2008 => AtmosphereModelType::Jb2008,
            Self::Jb2008Lookup => AtmosphereModelType::Jb2008Lookup,
            Self::HarrisPriester => AtmosphereModelType::HarrisPriester,
            Self::HarrisPriesterLookup => AtmosphereModelType::HarrisPriesterLookup,
            Self::Exponential => AtmosphereModelType::Exponential,
            Self::ExponentialLookup => AtmosphereModelType::ExponentialLookup,
        }
    }
}

impl AtmosphereLookupAccuracy {
    fn to_lookup_accuracy(self) -> LookupAccuracy {
        match self {
            Self::Low => LookupAccuracy::Low,
            Self::Medium => LookupAccuracy::Medium,
            Self::High => LookupAccuracy::High,
        }
    }
}

impl From<LookupAccuracy> for AtmosphereLookupAccuracy {
    fn from(value: LookupAccuracy) -> Self {
        match value {
            LookupAccuracy::Low => Self::Low,
            LookupAccuracy::Medium => Self::Medium,
            LookupAccuracy::High => Self::High,
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

#[derive(Debug, Serialize, Deserialize, Clone)]
struct CollisionEvent {
    norad_a: u32,
    norad_b: u32,
    time_utc: String,
    distance_km: f64,
    relative_speed_kms: f64,
    score: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct CollisionDatabase {
    generated_at: String,
    start_time_utc: String,
    hours: f64,
    step_seconds: u64,
    distance_km: f64,
    #[serde(default)]
    propagator: String,
    #[serde(default)]
    atmosphere: Option<String>,
    #[serde(default)]
    gravity: Option<String>,
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
#[derive(Debug, Serialize, Deserialize, Clone)]
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
#[derive(Debug, Serialize, Deserialize)]
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

    // Initialize satkit data if using HiFi propagator
    if args.propagator.is_hifi() {
        ensure_satkit_data();
    }

    log::info!("Loading database...");
    let db_path = PathBuf::from("out/space_objects.json");
    let discos_path = PathBuf::from("data/cache/discos_objects_by_satno.json.gz");
    let database = load_complete_database(&db_path, &discos_path)?;

    let mut filter = ObjectFilter::default();
    filter.has_tle_only = true;
    filter.exclude_decayed = true;

    let mut ids = if !args.norad_ids.is_empty() {
        args.norad_ids.clone()
    } else {
        collect_ids(&database, &filter)
    };

    // Apply limit
    if args.limit > 0 && ids.len() > args.limit {
        ids.truncate(args.limit);
    }

    if ids.is_empty() {
        return Err(anyhow!("no objects matched filters"));
    }

    let steps = ((args.hours * 3600.0) / args.step_seconds as f64).ceil() as u64;
    let total_steps = steps + 1;
    let threshold_km = args.distance_km as f32;
    let threshold_sq = threshold_km * threshold_km;
    let cell_size = threshold_km.max(1.0);
    let inv_cell = 1.0 / cell_size;

    let mut events: HashMap<(u32, u32), CollisionEvent> = HashMap::new();

    // Get propagator name for output
    let propagator_name = match args.propagator {
        CollisionPropagatorType::Sgp4 => "SGP4",
        CollisionPropagatorType::NativeRk4 => "Native RK4",
        CollisionPropagatorType::SatkitRk98 => "Satkit RK9(8)",
    };

    let (atmosphere_name, gravity_name) = if args.propagator.is_hifi() {
        (
            Some(args.atmosphere.to_atmosphere_type().name().to_string()),
            Some(args.gravity.to_gravity_choice().name().to_string()),
        )
    } else {
        (None, None)
    };

    log::info!(
        "Scanning {} objects for {} hours ({} steps) using {}...",
        ids.len(),
        args.hours,
        total_steps,
        propagator_name
    );

    let start_time = satkit::Instant::now();

    // Dispatch to appropriate scan method based on propagator type
    if args.propagator.is_hifi() {
        run_collision_scan_hifi(
            &args,
            &database,
            &ids,
            total_steps,
            start_time,
            threshold_sq,
            inv_cell,
            &mut events,
        )?;
    } else {
        run_collision_scan_sgp4(
            &args,
            &database,
            &ids,
            total_steps,
            start_time,
            threshold_sq,
            inv_cell,
            &mut events,
        )?;
    }

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
        propagator: propagator_name.to_string(),
        atmosphere: atmosphere_name,
        gravity: gravity_name,
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
    log::info!("Found {} collision events", db.total_pairs);
    Ok(())
}

/// SGP4 collision scan - uses batch propagation (fast)
fn run_collision_scan_sgp4(
    args: &CollisionArgs,
    database: &SpaceObjectDatabase,
    ids: &[u32],
    total_steps: u64,
    start_time: satkit::Instant,
    threshold_sq: f32,
    inv_cell: f32,
    events: &mut HashMap<(u32, u32), CollisionEvent>,
) -> Result<()> {
    log::info!("Initializing SGP4 propagator...");
    let mut propagator = Propagator::new();
    propagator.load_tles(&database.objects);

    let progress = ProgressBar::new(total_steps);
    progress.set_style(
        ProgressStyle::with_template(
            "{elapsed_precise} {bar:40.cyan/blue} {pos}/{len} {percent}% ETA {eta_precise}",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    for step in 0..total_steps {
        let time =
            start_time + satkit::Duration::from_seconds(step as f64 * args.step_seconds as f64);
        propagator.set_time(time);
        let states = propagator.propagate_subset(ids);
        if states.is_empty() {
            progress.inc(1);
            continue;
        }

        let objects = build_object_states_sgp4(&states);
        scan_for_collisions(&objects, time, threshold_sq, inv_cell, events);
        progress.inc(1);
    }

    progress.finish_and_clear();
    Ok(())
}

/// HiFi collision scan - pre-computes trajectories in parallel, then scans
fn run_collision_scan_hifi(
    args: &CollisionArgs,
    database: &SpaceObjectDatabase,
    ids: &[u32],
    total_steps: u64,
    start_time: satkit::Instant,
    threshold_sq: f32,
    inv_cell: f32,
    events: &mut HashMap<(u32, u32), CollisionEvent>,
) -> Result<()> {
    // Build HiFi settings
    let config = PropagatorConfig {
        step_size: args.step_seconds as f64,
        tolerance: 1e-9,
        max_steps: 10_000_000,
        store_history: false,
        history_interval: 0.0,
        reentry_altitude: 70_000.0, // 70 km
        escape_altitude: 1_000_000_000.0,
    };

    let settings = Arc::new(HiFiSettings {
        propagator: args
            .propagator
            .to_propagator_type()
            .unwrap_or(PropagatorType::NativeRk4),
        integrator: match args.propagator {
            CollisionPropagatorType::NativeRk4 => IntegratorType::NativeRk4,
            CollisionPropagatorType::SatkitRk98 => IntegratorType::SatkitRk98,
            CollisionPropagatorType::Sgp4 => IntegratorType::NativeRk4, // shouldn't happen
        },
        atmosphere: args.atmosphere.to_atmosphere_type(),
        lookup_accuracy: args.lookup_accuracy.to_lookup_accuracy(),
        gravity: args.gravity.to_gravity_choice(),
        include_srp: args.srp,
        include_third_body: args.third_body,
        ephemeris: args.ephemeris.to_ephemeris_type(),
        config,
        decay_horizon_days: args.hours / 24.0,
    });

    // Generate time steps
    let time_steps: Vec<satkit::Instant> = (0..total_steps)
        .map(|step| {
            start_time + satkit::Duration::from_seconds(step as f64 * args.step_seconds as f64)
        })
        .collect();

    log::info!(
        "Pre-computing {} trajectories ({} time steps each) in parallel...",
        ids.len(),
        total_steps
    );

    // Pre-compute all trajectories in parallel
    // Each entry is (norad_id, Vec<Option<(pos_km, vel_km_s)>>)
    let database = Arc::new(database.clone());
    let time_steps = Arc::new(time_steps);

    let progress = Arc::new(ProgressBar::new(ids.len() as u64));
    progress.set_style(
        ProgressStyle::with_template(
            "{elapsed_precise} {bar:40.cyan/blue} {pos}/{len} {percent}% ETA {eta_precise} (propagating)",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    let trajectories: Vec<(u32, Vec<Option<(Vec3, Vec3)>>)> = ids
        .par_iter()
        .map(|norad_id| {
            let norad_str = norad_id.to_string();
            let obj = database.objects.get(&norad_str);

            // Get TLE
            let tle = match obj.and_then(|o| o.tle.as_ref()) {
                Some(tle) => tle,
                None => {
                    progress.inc(1);
                    return (*norad_id, vec![None; time_steps.len()]);
                }
            };

            // Parse TLE
            let satkit_tle = match satkit::TLE::load_2line(&tle.line1, &tle.line2) {
                Ok(t) => t,
                Err(_) => {
                    progress.inc(1);
                    return (*norad_id, vec![None; time_steps.len()]);
                }
            };

            // Build initial state
            let orbital_state = match orbital_state_from_tle(&satkit_tle, &start_time) {
                Some(s) => s,
                None => {
                    progress.inc(1);
                    return (*norad_id, vec![None; time_steps.len()]);
                }
            };

            // Get DISCOS data
            let (mass, area) = obj
                .and_then(|o| o.discos.as_ref())
                .map(|d| (d.mass, d.x_sect_avg))
                .unwrap_or((None, None));

            let initial_state = SpacecraftState::from_discos(orbital_state, mass, area);

            // Build propagator for this thread
            let propagator = match settings.build_propagator() {
                Some(p) => p,
                None => {
                    progress.inc(1);
                    return (*norad_id, vec![None; time_steps.len()]);
                }
            };

            // Propagate to each time step
            let mut positions: Vec<Option<(Vec3, Vec3)>> = Vec::with_capacity(time_steps.len());
            let mut current_state = initial_state;

            for (i, &target_time) in time_steps.iter().enumerate() {
                if i == 0 {
                    // First step - use initial state
                    let pos_km = Vec3::new(
                        current_state.orbital.position.x as f32 / 1000.0,
                        current_state.orbital.position.y as f32 / 1000.0,
                        current_state.orbital.position.z as f32 / 1000.0,
                    );
                    let vel_kms = Vec3::new(
                        current_state.orbital.velocity.x as f32 / 1000.0,
                        current_state.orbital.velocity.y as f32 / 1000.0,
                        current_state.orbital.velocity.z as f32 / 1000.0,
                    );
                    positions.push(Some((pos_km, vel_kms)));
                    continue;
                }

                // Propagate from current state to target time
                match propagator.propagate(current_state.clone(), target_time) {
                    Ok(result) => {
                        current_state = result.final_state;
                        let pos_km = Vec3::new(
                            current_state.orbital.position.x as f32 / 1000.0,
                            current_state.orbital.position.y as f32 / 1000.0,
                            current_state.orbital.position.z as f32 / 1000.0,
                        );
                        let vel_kms = Vec3::new(
                            current_state.orbital.velocity.x as f32 / 1000.0,
                            current_state.orbital.velocity.y as f32 / 1000.0,
                            current_state.orbital.velocity.z as f32 / 1000.0,
                        );
                        positions.push(Some((pos_km, vel_kms)));
                    }
                    Err(_) => {
                        // Satellite decayed or other error - no more positions
                        positions.push(None);
                        // Fill rest with None
                        while positions.len() < time_steps.len() {
                            positions.push(None);
                        }
                        break;
                    }
                }
            }

            progress.inc(1);
            (*norad_id, positions)
        })
        .collect();

    progress.finish_and_clear();

    log::info!("Scanning for collisions...");

    let scan_progress = ProgressBar::new(total_steps);
    scan_progress.set_style(
        ProgressStyle::with_template(
            "{elapsed_precise} {bar:40.cyan/blue} {pos}/{len} {percent}% ETA {eta_precise} (scanning)",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    // Scan for collisions at each time step
    for step in 0..total_steps as usize {
        let time = time_steps[step];

        // Build object states for this time step
        let mut objects: Vec<ObjectState> = Vec::with_capacity(trajectories.len());
        for (norad_id, positions) in &trajectories {
            if let Some(Some((pos_km, vel_kms))) = positions.get(step) {
                objects.push(ObjectState {
                    norad: *norad_id,
                    pos_km: *pos_km,
                    vel_kms: *vel_kms,
                });
            }
        }

        if !objects.is_empty() {
            scan_for_collisions(&objects, time, threshold_sq, inv_cell, events);
        }

        scan_progress.inc(1);
    }

    scan_progress.finish_and_clear();
    Ok(())
}

/// Build object states from SGP4 propagation results
fn build_object_states_sgp4(
    states: &HashMap<u32, crate::propagation::SatelliteState>,
) -> Vec<ObjectState> {
    let mut objects: Vec<ObjectState> = Vec::with_capacity(states.len());
    for (norad, state) in states.iter() {
        let pos_km = state.position * EARTH_RADIUS_KM as f32;
        objects.push(ObjectState {
            norad: *norad,
            pos_km,
            vel_kms: state.velocity,
        });
    }
    objects
}

fn make_progress_bar(
    len: u64,
    template: &str,
    multi_progress: Option<&MultiProgress>,
) -> ProgressBar {
    let bar = ProgressBar::new(len);
    let bar = if let Some(multi) = multi_progress {
        multi.add(bar)
    } else {
        bar
    };
    bar.set_style(
        ProgressStyle::with_template(template)
            .unwrap()
            .progress_chars("##-"),
    );
    bar
}

fn record_collision(
    a: &ObjectState,
    b: &ObjectState,
    time: satkit::Instant,
    threshold_sq: f32,
    events: &mut HashMap<(u32, u32), CollisionEvent>,
) {
    let delta = a.pos_km - b.pos_km;
    let dist_sq = delta.length_squared();
    if dist_sq > threshold_sq {
        return;
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

/// Scan for collisions using spatial hashing
fn scan_for_collisions(
    objects: &[ObjectState],
    time: satkit::Instant,
    threshold_sq: f32,
    inv_cell: f32,
    events: &mut HashMap<(u32, u32), CollisionEvent>,
) {
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

                    for (pos_i, &i) in list_a.iter().enumerate() {
                        let start_pos = if neighbor == key { pos_i + 1 } else { 0 };
                        for &j in list_b.iter().skip(start_pos) {
                            record_collision(&objects[i], &objects[j], time, threshold_sq, events);
                        }
                    }
                }
            }
        }
    }
}

#[derive(Default)]
struct CellLists {
    all: Vec<usize>,
    targets: Vec<usize>,
    non_targets: Vec<usize>,
}

/// Scan for collisions, keeping only pairs where at least one object is in target_ids
fn scan_targets_vs_catalog(
    objects: &[ObjectState],
    target_ids: &HashSet<u32>,
    time: satkit::Instant,
    threshold_sq: f32,
    inv_cell: f32,
    events: &mut HashMap<(u32, u32), CollisionEvent>,
) {
    let mut cells: HashMap<(i32, i32, i32), CellLists> = HashMap::new();
    let mut is_target = vec![false; objects.len()];

    for (idx, obj) in objects.iter().enumerate() {
        let key = cell_key(obj.pos_km, inv_cell);
        let target = target_ids.contains(&obj.norad);
        is_target[idx] = target;

        let entry = cells.entry(key).or_default();
        entry.all.push(idx);
        if target {
            entry.targets.push(idx);
        } else {
            entry.non_targets.push(idx);
        }
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

                    if neighbor == key {
                        let list = &list_a.all;
                        for (pos_i, &i) in list.iter().enumerate() {
                            for &j in list.iter().skip(pos_i + 1) {
                                if !is_target[i] && !is_target[j] {
                                    continue;
                                }
                                record_collision(
                                    &objects[i],
                                    &objects[j],
                                    time,
                                    threshold_sq,
                                    events,
                                );
                            }
                        }
                        continue;
                    }

                    for &i in &list_a.targets {
                        for &j in &list_b.targets {
                            record_collision(&objects[i], &objects[j], time, threshold_sq, events);
                        }
                        for &j in &list_b.non_targets {
                            record_collision(&objects[i], &objects[j], time, threshold_sq, events);
                        }
                    }

                    for &j in &list_b.targets {
                        for &i in &list_a.non_targets {
                            record_collision(&objects[i], &objects[j], time, threshold_sq, events);
                        }
                    }
                }
            }
        }
    }
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
        lookup_accuracy: args.lookup_accuracy.to_lookup_accuracy(),
        gravity: args.gravity.to_gravity_choice(),
        include_srp: args.srp,
        include_third_body: args.third_body,
        ephemeris: args.ephemeris.to_ephemeris_type(),
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
    if settings.include_third_body {
        log::info!("Third-body ephemeris: {}", settings.ephemeris.name());
    }

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
            "{elapsed_precise} {bar:40.cyan/blue} {pos}/{len} {percent}% ETA {eta_precise} ({per_sec})",
        )
        .unwrap()
        .progress_chars("##-"),
    );

    // Shared state for progress tracking
    let progress = Arc::new(progress);
    let database = Arc::new(database);
    let settings = Arc::new(settings);

    log::info!(
        "Starting parallel decay prediction with {} threads",
        rayon::current_num_threads()
    );

    // Process satellites in parallel
    let predictions: Vec<DecayPrediction> = ids
        .par_iter()
        .map(|norad_id| {
            let norad_str = norad_id.to_string();
            let obj = database.objects.get(&norad_str);

            // Get object metadata
            let name = obj.map(|o| o.display_name());
            let object_type = obj.and_then(|o| o.object_type.clone());
            let country = obj.and_then(|o| o.country.clone());
            let launch_date = obj.and_then(|o| o.launch_date.clone());

            // Get TLE
            let tle = match obj.and_then(|o| o.tle.as_ref()) {
                Some(tle) => tle,
                None => {
                    progress.inc(1);
                    return DecayPrediction {
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
                    };
                }
            };

            // Parse TLE
            let satkit_tle = match satkit::TLE::load_2line(&tle.line1, &tle.line2) {
                Ok(t) => t,
                Err(e) => {
                    progress.inc(1);
                    return DecayPrediction {
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
                    };
                }
            };

            // Build initial state
            let orbital_state = match orbital_state_from_tle(&satkit_tle, &start_time) {
                Some(s) => s,
                None => {
                    progress.inc(1);
                    return DecayPrediction {
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
                    };
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

            // Build a propagator for this thread (each thread gets its own)
            let propagator = match settings.build_propagator() {
                Some(p) => p,
                None => {
                    progress.inc(1);
                    return DecayPrediction {
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
                        error_message: Some("SGP4 cannot be used for decay prediction".to_string()),
                        propagation_seconds: 0.0,
                    };
                }
            };

            // Run propagation
            let prop_start = std::time::Instant::now();
            let result = propagator.propagate(initial_state, target_epoch);
            let propagation_seconds = prop_start.elapsed().as_secs_f64();

            progress.inc(1);

            match result {
                Ok(_) => {
                    // Satellite survived the entire horizon without decay
                    DecayPrediction {
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
                    }
                }
                Err(crate::propagation::hifi::PropagationError::Reentry { altitude_km, epoch }) => {
                    // Satellite decayed!
                    let days_to_decay = (epoch - start_time).as_seconds() / 86400.0;
                    DecayPrediction {
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
                    }
                }
                Err(e) => {
                    // Other error (escape, integration failure, etc.)
                    DecayPrediction {
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
                    }
                }
            }
        })
        .collect();

    progress.finish_and_clear();

    // Count results
    let decayed_count = predictions.iter().filter(|p| p.status == "decayed").count();
    let survived_count = predictions
        .iter()
        .filter(|p| p.status == "survived")
        .count();
    let error_count = predictions
        .iter()
        .filter(|p| p.status == "error" || p.status == "no_tle")
        .count();

    // Sort predictions: decayed first (by days_to_decay), then survived, then errors
    let mut predictions = predictions;
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

// ============================================================================
// Value Analysis - Debris Deorbit Prioritization
// ============================================================================

/// TOML configuration for value analysis
#[derive(Debug, Clone, Deserialize)]
pub struct ValueConfig {
    pub metadata: ValueMetadata,
    pub filter: FilterConfig,
    pub decay: DecayConfig,
    pub collision: CollisionConfig,
    pub scoring: ScoringConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ValueMetadata {
    pub name: String,
    #[serde(default = "default_version")]
    pub version: String,
}

fn default_version() -> String {
    "1.0".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct FilterConfig {
    #[serde(default)]
    pub object_types: Vec<String>,
    #[serde(default)]
    pub countries: Vec<String>,
    #[serde(default = "default_true")]
    pub has_tle_only: bool,
    #[serde(default = "default_true")]
    pub exclude_decayed: bool,
    #[serde(default)]
    pub altitude_filter_enabled: bool,
    #[serde(default = "default_altitude_min")]
    pub altitude_min_km: f64,
    #[serde(default = "default_altitude_max")]
    pub altitude_max_km: f64,
    #[serde(default)]
    pub size_filter_enabled: bool,
    #[serde(default)]
    pub size_min_m: f64,
    #[serde(default)]
    pub size_max_m: f64,
    #[serde(default = "default_true")]
    pub include_unknown_size: bool,
    #[serde(default)]
    pub mass_filter_enabled: bool,
    #[serde(default)]
    pub mass_min_kg: f64,
    #[serde(default)]
    pub mass_max_kg: f64,
    #[serde(default = "default_true")]
    pub include_unknown_mass: bool,
}

fn default_true() -> bool {
    true
}

fn default_altitude_min() -> f64 {
    200.0
}

fn default_altitude_max() -> f64 {
    2000.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct DecayConfig {
    #[serde(default = "default_decay_propagator")]
    pub propagator: String,
    #[serde(default = "default_atmosphere")]
    pub atmosphere: String,
    #[serde(default = "default_lookup_accuracy")]
    pub lookup_accuracy: String,
    #[serde(default = "default_gravity")]
    pub gravity: String,
    #[serde(default = "default_decay_horizon")]
    pub horizon_days: f64,
    #[serde(default = "default_true")]
    pub include_srp: bool,
    #[serde(default)]
    pub include_third_body: bool,
}

fn default_decay_propagator() -> String {
    "satkit-rk98".to_string()
}

fn default_atmosphere() -> String {
    "nrlmsise00".to_string()
}

fn default_lookup_accuracy() -> String {
    "medium".to_string()
}

fn default_gravity() -> String {
    "j2".to_string()
}

fn default_decay_horizon() -> f64 {
    30.0 // 30 days default; increase to 3650.0 for production analysis
}

#[derive(Debug, Clone, Deserialize)]
pub struct CollisionConfig {
    #[serde(default = "default_collision_propagator")]
    pub propagator: String,
    #[serde(default = "default_collision_hours")]
    pub hours: f64,
    #[serde(default = "default_distance_threshold")]
    pub distance_threshold_km: f64,
    #[serde(default = "default_collision_atmosphere")]
    pub atmosphere: String,
    #[serde(default = "default_lookup_accuracy")]
    pub lookup_accuracy: String,
}

fn default_collision_propagator() -> String {
    "satkit-rk98".to_string()
}

fn default_collision_hours() -> f64 {
    168.0
}

fn default_distance_threshold() -> f64 {
    10.0
}

fn default_collision_atmosphere() -> String {
    "harris-priester".to_string()
}

fn parse_lookup_accuracy(scope: &str, value: &str) -> LookupAccuracy {
    match value.parse::<LookupAccuracy>() {
        Ok(acc) => acc,
        Err(_) => {
            log::warn!(
                "Unknown {} lookup accuracy '{}', using medium",
                scope,
                value
            );
            LookupAccuracy::Medium
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScoringConfig {
    pub weights: ScoringWeights,
    #[serde(default)]
    pub altitude_bands: AltitudeBandsConfig,
    #[serde(default)]
    pub object_type: ObjectTypeScores,
    #[serde(default)]
    pub missing_data: MissingDataConfig,
    #[serde(default)]
    pub decay_timeline: DecayTimelineConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ScoringWeights {
    #[serde(default = "default_mass_weight")]
    pub mass_size: f64,
    #[serde(default = "default_collision_weight")]
    pub collision_risk: f64,
    #[serde(default = "default_decay_weight")]
    pub decay_timeline: f64,
    #[serde(default = "default_object_type_weight")]
    pub object_type: f64,
    #[serde(default = "default_country_weight")]
    pub country_operator: f64,
}

fn default_mass_weight() -> f64 {
    0.25
}
fn default_collision_weight() -> f64 {
    0.35
}
fn default_decay_weight() -> f64 {
    0.25
}
fn default_object_type_weight() -> f64 {
    0.10
}
fn default_country_weight() -> f64 {
    0.05
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct AltitudeBandsConfig {
    #[serde(default = "default_altitude_bands")]
    pub bands: Vec<AltitudeBand>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AltitudeBand {
    pub min: f64,
    pub max: f64,
    pub weight: f64,
    pub name: String,
}

fn default_altitude_bands() -> Vec<AltitudeBand> {
    vec![
        AltitudeBand {
            min: 200.0,
            max: 450.0,
            weight: 1.5,
            name: "Very Low LEO".to_string(),
        },
        AltitudeBand {
            min: 450.0,
            max: 600.0,
            weight: 2.0,
            name: "Dense LEO (Starlink)".to_string(),
        },
        AltitudeBand {
            min: 600.0,
            max: 800.0,
            weight: 1.8,
            name: "Sun-Sync LEO".to_string(),
        },
        AltitudeBand {
            min: 800.0,
            max: 2000.0,
            weight: 1.2,
            name: "Upper LEO".to_string(),
        },
    ]
}

#[derive(Debug, Clone, Deserialize)]
pub struct ObjectTypeScores {
    #[serde(default = "default_payload_score")]
    pub payload: f64,
    #[serde(default = "default_rocket_body_score")]
    pub rocket_body: f64,
    #[serde(default = "default_debris_score")]
    pub debris: f64,
    #[serde(default = "default_unknown_score")]
    pub unknown: f64,
}

impl Default for ObjectTypeScores {
    fn default() -> Self {
        Self {
            payload: default_payload_score(),
            rocket_body: default_rocket_body_score(),
            debris: default_debris_score(),
            unknown: default_unknown_score(),
        }
    }
}

fn default_payload_score() -> f64 {
    0.3
}
fn default_rocket_body_score() -> f64 {
    1.0
}
fn default_debris_score() -> f64 {
    0.5
}
fn default_unknown_score() -> f64 {
    0.4
}

#[derive(Debug, Clone, Deserialize)]
pub struct MissingDataConfig {
    #[serde(default = "default_missing_strategy")]
    pub strategy: String,
    #[serde(default = "default_penalty_factor")]
    pub penalty_factor: f64,
    #[serde(default = "default_median_mass")]
    pub median_mass_kg: f64,
    #[serde(default = "default_median_size")]
    pub median_size_m: f64,
}

impl Default for MissingDataConfig {
    fn default() -> Self {
        Self {
            strategy: default_missing_strategy(),
            penalty_factor: default_penalty_factor(),
            median_mass_kg: default_median_mass(),
            median_size_m: default_median_size(),
        }
    }
}

fn default_missing_strategy() -> String {
    "penalize".to_string()
}
fn default_penalty_factor() -> f64 {
    0.3
}
fn default_median_mass() -> f64 {
    100.0
}
fn default_median_size() -> f64 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct DecayTimelineConfig {
    #[serde(default = "default_short_decay_days")]
    pub short_decay_days: f64,
    #[serde(default = "default_medium_decay_days")]
    pub medium_decay_days: f64,
    #[serde(default = "default_short_decay_score")]
    pub short_decay_score: f64,
    #[serde(default = "default_medium_decay_score")]
    pub medium_decay_score: f64,
    #[serde(default = "default_long_decay_score")]
    pub long_decay_score: f64,
}

impl Default for DecayTimelineConfig {
    fn default() -> Self {
        Self {
            short_decay_days: default_short_decay_days(),
            medium_decay_days: default_medium_decay_days(),
            short_decay_score: default_short_decay_score(),
            medium_decay_score: default_medium_decay_score(),
            long_decay_score: default_long_decay_score(),
        }
    }
}

fn default_short_decay_days() -> f64 {
    365.0
}
fn default_medium_decay_days() -> f64 {
    3650.0
}
fn default_short_decay_score() -> f64 {
    0.1
}
fn default_medium_decay_score() -> f64 {
    0.5
}
fn default_long_decay_score() -> f64 {
    1.0
}

impl ValueConfig {
    /// Load configuration from a TOML file
    pub fn load(path: &PathBuf) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow!("Failed to read config file {:?}: {}", path, e))?;
        let config: ValueConfig = toml::from_str(&content)
            .map_err(|e| anyhow!("Failed to parse config file {:?}: {}", path, e))?;
        Ok(config)
    }

    /// Build an ObjectFilter from the filter config
    pub fn build_object_filter(&self) -> ObjectFilter {
        ObjectFilter {
            object_types: self.filter.object_types.clone(),
            countries: self.filter.countries.clone(),
            has_tle_only: self.filter.has_tle_only,
            exclude_decayed: self.filter.exclude_decayed,
            size_filter_enabled: self.filter.size_filter_enabled,
            size_min_m: self.filter.size_min_m,
            size_max_m: self.filter.size_max_m,
            include_unknown_size: self.filter.include_unknown_size,
            mass_filter_enabled: self.filter.mass_filter_enabled,
            mass_min_kg: self.filter.mass_min_kg,
            mass_max_kg: self.filter.mass_max_kg,
            include_unknown_mass: self.filter.include_unknown_mass,
        }
    }
}

/// Summary of collision events for a single object
#[derive(Debug, Clone, Default)]
struct CollisionSummary {
    approach_count: usize,
    min_distance_km: f64,
    max_relative_speed_kms: f64,
    total_score: f64,
}

/// Output structures for value analysis
#[derive(Debug, Serialize)]
pub struct ValueDatabase {
    pub metadata: ValueOutputMetadata,
    pub summary: ValueSummary,
    pub objects: Vec<ScoredObject>,
}

#[derive(Debug, Serialize)]
pub struct ValueOutputMetadata {
    pub generated_at: String,
    pub config_name: String,
    pub config_version: String,
    pub total_processing_time_seconds: f64,
}

#[derive(Debug, Serialize)]
pub struct ValueSummary {
    pub total_objects_analyzed: usize,
    pub objects_with_collisions: usize,
    pub objects_decayed: usize,
    pub average_score: f64,
    pub max_score: f64,
}

#[derive(Debug, Serialize)]
pub struct ScoredObject {
    pub norad_id: u32,
    pub name: String,
    pub object_type: Option<String>,
    pub country: Option<String>,
    pub altitude_km: f64,
    pub mass_kg: Option<f64>,
    pub collision_count: usize,
    pub decay_status: String,
    pub days_to_decay: Option<f64>,
    pub scores: ValueScore,
    pub rank: usize,
}

#[derive(Debug, Serialize)]
pub struct ValueScore {
    pub mass_size_score: f64,
    pub collision_risk_score: f64,
    pub decay_timeline_score: f64,
    pub object_type_score: f64,
    pub country_score: f64,
    pub altitude_multiplier: f64,
    pub altitude_band: String,
    pub total_score: f64,
}

/// Value scorer that computes scores for debris objects
struct ValueScorer<'a> {
    config: &'a ScoringConfig,
    max_collision_count: usize,
    max_collision_score: f64,
}

impl<'a> ValueScorer<'a> {
    fn new(config: &'a ScoringConfig, collision_summaries: &HashMap<u32, CollisionSummary>) -> Self {
        let max_collision_count = collision_summaries
            .values()
            .map(|s| s.approach_count)
            .max()
            .unwrap_or(1)
            .max(1);
        let max_collision_score = collision_summaries
            .values()
            .map(|s| s.total_score)
            .fold(1.0, f64::max);

        Self {
            config,
            max_collision_count,
            max_collision_score,
        }
    }

    fn compute_score(
        &self,
        obj: &SpaceObject,
        altitude_km: f64,
        collision_summary: Option<&CollisionSummary>,
        decay_prediction: Option<&DecayPrediction>,
    ) -> ValueScore {
        // 1. Mass/Size score
        let mass_size_score = self.compute_mass_size_score(obj);

        // 2. Collision risk score
        let collision_risk_score = self.compute_collision_risk_score(collision_summary);

        // 3. Decay timeline score
        let decay_timeline_score = self.compute_decay_timeline_score(decay_prediction);

        // 4. Object type score
        let object_type_score = self.compute_object_type_score(obj);

        // 5. Country score (default 1.0, no geopolitical weighting by default)
        let country_score = 1.0;

        // Altitude multiplier
        let (altitude_multiplier, altitude_band) = self.get_altitude_multiplier(altitude_km);

        // Compute base score as weighted sum
        let weights = &self.config.weights;
        let base_score = weights.mass_size * mass_size_score
            + weights.collision_risk * collision_risk_score
            + weights.decay_timeline * decay_timeline_score
            + weights.object_type * object_type_score
            + weights.country_operator * country_score;

        // Final score (0-100 range)
        let total_score = (altitude_multiplier * base_score * 100.0).clamp(0.0, 100.0);

        ValueScore {
            mass_size_score,
            collision_risk_score,
            decay_timeline_score,
            object_type_score,
            country_score,
            altitude_multiplier,
            altitude_band,
            total_score,
        }
    }

    fn compute_mass_size_score(&self, obj: &SpaceObject) -> f64 {
        if let Some(discos) = &obj.discos {
            if let Some(mass) = discos.mass {
                // Log-normalized mass score
                // Assuming max debris mass ~10000 kg, min ~1 kg
                let log_mass = (mass.max(1.0)).log10();
                let normalized = (log_mass / 4.0).clamp(0.0, 1.0); // log10(10000) = 4

                // Add size factor if available
                let size_factor = if let Some(area) = discos.cross_section_m2() {
                    // Normalize cross-section (assuming max ~100 m²)
                    (area / 100.0).clamp(0.0, 0.5)
                } else {
                    0.0
                };

                return (normalized + size_factor).clamp(0.0, 1.0);
            }
        }

        // Handle missing data
        match self.config.missing_data.strategy.as_str() {
            "median" => {
                let log_mass = self.config.missing_data.median_mass_kg.log10();
                (log_mass / 4.0).clamp(0.0, 1.0)
            }
            "exclude" => 0.0,
            _ => self.config.missing_data.penalty_factor,
        }
    }

    fn compute_collision_risk_score(&self, summary: Option<&CollisionSummary>) -> f64 {
        match summary {
            Some(s) if s.approach_count > 0 => {
                // Normalize approach count
                let count_factor =
                    (s.approach_count as f64) / (self.max_collision_count as f64).max(1.0);

                // Velocity factor (higher relative speed = more dangerous)
                // Normalize to typical LEO collision speeds (0-15 km/s)
                let velocity_factor = (s.max_relative_speed_kms / 15.0).clamp(0.0, 1.0);

                // Distance factor (closer = more dangerous)
                // Inverse relationship: smaller distance = higher score
                let distance_factor = (1.0 - (s.min_distance_km / 10.0).clamp(0.0, 1.0)).max(0.0);

                // Combined score with score normalization
                let score_factor = s.total_score / self.max_collision_score;

                // Weight the factors
                (0.4 * count_factor + 0.3 * velocity_factor + 0.2 * distance_factor + 0.1 * score_factor)
                    .clamp(0.0, 1.0)
            }
            _ => 0.0,
        }
    }

    fn compute_decay_timeline_score(&self, prediction: Option<&DecayPrediction>) -> f64 {
        let timeline = &self.config.decay_timeline;

        match prediction {
            Some(p) => match p.status.as_str() {
                "decayed" => {
                    if let Some(days) = p.days_to_decay {
                        if days <= timeline.short_decay_days {
                            timeline.short_decay_score
                        } else if days <= timeline.medium_decay_days {
                            timeline.medium_decay_score
                        } else {
                            timeline.long_decay_score
                        }
                    } else {
                        timeline.medium_decay_score
                    }
                }
                "survived" => timeline.long_decay_score, // Won't decay naturally
                _ => timeline.medium_decay_score,         // Error or unknown
            },
            None => timeline.long_decay_score, // No prediction = assume won't decay
        }
    }

    fn compute_object_type_score(&self, obj: &SpaceObject) -> f64 {
        let obj_type = obj.object_type.as_deref().unwrap_or("UNKNOWN");
        let scores = &self.config.object_type;

        match obj_type.to_uppercase().as_str() {
            "PAYLOAD" => scores.payload,
            "ROCKET BODY" | "R/B" => scores.rocket_body,
            "DEBRIS" | "DEB" => scores.debris,
            _ => scores.unknown,
        }
    }

    fn get_altitude_multiplier(&self, altitude_km: f64) -> (f64, String) {
        for band in &self.config.altitude_bands.bands {
            if altitude_km >= band.min && altitude_km < band.max {
                return (band.weight, band.name.clone());
            }
        }
        (1.0, "Other".to_string())
    }
}

/// Run the value analysis
pub fn run_value_analysis(args: ValueArgs) -> Result<()> {
    let start_time = std::time::Instant::now();

    // Load config
    log::info!("Loading configuration from {:?}...", args.config);
    let config = ValueConfig::load(&args.config)?;

    if args.verbose {
        log::info!("Config: {} v{}", config.metadata.name, config.metadata.version);
    }

    // Load database
    log::info!("Loading database...");
    let db_path = PathBuf::from("out/space_objects.json");
    let discos_path = PathBuf::from("data/cache/discos_objects_by_satno.json.gz");
    let database = load_complete_database(&db_path, &discos_path)?;

    // Build filter and collect IDs
    let filter = config.build_object_filter();
    let mut ids: Vec<u32> = if !args.norad_ids.is_empty() {
        args.norad_ids.clone()
    } else {
        collect_ids(&database, &filter)
    };

    // Apply limit
    if args.limit > 0 && ids.len() > args.limit {
        ids.truncate(args.limit);
    }

    if ids.is_empty() {
        return Err(anyhow!("No objects matched filters"));
    }

    log::info!("Processing {} objects...", ids.len());

    let multi_progress = Arc::new(MultiProgress::new());

    let (decay_predictions, collision_summaries) =
        std::thread::scope(|s| -> Result<(HashMap<u32, DecayPrediction>, HashMap<u32, CollisionSummary>)> {
            let decay_handle = s.spawn(|| {
                if let Some(decay_path) = &args.decay_input {
                    log::info!("Loading decay predictions from {:?}...", decay_path);
                    load_decay_predictions(decay_path)
                } else {
                    log::info!("Running decay predictions...");
                    run_decay_for_value(
                        &config,
                        &database,
                        &ids,
                        args.verbose,
                        Some(multi_progress.clone()),
                    )
                }
            });

            let collision_handle = s.spawn(|| {
                if let Some(collision_path) = &args.collision_input {
                    log::info!("Loading collision events from {:?}...", collision_path);
                    load_collision_summaries(collision_path)
                } else {
                    log::info!("Running collision scan...");
                    run_collisions_for_value(
                        &config,
                        &database,
                        &ids,
                        args.verbose,
                        Some(multi_progress.clone()),
                    )
                }
            });

            let decay_predictions = decay_handle
                .join()
                .map_err(|_| anyhow!("Decay thread panicked"))??;
            let collision_summaries = collision_handle
                .join()
                .map_err(|_| anyhow!("Collision thread panicked"))??;

            Ok((decay_predictions, collision_summaries))
        })?;

    // Initialize propagator for altitude calculations
    let mut propagator = Propagator::new();
    propagator.load_tles(&database.objects);
    let current_time = satkit::Instant::now();

    // Score each object
    log::info!("Scoring objects...");

    let progress = make_progress_bar(
        ids.len() as u64,
        "{elapsed_precise} {bar:40.cyan/blue} {pos}/{len} {percent}% ETA {eta_precise}",
        Some(multi_progress.as_ref()),
    );

    let database_arc = Arc::new(database);
    let decay_predictions_arc = Arc::new(decay_predictions);
    let collision_summaries_arc = Arc::new(collision_summaries);
    let config_arc = Arc::new(config.clone());
    let progress_arc = Arc::new(progress);

    let mut scored_objects: Vec<ScoredObject> = ids
        .par_iter()
        .filter_map(|norad_id| {
            let norad_str = norad_id.to_string();
            let obj = database_arc.objects.get(&norad_str)?;

            // Get altitude from TLE
            let altitude_km = if let Some(tle) = &obj.tle {
                if let Ok(satkit_tle) = satkit::TLE::load_2line(&tle.line1, &tle.line2) {
                    if let Some(state) = orbital_state_from_tle(&satkit_tle, &current_time) {
                        state.altitude() / 1000.0
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            } else {
                return None;
            };

            // Apply altitude filter if enabled
            if config_arc.filter.altitude_filter_enabled {
                if altitude_km < config_arc.filter.altitude_min_km
                    || altitude_km > config_arc.filter.altitude_max_km
                {
                    progress_arc.inc(1);
                    return None;
                }
            }

            let decay_pred = decay_predictions_arc.get(norad_id);
            let collision_sum = collision_summaries_arc.get(norad_id);

            let scorer = ValueScorer::new(&config_arc.scoring, &collision_summaries_arc);
            let scores = scorer.compute_score(obj, altitude_km, collision_sum, decay_pred);

            let scored = ScoredObject {
                norad_id: *norad_id,
                name: obj.display_name(),
                object_type: obj.object_type.clone(),
                country: obj.country.clone(),
                altitude_km,
                mass_kg: obj.discos.as_ref().and_then(|d| d.mass),
                collision_count: collision_sum.map(|s| s.approach_count).unwrap_or(0),
                decay_status: decay_pred
                    .map(|p| p.status.clone())
                    .unwrap_or_else(|| "unknown".to_string()),
                days_to_decay: decay_pred.and_then(|p| p.days_to_decay),
                scores,
                rank: 0, // Will be set after sorting
            };

            progress_arc.inc(1);
            Some(scored)
        })
        .collect();

    progress_arc.finish_and_clear();

    // Sort by total score descending and assign ranks
    scored_objects.sort_by(|a, b| {
        b.scores
            .total_score
            .partial_cmp(&a.scores.total_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (i, obj) in scored_objects.iter_mut().enumerate() {
        obj.rank = i + 1;
    }

    // Compute summary statistics
    let total_objects = scored_objects.len();
    let objects_with_collisions = scored_objects.iter().filter(|o| o.collision_count > 0).count();
    let objects_decayed = scored_objects
        .iter()
        .filter(|o| o.decay_status == "decayed")
        .count();
    let average_score = if total_objects > 0 {
        scored_objects.iter().map(|o| o.scores.total_score).sum::<f64>() / total_objects as f64
    } else {
        0.0
    };
    let max_score = scored_objects
        .first()
        .map(|o| o.scores.total_score)
        .unwrap_or(0.0);

    let processing_time = start_time.elapsed().as_secs_f64();

    let output = ValueDatabase {
        metadata: ValueOutputMetadata {
            generated_at: chrono::Utc::now().to_rfc3339(),
            config_name: config.metadata.name,
            config_version: config.metadata.version,
            total_processing_time_seconds: processing_time,
        },
        summary: ValueSummary {
            total_objects_analyzed: total_objects,
            objects_with_collisions,
            objects_decayed,
            average_score,
            max_score,
        },
        objects: scored_objects,
    };

    // Write output
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(&args.output)?;
    serde_json::to_writer_pretty(file, &output)?;

    log::info!("Wrote value scores to {:?}", args.output);
    log::info!(
        "Results: {} objects scored, avg={:.1}, max={:.1}, {} with collisions, {} decayed",
        total_objects,
        average_score,
        max_score,
        objects_with_collisions,
        objects_decayed
    );

    if args.verbose && !output.objects.is_empty() {
        log::info!("Top 5 highest priority debris:");
        for obj in output.objects.iter().take(5) {
            log::info!(
                "  #{}: {} (NORAD {}) - Score: {:.1}, Alt: {:.0}km, Collisions: {}",
                obj.rank,
                obj.name,
                obj.norad_id,
                obj.scores.total_score,
                obj.altitude_km,
                obj.collision_count
            );
        }
    }

    Ok(())
}

/// Load pre-computed decay predictions from JSON file
fn load_decay_predictions(path: &PathBuf) -> Result<HashMap<u32, DecayPrediction>> {
    let file = std::fs::File::open(path)
        .map_err(|e| anyhow!("Failed to open decay predictions file: {}", e))?;
    let db: DecayDatabase = serde_json::from_reader(file)
        .map_err(|e| anyhow!("Failed to parse decay predictions: {}", e))?;

    let mut map = HashMap::new();
    for pred in db.predictions {
        map.insert(pred.norad_id, pred);
    }
    Ok(map)
}

/// Load pre-computed collision events and summarize per object
fn load_collision_summaries(path: &PathBuf) -> Result<HashMap<u32, CollisionSummary>> {
    let file = std::fs::File::open(path)
        .map_err(|e| anyhow!("Failed to open collision events file: {}", e))?;
    let db: CollisionDatabase = serde_json::from_reader(file)
        .map_err(|e| anyhow!("Failed to parse collision events: {}", e))?;

    let mut summaries: HashMap<u32, CollisionSummary> = HashMap::new();

    for event in &db.events {
        // Update summary for object A
        let summary_a = summaries.entry(event.norad_a).or_default();
        summary_a.approach_count += 1;
        if summary_a.min_distance_km == 0.0 || event.distance_km < summary_a.min_distance_km {
            summary_a.min_distance_km = event.distance_km;
        }
        if event.relative_speed_kms > summary_a.max_relative_speed_kms {
            summary_a.max_relative_speed_kms = event.relative_speed_kms;
        }
        summary_a.total_score += event.score;

        // Update summary for object B
        let summary_b = summaries.entry(event.norad_b).or_default();
        summary_b.approach_count += 1;
        if summary_b.min_distance_km == 0.0 || event.distance_km < summary_b.min_distance_km {
            summary_b.min_distance_km = event.distance_km;
        }
        if event.relative_speed_kms > summary_b.max_relative_speed_kms {
            summary_b.max_relative_speed_kms = event.relative_speed_kms;
        }
        summary_b.total_score += event.score;
    }

    Ok(summaries)
}

/// Run decay predictions for value analysis
fn run_decay_for_value(
    config: &ValueConfig,
    database: &SpaceObjectDatabase,
    ids: &[u32],
    verbose: bool,
    multi_progress: Option<Arc<MultiProgress>>,
) -> Result<HashMap<u32, DecayPrediction>> {
    // Parse propagator type
    let propagator_type = match config.decay.propagator.to_lowercase().as_str() {
        "native-rk4" => DecayPropagatorType::NativeRk4,
        "satkit-rk98" => DecayPropagatorType::SatkitRk98,
        _ => {
            log::warn!(
                "Unknown decay propagator '{}', using satkit-rk98",
                config.decay.propagator
            );
            DecayPropagatorType::SatkitRk98
        }
    };

    let atmosphere_type = match config.decay.atmosphere.to_lowercase().as_str() {
        "nrlmsise00" => DecayAtmosphereType::Nrlmsise00,
        "nrlmsise00-lookup" => DecayAtmosphereType::Nrlmsise00Lookup,
        "jb2008" => DecayAtmosphereType::Jb2008,
        "jb2008-lookup" => DecayAtmosphereType::Jb2008Lookup,
        "harris-priester" => DecayAtmosphereType::HarrisPriester,
        "harris-priester-lookup" => DecayAtmosphereType::HarrisPriesterLookup,
        "exponential" => DecayAtmosphereType::Exponential,
        "exponential-lookup" => DecayAtmosphereType::ExponentialLookup,
        _ => {
            log::warn!(
                "Unknown atmosphere model '{}', using nrlmsise00",
                config.decay.atmosphere
            );
            DecayAtmosphereType::Nrlmsise00
        }
    };

    let lookup_accuracy =
        parse_lookup_accuracy("decay", &config.decay.lookup_accuracy);

    let gravity_type = match config.decay.gravity.to_lowercase().as_str() {
        "point-mass" => DecayGravityType::PointMass,
        "j2" => DecayGravityType::J2,
        "full-field-20" => DecayGravityType::FullField20,
        _ => {
            log::warn!(
                "Unknown gravity model '{}', using j2",
                config.decay.gravity
            );
            DecayGravityType::J2
        }
    };

    let decay_args = DecayArgs {
        output: PathBuf::from("/dev/null"), // We don't need file output
        days: config.decay.horizon_days,
        propagator: propagator_type,
        atmosphere: atmosphere_type,
        lookup_accuracy: lookup_accuracy.into(),
        gravity: gravity_type,
        srp: config.decay.include_srp,
        third_body: config.decay.include_third_body,
        ephemeris: DecayEphemerisType::LowPrecision,
        step_seconds: 60.0,
        tolerance: 1e-9,
        reentry_altitude_km: 70.0,
        limit: 0,
        norad_ids: ids.to_vec(),
    };

    // Initialize satkit data
    ensure_satkit_data();

    // Build HiFi settings
    let mut hifi_config = PropagatorConfig::high_precision();
    hifi_config.step_size = decay_args.step_seconds;
    hifi_config.tolerance = decay_args.tolerance;
    hifi_config.reentry_altitude = decay_args.reentry_altitude_km * 1000.0;

    let settings = HiFiSettings {
        propagator: propagator_type.to_propagator_type(),
        integrator: match propagator_type {
            DecayPropagatorType::NativeRk4 => IntegratorType::NativeRk4,
            DecayPropagatorType::SatkitRk98 => IntegratorType::SatkitRk98,
        },
        atmosphere: atmosphere_type.to_atmosphere_type(),
        lookup_accuracy,
        gravity: gravity_type.to_gravity_choice(),
        include_srp: decay_args.srp,
        include_third_body: decay_args.third_body,
        ephemeris: decay_args.ephemeris.to_ephemeris_type(),
        config: hifi_config,
        decay_horizon_days: decay_args.days,
    };

    let start_time = satkit::Instant::now();
    let target_epoch = start_time + satkit::Duration::from_days(decay_args.days);

    if verbose {
        log::info!(
            "Decay prediction: {} days horizon, {} propagator",
            decay_args.days,
            settings.propagator.name()
        );
    }

    let progress = make_progress_bar(
        ids.len() as u64,
        "{elapsed_precise} {bar:40.cyan/blue} {pos}/{len} {percent}% ETA {eta_precise} (decay)",
        multi_progress.as_deref(),
    );

    let database = Arc::new(database.clone());
    let settings = Arc::new(settings);

    let predictions: Vec<DecayPrediction> = ids
        .par_iter()
        .map(|norad_id| {
            let norad_str = norad_id.to_string();
            let obj = database.objects.get(&norad_str);

            let name = obj.map(|o| o.display_name());
            let object_type = obj.and_then(|o| o.object_type.clone());
            let country = obj.and_then(|o| o.country.clone());
            let launch_date = obj.and_then(|o| o.launch_date.clone());

            let tle = match obj.and_then(|o| o.tle.as_ref()) {
                Some(tle) => tle,
                None => {
                    progress.inc(1);
                    return DecayPrediction {
                        norad_id: *norad_id,
                        name,
                        object_type,
                        country,
                        launch_date,
                        tle_epoch: "N/A".to_string(),
                        tle_age_days: 0.0,
                        initial_altitude_km: 0.0,
                        decay_epoch: None,
                        decay_altitude_km: None,
                        days_to_decay: None,
                        status: "no_tle".to_string(),
                        error_message: Some("No TLE available".to_string()),
                        propagation_seconds: 0.0,
                    };
                }
            };

            let satkit_tle = match satkit::TLE::load_2line(&tle.line1, &tle.line2) {
                Ok(t) => t,
                Err(e) => {
                    progress.inc(1);
                    return DecayPrediction {
                        norad_id: *norad_id,
                        name,
                        object_type,
                        country,
                        launch_date,
                        tle_epoch: tle.epoch.clone(),
                        tle_age_days: 0.0,
                        initial_altitude_km: 0.0,
                        decay_epoch: None,
                        decay_altitude_km: None,
                        days_to_decay: None,
                        status: "error".to_string(),
                        error_message: Some(format!("Failed to parse TLE: {}", e)),
                        propagation_seconds: 0.0,
                    };
                }
            };

            let orbital_state = match orbital_state_from_tle(&satkit_tle, &start_time) {
                Some(s) => s,
                None => {
                    progress.inc(1);
                    return DecayPrediction {
                        norad_id: *norad_id,
                        name,
                        object_type,
                        country,
                        launch_date,
                        tle_epoch: tle.epoch.clone(),
                        tle_age_days: 0.0,
                        initial_altitude_km: 0.0,
                        decay_epoch: None,
                        decay_altitude_km: None,
                        days_to_decay: None,
                        status: "error".to_string(),
                        error_message: Some("Failed to convert TLE to orbital state".to_string()),
                        propagation_seconds: 0.0,
                    };
                }
            };

            let (mass, area) = obj
                .and_then(|o| o.discos.as_ref())
                .map(|d| (d.mass, d.x_sect_avg))
                .unwrap_or((None, None));

            let initial_state = SpacecraftState::from_discos(orbital_state, mass, area);
            let initial_altitude_km = initial_state.orbital.altitude() / 1000.0;
            let tle_age_days = (start_time - satkit_tle.epoch).as_seconds() / 86400.0;

            let propagator = match settings.build_propagator() {
                Some(p) => p,
                None => {
                    progress.inc(1);
                    return DecayPrediction {
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
                        error_message: Some("Failed to build propagator".to_string()),
                        propagation_seconds: 0.0,
                    };
                }
            };

            let prop_start = std::time::Instant::now();
            let result = propagator.propagate(initial_state, target_epoch);
            let propagation_seconds = prop_start.elapsed().as_secs_f64();

            progress.inc(1);

            match result {
                Ok(_) => DecayPrediction {
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
                },
                Err(crate::propagation::hifi::PropagationError::Reentry { altitude_km, epoch }) => {
                    let days_to_decay = (epoch - start_time).as_seconds() / 86400.0;
                    DecayPrediction {
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
                    }
                }
                Err(e) => DecayPrediction {
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
                },
            }
        })
        .collect();

    progress.finish_and_clear();

    let mut map = HashMap::new();
    for pred in predictions {
        map.insert(pred.norad_id, pred);
    }
    Ok(map)
}

/// Run collision scan for value analysis
fn run_collisions_for_value(
    config: &ValueConfig,
    database: &SpaceObjectDatabase,
    ids: &[u32],
    verbose: bool,
    multi_progress: Option<Arc<MultiProgress>>,
) -> Result<HashMap<u32, CollisionSummary>> {
    let propagator_type = match config.collision.propagator.to_lowercase().as_str() {
        "sgp4" => CollisionPropagatorType::Sgp4,
        "native-rk4" => CollisionPropagatorType::NativeRk4,
        "satkit-rk98" => CollisionPropagatorType::SatkitRk98,
        _ => {
            log::warn!(
                "Unknown collision propagator '{}', using satkit-rk98",
                config.collision.propagator
            );
            CollisionPropagatorType::SatkitRk98
        }
    };

    if verbose {
        log::info!(
            "Collision scan: {} hours, {:.1} km threshold",
            config.collision.hours,
            config.collision.distance_threshold_km
        );
    }

    let collision_args = CollisionArgs {
        output: PathBuf::from("/dev/null"),
        hours: config.collision.hours,
        step_seconds: 60,
        distance_km: config.collision.distance_threshold_km,
        max_events: 100000,
        propagator: propagator_type,
        atmosphere: match config.collision.atmosphere.to_lowercase().as_str() {
            "nrlmsise00" => DecayAtmosphereType::Nrlmsise00,
            "nrlmsise00-lookup" => DecayAtmosphereType::Nrlmsise00Lookup,
            "jb2008" => DecayAtmosphereType::Jb2008,
            "jb2008-lookup" => DecayAtmosphereType::Jb2008Lookup,
            "harris-priester" => DecayAtmosphereType::HarrisPriester,
            "harris-priester-lookup" => DecayAtmosphereType::HarrisPriesterLookup,
            "exponential" => DecayAtmosphereType::Exponential,
            "exponential-lookup" => DecayAtmosphereType::ExponentialLookup,
            _ => {
                log::warn!(
                    "Unknown collision atmosphere model '{}', using harris-priester",
                    config.collision.atmosphere
                );
                DecayAtmosphereType::HarrisPriester
            }
        },
        lookup_accuracy: parse_lookup_accuracy("collision", &config.collision.lookup_accuracy).into(),
        gravity: DecayGravityType::J2,
        srp: false,
        third_body: false,
        ephemeris: DecayEphemerisType::LowPrecision,
        limit: 0,
        norad_ids: ids.to_vec(),
    };

    // Run collision scan inline
    let steps = ((collision_args.hours * 3600.0) / collision_args.step_seconds as f64).ceil() as u64;
    let total_steps = steps + 1;
    let threshold_km = collision_args.distance_km as f32;
    let threshold_sq = threshold_km * threshold_km;
    let cell_size = threshold_km.max(1.0);
    let inv_cell = 1.0 / cell_size;
    let target_set: HashSet<u32> = ids.iter().copied().collect();

    let mut events: HashMap<(u32, u32), CollisionEvent> = HashMap::new();
    let start_time = satkit::Instant::now();

    if collision_args.propagator.is_hifi() {
        if verbose {
            log::info!(
                "Collision scan: HiFi targets vs SGP4 catalog ({} targets)",
                ids.len()
            );
        }

        run_collisions_for_value_hifi(
            &collision_args,
            database,
            ids,
            &target_set,
            multi_progress.clone(),
            total_steps,
            start_time,
            threshold_sq,
            inv_cell,
            &mut events,
        )?;
    } else {
        // Use SGP4 for collision scan (fast)
        let mut propagator = Propagator::new();
        propagator.load_tles(&database.objects);

        let progress = make_progress_bar(
            total_steps,
            "{elapsed_precise} {bar:40.cyan/blue} {pos}/{len} {percent}% ETA {eta_precise} (collisions)",
            multi_progress.as_deref(),
        );

        for step in 0..total_steps {
            let time = start_time
                + satkit::Duration::from_seconds(step as f64 * collision_args.step_seconds as f64);
            propagator.set_time(time);
            let states = propagator.propagate_all();
            if states.is_empty() {
                progress.inc(1);
                continue;
            }

            let objects = build_object_states_sgp4(&states);
            scan_targets_vs_catalog(&objects, &target_set, time, threshold_sq, inv_cell, &mut events);
            progress.inc(1);
        }

        progress.finish_and_clear();
    }

    // Convert events to summaries
    let mut summaries: HashMap<u32, CollisionSummary> = HashMap::new();

    for event in events.values() {
        let summary_a = summaries.entry(event.norad_a).or_default();
        summary_a.approach_count += 1;
        if summary_a.min_distance_km == 0.0 || event.distance_km < summary_a.min_distance_km {
            summary_a.min_distance_km = event.distance_km;
        }
        if event.relative_speed_kms > summary_a.max_relative_speed_kms {
            summary_a.max_relative_speed_kms = event.relative_speed_kms;
        }
        summary_a.total_score += event.score;

        let summary_b = summaries.entry(event.norad_b).or_default();
        summary_b.approach_count += 1;
        if summary_b.min_distance_km == 0.0 || event.distance_km < summary_b.min_distance_km {
            summary_b.min_distance_km = event.distance_km;
        }
        if event.relative_speed_kms > summary_b.max_relative_speed_kms {
            summary_b.max_relative_speed_kms = event.relative_speed_kms;
        }
        summary_b.total_score += event.score;
    }

    Ok(summaries)
}

fn run_collisions_for_value_hifi(
    args: &CollisionArgs,
    database: &SpaceObjectDatabase,
    target_ids: &[u32],
    target_set: &HashSet<u32>,
    multi_progress: Option<Arc<MultiProgress>>,
    total_steps: u64,
    start_time: satkit::Instant,
    threshold_sq: f32,
    inv_cell: f32,
    events: &mut HashMap<(u32, u32), CollisionEvent>,
) -> Result<()> {
    ensure_satkit_data();

    let config = PropagatorConfig {
        step_size: args.step_seconds as f64,
        tolerance: 1e-9,
        max_steps: 10_000_000,
        store_history: false,
        history_interval: 0.0,
        reentry_altitude: 70_000.0,
        escape_altitude: 1_000_000_000.0,
    };

    let settings = Arc::new(HiFiSettings {
        propagator: args
            .propagator
            .to_propagator_type()
            .unwrap_or(PropagatorType::NativeRk4),
        integrator: match args.propagator {
            CollisionPropagatorType::NativeRk4 => IntegratorType::NativeRk4,
            CollisionPropagatorType::SatkitRk98 => IntegratorType::SatkitRk98,
            CollisionPropagatorType::Sgp4 => IntegratorType::NativeRk4,
        },
        atmosphere: args.atmosphere.to_atmosphere_type(),
        lookup_accuracy: args.lookup_accuracy.to_lookup_accuracy(),
        gravity: args.gravity.to_gravity_choice(),
        include_srp: args.srp,
        include_third_body: args.third_body,
        ephemeris: args.ephemeris.to_ephemeris_type(),
        config,
        decay_horizon_days: args.hours / 24.0,
    });

    let time_steps: Vec<satkit::Instant> = (0..total_steps)
        .map(|step| start_time + satkit::Duration::from_seconds(step as f64 * args.step_seconds as f64))
        .collect();

    let progress = make_progress_bar(
        target_ids.len() as u64,
        "{elapsed_precise} {bar:40.cyan/blue} {pos}/{len} {percent}% ETA {eta_precise} (propagating)",
        multi_progress.as_deref(),
    );

    let database = Arc::new(database.clone());
    let time_steps = Arc::new(time_steps);

    let trajectories: Vec<(u32, Vec<Option<(Vec3, Vec3)>>)> = target_ids
        .par_iter()
        .map(|norad_id| {
            let norad_str = norad_id.to_string();
            let obj = database.objects.get(&norad_str);

            let tle = match obj.and_then(|o| o.tle.as_ref()) {
                Some(tle) => tle,
                None => {
                    progress.inc(1);
                    return (*norad_id, vec![None; time_steps.len()]);
                }
            };

            let satkit_tle = match satkit::TLE::load_2line(&tle.line1, &tle.line2) {
                Ok(t) => t,
                Err(_) => {
                    progress.inc(1);
                    return (*norad_id, vec![None; time_steps.len()]);
                }
            };

            let orbital_state = match orbital_state_from_tle(&satkit_tle, &start_time) {
                Some(s) => s,
                None => {
                    progress.inc(1);
                    return (*norad_id, vec![None; time_steps.len()]);
                }
            };

            let (mass, area) = obj
                .and_then(|o| o.discos.as_ref())
                .map(|d| (d.mass, d.x_sect_avg))
                .unwrap_or((None, None));

            let initial_state = SpacecraftState::from_discos(orbital_state, mass, area);

            let propagator = match settings.build_propagator() {
                Some(p) => p,
                None => {
                    progress.inc(1);
                    return (*norad_id, vec![None; time_steps.len()]);
                }
            };

            let mut positions: Vec<Option<(Vec3, Vec3)>> = Vec::with_capacity(time_steps.len());
            let mut current_state = initial_state;

            for (i, &target_time) in time_steps.iter().enumerate() {
                if i == 0 {
                    let pos_km = Vec3::new(
                        current_state.orbital.position.x as f32 / 1000.0,
                        current_state.orbital.position.z as f32 / 1000.0,
                        -current_state.orbital.position.y as f32 / 1000.0,
                    );
                    let vel_kms = Vec3::new(
                        current_state.orbital.velocity.x as f32 / 1000.0,
                        current_state.orbital.velocity.z as f32 / 1000.0,
                        -current_state.orbital.velocity.y as f32 / 1000.0,
                    );
                    positions.push(Some((pos_km, vel_kms)));
                    continue;
                }

                match propagator.propagate(current_state.clone(), target_time) {
                    Ok(result) => {
                        current_state = result.final_state;
                        let pos_km = Vec3::new(
                            current_state.orbital.position.x as f32 / 1000.0,
                            current_state.orbital.position.z as f32 / 1000.0,
                            -current_state.orbital.position.y as f32 / 1000.0,
                        );
                        let vel_kms = Vec3::new(
                            current_state.orbital.velocity.x as f32 / 1000.0,
                            current_state.orbital.velocity.z as f32 / 1000.0,
                            -current_state.orbital.velocity.y as f32 / 1000.0,
                        );
                        positions.push(Some((pos_km, vel_kms)));
                    }
                    Err(_) => {
                        positions.push(None);
                        while positions.len() < time_steps.len() {
                            positions.push(None);
                        }
                        break;
                    }
                }
            }

            progress.inc(1);
            (*norad_id, positions)
        })
        .collect();

    progress.finish_and_clear();

    let scan_progress = make_progress_bar(
        total_steps,
        "{elapsed_precise} {bar:40.cyan/blue} {pos}/{len} {percent}% ETA {eta_precise} (collisions)",
        multi_progress.as_deref(),
    );

    let mut sgp4 = Propagator::new();
    sgp4.load_tles(&database.objects);

    for step in 0..total_steps as usize {
        let time = time_steps[step];
        sgp4.set_time(time);
        let states = sgp4.propagate_all();
        if states.is_empty() {
            scan_progress.inc(1);
            continue;
        }

        let mut objects: Vec<ObjectState> = Vec::with_capacity(states.len());
        for (norad, state) in states.iter() {
            if target_set.contains(norad) {
                continue;
            }
            let pos_km = state.position * EARTH_RADIUS_KM as f32;
            objects.push(ObjectState {
                norad: *norad,
                pos_km,
                vel_kms: state.velocity,
            });
        }

        for (norad_id, positions) in &trajectories {
            if let Some(Some((pos_km, vel_kms))) = positions.get(step) {
                objects.push(ObjectState {
                    norad: *norad_id,
                    pos_km: *pos_km,
                    vel_kms: *vel_kms,
                });
            }
        }

        if !objects.is_empty() {
            scan_targets_vs_catalog(&objects, target_set, time, threshold_sq, inv_cell, events);
        }

        scan_progress.inc(1);
    }

    scan_progress.finish_and_clear();
    Ok(())
}
