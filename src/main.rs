//! SpaceDB - Space Object Database Visualizer
//!
//! A 3D visualization tool for tracking satellites and space debris
//! using data from multiple sources including Space-Track, GCAT, and DISCOS.

mod data;
mod propagation;
mod renderer;
mod ui;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;

use anyhow::Result;
use eframe::egui;
use glam::Vec3;

use data::{load_complete_database, DatabaseStats, ObjectFilter, SearchIndex, SpaceObjectDatabase};
use propagation::{generate_orbit_track_from_tle, Propagator, SatelliteState, EARTH_RADIUS_KM};
use renderer::{
    altitude_to_color, Camera, OrbitVertex, SatelliteInstance, SceneCallback, SceneRenderData,
    SceneRenderResources,
};
use ui::{BrowserPanel, DetailPanel, SearchPanel, TimeControls, VelocityFilter};

#[derive(Debug)]
enum Sgp4Command {
    SetIds(Vec<u32>),
    Propagate { time: satkit::Instant },
    Stop,
}

#[derive(Debug)]
struct Sgp4Result {
    time: satkit::Instant,
    states: HashMap<u32, SatelliteState>,
}

struct Sgp4Worker {
    sender: Sender<Sgp4Command>,
    receiver: Receiver<Sgp4Result>,
    _handle: thread::JoinHandle<()>,
}

impl Sgp4Worker {
    fn new(mut propagator: Propagator) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<Sgp4Command>();
        let (result_tx, result_rx) = mpsc::channel::<Sgp4Result>();

        let handle = thread::spawn(move || {
            let mut current_ids: Vec<u32> = Vec::new();

            while let Ok(command) = cmd_rx.recv() {
                match command {
                    Sgp4Command::SetIds(ids) => {
                        current_ids = ids;
                    }
                    Sgp4Command::Propagate { time } => {
                        if current_ids.is_empty() {
                            let _ = result_tx.send(Sgp4Result {
                                time,
                                states: HashMap::new(),
                            });
                            continue;
                        }

                        propagator.set_time(time);
                        let states = propagator.propagate_subset(&current_ids);
                        if result_tx.send(Sgp4Result { time, states }).is_err() {
                            break;
                        }
                    }
                    Sgp4Command::Stop => break,
                }
            }
        });

        Self {
            sender: cmd_tx,
            receiver: result_rx,
            _handle: handle,
        }
    }
}

/// Application state
pub struct SpaceDbApp {
    // Data
    database: SpaceObjectDatabase,
    search_index: SearchIndex,
    stats: DatabaseStats,

    // Propagation
    propagator: Propagator,

    // UI state
    search_panel: SearchPanel,
    browser_panel: BrowserPanel,
    time_controls: TimeControls,
    selected_object: Option<u32>,
    show_settings_window: bool,

    // Camera
    camera: Camera,
    camera_drag: Option<egui::Pos2>,

    // Cached satellite instances for rendering
    satellite_instances: Arc<Vec<SatelliteInstance>>,
    satellite_positions: HashMap<u32, Vec3>,
    satellite_states: HashMap<u32, SatelliteState>,
    satellite_update_accumulator: f64,
    last_states_time: Option<satkit::Instant>,

    // Selected object orbit track
    orbit_track: Arc<Vec<OrbitVertex>>,
    orbit_track_update_accumulator: f64,
    last_orbit_target: Option<u32>,
    last_orbit_points: u32,

    // Filtering and worker state
    filtered_norad_ids: Vec<u32>,
    filtered_norad_set: HashSet<u32>,
    active_filter: ObjectFilter,
    active_velocity_filter: VelocityFilter,
    sgp4_worker: Option<Sgp4Worker>,

    // 3D Renderer state
    wgpu_initialized: bool,

    // Frame timing
    last_frame_time: std::time::Instant,
    last_log_time: std::time::Instant,
    last_frame_delta: f64,
}

impl SpaceDbApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Result<Self> {
        // Load database
        let db_path = PathBuf::from("out/space_objects.json");
        let discos_path = PathBuf::from("data/cache/discos_objects_by_satno.json.gz");

        log::info!("Loading database...");
        let database = load_complete_database(&db_path, &discos_path)?;

        log::info!("Building search index...");
        let mut search_index = SearchIndex::build(&database);

        let stats = DatabaseStats::from_database(&database);
        log::info!("Database stats: {:?}", stats);

        // Initialize propagator
        let mut propagator = Propagator::new();
        propagator.load_tles(&database.objects);
        log::info!("Loaded {} TLEs for propagation", propagator.tle_count());

        // Initialize search with all objects
        let mut search_panel = SearchPanel::default();
        search_panel.filter.has_tle_only = true;
        search_panel.filter.exclude_decayed = true;
        let active_filter = search_panel.filter.clone();
        let active_velocity_filter = search_panel.velocity_filter.clone();
        let filtered_norad_ids = build_filtered_ids(&database, &search_index, &active_filter);
        let filtered_norad_set: HashSet<u32> = filtered_norad_ids.iter().copied().collect();
        search_panel.results = search_index.search("", usize::MAX, Some(&filtered_norad_set));

        // Initialize wgpu renderer if available
        let wgpu_initialized = if let Some(wgpu_render_state) = &cc.wgpu_render_state {
            let device = &wgpu_render_state.device;
            let queue = &wgpu_render_state.queue;
            let target_format = wgpu_render_state.target_format;

            let assets_path = PathBuf::from("assets");

            match SceneRenderResources::new(device, queue, target_format, 1280, 720, &assets_path) {
                Ok(resources) => {
                    wgpu_render_state
                        .renderer
                        .write()
                        .callback_resources
                        .insert(resources);
                    log::info!("wgpu 3D renderer initialized successfully");
                    true
                }
                Err(e) => {
                    log::error!("Failed to initialize wgpu renderer: {}", e);
                    false
                }
            }
        } else {
            log::warn!("No wgpu render state available, using 2D fallback");
            false
        };

        Ok(Self {
            database,
            search_index,
            stats,
            propagator,
            search_panel,
            browser_panel: BrowserPanel::default(),
            time_controls: TimeControls::default(),
            selected_object: None,
            show_settings_window: false,
            camera: Camera::default(),
            camera_drag: None,
            satellite_instances: Arc::new(Vec::new()),
            satellite_positions: HashMap::new(),
            satellite_states: HashMap::new(),
            satellite_update_accumulator: 0.0,
            last_states_time: None,
            orbit_track: Arc::new(Vec::new()),
            orbit_track_update_accumulator: 0.0,
            last_orbit_target: None,
            last_orbit_points: 0,
            filtered_norad_ids,
            filtered_norad_set,
            active_filter,
            active_velocity_filter,
            sgp4_worker: None,
            wgpu_initialized,
            last_frame_time: std::time::Instant::now(),
            last_log_time: std::time::Instant::now(),
            last_frame_delta: 0.0,
        })
    }

    fn ensure_sgp4_worker(&mut self) {
        if self.sgp4_worker.is_none() {
            self.sgp4_worker = Some(Sgp4Worker::new(self.propagator.clone_for_worker()));
            if let Some(worker) = &self.sgp4_worker {
                let _ = worker
                    .sender
                    .send(Sgp4Command::SetIds(self.filtered_norad_ids.clone()));
            }
        }
    }

    fn apply_filters(&mut self) {
        self.active_filter = self.search_panel.filter.clone();
        self.active_velocity_filter = self.search_panel.velocity_filter.clone();

        self.filtered_norad_ids =
            build_filtered_ids(&self.database, &self.search_index, &self.active_filter);
        self.filtered_norad_set = self.filtered_norad_ids.iter().copied().collect();

        self.search_panel.results = self.search_index.search(
            self.search_panel.query.as_str(),
            usize::MAX,
            Some(&self.filtered_norad_set),
        );

        if let Some(worker) = &self.sgp4_worker {
            let _ = worker
                .sender
                .send(Sgp4Command::SetIds(self.filtered_norad_ids.clone()));
            let _ = worker.sender.send(Sgp4Command::Propagate {
                time: *self.propagator.current_time(),
            });
        }

        if let Some(selected) = self.selected_object {
            if !self.filtered_norad_set.contains(&selected) {
                self.selected_object = None;
                self.camera.reset_to_earth();
                self.orbit_track = Arc::new(Vec::new());
                self.last_orbit_target = None;
            }
        }

        self.satellite_states.clear();
        self.satellite_instances = Arc::new(Vec::new());
        self.satellite_positions.clear();
        self.last_states_time = None;
        let update_hz = self.time_controls.sgp4_update_hz.clamp(1.0, 20.0) as f64;
        self.satellite_update_accumulator = 1.0 / update_hz;
    }

    fn process_sgp4_worker(&mut self, frame_time: f64) {
        self.ensure_sgp4_worker();

        let update_hz = self.time_controls.sgp4_update_hz.clamp(1.0, 20.0) as f64;
        let update_interval = 1.0 / update_hz;

        self.satellite_update_accumulator += frame_time.max(0.0);

        if self.satellite_update_accumulator >= update_interval {
            let time = *self.propagator.current_time();
            if let Some(worker) = &self.sgp4_worker {
                let _ = worker.sender.send(Sgp4Command::Propagate { time });
            }
            self.satellite_update_accumulator = 0.0;
        }

        let mut latest_result = None;
        if let Some(worker) = &self.sgp4_worker {
            while let Ok(result) = worker.receiver.try_recv() {
                latest_result = Some(result);
            }
        }

        if let Some(result) = latest_result {
            self.satellite_states = result.states;
            self.last_states_time = Some(result.time);
            self.rebuild_satellite_instances();
        }
    }

    fn rebuild_satellite_instances(&mut self) {
        let velocity_filter = &self.active_velocity_filter;
        let mut min_speed = velocity_filter.min_kms;
        let mut max_speed = velocity_filter.max_kms;
        if max_speed < min_speed {
            std::mem::swap(&mut min_speed, &mut max_speed);
        }

        let (low_threshold, high_threshold) = if velocity_filter.enabled
            && (velocity_filter.slow_percent > 0.0 || velocity_filter.fast_percent > 0.0)
            && !self.satellite_states.is_empty()
        {
            let mut speeds: Vec<f32> = self
                .satellite_states
                .values()
                .map(|state| state.velocity.length())
                .collect();
            speeds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let count = speeds.len() as f32;

            let low = if velocity_filter.slow_percent > 0.0 {
                let idx = ((velocity_filter.slow_percent / 100.0) * (count - 1.0))
                    .round()
                    .clamp(0.0, count - 1.0) as usize;
                Some(speeds[idx])
            } else {
                None
            };

            let high = if velocity_filter.fast_percent > 0.0 {
                let idx = ((1.0 - (velocity_filter.fast_percent / 100.0)) * (count - 1.0))
                    .round()
                    .clamp(0.0, count - 1.0) as usize;
                Some(speeds[idx])
            } else {
                None
            };

            (low, high)
        } else {
            (None, None)
        };

        let current_time = *self.propagator.current_time();
        let dt = if let Some(last_time) = self.last_states_time {
            current_time.as_unixtime() - last_time.as_unixtime()
        } else {
            0.0
        };
        let dt = dt.clamp(-5.0, 5.0) as f32;
        let extrapolate = dt / EARTH_RADIUS_KM as f32;

        let camera_pos = self.camera.position();

        let mut instances = Vec::with_capacity(self.satellite_states.len());
        self.satellite_positions.clear();

        for (norad_id, state) in self.satellite_states.iter() {
            let is_selected = self.selected_object == Some(*norad_id);
            let speed = state.velocity.length();

            if velocity_filter.enabled && !is_selected {
                if speed < min_speed || speed > max_speed {
                    continue;
                }

                if let Some(low) = low_threshold {
                    if speed > low {
                        if let Some(high) = high_threshold {
                            if speed < high {
                                continue;
                            }
                        } else {
                            continue;
                        }
                    }
                } else if let Some(high) = high_threshold {
                    if speed < high {
                        continue;
                    }
                }
            }

            let position = state.position + state.velocity * extrapolate;

            if is_selected {
                self.satellite_positions.insert(*norad_id, position);
            } else if is_occluded_by_earth(camera_pos, position) {
                continue;
            } else {
                self.satellite_positions.insert(*norad_id, position);
            }

            // Create instance for rendering
            let altitude_km = state.altitude_km;
            let mut color = altitude_to_color(altitude_km);

            // Dim satellites with old TLEs
            if state.tle_age_days > 30.0 {
                let fade = (1.0 - (state.tle_age_days - 30.0) as f32 / 60.0).max(0.3);
                color[0] *= fade;
                color[1] *= fade;
                color[2] *= fade;
            }

            let size = if is_selected { 3.0 } else { 1.0 };

            instances.push(SatelliteInstance {
                position: position.to_array(),
                color,
                size,
            });
        }

        self.satellite_instances = Arc::new(instances);
    }

    fn update_orbit_track(&mut self, time_delta: f64) {
        let current_target = self.selected_object;
        let points = self.time_controls.orbit_points;

        self.orbit_track_update_accumulator += time_delta;

        let target_changed = current_target != self.last_orbit_target;
        let points_changed = points != self.last_orbit_points;
        let time_refresh = self.orbit_track_update_accumulator.abs() >= 2.0;

        if !(target_changed || points_changed || time_refresh) {
            return;
        }

        self.last_orbit_target = current_target;
        self.last_orbit_points = points;
        self.orbit_track_update_accumulator = 0.0;

        if let Some(norad_id) = current_target {
            // Get the TLE for this satellite directly from propagator
            if let Some(tle) = self.propagator.get_tle(norad_id) {
                let color = [1.0, 0.15, 0.15, 1.0]; // Bright red
                let track = generate_orbit_track_from_tle(
                    tle,
                    self.propagator.current_time(),
                    points,
                    color,
                );
                self.orbit_track = Arc::new(track);
                return;
            }
        }

        self.orbit_track = Arc::new(Vec::new());
    }

    fn handle_camera_input(&mut self, ctx: &egui::Context, viewport_rect: egui::Rect) {
        let input = ctx.input(|i| i.clone());

        if let Some(pos) = input.pointer.hover_pos() {
            if viewport_rect.contains(pos) {
                // Scroll to zoom
                let scroll = input.raw_scroll_delta.y;
                if scroll != 0.0 {
                    self.camera.zoom(scroll * 0.1);
                }

                // Drag to orbit
                if input.pointer.button_down(egui::PointerButton::Primary) {
                    if let Some(last_pos) = self.camera_drag {
                        let delta = pos - last_pos;
                        if input.modifiers.shift {
                            self.camera.pan(delta.x, -delta.y);
                        } else {
                            self.camera.orbit(delta.x, delta.y);
                        }
                    }
                    self.camera_drag = Some(pos);
                } else {
                    self.camera_drag = None;
                }
            }
        }
    }

    fn get_tle_age_days(&self, norad_id: u32) -> Option<f64> {
        self.satellite_positions.get(&norad_id).map(|_| {
            // Get from propagator state if available
            if let Some(state) = self.propagator.propagate(norad_id) {
                state.tle_age_days
            } else {
                0.0
            }
        })
    }

    fn get_sun_direction(&self) -> Vec3 {
        // Simplified sun direction - rotate based on time
        // In reality this should be calculated from ephemeris
        let time_hours = self.propagator.current_time().as_jd() * 24.0;
        let angle = (time_hours % 24.0) / 24.0 * 2.0 * std::f64::consts::PI;
        Vec3::new(angle.cos() as f32, 0.3, angle.sin() as f32).normalize()
    }

    fn update_wgpu_render_data(&mut self, frame: &eframe::Frame, aspect_ratio: f32) {
        if let Some(wgpu_render_state) = frame.wgpu_render_state() {
            let renderer = wgpu_render_state.renderer.read();
            if let Some(resources) = renderer.callback_resources.get::<SceneRenderResources>() {
                let camera_pos = self.camera.position();
                let show_satellites = self.time_controls.show_satellites;
                let satellite_instances = if show_satellites {
                    Arc::clone(&self.satellite_instances)
                } else {
                    Arc::new(Vec::new())
                };
                let orbit_track = if show_satellites {
                    Arc::clone(&self.orbit_track)
                } else {
                    Arc::new(Vec::new())
                };

                // Logging
                if self.last_log_time.elapsed().as_secs_f32() > 1.0 {
                    self.last_log_time = std::time::Instant::now();
                    log::info!("--- Render State Log ---");
                    log::info!("Time: {}", self.propagator.format_time());
                    let gmst = self.propagator.get_gmst();
                    log::info!("GMST: {:.4} rad ({:.2} deg)", gmst, gmst.to_degrees());
                    let sun = self.propagator.get_sun_position();
                    log::info!(
                        "Sun Pos (Render Frame): {:.4}, {:.4}, {:.4}",
                        sun.x,
                        sun.y,
                        sun.z
                    );
                    log::info!(
                        "Camera Pos: {:.2}, {:.2}, {:.2}",
                        camera_pos.x,
                        camera_pos.y,
                        camera_pos.z
                    );
                }

                let render_data = SceneRenderData {
                    camera: self.camera.clone(),
                    aspect_ratio,
                    sun_direction: self.propagator.get_sun_position(),
                    time: (self.propagator.current_time().as_jd() % 1.0) as f32,
                    earth_rotation: self.propagator.get_gmst() as f32,
                    satellites: satellite_instances,
                    orbit_track,
                };
                resources.set_render_data(render_data);
            }
        }
    }

    fn render_3d_viewport(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        let viewport_rect = ui.available_rect_before_wrap();
        let pixels_per_point = ui.ctx().pixels_per_point();
        let viewport_width = (viewport_rect.width() * pixels_per_point).round().max(1.0) as u32;
        let viewport_height = (viewport_rect.height() * pixels_per_point).round().max(1.0) as u32;

        // Handle camera input
        self.handle_camera_input(ui.ctx(), viewport_rect);

        // Update aspect ratio in render data
        let aspect_ratio = viewport_rect.width() / viewport_rect.height();
        self.update_wgpu_render_data(frame, aspect_ratio);

        // Allocate space and add the wgpu callback
        let (response, painter) =
            ui.allocate_painter(viewport_rect.size(), egui::Sense::click_and_drag());

        // Add the 3D rendering callback
        painter.add(egui_wgpu::Callback::new_paint_callback(
            response.rect,
            SceneCallback {
                viewport_size: (viewport_width, viewport_height),
            },
        ));

        // Draw overlay info
        self.draw_viewport_overlay(&painter, response.rect, viewport_rect);
    }

    fn draw_viewport_overlay(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        _viewport_rect: egui::Rect,
    ) {
        let frame_time = self.last_frame_delta.max(0.001);

        // Info overlay
        painter.text(
            rect.left_top() + egui::vec2(10.0, 10.0),
            egui::Align2::LEFT_TOP,
            format!(
                "Camera: dist={:.2} az={:.1}° el={:.1}°\n\
                 Drag to orbit | Shift+drag to pan | Scroll to zoom\n\
                 FPS: {:.0} | 3D GPU Rendering",
                self.camera.distance,
                self.camera.azimuth.to_degrees(),
                self.camera.elevation.to_degrees(),
                1.0 / frame_time,
            ),
            egui::FontId::monospace(12.0),
            egui::Color32::from_rgb(150, 150, 150),
        );

        // Altitude legend
        let legend_x = rect.right() - 130.0;
        let legend_y = rect.top() + 20.0;
        let legend_items = [
            ("LEO < 2000km", [0.2, 0.6, 1.0, 1.0]),
            ("MEO 2-20Mm", [0.0, 1.0, 0.5, 1.0]),
            ("GEO ~36Mm", [1.0, 1.0, 0.0, 1.0]),
            ("HEO > 40Mm", [1.0, 0.5, 0.0, 1.0]),
        ];

        for (i, (label, color)) in legend_items.iter().enumerate() {
            let y = legend_y + i as f32 * 18.0;
            painter.circle_filled(
                egui::pos2(legend_x, y),
                5.0,
                egui::Color32::from_rgba_unmultiplied(
                    (color[0] * 255.0) as u8,
                    (color[1] * 255.0) as u8,
                    (color[2] * 255.0) as u8,
                    255,
                ),
            );
            painter.text(
                egui::pos2(legend_x + 12.0, y),
                egui::Align2::LEFT_CENTER,
                *label,
                egui::FontId::proportional(11.0),
                egui::Color32::from_rgb(180, 180, 180),
            );
        }
    }

    fn render_2d_fallback(&mut self, ui: &mut egui::Ui) {
        let viewport_rect = ui.available_rect_before_wrap();
        self.handle_camera_input(ui.ctx(), viewport_rect);

        // 2D fallback rendering (same as before)
        let (response, painter) =
            ui.allocate_painter(viewport_rect.size(), egui::Sense::click_and_drag());

        // Draw background
        painter.rect_filled(response.rect, 0.0, egui::Color32::from_rgb(5, 5, 15));

        // Draw stars (simple random pattern)
        let center = response.rect.center();

        // Draw Earth
        let scale = response.rect.height().min(response.rect.width()) * 0.35;
        let earth_radius = scale / self.camera.distance;

        // Earth with gradient
        painter.circle_filled(center, earth_radius, egui::Color32::from_rgb(25, 60, 120));
        painter.circle_stroke(
            center,
            earth_radius,
            egui::Stroke::new(2.0, egui::Color32::from_rgb(50, 100, 180)),
        );

        // Terminator line (day/night boundary)
        let sun_dir = self.get_sun_direction();
        let _terminator_x = center.x + sun_dir.x * earth_radius * 0.3;

        // Atmosphere glow
        for i in 1..=3 {
            let r = earth_radius * (1.0 + i as f32 * 0.03);
            let alpha = (40 - i * 10) as u8;
            painter.circle_stroke(
                center,
                r,
                egui::Stroke::new(
                    2.0,
                    egui::Color32::from_rgba_unmultiplied(100, 150, 255, alpha),
                ),
            );
        }

        // Draw orbit track
        let aspect = response.rect.width() / response.rect.height();
        let vp_matrix = self.camera.view_projection_matrix(aspect);

        if self.orbit_track.len() > 1 {
            let mut screen_points: Vec<egui::Pos2> = Vec::new();
            for vertex in self.orbit_track.iter() {
                let pos = Vec3::from_array(vertex.position);
                let clip = vp_matrix * glam::Vec4::new(pos.x, pos.y, pos.z, 1.0);
                if clip.w > 0.0 {
                    let ndc = glam::Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
                    if ndc.z > 0.0 && ndc.z < 1.0 {
                        let screen_x = center.x + ndc.x * response.rect.width() * 0.5;
                        let screen_y = center.y - ndc.y * response.rect.height() * 0.5;
                        screen_points.push(egui::pos2(screen_x, screen_y));
                    }
                }
            }

            // Draw orbit as line segments
            for window in screen_points.windows(2) {
                painter.line_segment(
                    [window[0], window[1]],
                    egui::Stroke::new(1.5, egui::Color32::from_rgba_unmultiplied(0, 255, 150, 180)),
                );
            }
        }

        // Draw satellites
        for instance in self.satellite_instances.iter() {
            let pos = Vec3::from_array(instance.position);
            let clip = vp_matrix * glam::Vec4::new(pos.x, pos.y, pos.z, 1.0);
            if clip.w > 0.0 {
                let ndc = glam::Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
                if ndc.z > 0.0 && ndc.z < 1.0 {
                    let screen_x = center.x + ndc.x * response.rect.width() * 0.5;
                    let screen_y = center.y - ndc.y * response.rect.height() * 0.5;

                    let color = egui::Color32::from_rgba_unmultiplied(
                        (instance.color[0] * 255.0) as u8,
                        (instance.color[1] * 255.0) as u8,
                        (instance.color[2] * 255.0) as u8,
                        (instance.color[3] * 255.0) as u8,
                    );

                    let size = instance.size * 2.0;
                    painter.circle_filled(egui::pos2(screen_x, screen_y), size, color);
                }
            }
        }

        // Draw selected satellite highlight
        if let Some(norad_id) = self.selected_object {
            if let Some(&pos) = self.satellite_positions.get(&norad_id) {
                let clip = vp_matrix * glam::Vec4::new(pos.x, pos.y, pos.z, 1.0);
                if clip.w > 0.0 {
                    let ndc = glam::Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
                    if ndc.z > 0.0 && ndc.z < 1.0 {
                        let screen_x = center.x + ndc.x * response.rect.width() * 0.5;
                        let screen_y = center.y - ndc.y * response.rect.height() * 0.5;

                        // Pulsing selection ring
                        let pulse = (self.propagator.current_time().as_jd() * 10.0).sin() as f32
                            * 0.5
                            + 0.5;
                        let ring_size = 12.0 + pulse * 4.0;

                        painter.circle_stroke(
                            egui::pos2(screen_x, screen_y),
                            ring_size,
                            egui::Stroke::new(2.0, egui::Color32::WHITE),
                        );
                    }
                }
            }
        }

        let frame_time = self.last_frame_delta.max(0.001);

        // Info overlay
        painter.text(
            response.rect.left_top() + egui::vec2(10.0, 10.0),
            egui::Align2::LEFT_TOP,
            format!(
                "Camera: dist={:.2} az={:.1}° el={:.1}°\n\
                 Drag to orbit | Shift+drag to pan | Scroll to zoom\n\
                 FPS: {:.0} | 2D Fallback (wgpu not available)",
                self.camera.distance,
                self.camera.azimuth.to_degrees(),
                self.camera.elevation.to_degrees(),
                1.0 / frame_time,
            ),
            egui::FontId::monospace(12.0),
            egui::Color32::from_rgb(150, 150, 150),
        );

        // Altitude legend
        let legend_x = response.rect.right() - 130.0;
        let legend_y = response.rect.top() + 20.0;
        let legend_items = [
            ("LEO < 2000km", [0.2, 0.6, 1.0, 1.0]),
            ("MEO 2-20Mm", [0.0, 1.0, 0.5, 1.0]),
            ("GEO ~36Mm", [1.0, 1.0, 0.0, 1.0]),
            ("HEO > 40Mm", [1.0, 0.5, 0.0, 1.0]),
        ];

        for (i, (label, color)) in legend_items.iter().enumerate() {
            let y = legend_y + i as f32 * 18.0;
            painter.circle_filled(
                egui::pos2(legend_x, y),
                5.0,
                egui::Color32::from_rgba_unmultiplied(
                    (color[0] * 255.0) as u8,
                    (color[1] * 255.0) as u8,
                    (color[2] * 255.0) as u8,
                    255,
                ),
            );
            painter.text(
                egui::pos2(legend_x + 12.0, y),
                egui::Align2::LEFT_CENTER,
                *label,
                egui::FontId::proportional(11.0),
                egui::Color32::from_rgb(180, 180, 180),
            );
        }
    }
}

impl eframe::App for SpaceDbApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        // Calculate frame time
        let max_fps = self.time_controls.max_fps.clamp(20.0, 500.0) as f64;
        let min_frame_time = 1.0 / max_fps;
        let mut now = std::time::Instant::now();
        let mut frame_time = (now - self.last_frame_time).as_secs_f64();

        if frame_time < min_frame_time {
            let sleep_time = min_frame_time - frame_time;
            std::thread::sleep(std::time::Duration::from_secs_f64(sleep_time));
            now = std::time::Instant::now();
            frame_time = (now - self.last_frame_time).as_secs_f64();
        }

        self.last_frame_time = now;
        self.last_frame_delta = frame_time;

        // Advance simulation time
        let time_delta = self.time_controls.time_delta(frame_time);
        self.propagator.advance_time(time_delta);

        // Update satellite positions via SGP4 worker
        if self.time_controls.compute_satellites {
            self.process_sgp4_worker(frame_time);
            // Update orbit track if selected
            self.update_orbit_track(time_delta);
        } else {
            self.satellite_positions.clear();
            self.satellite_instances = Arc::new(Vec::new());
            self.satellite_states.clear();
            self.last_states_time = None;
            self.satellite_update_accumulator = 0.0;
            self.orbit_track = Arc::new(Vec::new());
            self.orbit_track_update_accumulator = 0.0;
            self.last_orbit_target = None;
        }

        if let Some(norad_id) = self.selected_object {
            if let Some(pos) = self.satellite_positions.get(&norad_id) {
                self.camera.target = *pos;
                self.camera.enable_close_zoom(true);
            }
        }

        // Top panel with time controls
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("SpaceDB");
                ui.separator();
                if ui.button("Settings").clicked() {
                    self.show_settings_window = true;
                }
                ui.separator();
                ui.label(format!("Time: {}", self.propagator.format_time()));
                ui.separator();
                let visible_count = if self.time_controls.show_satellites {
                    self.satellite_instances.len()
                } else {
                    0
                };
                ui.label(format!(
                    "Objects: {} | Propagating: {} | Visible: {}",
                    self.stats.total_objects,
                    self.propagator.tle_count(),
                    visible_count
                ));
                if self.wgpu_initialized {
                    ui.separator();
                    ui.label(egui::RichText::new("GPU").color(egui::Color32::GREEN));
                }
            });
        });

        if self.show_settings_window {
            egui::Window::new("Settings")
                .open(&mut self.show_settings_window)
                .resizable(true)
                .show(ctx, |ui| {
                    self.time_controls.show(ui, &self.propagator.format_time());
                });
        }

        // Left panel with search and browser
        egui::SidePanel::left("left_panel")
            .default_width(300.0)
            .show(ctx, |ui| {
                self.search_panel
                    .show(ui, &mut self.search_index, Some(&self.filtered_norad_set));
                if self.search_panel.take_apply_filters() {
                    self.apply_filters();
                }
                ui.separator();

                if let Some(new_sel) = self.browser_panel.show(
                    ui,
                    &self.database,
                    &self.search_panel.results,
                    self.selected_object,
                ) {
                    self.selected_object = Some(new_sel);
                    self.camera.enable_close_zoom(true);

                    // Zoom to selected object
                    if let Some(pos) = self.satellite_positions.get(&new_sel) {
                        let alt = pos.length();
                        self.camera.focus_on_altitude(*pos, alt);
                    }
                }
            });

        // Right panel with object details (if selected)
        if let Some(norad_id) = self.selected_object {
            let norad_str = norad_id.to_string();
            let obj_clone = self.database.objects.get(&norad_str).cloned();
            if let Some(obj) = obj_clone {
                let tle_age = self.get_tle_age_days(norad_id);
                let mut deselect = false;

                // Get current state for display
                let current_state = if self.time_controls.compute_satellites {
                    self.propagator.propagate(norad_id)
                } else {
                    None
                };

                egui::SidePanel::right("right_panel")
                    .default_width(320.0)
                    .show(ctx, |ui| {
                        DetailPanel::show(ui, &obj, tle_age);

                        // Show current orbital state
                        if !self.time_controls.compute_satellites {
                            ui.separator();
                            ui.colored_label(
                                egui::Color32::from_rgb(200, 120, 120),
                                "Satellite propagation disabled",
                            );
                        } else if let Some(state) = current_state {
                            ui.separator();
                            ui.heading("Current State");
                            egui::Grid::new("state_grid")
                                .num_columns(2)
                                .spacing([10.0, 4.0])
                                .show(ui, |ui| {
                                    ui.label("Altitude:");
                                    ui.label(format!("{:.1} km", state.altitude_km));
                                    ui.end_row();

                                    ui.label("Speed:");
                                    ui.label(format!("{:.2} km/s", state.velocity.length()));
                                    ui.end_row();

                                    ui.label("Position (TEME):");
                                    ui.label(format!(
                                        "({:.0}, {:.0}, {:.0}) km",
                                        state.position.x * EARTH_RADIUS_KM as f32,
                                        state.position.y * EARTH_RADIUS_KM as f32,
                                        state.position.z * EARTH_RADIUS_KM as f32,
                                    ));
                                    ui.end_row();
                                });
                        }

                        ui.separator();
                        if ui.button("Deselect").clicked() {
                            deselect = true;
                        }
                    });

                if deselect {
                    self.selected_object = None;
                    self.camera.reset_to_earth();
                }
            }
        }

        // Central panel with 3D viewport
        egui::CentralPanel::default().show(ctx, |ui| {
            if self.wgpu_initialized {
                self.render_3d_viewport(ui, frame);
            } else {
                self.render_2d_fallback(ui);
            }
        });

        // Request continuous repaint for animation (with FPS cap)
        let max_fps = self.time_controls.max_fps.clamp(20.0, 500.0) as f64;
        let frame_delay = 1.0 / max_fps;
        ctx.request_repaint_after(std::time::Duration::from_secs_f64(frame_delay));
    }
}

impl Drop for SpaceDbApp {
    fn drop(&mut self) {
        if let Some(worker) = &self.sgp4_worker {
            let _ = worker.sender.send(Sgp4Command::Stop);
        }
    }
}

fn is_occluded_by_earth(camera_pos: Vec3, sat_pos: Vec3) -> bool {
    if camera_pos.length_squared() <= 1.0 {
        return false;
    }
    let dir = sat_pos - camera_pos;
    let a = dir.dot(dir);
    if a <= 0.0 {
        return false;
    }

    let b = 2.0 * camera_pos.dot(dir);
    let c = camera_pos.dot(camera_pos) - 1.0; // Earth radius = 1 in render units
    let disc = b * b - 4.0 * a * c;
    if disc <= 0.0 {
        return false;
    }

    let sqrt_disc = disc.sqrt();
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);
    let (tmin, tmax) = if t1 < t2 { (t1, t2) } else { (t2, t1) };

    (tmin >= 0.0 && tmin <= 1.0) || (tmax >= 0.0 && tmax <= 1.0)
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    log::info!("Starting SpaceDB...");

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 900.0])
            .with_title("SpaceDB - Space Object Database Visualizer"),
        renderer: eframe::Renderer::Wgpu, // Force wgpu renderer
        vsync: false,
        wgpu_options: egui_wgpu::WgpuConfiguration {
            present_mode: wgpu::PresentMode::Immediate,
            desired_maximum_frame_latency: Some(1),
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        "SpaceDB",
        options,
        Box::new(|cc| match SpaceDbApp::new(cc) {
            Ok(app) => Ok(Box::new(app)),
            Err(e) => {
                log::error!("Failed to initialize app: {}", e);
                Err(e.into())
            }
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {}", e))
}

fn build_filtered_ids(
    database: &SpaceObjectDatabase,
    search_index: &SearchIndex,
    filter: &ObjectFilter,
) -> Vec<u32> {
    let mut ids = Vec::new();
    for &norad_id in search_index.all_sorted() {
        let norad_str = norad_id.to_string();
        if let Some(obj) = database.objects.get(&norad_str) {
            if filter.matches(obj) {
                ids.push(norad_id);
            }
        }
    }
    ids
}
