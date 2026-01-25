//! SpaceDB - Space Object Database Visualizer
//!
//! A 3D visualization tool for tracking satellites and space debris
//! using data from multiple sources including Space-Track, GCAT, and DISCOS.

mod data;
mod propagation;
mod renderer;
mod ui;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use eframe::egui;
use glam::Vec3;

use data::{load_complete_database, DatabaseStats, SearchIndex, SpaceObjectDatabase};
use propagation::{generate_orbit_track_from_tle, Propagator, EARTH_RADIUS_KM};
use renderer::{
    altitude_to_color, Camera, OrbitVertex, SatelliteInstance, SceneCallback, SceneRenderData,
    SceneRenderResources,
};
use ui::{BrowserPanel, DetailPanel, SearchPanel, TimeControls};

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

    // Camera
    camera: Camera,
    camera_drag: Option<egui::Pos2>,

    // Cached satellite instances for rendering
    satellite_instances: Vec<SatelliteInstance>,
    satellite_positions: HashMap<u32, Vec3>,

    // Selected object orbit track
    orbit_track: Vec<OrbitVertex>,

    // 3D Renderer state
    wgpu_initialized: bool,

    // Frame timing
    last_frame_time: std::time::Instant,
}

impl SpaceDbApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Result<Self> {
        // Load database
        let db_path = PathBuf::from("out/space_objects.json");
        let discos_path = PathBuf::from("data/cache/discos_objects_by_satno.json.gz");

        log::info!("Loading database...");
        let database = load_complete_database(&db_path, &discos_path)?;

        log::info!("Building search index...");
        let search_index = SearchIndex::build(&database);

        let stats = DatabaseStats::from_database(&database);
        log::info!("Database stats: {:?}", stats);

        // Initialize propagator
        let mut propagator = Propagator::new();
        propagator.load_tles(&database.objects);
        log::info!("Loaded {} TLEs for propagation", propagator.tle_count());

        // Initialize search with all objects
        let mut search_panel = SearchPanel::default();
        search_panel.results = search_index.all_sorted().to_vec();
        search_panel.filter.has_tle_only = true;
        search_panel.filter.exclude_decayed = true;

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
            camera: Camera::default(),
            camera_drag: None,
            satellite_instances: Vec::new(),
            satellite_positions: HashMap::new(),
            orbit_track: Vec::new(),
            wgpu_initialized,
            last_frame_time: std::time::Instant::now(),
        })
    }

    fn update_satellite_positions(&mut self) {
        // Propagate all satellites using SGP4
        let states = self.propagator.propagate_all();

        self.satellite_instances.clear();
        self.satellite_positions.clear();

        for (norad_id, state) in states {
            self.satellite_positions.insert(norad_id, state.position);

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

            // Highlight selected satellite
            let is_selected = self.selected_object == Some(norad_id);
            let size = if is_selected { 3.0 } else { 1.0 };

            self.satellite_instances.push(SatelliteInstance {
                position: state.position.to_array(),
                color,
                size,
            });
        }
    }

    fn update_orbit_track(&mut self) {
        self.orbit_track.clear();

        if let Some(norad_id) = self.selected_object {
            // Get the TLE for this satellite directly from propagator
            if let Some(tle) = self.propagator.get_tle(norad_id) {
                let color = [0.0, 1.0, 0.5, 0.8]; // Cyan-green
                self.orbit_track = generate_orbit_track_from_tle(
                    tle,
                    self.propagator.current_time(),
                    self.time_controls.orbit_points,
                    color,
                );
            }
        }
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

    fn update_wgpu_render_data(&self, frame: &eframe::Frame) {
        if let Some(wgpu_render_state) = frame.wgpu_render_state() {
            let renderer = wgpu_render_state.renderer.read();
            if let Some(resources) = renderer.callback_resources.get::<SceneRenderResources>() {
                let camera_pos = self.camera.position();
                let mut sorted_instances = self.satellite_instances.clone();
                sorted_instances.sort_by(|a, b| {
                    let a_pos = Vec3::from_array(a.position);
                    let b_pos = Vec3::from_array(b.position);
                    let a_dist = (a_pos - camera_pos).length_squared();
                    let b_dist = (b_pos - camera_pos).length_squared();
                    b_dist
                        .partial_cmp(&a_dist)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let render_data = SceneRenderData {
                    camera: self.camera.clone(),
                    aspect_ratio: 16.0 / 9.0, // Will be updated by viewport
                    sun_direction: self.get_sun_direction(),
                    time: (self.propagator.current_time().as_jd() % 1.0) as f32,
                    satellites: Arc::new(sorted_instances),
                    orbit_track: Arc::new(self.orbit_track.clone()),
                };
                resources.set_render_data(render_data);
            }
        }
    }

    fn render_3d_viewport(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        let viewport_rect = ui.available_rect_before_wrap();
        let viewport_width = viewport_rect.width() as u32;
        let viewport_height = viewport_rect.height() as u32;

        // Handle camera input
        self.handle_camera_input(ui.ctx(), viewport_rect);

        // Update aspect ratio in render data
        if let Some(wgpu_render_state) = frame.wgpu_render_state() {
            let renderer = wgpu_render_state.renderer.read();
            if let Some(resources) = renderer.callback_resources.get::<SceneRenderResources>() {
                let aspect_ratio = viewport_rect.width() / viewport_rect.height();
                let camera_pos = self.camera.position();
                let mut sorted_instances = self.satellite_instances.clone();
                sorted_instances.sort_by(|a, b| {
                    let a_pos = Vec3::from_array(a.position);
                    let b_pos = Vec3::from_array(b.position);
                    let a_dist = (a_pos - camera_pos).length_squared();
                    let b_dist = (b_pos - camera_pos).length_squared();
                    b_dist
                        .partial_cmp(&a_dist)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let render_data = SceneRenderData {
                    camera: self.camera.clone(),
                    aspect_ratio,
                    sun_direction: self.get_sun_direction(),
                    time: (self.propagator.current_time().as_jd() % 1.0) as f32,
                    satellites: Arc::new(sorted_instances),
                    orbit_track: Arc::new(self.orbit_track.clone()),
                };
                resources.set_render_data(render_data);
            }
        }

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
        let frame_time = (std::time::Instant::now() - self.last_frame_time)
            .as_secs_f64()
            .max(0.001);

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
            for vertex in &self.orbit_track {
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
        for instance in &self.satellite_instances {
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

        let frame_time = (std::time::Instant::now() - self.last_frame_time)
            .as_secs_f64()
            .max(0.001);

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
        let now = std::time::Instant::now();
        let frame_time = (now - self.last_frame_time).as_secs_f64();
        self.last_frame_time = now;

        // Advance simulation time
        let time_delta = self.time_controls.time_delta(frame_time);
        self.propagator.advance_time(time_delta);

        // Update satellite positions via SGP4
        self.update_satellite_positions();

        // Update orbit track if selected
        self.update_orbit_track();

        // Update wgpu render data
        self.update_wgpu_render_data(frame);

        // Top panel with time controls
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("SpaceDB");
                ui.separator();
                self.time_controls.show(ui, &self.propagator.format_time());
                ui.separator();
                ui.label(format!(
                    "Objects: {} | Propagating: {} | Visible: {}",
                    self.stats.total_objects,
                    self.propagator.tle_count(),
                    self.satellite_instances.len()
                ));
                if self.wgpu_initialized {
                    ui.separator();
                    ui.label(egui::RichText::new("GPU").color(egui::Color32::GREEN));
                }
            });
        });

        // Left panel with search and browser
        egui::SidePanel::left("left_panel")
            .default_width(300.0)
            .show(ctx, |ui| {
                self.search_panel.show(ui, &self.search_index);
                ui.separator();

                if let Some(new_sel) = self.browser_panel.show(
                    ui,
                    &self.database,
                    &self.search_panel.results,
                    &self.search_panel.filter,
                    self.selected_object,
                ) {
                    self.selected_object = Some(new_sel);

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
                let current_state = self.propagator.propagate(norad_id);

                egui::SidePanel::right("right_panel")
                    .default_width(320.0)
                    .show(ctx, |ui| {
                        DetailPanel::show(ui, &obj, tle_age);

                        // Show current orbital state
                        if let Some(state) = current_state {
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

        // Request continuous repaint for animation
        ctx.request_repaint();
    }
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    log::info!("Starting SpaceDB...");

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1600.0, 900.0])
            .with_title("SpaceDB - Space Object Database Visualizer"),
        renderer: eframe::Renderer::Wgpu, // Force wgpu renderer
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
