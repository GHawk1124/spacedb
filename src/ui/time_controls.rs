//! Time controls for simulation playback

use egui::Ui;

use crate::propagation::hifi::{
    AtmosphereModelType, GravityModelChoice, HiFiSettings, LookupAccuracy, PropagatorType,
};

/// Time control state
#[derive(Clone)]
pub struct TimeControls {
    /// Is the simulation playing?
    pub playing: bool,
    /// Playback speed multiplier (can be negative for reverse)
    pub speed: f64,
    /// Available speed presets
    pub speed_presets: Vec<f64>,
    /// Current preset index
    pub speed_index: usize,
    /// Orbit track resolution (points per orbit)
    pub orbit_points: u32,
    /// SGP4 update rate (Hz)
    pub sgp4_update_hz: f32,
    /// UI max FPS cap
    pub max_fps: f32,
    /// Show satellites in the renderer
    pub show_satellites: bool,
    /// Compute satellite propagation each frame
    pub compute_satellites: bool,
}

impl Default for TimeControls {
    fn default() -> Self {
        Self {
            playing: false,
            speed: 1.0,
            speed_presets: vec![
                -10000.0, -1000.0, -100.0, -10.0, -1.0, 1.0, 10.0, 100.0, 1000.0, 10000.0,
            ],
            speed_index: 5, // 1x
            orbit_points: 360,
            sgp4_update_hz: 20.0,
            max_fps: 120.0,
            show_satellites: true,
            compute_satellites: true,
        }
    }
}

impl TimeControls {
    pub fn show_top_bar(&mut self, ui: &mut Ui, current_time: &str) {
        ui.horizontal(|ui| {
            let play_text = if self.playing { "⏸" } else { "▶" };
            if ui.button(play_text).clicked() {
                self.playing = !self.playing;
            }

            if ui.button("⏪").clicked() {
                self.step_time_scale(-1);
            }

            if ui.button("⏩").clicked() {
                self.step_time_scale(1);
            }

            ui.separator();
            ui.label("Speed:");
            egui::ComboBox::from_id_salt("speed_select")
                .selected_text(format_speed(self.speed))
                .show_ui(ui, |ui| {
                    let presets = self.speed_presets.clone();
                    for (i, preset) in presets.into_iter().enumerate() {
                        if ui
                            .selectable_value(&mut self.speed_index, i, format_speed(preset))
                            .clicked()
                        {
                            self.set_speed(preset);
                            self.playing = true;
                        }
                    }
                });

            ui.separator();
            ui.label(format!("Time: {}", current_time));
        });
    }

    pub fn show_settings(&mut self, ui: &mut Ui, hifi_settings: &mut HiFiSettings) {
        ui.label("Satellites");
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.show_satellites, "Show");
            ui.checkbox(&mut self.compute_satellites, "Compute");
        });
        ui.add(egui::Slider::new(&mut self.orbit_points, 36..=720).text("Orbit points"));
        ui.add(egui::Slider::new(&mut self.sgp4_update_hz, 1.0..=60.0).text("SGP4 Hz"));

        ui.separator();
        ui.label("Performance");
        ui.add(egui::Slider::new(&mut self.max_fps, 20.0..=500.0).text("Max FPS"));

        ui.separator();
        ui.label("Propagation Method");

        ui.horizontal(|ui| {
            ui.label("Propagator");
            egui::ComboBox::from_id_salt("propagator_type")
                .selected_text(hifi_settings.propagator.name())
                .show_ui(ui, |ui| {
                    for option in PropagatorType::all() {
                        let response = ui.selectable_value(
                            &mut hifi_settings.propagator,
                            *option,
                            option.name(),
                        );
                        if response.clicked() {
                            hifi_settings.sync_integrator();
                        }
                        // Show description on hover
                        response.on_hover_text(option.description());
                    }
                });
        });

        // Show high-fidelity settings only when a high-fidelity propagator is selected
        if hifi_settings.propagator.is_high_fidelity() {
            ui.separator();
            ui.label("High-Fidelity Propagation Settings");

            ui.horizontal(|ui| {
                ui.label("Atmosphere");
                egui::ComboBox::from_id_salt("hifi_atmosphere")
                    .selected_text(hifi_settings.atmosphere.name())
                    .show_ui(ui, |ui| {
                        for option in [
                            AtmosphereModelType::Nrlmsise00,
                            AtmosphereModelType::Nrlmsise00Lookup,
                            AtmosphereModelType::Jb2008,
                            AtmosphereModelType::Jb2008Lookup,
                            AtmosphereModelType::HarrisPriester,
                            AtmosphereModelType::HarrisPriesterLookup,
                            AtmosphereModelType::Exponential,
                            AtmosphereModelType::ExponentialLookup,
                        ] {
                            ui.selectable_value(&mut hifi_settings.atmosphere, option, option.name());
                        }
                    });
            });

            if hifi_settings.atmosphere.is_lookup() {
                ui.horizontal(|ui| {
                    ui.label("Lookup Accuracy");
                    egui::ComboBox::from_id_salt("hifi_atmosphere_lookup")
                        .selected_text(hifi_settings.lookup_accuracy.name())
                        .show_ui(ui, |ui| {
                            for option in LookupAccuracy::all() {
                                let response = ui.selectable_value(
                                    &mut hifi_settings.lookup_accuracy,
                                    *option,
                                    option.name(),
                                );
                                response.on_hover_text(option.description());
                            }
                        });
                });
            }

            ui.horizontal(|ui| {
                ui.label("Gravity");
                egui::ComboBox::from_id_salt("hifi_gravity")
                    .selected_text(hifi_settings.gravity.name())
                    .show_ui(ui, |ui| {
                        for option in GravityModelChoice::all() {
                            ui.selectable_value(&mut hifi_settings.gravity, *option, option.name());
                        }
                    });
            });

            ui.horizontal(|ui| {
                ui.checkbox(&mut hifi_settings.include_srp, "SRP");
                ui.checkbox(&mut hifi_settings.include_third_body, "Sun/Moon");
            });

            ui.add(
                egui::Slider::new(&mut hifi_settings.config.step_size, 5.0..=600.0).text("Step (s)"),
            );
            ui.add(
                egui::Slider::new(&mut hifi_settings.config.tolerance, 1e-12..=1e-6)
                    .logarithmic(true)
                    .text("Tolerance"),
            );
            ui.add(
                egui::Slider::new(&mut hifi_settings.decay_horizon_days, 30.0..=3650.0)
                    .text("Decay horizon (days)"),
            );
        }
    }

    fn set_speed(&mut self, speed: f64) {
        self.speed = speed;
        if let Some((index, _)) = self
            .speed_presets
            .iter()
            .enumerate()
            .find(|(_, &preset)| (preset - speed).abs() < f64::EPSILON)
        {
            self.speed_index = index;
        }
    }

    fn step_time_scale(&mut self, direction: i32) {
        let mut magnitudes: Vec<f64> = self.speed_presets.iter().map(|s| s.abs()).collect();
        magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        magnitudes.dedup_by(|a, b| (*a - *b).abs() < f64::EPSILON);

        let current_mag = self.speed.abs();
        let mut idx = magnitudes
            .iter()
            .position(|m| (*m - current_mag).abs() < f64::EPSILON)
            .unwrap_or_else(|| {
                let mut best = 0usize;
                let mut best_diff = f64::MAX;
                for (i, m) in magnitudes.iter().enumerate() {
                    let diff = (*m - current_mag).abs();
                    if diff < best_diff {
                        best = i;
                        best_diff = diff;
                    }
                }
                best
            });

        if direction > 0 && idx + 1 < magnitudes.len() {
            idx += 1;
        } else if direction < 0 && idx > 0 {
            idx -= 1;
        }

        let sign = if self.speed < 0.0 { -1.0 } else { 1.0 };
        let next_speed = magnitudes[idx] * sign;
        self.set_speed(next_speed);
        self.playing = true;
    }

    /// Get the time delta for this frame
    pub fn time_delta(&self, frame_time: f64) -> f64 {
        if self.playing {
            frame_time * self.speed
        } else {
            0.0
        }
    }
}

fn format_speed(speed: f64) -> String {
    if speed == 0.0 {
        "Paused".to_string()
    } else if speed == 1.0 {
        "1x".to_string()
    } else if speed == -1.0 {
        "-1x".to_string()
    } else if speed.abs() >= 1000.0 {
        format!("{:.0}x", speed)
    } else if speed.abs() >= 1.0 {
        format!("{:.0}x", speed)
    } else {
        format!("{:.2}x", speed)
    }
}
