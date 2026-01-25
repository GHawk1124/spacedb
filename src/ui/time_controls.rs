//! Time controls for simulation playback

use egui::Ui;

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
}

impl Default for TimeControls {
    fn default() -> Self {
        Self {
            playing: true,
            speed: 1.0,
            speed_presets: vec![-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0],
            speed_index: 4, // 1x
            orbit_points: 360,
        }
    }
}

impl TimeControls {
    pub fn show(&mut self, ui: &mut Ui, current_time: &str) {
        ui.horizontal(|ui| {
            // Play/Pause button
            let play_text = if self.playing { "⏸" } else { "▶" };
            if ui.button(play_text).clicked() {
                self.playing = !self.playing;
            }

            // Reverse button
            if ui.button("⏪").clicked() {
                if self.speed > 0.0 {
                    self.speed = -self.speed;
                }
            }

            // Forward button
            if ui.button("⏩").clicked() {
                if self.speed < 0.0 {
                    self.speed = -self.speed;
                }
            }

            ui.separator();

            // Speed selector
            ui.label("Speed:");
            egui::ComboBox::from_id_salt("speed_select")
                .selected_text(format_speed(self.speed))
                .show_ui(ui, |ui| {
                    for (i, &preset) in self.speed_presets.iter().enumerate() {
                        if ui
                            .selectable_value(&mut self.speed_index, i, format_speed(preset))
                            .clicked()
                        {
                            self.speed = preset;
                        }
                    }
                });

            ui.separator();

            // Current time display
            ui.label(format!("Time: {}", current_time));
        });

        // Orbit track settings
        ui.horizontal(|ui| {
            ui.label("Orbit points:");
            ui.add(egui::Slider::new(&mut self.orbit_points, 36..=720));
        });
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
