//! UI panels for object browser, search, and details

use crate::data::{ObjectFilter, SearchIndex, SpaceObject, SpaceObjectDatabase};
use egui::{Color32, RichText, Ui};
use std::collections::HashSet;

/// Search and filter panel state
#[derive(Default)]
pub struct SearchPanel {
    pub query: String,
    pub results: Vec<u32>,
    pub filter: ObjectFilter,
    pub show_filters: bool,
    pub velocity_filter: VelocityFilter,
    pub filters_dirty: bool,
    pub apply_filters_requested: bool,
}

/// Velocity filtering controls (km/s)
#[derive(Debug, Clone)]
pub struct VelocityFilter {
    pub enabled: bool,
    pub min_kms: f32,
    pub max_kms: f32,
    pub slow_percent: f32,
    pub fast_percent: f32,
}

impl Default for VelocityFilter {
    fn default() -> Self {
        Self {
            enabled: false,
            min_kms: 0.0,
            max_kms: 15.0,
            slow_percent: 0.0,
            fast_percent: 0.0,
        }
    }
}

impl SearchPanel {
    pub fn show(
        &mut self,
        ui: &mut Ui,
        index: &mut SearchIndex,
        allowed_ids: Option<&HashSet<u32>>,
    ) -> bool {
        let mut changed = false;
        ui.heading("Search");

        // Search box
        let response = ui.text_edit_singleline(&mut self.query);
        let query = self.query.trim();
        if response.changed() {
            self.results = index.search(query, usize::MAX, allowed_ids);
            changed = true;
        } else if !query.is_empty() && index.matcher_is_running() {
            self.results = index.search(query, usize::MAX, allowed_ids);
            changed = true;
        } else if query.is_empty() && self.results.is_empty() {
            self.results = index.search("", usize::MAX, allowed_ids);
            changed = true;
        }

        // Filter toggle
        if ui
            .button(if self.show_filters {
                "Hide Filters"
            } else {
                "Show Filters"
            })
            .clicked()
        {
            self.show_filters = !self.show_filters;
            changed = true;
        }

        if self.show_filters {
            ui.separator();
            if ui
                .checkbox(&mut self.filter.has_tle_only, "Has TLE only")
                .changed()
            {
                self.filters_dirty = true;
                changed = true;
            }
            if ui
                .checkbox(&mut self.filter.exclude_decayed, "Exclude decayed")
                .changed()
            {
                self.filters_dirty = true;
                changed = true;
            }

            ui.separator();
            ui.label("Object types:");
            let type_payload = toggle_object_type(ui, &mut self.filter, "Payload", "PAYLOAD");
            let type_rocket =
                toggle_object_type(ui, &mut self.filter, "Rocket Body", "ROCKET BODY");
            let type_debris = toggle_object_type(ui, &mut self.filter, "Debris", "DEBRIS");
            let types_changed = type_payload || type_rocket || type_debris;
            if types_changed {
                self.filters_dirty = true;
                changed = true;
            }

            ui.separator();
            if ui
                .checkbox(&mut self.filter.size_filter_enabled, "Size filter (meters)")
                .changed()
            {
                if self.filter.size_filter_enabled && self.filter.size_max_m <= 0.0 {
                    self.filter.size_max_m = 100.0;
                }
                self.filters_dirty = true;
                changed = true;
            }
            if self.filter.size_filter_enabled {
                let min_resp = ui
                    .add(egui::Slider::new(&mut self.filter.size_min_m, 0.0..=100.0).text("Min m"));
                let max_resp = ui
                    .add(egui::Slider::new(&mut self.filter.size_max_m, 0.0..=100.0).text("Max m"));
                if min_resp.changed() || max_resp.changed() {
                    self.filters_dirty = true;
                    changed = true;
                }
                if ui
                    .checkbox(
                        &mut self.filter.include_unknown_size,
                        "Include unknown size",
                    )
                    .changed()
                {
                    self.filters_dirty = true;
                    changed = true;
                }
            }

            ui.separator();
            if ui
                .checkbox(&mut self.velocity_filter.enabled, "Velocity filter (km/s)")
                .changed()
            {
                if self.velocity_filter.enabled && self.velocity_filter.max_kms <= 0.0 {
                    self.velocity_filter.max_kms = 15.0;
                }
                self.filters_dirty = true;
                changed = true;
            }
            if self.velocity_filter.enabled {
                let min_resp = ui.add(
                    egui::Slider::new(&mut self.velocity_filter.min_kms, 0.0..=15.0)
                        .text("Min km/s"),
                );
                let max_resp = ui.add(
                    egui::Slider::new(&mut self.velocity_filter.max_kms, 0.0..=15.0)
                        .text("Max km/s"),
                );
                let slow_resp = ui.add(
                    egui::Slider::new(&mut self.velocity_filter.slow_percent, 0.0..=100.0)
                        .text("Slowest %"),
                );
                let fast_resp = ui.add(
                    egui::Slider::new(&mut self.velocity_filter.fast_percent, 0.0..=100.0)
                        .text("Fastest %"),
                );
                if min_resp.changed()
                    || max_resp.changed()
                    || slow_resp.changed()
                    || fast_resp.changed()
                {
                    self.filters_dirty = true;
                    changed = true;
                }
            }

            ui.separator();
            let apply_button =
                ui.add_enabled(self.filters_dirty, egui::Button::new("Apply Filters"));
            if apply_button.clicked() {
                self.apply_filters_requested = true;
                self.filters_dirty = false;
                changed = true;
            }
        }

        ui.separator();
        ui.label(format!("{} results", self.results.len()));

        changed
    }

    pub fn take_apply_filters(&mut self) -> bool {
        let requested = self.apply_filters_requested;
        self.apply_filters_requested = false;
        requested
    }
}

fn toggle_object_type(ui: &mut Ui, filter: &mut ObjectFilter, label: &str, value: &str) -> bool {
    let mut enabled = filter.object_types.iter().any(|t| t == value);
    let response = ui.checkbox(&mut enabled, label);
    if response.changed() {
        if enabled {
            if !filter.object_types.iter().any(|t| t == value) {
                filter.object_types.push(value.to_string());
            }
        } else {
            filter.object_types.retain(|t| t != value);
        }
        return true;
    }
    false
}

/// Object browser panel
#[derive(Default)]
pub struct BrowserPanel;

impl BrowserPanel {
    pub fn show(
        &mut self,
        ui: &mut Ui,
        db: &SpaceObjectDatabase,
        results: &[u32],
        selected: Option<u32>,
    ) -> Option<u32> {
        let mut new_selection = None;

        let row_height = ui.text_style_height(&egui::TextStyle::Body) + ui.spacing().item_spacing.y;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show_rows(ui, row_height, results.len(), |ui, row_range| {
                for row in row_range {
                    let norad_id = results[row];
                    let norad_str = norad_id.to_string();
                    if let Some(obj) = db.objects.get(&norad_str) {
                        let is_selected = selected == Some(norad_id);
                        let name = obj.display_name();

                        // Color based on object type
                        let color = match obj.object_type.as_deref() {
                            Some("PAYLOAD") => Color32::from_rgb(100, 200, 100),
                            Some("ROCKET BODY") => Color32::from_rgb(200, 150, 100),
                            Some("DEBRIS") => Color32::from_rgb(150, 150, 150),
                            _ => Color32::WHITE,
                        };

                        let text = RichText::new(&name).color(color);

                        let response = ui.selectable_label(is_selected, text);

                        if response.clicked() {
                            new_selection = Some(norad_id);
                        }

                        // Hover tooltip
                        response.on_hover_ui(|ui| {
                            ui.label(format!("NORAD: {}", norad_id));
                            if let Some(t) = &obj.object_type {
                                ui.label(format!("Type: {}", t));
                            }
                            if let Some(c) = &obj.country {
                                ui.label(format!("Country: {}", c));
                            }
                        });
                    }
                }
            });

        new_selection
    }
}

/// Object detail panel
pub struct DetailPanel;

impl DetailPanel {
    pub fn show(ui: &mut Ui, obj: &SpaceObject, tle_age_days: Option<f64>) {
        ui.heading(&obj.display_name());
        ui.separator();

        egui::Grid::new("detail_grid")
            .num_columns(2)
            .spacing([10.0, 4.0])
            .show(ui, |ui| {
                ui.label("NORAD ID:");
                ui.label(format!("{}", obj.norad_cat_id));
                ui.end_row();

                if let Some(cospar) = &obj.cospar_id {
                    ui.label("COSPAR ID:");
                    ui.label(cospar);
                    ui.end_row();
                }

                if let Some(obj_type) = &obj.object_type {
                    ui.label("Type:");
                    ui.label(obj_type);
                    ui.end_row();
                }

                if let Some(kind) = &obj.analyst_kind {
                    ui.label("Analyst kind:");
                    ui.label(kind);
                    ui.end_row();
                }

                if let Some(country) = &obj.country {
                    ui.label("Country:");
                    ui.label(country);
                    ui.end_row();
                }

                if let Some(launch) = &obj.launch_date {
                    ui.label("Launch:");
                    ui.label(launch);
                    ui.end_row();
                }

                if let Some(decay) = &obj.decay_date {
                    ui.label("Decay:");
                    ui.colored_label(Color32::from_rgb(200, 100, 100), decay);
                    ui.end_row();
                }

                ui.label("In SATCAT:");
                ui.label(if obj.in_satcat { "Yes" } else { "No" });
                ui.end_row();

                ui.label("Sources:");
                let mut sources = Vec::new();
                if obj.sources.spacetrack_satcat {
                    sources.push("Space-Track SATCAT");
                }
                if obj.sources.spacetrack_gp {
                    sources.push("Space-Track GP");
                }
                if obj.sources.gcat {
                    sources.push("GCAT");
                }
                if obj.sources.discos {
                    sources.push("DISCOS");
                }
                let sources_text = if sources.is_empty() {
                    "Unknown".to_string()
                } else {
                    sources.join(", ")
                };
                ui.label(sources_text);
                ui.end_row();
            });

        // TLE information
        if let Some(tle) = &obj.tle {
            ui.separator();
            ui.heading("TLE Data");

            egui::Grid::new("tle_grid")
                .num_columns(2)
                .spacing([10.0, 4.0])
                .show(ui, |ui| {
                    ui.label("Epoch:");
                    ui.label(&tle.epoch);
                    ui.end_row();

                    if let Some(age) = tle_age_days {
                        ui.label("TLE Age:");
                        let color = if age < 7.0 {
                            Color32::from_rgb(100, 200, 100)
                        } else if age < 30.0 {
                            Color32::from_rgb(200, 200, 100)
                        } else {
                            Color32::from_rgb(200, 100, 100)
                        };
                        ui.colored_label(color, format!("{:.1} days", age));
                        ui.end_row();
                    }
                });

            // Show TLE lines in monospace
            ui.separator();
            ui.label("TLE Lines:");
            ui.monospace(&tle.line1);
            ui.monospace(&tle.line2);
        } else {
            ui.separator();
            ui.colored_label(Color32::from_rgb(200, 100, 100), "No TLE available");
        }

        // DISCOS data
        if let Some(discos) = &obj.discos {
            ui.separator();
            ui.heading("Physical Properties (DISCOS)");

            egui::Grid::new("discos_grid")
                .num_columns(2)
                .spacing([10.0, 4.0])
                .show(ui, |ui| {
                    if let Some(name) = &discos.name {
                        ui.label("DISCOS name:");
                        ui.label(name);
                        ui.end_row();
                    }

                    if let Some(class) = &discos.object_class {
                        ui.label("Object class:");
                        ui.label(class);
                        ui.end_row();
                    }

                    if let Some(cospar) = &discos.cospar_id {
                        ui.label("DISCOS COSPAR:");
                        ui.label(cospar);
                        ui.end_row();
                    }

                    if let Some(satno) = discos.satno {
                        ui.label("DISCOS SATNO:");
                        ui.label(format!("{}", satno));
                        ui.end_row();
                    }

                    if let Some(mass) = discos.mass {
                        ui.label("Mass:");
                        ui.label(format!("{:.1} kg", mass));
                        ui.end_row();
                    }

                    if let Some(shape) = &discos.shape {
                        ui.label("Shape:");
                        ui.label(shape);
                        ui.end_row();
                    }

                    if let Some(width) = discos.width {
                        ui.label("Width:");
                        ui.label(format!("{:.2} m", width));
                        ui.end_row();
                    }

                    if let Some(height) = discos.height {
                        ui.label("Height:");
                        ui.label(format!("{:.2} m", height));
                        ui.end_row();
                    }

                    if let Some(depth) = discos.depth {
                        ui.label("Depth:");
                        ui.label(format!("{:.2} m", depth));
                        ui.end_row();
                    }

                    if let Some(diameter) = discos.diameter {
                        ui.label("Diameter:");
                        ui.label(format!("{:.2} m", diameter));
                        ui.end_row();
                    }

                    if let Some(span) = discos.span {
                        ui.label("Span:");
                        ui.label(format!("{:.2} m", span));
                        ui.end_row();
                    }

                    if let Some(xsect) = discos.x_sect_min {
                        ui.label("Cross-section min:");
                        ui.label(format!("{:.2} m²", xsect));
                        ui.end_row();
                    }

                    if let Some(xsect) = discos.x_sect_avg {
                        ui.label("Cross-section avg:");
                        ui.label(format!("{:.2} m²", xsect));
                        ui.end_row();
                    }

                    if let Some(xsect) = discos.x_sect_max {
                        ui.label("Cross-section max:");
                        ui.label(format!("{:.2} m²", xsect));
                        ui.end_row();
                    }
                });
        }
    }
}
