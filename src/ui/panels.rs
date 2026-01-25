//! UI panels for object browser, search, and details

use crate::data::{ObjectFilter, SearchIndex, SpaceObject, SpaceObjectDatabase};
use egui::{Color32, RichText, Ui};

/// Search and filter panel state
#[derive(Default)]
pub struct SearchPanel {
    pub query: String,
    pub results: Vec<u32>,
    pub filter: ObjectFilter,
    pub show_filters: bool,
}

impl SearchPanel {
    pub fn show(&mut self, ui: &mut Ui, index: &SearchIndex) {
        ui.heading("Search");

        // Search box
        let response = ui.text_edit_singleline(&mut self.query);
        if response.changed() {
            self.results = index.search(&self.query, 100);
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
        }

        if self.show_filters {
            ui.separator();
            ui.checkbox(&mut self.filter.has_tle_only, "Has TLE only");
            ui.checkbox(&mut self.filter.exclude_decayed, "Exclude decayed");
        }

        ui.separator();
        ui.label(format!("{} results", self.results.len()));
    }
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
        filter: &ObjectFilter,
        selected: Option<u32>,
    ) -> Option<u32> {
        let mut new_selection = None;

        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for &norad_id in results {
                    let norad_str = norad_id.to_string();
                    if let Some(obj) = db.objects.get(&norad_str) {
                        // Apply filter
                        if !filter.matches(obj) {
                            continue;
                        }

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

                    let (w, h, d) = discos.dimensions();
                    ui.label("Dimensions:");
                    ui.label(format!("{:.2} x {:.2} x {:.2} m", w, h, d));
                    ui.end_row();

                    if let Some(xsect) = discos.x_sect_avg {
                        ui.label("Cross-section:");
                        ui.label(format!("{:.2} m²", xsect));
                        ui.end_row();
                    }
                });
        }
    }
}
