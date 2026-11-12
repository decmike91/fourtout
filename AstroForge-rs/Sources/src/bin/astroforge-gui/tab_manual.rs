//! Onglet « Planétaire » : traitement planétaire réglé à la main (6 étapes)
//! et finitions (ondelettes, alignement RVB, niveaux).

use crate::job::Job;
use crate::plot;
use crate::video::VideoSlot;
use crate::viewer::Viewer;
use astroforge::image::Image;
use astroforge::io;
use astroforge::planetary::{self, PlanetOptions, PlanetStats, ALIGN_MODES, QUALITY_METHODS};
use astroforge::registration;
use astroforge::wavelets;
use eframe::egui::{self, Color32, RichText};

const STACK_METHODS: [&str; 5] = ["Moyenne", "Moyenne (meilleurs %)", "Sigma-clipping", "Médiane", "Winsorisé"];

pub struct ManualTab {
    opts: PlanetOptions,
    keep_percent: f64,
    rgb_align: bool,
    gains: [f64; 7],
    denoise: f64,
    auto_levels: bool,
    job: Option<Job<(Image, PlanetStats)>>,
    stack: Option<Image>,
    stats: Option<PlanetStats>,
    result: Option<Image>,
    show_stack: bool,
    viewer: Viewer,
    key: u64,
    error: Option<String>,
    status: Option<String>,
    from_auto: bool,
}

impl Default for ManualTab {
    fn default() -> Self {
        ManualTab {
            opts: PlanetOptions::default(),
            keep_percent: 25.0,
            rgb_align: true,
            gains: [1.6, 1.3, 1.1, 1.0, 1.0, 1.0, 1.0],
            denoise: 0.0,
            auto_levels: true,
            job: None,
            stack: None,
            stats: None,
            result: None,
            show_stack: false,
            viewer: Viewer::default(),
            key: 1,
            error: None,
            status: None,
            from_auto: false,
        }
    }
}

impl ManualTab {
    /// Reprend les réglages choisis par l'onglet automatique.
    pub fn apply_from_auto(&mut self, o: PlanetOptions, gains: Vec<f64>, n_frames: usize) {
        self.keep_percent = (o.n_align as f64 * 100.0 / n_frames.max(1) as f64).clamp(0.1, 100.0);
        self.opts = o;
        for (i, g) in gains.iter().take(7).enumerate() {
            self.gains[i] = g.clamp(0.0, 12.0);
        }
        self.denoise = 0.0;
        self.rgb_align = true;
        self.from_auto = true;
    }

    fn finish(&mut self) {
        let Some(st) = &self.stack else { return };
        let mut img = st.clone();
        if self.rgb_align && img.is_color() {
            img = registration::align_channels(&img, Some(8.0));
        }
        match wavelets::registax_enhance(&img, Some(&self.gains), self.denoise, None, None) {
            Ok(i) => img = i,
            Err(e) => self.error = Some(format!("{:#}", e)),
        }
        if self.auto_levels {
            if let Ok((t, _, _)) = planetary::auto_contrast_params(&img, 0.0005) {
                img = t.apply(&img);
            }
        }
        self.result = Some(img.clip01());
        self.key += 1;
    }

    pub fn ui(&mut self, ctx: &egui::Context, slot: &mut VideoSlot) {
        if let Some(j) = &mut self.job {
            if let Some(r) = j.poll() {
                let secs = j.started.elapsed().as_secs_f64();
                self.job = None;
                match r {
                    Ok((img, st)) => {
                        self.status = Some(format!("{} frames empilées sur {} en {:.1} s ({} points d'alignement)", st.stacked, st.n_frames, secs, st.n_points));
                        self.stack = Some(img);
                        self.stats = Some(st);
                        self.finish();
                    }
                    Err(e) => self.error = Some(e),
                }
            }
        }
        egui::SidePanel::left("manual_left").resizable(false).exact_width(330.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| self.left(ui, slot));
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.add_enabled_ui(self.stack.is_some(), |ui| {
                    ui.selectable_value(&mut self.show_stack, false, "Résultat");
                    ui.selectable_value(&mut self.show_stack, true, "Empilement brut");
                });
                ui.separator();
                self.viewer.toolbar(ui);
            });
            ui.separator();
            let img = if self.show_stack { self.stack.clone() } else { self.result.clone() };
            match img {
                Some(i) => {
                    let k = self.key * 2 + self.show_stack as u64;
                    self.viewer.show(ui, &i, k, self.show_stack);
                }
                None => {
                    ui.centered_and_justified(|ui| {
                        ui.label(RichText::new("Réglez les 6 étapes puis lancez l'empilement.\nL'onglet « Planétaire IA » fait tout cela automatiquement.").size(16.0).weak());
                    });
                }
            }
        });
    }

    fn left(&mut self, ui: &mut egui::Ui, slot: &mut VideoSlot) {
        let busy = self.job.is_some();
        ui.add_space(6.0);
        ui.heading("Planétaire");
        ui.label(RichText::new("Réglages manuels (principes de Registax 6, AutoStakkert!).").small());
        if self.from_auto {
            ui.label(RichText::new("Réglages repris de l'onglet Planétaire IA.").small().color(plot::ACCENT));
        }
        ui.add_space(8.0);
        ui.label(RichText::new("Vidéo").strong().color(plot::ACCENT));
        ui.add_enabled_ui(!busy, |ui| slot.ui(ui));
        ui.add_space(8.0);
        ui.label(RichText::new("Tri et alignement").strong().color(plot::ACCENT));
        ui.add_enabled_ui(!busy, |ui| {
            combo(ui, "Qualité", "q", &mut self.opts.quality_method, &QUALITY_METHODS);
            ui.add(egui::Slider::new(&mut self.keep_percent, 0.5..=100.0).text("% de frames gardées").logarithmic(true));
            combo(ui, "Alignement", "a", &mut self.opts.align_mode, &ALIGN_MODES);
            if self.opts.align_mode.starts_with("Multi-points (grille") {
                ui.add(egui::Slider::new(&mut self.opts.grid_n, 2..=8).text("grille N×N"));
            }
            ui.add(egui::Slider::new(&mut self.opts.box_size, 0..=128).text("taille des boîtes (0 = auto)"));
            combo(ui, "Empilement", "s", &mut self.opts.stack_method, &STACK_METHODS);
            if self.opts.align_mode.starts_with("Multi-points local") || self.opts.stack_method.starts_with("Moyenne (meilleurs") {
                ui.add(egui::Slider::new(&mut self.opts.stack_percent, 5.0..=100.0).text("% par point / meilleurs"));
            }
        });
        ui.add_space(8.0);
        if let Some(j) = &self.job {
            let (p, m) = j.progress();
            ui.add(egui::ProgressBar::new(p).text(m).animate(true));
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(250));
        } else {
            let enabled = slot.video.is_some() && !slot.is_loading();
            if ui.add_enabled(enabled, egui::Button::new(RichText::new("Empiler").strong()).min_size(egui::vec2(ui.available_width(), 32.0))).clicked() {
                self.run(ui.ctx(), slot);
            }
        }
        if let Some(s) = &self.status {
            ui.label(RichText::new(s).small().color(Color32::from_rgb(140, 200, 140)));
        }
        if let Some(e) = &self.error {
            ui.colored_label(Color32::from_rgb(230, 110, 100), e);
        }
        ui.add_space(8.0);
        ui.label(RichText::new("Finitions").strong().color(plot::ACCENT));
        let mut changed = false;
        let upd = |r: egui::Response| r.drag_stopped() || (r.changed() && !r.dragged());
        for i in 0..7 {
            let label = if i == 6 { "résidu".to_string() } else { format!("ondelette {}", i + 1) };
            changed |= upd(ui.add(egui::Slider::new(&mut self.gains[i], 0.0..=8.0).text(label).step_by(0.05)));
        }
        changed |= upd(ui.add(egui::Slider::new(&mut self.denoise, 0.0..=1.0).text("débruitage").step_by(0.05)));
        changed |= ui.checkbox(&mut self.rgb_align, "Aligner les canaux RVB").changed();
        changed |= ui.checkbox(&mut self.auto_levels, "Niveaux automatiques").changed();
        if ui.button("Gains par défaut").clicked() {
            self.gains = [1.6, 1.3, 1.1, 1.0, 1.0, 1.0, 1.0];
            changed = true;
        }
        if changed {
            self.finish();
        }
        ui.add_space(8.0);
        if let Some(img) = self.result.clone() {
            if ui.button("Enregistrer le résultat…").clicked() {
                if let Some(p) = rfd::FileDialog::new()
                    .add_filter("TIFF 16 bits", &["tif", "tiff"])
                    .add_filter("PNG 16 bits", &["png"])
                    .add_filter("FITS 32 bits", &["fits", "fit"])
                    .add_filter("JPEG", &["jpg", "jpeg"])
                    .set_file_name("planete.tif")
                    .save_file()
                {
                    let depth = if io::ext_of(&p) == "png" { "16" } else { "auto" };
                    match io::save_image(&p, &img, depth, None) {
                        Ok(_) => self.status = Some(format!("Écrit : {}", p.display())),
                        Err(e) => self.error = Some(format!("{:#}", e)),
                    }
                }
            }
        }
    }

    fn run(&mut self, ctx: &egui::Context, slot: &VideoSlot) {
        let Some(v) = slot.video.clone() else { return };
        self.error = None;
        let mut o = self.opts.clone();
        o.n_align = ((v.n as f64 * self.keep_percent / 100.0).round() as usize).clamp(1, v.n);
        self.job = Some(Job::spawn(ctx, move |rep| {
            let pf = |d: usize, t: usize, l: &str| rep.report(d as f64 / t.max(1) as f64, l);
            let r = planetary::run_planetary(&*v.frames, &o, Some(&pf))?;
            Ok((r.image, r.stats))
        }));
    }
}

fn combo(ui: &mut egui::Ui, label: &str, id: &str, value: &mut String, choices: &[&str]) {
    ui.horizontal(|ui| {
        ui.label(label);
        egui::ComboBox::from_id_salt(id).selected_text(value.as_str()).width(ui.available_width() - 8.0).show_ui(ui, |ui| {
            for c in choices {
                ui.selectable_value(value, c.to_string(), *c);
            }
        });
    });
}
