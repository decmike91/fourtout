//! Onglet « Planétaire IA » : traitement planétaire entièrement automatique.

use crate::job::Job;
use crate::plot;
use crate::video::VideoSlot;
use crate::viewer::Viewer;
use astroforge::image::Image;
use astroforge::io;
use astroforge::planetary::PlanetOptions;
use astroforge::planetary_auto::{self as pa, AutoResult, Effort, RestoreEval};
use eframe::egui::{self, Color32, RichText};
use std::sync::Arc;

#[derive(PartialEq, Clone, Copy)]
enum View {
    Video,
    Best,
    Stack,
    Result,
    Compare,
}

pub enum Action {
    None,
    /// envoie le plan retenu vers l'onglet manuel (options, gains d'ondelettes)
    SendToManual(PlanetOptions, Vec<f64>),
}

pub struct AutoTab {
    effort: Effort,
    use_memory: bool,
    learn: bool,
    onnx: bool,
    color_balance: bool,
    job: Option<Job<AutoResult>>,
    result: Option<Arc<AutoResult>>,
    result_video: String,
    view: View,
    viewer: Viewer,
    frame_idx: usize,
    frame: Option<(u64, usize, Image)>,
    // retouche
    strength: f64,
    denoise_k: f64,
    shown: Option<Image>,
    shown_eval: Option<RestoreEval>,
    tweak_job: Option<Job<(Image, RestoreEval)>>,
    tweak_dirty: bool,
    keys: u64,
    key_result: u64,
    key_stack: u64,
    key_best: u64,
    error: Option<String>,
    status: Option<String>,
    last_gen: u64,
    start_requested: bool,
}

impl Default for AutoTab {
    fn default() -> Self {
        AutoTab {
            effort: Effort::Equilibre,
            use_memory: true,
            learn: true,
            onnx: true,
            color_balance: true,
            job: None,
            result: None,
            result_video: String::new(),
            view: View::Video,
            viewer: Viewer::default(),
            frame_idx: 0,
            frame: None,
            strength: 0.0,
            denoise_k: 0.0,
            shown: None,
            shown_eval: None,
            tweak_job: None,
            tweak_dirty: false,
            keys: 1000,
            key_result: 0,
            key_stack: 0,
            key_best: 0,
            error: None,
            status: None,
            last_gen: 0,
            start_requested: false,
        }
    }
}

fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(10.0);
    ui.label(RichText::new(title).strong().color(plot::ACCENT));
    ui.add_space(2.0);
}

impl AutoTab {
    fn next_key(&mut self) -> u64 {
        self.keys += 1;
        self.keys
    }

    /// Lance le traitement dès que possible (option --auto).
    pub fn request_start(&mut self) {
        self.start_requested = true;
    }

    pub fn is_busy(&self) -> bool {
        self.job.is_some()
    }

    fn start(&mut self, ctx: &egui::Context, slot: &VideoSlot) {
        let Some(video) = slot.video.clone() else { return };
        self.error = None;
        self.status = None;
        let opts = pa::AutoOptions { effort: self.effort, use_memory: self.use_memory, learn: self.learn, onnx: self.onnx, color_balance: self.color_balance, ..Default::default() };
        self.result_video = video.name.clone();
        let job = Job::spawn(ctx, move |rep| {
            let mut o = opts;
            o.cancel = Some(rep.cancel.clone());
            let pf = |f: f64, m: &str| rep.report(f, m);
            pa::run_auto(&*video.frames, &o, Some(&pf))
        });
        self.job = Some(job);
    }

    fn poll(&mut self, ctx: &egui::Context) {
        if let Some(j) = &mut self.job {
            if let Some(r) = j.poll() {
                let secs = j.started.elapsed().as_secs_f64();
                self.job = None;
                match r {
                    Ok(res) => {
                        self.strength = res.plan.strength;
                        self.denoise_k = res.plan.denoise_k;
                        self.shown = Some(res.image.clone());
                        self.shown_eval = None;
                        self.key_result = self.next_key();
                        self.key_stack = self.next_key();
                        self.key_best = self.next_key();
                        self.status = Some(format!("Terminé en {:.1} s", secs));
                        self.result = Some(Arc::new(res));
                        self.view = View::Compare;
                    }
                    Err(e) => self.error = Some(e),
                }
            }
        }
        if let Some(j) = &mut self.tweak_job {
            if let Some(r) = j.poll() {
                self.tweak_job = None;
                if let Ok((img, ev)) = r {
                    self.shown = Some(img);
                    self.shown_eval = Some(ev);
                    self.key_result = self.next_key();
                }
            }
        }
        if self.tweak_dirty && self.tweak_job.is_none() {
            self.tweak_dirty = false;
            if let Some(r) = self.result.clone() {
                let (s, k) = (self.strength, self.denoise_k);
                self.tweak_job = Some(Job::spawn(ctx, move |_| {
                    let ev = r.finisher.evaluate(s, k);
                    Ok((r.finisher.render(s, k).image, ev))
                }));
            }
        }
    }

    pub fn ui(&mut self, ctx: &egui::Context, slot: &mut VideoSlot) -> Action {
        self.poll(ctx);
        if slot.generation != self.last_gen {
            self.last_gen = slot.generation;
            self.frame = None;
            self.frame_idx = 0;
            if self.result.is_none() {
                self.view = View::Video;
            }
        }
        if self.start_requested && self.job.is_none() && slot.video.is_some() {
            self.start_requested = false;
            self.start(ctx, slot);
        }
        let mut action = Action::None;
        egui::SidePanel::left("auto_left").resizable(false).exact_width(330.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                action = self.left_panel(ui, slot);
            });
        });
        if self.result.is_some() {
            egui::SidePanel::right("auto_right").resizable(false).exact_width(390.0).show(ctx, |ui| {
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| self.right_panel(ui));
            });
        }
        egui::CentralPanel::default().show(ctx, |ui| self.center(ui, slot));
        action
    }

    fn left_panel(&mut self, ui: &mut egui::Ui, slot: &mut VideoSlot) -> Action {
        let mut action = Action::None;
        ui.add_space(6.0);
        ui.heading("Planétaire IA");
        ui.label(RichText::new("Traitement entièrement automatique : le moteur local analyse la vidéo, trie les images, choisit l'alignement, l'empilement, les ondelettes et le débruitage, puis vérifie qu'aucun artefact n'est introduit.").small());
        section(ui, "1 · Vidéo");
        let busy = self.job.is_some();
        ui.add_enabled_ui(!busy, |ui| slot.ui(ui));

        section(ui, "2 · Moteur IA local");
        ui.add_enabled_ui(!busy, |ui| {
            ui.horizontal(|ui| {
                ui.label("Recherche :");
                egui::ComboBox::from_id_salt("effort").selected_text(self.effort.label()).show_ui(ui, |ui| {
                    for e in Effort::ALL {
                        ui.selectable_value(&mut self.effort, e, e.label());
                    }
                });
            });
            ui.label(
                RichText::new(match self.effort {
                    Effort::Rapide => "Règles expertes et mémoire, un seul alignement essayé.",
                    Effort::Equilibre => "Essaie les alignements et empilements les plus probables (recommandé).",
                    Effort::Maximal => "Essaie toutes les méthodes, tailles de boîtes et réglages fins.",
                })
                .small()
                .weak(),
            );
            ui.checkbox(&mut self.use_memory, "Réutiliser l'expérience des vidéos précédentes");
            ui.checkbox(&mut self.learn, "Mémoriser ce traitement (apprentissage local)");
            ui.checkbox(&mut self.onnx, "Essayer un modèle ONNX de débruitage installé");
            ui.checkbox(&mut self.color_balance, "Équilibre automatique des couleurs");
        });
        ui.add_space(10.0);
        if let Some(j) = &self.job {
            let (p, m) = j.progress();
            ui.add(egui::ProgressBar::new(p).text(m).animate(true));
            ui.label(RichText::new(format!("{:.0} s écoulées", j.started.elapsed().as_secs_f64())).small().weak());
            if ui.button("Annuler").clicked() {
                j.request_cancel();
            }
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(250));
        } else {
            let enabled = slot.video.is_some() && !slot.is_loading();
            let btn = egui::Button::new(RichText::new("Lancer le traitement automatique").strong().size(15.0).color(if enabled { Color32::BLACK } else { Color32::GRAY }))
                .fill(if enabled { plot::ACCENT } else { Color32::from_gray(60) })
                .min_size(egui::vec2(ui.available_width(), 38.0));
            if ui.add_enabled(enabled, btn).clicked() {
                self.start(ui.ctx(), slot);
            }
        }
        if let Some(s) = &self.status {
            ui.label(RichText::new(s).small().color(Color32::from_rgb(140, 200, 140)));
        }
        if let Some(e) = &self.error {
            ui.colored_label(Color32::from_rgb(230, 110, 100), e);
        }

        if let Some(r) = self.result.clone() {
            section(ui, "3 · Retouche (facultatif)");
            ui.label(RichText::new(format!("Choix de l'IA : force {:.2}, débruitage {:.2} σ", r.plan.strength, r.plan.denoise_k)).small().weak());
            let s1 = ui.add(egui::Slider::new(&mut self.strength, 0.0..=1.5).text("Force des détails").step_by(0.05));
            let s2 = ui.add(egui::Slider::new(&mut self.denoise_k, 0.0..=3.0).text("Débruitage (σ)").step_by(0.1));
            let changed = |r: &egui::Response| r.drag_stopped() || (r.changed() && !r.dragged());
            if changed(&s1) || changed(&s2) {
                self.tweak_dirty = true;
            }
            if ui.button("Revenir au choix de l'IA").clicked() {
                self.strength = r.plan.strength;
                self.denoise_k = r.plan.denoise_k;
                self.tweak_dirty = true;
            }
            if self.tweak_job.is_some() {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Calcul…");
                });
            } else if let Some(ev) = &self.shown_eval {
                let ok = ev.ok;
                ui.label(RichText::new(format!("Bruit visible {:.2} % · halo {:.1} % · détails ×{:.2}", ev.noise_rel * 100.0, ev.ring * 100.0, ev.detail)).small());
                ui.label(
                    RichText::new(if ok { "Dans les limites « sans perte » du moteur." } else { "Au-delà des limites du moteur : bruit ou halo visibles." })
                        .small()
                        .color(if ok { Color32::from_rgb(140, 200, 140) } else { Color32::from_rgb(235, 170, 90) }),
                );
            }

            section(ui, "4 · Exporter");
            let stem = std::path::Path::new(&self.result_video).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "planete".into());
            if ui.button("Enregistrer le résultat…").on_hover_text("TIFF 16 bits, PNG 16 bits, FITS 32 bits ou JPEG").clicked() {
                if let Some(img) = self.shown.clone() {
                    self.save_dialog(&img, &format!("{}_IA.tif", stem));
                }
            }
            if ui.button("Enregistrer l'empilement brut…").on_hover_text("Avant restauration, pour un traitement personnel").clicked() {
                let img = r.stack.clone();
                self.save_dialog(&img, &format!("{}_empilement.tif", stem));
            }
            if ui.button("Exporter le rapport (JSON)…").clicked() {
                if let Some(p) = rfd::FileDialog::new().add_filter("JSON", &["json"]).set_file_name(format!("{}_rapport.json", stem)).save_file() {
                    let mut j = r.report_json();
                    j["video"] = serde_json::json!(self.result_video);
                    j["retouche"] = serde_json::json!({"strength": self.strength, "denoise_k": self.denoise_k});
                    match std::fs::write(&p, serde_json::to_string_pretty(&j).unwrap_or_default()) {
                        Ok(_) => self.status = Some(format!("Rapport écrit : {}", p.display())),
                        Err(e) => self.error = Some(e.to_string()),
                    }
                }
            }
            if ui.button("Ouvrir ces réglages dans l'onglet Planétaire").on_hover_text("Reprend le tri, l'alignement, l'empilement et les gains d'ondelettes choisis par l'IA pour les ajuster à la main").clicked() {
                action = Action::SendToManual(r.manual_options(), r.finisher.gains(self.strength));
            }
        }
        action
    }

    fn save_dialog(&mut self, img: &Image, default: &str) {
        if let Some(p) = rfd::FileDialog::new()
            .add_filter("TIFF 16 bits", &["tif", "tiff"])
            .add_filter("PNG 16 bits", &["png"])
            .add_filter("FITS 32 bits", &["fits", "fit"])
            .add_filter("JPEG", &["jpg", "jpeg"])
            .set_file_name(default)
            .save_file()
        {
            let depth = if io::ext_of(&p) == "png" { "16" } else { "auto" };
            match io::save_image(&p, img, depth, None) {
                Ok(_) => self.status = Some(format!("Écrit : {}", p.display())),
                Err(e) => self.error = Some(format!("{:#}", e)),
            }
        }
    }

    fn right_panel(&mut self, ui: &mut egui::Ui) {
        let Some(r) = self.result.clone() else { return };
        ui.add_space(6.0);
        ui.heading("Bilan");
        let m = &r.metrics;
        let f = &r.features;
        egui::Grid::new("metrics").num_columns(2).spacing([16.0, 3.0]).show(ui, |ui| {
            let mut row = |k: &str, v: String| {
                ui.label(RichText::new(k).weak());
                ui.label(v);
                ui.end_row();
            };
            row("Frames gardées", format!("{} / {} ({:.1} %)", r.plan.keep, f.n_frames, r.plan.keep_percent));
            row("Rejetées", format!("{}", f.rejected));
            row("Alignement", r.plan.align_mode.clone());
            row("Empilement", r.plan.stack_method.clone());
            row("RSB d'une frame", format!("{:.0}", f.snr));
            row("RSB empilé", format!("{:.0} (×{:.1})", m.stack_snr, m.snr_gain));
            row("PSF (limbe)", format!("σ {:.2} px", r.plan.psf_sigma));
            row("Détails", format!("×{:.2}", m.detail_gain));
            row("Bruit visible", format!("{:.2} %", m.noise_out * 100.0));
            row("Écrêtage", format!("{:.3} %", m.clipped * 100.0));
            row("Mémoire", if r.memory_hits > 0 { format!("{} vidéo(s) proche(s)", r.memory_hits) } else { "aucune vidéo proche".into() });
        });

        section(ui, "Décisions de l'IA");
        for d in &r.decisions {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(&d.step).strong());
                ui.label(&d.choice);
            });
            ui.label(RichText::new(&d.reason).small().weak());
            ui.add_space(4.0);
        }

        section(ui, "Qualité des frames (triées)");
        let mut q: Vec<f64> = r.scores.iter().cloned().filter(|v| v.is_finite()).collect();
        q.sort_by(|a, b| b.total_cmp(a));
        let n = q.len().max(1) as f64;
        let xs: Vec<f64> = (0..q.len()).map(|i| (i as f64 + 0.5) * 100.0 / n).collect();
        let cut = r.plan.keep as f64 * 100.0 / n;
        plot::line(ui, &xs, &q, None, Some(cut), "rang (%) — trait : frames gardées", &|x| format!("{:.0} %", x), 90.0);

        section(ui, "Choix du nombre de frames");
        let xs: Vec<f64> = r.keep_curve.iter().map(|c| c.0).collect();
        let ys: Vec<f64> = r.keep_curve.iter().map(|c| c.1).collect();
        plot::line(ui, &xs, &ys, Some(r.plan.keep_percent), None, "frames gardées — score du critique", &|x| format!("{:.0} %", x), 100.0);

        section(ui, "Essais d'alignement et d'empilement");
        let best = r.trials.iter().enumerate().max_by(|a, b| a.1 .1.total_cmp(&b.1 .1)).map(|x| x.0);
        plot::bars(ui, &r.trials, best);

        section(ui, "Gains d'ondelettes");
        let gains = r.finisher.gains(self.strength);
        let items: Vec<(String, f64)> = gains.iter().enumerate().map(|(i, g)| (if i + 1 == gains.len() { "résidu".to_string() } else { format!("couche {}", i + 1) }, *g)).collect();
        plot::bars(ui, &items, None);
        ui.add_space(12.0);
    }

    fn center(&mut self, ui: &mut egui::Ui, slot: &VideoSlot) {
        let has = self.result.is_some();
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.view, View::Video, "Vidéo");
            ui.add_enabled_ui(has, |ui| {
                ui.selectable_value(&mut self.view, View::Best, "Meilleure frame");
                ui.selectable_value(&mut self.view, View::Stack, "Empilement brut");
                ui.selectable_value(&mut self.view, View::Result, "Résultat");
                ui.selectable_value(&mut self.view, View::Compare, "Avant / après");
            });
            ui.separator();
            self.viewer.toolbar(ui);
        });
        ui.separator();
        if !has && self.view != View::Video {
            self.view = View::Video;
        }
        match self.view {
            View::Video => {
                let Some(v) = slot.video.clone() else {
                    ui.centered_and_justified(|ui| {
                        ui.label(RichText::new("Ouvrez une vidéo SER (ou déposez-la ici), puis lancez le traitement automatique.").size(16.0).weak());
                    });
                    return;
                };
                if v.n > 1 {
                    ui.horizontal(|ui| {
                        ui.label("Frame");
                        ui.spacing_mut().slider_width = (ui.available_width() - 90.0).max(100.0);
                        ui.add(egui::Slider::new(&mut self.frame_idx, 0..=v.n - 1));
                    });
                }
                self.frame_idx = self.frame_idx.min(v.n.saturating_sub(1));
                let need = self.frame.as_ref().map(|f| f.0 != slot.generation || f.1 != self.frame_idx).unwrap_or(true);
                if need {
                    self.frame = Some((slot.generation, self.frame_idx, v.frames.get(self.frame_idx)));
                }
                let f = self.frame.as_ref().unwrap();
                // clé propre à (vidéo, frame), distincte des autres vues
                let k = (1u64 << 40) + f.0 * 1_000_000 + f.1 as u64;
                self.viewer.show(ui, &f.2, k, true);
            }
            View::Best => {
                let r = self.result.clone().unwrap();
                self.viewer.show(ui, &r.best_frame, self.key_best, true);
            }
            View::Stack => {
                let r = self.result.clone().unwrap();
                self.viewer.show(ui, &r.stack, self.key_stack, true);
            }
            View::Result => {
                if let Some(img) = self.shown.clone() {
                    self.viewer.show(ui, &img, self.key_result, false);
                }
            }
            View::Compare => {
                let r = self.result.clone().unwrap();
                if let Some(img) = self.shown.clone() {
                    self.viewer.show_compare(ui, &r.stack, self.key_stack, &img, self.key_result);
                }
            }
        }
    }
}
