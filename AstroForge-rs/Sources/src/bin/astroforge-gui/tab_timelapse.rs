//! Onglet « Time-lapse » : plusieurs vidéos -> une image chacune, même
//! traitement, même cadrage, même rendu -> animation fluide.

use crate::job::Job;
use crate::plot;
use crate::viewer::Viewer;
use astroforge::image::Image;
use astroforge::planetary_auto::Effort;
use astroforge::timelapse::{self as tl, Brightness, Framing, SeqInput, SeqProcessed};
use eframe::egui::{self, Color32, RichText};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

const FRAME_FORMATS: [&str; 4] = ["png", "tif", "fits", "jpg"];

pub struct TimelapseTab {
    inputs: Vec<SeqInput>,
    loading: Option<Job<Vec<SeqInput>>>,
    path_text: String,
    // traitement
    effort: Effort,
    reference: Option<usize>,
    framing: Framing,
    margin: f64,
    field_rotation: bool,
    use_memory: bool,
    onnx: bool,
    color_balance: bool,
    job: Option<Job<SeqProcessed>>,
    processed: Option<Arc<SeqProcessed>>,
    // rendu
    render: tl::RenderOptions,
    rendered: Vec<Image>,
    anim: Vec<Image>,
    dirty: bool,
    // animation
    fps: u32,
    between: usize,
    frame_format: String,
    playing: bool,
    pos: usize,
    last_tick: Instant,
    viewer: Viewer,
    key_base: u64,
    error: Option<String>,
    status: Option<String>,
    autorun: bool,
}

impl Default for TimelapseTab {
    fn default() -> Self {
        TimelapseTab {
            inputs: Vec::new(),
            loading: None,
            path_text: String::new(),
            effort: Effort::Equilibre,
            reference: None,
            framing: Framing::Auto,
            margin: 0.25,
            field_rotation: false,
            use_memory: true,
            onnx: true,
            color_balance: true,
            job: None,
            processed: None,
            render: tl::RenderOptions::default(),
            rendered: Vec::new(),
            anim: Vec::new(),
            dirty: false,
            fps: 10,
            between: 2,
            frame_format: "png".into(),
            playing: false,
            pos: 0,
            last_tick: Instant::now(),
            viewer: Viewer::default(),
            key_base: 1 << 50,
            error: None,
            status: None,
            autorun: false,
        }
    }
}

fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(10.0);
    ui.label(RichText::new(title).strong().color(plot::ACCENT));
    ui.add_space(2.0);
}

fn hms(t: &Option<chrono::NaiveDateTime>) -> String {
    t.map(|t| t.format("%H:%M:%S").to_string()).unwrap_or_else(|| "—".into())
}

impl TimelapseTab {
    pub fn is_busy(&self) -> bool {
        self.job.is_some()
    }

    /// Ouverture au lancement (option --timelapse) ; `run` lance aussitôt.
    pub fn preload(&mut self, ctx: &egui::Context, paths: Vec<PathBuf>, run: bool) {
        self.add_paths(ctx, paths);
        self.autorun = run;
    }

    fn add_paths(&mut self, ctx: &egui::Context, paths: Vec<PathBuf>) {
        if paths.is_empty() || self.loading.is_some() {
            return;
        }
        self.error = None;
        self.loading = Some(Job::spawn(ctx, move |rep| {
            rep.report(0.1, "Ouverture des vidéos…");
            tl::load_inputs(&paths, 0)
        }));
    }

    fn rerender(&mut self) {
        if let Some(p) = &self.processed {
            self.rendered = tl::render(p, &self.render);
            self.anim = tl::render_animation(p, &self.render, self.between);
            self.pos = self.pos.min(self.anim.len().saturating_sub(1));
            self.key_base += 1 << 20;
        }
        self.dirty = false;
    }

    fn poll(&mut self) {
        if let Some(j) = &mut self.loading {
            if let Some(r) = j.poll() {
                self.loading = None;
                match r {
                    Ok(v) => {
                        let known: std::collections::HashSet<String> = self.inputs.iter().map(|i| i.name.clone()).collect();
                        self.inputs.extend(v.into_iter().filter(|i| !known.contains(&i.name)));
                        tl::sort_inputs(&mut self.inputs);
                        self.reference = None;
                    }
                    Err(e) => self.error = Some(e),
                }
            }
        }
        if let Some(j) = &mut self.job {
            if let Some(r) = j.poll() {
                let secs = j.started.elapsed().as_secs_f64();
                self.job = None;
                match r {
                    Ok(p) => {
                        self.status = Some(format!("{} image(s) en {:.0} s", p.frames.len(), secs));
                        self.processed = Some(Arc::new(p));
                        self.pos = 0;
                        self.playing = true;
                        self.rerender();
                    }
                    Err(e) => self.error = Some(e),
                }
            }
        }
        if self.dirty {
            self.rerender();
        }
    }

    fn start(&mut self, ctx: &egui::Context) {
        if self.inputs.is_empty() {
            return;
        }
        self.error = None;
        self.status = None;
        let inputs = self.inputs.clone();
        let o = tl::SeqOptions {
            effort: self.effort,
            reference: self.reference,
            framing: self.framing,
            margin: self.margin,
            field_rotation: self.field_rotation,
            use_memory: self.use_memory,
            onnx: self.onnx,
            color_balance: self.color_balance,
            cancel: None,
        };
        self.job = Some(Job::spawn(ctx, move |rep| {
            let mut o = o;
            o.cancel = Some(rep.cancel.clone());
            let pf = |f: f64, m: &str| rep.report(f, m);
            tl::run_timelapse(&inputs, &o, Some(&pf))
        }));
    }

    pub fn ui(&mut self, ctx: &egui::Context) {
        self.poll();
        if self.autorun && self.loading.is_none() && !self.inputs.is_empty() && self.job.is_none() {
            self.autorun = false;
            self.start(ctx);
        }
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect());
        if !dropped.is_empty() && self.job.is_none() {
            self.add_paths(ctx, dropped);
        }
        egui::SidePanel::left("tl_left").resizable(false).exact_width(330.0).show(ctx, |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| self.left(ui));
        });
        if self.processed.is_some() {
            egui::SidePanel::right("tl_right").resizable(false).exact_width(390.0).show(ctx, |ui| {
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| self.right(ui));
            });
        }
        egui::CentralPanel::default().show(ctx, |ui| self.center(ui));
    }

    fn left(&mut self, ui: &mut egui::Ui) {
        let busy = self.job.is_some();
        let ctx = ui.ctx().clone();
        ui.add_space(6.0);
        ui.heading("Time-lapse multi-vidéos");
        ui.label(RichText::new("Une image par vidéo (éclipse, rotation d'une planète, passage d'ombre…), avec le même traitement, le même cadrage et le même rendu, puis une animation fluide.").small());

        section(ui, "1 · Vidéos de la série");
        ui.add_enabled_ui(!busy, |ui| {
            ui.horizontal(|ui| {
                if ui.button("Ajouter des vidéos SER…").clicked() {
                    if let Some(p) = rfd::FileDialog::new().add_filter("Vidéos SER", &["ser", "SER"]).pick_files() {
                        self.add_paths(&ctx, p);
                    }
                }
                if ui.button("Dossier…").on_hover_text("Toutes les vidéos SER d'un dossier").clicked() {
                    if let Some(p) = rfd::FileDialog::new().pick_folder() {
                        self.add_paths(&ctx, vec![p]);
                    }
                }
                if ui.add_enabled(!self.inputs.is_empty(), egui::Button::new("Vider")).clicked() {
                    self.inputs.clear();
                    self.reference = None;
                }
            });
            ui.horizontal(|ui| {
                let r = ui.add(egui::TextEdit::singleline(&mut self.path_text).hint_text("dossier ou fichier .ser").desired_width(ui.available_width() - 70.0));
                if ui.button("Ajouter").clicked() || (r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))) {
                    let p = PathBuf::from(self.path_text.trim());
                    self.add_paths(&ctx, vec![p]);
                }
            });
        });
        if self.loading.is_some() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Ouverture…");
            });
        }
        if self.inputs.is_empty() {
            ui.label(RichText::new("Aucune vidéo. Ajoutez des SER ou déposez-les sur la fenêtre ; elles sont triées par heure de prise de vue.").small().weak());
        } else {
            let span = match (self.inputs.first().and_then(|i| i.time), self.inputs.last().and_then(|i| i.time)) {
                (Some(a), Some(b)) => format!(" · {} - {} ({} min)", a.format("%H:%M"), b.format("%H:%M"), (b - a).num_minutes()),
                _ => String::new(),
            };
            ui.label(RichText::new(format!("{} vidéo(s){}", self.inputs.len(), span)).small());
            let mut remove = None;
            egui::ScrollArea::vertical().id_salt("list").max_height(200.0).show(ui, |ui| {
                egui::Grid::new("inputs").num_columns(4).spacing([8.0, 2.0]).striped(true).show(ui, |ui| {
                    for (i, v) in self.inputs.iter().enumerate() {
                        let is_ref = self.reference == Some(i);
                        ui.label(RichText::new(format!("{}", i + 1)).weak());
                        ui.label(RichText::new(&v.name).small()).on_hover_text(format!("{} frames", v.frames.len()));
                        ui.label(RichText::new(hms(&v.time)).small().weak());
                        ui.horizontal(|ui| {
                            if ui.add_enabled(!busy, egui::Button::selectable(is_ref, "réf.").small()).on_hover_text("Vidéo de référence (sinon choisie automatiquement)").clicked() {
                                self.reference = if is_ref { None } else { Some(i) };
                            }
                            if ui.add_enabled(!busy, egui::Button::new("×").small()).clicked() {
                                remove = Some(i);
                            }
                        });
                        ui.end_row();
                    }
                });
            });
            if let Some(i) = remove {
                self.inputs.remove(i);
                self.reference = None;
            }
        }

        section(ui, "2 · Traitement uniforme");
        ui.add_enabled_ui(!busy, |ui| {
            ui.horizontal(|ui| {
                ui.label("Recherche (référence) :");
                egui::ComboBox::from_id_salt("tl_effort").selected_text(self.effort.label()).show_ui(ui, |ui| {
                    for e in Effort::ALL {
                        ui.selectable_value(&mut self.effort, e, e.label());
                    }
                });
            });
            ui.label(
                RichText::new(match self.reference {
                    Some(r) => format!("Référence imposée : {}", self.inputs.get(r).map(|i| i.name.as_str()).unwrap_or("?")),
                    None => "Référence : la vidéo la plus propre, choisie automatiquement.".into(),
                })
                .small()
                .weak(),
            );
            ui.horizontal(|ui| {
                ui.label("Cadrage :");
                egui::ComboBox::from_id_salt("tl_framing").selected_text(self.framing.label()).show_ui(ui, |ui| {
                    for f in Framing::ALL {
                        ui.selectable_value(&mut self.framing, f, f.label());
                    }
                });
            });
            ui.add(egui::Slider::new(&mut self.margin, 0.05..=1.0).text("marge autour du disque"));
            ui.checkbox(&mut self.field_rotation, "Corriger la rotation de champ (altazimutal)");
            ui.checkbox(&mut self.use_memory, "Réutiliser l'expérience des vidéos précédentes");
            ui.checkbox(&mut self.onnx, "Essayer un modèle ONNX de débruitage");
            ui.checkbox(&mut self.color_balance, "Équilibre automatique des couleurs");
        });
        ui.add_space(8.0);
        if let Some(j) = &self.job {
            let (p, m) = j.progress();
            ui.add(egui::ProgressBar::new(p).text(m).animate(true));
            ui.label(RichText::new(format!("{:.0} s écoulées", j.started.elapsed().as_secs_f64())).small().weak());
            if ui.button("Annuler").clicked() {
                j.request_cancel();
            }
            ui.ctx().request_repaint_after(Duration::from_millis(250));
        } else {
            let enabled = !self.inputs.is_empty() && self.loading.is_none();
            let btn = egui::Button::new(RichText::new("Traiter la série").strong().size(15.0).color(if enabled { Color32::BLACK } else { Color32::GRAY }))
                .fill(if enabled { plot::ACCENT } else { Color32::from_gray(60) })
                .min_size(egui::vec2(ui.available_width(), 38.0));
            if ui.add_enabled(enabled, btn).clicked() {
                self.start(ui.ctx());
            }
        }
        if let Some(s) = &self.status {
            ui.label(RichText::new(s).small().color(Color32::from_rgb(140, 200, 140)));
        }
        if let Some(e) = &self.error {
            ui.colored_label(Color32::from_rgb(230, 110, 100), e);
        }

        if self.processed.is_some() {
            section(ui, "3 · Rendu commun");
            let mut changed = false;
            ui.horizontal(|ui| {
                ui.label("Luminosité :");
                egui::ComboBox::from_id_salt("tl_bright").selected_text(self.render.brightness.label()).show_ui(ui, |ui| {
                    for b in Brightness::ALL {
                        changed |= ui.selectable_value(&mut self.render.brightness, b, b.label()).changed();
                    }
                });
            });
            ui.label(
                RichText::new(match self.render.brightness {
                    Brightness::Deflicker => "Les voiles passagers sont gommés, l'évolution réelle (assombrissement d'une éclipse de Lune…) est gardée.",
                    Brightness::Real => "Mêmes niveaux pour toutes les images : chaque variation de luminosité reste visible.",
                    Brightness::Normalized => "Toutes les images à la même luminosité.",
                })
                .small()
                .weak(),
            );
            if self.render.brightness == Brightness::Deflicker {
                let r = ui.add(egui::Slider::new(&mut self.render.smoothing, 3..=15).text("lissage (images)"));
                changed |= r.drag_stopped() || (r.changed() && !r.dragged());
            }
            changed |= ui.checkbox(&mut self.render.timestamp, "Horodatage incrusté (UT)").changed();
            ui.horizontal(|ui| {
                ui.label("Légende :");
                let r = ui.add(egui::TextEdit::singleline(&mut self.render.caption).desired_width(ui.available_width()));
                changed |= r.lost_focus();
            });
            if changed {
                self.dirty = true;
            }

            section(ui, "4 · Animation");
            ui.add(egui::Slider::new(&mut self.fps, 1..=60).text("images / s"));
            let r = ui.add(egui::Slider::new(&mut self.between, 0..=8).text("images de transition"));
            if r.drag_stopped() || (r.changed() && !r.dragged()) {
                self.dirty = true;
            }
            ui.label(RichText::new(format!("{} images, {} images d'animation, {:.1} s", self.rendered.len(), self.anim.len(), self.anim.len() as f64 / self.fps.max(1) as f64)).small().weak());
            if ui.button("Exporter l'animation…").on_hover_text("MP4 ou AVI (ffmpeg), ou GIF").clicked() {
                if let Some(p) = rfd::FileDialog::new().add_filter("Vidéo MP4", &["mp4"]).add_filter("GIF animé", &["gif"]).add_filter("AVI", &["avi"]).set_file_name("timelapse.mp4").save_file() {
                    let res = match &self.processed {
                        Some(pr) => tl::export_animation(pr, &self.render, &p, self.fps, self.between),
                        None => Ok(()),
                    };
                    match res {
                        Ok(_) => self.status = Some(format!("Animation : {}", p.display())),
                        Err(e) => self.error = Some(format!("{:#}", e)),
                    }
                }
            }
            ui.horizontal(|ui| {
                if ui.button("Exporter les images…").on_hover_text("Une image par vidéo, numérotées, dans un dossier").clicked() {
                    if let Some(d) = rfd::FileDialog::new().pick_folder() {
                        match tl::export_frames(&self.rendered, &d, "timelapse", &self.frame_format) {
                            Ok(f) => self.status = Some(format!("{} images dans {}", f.len(), d.display())),
                            Err(e) => self.error = Some(format!("{:#}", e)),
                        }
                    }
                }
                egui::ComboBox::from_id_salt("tl_fmt").selected_text(self.frame_format.as_str()).width(60.0).show_ui(ui, |ui| {
                    for f in FRAME_FORMATS {
                        ui.selectable_value(&mut self.frame_format, f.to_string(), f);
                    }
                });
            });
            if ui.button("Exporter le rapport (JSON)…").clicked() {
                if let (Some(p), Some(pr)) = (rfd::FileDialog::new().add_filter("JSON", &["json"]).set_file_name("timelapse_rapport.json").save_file(), &self.processed) {
                    match std::fs::write(&p, serde_json::to_string_pretty(&tl::report_json(pr)).unwrap_or_default()) {
                        Ok(_) => self.status = Some(format!("Rapport : {}", p.display())),
                        Err(e) => self.error = Some(e.to_string()),
                    }
                }
            }
        }
    }

    fn right(&mut self, ui: &mut egui::Ui) {
        let Some(p) = self.processed.clone() else { return };
        ui.add_space(6.0);
        ui.heading("Bilan de la série");
        for d in &p.decisions {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(&d.step).strong());
                ui.label(&d.choice);
            });
            ui.label(RichText::new(&d.reason).small().weak());
            ui.add_space(4.0);
        }
        section(ui, "Luminosité : mesurée, puis rendue");
        let lv: Vec<f64> = p.items.iter().map(|i| i.level).collect();
        let g = tl::brightness_gains(&lv, self.render.brightness, self.render.smoothing, p.reference);
        let xs: Vec<f64> = (1..=lv.len()).map(|i| i as f64).collect();
        let m0 = lv.get(p.reference).cloned().unwrap_or(1.0).max(1e-9);
        let rel: Vec<f64> = lv.iter().map(|v| v / m0 * 100.0).collect();
        let after: Vec<f64> = lv.iter().zip(&g).map(|(v, g)| v * g / m0 * 100.0).collect();
        ui.label(RichText::new("mesurée (% de la référence)").small().weak());
        plot::line(ui, &xs, &rel, Some((p.reference + 1) as f64), None, "image", &|x| format!("{:.0}", x), 70.0);
        ui.label(RichText::new("après harmonisation").small().weak());
        plot::line(ui, &xs, &after, Some((p.reference + 1) as f64), None, "image", &|x| format!("{:.0}", x), 70.0);

        section(ui, "Images");
        egui::Grid::new("items").num_columns(4).spacing([10.0, 2.0]).striped(true).show(ui, |ui| {
            ui.label(RichText::new("vidéo").small().weak());
            ui.label(RichText::new("heure").small().weak());
            ui.label(RichText::new("frames").small().weak());
            ui.label(RichText::new("lum.").small().weak());
            ui.end_row();
            for (i, it) in p.items.iter().enumerate() {
                let name = if i == p.reference { format!("{} (réf.)", it.name) } else { it.name.clone() };
                ui.label(RichText::new(name).small());
                ui.label(RichText::new(it.time.as_deref().and_then(|t| t.split('T').nth(1)).map(|t| t.split('.').next().unwrap_or(t).to_string()).unwrap_or_else(|| "—".into())).small());
                ui.label(RichText::new(format!("{}/{}", it.keep, it.n_frames)).small());
                ui.label(RichText::new(format!("{:.0} %", rel[i])).small());
                ui.end_row();
            }
        });
        for f in &p.failed {
            ui.colored_label(Color32::from_rgb(230, 150, 100), format!("{} écartée : {}", f.name, f.error.clone().unwrap_or_default()));
        }
        section(ui, "Plan commun (référence)");
        for d in &p.reference_decisions {
            ui.label(RichText::new(format!("{} — {}", d.step, d.choice)).small());
        }
        ui.add_space(12.0);
    }

    fn center(&mut self, ui: &mut egui::Ui) {
        if self.anim.is_empty() {
            self.playing = false;
            ui.centered_and_justified(|ui| {
                ui.label(RichText::new("Ajoutez les vidéos de la série puis « Traiter la série ».\nChaque vidéo devient une image ; l'animation apparaît ici.").size(16.0).weak());
            });
            return;
        }
        let n = self.anim.len();
        // avance de la lecture avant tout affichage (étiquette et image cohérentes)
        if self.playing {
            let dt = Duration::from_secs_f64(1.0 / self.fps.max(1) as f64);
            if self.last_tick.elapsed() >= dt {
                self.pos = (self.pos + 1) % n;
                self.last_tick = Instant::now();
            }
            ui.ctx().request_repaint_after(dt);
        }
        ui.horizontal(|ui| {
            if ui.button(if self.playing { "Pause" } else { "Lecture" }).clicked() {
                self.playing = !self.playing;
                self.last_tick = Instant::now();
            }
            if ui.button("<").clicked() {
                self.playing = false;
                self.pos = (self.pos + n - 1) % n;
            }
            if ui.button(">").clicked() {
                self.playing = false;
                self.pos = (self.pos + 1) % n;
            }
            ui.spacing_mut().slider_width = (ui.available_width() - 260.0).max(80.0);
            if ui.add(egui::Slider::new(&mut self.pos, 0..=n - 1).show_value(false)).changed() {
                self.playing = false;
            }
            let src = self.pos / (self.between + 1);
            let label = self
                .processed
                .as_ref()
                .and_then(|p| p.items.get(src))
                .map(|it| format!("{} · {}", it.name, it.time.as_deref().map(|t| t.replace('T', " ")).unwrap_or_default().split('.').next().unwrap_or("").to_string()))
                .unwrap_or_default();
            ui.label(RichText::new(label).small());
            ui.separator();
            self.viewer.toolbar(ui);
        });
        ui.separator();
        let img = self.anim[self.pos].clone();
        self.viewer.show(ui, &img, self.key_base + self.pos as u64, false);
    }
}
