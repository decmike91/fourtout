//! Vidéo partagée entre les onglets : SER, ou série d'images.

use crate::job::Job;
use astroforge::frames::Frames;
use astroforge::image::Image;
use astroforge::io;
use eframe::egui;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub type SharedFrames = Arc<dyn Frames + Send + Sync>;

pub struct Video {
    pub frames: SharedFrames,
    pub name: String,
    pub info: String,
    pub n: usize,
}

const IMAGE_EXTS: [&str; 9] = ["fits", "fit", "fts", "tif", "tiff", "png", "jpg", "jpeg", "xisf"];

fn load_video(paths: Vec<PathBuf>, max_frames: usize) -> anyhow::Result<Video> {
    let mut paths = paths;
    if paths.len() == 1 && paths[0].is_dir() {
        let dir = paths[0].clone();
        paths = std::fs::read_dir(&dir)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| io::has_ext(p, &IMAGE_EXTS)).collect();
        paths.sort();
        if paths.is_empty() {
            anyhow::bail!("Aucune image dans {}", dir.display());
        }
    }
    if paths.is_empty() {
        anyhow::bail!("Aucun fichier");
    }
    if paths.len() == 1 && io::ext_of(&paths[0]) == "ser" {
        let (frames, meta) = io::ser::load_ser(&paths[0], max_frames)?;
        let n = Frames::len(&frames);
        let info = format!(
            "{} frames · {}×{} · {} bits{}{}",
            n,
            meta.width,
            meta.height,
            meta.bits,
            meta.bayer.as_ref().map(|b| format!(" · Bayer {}", b)).unwrap_or_default(),
            if meta.camera.trim().is_empty() { String::new() } else { format!(" · {}", meta.camera.trim()) }
        );
        return Ok(Video { frames: Arc::new(frames), name: file_name(&paths[0]), info, n });
    }
    if max_frames > 0 {
        paths.truncate(max_frames);
    }
    let imgs: Vec<Image> = paths.iter().map(io::load).collect::<anyhow::Result<_>>()?;
    let (w, h, c) = (imgs[0].w, imgs[0].h, imgs[0].c);
    if imgs.iter().any(|i| (i.w, i.h, i.c) != (w, h, c)) {
        anyhow::bail!("Toutes les images doivent avoir la même taille et le même nombre de canaux.");
    }
    let n = imgs.len();
    let name = if n == 1 { file_name(&paths[0]) } else { format!("{} images ({}…)", n, file_name(&paths[0])) };
    Ok(Video { frames: Arc::new(imgs), name, info: format!("{} images · {}×{} · {}", n, w, h, if c == 3 { "couleur" } else { "mono" }), n })
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| p.display().to_string())
}

#[derive(Default)]
pub struct VideoSlot {
    pub video: Option<Arc<Video>>,
    /// incrémenté à chaque nouvelle vidéo
    pub generation: u64,
    loading: Option<Job<Video>>,
    pub error: Option<String>,
    pub path_text: String,
    pub max_frames: usize,
}

impl VideoSlot {
    pub fn open(&mut self, ctx: &egui::Context, paths: Vec<PathBuf>) {
        if paths.is_empty() || self.loading.is_some() {
            return;
        }
        self.error = None;
        self.path_text = paths[0].display().to_string();
        let mf = self.max_frames;
        self.loading = Some(Job::spawn(ctx, move |rep| {
            rep.report(0.1, "Chargement de la vidéo…");
            load_video(paths, mf)
        }));
    }

    pub fn is_loading(&self) -> bool {
        self.loading.is_some()
    }

    pub fn poll(&mut self, ctx: &egui::Context, accept_drops: bool) {
        // fichiers déposés sur la fenêtre
        let dropped: Vec<PathBuf> = if accept_drops { ctx.input(|i| i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect()) } else { Vec::new() };
        if !dropped.is_empty() {
            self.open(ctx, dropped);
        }
        if let Some(j) = &mut self.loading {
            if let Some(r) = j.poll() {
                self.loading = None;
                match r {
                    Ok(v) => {
                        self.video = Some(Arc::new(v));
                        self.generation += 1;
                    }
                    Err(e) => self.error = Some(e),
                }
            }
        }
    }

    /// Bloc « Vidéo » : ouverture, chemin, informations.
    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        ui.horizontal(|ui| {
            if ui.button("Ouvrir une vidéo SER…").clicked() {
                if let Some(p) = rfd::FileDialog::new().add_filter("Vidéo SER", &["ser", "SER"]).pick_file() {
                    self.open(&ctx, vec![p]);
                }
            }
            if ui.button("Images…").on_hover_text("Série d'images (FITS, TIFF, PNG, XISF…)").clicked() {
                if let Some(p) = rfd::FileDialog::new().add_filter("Images", &IMAGE_EXTS).pick_files() {
                    self.open(&ctx, p);
                }
            }
        });
        ui.horizontal(|ui| {
            let r = ui.add(egui::TextEdit::singleline(&mut self.path_text).hint_text("chemin d'une vidéo ou d'un dossier").desired_width(ui.available_width() - 64.0));
            if ui.button("Charger").clicked() || (r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))) {
                let p = PathBuf::from(self.path_text.trim());
                self.open(&ctx, vec![p]);
            }
        });
        ui.horizontal(|ui| {
            ui.label("Frames max :");
            ui.add(egui::DragValue::new(&mut self.max_frames).range(0..=200_000).speed(50.0)).on_hover_text("0 = toute la vidéo (appliqué au prochain chargement)");
        });
        if self.loading.is_some() {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Chargement…");
            });
        }
        if let Some(v) = &self.video {
            ui.label(egui::RichText::new(&v.name).strong());
            ui.label(egui::RichText::new(&v.info).small());
        } else if self.loading.is_none() {
            ui.label(egui::RichText::new("Aucune vidéo. Ouvrez un fichier SER ou déposez-le sur la fenêtre.").small().weak());
        }
        if let Some(e) = &self.error {
            ui.colored_label(egui::Color32::from_rgb(230, 110, 100), e);
        }
    }
}
