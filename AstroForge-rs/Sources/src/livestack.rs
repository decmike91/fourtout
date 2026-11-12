//! Empilement en direct : chaque nouvelle image d'un dossier surveillé est
//! calibrée, dématricée, contrôlée (étoiles, FWHM), alignée sur la
//! première image valide puis ajoutée à une moyenne courante avec rejet
//! kappa-sigma progressif (Welford par pixel).

use crate::calibration::{self, Masters};
use crate::geom;
use crate::image::Image;
use crate::io;
use crate::registration;
use crate::subframes;
use crate::stats;
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub struct LiveStacker {
    pub masters: Masters,
    pub bayer: Option<String>,
    pub kappa: f64,
    pub min_frames_reject: usize,
    pub max_fwhm_ratio: f64,
    pub min_stars: usize,
    pub reference: Option<Image>,
    mean: Option<Vec<f64>>,
    m2: Option<Vec<f64>>,
    count: Option<Vec<f64>>,
    shape: (usize, usize, usize),
    pub n: usize,
    pub rejected: Vec<(String, String)>,
    pub accepted: Vec<String>,
    pub fwhms: Vec<f64>,
}

impl LiveStacker {
    pub fn new(masters: Masters, bayer: Option<String>, kappa: f64) -> Self {
        LiveStacker {
            masters,
            bayer,
            kappa,
            min_frames_reject: 5,
            max_fwhm_ratio: 1.6,
            min_stars: 5,
            reference: None,
            mean: None,
            m2: None,
            count: None,
            shape: (0, 0, 0),
            n: 0,
            rejected: vec![],
            accepted: vec![],
            fwhms: vec![],
        }
    }

    pub fn reset(&mut self) {
        let m = std::mem::take(&mut self.masters);
        *self = LiveStacker::new(m, self.bayer.clone(), self.kappa);
    }

    pub fn prepare_path(&self, p: &Path) -> Result<Image> {
        let (raw, pattern) = io::load_cfa(p)?;
        self.prepare(raw, pattern)
    }

    pub fn prepare(&self, raw: Image, pattern: Option<String>) -> Result<Image> {
        let pattern = if raw.c == 1 { self.bayer.clone().or(pattern) } else { None };
        let m = &self.masters;
        let cal = calibration::calibrate(&raw, m.bias.as_ref(), m.dark.as_ref(), m.flat.as_ref(), 1.0, true, 5.0, pattern.is_some())?;
        match pattern {
            Some(p) => io::debayer(&cal.clip01(), &p),
            None => Ok(cal),
        }
    }

    /// Ajoute une image (déjà préparée ou un fichier) ; false si rejetée.
    pub fn add_path(&mut self, p: &Path) -> bool {
        let name = p.to_string_lossy().to_string();
        match self.prepare_path(p) {
            Ok(img) => self.add_prepared(img, &name),
            Err(e) => {
                self.rejected.push((name, format!("illisible : {}", e)));
                false
            }
        }
    }

    pub fn add_prepared(&mut self, img: Image, name: &str) -> bool {
        let q = subframes::analyze_image(&img, false, 60);
        if q.stars < self.min_stars {
            self.rejected.push((name.into(), format!("{} étoiles", q.stars)));
            return false;
        }
        if !self.fwhms.is_empty() && q.fwhm.is_finite() && q.fwhm > self.max_fwhm_ratio * stats::median_f64(&self.fwhms) {
            self.rejected.push((name.into(), format!("FWHM {:.2} px", q.fwhm)));
            return false;
        }
        let aligned = match &self.reference {
            None => {
                self.reference = Some(img.clone());
                img
            }
            Some(r) => {
                if !img.same_shape(r) {
                    self.rejected.push((name.into(), "taille différente".into()));
                    return false;
                }
                let m = registration::estimate_transform(r, &img, true);
                if !geom::is_finite(&m) || (geom::det2(&m) - 1.0).abs() > 0.05 {
                    self.rejected.push((name.into(), "alignement impossible".into()));
                    return false;
                }
                registration::warp(&img, &m)
            }
        };
        if q.fwhm.is_finite() {
            self.fwhms.push(q.fwhm);
        }
        self.accumulate(&aligned);
        self.accepted.push(name.into());
        true
    }

    fn accumulate(&mut self, x: &Image) {
        let xv: Vec<f64> = x.data.iter().map(|&v| v as f64).collect();
        if self.mean.is_none() {
            self.shape = (x.w, x.h, x.c);
            self.m2 = Some(vec![0.0; xv.len()]);
            self.count = Some(vec![1.0; xv.len()]);
            self.mean = Some(xv);
            self.n = 1;
            return;
        }
        self.n += 1;
        let reject = self.n > self.min_frames_reject;
        let (mean, m2, count) = (self.mean.as_mut().unwrap(), self.m2.as_mut().unwrap(), self.count.as_mut().unwrap());
        for i in 0..xv.len() {
            let ok = if reject {
                let sd = (m2[i] / (count[i] - 1.0).max(1.0)).sqrt();
                (xv[i] - mean[i]).abs() <= self.kappa * sd.max(1e-4)
            } else {
                true
            };
            if !ok {
                continue;
            }
            let c = count[i] + 1.0;
            let delta = xv[i] - mean[i];
            mean[i] += delta / c;
            m2[i] += delta * (xv[i] - mean[i]);
            count[i] = c;
        }
    }

    pub fn image(&self) -> Option<Image> {
        let m = self.mean.as_ref()?;
        let (w, h, c) = self.shape;
        Some(Image::from_vec(w, h, c, m.iter().map(|&v| (v as f32).clamp(0.0, 1.0)).collect()))
    }

    pub fn status(&self) -> String {
        let mut s = format!("{} image(s) empilée(s), {} rejetée(s)", self.accepted.len(), self.rejected.len());
        if !self.fwhms.is_empty() {
            s += &format!(" — FWHM médiane {:.2} px", stats::median_f64(&self.fwhms));
        }
        s
    }
}

/// Nouveaux fichiers d'un dossier (taille stable entre deux passages).
pub struct FolderWatcher {
    pub folder: PathBuf,
    pub seen: HashSet<PathBuf>,
    pub sizes: HashMap<PathBuf, u64>,
}

impl FolderWatcher {
    pub fn new(folder: &Path, include_existing: bool) -> Self {
        let mut w = FolderWatcher { folder: folder.to_path_buf(), seen: HashSet::new(), sizes: HashMap::new() };
        if !include_existing {
            w.seen = w.list().into_iter().collect();
        }
        w
    }

    fn list(&self) -> Vec<PathBuf> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(&self.folder)
            .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_file() && io::has_ext(p, &io::CALIB_EXTENSIONS)).collect())
            .unwrap_or_default();
        v.sort();
        v
    }

    pub fn poll(&mut self) -> Vec<PathBuf> {
        let mut ready = Vec::new();
        for p in self.list() {
            if self.seen.contains(&p) {
                continue;
            }
            let Ok(md) = std::fs::metadata(&p) else { continue };
            let sz = md.len();
            if sz > 0 && self.sizes.get(&p) == Some(&sz) {
                self.seen.insert(p.clone());
                self.sizes.remove(&p);
                ready.push(p);
            } else {
                self.sizes.insert(p, sz);
            }
        }
        ready
    }
}

/// Boucle de surveillance ; duration_s = 0 : fichiers présents puis arrêt.
pub fn run_folder(folder: &Path, st: &mut LiveStacker, duration_s: f64, interval_s: f64, on_update: &mut dyn FnMut(&LiveStacker)) {
    let mut w = FolderWatcher::new(folder, true);
    let t0 = Instant::now();
    loop {
        let new = w.poll();
        if new.is_empty() && duration_s <= 0.0 && w.sizes.is_empty() {
            break;
        }
        for p in new {
            st.add_path(&p);
            on_update(st);
        }
        if duration_s > 0.0 && t0.elapsed().as_secs_f64() > duration_s {
            break;
        }
        std::thread::sleep(Duration::from_secs_f64(if duration_s > 0.0 { interval_s.max(0.05) } else { 0.05 }));
    }
}
