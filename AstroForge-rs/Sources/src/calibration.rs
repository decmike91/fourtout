//! Calibration : masters (bias, dark, dark-flat, flat) et calibration des
//! images brutes, correction cosmétique (pixels chauds), chaîne complète
//! lecture CFA -> calibration -> cosmétique -> dématriçage.

use crate::image::Image;
use crate::io::{self};
use crate::masters;
use crate::ndimage;
use crate::session;
use crate::stacking::combine_sigma_clip;
use crate::stats;
use anyhow::{bail, Result};
use rayon::prelude::*;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub type Progress<'a> = Option<&'a (dyn Fn(usize, usize, &str) + Sync)>;

pub fn make_master_bias(frames: &[Image], sigma: f64) -> Result<Image> {
    if frames.is_empty() {
        bail!("Aucun bias fourni");
    }
    let r: Vec<&Image> = frames.iter().collect();
    combine_sigma_clip(&r, sigma)
}

/// Master dark (thermique si un bias est fourni).
pub fn make_master_dark(frames: &[Image], bias: Option<&Image>, sigma: f64) -> Result<Image> {
    if frames.is_empty() {
        bail!("Aucune dark fournie");
    }
    match bias {
        Some(b) => {
            let d: Vec<Image> = frames.par_iter().map(|f| f.zip_map(b, |a, c| a - c)).collect();
            let r: Vec<&Image> = d.iter().collect();
            combine_sigma_clip(&r, sigma)
        }
        None => {
            let r: Vec<&Image> = frames.iter().collect();
            combine_sigma_clip(&r, sigma)
        }
    }
}

/// Master flat : chaque flat corrigé (bias + dark-flat), normalisé par sa
/// médiane, combiné puis renormalisé à 1.
pub fn make_master_flat(frames: &[Image], bias: Option<&Image>, darkflat: Option<&Image>, sigma: f64) -> Result<Image> {
    if frames.is_empty() {
        bail!("Aucune flat fournie");
    }
    let corrected: Vec<Image> = frames
        .par_iter()
        .map(|f| {
            let mut a = f.clone();
            if let Some(b) = bias {
                a = a.zip_map(b, |x, y| x - y);
            }
            if let Some(d) = darkflat {
                a = a.zip_map(d, |x, y| x - y);
            }
            let pos: Vec<f32> = a.data.iter().cloned().filter(|&v| v > 0.0).collect();
            let mut m = if pos.is_empty() { 1.0 } else { stats::median(&pos) };
            if m <= 0.0 {
                m = 1.0;
            }
            a.scale(1.0 / m)
        })
        .collect();
    let r: Vec<&Image> = corrected.iter().collect();
    let master = combine_sigma_clip(&r, sigma)?;
    let pos: Vec<f32> = master.data.iter().cloned().filter(|&v| v > 0.0).collect();
    let med = (stats::median(&pos) as f64).max(1e-9);
    let med = if med.is_nan() { 1e-9 } else { med };
    Ok(master.map(|v| ((v as f64 / med) as f32).max(1e-4)))
}

fn check_match(master: &Image, light: &Image, name: &str) -> Result<()> {
    if master.same_shape(light) {
        return Ok(());
    }
    if master.c == 1 && light.c == 3 && master.w == light.w && master.h == light.h {
        bail!(
            "Master {} mono (CFA) et image lumière couleur : calibrez les lumières AVANT dématriçage (chargement CFA), ou fournissez des masters de même type.",
            name
        );
    }
    bail!(
        "Taille du master {} {} différente de l'image {} (binning, recadrage ou orientation ?)",
        name,
        master.shape_str(),
        light.shape_str()
    )
}

/// (L − bias − k·dark) / flat, puis cosmétique des pixels chauds.
#[allow(clippy::too_many_arguments)]
pub fn calibrate(light: &Image, bias: Option<&Image>, dark: Option<&Image>, flat: Option<&Image>, dark_scale: f64, cosmetic: bool, cosmetic_sigma: f64, cfa: bool) -> Result<Image> {
    let mut l = light.clone();
    if let Some(b) = bias {
        check_match(b, &l, "bias")?;
        l = l.zip_map(b, |a, c| a - c);
    }
    if let Some(d) = dark {
        check_match(d, &l, "dark")?;
        let k = dark_scale as f32;
        l = l.zip_map(d, |a, c| a - c * k);
    }
    if let Some(f) = flat {
        check_match(f, &l, "flat")?;
        l = l.zip_map(f, |a, c| a / c.max(1e-4));
    }
    if cosmetic {
        l = remove_hot_pixels(&l, cosmetic_sigma, cfa);
    }
    Ok(l.clip_min(0.0))
}

fn hot_plane(p: &Image, sigma: f64) -> Image {
    let med = ndimage::median_filter(p, 3);
    let diff: Vec<f32> = p.data.iter().zip(&med.data).map(|(a, b)| a - b).collect();
    let dm = stats::median(&diff);
    let std = 1.4826 * stats::mad_about(&diff, dm) as f64 + 1e-9;
    let thr = (sigma * std) as f32;
    let mut out = p.clone();
    for i in 0..out.data.len() {
        if diff[i] > thr {
            out.data[i] = med.data[i];
        }
    }
    out
}

/// Pixels isolés (cosmiques/chauds) remplacés par la médiane locale 3×3,
/// par canal ou par sous-plan de même couleur du motif de Bayer (cfa).
pub fn remove_hot_pixels(img: &Image, sigma: f64, cfa: bool) -> Image {
    if img.c > 1 {
        return img.map_planes(|p| hot_plane(p, sigma));
    }
    if cfa && img.w >= 4 && img.h >= 4 {
        let mut out = img.clone();
        let subs: Vec<((usize, usize), Image)> = [(0, 0), (0, 1), (1, 0), (1, 1)]
            .par_iter()
            .map(|&(i, j)| ((i, j), hot_plane(&img.sub2(i, j), sigma)))
            .collect();
        for ((i, j), s) in subs {
            out.put_sub2(i, j, &s);
        }
        return out;
    }
    hot_plane(img, sigma)
}

// ------------------------------------------------------------- session complète
/// Fichiers image d'un dossier de calibration (triés).
pub fn list_frames(dir: &str) -> Vec<PathBuf> {
    if dir.is_empty() {
        return vec![];
    }
    let p = Path::new(dir);
    if !p.is_dir() {
        return vec![];
    }
    let mut v: Vec<PathBuf> = std::fs::read_dir(p)
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|f| f.is_file() && io::has_ext(f, &io::CALIB_EXTENSIONS)).collect())
        .unwrap_or_default();
    v.sort();
    v
}

pub fn load_frames_cfa(paths: &[PathBuf]) -> Result<Vec<Image>> {
    paths.par_iter().map(|p| io::load_cfa(p).map(|x| x.0)).collect()
}

/// Métadonnées médianes d'un lot (en-têtes FITS).
pub fn group_meta(paths: &[PathBuf]) -> Map<String, Value> {
    let infos: Vec<session::FrameInfo> = paths.iter().take(50).map(|p| session::frame_info(p)).collect();
    let mut out = Map::new();
    let med = |vals: Vec<f64>| -> Value {
        if vals.is_empty() {
            Value::Null
        } else {
            json!(stats::median_f64(&vals))
        }
    };
    out.insert("exptime".into(), med(infos.iter().filter_map(|i| i.exptime).collect()));
    out.insert("gain".into(), med(infos.iter().filter_map(|i| i.gain).collect()));
    out.insert("offset".into(), med(infos.iter().filter_map(|i| i.offset).collect()));
    out.insert("temp".into(), med(infos.iter().filter_map(|i| i.temp).collect()));
    out.insert("binning".into(), json!(infos.first().map(|i| i.binning).unwrap_or(1)));
    out.insert("filter".into(), json!(infos.first().map(|i| i.filter.clone()).unwrap_or_default()));
    out.insert("bayer".into(), infos.first().and_then(|i| i.bayer.clone()).map(Value::String).unwrap_or(Value::Null));
    out.insert("ncombine".into(), json!(paths.len()));
    out
}

/// Masters construits (ou repris de la bibliothèque).
#[derive(Default, Clone)]
pub struct Masters {
    pub bias: Option<Image>,
    pub dark: Option<Image>,
    pub darkflat: Option<Image>,
    pub flat: Option<Image>,
    pub meta: HashMap<String, Map<String, Value>>,
    pub sources: HashMap<String, String>,
}

impl Masters {
    pub fn get(&self, kind: &str) -> Option<&Image> {
        match kind {
            "bias" => self.bias.as_ref(),
            "dark" => self.dark.as_ref(),
            "darkflat" => self.darkflat.as_ref(),
            _ => self.flat.as_ref(),
        }
    }
    fn set(&mut self, kind: &str, img: Image) {
        match kind {
            "bias" => self.bias = Some(img),
            "dark" => self.dark = Some(img),
            "darkflat" => self.darkflat = Some(img),
            _ => self.flat = Some(img),
        }
    }
    pub fn describe(&self) -> String {
        let done: Vec<String> = ["bias", "dark", "darkflat", "flat"]
            .iter()
            .filter(|k| self.get(k).is_some())
            .map(|k| format!("{} ({})", k, self.sources.get(*k).cloned().unwrap_or_default()))
            .collect();
        format!("Masters : {}", if done.is_empty() { "aucun".to_string() } else { done.join(", ") })
    }
}

#[derive(Default, Clone)]
pub struct MasterOptions {
    pub bias_dir: String,
    pub dark_dir: String,
    pub flat_dir: String,
    pub darkflat_dir: String,
    pub sigma: f64,
    pub paths: Option<HashMap<String, Vec<PathBuf>>>,
    pub light_meta: Option<Map<String, Value>>,
    pub use_library: bool,
    pub save_to_library: bool,
    pub library: Option<PathBuf>,
}

/// Construit les masters (chaque lot lu une fois, en CFA).
pub fn build_masters(o: &MasterOptions) -> Result<Masters> {
    let sigma = if o.sigma > 0.0 { o.sigma } else { 3.0 };
    let dirs: HashMap<&str, &str> =
        [("bias", o.bias_dir.as_str()), ("dark", o.dark_dir.as_str()), ("darkflat", o.darkflat_dir.as_str()), ("flat", o.flat_dir.as_str())].into_iter().collect();
    let mut out = Masters::default();
    for kind in ["bias", "dark", "darkflat", "flat"] {
        let files: Vec<PathBuf> = match o.paths.as_ref().and_then(|p| p.get(kind)) {
            Some(v) if !v.is_empty() => v.clone(),
            _ => list_frames(dirs[kind]),
        };
        if !files.is_empty() {
            let frames = load_frames_cfa(&files)?;
            let m = match kind {
                "bias" => make_master_bias(&frames, sigma)?,
                "dark" => make_master_dark(&frames, out.bias.as_ref(), sigma)?,
                "darkflat" => make_master_dark(&frames, out.bias.as_ref(), sigma)?,
                _ => make_master_flat(&frames, out.bias.as_ref(), out.darkflat.as_ref(), sigma)?,
            };
            drop(frames);
            let mut meta = group_meta(&files);
            meta.insert("thermal".into(), json!((kind == "dark" || kind == "darkflat") && out.bias.is_some()));
            if o.save_to_library {
                masters::save_master(&m, kind, &meta, o.library.as_deref())?;
            }
            out.set(kind, m);
            out.meta.insert(kind.into(), meta);
            out.sources.insert(kind.into(), "calculé".into());
        } else if o.use_library {
            if let Some(lm) = &o.light_meta {
                if let (Some(entry), _scale) = masters::find_master(kind, lm, o.library.as_deref(), 3.0, true) {
                    if kind == "dark" && entry.get("thermal").and_then(|v| v.as_bool()).unwrap_or(false) && out.bias.is_none() {
                        match masters::find_master("bias", lm, o.library.as_deref(), 3.0, true) {
                            (Some(b), _) => {
                                out.bias = Some(masters::load_master(&b, o.library.as_deref())?);
                                out.sources.insert("bias".into(), "bibliothèque".into());
                                out.meta.insert("bias".into(), b);
                            }
                            _ => continue,
                        }
                    }
                    out.set(kind, masters::load_master(&entry, o.library.as_deref())?);
                    out.meta.insert(kind.into(), entry);
                    out.sources.insert(kind.into(), "bibliothèque".into());
                }
            }
        }
    }
    Ok(out)
}

/// Facteur du dark = EXPTIME lumière / EXPTIME dark (dark thermique).
pub fn auto_dark_scale(light: &Path, m: &Masters) -> f64 {
    let Some(meta) = m.meta.get("dark") else { return 1.0 };
    if m.dark.is_none() || !meta.get("thermal").and_then(|v| v.as_bool()).unwrap_or(false) {
        return 1.0;
    }
    let Some(dexp) = meta.get("exptime").and_then(|v| v.as_f64()).filter(|e| *e != 0.0) else { return 1.0 };
    match session::frame_info(light).exptime {
        Some(e) if e != 0.0 => e / dexp,
        _ => 1.0,
    }
}

pub const OUTPUTS: [&str; 3] = ["RGB (dématriçage)", "Ha (duo-bande)", "OIII (duo-bande)"];

#[derive(Clone)]
pub enum DarkScale {
    Auto,
    Fixed(f64),
}

/// Correction optionnelle des défauts : (image, cfa) -> image.
pub type DefectFn<'a> = Option<&'a (dyn Fn(&Image, bool) -> Image + Sync)>;

/// Chaîne de pré-traitement image par image : lecture CFA -> calibration ->
/// cosmétique et défauts (même couleur) -> dématriçage (ou Ha / OIII).
#[allow(clippy::too_many_arguments)]
pub fn calibrate_paths(paths: &[PathBuf], m: &Masters, bayer: Option<&str>, dark_scale: DarkScale, cosmetic: bool, defects: DefectFn, progress: Progress, output: &str) -> Result<Vec<Image>> {
    let kind = if output.starts_with("Ha") {
        "Ha"
    } else if output.starts_with("OIII") {
        "OIII"
    } else {
        "RGB"
    };
    let n = paths.len();
    let done = std::sync::atomic::AtomicUsize::new(0);
    paths
        .par_iter()
        .map(|lp| -> Result<Image> {
            let (raw, detected) = io::load_cfa(lp)?;
            let pattern: Option<String> = if raw.c == 1 { bayer.map(|s| s.to_string()).or(detected) } else { None };
            let k = match dark_scale {
                DarkScale::Auto => auto_dark_scale(lp, m),
                DarkScale::Fixed(v) => v,
            };
            let mut cal = calibrate(&raw, m.bias.as_ref(), m.dark.as_ref(), m.flat.as_ref(), k, cosmetic, 5.0, pattern.is_some())?;
            if let Some(f) = defects {
                cal = f(&cal, pattern.is_some());
            }
            let out = match &pattern {
                Some(p) => {
                    if kind == "RGB" {
                        io::debayer(&cal.clip01(), p)?
                    } else {
                        let (ha, oiii) = crate::narrowband::extract_ha_oiii(&cal, p, true)?;
                        if kind == "Ha" {
                            ha
                        } else {
                            oiii
                        }
                    }
                }
                None => {
                    if kind != "RGB" {
                        bail!("Extraction Ha/OIII : images de capteur couleur (motif de Bayer) requises");
                    }
                    cal
                }
            };
            let d = done.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
            if let Some(p) = progress {
                p(d, n, "calibration");
            }
            Ok(out)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn calib_chain() {
        let bias = Image::filled(32, 32, 1, 0.02);
        let darks: Vec<Image> = (0..5).map(|k| Image::from_fn(32, 32, move |x, y| 0.07 + 0.001 * ((x + y + k) % 3) as f32)).collect();
        let flats: Vec<Image> = (0..5).map(|_| Image::from_fn(32, 32, |x, _| 0.5 * (1.0 + 0.1 * x as f32 / 32.0))).collect();
        let mb = make_master_bias(&darks, 3.0).unwrap();
        let md = make_master_dark(&darks, Some(&mb), 3.0).unwrap();
        let mf = make_master_flat(&flats, Some(&mb), None, 3.0).unwrap();
        let light = Image::filled(32, 32, 1, 0.3);
        let cal = calibrate(&light, Some(&bias), Some(&md), Some(&mf), 1.0, true, 5.0, false).unwrap();
        assert!(cal.min_value() >= 0.0);
        // hot pixel CFA
        let mut l = Image::filled(16, 16, 1, 0.1);
        l.data[5 * 16 + 6] = 0.9;
        let c = remove_hot_pixels(&l, 5.0, true);
        assert!((c.data[5 * 16 + 6] - 0.1).abs() < 1e-6);
    }
}
