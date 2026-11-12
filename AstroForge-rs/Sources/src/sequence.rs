//! Séquences batch (esprit Registax) : chaîne d'opérations enregistrée en
//! JSON {"mode": "image"|"ser", "ops": [{"op": …, "params": {…}}]} et
//! rejouée sur tout un dossier d'images ou de vidéos SER.

use crate::colorcal;
use crate::display;
use crate::effects;
use crate::finishing;
use crate::geom::{self, Interp};
use crate::image::Image;
use crate::io;
use crate::planetary::{self, PlanetOptions, Tone, ToneFrames};
use crate::registration;
use crate::restoration;
use crate::stacking;
use crate::wavelets;
use anyhow::{bail, Result};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

/// (nom, libellé, paramètres par défaut).
pub fn ops() -> Vec<(&'static str, &'static str, Value)> {
    vec![
        ("brightness_contrast", "Luminosité / contraste", json!({"bright": 0.0, "contrast": 0.0})),
        ("gamma", "Gamma", json!({"gamma": 1.0})),
        ("saturation", "Saturation", json!({"sat": 1.0})),
        ("white_balance", "Balance des blancs", json!({"wr": 1.0, "wg": 1.0, "wb": 1.0})),
        ("color_calibrate", "Calibration couleur (fond ciel)", json!({"use_background": true, "percentile": 25.0})),
        ("background_neutralize", "Neutralisation du fond", json!({"percentile": 25.0})),
        ("scnr_green", "SCNR vert", json!({})),
        ("autostretch", "Étirement automatique (affichage/traitement)", json!({})),
        ("denoise", "Débruitage classique", json!({"method": "Non-local means", "strength": 0.5})),
        ("unsharp", "Masque flou", json!({"amount": 0.6, "radius": 2.0})),
        ("deconv_rl", "Déconvolution Richardson-Lucy", json!({"sigma": 1.5, "iterations": 20})),
        ("remove_satellite_trails", "Suppression traces satellites", json!({"min_len_frac": 0.25, "threshold": 8.0, "width": 5})),
        ("apply_palette", "Fausses couleurs", json!({"palette": "Solaire (orange)"})),
        ("superres_lanczos", "Super-résolution Lanczos", json!({"scale": 2})),
        ("ai_run", "Outil IA local", json!({"task": "denoise", "provider": "onnx", "strength": 0.5})),
        (
            "gimp_tone",
            "Finition GIMP (tonalité + netteté)",
            json!({"bright": 0.0, "contrast": 0.0, "sat": 1.0, "gamma": 1.0, "wr": 1.0, "wg": 1.0, "wb": 1.0, "usm_amount": 0.0, "usm_radius": 2.0}),
        ),
        ("fin_saturation", "Finition : saturation", json!({"amount": 1.0})),
        ("fin_white_balance", "Finition : balance des blancs", json!({"wr": 1.0, "wg": 1.0, "wb": 1.0})),
        ("fin_background_neutral", "Finition : neutralisation du fond", json!({"percentile": 5.0, "strength": 1.0})),
        ("fin_scnr_green", "Finition : suppression du vert (SCNR)", json!({"strength": 1.0})),
        ("fin_autostretch", "Finition : étirement automatique (médiane/MAD)", json!({"shadows_clip": -2.8, "target_bg": 0.25})),
        ("fin_gradient", "Finition : suppression de gradient (extraction de fond)", json!({"kernel": 64, "method": "Soustraire"})),
        ("fin_median_denoise", "Finition : débruitage médian (répété)", json!({"radius": 1, "iterations": 1})),
        ("fin_unsharp", "Finition : masque flou (netteté)", json!({"radius": 2.0, "gain": 0.6})),
        ("fin_levels", "Finition : niveaux", json!({"black": 0.0, "gamma": 1.0, "white": 1.0})),
        ("fin_shadows_highlights", "Finition : tons clairs / tons sombres", json!({"shadows": 0.0, "highlights": 0.0, "radius": 24})),
        ("fin_vibrance", "Finition : vibrance", json!({"amount": 1.0})),
        (
            "fin_hsl",
            "Finition : teinte/saturation/luminosité (HSL)",
            json!({"hue": 0, "sat": 1.0, "lum": 0.0, "r_hue": 0, "r_sat": 1.0, "y_hue": 0, "y_sat": 1.0, "g_hue": 0, "g_sat": 1.0, "c_hue": 0, "c_sat": 1.0, "b_hue": 0, "b_sat": 1.0, "m_hue": 0, "m_sat": 1.0}),
        ),
        ("fin_local_contrast", "Finition : contraste local (clarté)", json!({"radius": 32, "gain": 0.0})),
        (
            "fin_split_tone",
            "Finition : tonalité divisée",
            json!({"sh_r": 0.5, "sh_g": 0.5, "sh_b": 0.5, "hl_r": 0.5, "hl_g": 0.5, "hl_b": 0.5, "balance": 0.0, "force": 0.0}),
        ),
        (
            "fin_gradient_map",
            "Finition : mappage de dégradé",
            json!({"r0": 0.0, "g0": 0.0, "b0": 0.0, "r1": 0.5, "g1": 0.5, "b1": 0.5, "r2": 1.0, "g2": 1.0, "b2": 1.0, "intensity": 0.0}),
        ),
        ("fin_ghs", "Finition : étirement GHS", json!({"D": 0.0, "b": 0.0, "SP": 0.0, "LP": 0.0, "HP": 1.0, "method": "Canaux"})),
        ("fin_arcsinh", "Finition : étirement arcsinh", json!({"stretch": 0.0, "black": 0.0})),
        ("fin_star_reduction", "Finition : réduction d'étoiles", json!({"amount": 0.0, "iterations": 1})),
        ("fin_chroma_denoise", "Finition : débruitage de la chrominance", json!({"strength": 0.0, "scales": 4})),
        ("fin_masked", "Finition : filtre à travers un masque", json!({"filter": "none", "params": {}, "mask": {"kind": "Aucun"}})),
    ]
}

pub fn op_names() -> Vec<&'static str> {
    ops().into_iter().map(|o| o.0).collect()
}

fn fin_id(op: &str) -> Option<&'static str> {
    Some(match op {
        "fin_saturation" => "saturation",
        "fin_white_balance" => "white_balance",
        "fin_background_neutral" => "background_neutral",
        "fin_scnr_green" => "scnr_green",
        "fin_autostretch" => "autostretch",
        "fin_gradient" => "remove_gradient",
        "fin_median_denoise" => "denoise_median",
        "fin_unsharp" => "unsharp_mask",
        "fin_levels" => "levels",
        "fin_shadows_highlights" => "shadows_highlights",
        "fin_vibrance" => "vibrance",
        "fin_hsl" => "hsl",
        "fin_local_contrast" => "local_contrast",
        "fin_split_tone" => "split_tone",
        "fin_gradient_map" => "gradient_map",
        "fin_ghs" => "ghs",
        "fin_arcsinh" => "arcsinh",
        "fin_star_reduction" => "star_reduction",
        "fin_chroma_denoise" => "chroma_denoise",
        _ => return None,
    })
}

fn f(p: &Map<String, Value>, k: &str) -> f64 {
    match p.get(k) {
        Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0),
        Some(Value::Bool(b)) => *b as i64 as f64,
        Some(Value::String(s)) => s.parse().unwrap_or(0.0),
        _ => 0.0,
    }
}

fn s<'a>(p: &'a Map<String, Value>, k: &str, d: &'a str) -> &'a str {
    p.get(k).and_then(|v| v.as_str()).unwrap_or(d)
}

/// Applique une opération (paramètres = défauts + valeurs fournies).
pub fn apply_op(img: &Image, op: &str, params: &Map<String, Value>) -> Result<Image> {
    let Some((_, _, defaults)) = ops().into_iter().find(|o| o.0 == op) else { bail!("Opération inconnue dans la séquence : {}", op) };
    let mut p = defaults.as_object().cloned().unwrap_or_default();
    for (k, v) in params {
        p.insert(k.clone(), v.clone());
    }
    let a = img;
    Ok(match op {
        "brightness_contrast" => {
            let (b, c) = (f(&p, "bright") as f32, f(&p, "contrast") as f32);
            a.map(|v| ((v - 0.5) * (1.0 + c) + 0.5 + b).clamp(0.0, 1.0))
        }
        "gamma" => {
            let g = f(&p, "gamma").max(0.01) as f32;
            a.map(|v| v.clamp(0.0, 1.0).powf(1.0 / g))
        }
        "saturation" => {
            if a.c != 3 {
                a.clone()
            } else {
                let k = f(&p, "sat") as f32;
                let mut o = a.clone();
                for px in o.data.chunks_mut(3) {
                    let g = (px[0] + px[1] + px[2]) / 3.0;
                    for v in px.iter_mut() {
                        *v = (g + (*v - g) * k).clamp(0.0, 1.0);
                    }
                }
                o
            }
        }
        "white_balance" => {
            let mut o = a.clone();
            if o.c == 3 {
                let g = [f(&p, "wr") as f32, f(&p, "wg") as f32, f(&p, "wb") as f32];
                for px in o.data.chunks_mut(3) {
                    for k in 0..3 {
                        px[k] *= g[k];
                    }
                }
            }
            o.clip01()
        }
        "color_calibrate" => colorcal::color_calibrate(a, [1.0, 1.0, 1.0], p.get("use_background").and_then(|v| v.as_bool()).unwrap_or(true), f(&p, "percentile")),
        "background_neutralize" => colorcal::background_neutralize(a, f(&p, "percentile")),
        "scnr_green" => colorcal::scnr_green(a),
        "autostretch" => display::autostretch(a),
        "denoise" => restoration::denoise(a, s(&p, "method", "Non-local means"), f(&p, "strength"), 3),
        "unsharp" => restoration::unsharp(a, f(&p, "amount"), f(&p, "radius")),
        "deconv_rl" => restoration::richardson_lucy(a, f(&p, "sigma"), f(&p, "iterations") as usize, 0.2),
        "remove_satellite_trails" => effects::remove_satellite_trails(a, None, f(&p, "min_len_frac"), f(&p, "threshold"), f(&p, "width") as i32)?,
        "apply_palette" => {
            if a.is_color() {
                a.clone()
            } else {
                effects::apply_palette(a, s(&p, "palette", "Solaire (orange)"))?
            }
        }
        "superres_lanczos" => {
            let k = f(&p, "scale");
            geom::resize(a, (a.w as f64 * k).round() as usize, (a.h as f64 * k).round() as usize, Interp::Lanczos4)
        }
        "ai_run" => crate::ai::run(s(&p, "task", "denoise"), a, s(&p, "provider", "onnx"), f(&p, "strength"), false)?,
        "gimp_tone" => {
            let mut o = a.clone();
            if o.c == 3 {
                let g = [f(&p, "wr") as f32, f(&p, "wg") as f32, f(&p, "wb") as f32];
                for px in o.data.chunks_mut(3) {
                    for k in 0..3 {
                        px[k] *= g[k];
                    }
                }
            }
            let (b, c) = (f(&p, "bright") as f32, f(&p, "contrast") as f32);
            o.map_inplace(|v| (v - 0.5) * (1.0 + c) + 0.5 + b);
            let sat = f(&p, "sat");
            if o.c == 3 && (sat - 1.0).abs() > 1e-3 {
                for px in o.data.chunks_mut(3) {
                    let g = (px[0] + px[1] + px[2]) / 3.0;
                    for v in px.iter_mut() {
                        *v = g + (*v - g) * sat as f32;
                    }
                }
            }
            let gm = f(&p, "gamma").max(0.01) as f32;
            o.map_inplace(|v| v.clamp(0.0, 1.0).powf(1.0 / gm));
            if f(&p, "usm_amount") > 0.0 {
                o = restoration::unsharp(&o, f(&p, "usm_amount"), f(&p, "usm_radius"));
            }
            o.clip01()
        }
        "fin_masked" => {
            let filter = s(&p, "filter", "none").to_string();
            let fp = p.get("params").and_then(|v| v.as_object()).cloned().unwrap_or_default();
            let mk = p.get("mask").and_then(|v| v.as_object()).cloned().unwrap_or_default();
            finishing::apply_filter_masked(a, &filter, &fp, &mk)?
        }
        other => match fin_id(other) {
            Some(fid) => finishing::apply_filter(a, fid, &p)?,
            None => bail!("Opération inconnue dans la séquence : {}", other),
        },
    })
}

pub fn save_sequence(path: &Path, ops: &[Value], mode: &str) -> Result<()> {
    std::fs::write(path, serde_json::to_string_pretty(&json!({"mode": mode, "ops": ops}))?)?;
    Ok(())
}

pub fn load_sequence(path: &Path) -> Result<Value> {
    let seq: Value = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    let mode = seq.get("mode").and_then(|v| v.as_str()).unwrap_or("image");
    let names = op_names();
    for o in seq.get("ops").and_then(|v| v.as_array()).cloned().unwrap_or_default() {
        let op = o.get("op").and_then(|v| v.as_str()).unwrap_or("");
        if !names.contains(&op) && !(mode == "ser" && op == "planet_chain") {
            bail!("Opération inconnue dans la séquence : {}", op);
        }
    }
    Ok(seq)
}

pub fn apply_to_image(seq: &Value, img: &Image) -> Result<Image> {
    let mut im = img.clone();
    for o in seq.get("ops").and_then(|v| v.as_array()).cloned().unwrap_or_default() {
        let op = o.get("op").and_then(|v| v.as_str()).unwrap_or("");
        let p = o.get("params").and_then(|v| v.as_object()).cloned().unwrap_or_default();
        im = apply_op(&im, op, &p)?;
    }
    Ok(im)
}

/// Fichiers produits + échecs ; rapport texte écrit dans le dossier de sortie.
#[derive(Debug, Default)]
pub struct BatchResult {
    pub outputs: Vec<String>,
    pub errors: Vec<(String, String)>,
}

impl BatchResult {
    pub fn summary(&self) -> String {
        let mut t = format!("{} fichier(s) produit(s)", self.outputs.len());
        if !self.errors.is_empty() {
            let parts: Vec<String> = self.errors.iter().take(5).map(|(f, m)| format!("{} ({})", f, m.lines().next().unwrap_or("?"))).collect();
            t += &format!(", {} échec(s) : {}", self.errors.len(), parts.join("; "));
            if self.errors.len() > 5 {
                t += " …";
            }
        }
        t
    }
}

fn finish_batch(out_dir: &Path, res: BatchResult, total: usize) -> Result<BatchResult> {
    let mut txt = format!("AstroForge — rapport de traitement par lot\n{}\n\n", res.summary());
    for n in &res.outputs {
        txt += &format!("OK     {}\n", n);
    }
    for (n, m) in &res.errors {
        txt += &format!("ÉCHEC  {}\n{}\n", n, m);
    }
    let _ = std::fs::write(out_dir.join("rapport_batch.txt"), txt);
    if total > 0 && res.outputs.is_empty() {
        let detail: Vec<String> = res.errors.iter().take(10).map(|(f, m)| format!("- {} : {}", f, m.lines().next().unwrap_or("?"))).collect();
        bail!("Aucun fichier traité avec succès sur {}.\n{}", total, detail.join("\n"));
    }
    Ok(res)
}

fn list_with(dir: &Path, exts: &[&str]) -> Result<Vec<PathBuf>> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.is_file() && io::has_ext(p, exts)).collect();
    v.sort();
    Ok(v)
}

pub const IMAGE_EXTS: [&str; 10] = ["fits", "fit", "fts", "fz", "tif", "tiff", "png", "jpg", "jpeg", "xisf"];

type Prog<'a> = Option<&'a (dyn Fn(usize, usize, &str) + Sync)>;

/// Applique la séquence à toutes les images du dossier (sortie TIFF).
pub fn run_batch_image(seq: &Value, in_dir: &Path, out_dir: &Path, bit_depth: &str, progress: Prog) -> Result<BatchResult> {
    let files = list_with(in_dir, &IMAGE_EXTS)?;
    if files.is_empty() {
        bail!("Aucune image dans {}", in_dir.display());
    }
    std::fs::create_dir_all(out_dir)?;
    let mut res = BatchResult::default();
    for (i, f) in files.iter().enumerate() {
        let name = format!("{}.tif", f.file_stem().unwrap().to_string_lossy());
        let r = (|| -> Result<()> {
            let img = io::load(f)?;
            let img = apply_to_image(seq, &img)?;
            io::save_image(&out_dir.join(&name), &img, bit_depth, None)
        })();
        match r {
            Ok(()) => res.outputs.push(name),
            Err(e) => res.errors.push((f.file_name().unwrap().to_string_lossy().to_string(), format!("{:#}", e))),
        }
        if let Some(p) = progress {
            p(i + 1, files.len(), "Lot d'images");
        }
    }
    finish_batch(out_dir, res, files.len())
}

fn get_f(m: &Map<String, Value>, k: &str, d: f64) -> f64 {
    m.get(k).and_then(|v| v.as_f64()).unwrap_or(d)
}

fn get_vec(m: &Map<String, Value>, k: &str) -> Option<Vec<f64>> {
    m.get(k).and_then(|v| v.as_array()).map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect())
}

/// Chaîne planétaire complète par SER : classement -> alignement ->
/// empilement -> ondelettes -> palette, avec les réglages enregistrés.
pub fn run_batch_ser_chain(seq: &Value, in_dir: &Path, out_dir: &Path, max_frames: usize, progress: Prog) -> Result<BatchResult> {
    let files = list_with(in_dir, &["ser"])?;
    if files.is_empty() {
        bail!("Aucun fichier SER dans {}", in_dir.display());
    }
    std::fs::create_dir_all(out_dir)?;
    let st: Map<String, Value> = seq
        .get("ops")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .and_then(|o| o.get("params"))
        .and_then(|p| p.as_object())
        .cloned()
        .unwrap_or_default();
    let mut res = BatchResult::default();
    for (i, f) in files.iter().enumerate() {
        let name = format!("{}_stacked.fits", f.file_stem().unwrap().to_string_lossy());
        let r = (|| -> Result<()> {
            let (frames, _m) = io::ser::load_ser(f, max_frames)?;
            let tf = ToneFrames { originals: frames, tone: Tone::from_json(st.get("tone"))?, boxr: None };
            tf.validate()?;
            let n_all = crate::frames::Frames::len(&tf);
            let n_sel = match st.get("n_align").and_then(|v| v.as_f64()) {
                Some(v) if v > 0.0 => v as usize,
                _ => ((n_all as f64 * get_f(&st, "keep_percent", 75.0) / 100.0).round() as usize).max(1),
            };
            let quality = st.get("quality_method").or_else(|| st.get("quality_mode")).and_then(|v| v.as_str()).unwrap_or("Contraste HF (AutoStakkert)");
            let opts = PlanetOptions {
                quality_method: quality.into(),
                roi_mode: s(&st, "roi_mode", "Disque auto-détecté").into(),
                n_align: n_sel,
                align_mode: s(&st, "align_mode", "Multi-points (grille NxN)").into(),
                grid_n: get_f(&st, "grid_n", 4.0) as usize,
                box_size: get_f(&st, "box_size", 0.0) as usize,
                manual_points: None,
                stack_method: s(&st, "stack_method", "Moyenne").into(),
                stack_percent: get_f(&st, "stack_percent", 100.0),
            };
            let r = planetary::run_planetary(&tf, &opts, progress)?;
            let mut img = r.image;
            if st.get("rgb_align").map(|v| v.as_bool().unwrap_or(false)).unwrap_or(false) && img.is_color() {
                img = registration::align_channels(&img, None);
            }
            if st.get("ca").map(|v| v.as_bool().unwrap_or(false)).unwrap_or(false) {
                img = colorcal::correct_chromatic_aberration(&img, get_f(&st, "kr", 0.0), get_f(&st, "kb", 0.0), None);
            }
            let gains = get_vec(&st, "gains").unwrap_or_else(|| vec![1.0; 7]);
            let ld = get_vec(&st, "layer_denoise");
            let ls = get_vec(&st, "layer_sharpen");
            img = wavelets::registax_enhance(&img, Some(&gains), get_f(&st, "wave_denoise", 0.0), ld.as_deref(), ls.as_deref())?;
            if st.get("false_color").map(|v| v.as_bool().unwrap_or(false)).unwrap_or(false) && !img.is_color() {
                img = effects::apply_palette(&img, s(&st, "palette", "Solaire (orange)"))?;
            }
            io::save(out_dir.join(&name), &img)
        })();
        match r {
            Ok(()) => res.outputs.push(name),
            Err(e) => res.errors.push((f.file_name().unwrap().to_string_lossy().to_string(), format!("{:#}", e))),
        }
        if let Some(p) = progress {
            p(i + 1, files.len(), "Lot SER");
        }
    }
    finish_batch(out_dir, res, files.len())
}

/// Ancienne chaîne SER : classement -> alignement -> moyenne -> opérations.
pub fn run_batch_ser(seq: &Value, in_dir: &Path, out_dir: &Path, max_frames: usize, keep_percent: f64) -> Result<BatchResult> {
    let files = list_with(in_dir, &["ser"])?;
    if files.is_empty() {
        bail!("Aucun fichier SER dans {}", in_dir.display());
    }
    std::fs::create_dir_all(out_dir)?;
    let mut res = BatchResult::default();
    for f in &files {
        let name = format!("{}_stacked.fits", f.file_stem().unwrap().to_string_lossy());
        let r = (|| -> Result<()> {
            let (frames, _) = io::ser::load_ser(f, max_frames)?;
            let arr = frames.to_vec()?;
            let (idx, _) = stacking::rank_frames(&arr, "Netteté (Registax)", keep_percent);
            let sel: Vec<Image> = idx.iter().map(|&j| arr[j].clone()).collect();
            let r = sel[sel.len() / 2].clone();
            let al = registration::align_stack(&r, &sel, false, 3.0);
            let img = stacking::stack_owned(&al, "Moyenne", 3.0, None)?;
            let img = apply_to_image(seq, &img)?;
            io::save(out_dir.join(&name), &img)
        })();
        match r {
            Ok(()) => res.outputs.push(name),
            Err(e) => res.errors.push((f.file_name().unwrap().to_string_lossy().to_string(), format!("{:#}", e))),
        }
    }
    finish_batch(out_dir, res, files.len())
}
