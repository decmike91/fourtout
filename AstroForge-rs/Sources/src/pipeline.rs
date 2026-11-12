//! Processus de traitement automatique (liste d'étapes enregistrable en
//! JSON : [{"name": …, "params": {…}, "enabled": true}, …]) et registre
//! des étapes ciel profond / planétaire.

use crate::animation;
use crate::calibration::{self, DarkScale, MasterOptions};
use crate::drizzle;
use crate::effects;
use crate::gradient;
use crate::image::Image;
use crate::registration;
use crate::restoration;
use crate::session;
use crate::stacking::{self, StackParams};
use crate::wavelets;
use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Contexte partagé entre les étapes.
#[derive(Default)]
pub struct Ctx {
    pub light_paths: Vec<PathBuf>,
    pub calib_paths: Option<HashMap<String, Vec<PathBuf>>>,
    pub bayer: Option<String>,
    pub images: Vec<Image>,
    pub pre_align: Option<Vec<Image>>,
    pub image: Option<Image>,
    pub weights: Option<Vec<f64>>,
    pub frames: Vec<Image>,
    pub hdr_images: Vec<Image>,
    pub background_model: Option<Image>,
    pub kept_indices: Vec<usize>,
    pub quality_scores: Vec<f64>,
    pub animation_path: Option<String>,
}

/// (nom, libellé, catégorie, paramètres par défaut).
pub fn steps() -> Vec<(&'static str, &'static str, &'static str, Value)> {
    vec![
        (
            "calibration",
            "Calibration (masters)",
            "Pré-traitement",
            json!({"bias_dir": "", "dark_dir": "", "flat_dir": "", "darkflat_dir": "", "dark_scale": "auto", "cosmetic": true, "use_library": false, "save_to_library": false}),
        ),
        ("registration", "Alignement des images", "Pré-traitement", json!({"with_rotation": false, "max_deg": 2.0})),
        (
            "drizzle",
            "Drizzle + empilement",
            "Empilement",
            json!({"enabled_drizzle": true, "scale": 2, "pixfrac": 0.7, "method": "Moyenne (drizzle)", "stack_method": "Sigma-clipping", "sigma": 3.0}),
        ),
        ("gradient", "Extraction de fond / gradient", "Traitement", json!({"method": "Polynôme 2D", "degree": 3, "grid": 12, "sigma": 2.5, "correction": "Soustraction"})),
        ("denoise", "Débruitage", "Traitement", json!({"provider": "classique", "method": "Non-local means", "strength": 0.5})),
        ("deconvolution", "Déconvolution (Richardson-Lucy)", "Traitement", json!({"sigma_psf": 1.5, "iterations": 20, "damping": 0.2})),
        ("sharpen", "Netteté (masque flou)", "Traitement", json!({"amount": 0.6, "radius": 2.0})),
        ("ai_tool", "Outil IA (netteté/super-résolution/étoiles…)", "IA", json!({"task": "sharpen", "provider": "onnx", "strength": 0.5})),
        ("satellite", "Suppression traces satellites", "Traitement", json!({"min_len_frac": 0.25, "threshold": 8.0, "width": 5})),
        ("hdr", "Composition HDR", "Traitement", json!({"contrast": 1.0, "saturation": 1.0, "well_exposed": 1.0})),
        ("falsecolor", "Fausses couleurs", "Finition", json!({"palette": "Solaire (orange)"})),
        ("quality_rank", "Classement qualité des frames", "Planétaire", json!({"mode": "Netteté (Registax)", "keep_percent": 75.0})),
        ("derotation", "Dérotation du champ", "Planétaire", json!({"mode": "auto", "rate_deg_min": 0.0, "interval_s": 0.0, "auto_max_deg": 2.0})),
        ("align_planetary", "Alignement fin (planétaire)", "Planétaire", json!({"with_rotation": false, "max_deg": 1.0})),
        ("stack_planetary", "Empilement planétaire", "Planétaire", json!({"method": "Moyenne", "sigma": 3.0, "percent": 90.0})),
        ("rgb_align", "Alignement des canaux couleur", "Planétaire", json!({"max_shift": 12.0})),
        ("wavelets", "Ondelettes (style Registax6)", "Planétaire", json!({"levels": 6, "gains": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]})),
        ("export_animation", "Film stabilisé (aligner + recadrer)", "Planétaire", json!({"path": "animation.mp4", "fps": 24, "quality": 90, "crop_size": 0})),
    ]
}

pub const DEFAULT_CHAIN_SKY: [&str; 5] = ["calibration", "registration", "drizzle", "gradient", "denoise"];
pub const DEFAULT_CHAIN_PLANET: [&str; 5] = ["quality_rank", "derotation", "align_planetary", "stack_planetary", "wavelets"];

fn f(p: &Map<String, Value>, k: &str) -> f64 {
    p.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0)
}
fn b(p: &Map<String, Value>, k: &str) -> bool {
    p.get(k).and_then(|v| v.as_bool()).unwrap_or(false)
}
fn s<'a>(p: &'a Map<String, Value>, k: &str, d: &'a str) -> &'a str {
    p.get(k).and_then(|v| v.as_str()).unwrap_or(d)
}

fn image_mut(ctx: &mut Ctx) -> Result<&mut Image> {
    ctx.image.as_mut().ok_or_else(|| anyhow!("Aucune image empilée : ajoutez une étape d'empilement avant"))
}

/// Drizzle sur les images AVANT alignement, rejet contre l'empilement classique.
pub fn drizzle_images(ctx: &Ctx, scale: usize, pixfrac: f64, method: &str, sigma: f64, with_rotation: bool, weights: Option<&[f64]>) -> Result<Image> {
    let src: &Vec<Image> = ctx.pre_align.as_ref().unwrap_or(&ctx.images);
    if src.is_empty() {
        bail!("Drizzle : aucune image");
    }
    let r = &src[src.len() / 2];
    let mats: Vec<_> = {
        use rayon::prelude::*;
        src.par_iter().map(|im| registration::estimate_transform(r, im, with_rotation)).collect()
    };
    let aligned: Vec<Image> = if ctx.pre_align.is_some() { ctx.images.clone() } else { src.iter().zip(&mats).map(|(im, m)| registration::warp(im, m)).collect() };
    let w = weights.filter(|w| w.len() == src.len());
    let reference = stacking::stack(&aligned.iter().collect::<Vec<_>>(), method, &StackParams { sigma, ..Default::default() }, w)?;
    let (img, _) = drizzle::drizzle_stack(src, &mats, scale, pixfrac, w, Some(&reference), Some(sigma), None)?;
    Ok(img)
}

fn run_step(name: &str, ctx: &mut Ctx, p: &Map<String, Value>) -> Result<()> {
    match name {
        "calibration" => {
            let mut light_meta = None;
            if b(p, "use_library") {
                if let Some(l0) = ctx.light_paths.first() {
                    let fi = session::frame_info(l0);
                    light_meta = Some(match json!({"exptime": fi.exptime, "gain": fi.gain, "temp": fi.temp, "binning": fi.binning, "width": fi.width, "height": fi.height, "filter": fi.filter}) {
                        Value::Object(m) => m,
                        _ => unreachable!(),
                    });
                }
            }
            let m = calibration::build_masters(&MasterOptions {
                bias_dir: s(p, "bias_dir", "").into(),
                dark_dir: s(p, "dark_dir", "").into(),
                flat_dir: s(p, "flat_dir", "").into(),
                darkflat_dir: s(p, "darkflat_dir", "").into(),
                sigma: 3.0,
                paths: ctx.calib_paths.clone(),
                light_meta,
                use_library: b(p, "use_library"),
                save_to_library: b(p, "save_to_library"),
                library: None,
            })?;
            let ds = match p.get("dark_scale") {
                Some(Value::Number(n)) => DarkScale::Fixed(n.as_f64().unwrap_or(1.0)),
                _ => DarkScale::Auto,
            };
            ctx.images = calibration::calibrate_paths(&ctx.light_paths, &m, ctx.bayer.as_deref(), ds, p.get("cosmetic").and_then(|v| v.as_bool()).unwrap_or(true), None, None, "RGB")?;
            ctx.pre_align = None;
        }
        "registration" => {
            if ctx.images.is_empty() {
                bail!("Aucune image à aligner");
            }
            let r = ctx.images[ctx.images.len() / 2].clone();
            ctx.pre_align = Some(ctx.images.clone());
            ctx.images = registration::align_stack(&r, &ctx.images, b(p, "with_rotation"), f(p, "max_deg"));
        }
        "drizzle" => {
            let w = ctx.weights.clone();
            let img = if b(p, "enabled_drizzle") && s(p, "method", "").starts_with("Moyenne (drizzle)") {
                drizzle_images(ctx, f(p, "scale") as usize, f(p, "pixfrac"), s(p, "stack_method", "Sigma-clipping"), f(p, "sigma"), false, w.as_deref())?
            } else {
                let wv = w.filter(|w| w.len() == ctx.images.len());
                stacking::stack(&ctx.images.iter().collect::<Vec<_>>(), s(p, "stack_method", "Sigma-clipping"), &StackParams { sigma: f(p, "sigma"), ..Default::default() }, wv.as_deref())?
            };
            ctx.image = Some(img);
        }
        "gradient" => {
            let img = image_mut(ctx)?.clone();
            let (o, m) = gradient::extract_gradient(&img, s(p, "method", "Polynôme 2D"), f(p, "degree") as usize, f(p, "grid") as usize, f(p, "sigma"), s(p, "correction", "Soustraction"))?;
            ctx.image = Some(o);
            ctx.background_model = Some(m);
        }
        "denoise" => {
            let img = image_mut(ctx)?.clone();
            let prov = s(p, "provider", "classique");
            ctx.image = Some(if ["classique", "onnx-classique", ""].contains(&prov) {
                restoration::denoise(&img, s(p, "method", "Non-local means"), f(p, "strength"), 3)
            } else {
                crate::ai::run("denoise", &img, prov, f(p, "strength"), false)?
            });
        }
        "deconvolution" => {
            let img = image_mut(ctx)?.clone();
            ctx.image = Some(restoration::richardson_lucy(&img, f(p, "sigma_psf"), f(p, "iterations") as usize, f(p, "damping")));
        }
        "sharpen" => {
            let img = image_mut(ctx)?.clone();
            ctx.image = Some(restoration::unsharp(&img, f(p, "amount"), f(p, "radius")));
        }
        "ai_tool" => {
            let img = image_mut(ctx)?.clone();
            ctx.image = Some(crate::ai::run(s(p, "task", "sharpen"), &img, s(p, "provider", "onnx"), f(p, "strength"), false)?);
        }
        "satellite" => {
            let img = image_mut(ctx)?.clone();
            ctx.image = Some(effects::remove_satellite_trails(&img, None, f(p, "min_len_frac"), f(p, "threshold"), f(p, "width") as i32)?);
        }
        "hdr" => {
            if ctx.hdr_images.len() >= 2 {
                ctx.image = Some(effects::merge_hdr(&ctx.hdr_images, f(p, "contrast"), f(p, "saturation"), f(p, "well_exposed"))?);
            }
        }
        "falsecolor" => {
            let img = image_mut(ctx)?.clone();
            if !img.is_color() {
                ctx.image = Some(effects::apply_palette(&img, s(p, "palette", "Solaire (orange)"))?);
            }
        }
        "quality_rank" => {
            let (idx, scores) = stacking::rank_frames(&ctx.frames, s(p, "mode", "Netteté (Registax)"), f(p, "keep_percent"));
            ctx.frames = idx.iter().map(|&i| ctx.frames[i].clone()).collect();
            ctx.kept_indices = idx;
            ctx.quality_scores = scores;
        }
        "derotation" => {
            ctx.frames = if s(p, "mode", "auto") == "auto" {
                registration::derotate(&ctx.frames, None, None, f(p, "auto_max_deg"), 0)
            } else {
                registration::derotate(&ctx.frames, Some(f(p, "rate_deg_min")), Some(f(p, "interval_s")), 0.0, 0)
            };
        }
        "align_planetary" => {
            if ctx.frames.is_empty() {
                bail!("Aucune frame");
            }
            let r = ctx.frames[ctx.frames.len() / 2].clone();
            ctx.frames = registration::align_stack(&r, &ctx.frames, b(p, "with_rotation"), f(p, "max_deg"));
        }
        "stack_planetary" => {
            let m = s(p, "method", "Moyenne");
            let mut frames: Vec<&Image> = ctx.frames.iter().collect();
            if m == "Moyenne (meilleurs %)" {
                let k = ((frames.len() as f64 * f(p, "percent") / 100.0) as usize).max(1);
                frames.truncate(k);
            }
            let method = if m.starts_with("Moyenne") { "Moyenne" } else { m };
            ctx.image = Some(stacking::stack(&frames, method, &StackParams { sigma: f(p, "sigma"), ..Default::default() }, None)?);
        }
        "rgb_align" => {
            let img = image_mut(ctx)?.clone();
            if img.is_color() {
                ctx.image = Some(registration::align_channels(&img, Some(f(p, "max_shift"))));
            }
        }
        "wavelets" => {
            let img = image_mut(ctx)?.clone();
            let layers = wavelets::atrous_decompose(&img, f(p, "levels") as usize);
            let mut gains: Vec<f64> = p.get("gains").and_then(|v| v.as_array()).map(|a| a.iter().map(|x| x.as_f64().unwrap_or(1.0)).collect()).unwrap_or_default();
            gains.resize(layers.len(), 1.0);
            ctx.image = Some(wavelets::atrous_recompose(&layers, Some(&gains), None));
        }
        "export_animation" => {
            let size = f(p, "crop_size") as usize;
            let st = animation::stabilized_frames(&ctx.frames, if size > 0 { Some(size) } else { None }, None)?;
            let path = s(p, "path", "animation.mp4").to_string();
            animation::export_video(&st, Path::new(&path), f(p, "fps") as u32, f(p, "quality") as u32)?;
            ctx.animation_path = Some(path);
        }
        _ => bail!("Étape inconnue : {}", name),
    }
    Ok(())
}

/// Liste ordonnée d'étapes {name, params, enabled}.
#[derive(Clone, Debug, Default)]
pub struct Pipeline {
    pub steps: Vec<Value>,
}

impl Pipeline {
    pub fn default_for(mode: &str) -> Pipeline {
        let names: &[&str] = if mode == "sky" { &DEFAULT_CHAIN_SKY } else { &DEFAULT_CHAIN_PLANET };
        let reg = steps();
        Pipeline {
            steps: names
                .iter()
                .map(|n| {
                    let d = reg.iter().find(|s| s.0 == *n).unwrap().3.clone();
                    json!({"name": n, "params": d, "enabled": true})
                })
                .collect(),
        }
    }

    pub fn load(path: &Path) -> Result<Pipeline> {
        let v: Value = serde_json::from_str(&std::fs::read_to_string(path).with_context(|| format!("Lecture impossible : {}", path.display()))?)?;
        let arr = v.as_array().cloned().or_else(|| v.get("steps").and_then(|s| s.as_array()).cloned()).ok_or_else(|| anyhow!("Processus : liste d'étapes attendue"))?;
        Ok(Pipeline { steps: arr })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        std::fs::write(path, serde_json::to_string_pretty(&self.steps)?)?;
        Ok(())
    }

    pub fn run(&self, ctx: &mut Ctx, progress: Option<&dyn Fn(usize, usize, &str)>) -> Result<()> {
        let reg = steps();
        let enabled: Vec<&Value> = self.steps.iter().filter(|s| s.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true)).collect();
        let total = enabled.len();
        for (done, st) in enabled.into_iter().enumerate() {
            let name = st.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let Some((_, label, _, defaults)) = reg.iter().find(|s| s.0 == name) else { bail!("Étape inconnue : {}", name) };
            let mut merged = defaults.as_object().cloned().unwrap_or_default();
            if let Some(Value::Object(pm)) = st.get("params") {
                for (k, v) in pm {
                    merged.insert(k.clone(), v.clone());
                }
            }
            if let Some(p) = progress {
                p(done, total, label);
            }
            run_step(name, ctx, &merged).with_context(|| format!("Étape « {} » a échoué", label))?;
            if let Some(p) = progress {
                p(done + 1, total, label);
            }
        }
        Ok(())
    }
}
