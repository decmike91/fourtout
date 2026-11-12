//! Filtres de finition (implémentations originales inspirées des principes
//! de Photoshop, PixInsight, Siril, APP) : saturation, balance des blancs,
//! neutralisation du fond, SCNR, étirement automatique, gradient, médian,
//! masque flou, niveaux, tons clairs/sombres, vibrance, HSL, contraste
//! local, tonalité divisée, mappage de dégradé, GHS, arcsinh, réduction
//! d'étoiles, débruitage de chrominance — mono ou RGB, sortie [0, 1].

use crate::display::mtf;
use crate::image::Image;
use crate::masks;
use crate::ndimage;
use crate::stats;
use crate::stretch;
use anyhow::{bail, Result};
use serde_json::{Map, Value};

fn check(img: &Image) -> Result<()> {
    img.check_mono_or_rgb()
}

fn out(a: Image) -> Image {
    a.clip01()
}

pub fn luminance(img: &Image) -> Image {
    img.rec709()
}

pub fn saturation(img: &Image, amount: f64) -> Result<Image> {
    check(img)?;
    let k = amount.clamp(0.0, 2.0) as f32;
    if img.c == 1 || k == 1.0 {
        return Ok(out(img.clone()));
    }
    let mut a = img.clone();
    for p in a.data.chunks_mut(3) {
        let l = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
        for v in p.iter_mut() {
            *v = l + (*v - l) * k;
        }
    }
    Ok(out(a))
}

pub fn white_balance(img: &Image, wr: f64, wg: f64, wb: f64) -> Result<Image> {
    check(img)?;
    let mut a = img.clone();
    if a.c == 3 {
        let g = [wr.clamp(0.0, 2.0) as f32, wg.clamp(0.0, 2.0) as f32, wb.clamp(0.0, 2.0) as f32];
        for p in a.data.chunks_mut(3) {
            for k in 0..3 {
                p[k] *= g[k];
            }
        }
    }
    Ok(out(a))
}

pub fn neutralize_background(img: &Image, percentile: f64, strength: f64) -> Result<Image> {
    check(img)?;
    if img.c == 1 {
        return Ok(out(img.clone()));
    }
    let p = percentile.clamp(0.0, 49.0);
    let s = strength.clamp(0.0, 1.0);
    if s <= 0.0 {
        return Ok(out(img.clone()));
    }
    let bg: Vec<f64> = (0..3).map(|k| stats::percentile(&img.channel(k).data, p) as f64).collect();
    let target = stats::median_f64(&bg);
    let mut a = img.clone();
    for px in a.data.chunks_mut(3) {
        for k in 0..3 {
            px[k] -= ((bg[k] - target) * s) as f32;
        }
    }
    Ok(out(a))
}

pub fn scnr_green(img: &Image, strength: f64) -> Result<Image> {
    check(img)?;
    let mut a = img.clone();
    if a.c == 3 {
        let s = strength.clamp(0.0, 1.0) as f32;
        if s > 0.0 {
            for p in a.data.chunks_mut(3) {
                let g = p[1].min((p[0] + p[2]) * 0.5);
                p[1] = p[1] * (1.0 - s) + g * s;
            }
        }
    }
    Ok(out(a))
}

/// Étirement automatique (médiane/MAD de la luminance + MTF).
pub fn autostretch(img: &Image, shadows_clip: f64, target_bg: f64) -> Result<Image> {
    check(img)?;
    let lum = luminance(img);
    let med = stats::median(&lum.data) as f64;
    let sigma = stats::mad_about(&lum.data, med as f32) as f64 * 1.4826;
    if sigma < 1e-6 {
        return Ok(out(img.clone()));
    }
    let black = (med + shadows_clip * sigma).clamp(0.0, (med - 1e-6).max(0.0));
    let x0 = (med - black) / (1.0 - black).max(1e-6);
    let t = target_bg.clamp(0.05, 0.5);
    if x0 <= 1e-7 {
        return Ok(out(img.clone()));
    }
    let m = (x0 * (1.0 - t) / (x0 - 2.0 * t * x0 + t)).clamp(1e-6, 1.0 - 1e-6);
    let den = (1.0 - black).max(1e-6) as f32;
    let b = black as f32;
    Ok(img.map(|v| mtf(m, ((v - b) / den).clamp(0.0, 1.0)).clamp(0.0, 1.0)))
}

pub fn remove_gradient(img: &Image, kernel: f64, method: &str) -> Result<Image> {
    check(img)?;
    let sigma = (kernel / 6.0).max(1.0);
    let bg = ndimage::gaussian_img(img, sigma);
    let level = stats::median(&luminance(&bg).data);
    if method.starts_with("Div") {
        return Ok(out(img.zip_map(&bg, |a, b| a * (level / b.max(1e-3)))));
    }
    Ok(out(img.zip_map(&bg, |a, b| a - b + level)))
}

pub fn denoise_median(img: &Image, radius: i64, iterations: i64) -> Result<Image> {
    check(img)?;
    let r = radius.clamp(0, 20) as usize;
    let n = iterations.clamp(1, 8) as usize;
    let mut o = out(img.clone());
    if r == 0 {
        return Ok(o);
    }
    for _ in 0..n {
        o = o.map_planes(|p| ndimage::median_filter(p, 2 * r + 1));
    }
    Ok(o)
}

pub fn unsharp_mask(img: &Image, radius: f64, gain: f64) -> Result<Image> {
    check(img)?;
    let g = gain.clamp(0.0, 3.0) as f32;
    if g <= 0.0 || radius <= 0.0 {
        return Ok(out(img.clone()));
    }
    let blur = ndimage::gaussian_img(img, radius.max(0.3));
    Ok(out(img.zip_map(&blur, |a, b| a + g * (a - b))))
}

/// Courbe [[x, y], …] normalisée (identité si None).
pub fn normalize_curve(curve: Option<&Value>) -> Result<Vec<(f64, f64)>> {
    let id = vec![(0.0, 0.0), (1.0, 1.0)];
    let Some(c) = curve else { return Ok(id) };
    if c.is_null() {
        return Ok(id);
    }
    let arr = c.as_array().ok_or_else(|| anyhow::anyhow!("Courbe : liste d'au moins deux paires (x, y)."))?;
    let mut pts: Vec<(f64, f64)> = if arr.iter().all(|v| v.is_number()) {
        if arr.len() != 5 {
            bail!("Courbe héritée : exactement cinq valeurs.");
        }
        [0.0, 0.25, 0.5, 0.75, 1.0].iter().zip(arr).map(|(&x, y)| (x, y.as_f64().unwrap_or(0.0))).collect()
    } else {
        let mut v = Vec::new();
        for p in arr {
            let pa = p.as_array().ok_or_else(|| anyhow::anyhow!("Courbe : liste d'au moins deux paires (x, y)."))?;
            if pa.len() != 2 {
                bail!("Courbe : liste d'au moins deux paires (x, y).");
            }
            v.push((pa[0].as_f64().unwrap_or(f64::NAN), pa[1].as_f64().unwrap_or(f64::NAN)));
        }
        v
    };
    if pts.len() < 2 {
        bail!("Courbe : liste d'au moins deux paires (x, y).");
    }
    if pts.iter().any(|p| !p.0.is_finite() || !p.1.is_finite()) {
        bail!("Courbe : valeurs finies requises.");
    }
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    for p in pts.iter_mut() {
        *p = (p.0.clamp(0.0, 1.0), p.1.clamp(0.0, 1.0));
    }
    let mut prev = pts[0].0;
    for p in pts.iter_mut().skip(1) {
        p.0 = p.0.max(prev + 1e-6).min(1.0);
        prev = p.0;
    }
    Ok(pts)
}

pub fn is_identity_curve(c: &[(f64, f64)]) -> bool {
    c == [(0.0, 0.0), (1.0, 1.0)]
}

pub fn apply_curve_value(v: f32, c: &[(f64, f64)]) -> f32 {
    let xs: Vec<f64> = c.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = c.iter().map(|p| p.1).collect();
    stats::interp(v as f64, &xs, &ys) as f32
}

/// Courbes « commune », « r », « g », « b ».
pub fn apply_curves(img: &Image, curves: &Map<String, Value>) -> Result<Image> {
    check(img)?;
    let mut norm = std::collections::HashMap::new();
    for key in ["commune", "r", "g", "b"] {
        if let Some(c) = curves.get(key) {
            if !c.is_null() {
                let pts = normalize_curve(Some(c))?;
                if !is_identity_curve(&pts) {
                    norm.insert(key, pts);
                }
            }
        }
    }
    if norm.is_empty() {
        return Ok(out(img.clone()));
    }
    let mut o = img.clone();
    if let Some(c) = norm.get("commune") {
        let xs: Vec<f64> = c.iter().map(|p| p.0).collect();
        let ys: Vec<f64> = c.iter().map(|p| p.1).collect();
        o.map_inplace(|v| stats::interp(v as f64, &xs, &ys) as f32);
    }
    if o.c == 3 {
        for (idx, key) in [(0usize, "r"), (1, "g"), (2, "b")] {
            if let Some(c) = norm.get(key) {
                let xs: Vec<f64> = c.iter().map(|p| p.0).collect();
                let ys: Vec<f64> = c.iter().map(|p| p.1).collect();
                for p in o.data.chunks_mut(3) {
                    p[idx] = stats::interp(p[idx] as f64, &xs, &ys) as f32;
                }
            }
        }
    }
    Ok(out(o))
}

pub fn levels(img: &Image, black: f64, gamma: f64, white: f64) -> Result<Image> {
    check(img)?;
    let b = black.clamp(0.0, 0.5);
    let w = white.clamp(0.5, 1.0);
    let g = gamma.clamp(0.2, 5.0);
    if w - b < 1e-4 {
        return Ok(out(img.clone()));
    }
    Ok(img.map(|v| {
        let t = ((v as f64 - b) / (w - b)).clamp(0.0, 1.0);
        (if (g - 1.0).abs() > 1e-9 { t.powf(1.0 / g) } else { t }) as f32
    }))
}

pub fn shadows_highlights(img: &Image, shadows: f64, highlights: f64, radius: f64) -> Result<Image> {
    check(img)?;
    let fs = shadows.clamp(-1.0, 1.0);
    let fh = highlights.clamp(-1.0, 1.0);
    if fs == 0.0 && fh == 0.0 {
        return Ok(out(img.clone()));
    }
    let lum = luminance(img);
    let ls = ndimage::gaussian(&lum, radius.max(0.5));
    let c = img.c;
    let mut o = img.clone();
    for i in 0..lum.data.len() {
        let s = ls.data[i] as f64;
        let wsh = (1.0 - s).clamp(0.0, 1.0).powi(2);
        let whl = s.clamp(0.0, 1.0).powi(2);
        let l = lum.data[i] as f64;
        let nl = (l + (fs * wsh + fh * whl) * 0.4).clamp(0.0, 1.0);
        let ratio = (nl / l.max(1e-3)) as f32;
        for k in 0..c {
            o.data[i * c + k] *= ratio;
        }
    }
    Ok(out(o))
}

pub fn vibrance(img: &Image, amount: f64) -> Result<Image> {
    check(img)?;
    let k = (amount.clamp(0.0, 2.0) - 1.0) as f32;
    if img.c == 1 || k == 0.0 {
        return Ok(out(img.clone()));
    }
    let mut a = img.clone();
    for p in a.data.chunks_mut(3) {
        let l = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
        let mx = p[0].max(p[1]).max(p[2]);
        let mn = p[0].min(p[1]).min(p[2]);
        let s = ((mx - mn) / mx.max(1e-3)).clamp(0.0, 1.0);
        for v in p.iter_mut() {
            *v += k * (1.0 - s) * (*v - l);
        }
    }
    Ok(out(a))
}

fn rgb_to_hsv(p: &[f32]) -> (f64, f64, f64) {
    let (r, g, b) = (p[0] as f64, p[1] as f64, p[2] as f64);
    let mx = r.max(g).max(b);
    let mn = r.min(g).min(b);
    let d = mx - mn;
    let safe = d.max(1e-12);
    // argmax : premier maximum (ordre R, G, B) comme NumPy
    let cmax = if r >= g && r >= b { 0 } else if g >= b { 1 } else { 2 };
    let mut h = match cmax {
        0 => ((g - b) / safe).rem_euclid(6.0),
        1 => (b - r) / safe + 2.0,
        _ => (r - g) / safe + 4.0,
    };
    h = (h / 6.0).rem_euclid(1.0);
    let s = if mx > 1e-12 { d / mx.max(1e-12) } else { 0.0 };
    (h, s, mx)
}

fn hsv_to_rgb(h: f64, s: f64, v: f64) -> [f64; 3] {
    let i = ((h * 6.0).floor() as i64).rem_euclid(6);
    let f = h * 6.0 - (h * 6.0).floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - f * s);
    let t = v * (1.0 - (1.0 - f) * s);
    match i {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HslParams {
    pub hue: f64,
    pub sat: f64,
    pub lum: f64,
    pub ranges: [(f64, f64); 6], // (décalage de teinte, facteur de saturation) R, Y, G, C, B, M
}

impl Default for HslParams {
    fn default() -> Self {
        HslParams { hue: 0.0, sat: 1.0, lum: 0.0, ranges: [(0.0, 1.0); 6] }
    }
}

pub fn hsl(img: &Image, p: &HslParams) -> Result<Image> {
    check(img)?;
    let hue = p.hue.clamp(-180.0, 180.0);
    let sat = p.sat.clamp(0.0, 2.0);
    let lum = p.lum.clamp(-1.0, 1.0);
    let centers = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0];
    let neutral = hue.abs() < 1e-9 && (sat - 1.0).abs() < 1e-9 && lum.abs() < 1e-9 && p.ranges.iter().all(|&(dh, ds)| dh.abs() < 1e-9 && (ds - 1.0).abs() < 1e-9);
    if neutral {
        return Ok(out(img.clone()));
    }
    if img.c == 1 {
        return Ok(if lum != 0.0 { out(img.add_scalar(lum as f32)) } else { out(img.clone()) });
    }
    let active: Vec<(f64, f64, f64)> = centers
        .iter()
        .zip(&p.ranges)
        .filter(|(_, &(dh, ds))| !(dh.abs() < 1e-9 && (ds - 1.0).abs() < 1e-9))
        .map(|(&c, &(dh, ds))| (c, dh, ds))
        .collect();
    let mut o = img.clone();
    for px in o.data.chunks_mut(3) {
        let (mut h, mut s, mut v) = rgb_to_hsv(px);
        let mut ws: Vec<(f64, f64, f64)> = Vec::new();
        for &(center, dh, ds) in &active {
            let deg = h * 360.0;
            let mut d = (deg - center).abs();
            d = d.min(360.0 - d);
            let w = (1.0 - d / 75.0).clamp(0.0, 1.0);
            ws.push((w * w * (3.0 - 2.0 * w), dh, ds));
        }
        if hue != 0.0 {
            h = (h + hue / 360.0).rem_euclid(1.0);
        }
        if !ws.is_empty() {
            let wsum = ws.iter().map(|x| x.0).sum::<f64>().max(1e-6);
            let shift: f64 = ws.iter().map(|x| x.0 * x.1).sum();
            h = (h + (shift / wsum) / 360.0).rem_euclid(1.0);
            let factor = 1.0 + ws.iter().map(|x| x.0 * (x.2 - 1.0)).sum::<f64>() / wsum;
            s = (s * factor).clamp(0.0, 1.0);
        }
        s = (s * sat).clamp(0.0, 1.0);
        v = (v + lum).clamp(0.0, 1.0);
        let rgb = hsv_to_rgb(h, s, v);
        for k in 0..3 {
            px[k] = rgb[k] as f32;
        }
    }
    Ok(out(o))
}

pub fn local_contrast(img: &Image, radius: f64, gain: f64) -> Result<Image> {
    check(img)?;
    let g = gain.clamp(0.0, 2.0) as f32;
    if g <= 0.0 {
        return Ok(out(img.clone()));
    }
    let blur = ndimage::gaussian_img(img, radius.clamp(4.0, 200.0));
    Ok(out(img.zip_map(&blur, |a, b| a + g * (a - b))))
}

#[allow(clippy::too_many_arguments)]
pub fn split_tone(img: &Image, sh: [f64; 3], hl: [f64; 3], balance: f64, force: f64) -> Result<Image> {
    check(img)?;
    let f = force.clamp(0.0, 1.0) as f32;
    if f <= 0.0 {
        return Ok(out(img.clone()));
    }
    let bal = balance.clamp(-1.0, 1.0) as f32;
    let csh: Vec<f32> = sh.iter().map(|v| v.clamp(0.0, 1.0) as f32).collect();
    let chl: Vec<f32> = hl.iter().map(|v| v.clamp(0.0, 1.0) as f32).collect();
    if img.c == 1 {
        let c_sh = 0.2126 * csh[0] + 0.7152 * csh[1] + 0.0722 * csh[2];
        let c_hl = 0.2126 * chl[0] + 0.7152 * chl[1] + 0.0722 * chl[2];
        let t = ndimage::gaussian(img, 8.0).map(|v| (v + bal).clamp(0.0, 1.0));
        return Ok(out(img.zip_map(&t, |a, t| {
            let (ash, ahl) = (f * (1.0 - t), f * t);
            a * (1.0 - ash - ahl) + c_sh * ash + c_hl * ahl
        })));
    }
    let lum = luminance(img);
    let t = ndimage::gaussian(&lum, 8.0).map(|v| (v + bal).clamp(0.0, 1.0));
    let mut o = img.clone();
    for i in 0..t.data.len() {
        let (ash, ahl) = (f * (1.0 - t.data[i]), f * t.data[i]);
        for k in 0..3 {
            let j = i * 3 + k;
            o.data[j] = img.data[j] * (1.0 - ash - ahl) + csh[k] * ash + chl[k] * ahl;
        }
    }
    Ok(out(o))
}

pub fn gradient_map(img: &Image, stops: [[f64; 3]; 3], intensity: f64) -> Result<Image> {
    check(img)?;
    let it = intensity.clamp(0.0, 1.0) as f32;
    if it <= 0.0 {
        return Ok(out(img.clone()));
    }
    let lum = luminance(img).clip01();
    let xs = [0.0, 0.5, 1.0];
    let mut o = img.clone();
    if img.c == 3 {
        for i in 0..lum.data.len() {
            let l = lum.data[i];
            for k in 0..3 {
                let ys = [stops[0][k], stops[1][k], stops[2][k]];
                let m = stats::interp(l as f64, &xs, &ys) as f32 + (img.data[i * 3 + k] - l);
                o.data[i * 3 + k] = img.data[i * 3 + k] * (1.0 - it) + m * it;
            }
        }
    } else {
        let gray: Vec<f64> = stops.iter().map(|s| 0.2126 * s[0] + 0.7152 * s[1] + 0.0722 * s[2]).collect();
        for i in 0..lum.data.len() {
            let m = stats::interp(lum.data[i] as f64, &xs, &gray) as f32;
            o.data[i] = img.data[i] * (1.0 - it) + m * it;
        }
    }
    Ok(out(o))
}

// ------------------------------------------------------------------ registre
type Spec = (&'static str, f64); // (clé, défaut)

/// Filtres : (id, libellé, opération de séquence, paramètres par défaut).
pub fn filters() -> Vec<(&'static str, &'static str, &'static str, Vec<Spec>)> {
    vec![
        ("saturation", "Saturation", "fin_saturation", vec![("amount", 1.0)]),
        ("white_balance", "Balance des blancs (R/V/B)", "fin_white_balance", vec![("wr", 1.0), ("wg", 1.0), ("wb", 1.0)]),
        ("background_neutral", "Neutralisation du fond", "fin_background_neutral", vec![("percentile", 5.0), ("strength", 1.0)]),
        ("scnr_green", "Suppression du vert (SCNR)", "fin_scnr_green", vec![("strength", 1.0)]),
        ("autostretch", "Étirement automatique (médiane/MAD, tons moyens)", "fin_autostretch", vec![("shadows_clip", -2.8), ("target_bg", 0.25)]),
        ("remove_gradient", "Suppression de gradient (extraction de fond)", "fin_gradient", vec![("kernel", 64.0)]),
        ("denoise_median", "Débruitage médian (répété)", "fin_median_denoise", vec![("radius", 1.0), ("iterations", 1.0)]),
        ("unsharp_mask", "Masque flou (netteté)", "fin_unsharp", vec![("radius", 2.0), ("gain", 0.6)]),
        ("levels", "Niveaux", "fin_levels", vec![("black", 0.0), ("gamma", 1.0), ("white", 1.0)]),
        ("shadows_highlights", "Tons clairs / tons sombres", "fin_shadows_highlights", vec![("shadows", 0.0), ("highlights", 0.0), ("radius", 24.0)]),
        ("vibrance", "Vibrance", "fin_vibrance", vec![("amount", 1.0)]),
        (
            "hsl",
            "Teinte / saturation / luminosité (HSL)",
            "fin_hsl",
            vec![
                ("hue", 0.0), ("sat", 1.0), ("lum", 0.0), ("r_hue", 0.0), ("r_sat", 1.0), ("y_hue", 0.0), ("y_sat", 1.0), ("g_hue", 0.0), ("g_sat", 1.0), ("c_hue", 0.0),
                ("c_sat", 1.0), ("b_hue", 0.0), ("b_sat", 1.0), ("m_hue", 0.0), ("m_sat", 1.0),
            ],
        ),
        ("local_contrast", "Contraste local (clarté, esprit LHE)", "fin_local_contrast", vec![("radius", 32.0), ("gain", 0.0)]),
        (
            "split_tone",
            "Tonalité divisée",
            "fin_split_tone",
            vec![("sh_r", 0.5), ("sh_g", 0.5), ("sh_b", 0.5), ("hl_r", 0.5), ("hl_g", 0.5), ("hl_b", 0.5), ("balance", 0.0), ("force", 0.0)],
        ),
        (
            "gradient_map",
            "Mappage de dégradé",
            "fin_gradient_map",
            vec![("r0", 0.0), ("g0", 0.0), ("b0", 0.0), ("r1", 0.5), ("g1", 0.5), ("b1", 0.5), ("r2", 1.0), ("g2", 1.0), ("b2", 1.0), ("intensity", 0.0)],
        ),
        ("ghs", "Étirement hyperbolique généralisé (GHS)", "fin_ghs", vec![("D", 0.0), ("b", 0.0), ("SP", 0.0), ("LP", 0.0), ("HP", 1.0)]),
        ("arcsinh", "Étirement arcsinh (couleurs préservées)", "fin_arcsinh", vec![("stretch", 0.0), ("black", 0.0)]),
        ("star_reduction", "Réduction d'étoiles", "fin_star_reduction", vec![("amount", 0.0), ("iterations", 1.0)]),
        ("chroma_denoise", "Débruitage de la chrominance (luminance intacte)", "fin_chroma_denoise", vec![("strength", 0.0), ("scales", 4.0)]),
    ]
}

/// Paramètres effectifs (défauts du filtre + valeurs fournies).
pub struct P<'a> {
    pub params: &'a Map<String, Value>,
    pub defaults: Vec<Spec>,
}

impl P<'_> {
    pub fn f(&self, k: &str) -> f64 {
        match self.params.get(k) {
            Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0),
            Some(Value::Bool(b)) => *b as i64 as f64,
            Some(Value::String(s)) => s.parse().unwrap_or_else(|_| self.default(k)),
            _ => self.default(k),
        }
    }
    fn default(&self, k: &str) -> f64 {
        self.defaults.iter().find(|d| d.0 == k).map(|d| d.1).unwrap_or(0.0)
    }
    pub fn s(&self, k: &str, def: &str) -> String {
        self.params.get(k).and_then(|v| v.as_str()).unwrap_or(def).to_string()
    }
}

/// Applique le filtre `filter_id` (« none » : copie).
pub fn apply_filter(img: &Image, filter_id: &str, params: &Map<String, Value>) -> Result<Image> {
    if filter_id.is_empty() || filter_id == "none" {
        return Ok(img.clone());
    }
    let Some((_, _, _, defaults)) = filters().into_iter().find(|f| f.0 == filter_id) else { bail!("Filtre de finition inconnu : {}", filter_id) };
    let p = P { params, defaults };
    match filter_id {
        "saturation" => saturation(img, p.f("amount")),
        "white_balance" => white_balance(img, p.f("wr"), p.f("wg"), p.f("wb")),
        "background_neutral" => neutralize_background(img, p.f("percentile"), p.f("strength")),
        "scnr_green" => scnr_green(img, p.f("strength")),
        "autostretch" => autostretch(img, p.f("shadows_clip"), p.f("target_bg")),
        "remove_gradient" => remove_gradient(img, p.f("kernel"), &p.s("method", "Soustraire")),
        "denoise_median" => denoise_median(img, p.f("radius") as i64, p.f("iterations") as i64),
        "unsharp_mask" => unsharp_mask(img, p.f("radius"), p.f("gain")),
        "levels" => levels(img, p.f("black"), p.f("gamma"), p.f("white")),
        "shadows_highlights" => shadows_highlights(img, p.f("shadows"), p.f("highlights"), p.f("radius")),
        "vibrance" => vibrance(img, p.f("amount")),
        "hsl" => {
            let mut hp = HslParams { hue: p.f("hue"), sat: p.f("sat"), lum: p.f("lum"), ..Default::default() };
            for (i, pre) in ["r", "y", "g", "c", "b", "m"].iter().enumerate() {
                hp.ranges[i] = (p.f(&format!("{}_hue", pre)), p.f(&format!("{}_sat", pre)));
            }
            hsl(img, &hp)
        }
        "local_contrast" => local_contrast(img, p.f("radius"), p.f("gain")),
        "split_tone" => split_tone(img, [p.f("sh_r"), p.f("sh_g"), p.f("sh_b")], [p.f("hl_r"), p.f("hl_g"), p.f("hl_b")], p.f("balance"), p.f("force")),
        "gradient_map" => gradient_map(
            img,
            [[p.f("r0"), p.f("g0"), p.f("b0")], [p.f("r1"), p.f("g1"), p.f("b1")], [p.f("r2"), p.f("g2"), p.f("b2")]],
            p.f("intensity"),
        ),
        "ghs" => {
            check(img)?;
            let mode = if p.s("method", "Canaux").starts_with("Lum") { "Luminance" } else { "Canaux" };
            Ok(out(stretch::ghs(img, p.f("D"), p.f("b"), p.f("SP"), p.f("LP"), p.f("HP"), mode)))
        }
        "arcsinh" => {
            check(img)?;
            Ok(out(stretch::arcsinh_stretch(img, p.f("stretch"), p.f("black"))))
        }
        "star_reduction" => {
            check(img)?;
            Ok(out(masks::reduce_stars(img, p.f("amount"), p.f("iterations") as usize, 4.0)))
        }
        "chroma_denoise" => {
            check(img)?;
            Ok(out(masks::chroma_denoise(img, p.f("strength"), p.f("scales") as usize)))
        }
        _ => bail!("Filtre de finition inconnu : {}", filter_id),
    }
}

/// Filtre appliqué à travers un masque {kind, low, high, fuzz, blur}.
pub fn apply_filter_masked(img: &Image, filter_id: &str, params: &Map<String, Value>, mask: &Map<String, Value>) -> Result<Image> {
    let o = apply_filter(img, filter_id, params)?;
    let kind = mask.get("kind").and_then(|v| v.as_str()).unwrap_or("Aucun");
    if kind == "Aucun" || filter_id.is_empty() || filter_id == "none" {
        return Ok(o);
    }
    let g = |k: &str, d: f64| mask.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
    let m = masks::make_mask(img, kind, g("low", 0.25), g("high", 1.0), g("fuzz", 0.1), g("blur", 2.0))?;
    Ok(out(masks::apply_masked(img, &o, m.as_ref())?))
}
