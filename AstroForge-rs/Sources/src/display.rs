//! Réglages d'affichage : étirement automatique (STF), histogramme.

use crate::image::Image;
use crate::stats;

/// Fonction de transfert des tons moyens : MTF(m, 0) = 0, MTF(m, m) = 0,5, MTF(m, 1) = 1.
#[inline]
pub fn mtf(m: f64, x: f32) -> f32 {
    let m = m.clamp(1e-6, 1.0 - 1e-6) as f32;
    ((m - 1.0) * x) / ((2.0 * m - 1.0) * x - m)
}

/// (point noir, tons moyens m) d'un étirement automatique type STF.
pub fn stf_params(img: &Image, shadows_clip: f64, target_bg: f64) -> (f64, f64) {
    let lum = img.mean_lum();
    let sy = (lum.h / 512).max(1);
    let sx = (lum.w / 512).max(1);
    let mut sample = Vec::with_capacity((lum.h / sy + 1) * (lum.w / sx + 1));
    for y in (0..lum.h).step_by(sy) {
        for x in (0..lum.w).step_by(sx) {
            sample.push(lum.data[y * lum.w + x]);
        }
    }
    let med = stats::median(&sample) as f64;
    let sig = 1.4826 * stats::mad_about(&sample, med as f32) as f64;
    let black = (med + shadows_clip * sig).clamp(0.0, (med - 1e-6).max(0.0));
    let x0 = (med - black) / (1.0 - black).max(1e-6);
    let t = target_bg.clamp(0.01, 0.99);
    if x0 <= 1e-7 {
        return (black, 0.5);
    }
    let m = x0 * (1.0 - t) / (x0 - 2.0 * t * x0 + t);
    (black, m.clamp(1e-6, 1.0 - 1e-6))
}

/// Étirement automatique non linéaire (point noir + MTF), fond à ~25 %.
pub fn autostretch(img: &Image) -> Image {
    autostretch_params(img, -2.8, 0.25)
}

pub fn autostretch_params(img: &Image, shadows_clip: f64, target_bg: f64) -> Image {
    let (black, m) = stf_params(img, shadows_clip, target_bg);
    let den = (1.0 - black).max(1e-6) as f32;
    let b = black as f32;
    img.map(|v| {
        let x = ((v - b) / den).clamp(0.0, 1.0);
        mtf(m, x).clamp(0.0, 1.0)
    })
}

/// Histogramme de la luminance moyenne sur [0, 1].
pub fn histogram(img: &Image, bins: usize) -> Vec<u64> {
    let lum = img.mean_lum();
    let mut h = vec![0u64; bins];
    for &v in &lum.data {
        let v = v.clamp(0.0, 1.0);
        let i = ((v * bins as f32) as usize).min(bins - 1);
        h[i] += 1;
    }
    h
}

/// Étirement linéaire d'affichage (image_io.stretch_display).
pub fn stretch_display(img: &Image, black: f32, gamma: f32, max_val: Option<f32>) -> Image {
    let mv = max_val.unwrap_or_else(|| {
        let med = stats::median(&img.data);
        let mv = if med > 0.0 { med + 3.0 * stats::std(&img.data) as f32 } else { img.max_value() };
        mv.max(1e-6)
    });
    img.map(|v| {
        let a = ((v - black) / (mv - black)).clamp(0.0, 1.0);
        if gamma != 1.0 {
            a.powf(1.0 / gamma)
        } else {
            a
        }
    })
}
