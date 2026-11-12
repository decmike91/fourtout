//! Ondelettes à trous (spline B3), contrôles comparables à Registax6 :
//! gains, débruitage et accentuation par couche.

use crate::image::Image;
use crate::ndimage::{correlate_taps, Border};
use crate::stats;
use anyhow::{bail, Result};

const B3: [f64; 9] = [1.0 / 256.0, 8.0 / 256.0, 28.0 / 256.0, 56.0 / 256.0, 70.0 / 256.0, 56.0 / 256.0, 28.0 / 256.0, 8.0 / 256.0, 1.0 / 256.0];

fn taps(level: usize) -> Vec<(isize, f64)> {
    let step = if level == 0 { 1 } else { 1isize << level };
    (0..9).map(|k| ((k as isize - 4) * step, B3[k])).collect()
}

/// Décompose en [détail1, …, détailN, résidu] (canal par canal).
pub fn atrous_decompose(img: &Image, levels: usize) -> Vec<Image> {
    let mut layers = Vec::with_capacity(levels + 1);
    let mut current = img.clone();
    for lvl in 0..levels {
        let t = taps(lvl);
        let smooth = current.map_planes(|p| {
            let a = correlate_taps(p, &t, 0, Border::Reflect);
            correlate_taps(&a, &t, 1, Border::Reflect)
        });
        layers.push(current.zip_map(&smooth, |a, b| a - b));
        current = smooth;
    }
    layers.push(current);
    layers
}

/// Recompose avec gain par couche et seuil doux optionnel (détails seuls).
pub fn atrous_recompose(layers: &[Image], gains: Option<&[f64]>, thresholds: Option<&[f64]>) -> Image {
    let n = layers.len();
    let mut out = vec![0.0f64; layers[0].data.len()];
    for i in 0..n {
        let g = gains.and_then(|g| g.get(i)).cloned().unwrap_or(1.0);
        let t = thresholds.and_then(|t| t.get(i)).cloned().unwrap_or(0.0);
        for (o, &v) in out.iter_mut().zip(&layers[i].data) {
            let mut l = v as f64;
            if t != 0.0 && i < n - 1 {
                l = l.signum() * (l.abs() - t).max(0.0);
            }
            *o += l * g;
        }
    }
    Image::from_vec(layers[0].w, layers[0].h, layers[0].c, out.iter().map(|&v| v as f32).collect())
}

fn soft(v: f32, t: f64) -> f32 {
    (v.signum() as f64 * ((v.abs() as f64) - t).max(0.0)) as f32
}

/// Seuillage doux (seuil universel) des couches de détail.
pub fn wavelet_denoise(img: &Image, levels: usize, strength: f64) -> Image {
    let layers = atrous_decompose(img, levels);
    let n = layers.len();
    let mut new = Vec::with_capacity(n);
    for l in &layers[..n - 1] {
        let mut a: Vec<f32> = l.data.iter().map(|v| v.abs()).collect();
        let sigma = stats::median_mut(&mut a) as f64 / 0.6745 + 1e-9;
        let t = sigma * (2.0 * (l.data.len() as f64).ln()).sqrt() * strength;
        new.push(l.map(|v| soft(v, t)));
    }
    new.push(layers[n - 1].clone());
    atrous_recompose(&new, None, None)
}

/// Seuil doux MAD par couche (0..1) puis amplification (1 + netteté 0..3).
pub fn enhance_layers(layers: &[Image], denoise: Option<&[f64]>, sharpen: Option<&[f64]>) -> Result<Vec<Image>> {
    let dn: Vec<f64> = denoise.map(|d| d.to_vec()).unwrap_or_else(|| vec![0.0; 6]);
    let sh: Vec<f64> = sharpen.map(|d| d.to_vec()).unwrap_or_else(|| vec![0.0; 6]);
    if dn.len() != 6 || sh.len() != 6 || !dn.iter().chain(&sh).all(|v| v.is_finite()) {
        bail!("Six valeurs finies requises par contrôle d'ondelettes.");
    }
    let mut out = Vec::new();
    let n = layers.len();
    for (i, layer) in layers[..n - 1].iter().enumerate() {
        let mut detail = layer.clone();
        let strength = dn.get(i).cloned().unwrap_or(0.0).clamp(0.0, 1.0);
        if strength != 0.0 {
            // statistiques par canal (axes spatiaux)
            let c = detail.c;
            let npx = detail.w * detail.h;
            for k in 0..c {
                let ch: Vec<f32> = detail.data.iter().skip(k).step_by(c).cloned().collect();
                let med = stats::median(&ch);
                let sigma = stats::mad_about(&ch, med) as f64 / 0.6745;
                let thr = sigma * (2.0 * (npx.max(2) as f64).ln()).sqrt() * strength;
                for i2 in 0..npx {
                    let v = detail.data[i2 * c + k];
                    detail.data[i2 * c + k] = soft(v, thr);
                }
            }
        }
        let amount = sh.get(i).cloned().unwrap_or(0.0).clamp(0.0, 3.0);
        if amount != 0.0 {
            detail = detail.scale((1.0 + amount) as f32);
        }
        out.push(detail);
    }
    out.push(layers[n - 1].clone());
    Ok(out)
}

/// Six couches indépendantes + compatibilité des réglages historiques.
pub fn registax_enhance(img: &Image, gains: Option<&[f64]>, denoise_strength: f64, layer_denoise: Option<&[f64]>, layer_sharpen: Option<&[f64]>) -> Result<Image> {
    let mut a = img.clone();
    if denoise_strength > 0.0 {
        a = wavelet_denoise(&a, 4, denoise_strength.clamp(0.0, 1.0));
    }
    let mut layers = atrous_decompose(&a, 6);
    if layer_denoise.is_some() || layer_sharpen.is_some() {
        layers = enhance_layers(&layers, layer_denoise, layer_sharpen)?;
    }
    let g: Vec<f64> = match gains {
        Some(gs) => {
            let mut v = gs.to_vec();
            v.resize(7.max(v.len()), 1.0);
            v.truncate(7);
            v
        }
        None => vec![1.0; 7],
    };
    Ok(atrous_recompose(&layers, Some(&g), None))
}
