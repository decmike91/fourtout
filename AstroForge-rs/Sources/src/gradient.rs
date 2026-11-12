//! Extraction du fond de ciel / gradients : polynôme 2D ou interpolation
//! RBF (plaque mince) sur une grille d'échantillons (esprit ABE/DBE).

use crate::geom::{self, Interp};
use crate::image::Image;
use crate::stats;
use anyhow::{bail, Result};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

pub const METHODS: [&str; 2] = ["Polynôme 2D", "RBF (thin plate)"];

/// Échantillons du fond : médiane par case avec rejet sigma (luminance).
/// Retourne (points (x, y), valeurs par canal).
pub fn background_samples(img: &Image, grid: usize, sigma: f64, min_frac: f64) -> Result<(Vec<(f64, f64)>, Vec<Vec<f64>>)> {
    let mono = img.mean_lum();
    let (w, h) = (mono.w, mono.h);
    let g = grid.max(3);
    let ey: Vec<usize> = stats::linspace(0.0, h as f64, g + 1).iter().map(|&v| v as usize).collect();
    let ex: Vec<usize> = stats::linspace(0.0, w as f64, g + 1).iter().map(|&v| v as usize).collect();
    let c = img.c;
    let cells: Vec<Option<((f64, f64), Vec<f64>)>> = (0..g * g)
        .into_par_iter()
        .map(|k| {
            let (i, j) = (k / g, k % g);
            let mut idx = Vec::new();
            for y in ey[i]..ey[i + 1] {
                for x in ex[j]..ex[j + 1] {
                    idx.push(y * w + x);
                }
            }
            if idx.is_empty() {
                return None;
            }
            let block: Vec<f32> = idx.iter().map(|&p| mono.data[p]).collect();
            let med = stats::median(&block);
            let mad = 1.4826 * stats::mad_about(&block, med) as f64 + 1e-9;
            let ok: Vec<usize> = (0..idx.len()).filter(|&q| ((block[q] - med).abs() as f64) <= sigma * mad).collect();
            if (ok.len() as f64) / (idx.len() as f64) < min_frac {
                return None;
            }
            let pt = ((ex[j] + ex[j + 1]) as f64 / 2.0, (ey[i] + ey[i + 1]) as f64 / 2.0);
            let vals: Vec<f64> = (0..c)
                .map(|ch| {
                    let v: Vec<f32> = ok.iter().map(|&q| img.data[idx[q] * c + ch]).collect();
                    stats::median(&v) as f64
                })
                .collect();
            Some((pt, vals))
        })
        .collect();
    let (pts, vals): (Vec<(f64, f64)>, Vec<Vec<f64>>) = cells.into_iter().flatten().unzip();
    if pts.len() < 6 {
        bail!("Pas assez d'échantillons de fond — réduisez la grille ou sigma");
    }
    Ok((pts, vals))
}

fn fit_poly(pts: &[(f64, f64)], vals: &[f64], degree: usize, w: usize, h: usize) -> Image {
    let terms: Vec<(usize, usize)> = (0..=degree).flat_map(|dy| (0..=(degree - dy)).map(move |dx| (dx, dy))).collect();
    let n = pts.len();
    let mut a = DMatrix::<f64>::zeros(n, terms.len());
    for (i, &(x, y)) in pts.iter().enumerate() {
        let (xn, yn) = (x / w as f64, y / h as f64);
        for (k, &(dx, dy)) in terms.iter().enumerate() {
            a[(i, k)] = xn.powi(dx as i32) * yn.powi(dy as i32);
        }
    }
    let b = DVector::from_vec(vals.to_vec());
    let coef = a.svd(true, true).solve(&b, 1e-12).unwrap_or_else(|_| DVector::zeros(terms.len()));
    Image::from_fn(w, h, |x, y| {
        let (xn, yn) = (x as f64 / w as f64, y as f64 / h as f64);
        terms.iter().enumerate().map(|(k, &(dx, dy))| coef[k] * xn.powi(dx as i32) * yn.powi(dy as i32)).sum::<f64>() as f32
    })
}

fn tps(r: f64) -> f64 {
    if r <= 0.0 {
        0.0
    } else {
        r * r * r.ln()
    }
}

/// RBF plaque mince (lissage 1, polynôme de degré 1), évaluée sur une grille
/// réduite en coordonnées pixels puis agrandie.
fn fit_rbf(pts: &[(f64, f64)], vals: &[f64], w: usize, h: usize, smooth: f64) -> Image {
    let n = pts.len();
    let m = n + 3;
    let mut a = DMatrix::<f64>::zeros(m, m);
    for i in 0..n {
        for j in 0..n {
            let r = (pts[i].0 - pts[j].0).hypot(pts[i].1 - pts[j].1);
            a[(i, j)] = tps(r) + if i == j { smooth } else { 0.0 };
        }
        a[(i, n)] = 1.0;
        a[(i, n + 1)] = pts[i].0;
        a[(i, n + 2)] = pts[i].1;
        a[(n, i)] = 1.0;
        a[(n + 1, i)] = pts[i].0;
        a[(n + 2, i)] = pts[i].1;
    }
    let mut b = DVector::<f64>::zeros(m);
    for i in 0..n {
        b[i] = vals[i];
    }
    let sol = a.lu().solve(&b).unwrap_or_else(|| DVector::zeros(m));
    let step = 1usize.max(((h * w) as f64 / 4e6).sqrt() as usize);
    let (sw, sh) = (w.div_ceil(step), h.div_ceil(step));
    let small = Image::from_fn(sw, sh, |x, y| {
        let (px, py) = ((x * step) as f64, (y * step) as f64);
        let mut v = sol[n] + sol[n + 1] * px + sol[n + 2] * py;
        for i in 0..n {
            v += sol[i] * tps((px - pts[i].0).hypot(py - pts[i].1));
        }
        v as f32
    });
    geom::resize(&small, w, h, Interp::Cubic)
}

/// Retire le fond : (image corrigée, modèle de fond). Couleur : un modèle
/// par canal. Correction « Soustraction » (piédestal neutre) ou « Division ».
pub fn extract_gradient(img: &Image, method: &str, degree: usize, grid: usize, sigma: f64, correction: &str) -> Result<(Image, Image)> {
    let (w, h, c) = (img.w, img.h, img.c);
    let (pts, vals) = background_samples(img, grid, sigma, 0.25)?;
    let degree = degree.clamp(1, 5);
    let planes: Vec<Image> = (0..c)
        .into_par_iter()
        .map(|ch| {
            let v: Vec<f64> = vals.iter().map(|x| x[ch]).collect();
            if method.starts_with("Polynôme") {
                fit_poly(&pts, &v, degree, w, h)
            } else {
                fit_rbf(&pts, &v, w, h, 1.0)
            }
        })
        .collect();
    let model = Image::from_planes(&planes).clip_min(0.0);
    let level: Vec<f32> = (0..c).map(|k| stats::median(&model.channel(k).data)).collect();
    let out = if correction.starts_with("Division") {
        let mut o = img.clone();
        for i in 0..w * h {
            for k in 0..c {
                let j = i * c + k;
                o.data[j] = img.data[j] / model.data[j].max(1e-6) * level[k];
            }
        }
        o
    } else {
        let ped = level.iter().sum::<f32>() / c as f32;
        img.zip_map(&model, |a, m| a - m + ped)
    };
    Ok((out.clip_min(0.0), model))
}
