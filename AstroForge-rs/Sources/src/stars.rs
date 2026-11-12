//! Détection d'étoiles : portage de l'algorithme DAOFIND tel
//! qu'implémenté par photutils (DAOStarFinder ≥ 3) — noyau gaussien
//! « abaissé » (somme nulle, empreinte r ≤ max(1,5σ ; 2)), seuil × erreur
//! relative du noyau, maxima dans un cercle de rayon 2,5·FWHM, filtres de
//! piqué (0,2..1) et de rondeur (−1..1), centroïde par ajustement de
//! gaussiennes marginales, tri par flux décroissant.

use crate::image::Image;
use crate::stats;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct Star {
    pub x: f64,
    pub y: f64,
    pub flux: f64,
    pub peak: f64,
    pub sharpness: f64,
    pub roundness1: f64,
    pub roundness2: f64,
}

struct Kernel {
    r: usize,         // demi-largeur (noyau (2r+1)²)
    n: usize,         // côté
    mask: Vec<bool>,  // empreinte
    data: Vec<f64>,   // noyau normalisé (somme nulle sur l'empreinte)
    gauss: Vec<f64>,  // gaussienne non masquée (pic = 1)
    sigma: f64,
    rel_err: f64,
    npix: usize,
}

fn dao_kernel(fwhm: f64) -> Kernel {
    let sigma = fwhm / 2.354_820_045_030_949_4;
    let a = 1.0 / (2.0 * sigma * sigma);
    let f = 1.5f64.powi(2) / 2.0;
    let denom = a * a;
    let r = (2.0f64.max((a * f / denom).sqrt())) as usize;
    let n = 2 * r + 1;
    let mut mask = vec![false; n * n];
    let mut gauss = vec![0.0; n * n];
    for y in 0..n {
        for x in 0..n {
            let (dx, dy) = (x as f64 - r as f64, y as f64 - r as f64);
            let ell = a * dx * dx + a * dy * dy;
            let circ = (dx * dx + dy * dy).sqrt();
            mask[y * n + x] = ell <= f || circ <= 2.0;
            gauss[y * n + x] = (-ell).exp();
        }
    }
    let npix = mask.iter().filter(|&&m| m).count();
    let gk: Vec<f64> = (0..n * n).map(|i| if mask[i] { gauss[i] } else { 0.0 }).collect();
    let s1: f64 = gk.iter().sum();
    let s2: f64 = gk.iter().map(|v| v * v).sum();
    let dnm = s2 - s1 * s1 / npix as f64;
    let rel_err = 1.0 / dnm.sqrt();
    let data: Vec<f64> = (0..n * n).map(|i| if mask[i] { (gk[i] - s1 / npix as f64) / dnm } else { 0.0 }).collect();
    Kernel { r, n, mask, data, gauss, sigma, rel_err, npix }
}

/// Ajustement d'une gaussienne marginale (photutils _marginal_lstsq) :
/// (décalage, hauteur) ou None si l'amplitude n'est pas positive.
fn marginal_fit(k: &Kernel, cut: &[f64], axis_x: bool) -> Option<(f64, f64)> {
    let n = k.n;
    let c = k.r as f64;
    let wt: Vec<f64> = (0..n).map(|i| c - (i as f64 - c).abs() + 1.0).collect();
    // marginale : somme sur l'axe perpendiculaire pondérée par wt (2D)
    let mut kern1 = vec![0.0; n];
    let mut data1 = vec![0.0; n];
    for y in 0..n {
        for x in 0..n {
            let (i_par, i_perp) = if axis_x { (x, y) } else { (y, x) };
            let w2 = wt[i_perp];
            kern1[i_par] += k.gauss[y * n + x] * w2;
            data1[i_par] += cut[y * n + x] * w2;
        }
    }
    let dx_k: Vec<f64> = (0..n).map(|i| c - i as f64).collect();
    let dxx: Vec<f64> = if axis_x { dx_k.clone() } else { (0..n).map(|i| i as f64 - c).collect() };
    let wt_sum: f64 = wt.iter().sum();
    let kern_sum: f64 = (0..n).map(|i| kern1[i] * wt[i]).sum();
    let kern2_sum: f64 = (0..n).map(|i| kern1[i] * kern1[i] * wt[i]).sum();
    let dkern: Vec<f64> = (0..n).map(|i| kern1[i] * dx_k[i]).collect();
    let dkern_sum: f64 = (0..n).map(|i| dkern[i] * wt[i]).sum();
    let dkern2_sum: f64 = (0..n).map(|i| dkern[i] * dkern[i] * wt[i]).sum();
    let kern_dkern_sum: f64 = (0..n).map(|i| kern1[i] * dkern[i] * wt[i]).sum();
    let data_sum: f64 = (0..n).map(|i| data1[i] * wt[i]).sum();
    let data_kern_sum: f64 = (0..n).map(|i| data1[i] * kern1[i] * wt[i]).sum();
    let data_dkern_sum: f64 = (0..n).map(|i| data1[i] * dkern[i] * wt[i]).sum();
    let data_dx_sum: f64 = (0..n).map(|i| data1[i] * dxx[i] * wt[i]).sum();
    let hx_num = data_kern_sum - data_sum * kern_sum / wt_sum;
    let hx_den = kern2_sum - kern_sum * kern_sum / wt_sum;
    if hx_num <= 0.0 || hx_den <= 0.0 {
        return None;
    }
    let hx = hx_num / hx_den;
    let mut dx = (kern_dkern_sum - (data_dkern_sum - dkern_sum * data_sum)) / (hx * dkern2_sum / (k.sigma * k.sigma));
    let hsize = n as f64 / 2.0;
    if dx.abs() > hsize || !dx.is_finite() {
        dx = if data_sum == 0.0 { 0.0 } else { data_dx_sum / data_sum };
    }
    if dx.abs() > hsize || !dx.is_finite() {
        dx = 0.0;
    }
    Some((dx, hx))
}

/// Étoiles détectées (données = image − médiane, seuil absolu `threshold`).
pub fn dao_find(mono: &Image, threshold: f64, fwhm: f64) -> Vec<Star> {
    let (w, h) = (mono.w, mono.h);
    let k = dao_kernel(fwhm);
    let r = k.r as isize;
    let med = stats::median(&mono.data) as f64;
    let data: Vec<f64> = mono.data.iter().map(|&v| v as f64 - med).collect();
    let at = |x: isize, y: isize| -> f64 {
        if x < 0 || y < 0 || x >= w as isize || y >= h as isize {
            0.0
        } else {
            data[y as usize * w + x as usize]
        }
    };
    // convolution (bord nul)
    let taps: Vec<(isize, isize, f64)> = (0..k.n * k.n).filter(|&i| k.data[i] != 0.0).map(|i| ((i % k.n) as isize - r, (i / k.n) as isize - r, k.data[i])).collect();
    let mut conv = vec![0.0f64; w * h];
    conv.par_chunks_mut(w.max(1)).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let mut s = 0.0;
            for &(dx, dy, kv) in &taps {
                s += kv * at(x as isize + dx, y as isize + dy);
            }
            *o = s;
        }
    });
    let thr = threshold * k.rel_err;
    let rsep = 2.5 * fwhm;
    let ri = rsep.floor() as isize;
    let circle: Vec<(isize, isize)> = (-ri..=ri).flat_map(|dy| (-ri..=ri).map(move |dx| (dx, dy))).filter(|&(dx, dy)| (dx != 0 || dy != 0) && ((dx * dx + dy * dy) as f64) <= rsep * rsep).collect();
    let cv = |x: isize, y: isize| -> Option<f64> {
        if x < 0 || y < 0 || x >= w as isize || y >= h as isize {
            None
        } else {
            Some(conv[y as usize * w + x as usize])
        }
    };
    let peaks: Vec<(usize, usize)> = (0..h)
        .into_par_iter()
        .flat_map_iter(|y| {
            let mut out = Vec::new();
            for x in 0..w {
                let c = conv[y * w + x];
                if c <= thr {
                    continue;
                }
                let mut ok = true;
                for &(dx, dy) in &circle {
                    if let Some(v) = cv(x as isize + dx, y as isize + dy) {
                        if v > c || (v == c && (dy < 0 || (dy == 0 && dx < 0))) {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    out.push((x, y));
                }
            }
            out
        })
        .collect();
    let n = k.n;
    let center = k.r;
    let mut stars: Vec<Star> = peaks
        .par_iter()
        .filter_map(|&(px, py)| {
            let mut cut = vec![0.0f64; n * n];
            let mut ccut = vec![0.0f64; n * n];
            for yy in 0..n {
                for xx in 0..n {
                    let (sx, sy) = (px as isize + xx as isize - r, py as isize + yy as isize - r);
                    cut[yy * n + xx] = at(sx, sy);
                    ccut[yy * n + xx] = cv(sx, sy).unwrap_or(0.0);
                }
            }
            let peak = cut[center * n + center];
            let conv_peak = ccut[center * n + center];
            let masked_sum: f64 = (0..n * n).filter(|&i| k.mask[i]).map(|i| cut[i]).sum();
            let data_mean = (masked_sum - peak) / (k.npix as f64 - 1.0);
            let sharp = (peak - data_mean) / conv_peak;
            if !(0.2..=1.0).contains(&sharp) {
                return None;
            }
            // rondeur 1 (symétrie du noyau convolué)
            let mut cc = ccut.clone();
            cc[center * n + center] = 0.0;
            let q = |r0: usize, r1: usize, c0: usize, c1: usize| -> f64 {
                let mut s = 0.0;
                for yy in r0..r1 {
                    for xx in c0..c1 {
                        s += cc[yy * n + xx];
                    }
                }
                s
            };
            let q1 = q(0, center + 1, center + 1, n);
            let q2 = q(0, center, 0, center + 1);
            let q3 = q(center, n, 0, center);
            let q4 = q(center + 1, n, center, n);
            let sum2 = -q1 + q2 - q3 + q4;
            let sum4: f64 = cc.iter().map(|v| v.abs()).sum();
            let round1 = 2.0 * sum2 / sum4;
            if !(-1.0..=1.0).contains(&round1) || !round1.is_finite() {
                return None;
            }
            let (dx, hx) = marginal_fit(&k, &cut, true)?;
            let (dy, hy) = marginal_fit(&k, &cut, false)?;
            let round2 = 2.0 * (hx - hy) / (hx + hy);
            if !(-1.0..=1.0).contains(&round2) || !round2.is_finite() {
                return None;
            }
            Some(Star { x: px as f64 + dx, y: py as f64 + dy, flux: masked_sum, peak, sharpness: sharp, roundness1: round1, roundness2: round2 })
        })
        .collect();
    stars.sort_by(|a, b| b.flux.total_cmp(&a.flux));
    stars
}

/// Étoiles triées du plus brillant au plus faible (seuil en σ du MAD).
pub fn detect(img: &Image, n_max: usize, fwhm: f64, threshold_sigma: f64) -> Vec<Star> {
    let mono = img.mean_lum();
    if mono.w < 5 || mono.h < 5 {
        return vec![];
    }
    let med = stats::median(&mono.data);
    let mut absd: Vec<f32> = mono.data.iter().map(|&v| (v - med).abs()).collect();
    let mad = 1.4826 * stats::median_mut(&mut absd) as f64 + 1e-9;
    let mut s = dao_find(&mono, threshold_sigma * mad, fwhm);
    s.truncate(n_max);
    s
}

/// Positions (x, y) des étoiles (detect_stars du programme d'origine).
pub fn detect_stars(img: &Image, n_max: usize) -> Vec<(f64, f64)> {
    detect(img, n_max, 3.0, 8.0).iter().map(|s| (s.x, s.y)).collect()
}

pub fn detect_stars_opts(img: &Image, n_max: usize, fwhm: f64, threshold_sigma: f64) -> Vec<(f64, f64)> {
    detect(img, n_max, fwhm, threshold_sigma).iter().map(|s| (s.x, s.y)).collect()
}

#[cfg(test)]
pub mod tests {
    use super::*;
    /// Champ d'étoiles gaussiennes (x, y, amplitude).
    pub fn field(w: usize, h: usize, stars: &[(f64, f64, f64)], fwhm: f64, bg: f32) -> Image {
        let s = fwhm / 2.3548;
        Image::from_fn(w, h, |x, y| {
            let mut v = bg as f64;
            for &(sx, sy, a) in stars {
                v += a * (-((x as f64 - sx).powi(2) + (y as f64 - sy).powi(2)) / (2.0 * s * s)).exp();
            }
            v as f32
        })
    }
    #[test]
    fn finds_subpixel() {
        let img = field(64, 48, &[(20.3, 15.7, 0.8), (40.0, 30.2, 0.4)], 3.0, 0.05);
        let st = detect_stars(&img, 10);
        assert_eq!(st.len(), 2);
        assert!((st[0].0 - 20.3).abs() < 0.15 && (st[0].1 - 15.7).abs() < 0.15, "{:?}", st);
    }
    #[test]
    fn rejects_hot_pixel() {
        let mut img = field(64, 48, &[(20.0, 15.0, 0.5)], 3.0, 0.05);
        img.data[30 * 64 + 40] = 1.0;
        let st = detect_stars(&img, 10);
        assert_eq!(st.len(), 1, "{:?}", st);
    }
}
