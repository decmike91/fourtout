//! Restauration : débruitage (non-local means, bilatéral, médiane,
//! ondelettes), déconvolution de Richardson-Lucy, masque flou.

use crate::image::Image;
use crate::ndimage::{self, correlate_taps, fold, Border};
use crate::stats;
use crate::wavelets;
use rayon::prelude::*;

pub const DENOISE_METHODS: [&str; 4] = ["Non-local means", "Bilatéral", "Médiane", "Ondelettes"];

/// Noyau gaussien 1D normalisé de la PSF (taille max(3, int(6σ)|1)).
pub fn gaussian_psf_1d(sigma: f64) -> Vec<(isize, f64)> {
    let size = 3usize.max((6.0 * sigma) as usize | 1);
    let c = (size as f64 - 1.0) / 2.0;
    let mut g: Vec<f64> = (0..size).map(|i| (-((i as f64 - c).powi(2)) / (2.0 * sigma * sigma)).exp()).collect();
    let s: f64 = g.iter().sum();
    g.iter_mut().for_each(|v| *v /= s);
    let half = (size / 2) as isize;
    g.into_iter().enumerate().map(|(i, v)| (i as isize - half, v)).collect()
}

fn conv_same(p: &Image, k: &[(isize, f64)]) -> Image {
    // convolution « same » à bord nul (fftconvolve), noyau séparable symétrique
    let a = correlate_taps(p, k, 0, Border::Constant);
    correlate_taps(&a, k, 1, Border::Constant)
}

fn rl_plane(a: &Image, k: &[(isize, f64)], iterations: usize, damping: f64) -> Image {
    let mut est = a.clone();
    for _ in 0..iterations.max(1) {
        let conv = conv_same(&est, k);
        let rel = a.zip_map(&conv, |x, c| x / c.max(1e-7));
        let corr = conv_same(&rel, k);
        let mut new = est.zip_map(&corr, |e, c| e * c);
        if damping != 0.0 {
            let d = damping as f32;
            new = new.zip_map(&est, |n, e| (1.0 - d) * n + d * e);
        }
        est = new.clip_min(0.0);
    }
    let k = (a.sum() / est.sum().max(1e-9)) as f32;
    est.scale(k)
}

/// Richardson-Lucy, PSF gaussienne, amortissement, canal par canal.
pub fn richardson_lucy(img: &Image, sigma_psf: f64, iterations: usize, damping: f64) -> Image {
    let a = img.clip_min(0.0);
    let k = gaussian_psf_1d(sigma_psf);
    a.map_planes(|p| rl_plane(p, &k, iterations, damping))
}

/// Écart-type du bruit (MAD du résidu passe-haut).
pub fn noise_sigma(img: &Image) -> f64 {
    let mono = img.mean_lum();
    let med = ndimage::median_filter(&mono, 3);
    let hp: Vec<f32> = mono.data.iter().zip(&med.data).map(|(a, b)| a - b).collect();
    let m = stats::median(&hp);
    1.4826 * stats::mad_about(&hp, m) as f64 / 0.85 + 1e-6
}

/// Masque flou classique.
pub fn unsharp(img: &Image, amount: f64, radius: f64) -> Image {
    let blur = ndimage::gaussian_img(img, radius);
    let a = amount as f32;
    let out = img.zip_map(&blur, |x, b| x + a * (x - b));
    if img.max_value() <= 1.5 {
        out.clip01()
    } else {
        out
    }
}

/// Filtre bilatéral (d = 9, comme cv2.bilateralFilter), distance de
/// couleur L1 en RGB.
pub fn bilateral(img: &Image, sigma_color: f64, sigma_space: f64) -> Image {
    let r = 4isize;
    let (w, h, c) = (img.w, img.h, img.c);
    let gc = -0.5 / (sigma_color * sigma_color).max(1e-30);
    let gs = -0.5 / (sigma_space * sigma_space);
    let offs: Vec<(isize, isize, f64)> = (-r..=r)
        .flat_map(|dy| (-r..=r).map(move |dx| (dx, dy)))
        .filter(|&(dx, dy)| ((dx * dx + dy * dy) as f64).sqrt() <= r as f64)
        .map(|(dx, dy)| (dx, dy, ((dx * dx + dy * dy) as f64 * gs).exp()))
        .collect();
    let mut out = Image::new(w, h, c);
    out.data.par_chunks_mut(w * c).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let p0 = img.pixel(x, y);
            let mut acc = [0.0f64; 3];
            let mut wt = 0.0f64;
            for &(dx, dy, ws) in &offs {
                let xx = fold(x as isize + dx, w, Border::Mirror).unwrap();
                let yy = fold(y as isize + dy, h, Border::Mirror).unwrap();
                let q = img.pixel(xx, yy);
                let d: f64 = (0..c).map(|k| (q[k] - p0[k]).abs() as f64).sum();
                let wv = ws * (d * d * gc).exp();
                wt += wv;
                for k in 0..c {
                    acc[k] += wv * q[k] as f64;
                }
            }
            for k in 0..c {
                row[x * c + k] = (acc[k] / wt) as f32;
            }
        }
    });
    out
}

/// Non-local means (fenêtres 7×7 / 21×21, norme L1, poids exp(−d²/h²)),
/// implémentation par images intégrales des distances.
pub fn nl_means(plane: &Image, h: f64) -> Image {
    let (w, ht) = (plane.w, plane.h);
    let tr = 3isize; // gabarit 7×7
    let sr = 10isize; // recherche 21×21
    let pad = tr + sr;
    let pw = w + 2 * pad as usize;
    let ph = ht + 2 * pad as usize;
    let mut p = vec![0.0f64; pw * ph];
    for y in 0..ph {
        for x in 0..pw {
            let sx = fold(x as isize - pad, w, Border::Mirror).unwrap();
            let sy = fold(y as isize - pad, ht, Border::Mirror).unwrap();
            p[y * pw + x] = plane.data[sy * w + sx] as f64;
        }
    }
    let h2 = (h * h).max(1e-12);
    let area = ((2 * tr + 1) * (2 * tr + 1)) as f64;
    let band = 32usize;
    let bands: Vec<(usize, Vec<f32>)> = (0..ht.div_ceil(band))
        .into_par_iter()
        .map(|b| {
            let (y0, y1) = (b * band, ((b + 1) * band).min(ht));
            let bh = y1 - y0;
            let mut num = vec![0.0f64; bh * w];
            let mut den = vec![0.0f64; bh * w];
            // lignes de travail : y0 - tr .. y1 + tr (coordonnées image)
            let rows = bh + 2 * tr as usize;
            let cols = w + 2 * tr as usize;
            let mut diff = vec![0.0f64; rows * cols];
            let mut integ = vec![0.0f64; (rows + 1) * (cols + 1)];
            for dy in -sr..=sr {
                for dx in -sr..=sr {
                    for r in 0..rows {
                        let py = (y0 as isize + r as isize - tr + pad) as usize;
                        for cc in 0..cols {
                            let px = (cc as isize - tr + pad) as usize;
                            let a = p[py * pw + px];
                            let q = p[((py as isize + dy) as usize) * pw + (px as isize + dx) as usize];
                            diff[r * cols + cc] = (a - q).abs();
                        }
                    }
                    for r in 0..rows {
                        let mut s = 0.0;
                        for cc in 0..cols {
                            s += diff[r * cols + cc];
                            integ[(r + 1) * (cols + 1) + cc + 1] = integ[r * (cols + 1) + cc + 1] + s;
                        }
                    }
                    let k = (2 * tr + 1) as usize;
                    for yy in 0..bh {
                        for xx in 0..w {
                            let (r0, c0) = (yy, xx);
                            let s = integ[(r0 + k) * (cols + 1) + c0 + k] - integ[r0 * (cols + 1) + c0 + k] - integ[(r0 + k) * (cols + 1) + c0] + integ[r0 * (cols + 1) + c0];
                            let d = s / area;
                            let wv = (-d * d / h2).exp();
                            let i = yy * w + xx;
                            let py = (y0 + yy) as isize + pad + dy;
                            let px = xx as isize + pad + dx;
                            num[i] += wv * p[py as usize * pw + px as usize];
                            den[i] += wv;
                        }
                    }
                }
            }
            let out: Vec<f32> = (0..bh * w).map(|i| (num[i] / den[i].max(1e-300)) as f32).collect();
            (b, out)
        })
        .collect();
    let mut out = Image::new(w, ht, 1);
    for (b, v) in bands {
        let y0 = b * band;
        out.data[y0 * w..y0 * w + v.len()].copy_from_slice(&v);
    }
    out
}

/// Débruitage classique en flottant ; force ∈ [0, 1] relative au bruit mesuré.
pub fn denoise(img: &Image, method: &str, strength: f64, median_size: usize) -> Image {
    let s = strength.clamp(0.0, 1.0);
    match method {
        "Médiane" => {
            let k = median_size | 1;
            img.map_planes(|p| ndimage::median_filter(p, k))
        }
        "Ondelettes" => img.map_planes(|p| wavelets::wavelet_denoise(p, 4, s)),
        "Bilatéral" => {
            let sn = noise_sigma(img);
            bilateral(img, sn * (1.0 + 5.0 * s), 3.0 + 4.0 * s)
        }
        _ => {
            let sn = noise_sigma(img);
            let hi = (img.max_value() as f64).max(1e-6);
            let scale = 65535.0 / hi;
            let h = (sn * scale * (0.6 + 2.4 * s)).clamp(1.0, 65535.0);
            img.map_planes(|p| {
                // quantification 16 bits comme le programme d'origine
                let q = p.map(|v| ((v as f64 * scale).clamp(0.0, 65535.0)).floor() as f32);
                nl_means(&q, h).map(|v| (v as f64 / scale) as f32)
            })
        }
    }
}
