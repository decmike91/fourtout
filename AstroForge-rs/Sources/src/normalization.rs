//! Normalisation avant empilement : globale (médiane/MAD) ou locale (fond
//! par tuiles, étoiles rejetées, cartes lissées, facteur de transparence).

use crate::geom::{self, Interp};
use crate::image::Image;
use crate::ndimage::{self, Border};
use crate::stats;
use anyhow::{bail, Result};
use rayon::prelude::*;

pub const NORMALIZATIONS: [&str; 3] = ["Aucune", "Globale", "Locale"];

fn robust(a: &[f32]) -> (f64, f64) {
    let (m, s) = stats::robust(a);
    (m, s + 1e-9)
}

pub fn global_normalize(img: &Image, ref_: &Image) -> Image {
    if img.c > 1 {
        let planes: Vec<Image> = (0..img.c).map(|c| global_normalize(&img.channel(c), &ref_.channel(c))).collect();
        return Image::from_planes(&planes);
    }
    let (ma, sa) = robust(&img.data);
    let (mr, sr) = robust(&ref_.data);
    let k = (sr / sa) as f32;
    img.map(|v| (v - ma as f32) * k + mr as f32)
}

fn tile_stats(a: &Image, tile: usize, sigma: f64) -> (usize, usize, Vec<f64>) {
    let (w, h) = (a.w, a.h);
    let ny = (h / tile).max(1);
    let nx = (w / tile).max(1);
    let ys: Vec<usize> = stats::linspace(0.0, h as f64, ny + 1).iter().map(|&v| v as usize).collect();
    let xs: Vec<usize> = stats::linspace(0.0, w as f64, nx + 1).iter().map(|&v| v as usize).collect();
    let med: Vec<f64> = (0..ny * nx)
        .into_par_iter()
        .map(|k| {
            let (i, j) = (k / nx, k % nx);
            let mut blk = Vec::with_capacity((ys[i + 1] - ys[i]) * (xs[j + 1] - xs[j]));
            for y in ys[i]..ys[i + 1] {
                blk.extend_from_slice(&a.data[y * w + xs[j]..y * w + xs[j + 1]]);
            }
            let (m, s) = robust(&blk);
            let ok: Vec<f32> = blk.iter().cloned().filter(|&v| ((v as f64) - m).abs() <= sigma * s).collect();
            if ok.len() as f64 > 0.3 * blk.len() as f64 {
                robust(&ok).0
            } else {
                f64::NAN
            }
        })
        .collect();
    (nx, ny, med)
}

fn fill_smooth(nx: usize, ny: usize, grid: &[f64], w: usize, h: usize) -> Image {
    let mut g = grid.to_vec();
    if g.iter().all(|v| v.is_nan()) {
        g.iter_mut().for_each(|v| *v = 0.0);
    } else if g.iter().any(|v| v.is_nan()) {
        let mask: Vec<bool> = g.iter().map(|v| v.is_nan()).collect();
        let idx = ndimage::nearest_false_indices(&mask, nx, ny);
        g = idx.iter().map(|&i| grid[i]).collect();
    }
    let mut img = Image::from_vec(nx, ny, 1, g.iter().map(|&v| v as f32).collect());
    if nx.min(ny) >= 3 {
        img = ndimage::median_filter_mode(&img, 3, Border::Nearest);
    }
    geom::resize(&img, w, h, Interp::Cubic)
}

/// Fond lissé : médianes par tuiles (étoiles rejetées), interpolées.
pub fn background_map(a: &Image, tile: usize) -> Image {
    let (nx, ny, med) = tile_stats(a, tile, 2.5);
    fill_smooth(nx, ny, &med, a.w, a.h)
}

/// Rapport d'intensité du signal entre deux images sans fond.
pub fn signal_scale(det: &Image, det_ref: &Image) -> f64 {
    let (_, n_ref) = stats::robust(&det_ref.data);
    let (_, n_det) = stats::robust(&det.data);
    let (n_ref, n_det) = (n_ref + 1e-9, n_det + 1e-9);
    let ratios: Vec<f64> = det
        .data
        .iter()
        .zip(&det_ref.data)
        .filter(|(&d, &r)| (r as f64) > 8.0 * n_ref && (d as f64) > 8.0 * n_det)
        .map(|(&d, &r)| r as f64 / d as f64)
        .collect();
    if ratios.len() >= 50 {
        return stats::median_f64(&ratios).clamp(0.05, 20.0);
    }
    (n_ref / n_det).clamp(0.05, 20.0)
}

/// out = s·(img − fond_img) + fond_ref.
pub fn local_normalize(img: &Image, ref_: &Image, tile: usize) -> Result<Image> {
    if !img.same_shape(ref_) {
        bail!("Normalisation locale : images de même taille requises");
    }
    if img.c > 1 {
        let planes: Result<Vec<Image>> = (0..img.c).map(|c| local_normalize(&img.channel(c), &ref_.channel(c), tile)).collect();
        return Ok(Image::from_planes(&planes?));
    }
    let tile = 16usize.max(tile.min(img.w.min(img.h) / 2));
    let (bg_a, bg_r) = rayon::join(|| background_map(img, tile), || background_map(ref_, tile));
    let da = img.zip_map(&bg_a, |a, b| a - b);
    let dr = ref_.zip_map(&bg_r, |a, b| a - b);
    let k = signal_scale(&da, &dr) as f32;
    Ok(da.zip_map(&bg_r, |d, b| k * d + b))
}

pub fn normalize_frames(frames: Vec<Image>, ref_: &Image, mode: &str, tile: usize) -> Result<Vec<Image>> {
    match mode {
        "Globale" => Ok(frames.par_iter().map(|f| global_normalize(f, ref_)).collect()),
        "Locale" => frames.par_iter().map(|f| local_normalize(f, ref_, tile)).collect(),
        _ => Ok(frames),
    }
}
