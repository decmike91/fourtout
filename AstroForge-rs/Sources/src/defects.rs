//! Pixels/colonnes/lignes défectueux : carte fournie (FITS/PNG : pixels
//! > 0,5 ; texte : une ligne « x,y » par pixel) ou détection automatique sur
//! master dark/flat, correction par interpolation locale.

use crate::image::Image;
use crate::io;
use crate::ndimage;
use crate::stats;
use anyhow::{bail, Result};
use std::path::Path;

/// Masque booléen (largeur w, hauteur h).
#[derive(Clone, Debug)]
pub struct Mask {
    pub w: usize,
    pub h: usize,
    pub data: Vec<bool>,
}

impl Mask {
    pub fn count(&self) -> usize {
        self.data.iter().filter(|&&b| b).count()
    }
    pub fn any(&self) -> bool {
        self.data.iter().any(|&b| b)
    }
}

/// Charge une carte de défauts (redimensionnée par recadrage/complément à `shape`).
pub fn load_bad_pixel_map(path: &Path, shape: Option<(usize, usize)>) -> Result<Mask> {
    let ext = io::ext_of(path);
    let mut mask = if ["txt", "csv", "dat"].contains(&ext.as_str()) {
        let txt = std::fs::read_to_string(path)?;
        let mut pts: Vec<(i64, i64)> = Vec::new();
        for line in txt.lines() {
            let s = line.trim();
            if s.is_empty() || s.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = s.split(|c: char| c == ',' || c == ';' || c.is_whitespace()).filter(|t| !t.is_empty()).collect();
            if parts.len() >= 2 {
                if let (Ok(x), Ok(y)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) {
                    pts.push((x as i64, y as i64));
                }
            }
        }
        if pts.is_empty() {
            bail!("Carte de pixels vide : {}", path.display());
        }
        let (w, h) = match shape {
            Some((w, h)) => (w, h),
            None => ((pts.iter().map(|p| p.0).max().unwrap() + 2) as usize, (pts.iter().map(|p| p.1).max().unwrap() + 2) as usize),
        };
        let mut m = Mask { w, h, data: vec![false; w * h] };
        for (x, y) in pts {
            if x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h {
                m.data[y as usize * w + x as usize] = true;
            }
        }
        m
    } else {
        let img = io::load(path)?;
        let c = img.c;
        Mask { w: img.w, h: img.h, data: img.data.chunks(c).map(|p| p.iter().any(|&v| v > 0.5)).collect() }
    };
    if let Some((w, h)) = shape {
        if (mask.w, mask.h) != (w, h) {
            let mut out = Mask { w, h, data: vec![false; w * h] };
            for y in 0..h.min(mask.h) {
                for x in 0..w.min(mask.w) {
                    out.data[y * w + x] = mask.data[y * mask.w + x];
                }
            }
            mask = out;
        }
    }
    Ok(mask)
}

/// Pixels à plus de `sigma` MAD de la médiane globale (chauds + froids).
pub fn detect_bad_pixels(reference: &Image, sigma: f64) -> Mask {
    let a = reference.mean_lum();
    let med = stats::median(&a.data) as f64;
    let mad = 1.4826 * stats::mad_about(&a.data, med as f32) as f64 + 1e-9;
    let data = a
        .data
        .iter()
        .map(|&v| {
            let v = v as f64;
            v > med + sigma * mad || (med > 1e-6 && v < med - sigma * mad)
        })
        .collect();
    Mask { w: a.w, h: a.h, data }
}

fn profile_outliers(prof: &[f32], sigma: f64) -> Vec<bool> {
    let med = stats::median(prof) as f64;
    let mad = 1.4826 * stats::mad_about(prof, med as f32) as f64 + 1e-9;
    prof.iter().map(|&p| ((p as f64) - med).abs() > sigma * mad).collect()
}

pub fn detect_bad_columns(reference: &Image, sigma: f64) -> Vec<bool> {
    let a = reference.mean_lum();
    let prof: Vec<f32> = (0..a.w).map(|x| stats::median(&(0..a.h).map(|y| a.data[y * a.w + x]).collect::<Vec<_>>())).collect();
    profile_outliers(&prof, sigma)
}

pub fn detect_bad_rows(reference: &Image, sigma: f64) -> Vec<bool> {
    let a = reference.mean_lum();
    let prof: Vec<f32> = (0..a.h).map(|y| stats::median(&a.data[y * a.w..(y + 1) * a.w])).collect();
    profile_outliers(&prof, sigma)
}

/// Corrige colonnes/lignes (moyenne des voisines) et pixels (médiane des
/// voisins, `passes` passes) ; en CFA, voisins de même couleur (pas de 2).
pub fn fix_defects(img: &Image, bad: Option<&Mask>, cols: Option<&[bool]>, rows: Option<&[bool]>, passes: usize, cfa: bool) -> Image {
    let mut a = img.clone();
    let (w, h, ch) = (a.w, a.h, a.c);
    let step = if cfa && ch == 1 { 2isize } else { 1 };
    if let Some(cols) = cols {
        for (c, &b) in cols.iter().enumerate() {
            if !b || c >= w {
                continue;
            }
            let (mut l, mut r) = (c as isize - step, c as isize + step);
            if l < 0 {
                l = r;
            }
            if r >= w as isize {
                r = l;
            }
            if l >= 0 && (l as usize) < w && r >= 0 && (r as usize) < w {
                for y in 0..h {
                    for k in 0..ch {
                        let v = (a.get(l as usize, y, k) + a.get(r as usize, y, k)) * 0.5;
                        a.set(c, y, k, v);
                    }
                }
            }
        }
    }
    if let Some(rows) = rows {
        for (r0, &b) in rows.iter().enumerate() {
            if !b || r0 >= h {
                continue;
            }
            let (mut u, mut d) = (r0 as isize - step, r0 as isize + step);
            if u < 0 {
                u = d;
            }
            if d >= h as isize {
                d = u;
            }
            if u >= 0 && (u as usize) < h && d >= 0 && (d as usize) < h {
                for x in 0..w {
                    for k in 0..ch {
                        let v = (a.get(x, u as usize, k) + a.get(x, d as usize, k)) * 0.5;
                        a.set(x, r0, k, v);
                    }
                }
            }
        }
    }
    if let Some(m) = bad {
        if m.any() && m.w == w && m.h == h {
            if step == 2 {
                for (i, j) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
                    let mut sub = a.sub2(i, j);
                    let msub: Vec<bool> = (0..sub.h).flat_map(|y| (0..sub.w).map(move |x| (y, x))).map(|(y, x)| m.data[(2 * y + i) * w + 2 * x + j]).collect();
                    if !msub.iter().any(|&b| b) {
                        continue;
                    }
                    for _ in 0..passes.max(1) {
                        let med = ndimage::median_filter(&sub, 3);
                        for (k, &b) in msub.iter().enumerate() {
                            if b {
                                sub.data[k] = med.data[k];
                            }
                        }
                    }
                    a.put_sub2(i, j, &sub);
                }
            } else {
                for _ in 0..passes.max(1) {
                    let med = a.map_planes(|p| ndimage::median_filter(p, 5));
                    for (pix, &b) in m.data.iter().enumerate() {
                        if b {
                            for k in 0..ch {
                                a.data[pix * ch + k] = med.data[pix * ch + k];
                            }
                        }
                    }
                }
            }
        }
    }
    a
}

#[derive(Clone, Debug, Default)]
pub struct DefectInfo {
    pub bad_pixels: usize,
    pub bad_columns: usize,
    pub bad_rows: usize,
}

/// Tout-en-un : carte depuis `bpm` sinon détection sur dark/flat.
pub fn auto_correct(img: &Image, master_dark: Option<&Image>, master_flat: Option<&Image>, bpm: Option<&Path>, sigma: f64, fix_cols_rows: bool) -> Result<(Image, DefectInfo)> {
    let reference = master_dark.or(master_flat);
    let mut bad = match bpm {
        Some(p) => Some(load_bad_pixel_map(p, Some((img.w, img.h)))?),
        None => None,
    };
    let (mut cols, mut rows) = (None, None);
    if bad.is_none() {
        if let Some(r) = reference {
            bad = Some(detect_bad_pixels(r, sigma));
            if fix_cols_rows {
                cols = Some(detect_bad_columns(r, sigma));
                rows = Some(detect_bad_rows(r, sigma));
            }
        }
    }
    let out = fix_defects(img, bad.as_ref(), cols.as_deref(), rows.as_deref(), 2, false);
    let info = DefectInfo {
        bad_pixels: bad.as_ref().map(|b| b.count()).unwrap_or(0),
        bad_columns: cols.as_ref().map(|c| c.iter().filter(|&&b| b).count()).unwrap_or(0),
        bad_rows: rows.as_ref().map(|c| c.iter().filter(|&&b| b).count()).unwrap_or(0),
    };
    Ok((out, info))
}
