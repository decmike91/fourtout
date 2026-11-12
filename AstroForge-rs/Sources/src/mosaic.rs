//! Mosaïques : panneaux reprojetés par astrométrie (WCS) sur une projection
//! tangente commune, ou recalés par appariement d'étoiles ; égalisation du
//! fond dans les recouvrements et fondu progressif.

use crate::geom::{self, Affine, Interp};
use crate::image::Image;
use crate::io;
use crate::ndimage::{self, Border};
use crate::starmatch;
use crate::stars;
use crate::stats;
use crate::wcs::{self, Wcs};
use anyhow::{bail, Result};
use std::path::Path;

fn feather(valid: &[bool], w: usize, h: usize, width: f64) -> Vec<f32> {
    let d = ndimage::distance_transform_edt(valid, w, h);
    d.iter().map(|&v| ((v as f64 / width.max(1.0)).clamp(0.0, 1.0) as f32).powf(1.5)).collect()
}

fn blend_into(acc: &mut Image, wacc: &mut [f32], img: &Image, valid: &[bool], match_bg: bool, fw: f64) {
    let (w, h, c) = (img.w, img.h, img.c);
    let wt = feather(valid, w, h, fw);
    let mut a = img.clone();
    if match_bg {
        let ov: Vec<usize> = (0..w * h).filter(|&i| valid[i] && wacc[i] > 1e-6).collect();
        if ov.len() > 100 {
            for k in 0..c {
                let cur: Vec<f32> = ov.iter().map(|&i| acc.data[i * c + k] / wacc[i].max(1e-9)).collect();
                let mine: Vec<f32> = ov.iter().map(|&i| a.data[i * c + k]).collect();
                let off = stats::median(&cur) - stats::median(&mine);
                for i in 0..w * h {
                    a.data[i * c + k] += off;
                }
            }
        }
    }
    for i in 0..w * h {
        for k in 0..c {
            acc.data[i * c + k] += a.data[i * c + k] * wt[i];
        }
        wacc[i] += wt[i];
    }
}

fn finish(acc: &Image, wacc: &[f32]) -> (Image, Vec<bool>) {
    let c = acc.c;
    let mut out = acc.clone();
    for i in 0..wacc.len() {
        for k in 0..c {
            out.data[i * c + k] = (acc.data[i * c + k] / wacc[i].max(1e-9)).clamp(0.0, 1.0);
        }
    }
    (out, wacc.iter().map(|&v| v > 1e-6).collect())
}

/// WCS céleste de l'en-tête d'un FITS, ou None.
pub fn wcs_from_fits(path: &Path) -> Option<Wcs> {
    if !io::has_ext(path, &io::FITS_EXTENSIONS) {
        return None;
    }
    let h = io::fits::read_header(path).ok()?;
    Wcs::from_header(&h)
}

/// Projection TAN commune couvrant tous les panneaux : (WCS, (largeur, hauteur)).
pub fn output_wcs(wcs_list: &[Wcs], shapes: &[(usize, usize)], scale_deg: Option<f64>) -> Result<(Wcs, (usize, usize))> {
    let mut v = [0.0f64; 3];
    for (w, &(wd, h)) in wcs_list.iter().zip(shapes) {
        let (ra, dec) = w.pixel_to_world_linear(wd as f64 / 2.0, h as f64 / 2.0);
        let (r, d) = (ra.to_radians(), dec.to_radians());
        v[0] += d.cos() * r.cos();
        v[1] += d.cos() * r.sin();
        v[2] += d.sin();
    }
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let ra0 = v[1].atan2(v[0]).to_degrees().rem_euclid(360.0);
    let dec0 = (v[2] / n).clamp(-1.0, 1.0).asin().to_degrees();
    let scale = scale_deg.unwrap_or_else(|| wcs_list.iter().map(|w| (w.pixel_scales()[0] + w.pixel_scales()[1]) / 2.0).fold(f64::INFINITY, f64::min));
    let mut out = Wcs::new_tan([1.0, 1.0], [ra0, dec0], [[-scale, 0.0], [0.0, scale]]);
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for (w, &(wd, h)) in wcs_list.iter().zip(shapes) {
        let (wd, h) = (wd as f64, h as f64);
        for (x, y) in [(0.0, 0.0), (wd - 1.0, 0.0), (0.0, h - 1.0), (wd - 1.0, h - 1.0), (wd / 2.0, 0.0), (wd / 2.0, h - 1.0), (0.0, h / 2.0), (wd - 1.0, h / 2.0)] {
            let (ra, dec) = w.pixel_to_world_linear(x, y);
            if let Some(p) = out.world_to_pixel_linear(ra, dec) {
                xs.push(p.0);
                ys.push(p.1);
            }
        }
    }
    let x0 = xs.iter().cloned().fold(f64::INFINITY, f64::min).floor();
    let y0 = ys.iter().cloned().fold(f64::INFINITY, f64::min).floor();
    let width = (xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - x0).ceil() as usize + 2;
    let height = (ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - y0).ceil() as usize + 2;
    out.crpix = [1.0 - x0 + 1.0, 1.0 - y0 + 1.0];
    if (width * height) as f64 > 400e6 {
        bail!("Mosaïque trop grande ({}×{}) — vérifiez les WCS", width, height);
    }
    Ok((out, (width, height)))
}

/// Reprojection bicubique sur la grille de sortie : (image, validité).
pub fn reproject(img: &Image, w_in: &Wcs, w_out: &Wcs, shape: (usize, usize)) -> (Image, Vec<bool>) {
    let (w, h) = shape;
    let maps: Vec<(f64, f64)> = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            let (ra, dec) = w_out.pixel_to_world_linear(x, y);
            w_in.world_to_pixel_linear(ra, dec).unwrap_or((f64::NAN, f64::NAN))
        })
        .collect();
    let valid: Vec<bool> = maps.iter().map(|&(x, y)| x >= 0.0 && y >= 0.0 && x <= (img.w - 1) as f64 && y <= (img.h - 1) as f64).collect();
    let out = geom::remap_fn(img, w, h, Interp::Cubic, Border::Constant, |x, y| {
        let m = maps[y * w + x];
        if m.0.is_finite() { m } else { (-1e9, -1e9) }
    });
    (out, valid)
}

/// Mosaïque par astrométrie : (image, WCS de sortie, couverture).
pub fn mosaic_wcs(images: &[Image], wcs_list: &[Wcs], scale_deg: Option<f64>, fw: f64) -> Result<(Image, Wcs, Vec<bool>)> {
    if images.len() != wcs_list.len() || images.is_empty() {
        bail!("Un WCS par panneau est requis");
    }
    let shapes: Vec<(usize, usize)> = images.iter().map(|i| (i.w, i.h)).collect();
    let (wout, shape) = output_wcs(wcs_list, &shapes, scale_deg)?;
    let c = images[0].c;
    let mut acc = Image::new(shape.0, shape.1, c);
    let mut wacc = vec![0.0f32; shape.0 * shape.1];
    for (i, (img, w)) in images.iter().zip(wcs_list).enumerate() {
        let (rep, valid) = reproject(img, w, &wout, shape);
        blend_into(&mut acc, &mut wacc, &rep, &valid, i > 0, fw);
    }
    let (out, cov) = finish(&acc, &wacc);
    Ok((out, wout, cov))
}

/// Similitude envoyant img_b sur img_a par appariement d'étoiles (recouvrement partiel).
pub fn match_similarity(a: &Image, b: &Image, min_matches: usize) -> Option<Affine> {
    // fond retiré (gradients, nébulosité) avant détection linéaire
    let hp = |i: &Image| {
        let l = i.mean_lum();
        let bg = ndimage::gaussian(&l, 10.0);
        l.zip_map(&bg, |v, b| v - b)
    };
    let sa = stars::detect_stars(&hp(a), 150);
    let sb = stars::detect_stars(&hp(b), 150);
    let m = starmatch::match_points(&sb, &sa, false, 2.0, None, true)?;
    if m.n_inliers < min_matches.min(8).max(4) {
        return None;
    }
    Some(m.sim.to_affine())
}

/// Mosaïque sans astrométrie (champs étoilés) : chaque panneau est recalé
/// sur un panneau déjà placé. Retourne (image, couverture, matrices).
pub fn mosaic_features(images: &[Image], fw: f64) -> Result<(Image, Vec<bool>, Vec<Affine>)> {
    if images.is_empty() {
        bail!("Aucun panneau");
    }
    let n = images.len();
    let mut mats: Vec<Option<Affine>> = vec![None; n];
    mats[0] = Some(geom::IDENTITY);
    let mut placed = vec![0usize];
    let mut pending: Vec<usize> = (1..n).collect();
    while !pending.is_empty() {
        let mut progress = false;
        for j in pending.clone() {
            for &i in &placed.clone() {
                let Some(m) = match_similarity(&images[i], &images[j], 12) else { continue };
                mats[j] = Some(geom::compose(&mats[i].unwrap(), &m));
                placed.push(j);
                pending.retain(|&p| p != j);
                progress = true;
                break;
            }
        }
        if !progress {
            bail!("Panneau(x) sans recouvrement exploitable : {}", pending.iter().map(|p| (p + 1).to_string()).collect::<Vec<_>>().join(", "));
        }
    }
    let mats: Vec<Affine> = mats.into_iter().map(|m| m.unwrap()).collect();
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for (img, m) in images.iter().zip(&mats) {
        for (x, y) in [(0.0, 0.0), (img.w as f64, 0.0), (0.0, img.h as f64), (img.w as f64, img.h as f64)] {
            let p = geom::apply(m, x, y);
            xs.push(p.0);
            ys.push(p.1);
        }
    }
    let x0 = xs.iter().cloned().fold(f64::INFINITY, f64::min).floor();
    let y0 = ys.iter().cloned().fold(f64::INFINITY, f64::min).floor();
    let wd = (xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - x0).ceil() as usize;
    let ht = (ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - y0).ceil() as usize;
    if (wd * ht) as f64 > 400e6 {
        bail!("Mosaïque trop grande — recouvrements incohérents ?");
    }
    let c = images[0].c;
    let mut acc = Image::new(wd, ht, c);
    let mut wacc = vec![0.0f32; wd * ht];
    for (k, (img, m)) in images.iter().zip(&mats).enumerate() {
        let mut t = *m;
        t[0][2] -= x0;
        t[1][2] -= y0;
        let rep = geom::warp_affine(img, &t, wd, ht, Interp::Cubic, Border::Constant);
        let inv = geom::invert(&t);
        let valid: Vec<bool> = (0..wd * ht)
            .map(|i| {
                let (sx, sy) = geom::apply(&inv, (i % wd) as f64, (i / wd) as f64);
                sx >= 0.0 && sy >= 0.0 && sx <= (img.w - 1) as f64 && sy <= (img.h - 1) as f64
            })
            .collect();
        let valid = ndimage::binary_erosion(&valid, wd, ht, 2);
        blend_into(&mut acc, &mut wacc, &rep, &valid, k > 0, fw);
    }
    let (out, cov) = finish(&acc, &wacc);
    Ok((out, cov, mats))
}

pub fn separation(a: (f64, f64), b: (f64, f64)) -> f64 {
    wcs::separation(a.0, a.1, b.0, b.1)
}
