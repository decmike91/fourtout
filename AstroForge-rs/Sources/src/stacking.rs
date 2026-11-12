//! Empilement : moyenne, médiane, sigma-clipping, winsorisé, rejet min/max,
//! linear fit, MAD clipping (pondérés), classement de qualité (planétaire)
//! et empilement « basé AutoStakkert! ». Calcul pixel par pixel, en
//! parallèle sur tous les cœurs.

use crate::geom::{self, Interp};
use crate::image::Image;
use crate::ndimage::{self, Border};
use crate::stats;
use anyhow::{bail, Result};
use rayon::prelude::*;

pub const METHODS: [&str; 7] = ["Moyenne", "Médiane", "Sigma-clipping", "Winsorisé", "Rejet min/max", "Linear fit", "MAD clipping"];

#[derive(Clone, Copy, Debug)]
pub struct StackParams {
    pub sigma: f64,
    pub iters: usize,
    pub low: f64,
    pub high: f64,
}

impl Default for StackParams {
    fn default() -> Self {
        StackParams { sigma: 3.0, iters: 3, low: 1.0, high: 1.0 }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Method {
    Mean,
    Median,
    Sigma,
    Winsor,
    MinMax,
    LinearFit,
    Mad,
}

fn parse_method(m: &str) -> Result<Method> {
    Ok(match m {
        "Moyenne" => Method::Mean,
        "Médiane" => Method::Median,
        "Sigma-clipping" => Method::Sigma,
        "Winsorisé" => Method::Winsor,
        "Rejet min/max" => Method::MinMax,
        "Linear fit" => Method::LinearFit,
        "MAD clipping" => Method::Mad,
        _ => bail!("Méthode d'empilement inconnue : {}", m),
    })
}

struct Scratch {
    a: Vec<f32>,
    tmp: Vec<f32>,
    mask: Vec<bool>,
    wz: Vec<f32>,
}

impl Scratch {
    fn new(n: usize) -> Self {
        Scratch { a: vec![0.0; n], tmp: vec![0.0; n], mask: vec![false; n], wz: vec![0.0; n] }
    }
}

#[inline]
fn median_of(src: &[f32], tmp: &mut [f32]) -> f32 {
    tmp[..src.len()].copy_from_slice(src);
    stats::median_mut(&mut tmp[..src.len()])
}

#[inline]
fn mad_of(src: &[f32], med: f32, tmp: &mut [f32]) -> f32 {
    for (t, &v) in tmp.iter_mut().zip(src) {
        *t = (v - med).abs();
    }
    stats::median_mut(&mut tmp[..src.len()])
}

/// Moyenne (pondérée) des valeurs retenues ; (valeur, poids total).
#[inline]
fn wmean(a: &[f32], mask: &[bool], w: Option<&[f64]>) -> (f64, f64) {
    let (mut s, mut t) = (0.0f64, 0.0f64);
    match w {
        None => {
            for (i, &v) in a.iter().enumerate() {
                if mask[i] {
                    s += v as f64;
                    t += 1.0;
                }
            }
            (s / t.max(1.0), t)
        }
        Some(w) => {
            for (i, &v) in a.iter().enumerate() {
                if mask[i] {
                    s += v as f64 * w[i];
                    t += w[i];
                }
            }
            (s / t.max(1e-12), t)
        }
    }
}

fn px_sigma(sc: &mut Scratch, n: usize, sigma: f64, iters: usize, w: Option<&[f64]>) -> f32 {
    let a = &sc.a[..n];
    let med = median_of(a, &mut sc.tmp);
    let rsd = 1.4826 * mad_of(a, med, &mut sc.tmp) as f64;
    let thr0 = 4.0f64.max(sigma + 1.0) * rsd.max(1e-5);
    let mask = &mut sc.mask[..n];
    for i in 0..n {
        mask[i] = ((a[i] - med).abs() as f64) <= thr0;
    }
    for _ in 0..iters.max(1) {
        let mut cnt = 0usize;
        let mut s = 0.0f64;
        for i in 0..n {
            if mask[i] {
                cnt += 1;
                s += a[i] as f64;
            }
        }
        let c = s / (cnt.max(1)) as f64;
        let mut var = 0.0f64;
        for i in 0..n {
            if mask[i] {
                var += (a[i] as f64 - c).powi(2);
            }
        }
        var /= (cnt as f64 - 1.0).max(1.0);
        let sd = var.sqrt();
        let c = if cnt > 0 { c } else { med as f64 };
        let thr = sigma * sd.max(1e-5);
        let mut changed = false;
        for i in 0..n {
            let nm = (a[i] as f64 - c).abs() <= thr;
            if nm != mask[i] {
                changed = true;
                mask[i] = nm;
            }
        }
        if !changed {
            break;
        }
    }
    let (m, t) = wmean(a, mask, w);
    if t > 0.0 {
        m as f32
    } else {
        med
    }
}

fn px_winsor(sc: &mut Scratch, n: usize, sigma: f64, iters: usize, w: Option<&[f64]>) -> f32 {
    let a = &sc.a[..n];
    let mut med = median_of(a, &mut sc.tmp) as f64;
    let mut sd = (1.4826 * mad_of(a, med as f32, &mut sc.tmp) as f64).max(1e-6);
    for _ in 0..iters.max(1) * 3 {
        let (lo, hi) = (med - 1.5 * sd, med + 1.5 * sd);
        for i in 0..n {
            sc.wz[i] = (a[i] as f64).clamp(lo, hi) as f32;
        }
        let new = (1.134 * stats::std(&sc.wz[..n])).max(1e-6);
        med = median_of(&sc.wz[..n], &mut sc.tmp) as f64;
        let conv = stats::allclose(new, sd, 5e-4, 1e-7);
        sd = new;
        if conv {
            break;
        }
    }
    let (lo, hi) = (med - sigma * sd, med + sigma * sd);
    let (mut s, mut t) = (0.0f64, 0.0f64);
    for i in 0..n {
        let v = (a[i] as f64).clamp(lo, hi);
        let wi = w.map(|w| w[i]).unwrap_or(1.0);
        s += v * wi;
        t += wi;
    }
    (s / t.max(if w.is_some() { 1e-12 } else { 1.0 })) as f32
}

fn px_linear_fit(sc: &mut Scratch, n: usize, norm: &[f64], k: f64, w: Option<&[f64]>) -> f32 {
    let a = &sc.a[..n];
    let r = median_of(a, &mut sc.tmp);
    for i in 0..n {
        sc.wz[i] = (a[i] as f64 - norm[i] * r as f64) as f32;
    }
    let dmed = median_of(&sc.wz[..n], &mut sc.tmp);
    let sig = 1.4826 * mad_of(&sc.wz[..n], dmed, &mut sc.tmp) as f64 + 1e-6;
    let mask = &mut sc.mask[..n];
    for i in 0..n {
        mask[i] = (sc.wz[i].abs() as f64) <= k * sig;
    }
    let (m, t) = wmean(a, mask, w);
    if t > 0.0 {
        m as f32
    } else {
        r
    }
}

fn px_mad(sc: &mut Scratch, n: usize, k: f64, w: Option<&[f64]>) -> f32 {
    let a = &sc.a[..n];
    let med = median_of(a, &mut sc.tmp);
    let mad = 1.4826 * mad_of(a, med, &mut sc.tmp) as f64 + 1e-6;
    let mask = &mut sc.mask[..n];
    for i in 0..n {
        mask[i] = ((a[i] - med).abs() as f64) <= k * mad;
    }
    let (m, t) = wmean(a, mask, w);
    if t > 0.0 {
        m as f32
    } else {
        med
    }
}

fn px_minmax(sc: &mut Scratch, n: usize, low: f64, high: f64) -> f32 {
    let k_lo = (low / 100.0 * n as f64).floor() as usize;
    let k_hi = (high / 100.0 * n as f64).floor() as usize;
    let t = &mut sc.tmp[..n];
    t.copy_from_slice(&sc.a[..n]);
    t.sort_by(|a, b| a.total_cmp(b));
    let part: &[f32] = if k_lo + k_hi > 0 && k_lo + k_hi < n { &t[k_lo..n - k_hi] } else { t };
    (part.iter().map(|&v| v as f64).sum::<f64>() / part.len() as f64) as f32
}

/// Empile des images de même forme (N ≥ 1).
pub fn stack(frames: &[&Image], method: &str, p: &StackParams, weights: Option<&[f64]>) -> Result<Image> {
    let m = parse_method(method)?;
    if frames.is_empty() {
        bail!("Aucune image à empiler");
    }
    let first = frames[0];
    for f in frames {
        if !f.same_shape(first) {
            bail!("Taille différente : {} ≠ {}", f.shape_str(), first.shape_str());
        }
    }
    // poids : exclusion des poids nuls, poids égaux ignorés
    let mut sel: Vec<&Image> = frames.to_vec();
    let mut w: Option<Vec<f64>> = None;
    if let Some(ws) = weights {
        if ws.len() != frames.len() {
            bail!("{} poids pour {} images", ws.len(), frames.len());
        }
        if !ws.iter().any(|&x| x > 0.0) {
            bail!("Toutes les images ont un poids nul");
        }
        let keep: Vec<usize> = (0..ws.len()).filter(|&i| ws[i] > 0.0).collect();
        sel = keep.iter().map(|&i| frames[i]).collect();
        let wk: Vec<f64> = keep.iter().map(|&i| ws[i]).collect();
        let w0 = wk[0];
        if !wk.iter().all(|&x| stats::allclose(x, w0, 1e-5, 1e-8)) {
            w = Some(wk);
        }
    }
    let n = sel.len();
    let norm: Vec<f64> = if m == Method::LinearFit {
        let v: Vec<f64> = sel.par_iter().map(|f| stats::median(&f.data) as f64).collect();
        let mv = stats::median_f64(&v).max(1e-9);
        v.iter().map(|x| x / mv).collect()
    } else {
        vec![]
    };
    let len = first.data.len();
    let mut out = vec![0.0f32; len];
    let wr = w.as_deref();
    let chunk = 4096;
    out.par_chunks_mut(chunk).enumerate().for_each_init(
        || Scratch::new(n),
        |sc, (ci, dst)| {
            let base = ci * chunk;
            for (j, o) in dst.iter_mut().enumerate() {
                let i = base + j;
                for (k, f) in sel.iter().enumerate() {
                    sc.a[k] = f.data[i];
                }
                *o = match m {
                    Method::Mean => {
                        let (mut s, mut t) = (0.0f64, 0.0f64);
                        for k in 0..n {
                            let wk = wr.map(|w| w[k]).unwrap_or(1.0);
                            s += sc.a[k] as f64 * wk;
                            t += wk;
                        }
                        (s / t.max(if wr.is_some() { 1e-12 } else { 1.0 })) as f32
                    }
                    Method::Median => {
                        let (a, tmp) = (&sc.a[..n], &mut sc.tmp);
                        median_of(a, tmp)
                    }
                    Method::Sigma => px_sigma(sc, n, p.sigma, p.iters, wr),
                    Method::Winsor => px_winsor(sc, n, p.sigma, p.iters, wr),
                    Method::MinMax => px_minmax(sc, n, p.low, p.high),
                    Method::LinearFit => px_linear_fit(sc, n, &norm, p.sigma, wr),
                    Method::Mad => px_mad(sc, n, p.sigma, wr),
                };
            }
        },
    );
    Ok(Image::from_vec(first.w, first.h, first.c, out))
}

/// Variante pratique : images possédées, sigma seul.
pub fn stack_owned(frames: &[Image], method: &str, sigma: f64, weights: Option<&[f64]>) -> Result<Image> {
    let refs: Vec<&Image> = frames.iter().collect();
    stack(&refs, method, &StackParams { sigma, ..Default::default() }, weights)
}

/// Sigma-clipping par défaut (masters de calibration).
pub fn combine_sigma_clip(frames: &[&Image], sigma: f64) -> Result<Image> {
    stack(frames, "Sigma-clipping", &StackParams { sigma, ..Default::default() }, None)
}

/// Empilement depuis le disque (chaque image lue une fois).
pub fn stack_paths(paths: &[std::path::PathBuf], method: &str, p: &StackParams, bayer: Option<&str>) -> Result<Image> {
    if paths.is_empty() {
        bail!("Aucune image à empiler");
    }
    if method == "Moyenne" {
        let first = crate::io::load_image(&paths[0], bayer)?;
        let mut acc: Vec<f64> = first.data.iter().map(|&v| v as f64).collect();
        for pth in &paths[1..] {
            let im = crate::io::load_image(pth, bayer)?;
            if !im.same_shape(&first) {
                bail!("Taille différente : {} {} ≠ {}", pth.display(), im.shape_str(), first.shape_str());
            }
            for (a, &v) in acc.iter_mut().zip(&im.data) {
                *a += v as f64;
            }
        }
        let n = paths.len() as f64;
        return Ok(Image::from_vec(first.w, first.h, first.c, acc.iter().map(|&v| (v / n) as f32).collect()));
    }
    let imgs: Result<Vec<Image>> = paths.par_iter().map(|p| crate::io::load_image(p, bayer)).collect();
    let imgs = imgs?;
    let refs: Vec<&Image> = imgs.iter().collect();
    stack(&refs, method, p, None)
}

// ------------------------------------------------------------------ qualité
fn green_or_mono(img: &Image) -> Image {
    if img.is_color() {
        img.channel(1)
    } else {
        img.clone()
    }
}

/// Netteté : moyenne du laplacien absolu (luminance, échelle 16 bits).
pub fn sharpness_score(img: &Image) -> f64 {
    let a = green_or_mono(img);
    let (w, h) = (a.w, a.h);
    let mut s = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            let g = |dx: isize, dy: isize| {
                let xx = ndimage::fold(x as isize + dx, w, Border::Mirror).unwrap();
                let yy = ndimage::fold(y as isize + dy, h, Border::Mirror).unwrap();
                (a.data[yy * w + xx].clamp(0.0, 1.0) as f64 * 65535.0).floor()
            };
            s += (g(-1, 0) + g(1, 0) + g(0, -1) + g(0, 1) - 4.0 * g(0, 0)).abs();
        }
    }
    s / (w * h).max(1) as f64 + 1e-9
}

pub fn brightness_score(img: &Image) -> f64 {
    let a = green_or_mono(img);
    1.0 - (a.mean() - 0.5).abs() * 2.0
}

pub fn quality_scores(frames: &[Image], mode: &str) -> Vec<f64> {
    frames
        .par_iter()
        .map(|f| {
            let s = sharpness_score(f);
            let b = brightness_score(f);
            if mode.starts_with("Netteté") {
                s * (0.5 + 0.5 * b.max(0.0))
            } else if mode.starts_with("Luminosité") {
                b.max(1e-9)
            } else {
                s * b.max(1e-9)
            }
        })
        .collect()
}

/// (indices du meilleur au moins bon limités à keep_percent %, scores).
pub fn rank_frames(frames: &[Image], mode: &str, keep_percent: f64) -> (Vec<usize>, Vec<f64>) {
    let scores = quality_scores(frames, mode);
    let order = stats::argsort_desc(&scores);
    let k = ((order.len() as f64 * keep_percent / 100.0).round() as usize).max(1);
    (order[..k.min(order.len())].to_vec(), scores)
}

// ------------------------------------------- empilement basé AutoStakkert!
pub fn detail_score(img: &Image) -> f64 {
    let mono = img.mean_lum();
    let med = ndimage::median_filter(&mono, 5);
    mono.data.iter().zip(&med.data).map(|(&a, &b)| (a - b).abs() as f64).sum::<f64>() / mono.data.len().max(1) as f64
}

pub fn as_quality_scores(frames: &[Image]) -> Vec<f64> {
    frames.par_iter().map(detail_score).collect()
}

fn box_shifts(ref_: &Image, mov: &Image, grid_n: usize, bx: usize) -> Vec<(f64, f64)> {
    let (w, h) = (ref_.w, ref_.h);
    let ys = stats::linspace((bx / 2) as f64, (h - bx / 2) as f64, grid_n);
    let xs = stats::linspace((bx / 2) as f64, (w - bx / 2) as f64, grid_n);
    let mut out = Vec::with_capacity(grid_n * grid_n);
    for &cy in &ys {
        for &cx in &xs {
            let y0 = (cy - bx as f64 / 2.0).max(0.0) as usize;
            let x0 = (cx - bx as f64 / 2.0).max(0.0) as usize;
            let rb = ref_.crop(x0, y0, bx, bx);
            let mb = mov.crop(x0, y0, bx, bx);
            let (dx, dy, _) = crate::registration::phase_corr(&rb, &mb);
            out.push((dx, dy));
        }
    }
    out
}

/// Qualité par frame, sélection des meilleures, alignement multipoint
/// (champ de décalage lissé bicubique) puis moyenne.
pub fn autostacker_stack(frames: &[Image], keep_percent: f64, grid_n: usize, box_frac: f64) -> Result<Image> {
    let n = frames.len();
    if n == 0 {
        bail!("Aucune frame");
    }
    let order = stats::argsort_desc(&as_quality_scores(frames));
    let k = ((n as f64 * keep_percent.clamp(5.0, 100.0) / 100.0).round() as usize).max(2).min(n);
    let kept = &order[..k];
    let ref_ = &frames[kept[0]];
    let (w, h) = (ref_.w, ref_.h);
    let bx = 32usize.max((w.min(h) as f64 * box_frac) as usize);
    let grid_n = grid_n.max(1);
    let warped: Vec<Image> = kept
        .par_iter()
        .enumerate()
        .map(|(j, &fi)| {
            let f = &frames[fi];
            let field = if j == 0 { vec![(0.0, 0.0); grid_n * grid_n] } else { box_shifts(ref_, f, grid_n, bx) };
            let fx = Image::from_vec(grid_n, grid_n, 1, field.iter().map(|s| s.0 as f32).collect());
            let fy = Image::from_vec(grid_n, grid_n, 1, field.iter().map(|s| s.1 as f32).collect());
            let mx = geom::resize(&fx, w, h, Interp::Cubic);
            let my = geom::resize(&fy, w, h, Interp::Cubic);
            geom::remap_fn(f, w, h, Interp::Cubic, Border::Reflect, |x, y| {
                let i = y * w + x;
                (x as f64 - mx.data[i] as f64, y as f64 - my.data[i] as f64)
            })
        })
        .collect();
    let refs: Vec<&Image> = warped.iter().collect();
    Ok(crate::image::mean_images(&refs))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn methods_finite() {
        let frames: Vec<Image> = (0..9)
            .map(|k| Image::from_fn(16, 16, move |x, y| 0.1 + 0.01 * ((x * 7 + y * 3 + k * 5) % 11) as f32))
            .collect();
        let refs: Vec<&Image> = frames.iter().collect();
        for m in METHODS {
            let o = stack(&refs, m, &StackParams::default(), None).unwrap();
            assert!(o.all_finite(), "{}", m);
            assert!((o.mean() - 0.15).abs() < 0.03, "{} {}", m, o.mean());
        }
    }
    #[test]
    fn sigma_rejects_outlier() {
        let mut frames: Vec<Image> = (0..5).map(|_| Image::filled(4, 4, 1, 0.2)).collect();
        frames[2].data[5] = 1.0;
        let refs: Vec<&Image> = frames.iter().collect();
        let o = stack(&refs, "Sigma-clipping", &StackParams::default(), None).unwrap();
        assert!((o.data[5] - 0.2).abs() < 1e-6);
        let w = [1.0, 2.0, 0.0, 1.0, 1.0];
        let o = stack(&refs, "Moyenne", &StackParams::default(), Some(&w)).unwrap();
        assert!((o.data[5] - 0.2).abs() < 1e-6);
    }
}
