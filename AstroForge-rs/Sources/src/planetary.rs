//! Traitement planétaire en 6 étapes (principes de Registax 6, AstroSurface,
//! AutoStakkert!) : vidéo, qualité + tri, sélection, méthode d'alignement
//! (globale, multi-points NxN, points manuels, multipoint local), alignement,
//! empilement.

use crate::finishing::{is_identity_curve, normalize_curve};
use crate::frames::Frames;
use crate::geom::{self, Interp};
use crate::image::Image;
use crate::ndimage::{self, Border};
use crate::registration::{apply_shift, phase_corr};
use crate::stacking::{self, StackParams};
use crate::stats;
use anyhow::{bail, Result};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use serde_json::Value;

pub const QUALITY_METHODS: [&str; 5] = ["Contraste HF (AutoStakkert)", "Netteté (Laplacien)", "Gradient (Sobel)", "Luminosité", "Mixte"];
pub const ALIGN_MODES: [&str; 4] = ["Multi-points (grille NxN)", "Global (une translation)", "Points manuels (carrés)", "Multi-points local (AS!)"];

#[derive(Clone, Debug, PartialEq)]
pub struct Tone {
    pub brightness: f64,
    pub contrast: f64,
    pub curve: Vec<(f64, f64)>,
}

impl Default for Tone {
    fn default() -> Self {
        Tone { brightness: 0.0, contrast: 1.0, curve: vec![(0.0, 0.0), (1.0, 1.0)] }
    }
}

impl Tone {
    pub fn new(brightness: f64, contrast: f64, curve: Option<&Value>) -> Result<Tone> {
        if !brightness.is_finite() || !contrast.is_finite() {
            bail!("Luminosité et contraste doivent être finis.");
        }
        Ok(Tone { brightness: brightness.clamp(-1.0, 1.0), contrast: contrast.clamp(0.0, 3.0), curve: normalize_curve(curve)? })
    }

    /// Réglages « tone » d'une séquence enregistrée.
    pub fn from_json(v: Option<&Value>) -> Result<Tone> {
        match v {
            Some(Value::Object(m)) => Tone::new(
                m.get("brightness").and_then(|x| x.as_f64()).unwrap_or(0.0),
                m.get("contrast").and_then(|x| x.as_f64()).unwrap_or(1.0),
                m.get("curve"),
            ),
            _ => Ok(Tone::default()),
        }
    }

    pub fn is_identity(&self) -> bool {
        self.brightness == 0.0 && self.contrast == 1.0 && is_identity_curve(&self.curve)
    }

    pub fn apply(&self, img: &Image) -> Image {
        let (xs, ys): (Vec<f64>, Vec<f64>) = self.curve.iter().cloned().unzip();
        let id = is_identity_curve(&self.curve);
        let (c, b) = (self.contrast as f32, self.brightness as f32);
        img.map(|v| {
            let mut a = v;
            if c != 1.0 {
                a = (a - 0.5) * c + 0.5;
            }
            if b != 0.0 {
                a += b;
            }
            a = a.clamp(0.0, 1.0);
            if !id {
                a = stats::interp(a as f64, &xs, &ys) as f32;
            }
            a
        })
    }
}

/// Vue différée : chaque frame est ajustée (tonalité, recadrage) à la demande.
pub struct ToneFrames<F: Frames> {
    pub originals: F,
    pub tone: Tone,
    pub boxr: Option<(usize, usize, usize, usize)>,
}

impl<F: Frames> Frames for ToneFrames<F> {
    fn len(&self) -> usize {
        self.originals.len()
    }
    fn get(&self, i: usize) -> Image {
        let mut im = self.originals.get(i);
        if !self.tone.is_identity() {
            im = self.tone.apply(&im);
        }
        if let Some((x, y, w, h)) = self.boxr {
            im = im.crop(x, y, w, h);
        }
        im
    }
}

impl<F: Frames> ToneFrames<F> {
    /// Vérifie toutes les frames (pixels finis).
    pub fn validate(&self) -> Result<()> {
        for i in 0..self.len() {
            if !self.get(i).all_finite() {
                bail!("Image {} : pixels non finis.", i + 1);
            }
        }
        Ok(())
    }
}

pub fn luminance(img: &Image) -> Image {
    img.mean_lum()
}

/// Moyenne échantillonnée (une frame sur `stride`).
pub fn mean_sample(frames: &dyn Frames, stride: usize) -> Image {
    let idx: Vec<usize> = (0..frames.len()).step_by(stride.max(1)).collect();
    let imgs: Vec<Image> = idx.par_iter().map(|&i| frames.get(i)).collect();
    let refs: Vec<&Image> = imgs.iter().collect();
    crate::image::mean_images(&refs)
}

/// Boîte (x0, y0, x1, y1) du disque planétaire, ou None.
pub fn detect_planet_box(img: &Image, pad: f64) -> Option<(usize, usize, usize, usize)> {
    let a = ndimage::gaussian(&luminance(img), 2.0);
    let thr = 0.5 * (stats::percentile(&a.data, 50.0) + stats::percentile(&a.data, 99.5));
    let (w, h) = (a.w, a.h);
    let (mut xmin, mut xmax, mut ymin, mut ymax, mut n) = (usize::MAX, 0, usize::MAX, 0, 0usize);
    for y in 0..h {
        for x in 0..w {
            if a.data[y * w + x] > thr {
                xmin = xmin.min(x);
                xmax = xmax.max(x);
                ymin = ymin.min(y);
                ymax = ymax.max(y);
                n += 1;
            }
        }
    }
    if n < 30 {
        return None;
    }
    let (cx, cy) = ((xmin + xmax) as f64 / 2.0, (ymin + ymax) as f64 / 2.0);
    let r = ((xmax - xmin).max(ymax - ymin)) as f64 / 2.0 * (1.0 + pad) + 2.0;
    Some((((cx - r) as i64).max(0) as usize, ((cy - r) as i64).max(0) as usize, ((cx + r) as usize).min(w), ((cy + r) as usize).min(h)))
}

pub fn q_hf(a: &Image) -> f64 {
    let m = ndimage::median_filter(a, 5);
    a.data.iter().zip(&m.data).map(|(x, y)| (x - y).abs() as f64).sum::<f64>() / a.data.len().max(1) as f64 + 1e-9
}

pub fn q_lap(a: &Image) -> f64 {
    let l = ndimage::laplace(a);
    l.data.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / l.data.len().max(1) as f64 + 1e-9
}

pub fn q_grad(a: &Image) -> f64 {
    let (w, h) = (a.w, a.h);
    let (mut gx, mut gy) = (0.0f64, 0.0f64);
    for y in 0..h {
        for x in 0..w.saturating_sub(1) {
            gx += (a.data[y * w + x + 1] - a.data[y * w + x]).abs() as f64;
        }
    }
    for y in 0..h.saturating_sub(1) {
        for x in 0..w {
            gy += (a.data[(y + 1) * w + x] - a.data[y * w + x]).abs() as f64;
        }
    }
    gx / ((w.saturating_sub(1)) * h).max(1) as f64 + gy / (w * h.saturating_sub(1)).max(1) as f64 + 1e-9
}

/// Score de qualité de chaque frame (dans la ROI si fournie).
pub fn measure_quality(frames: &dyn Frames, method: &str, roi: Option<(usize, usize, usize, usize)>) -> Vec<f64> {
    (0..frames.len())
        .into_par_iter()
        .map(|i| {
            let mut a = luminance(&frames.get(i));
            if let Some((x0, y0, x1, y1)) = roi {
                let (x1, y1) = (x1.max(1), y1.max(1));
                let sub = a.crop(x0, y0, x1.saturating_sub(x0), y1.saturating_sub(y0));
                if sub.data.len() >= 64 {
                    a = sub;
                }
            }
            let m = a.mean();
            if method.starts_with("Luminosité") {
                (1.0 - (m - 0.5).abs() * 2.0).max(0.0)
            } else if method.starts_with("Netteté") {
                q_lap(&a)
            } else if method.starts_with("Gradient") {
                q_grad(&a)
            } else if method.starts_with("Mixte") {
                q_lap(&a) * (1.0 - (m - 0.5).abs() * 2.0).max(1e-9)
            } else {
                q_hf(&a)
            }
        })
        .collect()
}

pub fn sort_quality(scores: &[f64]) -> Vec<usize> {
    stats::argsort_desc(scores)
}

/// Grille N×N de centres de boîtes (image ou ROI).
pub fn grid_points(w: usize, h: usize, n: usize, bx: usize, roi: Option<(usize, usize, usize, usize)>) -> Vec<(f64, f64)> {
    let (x0, y0, x1, y1) = roi.unwrap_or((0, 0, w, h));
    let n = n.max(1);
    let b = bx as f64;
    let xs = if n > 1 { stats::linspace(x0 as f64 + b / 2.0, x1 as f64 - b / 2.0, n) } else { vec![(x0 + x1) as f64 / 2.0] };
    let ys = if n > 1 { stats::linspace(y0 as f64 + b / 2.0, y1 as f64 - b / 2.0, n) } else { vec![(y0 + y1) as f64 / 2.0] };
    ys.iter().flat_map(|&cy| xs.iter().map(move |&cx| (cx, cy))).collect()
}

fn lstsq(a: &DMatrix<f64>, b: &DVector<f64>) -> DVector<f64> {
    a.clone().svd(true, true).solve(b, 1e-12).unwrap_or_else(|_| DVector::zeros(a.ncols()))
}

/// Champ de décalage affine s = A·p + b (consensus, rejet MAD) :
/// (coefficients [a1 a2 a3 b1 b2 b3], rejetées, rms).
pub fn fit_shift_field(pts: &[(f64, f64)], shifts: &[(f64, f64)]) -> ([f64; 6], usize, f64) {
    let n = pts.len();
    let mut keep = vec![true; n];
    if n >= 3 {
        let counts: Vec<usize> = (0..n).map(|i| (0..n).filter(|&j| (shifts[i].0 - shifts[j].0).hypot(shifts[i].1 - shifts[j].1) <= 1.0).count()).collect();
        let ci = (0..n).max_by_key(|&i| (counts[i], std::cmp::Reverse(i))).unwrap();
        let center = shifts[ci];
        let cons: Vec<bool> = shifts.iter().map(|s| (s.0 - center.0).hypot(s.1 - center.1) <= 1.0).collect();
        if cons.iter().filter(|&&b| b).count() >= 3 {
            keep = cons;
        }
    }
    let nk = keep.iter().filter(|&&b| b).count();
    let mean_k = |f: &dyn Fn(&(f64, f64)) -> f64| -> f64 {
        if nk == 0 {
            0.0
        } else {
            (0..n).filter(|&i| keep[i]).map(|i| f(&shifts[i])).sum::<f64>() / nk as f64
        }
    };
    let mut cx = DVector::from_vec(vec![0.0, 0.0, mean_k(&|s| s.0)]);
    let mut cy = DVector::from_vec(vec![0.0, 0.0, mean_k(&|s| s.1)]);
    let build = |keep: &[bool]| -> (DMatrix<f64>, DVector<f64>, DVector<f64>) {
        let idx: Vec<usize> = (0..n).filter(|&i| keep[i]).collect();
        let mut a = DMatrix::<f64>::zeros(idx.len(), 3);
        for (r, &i) in idx.iter().enumerate() {
            a[(r, 0)] = pts[i].0;
            a[(r, 1)] = pts[i].1;
            a[(r, 2)] = 1.0;
        }
        (a, DVector::from_iterator(idx.len(), idx.iter().map(|&i| shifts[i].0)), DVector::from_iterator(idx.len(), idx.iter().map(|&i| shifts[i].1)))
    };
    let resid = |cx: &DVector<f64>, cy: &DVector<f64>| -> Vec<f64> {
        (0..n)
            .map(|i| {
                let (x, y) = pts[i];
                ((cx[0] * x + cx[1] * y + cx[2]) - shifts[i].0).hypot((cy[0] * x + cy[1] * y + cy[2]) - shifts[i].1)
            })
            .collect()
    };
    if keep.iter().filter(|&&b| b).count() >= 3 {
        for _ in 0..2 {
            let (a, bx, by) = build(&keep);
            cx = lstsq(&a, &bx);
            cy = lstsq(&a, &by);
            let r = resid(&cx, &cy);
            let rk: Vec<f64> = (0..n).filter(|&i| keep[i]).map(|i| r[i]).collect();
            let med = stats::median_f64(&rk);
            let mad = 1.4826 * stats::median_f64(&rk.iter().map(|v| (v - med).abs()).collect::<Vec<_>>()) + 0.05;
            let nk2: Vec<bool> = r.iter().map(|&v| v <= med + 3.0 * mad).collect();
            if nk2.iter().filter(|&&b| b).count() < 3 || nk2 == keep {
                break;
            }
            keep = nk2;
        }
        let (a, bx, by) = build(&keep);
        cx = lstsq(&a, &bx);
        cy = lstsq(&a, &by);
    }
    let r = resid(&cx, &cy);
    let rk: Vec<f64> = (0..n).filter(|&i| keep[i]).map(|i| r[i] * r[i]).collect();
    let rms = if rk.is_empty() { 0.0 } else { (rk.iter().sum::<f64>() / rk.len() as f64).sqrt() };
    ([cx[0], cx[1], cx[2], cy[0], cy[1], cy[2]], keep.iter().filter(|&&b| !b).count(), rms)
}

#[derive(Clone, Debug, Default)]
pub struct AlignStats {
    pub n_points: usize,
    pub rejected: usize,
    pub rms: f64,
    pub dx: f64,
    pub dy: f64,
}

/// Aligne `mov` sur `ref_` (global, multi-points ou points manuels).
pub fn align_frame(ref_: &Image, mov: &Image, points: Option<&[(f64, f64)]>, bx: usize, mode: &str) -> (Image, AlignStats) {
    let (rl, ml) = (luminance(ref_), luminance(mov));
    let (w, h) = (rl.w, rl.h);
    let bx = bx.max(16);
    let (coeffs, n_rej, rms, n_pts) = if mode.starts_with("Global") {
        let (dx, dy, _) = phase_corr(ref_, mov);
        ([0.0, 0.0, dx, 0.0, 0.0, dy], 0, 0.0, 1)
    } else {
        let pts: Vec<(f64, f64)> = match points {
            Some(p) if !p.is_empty() => p.to_vec(),
            _ => grid_points(w, h, 4, bx, None),
        };
        let shifts: Vec<(f64, f64)> = pts
            .par_iter()
            .map(|&(cx, cy)| {
                let (x0, y0) = ((cx - bx as f64 / 2.0) as i64, (cy - bx as f64 / 2.0) as i64);
                let (x0, y0) = (x0.max(0) as usize, y0.max(0) as usize);
                let rb = rl.crop(x0, y0, bx, bx);
                let mb = ml.crop(x0, y0, bx, bx);
                if rb.w < 2 || rb.h < 2 {
                    return (0.0, 0.0);
                }
                let (dx, dy, _) = phase_corr(&rb, &mb);
                (dx, dy)
            })
            .collect();
        let (c, nr, rms) = fit_shift_field(&pts, &shifts);
        (c, nr, rms, pts.len())
    };
    let [a1, a2, a3, b1, b2, b3] = coeffs;
    let out = geom::remap_fn(mov, w, h, Interp::Cubic, Border::Reflect, |x, y| {
        let (xf, yf) = (x as f64, y as f64);
        (xf - (a1 * xf + a2 * yf + a3), yf - (b1 * xf + b2 * yf + b3))
    });
    (out, AlignStats { n_points: n_pts, rejected: n_rej, rms, dx: a3, dy: b3 })
}

fn hann2(n: usize) -> Vec<f32> {
    let w = crate::fft::hanning(n + 2);
    let w = &w[1..n + 1];
    let mut out = vec![0.0f32; n * n];
    for y in 0..n {
        for x in 0..n {
            out[y * n + x] = (w[y] * w[x]) as f32;
        }
    }
    out
}

pub struct LocalResult {
    pub image: Image,
    pub n_aps: usize,
    pub frames_per_ap: usize,
    pub reference: Image,
}

/// Empilement multipoint LOCAL (principe d'AutoStakkert!).
#[allow(clippy::too_many_arguments)]
pub fn local_multipoint_stack(frames: &dyn Frames, order: &[usize], n_sel: usize, roi: Option<(usize, usize, usize, usize)>, bx: usize, _grid_n: usize, ap_percent: f64, min_contrast: f64) -> LocalResult {
    let sel: Vec<usize> = order.iter().take(n_sel.max(2)).cloned().collect();
    let first = frames.get(sel[0]);
    let ref_l = luminance(&first);
    let shifts: Vec<(f64, f64)> = sel
        .par_iter()
        .enumerate()
        .map(|(k, &i)| if k == 0 { (0.0, 0.0) } else { let (dx, dy, _) = phase_corr(&ref_l, &luminance(&frames.get(i))); (dx, dy) })
        .collect();
    let n_ref = (sel.len() / 10).max(2).min(sel.len());
    let shifted = |k: usize| apply_shift(&frames.get(sel[k]), shifts[k].0, shifts[k].1);
    let ref_imgs: Vec<Image> = (0..n_ref).into_par_iter().map(shifted).collect();
    let reference = crate::image::mean_images(&ref_imgs.iter().collect::<Vec<_>>());
    let n_base = n_ref.max((sel.len() as f64 * ap_percent / 100.0) as usize).min(sel.len());
    let base_imgs: Vec<Image> = (0..n_base).into_par_iter().map(shifted).collect();
    let base = crate::image::mean_images(&base_imgs.iter().collect::<Vec<_>>());
    drop(base_imgs);
    let (image, n_aps, k_ap) = local_ap_core(frames, &sel, &shifts, &reference, &base, roi, bx, ap_percent, min_contrast);
    LocalResult { image, n_aps, frames_per_ap: k_ap, reference }
}

/// Cœur de l'empilement multipoint local : `sel` (frames triées), décalages
/// globaux (`apply_shift(frame, dx, dy)` ≈ référence), référence et image de
/// base déjà calculées. Retourne (image, nombre de points, frames par point).
#[allow(clippy::too_many_arguments)]
pub fn local_ap_core(frames: &dyn Frames, sel: &[usize], shifts: &[(f64, f64)], reference: &Image, base: &Image, roi: Option<(usize, usize, usize, usize)>, bx: usize, ap_percent: f64, min_contrast: f64) -> (Image, usize, usize) {
    let (w, h, c) = (base.w, base.h, base.c);
    let ref_lum = luminance(&reference);
    let bx = bx.max(16);
    let (x0, y0, x1, y1) = roi.unwrap_or((0, 0, w, h));
    let step = 8usize.max(bx / 2);
    let contrast_all = stats::std(&ref_lum.data) + 1e-9;
    let mut aps = Vec::new();
    let mut cy = y0 + bx / 2;
    while cy + bx / 2 <= y1 {
        let mut cx = x0 + bx / 2;
        while cx + bx / 2 <= x1 {
            if cx >= bx / 2 && cy >= bx / 2 {
                let patch = ref_lum.crop(cx - bx / 2, cy - bx / 2, bx, bx);
                if patch.w == bx && patch.h == bx && stats::std(&patch.data) >= min_contrast * contrast_all {
                    aps.push((cx, cy));
                }
            }
            cx += step;
        }
        cy += step;
    }
    let win = hann2(bx);
    let k_ap = ((sel.len() as f64 * ap_percent.clamp(1.0, 100.0) / 100.0).round() as usize).max(1);
    let margin = 8usize;
    // frames gardées en mémoire (si la sélection tient en ~1,5 Go), sinon relues
    let cache = (sel.len() * w * h * c * 4) < 1_500_000_000;
    let frames_sel: Vec<Image> = if cache { sel.par_iter().map(|&i| frames.get(i)).collect() } else { vec![] };
    let crop_of = |k: usize, x: usize, y: usize, cw: usize, ch: usize| -> Image {
        if cache {
            frames_sel[k].crop(x, y, cw, ch)
        } else {
            frames.get(sel[k]).crop(x, y, cw, ch)
        }
    };
    let patches: Vec<Option<(usize, usize, Image)>> = aps
        .par_iter()
        .map(|&(cx, cy)| {
            let (xs, ys) = (cx - bx / 2, cy - bx / 2);
            let rp = ref_lum.crop(xs, ys, bx, bx);
            let mut cands: Vec<(f64, usize, Image)> = Vec::new();
            for (k, _) in sel.iter().enumerate() {
                let (dx, dy) = shifts[k];
                let sx = (xs as f64 - dx).round() as i64;
                let sy = (ys as f64 - dy).round() as i64;
                if sx - (margin as i64) < 0 || sy - (margin as i64) < 0 || sx as usize + bx + margin > w || sy as usize + bx + margin > h {
                    continue;
                }
                let big = crop_of(k, sx as usize - margin, sy as usize - margin, bx + 2 * margin, bx + 2 * margin);
                let q = q_hf(&luminance(&big).crop(margin, margin, bx, bx));
                cands.push((q, k, big));
            }
            if cands.is_empty() {
                return None;
            }
            cands.sort_by(|a, b| b.0.total_cmp(&a.0));
            let mut stack = Vec::new();
            for (_, _, big) in cands.into_iter().take(k_ap) {
                let (ddx, ddy, _) = phase_corr(&rp, &luminance(&big).crop(margin, margin, bx, bx));
                if ddx.hypot(ddy) > margin as f64 - 1.0 {
                    continue;
                }
                let al = apply_shift(&big, ddx, ddy);
                stack.push(al.crop(margin, margin, bx, bx));
            }
            if stack.is_empty() {
                return None;
            }
            Some((xs, ys, crate::image::mean_images(&stack.iter().collect::<Vec<_>>())))
        })
        .collect();
    let mut acc = vec![0.0f64; w * h * c];
    let mut wacc = vec![0.0f64; w * h];
    for (xs, ys, patch) in patches.into_iter().flatten() {
        for y in 0..bx {
            for x in 0..bx {
                let wv = win[y * bx + x] as f64;
                let o = (ys + y) * w + xs + x;
                wacc[o] += wv;
                for k in 0..c {
                    acc[o * c + k] += patch.data[(y * bx + x) * c + k] as f64 * wv;
                }
            }
        }
    }
    let pos: Vec<f64> = wacc.iter().cloned().filter(|&v| v > 0.0).collect();
    let norm = if pos.is_empty() { 1.0 } else { stats::median_f64(&pos) }.max(1e-9);
    let mut out = base.clone();
    for i in 0..w * h {
        let wn = (wacc[i] / norm).clamp(0.0, 1.0);
        if wn > 0.0 {
            for k in 0..c {
                let local = acc[i * c + k] / wacc[i].max(1e-12);
                out.data[i * c + k] = (local * wn + base.data[i * c + k] as f64 * (1.0 - wn)) as f32;
            }
        }
    }
    (out.clip01(), aps.len(), k_ap)
}

#[derive(Clone, Debug, Default)]
pub struct PlanetStats {
    pub n_frames: usize,
    pub n_selected: usize,
    pub roi: Option<(usize, usize, usize, usize)>,
    pub box_size: usize,
    pub n_points: usize,
    pub rejected_boxes: usize,
    pub rms_boxes: f64,
    pub shift_mean: f64,
    pub shift_max: f64,
    pub best_frame: usize,
    pub stacked: usize,
}

pub struct PlanetResult {
    pub image: Image,
    pub order: Vec<usize>,
    pub scores: Vec<f64>,
    pub selected: Vec<usize>,
    pub stats: PlanetStats,
}

#[derive(Clone, Debug)]
pub struct PlanetOptions {
    pub quality_method: String,
    pub roi_mode: String,
    pub n_align: usize,
    pub align_mode: String,
    pub grid_n: usize,
    pub box_size: usize,
    pub manual_points: Option<Vec<(f64, f64)>>,
    pub stack_method: String,
    pub stack_percent: f64,
}

impl Default for PlanetOptions {
    fn default() -> Self {
        PlanetOptions {
            quality_method: QUALITY_METHODS[0].into(),
            roi_mode: "Disque auto-détecté".into(),
            n_align: 30,
            align_mode: ALIGN_MODES[0].into(),
            grid_n: 4,
            box_size: 0,
            manual_points: None,
            stack_method: "Moyenne".into(),
            stack_percent: 100.0,
        }
    }
}

/// Enchaîne les 6 étapes ; retourne image, ordre, scores, sélection, stats.
pub fn run_planetary(frames: &dyn Frames, o: &PlanetOptions, progress: Option<&(dyn Fn(usize, usize, &str) + Sync)>) -> Result<PlanetResult> {
    let n = frames.len();
    if n == 0 {
        bail!("Aucune frame chargée");
    }
    let mut roi = None;
    if o.roi_mode.starts_with("Disque") {
        let mid = mean_sample(frames, (n / 20).max(1));
        roi = detect_planet_box(&mid, 0.2);
    }
    if let Some(p) = progress {
        p(0, 3, "Qualité des images");
    }
    let scores = measure_quality(frames, &o.quality_method, roi);
    let order = sort_quality(&scores);
    let n_sel = o.n_align.clamp(1, n);
    let sel: Vec<usize> = order[..n_sel].to_vec();
    if o.align_mode.starts_with("Multi-points local") {
        let ref0 = frames.get(sel[0]);
        let bx = if o.box_size >= 16 { o.box_size } else { 32usize.max(ref0.w.min(ref0.h) / 8) };
        if let Some(p) = progress {
            p(1, 3, "Points d'alignement locaux");
        }
        let loc = local_multipoint_stack(frames, &order, n_sel, roi, bx, o.grid_n, o.stack_percent.clamp(5.0, 100.0), 0.15);
        return Ok(PlanetResult {
            image: loc.image,
            stats: PlanetStats {
                n_frames: n,
                n_selected: n_sel,
                roi,
                box_size: bx,
                n_points: loc.n_aps,
                best_frame: order[0],
                stacked: loc.frames_per_ap,
                ..Default::default()
            },
            order,
            scores,
            selected: sel,
        });
    }
    let ref_ = frames.get(sel[0]);
    let (w, h) = (ref_.w, ref_.h);
    let bx = if o.box_size >= 16 { o.box_size } else { 24usize.max(w.min(h) / 6) };
    let pts: Option<Vec<(f64, f64)>> = if o.align_mode.starts_with("Points manuels") {
        Some(o.manual_points.clone().filter(|p| !p.is_empty()).unwrap_or_else(|| vec![(w as f64 / 2.0, h as f64 / 2.0)]))
    } else if o.align_mode.starts_with("Multi-points") {
        Some(grid_points(w, h, o.grid_n.max(2), bx, roi))
    } else {
        None
    };
    if let Some(p) = progress {
        p(1, 3, "Alignement");
    }
    let results: Vec<(Image, Option<AlignStats>)> = sel
        .par_iter()
        .enumerate()
        .map(|(j, &fi)| {
            if j == 0 {
                (ref_.clone(), None)
            } else {
                let (im, st) = align_frame(&ref_, &frames.get(fi), pts.as_deref(), bx, &o.align_mode);
                (im, Some(st))
            }
        })
        .collect();
    let mut aligned = Vec::with_capacity(n_sel);
    let (mut tot_rej, mut rms_sum, mut shifts) = (0usize, 0.0f64, Vec::new());
    for (im, st) in results {
        if let Some(s) = st {
            tot_rej += s.rejected;
            rms_sum += s.rms;
            shifts.push(s.dx.hypot(s.dy));
        }
        aligned.push(im);
    }
    let mut method = o.stack_method.clone();
    if method.starts_with("Moyenne (meilleurs") {
        let k = ((n_sel as f64 * o.stack_percent.clamp(5.0, 100.0) / 100.0) as usize).max(1);
        aligned.truncate(k);
        method = "Moyenne".into();
    }
    if let Some(p) = progress {
        p(2, 3, "Empilement");
    }
    let refs: Vec<&Image> = aligned.iter().collect();
    let img = stacking::stack(&refs, &method, &StackParams::default(), None)?;
    let stats = PlanetStats {
        n_frames: n,
        n_selected: n_sel,
        roi,
        box_size: bx,
        n_points: if o.align_mode.starts_with("Global") { 1 } else { pts.as_ref().map(|p| p.len()).unwrap_or(0) },
        rejected_boxes: tot_rej,
        rms_boxes: rms_sum / (n_sel.saturating_sub(1)).max(1) as f64,
        shift_mean: if shifts.is_empty() { 0.0 } else { shifts.iter().sum::<f64>() / shifts.len() as f64 },
        shift_max: shifts.iter().cloned().fold(0.0, f64::max),
        best_frame: order[0],
        stacked: aligned.len(),
    };
    if let Some(p) = progress {
        p(3, 3, "Terminé");
    }
    Ok(PlanetResult { image: img, order, scores, selected: sel, stats })
}

/// Contraste automatique : points noir/blanc aux centiles `clip` de la luminance.
pub fn auto_contrast_params(img: &Image, clip: f64) -> Result<(Tone, f64, f64)> {
    let lum = luminance(img);
    let c = clip.clamp(0.0, 0.49);
    let black = stats::quantile(&lum.data, c) as f64;
    let white = stats::quantile(&lum.data, 1.0 - c) as f64;
    if white - black < 1e-3 {
        return Ok((Tone::default(), black, white));
    }
    let k = 1.0 / (white - black);
    Ok((Tone::new(k * (0.5 + black) - 0.5, k, None)?, black, white))
}

pub fn gamma_curve(gamma: f64, knots: usize) -> Vec<(f64, f64)> {
    let g = gamma.clamp(0.1, 10.0);
    stats::linspace(0.0, 1.0, knots).into_iter().map(|x| (x, x.powf(g))).collect()
}

/// Luminosité automatique : gamma visant le gris moyen `target`.
pub fn auto_brightness_params(img: &Image, target: f64, knots: usize) -> (Tone, f64) {
    let m = luminance(img).mean().clamp(1e-6, 1.0 - 1e-6);
    let gamma = target.ln() / m.ln();
    let curve = gamma_curve(gamma, knots);
    let v = Value::Array(curve.iter().map(|&(x, y)| Value::Array(vec![x.into(), y.into()])).collect());
    (Tone::new(0.0, 1.0, Some(&v)).unwrap_or_default(), gamma.clamp(0.1, 10.0))
}
