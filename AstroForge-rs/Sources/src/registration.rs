//! Alignement : corrélation de phase passe-haut sous-pixel, rotation
//! (Fourier–Mellin log-polaire + raffinement), dérotation planétaire,
//! alignement des canaux, contours (limbe), similitude et distorsions
//! polynomiales par étoiles, transformation globale pour le drizzle.

use crate::display;
use crate::fft::{self, C64};
use crate::features;
use crate::geom::{self, Affine, Interp};
use crate::image::Image;
use crate::ndimage::{self, Border};
use crate::starmatch::{self, Pt};
use crate::stars;
use crate::stats;
use anyhow::{bail, Result};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

pub const ALIGN_METHODS: [&str; 3] = ["Phase (rapide)", "Limbe (précis)", "Étoiles (large champ)"];

fn lum(img: &Image) -> Image {
    img.mean_lum()
}

/// Corrélation croisée passe-haut pondérée (fenêtre de Hann) ; retourne
/// (dx, dy, pic) : translation à appliquer à `mov` pour le recaler sur `ref_`
/// (apply_shift(mov, dx, dy) ≈ ref_).
pub fn phase_corr(ref_: &Image, mov: &Image) -> (f64, f64, f64) {
    let r = lum(ref_);
    let mut m = lum(mov);
    let (w, h) = (r.w, r.h);
    if w == 0 || h == 0 {
        return (0.0, 0.0, 0.0);
    }
    if m.w != w || m.h != h {
        let med = stats::median(&m.data);
        let mut tmp = Image::filled(w, h, 1, med);
        let src = m.crop(0, 0, w.min(m.w), h.min(m.h));
        tmp.paste(&src, 0, 0);
        m = tmp;
    }
    let hy = fft::hanning(h);
    let hx = fft::hanning(w);
    let prep = |x: &Image| -> Vec<C64> {
        let g = ndimage::gaussian(x, 4.0);
        let mut v = Vec::with_capacity(w * h);
        for y in 0..h {
            for xx in 0..w {
                let i = y * w + xx;
                v.push(C64::new((x.data[i] - g.data[i]) as f64 * hy[y] * hx[xx], 0.0));
            }
        }
        v
    };
    let (mut fa, mut fb) = rayon::join(|| prep(&r), || prep(&m));
    rayon::join(|| fft::fft2(&mut fa, w, h, false), || fft::fft2(&mut fb, w, h, false));
    let mut p: Vec<C64> = fa.iter().zip(&fb).map(|(a, b)| a * b.conj()).collect();
    fft::fft2(&mut p, w, h, true);
    let corr: Vec<f64> = p.iter().map(|c| c.re).collect();
    let (mut best, mut bi) = (f64::NEG_INFINITY, 0usize);
    for (i, &v) in corr.iter().enumerate() {
        if v > best {
            best = v;
            bi = i;
        }
    }
    let (y0, x0) = (bi / w, bi % w);
    let foldc = |v: usize, n: usize| -> f64 { if v > n / 2 { v as f64 - n as f64 } else { v as f64 } };
    let c = best;
    let mut dy = foldc(y0, h);
    let mut dx = foldc(x0, w);
    if h >= 3 {
        let cm = corr[((y0 + h - 1) % h) * w + x0];
        let cp = corr[((y0 + 1) % h) * w + x0];
        let den = cm - 2.0 * c + cp;
        if den.abs() > 1e-12 {
            dy += (0.5 * (cm - cp) / den).clamp(-0.5, 0.5);
        }
    }
    if w >= 3 {
        let cm = corr[y0 * w + (x0 + w - 1) % w];
        let cp = corr[y0 * w + (x0 + 1) % w];
        let den = cm - 2.0 * c + cp;
        if den.abs() > 1e-12 {
            dx += (0.5 * (cm - cp) / den).clamp(-0.5, 0.5);
        }
    }
    (dx, dy, c)
}

pub fn apply_shift(img: &Image, dx: f64, dy: f64) -> Image {
    geom::apply_shift(img, dx, dy, 0.0, Border::Reflect)
}

pub fn apply_shift_rot(img: &Image, dx: f64, dy: f64, angle_deg: f64) -> Image {
    geom::apply_shift(img, dx, dy, angle_deg, Border::Reflect)
}

/// Corrélation de phase normalisée (cv2.phaseCorrelate) : (sx, sy, réponse).
fn cv_phase_correlate(a: &[f64], b: &[f64], w: usize, h: usize) -> (f64, f64, f64) {
    let mut fa: Vec<C64> = fft::to_complex(a);
    let mut fb: Vec<C64> = fft::to_complex(b);
    fft::fft2(&mut fa, w, h, false);
    fft::fft2(&mut fb, w, h, false);
    let mut p: Vec<C64> = fa
        .iter()
        .zip(&fb)
        .map(|(x, y)| {
            let v = x * y.conj();
            let n = v.norm();
            if n > 1e-300 { v / n } else { C64::new(0.0, 0.0) }
        })
        .collect();
    fft::fft2(&mut p, w, h, true);
    // fftshift
    let mut c = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let (sy, sx) = ((y + h / 2) % h, (x + w / 2) % w);
            c[sy * w + sx] = p[y * w + x].re * (w * h) as f64;
        }
    }
    let (mut best, mut bi) = (f64::NEG_INFINITY, 0usize);
    for (i, &v) in c.iter().enumerate() {
        if v > best {
            best = v;
            bi = i;
        }
    }
    let (py, px) = ((bi / w) as isize, (bi % w) as isize);
    let (mut sx, mut sy, mut st) = (0.0, 0.0, 0.0);
    for yy in (py - 2).max(0)..=(py + 2).min(h as isize - 1) {
        for xx in (px - 2).max(0)..=(px + 2).min(w as isize - 1) {
            let v = c[yy as usize * w + xx as usize];
            sx += v * xx as f64;
            sy += v * yy as f64;
            st += v;
        }
    }
    let (tx, ty) = if st.abs() > 1e-300 { (sx / st, sy / st) } else { (px as f64, py as f64) };
    let resp = st / (w * h) as f64;
    (w as f64 / 2.0 - tx, h as f64 / 2.0 - ty, resp)
}

/// cv2.warpPolar (WARP_POLAR_LOG, bilinéaire, extérieur à 0).
fn warp_polar_log(src: &Image, n_rad: usize, n_ang: usize, cx: f64, cy: f64, max_r: f64) -> Vec<f64> {
    let klog = n_rad as f64 / max_r.ln();
    let mut out = vec![0.0f64; n_rad * n_ang];
    out.par_chunks_mut(n_rad).enumerate().for_each(|(row, dst)| {
        let ang = row as f64 * 2.0 * std::f64::consts::PI / n_ang as f64;
        let (ca, sa) = (ang.cos(), ang.sin());
        let mut px = [0.0f32; 1];
        for (rho, d) in dst.iter_mut().enumerate() {
            let r = (rho as f64 / klog).exp();
            let (x, y) = (cx + r * ca, cy + r * sa);
            if x < 0.0 || y < 0.0 || x > (src.w - 1) as f64 || y > (src.h - 1) as f64 {
                *d = 0.0;
                continue;
            }
            geom::sample(src, x, y, Interp::Linear, Border::Constant, &mut px);
            *d = px[0] as f64;
        }
    });
    out
}

/// Rotation + échelle directes (Fourier–Mellin) : (angle_deg, échelle, réponse).
pub fn logpolar_rotation(ref_: &Image, mov: &Image) -> (f64, f64, f64) {
    let r = lum(ref_);
    let mut m = lum(mov);
    let (w, h) = (r.w, r.h);
    if m.w != w || m.h != h {
        m = geom::resize(&m, w, h, Interp::Area);
    }
    let hy = fft::hanning(h);
    let hx = fft::hanning(w);
    let ys = stats::linspace(-0.5, 0.5, h);
    let xs = stats::linspace(-0.5, 0.5, w);
    let mag = |x: &Image| -> Image {
        let g = ndimage::gaussian(x, 4.0);
        let mut v: Vec<C64> = Vec::with_capacity(w * h);
        for y in 0..h {
            for xx in 0..w {
                let i = y * w + xx;
                v.push(C64::new((x.data[i] - g.data[i]) as f64 * hy[y] * hx[xx], 0.0));
            }
        }
        fft::fft2(&mut v, w, h, false);
        let mut out = Image::new(w, h, 1);
        for y in 0..h {
            for xx in 0..w {
                let (sy, sx) = ((y + h / 2) % h, (xx + w / 2) % w);
                let f = v[y * w + xx].norm();
                let c = (std::f64::consts::PI * ys[sy]).cos() * (std::f64::consts::PI * xs[sx]).cos();
                out.data[sy * w + sx] = ((f.ln_1p()) * (1.0 - c) * (2.0 - c)) as f32;
            }
        }
        out
    };
    let (ar, am) = rayon::join(|| mag(&r), || mag(&m));
    let n_ang = 720usize;
    let radius = w.min(h) as f64 / 2.0;
    let n_rad = (64.0f64.max(radius)) as usize;
    let pr = warp_polar_log(&ar, n_rad, n_ang, w as f64 / 2.0, h as f64 / 2.0, radius);
    let pm = warp_polar_log(&am, n_rad, n_ang, w as f64 / 2.0, h as f64 / 2.0, radius);
    let (sx, sy, resp) = cv_phase_correlate(&pr, &pm, n_rad, n_ang);
    let mut ang = sy * 360.0 / n_ang as f64;
    ang = (ang + 90.0).rem_euclid(180.0) - 90.0;
    let scale = (sx * radius.ln() / n_rad as f64).exp();
    (ang, scale, resp)
}

fn rot_score(r_hp: &Image, mov: &Image, ang: f64) -> f64 {
    let m = apply_shift_rot(mov, 0.0, 0.0, ang);
    let m = lum(&m);
    let g = ndimage::gaussian(&m, 6.0);
    let m = m.zip_map(&g, |a, b| a - b);
    phase_corr(r_hp, &m).2
}

/// Rotation (± max_deg) qui recale `mov` sur `ref_`. Méthode « auto »
/// (log-polaire + raffinement) ou « sweep » (balayage).
pub fn estimate_rotation(ref_: &Image, mov: &Image, max_deg: f64, step: f64, method: &str) -> f64 {
    let r = lum(ref_);
    let g = ndimage::gaussian(&r, 6.0);
    let r = r.zip_map(&g, |a, b| a - b);
    let mut cand: Option<f64> = None;
    if method == "auto" {
        let (a, _s, _resp) = logpolar_rotation(ref_, mov);
        if a.is_finite() && a.abs() <= max_deg + step {
            cand = Some(a);
        }
    }
    let mut cand = match cand {
        Some(c) => c,
        None => {
            let n = ((2.0 * max_deg) / step + 1e-9).floor() as usize + 1;
            let angs: Vec<f64> = (0..n).map(|i| -max_deg + i as f64 * step).collect();
            let scores: Vec<f64> = angs.par_iter().map(|&a| rot_score(&r, mov, a)).collect();
            let mut best = (f64::NEG_INFINITY, 0.0);
            for (a, s) in angs.iter().zip(scores) {
                if s > best.0 {
                    best = (s, *a);
                }
            }
            best.1
        }
    };
    let mut hstep = step;
    for _ in 0..2 {
        let (s0, (sm, sp)) = rayon::join(
            || rot_score(&r, mov, cand),
            || rayon::join(|| rot_score(&r, mov, cand - hstep), || rot_score(&r, mov, cand + hstep)),
        );
        if sm > s0 && sm >= sp {
            cand -= hstep;
            continue;
        }
        if sp > s0 && sp > sm {
            cand += hstep;
            continue;
        }
        let den = sm - 2.0 * s0 + sp;
        if den.abs() > 1e-12 {
            cand += (0.5 * (sm - sp) / den).clamp(-0.5, 0.5) * hstep;
        }
        hstep /= 4.0;
    }
    cand.clamp(-max_deg, max_deg)
}

/// Aligne toutes les frames sur la référence (translation, + rotation).
pub fn align_stack(ref_: &Image, frames: &[Image], with_rotation: bool, max_deg: f64) -> Vec<Image> {
    frames
        .par_iter()
        .map(|f| {
            let ang = if with_rotation { estimate_rotation(ref_, f, max_deg, 0.25, "auto") } else { 0.0 };
            let rot = if ang != 0.0 { apply_shift_rot(f, 0.0, 0.0, ang) } else { f.clone() };
            let (dx, dy, _) = phase_corr(ref_, &rot);
            if dx != 0.0 || dy != 0.0 {
                apply_shift(&rot, dx, dy)
            } else {
                rot
            }
        })
        .collect()
}

/// Dérotation du champ : taux fixe (°/min × temps) ou estimation auto
/// (sous-échantillon + interpolation linéaire).
pub fn derotate(frames: &[Image], rate_deg_per_min: Option<f64>, interval_s: Option<f64>, auto_max_deg: f64, ref_index: usize) -> Vec<Image> {
    let n = frames.len();
    if n == 0 {
        return vec![];
    }
    let ref_ = &frames[ref_index.min(n - 1)];
    let fixed = matches!((rate_deg_per_min, interval_s), (Some(r), Some(_)) if r != 0.0);
    let angles: Vec<f64> = if auto_max_deg > 0.0 && !fixed {
        let step = (n / 12).max(1);
        let mut sample: Vec<usize> = (0..n).step_by(step).collect();
        if !sample.contains(&(n - 1)) {
            sample.push(n - 1);
        }
        let vals: Vec<f64> = sample.par_iter().map(|&i| estimate_rotation(ref_, &frames[i], auto_max_deg, 0.25, "auto")).collect();
        let xs: Vec<f64> = sample.iter().map(|&i| i as f64).collect();
        (0..n).map(|i| stats::interp(i as f64, &xs, &vals)).collect()
    } else {
        (0..n)
            .map(|i| match (rate_deg_per_min, interval_s) {
                (Some(r), Some(dt)) if r != 0.0 => r * (i as f64 - ref_index as f64) * dt / 60.0,
                _ => 0.0,
            })
            .collect()
    };
    frames.par_iter().zip(angles.par_iter()).map(|(f, &a)| apply_shift_rot(f, 0.0, 0.0, a)).collect()
}

/// Recale R et B sur G (aberration chromatique planétaire).
pub fn align_channels(rgb: &Image, max_shift: Option<f64>) -> Image {
    if !rgb.is_color() {
        return rgb.clone();
    }
    let g = rgb.channel(1);
    let mut out = rgb.clone();
    for idx in [0usize, 2] {
        let ch = rgb.channel(idx);
        let (dx, dy, _) = phase_corr(&g, &ch);
        if let Some(ms) = max_shift {
            if (dx * dx + dy * dy).sqrt() > ms {
                continue;
            }
        }
        out.set_channel(idx, &apply_shift(&ch, dx, dy));
    }
    out
}

/// Contours (limbe planétaire) pour un recalage insensible au contraste doux.
pub fn edges(img: &Image) -> Image {
    let mono = lum(img);
    let st = display::autostretch(&mono);
    let u8v = features::to_u8(&st);
    let e = features::canny(&u8v, mono.w, mono.h, 40.0, 140.0);
    Image::from_vec(mono.w, mono.h, 1, e.iter().map(|&v| v as f32).collect())
}

/// Aligne avec l'algorithme choisi (référence = frame centrale).
pub fn align_frames(frames: &[Image], method: &str) -> Vec<Image> {
    let n = frames.len();
    if n == 0 {
        return vec![];
    }
    let ri = n / 2;
    let ref_ = &frames[ri];
    if method.starts_with("Étoiles") {
        return frames
            .par_iter()
            .enumerate()
            .map(|(i, f)| if i == ri { f.clone() } else { align_by_stars(ref_, f) })
            .collect();
    }
    let use_edges = method.starts_with("Limbe");
    let ref_p = if use_edges { edges(ref_) } else { ref_.clone() };
    frames
        .par_iter()
        .enumerate()
        .map(|(i, f)| {
            if i == ri {
                return f.clone();
            }
            let mov_p = if use_edges { edges(f) } else { f.clone() };
            let (dx, dy, _) = phase_corr(&ref_p, &mov_p);
            apply_shift(f, dx, dy)
        })
        .collect()
}

// ------------------------------------------------ similitude (étoiles)
/// Moindres carrés similitude 2D (complexes) : (M, t, résidu médian).
pub fn similarity_ls(p: &[Pt], s: &[Pt], flip: bool) -> Option<([[f64; 2]; 2], [f64; 2], f64)> {
    let n = p.len() as f64;
    let zp: Vec<(f64, f64)> = p.iter().map(|&(x, y)| if flip { (x, -y) } else { (x, y) }).collect();
    let mp = (zp.iter().map(|z| z.0).sum::<f64>() / n, zp.iter().map(|z| z.1).sum::<f64>() / n);
    let ms = (s.iter().map(|z| z.0).sum::<f64>() / n, s.iter().map(|z| z.1).sum::<f64>() / n);
    let (mut den, mut ar, mut ai) = (0.0, 0.0, 0.0);
    for (a, b) in zp.iter().zip(s) {
        let (dpx, dpy) = (a.0 - mp.0, a.1 - mp.1);
        let (dsx, dsy) = (b.0 - ms.0, b.1 - ms.1);
        den += dpx * dpx + dpy * dpy;
        // conj(dp)·ds
        ar += dpx * dsx + dpy * dsy;
        ai += dpx * dsy - dpy * dsx;
    }
    if den < 1e-9 {
        return None;
    }
    let (alr, ali) = (ar / den, ai / den);
    let (br, bi) = (ms.0 - (alr * mp.0 - ali * mp.1), ms.1 - (alr * mp.1 + ali * mp.0));
    let mut res: Vec<f64> = zp
        .iter()
        .zip(s)
        .map(|(a, b)| ((alr * a.0 - ali * a.1 + br - b.0).powi(2) + (alr * a.1 + ali * a.0 + bi - b.1).powi(2)).sqrt())
        .collect();
    let med = stats::median_f64_mut(&mut res);
    // comme le programme d'origine : la matrice (rotation + échelle) est
    // appliquée aux points non retournés — l'hypothèse « miroir » fournit
    // simplement d'autres rotations candidates.
    let m = [[alr, -ali], [ali, alr]];
    Some((m, [br, bi], med))
}

fn apply_m(m: &[[f64; 2]; 2], t: &[f64; 2], p: Pt) -> Pt {
    (m[0][0] * p.0 + m[0][1] * p.1 + t[0], m[1][0] * p.0 + m[1][1] * p.1 + t[1])
}

fn nearest(q: Pt, pts: &[Pt]) -> (usize, f64) {
    let mut best = (0usize, f64::INFINITY);
    for (k, p) in pts.iter().enumerate() {
        let d = ((q.0 - p.0).powi(2) + (q.1 - p.1).powi(2)).sqrt();
        if d < best.1 {
            best = (k, d);
        }
    }
    best
}

type PairBest = ((usize, f64), Vec<usize>, Vec<bool>);

fn pair_hypotheses(sm: &[Pt], sr: &[Pt], tol: f64) -> Option<PairBest> {
    let (ni, nj) = (5.min(sm.len()), 12.min(sr.len()));
    let mut best: Option<PairBest> = None;
    for flip in [false, true] {
        for i in 0..ni {
            for j in i + 1..ni {
                for a in 0..nj {
                    for b in a + 1..nj {
                        let Some((m, t, _)) = similarity_ls(&[sm[i], sm[j]], &[sr[a], sr[b]], flip) else { continue };
                        let mut matches = Vec::with_capacity(sm.len());
                        let mut inl = Vec::with_capacity(sm.len());
                        let mut cnt = 0usize;
                        let mut dsum = 0.0;
                        for &p in sm {
                            let (k, d) = nearest(apply_m(&m, &t, p), sr);
                            matches.push(k);
                            let ok = d < tol;
                            if ok {
                                cnt += 1;
                                dsum += d;
                            }
                            inl.push(ok);
                        }
                        let score = (cnt, if cnt > 0 { -dsum / cnt as f64 } else { 0.0 });
                        let better = match &best {
                            None => true,
                            Some(b) => score.0 > b.0 .0 || (score.0 == b.0 .0 && score.1 > b.0 .1),
                        };
                        if better {
                            best = Some((score, matches, inl));
                        }
                    }
                }
            }
        }
    }
    best
}

fn stars_mono(img: &Image, n: usize) -> Vec<Pt> {
    stars::detect_stars(&img.mean_lum(), n)
}

fn phase_fallback(ref_: &Image, mov: &Image) -> Image {
    let (dx, dy, _) = phase_corr(ref_, mov);
    apply_shift(mov, dx, dy)
}

/// Alignement en similitude par appariement de paires d'étoiles (direct /
/// miroir), moindres carrés sur les inliers ; repli sur la phase.
pub fn align_similarity(ref_: &Image, mov: &Image, max_stars: usize) -> Image {
    let sr = stars_mono(ref_, max_stars);
    let sm = stars_mono(mov, max_stars);
    if sr.len() < 3 || sm.len() < 3 {
        return phase_fallback(ref_, mov);
    }
    let Some((score, matches, inl)) = pair_hypotheses(&sm, &sr, 3.0) else { return phase_fallback(ref_, mov) };
    if score.0 < 3 {
        return phase_fallback(ref_, mov);
    }
    let keep: Vec<usize> = (0..sm.len()).filter(|&i| inl[i]).collect();
    let p: Vec<Pt> = keep.iter().map(|&i| sm[i]).collect();
    let s: Vec<Pt> = keep.iter().map(|&i| sr[matches[i]]).collect();
    let Some((m, t, _)) = similarity_ls(&p, &s, false) else { return phase_fallback(ref_, mov) };
    let aff: Affine = [[m[0][0], m[0][1], t[0]], [m[1][0], m[1][1], t[1]]];
    geom::warp(mov, &aff)
}

// --------------------------- plate solving par image + distorsions polynomiales
pub fn poly_terms(x: f64, y: f64, order: usize) -> Vec<f64> {
    let mut t = Vec::new();
    for dy in 0..=order {
        for dx in 0..=(order - dy) {
            t.push(x.powi(dx as i32) * y.powi(dy as i32));
        }
    }
    t
}

fn poly_eval(c: &[f64], x: f64, y: f64, order: usize) -> f64 {
    poly_terms(x, y, order).iter().zip(c).map(|(a, b)| a * b).sum()
}

fn ridge_solve(p: &[Pt], s: &[Pt], order: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    let nt = (order + 1) * (order + 2) / 2;
    let n = p.len();
    let mut a = DMatrix::<f64>::zeros(n, nt);
    for (i, &(x, y)) in p.iter().enumerate() {
        for (k, v) in poly_terms(x, y, order).into_iter().enumerate() {
            a[(i, k)] = v;
        }
    }
    let lam = 1e-6 * n as f64;
    let ata = a.transpose() * &a + DMatrix::<f64>::identity(nt, nt) * lam;
    let bx = DVector::from_iterator(n, s.iter().map(|q| q.0));
    let by = DVector::from_iterator(n, s.iter().map(|q| q.1));
    let lu = ata.lu();
    let cx = lu.solve(&(a.transpose() * bx))?;
    let cy = lu.solve(&(a.transpose() * by))?;
    Some((cx.iter().cloned().collect(), cy.iter().cloned().collect()))
}

#[derive(Clone, Debug)]
pub struct PolyFit {
    pub cx: Vec<f64>,
    pub cy: Vec<f64>,
    pub resid: f64,
    pub n: usize,
    pub order: usize,
}

/// Appariement d'étoiles mov -> ref puis ajustement polynomial (degré `order`).
pub fn fit_poly_transform(ref_: &Image, mov: &Image, order: usize, max_stars: usize) -> Result<PolyFit> {
    let sr = stars_mono(ref_, max_stars);
    let sm = stars_mono(mov, max_stars);
    if sr.len() < 6 || sm.len() < 6 {
        bail!("Pas assez d'étoiles pour le plate solving par image");
    }
    let Some((score, matches, inl)) = pair_hypotheses(&sm, &sr, 8.0) else { bail!("Appariement insuffisant pour le poly (≥6 inliers requis)") };
    if score.0 < 6 {
        bail!("Appariement insuffisant pour le poly (≥6 inliers requis)");
    }
    let mut keep_mov: Vec<usize> = (0..sm.len()).filter(|&i| inl[i]).collect();
    let mut keep_ref: Vec<usize> = keep_mov.iter().map(|&i| matches[i]).collect();
    let mut order_eff = order.min(1usize.max(((-3.0 + (4.0 * keep_mov.len() as f64 + 1.0).sqrt()) / 2.0) as usize));
    let mut cx: Vec<f64> = vec![];
    let mut cy: Vec<f64> = vec![];
    for thr in [8.0f64, 5.0, 3.0] {
        let nterm = (order_eff + 1) * (order_eff + 2) / 2;
        if keep_mov.len() < 2 * nterm {
            let v = ((-1.0 + (1.0 + 2.0 * keep_mov.len() as f64).sqrt()) - 3.0) / 2.0;
            order_eff = 1usize.max(if v > 0.0 { v as usize } else { 0 });
        }
        let p: Vec<Pt> = keep_mov.iter().map(|&i| sm[i]).collect();
        let s: Vec<Pt> = keep_ref.iter().map(|&j| sr[j]).collect();
        let Some((ncx, ncy)) = ridge_solve(&p, &s, order_eff) else { bail!("Poly : système singulier") };
        cx = ncx;
        cy = ncy;
        let rr: Vec<f64> = p
            .iter()
            .zip(&s)
            .map(|(a, b)| ((poly_eval(&cx, a.0, a.1, order_eff) - b.0).powi(2) + (poly_eval(&cy, a.0, a.1, order_eff) - b.1).powi(2)).sqrt())
            .collect();
        let med = stats::median_f64(&rr);
        let dev: Vec<f64> = rr.iter().map(|r| (r - med).abs()).collect();
        let mad = 1.4826 * stats::median_f64(&dev) + 1e-6;
        let good: Vec<bool> = rr.iter().map(|&r| r <= med + (3.0 * mad).max(0.5)).collect();
        let ng = good.iter().filter(|&&g| g).count();
        if ng >= 6 && ng < keep_mov.len() {
            keep_mov = keep_mov.iter().zip(&good).filter(|x| *x.1).map(|x| *x.0).collect();
            keep_ref = keep_ref.iter().zip(&good).filter(|x| *x.1).map(|x| *x.0).collect();
        }
        if thr == 3.0 {
            break;
        }
        // ré-association mutuelle (type ICP)
        let q: Vec<Pt> = sm.iter().map(|p| (poly_eval(&cx, p.0, p.1, order_eff), poly_eval(&cy, p.0, p.1, order_eff))).collect();
        let cand: Vec<(usize, f64)> = q.iter().map(|&qq| nearest(qq, &sr)).collect();
        let back: Vec<usize> = sr.iter().map(|&r| nearest(r, &q).0).collect();
        let pairs: Vec<(usize, usize)> = (0..sm.len()).filter(|&i| cand[i].1 < thr && back[cand[i].0] == i).map(|i| (i, cand[i].0)).collect();
        if pairs.len() >= 6 {
            keep_mov = pairs.iter().map(|p| p.0).collect();
            keep_ref = pairs.iter().map(|p| p.1).collect();
        }
        if keep_mov.len() < 6 {
            bail!("Poly : raffinement insuffisant");
        }
    }
    if cx.is_empty() || keep_mov.len() < 6 {
        bail!("Poly : raffinement insuffisant");
    }
    let r2: Vec<f64> = keep_mov
        .iter()
        .zip(&keep_ref)
        .map(|(&i, &j)| (poly_eval(&cx, sm[i].0, sm[i].1, order_eff) - sr[j].0).powi(2) + (poly_eval(&cy, sm[i].0, sm[i].1, order_eff) - sr[j].1).powi(2))
        .collect();
    let resid = stats::median_f64(&r2).sqrt();
    Ok(PolyFit { cx, cy, resid, n: keep_mov.len(), order: order_eff })
}

fn poly_inverse(ref_: &Image, mov: &Image, direct: &PolyFit, max_stars: usize) -> Result<PolyFit> {
    let order = direct.order;
    let sr = stars_mono(ref_, max_stars);
    let sm = stars_mono(mov, max_stars);
    let mut p = Vec::new();
    let mut s = Vec::new();
    for &m in &sm {
        let q = (poly_eval(&direct.cx, m.0, m.1, order), poly_eval(&direct.cy, m.0, m.1, order));
        let (k, d) = nearest(q, &sr);
        if d < 3.0 {
            p.push(sr[k]);
            s.push(m);
        }
    }
    if p.len() < 6 {
        bail!("poly inverse : appariement insuffisant");
    }
    let Some((cx, cy)) = ridge_solve(&p, &s, order) else { bail!("poly inverse : système singulier") };
    let r2: Vec<f64> = p.iter().zip(&s).map(|(a, b)| (poly_eval(&cx, a.0, a.1, order) - b.0).powi(2) + (poly_eval(&cy, a.0, a.1, order) - b.1).powi(2)).collect();
    Ok(PolyFit { resid: stats::median_f64(&r2).sqrt(), cx, cy, n: p.len(), order })
}

/// Plate solving par image : distorsions polynomiales (degré 1..order
/// choisi au résidu médian) ; repli similitude puis phase.
pub fn align_poly(ref_: &Image, mov: &Image, order: usize, max_stars: usize) -> Image {
    let mut fits = Vec::new();
    for o in 1..=order.max(1) {
        match fit_poly_transform(ref_, mov, o, max_stars) {
            Ok(f) => fits.push(f),
            Err(_) => return align_similarity(ref_, mov, 40),
        }
    }
    let best = fits.into_iter().min_by(|a, b| a.resid.total_cmp(&b.resid)).unwrap();
    let inv = match poly_inverse(ref_, mov, &best, max_stars) {
        Ok(i) => i,
        Err(_) => return align_similarity(ref_, mov, 40),
    };
    geom::remap_fn(mov, mov.w, mov.h, Interp::Cubic, Border::Reflect, |x, y| {
        let (xf, yf) = (x as f64, y as f64);
        (poly_eval(&inv.cx, xf, yf, inv.order), poly_eval(&inv.cy, xf, yf, inv.order))
    })
}

// ------------------------------------------- transformation globale (drizzle)
pub fn warp(img: &Image, m: &Affine) -> Image {
    geom::warp(img, m)
}

/// Similitude mov -> ref par triangles d'étoiles (équivalent d'astroalign).
pub fn star_transform(ref_: &Image, mov: &Image) -> Option<Affine> {
    let sr = stars_mono(ref_, 60);
    let sm = stars_mono(mov, 60);
    if sr.len() < 4 || sm.len() < 4 {
        return None;
    }
    let m = starmatch::match_points(&sm, &sr, false, 2.0, None, true)?;
    let need = 4usize.max((0.2 * sm.len().min(sr.len()) as f64) as usize);
    if m.n_inliers < need {
        return None;
    }
    // affinement final : moindres carrés sur les paires à moins de 1 px
    let mut sim = m.sim;
    for _ in 0..2 {
        let (ok, j, _) = starmatch::inliers(&sim, &sm, &sr, 1.0);
        let p: Vec<Pt> = (0..sm.len()).filter(|&i| ok[i]).map(|i| sm[i]).collect();
        let q: Vec<Pt> = (0..sm.len()).filter(|&i| ok[i]).map(|i| sr[j[i]]).collect();
        if p.len() < need {
            break;
        }
        sim = starmatch::umeyama(&p, &q, false);
    }
    let a = sim.to_affine();
    if geom::is_finite(&a) {
        Some(a)
    } else {
        None
    }
}

/// Matrice 2×3 envoyant les coordonnées de `mov` sur celles de `ref_` :
/// similitude par étoiles, sinon phase (+ rotation estimée si demandé).
pub fn estimate_transform(ref_: &Image, mov: &Image, with_rotation: bool) -> Affine {
    if std::ptr::eq(ref_, mov) || ref_ == mov {
        return geom::IDENTITY;
    }
    let r = lum(ref_);
    let m = lum(mov);
    if let Some(a) = star_transform(&r, &m) {
        return a;
    }
    let ang = if with_rotation { estimate_rotation(&r, &m, 2.0, 0.25, "auto") } else { 0.0 };
    let rot = if ang != 0.0 { apply_shift_rot(&m, 0.0, 0.0, ang) } else { m.clone() };
    let (dx, dy, _) = phase_corr(&r, &rot);
    let mut a = geom::rotation_matrix(m.w as f64 / 2.0, m.h as f64 / 2.0, ang, 1.0);
    a[0][2] += dx;
    a[1][2] += dy;
    a
}

/// Alignement par étoiles avec replis (similitude par paires, phase).
pub fn align_by_stars(ref_: &Image, mov: &Image) -> Image {
    if let Some(a) = star_transform(&lum(ref_), &lum(mov)) {
        return geom::warp(mov, &a);
    }
    align_similarity(ref_, mov, 40)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stars::tests::field;
    fn synth(dx: f64, dy: f64) -> Image {
        let st: Vec<(f64, f64, f64)> = [(60.0, 40.0), (30.0, 80.0), (100.0, 64.0), (20.0, 20.0), (90.0, 100.0), (45.0, 110.0)]
            .iter()
            .enumerate()
            .map(|(k, &(x, y))| (x + dx, y + dy, 0.8 - 0.1 * k as f64))
            .collect();
        let img = field(128, 128, &st, 3.0, 0.05);
        Image::from_fn(128, 128, |x, y| img.get(x, y, 0) + 0.05 * x as f32 / 128.0)
    }
    #[test]
    fn phase_finds_shift() {
        let r = synth(0.0, 0.0);
        let m = synth(4.0, -7.0);
        let (dx, dy, _) = phase_corr(&r, &m);
        assert!((dx + 4.0).abs() < 0.6 && (dy - 7.0).abs() < 0.6, "{} {}", dx, dy);
        let al = apply_shift(&m, dx, dy);
        let err = al.data.iter().zip(&r.data).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / r.data.len() as f64;
        assert!(err < 0.02, "{}", err);
    }
    #[test]
    fn star_transform_works() {
        let r = synth(0.0, 0.0);
        let m = synth(3.3, 2.1);
        let a = star_transform(&r, &m).expect("transform");
        assert!((a[0][2] + 3.3).abs() < 0.2 && (a[1][2] + 2.1).abs() < 0.2, "{:?}", a);
    }
}
