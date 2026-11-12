//! Analyse des brutes (esprit SubframeSelector / PSF de Siril) : étoiles,
//! FWHM et excentricité (moments adaptatifs), fond, bruit, SNR ; rejet des
//! images aberrantes ; poids d'empilement.

use crate::image::Image;
use crate::io;
use crate::stars;
use crate::stats;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

pub const WEIGHT_MODES: [&str; 4] = ["Aucune", "SNR", "FWHM", "Mixte"];

#[derive(Clone, Debug, Default)]
pub struct FrameStats {
    pub path: String,
    pub stars: usize,
    pub measured: usize,
    pub fwhm: f64,
    pub ecc: f64,
    pub background: f64,
    pub noise: f64,
    pub signal: f64,
    pub snr: f64,
    pub error: String,
    pub keep: bool,
    pub reasons: Vec<String>,
    pub weight: f64,
}

fn bin2(a: &Image) -> Image {
    let (w, h) = (a.w / 2, a.h / 2);
    let mut out = Image::new(w, h, 1);
    for y in 0..h {
        for x in 0..w {
            let s = a.data[2 * y * a.w + 2 * x] + a.data[2 * y * a.w + 2 * x + 1] + a.data[(2 * y + 1) * a.w + 2 * x] + a.data[(2 * y + 1) * a.w + 2 * x + 1];
            out.data[y * w + x] = s / 4.0;
        }
    }
    out
}

fn eig2(c: [[f64; 2]; 2]) -> (f64, f64) {
    // valeurs propres d'une matrice symétrique 2×2 (croissantes) ; la plus
    // petite est déduite du déterminant (précision quand elle est ≈ 0)
    let (a, b, d) = (c[0][0], 0.5 * (c[0][1] + c[1][0]), c[1][1]);
    let half = 0.5 * (a + d);
    let disc = (0.25 * (a - d) * (a - d) + b * b).sqrt();
    let big = if half >= 0.0 { half + disc } else { half - disc };
    let det = a * d - b * b;
    let small = if big != 0.0 { det / big } else { 0.0 };
    if big >= small {
        (small, big)
    } else {
        (big, small)
    }
}

/// Moments adaptatifs elliptiques (HSM) : (fwhm, excentricité, x, y).
pub fn star_shape(img: &Image, x: f64, y: f64, bg: f64, s0: f64, iters: usize) -> Option<(f64, f64, f64, f64)> {
    let (w, h) = (img.w as isize, img.h as isize);
    let mut c = [[s0 * s0, 0.0], [0.0, s0 * s0]];
    let (mut cx, mut cy) = (x, y);
    for _ in 0..iters {
        let (emin, emax) = eig2(c);
        if emin <= 0.0 || emax > 900.0 {
            return None;
        }
        let r = 4isize.max((4.0 * emax.sqrt()).ceil() as isize);
        let (x0, x1) = (cx.round_ties_even() as isize - r, cx.round_ties_even() as isize + r + 1);
        let (y0, y1) = (cy.round_ties_even() as isize - r, cy.round_ties_even() as isize + r + 1);
        if x0 < 0 || y0 < 0 || x1 > w || y1 > h {
            return None;
        }
        let det = c[0][0] * c[1][1] - c[0][1] * c[1][0];
        let ci = [[c[1][1] / det, -c[0][1] / det], [-c[1][0] / det, c[0][0] / det]];
        let mut wts = Vec::with_capacity(((x1 - x0) * (y1 - y0)) as usize);
        let (mut tot, mut sx, mut sy) = (0.0f64, 0.0f64, 0.0f64);
        for yy in y0..y1 {
            for xx in x0..x1 {
                let v = img.data[yy as usize * img.w + xx as usize] as f64 - bg;
                let (dx, dy) = (xx as f64 - cx, yy as f64 - cy);
                let q = ci[0][0] * dx * dx + 2.0 * ci[0][1] * dx * dy + ci[1][1] * dy * dy;
                let wt = (-0.5 * q).exp() * v;
                wts.push((xx as f64, yy as f64, wt));
                tot += wt;
                sx += wt * xx as f64;
                sy += wt * yy as f64;
            }
        }
        if tot <= 0.0 {
            return None;
        }
        let (ncx, ncy) = (sx / tot, sy / tot);
        let (mut mxx, mut mxy, mut myy) = (0.0, 0.0, 0.0);
        for &(xx, yy, wt) in &wts {
            let (dx, dy) = (xx - ncx, yy - ncy);
            mxx += wt * dx * dx;
            mxy += wt * dx * dy;
            myy += wt * dy * dy;
        }
        let newc = [[2.0 * mxx / tot, 2.0 * mxy / tot], [2.0 * mxy / tot, 2.0 * myy / tot]];
        if eig2(newc).0 <= 0.0 {
            return None;
        }
        let diff = (newc[0][0] - c[0][0]).abs().max((newc[0][1] - c[0][1]).abs()).max((newc[1][1] - c[1][1]).abs());
        let done = diff < 1e-4 && (ncx - cx).abs() < 1e-3 && (ncy - cy).abs() < 1e-3;
        c = [[0.5 * (c[0][0] + newc[0][0]), 0.5 * (c[0][1] + newc[0][1])], [0.5 * (c[1][0] + newc[1][0]), 0.5 * (c[1][1] + newc[1][1])]];
        cx = ncx;
        cy = ncy;
        if done {
            c = newc;
            break;
        }
    }
    let (l2, l1) = eig2(c);
    if l2 <= 0.0 {
        return None;
    }
    let sigma = ((l1 + l2) / 2.0).sqrt();
    let ecc = (1.0 - l2 / l1).max(0.0).sqrt();
    Some((2.3548 * sigma, ecc, cx, cy))
}

/// Mesures d'une image (mono, couleur, ou CFA si `cfa`).
pub fn analyze_image(img: &Image, cfa: bool, max_stars: usize) -> FrameStats {
    let mut a = img.mean_lum();
    let mut scale = 1.0;
    if cfa {
        a = bin2(&a);
        scale = 2.0;
    }
    let bg = stats::median(&a.data) as f64;
    let noise = 1.4826 * stats::mad_about(&a.data, bg as f32) as f64 + 1e-9;
    let st = stars::detect_stars_opts(&a, max_stars * 2, 3.0, 6.0);
    let sat = stats::percentile(&a.data, 99.99) as f64;
    let mut shapes = Vec::new();
    let mut peaks = Vec::new();
    for &(x, y) in &st {
        let (xi, yi) = (x.round_ties_even() as isize, y.round_ties_even() as isize);
        if xi < 0 || yi < 0 || xi >= a.w as isize || yi >= a.h as isize {
            continue;
        }
        let pk = a.data[yi as usize * a.w + xi as usize] as f64;
        if pk >= 0.98 * sat.max(1e-6) && pk > 0.9 {
            continue;
        }
        match star_shape(&a, x, y, bg, 1.5, 20) {
            Some(r) if r.0 > 0.5 && r.0 < 40.0 => {
                shapes.push(r);
                peaks.push(pk - bg);
            }
            _ => continue,
        }
        if shapes.len() >= max_stars {
            break;
        }
    }
    let (fwhm, ecc, signal) = if !shapes.is_empty() {
        (
            stats::median_f64(&shapes.iter().map(|s| s.0).collect::<Vec<_>>()) * scale,
            stats::median_f64(&shapes.iter().map(|s| s.1).collect::<Vec<_>>()),
            stats::median_f64(&peaks),
        )
    } else {
        (f64::NAN, f64::NAN, 0.0)
    };
    FrameStats {
        stars: st.len(),
        measured: shapes.len(),
        fwhm,
        ecc,
        background: bg,
        noise,
        signal,
        snr: if noise > 0.0 { signal / noise } else { 0.0 },
        keep: true,
        weight: 1.0,
        ..Default::default()
    }
}

/// Analyse une liste de fichiers (CFA mesuré après binning 2×2).
pub fn analyze_paths(paths: &[PathBuf], bayer: Option<&str>, progress: Option<&(dyn Fn(usize, usize, &str) + Sync)>) -> Vec<FrameStats> {
    let n = paths.len();
    let done = std::sync::atomic::AtomicUsize::new(0);
    paths
        .par_iter()
        .map(|p| {
            let mut r = match io::load_cfa(p) {
                Ok((img, pattern)) => {
                    let cfa = img.c == 1 && (pattern.is_some() || bayer.is_some());
                    analyze_image(&img, cfa, 60)
                }
                Err(e) => FrameStats {
                    fwhm: f64::NAN,
                    ecc: f64::NAN,
                    background: f64::NAN,
                    noise: f64::NAN,
                    error: e.to_string(),
                    keep: true,
                    weight: 1.0,
                    ..Default::default()
                },
            };
            r.path = p.to_string_lossy().to_string();
            let d = done.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
            if let Some(pr) = progress {
                pr(d, n, "Analyse des brutes");
            }
            r
        })
        .collect()
}

fn robust(vals: &[f64]) -> (f64, f64) {
    let v: Vec<f64> = vals.iter().cloned().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let med = stats::median_f64(&v);
    let dev: Vec<f64> = v.iter().map(|x| (x - med).abs()).collect();
    (med, 1.4826 * stats::median_f64(&dev) + 1e-9)
}

/// Ajoute `keep` et `reasons` à chaque ligne.
pub fn flag_outliers(rows: &mut [FrameStats], k: f64, max_ecc: f64, min_star_frac: f64) {
    let (f_med, f_sd) = robust(&rows.iter().map(|r| r.fwhm).collect::<Vec<_>>());
    let (b_med, b_sd) = robust(&rows.iter().map(|r| r.background).collect::<Vec<_>>());
    let n_med = if rows.is_empty() { 0.0 } else { stats::median_f64(&rows.iter().map(|r| r.stars as f64).collect::<Vec<_>>()) };
    for r in rows.iter_mut() {
        let mut reasons = Vec::new();
        if !r.error.is_empty() {
            reasons.push("illisible".to_string());
        } else {
            if r.fwhm.is_finite() && r.fwhm > f_med + k * f_sd.max(0.05 * f_med) {
                reasons.push(format!("FWHM {:.2} px (médiane {:.2})", r.fwhm, f_med));
            }
            if !r.fwhm.is_finite() {
                reasons.push("aucune étoile mesurable".into());
            }
            if r.background.is_finite() && r.background > b_med + k * b_sd.max(0.02 * b_med.abs()) {
                reasons.push("fond élevé (nuage, lune, lumière)".into());
            }
            if n_med > 0.0 && (r.stars as f64) < min_star_frac * n_med {
                reasons.push(format!("{} étoiles (médiane {:.0})", r.stars, n_med));
            }
            if r.ecc.is_finite() && r.ecc > max_ecc {
                reasons.push(format!("étoiles allongées (excentricité {:.2})", r.ecc));
            }
        }
        r.keep = reasons.is_empty();
        r.reasons = reasons;
    }
}

pub fn flag_outliers_default(rows: &mut [FrameStats]) {
    flag_outliers(rows, 3.0, 0.65, 0.5)
}

/// Poids normalisés (moyenne 1 sur les images gardées).
pub fn compute_weights(rows: &[FrameStats], mode: &str) -> Vec<f64> {
    if !WEIGHT_MODES.contains(&mode) || mode == "Aucune" {
        return rows.iter().map(|r| if r.keep { 1.0 } else { 0.0 }).collect();
    }
    let f_med = robust(&rows.iter().map(|r| r.fwhm).collect::<Vec<_>>()).0;
    let mut w: Vec<f64> = rows
        .iter()
        .map(|r| {
            if !r.keep {
                return 0.0;
            }
            let snr = r.snr.max(1e-6);
            let fw = if r.fwhm.is_finite() && r.fwhm > 0.0 { r.fwhm } else { f_med };
            let fw = if fw.is_finite() && fw > 0.0 { fw } else { 1.0 };
            match mode {
                "SNR" => snr * snr,
                "FWHM" => 1.0 / (fw * fw),
                _ => snr * snr / (fw * fw),
            }
        })
        .collect();
    let kept: Vec<f64> = w.iter().cloned().filter(|&x| x > 0.0).collect();
    if !kept.is_empty() {
        let m = kept.iter().sum::<f64>() / kept.len() as f64;
        for x in w.iter_mut() {
            *x /= m;
        }
    }
    w
}

fn fmt_f(v: f64, width: usize, prec: usize) -> String {
    if v.is_nan() {
        format!("{:>width$}", "nan", width = width)
    } else {
        format!("{:>width$.prec$}", v, width = width, prec = prec)
    }
}

/// Tableau texte des mesures.
pub fn report(rows: &[FrameStats]) -> String {
    let mut lines = vec![format!(
        "{:<32} {:>7} {:>6} {:>5} {:>7} {:>7} {:>7} {:>6}  Statut",
        "Fichier", "Étoiles", "FWHM", "Exc.", "Fond", "Bruit", "SNR", "Poids"
    )];
    for r in rows {
        let name: String = Path::new(&r.path).file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default().chars().take(32).collect();
        let status = if r.keep { "garder".to_string() } else { format!("rejeter : {}", r.reasons.join("; ")) };
        lines.push(format!(
            "{:<32} {:>7} {} {} {} {} {} {}  {}",
            name,
            r.stars,
            fmt_f(r.fwhm, 6, 2),
            fmt_f(r.ecc, 5, 2),
            fmt_f(r.background, 7, 4),
            fmt_f(r.noise, 7, 5),
            fmt_f(r.snr, 7, 1),
            fmt_f(r.weight, 6, 2),
            status
        ));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stars::tests::field;
    #[test]
    fn fwhm_measured() {
        let st: Vec<(f64, f64, f64)> = (0..12).map(|k| (15.0 + (k % 4) as f64 * 30.0, 15.0 + (k / 4) as f64 * 30.0, 0.5)).collect();
        let img = field(128, 100, &st, 3.5, 0.05);
        let r = analyze_image(&img, false, 60);
        assert!((r.fwhm - 3.5).abs() < 0.3, "{}", r.fwhm);
        assert!(r.ecc < 0.2);
    }
}
