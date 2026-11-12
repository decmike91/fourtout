//! Calibration couleur & neutralisation du fond, SCNR vert, aberration
//! chromatique latérale, calibration photométrique (PCC, couleurs Gaia BP/RP).

use crate::astrometry;
use crate::geom::{self, Interp};
use crate::image::Image;
use crate::ndimage::Border;
use crate::stars;
use crate::stats;
use crate::wcs::Wcs;
use anyhow::{bail, Result};
use std::path::Path;

/// Masque du fond : pixels sous le percentile de luminance.
pub fn background_mask(img: &Image, percentile: f64) -> Vec<bool> {
    let l = img.mean_lum();
    let thr = stats::percentile(&l.data, percentile.clamp(1.0, 99.0));
    l.data.iter().map(|&v| v <= thr).collect()
}

fn masked_medians(img: &Image, mask: &[bool]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for (k, o) in out.iter_mut().enumerate() {
        let v: Vec<f32> = (0..mask.len()).filter(|&i| mask[i]).map(|i| img.data[i * img.c + k]).collect();
        *o = stats::median(&v) as f64;
    }
    out
}

/// Égalise les médianes des canaux (fond ou global) vers les ratios `white`.
pub fn color_calibrate(img: &Image, white: [f64; 3], use_background: bool, percentile: f64) -> Image {
    if img.c < 3 {
        return img.clone();
    }
    let mask = if use_background {
        let m = background_mask(img, percentile);
        if m.iter().filter(|&&b| b).count() < 100 {
            vec![true; img.npix()]
        } else {
            m
        }
    } else {
        vec![true; img.npix()]
    };
    let m = masked_medians(img, &mask);
    let rf = m[1].max(1e-9);
    let mut out = img.clone();
    for c in 0..3 {
        if m[c] > 1e-9 {
            let k = (white[c] * rf / m[c]) as f32;
            for i in 0..img.npix() {
                out.data[i * img.c + c] = img.data[i * img.c + c] * k;
            }
        }
    }
    out.clip01()
}

/// Le ciel devient strictement neutre (R = G = B sur le fond).
pub fn background_neutralize(img: &Image, percentile: f64) -> Image {
    if img.c < 3 {
        return img.clone();
    }
    let mut mask = background_mask(img, percentile);
    if mask.iter().filter(|&&b| b).count() < 100 {
        mask = vec![true; img.npix()];
    }
    let m = masked_medians(img, &mask);
    let mut out = img.clone();
    for c in 0..3 {
        let off = (m[c] - m[1]) as f32;
        for i in 0..img.npix() {
            out.data[i * img.c + c] = (img.data[i * img.c + c] - off).max(0.0);
        }
    }
    out
}

/// SCNR « moyenne neutre » : G = min(G, (R+B)/2).
pub fn scnr_green(img: &Image) -> Image {
    if img.c < 3 {
        return img.clone();
    }
    let mut out = img.clone();
    for p in out.data.chunks_mut(img.c) {
        p[1] = p[1].min((p[0] + p[2]) * 0.5);
    }
    out.clip01()
}

/// R et B remis à l'échelle radialement autour du centre (k ∈ [−0,1 ; 0,1]).
pub fn correct_chromatic_aberration(img: &Image, k_r: f64, k_b: f64, center: Option<(f64, f64)>) -> Image {
    if img.c < 3 {
        return img.clone();
    }
    let (cx, cy) = center.unwrap_or(((img.w as f64 - 1.0) / 2.0, (img.h as f64 - 1.0) / 2.0));
    let mut out = img.clone();
    for (c, k) in [(0usize, k_r), (2, k_b)] {
        let k = k.clamp(-0.1, 0.1);
        if k.abs() < 1e-4 {
            continue;
        }
        let s = 1.0 - k;
        let m = [[s, 0.0, cx * (1.0 - s)], [0.0, s, cy * (1.0 - s)]];
        let ch = geom::warp_affine(&img.channel(c), &m, img.w, img.h, Interp::Cubic, Border::Reflect);
        out.set_channel(c, &ch);
    }
    out
}

#[derive(Clone, Debug)]
pub struct PccInfo {
    pub n: usize,
    pub k_r: f64,
    pub k_b: f64,
}

/// Calibration photométrique des couleurs (étoiles appariées au catalogue
/// via le WCS, couleurs BP/RP de Gaia).
pub fn photometric_color_calibration(img: &Image, w: &Wcs, catalog: &Path, r: f64, r_in: f64, r_out: f64, n_max: usize, neutralize: bool) -> Result<(Image, PccInfo)> {
    if img.c < 3 {
        bail!("PCC : une image couleur (RGB) est requise");
    }
    let (wd, ht) = (img.w, img.h);
    let st = stars::detect_stars(&img.mean_lum(), n_max);
    if st.len() < 5 {
        bail!("PCC : moins de 5 étoiles détectées");
    }
    let cat = astrometry::load_catalog(catalog)?;
    let (Some(bp), Some(rp), Some(mag)) = (cat.bp.clone(), cat.rp.clone(), cat.mag.clone()) else {
        bail!("Catalogue sans couleurs BP/RP — régénérez le catalogue Gaia (onglet Configuration)")
    };
    let px: Vec<Option<(f64, f64)>> = (0..cat.len()).map(|i| w.world_to_pixel(cat.ra[i], cat.dec[i])).collect();
    let mut matched = Vec::new();
    for (k, s) in st.iter().enumerate() {
        let mut best = (usize::MAX, f64::INFINITY);
        for (j, p) in px.iter().enumerate() {
            if let Some(p) = p {
                let d = (p.0 - s.0).hypot(p.1 - s.1);
                if d < best.1 {
                    best = (j, d);
                }
            }
        }
        if best.1 < 4.0 {
            matched.push((k, best.0));
        }
    }
    if matched.len() < 5 {
        bail!("PCC : moins de 5 étoiles appariées au catalogue — vérifiez le plate solving et le catalogue");
    }
    let rr = r_out.ceil() as isize + 1;
    let (mut ratios_b, mut ratios_r) = (Vec::new(), Vec::new());
    for (ii, ic) in matched {
        let (sx, sy) = st[ii];
        let (x0, x1) = (sx as isize - rr, sx as isize + rr + 1);
        let (y0, y1) = (sy as isize - rr, sy as isize + rr + 1);
        if x0 < 0 || y0 < 0 || x1 > wd as isize || y1 > ht as isize {
            continue;
        }
        let mut ap: Vec<[f32; 3]> = Vec::new();
        let mut an: Vec<[f32; 3]> = Vec::new();
        for y in y0..y1 {
            for x in x0..x1 {
                let d2 = (x as f64 - sx).powi(2) + (y as f64 - sy).powi(2);
                let p = img.pixel(x as usize, y as usize);
                let v = [p[0], p[1], p[2]];
                if d2 <= r * r {
                    ap.push(v);
                } else if d2 > r_in * r_in && d2 <= r_out * r_out {
                    an.push(v);
                }
            }
        }
        if ap.is_empty() || an.len() < 8 {
            continue;
        }
        let mut sat = false;
        for c in 0..3 {
            let peak = ap.iter().map(|v| v[c]).fold(f32::NEG_INFINITY, f32::max);
            let n = ap.iter().filter(|v| v[c] >= 0.995 * peak && peak > 0.0).count();
            if n >= 3 {
                sat = true;
            }
        }
        if sat {
            continue;
        }
        let mut flux = [0.0f64; 3];
        for c in 0..3 {
            let bg = stats::median(&an.iter().map(|v| v[c]).collect::<Vec<_>>()) as f64;
            flux[c] = ap.iter().map(|v| v[c] as f64).sum::<f64>() - bg * ap.len() as f64;
        }
        let (fr, fg, fb) = (flux[0], flux[1], flux[2]);
        if fg <= 0.0 || fb <= 0.0 || fr <= 0.0 {
            continue;
        }
        if !(bp[ic].is_finite() && rp[ic].is_finite() && mag[ic].is_finite()) {
            continue;
        }
        let eb = 10f64.powf(-0.4 * (bp[ic] - mag[ic]));
        let er = 10f64.powf(-0.4 * (rp[ic] - mag[ic]));
        ratios_b.push(eb / (fb / fg));
        ratios_r.push(er / (fr / fg));
    }
    if ratios_b.len() < 5 {
        bail!("PCC : pas assez d'étoiles mesurables (saturation ?)");
    }
    let kb = stats::median_f64(&ratios_b);
    let kr = stats::median_f64(&ratios_r);
    let mut out = img.clone();
    for p in out.data.chunks_mut(img.c) {
        p[0] *= kr as f32;
        p[2] *= kb as f32;
    }
    if neutralize {
        out = background_neutralize(&out, 25.0);
    }
    Ok((out.clip01(), PccInfo { n: ratios_b.len(), k_r: kr, k_b: kb }))
}
