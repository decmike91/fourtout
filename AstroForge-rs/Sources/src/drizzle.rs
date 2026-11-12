//! Drizzle (Fruchter & Hook 2002), noyau carré, pixfrac réglable, ×2/×3 :
//! chaque pixel d'entrée (image calibrée NON rééchantillonnée) est projeté
//! sur la grille suréchantillonnée par la transformation de sa frame, sa
//! « goutte » répartie au prorata des surfaces. Rejet optionnel contre une
//! image de référence (empilement classique).

use crate::geom::{self, Affine, Interp};
use crate::image::Image;
use crate::ndimage::Border;
use crate::stats;
use anyhow::{bail, Result};
use rayon::prelude::*;

/// Matrice frame -> référence identique à apply_shift (rotation au centre puis translation).
pub fn shift_matrix(dx: f64, dy: f64, angle_deg: f64, shape: Option<(usize, usize)>) -> Affine {
    let mut m = match shape {
        Some((w, h)) if angle_deg != 0.0 => geom::rotation_matrix(w as f64 / 2.0, h as f64 / 2.0, angle_deg, 1.0),
        _ => geom::IDENTITY,
    };
    m[0][2] += dx;
    m[1][2] += dy;
    m
}

/// Combine des frames NON alignées ; retourne (image, couverture normalisée).
#[allow(clippy::too_many_arguments)]
pub fn drizzle_stack(
    frames: &[Image],
    matrices: &[Affine],
    scale: usize,
    pixfrac: f64,
    weights: Option<&[f64]>,
    reference: Option<&Image>,
    reject_sigma: Option<f64>,
    progress: Option<&(dyn Fn(usize, usize, &str) + Sync)>,
) -> Result<(Image, Image)> {
    if frames.is_empty() {
        bail!("Drizzle : aucune frame");
    }
    let n = frames.len();
    if matrices.len() != n {
        bail!("Drizzle : une matrice par frame est requise");
    }
    let scale = scale.max(1);
    let pixfrac = pixfrac.clamp(0.05, 1.0);
    let (w, h, chans) = (frames[0].w, frames[0].h, frames[0].c);
    let (ow, oh) = (w * scale, h * scale);
    let wts: Vec<f64> = weights.map(|w| w.to_vec()).unwrap_or_else(|| vec![1.0; n]);
    let mut acc = vec![0.0f64; chans * ow * oh];
    let mut cov = vec![0.0f64; ow * oh];
    let ref_img = match (reference, reject_sigma) {
        (Some(r), Some(s)) if s > 0.0 => Some(r),
        _ => None,
    };
    let sc = scale as f64;
    for k in 0..n {
        let img = &frames[k];
        if img.w != w || img.h != h || img.c != chans {
            bail!("Drizzle : frames de tailles différentes");
        }
        let m = &matrices[k];
        let lin = geom::det2(m).abs().sqrt();
        let lin = if lin == 0.0 { 1.0 } else { lin };
        let half = pixfrac * sc * lin / 2.0;
        // rejet : résidu contre la référence rééchantillonnée en (u, v)
        let valid: Vec<bool> = if let Some(r) = ref_img {
            let refv = geom::remap_fn(r, w, h, Interp::Linear, Border::Reflect, |x, y| geom::apply(m, x as f64, y as f64));
            let refv = if refv.c != chans { refv.mean_lum().to_rgb().take_channels(chans) } else { refv };
            let resid: Vec<f32> = img.data.iter().zip(&refv.data).map(|(a, b)| a - b).collect();
            let mut sig = vec![0.0f64; chans];
            for c in 0..chans {
                let col: Vec<f32> = resid.iter().skip(c).step_by(chans).cloned().collect();
                let med = stats::median(&col);
                sig[c] = 1.4826 * stats::mad_about(&col, med) as f64 + 1e-6;
            }
            let rs = reject_sigma.unwrap();
            (0..w * h).map(|i| (0..chans).all(|c| (resid[i * chans + c].abs() as f64) <= rs * sig[c])).collect()
        } else {
            vec![true; w * h]
        };
        let kmax = (2.0 * half).ceil() as isize + 1;
        let wk = wts[k];
        // dépôt parallèle par bandes de lignes de sortie : chaque bande ne
        // reçoit que les gouttes qui la touchent (aucun conflit d'écriture)
        let inv = geom::invert(m);
        let band = 32usize;
        let margin = (half / sc) * 2.0 + 2.0;
        let acc_bands: Vec<(usize, Vec<f64>, Vec<f64>)> = (0..oh.div_ceil(band))
            .into_par_iter()
            .map(|b| {
                let (by0, by1) = (b * band, ((b + 1) * band).min(oh));
                let mut a = vec![0.0f64; (by1 - by0) * ow * chans];
                let mut cv = vec![0.0f64; (by1 - by0) * ow];
                let mut ymin = f64::INFINITY;
                let mut ymax = f64::NEG_INFINITY;
                for &(xo, yo) in &[(0.0, by0 as f64), (ow as f64, by0 as f64), (0.0, by1 as f64), (ow as f64, by1 as f64)] {
                    let (u, v) = (xo / sc - 0.5, yo / sc - 0.5);
                    let (_, yi) = geom::apply(&inv, u, v);
                    ymin = ymin.min(yi);
                    ymax = ymax.max(yi);
                }
                let y_lo = ((ymin - margin).floor().max(0.0)) as usize;
                let y_hi = ((ymax + margin).ceil().min(h as f64 - 1.0)).max(-1.0);
                if y_hi < 0.0 || y_lo as f64 > y_hi {
                    return (b, a, cv);
                }
                for y in y_lo..=(y_hi as usize) {
                    for x in 0..w {
                        let i = y * w + x;
                        if !valid[i] {
                            continue;
                        }
                        let (u, v) = geom::apply(m, x as f64, y as f64);
                        let cx = (u + 0.5) * sc;
                        let cy = (v + 0.5) * sc;
                        let x0 = (cx - half).floor() as isize;
                        let y0 = (cy - half).floor() as isize;
                        for oy in 0..kmax {
                            let iy = y0 + oy;
                            if iy < by0 as isize || iy >= by1 as isize {
                                continue;
                            }
                            let ovy = ((iy + 1) as f64).min(cy + half) - (iy as f64).max(cy - half);
                            if ovy <= 0.0 {
                                continue;
                            }
                            for ox in 0..kmax {
                                let ix = x0 + ox;
                                if ix < 0 || ix >= ow as isize {
                                    continue;
                                }
                                let ovx = ((ix + 1) as f64).min(cx + half) - (ix as f64).max(cx - half);
                                if ovx <= 0.0 {
                                    continue;
                                }
                                let wv = ovx * ovy * wk;
                                let o = (iy as usize - by0) * ow + ix as usize;
                                cv[o] += wv;
                                for c in 0..chans {
                                    a[o * chans + c] += img.data[i * chans + c] as f64 * wv;
                                }
                            }
                        }
                    }
                }
                (b, a, cv)
            })
            .collect();
        for (b, a, cv) in acc_bands {
            let by0 = b * band;
            let o0 = by0 * ow;
            for (j, v) in cv.iter().enumerate() {
                cov[o0 + j] += v;
            }
            for (j, v) in a.iter().enumerate() {
                acc[o0 * chans + j] += v;
            }
        }
        if let Some(p) = progress {
            p(k + 1, n, "Drizzle");
        }
    }
    let mut out = Image::new(ow, oh, chans);
    out.data.par_chunks_mut(chans).enumerate().for_each(|(o, px)| {
        let cv = cov[o].max(1e-12);
        for c in 0..chans {
            px[c] = (acc[o * chans + c] / cv) as f32;
        }
    });
    if cov.iter().any(|&c| c <= 1e-9) {
        let fill_src = match reference {
            Some(r) => r.clone(),
            None => {
                let refs: Vec<&Image> = frames.iter().take(n.min(8)).collect();
                crate::image::mean_images(&refs)
            }
        };
        let fill = geom::resize(&fill_src, ow, oh, Interp::Cubic);
        let fill = if fill.c != chans { fill.to_rgb().take_channels(chans) } else { fill };
        for o in 0..ow * oh {
            if cov[o] <= 1e-9 {
                for c in 0..chans {
                    out.data[o * chans + c] = fill.data[o * chans + c];
                }
            }
        }
    }
    let mx = cov.iter().cloned().fold(0.0f64, f64::max).max(1e-9);
    let coverage = Image::from_vec(ow, oh, 1, cov.iter().map(|&c| (c / mx).clamp(0.0, 1.0) as f32).collect());
    Ok((out, coverage))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn drizzle_identity_flat() {
        let frames: Vec<Image> = (0..4).map(|_| Image::filled(10, 8, 1, 0.3)).collect();
        let mats: Vec<Affine> = (0..4).map(|k| shift_matrix(0.25 * k as f64, -0.2 * k as f64, 0.0, None)).collect();
        let (o, cov) = drizzle_stack(&frames, &mats, 2, 0.7, None, None, None, None).unwrap();
        assert_eq!((o.w, o.h), (20, 16));
        assert!(o.data.iter().all(|&v| (v - 0.3).abs() < 1e-5));
        assert!(cov.max_value() <= 1.0);
    }
}
