//! Effets : suppression des traces de satellites (Hough + inpainting),
//! fusion HDR (Mertens), fausses couleurs (soleil, palettes scientifiques).

use crate::display;
use crate::features;
use crate::image::Image;
use crate::ndimage::{correlate_taps, Border};
use crate::stats;
use anyhow::{bail, Result};

/// Lignes brillantes fines (satellites/avions) -> masque.
pub fn detect_trail_mask(img: &Image, min_len_frac: f64, threshold: f64, width: i32) -> Vec<bool> {
    let mono = img.mean_lum();
    let (w, h) = (mono.w, mono.h);
    let u8v = features::to_u8(&display::autostretch(&mono));
    let med = features::median_blur_u8(&u8v, w, h, 21);
    let high: Vec<u8> = u8v.iter().zip(&med).map(|(&a, &b)| a.saturating_sub(b)).collect();
    let t = features::otsu(&high);
    let th: Vec<u8> = high.iter().map(|&v| if v > t { 255 } else { 0 }).collect();
    let min_len = (min_len_frac * w.min(h) as f64) as usize;
    let lines = features::hough_lines_p(&th, w, h, threshold as usize, min_len, 6);
    let mut mask = vec![0u8; w * h];
    for (x1, y1, x2, y2) in lines {
        features::draw_line_u8(&mut mask, w, h, (x1, y1), (x2, y2), width);
    }
    mask.iter().map(|&v| v > 0).collect()
}

/// Inpainting des pixels du masque (détecté ou fourni), canal par canal.
pub fn remove_satellite_trails(img: &Image, mask: Option<&[bool]>, min_len_frac: f64, threshold: f64, width: i32) -> Result<Image> {
    let m: Vec<bool> = match mask {
        Some(m) => m.to_vec(),
        None => detect_trail_mask(img, min_len_frac, threshold, width),
    };
    if m.len() != img.npix() {
        bail!("Masque ≠ image");
    }
    if !m.iter().any(|&b| b) {
        return Ok(img.clone());
    }
    Ok(img.map_planes(|p| features::inpaint(p, &m, 7)))
}

/// Médiane de frames alignées : efface les traces mobiles.
pub fn remove_trails_by_median(frames: &[Image]) -> Result<Image> {
    if frames.is_empty() {
        bail!("Aucune frame");
    }
    crate::stacking::stack_owned(frames, "Médiane", 3.0, None)
}

// ------------------------------------------------------------------ HDR
const K5: [(isize, f64); 5] = [(-2, 1.0 / 16.0), (-1, 4.0 / 16.0), (0, 6.0 / 16.0), (1, 4.0 / 16.0), (2, 1.0 / 16.0)];

fn blur5(p: &Image) -> Image {
    let a = correlate_taps(p, &K5, 0, Border::Mirror);
    correlate_taps(&a, &K5, 1, Border::Mirror)
}

fn pyr_down(p: &Image) -> Image {
    let b = p.map_planes(blur5);
    let (nw, nh) = (p.w.div_ceil(2), p.h.div_ceil(2));
    let c = p.c;
    let mut out = Image::new(nw, nh, c);
    for y in 0..nh {
        for x in 0..nw {
            for k in 0..c {
                out.data[(y * nw + x) * c + k] = b.data[((2 * y).min(p.h - 1) * p.w + (2 * x).min(p.w - 1)) * c + k];
            }
        }
    }
    out
}

fn pyr_up(p: &Image, w: usize, h: usize) -> Image {
    let c = p.c;
    let mut up = Image::new(w, h, c);
    for y in 0..p.h {
        for x in 0..p.w {
            if 2 * y < h && 2 * x < w {
                for k in 0..c {
                    up.data[((2 * y) * w + 2 * x) * c + k] = 4.0 * p.data[(y * p.w + x) * c + k];
                }
            }
        }
    }
    up.map_planes(blur5)
}

/// Fusion d'expositions de Mertens (contraste, saturation, bonne exposition).
pub fn merge_hdr(images: &[Image], contrast: f64, saturation: f64, well_exposed: f64) -> Result<Image> {
    if images.len() < 2 {
        bail!("HDR : fournir au moins 2 expositions");
    }
    if images.iter().any(|i| !i.same_shape(&images[0])) {
        bail!("HDR : toutes les expositions doivent avoir la même taille");
    }
    let mono = images[0].c == 1;
    let imgs: Vec<Image> = images.iter().map(|i| i.clip01().to_rgb()).collect();
    let (w, h) = (imgs[0].w, imgs[0].h);
    // poids
    let mut weights: Vec<Image> = imgs
        .iter()
        .map(|im| {
            let gray = Image::from_vec(w, h, 1, im.data.chunks(3).map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]).collect());
            let lap = crate::ndimage::laplace(&gray);
            let mut wv = Image::new(w, h, 1);
            for i in 0..w * h {
                let p = &im.data[i * 3..i * 3 + 3];
                let c = (lap.data[i].abs() as f64).powf(contrast);
                let m = (p[0] + p[1] + p[2]) as f64 / 3.0;
                let sd = (((p[0] as f64 - m).powi(2) + (p[1] as f64 - m).powi(2) + (p[2] as f64 - m).powi(2)) / 3.0).sqrt();
                let e: f64 = p.iter().map(|&v| (-((v as f64 - 0.5).powi(2)) / (2.0 * 0.04)).exp()).product();
                wv.data[i] = (c * sd.powf(saturation) * e.powf(well_exposed) + 1e-12) as f32;
            }
            wv
        })
        .collect();
    for i in 0..w * h {
        let s: f32 = weights.iter().map(|wv| wv.data[i]).sum();
        for wv in weights.iter_mut() {
            wv.data[i] /= s;
        }
    }
    let levels = ((w.min(h) as f64).log2() as usize).max(1);
    let mut result: Option<Vec<Image>> = None;
    for (im, wv) in imgs.iter().zip(&weights) {
        // pyramides
        let mut gw = vec![wv.clone()];
        let mut gi = vec![im.clone()];
        for _ in 0..levels {
            gw.push(pyr_down(gw.last().unwrap()));
            gi.push(pyr_down(gi.last().unwrap()));
        }
        let mut lap = Vec::new();
        for l in 0..levels {
            let up = pyr_up(&gi[l + 1], gi[l].w, gi[l].h);
            lap.push(gi[l].zip_map(&up, |a, b| a - b));
        }
        lap.push(gi[levels].clone());
        let contrib: Vec<Image> = lap
            .iter()
            .zip(&gw)
            .map(|(li, wl)| {
                let mut o = li.clone();
                for i in 0..wl.data.len() {
                    for k in 0..3 {
                        o.data[i * 3 + k] *= wl.data[i];
                    }
                }
                o
            })
            .collect();
        result = Some(match result {
            None => contrib,
            Some(r) => r.iter().zip(&contrib).map(|(a, b)| a.zip_map(b, |x, y| x + y)).collect(),
        });
    }
    let res = result.unwrap();
    let mut out = res[levels].clone();
    for l in (0..levels).rev() {
        out = pyr_up(&out, res[l].w, res[l].h).zip_map(&res[l], |a, b| a + b);
    }
    let out = out.clip01();
    Ok(if mono { out.mean_lum() } else { out })
}

// ------------------------------------------------------------- fausses couleurs
pub const PALETTES: [&str; 7] = ["Solaire (orange)", "Inferno", "Magma", "Plasma", "Viridis", "Turbo", "Jet"];

const SOLAR: [[f64; 3]; 8] = [[0.0, 0.0, 0.0], [60.0, 20.0, 0.0], [140.0, 60.0, 0.0], [210.0, 120.0, 0.0], [255.0, 180.0, 20.0], [255.0, 230.0, 120.0], [255.0, 255.0, 230.0], [255.0, 255.0, 255.0]];

fn jet(t: f64) -> [f64; 3] {
    let seg = |pts: &[(f64, f64)]| -> f64 {
        let xs: Vec<f64> = pts.iter().map(|p| p.0).collect();
        let ys: Vec<f64> = pts.iter().map(|p| p.1).collect();
        stats::interp(t, &xs, &ys)
    };
    [
        seg(&[(0.0, 0.0), (0.35, 0.0), (0.66, 1.0), (0.89, 1.0), (1.0, 0.5)]),
        seg(&[(0.0, 0.0), (0.125, 0.0), (0.375, 1.0), (0.64, 1.0), (0.91, 0.0), (1.0, 0.0)]),
        seg(&[(0.0, 0.5), (0.11, 1.0), (0.34, 1.0), (0.65, 0.0), (1.0, 0.0)]),
    ]
}

/// Table de 256 couleurs de la palette.
pub fn palette_lut(name: &str) -> Result<Vec<[f32; 3]>> {
    let lin: Vec<f64> = stats::linspace(0.0, 1.0, 256);
    let lut: Vec<[f32; 3]> = if name.starts_with("Solaire") {
        let xs = stats::linspace(0.0, 1.0, SOLAR.len());
        lin.iter()
            .map(|&t| {
                let mut c = [0.0f32; 3];
                for (k, v) in c.iter_mut().enumerate() {
                    *v = (stats::interp(t, &xs, &SOLAR.iter().map(|s| s[k] / 255.0).collect::<Vec<_>>())) as f32;
                }
                c
            })
            .collect()
    } else {
        let g = match name.to_lowercase().as_str() {
            "inferno" => Some(colorous::INFERNO),
            "magma" => Some(colorous::MAGMA),
            "plasma" => Some(colorous::PLASMA),
            "viridis" => Some(colorous::VIRIDIS),
            "turbo" => Some(colorous::TURBO),
            "jet" => None,
            _ => bail!("Palette inconnue : {}", name),
        };
        lin.iter()
            .map(|&t| match g {
                Some(g) => {
                    let c = g.eval_continuous(t);
                    [c.r as f32 / 255.0, c.g as f32 / 255.0, c.b as f32 / 255.0]
                }
                None => {
                    let c = jet(t);
                    [c[0] as f32, c[1] as f32, c[2] as f32]
                }
            })
            .collect()
    };
    Ok(lut)
}

/// Fausses couleurs sur une image mono (ex. Soleil H-alpha).
pub fn apply_palette(mono: &Image, palette: &str) -> Result<Image> {
    let a = mono.mean_lum();
    let lut = palette_lut(palette)?;
    let mn = a.min_value();
    let mx = (a.max_value() - mn).max(1e-9);
    let mut out = Image::new(a.w, a.h, 3);
    for (i, &v) in a.data.iter().enumerate() {
        let n = (v - mn) / mx;
        let idx = ((n * 255.0) as i32).clamp(0, 255) as usize;
        out.data[i * 3..i * 3 + 3].copy_from_slice(&lut[idx]);
    }
    Ok(out)
}
