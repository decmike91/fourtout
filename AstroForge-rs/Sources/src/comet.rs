//! Empilement de comètes (« étoiles + comète ») : images alignées sur les
//! étoiles, position du noyau interpolée dans le temps, deux empilements avec
//! rejet (étoiles / comète), composition.

use crate::image::Image;
use crate::ndimage;
use crate::registration::apply_shift;
use crate::stacking::{self, StackParams};
use crate::stats;
use crate::timeutil;
use anyhow::{bail, Result};

pub const COMPOSITE_MODES: [&str; 4] = ["Étoiles + comète (maximum)", "Étoiles + comète (addition)", "Comète seule", "Étoiles seules"];

/// DATE-OBS -> secondes depuis la première image ; None si absent.
pub fn parse_times(dates: &[String]) -> Option<Vec<f64>> {
    let mut out = Vec::new();
    for d in dates {
        if d.is_empty() {
            return None;
        }
        out.push(timeutil::timestamp(&timeutil::parse_iso_utc(d).ok()?));
    }
    let t0 = *out.first()?;
    Some(out.iter().map(|t| t - t0).collect())
}

pub fn interpolate_positions(n: usize, first: (f64, f64), last: (f64, f64), times: Option<&[f64]>) -> Vec<(f64, f64)> {
    let t: Vec<f64> = times.map(|t| t.to_vec()).unwrap_or_else(|| (0..n).map(|i| i as f64).collect());
    if t.is_empty() {
        return vec![];
    }
    let span = t[t.len() - 1] - t[0];
    let span = if span == 0.0 { 1.0 } else { span };
    t.iter()
        .map(|&ti| {
            let f = (ti - t[0]) / span;
            (first.0 + f * (last.0 - first.0), first.1 + f * (last.1 - first.1))
        })
        .collect()
}

/// Centre photométrique du noyau autour de (x, y).
pub fn refine_nucleus(img: &Image, x: f64, y: f64, radius: usize) -> (f64, f64) {
    let mono = img.mean_lum();
    let (w, h) = (mono.w, mono.h);
    let x0 = (x as i64 - radius as i64).max(0) as usize;
    let x1 = ((x as i64 + radius as i64 + 1) as usize).min(w);
    let y0 = (y as i64 - radius as i64).max(0) as usize;
    let y1 = ((y as i64 + radius as i64 + 1) as usize).min(h);
    if x1 <= x0 || y1 <= y0 {
        return (x, y);
    }
    let win = ndimage::gaussian(&mono.crop(x0, y0, x1 - x0, y1 - y0), 2.0);
    let med = stats::median(&win.data);
    let (mut sx, mut sy, mut st) = (0.0, 0.0, 0.0);
    for yy in 0..win.h {
        for xx in 0..win.w {
            let v = (win.data[yy * win.w + xx] - med).max(0.0) as f64;
            sx += v * (xx + x0) as f64;
            sy += v * (yy + y0) as f64;
            st += v;
        }
    }
    if st <= 0.0 {
        return (x, y);
    }
    (sx / st, sy / st)
}

pub struct CometResult {
    pub stars: Image,
    pub comet: Image,
    pub composite: Image,
    pub positions: Vec<(f64, f64)>,
}

pub fn comet_stack(frames: &[Image], positions: &[(f64, f64)], method: &str, sigma: f64, mode: &str, refine: bool) -> Result<CometResult> {
    if frames.len() != positions.len() || frames.len() < 3 {
        bail!("Comète : au moins 3 images et une position par image");
    }
    let pos: Vec<(f64, f64)> = frames.iter().zip(positions).map(|(f, p)| if refine { refine_nucleus(f, p.0, p.1, 12) } else { *p }).collect();
    let r = pos[pos.len() / 2];
    let p = StackParams { sigma, ..Default::default() };
    let stars = stacking::stack(&frames.iter().collect::<Vec<_>>(), method, &p, None)?;
    let shifted: Vec<Image> = frames.iter().zip(&pos).map(|(f, q)| apply_shift(f, r.0 - q.0, r.1 - q.1)).collect();
    let comet = stacking::stack(&shifted.iter().collect::<Vec<_>>(), method, &p, None)?;
    let comp = if mode.starts_with("Comète seule") {
        comet.clone()
    } else if mode.starts_with("Étoiles seules") {
        stars.clone()
    } else if mode.contains("addition") {
        let c = comet.c;
        let bg: Vec<f32> = (0..c).map(|k| stats::median(&comet.channel(k).data)).collect();
        let mut o = stars.clone();
        for i in 0..o.npix() {
            for k in 0..c {
                o.data[i * c + k] += (comet.data[i * c + k] - bg[k]).max(0.0);
            }
        }
        o
    } else {
        stars.zip_map(&comet, f32::max)
    };
    Ok(CometResult { stars, comet, composite: comp.clip01(), positions: pos })
}
