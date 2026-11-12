//! Outils solaires : limbe, assombrissement centre-bord compensé, disque
//! inversé + protubérances rehaussées avec raccord doux au limbe.

use crate::derotation::fit_disc;
use crate::image::Image;
use crate::ndimage;
use crate::stats;
use anyhow::Result;

/// (intérieur doux 0→1, distance normalisée r/R).
pub fn disc_masks(w: usize, h: usize, disc: (f64, f64, f64), feather: f64) -> (Vec<f32>, Vec<f32>) {
    let (cx, cy, r) = disc;
    let mut inside = vec![0.0f32; w * h];
    let mut rn = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let rr = (x as f64 - cx).hypot(y as f64 - cy);
            inside[y * w + x] = ((r - rr) / feather.max(0.5) + 0.5).clamp(0.0, 1.0) as f32;
            rn[y * w + x] = (rr / r.max(1e-6)) as f32;
        }
    }
    (inside, rn)
}

fn polyfit(xs: &[f64], ys: &[f64], deg: usize) -> Vec<f64> {
    let n = xs.len();
    let mut a = nalgebra::DMatrix::<f64>::zeros(n, deg + 1);
    for i in 0..n {
        for k in 0..=deg {
            a[(i, k)] = xs[i].powi((deg - k) as i32);
        }
    }
    let b = nalgebra::DVector::from_vec(ys.to_vec());
    a.svd(true, true).solve(&b, 1e-12).map(|v| v.iter().cloned().collect()).unwrap_or_else(|_| vec![0.0; deg + 1])
}

fn polyval(c: &[f64], x: f64) -> f64 {
    c.iter().fold(0.0, |acc, &k| acc * x + k)
}

/// Compense l'assombrissement centre-bord (profil radial médian, polynôme).
pub fn flatten_limb_darkening(img: &Image, disc: (f64, f64, f64), degree: usize) -> Image {
    let mono = img.mean_lum();
    let (w, h) = (mono.w, mono.h);
    let (inside, rn) = disc_masks(w, h, disc, 1.0);
    let sel: Vec<usize> = (0..w * h).filter(|&i| inside[i] > 0.99 && rn[i] < 0.98).collect();
    if sel.len() < 100 {
        return img.clone();
    }
    let bins = stats::linspace(0.0, 0.98, 40);
    let mut groups: Vec<Vec<f32>> = vec![Vec::new(); bins.len()];
    for &i in &sel {
        let r = rn[i] as f64;
        let b = bins.iter().filter(|&&e| e <= r).count();
        if b >= 1 && b < bins.len() {
            groups[b].push(mono.data[i]);
        }
    }
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for b in 1..bins.len() {
        if groups[b].len() > 20 {
            xs.push(0.5 * (bins[b - 1] + bins[b]));
            ys.push(stats::median(&groups[b]) as f64);
        }
    }
    if xs.len() < degree + 2 {
        return img.clone();
    }
    let c = polyfit(&xs, &ys, degree);
    let p0 = polyval(&c, 0.0).max(1e-4);
    let ch = img.c;
    let mut out = img.clone();
    for i in 0..w * h {
        let prof = polyval(&c, (rn[i] as f64).clamp(0.0, 0.999)).max(1e-4) / p0;
        let mut k = if inside[i] > 0.0 { 1.0 / prof } else { 1.0 };
        k = 1.0 + (k - 1.0) * inside[i] as f64;
        for q in 0..ch {
            out.data[i * ch + q] = ((img.data[i * ch + q] as f64 * k) as f32).clamp(0.0, 1.0);
        }
    }
    out
}

/// Composite solaire : disque (inversé ou non) + protubérances.
#[allow(clippy::too_many_arguments)]
pub fn prominence_composite(img: &Image, disc: Option<(f64, f64, f64)>, invert_disc: bool, disc_gamma: f64, prom_gain: f64, prom_gamma: f64, feather: f64, flatten: bool) -> Result<(Image, (f64, f64, f64))> {
    let mut mono = img.mean_lum();
    let disc = match disc {
        Some(d) => d,
        None => fit_disc(&mono, None)?,
    };
    if flatten {
        mono = flatten_limb_darkening(&mono, disc, 4);
    }
    let (w, h) = (mono.w, mono.h);
    let (inside, rn) = disc_masks(w, h, disc, feather);
    let vals: Vec<f32> = (0..w * h).filter(|&i| inside[i] > 0.99).map(|i| mono.data[i]).collect();
    let (lo, hi) = if vals.is_empty() { (0.0, 1.0) } else { (stats::percentile(&vals, 0.5), stats::percentile(&vals, 99.5)) };
    let mut d = mono.map(|v| ((v - lo) / (hi - lo).max(1e-6)).clamp(0.0, 1.0));
    if invert_disc {
        d.map_inplace(|v| 1.0 - v);
    }
    let dg = disc_gamma.max(1e-3) as f32;
    d.map_inplace(|v| v.powf(1.0 / dg));
    let ring: Vec<f32> = (0..w * h).filter(|&i| rn[i] > 1.15 && rn[i] < 2.0).map(|i| mono.data[i]).collect();
    let bg = if ring.is_empty() { 0.0 } else { stats::median(&ring) };
    let pg = prom_gamma.max(1e-3) as f32;
    let mut p = mono.map(|v| ((v - bg) * prom_gain as f32).clamp(0.0, 1.0).powf(pg));
    if (0..w * h).any(|i| inside[i] < 0.01 && rn[i] < 2.0) {
        p = ndimage::gaussian(&p, 0.5);
    }
    let comp = Image::from_vec(w, h, 1, (0..w * h).map(|i| (d.data[i] * inside[i] + p.data[i] * (1.0 - inside[i])).clamp(0.0, 1.0)).collect());
    Ok((comp, disc))
}
