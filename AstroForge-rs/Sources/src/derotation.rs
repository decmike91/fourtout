//! Planétaire : disque (cercle ajusté), dérotation de séries d'images
//! (esprit WinJUPOS) et noms de fichiers WinJUPOS.

use crate::features;
use crate::geom::{self, Interp};
use crate::image::Image;
use crate::ndimage::{self, Border};
use crate::registration::apply_shift;
use crate::stats;
use crate::timeutil;
use anyhow::{bail, Result};
use chrono::{NaiveDateTime, Timelike};

/// (nom, période de rotation en heures, aplatissement).
pub const PLANETS: [(&str, f64, f64); 6] = [
    ("Jupiter (Système II)", 9.0 + 55.0 / 60.0 + 40.632 / 3600.0, 0.06487),
    ("Jupiter (Système I)", 9.0 + 50.0 / 60.0 + 30.0034 / 3600.0, 0.06487),
    ("Saturne", 10.0 + 33.0 / 60.0 + 38.0 / 3600.0, 0.09796),
    ("Mars", 24.0 + 37.0 / 60.0 + 22.66 / 3600.0, 0.00589),
    ("Vénus", -5832.5, 0.0),
    ("Lune (libration ignorée)", 655.72, 0.0012),
];

/// (cx, cy, r) : seuil d'Otsu, plus grand objet, contour, cercle de Kåsa.
pub fn fit_disc(img: &Image, threshold: Option<f64>) -> Result<(f64, f64, f64)> {
    let mono = img.mean_lum();
    let sm = ndimage::gaussian(&mono, 1.5);
    let (w, h) = (sm.w, sm.h);
    let mx = (sm.max_value() as f64).max(1e-9);
    let thr = match threshold {
        Some(t) => t,
        None => {
            let u8v: Vec<u8> = sm.data.iter().map(|&v| ((v as f64 / mx).clamp(0.0, 1.0) * 255.0) as u8).collect();
            features::otsu(&u8v) as f64 / 255.0 * mx
        }
    };
    let mask: Vec<bool> = sm.data.iter().map(|&v| (v as f64) > thr).collect();
    let (lab, n) = ndimage::label(&mask, w, h);
    if n == 0 {
        bail!("Aucun disque détecté");
    }
    let mut sizes = vec![0usize; n + 1];
    for &l in &lab {
        sizes[l as usize] += 1;
    }
    let big = (1..=n).max_by_key(|&k| (sizes[k], std::cmp::Reverse(k))).unwrap() as u32;
    let m: Vec<bool> = lab.iter().map(|&l| l == big).collect();
    let m = ndimage::binary_fill_holes(&m, w, h);
    let er = ndimage::binary_erosion(&m, w, h, 1);
    let (mut xs, mut ys) = (Vec::new(), Vec::new());
    for i in 0..w * h {
        if m[i] && !er[i] {
            xs.push((i % w) as f64);
            ys.push((i / w) as f64);
        }
    }
    if xs.len() < 10 {
        bail!("Contour du disque insuffisant");
    }
    let mut keep = vec![true; xs.len()];
    let (mut cx, mut cy, mut r) = (0.0, 0.0, 0.0);
    for _ in 0..4 {
        let idx: Vec<usize> = (0..xs.len()).filter(|&i| keep[i]).collect();
        let mut a = nalgebra::DMatrix::<f64>::zeros(idx.len(), 3);
        let mut b = nalgebra::DVector::<f64>::zeros(idx.len());
        for (k, &i) in idx.iter().enumerate() {
            a[(k, 0)] = xs[i];
            a[(k, 1)] = ys[i];
            a[(k, 2)] = 1.0;
            b[k] = xs[i] * xs[i] + ys[i] * ys[i];
        }
        let c = a.svd(true, true).solve(&b, 1e-12).map_err(|e| anyhow::anyhow!(e))?;
        cx = c[0] / 2.0;
        cy = c[1] / 2.0;
        r = (c[2] + cx * cx + cy * cy).max(1e-9).sqrt();
        let res: Vec<f64> = (0..xs.len()).map(|i| ((xs[i] - cx).hypot(ys[i] - cy) - r).abs()).collect();
        let rk: Vec<f64> = idx.iter().map(|&i| res[i]).collect();
        let lim = 2.0f64.max(3.0 * 1.4826 * stats::median_f64(&rk));
        let new: Vec<bool> = res.iter().map(|&v| v < lim).collect();
        if new.iter().filter(|&&b| b).count() < 10 || new == keep {
            break;
        }
        keep = new;
    }
    Ok((cx, cy, r))
}

#[doc(hidden)]
pub fn image_to_sphere(x: f64, y: f64, d: (f64, f64, f64), pa: f64, flat: f64, b: f64) -> (f64, f64, bool, f64) {
    let (c, s) = (pa.to_radians().cos(), pa.to_radians().sin());
    let (dx, dy) = ((x - d.0) / d.2, (d.1 - y) / d.2);
    let u = c * dx - s * dy;
    let mut v = s * dx + c * dy;
    v /= 1.0 - flat;
    let rho2 = u * u + v * v;
    let vis = rho2 < 1.0;
    let z = (1.0 - rho2).clamp(0.0, 1.0).sqrt();
    let br = b.to_radians();
    let yy = v * br.cos() + z * br.sin();
    let zz = -v * br.sin() + z * br.cos();
    (u.atan2(zz), yy.clamp(-1.0, 1.0).asin(), vis, z)
}

fn sphere_to_image(lon: f64, lat: f64, d: (f64, f64, f64), pa: f64, flat: f64, b: f64) -> (f64, f64, f64) {
    let u = lat.cos() * lon.sin();
    let yy = lat.sin();
    let zz = lat.cos() * lon.cos();
    let br = b.to_radians();
    let mut v = yy * br.cos() - zz * br.sin();
    let z = yy * br.sin() + zz * br.cos();
    v *= 1.0 - flat;
    let (c, s) = (pa.to_radians().cos(), pa.to_radians().sin());
    let dx = c * u + s * v;
    let dy = -s * u + c * v;
    (d.0 + dx * d.2, d.1 - dy * d.2, z)
}

/// Ramène une image (prise à t) à t0 = t + dt : (image, poids).
#[allow(clippy::too_many_arguments)]
pub fn derotate_image(img: &Image, disc: (f64, f64, f64), dt_seconds: f64, period_hours: f64, pa: f64, flat: f64, b: f64, direction: i32, limb_cut: f64) -> (Image, Image) {
    let (w, h) = (img.w, img.h);
    let dlon = direction as f64 * 2.0 * std::f64::consts::PI * dt_seconds / (period_hours * 3600.0);
    let mut maps = vec![(0.0, 0.0); w * h];
    let mut wgt = Image::new(w, h, 1);
    let mut outside = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let (lon, lat, vis, z0) = image_to_sphere(x as f64, y as f64, disc, pa, flat, b);
            let (sx, sy, zs) = sphere_to_image(lon - dlon, lat, disc, pa, flat, b);
            let i = y * w + x;
            maps[i] = (sx, sy);
            let ok = vis && zs > limb_cut && z0 > 0.0;
            wgt.data[i] = if ok { zs.clamp(0.0, 1.0) as f32 } else { 0.0 };
            if !vis {
                outside[i] = true;
                wgt.data[i] = 1e-3;
            }
        }
    }
    let mut out = geom::remap_fn(img, w, h, Interp::Cubic, Border::Reflect, |x, y| maps[y * w + x]);
    let c = img.c;
    for i in 0..w * h {
        if outside[i] {
            out.data[i * c..i * c + c].copy_from_slice(&img.data[i * c..i * c + c]);
        }
    }
    (out, wgt)
}

pub fn planet_params(name: &str) -> (f64, f64) {
    PLANETS.iter().find(|p| p.0 == name).map(|p| (p.1, p.2)).unwrap_or((9.925, 0.0649))
}

/// Combine des images prises à des instants différents, ramenées à t0
/// (milieu de la série par défaut) : (image, t0 ISO, disque).
#[allow(clippy::too_many_arguments)]
pub fn derotate_series(images: &[Image], times: &[String], t0: Option<&str>, planet: &str, pa: f64, b: f64, direction: i32, period_hours: Option<f64>, flat: Option<f64>) -> Result<(Image, String, (f64, f64, f64))> {
    if images.len() != times.len() || images.is_empty() {
        bail!("Une date par image est requise");
    }
    let ts: Vec<NaiveDateTime> = times.iter().map(|t| timeutil::parse_iso_utc(t)).collect::<Result<_>>()?;
    let mid = match t0 {
        Some(t) => timeutil::parse_iso_utc(t)?,
        None => {
            let mn = *ts.iter().min().unwrap();
            let mx = *ts.iter().max().unwrap();
            mn + (mx - mn) / 2
        }
    };
    let (p0, f0) = planet_params(planet);
    let p = period_hours.filter(|v| *v != 0.0).unwrap_or(p0);
    let f = flat.unwrap_or(f0);
    let ri = (0..ts.len()).min_by_key(|&i| (ts[i] - mid).num_milliseconds().abs()).unwrap();
    let (cx, cy, r) = fit_disc(&images[ri], None)?;
    let disc = (cx, cy, r / (1.0 - f / 2.0));
    let c = images[0].c;
    let (w, h) = (images[0].w, images[0].h);
    let mut acc = vec![0.0f64; w * h * c];
    let mut wacc = vec![0.0f64; w * h];
    for (img, t) in images.iter().zip(&ts) {
        let d = fit_disc(img, None)?;
        let a = apply_shift(img, disc.0 - d.0, disc.1 - d.1);
        let dt = (mid - *t).num_microseconds().unwrap_or(0) as f64 / 1e6;
        let (o, wg) = derotate_image(&a, disc, dt, p, pa, f, b, direction, 0.15);
        for i in 0..w * h {
            let wv = wg.data[i] as f64;
            wacc[i] += wv;
            for k in 0..c {
                acc[i * c + k] += o.data[i * c + k] as f64 * wv;
            }
        }
    }
    let mut out = Image::new(w, h, c);
    for i in 0..w * h {
        for k in 0..c {
            out.data[i * c + k] = ((acc[i * c + k] / wacc[i].max(1e-9)) as f32).clamp(0.0, 1.0);
        }
    }
    Ok((out, timeutil::iso_s(&mid), disc))
}

/// « AAAA-MM-JJ-HHMM_D[-Observateur][-Filtre].ext » (D = dixièmes de minute, UT).
pub fn winjupos_name(when: &str, observer: &str, filt: &str, ext: &str) -> Result<String> {
    let d = timeutil::parse_iso_utc(when)?;
    let tenth = ((d.second() as f64 + d.nanosecond() as f64 / 1e9) / 6.0) as u32;
    let mut parts = vec![format!("{}_{}", d.format("%Y-%m-%d-%H%M"), tenth)];
    for extra in [observer, filt] {
        let clean: String = extra.chars().filter(|c| c.is_alphanumeric() || "+._".contains(*c)).collect();
        if !clean.is_empty() {
            parts.push(clean);
        }
    }
    Ok(format!("{}.{}", parts.join("-"), ext.trim_start_matches('.')))
}

pub fn parse_winjupos_name(name: &str) -> Option<NaiveDateTime> {
    let base = std::path::Path::new(name).file_name()?.to_string_lossy().to_string();
    let rx = regex::Regex::new(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})(\d{2})_(\d)").ok()?;
    let c = rx.captures(&base)?;
    let g = |i: usize| c.get(i).unwrap().as_str().parse::<u32>().unwrap();
    let d = chrono::NaiveDate::from_ymd_opt(g(1) as i32, g(2), g(3))?.and_hms_opt(g(4), g(5), 0)?;
    Some(d + chrono::Duration::seconds(6 * g(6) as i64))
}

/// Instant médian d'une liste d'horodatages ISO.
pub fn mid_time(ts: &[String]) -> Option<String> {
    let mut v: Vec<NaiveDateTime> = ts.iter().filter(|t| !t.is_empty()).filter_map(|t| timeutil::parse_iso_utc(t).ok()).collect();
    if v.is_empty() {
        return None;
    }
    v.sort();
    let m = v[0] + (*v.last().unwrap() - v[0]) / 2;
    Some(timeutil::iso_ms(&m))
}
