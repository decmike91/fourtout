//! Annotation des objets du ciel profond (Messier, NGC, IC — sous-ensemble
//! d'OpenNGC, CC BY-SA 4.0, voir data/OPENNGC_LICENSE.txt) sur une image
//! résolue (WCS) : liste des objets du champ et image annotée.

use crate::draw::Canvas;
use crate::image::Image;
use crate::wcs::{self, Wcs};
use anyhow::Result;
use std::path::Path;

const CATALOG: &str = include_str!("../data/openngc_lite.csv");

#[derive(Clone, Debug)]
pub struct DsoObject {
    pub label: String,
    pub name: String,
    pub kind: String,
    pub ra: f64,
    pub dec: f64,
    pub maj_arcmin: Option<f64>,
    pub min_arcmin: Option<f64>,
    pub pa: Option<f64>,
    pub mag: Option<f64>,
    pub common: String,
    pub x: f64,
    pub y: f64,
    pub maj_px: f64,
    pub min_px: f64,
    pub type_fr: String,
}

pub fn type_fr(t: &str) -> String {
    match t {
        "G" => "galaxie",
        "GPair" => "paire de galaxies",
        "GTrpl" => "triplet de galaxies",
        "GGroup" => "groupe de galaxies",
        "OCl" => "amas ouvert",
        "GCl" => "amas globulaire",
        "Cl+N" => "amas + nébuleuse",
        "PN" => "nébuleuse planétaire",
        "HII" => "région HII",
        "Neb" => "nébuleuse",
        "EmN" => "nébuleuse en émission",
        "RfN" => "nébuleuse par réflexion",
        "SNR" => "reste de supernova",
        "*Ass" => "association stellaire",
        "DrkN" => "nébuleuse obscure",
        other => other,
    }
    .to_string()
}

pub fn load_objects(path: Option<&Path>) -> Result<Vec<DsoObject>> {
    let txt = match path {
        Some(p) => std::fs::read_to_string(p)?,
        None => CATALOG.to_string(),
    };
    let mut lines = txt.lines();
    let head: Vec<String> = lines.next().unwrap_or("").split(',').map(|s| s.trim().to_string()).collect();
    let col = |n: &str| head.iter().position(|h| h == n);
    let idx: Vec<Option<usize>> = ["label", "name", "type", "ra", "dec", "maj_arcmin", "min_arcmin", "pa", "mag", "common"].iter().map(|n| col(n)).collect();
    let mut out = Vec::new();
    for l in lines {
        if l.trim().is_empty() {
            continue;
        }
        let f: Vec<&str> = l.split(',').collect();
        let s = |k: usize| idx[k].and_then(|i| f.get(i)).map(|v| v.trim().to_string()).unwrap_or_default();
        let n = |k: usize| {
            let v = s(k);
            if v.is_empty() {
                None
            } else {
                v.parse::<f64>().ok()
            }
        };
        let (Some(ra), Some(dec)) = (n(3), n(4)) else { continue };
        out.push(DsoObject {
            label: s(0),
            name: s(1),
            kind: s(2),
            ra,
            dec,
            maj_arcmin: n(5),
            min_arcmin: n(6),
            pa: n(7),
            mag: n(8),
            common: s(9),
            x: 0.0,
            y: 0.0,
            maj_px: 0.0,
            min_px: 0.0,
            type_fr: String::new(),
        });
    }
    Ok(out)
}

/// Objets dont le centre tombe dans l'image.
pub fn objects_in_field(w: &Wcs, width: usize, height: usize, mag_limit: f64, path: Option<&Path>, margin: f64) -> Result<Vec<DsoObject>> {
    let objs = load_objects(path)?;
    let (cra, cdec) = w.pixel_to_world(width as f64 / 2.0, height as f64 / 2.0);
    let scale = w.scale_deg();
    let mut found = Vec::new();
    for mut o in objs {
        if wcs::separation(o.ra, o.dec, cra, cdec) > 90.0 {
            continue;
        }
        let Some((x, y)) = w.world_to_pixel(o.ra, o.dec) else { continue };
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        if -margin <= x && x < width as f64 + margin && -margin <= y && y < height as f64 + margin {
            if !o.label.starts_with('M') {
                if let Some(m) = o.mag {
                    if m > mag_limit {
                        continue;
                    }
                }
            }
            o.x = x;
            o.y = y;
            o.maj_px = o.maj_arcmin.unwrap_or(0.0) / 60.0 / scale;
            o.min_px = o.min_arcmin.or(o.maj_arcmin).unwrap_or(0.0) / 60.0 / scale;
            o.type_fr = type_fr(&o.kind);
            found.push(o);
        }
    }
    found.sort_by(|a, b| {
        (!a.label.starts_with('M'), a.mag.unwrap_or(99.0)).partial_cmp(&(!b.label.starts_with('M'), b.mag.unwrap_or(99.0))).unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(found)
}

/// Angle (degrés image) de la direction du nord céleste au point (x, y).
pub fn north_angle(w: &Wcs, x: f64, y: f64) -> f64 {
    let (ra, dec) = w.pixel_to_world(x, y);
    match w.world_to_pixel(ra, dec + 1.0 / 60.0) {
        Some((xn, yn)) => (yn - y).atan2(xn - x).to_degrees(),
        None => -90.0,
    }
}

/// Image RGB annotée + liste des objets.
pub fn annotate(img: &Image, w: &Wcs, mag_limit: f64, color: [f64; 3], font_scale: Option<f64>, path: Option<&Path>) -> Result<(Image, Vec<DsoObject>)> {
    let mut cv = Canvas::from_rgb_f32(img);
    let objs = objects_in_field(w, img.w, img.h, mag_limit, path, 0.0)?;
    let col = [(color[0] * 255.0) as u8, (color[1] * 255.0) as u8, (color[2] * 255.0) as u8];
    let fs = font_scale.unwrap_or_else(|| (img.w.min(img.h) as f64 / 1400.0).max(0.4));
    let th = (fs * 2.0).round().max(1.0);
    for o in &objs {
        let (x, y) = (o.x.round(), o.y.round());
        let a = (o.maj_px / 2.0).round().max(6.0);
        let b = (o.min_px / 2.0).round().max(6.0);
        let ang = north_angle(w, o.x, o.y) + 90.0 - o.pa.unwrap_or(0.0);
        cv.ellipse(x, y, a, b, ang, col, th);
        let mut label = o.label.clone();
        if !o.common.is_empty() && o.label.starts_with('M') {
            label += &format!(" ({})", o.common);
        }
        let label: String = label.chars().map(|c| if c.is_ascii() { c } else { '?' }).collect();
        // Hershey simplex d'OpenCV : hauteur ≈ 22·fs px
        cv.text(&label, x + a + 4.0, y - 4.0, fs * 22.0 / 8.0, col);
    }
    Ok((cv.to_image(), objs))
}
