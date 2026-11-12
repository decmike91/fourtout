//! Masques d'application (0 = protégé, 1 = traité) et outils d'étoiles :
//! masques de luminance / plage / étoiles, mélange à travers un masque,
//! réduction d'étoiles, extraction et recombinaison, débruitage de chrominance.

use crate::image::Image;
use crate::ndimage;
use crate::stats;
use crate::wavelets;
use anyhow::{bail, Result};

pub const MASK_TYPES: [&str; 6] = ["Aucun", "Luminance", "Luminance inversée (fond)", "Étoiles", "Sans étoiles (inverse)", "Plage"];

fn lum(img: &Image) -> Image {
    img.rec709()
}

fn smoothstep(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

pub fn luminance_mask(img: &Image, low: f64, high: f64, fuzz: f64, blur: f64) -> Image {
    let l = lum(img);
    let f = fuzz.max(1e-4) as f32;
    let (lo_v, hi_v) = (low as f32, high as f32);
    let mut m = l.map(|v| {
        let lo = if low <= 0.0 { 1.0 } else { smoothstep((v - (lo_v - f)) / f) };
        let hi = if high >= 1.0 { 1.0 } else { smoothstep(((hi_v + f) - v) / f) };
        lo * hi
    });
    if blur > 0.0 {
        m = ndimage::gaussian(&m, blur);
    }
    m.clip01()
}

/// Masque d'étoiles (chapeau haut-de-forme, composantes compactes, dilatation, flou).
pub fn star_mask(img: &Image, threshold: f64, grow: usize, blur: f64, max_size: usize) -> Image {
    let l = lum(img);
    let bg = ndimage::grey_opening(&l, 9);
    let th = l.zip_map(&bg, |a, b| a - b);
    let med = stats::median(&th.data);
    let sig = 1.4826 * stats::mad_about(&th.data, med) as f64 + 1e-6;
    let thr = (threshold * sig) as f32;
    let (w, h) = (l.w, l.h);
    let core: Vec<bool> = th.data.iter().map(|&v| v > thr).collect();
    let (lab, n) = ndimage::label(&core, w, h);
    let mut core = core;
    if n > 0 {
        let objs = ndimage::find_objects(&lab, w, n);
        let keep: Vec<bool> = std::iter::once(false)
            .chain(objs.iter().map(|o| match o {
                Some((y0, y1, x0, x1)) => (y1 - y0) <= max_size && (x1 - x0) <= max_size,
                None => false,
            }))
            .collect();
        core = lab.iter().map(|&l| keep[l as usize]).collect();
    }
    if grow > 0 {
        core = ndimage::binary_dilation(&core, w, h, grow);
    }
    let mut m = Image::from_vec(w, h, 1, core.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect());
    if blur > 0.0 {
        m = ndimage::gaussian(&m, blur).map(|v| (v * 1.5).clamp(0.0, 1.0));
    }
    m
}

pub fn make_mask(img: &Image, kind: &str, low: f64, high: f64, fuzz: f64, blur: f64) -> Result<Option<Image>> {
    Ok(match kind {
        "" | "Aucun" => None,
        "Luminance" => Some(luminance_mask(img, low, high, fuzz, blur)),
        k if k.starts_with("Luminance inversée") => Some(luminance_mask(img, low, high, fuzz, blur).map(|v| 1.0 - v)),
        "Étoiles" => Some(star_mask(img, 4.0, 2, blur.max(0.5), 40)),
        k if k.starts_with("Sans étoiles") => Some(star_mask(img, 4.0, 2, blur.max(0.5), 40).map(|v| 1.0 - v)),
        "Plage" => {
            let l = lum(img);
            let m = l.map(|v| if v as f64 >= low && v as f64 <= high { 1.0 } else { 0.0 });
            Some(if blur > 0.0 { ndimage::gaussian(&m, blur).clip01() } else { m })
        }
        _ => bail!("Masque inconnu : {}", kind),
    })
}

/// Mélange : 0 -> original, 1 -> traité.
pub fn apply_masked(original: &Image, processed: &Image, mask: Option<&Image>) -> Result<Image> {
    let Some(m) = mask else { return Ok(processed.clone()) };
    if processed.w != m.w || processed.h != m.h {
        bail!("Le masque et l'image n'ont pas la même taille");
    }
    if !original.same_shape(processed) {
        return Ok(processed.clone());
    }
    let c = processed.c;
    let mut out = processed.clone();
    for i in 0..m.data.len() {
        let mv = m.data[i];
        for k in 0..c {
            let j = i * c + k;
            out.data[j] = original.data[j] * (1.0 - mv) + processed.data[j] * mv;
        }
    }
    Ok(out)
}

/// Réduction d'étoiles : érosion 3×3 répétée dans le masque d'étoiles.
pub fn reduce_stars(img: &Image, amount: f64, iterations: usize, threshold: f64) -> Image {
    let k = amount.clamp(0.0, 1.0);
    if k <= 0.0 {
        return img.clip01();
    }
    let m = star_mask(img, threshold, 1, 1.0, 40).scale(k as f32);
    let mut er = img.clone();
    for _ in 0..iterations.max(1) {
        er = er.map_planes(|p| ndimage::minimum_filter(p, 3));
    }
    let l = lum(img);
    let bgk = ndimage::median_filter(&l, 9);
    let c = img.c;
    for i in 0..l.data.len() {
        let floor = bgk.data[i].min(l.data[i]);
        for kk in 0..c {
            let j = i * c + kk;
            let base = if c == 3 { img.data[j] / l.data[i].max(1e-6) * floor } else { floor };
            er.data[j] = er.data[j].max(base);
        }
    }
    apply_masked(img, &er, Some(&m)).map(|x| x.clip01()).unwrap_or_else(|_| img.clip01())
}

pub fn extract_stars(img: &Image, starless: &Image) -> Image {
    img.zip_map(starless, |a, b| (a - b).clamp(0.0, 1.0))
}

/// Recombinaison « Addition » ou « Écran » ; gain des étoiles réglable.
pub fn recombine(starless: &Image, stars: &Image, mode: &str, stars_gain: f64) -> Result<Image> {
    if !starless.same_shape(stars) {
        bail!("Image sans étoiles et étoiles de tailles différentes");
    }
    let g = stars_gain as f32;
    Ok(if mode.starts_with("Add") {
        starless.zip_map(stars, |a, b| (a + (b * g).clamp(0.0, 1.0)).clamp(0.0, 1.0))
    } else {
        starless.zip_map(stars, |a, b| 1.0 - (1.0 - a.clamp(0.0, 1.0)) * (1.0 - (b * g).clamp(0.0, 1.0)))
    })
}

/// Débruitage de la chrominance seule (ondelettes, luminance intacte).
pub fn chroma_denoise(img: &Image, strength: f64, scales: usize) -> Image {
    let s = strength.clamp(0.0, 1.0);
    if img.c != 3 || s <= 0.0 {
        return img.clip01();
    }
    let l = lum(img);
    let r = img.channel(0);
    let b = img.channel(2);
    let c1 = r.zip_map(&l, |a, b| a - b);
    let c2 = b.zip_map(&l, |a, b| a - b);
    let den = |c: &Image| -> Image {
        let layers = wavelets::atrous_decompose(c, scales);
        let n = layers.len();
        let mut new = Vec::new();
        for (i, layer) in layers[..n - 1].iter().enumerate() {
            let mut a: Vec<f32> = layer.data.iter().map(|v| v.abs()).collect();
            let sig = stats::median_mut(&mut a) as f64 / 0.6745 + 1e-9;
            let t = sig * (3.0 * s) / (1.0 + 0.5 * i as f64);
            new.push(layer.map(|v| (v.signum() as f64 * ((v.abs() as f64) - t).max(0.0)) as f32));
        }
        new.push(layers[n - 1].clone());
        wavelets::atrous_recompose(&new, None, None)
    };
    let (o1, o2) = rayon::join(|| den(&c1), || den(&c2));
    let mut out = Image::new(img.w, img.h, 3);
    for i in 0..l.data.len() {
        let rr = l.data[i] + o1.data[i];
        let bb = l.data[i] + o2.data[i];
        let gg = (l.data[i] - 0.2126 * rr - 0.0722 * bb) / 0.7152;
        out.data[i * 3] = rr.clamp(0.0, 1.0);
        out.data[i * 3 + 1] = gg.clamp(0.0, 1.0);
        out.data[i * 3 + 2] = bb.clamp(0.0, 1.0);
    }
    out
}
