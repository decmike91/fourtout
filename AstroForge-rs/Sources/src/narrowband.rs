//! Bande étroite et composition couleur : extraction Ha/OIII d'un capteur
//! couleur (filtre duo-bande), normalisation des canaux, palettes HOO, SHO,
//! HSO, OSH…, combinaison LRGB, alignement des canaux.

use crate::geom::{self, Interp};
use crate::image::Image;
use crate::registration;
use crate::stats;
use anyhow::{bail, Result};
use std::collections::BTreeMap;

pub type Mix = Vec<(String, f64)>;
pub type Palette = BTreeMap<String, Mix>; // sortie "R"/"G"/"B" -> [(source, poids)]

pub const PALETTE_NAMES: [&str; 6] = ["HOO", "HOO équilibrée", "SHO (Hubble)", "HSO", "OSH", "RGB"];

/// Ordre canonique des canaux (ordre de saisie du programme d'origine).
pub const CHANNEL_ORDER: [&str; 7] = ["Ha", "OIII", "SII", "R", "G", "B", "L"];

fn first_key<'a, V>(m: &'a BTreeMap<String, V>) -> Option<String> {
    for k in CHANNEL_ORDER {
        if m.contains_key(k) {
            return Some(k.to_string());
        }
    }
    m.keys().next().cloned()
}

pub fn palette(name: &str) -> Result<Palette> {
    let p: Vec<(&str, Vec<(&str, f64)>)> = match name {
        "HOO" => vec![("R", vec![("Ha", 1.0)]), ("G", vec![("OIII", 1.0)]), ("B", vec![("OIII", 1.0)])],
        "HOO équilibrée" => vec![("R", vec![("Ha", 1.0)]), ("G", vec![("Ha", 0.3), ("OIII", 0.7)]), ("B", vec![("OIII", 1.0)])],
        "SHO (Hubble)" | "SHO" => vec![("R", vec![("SII", 1.0)]), ("G", vec![("Ha", 1.0)]), ("B", vec![("OIII", 1.0)])],
        "HSO" => vec![("R", vec![("Ha", 1.0)]), ("G", vec![("SII", 1.0)]), ("B", vec![("OIII", 1.0)])],
        "OSH" => vec![("R", vec![("OIII", 1.0)]), ("G", vec![("SII", 1.0)]), ("B", vec![("Ha", 1.0)])],
        "RGB" => vec![("R", vec![("R", 1.0)]), ("G", vec![("G", 1.0)]), ("B", vec![("B", 1.0)])],
        _ => bail!("Palette inconnue : {} (choix : {})", name, PALETTE_NAMES.join(", ")),
    };
    Ok(p.into_iter().map(|(k, v)| (k.to_string(), v.into_iter().map(|(s, w)| (s.to_string(), w)).collect())).collect())
}

fn red_offset(pattern: &str) -> Result<(usize, usize)> {
    Ok(match pattern.to_uppercase().as_str() {
        "RGGB" => (0, 0),
        "BGGR" => (1, 1),
        "GRBG" => (0, 1),
        "GBRG" => (1, 0),
        _ => bail!("Motif de Bayer inconnu : {}", pattern),
    })
}

/// CFA mono -> (Ha, OIII) ; demi-résolution, ou agrandis (bicubique).
pub fn extract_ha_oiii(cfa: &Image, pattern: &str, upscale: bool) -> Result<(Image, Image)> {
    if cfa.c != 1 {
        bail!("Extraction Ha/OIII : image CFA mono (non dématricée) attendue");
    }
    let (h, w) = ((cfa.h / 2) * 2, (cfa.w / 2) * 2);
    let a = cfa.crop(0, 0, w, h);
    let (ry, rx) = red_offset(pattern)?;
    let (by, bx) = (1 - ry, 1 - rx);
    let ha = a.sub2(ry, rx);
    let g1 = a.sub2(ry, bx);
    let g2 = a.sub2(by, rx);
    let b = a.sub2(by, bx);
    let mut oiii = g1.clone();
    for i in 0..oiii.data.len() {
        oiii.data[i] = (g1.data[i] + g2.data[i] + b.data[i]) / 3.0;
    }
    let (ha, oiii) = if upscale { (geom::resize(&ha, w, h, Interp::Cubic), geom::resize(&oiii, w, h, Interp::Cubic)) } else { (ha, oiii) };
    Ok((ha.clip_min(0.0), oiii.clip_min(0.0)))
}

/// Approximation sur une image dématricée : Ha = R, OIII = (G+B)/2.
pub fn extract_from_rgb(rgb: &Image) -> Result<(Image, Image)> {
    if rgb.c < 3 {
        bail!("Image couleur attendue");
    }
    let r = rgb.channel(0);
    let g = rgb.channel(1);
    let b = rgb.channel(2);
    Ok((r, g.zip_map(&b, |x, y| (x + y) * 0.5)))
}

/// Égalise fond (médiane) et dispersion (MAD) de `img` sur `ref_`.
pub fn normalize_to(img: &Image, ref_: &Image) -> Image {
    let ma = stats::median(&img.data) as f64;
    let mr = stats::median(&ref_.data) as f64;
    let sa = stats::mad_about(&img.data, ma as f32) as f64 + 1e-9;
    let sr = stats::mad_about(&ref_.data, mr as f32) as f64 + 1e-9;
    img.map(|v| (((v as f64 - ma) * (sr / sa) + mr) as f32).max(0.0))
}

/// Aligne tous les canaux sur le canal de référence.
pub fn align_channels(channels: &BTreeMap<String, Image>, ref_key: Option<&str>) -> BTreeMap<String, Image> {
    let mut out = BTreeMap::new();
    let Some(first) = first_key(channels) else { return out };
    let rk = ref_key.map(|s| s.to_string()).filter(|k| channels.contains_key(k)).unwrap_or(first);
    let r = channels[&rk].clone();
    for (k, img) in channels {
        if *k == rk {
            out.insert(k.clone(), r.clone());
            continue;
        }
        let im = if img.w != r.w || img.h != r.h { geom::resize(img, r.w, r.h, Interp::Cubic) } else { img.clone() };
        let m = registration::estimate_transform(&r, &im, true);
        out.insert(k.clone(), geom::warp(&im, &m));
    }
    out
}

/// Compose une image RGB à partir de canaux mono alignés.
pub fn compose(channels: &BTreeMap<String, Image>, pal: &Palette, normalize: bool, ref_key: Option<&str>) -> Result<Image> {
    let needed: std::collections::BTreeSet<String> = pal.values().flat_map(|m| m.iter().map(|x| x.0.clone())).collect();
    let missing: Vec<String> = needed.iter().filter(|k| !channels.contains_key(*k)).cloned().collect();
    if !missing.is_empty() {
        bail!("Canaux manquants pour la palette : {}", missing.join(", "));
    }
    let mut chans: BTreeMap<String, Image> = channels.iter().filter(|(k, _)| needed.contains(*k)).map(|(k, v)| (k.clone(), v.clone())).collect();
    let first = chans.values().next().unwrap();
    let (w, h) = (first.w, first.h);
    if chans.values().any(|v| v.c != 1 || v.w != w || v.h != h) {
        bail!("Les canaux doivent être des images mono de même taille");
    }
    if normalize && chans.len() > 1 {
        let rk = match ref_key {
            Some(k) if chans.contains_key(k) => k.to_string(),
            _ => {
                first_key(&chans).unwrap()
            }
        };
        let r = chans[&rk].clone();
        for (k, v) in chans.iter_mut() {
            if *k != rk {
                *v = normalize_to(v, &r);
            }
        }
    }
    let mut planes = Vec::new();
    for ch in ["R", "G", "B"] {
        let mix = pal.get(ch).cloned().unwrap_or_default();
        if mix.is_empty() {
            planes.push(Image::new(w, h, 1));
            continue;
        }
        let tot: f64 = mix.iter().map(|m| m.1).sum::<f64>();
        let tot = if tot == 0.0 { 1.0 } else { tot };
        let mut p = Image::new(w, h, 1);
        for (src, wt) in &mix {
            let s = &chans[src];
            for i in 0..p.data.len() {
                p.data[i] += s.data[i] * *wt as f32;
            }
        }
        planes.push(p.scale((1.0 / tot) as f32));
    }
    Ok(Image::from_planes(&planes).clip01())
}

/// LRGB : la luminance de l'image couleur est remplacée par L.
pub fn lrgb(lum: &Image, rgb: &Image, saturation: f64) -> Result<Image> {
    if rgb.c < 3 || lum.w != rgb.w || lum.h != rgb.h || lum.c != 1 {
        bail!("LRGB : L mono et RGB couleur de même taille requis");
    }
    let mut out = Image::new(rgb.w, rgb.h, 3);
    let s = saturation as f32;
    for i in 0..lum.data.len() {
        let p = &rgb.data[i * rgb.c..i * rgb.c + 3];
        let y = 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2];
        let l = lum.data[i];
        let k = l / y.max(1e-4);
        let mut v = [l + (p[0] - y) * s * k, l + (p[1] - y) * s * k, l + (p[2] - y) * s * k];
        let mx = v[0].max(v[1]).max(v[2]);
        if mx > 1.0 {
            for x in v.iter_mut() {
                *x /= mx;
            }
        }
        for k in 0..3 {
            out.data[i * 3 + k] = v[k].clamp(0.0, 1.0);
        }
    }
    Ok(out)
}
