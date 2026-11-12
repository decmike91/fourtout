//! Outils communs des tests : générateur pseudo-aléatoire, champs d'étoiles.
#![allow(dead_code)]
use astroforge::io::{fits, HVal, Header};
use astroforge::Image;
use std::path::{Path, PathBuf};

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xD1B54A32D192ED03)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    pub fn uniform(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64)
    }
    pub fn normal(&mut self, mu: f64, sigma: f64) -> f64 {
        let u1 = self.uniform(1e-12, 1.0);
        let u2 = self.uniform(0.0, 1.0);
        mu + sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    pub fn noise(&mut self, img: &Image, sigma: f64) -> Image {
        let mut o = img.clone();
        for v in o.data.iter_mut() {
            *v += self.normal(0.0, sigma) as f32;
        }
        o
    }
}

/// Champ d'étoiles gaussiennes décalable (comme la fixture Python star_field).
pub fn star_field(h: usize, w: usize, dx: f64, dy: f64, n: usize, seed: u64, fwhm: f64, bg: f32) -> Image {
    let mut r = Rng::new(seed);
    let stars: Vec<(f64, f64, f64)> = (0..n).map(|_| (r.uniform(10.0, w as f64 - 10.0), r.uniform(10.0, h as f64 - 10.0), r.uniform(0.2, 0.8))).collect();
    let s = fwhm / 2.3548;
    Image::from_fn(w, h, |x, y| {
        let mut v = bg as f64;
        for &(sx, sy, a) in &stars {
            let d2 = (x as f64 - sx - dx).powi(2) + (y as f64 - sy - dy).powi(2);
            if d2 < 200.0 {
                v += a * (-d2 / (2.0 * s * s)).exp();
            }
        }
        v as f32
    })
}

pub fn sf(dx: f64, dy: f64) -> Image {
    star_field(128, 128, dx, dy, 25, 3, 3.0, 0.05)
}

/// Écrit un FITS uint16 (comme une caméra) avec en-tête.
pub fn write_fits(dir: &Path, name: &str, img: &Image, hdr: &[(&str, HVal)]) -> PathBuf {
    let p = dir.join(name);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    let mut h = Header::new();
    for (k, v) in hdr {
        h.set(k, v.clone());
    }
    fits::write_fits_u16(&p, &img.clip01(), Some(&h)).unwrap();
    p
}

pub fn mean_abs_diff(a: &Image, b: &Image) -> f64 {
    a.data.iter().zip(&b.data).map(|(x, y)| (x - y).abs() as f64).sum::<f64>() / a.data.len() as f64
}

pub fn corr(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mb = b.iter().map(|&v| v as f64).sum::<f64>() / n;
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (&x, &y) in a.iter().zip(b) {
        let (dx, dy) = (x as f64 - ma, y as f64 - mb);
        sab += dx * dy;
        saa += dx * dx;
        sbb += dy * dy;
    }
    sab / (saa * sbb).sqrt()
}

pub fn inner(img: &Image, m: usize) -> Image {
    img.crop(m, m, img.w - 2 * m, img.h - 2 * m)
}

/// Image RGB construite pixel par pixel.
pub fn rgb_fn(w: usize, h: usize, f: impl Fn(usize, usize) -> [f32; 3]) -> Image {
    let mut d = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            d.extend_from_slice(&f(x, y));
        }
    }
    Image::from_vec(w, h, 3, d)
}
