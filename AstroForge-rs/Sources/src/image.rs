//! Type image commun : flottants 32 bits dans [0..1] (convention), mono
//! (`c == 1`) ou couleur RGB (`c == 3`), stockage entrelacé ligne par ligne
//! (indice = (y·w + x)·c + canal) — l'équivalent des tableaux NumPy HxW /
//! HxWx3 du programme Python.

use anyhow::{bail, Result};
use rayon::prelude::*;

#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub c: usize,
    pub data: Vec<f32>,
}

impl Image {
    pub fn new(w: usize, h: usize, c: usize) -> Self {
        Image { w, h, c, data: vec![0.0; w * h * c] }
    }

    pub fn filled(w: usize, h: usize, c: usize, v: f32) -> Self {
        Image { w, h, c, data: vec![v; w * h * c] }
    }

    pub fn from_vec(w: usize, h: usize, c: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), w * h * c, "taille de données incohérente");
        Image { w, h, c, data }
    }

    /// Image mono construite par une fonction (x, y) -> valeur.
    pub fn from_fn(w: usize, h: usize, f: impl Fn(usize, usize) -> f32 + Sync) -> Self {
        let mut data = vec![0.0f32; w * h];
        data.par_chunks_mut(w.max(1)).enumerate().for_each(|(y, row)| {
            for (x, v) in row.iter_mut().enumerate() {
                *v = f(x, y);
            }
        });
        Image { w, h, c: 1, data }
    }

    #[inline]
    pub fn is_color(&self) -> bool {
        self.c >= 3
    }

    #[inline]
    pub fn npix(&self) -> usize {
        self.w * self.h
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn idx(&self, x: usize, y: usize, ch: usize) -> usize {
        (y * self.w + x) * self.c + ch
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, ch: usize) -> f32 {
        self.data[(y * self.w + x) * self.c + ch]
    }

    #[inline]
    pub fn set(&mut self, x: usize, y: usize, ch: usize, v: f32) {
        let i = (y * self.w + x) * self.c + ch;
        self.data[i] = v;
    }

    /// Forme (h, w, c) — pour les messages.
    pub fn shape_str(&self) -> String {
        if self.c == 1 {
            format!("({}, {})", self.h, self.w)
        } else {
            format!("({}, {}, {})", self.h, self.w, self.c)
        }
    }

    pub fn same_shape(&self, o: &Image) -> bool {
        self.w == o.w && self.h == o.h && self.c == o.c
    }

    /// Plan d'un canal (image mono).
    pub fn channel(&self, ch: usize) -> Image {
        if self.c == 1 {
            return self.clone();
        }
        let c = self.c;
        let data: Vec<f32> = self.data.par_chunks(c).map(|p| p[ch]).collect();
        Image { w: self.w, h: self.h, c: 1, data }
    }

    pub fn channels(&self) -> Vec<Image> {
        (0..self.c).map(|k| self.channel(k)).collect()
    }

    pub fn set_channel(&mut self, ch: usize, plane: &Image) {
        assert!(plane.c == 1 && plane.w == self.w && plane.h == self.h);
        let c = self.c;
        self.data.par_chunks_mut(c).zip(plane.data.par_iter()).for_each(|(p, v)| p[ch] = *v);
    }

    /// Assemble des plans mono en une image multi-canaux.
    pub fn from_planes(planes: &[Image]) -> Image {
        assert!(!planes.is_empty());
        let (w, h, c) = (planes[0].w, planes[0].h, planes.len());
        if c == 1 {
            return planes[0].clone();
        }
        let mut data = vec![0.0f32; w * h * c];
        data.par_chunks_mut(c).enumerate().for_each(|(i, p)| {
            for (k, pl) in planes.iter().enumerate() {
                p[k] = pl.data[i];
            }
        });
        Image { w, h, c, data }
    }

    /// Applique `f` à chaque canal (plan mono) puis réassemble.
    pub fn map_planes(&self, f: impl Fn(&Image) -> Image + Sync) -> Image {
        if self.c == 1 {
            return f(self);
        }
        let planes: Vec<Image> = (0..self.c).into_par_iter().map(|k| f(&self.channel(k))).collect();
        Image::from_planes(&planes)
    }

    /// Luminance = moyenne des 3 premiers canaux (convention `_lum` Python).
    pub fn mean_lum(&self) -> Image {
        if self.c == 1 {
            return self.clone();
        }
        let c = self.c;
        let n = c.min(3) as f32;
        let data: Vec<f32> = self
            .data
            .par_chunks(c)
            .map(|p| p[..c.min(3)].iter().sum::<f32>() / n)
            .collect();
        Image { w: self.w, h: self.h, c: 1, data }
    }

    /// Luminance Rec.709 (0,2126 R + 0,7152 G + 0,0722 B).
    pub fn rec709(&self) -> Image {
        if self.c == 1 {
            return self.clone();
        }
        let c = self.c;
        let data: Vec<f32> = self
            .data
            .par_chunks(c)
            .map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2])
            .collect();
        Image { w: self.w, h: self.h, c: 1, data }
    }

    /// Convertit en mono si besoin (moyenne des canaux).
    pub fn to_mono(&self) -> Image {
        self.mean_lum()
    }

    /// Mono -> RGB (canal dupliqué).
    pub fn to_rgb(&self) -> Image {
        if self.c == 3 {
            return self.clone();
        }
        if self.c > 3 {
            return self.take_channels(3);
        }
        let data: Vec<f32> = self.data.iter().flat_map(|&v| [v, v, v]).collect();
        Image { w: self.w, h: self.h, c: 3, data }
    }

    pub fn take_channels(&self, n: usize) -> Image {
        if n >= self.c {
            return self.clone();
        }
        let c = self.c;
        let data: Vec<f32> = self.data.chunks(c).flat_map(|p| p[..n].to_vec()).collect();
        Image { w: self.w, h: self.h, c: n, data }
    }

    pub fn map(&self, f: impl Fn(f32) -> f32 + Sync) -> Image {
        let data: Vec<f32> = self.data.par_iter().map(|&v| f(v)).collect();
        Image { w: self.w, h: self.h, c: self.c, data }
    }

    pub fn map_inplace(&mut self, f: impl Fn(f32) -> f32 + Sync) {
        self.data.par_iter_mut().for_each(|v| *v = f(*v));
    }

    pub fn zip_map(&self, o: &Image, f: impl Fn(f32, f32) -> f32 + Sync) -> Image {
        assert_eq!(self.data.len(), o.data.len(), "tailles différentes");
        let data: Vec<f32> = self.data.par_iter().zip(o.data.par_iter()).map(|(&a, &b)| f(a, b)).collect();
        Image { w: self.w, h: self.h, c: self.c, data }
    }

    pub fn clip01(&self) -> Image {
        self.map(|v| if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) })
    }

    pub fn clip(&self, lo: f32, hi: f32) -> Image {
        self.map(|v| v.clamp(lo, hi))
    }

    pub fn clip_min(&self, lo: f32) -> Image {
        self.map(|v| if v < lo { lo } else { v })
    }

    pub fn max_value(&self) -> f32 {
        self.data.par_iter().cloned().reduce(|| f32::NEG_INFINITY, f32::max)
    }

    pub fn min_value(&self) -> f32 {
        self.data.par_iter().cloned().reduce(|| f32::INFINITY, f32::min)
    }

    pub fn mean(&self) -> f64 {
        if self.data.is_empty() {
            return f64::NAN;
        }
        self.data.par_iter().map(|&v| v as f64).sum::<f64>() / self.data.len() as f64
    }

    pub fn sum(&self) -> f64 {
        self.data.par_iter().map(|&v| v as f64).sum()
    }

    pub fn all_finite(&self) -> bool {
        self.data.par_iter().all(|v| v.is_finite())
    }

    /// Recadrage (x, y, largeur, hauteur), borné à l'image.
    pub fn crop(&self, x0: usize, y0: usize, cw: usize, ch: usize) -> Image {
        let x1 = (x0 + cw).min(self.w);
        let y1 = (y0 + ch).min(self.h);
        let x0 = x0.min(x1);
        let y0 = y0.min(y1);
        let (nw, nh) = (x1 - x0, y1 - y0);
        let c = self.c;
        let mut out = Image::new(nw, nh, c);
        for y in 0..nh {
            let src = &self.data[((y + y0) * self.w + x0) * c..((y + y0) * self.w + x1) * c];
            out.data[y * nw * c..(y + 1) * nw * c].copy_from_slice(src);
        }
        out
    }

    /// Copie `src` dans l'image à partir de (x0, y0) (zone bornée).
    pub fn paste(&mut self, src: &Image, x0: usize, y0: usize) {
        assert_eq!(self.c, src.c);
        let c = self.c;
        for y in 0..src.h {
            let yy = y + y0;
            if yy >= self.h {
                break;
            }
            let nw = src.w.min(self.w.saturating_sub(x0));
            if nw == 0 {
                break;
            }
            let d0 = (yy * self.w + x0) * c;
            self.data[d0..d0 + nw * c].copy_from_slice(&src.data[y * src.w * c..(y * src.w + nw) * c]);
        }
    }

    /// Sous-échantillonnage d'un plan selon un motif 2×2 (offset oy, ox).
    pub fn sub2(&self, oy: usize, ox: usize) -> Image {
        assert_eq!(self.c, 1);
        let nh = (self.h + 1 - oy.min(self.h)) / 2;
        let nw = (self.w + 1 - ox.min(self.w)) / 2;
        let nh = if oy >= self.h { 0 } else { nh };
        let nw = if ox >= self.w { 0 } else { nw };
        let mut out = Image::new(nw, nh, 1);
        for y in 0..nh {
            for x in 0..nw {
                out.data[y * nw + x] = self.data[(2 * y + oy) * self.w + 2 * x + ox];
            }
        }
        out
    }

    /// Réinjecte un sous-plan 2×2 (inverse de `sub2`).
    pub fn put_sub2(&mut self, oy: usize, ox: usize, sub: &Image) {
        for y in 0..sub.h {
            for x in 0..sub.w {
                let (yy, xx) = (2 * y + oy, 2 * x + ox);
                if yy < self.h && xx < self.w {
                    self.data[yy * self.w + xx] = sub.data[y * sub.w + x];
                }
            }
        }
    }

    pub fn add_scalar(&self, s: f32) -> Image {
        self.map(|v| v + s)
    }

    pub fn scale(&self, s: f32) -> Image {
        self.map(|v| v * s)
    }

    pub fn check_mono_or_rgb(&self) -> Result<()> {
        if !(self.c == 1 || self.c == 3) {
            bail!("Image attendue : mono ou RGB.");
        }
        if self.data.is_empty() {
            bail!("Image vide.");
        }
        Ok(())
    }

    /// Valeurs du pixel (x, y) (tous canaux).
    #[inline]
    pub fn pixel(&self, x: usize, y: usize) -> &[f32] {
        let i = (y * self.w + x) * self.c;
        &self.data[i..i + self.c]
    }

    #[inline]
    pub fn pixel_mut(&mut self, x: usize, y: usize) -> &mut [f32] {
        let i = (y * self.w + x) * self.c;
        let c = self.c;
        &mut self.data[i..i + c]
    }

    /// Ligne y (tous canaux).
    #[inline]
    pub fn row(&self, y: usize) -> &[f32] {
        &self.data[y * self.w * self.c..(y + 1) * self.w * self.c]
    }
}

/// Conversion flottant -> u16 (écrêtage [0,1], troncature comme NumPy astype).
#[inline]
pub fn f_to_u16(v: f32) -> u16 {
    let x = (v as f64).clamp(0.0, 1.0) * 65535.0;
    if x.is_nan() {
        0
    } else {
        x as u16
    }
}

#[inline]
pub fn f_to_u8(v: f32) -> u8 {
    let x = (v as f64).clamp(0.0, 1.0) * 255.0;
    if x.is_nan() {
        0
    } else {
        x as u8
    }
}

/// Moyenne de plusieurs images de même forme.
pub fn mean_images(imgs: &[&Image]) -> Image {
    assert!(!imgs.is_empty());
    let n = imgs.len() as f64;
    let len = imgs[0].data.len();
    let data: Vec<f32> = (0..len)
        .into_par_iter()
        .map(|i| (imgs.iter().map(|im| im.data[i] as f64).sum::<f64>() / n) as f32)
        .collect();
    Image { w: imgs[0].w, h: imgs[0].h, c: imgs[0].c, data }
}
