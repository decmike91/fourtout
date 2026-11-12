//! Dématriçage bilinéaire d'une image CFA (matrice de Bayer 2×2 nommée par
//! son coin haut-gauche : RGGB, BGGR, GRBG, GBRG).

use crate::image::Image;
use crate::ndimage::{fold, Border};
use anyhow::{bail, Result};
use rayon::prelude::*;

pub const BAYER_PATTERNS: [&str; 4] = ["RGGB", "BGGR", "GRBG", "GBRG"];

/// Couleur (0 R, 1 G, 2 B) des 4 positions du motif [(0,0), (0,1), (1,0), (1,1)].
pub fn pattern_colors(pattern: &str) -> Result<[usize; 4]> {
    let p = pattern.to_uppercase().replace(' ', "");
    if !BAYER_PATTERNS.contains(&p.as_str()) {
        bail!("Matrice de Bayer inconnue : {} (choix: {:?})", pattern, BAYER_PATTERNS);
    }
    let mut out = [0usize; 4];
    for (i, ch) in p.chars().enumerate() {
        out[i] = match ch {
            'R' => 0,
            'G' => 1,
            _ => 2,
        };
    }
    Ok(out)
}

pub fn debayer(cfa: &Image, pattern: &str) -> Result<Image> {
    if cfa.c != 1 {
        bail!("Dématriçage : image CFA mono attendue");
    }
    let col = pattern_colors(pattern)?;
    let (w, h) = (cfa.w, cfa.h);
    let src = &cfa.data;
    let color_at = |x: usize, y: usize| col[(y % 2) * 2 + (x % 2)];
    let mut out = Image::new(w, h, 3);
    out.data.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let own = color_at(x, y);
            let v = src[y * w + x].clamp(0.0, 1.0);
            let mut acc = [0.0f32; 3];
            let mut cnt = [0u32; 3];
            for dy in -1isize..=1 {
                let yy = fold(y as isize + dy, h, Border::Mirror).unwrap();
                for dx in -1isize..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let xx = fold(x as isize + dx, w, Border::Mirror).unwrap();
                    let k = color_at(xx, yy);
                    if k == own {
                        continue;
                    }
                    acc[k] += src[yy * w + xx].clamp(0.0, 1.0);
                    cnt[k] += 1;
                }
            }
            for k in 0..3 {
                row[x * 3 + k] = if k == own {
                    v
                } else if cnt[k] > 0 {
                    acc[k] / cnt[k] as f32
                } else {
                    v
                };
            }
        }
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn flat_grey() {
        let cfa = Image::filled(8, 6, 1, 0.5);
        let rgb = debayer(&cfa, "RGGB").unwrap();
        assert!(rgb.data.iter().all(|&v| (v - 0.5).abs() < 1e-6));
    }
    #[test]
    fn red_only() {
        // capteur RGGB éclairé en rouge pur
        let cfa = Image::from_fn(8, 8, |x, y| if x % 2 == 0 && y % 2 == 0 { 1.0 } else { 0.0 });
        let rgb = debayer(&cfa, "RGGB").unwrap();
        let p = rgb.pixel(3, 3);
        assert!(p[0] > 0.9 && p[1] < 0.1 && p[2] < 0.1);
    }
}
