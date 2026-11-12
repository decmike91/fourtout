//! Géométrie : matrices affines 2×3 (convention OpenCV), rééchantillonnage
//! (warpAffine, remap, resize) en interpolation bicubique (a = −0,75 comme
//! OpenCV), bilinéaire, Lanczos-4 ou par surface.

use crate::image::Image;
use crate::ndimage::{fold, Border};
use rayon::prelude::*;

/// Matrice affine 2×3 : [[a, b, tx], [c, d, ty]] (coordonnées source -> destination).
pub type Affine = [[f64; 3]; 2];

pub const IDENTITY: Affine = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Interp {
    Nearest,
    Linear,
    Cubic,
    Lanczos4,
    Area,
}

/// cv2.getRotationMatrix2D : rotation de `angle_deg` (sens trigonométrique,
/// axe y vers le bas) et mise à l'échelle autour de (cx, cy).
pub fn rotation_matrix(cx: f64, cy: f64, angle_deg: f64, scale: f64) -> Affine {
    let a = angle_deg.to_radians();
    let alpha = scale * a.cos();
    let beta = scale * a.sin();
    [
        [alpha, beta, (1.0 - alpha) * cx - beta * cy],
        [-beta, alpha, beta * cx + (1.0 - alpha) * cy],
    ]
}

pub fn invert(m: &Affine) -> Affine {
    let (a, b, c, d) = (m[0][0], m[0][1], m[1][0], m[1][1]);
    let det = a * d - b * c;
    let det = if det.abs() < 1e-300 { 1e-300 } else { det };
    let ia = d / det;
    let ib = -b / det;
    let ic = -c / det;
    let id = a / det;
    [
        [ia, ib, -(ia * m[0][2] + ib * m[1][2])],
        [ic, id, -(ic * m[0][2] + id * m[1][2])],
    ]
}

/// Composition : a ∘ b (applique b puis a).
pub fn compose(a: &Affine, b: &Affine) -> Affine {
    [
        [
            a[0][0] * b[0][0] + a[0][1] * b[1][0],
            a[0][0] * b[0][1] + a[0][1] * b[1][1],
            a[0][0] * b[0][2] + a[0][1] * b[1][2] + a[0][2],
        ],
        [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
            a[1][0] * b[0][1] + a[1][1] * b[1][1],
            a[1][0] * b[0][2] + a[1][1] * b[1][2] + a[1][2],
        ],
    ]
}

#[inline]
pub fn apply(m: &Affine, x: f64, y: f64) -> (f64, f64) {
    (m[0][0] * x + m[0][1] * y + m[0][2], m[1][0] * x + m[1][1] * y + m[1][2])
}

pub fn det2(m: &Affine) -> f64 {
    m[0][0] * m[1][1] - m[0][1] * m[1][0]
}

pub fn is_finite(m: &Affine) -> bool {
    m.iter().all(|r| r.iter().all(|v| v.is_finite()))
}

#[inline]
fn cubic_w(t: f64) -> [f64; 4] {
    const A: f64 = -0.75;
    let x0 = 1.0 + t;
    let x1 = t;
    let x2 = 1.0 - t;
    let w0 = ((A * x0 - 5.0 * A) * x0 + 8.0 * A) * x0 - 4.0 * A;
    let w1 = ((A + 2.0) * x1 - (A + 3.0)) * x1 * x1 + 1.0;
    let w2 = ((A + 2.0) * x2 - (A + 3.0)) * x2 * x2 + 1.0;
    let w3 = 1.0 - w0 - w1 - w2;
    [w0, w1, w2, w3]
}

#[inline]
fn lanczos4_w(t: f64) -> [f64; 8] {
    let mut w = [0.0f64; 8];
    if t.abs() < 1e-12 {
        w[3] = 1.0;
        return w;
    }
    let mut sum = 0.0;
    for (i, wi) in w.iter_mut().enumerate() {
        let x = (3.0 - i as f64 + t) * std::f64::consts::PI; // distance (i-3) - t
        let x = -x;
        let v = if x.abs() < 1e-12 { 1.0 } else { 4.0 * (x).sin() * (x / 4.0).sin() / (x * x) };
        *wi = v;
        sum += v;
    }
    for wi in w.iter_mut() {
        *wi /= sum;
    }
    w
}

/// Échantillonne tous les canaux au point (x, y).
#[inline]
pub fn sample(img: &Image, x: f64, y: f64, interp: Interp, border: Border, out: &mut [f32]) {
    let (w, h, c) = (img.w, img.h, img.c);
    match interp {
        Interp::Nearest => {
            let xi = fold(x.round() as isize, w, border);
            let yi = fold(y.round() as isize, h, border);
            match (xi, yi) {
                (Some(xi), Some(yi)) => out.copy_from_slice(img.pixel(xi, yi)),
                _ => out.iter_mut().for_each(|v| *v = 0.0),
            }
        }
        Interp::Linear | Interp::Area => {
            let x0 = x.floor();
            let y0 = y.floor();
            let fx = x - x0;
            let fy = y - y0;
            let (x0, y0) = (x0 as isize, y0 as isize);
            let mut acc = [0.0f64; 4];
            for (dy, wy) in [(0isize, 1.0 - fy), (1, fy)] {
                let yy = fold(y0 + dy, h, border);
                for (dx, wx) in [(0isize, 1.0 - fx), (1, fx)] {
                    let xx = fold(x0 + dx, w, border);
                    if let (Some(xx), Some(yy)) = (xx, yy) {
                        let p = img.pixel(xx, yy);
                        for k in 0..c {
                            acc[k] += wy * wx * p[k] as f64;
                        }
                    }
                }
            }
            for k in 0..c {
                out[k] = acc[k] as f32;
            }
        }
        Interp::Cubic => {
            let x0 = x.floor();
            let y0 = y.floor();
            let wx = cubic_w(x - x0);
            let wy = cubic_w(y - y0);
            let (x0, y0) = (x0 as isize, y0 as isize);
            let mut acc = [0.0f64; 4];
            let xs: [Option<usize>; 4] = [
                fold(x0 - 1, w, border),
                fold(x0, w, border),
                fold(x0 + 1, w, border),
                fold(x0 + 2, w, border),
            ];
            for j in 0..4 {
                let yy = match fold(y0 - 1 + j as isize, h, border) {
                    Some(v) => v,
                    None => continue,
                };
                let row = img.row(yy);
                for i in 0..4 {
                    if let Some(xx) = xs[i] {
                        let wgt = wy[j] * wx[i];
                        let base = xx * c;
                        for k in 0..c {
                            acc[k] += wgt * row[base + k] as f64;
                        }
                    }
                }
            }
            for k in 0..c {
                out[k] = acc[k] as f32;
            }
        }
        Interp::Lanczos4 => {
            let x0 = x.floor();
            let y0 = y.floor();
            let wx = lanczos4_w(x - x0);
            let wy = lanczos4_w(y - y0);
            let (x0, y0) = (x0 as isize, y0 as isize);
            let mut acc = [0.0f64; 4];
            for j in 0..8 {
                let yy = match fold(y0 - 3 + j as isize, h, border) {
                    Some(v) => v,
                    None => continue,
                };
                for i in 0..8 {
                    if let Some(xx) = fold(x0 - 3 + i as isize, w, border) {
                        let p = img.pixel(xx, yy);
                        let wgt = wy[j] * wx[i];
                        for k in 0..c {
                            acc[k] += wgt * p[k] as f64;
                        }
                    }
                }
            }
            for k in 0..c {
                out[k] = acc[k] as f32;
            }
        }
    }
}

/// cv2.warpAffine(img, M, (out_w, out_h)) : dst(x, y) = src(M⁻¹·(x, y)).
pub fn warp_affine(img: &Image, m: &Affine, out_w: usize, out_h: usize, interp: Interp, border: Border) -> Image {
    let inv = invert(m);
    let c = img.c;
    let mut out = Image::new(out_w, out_h, c);
    out.data.par_chunks_mut((out_w * c).max(1)).enumerate().for_each(|(y, row)| {
        let mut px = [0.0f32; 4];
        for x in 0..out_w {
            let (sx, sy) = apply(&inv, x as f64, y as f64);
            sample(img, sx, sy, interp, border, &mut px[..c]);
            row[x * c..x * c + c].copy_from_slice(&px[..c]);
        }
    });
    out
}

/// Recalage bicubique avec bord réfléchi (usage courant du programme).
pub fn warp(img: &Image, m: &Affine) -> Image {
    warp_affine(img, m, img.w, img.h, Interp::Cubic, Border::Reflect)
}

/// cv2.remap : dst(x, y) = src(map_x(x, y), map_y(x, y)).
pub fn remap(img: &Image, map_x: &[f32], map_y: &[f32], out_w: usize, out_h: usize, interp: Interp, border: Border) -> Image {
    let c = img.c;
    let mut out = Image::new(out_w, out_h, c);
    out.data.par_chunks_mut((out_w * c).max(1)).enumerate().for_each(|(y, row)| {
        let mut px = [0.0f32; 4];
        for x in 0..out_w {
            let i = y * out_w + x;
            sample(img, map_x[i] as f64, map_y[i] as f64, interp, border, &mut px[..c]);
            row[x * c..x * c + c].copy_from_slice(&px[..c]);
        }
    });
    out
}

/// Remap avec des cartes calculées à la volée (évite d'allouer les cartes).
pub fn remap_fn(img: &Image, out_w: usize, out_h: usize, interp: Interp, border: Border, f: impl Fn(usize, usize) -> (f64, f64) + Sync) -> Image {
    let c = img.c;
    let mut out = Image::new(out_w, out_h, c);
    out.data.par_chunks_mut((out_w * c).max(1)).enumerate().for_each(|(y, row)| {
        let mut px = [0.0f32; 4];
        for x in 0..out_w {
            let (sx, sy) = f(x, y);
            sample(img, sx, sy, interp, border, &mut px[..c]);
            row[x * c..x * c + c].copy_from_slice(&px[..c]);
        }
    });
    out
}

/// cv2.resize : centres de pixels alignés ((x + 0,5)·échelle − 0,5),
/// bords répliqués. INTER_AREA en réduction = moyenne par surface.
pub fn resize(img: &Image, new_w: usize, new_h: usize, interp: Interp) -> Image {
    if new_w == img.w && new_h == img.h {
        return img.clone();
    }
    let sx = img.w as f64 / new_w as f64;
    let sy = img.h as f64 / new_h as f64;
    if interp == Interp::Area && sx >= 1.0 && sy >= 1.0 {
        return resize_area(img, new_w, new_h);
    }
    let interp = if interp == Interp::Area { Interp::Linear } else { interp };
    remap_fn(img, new_w, new_h, interp, Border::Nearest, |x, y| {
        ((x as f64 + 0.5) * sx - 0.5, (y as f64 + 0.5) * sy - 0.5)
    })
}

fn area_weights(n_in: usize, n_out: usize) -> Vec<Vec<(usize, f64)>> {
    let s = n_in as f64 / n_out as f64;
    (0..n_out)
        .map(|o| {
            let a = o as f64 * s;
            let b = (o as f64 + 1.0) * s;
            let mut v = Vec::new();
            let mut i = a.floor() as usize;
            while (i as f64) < b && i < n_in {
                let lo = a.max(i as f64);
                let hi = b.min(i as f64 + 1.0);
                if hi > lo {
                    v.push((i, (hi - lo) / s));
                }
                i += 1;
            }
            v
        })
        .collect()
}

fn resize_area(img: &Image, new_w: usize, new_h: usize) -> Image {
    let wx = area_weights(img.w, new_w);
    let wy = area_weights(img.h, new_h);
    let c = img.c;
    let mut out = Image::new(new_w, new_h, c);
    out.data.par_chunks_mut((new_w * c).max(1)).enumerate().for_each(|(y, row)| {
        for x in 0..new_w {
            let mut acc = [0.0f64; 4];
            for &(yy, a) in &wy[y] {
                for &(xx, b) in &wx[x] {
                    let p = img.pixel(xx, yy);
                    for k in 0..c {
                        acc[k] += a * b * p[k] as f64;
                    }
                }
            }
            for k in 0..c {
                row[x * c + k] = acc[k] as f32;
            }
        }
    });
    out
}

/// Translation (dx, dy) après rotation de `angle_deg` autour du centre
/// (registration.apply_shift) : bicubique, bord réfléchi ou constant.
pub fn shift_matrix(w: usize, h: usize, dx: f64, dy: f64, angle_deg: f64) -> Affine {
    let mut m = rotation_matrix(w as f64 / 2.0, h as f64 / 2.0, angle_deg, 1.0);
    m[0][2] += dx;
    m[1][2] += dy;
    m
}

pub fn apply_shift(img: &Image, dx: f64, dy: f64, angle_deg: f64, border: Border) -> Image {
    let m = shift_matrix(img.w, img.h, dx, dy, angle_deg);
    warp_affine(img, &m, img.w, img.h, Interp::Cubic, border)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inv() {
        let m = rotation_matrix(10.0, 20.0, 30.0, 1.2);
        let i = invert(&m);
        let c = compose(&m, &i);
        assert!((c[0][0] - 1.0).abs() < 1e-12 && c[0][2].abs() < 1e-9);
    }
    #[test]
    fn shift_int() {
        let img = Image::from_fn(16, 16, |x, y| (x * 16 + y) as f32);
        let s = apply_shift(&img, 2.0, 0.0, 0.0, Border::Reflect);
        assert!((s.get(5, 5, 0) - img.get(3, 5, 0)).abs() < 1e-4);
    }
}
