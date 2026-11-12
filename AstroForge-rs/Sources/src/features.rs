//! Outils de vision : seuil d'Otsu, contours de Canny, transformée de Hough
//! probabiliste, tracé de segments, inpainting par marche rapide.

use crate::image::Image;
use crate::ndimage::{fold, Border};

/// Seuil d'Otsu sur une image 8 bits (valeur 0..255).
pub fn otsu(u8img: &[u8]) -> u8 {
    let mut hist = [0u64; 256];
    for &v in u8img {
        hist[v as usize] += 1;
    }
    let total = u8img.len() as f64;
    let sum_all: f64 = (0..256).map(|i| i as f64 * hist[i] as f64).sum();
    let (mut w0, mut sum0) = (0.0f64, 0.0f64);
    let (mut best, mut best_t) = (-1.0f64, 0u8);
    for t in 0..256 {
        w0 += hist[t] as f64;
        if w0 == 0.0 {
            continue;
        }
        let w1 = total - w0;
        if w1 == 0.0 {
            break;
        }
        sum0 += t as f64 * hist[t] as f64;
        let m0 = sum0 / w0;
        let m1 = (sum_all - sum0) / w1;
        let var = w0 * w1 * (m0 - m1).powi(2);
        if var > best {
            best = var;
            best_t = t as u8;
        }
    }
    best_t
}

pub fn to_u8(img: &Image) -> Vec<u8> {
    img.data.iter().map(|&v| crate::image::f_to_u8(v)).collect()
}

/// Filtre médian 8 bits (cv2.medianBlur, bord répliqué).
pub fn median_blur_u8(src: &[u8], w: usize, h: usize, k: usize) -> Vec<u8> {
    let r = (k / 2) as isize;
    let mut out = vec![0u8; w * h];
    use rayon::prelude::*;
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let mut hist = [0u32; 256];
        let n = (k * k) as u32;
        let half = n / 2;
        // histogramme glissant le long de la ligne
        for x in 0..w {
            if x == 0 {
                hist = [0u32; 256];
                for dy in -r..=r {
                    let yy = fold(y as isize + dy, h, Border::Nearest).unwrap();
                    for dx in -r..=r {
                        let xx = fold(dx, w, Border::Nearest).unwrap();
                        hist[src[yy * w + xx] as usize] += 1;
                    }
                }
            } else {
                for dy in -r..=r {
                    let yy = fold(y as isize + dy, h, Border::Nearest).unwrap();
                    let xo = fold(x as isize - r - 1, w, Border::Nearest).unwrap();
                    let xi = fold(x as isize + r, w, Border::Nearest).unwrap();
                    hist[src[yy * w + xo] as usize] -= 1;
                    hist[src[yy * w + xi] as usize] += 1;
                }
            }
            let mut acc = 0u32;
            let mut m = 0u8;
            for (v, &c) in hist.iter().enumerate() {
                acc += c;
                if acc > half {
                    m = v as u8;
                    break;
                }
            }
            row[x] = m;
        }
    });
    out
}

/// Contours de Canny (Sobel 3×3, norme L1, suppression des non-maxima,
/// hystérésis) sur une image 8 bits -> masque 0/255.
pub fn canny(src: &[u8], w: usize, h: usize, low: f64, high: f64) -> Vec<u8> {
    let g = |x: isize, y: isize| -> f64 {
        let xx = fold(x, w, Border::Mirror).unwrap();
        let yy = fold(y, h, Border::Mirror).unwrap();
        src[yy * w + xx] as f64
    };
    let mut gx = vec![0.0f64; w * h];
    let mut gy = vec![0.0f64; w * h];
    let mut mag = vec![0.0f64; w * h];
    for y in 0..h as isize {
        for x in 0..w as isize {
            let dx = (g(x + 1, y - 1) + 2.0 * g(x + 1, y) + g(x + 1, y + 1)) - (g(x - 1, y - 1) + 2.0 * g(x - 1, y) + g(x - 1, y + 1));
            let dy = (g(x - 1, y + 1) + 2.0 * g(x, y + 1) + g(x + 1, y + 1)) - (g(x - 1, y - 1) + 2.0 * g(x, y - 1) + g(x + 1, y - 1));
            let i = y as usize * w + x as usize;
            gx[i] = dx;
            gy[i] = dy;
            mag[i] = dx.abs() + dy.abs();
        }
    }
    // 0 = non contour, 1 = faible, 2 = fort
    let mut state = vec![0u8; w * h];
    let at = |x: isize, y: isize, m: &Vec<f64>| -> f64 {
        if x < 0 || y < 0 || x >= w as isize || y >= h as isize {
            0.0
        } else {
            m[y as usize * w + x as usize]
        }
    };
    for y in 0..h as isize {
        for x in 0..w as isize {
            let i = y as usize * w + x as usize;
            let m = mag[i];
            if m <= low {
                continue;
            }
            let (ax, ay) = (gx[i].abs(), gy[i].abs());
            let tg22 = 0.414_213_562;
            let (n1, n2) = if ay <= ax * tg22 {
                (at(x - 1, y, &mag), at(x + 1, y, &mag))
            } else if ay >= ax / tg22 {
                (at(x, y - 1, &mag), at(x, y + 1, &mag))
            } else if (gx[i] > 0.0) == (gy[i] > 0.0) {
                (at(x - 1, y - 1, &mag), at(x + 1, y + 1, &mag))
            } else {
                (at(x + 1, y - 1, &mag), at(x - 1, y + 1, &mag))
            };
            if m > n1 && m >= n2 {
                state[i] = if m > high { 2 } else { 1 };
            }
        }
    }
    let mut out = vec![0u8; w * h];
    let mut stack: Vec<usize> = (0..w * h).filter(|&i| state[i] == 2).collect();
    for &i in &stack {
        out[i] = 255;
    }
    while let Some(i) = stack.pop() {
        let (x, y) = ((i % w) as isize, (i / w) as isize);
        for dy in -1..=1 {
            for dx in -1..=1 {
                let (xx, yy) = (x + dx, y + dy);
                if xx < 0 || yy < 0 || xx >= w as isize || yy >= h as isize {
                    continue;
                }
                let j = yy as usize * w + xx as usize;
                if state[j] == 1 && out[j] == 0 {
                    out[j] = 255;
                    stack.push(j);
                }
            }
        }
    }
    out
}

/// Transformée de Hough probabiliste (esprit cv2.HoughLinesP) : segments
/// (x1, y1, x2, y2) d'au moins `min_len` pixels, trous ≤ `max_gap`.
pub fn hough_lines_p(mask: &[u8], w: usize, h: usize, threshold: usize, min_len: usize, max_gap: usize) -> Vec<(i32, i32, i32, i32)> {
    let n_theta = 180usize;
    let diag = ((w * w + h * h) as f64).sqrt().ceil() as usize;
    let n_rho = 2 * diag + 1;
    let thetas: Vec<(f64, f64)> = (0..n_theta).map(|t| {
        let a = t as f64 * std::f64::consts::PI / n_theta as f64;
        (a.cos(), a.sin())
    }).collect();
    let mut acc = vec![0u32; n_theta * n_rho];
    let mut pts: Vec<(usize, usize)> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] > 0 {
                pts.push((x, y));
            }
        }
    }
    // ordre pseudo-aléatoire déterministe des points
    let mut order: Vec<usize> = (0..pts.len()).collect();
    let mut seed: u64 = 0x2545F4914F6CDD1D;
    for i in (1..order.len()).rev() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let j = (seed % (i as u64 + 1)) as usize;
        order.swap(i, j);
    }
    let mut live = mask.iter().map(|&v| v > 0).collect::<Vec<bool>>();
    let mut voted = vec![false; w * h];
    let mut lines = Vec::new();
    for &pi in &order {
        let (x, y) = pts[pi];
        if !live[y * w + x] {
            continue;
        }
        let mut best = (0u32, 0usize);
        for (t, &(c, s)) in thetas.iter().enumerate() {
            let r = (x as f64 * c + y as f64 * s).round() as isize + diag as isize;
            let k = t * n_rho + r as usize;
            acc[k] += 1;
            if acc[k] > best.0 {
                best = (acc[k], t);
            }
        }
        voted[y * w + x] = true;
        if (best.0 as usize) < threshold {
            continue;
        }
        // suivi du segment le long de la direction de la droite
        let (c, s) = thetas[best.1];
        let (dx, dy) = (-s, c);
        let mut ends = [(x as i32, y as i32); 2];
        for (side, sign) in [1.0f64, -1.0].iter().enumerate() {
            let mut gap = 0usize;
            let mut k = 1usize;
            loop {
                let px = (x as f64 + sign * dx * k as f64).round() as isize;
                let py = (y as f64 + sign * dy * k as f64).round() as isize;
                if px < 0 || py < 0 || px >= w as isize || py >= h as isize {
                    break;
                }
                let j = py as usize * w + px as usize;
                if mask[j] > 0 {
                    gap = 0;
                    ends[side] = (px as i32, py as i32);
                } else {
                    gap += 1;
                    if gap > max_gap {
                        break;
                    }
                }
                k += 1;
            }
        }
        let len = (((ends[0].0 - ends[1].0).pow(2) + (ends[0].1 - ends[1].1).pow(2)) as f64).sqrt();
        // retire les points du segment (et leurs votes)
        let steps = len.ceil() as usize + 1;
        for k in 0..=steps {
            let t = k as f64 / steps.max(1) as f64;
            let px = (ends[1].0 as f64 + (ends[0].0 - ends[1].0) as f64 * t).round() as isize;
            let py = (ends[1].1 as f64 + (ends[0].1 - ends[1].1) as f64 * t).round() as isize;
            for oy in -1..=1isize {
                for ox in -1..=1isize {
                    let (qx, qy) = (px + ox, py + oy);
                    if qx < 0 || qy < 0 || qx >= w as isize || qy >= h as isize {
                        continue;
                    }
                    let j = qy as usize * w + qx as usize;
                    if live[j] && mask[j] > 0 {
                        live[j] = false;
                        if voted[j] {
                            for (tt, &(cc, ss)) in thetas.iter().enumerate() {
                                let r = (qx as f64 * cc + qy as f64 * ss).round() as isize + diag as isize;
                                let kk = tt * n_rho + r as usize;
                                acc[kk] = acc[kk].saturating_sub(1);
                            }
                        }
                    }
                }
            }
        }
        if len >= min_len as f64 {
            lines.push((ends[1].0, ends[1].1, ends[0].0, ends[0].1));
        }
    }
    lines
}

/// Trace un segment épais (valeur 255) dans un masque 8 bits.
pub fn draw_line_u8(mask: &mut [u8], w: usize, h: usize, p1: (i32, i32), p2: (i32, i32), thickness: i32) {
    let r = (thickness as f64 / 2.0).max(0.5);
    let (x0, y0, x1, y1) = (p1.0 as f64, p1.1 as f64, p2.0 as f64, p2.1 as f64);
    let (minx, maxx) = ((x0.min(x1) - r).floor() as isize, (x0.max(x1) + r).ceil() as isize);
    let (miny, maxy) = ((y0.min(y1) - r).floor() as isize, (y0.max(y1) + r).ceil() as isize);
    let (vx, vy) = (x1 - x0, y1 - y0);
    let l2 = (vx * vx + vy * vy).max(1e-12);
    for y in miny.max(0)..=maxy.min(h as isize - 1) {
        for x in minx.max(0)..=maxx.min(w as isize - 1) {
            let t = (((x as f64 - x0) * vx + (y as f64 - y0) * vy) / l2).clamp(0.0, 1.0);
            let (px, py) = (x0 + t * vx, y0 + t * vy);
            if ((x as f64 - px).powi(2) + (y as f64 - py).powi(2)).sqrt() <= r {
                mask[y as usize * w + x as usize] = 255;
            }
        }
    }
}

/// Inpainting par marche rapide (principe de Telea) : les pixels masqués
/// sont remplis du bord vers l'intérieur par moyenne pondérée (distance,
/// direction) des pixels connus dans un rayon `radius`.
pub fn inpaint(plane: &Image, mask: &[bool], radius: usize) -> Image {
    let (w, h) = (plane.w, plane.h);
    let mut out = plane.clone();
    let mut known: Vec<bool> = mask.iter().map(|&m| !m).collect();
    // distance au bord connu (ordre de remplissage)
    let dist = crate::ndimage::distance_transform_edt(mask, w, h);
    let mut order: Vec<usize> = (0..w * h).filter(|&i| mask[i]).collect();
    order.sort_by(|&a, &b| dist[a].total_cmp(&dist[b]));
    let r = radius as isize;
    for &i in &order {
        let (x, y) = ((i % w) as isize, (i / w) as isize);
        let (mut s, mut t) = (0.0f64, 0.0f64);
        for dy in -r..=r {
            for dx in -r..=r {
                let (xx, yy) = (x + dx, y + dy);
                if xx < 0 || yy < 0 || xx >= w as isize || yy >= h as isize {
                    continue;
                }
                let j = yy as usize * w + xx as usize;
                if !known[j] {
                    continue;
                }
                let d2 = (dx * dx + dy * dy) as f64;
                if d2 > (r * r) as f64 || d2 == 0.0 {
                    continue;
                }
                // pondération : proximité géométrique et niveau (distance au bord)
                let wdst = 1.0 / d2;
                let wlev = 1.0 / (1.0 + (dist[j] - dist[i]).abs() as f64);
                let wt = wdst * wlev;
                s += wt * out.data[j] as f64;
                t += wt;
            }
        }
        if t > 0.0 {
            out.data[i] = (s / t) as f32;
        }
        known[i] = true;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn otsu_bimodal() {
        let mut v = vec![10u8; 500];
        v.extend(vec![200u8; 500]);
        let t = otsu(&v);
        assert!((10..200).contains(&t));
    }
    #[test]
    fn hough_finds_line() {
        let (w, h) = (100, 80);
        let mut m = vec![0u8; w * h];
        draw_line_u8(&mut m, w, h, (5, 60), (95, 20), 1);
        let l = hough_lines_p(&m, w, h, 8, 25, 6);
        assert!(!l.is_empty());
    }
}
