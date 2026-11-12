//! Filtres d'image (équivalents de scipy.ndimage) sur des plans mono.
//! Mode de bord par défaut « reflect » de SciPy : (d c b a | a b c d | d c b a),
//! identique au BORDER_REFLECT d'OpenCV.

use crate::image::Image;
use crate::stats;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Border {
    /// d c b a | a b c d (scipy « reflect », OpenCV BORDER_REFLECT)
    Reflect,
    /// d c b | a b c d (scipy « mirror », OpenCV BORDER_REFLECT_101)
    Mirror,
    /// a a a | a b c d (scipy « nearest », OpenCV BORDER_REPLICATE)
    Nearest,
    /// valeur constante (0)
    Constant,
}

/// Indice replié selon le mode ; None = hors image (mode constant).
#[inline]
pub fn fold(i: isize, n: usize, mode: Border) -> Option<usize> {
    let n_i = n as isize;
    if i >= 0 && i < n_i {
        return Some(i as usize);
    }
    if n == 1 {
        return match mode {
            Border::Constant => None,
            _ => Some(0),
        };
    }
    match mode {
        Border::Constant => None,
        Border::Nearest => Some(i.clamp(0, n_i - 1) as usize),
        Border::Reflect => {
            let p = 2 * n_i;
            let mut k = i.rem_euclid(p);
            if k >= n_i {
                k = p - 1 - k;
            }
            Some(k as usize)
        }
        Border::Mirror => {
            let p = 2 * n_i - 2;
            let mut k = i.rem_euclid(p);
            if k >= n_i {
                k = p - k;
            }
            Some(k as usize)
        }
    }
}

#[inline]
pub fn reflect(i: isize, n: usize) -> usize {
    fold(i, n, Border::Reflect).unwrap()
}

/// Corrélation 1D selon un axe (0 = lignes/vertical, 1 = colonnes/horizontal)
/// avec un noyau centré défini par (décalages, poids).
pub fn correlate_taps(p: &Image, taps: &[(isize, f64)], axis: usize, mode: Border) -> Image {
    assert_eq!(p.c, 1);
    let (w, h) = (p.w, p.h);
    let mut out = vec![0.0f32; w * h];
    if w == 0 || h == 0 {
        return Image::from_vec(w, h, 1, out);
    }
    if axis == 1 {
        out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let src = &p.data[y * w..(y + 1) * w];
            for x in 0..w {
                let mut acc = 0.0f64;
                for &(o, k) in taps {
                    if let Some(xx) = fold(x as isize + o, w, mode) {
                        acc += k * src[xx] as f64;
                    }
                }
                row[x] = acc as f32;
            }
        });
    } else {
        // précalcul des indices de lignes repliés
        out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let rows: Vec<(Option<usize>, f64)> =
                taps.iter().map(|&(o, k)| (fold(y as isize + o, h, mode), k)).collect();
            let mut acc = vec![0.0f64; w];
            for (yy, k) in rows {
                if let Some(yy) = yy {
                    let src = &p.data[yy * w..(yy + 1) * w];
                    for x in 0..w {
                        acc[x] += k * src[x] as f64;
                    }
                }
            }
            for x in 0..w {
                row[x] = acc[x] as f32;
            }
        });
    }
    Image::from_vec(w, h, 1, out)
}

/// Noyau gaussien 1D normalisé (scipy : rayon = int(4σ + 0,5)).
pub fn gaussian_kernel(sigma: f64, truncate: f64) -> Vec<(isize, f64)> {
    let r = (truncate * sigma + 0.5) as isize;
    let r = r.max(0);
    let s2 = sigma * sigma;
    let mut taps: Vec<(isize, f64)> = (-r..=r).map(|i| (i, (-0.5 * (i * i) as f64 / s2).exp())).collect();
    let sum: f64 = taps.iter().map(|t| t.1).sum();
    for t in taps.iter_mut() {
        t.1 /= sum;
    }
    taps
}

/// Flou gaussien d'un plan mono (scipy.ndimage.gaussian_filter, mode reflect).
pub fn gaussian(p: &Image, sigma: f64) -> Image {
    gaussian_mode(p, sigma, Border::Reflect)
}

pub fn gaussian_mode(p: &Image, sigma: f64, mode: Border) -> Image {
    if sigma <= 0.0 {
        return p.clone();
    }
    let k = gaussian_kernel(sigma, 4.0);
    let t = correlate_taps(p, &k, 0, mode);
    correlate_taps(&t, &k, 1, mode)
}

/// Flou gaussien canal par canal (axes spatiaux uniquement).
pub fn gaussian_img(img: &Image, sigma: f64) -> Image {
    img.map_planes(|p| gaussian(p, sigma))
}

/// Laplacien discret (scipy.ndimage.laplace, mode reflect).
pub fn laplace(p: &Image) -> Image {
    let taps = [(-1isize, 1.0), (0, -2.0), (1, 1.0)];
    let a = correlate_taps(p, &taps, 0, Border::Reflect);
    let b = correlate_taps(p, &taps, 1, Border::Reflect);
    a.zip_map(&b, |x, y| x + y)
}

/// Filtre médian carré `size`×`size` (scipy.ndimage.median_filter).
pub fn median_filter(p: &Image, size: usize) -> Image {
    median_filter_mode(p, size, Border::Reflect)
}

pub fn median_filter_mode(p: &Image, size: usize, mode: Border) -> Image {
    assert_eq!(p.c, 1);
    let size = size.max(1);
    if size == 1 {
        return p.clone();
    }
    let (w, h) = (p.w, p.h);
    let r0 = (size / 2) as isize; // origine scipy : centre (taille paire : décalé)
    let lo = -r0;
    let hi = size as isize - 1 - r0;
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w.max(1)).enumerate().for_each(|(y, row)| {
        let mut buf: Vec<f32> = Vec::with_capacity(size * size);
        let ys: Vec<Option<usize>> = (lo..=hi).map(|o| fold(y as isize + o, h, mode)).collect();
        for x in 0..w {
            buf.clear();
            for yy in &ys {
                for o in lo..=hi {
                    let xx = fold(x as isize + o, w, mode);
                    match (yy, xx) {
                        (Some(yy), Some(xx)) => buf.push(p.data[yy * w + xx]),
                        _ => buf.push(0.0),
                    }
                }
            }
            // scipy : rang = n/2 (élément central du tri, pas de moyenne)
            let k = buf.len() / 2;
            let (_, m, _) = buf.select_nth_unstable_by(k, |a, b| a.total_cmp(b));
            row[x] = *m;
        }
    });
    Image::from_vec(w, h, 1, out)
}

pub fn median_filter_img(img: &Image, size: usize) -> Image {
    img.map_planes(|p| median_filter(p, size))
}

fn minmax_1d(p: &Image, size: usize, axis: usize, is_max: bool, mode: Border) -> Image {
    let r0 = (size / 2) as isize;
    let taps: Vec<(isize, f64)> = (-r0..=(size as isize - 1 - r0)).map(|o| (o, 0.0)).collect();
    let (w, h) = (p.w, p.h);
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w.max(1)).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut best = if is_max { f32::NEG_INFINITY } else { f32::INFINITY };
            for &(o, _) in &taps {
                let v = if axis == 1 {
                    fold(x as isize + o, w, mode).map(|xx| p.data[y * w + xx])
                } else {
                    fold(y as isize + o, h, mode).map(|yy| p.data[yy * w + x])
                };
                let v = v.unwrap_or(0.0);
                if is_max {
                    if v > best {
                        best = v;
                    }
                } else if v < best {
                    best = v;
                }
            }
            row[x] = best;
        }
    });
    Image::from_vec(w, h, 1, out)
}

/// Filtre maximum carré (scipy.ndimage.maximum_filter / grey_dilation).
pub fn maximum_filter(p: &Image, size: usize) -> Image {
    let t = minmax_1d(p, size, 0, true, Border::Reflect);
    minmax_1d(&t, size, 1, true, Border::Reflect)
}

/// Filtre minimum carré (scipy.ndimage.minimum_filter / grey_erosion).
pub fn minimum_filter(p: &Image, size: usize) -> Image {
    let t = minmax_1d(p, size, 0, false, Border::Reflect);
    minmax_1d(&t, size, 1, false, Border::Reflect)
}

/// Ouverture morphologique en niveaux de gris (érosion puis dilatation).
pub fn grey_opening(p: &Image, size: usize) -> Image {
    maximum_filter(&minimum_filter(p, size), size)
}

// ---------------------------------------------------------------- binaire
/// Composantes connexes (connexité 4, comme la structure par défaut de
/// SciPy). Retourne (étiquettes 1..=n, 0 = fond ; n).
pub fn label(mask: &[bool], w: usize, h: usize) -> (Vec<u32>, usize) {
    let mut lab = vec![0u32; w * h];
    let mut n = 0u32;
    let mut stack: Vec<usize> = Vec::new();
    for start in 0..w * h {
        if !mask[start] || lab[start] != 0 {
            continue;
        }
        n += 1;
        lab[start] = n;
        stack.push(start);
        while let Some(i) = stack.pop() {
            let (x, y) = (i % w, i / w);
            let mut push = |j: usize| {
                if mask[j] && lab[j] == 0 {
                    lab[j] = n;
                    stack.push(j);
                }
            };
            if x > 0 {
                push(i - 1);
            }
            if x + 1 < w {
                push(i + 1);
            }
            if y > 0 {
                push(i - w);
            }
            if y + 1 < h {
                push(i + w);
            }
        }
    }
    (lab, n as usize)
}

/// Boîtes englobantes des étiquettes 1..=n : (y0, y1, x0, x1) exclusifs.
pub fn find_objects(lab: &[u32], w: usize, n: usize) -> Vec<Option<(usize, usize, usize, usize)>> {
    let mut out: Vec<Option<(usize, usize, usize, usize)>> = vec![None; n];
    for (i, &l) in lab.iter().enumerate() {
        if l == 0 {
            continue;
        }
        let (x, y) = (i % w, i / w);
        let e = &mut out[(l - 1) as usize];
        *e = Some(match *e {
            None => (y, y + 1, x, x + 1),
            Some((y0, y1, x0, x1)) => (y0.min(y), y1.max(y + 1), x0.min(x), x1.max(x + 1)),
        });
    }
    out
}

/// Dilatation binaire (structure en croix), `iterations` passes.
pub fn binary_dilation(mask: &[bool], w: usize, h: usize, iterations: usize) -> Vec<bool> {
    let mut cur = mask.to_vec();
    for _ in 0..iterations {
        let prev = cur.clone();
        cur.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let i = y * w + x;
                row[x] = prev[i]
                    || (x > 0 && prev[i - 1])
                    || (x + 1 < w && prev[i + 1])
                    || (y > 0 && prev[i - w])
                    || (y + 1 < h && prev[i + w]);
            }
        });
    }
    cur
}

/// Érosion binaire (structure en croix, hors image = faux).
pub fn binary_erosion(mask: &[bool], w: usize, h: usize, iterations: usize) -> Vec<bool> {
    let mut cur = mask.to_vec();
    for _ in 0..iterations {
        let prev = cur.clone();
        cur.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let i = y * w + x;
                row[x] = prev[i]
                    && x > 0
                    && prev[i - 1]
                    && x + 1 < w
                    && prev[i + 1]
                    && y > 0
                    && prev[i - w]
                    && y + 1 < h
                    && prev[i + w];
            }
        });
    }
    cur
}

/// Remplissage des trous (fond non relié au bord), connexité 4.
pub fn binary_fill_holes(mask: &[bool], w: usize, h: usize) -> Vec<bool> {
    let mut outside = vec![false; w * h];
    let mut stack: Vec<usize> = Vec::new();
    let seed = |i: usize, outside: &mut Vec<bool>, stack: &mut Vec<usize>| {
        if !mask[i] && !outside[i] {
            outside[i] = true;
            stack.push(i);
        }
    };
    for x in 0..w {
        seed(x, &mut outside, &mut stack);
        seed((h - 1) * w + x, &mut outside, &mut stack);
    }
    for y in 0..h {
        seed(y * w, &mut outside, &mut stack);
        seed(y * w + w - 1, &mut outside, &mut stack);
    }
    while let Some(i) = stack.pop() {
        let (x, y) = (i % w, i / w);
        let mut nb = Vec::with_capacity(4);
        if x > 0 {
            nb.push(i - 1);
        }
        if x + 1 < w {
            nb.push(i + 1);
        }
        if y > 0 {
            nb.push(i - w);
        }
        if y + 1 < h {
            nb.push(i + w);
        }
        for j in nb {
            if !mask[j] && !outside[j] {
                outside[j] = true;
                stack.push(j);
            }
        }
    }
    outside.iter().map(|&o| !o).collect()
}

/// Transformée de distance euclidienne exacte (scipy distance_transform_edt) :
/// pour chaque pixel vrai, distance au pixel faux le plus proche (0 sinon).
pub fn distance_transform_edt(mask: &[bool], w: usize, h: usize) -> Vec<f32> {
    const INF: f64 = 1e20;
    // passe 1 : colonnes
    let mut f = vec![0.0f64; w * h];
    for i in 0..w * h {
        f[i] = if mask[i] { INF } else { 0.0 };
    }
    let mut g = vec![0.0f64; w * h];
    // transformée 1D (Felzenszwalb & Huttenlocher) le long des colonnes
    let cols: Vec<Vec<f64>> = (0..w)
        .into_par_iter()
        .map(|x| {
            let col: Vec<f64> = (0..h).map(|y| f[y * w + x]).collect();
            dt1d(&col)
        })
        .collect();
    for x in 0..w {
        for y in 0..h {
            g[y * w + x] = cols[x][y];
        }
    }
    let rows: Vec<Vec<f64>> = (0..h).into_par_iter().map(|y| dt1d(&g[y * w..(y + 1) * w])).collect();
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            out[y * w + x] = rows[y][x].sqrt() as f32;
        }
    }
    out
}

fn dt1d(f: &[f64]) -> Vec<f64> {
    let n = f.len();
    let mut d = vec![0.0f64; n];
    if n == 0 {
        return d;
    }
    let mut v = vec![0usize; n];
    let mut z = vec![0.0f64; n + 1];
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f64::NEG_INFINITY;
    z[1] = f64::INFINITY;
    let inter = |q: usize, p: usize| -> f64 {
        ((f[q] + (q * q) as f64) - (f[p] + (p * p) as f64)) / (2.0 * q as f64 - 2.0 * p as f64)
    };
    for q in 1..n {
        let mut s = inter(q, v[k]);
        while s <= z[k] {
            k -= 1;
            s = inter(q, v[k]);
        }
        k += 1;
        v[k] = q;
        z[k] = s;
        z[k + 1] = f64::INFINITY;
    }
    k = 0;
    for q in 0..n {
        while z[k + 1] < q as f64 {
            k += 1;
        }
        let p = v[k];
        d[q] = (q as f64 - p as f64).powi(2) + f[p];
    }
    d
}

/// Pour une petite grille : indice du pixel « faux » le plus proche de
/// chaque pixel « vrai » (équivalent de return_indices=True).
pub fn nearest_false_indices(mask: &[bool], w: usize, h: usize) -> Vec<usize> {
    let falses: Vec<usize> = (0..w * h).filter(|&i| !mask[i]).collect();
    (0..w * h)
        .map(|i| {
            if !mask[i] || falses.is_empty() {
                return i;
            }
            let (x, y) = ((i % w) as isize, (i / w) as isize);
            let mut best = (isize::MAX, i);
            for &j in &falses {
                let (xx, yy) = ((j % w) as isize, (j / w) as isize);
                let d = (xx - x).pow(2) + (yy - y).pow(2);
                if d < best.0 {
                    best = (d, j);
                }
            }
            best.1
        })
        .collect()
}

/// Statistique robuste d'un plan : (médiane, 1,4826·MAD).
pub fn robust_plane(p: &Image) -> (f64, f64) {
    stats::robust(&p.data)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fold_modes() {
        assert_eq!(fold(-1, 4, Border::Reflect), Some(0));
        assert_eq!(fold(-2, 4, Border::Reflect), Some(1));
        assert_eq!(fold(4, 4, Border::Reflect), Some(3));
        assert_eq!(fold(-1, 4, Border::Mirror), Some(1));
        assert_eq!(fold(4, 4, Border::Mirror), Some(2));
    }
    #[test]
    fn edt() {
        let w = 5;
        let h = 1;
        let m = vec![false, true, true, true, false];
        let d = distance_transform_edt(&m, w, h);
        assert_eq!(d, vec![0.0, 1.0, 2.0, 1.0, 0.0]);
    }
}
