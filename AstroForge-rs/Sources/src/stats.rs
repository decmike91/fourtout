//! Statistiques robustes (équivalents de np.median, np.percentile…).

/// Médiane façon NumPy (moyenne des deux valeurs centrales si n pair).
/// Les NaN sont ignorés ; tableau vide -> NaN. Réordonne `v`.
pub fn median_mut(v: &mut [f32]) -> f32 {
    let n = v.len();
    if n == 0 {
        return f32::NAN;
    }
    let mid = n / 2;
    let (_, m, _) = v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let hi = *m;
    if n % 2 == 1 {
        return hi;
    }
    let lo = v[..mid].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    ((lo as f64 + hi as f64) * 0.5) as f32
}

pub fn median(v: &[f32]) -> f32 {
    let mut c: Vec<f32> = v.iter().cloned().filter(|x| !x.is_nan()).collect();
    median_mut(&mut c)
}

pub fn median_f64_mut(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let mid = n / 2;
    let (_, m, _) = v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    let hi = *m;
    if n % 2 == 1 {
        return hi;
    }
    let lo = v[..mid].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (lo + hi) * 0.5
}

pub fn median_f64(v: &[f64]) -> f64 {
    let mut c: Vec<f64> = v.iter().cloned().filter(|x| !x.is_nan()).collect();
    median_f64_mut(&mut c)
}

/// Écart absolu médian (non mis à l'échelle) autour de `med`.
pub fn mad_about(v: &[f32], med: f32) -> f32 {
    let mut d: Vec<f32> = v.iter().map(|&x| (x - med).abs()).collect();
    median_mut(&mut d)
}

/// (médiane, 1,4826·MAD).
pub fn robust(v: &[f32]) -> (f64, f64) {
    let med = median(v);
    let mad = mad_about(v, med);
    (med as f64, 1.4826 * mad as f64)
}

/// Percentile façon np.percentile (interpolation linéaire), q en %.
pub fn percentile(v: &[f32], q: f64) -> f32 {
    let mut c: Vec<f32> = v.iter().cloned().filter(|x| !x.is_nan()).collect();
    percentile_mut(&mut c, q)
}

pub fn percentile_mut(c: &mut [f32], q: f64) -> f32 {
    let n = c.len();
    if n == 0 {
        return f32::NAN;
    }
    let q = q.clamp(0.0, 100.0) / 100.0;
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    let (_, vlo, right) = c.select_nth_unstable_by(lo, |a, b| a.total_cmp(b));
    let vlo = *vlo as f64;
    if frac == 0.0 || hi == lo {
        return vlo as f32;
    }
    let vhi = right.iter().cloned().fold(f32::INFINITY, f32::min) as f64;
    (vlo + (vhi - vlo) * frac) as f32
}

/// Quantile (q dans [0,1]) façon np.quantile.
pub fn quantile(v: &[f32], q: f64) -> f32 {
    percentile(v, q * 100.0)
}

pub fn mean(v: &[f32]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64
}

pub fn mean_f64(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

/// Écart-type de population (ddof = 0).
pub fn std(v: &[f32]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let m = mean(v);
    (v.iter().map(|&x| (x as f64 - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

pub fn std_f64(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let m = mean_f64(v);
    (v.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
}

/// Indices triant `v` par ordre décroissant (stable).
pub fn argsort_desc(v: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
    idx
}

pub fn argsort_asc(v: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    idx
}

/// Interpolation linéaire par morceaux (np.interp) : xs croissants.
pub fn interp(x: f64, xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len();
    if n == 0 {
        return f64::NAN;
    }
    if x <= xs[0] {
        return ys[0];
    }
    if x >= xs[n - 1] {
        return ys[n - 1];
    }
    // recherche dichotomique
    let mut lo = 0usize;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if xs[mid] <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let dx = xs[hi] - xs[lo];
    if dx <= 0.0 {
        return ys[hi];
    }
    ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / dx
}

/// np.linspace(a, b, n).
pub fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![a];
    }
    let step = (b - a) / (n - 1) as f64;
    (0..n).map(|i| if i == n - 1 { b } else { a + step * i as f64 }).collect()
}

/// np.allclose(a, b, rtol, atol).
pub fn allclose(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

/// Formatage façon Python « {:g} ».
pub fn fmt_g(v: f64) -> String {
    fmt_g_prec(v, 6)
}

pub fn fmt_g_prec(v: f64, prec: usize) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if v == 0.0 {
        return if v.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    let p = prec.max(1);
    let exp = v.abs().log10().floor() as i32;
    // arrondi à p chiffres significatifs pour déterminer l'exposant réel
    let s = format!("{:.*e}", p - 1, v);
    let (mant, e) = s.split_once('e').unwrap();
    let e: i32 = e.parse().unwrap_or(exp);
    if e < -4 || e >= p as i32 {
        let mant = strip_zeros(mant);
        let sign = if e < 0 { "-" } else { "+" };
        format!("{}e{}{:02}", mant, sign, e.abs())
    } else {
        let decimals = (p as i32 - 1 - e).max(0) as usize;
        strip_zeros(&format!("{:.*}", decimals, v))
    }
}

fn strip_zeros(s: &str) -> String {
    if s.contains('.') {
        let t = s.trim_end_matches('0');
        t.trim_end_matches('.').to_string()
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn med() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        assert!((percentile(&[1.0, 2.0, 3.0, 4.0], 25.0) - 1.75).abs() < 1e-6);
    }
    #[test]
    fn g() {
        assert_eq!(fmt_g(300.0), "300");
        assert_eq!(fmt_g(0.5), "0.5");
        assert_eq!(fmt_g(120.25), "120.25");
        assert_eq!(fmt_g(1e7), "1e+07");
        assert_eq!(fmt_g(0.0001), "0.0001");
        assert_eq!(fmt_g(0.00001), "1e-05");
    }
}
