//! FFT 2D (rustfft) : transformées directe/inverse d'un tableau complexe
//! ligne par ligne puis colonne par colonne.

use rayon::prelude::*;
use rustfft::num_complex::Complex64;
use rustfft::FftPlanner;

pub type C64 = Complex64;

/// FFT 2D en place ; `inverse` normalise par 1/(w·h) comme NumPy.
pub fn fft2(data: &mut [C64], w: usize, h: usize, inverse: bool) {
    assert_eq!(data.len(), w * h);
    let mut planner = FftPlanner::<f64>::new();
    let frow = if inverse { planner.plan_fft_inverse(w) } else { planner.plan_fft_forward(w) };
    let fcol = if inverse { planner.plan_fft_inverse(h) } else { planner.plan_fft_forward(h) };
    data.par_chunks_mut(w).for_each(|row| frow.process(row));
    // transposition -> colonnes contiguës
    let mut t = vec![C64::new(0.0, 0.0); w * h];
    for y in 0..h {
        for x in 0..w {
            t[x * h + y] = data[y * w + x];
        }
    }
    t.par_chunks_mut(h).for_each(|col| fcol.process(col));
    let norm = if inverse { 1.0 / (w * h) as f64 } else { 1.0 };
    for x in 0..w {
        for y in 0..h {
            data[y * w + x] = t[x * h + y] * norm;
        }
    }
}

pub fn to_complex(v: &[f64]) -> Vec<C64> {
    v.iter().map(|&x| C64::new(x, 0.0)).collect()
}

/// np.hanning(n).
pub fn hanning(n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![1.0];
    }
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos())
        .collect()
}
