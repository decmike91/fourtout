//! Vidéo SER synthétique de Jupiter pour tester le planétaire automatique.
//!
//!     cargo run --release --example synth_planet -- jupiter_test.ser [frames] [vérité.tif]
//!
//! Variables facultatives : SYN_R (rayon, 78), SYN_GAIN (électrons pour
//! 1.0 : 300 ; plus petit = plus bruité), SYN_TURB (distorsion, 0.6),
//! SYN_SEEING (flou médian σ, 0.9), SYN_MONO=1, SYN_SURFACE=1 (texture
//! lunaire plein champ), SYN_SEED, SYN_DX / SYN_DY (décalage du champ).
//!
//! Seeing variable (flou et distorsion locale par frame), agitation,
//! bruit photonique + lecture, dispersion atmosphérique, quelques frames
//! voilées par un nuage et quelques frames où la planète sort du champ.

use astroforge::geom::{self, Interp};
use astroforge::image::Image;
use astroforge::ndimage::{self, Border};
use std::path::PathBuf;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn u(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn n(&mut self) -> f64 {
        let (a, b) = (self.u(), self.u());
        (-2.0 * a.ln()).sqrt() * (6.283185307179586 * b).cos()
    }
}

/// Jupiter « vraie » (RVB), centrée en (cx, cy), rayon équatorial r.
pub fn jupiter(w: usize, h: usize, cx: f64, cy: f64, r: f64) -> Image {
    let mut rng = Rng(42);
    let spots: Vec<(f64, f64, f64, f64)> = (0..40).map(|_| (rng.u() * 2.0 - 1.0, rng.u() * 1.6 - 0.8, 0.02 + 0.05 * rng.u(), if rng.u() < 0.5 { 0.25 } else { -0.2 })).collect();
    let flat = 0.935;
    let mut img = Image::new(w, h, 3);
    for y in 0..h {
        for x in 0..w {
            let u = (x as f64 - cx) / r;
            let v = (y as f64 - cy) / (r * flat);
            let rr = u * u + v * v;
            let px = img.pixel_mut(x, y);
            if rr >= 1.0 {
                px.copy_from_slice(&[0.01, 0.01, 0.012]);
                continue;
            }
            let mu = (1.0 - rr).sqrt();
            let limb = 0.35 + 0.65 * mu.powf(0.8);
            let lat = v;
            // bandes
            let mut b = 0.0;
            for (c, wdt, a) in [(0.18, 0.07, -0.35), (-0.22, 0.08, -0.4), (0.45, 0.05, -0.2), (-0.5, 0.05, -0.22), (0.0, 0.1, 0.12), (0.7, 0.08, -0.15), (-0.72, 0.08, -0.15)] {
                b += a * (-((lat - c) / wdt).powi(2)).exp();
            }
            // turbulence fine des bandes
            b += 0.05 * ((u * 23.0 + lat * 7.0).sin() * (lat * 41.0).cos()) * (-((lat.abs() - 0.2) / 0.1).powi(2)).exp();
            // spots et GTR
            for &(su, sv, s, a) in &spots {
                b += a * (-((u - su).powi(2) + (v - sv).powi(2)) / (s * s)).exp();
            }
            let grs = (-(((u - 0.3) / 0.12).powi(2) + ((v + 0.3) / 0.07).powi(2))).exp();
            let base = (0.62 + b).clamp(0.05, 1.0) * limb;
            let rch = base * (1.0 + 0.08 + 0.25 * grs);
            let gch = base * (0.95 - 0.1 * grs);
            let bch = base * (0.80 - 0.2 * grs + 0.1 * (b.min(0.0)));
            px.copy_from_slice(&[rch as f32, gch as f32, bch as f32]);
        }
    }
    img.map(|v| v * 0.75)
}

/// Texture lunaire (cratères) plein champ, mono.
pub fn moon(w: usize, h: usize) -> Image {
    let mut rng = Rng(9);
    let craters: Vec<(f64, f64, f64)> = (0..140).map(|_| (rng.u() * w as f64, rng.u() * h as f64, 3.0 + 25.0 * rng.u().powi(3))).collect();
    let img = Image::from_fn(w, h, |x, y| {
        let (xf, yf) = (x as f64, y as f64);
        let mut v = 0.45 + 0.05 * ((xf * 0.05).sin() + (yf * 0.037).cos());
        for &(cx, cy, r) in &craters {
            let d = (xf - cx).hypot(yf - cy) / r;
            if d < 1.0 {
                v -= 0.12 * (1.0 - d * d);
            } else if d < 1.25 {
                v += 0.15 * (1.0 - (d - 1.0) / 0.25);
            }
        }
        v.clamp(0.02, 1.0) as f32
    });
    Image::from_planes(&[img.clone(), img.clone(), img])
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let out = PathBuf::from(args.get(1).cloned().unwrap_or_else(|| "jupiter_test.ser".into()));
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(300);
    let env = |k: &str, d: f64| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(d);
    let (w, h) = (360usize, 280usize);
    let r = env("SYN_R", 78.0);
    let gain = env("SYN_GAIN", 300.0);
    let turb = env("SYN_TURB", 0.6);
    let seeing = env("SYN_SEEING", 0.9);
    let mono = env("SYN_MONO", 0.0) > 0.0;
    let surface = env("SYN_SURFACE", 0.0) > 0.0;
    let mut truth = if surface { moon(w, h) } else { jupiter(w, h, w as f64 / 2.0, h as f64 / 2.0, r) };
    if mono {
        truth = truth.mean_lum();
    }
    if let Some(t) = args.get(3) {
        astroforge::io::save(t, &truth)?;
    }
    let mut rng = Rng(env("SYN_SEED", 7.0) as u64);
    let (odx, ody) = (env("SYN_DX", 0.0), env("SYN_DY", 0.0));
    let mut frames = Vec::with_capacity(n);
    for i in 0..n {
        // seeing : flou log-normal, distorsion locale corrélée
        let blur = (seeing * (0.45 * rng.n()).exp()).clamp(0.4, 5.0);
        let amp = turb * blur;
        let small = |rng: &mut Rng| Image::from_vec(5, 5, 1, (0..25).map(|_| (rng.n() * amp) as f32).collect());
        let fx = geom::resize(&small(&mut rng), w, h, Interp::Cubic);
        let fy = geom::resize(&small(&mut rng), w, h, Interp::Cubic);
        let (mut gx, gy) = (rng.n() * 6.0 + odx, rng.n() * 6.0 + ody);
        if i % 97 == 50 && !surface {
            gx += 120.0; // planète qui sort du champ
        }
        let disp = if surface { 0.0 } else { 1.2 };
        let nc = truth.c;
        let mut f = Image::new(w, h, nc);
        for c in 0..nc {
            let ch = truth.channel(c);
            let dy = if nc == 3 { (c as f64 - 1.0) * disp } else { 0.0 };
            let warped = geom::remap_fn(&ch, w, h, Interp::Cubic, Border::Nearest, |x, y| (x as f64 + fx.data[y * w + x] as f64 + gx, y as f64 + fy.data[y * w + x] as f64 + gy + dy));
            f.set_channel(c, &ndimage::gaussian(&warped, blur));
        }
        let veil = if i % 61 == 30 { 0.25 } else { 1.0 };
        for v in f.data.iter_mut() {
            let s = (*v as f64 * veil).max(0.0) * gain;
            let e = s + s.sqrt() * rng.n() + 3.0 * rng.n();
            *v = (e / gain).clamp(0.0, 1.0) as f32;
        }
        frames.push(f);
    }
    astroforge::io::ser::write_ser(&out, &frames, "AstroForge", "synthétique", "", None)?;
    println!("Écrit : {} ({} frames {}×{})", out.display(), n, w, h);
    Ok(())
}
