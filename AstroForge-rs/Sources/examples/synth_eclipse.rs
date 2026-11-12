//! Série de vidéos SER synthétiques d'une éclipse partielle de Soleil, pour
//! tester le time-lapse multi-vidéos.
//!
//!     cargo run --release --example synth_eclipse -- DOSSIER [vidéos] [frames]
//!
//! Le Soleil (assombrissement centre-bord, taches) dérive dans le champ d'une
//! vidéo à l'autre (suivi imparfait) ; la Lune avance ; la transparence
//! varie (voiles) ; seeing et bruit variables ; une vidéo toutes les 3 min.
//! SYN_ROT=0.4 ajoute une rotation de champ (degrés par vidéo).

use astroforge::image::Image;
use astroforge::ndimage;
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

pub const W: usize = 320;
pub const H: usize = 260;
pub const R_SUN: f64 = 70.0;
pub const R_MOON: f64 = 72.0;

/// Soleil centré en (sx, sy), Lune en (mx, my), transparence `t`.
pub fn scene(sx: f64, sy: f64, mx: f64, my: f64, t: f64, rot_deg: f64) -> Image {
    let (ca, sa) = (rot_deg.to_radians().cos(), rot_deg.to_radians().sin());
    let spots: Vec<(f64, f64, f64)> = [(0.25, -0.2, 0.06), (-0.35, 0.1, 0.04), (0.05, 0.4, 0.05), (-0.1, -0.45, 0.03), (0.55, 0.3, 0.035), (-0.6, -0.25, 0.04)]
        .iter()
        .map(|&(u, v, s)| (ca * u - sa * v, sa * u + ca * v, s))
        .collect();
    let one = Image::from_fn(W, H, |x, y| {
        let (u, v) = ((x as f64 - sx) / R_SUN, (y as f64 - sy) / R_SUN);
        let rr = u * u + v * v;
        let dm = (x as f64 - mx).hypot(y as f64 - my);
        if rr >= 1.0 || dm < R_MOON {
            return 0.015;
        }
        let mu = (1.0 - rr).sqrt();
        let mut b = 0.3 + 0.7 * mu;
        for &(su, sv, s) in spots.iter() {
            let d2 = (u - su).powi(2) + (v - sv).powi(2);
            b *= 1.0 - 0.7 * (-d2 / (s * s)).exp() - 0.2 * (-d2 / (4.0 * s * s)).exp();
        }
        (0.75 * b * t) as f32
    });
    // teinte légèrement chaude (filtre solaire)
    let r = one.map(|v| v * 1.05);
    let b = one.map(|v| v * 0.85);
    Image::from_planes(&[r, one, b])
}

fn main() -> anyhow::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(a.get(1).cloned().unwrap_or_else(|| "eclipse_test".into()));
    let nv: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
    let nf: usize = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(40);
    // rotation de champ (degrés par vidéo), variable SYN_ROT
    let rot: f64 = std::env::var("SYN_ROT").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    std::fs::create_dir_all(&dir)?;
    let mut rng = Rng(11);
    let t0 = chrono::NaiveDate::from_ymd_opt(2026, 8, 12).unwrap().and_hms_opt(17, 30, 0).unwrap();
    for k in 0..nv {
        let f = k as f64 / (nv - 1).max(1) as f64;
        // dérive du suivi et position de la Lune (relative au Soleil)
        let (sx, sy) = (W as f64 / 2.0 + 25.0 * (f * 3.0).sin() + 8.0 * rng.n(), H as f64 / 2.0 - 18.0 * f + 6.0 * rng.n());
        let th = rot * k as f64;
        let (mdx0, mdy0) = (-150.0 + 230.0 * f, -60.0 + 70.0 * f);
        let (ct, st) = (th.to_radians().cos(), th.to_radians().sin());
        let (mdx, mdy) = (ct * mdx0 - st * mdy0, st * mdx0 + ct * mdy0);
        let trans = if k % 4 == 2 { 0.7 } else { 0.95 + 0.05 * rng.u() };
        let mut frames = Vec::new();
        let mut stamps = Vec::new();
        for i in 0..nf {
            let (jx, jy) = (rng.n() * 2.5, rng.n() * 2.5);
            let im = scene(sx + jx, sy + jy, sx + jx + mdx, sy + jy + mdy, trans, th);
            let blur = (1.0 * (0.4 * rng.n()).exp()).clamp(0.5, 3.0);
            let mut im = ndimage::gaussian_img(&im, blur);
            for v in im.data.iter_mut() {
                let e = (*v as f64) * 500.0;
                *v = ((e + e.sqrt() * rng.n() + 3.0 * rng.n()) / 500.0).clamp(0.0, 1.0) as f32;
            }
            frames.push(im);
            let t = t0 + chrono::Duration::seconds(180 * k as i64) + chrono::Duration::milliseconds(25 * i as i64);
            stamps.push(t.format("%Y-%m-%dT%H:%M:%S%.3f").to_string());
        }
        let p = dir.join(format!("eclipse_{:02}.ser", k + 1));
        astroforge::io::ser::write_ser(&p, &frames, "AstroForge", "synthétique", "", Some(&stamps))?;
        println!("{} : Soleil ({:.1}, {:.1})", p.display(), sx, sy);
    }
    Ok(())
}
