//! Planétaire entièrement automatique : vidéo synthétique avec seeing
//! variable, nuages et planète sortant du champ.
mod common;
use astroforge::geom::{self, Interp};
use astroforge::ndimage::{self, Border};
use astroforge::planetary_auto as pa;
use astroforge::registration::{apply_shift, phase_corr};
use astroforge::Image;
use common::Rng;

fn corr(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mb = b.iter().map(|&v| v as f64).sum::<f64>() / n;
    let (mut s, mut x, mut y) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        let (p, q) = (a[i] as f64 - ma, b[i] as f64 - mb);
        s += p * q;
        x += p * p;
        y += q * q;
    }
    s / (x * y).sqrt()
}

/// Fidélité des détails (passe-haut) à la vérité, après recalage.
fn fidelity(img: &Image, truth: &Image) -> f64 {
    let t = truth.mean_lum();
    let a = img.mean_lum();
    let (dx, dy, _) = phase_corr(&t, &a);
    let a = apply_shift(&a, dx, dy);
    let hp = |z: &Image| z.zip_map(&ndimage::gaussian(z, 2.0), |p, q| p - q);
    let b = a.w / 5;
    let c = |z: &Image| z.crop(b, b, z.w - 2 * b, z.h - 2 * b).data;
    corr(&c(&hp(&a)), &c(&hp(&t)))
}

fn video(n: usize, seed: u64) -> (Image, Vec<Image>) {
    let (w, h, r) = (200usize, 170usize, 45.0);
    let mut rng = Rng::new(seed);
    let blobs: Vec<(f64, f64, f64, f64)> = (0..50).map(|_| (rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), rng.uniform(-0.25, 0.25), rng.uniform(0.04, 0.12))).collect();
    let truth = Image::from_fn(w, h, |x, y| {
        let (u, v) = ((x as f64 - 100.0) / r, (y as f64 - 85.0) / r);
        let rr = u * u + v * v;
        if rr >= 1.0 {
            return 0.01;
        }
        let mut b = 0.55 + 0.15 * (v * 9.0).sin();
        for &(bu, bv, a, s) in &blobs {
            b += a * (-((u - bu).powi(2) + (v - bv).powi(2)) / (s * s)).exp();
        }
        (b.clamp(0.05, 1.0) * (0.4 + 0.6 * (1.0 - rr).sqrt())) as f32
    });
    let mut frames = Vec::new();
    for i in 0..n {
        let blur = (0.9 * (0.45 * rng.normal(0.0, 1.0)).exp()).clamp(0.5, 3.5);
        let small = |rng: &mut Rng| Image::from_vec(4, 4, 1, (0..16).map(|_| rng.normal(0.0, 0.5 * blur) as f32).collect());
        let (fx, fy) = (geom::resize(&small(&mut rng), w, h, Interp::Cubic), geom::resize(&small(&mut rng), w, h, Interp::Cubic));
        let mut gx = rng.normal(0.0, 4.0);
        let gy = rng.normal(0.0, 4.0);
        if i == 17 || i == 41 {
            gx += 75.0; // planète coupée par le bord
        }
        let f = geom::remap_fn(&truth, w, h, Interp::Cubic, Border::Nearest, |x, y| (x as f64 + fx.data[y * w + x] as f64 + gx, y as f64 + fy.data[y * w + x] as f64 + gy));
        let mut f = ndimage::gaussian(&f, blur);
        if i == 9 || i == 33 || i == 52 {
            f = f.scale(0.25); // nuage
        }
        frames.push(rng.noise(&f, 0.012).clip01());
    }
    (truth, frames)
}

#[test]
fn automatic_planetary_end_to_end() {
    let (truth, frames) = video(64, 5);
    let dir = tempfile::tempdir().unwrap();
    let opts = pa::AutoOptions { effort: pa::Effort::Equilibre, memory_path: Some(dir.path().join("mem.json")), ..Default::default() };
    let r = pa::run_auto(&frames, &opts, None).unwrap();
    // frames voilées et planète coupée : rejetées
    for i in [9, 17, 33, 41, 52] {
        assert!(r.rejected.contains(&i), "frame {} non rejetée : {:?}", i, r.rejected);
    }
    assert!(r.rejected.len() <= 8, "{:?}", r.rejected);
    assert_eq!(r.features.mode, "disque");
    assert!((r.features.disc_radius - 45.0).abs() < 8.0, "{}", r.features.disc_radius);
    assert!(r.image.all_finite() && r.image.max_value() <= 1.0 && r.image.min_value() >= 0.0);
    assert!(r.plan.keep >= 5 && r.plan.keep <= 64 - r.rejected.len());
    assert!(r.plan.psf_sigma > 0.4 && r.plan.psf_sigma < 3.6);
    assert!(r.decisions.len() >= 8);
    // l'empilement bat nettement la meilleure frame, la restauration ne dégrade pas
    let s = r.image.w;
    let truth = truth.crop(100 - s / 2, 85 - s / 2, s, s);
    let (fb, fs, ff) = (fidelity(&r.best_frame, &truth), fidelity(&r.stack, &truth), fidelity(&r.image, &truth));
    eprintln!("fidélité : meilleure frame {:.3}, empilement {:.3}, final {:.3}", fb, fs, ff);
    assert!(fs > fb + 0.015, "empilement {} meilleure frame {}", fs, fb);
    assert!(ff > fs - 0.01, "final {} empilement {}", ff, fs);
    assert!(r.metrics.snr_gain > 1.5, "{:?}", r.metrics);
    // retouche rejouable
    let again = r.finisher.render(r.plan.strength, r.plan.denoise_k);
    assert_eq!(again.image.data, r.image.data);
    // la mémoire a été écrite et sert au traitement suivant
    let r2 = pa::run_auto(&frames, &pa::AutoOptions { effort: pa::Effort::Rapide, ..opts.clone() }, None).unwrap();
    assert!(r2.memory_hits >= 1);
    let txt = serde_json::to_string(&r2.report_json()).unwrap();
    assert!(txt.contains("decisions") && txt.contains("plan"));
}

#[test]
fn surface_mode_without_disc() {
    let mut rng = Rng::new(3);
    let tex = ndimage::gaussian(&Image::from_vec(150, 120, 1, (0..150 * 120).map(|_| rng.uniform(0.3, 0.7) as f32).collect()), 2.0);
    let frames: Vec<Image> = (0..24)
        .map(|_| {
            let (dx, dy) = (rng.normal(0.0, 2.0), rng.normal(0.0, 2.0));
            let f = apply_shift(&tex, dx, dy);
            let b = rng.uniform(0.5, 1.5);
            rng.noise(&ndimage::gaussian(&f, b), 0.01)
        })
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let opts = pa::AutoOptions { effort: pa::Effort::Rapide, memory_path: Some(dir.path().join("m.json")), learn: false, ..Default::default() };
    let r = pa::run_auto(&frames, &opts, None).unwrap();
    assert_eq!(r.features.mode, "surface");
    assert!(r.image.all_finite());
    assert!(!dir.path().join("m.json").exists());
}

#[test]
fn cancellation_stops_processing() {
    let (_, frames) = video(12, 2);
    let flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let opts = pa::AutoOptions { cancel: Some(flag), learn: false, use_memory: false, ..Default::default() };
    assert!(pa::run_auto(&frames, &opts, None).is_err());
}
