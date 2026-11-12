//! Time-lapse multi-vidéos : éclipse partielle synthétique avec dérive du
//! suivi, voiles et seeing variable.
mod common;
use astroforge::frames::Frames;
use astroforge::image::Image;
use astroforge::ndimage;
use astroforge::planetary_auto::robust_disc;
use astroforge::registration::phase_corr;
use astroforge::timelapse as tl;
use common::Rng;
use std::sync::Arc;

const R: f64 = 45.0;

fn scene(w: usize, h: usize, sx: f64, sy: f64, mx: f64, my: f64, t: f64) -> Image {
    let spots = [(0.3, -0.2, 0.08), (-0.35, 0.15, 0.06), (0.0, 0.4, 0.07)];
    Image::from_fn(w, h, |x, y| {
        let (u, v) = ((x as f64 - sx) / R, (y as f64 - sy) / R);
        let rr = u * u + v * v;
        if rr >= 1.0 || (x as f64 - mx).hypot(y as f64 - my) < 46.0 {
            return 0.015;
        }
        let mut b = 0.3 + 0.7 * (1.0 - rr).sqrt();
        for &(su, sv, s) in &spots {
            b *= 1.0 - 0.7 * (-((u - su).powi(2) + (v - sv).powi(2)) / (s * s)).exp();
        }
        (0.7 * b * t) as f32
    })
}

fn series(n: usize) -> (Vec<tl::SeqInput>, Vec<(f64, f64)>) {
    let mut rng = Rng::new(21);
    let (w, h) = (200usize, 170usize);
    let mut inputs = Vec::new();
    let mut suns = Vec::new();
    let t0 = chrono::NaiveDate::from_ymd_opt(2026, 8, 12).unwrap().and_hms_opt(17, 0, 0).unwrap();
    for k in 0..n {
        let f = k as f64 / (n - 1) as f64;
        let (sx, sy) = (100.0 + rng.uniform(-15.0, 15.0), 85.0 + rng.uniform(-12.0, 12.0));
        let (mdx, mdy) = (-100.0 + 150.0 * f, -30.0 + 40.0 * f);
        let trans = if k == 3 { 0.6 } else { 1.0 };
        let frames: Vec<Image> = (0..16)
            .map(|_| {
                let (jx, jy) = (rng.normal(0.0, 1.5), rng.normal(0.0, 1.5));
                let im = scene(w, h, sx + jx, sy + jy, sx + jx + mdx, sy + jy + mdy, trans);
                let b = rng.uniform(0.6, 1.6);
                rng.noise(&ndimage::gaussian(&im, b), 0.008).clip01()
            })
            .collect();
        suns.push((sx, sy));
        inputs.push(tl::SeqInput { name: format!("v{:02}.ser", k), frames: Arc::new(frames), time: Some(t0 + chrono::Duration::minutes(4 * k as i64)) });
    }
    (inputs, suns)
}

#[test]
fn eclipse_series_same_framing_and_rendering() {
    let (inputs, _) = series(8);
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("ASTROFORGE_HOME", dir.path());
    let opts = tl::SeqOptions { effort: astroforge::planetary_auto::Effort::Rapide, ..Default::default() };
    let p = tl::run_timelapse(&inputs, &opts, None).unwrap();
    assert_eq!(p.frames.len(), 8);
    assert_eq!(p.framing, tl::Framing::Disc);
    assert!((p.disc_radius - R).abs() < 3.0, "rayon {}", p.disc_radius);
    let (w, h) = (p.frames[0].w, p.frames[0].h);
    assert!(p.frames.iter().all(|f| f.w == w && f.h == h), "tailles différentes");
    // le limbe est au centre exact du canevas sur toutes les images
    for (i, f) in p.frames.iter().enumerate() {
        let (cx, cy, _) = robust_disc(f, Some(p.disc_radius)).unwrap();
        assert!((cx - (w as f64 / 2.0 - 0.5)).abs() < 0.6 && (cy - (h as f64 / 2.0 - 0.5)).abs() < 0.6, "image {} : centre ({:.2}, {:.2})", i, cx, cy);
    }
    // les détails (taches) restent immobiles entre images successives non éclipsées
    let (dx, dy, _) = phase_corr(&p.frames[0], &p.frames[1]);
    assert!(dx.hypot(dy) < 0.5, "dérive {} {}", dx, dy);
    // le voile de la vidéo 4 est mesuré…
    let lv: Vec<f64> = p.items.iter().map(|i| i.level).collect();
    assert!(lv[3] < 0.75 * lv[2] && lv[3] < 0.75 * lv[4], "{:?}", lv);
    // … sans que le croissant ne fausse la mesure des autres
    let others: Vec<f64> = lv.iter().enumerate().filter(|(i, _)| *i != 3).map(|(_, v)| *v).collect();
    let m = others.iter().sum::<f64>() / others.len() as f64;
    assert!(others.iter().all(|v| (v / m - 1.0).abs() < 0.06), "{:?}", lv);
    // dé-scintillement : le voile est corrigé
    let g = tl::brightness_gains(&lv, tl::Brightness::Deflicker, 5, p.reference);
    let after: Vec<f64> = lv.iter().zip(&g).map(|(a, b)| a * b).collect();
    assert!((after[3] / after[2] - 1.0).abs() < 0.08, "{:?}", after);
    // rendu : même taille, valeurs bornées, horodatage incrusté
    let out = tl::render(&p, &tl::RenderOptions::default());
    assert!(out.iter().all(|f| f.all_finite() && f.max_value() <= 1.0 && f.w == w));
    let anim = tl::interpolate(&out, 2);
    assert_eq!(anim.len(), 8 + 7 * 2);
    let gif = dir.path().join("a.gif");
    tl::export_animation(&p, &tl::RenderOptions::default(), &gif, 8, 1).unwrap();
    assert_eq!(tl::render_animation(&p, &tl::RenderOptions::default(), 3).len(), 8 + 7 * 3);
    assert!(std::fs::metadata(&gif).unwrap().len() > 1000);
    let files = tl::export_frames(&out, &dir.path().join("img"), "eclipse", "png").unwrap();
    assert_eq!(files.len(), 8);
    let _ = Frames::len(&out);
}
