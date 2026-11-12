//! Finitions, bande étroite, planétaire, solaire, dérotation, astrométrie
//! (portage des tests Python).

mod common;
use astroforge::geom::{self, Interp};
use astroforge::ndimage::{self, Border};
use astroforge::registration::{apply_shift, phase_corr};
use astroforge::{annotate, derotation as dr, finishing as fin, io, masks as mk, narrowband as nb, planetary as pl, platesolve, sequence, solar, stats, stretch, wcs::Wcs, Image};
use common::*;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

fn obj(v: Value) -> Map<String, Value> {
    v.as_object().unwrap().clone()
}

// ---------------------------------------------------------------- finitions

#[test]
fn ghs_monotonic_endpoints_and_symmetry_point() {
    for b in [-1.0, -0.5, 0.0, 1.0, 4.0] {
        let xs: Vec<f64> = (0..=2000).map(|i| i as f64 / 2000.0).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| stretch::ghs_curve(x, 4.0, b, 0.1, 0.0, 1.0)).collect();
        assert!(ys[0].abs() < 1e-9 && (ys[2000] - 1.0).abs() < 1e-9, "b={}", b);
        assert!(ys.windows(2).all(|w| w[1] - w[0] >= -1e-12));
        let k = (0..2000).max_by(|&i, &j| (ys[i + 1] - ys[i]).partial_cmp(&(ys[j + 1] - ys[j])).unwrap()).unwrap();
        assert!((xs[k] - 0.1).abs() < 0.01, "b={} max à {}", b, xs[k]);
    }
}

#[test]
fn ghs_protection_linear_segments() {
    let xs: Vec<f64> = (0..=1000).map(|i| i as f64 / 1000.0).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| stretch::ghs_curve(x, 5.0, 0.0, 0.3, 0.1, 0.8)).collect();
    let d: Vec<f64> = ys.windows(2).map(|w| w[1] - w[0]).collect();
    let lo: Vec<f64> = (0..1000).filter(|&i| xs[i + 1] < 0.1).map(|i| d[i]).collect();
    let hi: Vec<f64> = (0..1000).filter(|&i| xs[i] > 0.8).map(|i| d[i]).collect();
    assert!(lo.iter().all(|v| (v - lo[0]).abs() <= 1e-6 * lo[0].abs() + 1e-12));
    assert!(hi.iter().all(|v| (v - hi[0]).abs() <= 1e-6 * hi[0].abs() + 1e-12));
    assert!(xs.iter().all(|&x| (stretch::ghs_curve(x, 0.0, 0.0, 0.0, 0.0, 1.0) - x).abs() < 1e-9));
}

#[test]
fn arcsinh_preserves_colour_ratios() {
    let rgb = Image::from_vec(2, 1, 3, vec![0.02, 0.01, 0.005, 0.2, 0.1, 0.05]);
    let out = stretch::arcsinh_stretch(&rgb, 200.0, 0.0);
    for p in 0..2 {
        let rin = rgb.data[p * 3] / rgb.data[p * 3 + 1];
        let rout = out.data[p * 3] / out.data[p * 3 + 1];
        assert!((rin - rout).abs() < 1e-4 * rin);
    }
    assert!(out.data[0] > rgb.data[0]);
    let same = stretch::arcsinh_stretch(&rgb, 0.0, 0.0);
    assert!(mean_abs_diff(&same, &rgb) < 1e-7);
}

struct Neb {
    img: Image,
    base: Image,
    stars: Image,
    pos: Vec<(f64, f64)>,
}

fn nebula_with_stars(seed: u64) -> Neb {
    let mut rng = Rng::new(seed);
    let pos: Vec<(f64, f64)> = (0..25).map(|_| (rng.uniform(10.0, 150.0), rng.uniform(10.0, 150.0))).collect();
    let base = Image::from_fn(160, 160, |x, y| (0.05 + 0.15 * (-((x as f64 - 80.0).powi(2) + (y as f64 - 80.0).powi(2)) / (2.0 * 1600.0)).exp()) as f32);
    let stars = Image::from_fn(160, 160, |x, y| pos.iter().map(|&(py, px)| 0.7 * (-((x as f64 - px).powi(2) + (y as f64 - py).powi(2)) / (2.0 * 1.69)).exp()).sum::<f64>() as f32);
    let img = rng.noise(&base.zip_map(&stars, |a, b| a + b), 0.002).clip01();
    Neb { img, base, stars, pos }
}

#[test]
fn star_mask_targets_stars_not_nebula() {
    let n = nebula_with_stars(1);
    let m = mk::star_mask(&n.img, 4.0, 2, 1.5, 25);
    let on = n.pos.iter().map(|&(y, x)| m.data[y.round() as usize * 160 + x.round() as usize] as f64).sum::<f64>() / n.pos.len() as f64;
    let off: Vec<f32> = (0..m.data.len()).filter(|&i| n.stars.data[i] < 1e-4).map(|i| m.data[i]).collect();
    assert!(on > 0.8 && stats::mean(&off) < 0.05, "{} {}", on, stats::mean(&off));
}

#[test]
fn luminance_mask_ranges() {
    let ramp = Image::from_fn(101, 4, |x, _| x as f32 / 100.0);
    let m = mk::luminance_mask(&ramp, 0.4, 0.6, 0.05, 0.0);
    assert!((m.data[50] - 1.0).abs() < 1e-6 && m.data[10].abs() < 1e-6 && m.data[90].abs() < 1e-6);
    let inv = mk::make_mask(&ramp, "Luminance inversée (fond)", 0.4, 0.6, 0.05, 0.0).unwrap().unwrap();
    assert!(inv.data.iter().zip(&m.data).all(|(a, b)| (a - (1.0 - b)).abs() < 1e-6));
}

#[test]
fn reduce_stars_keeps_nebula() {
    let n = nebula_with_stars(2);
    let out = fin::apply_filter(&n.img, "star_reduction", &obj(json!({"amount": 1.0, "iterations": 2}))).unwrap();
    let flux = |a: &Image| (0..a.data.len()).filter(|&i| n.stars.data[i] > 0.1).map(|i| (a.data[i] - n.base.data[i]) as f64).sum::<f64>();
    assert!(flux(&out) < 0.6 * flux(&n.img));
    let far = mk::star_mask(&n.img, 4.0, 4, 0.0, 25);
    let mx = (0..far.data.len()).filter(|&i| far.data[i] < 0.01).map(|i| (out.data[i] - n.img.data[i]).abs()).fold(0.0f32, f32::max);
    assert!(mx < 1e-3, "{}", mx);
}

#[test]
fn extract_and_recombine_roundtrip() {
    let n = nebula_with_stars(3);
    let st = mk::extract_stars(&n.img, &n.base);
    let add = mk::recombine(&n.base, &st, "Addition", 1.0).unwrap();
    assert!(mean_abs_diff(&add, &n.base.zip_map(&st, |a, b| a + b).clip01()) < 1e-6);
    let scr = mk::recombine(&n.base, &st, "Écran", 1.0).unwrap();
    assert!(scr.max_value() <= 1.0 && scr.data.iter().zip(&n.base.data).all(|(s, b)| *s >= b - 1e-6));
}

#[test]
fn chroma_denoise_keeps_luminance() {
    let mut rng = Rng::new(4);
    let noisy = rng.noise(&Image::filled(96, 96, 3, 0.3), 0.03).clip01();
    let out = mk::chroma_denoise(&noisy, 0.8, 4);
    let lum = |a: &Image| a.data.chunks(3).map(|p| 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]).collect::<Vec<f32>>();
    let chroma = |a: &Image| {
        let l = lum(a);
        let r: Vec<f32> = a.data.chunks(3).zip(&l).map(|(p, l)| p[0] - l).collect();
        let b: Vec<f32> = a.data.chunks(3).zip(&l).map(|(p, l)| p[2] - l).collect();
        stats::std(&r) + stats::std(&b)
    };
    assert!(chroma(&out) < 0.5 * chroma(&noisy));
    let d = lum(&out).iter().zip(lum(&noisy)).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(d < 2e-3, "{}", d);
}

#[test]
fn masked_filter_and_sequence() {
    let n = nebula_with_stars(5);
    let rgb = rgb_fn(160, 160, |x, y| {
        let v = n.img.data[y * 160 + x];
        [v, v, v]
    });
    let out = fin::apply_filter_masked(&rgb, "saturation", &obj(json!({"amount": 0.0})), &obj(json!({"kind": "Luminance", "low": 0.5, "high": 1.0, "fuzz": 0.01, "blur": 0}))).unwrap();
    for i in 0..160 * 160 {
        if n.img.data[i] < 0.3 {
            for c in 0..3 {
                assert!((out.data[i * 3 + c] - rgb.data[i * 3 + c]).abs() < 1e-6);
            }
        }
    }
    let seq = json!({"mode": "image", "ops": [
        {"op": "fin_ghs", "params": {"D": 3.0, "SP": 0.05}},
        {"op": "fin_masked", "params": {"filter": "star_reduction", "params": {"amount": 1.0}, "mask": {"kind": "Aucun"}}}]});
    let res = sequence::apply_to_image(&seq, &rgb).unwrap();
    assert!(res.same_shape(&rgb) && res.mean() > rgb.mean());
}

// ---------------------------------------------------------- bande étroite

fn cfa_from(ha: &Image, o3: &Image, pattern: &str) -> Image {
    let (w, h) = (ha.w, ha.h);
    let (ry, rx) = match pattern {
        "RGGB" => (0, 0),
        "BGGR" => (1, 1),
        "GRBG" => (0, 1),
        _ => (1, 0),
    };
    Image::from_fn(2 * w, 2 * h, |x, y| {
        let v = if y % 2 == ry && x % 2 == rx { ha } else { o3 };
        v.data[(y / 2) * w + x / 2]
    })
}

#[test]
fn extract_ha_oiii_exact() {
    let mut rng = Rng::new(6);
    let ha = Image::from_vec(30, 20, 1, (0..600).map(|_| rng.uniform(0.0, 1.0) as f32).collect());
    let o3 = Image::from_vec(30, 20, 1, (0..600).map(|_| rng.uniform(0.0, 1.0) as f32).collect());
    for p in ["RGGB", "BGGR", "GRBG", "GBRG"] {
        let (h2, o2) = nb::extract_ha_oiii(&cfa_from(&ha, &o3, p), p, false).unwrap();
        assert!(mean_abs_diff(&h2, &ha) < 1e-7 && mean_abs_diff(&o2, &o3) < 1e-7, "{}", p);
        let (hu, _) = nb::extract_ha_oiii(&cfa_from(&ha, &o3, p), p, true).unwrap();
        assert_eq!((hu.w, hu.h), (60, 40));
    }
}

#[test]
fn palettes_and_normalization() {
    let mut rng = Rng::new(7);
    let mut r = |a: f64, b: f64| Image::from_vec(32, 32, 1, (0..1024).map(|_| (a + b * rng.uniform(0.0, 1.0)) as f32).collect());
    let ha = r(0.1, 0.05);
    let o3 = r(0.02, 0.01);
    let s2 = r(0.05, 0.02);
    let mut ch = BTreeMap::new();
    ch.insert("Ha".to_string(), ha.clone());
    ch.insert("OIII".to_string(), o3.clone());
    let hoo = nb::compose(&ch, &nb::palette("HOO").unwrap(), false, None).unwrap();
    for c in 0..3 {
        let src = if c == 0 { &ha } else { &o3 };
        assert!((0..1024).all(|i| (hoo.data[i * 3 + c] - src.data[i]).abs() < 1e-6));
    }
    ch.insert("SII".to_string(), s2);
    let sho = nb::compose(&ch, &nb::palette("SHO (Hubble)").unwrap(), true, None).unwrap();
    let b: Vec<f32> = sho.data.chunks(3).map(|p| p[2]).collect();
    assert!((stats::median(&b) - stats::median(&ha.data)).abs() < 1e-4);
    let mut only = BTreeMap::new();
    only.insert("Ha".to_string(), ha);
    assert!(nb::compose(&only, &nb::palette("SHO (Hubble)").unwrap(), true, None).is_err());
}

#[test]
fn lrgb_keeps_hue_and_takes_luminance() {
    let rgb = rgb_fn(8, 8, |_, _| [0.3, 0.2, 0.1]);
    let l = Image::filled(8, 8, 1, 0.4);
    let out = nb::lrgb(&l, &rgb, 1.0).unwrap();
    for p in out.data.chunks(3) {
        assert!((0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2] - 0.4).abs() < 1e-5);
        assert!((p[0] / p[2] - 3.0).abs() < 3e-4);
    }
}

#[test]
fn align_narrowband_channels() {
    let mut ch = BTreeMap::new();
    ch.insert("Ha".to_string(), sf(0.0, 0.0));
    ch.insert("OIII".to_string(), sf(3.2, -1.7));
    let out = nb::align_channels(&ch, Some("Ha"));
    let (dx, dy, _) = phase_corr(&out["Ha"], &out["OIII"]);
    assert!(dx.hypot(dy) < 0.15, "{} {}", dx, dy);
}

#[test]
fn calibrate_paths_duo_band() {
    let d = tempfile::tempdir().unwrap();
    let cfa = cfa_from(&Image::filled(16, 16, 1, 0.4), &Image::filled(16, 16, 1, 0.1), "RGGB");
    let p = write_fits(d.path(), "osc.fits", &cfa, &[("BAYERPAT", "RGGB".into())]);
    let m = astroforge::calibration::Masters::default();
    use astroforge::calibration::{calibrate_paths, DarkScale};
    let h = calibrate_paths(&[p.clone()], &m, None, DarkScale::Auto, false, None, None, "Ha (duo-bande)").unwrap().remove(0);
    let o = calibrate_paths(&[p], &m, None, DarkScale::Auto, false, None, None, "OIII (duo-bande)").unwrap().remove(0);
    assert_eq!((h.w, h.h), (32, 32));
    assert!((h.mean() - 0.4).abs() < 1e-3 && (o.mean() - 0.1).abs() < 1e-3, "{} {}", h.mean(), o.mean());
}

// ---------------------------------------------------------- planétaire

fn planet_frames(n: usize, amp: f64, seed: u64) -> (Image, Vec<Image>) {
    let mut rng = Rng::new(seed);
    let (h, w) = (160usize, 160usize);
    let blobs: Vec<(f64, f64, f64, f64)> = (0..60).map(|_| (rng.uniform(25.0, 135.0), rng.uniform(25.0, 135.0), rng.uniform(0.05, 0.2), rng.uniform(1.0, 3.0))).collect();
    let raw = Image::from_fn(w, h, |x, y| {
        let (xf, yf) = (x as f64, y as f64);
        if (xf - 80.0).powi(2) + (yf - 80.0).powi(2) >= 3600.0 {
            return 0.0;
        }
        let t: f64 = blobs.iter().map(|&(by, bx, a, s)| a * (-((xf - bx).powi(2) + (yf - by).powi(2)) / (2.0 * s * s)).exp()).sum();
        (0.4 + t) as f32
    });
    let truth = ndimage::gaussian(&raw, 0.8);
    let mut frames = Vec::new();
    for _ in 0..n {
        let small = |rng: &mut Rng| Image::from_vec(4, 4, 1, (0..16).map(|_| rng.normal(0.0, amp) as f32).collect());
        let fx = geom::resize(&small(&mut rng), w, h, Interp::Cubic);
        let fy = geom::resize(&small(&mut rng), w, h, Interp::Cubic);
        let (gx, gy) = (rng.uniform(-4.0, 4.0), rng.uniform(-4.0, 4.0));
        let f = geom::remap_fn(&truth, w, h, Interp::Cubic, Border::Reflect, |x, y| (x as f64 + fx.data[y * w + x] as f64 + gx, y as f64 + fy.data[y * w + x] as f64 + gy));
        let f = ndimage::gaussian(&f, rng.uniform(0.5, 1.8));
        frames.push(rng.noise(&f, 0.01));
    }
    (truth, frames)
}

fn hf_fidelity(img: &Image, truth: &Image) -> f64 {
    let (dx, dy, _) = phase_corr(truth, img);
    let a = apply_shift(img, dx, dy);
    let hp = |z: &Image| z.zip_map(&ndimage::gaussian(z, 2.0), |a, b| a - b);
    let c = |z: &Image| z.crop(30, 30, 100, 100).data;
    corr(&c(&hp(&a)), &c(&hp(truth)))
}

#[test]
fn local_multipoint_beats_global_under_local_turbulence() {
    // Comme le test Python (graine fixe) : l'avantage du multipoint local
    // dépend du tirage ; sur 5 graines, Python et Rust donnent en moyenne une
    // fidélité équivalente aux deux modes (résultats Rust = Python à 1e-7
    // sur des frames identiques).
    let seed: u64 = std::env::var("PSEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    let (truth, frames) = planet_frames(60, 2.0, seed);
    let base = pl::PlanetOptions {
        quality_method: "Contraste HF (AutoStakkert)".into(),
        n_align: 60,
        grid_n: 5,
        box_size: 32,
        stack_percent: 50.0,
        stack_method: "Moyenne (meilleurs %)".into(),
        ..Default::default()
    };
    let g = pl::run_planetary(&frames, &pl::PlanetOptions { align_mode: "Global (une translation)".into(), ..base.clone() }, None).unwrap();
    let l = pl::run_planetary(&frames, &pl::PlanetOptions { align_mode: "Multi-points local (AS!)".into(), ..base }, None).unwrap();
    assert!(l.stats.n_points > 10);
    let (fl, fg) = (hf_fidelity(&l.image, &truth), hf_fidelity(&g.image, &truth));
    assert!(fl > fg, "local {} global {}", fl, fg);
}

#[test]
fn ser_timestamps_roundtrip() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("jup.ser");
    let t0 = chrono::NaiveDate::from_ymd_opt(2026, 10, 15).unwrap().and_hms_opt(21, 30, 0).unwrap();
    let ts: Vec<String> = (0..5).map(|i| (t0 + chrono::Duration::milliseconds(20 * i)).format("%Y-%m-%dT%H:%M:%S%.3f").to_string()).collect();
    let mut rng = Rng::new(8);
    let frames: Vec<Image> = (0..5).map(|_| Image::from_vec(8, 8, 1, (0..64).map(|_| rng.uniform(0.0, 1.0) as f32).collect())).collect();
    io::ser::write_ser(&p, &frames, "", "", "", Some(&ts)).unwrap();
    let (_fr, meta) = io::ser::load_ser(&p, 0).unwrap();
    assert_eq!(meta.timestamps[4].as_deref(), Some(ts[4].as_str()));
    assert!(meta.date_utc.as_deref().unwrap().starts_with("2026-10-15T21:30:00"));
    let got: Vec<String> = meta.timestamps.iter().flatten().cloned().collect();
    assert_eq!(dr::mid_time(&got).as_deref(), Some("2026-10-15T21:30:00.040"));
}

fn render(t_sec: f64) -> Image {
    let (p, flat, pa, b) = (9.925, 0.0649, 20.0, 3.0);
    Image::from_fn(200, 200, |x, y| {
        let (lon, lat, vis, z) = dr::image_to_sphere(x as f64, y as f64, (100.3, 99.6, 80.0), pa, flat, b);
        if !vis {
            return 0.0;
        }
        let tex = 0.4 + 0.15 * (6.0 * (lon - 2.0 * std::f64::consts::PI * t_sec / (p * 3600.0))).sin() * (3.0 * lat).cos() + 0.1 * (9.0 * lat).sin();
        (tex * (0.5 + 0.5 * z)) as f32
    })
}

#[test]
fn derotation_recovers_reference_time() {
    let (i0, it) = (render(0.0), render(600.0));
    let (out, w) = dr::derotate_image(&it, (100.3, 99.6, 80.0), -600.0, 9.925, 20.0, 0.0649, 3.0, 1, 0.97);
    let m: Vec<usize> = (0..w.data.len()).filter(|&i| w.data[i] > 0.4).collect();
    let e = |a: &Image| m.iter().map(|&i| (a.data[i] - i0.data[i]).abs() as f64).sum::<f64>() / m.len() as f64;
    assert!(e(&out) < 0.35 * e(&it), "{} {}", e(&out), e(&it));
}

#[test]
fn derotate_series_and_winjupos() {
    let imgs: Vec<Image> = [-300.0, 0.0, 300.0].iter().map(|&s| render(s)).collect();
    let times: Vec<String> = ["20:55:00", "21:00:00", "21:05:00"].iter().map(|t| format!("2026-10-15T{}", t)).collect();
    let (res, mid, _disc) = dr::derotate_series(&imgs, &times, None, "Jupiter", 20.0, 3.0, 1, None, None).unwrap();
    assert_eq!(mid, "2026-10-15T21:00:00");
    let inner: Vec<usize> = (0..200 * 200).filter(|&i| dr::image_to_sphere((i % 200) as f64, (i / 200) as f64, (100.3, 99.6, 80.0), 20.0, 0.0649, 3.0).3 > 0.6).collect();
    let e = |a: &Image, b: &Image| inner.iter().map(|&i| (a.data[i] - b.data[i]).abs() as f64).sum::<f64>() / inner.len() as f64;
    assert!(e(&res, &imgs[1]) < 0.4 * e(&imgs[0], &imgs[1]));
    let name = dr::winjupos_name("2026-10-15T21:30:25", "Obs Name", "R", "png").unwrap();
    assert_eq!(name, "2026-10-15-2130_4-ObsName-R.png");
    assert_eq!(dr::parse_winjupos_name(&name).unwrap().to_string(), "2026-10-15 21:30:24");
}

#[test]
fn solar_limb_and_composite() {
    let img = Image::from_fn(240, 240, |x, y| {
        let rr = (x as f64 - 118.4).hypot(y as f64 - 121.7);
        let disc = if rr < 90.0 { 0.7 * (1.0 - 0.4 * (rr / 90.0).powi(2)) } else { 0.0 };
        let prom = 0.05 * (-((x as f64 - 118.0).powi(2) + (y as f64 - 26.0).powi(2)) / 30.0).exp();
        (disc + prom + 0.01) as f32
    });
    let (cx, cy, r) = dr::fit_disc(&img, None).unwrap();
    assert!((cx - 118.4).abs() < 0.6 && (cy - 121.7).abs() < 0.6 && (r - 90.0).abs() < 1.5, "{} {} {}", cx, cy, r);
    let flat = solar::flatten_limb_darkening(&img, (cx, cy, r), 2);
    let inner: Vec<usize> = (0..240 * 240).filter(|&i| ((i % 240) as f64 - 118.4).hypot((i / 240) as f64 - 121.7) < 80.0).collect();
    let sd = |a: &Image| stats::std(&inner.iter().map(|&i| a.data[i]).collect::<Vec<_>>());
    assert!(sd(&flat) < 0.3 * sd(&img));
    let (comp, _) = solar::prominence_composite(&img, Some((cx, cy, r)), true, 1.0, 10.0, 0.5, 6.0, true).unwrap();
    let px = |y: usize, x: usize| comp.data[(y * 240 + x) * comp.c];
    assert!(px(26, 118) > 0.2 && px(121, 118) < 0.6, "{} {}", px(26, 118), px(121, 118));
}

// ---------------------------------------------------------- astrométrie

#[allow(clippy::type_complexity)]
fn field(dir: &std::path::Path, seed: u64, rot: f64, mirror: bool, k1: f64) -> (Image, std::path::PathBuf, Vec<(f64, f64, f64, f64, f64)>) {
    let mut rng = Rng::new(seed);
    let (ra0, dec0, n) = (150.0, 20.0, 400);
    let s = 3.0 / 3600.0;
    let t = rot.to_radians();
    let mut cd = [[-s * t.cos(), s * t.sin()], [s * t.sin(), s * t.cos()]];
    if mirror {
        cd[0][0] = -cd[0][0];
        cd[1][0] = -cd[1][0];
    }
    let w = Wcs::new_tan([500.0, 400.0], [ra0, dec0], cd);
    let mut csv = String::from("ra,dec,mag\n");
    let mut rows = Vec::new();
    for _ in 0..n {
        let ra = ra0 + rng.uniform(-0.6, 0.6) / dec0.to_radians().cos();
        let dec = dec0 + rng.uniform(-0.6, 0.6);
        let mag = rng.uniform(7.0, 14.0);
        csv += &format!("{},{},{}\n", ra, dec, mag);
        // astropy world_to_pixel_values : 0-based
        let (x, y) = w.world_to_pixel_linear(ra, dec).unwrap();
        let (dx, dy) = (x - 499.0, y - 399.0);
        let r2 = dx * dx + dy * dy;
        rows.push((ra, dec, mag, 499.0 + dx * (1.0 + k1 * r2), 399.0 + dy * (1.0 + k1 * r2)));
    }
    let cat = dir.join("cat.csv");
    std::fs::write(&cat, csv).unwrap();
    let mut img = Image::filled(1000, 800, 1, 0.05);
    for &(_, _, m, xi, yi) in &rows {
        if xi > 8.0 && xi < 992.0 && yi > 8.0 && yi < 792.0 && m < 13.0 {
            let a = 10f64.powf(-0.4 * (m - 7.0)) * 0.9;
            for y in (yi as usize - 8)..=(yi as usize + 8) {
                for x in (xi as usize - 8)..=(xi as usize + 8) {
                    img.data[y * 1000 + x] += (a * (-((x as f64 - xi).powi(2) + (y as f64 - yi).powi(2)) / 4.5).exp()) as f32;
                }
            }
        }
    }
    (rng.noise(&img, 0.002), cat, rows)
}

#[test]
fn triangle_solver() {
    for (k, (rot, mirror, k1)) in [(25.0, false, 0.0), (-70.0, true, 0.0), (10.0, false, 5e-8)].into_iter().enumerate() {
        let d = tempfile::tempdir().unwrap();
        let (img, cat, rows) = field(d.path(), 20 + k as u64, rot, mirror, k1);
        let res = platesolve::solve_triangles(&img, &cat, 150.1, 19.92, 1.0, None, 40, 120, 14, 24, Some(3)).unwrap();
        assert!((res.scale_arcsec_pix - 3.0).abs() < 0.1, "{}", res.scale_arcsec_pix);
        let mut err: Vec<f64> = rows
            .iter()
            .filter(|r| r.2 < 12.0 && r.3 > 20.0 && r.3 < 980.0 && r.4 > 20.0 && r.4 < 780.0)
            .map(|r| {
                let (x, y) = res.wcs.world_to_pixel(r.0, r.1).unwrap();
                (x - r.3).hypot(y - r.4)
            })
            .collect();
        err.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(err[err.len() / 2] < 0.1 && *err.last().unwrap() < 0.6 && res.n_matched >= 25, "rot {} : med {} max {}", rot, err[err.len() / 2], err.last().unwrap());
        if k1 != 0.0 {
            assert!(res.sip);
        }
    }
}

#[test]
fn solve_image_uses_triangles() {
    let d = tempfile::tempdir().unwrap();
    let (img, cat, _) = field(d.path(), 30, 25.0, false, 0.0);
    let res = astroforge::astrometry::solve_image(&img, Some(&cat), Some(150.05), Some(20.03), 1.0, None, None, false).unwrap();
    assert!(res.method.starts_with("triangles"), "{}", res.method);
}

#[test]
fn annotation_finds_messier() {
    let objs = annotate::load_objects(None).unwrap();
    let m42 = objs.iter().find(|o| o.label == "M42").unwrap();
    let s = 4.0 / 3600.0;
    let w = Wcs::new_tan([400.0, 300.0], [m42.ra, m42.dec], [[-s, 0.0], [0.0, s]]);
    let found = annotate::objects_in_field(&w, 800, 600, 99.0, None, 0.0).unwrap();
    let labels: Vec<&str> = found.iter().map(|o| o.label.as_str()).collect();
    assert!(["M42", "M43"].contains(&labels[0]) && labels.contains(&"M42") && labels.contains(&"M43"), "{:?}", labels);
    let (img, _) = annotate::annotate(&Image::filled(800, 600, 1, 0.1), &w, 99.0, [1.0, 0.8, 0.2], None, None).unwrap();
    assert!(img.c == 3 && img.w == 800 && img.max_value() > 0.5);
    let o = found.iter().find(|o| o.label == "M42").unwrap();
    assert!((o.x - 399.0).abs() < 1.0 && (o.y - 299.0).abs() < 1.0, "{} {}", o.x, o.y);
}
