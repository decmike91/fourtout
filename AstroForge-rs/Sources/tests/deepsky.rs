//! Ciel profond : calibration, empilement, alignement, drizzle, session,
//! bibliothèque de masters, analyse des brutes, normalisation, multi-nuits,
//! mosaïques, comète, empilement en direct (portage des tests Python).

mod common;
use astroforge::calibration::{self as C, DarkScale, MasterOptions};
use astroforge::geom::{self, Interp};
use astroforge::io;
use astroforge::ndimage::{self, Border};
use astroforge::registration::{apply_shift, phase_corr};
use astroforge::stacking::{self, StackParams};
use astroforge::{comet, drizzle, livestack, masters, mosaic, normalization, session, stats, subframes, Image};
use common::*;
use serde_json::{json, Map, Value};
use std::collections::HashMap;

fn obj(v: Value) -> Map<String, Value> {
    v.as_object().unwrap().clone()
}

#[test]
fn registration_phase_and_shift() {
    let r = sf(0.0, 0.0);
    let m = sf(4.0, -7.0);
    let (dx, dy, _) = phase_corr(&r, &m);
    assert!((dx + 4.0).abs() < 0.6 && (dy - 7.0).abs() < 0.6, "{} {}", dx, dy);
    let al = apply_shift(&m, dx, dy);
    assert!(mean_abs_diff(&inner(&al, 10), &inner(&r, 10)) < 0.02);
}

#[test]
fn rotation_estimate() {
    let r = sf(0.0, 0.0);
    let rot = geom::warp_affine(&r, &geom::rotation_matrix(64.0, 64.0, 1.5, 1.0), 128, 128, Interp::Cubic, Border::Reflect);
    let a = astroforge::registration::estimate_rotation(&r, &rot, 3.0, 0.25, "auto");
    assert!((a + 1.5).abs() < 0.2, "angle {}", a);
}

#[test]
fn all_stack_methods_and_weights() {
    let mut rng = Rng::new(1);
    let frames: Vec<Image> = (0..9).map(|_| rng.noise(&sf(0.0, 0.0), 0.03)).collect();
    let refs: Vec<&Image> = frames.iter().collect();
    for m in stacking::METHODS {
        let o = stacking::stack(&refs, m, &StackParams::default(), None).unwrap();
        assert!(o.w == 128 && o.all_finite(), "{}", m);
    }
    // pondération : bruit réduit
    let clean = Image::filled(64, 64, 1, 0.2);
    let mut a: Vec<Image> = (0..4).map(|_| rng.noise(&clean, 0.01)).collect();
    a.extend((0..2).map(|_| rng.noise(&clean, 0.05)));
    let refs: Vec<&Image> = a.iter().collect();
    let w = [1e4, 1e4, 1e4, 1e4, 400.0, 400.0];
    for m in ["Moyenne", "Sigma-clipping", "Winsorisé", "MAD clipping", "Linear fit"] {
        let sw = stats::std(&stacking::stack(&refs, m, &StackParams::default(), Some(&w)).unwrap().data);
        let s0 = stats::std(&stacking::stack(&refs, m, &StackParams::default(), None).unwrap().data);
        assert!(sw < s0, "{} {} {}", m, sw, s0);
    }
    assert!(stacking::stack(&refs, "Moyenne", &StackParams::default(), Some(&[1.0, 2.0])).is_err());
}

#[test]
fn drizzle_improves_sampling_and_rejects_trail() {
    let mut rng = Rng::new(2);
    let mut frames = Vec::new();
    let mut mats = Vec::new();
    for k in 0..8 {
        let (dx, dy) = (rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0));
        let mut f = rng.noise(&sf(dx, dy), 0.003);
        if k == 3 {
            for x in 5..120 {
                f.data[60 * 128 + x] += 0.5;
            }
        }
        frames.push(f);
        mats.push(drizzle::shift_matrix(-dx, -dy, 0.0, None));
    }
    let aligned: Vec<Image> = frames.iter().zip(&mats).map(|(f, m)| geom::warp(f, m)).collect();
    let reference = stacking::stack_owned(&aligned, "Sigma-clipping", 3.0, None).unwrap();
    let (out, cov) = drizzle::drizzle_stack(&frames, &mats, 2, 0.7, None, Some(&reference), Some(3.0), None).unwrap();
    assert_eq!((out.w, out.h), (256, 256));
    assert!(cov.max_value() <= 1.0 && out.all_finite());
    // trace rejetée
    let row: f64 = (20..230).map(|x| out.data[121 * 256 + x] as f64).sum::<f64>() / 210.0;
    let truth = geom::resize(&sf(0.0, 0.0), 256, 256, Interp::Cubic);
    let trow: f64 = (20..230).map(|x| truth.data[121 * 256 + x] as f64).sum::<f64>() / 210.0;
    assert!(row < trow + 0.03, "{} {}", row, trow);
}

fn session_files(dir: &std::path::Path, rng: &mut Rng) {
    let (h, w) = (32, 32);
    let f = |v: f32| Image::filled(w, h, 1, v);
    let rnd = |rng: &mut Rng, base: f32| Image::from_vec(w, h, 1, (0..w * h).map(|_| base + 0.02 * rng.uniform(0.0, 1.0) as f32).collect());
    for i in 0..3 {
        write_fits(dir, &format!("nuit/L_{}.fits", i), &rnd(rng, 0.4), &[("IMAGETYP", "Light Frame".into()), ("EXPTIME", 300.0.into()), ("FILTER", "L".into()), ("GAIN", 100i64.into()), ("CCD-TEMP", (-10.0).into()), ("OBJECT", "M31".into())]);
        write_fits(dir, &format!("nuit/Lr_{}.fits", i), &rnd(rng, 0.3), &[("IMAGETYP", "LIGHT".into()), ("EXPTIME", 300.0.into()), ("FILTER", "R".into()), ("GAIN", 100i64.into()), ("CCD-TEMP", (-10.0).into()), ("OBJECT", "M31".into())]);
        write_fits(dir, &format!("nuit/D_{}.fits", i), &f(0.12), &[("IMAGETYP", "Dark Frame".into()), ("EXPTIME", 300.0.into()), ("GAIN", 100i64.into()), ("CCD-TEMP", (-10.2).into())]);
        write_fits(dir, &format!("nuit/D60_{}.fits", i), &f(0.104), &[("IMAGETYP", "Dark Frame".into()), ("EXPTIME", 60.0.into()), ("GAIN", 100i64.into()), ("CCD-TEMP", (-10.0).into())]);
        write_fits(dir, &format!("nuit/Dg_{}.fits", i), &f(0.5), &[("IMAGETYP", "Dark Frame".into()), ("EXPTIME", 300.0.into()), ("GAIN", 200i64.into()), ("CCD-TEMP", (-10.0).into())]);
        write_fits(dir, &format!("nuit/F_{}.fits", i), &f(0.5), &[("IMAGETYP", "Flat Field".into()), ("EXPTIME", 2.0.into()), ("FILTER", "L".into()), ("GAIN", 100i64.into())]);
        write_fits(dir, &format!("nuit/B_{}.fits", i), &f(0.1), &[("IMAGETYP", "Bias Frame".into()), ("EXPTIME", 0.0.into()), ("GAIN", 100i64.into())]);
        write_fits(dir, &format!("nuit/flats_R/sans_entete_{}.fits", i), &f(0.4), &[]);
    }
}

#[test]
fn session_classify_and_plan() {
    let mut h = io::Header::new();
    h.set("IMAGETYP", "Flat Field");
    assert_eq!(session::classify(std::path::Path::new("x.fits"), &h), ("flat".to_string(), "en-tête".to_string()));
    h.set("IMAGETYP", "Offset");
    assert_eq!(session::classify(std::path::Path::new("x.fits"), &h).0, "bias");
    let e = io::Header::new();
    assert_eq!(session::classify(std::path::Path::new("/d/DarkFlats/a.fits"), &e).0, "darkflat");
    assert_eq!(session::classify(std::path::Path::new("/d/flats_R/a.fits"), &e).0, "flat");
    assert_eq!(session::classify(std::path::Path::new("/d/lights/a.cr2"), &e).0, "light");

    let d = tempfile::tempdir().unwrap();
    let mut rng = Rng::new(3);
    session_files(d.path(), &mut rng);
    let plan = session::plan_session(&d.path().join("nuit"), true, 3.0);
    let names: HashMap<String, &session::LightGroup> = plan.groups.iter().map(|g| (g.key.filter.clone(), g)).collect();
    assert_eq!(names.len(), 2);
    let gl = names["L"];
    assert_eq!((gl.lights.len(), gl.biases.len(), gl.flats.len()), (3, 3, 3));
    assert!(gl.darks.len() == 3 && gl.darks.iter().all(|d| d.exptime == Some(300.0) && d.gain == Some(100.0)));
    assert!(gl.summary().contains("M31"));
    let gr = names["R"];
    assert_eq!(gr.flats.len(), 3, "flats R déduits du nom de dossier");
}

#[test]
fn dark_scaling_when_exposure_differs() {
    let d = tempfile::tempdir().unwrap();
    for i in 0..3 {
        write_fits(d.path(), &format!("s/L{}.fits", i), &Image::filled(16, 16, 1, 0.5), &[("IMAGETYP", "Light".into()), ("EXPTIME", 120.0.into()), ("GAIN", 0i64.into())]);
        write_fits(d.path(), &format!("s/D{}.fits", i), &Image::filled(16, 16, 1, 0.11), &[("IMAGETYP", "Dark".into()), ("EXPTIME", 60.0.into()), ("GAIN", 0i64.into())]);
        write_fits(d.path(), &format!("s/B{}.fits", i), &Image::filled(16, 16, 1, 0.10), &[("IMAGETYP", "Bias".into()), ("GAIN", 0i64.into())]);
    }
    let g = session::plan_session(&d.path().join("s"), true, 3.0).groups.remove(0);
    assert!((g.dark_scale - 2.0).abs() < 1e-9 && !g.warnings.is_empty());
    let mut paths = HashMap::new();
    paths.insert("bias".to_string(), g.paths("bias"));
    paths.insert("dark".to_string(), g.paths("dark"));
    let m = C::build_masters(&MasterOptions { paths: Some(paths), sigma: 3.0, ..Default::default() }).unwrap();
    assert_eq!(m.meta["dark"]["thermal"], json!(true));
    assert_eq!(m.meta["dark"]["exptime"].as_f64(), Some(60.0));
    let cal = C::calibrate_paths(&g.paths("light"), &m, None, DarkScale::Auto, false, None, None, "RGB").unwrap();
    assert!((cal[0].mean() - 0.38).abs() < 2e-3, "{}", cal[0].mean());
}

#[test]
fn master_library_roundtrip() {
    let d = tempfile::tempdir().unwrap();
    let lib = d.path().join("lib");
    let dark = Image::filled(30, 20, 1, 0.01);
    masters::save_master(&dark, "dark", &obj(json!({"exptime": 60.0, "gain": 100, "temp": -10.0, "thermal": true})), Some(&lib)).unwrap();
    masters::save_master(&dark.scale(2.0), "dark", &obj(json!({"exptime": 300.0, "gain": 100, "temp": -10.0, "thermal": true})), Some(&lib)).unwrap();
    masters::save_master(&dark, "dark", &obj(json!({"exptime": 300.0, "gain": 200, "temp": -10.0})), Some(&lib)).unwrap();
    masters::save_master(&Image::filled(30, 20, 1, 0.1), "bias", &obj(json!({"gain": 100})), Some(&lib)).unwrap();
    let light = obj(json!({"exptime": 300.0, "gain": 100, "temp": -9.0, "width": 30, "height": 20, "binning": 1}));
    let (e, k) = masters::find_master("dark", &light, Some(&lib), 3.0, true);
    let e = e.unwrap();
    assert!(e["exptime"].as_f64() == Some(300.0) && e["gain"].as_f64() == Some(100.0) && k == 1.0);
    let mut l2 = light.clone();
    l2.insert("exptime".into(), json!(120.0));
    let (e2, k2) = masters::find_master("dark", &l2, Some(&lib), 3.0, true);
    let e2 = e2.unwrap();
    assert!((k2 * e2["exptime"].as_f64().unwrap() - 120.0).abs() < 1e-6);
    let mut l3 = light.clone();
    l3.insert("temp".into(), json!(5.0));
    assert!(masters::find_master("dark", &l3, Some(&lib), 3.0, true).0.is_none());
    let lm = masters::load_master(&e, Some(&lib)).unwrap();
    assert!((lm.data[0] - 0.02).abs() < 1e-7);
    let m = C::build_masters(&MasterOptions { light_meta: Some(light), use_library: true, library: Some(lib.clone()), sigma: 3.0, ..Default::default() }).unwrap();
    assert_eq!(m.sources.get("dark").map(|s| s.as_str()), Some("bibliothèque"));
    assert_eq!(m.sources.get("bias").map(|s| s.as_str()), Some("bibliothèque"));
}

#[test]
fn save_to_library() {
    let d = tempfile::tempdir().unwrap();
    let lib = d.path().join("lib2");
    let paths: Vec<_> = (0..3).map(|i| write_fits(d.path(), &format!("b/B{}.fits", i), &Image::filled(8, 8, 1, 0.1), &[("IMAGETYP", "Bias".into()), ("GAIN", 50i64.into())])).collect();
    let mut p = HashMap::new();
    p.insert("bias".to_string(), paths);
    C::build_masters(&MasterOptions { paths: Some(p), save_to_library: true, library: Some(lib.clone()), sigma: 3.0, ..Default::default() }).unwrap();
    let idx = masters::load_index(Some(&lib));
    assert_eq!(idx.len(), 1);
    assert_eq!(idx[0]["kind"], json!("bias"));
    assert_eq!(idx[0]["gain"].as_f64(), Some(50.0));
    assert_eq!(idx[0]["ncombine"].as_i64(), Some(3));
}

fn frame(rng: &mut Rng, fwhm: f64, ecc_ratio: f64, bg: f32, amp: f64, noise: f64, seed: u64) -> Image {
    let mut r = Rng::new(seed);
    let (h, w) = (200usize, 200usize);
    let sx = fwhm / 2.3548;
    let sy = sx / ecc_ratio;
    let stars: Vec<(f64, f64, f64)> = (0..50).map(|_| (r.uniform(12.0, w as f64 - 12.0), r.uniform(12.0, h as f64 - 12.0), amp * r.uniform(0.1, 0.6))).collect();
    let img = Image::from_fn(w, h, |x, y| {
        let mut v = bg as f64;
        for &(px, py, a) in &stars {
            v += a * (-((x as f64 - px).powi(2) / (2.0 * sx * sx) + (y as f64 - py).powi(2) / (2.0 * sy * sy))).exp();
        }
        v as f32
    });
    rng.noise(&img, noise)
}

#[test]
fn subframes_star_shape_and_flags() {
    for (sx, sy) in [(1.0, 1.0), (2.0, 2.0), (2.5, 1.2)] {
        let img = Image::from_fn(64, 64, |x, y| (0.05 + 0.8 * (-((x as f64 - 31.6).powi(2) / (2.0 * sx * sx) + (y as f64 - 32.4).powi(2) / (2.0 * sy * sy))).exp()) as f32);
        let (fwhm, ecc, cx, cy) = subframes::star_shape(&img, 32.0, 32.0, 0.05, 1.5, 20).unwrap();
        assert!((fwhm - 2.3548 * ((sx * sx + sy * sy) / 2.0f64).sqrt()).abs() < 0.03);
        assert!((ecc - (1.0 - (f64::min(sx, sy) / f64::max(sx, sy)).powi(2)).sqrt()).abs() < 0.01);
        assert!((cx - 31.6).abs() < 0.01 && (cy - 32.4).abs() < 0.01);
    }
    let d = tempfile::tempdir().unwrap();
    let mut rng = Rng::new(7);
    let mut frames: Vec<Image> = (0..6).map(|_| frame(&mut rng, 2.5, 1.0, 0.05, 1.0, 0.003, 1)).collect();
    frames.push(frame(&mut rng, 5.0, 1.0, 0.05, 1.0, 0.003, 1));
    frames.push(frame(&mut rng, 2.5, 1.0, 0.12, 0.25, 0.003, 1));
    frames.push(frame(&mut rng, 2.5, 2.5, 0.05, 1.0, 0.003, 1));
    let paths: Vec<_> = frames
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let p = d.path().join(format!("f{}.tif", i));
            io::save(&p, f).unwrap();
            p
        })
        .collect();
    let mut rows = subframes::analyze_paths(&paths, None, None);
    subframes::flag_outliers_default(&mut rows);
    let keep: Vec<bool> = rows.iter().map(|r| r.keep).collect();
    assert_eq!(keep, [true, true, true, true, true, true, false, false, false], "{:?}", rows.iter().map(|r| r.reasons.clone()).collect::<Vec<_>>());
    assert!((rows[0].fwhm - 2.5).abs() < 0.3 && rows[8].ecc > 0.65);
    let w = subframes::compute_weights(&rows, "Mixte");
    assert_eq!(&w[6..], &[0.0, 0.0, 0.0]);
    assert!((w[..6].iter().sum::<f64>() / 6.0 - 1.0).abs() < 1e-9);
    assert!(subframes::report(&rows).contains("rejeter"));
}

#[test]
fn local_normalization_removes_night_gradient() {
    let mut rng = Rng::new(5);
    let r = rng.noise(&star_field(256, 256, 0.0, 0.0, 25, 5, 3.0, 0.05), 0.002);
    let other = Image::from_fn(256, 256, |x, y| r.data[y * 256 + x] * 0.8 + 0.03 + 0.04 * x as f32 / 256.0);
    let ln = normalization::local_normalize(&other, &r, 64).unwrap();
    let gn = normalization::global_normalize(&other, &r);
    let lowpass = |a: &Image| inner(&ndimage::gaussian(&a.zip_map(&r, |x, y| x - y), 8.0).map(|v| v.abs()), 16).mean();
    let (rl, rg) = (lowpass(&ln), lowpass(&gn));
    assert!(rl < 0.3 * rg && rl < 0.002, "{} {}", rl, rg);
}

#[test]
fn multi_night_stack() {
    let d = tempfile::tempdir().unwrap();
    let mut rng = Rng::new(9);
    let truth = star_field(96, 96, 0.0, 0.0, 25, 9, 3.0, 0.05);
    for (night, (gain, offs)) in [(1.0f32, 0.0f32), (0.7, 0.04)].iter().enumerate() {
        for i in 0..4 {
            let (dx, dy) = (rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0));
            let img = apply_shift(&truth, dx, dy).map(|v| v * gain + offs + 0.01);
            let img = rng.noise(&img, 0.003);
            write_fits(d.path(), &format!("n{}/light_{}.fits", night, i), &img, &[("IMAGETYP", "Light".into()), ("EXPTIME", 60.0.into())]);
            write_fits(d.path(), &format!("n{}/dark_{}.fits", night, i), &Image::filled(96, 96, 1, 0.01), &[("IMAGETYP", "Dark".into()), ("EXPTIME", 60.0.into())]);
        }
    }
    let sess: Vec<_> = (0..2).map(|k| astroforge::multisession::session_from_folder(&d.path().join(format!("n{}", k))).unwrap()).collect();
    let (out, info) = astroforge::multisession::stack_sessions(&sess, "Sigma-clipping", 3.0, "Locale", None, "RGB", "Aucune", None, None).unwrap();
    assert_eq!(info.per_night, vec![4, 4]);
    assert_eq!((out.w, out.h), (96, 96));
    let (dx, dy, _) = phase_corr(&truth, &out);
    let t = if dx.hypot(dy) > 0.5 { apply_shift(&truth, -dx, -dy) } else { truth.clone() };
    assert!(corr(&inner(&out, 8).data, &inner(&t, 8).data) > 0.98);
}

fn sky(rng: &mut Rng, h: usize, w: usize) -> Image {
    let stars: Vec<(f64, f64, f64)> = (0..220).map(|_| (rng.uniform(5.0, w as f64 - 5.0), rng.uniform(5.0, h as f64 - 5.0), rng.uniform(0.1, 0.7))).collect();
    Image::from_fn(w, h, |x, y| {
        let (xf, yf) = (x as f64, y as f64);
        let mut v = 0.05 + 0.2 * (-((xf - w as f64 / 2.0).powi(2) + (yf - h as f64 / 2.0).powi(2)) / (2.0 * 80.0 * 80.0)).exp();
        for &(sx, sy, a) in &stars {
            let d2 = (xf - sx).powi(2) + (yf - sy).powi(2);
            if d2 < 60.0 {
                v += a * (-d2 / (2.0 * 1.2 * 1.2)).exp();
            }
        }
        v as f32
    })
}

#[test]
fn mosaic_wcs_and_features() {
    let mut rng = Rng::new(11);
    let s = sky(&mut rng, 300, 420);
    let sc = 2.0 / 3600.0;
    let mut panels = Vec::new();
    let mut wcss = Vec::new();
    for (k, (x0, y0)) in [(0usize, 0usize), (170, 0), (0, 120), (170, 120)].iter().enumerate() {
        panels.push(s.crop(*x0, *y0, 250, 180).add_scalar(0.01 * k as f32));
        wcss.push(astroforge::wcs::Wcs::new_tan([211.0 - *x0 as f64, 151.0 - *y0 as f64], [83.8, -5.4], [[-sc, 0.0], [0.0, sc]]));
    }
    let (out, wout, cov) = mosaic::mosaic_wcs(&panels, &wcss, None, 64.0).unwrap();
    assert!(cov.iter().filter(|&&b| b).count() as f64 / cov.len() as f64 > 0.95);
    let ref0 = astroforge::wcs::Wcs::new_tan([211.0, 151.0], [83.8, -5.4], [[-sc, 0.0], [0.0, sc]]);
    let (ra, dec) = ref0.pixel_to_world_linear(0.0, 0.0);
    let (px, py) = wout.world_to_pixel_linear(ra, dec).unwrap();
    let (x0, y0) = (px.round() as usize, py.round() as usize);
    let part = out.crop(x0 + 10, y0 + 10, 400, 280);
    assert!(corr(&part.data, &s.crop(10, 10, 400, 280).data) > 0.99);
    let (_o2, _c2, mats) = mosaic::mosaic_features(&panels, 48.0).unwrap();
    let offs: Vec<(i64, i64)> = mats.iter().map(|m| (m[0][2].round() as i64, m[1][2].round() as i64)).collect();
    assert_eq!(offs, vec![(0, 0), (170, 0), (0, 120), (170, 120)]);
}

#[test]
fn comet_stack_separates_stars_and_comet() {
    let mut rng = Rng::new(4);
    let n = 8;
    let stars = star_field(128, 128, 0.0, 0.0, 25, 4, 3.0, 0.05);
    let mut frames = Vec::new();
    let mut pos = Vec::new();
    for i in 0..n {
        let (cx, cy) = (30.0 + 8.0 * i as f64, 40.0 + 5.0 * i as f64);
        let f = Image::from_fn(128, 128, |x, y| stars.data[y * 128 + x] + (0.5 * (-((x as f64 - cx).powi(2) + (y as f64 - cy).powi(2)) / 18.0).exp()) as f32);
        frames.push(rng.noise(&f, 0.002));
        pos.push((cx + rng.normal(0.0, 0.5), cy + rng.normal(0.0, 0.5)));
    }
    let interp = comet::interpolate_positions(n, pos[0], pos[n - 1], None);
    let res = comet::comet_stack(&frames, &interp, "Sigma-clipping", 2.0, comet::COMPOSITE_MODES[0], true).unwrap();
    assert!(mean_abs_diff(&res.stars, &stars) < 0.01);
    let mid = res.positions[n / 2];
    let (cy, cx) = (mid.1.round() as usize, mid.0.round() as usize);
    let med = stats::median(&res.comet.data);
    assert!(res.comet.data[cy * 128 + cx] > 0.45 + med);
    assert!(res.composite.data[cy * 128 + cx] >= res.comet.data[cy * 128 + cx] - 1e-6);
    assert_eq!(comet::parse_times(&["2026-01-01T00:00:00".into(), "2026-01-01T00:01:00".into()]), Some(vec![0.0, 60.0]));
}

#[test]
fn live_stacker_rejects_bad_and_trails() {
    let d = tempfile::tempdir().unwrap();
    let mut rng = Rng::new(2);
    let base = star_field(128, 128, 0.0, 0.0, 25, 2, 3.0, 0.05);
    for i in 0..8 {
        let (sx, sy) = (rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0));
        let mut img = rng.noise(&apply_shift(&base, sx, sy), 0.003);
        if i == 6 {
            for y in 60..62 {
                for x in 0..128 {
                    img.data[y * 128 + x] += 0.5;
                }
            }
        }
        io::save(d.path().join(format!("f{:02}.tif", i)), &img.clip01()).unwrap();
    }
    io::save(d.path().join("f99_nuage.tif"), &Image::filled(128, 128, 1, 0.3)).unwrap();
    let mut st = livestack::LiveStacker::new(Default::default(), None, 3.0);
    livestack::run_folder(d.path(), &mut st, 0.0, 2.0, &mut |_| {});
    assert_eq!((st.accepted.len(), st.rejected.len()), (8, 1), "{:?}", st.rejected);
    let out = st.image().unwrap();
    let m = |im: &Image| (60..62).flat_map(|y| (20..108).map(move |x| (y, x))).map(|(y, x)| im.data[y * 128 + x] as f64).sum::<f64>() / 176.0;
    assert!(m(&out) < m(&base) + 0.03);
}

#[test]
fn defects_and_hot_pixels() {
    let mut img = Image::filled(40, 30, 1, 0.2);
    for y in 0..30 {
        img.data[y * 40 + 17] = 0.9;
    }
    img.data[10 * 40 + 5] = 1.0;
    let mut dark = Image::filled(40, 30, 1, 0.01);
    for y in 0..30 {
        dark.data[y * 40 + 17] = 0.3;
    }
    dark.data[10 * 40 + 5] = 0.5;
    let (out, info) = astroforge::defects::auto_correct(&img, Some(&dark), None, None, 5.0, true).unwrap();
    assert!(info.bad_columns >= 1 && info.bad_pixels >= 1);
    assert!((out.data[10 * 40 + 5] - 0.2).abs() < 1e-3 && (out.data[5 * 40 + 17] - 0.2).abs() < 1e-3);
}

#[test]
fn calibration_and_cosmetic() {
    let mut rng = Rng::new(1);
    let darks: Vec<Image> = (0..5).map(|_| rng.noise(&Image::filled(128, 128, 1, 0.07), 0.002)).collect();
    let flats: Vec<Image> = (0..5).map(|_| Image::from_fn(128, 128, |x, _| 0.5 * (1.0 + 0.1 * x as f32 / 128.0))).collect();
    let mb = C::make_master_bias(&darks, 3.0).unwrap();
    let md = C::make_master_dark(&darks, Some(&mb), 3.0).unwrap();
    let mf = C::make_master_flat(&flats, Some(&mb), None, 3.0).unwrap();
    let light = sf(0.0, 0.0).scale(0.4).add_scalar(0.05).zip_map(&md, |a, b| a + b);
    let cal = C::calibrate(&light, Some(&mb), Some(&md), Some(&mf), 1.0, true, 5.0, false).unwrap();
    assert!(cal.same_shape(&light) && cal.min_value() >= 0.0);
}
