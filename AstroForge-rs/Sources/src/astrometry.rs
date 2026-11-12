//! Astrométrie locale : catalogue CSV (ra, dec, mag[, bp, rp]), plate
//! solving (astrometry.net « solve-field » s'il est installé, sinon solveurs
//! internes : triangles + SIP, puis paires d'étoiles), photométrie
//! d'ouverture et photométrie différentielle.

use crate::image::Image;
use crate::io;
use crate::starmatch::{self, Pt};
use crate::stars;
use crate::stats;
use crate::wcs::{self, Wcs};
use anyhow::{anyhow, bail, Context, Result};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default)]
pub struct Catalog {
    pub ra: Vec<f64>,
    pub dec: Vec<f64>,
    pub mag: Option<Vec<f64>>,
    pub bp: Option<Vec<f64>>,
    pub rp: Option<Vec<f64>>,
}

impl Catalog {
    pub fn len(&self) -> usize {
        self.ra.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ra.is_empty()
    }
    pub fn mags_or(&self, default: f64) -> Vec<f64> {
        self.mag.clone().unwrap_or_else(|| vec![default; self.len()])
    }
}

fn split_csv(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut q = false;
    for ch in line.chars() {
        match ch {
            '"' => q = !q,
            ',' if !q => {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    out.push(cur.trim().to_string());
    out
}

/// Charge un catalogue CSV (colonnes ra, dec en degrés ; mag, bp, rp facultatives).
pub fn load_catalog(path: &Path) -> Result<Catalog> {
    let txt = std::fs::read_to_string(path).with_context(|| format!("Catalogue illisible : {}", path.display()))?;
    let mut lines = txt.lines().filter(|l| !l.trim().is_empty() && !l.starts_with('#'));
    let head: Vec<String> = split_csv(lines.next().ok_or_else(|| anyhow!("Catalogue vide : {}", path.display()))?).iter().map(|s| s.to_lowercase()).collect();
    let col = |n: &str| head.iter().position(|h| h == n);
    let (Some(ira), Some(idec)) = (col("ra"), col("dec")) else {
        let missing = if col("ra").is_none() { "ra" } else { "dec" };
        bail!("Catalogue sans colonne « {} » : {}", missing, path.display());
    };
    let (imag, ibp, irp) = (col("mag"), col("bp"), col("rp"));
    let mut c = Catalog { mag: imag.map(|_| vec![]), bp: ibp.map(|_| vec![]), rp: irp.map(|_| vec![]), ..Default::default() };
    let num = |v: Option<&String>| v.and_then(|s| s.parse::<f64>().ok()).unwrap_or(f64::NAN);
    for l in lines {
        let f = split_csv(l);
        let (ra, dec) = (num(f.get(ira)), num(f.get(idec)));
        if !ra.is_finite() || !dec.is_finite() {
            continue;
        }
        c.ra.push(ra);
        c.dec.push(dec);
        if let (Some(i), Some(v)) = (imag, c.mag.as_mut()) {
            v.push(num(f.get(i)));
        }
        if let (Some(i), Some(v)) = (ibp, c.bp.as_mut()) {
            v.push(num(f.get(i)));
        }
        if let (Some(i), Some(v)) = (irp, c.rp.as_mut()) {
            v.push(num(f.get(i)));
        }
    }
    Ok(c)
}

/// Plan tangent au centre -> secondes d'arc (ξ vers l'est, η vers le nord).
pub fn tangent_xy(ra: &[f64], dec: &[f64], ra0: f64, dec0: f64) -> Vec<Pt> {
    ra.iter()
        .zip(dec)
        .map(|(&r, &d)| wcs::tan_project(r, d, ra0, dec0).map(|(x, y)| (x * 3600.0, y * 3600.0)).unwrap_or((f64::NAN, f64::NAN)))
        .collect()
}

#[derive(Clone, Debug)]
pub struct Solution {
    pub wcs: Wcs,
    pub scale_arcsec_pix: f64,
    pub n_matched: i64,
    pub rms_arcsec: f64,
    pub center_ra: f64,
    pub center_dec: f64,
    pub method: String,
    pub sip: bool,
}

fn finish(wcs: Wcs, w: usize, h: usize, scale: f64, n: i64, rms: f64, method: &str, sip: bool) -> Solution {
    let (ra, dec) = wcs.pixel_to_world(w as f64 / 2.0, h as f64 / 2.0);
    Solution { wcs, scale_arcsec_pix: scale, n_matched: n, rms_arcsec: rms, center_ra: ra, center_dec: dec, method: method.into(), sip }
}

/// Étoiles du catalogue dans le rayon : (ra, dec, mag).
pub fn cone(cat: &Catalog, ra0: f64, dec0: f64, radius: f64, default_mag: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mags = cat.mags_or(default_mag);
    let mut out = (vec![], vec![], vec![]);
    for i in 0..cat.len() {
        if wcs::separation(ra0, dec0, cat.ra[i], cat.dec[i]) <= radius {
            out.0.push(cat.ra[i]);
            out.1.push(cat.dec[i]);
            out.2.push(mags[i]);
        }
    }
    out
}

/// Solveur interne par paires d'étoiles brillantes (hypothèses directe et
/// miroir), affinement Procrustes, WCS TAN.
#[allow(clippy::too_many_arguments)]
pub fn solve_internal(img: &Image, catalog: &Path, center_ra: f64, center_dec: f64, radius_deg: f64, focal_mm: Option<f64>, pixel_um: Option<f64>, max_stars: usize) -> Result<Solution> {
    let (w, h) = (img.w, img.h);
    let st = stars::detect_stars(img, max_stars);
    if st.len() < 4 {
        bail!("Moins de 4 étoiles détectées — image non résoluble");
    }
    let cat = load_catalog(catalog)?;
    let (ra, dec, mags) = cone(&cat, center_ra, center_dec, radius_deg, 0.0);
    if ra.len() < 4 {
        bail!("Moins de 4 étoiles de catalogue dans le rayon — élargissez radius_deg ou vérifiez le centre");
    }
    let sky_all = tangent_xy(&ra, &dec, center_ra, center_dec);
    let order = stats::argsort_asc(&mags.iter().map(|m| if m.is_nan() { f64::INFINITY } else { *m }).collect::<Vec<_>>());
    let sky: Vec<Pt> = order.iter().take(40).map(|&i| sky_all[i]).collect();
    let diag = ((w * w + h * h) as f64).sqrt();
    let s0 = match (focal_mm, pixel_um) {
        (Some(f), Some(p)) if f > 0.0 && p > 0.0 => 206.265 * p / f,
        _ => radius_deg * 3600.0 / (diag / 2.0) * 0.9,
    };
    type Hyp = (usize, starmatch::Sim);
    let hypothesis = |pi: Pt, pj: Pt, si: Pt, sj: Pt, flip: bool| -> Option<Hyp> {
        let vp = if flip { (pj.0 - pi.0, -(pj.1 - pi.1)) } else { (pj.0 - pi.0, pj.1 - pi.1) };
        let vs = (sj.0 - si.0, sj.1 - si.1);
        let (lp, ls) = (vp.0.hypot(vp.1), vs.0.hypot(vs.1));
        if lp < 1e-6 || ls < 1e-6 {
            return None;
        }
        let sc = ls / lp;
        let r = sc / s0.max(1e-9);
        if !(0.2 < r && r < 5.0) {
            return None;
        }
        let ang = vs.1.atan2(vs.0) - vp.1.atan2(vp.0);
        let (c, s) = (ang.cos(), ang.sin());
        let rm = [[c * sc, -s * sc], [s * sc, c * sc]];
        let m = if flip { [[rm[0][0], -rm[0][1]], [rm[1][0], -rm[1][1]]] } else { rm };
        let t = [si.0 - (m[0][0] * pi.0 + m[0][1] * pi.1), si.1 - (m[1][0] * pi.0 + m[1][1] * pi.1)];
        let sim = starmatch::Sim { m, t };
        let tol = (3.0 * sc).max(1.5);
        let mut n = 0;
        for &p in &st {
            let q = sim.apply(p);
            if sky.iter().any(|s| (q.0 - s.0).hypot(q.1 - s.1) < tol) {
                n += 1;
            }
        }
        Some((n, sim))
    };
    let (ni, ns) = (5.min(st.len()), 12.min(sky.len()));
    let mut best: Option<Hyp> = None;
    for flip in [false, true] {
        for i in 0..ni {
            for j in i + 1..ni {
                for a in 0..ns {
                    for b in a + 1..ns {
                        if let Some(r) = hypothesis(st[i], st[j], sky[a], sky[b], flip) {
                            if best.as_ref().map(|x| r.0 > x.0).unwrap_or(true) {
                                best = Some(r);
                            }
                        }
                    }
                }
            }
        }
    }
    let Some((nb, mut sim)) = best else { bail!("Plate solving interne échoué : pas d'appariement fiable (vérifiez RA/Dec/rayon, ou installez astrometry.net)") };
    if nb < 4 {
        bail!("Plate solving interne échoué : pas d'appariement fiable (vérifiez RA/Dec/rayon, ou installez astrometry.net)");
    }
    let nearest = |sim: &starmatch::Sim| -> (Vec<usize>, Vec<f64>) {
        let mut idx = vec![0; st.len()];
        let mut dist = vec![f64::INFINITY; st.len()];
        for (k, &p) in st.iter().enumerate() {
            let q = sim.apply(p);
            for (j, s) in sky.iter().enumerate() {
                let d = (q.0 - s.0).hypot(q.1 - s.1);
                if d < dist[k] {
                    dist[k] = d;
                    idx[k] = j;
                }
            }
        }
        (idx, dist)
    };
    let mut scale = sim.scale();
    let (mut idx, mut dist) = nearest(&sim);
    let mut inl: Vec<bool> = dist.iter().map(|&d| d < (3.0 * scale).max(1.5)).collect();
    for _ in 0..3 {
        let sel: Vec<usize> = (0..st.len()).filter(|&k| inl[k]).collect();
        if sel.len() < 3 {
            break;
        }
        let p: Vec<Pt> = sel.iter().map(|&k| st[k]).collect();
        let s: Vec<Pt> = sel.iter().map(|&k| sky[idx[k]]).collect();
        let s2 = starmatch::umeyama(&p, &s, true);
        let (i2, d2) = nearest(&s2);
        let sc2 = s2.scale();
        let mut inl2: Vec<bool> = d2.iter().map(|&d| d < (2.0 * sc2).max(1.5)).collect();
        if (sc2 / scale.max(1e-9) - 1.0).abs() > 0.2 {
            inl2 = vec![false; st.len()];
        }
        if inl2.iter().filter(|&&b| b).count() < inl.iter().filter(|&&b| b).count() {
            break;
        }
        sim = s2;
        idx = i2;
        dist = d2;
        inl = inl2;
        scale = sc2;
    }
    let sel: Vec<usize> = (0..st.len()).filter(|&k| inl[k]).collect();
    let rms = if sel.is_empty() { -1.0 } else { (sel.iter().map(|&k| dist[k].powi(2)).sum::<f64>() / sel.len() as f64).sqrt() };
    let wcs = wcs_from_sim(&sim, center_ra, center_dec);
    Ok(finish(wcs, w, h, scale, sel.len() as i64, rms, "interne", false))
}

/// WCS TAN à partir d'une similitude pixel -> secondes d'arc du plan tangent.
pub fn wcs_from_sim(sim: &starmatch::Sim, ra0: f64, dec0: f64) -> Wcs {
    let m = sim.m;
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let inv = [[m[1][1] / det, -m[0][1] / det], [-m[1][0] / det, m[0][0] / det]];
    let pc = (-(inv[0][0] * sim.t[0] + inv[0][1] * sim.t[1]), -(inv[1][0] * sim.t[0] + inv[1][1] * sim.t[1]));
    Wcs::new_tan([pc.0 + 1.0, pc.1 + 1.0], [ra0, dec0], [[m[0][0] / 3600.0, m[0][1] / 3600.0], [m[1][0] / 3600.0, m[1][1] / 3600.0]])
}

pub fn which(exe: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for d in std::env::split_paths(&path) {
        let p = d.join(exe);
        if p.is_file() {
            return Some(p);
        }
        #[cfg(windows)]
        {
            let p = d.join(format!("{}.exe", exe));
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

/// astrometry.net local (binaire solve-field + index files).
pub fn solve_with_solvefield(fits_path: &Path, work_dir: &Path, hint: Option<(f64, f64)>, radius_deg: f64) -> Result<Solution> {
    let exe = which("solve-field").ok_or_else(|| anyhow!("solve-field (astrometry.net) introuvable — installez-le ou utilisez le solveur interne avec un catalogue local"))?;
    let mut cmd = std::process::Command::new(exe);
    cmd.args(["--no-plots", "--overwrite", "--downsample", "2"]);
    if let Some((ra, dec)) = hint {
        cmd.args(["--ra", &ra.to_string(), "--dec", &dec.to_string(), "--radius", &radius_deg.to_string()]);
    }
    cmd.arg("-D").arg(work_dir).arg(fits_path);
    let out = cmd.output().context("Impossible de lancer solve-field")?;
    if !out.status.success() {
        bail!("solve-field a échoué : {}", String::from_utf8_lossy(&out.stderr).chars().rev().take(500).collect::<String>().chars().rev().collect::<String>());
    }
    let stem = fits_path.file_stem().unwrap().to_string_lossy().to_string();
    let mut newp = work_dir.join(format!("{}.new", stem));
    if !newp.is_file() {
        let cand = std::fs::read_dir(work_dir)?.filter_map(|e| e.ok().map(|e| e.path())).find(|p| p.extension().map(|e| e == "new").unwrap_or(false));
        newp = cand.ok_or_else(|| anyhow!("solve-field n'a pas résolu le champ"))?;
    }
    let h = io::fits::read_header(&newp)?;
    let wcs = Wcs::from_header(&h).ok_or_else(|| anyhow!("solve-field : WCS illisible"))?;
    let w = h.get("NAXIS1").and_then(|v| v.as_i64()).unwrap_or(0) as usize;
    let hh = h.get("NAXIS2").and_then(|v| v.as_i64()).unwrap_or(0) as usize;
    let scale = wcs.scale_deg() * 3600.0;
    let sip = wcs.sip.is_some();
    Ok(finish(wcs, w, hh, scale, -1, -1.0, "solve-field", sip))
}

/// Plate solving : solve-field local si disponible, sinon solveurs internes.
#[allow(clippy::too_many_arguments)]
pub fn solve_image(img: &Image, catalog: Option<&Path>, center_ra: Option<f64>, center_dec: Option<f64>, radius_deg: f64, focal_mm: Option<f64>, pixel_um: Option<f64>, prefer_solve_field: bool) -> Result<Solution> {
    if prefer_solve_field && which("solve-field").is_some() {
        let d = tempfile::Builder::new().prefix("astroforge_solve_").tempdir()?;
        let p = d.path().join("solve_in.fits");
        io::save(&p, img)?;
        let hint = match (center_ra, center_dec) {
            (Some(r), Some(dd)) => Some((r, dd)),
            _ => None,
        };
        let mut s = solve_with_solvefield(&p, d.path(), hint, radius_deg)?;
        if s.wcs.crpix == [0.0, 0.0] {
            s.center_ra = s.wcs.crval[0];
        }
        return Ok(s);
    }
    let Some(cat) = catalog else { bail!("Fournissez un catalogue local (CSV) ou installez astrometry.net (solve-field) pour le plate solving") };
    let (Some(ra), Some(dec)) = (center_ra, center_dec) else { bail!("Le solveur interne a besoin d'un centre approximatif (RA/Dec) et du rayon de recherche") };
    let hint = match (focal_mm, pixel_um) {
        (Some(f), Some(p)) if f > 0.0 && p > 0.0 => Some(206.265 * p / f),
        _ => None,
    };
    match crate::platesolve::solve_triangles(img, cat, ra, dec, radius_deg, hint, 40, 120, 14, 24, Some(3)) {
        Ok(s) => Ok(s),
        Err(_) => solve_internal(img, cat, ra, dec, radius_deg, focal_mm, pixel_um, 30),
    }
}

// ------------------------------------------------------------------ photométrie
/// Somme sur une ouverture circulaire (surface exacte approchée par
/// sur-échantillonnage 5×5) et aire effective.
pub fn aperture_sum(data: &Image, cx: f64, cy: f64, r_in: f64, r_out: f64) -> (f64, f64) {
    let sub = 5usize;
    let (x0, x1) = ((cx - r_out - 1.0).floor().max(0.0) as usize, ((cx + r_out + 1.0).ceil() as usize).min(data.w.saturating_sub(1)));
    let (y0, y1) = ((cy - r_out - 1.0).floor().max(0.0) as usize, ((cy + r_out + 1.0).ceil() as usize).min(data.h.saturating_sub(1)));
    let (mut s, mut area) = (0.0f64, 0.0f64);
    let (ri2, ro2) = (r_in * r_in, r_out * r_out);
    for y in y0..=y1 {
        for x in x0..=x1 {
            let mut frac = 0.0;
            for sy in 0..sub {
                for sx in 0..sub {
                    let px = x as f64 - 0.5 + (sx as f64 + 0.5) / sub as f64;
                    let py = y as f64 - 0.5 + (sy as f64 + 0.5) / sub as f64;
                    let d2 = (px - cx).powi(2) + (py - cy).powi(2);
                    if d2 <= ro2 && d2 > ri2 {
                        frac += 1.0;
                    }
                }
            }
            if frac > 0.0 {
                let f = frac / (sub * sub) as f64;
                s += f * data.data[y * data.w + x] as f64;
                area += f;
            }
        }
    }
    (s, area)
}

/// Flux d'ouverture moins le fond de la couronne.
pub fn aperture_fluxes(data: &Image, pts: &[Pt], r: f64, r_in: f64, r_out: f64) -> Vec<f64> {
    use rayon::prelude::*;
    pts.par_iter()
        .map(|&(x, y)| {
            let (sa, _) = aperture_sum(data, x, y, -1.0, r);
            let (sn, _) = aperture_sum(data, x, y, r_in, r_out);
            let ap_area = std::f64::consts::PI * r * r;
            let an_area = std::f64::consts::PI * (r_out * r_out - r_in * r_in);
            (sa - sn / an_area.max(1.0) * ap_area).max(1e-9)
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct PhotRow {
    pub x: f64,
    pub y: f64,
    pub flux: f64,
    pub mag: Option<f64>,
    pub zp: Option<f64>,
}

/// Photométrie d'ouverture ; zero point calibré si catalogue + WCS.
#[allow(clippy::too_many_arguments)]
pub fn aperture_photometry(img: &Image, wcs: Option<&Wcs>, catalog: Option<&Path>, r: f64, r_in: f64, r_out: f64, n_max: usize) -> Result<Vec<PhotRow>> {
    let st = stars::detect_stars(img, n_max);
    if st.is_empty() {
        bail!("Aucune étoile détectée");
    }
    let mono = img.mean_lum();
    let flux = aperture_fluxes(&mono, &st, r, r_in, r_out);
    let mut rows: Vec<PhotRow> = st.iter().zip(&flux).map(|(p, &f)| PhotRow { x: p.0, y: p.1, flux: f, mag: None, zp: None }).collect();
    if let (Some(cp), Some(w)) = (catalog, wcs) {
        let cat = load_catalog(cp)?;
        let mags = cat.mags_or(f64::NAN);
        let px: Vec<Option<(f64, f64)>> = (0..cat.len()).map(|i| w.world_to_pixel(cat.ra[i], cat.dec[i])).collect();
        let mut zps = Vec::new();
        for (k, s) in st.iter().enumerate() {
            let mut best = (usize::MAX, f64::INFINITY);
            for (j, p) in px.iter().enumerate() {
                if let Some(p) = p {
                    let d = (p.0 - s.0).hypot(p.1 - s.1);
                    if d < best.1 {
                        best = (j, d);
                    }
                }
            }
            if best.1 < 4.0 && mags[best.0].is_finite() {
                zps.push(mags[best.0] + 2.5 * flux[k].log10());
            }
        }
        if zps.len() >= 2 {
            let zp = stats::median_f64(&zps);
            for r in rows.iter_mut() {
                r.mag = Some(zp - 2.5 * r.flux.log10());
                r.zp = Some(zp);
            }
        }
    }
    Ok(rows)
}

#[derive(Clone, Debug)]
pub struct DiffRow {
    pub role: String,
    pub ra: f64,
    pub dec: f64,
    pub x: f64,
    pub y: f64,
    pub flux: f64,
    pub mag_catalogue: f64,
    pub dmag: f64,
    pub dmag_err: f64,
}

#[derive(Clone, Debug)]
pub struct DiffResult {
    pub rows: Vec<DiffRow>,
    pub dmag: f64,
    pub dmag_err: f64,
    pub mag_ref_ensemble: f64,
    pub n_comp: usize,
}

/// Photométrie différentielle (cible contre un ensemble d'étoiles de comparaison).
#[allow(clippy::too_many_arguments)]
pub fn differential_photometry(img: &Image, w: &Wcs, catalog: &Path, target: Option<(f64, f64)>, n_comp: usize, r: f64, r_in: f64, r_out: f64) -> Result<DiffResult> {
    let mono = img.mean_lum();
    let st = stars::detect_stars(&mono, 150);
    if st.is_empty() {
        bail!("Aucune étoile détectée");
    }
    let flux = aperture_fluxes(&mono, &st, r, r_in, r_out);
    let cat = load_catalog(catalog)?;
    let mags = cat.mags_or(f64::NAN);
    let px: Vec<Option<(f64, f64)>> = (0..cat.len()).map(|i| w.world_to_pixel(cat.ra[i], cat.dec[i])).collect();
    let mut idx_img = Vec::new();
    let mut idx_cat = Vec::new();
    for (k, s) in st.iter().enumerate() {
        let mut best = (usize::MAX, f64::INFINITY);
        for (j, p) in px.iter().enumerate() {
            if let Some(p) = p {
                let d = (p.0 - s.0).hypot(p.1 - s.1);
                if d < best.1 {
                    best = (j, d);
                }
            }
        }
        if best.1 < 4.0 {
            idx_img.push(k);
            idx_cat.push(best.0);
        }
    }
    if idx_img.len() < 4 {
        bail!("Photométrie différentielle : < 4 étoiles appariées au catalogue (plate solving requis)");
    }
    let m_inst: Vec<f64> = flux.iter().map(|f| -2.5 * f.log10()).collect();
    let it = if let Some((tra, tdec)) = target {
        let seps: Vec<f64> = idx_cat.iter().map(|&j| wcs::separation(cat.ra[j], cat.dec[j], tra, tdec) * 3600.0).collect();
        let it = (0..seps.len()).min_by(|&a, &b| seps[a].total_cmp(&seps[b])).unwrap();
        let ps = w.pixel_scales();
        let tol = 10.0f64.max(3.0 * (ps[0] + ps[1]) / 2.0 * 3600.0);
        if seps[it] > tol {
            bail!(
                "Cible introuvable : aucune étoile détectée et appariée à moins de {:.0}\" de la position demandée (la plus proche : {:.0}\").",
                tol,
                seps[it]
            );
        }
        it
    } else {
        (0..idx_img.len()).max_by(|&a, &b| flux[idx_img[a]].total_cmp(&flux[idx_img[b]])).unwrap()
    };
    let mag_t = mags[idx_cat[it]];
    let mut cand: Vec<usize> = (0..idx_img.len()).filter(|&i| i != it && mags[idx_cat[i]].is_finite() && (mags[idx_cat[i]] - mag_t).abs() < 1.5).collect();
    cand.sort_by(|&a, &b| flux[idx_img[b]].total_cmp(&flux[idx_img[a]]));
    let comp: Vec<usize> = cand.into_iter().take(n_comp.max(1)).collect();
    if comp.is_empty() {
        bail!("Photométrie différentielle : aucune étoile de comparaison de magnitude proche (±1,5) de la cible dans le champ.");
    }
    let dmag = m_inst[idx_img[it]] - comp.iter().map(|&i| m_inst[idx_img[i]]).sum::<f64>() / comp.len() as f64;
    let snr = |ii: usize| -> f64 {
        let (sx, sy) = st[idx_img[ii]];
        let mut an = Vec::new();
        let mut ap = 0usize;
        let ro = r_out.ceil() as isize + 1;
        for y in (sy as isize - ro).max(0)..=(sy as isize + ro).min(mono.h as isize - 1) {
            for x in (sx as isize - ro).max(0)..=(sx as isize + ro).min(mono.w as isize - 1) {
                let d2 = (x as f64 - sx).powi(2) + (y as f64 - sy).powi(2);
                if d2 <= r * r {
                    ap += 1;
                } else if d2 > r_in * r_in && d2 <= r_out * r_out {
                    an.push(mono.data[y as usize * mono.w + x as usize]);
                }
            }
        }
        let sig = stats::std(&an) + 1e-9;
        let f = flux[idx_img[ii]];
        f / (f + ap as f64 * sig * sig).sqrt()
    };
    let snr_t = snr(it);
    let snr_c = comp.iter().map(|&i| snr(i)).sum::<f64>() / comp.len() as f64;
    let dmag_err = 1.0857 * (snr_t.powi(-2) + snr_c.powi(-2) / comp.len() as f64).sqrt();
    let m_ref = comp.iter().map(|&i| mags[idx_cat[i]]).sum::<f64>() / comp.len() as f64;
    let mut rows = Vec::new();
    for (k, &i) in std::iter::once(&it).chain(comp.iter()).enumerate() {
        rows.push(DiffRow {
            role: if k == 0 { "Cible".into() } else { "Comp.".into() },
            ra: cat.ra[idx_cat[i]],
            dec: cat.dec[idx_cat[i]],
            x: st[idx_img[i]].0,
            y: st[idx_img[i]].1,
            flux: flux[idx_img[i]],
            mag_catalogue: mags[idx_cat[i]],
            dmag: if k == 0 { dmag } else { f64::NAN },
            dmag_err: if k == 0 { dmag_err } else { f64::NAN },
        });
    }
    Ok(DiffResult { rows, dmag, dmag_err, mag_ref_ensemble: m_ref, n_comp: comp.len() })
}

/// Alignement par étoiles (repli similitude puis phase).
pub fn align_by_stars(ref_: &Image, mov: &Image) -> Image {
    crate::registration::align_by_stars(ref_, mov)
}
