//! Solveur astrométrique par triangles (invariants de forme, réflexion
//! permise), vote par étoiles retrouvées, affinement Umeyama, WCS TAN puis
//! distorsions SIP de degré croissant.

use crate::astrometry::{self, Solution};
use crate::image::Image;
use crate::starmatch::{self, Pt, Sim};
use crate::stars;
use crate::stats;
use crate::wcs::{self, Wcs};
use anyhow::{bail, Result};
use std::path::Path;

fn resid(w: &Wcs, xy: &[Pt], world: &[(f64, f64)]) -> f64 {
    if xy.is_empty() {
        return -1.0;
    }
    let sc = w.scale_deg() * 3600.0;
    let mut s = 0.0;
    for (p, q) in xy.iter().zip(world) {
        match w.world_to_pixel(q.0, q.1) {
            Some(px) => s += ((px.0 - p.0).hypot(px.1 - p.1) * sc).powi(2),
            None => return f64::INFINITY,
        }
    }
    (s / xy.len() as f64).sqrt()
}

/// Centre (ra, dec) moyen d'un ensemble de positions (vecteurs unitaires).
fn mean_radec(pts: &[(f64, f64)]) -> (f64, f64) {
    let (mut x, mut y, mut z) = (0.0, 0.0, 0.0);
    for &(r, d) in pts {
        let (r, d) = (r.to_radians(), d.to_radians());
        x += d.cos() * r.cos();
        y += d.cos() * r.sin();
        z += d.sin();
    }
    let n = (x * x + y * y + z * z).sqrt().max(1e-300);
    (y.atan2(x).to_degrees().rem_euclid(360.0), (z / n).clamp(-1.0, 1.0).asin().to_degrees())
}

#[allow(clippy::too_many_arguments)]
pub fn solve_triangles(
    img: &Image,
    catalog: &Path,
    center_ra: f64,
    center_dec: f64,
    radius_deg: f64,
    scale_hint: Option<f64>,
    n_img: usize,
    n_cat: usize,
    k_img: usize,
    k_cat: usize,
    sip_degree: Option<usize>,
) -> Result<Solution> {
    let (w, h) = (img.w, img.h);
    let st = stars::detect_stars(img, n_img);
    if st.len() < 5 {
        bail!("Moins de 5 étoiles détectées — image non résoluble");
    }
    let cat = astrometry::load_catalog(catalog)?;
    let (ra, dec, mags) = astrometry::cone(&cat, center_ra, center_dec, radius_deg, 0.0);
    if ra.len() < 5 {
        bail!("Moins de 5 étoiles de catalogue dans le rayon");
    }
    let order = stats::argsort_asc(&mags.iter().map(|m| if m.is_nan() { 99.0 } else { *m }).collect::<Vec<_>>());
    let order: Vec<usize> = order.into_iter().take(n_cat).collect();
    let (ra, dec): (Vec<f64>, Vec<f64>) = (order.iter().map(|&i| ra[i]).collect(), order.iter().map(|&i| dec[i]).collect());
    let sky = astrometry::tangent_xy(&ra, &dec, center_ra, center_dec);
    let (di, ii) = starmatch::triangles_all(&st, k_img);
    let (dc, ic) = starmatch::triangles_all(&sky, k_cat);
    if di.is_empty() || dc.is_empty() {
        bail!("Pas assez de triangles exploitables");
    }
    let cell = 0.008;
    let mut grid: std::collections::HashMap<(i32, i32), Vec<usize>> = std::collections::HashMap::new();
    for (k, d) in dc.iter().enumerate() {
        grid.entry(((d.0 / cell) as i32, (d.1 / cell) as i32)).or_default().push(k);
    }
    let diag = ((w * w + h * h) as f64).sqrt();
    let mut best: Option<(usize, Sim)> = None;
    let target = 12usize.max((0.6 * st.len() as f64) as usize);
    for (d, tri) in di.iter().zip(&ii) {
        let (gx, gy) = ((d.0 / cell) as i32, (d.1 / cell) as i32);
        for ox in -1..=1 {
            for oy in -1..=1 {
                let Some(list) = grid.get(&(gx + ox, gy + oy)) else { continue };
                for &ci in list {
                    let e = dc[ci];
                    if (e.0 - d.0).hypot(e.1 - d.1) > 0.008 {
                        continue;
                    }
                    let src: Vec<Pt> = tri.iter().map(|&k| st[k]).collect();
                    let dst: Vec<Pt> = ic[ci].iter().map(|&k| sky[k]).collect();
                    let sim = starmatch::umeyama(&src, &dst, true);
                    if !sim.is_finite() {
                        continue;
                    }
                    let sc = sim.scale();
                    if let Some(hint) = scale_hint {
                        let r = sc / hint;
                        if !(0.5 < r && r < 1.5) {
                            continue;
                        }
                    }
                    if sc * diag / 3600.0 > 4.0 * radius_deg + 1.0 {
                        continue;
                    }
                    let (ok, _, _) = starmatch::inliers(&sim, &st, &sky, (3.0 * sc).max(2.0));
                    let n = ok.iter().filter(|&&b| b).count();
                    if best.as_ref().map(|b| n > b.0).unwrap_or(true) {
                        best = Some((n, sim));
                    }
                }
            }
        }
        if best.as_ref().map(|b| b.0 >= target).unwrap_or(false) {
            break;
        }
    }
    let Some((nb, mut sim)) = best else { bail!("Plate solving par triangles : aucun appariement fiable") };
    if nb < 5 {
        bail!("Plate solving par triangles : aucun appariement fiable");
    }
    for _ in 0..3 {
        let sc = sim.scale();
        let (ok, j, _) = starmatch::inliers(&sim, &st, &sky, (2.5 * sc).max(1.5));
        let sel: Vec<usize> = (0..st.len()).filter(|&k| ok[k]).collect();
        if sel.len() < 4 {
            break;
        }
        let p: Vec<Pt> = sel.iter().map(|&k| st[k]).collect();
        let s: Vec<Pt> = sel.iter().map(|&k| sky[j[k]]).collect();
        sim = starmatch::umeyama(&p, &s, true);
    }
    let sc = sim.scale();
    let (okm, jm, _) = starmatch::inliers(&sim, &st, &sky, (2.0 * sc).max(1.5));
    let mut wcs = astrometry::wcs_from_sim(&sim, center_ra, center_dec);
    let mut sip_used = false;
    let mut pairs: (Vec<usize>, Vec<usize>) = ((0..st.len()).filter(|&k| okm[k]).collect(), (0..st.len()).filter(|&k| okm[k]).map(|k| jm[k]).collect());
    let world: Vec<(f64, f64)> = ra.iter().cloned().zip(dec.iter().cloned()).collect();
    if let Some(sd) = sip_degree {
        let (ok0, j0, _) = starmatch::inliers(&sim, &st, &sky, (6.0 * sc).max(3.0));
        let n0 = ok0.iter().filter(|&&b| b).count();
        if n0 > pairs.0.len() {
            pairs = ((0..st.len()).filter(|&k| ok0[k]).collect(), (0..st.len()).filter(|&k| ok0[k]).map(|k| j0[k]).collect());
        }
        for deg in 2..=sd.max(2) {
            let nterm = (deg + 1) * (deg + 2) / 2;
            for _ in 0..4 {
                let (ii_, jj_) = (&pairs.0, &pairs.1);
                if ii_.len() < 2 * nterm + 4 {
                    break;
                }
                let xy: Vec<Pt> = ii_.iter().map(|&k| st[k]).collect();
                let wd: Vec<(f64, f64)> = jj_.iter().map(|&k| world[k]).collect();
                let crval = mean_radec(&wd);
                let Some(w2) = wcs::fit_wcs_sip(&xy, &wd, [crval.0, crval.1], deg) else { break };
                let pxs: Vec<Option<(f64, f64)>> = world.iter().map(|q| w2.world_to_pixel(q.0, q.1)).collect();
                let mut new_i = Vec::new();
                let mut new_j = Vec::new();
                for (k, s) in st.iter().enumerate() {
                    let mut b = (0usize, f64::INFINITY);
                    for (j, p) in pxs.iter().enumerate() {
                        if let Some(p) = p {
                            let d = (s.0 - p.0).hypot(s.1 - p.1);
                            if d < b.1 {
                                b = (j, d);
                            }
                        }
                    }
                    if b.1 < 2.0 {
                        new_i.push(k);
                        new_j.push(b.0);
                    }
                }
                if new_i.len() as f64 >= ii_.len() as f64 * 0.8 && resid(&w2, &xy, &wd) <= resid(&wcs, &xy, &wd) {
                    wcs = w2;
                    sip_used = true;
                }
                if (new_i.len() as f64) < ii_.len() as f64 * 0.8 {
                    break;
                }
                let same = new_i == *ii_;
                pairs = (new_i, new_j);
                if same {
                    break;
                }
            }
        }
    }
    let n_in = pairs.0.len();
    let xy: Vec<Pt> = pairs.0.iter().map(|&k| st[k]).collect();
    let wd: Vec<(f64, f64)> = pairs.1.iter().map(|&k| world[k]).collect();
    let rms = resid(&wcs, &xy, &wd);
    let (cra, cdec) = wcs.pixel_to_world(w as f64 / 2.0, h as f64 / 2.0);
    Ok(Solution {
        wcs,
        scale_arcsec_pix: sc,
        n_matched: n_in as i64,
        rms_arcsec: rms,
        center_ra: cra,
        center_dec: cdec,
        method: format!("triangles{}", if sip_used { " + SIP" } else { "" }),
        sip: sip_used,
    })
}
