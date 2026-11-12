//! Système de coordonnées célestes (WCS FITS) : projection gnomonique TAN
//! avec matrice CD et distorsions SIP facultatives (A/B, inverses AP/BP).
//! Pixels en convention 0 (premier pixel = 0), comme astropy *_world2pix(…, 0).

use crate::io::{HVal, Header};
use nalgebra::{DMatrix, DVector};

const D2R: f64 = std::f64::consts::PI / 180.0;

#[derive(Clone, Debug, PartialEq)]
pub struct Sip {
    pub order: usize,
    pub a: Vec<((usize, usize), f64)>,
    pub b: Vec<((usize, usize), f64)>,
    pub ap_order: usize,
    pub ap: Vec<((usize, usize), f64)>,
    pub bp: Vec<((usize, usize), f64)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Wcs {
    /// CRPIX (convention FITS : premier pixel = 1)
    pub crpix: [f64; 2],
    /// CRVAL (degrés)
    pub crval: [f64; 2],
    /// CD (degrés / pixel)
    pub cd: [[f64; 2]; 2],
    pub sip: Option<Sip>,
}

fn poly(coefs: &[((usize, usize), f64)], u: f64, v: f64) -> f64 {
    coefs.iter().map(|&((p, q), c)| c * u.powi(p as i32) * v.powi(q as i32)).sum()
}

/// Projection gnomonique : (ra, dec) -> coordonnées standard (ξ, η) en degrés.
pub fn tan_project(ra: f64, dec: f64, ra0: f64, dec0: f64) -> Option<(f64, f64)> {
    let (a, d, a0, d0) = (ra * D2R, dec * D2R, ra0 * D2R, dec0 * D2R);
    let cosc = d0.sin() * d.sin() + d0.cos() * d.cos() * (a - a0).cos();
    if cosc <= 1e-12 {
        return None;
    }
    let xi = d.cos() * (a - a0).sin() / cosc;
    let eta = (d0.cos() * d.sin() - d0.sin() * d.cos() * (a - a0).cos()) / cosc;
    Some((xi / D2R, eta / D2R))
}

/// Inverse de la projection gnomonique.
pub fn tan_deproject(xi_deg: f64, eta_deg: f64, ra0: f64, dec0: f64) -> (f64, f64) {
    let (x, y) = (xi_deg * D2R, eta_deg * D2R);
    let (a0, d0) = (ra0 * D2R, dec0 * D2R);
    let rho = (x * x + y * y).sqrt();
    if rho < 1e-15 {
        return (ra0, dec0);
    }
    let c = rho.atan();
    let (sc, cc) = (c.sin(), c.cos());
    let dec = (cc * d0.sin() + y * sc * d0.cos() / rho).asin();
    let ra = a0 + (x * sc).atan2(rho * d0.cos() * cc - y * d0.sin() * sc);
    ((ra / D2R).rem_euclid(360.0), dec / D2R)
}

/// Séparation angulaire (degrés).
pub fn separation(ra1: f64, dec1: f64, ra2: f64, dec2: f64) -> f64 {
    let (a1, d1, a2, d2) = (ra1 * D2R, dec1 * D2R, ra2 * D2R, dec2 * D2R);
    let sdl = ((d2 - d1) / 2.0).sin();
    let sal = ((a2 - a1) / 2.0).sin();
    let h = sdl * sdl + d1.cos() * d2.cos() * sal * sal;
    2.0 * h.sqrt().clamp(0.0, 1.0).asin() / D2R
}

impl Wcs {
    pub fn new_tan(crpix: [f64; 2], crval: [f64; 2], cd: [[f64; 2]; 2]) -> Self {
        Wcs { crpix, crval, cd, sip: None }
    }

    fn cd_inv(&self) -> [[f64; 2]; 2] {
        let c = self.cd;
        let det = c[0][0] * c[1][1] - c[0][1] * c[1][0];
        [[c[1][1] / det, -c[0][1] / det], [-c[1][0] / det, c[0][0] / det]]
    }

    /// Échelle moyenne (degrés / pixel) = √|det CD|.
    pub fn scale_deg(&self) -> f64 {
        (self.cd[0][0] * self.cd[1][1] - self.cd[0][1] * self.cd[1][0]).abs().sqrt()
    }

    /// Échelles par axe (proj_plane_pixel_scales, degrés).
    pub fn pixel_scales(&self) -> [f64; 2] {
        [(self.cd[0][0].powi(2) + self.cd[1][0].powi(2)).sqrt(), (self.cd[0][1].powi(2) + self.cd[1][1].powi(2)).sqrt()]
    }

    /// Pixel (0-based) -> (ra, dec) en degrés (SIP inclus).
    pub fn pixel_to_world(&self, x: f64, y: f64) -> (f64, f64) {
        let mut u = x + 1.0 - self.crpix[0];
        let mut v = y + 1.0 - self.crpix[1];
        if let Some(s) = &self.sip {
            let (du, dv) = (poly(&s.a, u, v), poly(&s.b, u, v));
            u += du;
            v += dv;
        }
        let xi = self.cd[0][0] * u + self.cd[0][1] * v;
        let eta = self.cd[1][0] * u + self.cd[1][1] * v;
        tan_deproject(xi, eta, self.crval[0], self.crval[1])
    }

    /// Pixel sans distorsion (wcs_pix2world).
    pub fn pixel_to_world_linear(&self, x: f64, y: f64) -> (f64, f64) {
        let u = x + 1.0 - self.crpix[0];
        let v = y + 1.0 - self.crpix[1];
        let xi = self.cd[0][0] * u + self.cd[0][1] * v;
        let eta = self.cd[1][0] * u + self.cd[1][1] * v;
        tan_deproject(xi, eta, self.crval[0], self.crval[1])
    }

    /// (ra, dec) -> pixel 0-based (inverse exacte de SIP par itérations).
    pub fn world_to_pixel(&self, ra: f64, dec: f64) -> Option<(f64, f64)> {
        let (xi, eta) = tan_project(ra, dec, self.crval[0], self.crval[1])?;
        let ci = self.cd_inv();
        let up = ci[0][0] * xi + ci[0][1] * eta;
        let vp = ci[1][0] * xi + ci[1][1] * eta;
        let (mut u, mut v) = (up, vp);
        if let Some(s) = &self.sip {
            if !s.ap.is_empty() || !s.bp.is_empty() {
                u = up + poly(&s.ap, up, vp);
                v = vp + poly(&s.bp, up, vp);
            }
            // point fixe : u + A(u, v) = u'
            for _ in 0..50 {
                let fu = u + poly(&s.a, u, v) - up;
                let fv = v + poly(&s.b, u, v) - vp;
                if fu.abs() < 1e-10 && fv.abs() < 1e-10 {
                    break;
                }
                // jacobien numérique
                let e = 1e-4;
                let j11 = 1.0 + (poly(&s.a, u + e, v) - poly(&s.a, u - e, v)) / (2.0 * e);
                let j12 = (poly(&s.a, u, v + e) - poly(&s.a, u, v - e)) / (2.0 * e);
                let j21 = (poly(&s.b, u + e, v) - poly(&s.b, u - e, v)) / (2.0 * e);
                let j22 = 1.0 + (poly(&s.b, u, v + e) - poly(&s.b, u, v - e)) / (2.0 * e);
                let det = j11 * j22 - j12 * j21;
                if det.abs() < 1e-12 {
                    break;
                }
                u -= (j22 * fu - j12 * fv) / det;
                v -= (-j21 * fu + j11 * fv) / det;
            }
        }
        Some((u + self.crpix[0] - 1.0, v + self.crpix[1] - 1.0))
    }

    pub fn world_to_pixel_linear(&self, ra: f64, dec: f64) -> Option<(f64, f64)> {
        let (xi, eta) = tan_project(ra, dec, self.crval[0], self.crval[1])?;
        let ci = self.cd_inv();
        Some((ci[0][0] * xi + ci[0][1] * eta + self.crpix[0] - 1.0, ci[1][0] * xi + ci[1][1] * eta + self.crpix[1] - 1.0))
    }

    /// Mots-clés FITS (forme CD, SIP si présent).
    pub fn to_header(&self) -> Header {
        let mut h = Header::new();
        let sip = self.sip.is_some();
        h.set("WCSAXES", 2i64);
        h.set("CTYPE1", if sip { "RA---TAN-SIP" } else { "RA---TAN" });
        h.set("CTYPE2", if sip { "DEC--TAN-SIP" } else { "DEC--TAN" });
        h.set("CUNIT1", "deg");
        h.set("CUNIT2", "deg");
        h.set("CRVAL1", self.crval[0]);
        h.set("CRVAL2", self.crval[1]);
        h.set("CRPIX1", self.crpix[0]);
        h.set("CRPIX2", self.crpix[1]);
        h.set("CD1_1", self.cd[0][0]);
        h.set("CD1_2", self.cd[0][1]);
        h.set("CD2_1", self.cd[1][0]);
        h.set("CD2_2", self.cd[1][1]);
        h.set("LONPOLE", 180.0);
        h.set("LATPOLE", self.crval[1]);
        h.set("RADESYS", "ICRS");
        if let Some(s) = &self.sip {
            h.set("A_ORDER", s.order as i64);
            for &((p, q), c) in &s.a {
                h.set(&format!("A_{}_{}", p, q), c);
            }
            h.set("B_ORDER", s.order as i64);
            for &((p, q), c) in &s.b {
                h.set(&format!("B_{}_{}", p, q), c);
            }
            if !s.ap.is_empty() {
                h.set("AP_ORDER", s.ap_order as i64);
                for &((p, q), c) in &s.ap {
                    h.set(&format!("AP_{}_{}", p, q), c);
                }
                h.set("BP_ORDER", s.ap_order as i64);
                for &((p, q), c) in &s.bp {
                    h.set(&format!("BP_{}_{}", p, q), c);
                }
            }
        }
        h
    }

    /// Lecture d'un en-tête FITS : TAN (± SIP), CD ou PC/CDELT ou CDELT/CROTA2.
    pub fn from_header(h: &Header) -> Option<Wcs> {
        let c1 = h.get_str("CTYPE1")?.to_uppercase();
        let c2 = h.get_str("CTYPE2")?.to_uppercase();
        if !(c1.starts_with("RA") && c2.starts_with("DEC")) || !c1.contains("TAN") || !c2.contains("TAN") {
            return None;
        }
        let f = |k: &str| h.get_f64(k);
        let crval = [f("CRVAL1")?, f("CRVAL2")?];
        let crpix = [f("CRPIX1").unwrap_or(0.0), f("CRPIX2").unwrap_or(0.0)];
        let cd = if let (Some(a), Some(d)) = (f("CD1_1"), f("CD2_2")) {
            [[a, f("CD1_2").unwrap_or(0.0)], [f("CD2_1").unwrap_or(0.0), d]]
        } else if f("PC1_1").is_some() || f("PC2_2").is_some() {
            let (s1, s2) = (f("CDELT1").unwrap_or(1.0), f("CDELT2").unwrap_or(1.0));
            [
                [s1 * f("PC1_1").unwrap_or(1.0), s1 * f("PC1_2").unwrap_or(0.0)],
                [s2 * f("PC2_1").unwrap_or(0.0), s2 * f("PC2_2").unwrap_or(1.0)],
            ]
        } else {
            let (s1, s2) = (f("CDELT1")?, f("CDELT2")?);
            let rot = f("CROTA2").unwrap_or(0.0) * D2R;
            [[s1 * rot.cos(), -s2 * rot.sin()], [s1 * rot.sin(), s2 * rot.cos()]]
        };
        let mut w = Wcs { crpix, crval, cd, sip: None };
        if c1.contains("SIP") {
            let read = |pre: &str, order: usize| -> Vec<((usize, usize), f64)> {
                let mut v = Vec::new();
                for p in 0..=order {
                    for q in 0..=(order - p) {
                        if let Some(c) = f(&format!("{}_{}_{}", pre, p, q)) {
                            if c != 0.0 {
                                v.push(((p, q), c));
                            }
                        }
                    }
                }
                v
            };
            let order = f("A_ORDER").unwrap_or(0.0) as usize;
            let ap_order = f("AP_ORDER").unwrap_or(0.0) as usize;
            w.sip = Some(Sip {
                order,
                a: read("A", order),
                b: read("B", f("B_ORDER").map(|v| v as usize).unwrap_or(order)),
                ap_order,
                ap: read("AP", ap_order),
                bp: read("BP", f("BP_ORDER").map(|v| v as usize).unwrap_or(ap_order)),
            });
        }
        Some(w)
    }
}

fn lstsq(a: &DMatrix<f64>, b: &DVector<f64>) -> Option<DVector<f64>> {
    let svd = a.clone().svd(true, true);
    svd.solve(b, 1e-12).ok()
}

/// Ajuste un WCS TAN + SIP (degré `deg` ≥ 2) sur des paires pixel (0-based)
/// / ciel (degrés), point de tangence `crval`. Équivalent de
/// astropy.wcs.utils.fit_wcs_from_points(projection="TAN", sip_degree=deg).
pub fn fit_wcs_sip(px: &[(f64, f64)], world: &[(f64, f64)], crval: [f64; 2], deg: usize) -> Option<Wcs> {
    let n = px.len();
    let proj: Vec<(f64, f64)> = world.iter().filter_map(|&(r, d)| tan_project(r, d, crval[0], crval[1])).collect();
    if proj.len() != n || n < 3 {
        return None;
    }
    // 1) ajustement linéaire : ξ = a·x + b·y + c (pixels 1-based)
    let mut a = DMatrix::<f64>::zeros(n, 3);
    for (i, p) in px.iter().enumerate() {
        a[(i, 0)] = p.0 + 1.0;
        a[(i, 1)] = p.1 + 1.0;
        a[(i, 2)] = 1.0;
    }
    let bx = DVector::from_iterator(n, proj.iter().map(|p| p.0));
    let by = DVector::from_iterator(n, proj.iter().map(|p| p.1));
    let lx = lstsq(&a, &bx)?;
    let ly = lstsq(&a, &by)?;
    let cd = [[lx[0], lx[1]], [ly[0], ly[1]]];
    let det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
    if det.abs() < 1e-30 {
        return None;
    }
    // CRPIX : pixel où (ξ, η) = (0, 0)
    let crpix = [(-(cd[1][1] * lx[2]) + cd[0][1] * ly[2]) / det, (cd[1][0] * lx[2] - cd[0][0] * ly[2]) / det];
    let mut w = Wcs { crpix, crval, cd, sip: None };
    if deg < 2 {
        return Some(w);
    }
    // 2) polynôme complet en (u, v) (terme constant compris) ; CRPIX déplacé
    //    jusqu'à annuler le terme constant, puis décomposition CD + SIP
    let terms: Vec<(usize, usize)> = (0..=deg).flat_map(|t| (0..=t).map(move |q| (t - q, q))).collect();
    let nt = terms.len();
    if n < nt + 2 {
        return Some(w);
    }
    let mut crpix = crpix;
    let i00 = terms.iter().position(|&t| t == (0, 0)).unwrap();
    let i10 = terms.iter().position(|&t| t == (1, 0)).unwrap();
    let i01 = terms.iter().position(|&t| t == (0, 1)).unwrap();
    let (mut cx, mut cy) = (DVector::<f64>::zeros(nt), DVector::<f64>::zeros(nt));
    for _ in 0..8 {
        let mut m = DMatrix::<f64>::zeros(n, nt);
        for (i, p) in px.iter().enumerate() {
            let (u, v) = (p.0 + 1.0 - crpix[0], p.1 + 1.0 - crpix[1]);
            for (k, &(pp, qq)) in terms.iter().enumerate() {
                m[(i, k)] = u.powi(pp as i32) * v.powi(qq as i32);
            }
        }
        cx = lstsq(&m, &bx)?;
        cy = lstsq(&m, &by)?;
        let (c0x, c0y) = (cx[i00], cy[i00]);
        let l = [[cx[i10], cx[i01]], [cy[i10], cy[i01]]];
        let dl = l[0][0] * l[1][1] - l[0][1] * l[1][0];
        if dl.abs() < 1e-30 {
            return None;
        }
        // déplacement δ tel que L·δ = −c0
        let du = (-(l[1][1] * c0x) + l[0][1] * c0y) / dl;
        let dv = (l[1][0] * c0x - l[0][0] * c0y) / dl;
        crpix = [crpix[0] + du, crpix[1] + dv];
        if du.abs() < 1e-9 && dv.abs() < 1e-9 {
            break;
        }
    }
    w.crpix = crpix;
    let cd = [[cx[i10], cx[i01]], [cy[i10], cy[i01]]];
    w.cd = cd;
    let ci = w.cd_inv();
    let mut sa = Vec::new();
    let mut sb = Vec::new();
    for (k, &t) in terms.iter().enumerate() {
        if t.0 + t.1 < 2 {
            continue;
        }
        sa.push((t, ci[0][0] * cx[k] + ci[0][1] * cy[k]));
        sb.push((t, ci[1][0] * cx[k] + ci[1][1] * cy[k]));
    }
    // 3) inverse AP/BP ajustée sur une grille couvrant les étoiles
    let (xmin, xmax) = px.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p.0), a.1.max(p.0)));
    let (ymin, ymax) = px.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p.1), a.1.max(p.1)));
    let ap_deg = deg + 1;
    let iterms: Vec<(usize, usize)> = (1..=ap_deg).flat_map(|t| (0..=t).map(move |q| (t - q, q))).collect();
    let g = 24usize;
    let mut rows = Vec::new();
    for iy in 0..g {
        for ix in 0..g {
            let x = xmin + (xmax - xmin) * ix as f64 / (g - 1) as f64;
            let y = ymin + (ymax - ymin) * iy as f64 / (g - 1) as f64;
            let (u, v) = (x + 1.0 - crpix[0], y + 1.0 - crpix[1]);
            let (up, vp) = (u + poly(&sa, u, v), v + poly(&sb, u, v));
            rows.push((up, vp, u - up, v - vp));
        }
    }
    let mut mi = DMatrix::<f64>::zeros(rows.len(), iterms.len());
    for (i, r) in rows.iter().enumerate() {
        for (k, &(pp, qq)) in iterms.iter().enumerate() {
            mi[(i, k)] = r.0.powi(pp as i32) * r.1.powi(qq as i32);
        }
    }
    let du = DVector::from_iterator(rows.len(), rows.iter().map(|r| r.2));
    let dv = DVector::from_iterator(rows.len(), rows.iter().map(|r| r.3));
    let (ap, bp) = match (lstsq(&mi, &du), lstsq(&mi, &dv)) {
        (Some(a), Some(b)) => (iterms.iter().cloned().zip(a.iter().cloned()).collect(), iterms.iter().cloned().zip(b.iter().cloned()).collect()),
        _ => (vec![], vec![]),
    };
    w.sip = Some(Sip { order: deg, a: sa, b: sb, ap_order: ap_deg, ap, bp });
    Some(w)
}

pub fn hval_to_f64(v: &HVal) -> Option<f64> {
    v.as_f64()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tan_roundtrip() {
        let w = Wcs::new_tan([512.0, 384.0], [83.82, -5.39], [[-2.0e-4, 1.0e-5], [1.0e-5, 2.0e-4]]);
        let (ra, dec) = w.pixel_to_world(100.0, 700.0);
        let (x, y) = w.world_to_pixel(ra, dec).unwrap();
        assert!((x - 100.0).abs() < 1e-7 && (y - 700.0).abs() < 1e-7);
        let h = w.to_header();
        let w2 = Wcs::from_header(&h).unwrap();
        assert_eq!(w, w2);
        assert!(separation(10.0, 20.0, 10.0, 21.0) - 1.0 < 1e-9);
    }
    #[test]
    fn sip_fit() {
        let truth = Wcs::new_tan([200.0, 150.0], [150.0, 30.0], [[-3e-4, 0.0], [0.0, 3e-4]]);
        let mut px = Vec::new();
        let mut world = Vec::new();
        for i in 0..15 {
            for j in 0..12 {
                let (x, y) = (i as f64 * 27.0, j as f64 * 25.0);
                let (u, v) = (x + 1.0 - 200.0, y + 1.0 - 150.0);
                let (ud, vd) = (x + 2e-6 * u * u, y + 1e-6 * u * v);
                let _ = (ud, vd);
                let (r, d) = truth.pixel_to_world(x + 2e-6 * u * u, y + 1e-6 * u * v);
                px.push((x, y));
                world.push((r, d));
            }
        }
        let w = fit_wcs_sip(&px, &world, [150.0, 30.0], 3).unwrap();
        let mut err: f64 = 0.0;
        for (p, q) in px.iter().zip(&world) {
            let (x, y) = w.world_to_pixel(q.0, q.1).unwrap();
            err = err.max(((x - p.0).powi(2) + (y - p.1).powi(2)).sqrt());
        }
        assert!(err < 0.01, "{}", err);
    }
}
