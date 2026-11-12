//! Courbe de lumière par photométrie différentielle sur une série d'images
//! alignées sur la référence résolue ; jour julien (milieu de pose), CSV.

use crate::astrometry;
use crate::image::Image;
use crate::io;
use crate::registration;
use crate::timeutil;
use crate::wcs::Wcs;
use anyhow::Result;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct LcRow {
    pub fichier: String,
    pub jd: Option<f64>,
    pub dmag: f64,
    pub dmag_err: f64,
    pub n_comp: usize,
    pub erreur: String,
}

pub fn julian_date(date_obs: &str, exptime: Option<f64>) -> Option<f64> {
    if date_obs.is_empty() {
        return None;
    }
    timeutil::julian_date(date_obs, exptime.unwrap_or(0.0) / 2.0)
}

#[allow(clippy::too_many_arguments)]
pub fn light_curve(paths: &[PathBuf], w: &Wcs, catalog: &Path, target: (f64, f64), n_comp: usize, align: bool, ref_img: Option<&Image>, r: f64, r_in: f64, r_out: f64) -> Result<Vec<LcRow>> {
    let reference = match ref_img {
        Some(i) => i.clone(),
        None => io::load(&paths[0])?,
    };
    let ref_m = reference.mean_lum();
    Ok(paths
        .par_iter()
        .map(|p| {
            let mut rec = LcRow { fichier: p.to_string_lossy().to_string(), jd: None, dmag: f64::NAN, dmag_err: f64::NAN, n_comp: 0, erreur: String::new() };
            let res = (|| -> Result<()> {
                let img = io::load(p)?;
                let mut mono = img.mean_lum();
                if align && mono.w == ref_m.w && mono.h == ref_m.h {
                    mono = registration::warp(&mono, &registration::estimate_transform(&ref_m, &mono, true));
                }
                let t = astrometry::differential_photometry(&mono, w, catalog, Some(target), n_comp, r, r_in, r_out)?;
                let h = io::read_header(p);
                let exp = h.get_f64("EXPTIME").filter(|v| *v != 0.0).or_else(|| h.get_f64("EXPOSURE"));
                rec.jd = julian_date(&h.get_str("DATE-OBS").unwrap_or_default(), exp);
                rec.dmag = t.dmag;
                rec.dmag_err = t.dmag_err;
                rec.n_comp = t.n_comp;
                Ok(())
            })();
            if let Err(e) = res {
                rec.erreur = e.to_string();
            }
            rec
        })
        .collect())
}

pub fn save_csv(rows: &[LcRow], path: &Path) -> Result<()> {
    let mut s = String::from("fichier;jd;dmag;dmag_err;n_comp;erreur\n");
    for r in rows {
        s += &format!(
            "{};{};{};{};{};{}\n",
            r.fichier,
            r.jd.map(|v| v.to_string()).unwrap_or_default(),
            r.dmag,
            r.dmag_err,
            r.n_comp,
            r.erreur
        );
    }
    std::fs::write(path, s)?;
    Ok(())
}
