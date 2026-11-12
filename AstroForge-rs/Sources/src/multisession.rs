//! Empilement multi-nuits : chaque nuit calibrée avec SES masters, puis
//! alignement commun, normalisation, pondération et empilement.

use crate::calibration::{self as C, DarkScale, MasterOptions};
use crate::image::Image;
use crate::normalization;
use crate::registration;
use crate::session;
use crate::stacking::{self, StackParams};
use crate::subframes;
use anyhow::{bail, Result};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct Session {
    pub name: String,
    pub lights: Vec<PathBuf>,
    pub calib: HashMap<String, Vec<PathBuf>>,
    pub summary: String,
}

pub fn session_from_folder(folder: &Path) -> Result<Session> {
    let plan = session::plan_session(folder, true, 3.0);
    let Some(g) = plan.groups.first() else { bail!("Aucune image lumière dans {}", folder.display()) };
    Ok(Session { name: g.name(), lights: g.paths("light"), calib: g.calib_paths(), summary: g.summary() })
}

pub struct MultiInfo {
    pub n_frames: usize,
    pub per_night: Vec<usize>,
    pub weights: Option<Vec<f64>>,
}

#[allow(clippy::too_many_arguments)]
pub fn stack_sessions(
    sessions: &[Session],
    method: &str,
    sigma: f64,
    normalization: &str,
    bayer: Option<&str>,
    output: &str,
    weight_mode: &str,
    progress: Option<&(dyn Fn(usize, usize, &str) + Sync)>,
    log: Option<&dyn Fn(&str)>,
) -> Result<(Image, MultiInfo)> {
    if sessions.is_empty() {
        bail!("Aucune session");
    }
    let mut frames: Vec<Image> = Vec::new();
    let mut origin: Vec<usize> = Vec::new();
    for (k, s) in sessions.iter().enumerate() {
        let masters = C::build_masters(&MasterOptions { paths: Some(s.calib.clone()), sigma: 3.0, ..Default::default() })?;
        let lab = format!("Nuit {} : calibration", k + 1);
        let prog = |d: usize, t: usize, _l: &str| {
            if let Some(p) = progress {
                p(d, t, &lab)
            }
        };
        let imgs = C::calibrate_paths(&s.lights, &masters, bayer, DarkScale::Auto, true, None, Some(&prog), output)?;
        if let Some(l) = log {
            l(&format!("Nuit {} ({}) : {} images calibrées", k + 1, s.name, imgs.len()));
        }
        origin.extend(std::iter::repeat_n(k, imgs.len()));
        frames.extend(imgs);
    }
    if frames.is_empty() {
        bail!("Aucune image calibrée");
    }
    let first: Vec<usize> = (0..origin.len()).filter(|&i| origin[i] == 0).collect();
    let ri = if first.is_empty() { 0 } else { first[first.len() / 2] };
    let ref_ = frames[ri].clone();
    let n = frames.len();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let aligned: Vec<Image> = frames
        .into_par_iter()
        .enumerate()
        .map(|(i, f)| {
            let out = if i == ri { f } else { registration::warp(&f, &registration::estimate_transform(&ref_, &f, true)) };
            let d = done.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
            if let Some(p) = progress {
                p(d, n, "Alignement commun");
            }
            out
        })
        .collect();
    let aligned = normalization::normalize_frames(aligned, &ref_, normalization, 128)?;
    let mut weights = None;
    if !weight_mode.is_empty() && weight_mode != "Aucune" {
        let mut rows: Vec<subframes::FrameStats> = aligned
            .par_iter()
            .enumerate()
            .map(|(i, f)| {
                let mut r = subframes::analyze_image(f, false, 60);
                r.path = i.to_string();
                r
            })
            .collect();
        subframes::flag_outliers_default(&mut rows);
        weights = Some(subframes::compute_weights(&rows, weight_mode));
    }
    let refs: Vec<&Image> = aligned.iter().collect();
    let out = stacking::stack(&refs, method, &StackParams { sigma, ..Default::default() }, weights.as_deref())?;
    let per_night = (0..sessions.len()).map(|k| origin.iter().filter(|&&o| o == k).count()).collect();
    Ok((out, MultiInfo { n_frames: aligned.len(), per_night, weights }))
}
