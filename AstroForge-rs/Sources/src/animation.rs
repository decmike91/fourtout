//! Animation planétaire : film stabilisé (recalage + recadrage commun) et
//! export GIF (natif) ou MP4/AVI (via ffmpeg s'il est installé).

use crate::frames::Frames;
use crate::geom::{self, Interp};
use crate::image::Image;
use crate::ndimage::Border;
use crate::registration::phase_corr;
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::io::Write;
use std::path::Path;

/// Rectangle commun (x0, y0, w, h) après translations (convention apply_shift).
pub fn common_crop(w: usize, h: usize, shifts: &[(f64, f64)], angles: &[f64], size: Option<usize>, center: Option<(i64, i64)>) -> (usize, usize, usize, usize) {
    let maxa = angles.iter().map(|a| a.abs()).fold(0.0, f64::max);
    let margin = if maxa > 0.0 { (1.0 - maxa.to_radians().cos()) * w.max(h) as f64 / 2.0 } else { 0.0 };
    let mut x0 = shifts.iter().map(|s| s.0.max(0.0)).fold(0.0, f64::max) + margin;
    let mut y0 = shifts.iter().map(|s| s.1.max(0.0)).fold(0.0, f64::max) + margin;
    let mut x1 = shifts.iter().map(|s| w as f64 + s.0.min(0.0)).fold(f64::INFINITY, f64::min) - margin;
    let mut y1 = shifts.iter().map(|s| h as f64 + s.1.min(0.0)).fold(f64::INFINITY, f64::min) - margin;
    if let Some(sz) = size {
        let (cx, cy) = center.unwrap_or((((x0 + x1) / 2.0) as i64, ((y0 + y1) / 2.0) as i64));
        let half = (sz / 2) as i64;
        x0 = (cx - half) as f64;
        y0 = (cy - half) as f64;
        x1 = x0 + sz as f64;
        y1 = y0 + sz as f64;
    }
    let (mut x0, mut y0) = (x0.max(0.0) as usize, y0.max(0.0) as usize);
    let (mut x1, mut y1) = ((x1.min(w as f64)) as usize, (y1.min(h as f64)) as usize);
    if x1 < x0 + 16 || y1 < y0 + 16 {
        x0 = 0;
        y0 = 0;
        x1 = w;
        y1 = h;
    }
    (x0, y0, x1 - x0, y1 - y0)
}

/// Frames recalées (phase) puis recadrées sur la zone commune (vue différée).
pub struct StabilizedFrames<'a> {
    pub frames: &'a dyn Frames,
    pub shifts: Vec<(f64, f64)>,
    pub boxr: (usize, usize, usize, usize),
}

impl Frames for StabilizedFrames<'_> {
    fn len(&self) -> usize {
        self.frames.len()
    }
    fn get(&self, i: usize) -> Image {
        let (dx, dy) = self.shifts[i];
        let mut f = self.frames.get(i);
        if dx != 0.0 || dy != 0.0 {
            f = geom::apply_shift(&f, dx, dy, 0.0, Border::Constant);
        }
        let (x0, y0, cw, ch) = self.boxr;
        f.crop(x0, y0, cw, ch)
    }
}

pub fn stabilized_frames<'a>(frames: &'a dyn Frames, size: Option<usize>, ref_index: Option<usize>) -> Result<StabilizedFrames<'a>> {
    let n = frames.len();
    if n == 0 {
        bail!("Aucune frame");
    }
    let ri = ref_index.unwrap_or(n / 2);
    let r = frames.get(ri);
    let shifts: Vec<(f64, f64)> = (0..n).into_par_iter().map(|i| if i == ri { (0.0, 0.0) } else { let (dx, dy, _) = phase_corr(&r, &frames.get(i)); (dx, dy) }).collect();
    let boxr = common_crop(r.w, r.h, &shifts, &[], size, None);
    Ok(StabilizedFrames { frames, shifts, boxr })
}

fn to_rgb8(f: &Image) -> Vec<u8> {
    f.to_rgb().data.iter().map(|&v| crate::image::f_to_u8(v)).collect()
}

/// GIF animé (largeur max 800 px, boucle infinie).
pub fn export_gif(frames: &dyn Frames, path: &Path, fps: u32, max_width: usize) -> Result<()> {
    use image::codecs::gif::{GifEncoder, Repeat};
    use image::{Delay, Frame, RgbaImage};
    let f = std::fs::File::create(path).with_context(|| format!("Écriture impossible : {}", path.display()))?;
    let mut enc = GifEncoder::new_with_speed(std::io::BufWriter::new(f), 10);
    enc.set_repeat(Repeat::Infinite)?;
    let delay = Delay::from_numer_denom_ms(1000, fps.max(1));
    for i in 0..frames.len() {
        let mut fr = frames.get(i).to_rgb();
        if fr.w > max_width {
            let nh = (fr.h as f64 * max_width as f64 / fr.w as f64) as usize;
            fr = geom::resize(&fr, max_width, nh.max(1), Interp::Lanczos4);
        }
        let rgb = to_rgb8(&fr);
        let rgba: Vec<u8> = rgb.chunks(3).flat_map(|p| [p[0], p[1], p[2], 255]).collect();
        let img = RgbaImage::from_raw(fr.w as u32, fr.h as u32, rgba).unwrap();
        enc.encode_frame(Frame::from_parts(img, 0, 0, delay))?;
    }
    Ok(())
}

/// MP4 (H.264) / AVI (MJPEG) via ffmpeg, GIF natif.
pub fn export_video(frames: &dyn Frames, path: &Path, fps: u32, quality: u32) -> Result<()> {
    if frames.is_empty() {
        bail!("Aucune frame à exporter");
    }
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    if ext == "gif" {
        return export_gif(frames, path, fps, 800);
    }
    let ffmpeg = crate::astrometry::which("ffmpeg").ok_or_else(|| anyhow::anyhow!("Export {} : ffmpeg introuvable (installez ffmpeg) — l'export GIF fonctionne sans dépendance", ext.to_uppercase()))?;
    let f0 = frames.get(0);
    let (w, h) = (f0.w, f0.h);
    let q = quality.clamp(1, 100);
    let mut cmd = std::process::Command::new(ffmpeg);
    cmd.args(["-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", &format!("{}x{}", w, h), "-r", &fps.to_string(), "-i", "-"]);
    if ext == "avi" {
        let qv = (31 - (q * 29 / 100)).max(2);
        cmd.args(["-c:v", "mjpeg", "-q:v", &qv.to_string()]);
    } else {
        let crf = 51 - (q * 33 / 100);
        cmd.args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", &crf.to_string(), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]);
    }
    cmd.arg(path).stdin(std::process::Stdio::piped());
    let mut child = cmd.spawn().context("Impossible de lancer ffmpeg")?;
    {
        let stdin = child.stdin.as_mut().unwrap();
        for i in 0..frames.len() {
            stdin.write_all(&to_rgb8(&frames.get(i)))?;
        }
    }
    let st = child.wait()?;
    if !st.success() {
        bail!("ffmpeg a échoué (code {:?})", st.code());
    }
    Ok(())
}
