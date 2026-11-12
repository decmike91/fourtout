//! RAW d'appareils photo (rawloader, pur Rust) : CFA non dématricé de la
//! zone visible, divisé par le niveau de blanc, et motif de Bayer réel.

use crate::image::Image;
#[cfg(feature = "raw")]
use anyhow::anyhow;
use anyhow::{bail, Result};
use std::path::Path;

#[cfg(feature = "raw")]
pub fn read_raw(path: &Path) -> Result<(Image, String)> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    let raw = rawloader::decode_file(path).map_err(|e| {
        if ext == "cr3" {
            anyhow!("RAW CR3 non pris en charge par le décodeur pur Rust : convertissez en DNG (Adobe DNG Converter) — {}", path.display())
        } else {
            anyhow!("RAW illisible ({}) : {}", e, path.display())
        }
    })?;
    if raw.cpp != 1 {
        bail!("RAW déjà dématricé ({} canaux) non pris en charge : {}", raw.cpp, path.display());
    }
    let (w, h) = (raw.width, raw.height);
    let [top, right, bottom, left] = raw.crops;
    let (x0, y0) = (left.min(w), top.min(h));
    let x1 = w.saturating_sub(right).max(x0);
    let y1 = h.saturating_sub(bottom).max(y0);
    let (vw, vh) = (x1 - x0, y1 - y0);
    let full: Vec<f32> = match &raw.data {
        rawloader::RawImageData::Integer(v) => v.iter().map(|&x| x as f32).collect(),
        rawloader::RawImageData::Float(v) => v.clone(),
    };
    let mut data = vec![0.0f32; vw * vh];
    let mut mx = 0.0f32;
    for y in 0..vh {
        for x in 0..vw {
            let v = full[(y + y0) * w + x + x0];
            data[y * vw + x] = v;
            mx = mx.max(v);
        }
    }
    let white = raw.whitelevels[0] as f32;
    let white = if white > 0.0 { white } else { mx.max(1.0) };
    for v in data.iter_mut() {
        *v = (*v / white.max(1.0)).clamp(0.0, 1.0);
    }
    let mut pattern = String::new();
    for (dy, dx) in [(0usize, 0usize), (0, 1), (1, 0), (1, 1)] {
        pattern.push(match raw.cfa.color_at(y0 + dy, x0 + dx) {
            0 => 'R',
            2 => 'B',
            _ => 'G',
        });
    }
    if !super::debayer::BAYER_PATTERNS.contains(&pattern.as_str()) {
        pattern = "RGGB".into();
    }
    Ok((Image::from_vec(vw, vh, 1, data), pattern))
}

#[cfg(not(feature = "raw"))]
pub fn read_raw(path: &Path) -> Result<(Image, String)> {
    bail!("Lecture des RAW désactivée (compilez avec --features raw) : {}", path.display())
}
