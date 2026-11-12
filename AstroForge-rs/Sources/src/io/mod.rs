//! Entrées/sorties d'images : FITS, XISF, TIFF, PNG, JPEG, BMP, RAW, SER.
//! Toutes les images sont converties en flottants [0..1] (mono ou RGB).

pub mod debayer;
pub mod fits;
pub mod header;
pub mod raster;
pub mod raw;
pub mod ser;
pub mod xisf;

pub use debayer::{debayer, BAYER_PATTERNS};
pub use header::{HVal, Header};

use crate::image::Image;
use anyhow::{bail, Result};
use std::path::Path;

pub const IMG_EXTENSIONS: [&str; 11] = ["fits", "fit", "fts", "fz", "tif", "tiff", "png", "jpg", "jpeg", "bmp", "xisf"];
pub const RAW_EXTENSIONS: [&str; 10] = ["cr2", "cr3", "nef", "arw", "dng", "raf", "orf", "rw2", "pef", "srw"];
pub const FITS_EXTENSIONS: [&str; 4] = ["fits", "fit", "fts", "fz"];
/// Extensions des images de calibration / lumières (calibration.CALIB_EXTENSIONS).
pub const CALIB_EXTENSIONS: [&str; 18] = [
    "fits", "fit", "fts", "fz", "tif", "tiff", "png", "xisf", "cr2", "cr3", "nef", "arw", "dng", "raf", "orf", "rw2", "pef", "srw",
];

pub fn ext_of(path: &Path) -> String {
    path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase()
}

pub fn has_ext(path: &Path, exts: &[&str]) -> bool {
    exts.contains(&ext_of(path).as_str())
}

/// Charge une image ; dématrice si `bayer` est fourni et l'image mono
/// (RAW : motif forcé ou motif réel du capteur).
pub fn load_image(path: &Path, bayer: Option<&str>) -> Result<Image> {
    let ext = ext_of(path);
    if FITS_EXTENSIONS.contains(&ext.as_str()) {
        let (img, _h) = fits::read_fits(path)?;
        return match bayer {
            Some(b) if img.c == 1 => debayer(&img, b),
            _ => Ok(img),
        };
    }
    if RAW_EXTENSIONS.contains(&ext.as_str()) {
        let (cfa, pattern) = raw::read_raw(path)?;
        return debayer(&cfa, bayer.unwrap_or(&pattern));
    }
    if ext == "xisf" {
        return xisf::read_xisf(path);
    }
    let img = if ext == "tif" || ext == "tiff" { raster::read_tiff(path)? } else { raster::read_other(path)? };
    match bayer {
        Some(b) if img.c == 1 => debayer(&img, b),
        _ => Ok(img),
    }
}

/// Motif de Bayer déclaré dans l'en-tête (BAYERPAT / COLORTYP / CFA-PAT).
pub fn fits_bayer_pattern(h: &Header) -> Option<String> {
    for key in ["BAYERPAT", "COLORTYP", "CFA-PAT"] {
        if let Some(v) = h.get(key) {
            let s = v.to_string().trim().to_uppercase().replace(' ', "");
            if BAYER_PATTERNS.contains(&s.as_str()) {
                return Some(s);
            }
        }
    }
    None
}

/// Charge une image SANS la dématricer (calibration en CFA) : (image, motif détecté).
pub fn load_cfa(path: &Path) -> Result<(Image, Option<String>)> {
    let ext = ext_of(path);
    if RAW_EXTENSIONS.contains(&ext.as_str()) {
        let (img, p) = raw::read_raw(path)?;
        return Ok((img, Some(p)));
    }
    if FITS_EXTENSIONS.contains(&ext.as_str()) {
        let (img, h) = fits::read_fits(path)?;
        let p = if img.c == 1 { fits_bayer_pattern(&h) } else { None };
        return Ok((img, p));
    }
    Ok((load_image(path, None)?, None))
}

/// Sauvegarde : FITS (float32, en-tête) / XISF / TIFF (16 bits, ou 32 bits
/// flottants avec bit_depth « 32f ») / PNG (8, ou 16 avec « 16 ») / JPEG / BMP.
pub fn save_image(path: &Path, img: &Image, bit_depth: &str, header: Option<&Header>) -> Result<()> {
    let ext = ext_of(path);
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    match ext.as_str() {
        "fits" | "fit" | "fts" => fits::write_fits(path, img, header, matches!(bit_depth, "auto" | "float32")),
        "xisf" => xisf::write_xisf(path, img, header),
        "tif" | "tiff" => {
            let d = match bit_depth {
                "32f" | "float32" => raster::TiffDepth::F32,
                "auto" | "16" => raster::TiffDepth::U16,
                _ => raster::TiffDepth::U8,
            };
            raster::write_tiff(path, img, d)
        }
        "png" => raster::write_png(path, img, bit_depth == "16"),
        "jpg" | "jpeg" => raster::write_jpeg_bmp(path, img, true),
        "bmp" => raster::write_jpeg_bmp(path, img, false),
        "" => raster::write_png(&path.with_extension("png"), img, bit_depth == "16"),
        _ => bail!("Format de sortie non pris en charge : .{}", ext),
    }
}

pub fn save(path: impl AsRef<Path>, img: &Image) -> Result<()> {
    save_image(path.as_ref(), img, "auto", None)
}

pub fn load(path: impl AsRef<Path>) -> Result<Image> {
    load_image(path.as_ref(), None)
}

/// Lit l'en-tête FITS (première HDU image) ; vide pour les autres formats
/// (mots-clés FITS d'un XISF inclus).
pub fn read_header(path: &Path) -> Header {
    let ext = ext_of(path);
    if FITS_EXTENSIONS.contains(&ext.as_str()) {
        return fits::read_header(path).unwrap_or_default();
    }
    Header::new()
}
