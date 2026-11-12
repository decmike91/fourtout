//! Images « bureautiques » : TIFF (8/16/32 bits entiers, 32/64 bits
//! flottants), PNG 8/16 bits, JPEG, BMP.

use crate::image::{f_to_u16, f_to_u8, Image};
use anyhow::{anyhow, bail, Context, Result};
use std::io::BufWriter;
use std::path::Path;

fn from_u8(w: usize, h: usize, c: usize, v: &[u8]) -> Image {
    shape(w, h, c, v.iter().map(|&x| x as f32 / 255.0).collect())
}

fn from_u16(w: usize, h: usize, c: usize, v: &[u16]) -> Image {
    shape(w, h, c, v.iter().map(|&x| x as f32 / 65535.0).collect())
}

/// Garde 1 canal (gris, gris+alpha) ou 3 (RGB, RGBA -> RGB).
fn shape(w: usize, h: usize, c: usize, data: Vec<f32>) -> Image {
    let img = Image::from_vec(w, h, c, data);
    match c {
        1 | 3 => img,
        2 => img.channel(0),
        _ => img.take_channels(3),
    }
}

pub fn read_tiff(path: &Path) -> Result<Image> {
    use tiff::decoder::{Decoder, DecodingResult, Limits};
    let f = std::fs::File::open(path).with_context(|| format!("Impossible de lire : {}", path.display()))?;
    let mut dec = Decoder::new(std::io::BufReader::new(f))
        .map_err(|e| anyhow!("TIFF illisible ({}) : {}", e, path.display()))?
        .with_limits(Limits::unlimited());
    let (w, h) = dec.dimensions()?;
    let (w, h) = (w as usize, h as usize);
    let ct = dec.colortype()?;
    let c = match ct {
        tiff::ColorType::Gray(_) => 1,
        tiff::ColorType::GrayA(_) => 2,
        tiff::ColorType::RGB(_) => 3,
        tiff::ColorType::RGBA(_) => 4,
        tiff::ColorType::CMYK(_) => 4,
        _ => bail!("TIFF : espace couleur non pris en charge ({:?})", ct),
    };
    let res = dec.read_image().map_err(|e| anyhow!("TIFF illisible ({}) : {}", e, path.display()))?;
    let img = match res {
        DecodingResult::U8(v) => from_u8(w, h, c, &v),
        DecodingResult::U16(v) => from_u16(w, h, c, &v),
        DecodingResult::U32(v) => {
            let mx = v.iter().cloned().max().unwrap_or(0) as f64;
            let s = if mx <= 65535.0 { 65535.0 } else { u32::MAX as f64 };
            shape(w, h, c, v.iter().map(|&x| ((x as f64 / s) as f32).clamp(0.0, 1.0)).collect())
        }
        DecodingResult::U64(v) => {
            let mx = v.iter().cloned().max().unwrap_or(0) as f64;
            let s = if mx <= 65535.0 { 65535.0 } else { u64::MAX as f64 };
            shape(w, h, c, v.iter().map(|&x| ((x as f64 / s) as f32).clamp(0.0, 1.0)).collect())
        }
        DecodingResult::I8(v) => shape(w, h, c, v.iter().map(|&x| (x as f32 / 127.0).clamp(0.0, 1.0)).collect()),
        DecodingResult::I16(v) => shape(w, h, c, v.iter().map(|&x| (x as f32 / 32767.0).clamp(0.0, 1.0)).collect()),
        DecodingResult::I32(v) => {
            let mx = v.iter().cloned().max().unwrap_or(0) as f64;
            let s = if mx <= 65535.0 { 65535.0 } else { i32::MAX as f64 };
            shape(w, h, c, v.iter().map(|&x| ((x as f64 / s) as f32).clamp(0.0, 1.0)).collect())
        }
        DecodingResult::I64(v) => {
            let mx = v.iter().cloned().max().unwrap_or(0) as f64;
            let s = if mx <= 65535.0 { 65535.0 } else { i64::MAX as f64 };
            shape(w, h, c, v.iter().map(|&x| ((x as f64 / s) as f32).clamp(0.0, 1.0)).collect())
        }
        DecodingResult::F32(v) => {
            let d = super::fits::normalize_float(&v.iter().map(|&x| x as f64).collect::<Vec<_>>());
            shape(w, h, c, d)
        }
        DecodingResult::F64(v) => shape(w, h, c, super::fits::normalize_float(&v)),

    };
    Ok(img)
}

pub fn read_other(path: &Path) -> Result<Image> {
    let bytes = std::fs::read(path).with_context(|| format!("Impossible de lire : {}", path.display()))?;
    if bytes.is_empty() {
        bail!("Impossible de lire : {}", path.display());
    }
    let dynimg = image::load_from_memory(&bytes).map_err(|e| anyhow!("Impossible de lire : {} ({})", path.display(), e))?;
    use image::DynamicImage as D;
    let (w, h) = (dynimg.width() as usize, dynimg.height() as usize);
    Ok(match dynimg {
        D::ImageLuma8(b) => from_u8(w, h, 1, b.as_raw()),
        D::ImageLumaA8(b) => from_u8(w, h, 2, b.as_raw()),
        D::ImageRgb8(b) => from_u8(w, h, 3, b.as_raw()),
        D::ImageRgba8(b) => from_u8(w, h, 4, b.as_raw()),
        D::ImageLuma16(b) => from_u16(w, h, 1, b.as_raw()),
        D::ImageLumaA16(b) => from_u16(w, h, 2, b.as_raw()),
        D::ImageRgb16(b) => from_u16(w, h, 3, b.as_raw()),
        D::ImageRgba16(b) => from_u16(w, h, 4, b.as_raw()),
        D::ImageRgb32F(b) => shape(w, h, 3, super::fits::normalize_float(&b.as_raw().iter().map(|&x| x as f64).collect::<Vec<_>>())),
        D::ImageRgba32F(b) => shape(w, h, 4, super::fits::normalize_float(&b.as_raw().iter().map(|&x| x as f64).collect::<Vec<_>>())),
        other => {
            let b = other.to_rgb16();
            from_u16(w, h, 3, b.as_raw())
        }
    })
}

/// Taille (largeur, hauteur) sans décoder l'image (PNG/JPEG/TIFF/BMP).
pub fn dimensions(path: &Path) -> Option<(usize, usize)> {
    let ext = path.extension()?.to_str()?.to_lowercase();
    if ext == "tif" || ext == "tiff" {
        let f = std::fs::File::open(path).ok()?;
        let mut d = tiff::decoder::Decoder::new(std::io::BufReader::new(f)).ok()?;
        let (w, h) = d.dimensions().ok()?;
        return Some((w as usize, h as usize));
    }
    image::image_dimensions(path).ok().map(|(w, h)| (w as usize, h as usize))
}

fn create(path: &Path) -> Result<BufWriter<std::fs::File>> {
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    Ok(BufWriter::new(std::fs::File::create(path).with_context(|| format!("Écriture impossible : {}", path.display()))?))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TiffDepth {
    U8,
    U16,
    F32,
}

pub fn write_tiff(path: &Path, img: &Image, depth: TiffDepth) -> Result<()> {
    use tiff::encoder::{colortype, compression::Lzw, TiffEncoder};
    let f = create(path)?;
    let mut enc = TiffEncoder::new(f)?;
    let (w, h) = (img.w as u32, img.h as u32);
    let a = if img.c == 1 || img.c == 3 { img.clone() } else { img.take_channels(3) };
    match (depth, a.c) {
        (TiffDepth::U8, 1) => {
            let v: Vec<u8> = a.data.iter().map(|&x| f_to_u8(x)).collect();
            enc.write_image_with_compression::<colortype::Gray8, _>(w, h, Lzw, &v)?;
        }
        (TiffDepth::U8, _) => {
            let v: Vec<u8> = a.data.iter().map(|&x| f_to_u8(x)).collect();
            enc.write_image_with_compression::<colortype::RGB8, _>(w, h, Lzw, &v)?;
        }
        (TiffDepth::U16, 1) => {
            let v: Vec<u16> = a.data.iter().map(|&x| f_to_u16(x)).collect();
            enc.write_image_with_compression::<colortype::Gray16, _>(w, h, Lzw, &v)?;
        }
        (TiffDepth::U16, _) => {
            let v: Vec<u16> = a.data.iter().map(|&x| f_to_u16(x)).collect();
            enc.write_image_with_compression::<colortype::RGB16, _>(w, h, Lzw, &v)?;
        }
        (TiffDepth::F32, 1) => {
            let v: Vec<f32> = a.data.iter().map(|&x| if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 }).collect();
            enc.write_image::<colortype::Gray32Float>(w, h, &v)?;
        }
        (TiffDepth::F32, _) => {
            let v: Vec<f32> = a.data.iter().map(|&x| if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 }).collect();
            enc.write_image::<colortype::RGB32Float>(w, h, &v)?;
        }
    }
    Ok(())
}

pub fn write_png(path: &Path, img: &Image, sixteen: bool) -> Result<()> {
    use image::{ImageBuffer, Luma, Rgb};
    let a = if img.c == 1 || img.c == 3 { img.clone() } else { img.take_channels(3) };
    let (w, h) = (a.w as u32, a.h as u32);
    let err = |e: image::ImageError| anyhow!("Encodage impossible : {} ({})", path.display(), e);
    let mut f = create(path)?;
    let fmt = image::ImageFormat::Png;
    if sixteen {
        let v: Vec<u16> = a.data.iter().map(|&x| f_to_u16(x)).collect();
        if a.c == 1 {
            ImageBuffer::<Luma<u16>, _>::from_raw(w, h, v).unwrap().write_to(&mut f, fmt).map_err(err)?;
        } else {
            ImageBuffer::<Rgb<u16>, _>::from_raw(w, h, v).unwrap().write_to(&mut f, fmt).map_err(err)?;
        }
    } else {
        let v: Vec<u8> = a.data.iter().map(|&x| f_to_u8(x)).collect();
        if a.c == 1 {
            ImageBuffer::<Luma<u8>, _>::from_raw(w, h, v).unwrap().write_to(&mut f, fmt).map_err(err)?;
        } else {
            ImageBuffer::<Rgb<u8>, _>::from_raw(w, h, v).unwrap().write_to(&mut f, fmt).map_err(err)?;
        }
    }
    Ok(())
}

pub fn write_jpeg_bmp(path: &Path, img: &Image, jpeg: bool) -> Result<()> {
    use image::codecs::jpeg::JpegEncoder;
    let a = if img.c == 1 || img.c == 3 { img.clone() } else { img.take_channels(3) };
    let v: Vec<u8> = a.data.iter().map(|&x| f_to_u8(x)).collect();
    let ct = if a.c == 1 { image::ExtendedColorType::L8 } else { image::ExtendedColorType::Rgb8 };
    let mut f = create(path)?;
    if jpeg {
        let mut enc = JpegEncoder::new_with_quality(&mut f, 96);
        enc.encode(&v, a.w as u32, a.h as u32, ct).map_err(|e| anyhow!("Encodage impossible : {} ({})", path.display(), e))?;
    } else {
        use image::ImageEncoder;
        image::codecs::bmp::BmpEncoder::new(&mut f)
            .write_image(&v, a.w as u32, a.h as u32, ct)
            .map_err(|e| anyhow!("Encodage impossible : {} ({})", path.display(), e))?;
    }
    Ok(())
}
