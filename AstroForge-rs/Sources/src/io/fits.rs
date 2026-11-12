//! FITS natif (lecture/écriture), sans dépendance externe :
//! - lecture de la première HDU contenant une image (primaire ou extension
//!   IMAGE), BITPIX 8/16/32/64/−32/−64, BZERO/BSCALE (entiers non signés des
//!   caméras : BZERO = 32768), cubes (C, H, W) ou entrelacés ;
//! - normalisation en flottants [0..1] identique au programme Python ;
//! - écriture Float32 avec en-tête complet ; mise à jour d'en-tête en place.

use super::header::{py_float_repr, HVal, Header};
use crate::image::Image;
use anyhow::{anyhow, bail, Context, Result};
use std::path::Path;

const BLOCK: usize = 2880;

/// Type d'échantillon après application de BZERO/BSCALE (comme astropy).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    U8,
    I8,
    I16,
    U16,
    I32,
    U32,
    I64,
    F32,
    F64,
    Scaled,
}

pub struct Hdu {
    pub header: Header,
    pub header_start: usize,
    pub header_len: usize,
    pub data_start: usize,
    pub data_len: usize,
}

fn parse_value(raw: &str) -> (HVal, String) {
    let s = raw.trim_start();
    if let Some(rest) = s.strip_prefix('\'') {
        // chaîne : '' = apostrophe
        let mut out = String::new();
        let mut chars = rest.chars().peekable();
        let mut consumed = 1usize;
        while let Some(ch) = chars.next() {
            consumed += ch.len_utf8();
            if ch == '\'' {
                if chars.peek() == Some(&'\'') {
                    chars.next();
                    consumed += 1;
                    out.push('\'');
                    continue;
                }
                break;
            }
            out.push(ch);
        }
        let after: String = s[consumed.min(s.len())..].to_string();
        let comment = after.split_once('/').map(|x| x.1.trim().to_string()).unwrap_or_default();
        return (HVal::Str(out.trim_end().to_string()), comment);
    }
    let (val, comment) = match s.split_once('/') {
        Some((a, b)) => (a.trim(), b.trim().to_string()),
        None => (s.trim(), String::new()),
    };
    if val == "T" {
        return (HVal::Bool(true), comment);
    }
    if val == "F" {
        return (HVal::Bool(false), comment);
    }
    if let Ok(i) = val.parse::<i64>() {
        return (HVal::Int(i), comment);
    }
    let fv = val.replace(['D', 'd'], "E");
    if let Ok(f) = fv.parse::<f64>() {
        return (HVal::Float(f), comment);
    }
    (HVal::Str(val.to_string()), comment)
}

/// Analyse un en-tête à partir de `start` ; retourne (en-tête, longueur en octets).
fn parse_header(buf: &[u8], start: usize) -> Result<(Header, usize)> {
    let mut h = Header::new();
    let mut pos = start;
    loop {
        if pos + 80 > buf.len() {
            bail!("FITS tronqué (mot-clé END absent)");
        }
        let card = &buf[pos..pos + 80];
        pos += 80;
        let text = String::from_utf8_lossy(card).to_string();
        let key = text[..8.min(text.len())].trim_end().to_string();
        if key == "END" {
            break;
        }
        if key.is_empty() || key == "COMMENT" || key == "HISTORY" {
            continue;
        }
        if text.len() >= 10 && &text[8..10] == "= " {
            let (v, c) = parse_value(&text[10..]);
            if !h.contains(&key) {
                h.cards.push((key, v, c));
            }
        } else if key == "HIERARCH" {
            if let Some((k, v)) = text[8..].split_once('=') {
                let (v, c) = parse_value(v);
                h.cards.push((k.trim().to_string(), v, c));
            }
        }
    }
    let len = (pos - start).div_ceil(BLOCK) * BLOCK;
    Ok((h, len))
}

fn data_size(h: &Header) -> usize {
    let bitpix = h.get("BITPIX").and_then(|v| v.as_i64()).unwrap_or(8).unsigned_abs() as usize;
    let naxis = h.get("NAXIS").and_then(|v| v.as_i64()).unwrap_or(0) as usize;
    if naxis == 0 {
        return 0;
    }
    let mut n: usize = 1;
    for i in 1..=naxis {
        n *= h.get(&format!("NAXIS{}", i)).and_then(|v| v.as_i64()).unwrap_or(0).max(0) as usize;
    }
    let pcount = h.get("PCOUNT").and_then(|v| v.as_i64()).unwrap_or(0).max(0) as usize;
    let gcount = h.get("GCOUNT").and_then(|v| v.as_i64()).unwrap_or(1).max(1) as usize;
    bitpix / 8 * gcount * (pcount + n)
}

pub fn list_hdus(buf: &[u8]) -> Result<Vec<Hdu>> {
    let mut out = Vec::new();
    let mut pos = 0usize;
    while pos + BLOCK <= buf.len() {
        let first = &buf[pos..pos + 8];
        if out.is_empty() && first != b"SIMPLE  " {
            bail!("Pas un fichier FITS");
        }
        if !out.is_empty() && first != b"XTENSION" {
            break;
        }
        let (header, hlen) = parse_header(buf, pos)?;
        let dlen = data_size(&header);
        let data_start = pos + hlen;
        out.push(Hdu { header, header_start: pos, header_len: hlen, data_start, data_len: dlen });
        pos = data_start + dlen.div_ceil(BLOCK) * BLOCK;
    }
    Ok(out)
}

fn axes(h: &Header) -> Vec<usize> {
    let naxis = h.get("NAXIS").and_then(|v| v.as_i64()).unwrap_or(0) as usize;
    (1..=naxis).map(|i| h.get(&format!("NAXIS{}", i)).and_then(|v| v.as_i64()).unwrap_or(0).max(0) as usize).collect()
}

fn is_image_hdu(h: &Hdu) -> bool {
    let ax = axes(&h.header);
    if let Some(HVal::Str(x)) = h.header.get("XTENSION") {
        if x.trim() != "IMAGE" {
            return false;
        }
    }
    ax.iter().filter(|&&n| n > 0).count() >= 2 && ax.iter().all(|&n| n > 0) && h.data_len > 0
}

/// Lit les valeurs brutes (big-endian) en f64 et détermine le type effectif.
fn decode(buf: &[u8], h: &Header, n: usize) -> Result<(Vec<f64>, Kind)> {
    let bitpix = h.get("BITPIX").and_then(|v| v.as_i64()).ok_or_else(|| anyhow!("BITPIX absent"))?;
    let bzero = h.get_f64("BZERO").unwrap_or(0.0);
    let bscale = h.get_f64("BSCALE").unwrap_or(1.0);
    let bpp = (bitpix.unsigned_abs() / 8) as usize;
    if buf.len() < n * bpp {
        bail!("FITS : données tronquées");
    }
    let mut v = vec![0.0f64; n];
    match bitpix {
        8 => {
            for i in 0..n {
                v[i] = buf[i] as f64;
            }
        }
        16 => {
            for i in 0..n {
                v[i] = i16::from_be_bytes([buf[2 * i], buf[2 * i + 1]]) as f64;
            }
        }
        32 => {
            for i in 0..n {
                v[i] = i32::from_be_bytes(buf[4 * i..4 * i + 4].try_into().unwrap()) as f64;
            }
        }
        64 => {
            for i in 0..n {
                v[i] = i64::from_be_bytes(buf[8 * i..8 * i + 8].try_into().unwrap()) as f64;
            }
        }
        -32 => {
            for i in 0..n {
                v[i] = f32::from_be_bytes(buf[4 * i..4 * i + 4].try_into().unwrap()) as f64;
            }
        }
        -64 => {
            for i in 0..n {
                v[i] = f64::from_be_bytes(buf[8 * i..8 * i + 8].try_into().unwrap());
            }
        }
        _ => bail!("FITS : BITPIX {} non pris en charge", bitpix),
    }
    let kind = if bscale == 1.0 {
        match (bitpix, bzero) {
            (8, z) if z == 0.0 => Kind::U8,
            (8, z) if z == -128.0 => Kind::I8,
            (16, z) if z == 0.0 => Kind::I16,
            (16, z) if z == 32768.0 => Kind::U16,
            (32, z) if z == 0.0 => Kind::I32,
            (32, z) if z == 2147483648.0 => Kind::U32,
            (64, z) if z == 0.0 => Kind::I64,
            (-32, z) if z == 0.0 => Kind::F32,
            (-64, z) if z == 0.0 => Kind::F64,
            _ => Kind::Scaled,
        }
    } else {
        Kind::Scaled
    };
    if bzero != 0.0 || bscale != 1.0 {
        for x in v.iter_mut() {
            *x = *x * bscale + bzero;
        }
    }
    Ok((v, kind))
}

/// Normalisation [0..1] (image_io.to_float).
fn to_unit(v: &[f64], kind: Kind) -> Vec<f32> {
    match kind {
        Kind::U8 => v.iter().map(|&x| (x / 255.0) as f32).collect(),
        Kind::U16 => v.iter().map(|&x| (x / 65535.0) as f32).collect(),
        Kind::I8 | Kind::I16 | Kind::I32 | Kind::U32 | Kind::I64 => {
            let max_t: f64 = match kind {
                Kind::I8 => 127.0,
                Kind::I16 => 32767.0,
                Kind::I32 => 2147483647.0,
                Kind::U32 => 4294967295.0,
                _ => 9.223372036854776e18,
            };
            let itemsize = match kind {
                Kind::I8 => 1,
                Kind::I16 => 2,
                _ => 4,
            };
            let mx = v.iter().cloned().fold(0.0f64, f64::max);
            let scale = if itemsize > 2 && mx <= 65535.0 { 65535.0 } else { max_t };
            v.iter().map(|&x| ((x as f32) / (scale as f32)).clamp(0.0, 1.0)).collect()
        }
        Kind::F32 | Kind::F64 | Kind::Scaled => normalize_float(v),
    }
}

/// image_io._normalize_float.
pub fn normalize_float(v: &[f64]) -> Vec<f32> {
    let a: Vec<f32> = v.iter().map(|&x| if x.is_finite() { x as f32 } else { 0.0 }).collect();
    let m = a.iter().cloned().fold(0.0f32, f32::max) as f64;
    let div = if m > 1.5 {
        if m <= 65535.0 * 1.001 {
            65535.0f32
        } else {
            m as f32
        }
    } else {
        1.0
    };
    a.iter().map(|&x| (x / div).clamp(0.0, 1.0)).collect()
}

/// Lit un FITS -> (image [0..1], en-tête).
pub fn read_fits(path: &Path) -> Result<(Image, Header)> {
    let buf = std::fs::read(path).with_context(|| format!("Impossible de lire : {}", path.display()))?;
    read_fits_bytes(&buf).with_context(|| format!("FITS : {}", path.display()))
}

pub fn read_fits_bytes(buf: &[u8]) -> Result<(Image, Header)> {
    read_fits_impl(buf, true)
}

/// Lecture sans normalisation (valeurs physiques après BZERO/BSCALE).
pub fn read_fits_raw_bytes(buf: &[u8]) -> Result<(Image, Header)> {
    read_fits_impl(buf, false)
}

fn read_fits_impl(buf: &[u8], normalize: bool) -> Result<(Image, Header)> {
    let hdus = list_hdus(buf)?;
    if hdus.iter().any(|h| h.header.get("ZIMAGE").map(|v| v.as_bool()).unwrap_or(false))
        && !hdus.iter().any(is_image_hdu)
    {
        bail!("FITS compressé par tuiles (.fz, fpack) non pris en charge : décompressez-le avec « funpack »");
    }
    let hdu = hdus.iter().find(|h| is_image_hdu(h)).ok_or_else(|| anyhow!("FITS sans données image"))?;
    let ax = axes(&hdu.header);
    // axes significatifs (on retire les axes de taille 1 au-delà du 2e)
    let mut dims: Vec<usize> = ax.clone();
    while dims.len() > 3 && *dims.last().unwrap() == 1 {
        dims.pop();
    }
    if dims.len() > 3 {
        bail!("FITS à {} dimensions non pris en charge", dims.len());
    }
    let n: usize = dims.iter().product();
    let (raw, kind) = decode(&buf[hdu.data_start..], &hdu.header, n)?;
    let d = if normalize { to_unit(&raw, kind) } else { raw.iter().map(|&v| v as f32).collect() };
    let img = if dims.len() == 2 {
        Image::from_vec(dims[0], dims[1], 1, d)
    } else {
        // astropy : forme (NAXIS3, NAXIS2, NAXIS1) = (C, H, W)
        let (n1, n2, n3) = (dims[0], dims[1], dims[2]);
        let planar = (n3 == 1 || n3 == 3) && n3 < n1;
        if planar || !(n1 == 1 || n1 == 3) {
            // plans (C, H, W) -> (H, W, C)
            let (w, h, c) = (n1, n2, n3);
            let mut data = vec![0.0f32; w * h * c];
            for k in 0..c {
                for i in 0..w * h {
                    data[i * c + k] = d[k * w * h + i];
                }
            }
            Image::from_vec(w, h, c, data)
        } else {
            // déjà entrelacé : (H, W, C) = (NAXIS3, NAXIS2, NAXIS1)
            Image::from_vec(n2, n3, n1, d)
        }
    };
    let img = normalize_channels(img);
    Ok((img, hdu.header.clone()))
}

/// 1 canal -> mono ; 2 canaux -> premier ; > 3 -> 3 premiers.
pub fn normalize_channels(img: Image) -> Image {
    match img.c {
        1 | 3 => img,
        2 => img.channel(0),
        _ => img.take_channels(3),
    }
}

/// En-tête seul de la première HDU image (ou de la primaire).
pub fn read_header(path: &Path) -> Result<Header> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    // lecture progressive : les en-têtes sont en tête de fichier
    let mut buf = Vec::new();
    let mut chunk = vec![0u8; BLOCK * 16];
    loop {
        let n = f.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
        // primaire complète ?
        if let Ok((h, hlen)) = parse_header(&buf, 0) {
            let naxis = h.get("NAXIS").and_then(|v| v.as_i64()).unwrap_or(0);
            if naxis >= 2 {
                return Ok(h);
            }
            // extension suivante
            let start = hlen + data_size(&h).div_ceil(BLOCK) * BLOCK;
            if buf.len() >= start + BLOCK {
                if let Ok((h2, _)) = parse_header(&buf, start) {
                    if h2.get("NAXIS").and_then(|v| v.as_i64()).unwrap_or(0) >= 2 {
                        return Ok(h2);
                    }
                    return Ok(h);
                }
            }
            if buf.len() > start + BLOCK * 64 {
                return Ok(h);
            }
        }
        if buf.len() > 64 * 1024 * 1024 {
            break;
        }
    }
    let hdus = list_hdus(&buf)?;
    Ok(hdus
        .iter()
        .find(|h| h.header.get("NAXIS").and_then(|v| v.as_i64()).unwrap_or(0) >= 2)
        .or(hdus.first())
        .map(|h| h.header.clone())
        .unwrap_or_default())
}

// ------------------------------------------------------------------ écriture
fn fmt_value(v: &HVal) -> String {
    match v {
        HVal::Bool(b) => format!("{:>20}", if *b { "T" } else { "F" }),
        HVal::Int(i) => format!("{:>20}", i),
        HVal::Float(f) => {
            let mut s = if f.is_finite() { py_float_repr(*f).to_uppercase() } else { "0.0".into() };
            if !s.contains('.') && !s.contains('E') {
                s.push_str(".0");
            }
            if s.len() > 20 {
                s = format!("{:.12E}", f);
            }
            format!("{:>20}", s)
        }
        HVal::Str(s) => {
            let mut t: String = s.chars().filter(|c| c.is_ascii() && !c.is_ascii_control()).collect();
            t = t.replace('\'', "''");
            if t.len() > 66 {
                t.truncate(66);
            }
            let inner = format!("{:<8}", t);
            format!("'{}'", inner)
        }
    }
}

pub fn card(key: &str, v: &HVal, comment: &str) -> String {
    let k: String = key.chars().take(8).collect::<String>().to_uppercase();
    let mut s = format!("{:<8}= {}", k, fmt_value(v));
    if !comment.is_empty() && s.len() < 77 {
        s.push_str(" / ");
        s.push_str(comment);
    }
    let mut s: String = s.chars().filter(|c| c.is_ascii()).collect();
    s.truncate(80);
    format!("{:<80}", s)
}

const STRUCTURAL: [&str; 9] = ["SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "EXTEND", "BZERO", "BSCALE"];

fn serialize_header(cards: &[String]) -> Vec<u8> {
    let mut out = String::new();
    for c in cards {
        out.push_str(c);
    }
    out.push_str(&format!("{:<80}", "END"));
    let mut b = out.into_bytes();
    let pad = b.len().div_ceil(BLOCK) * BLOCK - b.len();
    b.extend(std::iter::repeat_n(b' ', pad));
    b
}

/// Écrit un FITS Float32 (image écrêtée [0,1] si `clip`), plans (3, H, W) en couleur.
pub fn write_fits(path: &Path, img: &Image, header: Option<&Header>, clip: bool) -> Result<()> {
    let mut cards = vec![
        card("SIMPLE", &HVal::Bool(true), "conforms to FITS standard"),
        card("BITPIX", &HVal::Int(-32), "array data type"),
        card("NAXIS", &HVal::Int(if img.c > 1 { 3 } else { 2 }), "number of array dimensions"),
        card("NAXIS1", &HVal::Int(img.w as i64), ""),
        card("NAXIS2", &HVal::Int(img.h as i64), ""),
    ];
    if img.c > 1 {
        cards.push(card("NAXIS3", &HVal::Int(img.c as i64), ""));
    }
    cards.push(card("EXTEND", &HVal::Bool(true), ""));
    if let Some(h) = header {
        let mut seen = std::collections::HashSet::new();
        for (k, v, c) in &h.cards {
            let ku: String = k.chars().take(8).collect::<String>().to_uppercase();
            if STRUCTURAL.contains(&ku.as_str()) || ku.is_empty() || !seen.insert(ku.clone()) {
                continue;
            }
            cards.push(card(&ku, v, c));
        }
    }
    let mut bytes = serialize_header(&cards);
    let (w, h, c) = (img.w, img.h, img.c);
    let mut data = Vec::with_capacity(w * h * c * 4);
    for k in 0..c {
        for i in 0..w * h {
            let mut v = img.data[i * c + k];
            if clip {
                v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
            }
            data.extend_from_slice(&v.to_be_bytes());
        }
    }
    let pad = data.len().div_ceil(BLOCK) * BLOCK - data.len();
    data.extend(std::iter::repeat_n(0u8, pad));
    bytes.extend(data);
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    std::fs::write(path, bytes).with_context(|| format!("Écriture impossible : {}", path.display()))?;
    Ok(())
}

/// Met à jour l'en-tête de la HDU image d'un FITS existant (données intactes).
pub fn update_header(path: &Path, updates: &Header) -> Result<()> {
    let buf = std::fs::read(path)?;
    let hdus = list_hdus(&buf)?;
    let target = hdus
        .iter()
        .position(is_image_hdu)
        .unwrap_or(0);
    let hdu = &hdus[target];
    // relit les cartes brutes (COMMENT/HISTORY conservés)
    let mut cards: Vec<String> = Vec::new();
    let mut pos = hdu.header_start;
    loop {
        let c = String::from_utf8_lossy(&buf[pos..pos + 80]).to_string();
        pos += 80;
        if c[..8].trim_end() == "END" {
            break;
        }
        cards.push(c);
    }
    for (k, v, com) in &updates.cards {
        let ku: String = k.chars().take(8).collect::<String>().to_uppercase();
        let new = card(&ku, v, com);
        if let Some(c) = cards.iter_mut().find(|c| c[..8].trim_end() == ku) {
            *c = new;
        } else {
            cards.push(new);
        }
    }
    let mut out = Vec::with_capacity(buf.len() + BLOCK);
    out.extend_from_slice(&buf[..hdu.header_start]);
    out.extend(serialize_header(&cards));
    out.extend_from_slice(&buf[hdu.data_start..]);
    std::fs::write(path, out)?;
    Ok(())
}

/// Écrit un FITS uint16 (BZERO = 32768) — format des caméras (tests, outils).
pub fn write_fits_u16(path: &Path, img: &Image, header: Option<&Header>) -> Result<()> {
    let mut cards = vec![
        card("SIMPLE", &HVal::Bool(true), ""),
        card("BITPIX", &HVal::Int(16), ""),
        card("NAXIS", &HVal::Int(if img.c > 1 { 3 } else { 2 }), ""),
        card("NAXIS1", &HVal::Int(img.w as i64), ""),
        card("NAXIS2", &HVal::Int(img.h as i64), ""),
    ];
    if img.c > 1 {
        cards.push(card("NAXIS3", &HVal::Int(img.c as i64), ""));
    }
    cards.push(card("EXTEND", &HVal::Bool(true), ""));
    cards.push(card("BZERO", &HVal::Float(32768.0), ""));
    cards.push(card("BSCALE", &HVal::Float(1.0), ""));
    if let Some(h) = header {
        for (k, v, c) in &h.cards {
            let ku = k.to_uppercase();
            if STRUCTURAL.contains(&ku.as_str()) {
                continue;
            }
            cards.push(card(&ku, v, c));
        }
    }
    let mut bytes = serialize_header(&cards);
    let (w, h, c) = (img.w, img.h, img.c);
    let mut data = Vec::with_capacity(w * h * c * 2);
    for k in 0..c {
        for i in 0..w * h {
            let u = crate::image::f_to_u16(img.data[i * c + k]);
            let s = (u as i32 - 32768) as i16;
            data.extend_from_slice(&s.to_be_bytes());
        }
    }
    let pad = data.len().div_ceil(BLOCK) * BLOCK - data.len();
    data.extend(std::iter::repeat_n(0u8, pad));
    bytes.extend(data);
    std::fs::write(path, bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.fits");
        let img = Image::from_fn(7, 5, |x, y| (x + 7 * y) as f32 / 40.0);
        let mut h = Header::new();
        h.set("EXPTIME", 300.0);
        h.set("OBJECT", "M31");
        h.set("GAIN", 120i64);
        write_fits(&p, &img, Some(&h), true).unwrap();
        let (r, hh) = read_fits(&p).unwrap();
        assert_eq!(r.w, 7);
        assert!((r.get(3, 2, 0) - img.get(3, 2, 0)).abs() < 1e-7);
        assert_eq!(hh.get_f64("EXPTIME"), Some(300.0));
        assert_eq!(hh.get_str("OBJECT").unwrap(), "M31");
        let p2 = dir.path().join("u.fits");
        write_fits_u16(&p2, &img, None).unwrap();
        let (r2, _) = read_fits(&p2).unwrap();
        assert!((r2.get(3, 2, 0) - img.get(3, 2, 0)).abs() < 2e-5);
        let rgb = img.to_rgb();
        write_fits(&p, &rgb, None, true).unwrap();
        let (r3, _) = read_fits(&p).unwrap();
        assert_eq!(r3.c, 3);
    }
}
