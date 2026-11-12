//! XISF natif (PixInsight), spécification 1.0 : lecture des fichiers
//! monolithiques (attachment / inline / embedded, planaire ou entrelacé,
//! zlib, lz4, lz4hc, zstd, avec ou sans byte-shuffle ; ancien format
//! AstroForge sans longueur d'en-tête) ; écriture Float32 planaire.

use super::header::{py_float_repr, HVal, Header};
use crate::image::Image;
use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use std::io::Read;
use std::path::Path;

const SIGNATURE: &[u8] = b"XISF0100";

fn decompress(raw: &[u8], comp: &str, itemsize: usize) -> Result<Vec<u8>> {
    let parts: Vec<&str> = comp.split(':').collect();
    let codec = parts[0].trim().to_lowercase();
    let shuffled = codec.ends_with("+sh");
    let base = if shuffled { &codec[..codec.len() - 3] } else { codec.as_str() };
    let size: usize = parts.get(1).and_then(|s| s.trim().parse().ok()).unwrap_or(0);
    let item: usize = parts.get(2).and_then(|s| s.trim().parse().ok()).unwrap_or(itemsize);
    let mut out = match base {
        "zlib" => {
            let mut d = flate2::read::ZlibDecoder::new(raw);
            let mut v = Vec::with_capacity(size);
            d.read_to_end(&mut v).context("XISF : flux zlib invalide")?;
            v
        }
        "lz4" | "lz4hc" => lz4_flex::block::decompress(raw, size).map_err(|e| anyhow!("XISF : flux LZ4 invalide ({})", e))?,
        "zstd" => {
            let mut d = ruzstd::streaming_decoder::StreamingDecoder::new(raw).map_err(|e| anyhow!("XISF : flux Zstandard invalide ({})", e))?;
            let mut v = Vec::with_capacity(size);
            d.read_to_end(&mut v).context("XISF : flux Zstandard invalide")?;
            v
        }
        _ => bail!("XISF : compression « {} » non prise en charge", codec),
    };
    if shuffled && item > 1 {
        let n = out.len() / item;
        let mut un = vec![0u8; out.len()];
        for i in 0..n {
            for b in 0..item {
                un[i * item + b] = out[b * n + i];
            }
        }
        un[n * item..].copy_from_slice(&out[n * item..]);
        out = un;
    }
    Ok(out)
}

fn xml_text(blob: &[u8]) -> Result<String> {
    if !blob.starts_with(b"XISF") {
        bail!("Pas un fichier XISF");
    }
    if blob.len() >= 16 {
        let hlen = u32::from_le_bytes(blob[8..12].try_into().unwrap()) as usize;
        if 16 + hlen <= blob.len() {
            let x = &blob[16..16 + hlen];
            let t = String::from_utf8_lossy(x);
            let tt = t.trim_start();
            if tt.starts_with("<?xml") || tt.starts_with("<xisf") {
                return Ok(t.trim_end_matches(['\0', ' ', '\r', '\n', '\t']).to_string());
            }
        }
    }
    // ancien format AstroForge : XML juste après la signature
    let end = find(blob, b"</xisf>").ok_or_else(|| anyhow!("XISF corrompu (en-tête XML absent)"))?;
    Ok(String::from_utf8_lossy(&blob[8..end + 7]).trim_end_matches(['\0', ' ', '\r', '\n', '\t']).to_string())
}

fn find(h: &[u8], n: &[u8]) -> Option<usize> {
    h.windows(n.len()).position(|w| w == n)
}

pub fn read_xisf(path: &Path) -> Result<Image> {
    let blob = std::fs::read(path).with_context(|| format!("Impossible de lire : {}", path.display()))?;
    let text = xml_text(&blob)?;
    let doc = roxmltree::Document::parse(&text).map_err(|e| anyhow!("XISF illisible ({}) : {}", e, path.display()))?;
    let img = doc
        .descendants()
        .find(|n| n.tag_name().name() == "Image")
        .ok_or_else(|| anyhow!("XISF sans image : {}", path.display()))?;
    let geom: Vec<usize> = img
        .attribute("geometry")
        .ok_or_else(|| anyhow!("XISF : géométrie absente"))?
        .split(':')
        .map(|v| v.trim().parse::<usize>().unwrap_or(0))
        .collect();
    let (w, h) = (geom[0], *geom.get(1).unwrap_or(&0));
    let c = *geom.get(2).unwrap_or(&1);
    let fmt = img.attribute("sampleFormat").unwrap_or("Float32");
    let itemsize = match fmt {
        "UInt8" => 1,
        "UInt16" => 2,
        "UInt32" | "Float32" => 4,
        "Float64" => 8,
        _ => bail!("XISF : format d'échantillon {} non pris en charge", fmt),
    };
    let loc = img.attribute("location").unwrap_or("");
    let raw: Vec<u8> = if let Some(rest) = loc.strip_prefix("attachment:").or_else(|| loc.strip_prefix("payload:@")) {
        let p: Vec<usize> = rest.split(':').filter_map(|s| s.trim().parse().ok()).collect();
        if p.len() < 2 {
            bail!("XISF : emplacement invalide « {} »", loc);
        }
        let (off, len) = (p[0], p[1]);
        if off + len > blob.len() {
            bail!("XISF tronqué");
        }
        blob[off..off + len].to_vec()
    } else if let Some(enc) = loc.strip_prefix("inline:") {
        let txt: String = img.text().unwrap_or("").chars().filter(|c| !c.is_whitespace()).collect();
        if enc == "base64" {
            base64::engine::general_purpose::STANDARD.decode(txt.as_bytes())?
        } else {
            (0..txt.len() / 2).map(|i| u8::from_str_radix(&txt[2 * i..2 * i + 2], 16).unwrap_or(0)).collect()
        }
    } else if loc == "embedded" {
        let d = img
            .children()
            .find(|n| n.tag_name().name() == "Data")
            .ok_or_else(|| anyhow!("XISF : bloc embarqué sans élément Data"))?;
        let txt: String = d.text().unwrap_or("").chars().filter(|c| !c.is_whitespace()).collect();
        base64::engine::general_purpose::STANDARD.decode(txt.as_bytes())?
    } else {
        bail!("XISF : emplacement de données non pris en charge « {} »", loc);
    };
    let raw = match img.attribute("compression") {
        Some(comp) if !comp.is_empty() => decompress(&raw, comp, itemsize)?,
        _ => raw,
    };
    let big = img.attribute("byteOrder").unwrap_or("little") == "big";
    let n = w * h * c;
    if raw.len() < n * itemsize {
        bail!("XISF : données incomplètes");
    }
    let rd = |i: usize| -> f64 {
        let b = &raw[i * itemsize..(i + 1) * itemsize];
        match (fmt, big) {
            ("UInt8", _) => b[0] as f64,
            ("UInt16", false) => u16::from_le_bytes([b[0], b[1]]) as f64,
            ("UInt16", true) => u16::from_be_bytes([b[0], b[1]]) as f64,
            ("UInt32", false) => u32::from_le_bytes(b.try_into().unwrap()) as f64,
            ("UInt32", true) => u32::from_be_bytes(b.try_into().unwrap()) as f64,
            ("Float32", false) => f32::from_le_bytes(b.try_into().unwrap()) as f64,
            ("Float32", true) => f32::from_be_bytes(b.try_into().unwrap()) as f64,
            ("Float64", false) => f64::from_le_bytes(b.try_into().unwrap()),
            _ => f64::from_be_bytes(b.try_into().unwrap()),
        }
    };
    let normal = img.attribute("pixelStorage").unwrap_or("Planar").to_lowercase() == "normal";
    let cc = c.min(3).max(1);
    let mut vals = vec![0.0f64; w * h * cc];
    for i in 0..w * h {
        for k in 0..cc {
            let src = if c == 1 { i } else if normal { i * c + k } else { k * w * h + i };
            vals[i * cc + k] = rd(src);
        }
    }
    // bornes des flottants
    let is_float = fmt.starts_with("Float");
    if is_float {
        if let Some(b) = img.attribute("bounds") {
            let p: Vec<f64> = b.split(':').filter_map(|s| s.trim().parse().ok()).collect();
            if p.len() >= 2 && p[1] > p[0] && (p[0], p[1]) != (0.0, 1.0) {
                for v in vals.iter_mut() {
                    *v = (*v - p[0]) / (p[1] - p[0]);
                }
            }
        }
    }
    let data: Vec<f32> = match fmt {
        "UInt8" => vals.iter().map(|&v| (v / 255.0) as f32).collect(),
        "UInt16" => vals.iter().map(|&v| (v / 65535.0) as f32).collect(),
        "UInt32" => {
            let mx = vals.iter().cloned().fold(0.0, f64::max);
            let s = if mx <= 65535.0 { 65535.0 } else { 4294967295.0 };
            vals.iter().map(|&v| ((v / s) as f32).clamp(0.0, 1.0)).collect()
        }
        _ => super::fits::normalize_float(&vals),
    };
    let mut im = Image::from_vec(w, h, cc, data);
    if cc == 2 {
        im = im.channel(0);
    }
    Ok(im)
}

/// Mots-clés FITS (éléments FITSKeyword) d'un XISF.
pub fn fits_keywords(path: &Path) -> Result<Header> {
    let blob = std::fs::read(path)?;
    let text = xml_text(&blob)?;
    let doc = roxmltree::Document::parse(&text).map_err(|e| anyhow!("XISF illisible ({})", e))?;
    let mut out = Header::new();
    for el in doc.descendants().filter(|n| n.tag_name().name() == "FITSKeyword") {
        let v = el.attribute("value").unwrap_or("").trim();
        let val = if v.starts_with('\'') && v.ends_with('\'') && v.len() >= 2 {
            HVal::Str(v[1..v.len() - 1].replace("''", "'").trim_end().to_string())
        } else if let Ok(i) = v.parse::<i64>() {
            HVal::Int(i)
        } else if let Ok(f) = v.parse::<f64>() {
            HVal::Float(f)
        } else if v == "T" {
            HVal::Bool(true)
        } else if v == "F" {
            HVal::Bool(false)
        } else {
            HVal::Str(v.to_string())
        };
        out.set(el.attribute("name").unwrap_or(""), val);
    }
    Ok(out)
}

fn fits_value(v: &HVal) -> String {
    match v {
        HVal::Bool(b) => if *b { "T".into() } else { "F".into() },
        HVal::Int(i) => i.to_string(),
        HVal::Float(f) => py_float_repr(*f).to_uppercase().replace("INF", "Inf"),
        HVal::Str(s) => format!("'{}'", s.replace('\'', "''")),
    }
}

fn xml_attr(v: &str) -> String {
    v.replace('&', "&amp;").replace('"', "&quot;").replace('<', "&lt;").replace('>', "&gt;")
}

/// XISF 1.0 monolithique Float32 planaire non compressé, bornes 0:1.
pub fn write_xisf(path: &Path, img: &Image, header: Option<&Header>) -> Result<()> {
    let a = img.map(|v| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 });
    let (c, color) = if a.c == 1 { (1, "Gray") } else if a.c >= 3 { (3, "RGB") } else { (1, "Gray") };
    let mut data = Vec::with_capacity(a.w * a.h * c * 4);
    for k in 0..c {
        for i in 0..a.w * a.h {
            data.extend_from_slice(&a.data[i * a.c + k].to_le_bytes());
        }
    }
    let now = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
    let skip = ["SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "EXTEND", "BZERO", "BSCALE"];
    let mut kw = String::new();
    if let Some(h) = header {
        for (k, v, _) in &h.cards {
            let ku: String = k.chars().take(8).collect::<String>().to_uppercase();
            if skip.contains(&k.to_uppercase().as_str()) || k.trim().is_empty() {
                continue;
            }
            kw.push_str(&format!(
                "<FITSKeyword name=\"{}\" value=\"{}\" comment=\"\"/>",
                xml_attr(&ku),
                xml_attr(&fits_value(v))
            ));
        }
    }
    let make = |off: usize| -> String {
        format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<xisf version=\"1.0\" xmlns=\"http://www.pixinsight.com/xisf\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xsi:schemaLocation=\"http://www.pixinsight.com/xisf http://pixinsight.com/xisf/xisf-1.0.xsd\"><Image geometry=\"{}:{}:{}\" sampleFormat=\"Float32\" bounds=\"0:1\" colorSpace=\"{}\" location=\"attachment:{}:{}\">{}</Image><Metadata><Property id=\"XISF:CreationTime\" type=\"TimePoint\" value=\"{}\"/><Property id=\"XISF:CreatorApplication\" type=\"String\">AstroForge</Property></Metadata></xisf>",
            a.w, a.h, c, color, off, data.len(), kw, now
        )
    };
    let mut off = 0usize;
    for _ in 0..4 {
        let xml = make(off);
        let new = (16 + xml.len()).div_ceil(4096) * 4096;
        if new == off {
            break;
        }
        off = new;
    }
    let xml = make(off);
    if let Some(dir) = path.parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir)?;
        }
    }
    let mut out = Vec::with_capacity(off + data.len());
    out.extend_from_slice(SIGNATURE);
    out.extend_from_slice(&(xml.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(xml.as_bytes());
    out.resize(off, 0);
    out.extend(data);
    std::fs::write(path, out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.xisf");
        let img = Image::from_fn(9, 4, |x, y| (x * y) as f32 / 30.0).to_rgb();
        let mut h = Header::new();
        h.set("OBJECT", "M42");
        write_xisf(&p, &img, Some(&h)).unwrap();
        let r = read_xisf(&p).unwrap();
        assert_eq!((r.w, r.h, r.c), (9, 4, 3));
        assert!((r.get(5, 3, 1) - img.get(5, 3, 1)).abs() < 1e-7);
        let k = fits_keywords(&p).unwrap();
        assert_eq!(k.get_str("OBJECT").unwrap(), "M42");
    }
}
