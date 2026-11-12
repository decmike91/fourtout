//! Bibliothèque de masters réutilisables (FITS float32 + index JSON) :
//! `find_master` retrouve le master le plus adapté à une image lumière.

use crate::image::Image;
use crate::io::{fits, Header};
use crate::paths;
use anyhow::{bail, Result};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

pub const KINDS: [&str; 4] = ["bias", "dark", "darkflat", "flat"];
const META_KEYS: [&str; 14] = ["kind", "exptime", "gain", "offset", "temp", "binning", "width", "height", "filter", "ncombine", "date", "bayer", "thermal", "camera"];

fn lib_dir(lib: Option<&Path>) -> PathBuf {
    lib.map(|p| p.to_path_buf()).unwrap_or_else(paths::masters_dir)
}

fn index_path(lib: Option<&Path>) -> PathBuf {
    lib_dir(lib).join("index.json")
}

pub fn load_index(lib: Option<&Path>) -> Vec<Map<String, Value>> {
    let p = index_path(lib);
    let Ok(txt) = std::fs::read_to_string(&p) else { return vec![] };
    let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(&txt) else { return vec![] };
    let dir = lib_dir(lib);
    arr.into_iter()
        .filter_map(|v| match v {
            Value::Object(m) => Some(m),
            _ => None,
        })
        .filter(|m| dir.join(m.get("file").and_then(|f| f.as_str()).unwrap_or("")).is_file())
        .collect()
}

fn save_index(entries: &[Map<String, Value>], lib: Option<&Path>) -> Result<()> {
    let dir = lib_dir(lib);
    std::fs::create_dir_all(&dir)?;
    let tmp = index_path(lib).with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&entries)?)?;
    std::fs::rename(&tmp, index_path(lib))?;
    Ok(())
}

fn num(m: &Map<String, Value>, k: &str) -> Option<f64> {
    m.get(k).and_then(|v| v.as_f64())
}

/// Enregistre un master et retourne son entrée d'index.
pub fn save_master(img: &Image, kind: &str, meta: &Map<String, Value>, lib: Option<&Path>) -> Result<Map<String, Value>> {
    if !KINDS.contains(&kind) {
        bail!("Type de master inconnu : {}", kind);
    }
    let dir = lib_dir(lib);
    std::fs::create_dir_all(&dir)?;
    let mut m = Map::new();
    for k in META_KEYS {
        m.insert(k.into(), meta.get(k).cloned().unwrap_or(Value::Null));
    }
    m.insert("kind".into(), json!(kind));
    m.insert("height".into(), json!(img.h));
    m.insert("width".into(), json!(img.w));
    if m.get("date").map(|v| v.is_null()).unwrap_or(true) {
        m.insert("date".into(), json!(chrono::Local::now().format("%Y-%m-%dT%H:%M:%S").to_string()));
    }
    let g = |k: &str| crate::stats::fmt_g(num(&m, k).unwrap_or(0.0));
    let temp = num(&m, "temp").map(crate::stats::fmt_g).unwrap_or_else(|| "NA".into());
    let uid = format!("{:06x}", (std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() & 0xffffff) as u32);
    let name = format!("master_{}_{}s_g{}_{}C_{}.fits", kind, g("exptime"), g("gain"), temp, uid).replace(' ', "");
    let mut hdr = Header::new();
    hdr.set(
        "IMAGETYP",
        match kind {
            "bias" => "Master Bias",
            "dark" => "Master Dark",
            "darkflat" => "Master DarkFlat",
            _ => "Master Flat",
        },
    );
    for (key, fk) in [("exptime", "EXPTIME"), ("gain", "GAIN"), ("offset", "OFFSET"), ("temp", "CCD-TEMP"), ("binning", "XBINNING"), ("filter", "FILTER"), ("ncombine", "NCOMBINE"), ("bayer", "BAYERPAT")] {
        match m.get(key) {
            Some(Value::Number(n)) => {
                if let Some(i) = n.as_i64() {
                    hdr.set(fk, i);
                } else {
                    hdr.set(fk, n.as_f64().unwrap_or(0.0));
                }
            }
            Some(Value::String(s)) if !s.is_empty() => hdr.set(fk, s.as_str()),
            _ => {}
        }
    }
    hdr.set_with_comment("THERMAL", m.get("thermal").and_then(|v| v.as_bool()).unwrap_or(false), "dark without bias offset (thermal only)");
    hdr.set("CREATOR", "AstroForge");
    fits::write_fits(&dir.join(&name), img, Some(&hdr), false)?;
    m.insert("file".into(), json!(name));
    let mut entries = load_index(lib);
    entries.push(m.clone());
    save_index(&entries, lib)?;
    Ok(m)
}

pub fn load_master(entry: &Map<String, Value>, lib: Option<&Path>) -> Result<Image> {
    let f = entry.get("file").and_then(|v| v.as_str()).unwrap_or("");
    let p = lib_dir(lib).join(f);
    // lecture brute (sans normalisation : les masters sont déjà en [0..1] ou thermiques)
    let buf = std::fs::read(&p)?;
    let (img, _) = fits::read_fits_raw_bytes(&buf)?;
    Ok(img)
}

pub fn delete_master(entry: &Map<String, Value>, lib: Option<&Path>) -> Result<()> {
    let f = entry.get("file").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let _ = std::fs::remove_file(lib_dir(lib).join(&f));
    let rest: Vec<Map<String, Value>> = load_index(lib).into_iter().filter(|e| e.get("file").and_then(|v| v.as_str()) != Some(f.as_str())).collect();
    save_index(&rest, lib)
}

/// Master le plus adapté (gain, binning, taille, température, exposition).
/// Retourne (entrée, facteur d'échelle du dark).
pub fn find_master(kind: &str, light: &Map<String, Value>, lib: Option<&Path>, temp_tol: f64, allow_scaling: bool) -> (Option<Map<String, Value>>, f64) {
    let mut best: Option<Map<String, Value>> = None;
    let mut best_score = f64::INFINITY;
    let mut scale = 1.0;
    let truthy = |m: &Map<String, Value>, k: &str| match m.get(k) {
        Some(Value::Null) | None => false,
        Some(Value::Number(n)) => n.as_f64().unwrap_or(0.0) != 0.0,
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Bool(b)) => *b,
        _ => true,
    };
    for e in load_index(lib) {
        if e.get("kind").and_then(|v| v.as_str()) != Some(kind) {
            continue;
        }
        if truthy(light, "width") && truthy(&e, "width") && (num(&e, "width"), num(&e, "height")) != (num(light, "width"), num(light, "height")) {
            continue;
        }
        let eb = num(&e, "binning").filter(|b| *b != 0.0).unwrap_or(1.0);
        let lb = num(light, "binning").filter(|b| *b != 0.0).unwrap_or(1.0);
        if eb != lb {
            continue;
        }
        if let (Some(lg), Some(eg)) = (num(light, "gain"), num(&e, "gain")) {
            if (eg - lg).abs() >= 0.5 {
                continue;
            }
        }
        if kind == "flat" && truthy(light, "filter") && truthy(&e, "filter") && e.get("filter") != light.get("filter") {
            continue;
        }
        let mut dt = 0.0;
        if let (Some(lt), Some(et)) = (num(light, "temp"), num(&e, "temp")) {
            dt = (et - lt).abs();
            if ["dark", "bias", "darkflat"].contains(&kind) && dt > temp_tol {
                continue;
            }
        }
        let mut de = 0.0;
        if kind == "dark" && truthy(light, "exptime") && truthy(&e, "exptime") {
            let (le, ee) = (num(light, "exptime").unwrap(), num(&e, "exptime").unwrap());
            de = (ee - le).abs() / le;
            if de > 0.05 && !(allow_scaling && e.get("thermal").and_then(|v| v.as_bool()).unwrap_or(false)) {
                continue;
            }
        }
        let mut age = 0.0;
        if let Some(d) = e.get("date").and_then(|v| v.as_str()) {
            if let Ok(t) = crate::timeutil::parse_iso_utc(d) {
                age = (chrono::Local::now().naive_local() - t).num_days() as f64 / 365.0;
            }
        }
        let score = de * 10.0 + dt + age * 0.5;
        if score < best_score {
            best_score = score;
            scale = if kind == "dark" && truthy(light, "exptime") && truthy(&e, "exptime") && de > 0.05 {
                num(light, "exptime").unwrap() / num(&e, "exptime").unwrap()
            } else {
                1.0
            };
            best = Some(e);
        }
    }
    (best, scale)
}
