//! Profils d'équipement et réglages généraux (profiles/settings.json),
//! calculs optiques, conversion des coordonnées, catalogue Gaia local.

use crate::paths;
use anyhow::{bail, Result};
use regex::Regex;
use serde_json::{json, Map, Value};
use std::path::PathBuf;

pub fn equip_dir() -> PathBuf {
    paths::profiles_root().join("equipment")
}

pub fn settings_path() -> PathBuf {
    paths::profiles_root().join("settings.json")
}

pub const POLLUTION: [&str; 4] = ["Rural sombre (Bortle 3-4)", "Banlieue (Bortle 5-6)", "Urbain (Bortle 7-8)", "Centre-ville (Bortle 9)"];

pub fn default_equipment() -> Map<String, Value> {
    match json!({
        "name": "standard", "camera": "", "pixel_um": 3.8, "sensor_w_px": 5000, "sensor_h_px": 3500,
        "sensor_temp": -10.0, "bayer": "RGGB", "telescope": "", "aperture_mm": 100.0, "focal_mm": 500.0,
        "reducer": false, "reducer_factor": 0.8, "mode": "Ciel profond",
        "light_pollution": "Banlieue (Bortle 5-6)", "catalog": ""
    }) {
        Value::Object(m) => m,
        _ => unreachable!(),
    }
}

pub fn default_settings() -> Map<String, Value> {
    match json!({"use_solve_field": true, "gaia_catalog": "", "active_profile": "standard"}) {
        Value::Object(m) => m,
        _ => unreachable!(),
    }
}

pub fn load_settings() -> Map<String, Value> {
    let mut s = default_settings();
    if let Ok(txt) = std::fs::read_to_string(settings_path()) {
        if let Ok(Value::Object(m)) = serde_json::from_str::<Value>(&txt) {
            for (k, v) in m {
                s.insert(k, v);
            }
        }
    }
    s
}

pub fn save_settings(s: &Map<String, Value>) -> Result<()> {
    let p = settings_path();
    std::fs::create_dir_all(p.parent().unwrap())?;
    std::fs::write(p, serde_json::to_string_pretty(s)?)?;
    Ok(())
}

pub fn list_profiles() -> Vec<String> {
    let d = equip_dir();
    let _ = std::fs::create_dir_all(&d);
    let mut names: Vec<String> = std::fs::read_dir(&d)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter_map(|e| e.file_name().to_str().and_then(|n| n.strip_suffix(".json").map(|s| s.to_string())))
                .collect()
        })
        .unwrap_or_default();
    names.sort_by(|a, b| (a != "standard", a).cmp(&(b != "standard", b)));
    if names.is_empty() {
        names.push("standard".into());
    }
    names
}

pub fn load_profile(name: &str) -> Map<String, Value> {
    let mut p = default_equipment();
    p.insert("name".into(), json!(name));
    if name != "standard" {
        if let Ok(txt) = std::fs::read_to_string(equip_dir().join(format!("{}.json", name))) {
            if let Ok(Value::Object(m)) = serde_json::from_str::<Value>(&txt) {
                for (k, v) in m {
                    p.insert(k, v);
                }
            }
        }
    }
    p
}

pub fn save_profile(name: &str, params: &Map<String, Value>) -> Result<()> {
    let name = name.trim();
    if name.is_empty() {
        bail!("Nom de profil vide");
    }
    if name == "standard" {
        bail!("Le profil « standard » est protégé — enregistrez sous un autre nom");
    }
    std::fs::create_dir_all(equip_dir())?;
    let mut p = params.clone();
    p.insert("name".into(), json!(name));
    std::fs::write(equip_dir().join(format!("{}.json", name)), serde_json::to_string_pretty(&p)?)?;
    Ok(())
}

pub fn delete_profile(name: &str) -> Result<()> {
    if name == "standard" {
        bail!("Le profil « standard » ne peut pas être supprimé");
    }
    let p = equip_dir().join(format!("{}.json", name));
    if p.exists() {
        std::fs::remove_file(p)?;
    }
    Ok(())
}

fn f(p: &Map<String, Value>, k: &str) -> f64 {
    p.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0)
}

pub fn effective_focal(p: &Map<String, Value>) -> f64 {
    let reducer = p.get("reducer").and_then(|v| v.as_bool()).unwrap_or(false);
    f(p, "focal_mm") * if reducer { f(p, "reducer_factor") } else { 1.0 }
}

pub fn scale_from_focal(pixel_um: f64, focal_mm: f64) -> f64 {
    206.265 * pixel_um / focal_mm
}

pub fn focal_from_scale(pixel_um: f64, scale: f64) -> f64 {
    206.265 * pixel_um / scale
}

/// (largeur°, hauteur°, échelle "/px).
pub fn field_of_view(p: &Map<String, Value>) -> (f64, f64, f64) {
    let s = scale_from_focal(f(p, "pixel_um"), effective_focal(p));
    (f(p, "sensor_w_px") * s / 3600.0, f(p, "sensor_h_px") * s / 3600.0, s)
}

pub fn suggestion_denoise(p: &Map<String, Value>) -> f64 {
    match p.get("light_pollution").and_then(|v| v.as_str()).unwrap_or("") {
        x if x == POLLUTION[0] => 0.35,
        x if x == POLLUTION[1] => 0.5,
        x if x == POLLUTION[2] => 0.65,
        x if x == POLLUTION[3] => 0.8,
        _ => 0.5,
    }
}

/// Coordonnée -> degrés décimaux : « 10h25m03s », « 10:25:03 », « +19°48'30" »,
/// « 19d48m30s », « 156,25 »… (RA sexagésimale en heures, ×15).
pub fn parse_coord(txt: &str, is_ra: bool) -> Result<Option<f64>> {
    let t = txt.trim().replace(',', ".").replace('−', "-");
    if t.is_empty() {
        return Ok(None);
    }
    if let Ok(v) = t.parse::<f64>() {
        return Ok(Some(v));
    }
    let sign = if t.starts_with('-') { -1.0 } else { 1.0 };
    let rx_num = Regex::new(r"\d+(?:\.\d*)?|\.\d+").unwrap();
    let nums: Vec<f64> = rx_num.find_iter(&t).filter_map(|m| m.as_str().parse().ok()).collect();
    let rx_rest = Regex::new(r#"[\d.\s:+\-hHmMsSdD°º'"′″]"#).unwrap();
    let rest = rx_rest.replace_all(&t, "");
    if nums.is_empty() || nums.len() > 3 || !rest.is_empty() {
        bail!("Coordonnée illisible : « {} »", txt);
    }
    let a = nums[0];
    let m = *nums.get(1).unwrap_or(&0.0);
    let s = *nums.get(2).unwrap_or(&0.0);
    if m >= 60.0 || s >= 60.0 {
        bail!("Minutes/secondes ≥ 60 dans « {} »", txt);
    }
    let v = sign * (a + m / 60.0 + s / 3600.0);
    Ok(Some(if is_ra { v * 15.0 } else { v }))
}

/// Catalogue Gaia DR3 (étoiles < mag_limit dans le rayon) -> CSV
/// « ra,dec,mag,bp,rp » (service TAP de l'ESA, requête asynchrone).
#[cfg(feature = "net")]
pub fn make_gaia_catalog(ra: f64, dec: f64, radius: f64, out_csv: &std::path::Path, mag_limit: f64) -> Result<PathBuf> {
    use anyhow::Context;
    let q = format!(
        "SELECT ra, dec, phot_g_mean_mag AS mag, phot_bp_mean_mag AS bp, phot_rp_mean_mag AS rp FROM gaiadr3.gaia_source WHERE phot_g_mean_mag < {} AND CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {}, {}, {}))=1",
        mag_limit, ra, dec, radius
    );
    let base = "https://gea.esac.esa.int/tap-server/tap/async";
    let agent = ureq::AgentBuilder::new().redirects(0).build();
    let resp = agent
        .post(base)
        .send_form(&[("REQUEST", "doQuery"), ("LANG", "ADQL"), ("FORMAT", "csv"), ("PHASE", "RUN"), ("QUERY", &q)]);
    let job = match resp {
        Ok(r) => r.header("Location").map(|s| s.to_string()).context("Gaia : réponse sans adresse de tâche")?,
        Err(ureq::Error::Status(303, r)) | Err(ureq::Error::Status(302, r)) => r.header("Location").map(|s| s.to_string()).context("Gaia : réponse sans adresse de tâche")?,
        Err(e) => bail!("Échec de la requête Gaia : {}", e),
    };
    for _ in 0..600 {
        let phase = ureq::get(&format!("{}/phase", job)).call().map_err(|e| anyhow::anyhow!("Échec de la requête Gaia : {}", e))?.into_string()?;
        match phase.trim() {
            "COMPLETED" => break,
            "ERROR" | "ABORTED" => bail!("Échec de la requête Gaia (tâche {})", phase.trim()),
            _ => std::thread::sleep(std::time::Duration::from_millis(1000)),
        }
    }
    let csv = ureq::get(&format!("{}/results/result", job)).call().map_err(|e| anyhow::anyhow!("Échec de la requête Gaia : {}", e))?.into_string()?;
    let mut lines = csv.lines();
    let head = lines.next().unwrap_or("").to_lowercase();
    let mut out = String::from(head.trim());
    out.push('\n');
    for l in lines {
        out.push_str(l.trim());
        out.push('\n');
    }
    if let Some(d) = out_csv.parent() {
        if !d.as_os_str().is_empty() {
            std::fs::create_dir_all(d)?;
        }
    }
    std::fs::write(out_csv, out)?;
    Ok(out_csv.to_path_buf())
}

#[cfg(not(feature = "net"))]
pub fn make_gaia_catalog(_ra: f64, _dec: f64, _radius: f64, _out: &std::path::Path, _mag: f64) -> Result<PathBuf> {
    bail!("Fonction réseau désactivée (compilez avec --features net)")
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn coords() {
        assert!((parse_coord("10h42m", true).unwrap().unwrap() - 160.5).abs() < 1e-9);
        assert!((parse_coord("+41°16'", false).unwrap().unwrap() - 41.266666).abs() < 1e-5);
        assert!((parse_coord("-0 30 00", false).unwrap().unwrap() + 0.5).abs() < 1e-9);
        assert_eq!(parse_coord("156,25", true).unwrap(), Some(156.25));
        assert!(parse_coord("abc", true).is_err());
    }
}
