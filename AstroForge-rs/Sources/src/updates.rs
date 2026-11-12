//! Vérification des mises à jour : manifeste JSON publié
//! {"version", "url", "sha256", "notes"} (entrées par plate-forme possibles).
//! Adresse : settings.json « update_url » ou ASTROFORGE_UPDATE_URL.

use anyhow::{bail, Result};
use serde_json::Value;

/// « 2.0.10 » -> (2, 0, 10, final?) ; une pré-version est antérieure.
pub fn parse_version(v: &str) -> Vec<i64> {
    let re = regex::Regex::new(r"^\s*v?(\d+(?:\.\d+)*)(.*)").unwrap();
    let Some(c) = re.captures(v) else { return vec![0] };
    let mut nums: Vec<i64> = c[1].split('.').map(|x| x.parse().unwrap_or(0)).collect();
    while nums.len() < 3 {
        nums.push(0);
    }
    nums.push(if c[2].trim().is_empty() { 1 } else { 0 });
    nums
}

pub fn is_newer(remote: &str, local: &str) -> bool {
    parse_version(remote) > parse_version(local)
}

pub fn platform_key() -> &'static str {
    if cfg!(windows) {
        "windows"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else {
        "linux"
    }
}

pub fn update_url() -> String {
    if let Ok(e) = std::env::var("ASTROFORGE_UPDATE_URL") {
        return e.trim().to_string();
    }
    crate::equipment::load_settings().get("update_url").and_then(|v| v.as_str()).unwrap_or("").trim().to_string()
}

pub struct UpdateInfo {
    pub version: String,
    pub url: String,
    pub sha256: String,
    pub notes: String,
}

pub fn fetch_manifest(url: &str) -> Result<Value> {
    let low = url.to_lowercase();
    let txt = if let Some(p) = url.strip_prefix("file://").or_else(|| url.strip_prefix("file:")) {
        std::fs::read_to_string(p)?
    } else if low.starts_with("https://") {
        #[cfg(feature = "net")]
        {
            ureq::get(url).set("User-Agent", &format!("AstroForge/{}", crate::VERSION)).call().map_err(|e| anyhow::anyhow!("{}", e))?.into_string()?
        }
        #[cfg(not(feature = "net"))]
        bail!("Fonction réseau désactivée (compilez avec --features net)")
    } else {
        bail!("Adresse de mise à jour : https:// (ou file:) requis");
    };
    let v: Value = serde_json::from_str(&txt)?;
    if v.get("version").is_none() {
        bail!("Manifeste de mise à jour invalide");
    }
    Ok(v)
}

/// Some(info) si une version plus récente est publiée.
pub fn check(url: &str, local: &str) -> Result<Option<UpdateInfo>> {
    if url.is_empty() {
        return Ok(None);
    }
    let m = fetch_manifest(url)?;
    let ver = match &m["version"] {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    if !is_newer(&ver, local) {
        return Ok(None);
    }
    let plat = m.get(platform_key()).and_then(|v| v.as_object());
    let g = |k: &str| plat.and_then(|p| p.get(k)).or_else(|| m.get(k)).and_then(|v| v.as_str()).unwrap_or("").to_string();
    Ok(Some(UpdateInfo { version: ver, url: g("url"), sha256: g("sha256"), notes: m.get("notes").and_then(|v| v.as_str()).unwrap_or("").to_string() }))
}

/// Langue configurée (settings.json « language » ou ASTROFORGE_LANG).
pub fn configured_language() -> String {
    if let Ok(l) = std::env::var("ASTROFORGE_LANG") {
        if !l.is_empty() {
            return l;
        }
    }
    crate::equipment::load_settings().get("language").and_then(|v| v.as_str()).filter(|s| !s.is_empty()).unwrap_or("fr").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn versions() {
        assert!(is_newer("2.0.1", "2.0.0"));
        assert!(!is_newer("2.0.0", "2.0.0"));
        assert!(is_newer("2.0.0", "2.0.0rc1"));
        assert!(is_newer("2.0.10", "2.0.9"));
    }
}
