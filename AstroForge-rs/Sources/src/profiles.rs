//! Profils de paramètres par étape (profiles/<étape>/<nom>.json), profil
//! « standard » = valeurs par défaut.

use crate::paths;
use anyhow::{bail, Result};
use serde_json::{Map, Value};
use std::path::PathBuf;

fn dir_for(step: &str) -> PathBuf {
    let d = paths::profiles_root().join(step);
    let _ = std::fs::create_dir_all(&d);
    d
}

pub fn list_profiles(step: &str) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir_for(step))
        .map(|rd| rd.filter_map(|e| e.ok()).filter_map(|e| e.file_name().to_str().and_then(|n| n.strip_suffix(".json").map(|s| s.to_string()))).collect())
        .unwrap_or_default();
    if !names.iter().any(|n| n == "standard") {
        names.sort();
        let mut v = vec!["standard".to_string()];
        v.extend(names);
        return v;
    }
    names.sort_by(|a, b| (a != "standard", a).cmp(&(b != "standard", b)));
    names
}

pub fn load_profile(step: &str, name: &str, defaults: &Map<String, Value>) -> Map<String, Value> {
    let mut base = defaults.clone();
    if name == "standard" {
        return base;
    }
    if let Ok(t) = std::fs::read_to_string(dir_for(step).join(format!("{}.json", name))) {
        if let Ok(Value::Object(m)) = serde_json::from_str::<Value>(&t) {
            base.extend(m);
        }
    }
    base
}

pub fn save_profile(step: &str, name: &str, params: &Map<String, Value>) -> Result<()> {
    let name = name.trim();
    if name.is_empty() {
        bail!("Nom de profil vide");
    }
    if name == "standard" {
        bail!("Le profil « standard » est protégé — enregistrez sous un autre nom");
    }
    std::fs::write(dir_for(step).join(format!("{}.json", name)), serde_json::to_string_pretty(params)?)?;
    Ok(())
}

pub fn delete_profile(step: &str, name: &str) -> Result<()> {
    let p = dir_for(step).join(format!("{}.json", name));
    if p.exists() {
        std::fs::remove_file(p)?;
    }
    Ok(())
}
