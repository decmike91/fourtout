//! Emplacements des données de l'application (réglages, profils, masters,
//! modèles IA) :
//! - variable d'environnement ASTROFORGE_HOME si définie ;
//! - sinon mode « portable » : dossier de l'exécutable s'il contient un
//!   sous-dossier `profiles/` inscriptible ;
//! - sinon le dossier de configuration de l'utilisateur :
//!   Windows %APPDATA%\AstroForge, macOS ~/Library/Application Support/AstroForge,
//!   Linux $XDG_CONFIG_HOME/astroforge (~/.config/astroforge).

use std::path::PathBuf;

fn home() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn platform_dir() -> PathBuf {
    if cfg!(windows) {
        let base = std::env::var_os("APPDATA").map(PathBuf::from).unwrap_or_else(|| home().join("AppData").join("Roaming"));
        return base.join("AstroForge");
    }
    if cfg!(target_os = "macos") {
        return home().join("Library").join("Application Support").join("AstroForge");
    }
    let base = std::env::var_os("XDG_CONFIG_HOME").map(PathBuf::from).unwrap_or_else(|| home().join(".config"));
    base.join("astroforge")
}

fn writable(p: &std::path::Path) -> bool {
    std::fs::metadata(p).map(|m| !m.permissions().readonly()).unwrap_or(false)
}

/// Dossier racine des données utilisateur.
pub fn data_root() -> PathBuf {
    if let Some(env) = std::env::var_os("ASTROFORGE_HOME") {
        let p = PathBuf::from(env);
        return std::path::absolute(&p).unwrap_or(p);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let prof = dir.join("profiles");
            if prof.is_dir() && writable(&prof) {
                return dir.to_path_buf();
            }
        }
    }
    platform_dir()
}

pub fn profiles_root() -> PathBuf {
    data_root().join("profiles")
}

pub fn user_models_dir() -> PathBuf {
    data_root().join("models").join("ai")
}

/// Modèles livrés à côté de l'exécutable (models/ai).
pub fn app_models_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(|d| d.join("models").join("ai")))
        .unwrap_or_else(|| PathBuf::from("models/ai"))
}

pub fn masters_dir() -> PathBuf {
    data_root().join("masters")
}
