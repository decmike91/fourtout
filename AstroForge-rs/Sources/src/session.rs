//! Sessions d'acquisition : tri automatique (lumières, darks, flats, darks
//! de flats, bias) d'après les en-têtes FITS (IMAGETYP, EXPTIME, FILTER,
//! GAIN, CCD-TEMP, XBINNING, OBJECT…) ou les noms de fichiers et dossiers,
//! puis appariement des calibrations avec chaque groupe de lumières.

use crate::io::{self, HVal, Header};
use crate::stats::fmt_g;
use regex::Regex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub const FRAME_TYPES: [&str; 5] = ["light", "dark", "flat", "darkflat", "bias"];

fn type_words(kind: &str) -> &'static [&'static str] {
    match kind {
        "light" => &["light", "lightframe", "object", "science"],
        "dark" => &["dark", "darkframe"],
        "flat" => &["flat", "flatfield", "flatframe", "skyflat", "domeflat"],
        "darkflat" => &["darkflat", "flatdark", "darkflats", "flatdarks", "dflat"],
        _ => &["bias", "biasframe", "offset", "offsetframe", "zero"],
    }
}

fn name_hints() -> &'static [(&'static str, Regex)] {
    static H: OnceLock<Vec<(&'static str, Regex)>> = OnceLock::new();
    H.get_or_init(|| {
        vec![
            ("darkflat", Regex::new(r"(?i)(dark[\s_-]*flat|flat[\s_-]*dark|dflat)").unwrap()),
            ("bias", Regex::new(r"(?i)(bias|offset)").unwrap()),
            ("flat", Regex::new(r"(?i)flat").unwrap()),
            ("dark", Regex::new(r"(?i)dark").unwrap()),
            ("light", Regex::new(r"(?i)(light|lum|object|brut|target)").unwrap()),
        ]
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct FrameInfo {
    pub path: PathBuf,
    pub kind: String,
    pub exptime: Option<f64>,
    pub filter: String,
    pub gain: Option<f64>,
    pub offset: Option<f64>,
    pub temp: Option<f64>,
    pub binning: i64,
    pub width: Option<usize>,
    pub height: Option<usize>,
    pub date_obs: String,
    pub object: String,
    pub bayer: Option<String>,
    pub source: String,
}

/// Clé de regroupement des lumières.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GroupKey {
    pub object: String,
    pub filter: String,
    pub exptime: Option<i64>, // ×100 arrondi
    pub gain: Option<i64>,
    pub binning: i64,
    pub width: Option<usize>,
    pub height: Option<usize>,
}

fn round2(v: Option<f64>) -> Option<i64> {
    v.map(|x| (x * 100.0).round() as i64)
}

impl FrameInfo {
    pub fn key(&self) -> GroupKey {
        GroupKey {
            object: if self.object.is_empty() { "?".into() } else { self.object.clone() },
            filter: self.filter.clone(),
            exptime: round2(self.exptime),
            gain: round2(self.gain),
            binning: self.binning,
            width: self.width,
            height: self.height,
        }
    }
}

fn num(h: &Header, keys: &[&str]) -> Option<f64> {
    for k in keys {
        if let Some(v) = h.get(k) {
            if let Some(x) = v.as_f64() {
                return Some(x);
            }
        }
    }
    None
}

fn norm_word(v: &str) -> String {
    v.chars().filter(|c| !c.is_whitespace() && *c != '_' && *c != '-').collect::<String>().to_lowercase()
}

fn stem(s: &str) -> String {
    Path::new(s).file_stem().map(|x| x.to_string_lossy().to_string()).unwrap_or_else(|| s.to_string())
}

/// Type d'image -> (type, origine) : IMAGETYP/FRAME, puis nom du fichier,
/// puis noms des dossiers parents.
pub fn classify(path: &Path, h: &Header) -> (String, String) {
    for key in ["IMAGETYP", "FRAME", "FRAMETYP", "IMAGETYPE"] {
        if let Some(v) = h.get(key) {
            let w = norm_word(&v.to_string());
            for kind in ["light", "dark", "flat", "darkflat", "bias"] {
                if type_words(kind).contains(&w.as_str()) {
                    return (kind.into(), "en-tête".into());
                }
            }
            if w.contains("flat") && w.contains("dark") {
                return ("darkflat".into(), "en-tête".into());
            }
            for kind in ["bias", "flat", "dark", "light"] {
                if type_words(kind).iter().any(|x| w.contains(x)) {
                    return (kind.into(), "en-tête".into());
                }
            }
        }
    }
    let mut parts: Vec<String> = vec![path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default()];
    if let Some(dir) = path.parent() {
        let comps: Vec<String> = dir.components().map(|c| c.as_os_str().to_string_lossy().to_string()).collect();
        parts.extend(comps.iter().rev().take(3).cloned());
    }
    for part in &parts {
        let s = stem(part);
        for (kind, rx) in name_hints() {
            if rx.is_match(&s) {
                return (kind.to_string(), "nom".into());
            }
        }
    }
    ("light".into(), "défaut".into())
}

fn filter_alias(tok: &str) -> Option<&'static str> {
    Some(match tok {
        "l" | "lum" | "luminance" => "L",
        "r" | "red" | "rouge" => "R",
        "g" | "green" | "vert" => "G",
        "b" | "blue" | "bleu" => "B",
        "ha" | "halpha" | "h-alpha" => "Ha",
        "oiii" | "o3" => "OIII",
        "sii" | "s2" => "SII",
        _ => return None,
    })
}

/// Filtre déduit du nom (« flats_R », « M31_Ha_300s.fits ») ; vide sinon.
pub fn infer_filter(path: &Path) -> String {
    static SPLIT: OnceLock<Regex> = OnceLock::new();
    let rx = SPLIT.get_or_init(|| Regex::new(r"[\s_\-.()\[\]]+").unwrap());
    let parts = [
        path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default(),
        path.parent().and_then(|p| p.file_name()).map(|s| s.to_string_lossy().to_string()).unwrap_or_default(),
    ];
    for part in &parts {
        for tok in rx.split(part) {
            if let Some(a) = filter_alias(&tok.to_lowercase()) {
                return a.into();
            }
        }
    }
    String::new()
}

/// « 300s », « 120sec » dans un nom de fichier (lettre suivante interdite).
pub fn exptime_from_name(name: &str) -> Option<f64> {
    let b: Vec<char> = name.chars().collect();
    let n = b.len();
    for start in 0..n {
        if !b[start].is_ascii_digit() {
            continue;
        }
        let mut i = start;
        while i < n && b[i].is_ascii_digit() {
            i += 1;
        }
        let mut end_num = i;
        if i + 1 < n && b[i] == '.' && b[i + 1].is_ascii_digit() {
            let mut j = i + 1;
            while j < n && b[j].is_ascii_digit() {
                j += 1;
            }
            end_num = j;
        }
        for &e in &[end_num, i] {
            let mut k = e;
            while k < n && b[k].is_whitespace() {
                k += 1;
            }
            for suf in ["s", "sec", "secs"] {
                let sl = suf.len();
                if k + sl <= n && b[k..k + sl].iter().collect::<String>().to_lowercase() == suf {
                    let next = b.get(k + sl);
                    if next.map(|c| !c.is_ascii_alphabetic()).unwrap_or(true) {
                        let s: String = b[start..e].iter().collect();
                        if let Ok(v) = s.parse::<f64>() {
                            return Some(v);
                        }
                    }
                }
            }
            if e == i {
                break;
            }
        }
    }
    None
}

pub fn frame_info(path: &Path) -> FrameInfo {
    let h = io::read_header(path);
    frame_info_with(path, &h)
}

pub fn frame_info_with(path: &Path, h: &Header) -> FrameInfo {
    let (kind, src) = classify(path, h);
    let binning = num(h, &["XBINNING", "BINNING"]);
    let mut info = FrameInfo {
        path: path.to_path_buf(),
        kind,
        source: src,
        exptime: num(h, &["EXPTIME", "EXPOSURE", "EXP"]),
        filter: h.get("FILTER").map(|v| v.to_string().trim().to_string()).unwrap_or_default(),
        gain: num(h, &["GAIN", "EGAIN_SET", "ISOSPEED"]),
        offset: num(h, &["OFFSET", "BLKLEVEL"]),
        temp: num(h, &["CCD-TEMP", "CCD_TEMP", "SENSOR-TEMP", "SET-TEMP"]),
        binning: binning.filter(|b| *b != 0.0).map(|b| b as i64).unwrap_or(1),
        width: h.get("NAXIS1").and_then(|v| v.as_i64()).map(|v| v as usize),
        height: h.get("NAXIS2").and_then(|v| v.as_i64()).map(|v| v as usize),
        date_obs: h.get("DATE-OBS").map(|v| v.to_string()).unwrap_or_default(),
        object: h.get("OBJECT").map(|v| v.to_string().trim().to_string()).unwrap_or_default(),
        bayer: h.get("BAYERPAT").map(|v| v.to_string().trim().to_uppercase()).filter(|s| !s.is_empty()),
    };
    if info.filter.is_empty() {
        info.filter = infer_filter(path);
    }
    if info.exptime.is_none() {
        let name = path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        info.exptime = exptime_from_name(&name);
    }
    if info.width.is_none() && !io::has_ext(path, &io::RAW_EXTENSIONS) {
        if let Some((w, hh)) = io::raster::dimensions(path) {
            info.width = Some(w);
            info.height = Some(hh);
        }
    }
    info
}

/// Inventaire des fichiers image d'un dossier (récursif).
pub fn scan_folder(folder: &Path, recursive: bool) -> Vec<FrameInfo> {
    let mut files: Vec<PathBuf> = Vec::new();
    fn walk(d: &Path, rec: bool, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(d) else { return };
        let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
        entries.sort();
        let (dirs, fs): (Vec<PathBuf>, Vec<PathBuf>) = entries.into_iter().partition(|p| p.is_dir());
        for f in fs {
            if io::has_ext(&f, &io::CALIB_EXTENSIONS) {
                out.push(f);
            }
        }
        if rec {
            for d in dirs {
                walk(&d, rec, out);
            }
        }
    }
    walk(folder, recursive, &mut files);
    use rayon::prelude::*;
    files.par_iter().map(|p| frame_info(p)).collect()
}

#[derive(Clone, Debug)]
pub struct LightGroup {
    pub key: GroupKey,
    pub key_exptime: Option<f64>,
    pub key_gain: Option<f64>,
    pub lights: Vec<FrameInfo>,
    pub darks: Vec<FrameInfo>,
    pub flats: Vec<FrameInfo>,
    pub darkflats: Vec<FrameInfo>,
    pub biases: Vec<FrameInfo>,
    pub dark_scale: f64,
    pub warnings: Vec<String>,
}

impl LightGroup {
    pub fn name(&self) -> String {
        let mut parts = vec![self.key.object.clone()];
        if !self.key.filter.is_empty() {
            parts.push(format!("filtre {}", self.key.filter));
        }
        if let Some(e) = self.key_exptime {
            parts.push(format!("{} s", fmt_g(e)));
        }
        if let Some(g) = self.key_gain {
            parts.push(format!("gain {}", fmt_g(g)));
        }
        if self.key.binning != 0 && self.key.binning != 1 {
            parts.push(format!("bin {}", self.key.binning));
        }
        parts.join(" · ")
    }

    pub fn summary(&self) -> String {
        let mut t = format!(
            "{} : {} lumières, {} darks, {} flats, {} darks de flats, {} bias",
            self.name(),
            self.lights.len(),
            self.darks.len(),
            self.flats.len(),
            self.darkflats.len(),
            self.biases.len()
        );
        if (self.dark_scale - 1.0).abs() > 1e-6 {
            t += &format!(" (dark ×{:.3})", self.dark_scale);
        }
        if !self.warnings.is_empty() {
            t += &format!("\n  ⚠ {}", self.warnings.join("\n  ⚠ "));
        }
        t
    }

    pub fn list(&self, kind: &str) -> &Vec<FrameInfo> {
        match kind {
            "light" => &self.lights,
            "dark" => &self.darks,
            "flat" => &self.flats,
            "darkflat" => &self.darkflats,
            _ => &self.biases,
        }
    }

    pub fn paths(&self, kind: &str) -> Vec<PathBuf> {
        self.list(kind).iter().map(|f| f.path.clone()).collect()
    }

    pub fn calib_paths(&self) -> HashMap<String, Vec<PathBuf>> {
        ["bias", "dark", "darkflat", "flat"].iter().map(|k| (k.to_string(), self.paths(k))).collect()
    }
}

#[derive(Clone, Debug)]
pub struct SessionPlan {
    pub frames: Vec<FrameInfo>,
    pub groups: Vec<LightGroup>,
    pub unused: Vec<FrameInfo>,
}

impl SessionPlan {
    pub fn summary(&self) -> String {
        let counts: Vec<String> = FRAME_TYPES
            .iter()
            .map(|k| (k, self.frames.iter().filter(|f| f.kind == *k).count()))
            .filter(|(_, v)| *v > 0)
            .map(|(k, v)| format!("{} {}", v, k))
            .collect();
        let mut s = format!("{} fichiers ({}) — {} groupe(s) de lumières\n", self.frames.len(), counts.join(", "), self.groups.len());
        s += &self.groups.iter().map(|g| g.summary()).collect::<Vec<_>>().join("\n");
        s
    }
}

fn same_geometry(a: &FrameInfo, b: &FrameInfo) -> bool {
    if a.binning != b.binning {
        return false;
    }
    if let (Some(aw), Some(bw), Some(ah), Some(bh)) = (a.width, b.width, a.height, b.height) {
        return (aw, ah) == (bw, bh);
    }
    true
}

fn gain_ok(a: &FrameInfo, b: &FrameInfo) -> bool {
    match (a.gain, b.gain) {
        (Some(x), Some(y)) => (x - y).abs() < 0.5,
        _ => true,
    }
}

fn temp_ok(a: &FrameInfo, b: &FrameInfo, tol: f64) -> bool {
    match (a.temp, b.temp) {
        (Some(x), Some(y)) => (x - y).abs() <= tol,
        _ => true,
    }
}

/// Associe à un groupe de lumières ses images de calibration.
pub fn match_group(mut g: LightGroup, frames: &[FrameInfo], temp_tol: f64, exp_tol: f64) -> LightGroup {
    let r = g.lights[0].clone();
    let mut darks: Vec<FrameInfo> =
        frames.iter().filter(|f| f.kind == "dark" && same_geometry(f, &r) && gain_ok(f, &r) && temp_ok(f, &r, temp_tol)).cloned().collect();
    let biases: Vec<FrameInfo> = frames.iter().filter(|f| f.kind == "bias" && same_geometry(f, &r) && gain_ok(f, &r)).cloned().collect();
    if let (Some(rexp), false) = (r.exptime, darks.is_empty()) {
        let exact: Vec<FrameInfo> = darks
            .iter()
            .filter(|f| f.exptime.map(|e| (e - rexp).abs() <= exp_tol * rexp.max(1e-3)).unwrap_or(false))
            .cloned()
            .collect();
        if !exact.is_empty() {
            darks = exact;
        } else {
            let with_exp: Vec<FrameInfo> = darks.iter().filter(|f| f.exptime.map(|e| e != 0.0).unwrap_or(false)).cloned().collect();
            if !with_exp.is_empty() {
                let mut exps: Vec<f64> = with_exp.iter().map(|f| f.exptime.unwrap()).collect();
                exps.sort_by(|a, b| a.total_cmp(b));
                exps.dedup();
                let best = exps.iter().cloned().min_by(|a, b| (a - rexp).abs().total_cmp(&(b - rexp).abs())).unwrap();
                darks = with_exp.into_iter().filter(|f| f.exptime == Some(best)).collect();
                if !biases.is_empty() {
                    g.dark_scale = rexp / best;
                    g.warnings.push(format!(
                        "darks de {} s pour des lumières de {} s : dark thermique mis à l'échelle ×{:.3}",
                        fmt_g(best),
                        fmt_g(rexp),
                        g.dark_scale
                    ));
                } else {
                    g.warnings.push(format!(
                        "darks de {} s pour des lumières de {} s et aucun bias : mise à l'échelle impossible (dark utilisé tel quel)",
                        fmt_g(best),
                        fmt_g(rexp)
                    ));
                }
            }
        }
    }
    if let (false, Some(rt)) = (darks.is_empty(), r.temp) {
        let temps: Vec<f64> = darks.iter().filter_map(|f| f.temp).collect();
        if !temps.is_empty() && temps.iter().map(|t| (t - rt).abs()).fold(f64::INFINITY, f64::min) > 1.0 {
            g.warnings.push("température des darks différente de plus de 1 °C".into());
        }
    }
    let mut flats: Vec<FrameInfo> = frames.iter().filter(|f| f.kind == "flat" && same_geometry(f, &r)).cloned().collect();
    if !r.filter.is_empty() {
        let same: Vec<FrameInfo> = flats.iter().filter(|f| f.filter == r.filter).cloned().collect();
        flats = if !same.is_empty() { same } else { flats.into_iter().filter(|f| f.filter.is_empty()).collect() };
    }
    let mut darkflats = Vec::new();
    if !flats.is_empty() {
        let fexp = flats[0].exptime;
        darkflats = frames
            .iter()
            .filter(|f| {
                f.kind == "darkflat"
                    && same_geometry(f, &r)
                    && match (fexp, f.exptime) {
                        (Some(fe), Some(e)) => (e - fe).abs() <= exp_tol * fe.max(1e-3) + 0.01,
                        _ => true,
                    }
            })
            .cloned()
            .collect();
    }
    if darks.is_empty() {
        g.warnings.push("aucun dark compatible (gain/binning/taille/température)".into());
    }
    if flats.is_empty() {
        g.warnings.push("aucun flat compatible (filtre/binning/taille)".into());
    }
    g.darks = darks;
    g.biases = biases;
    g.flats = flats;
    g.darkflats = darkflats;
    g
}

/// Tri complet d'une session.
pub fn plan_session(folder: &Path, recursive: bool, temp_tol: f64) -> SessionPlan {
    plan_frames(scan_folder(folder, recursive), temp_tol)
}

pub fn plan_frames(frames: Vec<FrameInfo>, temp_tol: f64) -> SessionPlan {
    let mut order: Vec<GroupKey> = Vec::new();
    let mut groups: HashMap<GroupKey, LightGroup> = HashMap::new();
    for f in frames.iter().filter(|f| f.kind == "light") {
        let k = f.key();
        let g = groups.entry(k.clone()).or_insert_with(|| {
            order.push(k.clone());
            LightGroup {
                key: k.clone(),
                key_exptime: f.exptime.map(|e| (e * 100.0).round() / 100.0),
                key_gain: f.gain.map(|e| (e * 100.0).round() / 100.0),
                lights: vec![],
                darks: vec![],
                flats: vec![],
                darkflats: vec![],
                biases: vec![],
                dark_scale: 1.0,
                warnings: vec![],
            }
        });
        g.lights.push(f.clone());
    }
    let mut out: Vec<LightGroup> = order.into_iter().map(|k| match_group(groups.remove(&k).unwrap(), &frames, temp_tol, 0.05)).collect();
    out.sort_by(|a, b| b.lights.len().cmp(&a.lights.len()).then_with(|| a.name().cmp(&b.name())));
    let used: std::collections::HashSet<PathBuf> =
        out.iter().flat_map(|g| FRAME_TYPES.iter().flat_map(move |k| g.list(k).iter().map(|f| f.path.clone()))).collect();
    let unused = frames.iter().filter(|f| !used.contains(&f.path)).cloned().collect();
    SessionPlan { frames, groups: out, unused }
}

/// En-tête FITS d'une image empilée.
pub fn stack_header(light_paths: &[PathBuf], weights: Option<&[f64]>, extra: Option<&Header>) -> Header {
    let infos: Vec<FrameInfo> = light_paths.iter().map(|p| frame_info(p)).collect();
    let used: Vec<&FrameInfo> = infos
        .iter()
        .enumerate()
        .filter(|(i, _)| weights.map(|w| w.get(*i).cloned().unwrap_or(1.0) > 0.0).unwrap_or(true))
        .map(|x| x.1)
        .collect();
    let mut h = Header::new();
    let v = format!("AstroForge {}", crate::VERSION);
    h.set("CREATOR", v.as_str());
    h.set("SWCREATE", v.as_str());
    h.set("DATE", chrono::Utc::now().format("%Y-%m-%dT%H:%M:%S").to_string());
    h.set("NCOMBINE", used.len());
    h.set("IMAGETYP", "Master Light");
    let exps: Vec<f64> = used.iter().filter_map(|i| i.exptime).filter(|&e| e != 0.0).collect();
    if !exps.is_empty() {
        let tot: f64 = exps.iter().sum();
        h.set("EXPTIME", tot);
        h.set("EXPOSURE", tot);
        h.set("SUBEXP", exps[0]);
    }
    let mut dates: Vec<String> = used.iter().filter(|i| !i.date_obs.is_empty()).map(|i| i.date_obs.clone()).collect();
    dates.sort();
    if !dates.is_empty() {
        h.set("DATE-OBS", dates[0].as_str());
        h.set("DATE-END", dates.last().unwrap().as_str());
    }
    for (key, vals) in [
        ("OBJECT", used.iter().map(|i| i.object.clone()).collect::<Vec<_>>()),
        ("FILTER", used.iter().map(|i| i.filter.clone()).collect::<Vec<_>>()),
    ] {
        let set: std::collections::BTreeSet<String> = vals.into_iter().filter(|s| !s.is_empty()).collect();
        if set.len() == 1 {
            h.set(key, set.into_iter().next().unwrap());
        }
    }
    let gains: Vec<f64> = used.iter().filter_map(|i| i.gain).collect();
    if !gains.is_empty() && gains.iter().all(|g| *g == gains[0]) {
        h.set("GAIN", gains[0]);
    }
    if let Some(p0) = light_paths.first() {
        let first = io::read_header(p0);
        for k in ["INSTRUME", "TELESCOP", "FOCALLEN", "XPIXSZ", "YPIXSZ", "OBSERVER", "SITELAT", "SITELONG"] {
            if let Some(v) = first.get(k) {
                h.set(k, v.clone());
            }
        }
    }
    if let Some(e) = extra {
        h.extend(e);
    }
    h
}

pub fn hval_str(v: &HVal) -> String {
    v.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn names() {
        let h = Header::new();
        assert_eq!(classify(Path::new("/n/M31/darks/d_001.fits"), &h).0, "dark");
        assert_eq!(classify(Path::new("/n/M31/flat_dark/a.fits"), &h).0, "darkflat");
        assert_eq!(classify(Path::new("/n/M31/Offset/a.fits"), &h).0, "bias");
        assert_eq!(classify(Path::new("/n/x/a.fits"), &h).0, "light");
        assert_eq!(exptime_from_name("M31_Ha_300s.fits"), Some(300.0));
        assert_eq!(exptime_from_name("dark_120sec_x.fits"), Some(120.0));
        assert_eq!(exptime_from_name("img_30secret.fits"), None);
        assert_eq!(exptime_from_name("a_1.5s.fit"), Some(1.5));
        assert_eq!(infer_filter(Path::new("/x/flats_R/f1.fits")), "R");
        assert_eq!(infer_filter(Path::new("/x/M31_Ha_300s.fits")), "Ha");
        let mut h = Header::new();
        h.set("IMAGETYP", "Dark Flat");
        assert_eq!(classify(Path::new("a.fits"), &h).0, "darkflat");
        h.set("IMAGETYP", "Light Frame");
        assert_eq!(classify(Path::new("a.fits"), &h).0, "light");
    }
}
