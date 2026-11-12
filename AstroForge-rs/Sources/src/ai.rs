//! IA locale : modèles ONNX (models/ai/*.onnx + .json de configuration,
//! inférence par tuiles avec fondu ; moteur pur Rust « tract », option
//! `onnx`) et outils externes appelés en sous-processus (GraXpert,
//! StarNet2, DeepSNR). Catalogue de modèles téléchargeables (SHA-256 vérifié).

use crate::image::Image;
use crate::io;
use crate::paths;
use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

pub const TASKS: [&str; 6] = ["denoise", "background", "sharpen", "superres", "satellite", "starless"];

pub fn task_label(t: &str) -> &'static str {
    match t {
        "denoise" => "Débruitage",
        "background" => "Extraction de fond",
        "sharpen" => "Netteté / déconvolution",
        "superres" => "Super-résolution",
        "satellite" => "Traces de satellites",
        "starless" => "Suppression des étoiles",
        _ => "?",
    }
}

#[derive(Clone, Debug)]
pub struct Model {
    pub name: String,
    pub path: PathBuf,
    pub config: Map<String, Value>,
}

/// Modèles ONNX disponibles (dossier utilisateur puis dossier du programme).
pub fn list_models() -> Vec<Model> {
    let mut dirs = vec![paths::user_models_dir()];
    let app = paths::app_models_dir();
    if app != dirs[0] {
        dirs.push(app);
    }
    let mut out = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for d in dirs {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        let mut files: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
        files.sort();
        for f in files {
            let name = f.file_name().unwrap().to_string_lossy().to_string();
            if !name.to_lowercase().ends_with(".onnx") || !seen.insert(name.clone()) {
                continue;
            }
            let mut cfg = match json!({"task": "generic", "input_size": 256, "overlap": 32, "scale": 1, "norm": {"scale": 1.0, "offset": 0.0}}) {
                Value::Object(m) => m,
                _ => unreachable!(),
            };
            let j = f.with_extension("json");
            if let Ok(txt) = std::fs::read_to_string(&j) {
                match serde_json::from_str::<Value>(&txt) {
                    Ok(Value::Object(m)) => cfg.extend(m),
                    Ok(_) => {}
                    Err(e) => {
                        cfg.insert("config_error".into(), json!(e.to_string()));
                    }
                }
            }
            out.push(Model { name, path: f, config: cfg });
        }
    }
    out
}

/// La tâche doit être déclarée explicitement (« deconv » = netteté).
pub fn model_supports_task(m: &Model, task: &str) -> bool {
    let d = m.config.get("task").and_then(|v| v.as_str()).unwrap_or("");
    d == task || (d == "deconv" && task == "sharpen")
}

fn cfg_f(c: &Map<String, Value>, k: &str, d: f64) -> f64 {
    c.get(k).and_then(|v| v.as_f64()).unwrap_or(d)
}

#[cfg_attr(not(feature = "onnx"), allow(dead_code))]
fn norm(c: &Map<String, Value>) -> (f32, f32) {
    let n = c.get("norm").and_then(|v| v.as_object());
    (n.and_then(|m| m.get("scale")).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32, n.and_then(|m| m.get("offset")).and_then(|v| v.as_f64()).unwrap_or(0.0) as f32)
}

/// Inférence d'une tuile : NCHW -> NCHW (échelle `scale`).
pub trait TileModel {
    fn run_tile(&self, tile: &Image) -> Result<Image>;
}

/// Inférence par tuiles avec recouvrement et fondu triangulaire.
pub fn run_tiled(model: &dyn TileModel, cfg: &Map<String, Value>, img: &Image) -> Result<Image> {
    let size = cfg_f(cfg, "input_size", 256.0) as usize;
    let ov = cfg_f(cfg, "overlap", 32.0) as usize;
    let scale = cfg_f(cfg, "scale", 1.0) as usize;
    if size < 1 || ov >= size || scale < 1 {
        bail!("ONNX : input_size, overlap ou scale invalide.");
    }
    if !(img.c == 1 || img.c == 3) {
        bail!("ONNX : image mono ou RGB attendue.");
    }
    if img.is_empty() || !img.all_finite() {
        bail!("ONNX : image vide ou non finie.");
    }
    let (w, h) = (img.w, img.h);
    let check = |r: &Image, t: &Image| -> Result<()> {
        if r.c != t.c {
            bail!("ONNX : canaux de sortie incompatibles avec l'image mono/RGB.");
        }
        if (r.w, r.h) != (t.w * scale, t.h * scale) {
            bail!("ONNX : taille de sortie {}×{} incompatible avec l'échelle déclarée {} (attendue {}×{}).", r.w, r.h, scale, t.w * scale, t.h * scale);
        }
        Ok(())
    };
    if h <= size && w <= size {
        let r = model.run_tile(img)?;
        check(&r, img)?;
        return Ok(r.clip01());
    }
    let mut ys: Vec<usize> = (0..=h.saturating_sub(size)).step_by(size - ov).collect();
    let mut xs: Vec<usize> = (0..=w.saturating_sub(size)).step_by(size - ov).collect();
    if ys.is_empty() {
        ys.push(0);
    }
    if xs.is_empty() {
        xs.push(0);
    }
    if ys.last().unwrap() + size < h {
        ys.push(h - size);
    }
    if xs.last().unwrap() + size < w {
        xs.push(w - size);
    }
    let c = img.c;
    let (ow, oh) = (w * scale, h * scale);
    let mut out = vec![0.0f32; ow * oh * c];
    let mut wsum = vec![0.0f32; ow * oh];
    for &ty in &ys {
        for &tx in &xs {
            let tile = img.crop(tx, ty, size, size);
            let r = model.run_tile(&tile)?;
            check(&r, &tile)?;
            let (rw, rh) = (r.w, r.h);
            let ramp = |n: usize| -> Vec<f32> {
                let mut v = vec![1.0f32; n];
                if ov > 0 && n >= 2 {
                    let l = (ov * scale).min(n / 2);
                    if l > 0 {
                        let rp = crate::stats::linspace(0.05, 1.0, l);
                        for i in 0..l {
                            v[i] = v[i].min(rp[i] as f32);
                            v[n - 1 - i] = v[n - 1 - i].min(rp[i] as f32);
                        }
                    }
                }
                v
            };
            let (wy, wx) = (ramp(rh), ramp(rw));
            let (oy, ox) = (ty * scale, tx * scale);
            for y in 0..rh {
                for x in 0..rw {
                    let wv = wy[y] * wx[x];
                    let o = (oy + y) * ow + ox + x;
                    wsum[o] += wv;
                    for k in 0..c {
                        out[o * c + k] += r.data[(y * rw + x) * c + k].clamp(0.0, 1.0) * wv;
                    }
                }
            }
        }
    }
    for i in 0..ow * oh {
        let s = wsum[i].max(1e-6);
        for k in 0..c {
            out[i * c + k] = (out[i * c + k] / s).clamp(0.0, 1.0);
        }
    }
    Ok(Image::from_vec(ow, oh, c, out))
}

#[cfg(feature = "onnx")]
mod onnx_impl {
    use super::*;
    use tract_onnx::prelude::*;

    type Plan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

    pub struct OnnxRunner {
        model_path: PathBuf,
        cfg: Map<String, Value>,
        cache: std::sync::Mutex<std::collections::HashMap<(usize, usize, usize), std::sync::Arc<Plan>>>,
    }

    impl OnnxRunner {
        pub fn new(path: &Path, cfg: &Map<String, Value>) -> Result<Self> {
            Ok(OnnxRunner { model_path: path.to_path_buf(), cfg: cfg.clone(), cache: Default::default() })
        }
        fn plan(&self, c: usize, h: usize, w: usize) -> Result<std::sync::Arc<Plan>> {
            let mut cache = self.cache.lock().unwrap();
            if let Some(p) = cache.get(&(c, h, w)) {
                return Ok(p.clone());
            }
            let model = tract_onnx::onnx()
                .model_for_path(&self.model_path)
                .map_err(|e| anyhow!("Modèle ONNX invalide : {}", e))?
                .with_input_fact(0, f32::fact([1, c, h, w]).into())
                .map_err(|e| anyhow!("{}", e))?
                .into_optimized()
                .map_err(|e| anyhow!("{}", e))?
                .into_runnable()
                .map_err(|e| anyhow!("{}", e))?;
            let p = std::sync::Arc::new(model);
            cache.insert((c, h, w), p.clone());
            Ok(p)
        }
    }

    impl TileModel for OnnxRunner {
        fn run_tile(&self, tile: &Image) -> Result<Image> {
            let (ns, no) = norm(&self.cfg);
            let (c, h, w) = (tile.c, tile.h, tile.w);
            let plan = self.plan(c, h, w)?;
            let mut v = vec![0.0f32; c * h * w];
            for k in 0..c {
                for i in 0..h * w {
                    v[k * h * w + i] = tile.data[i * c + k] * ns + no;
                }
            }
            let input: Tensor = tract_ndarray::Array4::from_shape_vec((1, c, h, w), v)?.into();
            let res = plan.run(tvec!(input.into())).map_err(|e| anyhow!("ONNX : {}", e))?;
            let y = res[0].to_array_view::<f32>().map_err(|e| anyhow!("ONNX : {}", e))?;
            let sh = y.shape().to_vec();
            let (oc, oh, ow) = match sh.len() {
                4 => {
                    if sh[0] != 1 {
                        bail!("ONNX : sortie avec plusieurs images, attendue : une image NCHW.");
                    }
                    (sh[1], sh[2], sh[3])
                }
                3 => (sh[0], sh[1], sh[2]),
                2 => (1, sh[0], sh[1]),
                _ => bail!("ONNX : canaux de sortie incompatibles avec l'image mono/RGB."),
            };
            let flat: Vec<f32> = y.iter().cloned().collect();
            if !flat.iter().all(|v| v.is_finite()) || ns == 0.0 {
                bail!("ONNX : sortie ou normalisation invalide.");
            }
            let mut out = Image::new(ow, oh, oc);
            for k in 0..oc {
                for i in 0..oh * ow {
                    out.data[i * oc + k] = ((flat[k * oh * ow + i] - no) / ns).clamp(0.0, 1.0);
                }
            }
            Ok(out)
        }
    }
}

fn run_onnx(m: &Model, img: &Image) -> Result<Image> {
    #[cfg(feature = "onnx")]
    {
        let r = onnx_impl::OnnxRunner::new(&m.path, &m.config)?;
        run_tiled(&r, &m.config, img)
    }
    #[cfg(not(feature = "onnx"))]
    {
        let _ = (m, img);
        bail!("Inférence ONNX non compilée : recompilez avec « cargo build --release --features onnx »")
    }
}

// ------------------------------------------------------------------ outils
pub fn settings_path() -> PathBuf {
    paths::profiles_root().join("ai_tools.json")
}

pub fn default_tools() -> Map<String, Value> {
    match json!({
        "graxpert": {"label": "GraXpert", "binary": "", "extra_args": ""},
        "cosmicclarity": {"label": "Cosmic Clarity (SetiAstro)", "binary": "", "denoise": "", "sharpen": "", "superres": "", "satellite": ""},
        "starnet2": {"label": "StarNet2 (suppression des étoiles)", "binary": "", "args": ""},
        "deepsnr": {"label": "DeepSNR (débruitage)", "binary": "", "args": ""}
    }) {
        Value::Object(m) => m,
        _ => unreachable!(),
    }
}

/// Complète les sections ; migration du schéma 1 (chemin DeepSNR mal rangé).
pub fn normalize_tool_settings(raw: &Value) -> Map<String, Value> {
    let mut cfg = default_tools();
    if let Value::Object(m) = raw {
        for (k, v) in m {
            match (cfg.get_mut(k), v) {
                (Some(Value::Object(dst)), Value::Object(src)) => dst.extend(src.clone()),
                (Some(_), _) => {}
                (None, _) => {
                    cfg.insert(k.clone(), v.clone());
                }
            }
        }
    }
    let version = raw.get("schema_version").and_then(|v| v.as_i64()).unwrap_or(0);
    if version < 2 {
        let old = cfg["starnet2"].get("binary").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let name = old.replace('\\', "/").rsplit('/').next().unwrap_or("").to_lowercase();
        if !old.is_empty() && name.contains("deepsnr") && !name.contains("starnet") {
            if cfg["deepsnr"].get("binary").and_then(|v| v.as_str()).unwrap_or("").is_empty() {
                cfg["deepsnr"]["binary"] = json!(old);
            }
            cfg["starnet2"]["legacy_binary"] = json!(old);
            cfg["starnet2"]["binary"] = json!("");
        } else if !old.is_empty() && (!name.contains("starnet") || name.contains("deepsnr")) {
            cfg["starnet2"]["legacy_binary"] = json!(old);
            cfg["starnet2"]["binary"] = json!("");
        }
    }
    let d = default_tools();
    for key in ["starnet2", "deepsnr"] {
        cfg[key]["label"] = d[key]["label"].clone();
    }
    cfg.insert("schema_version".into(), json!(2));
    cfg
}

pub fn load_tool_settings() -> Map<String, Value> {
    let raw = std::fs::read_to_string(settings_path()).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()).unwrap_or(Value::Null);
    normalize_tool_settings(&raw)
}

pub fn save_tool_settings(cfg: &Map<String, Value>) -> Result<()> {
    let p = settings_path();
    std::fs::create_dir_all(p.parent().unwrap())?;
    std::fs::write(p, serde_json::to_string_pretty(&normalize_tool_settings(&Value::Object(cfg.clone())))?)?;
    Ok(())
}

fn resolve_exe(binary: &str, label: &str) -> Result<PathBuf> {
    let p = if let Some(rest) = binary.strip_prefix("~/") { dirs_home().join(rest) } else { PathBuf::from(binary) };
    if !binary.is_empty() && p.is_file() {
        return Ok(std::path::absolute(&p).unwrap_or(p));
    }
    if !binary.is_empty() {
        if let Some(w) = crate::astrometry::which(binary) {
            return Ok(w);
        }
    }
    bail!("Binaire {} introuvable ou non configuré (menu Outils IA).", label)
}

fn dirs_home() -> PathBuf {
    std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from).unwrap_or_default()
}

fn execute(cmd: &[String], timeout_s: u64) -> Result<()> {
    let exe = PathBuf::from(&cmd[0]);
    let mut child = std::process::Command::new(&exe)
        .args(&cmd[1..])
        .current_dir(exe.parent().unwrap_or(Path::new(".")))
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .with_context(|| format!("Impossible de lancer {}", cmd[0]))?;
    let t0 = std::time::Instant::now();
    loop {
        if let Some(st) = child.try_wait()? {
            if !st.success() {
                let out = child.wait_with_output()?;
                let mut detail = String::from_utf8_lossy(&out.stderr).to_string();
                if detail.trim().is_empty() {
                    detail = String::from_utf8_lossy(&out.stdout).to_string();
                }
                let tail: String = detail.chars().rev().take(2000).collect::<String>().chars().rev().collect();
                bail!("{} : code {:?} — {}", exe.file_name().unwrap_or_default().to_string_lossy(), st.code(), tail.trim());
            }
            return Ok(());
        }
        if t0.elapsed().as_secs() > timeout_s {
            let _ = child.kill();
            bail!("{} : délai dépassé ({} s).", exe.file_name().unwrap_or_default().to_string_lossy(), timeout_s);
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}

fn with_temp(img: &Image, ext: &str, f: impl FnOnce(&Path, &Path) -> Result<Vec<PathBuf>>) -> Result<Image> {
    if !(img.c == 1 || img.c == 3) {
        bail!("L'outil externe accepte une image mono ou RGB uniquement.");
    }
    if img.is_empty() || !img.all_finite() {
        bail!("Image IA vide ou contenant des valeurs non finies.");
    }
    let d = tempfile::Builder::new().prefix("astroforge_ai_").tempdir()?;
    let inp = d.path().join(format!("in{}", ext));
    io::save(&inp, &img.clip01())?;
    let started = std::time::SystemTime::now() - std::time::Duration::from_secs(2);
    let outs = f(&inp, d.path())?;
    for o in outs {
        let Ok(md) = std::fs::metadata(&o) else { continue };
        if !md.is_file() || md.len() == 0 || o == inp || md.modified().map(|m| m < started).unwrap_or(true) || o.parent() != Some(d.path()) {
            continue;
        }
        let r = io::load(&o)?;
        if r.is_empty() || !r.all_finite() {
            bail!("L'outil externe a produit une image vide ou non finie.");
        }
        return Ok(r);
    }
    bail!("L'outil externe n'a pas produit de sortie nouvelle valide (l'entrée et les fichiers anciens sont exclus).")
}

/// Point d'entrée unique : tâche IA par fournisseur (onnx, onnx:<modèle>,
/// graxpert, starnet2, deepsnr).
pub fn run(task: &str, img: &Image, provider: &str, strength: f64, linear: bool) -> Result<Image> {
    if !strength.is_finite() || !(0.0..=1.0).contains(&strength) {
        bail!("Force IA attendue entre 0 et 1.");
    }
    if !TASKS.contains(&task) {
        bail!("Tâche IA inconnue : {}", task);
    }
    let st = load_tool_settings();
    if provider == "onnx" || provider.starts_with("onnx:") {
        let models = list_models();
        let name = match provider.split_once(':') {
            Some((_, n)) => Some(n.to_string()),
            None => models.iter().find(|m| model_supports_task(m, task)).map(|m| m.name.clone()),
        };
        let model = name.as_ref().and_then(|n| models.iter().find(|m| &m.name == n)).ok_or_else(|| {
            anyhow!(
                "Aucun modèle ONNX disponible pour « {} » dans {}. Ajoutez un modèle et ses métadonnées de tâche, ou configurez un outil externe.",
                task_label(task),
                paths::user_models_dir().display()
            )
        })?;
        if !model_supports_task(model, task) {
            bail!("Modèle ONNX {} incompatible avec la tâche {} (métadonnées absentes ou tâche différente).", model.name, task);
        }
        let out = run_onnx(model, img)?;
        if out.same_shape(img) && task != "superres" && strength < 1.0 {
            let s = strength as f32;
            return Ok(img.zip_map(&out, |a, b| a + s * (b - a)));
        }
        return Ok(out);
    }
    match provider {
        "graxpert" => {
            if task != "denoise" && task != "background" {
                bail!("GraXpert ne prend pas en charge la tâche {}.", task);
            }
            let g = &st["graxpert"];
            let exe = resolve_exe(g.get("binary").and_then(|v| v.as_str()).unwrap_or(""), "GraXpert")?;
            let extra: Vec<String> = g.get("extra_args").and_then(|v| v.as_str()).unwrap_or("").split_whitespace().map(|s| s.to_string()).collect();
            let reserved = ["-cli", "-cmd", "-output", "-strength", "-smoothing"];
            if extra.iter().any(|a| reserved.contains(&a.split('=').next().unwrap_or(""))) {
                bail!("GraXpert : extra_args ne doit pas modifier la tâche, le mode CLI, la force ou le chemin de sortie.");
            }
            with_temp(img, ".fits", |inp, d| {
                let mut cmd = vec![
                    exe.to_string_lossy().to_string(),
                    inp.to_string_lossy().to_string(),
                    "-cli".into(),
                    "-cmd".into(),
                    if task == "denoise" { "denoising" } else { "background-extraction" }.into(),
                    if task == "denoise" { "-strength" } else { "-smoothing" }.into(),
                    strength.to_string(),
                    "-output".into(),
                    d.join("out").to_string_lossy().to_string(),
                ];
                cmd.extend(extra);
                execute(&cmd, 1800)?;
                Ok(vec![d.join("out.fits"), d.join("out_GraXpert.fits")])
            })
        }
        "cosmicclarity" => bail!(
            "Cosmic Clarity : automatisation non prise en charge pour cette installation. Le workflow officiel Input/Output est interactif. Utilisez l'outil manuellement puis importez sa sortie ; aucun fichier utilisateur n'a été touché."
        ),
        "starnet2" | "deepsnr" => {
            let expected = if provider == "starnet2" { "starless" } else { "denoise" };
            if task != expected {
                bail!("{} est incompatible avec la tâche {} ; tâche attendue : {}.", provider, task, expected);
            }
            let label = if provider == "starnet2" { "StarNet2" } else { "DeepSNR" };
            let exe = resolve_exe(st[provider].get("binary").and_then(|v| v.as_str()).unwrap_or(""), label)?;
            if img.w.min(img.h) < 512 {
                bail!("{} : image trop petite, minimum 512 × 512 pixels.", label);
            }
            let ext = if provider == "starnet2" { ".tif" } else { ".fits" };
            let template = st[provider].get("args").and_then(|v| v.as_str()).unwrap_or("").to_string();
            with_temp(img, ext, |inp, d| {
                let outp = d.join(format!("out{}", ext));
                let (i, o) = (inp.to_string_lossy().to_string(), outp.to_string_lossy().to_string());
                let mut cmd = vec![exe.to_string_lossy().to_string()];
                if !template.is_empty() {
                    cmd.extend(template.split_whitespace().map(|a| a.replace("{input}", &i).replace("{output}", &o)));
                } else {
                    cmd.extend(["--input".to_string(), i.clone(), "--output".into(), o.clone()]);
                    if provider == "deepsnr" {
                        cmd.extend(["--model".to_string(), "2".into()]);
                    }
                }
                if linear {
                    cmd.push("--linear".into());
                }
                execute(&cmd, 1800)?;
                Ok(vec![outp])
            })
        }
        _ => bail!("Fournisseur IA inconnu : {}", provider),
    }
}

// ------------------------------------------------------------- catalogue
pub const REQUIRED: [&str; 5] = ["id", "name", "task", "url", "sha256"];

pub fn sha256_file(p: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(p)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}

fn fetch_text(src: &str) -> Result<String> {
    if Path::new(src).is_file() {
        return Ok(std::fs::read_to_string(src)?);
    }
    if let Some(p) = src.strip_prefix("file://") {
        return Ok(std::fs::read_to_string(p)?);
    }
    if !src.starts_with("https://") {
        bail!("Catalogue : URL https:// (ou fichier local) requise");
    }
    #[cfg(feature = "net")]
    {
        Ok(ureq::get(src).set("User-Agent", "AstroForge").call().map_err(|e| anyhow!("{}", e))?.into_string()?)
    }
    #[cfg(not(feature = "net"))]
    bail!("Fonction réseau désactivée (compilez avec --features net)")
}

pub fn load_catalog(source: &str) -> Result<Vec<Map<String, Value>>> {
    if source.is_empty() {
        return Ok(vec![]);
    }
    let v: Value = serde_json::from_str(&fetch_text(source)?)?;
    let models = v.get("models").and_then(|m| m.as_array()).cloned().unwrap_or_default();
    Ok(models
        .into_iter()
        .filter_map(|m| match m {
            Value::Object(o) if REQUIRED.iter().all(|k| o.get(*k).map(|x| !x.is_null() && x != "").unwrap_or(false)) => Some(o),
            _ => None,
        })
        .collect())
}

fn target_path(entry: &Map<String, Value>, dir: &Path) -> PathBuf {
    let id = entry.get("id").and_then(|v| v.as_str()).unwrap_or("");
    let mut name: String = id.chars().filter(|c| c.is_alphanumeric() || "-_.".contains(*c)).collect();
    if name.is_empty() {
        name = "modele".into();
    }
    dir.join(format!("{}.onnx", name))
}

pub fn is_installed(entry: &Map<String, Value>, dir: Option<&Path>) -> bool {
    target_path(entry, &dir.map(|d| d.to_path_buf()).unwrap_or_else(paths::user_models_dir)).exists()
}

/// Télécharge, vérifie (SHA-256) et installe un modèle + sa configuration.
#[cfg(feature = "net")]
pub fn download_model(entry: &Map<String, Value>, dir: Option<&Path>) -> Result<PathBuf> {
    use sha2::{Digest, Sha256};
    use std::io::{Read, Write};
    let dest = dir.map(|d| d.to_path_buf()).unwrap_or_else(paths::user_models_dir);
    std::fs::create_dir_all(&dest)?;
    let url = entry.get("url").and_then(|v| v.as_str()).unwrap_or("");
    if !url.starts_with("https://") && !url.starts_with("file://") {
        bail!("Téléchargement refusé : URL non https");
    }
    let target = target_path(entry, &dest);
    let mut tmp = tempfile::Builder::new().prefix(".dl_").suffix(".onnx").tempfile_in(&dest)?;
    let mut h = Sha256::new();
    if let Some(p) = url.strip_prefix("file://") {
        let data = std::fs::read(p)?;
        h.update(&data);
        tmp.write_all(&data)?;
    } else {
        let resp = ureq::get(url).set("User-Agent", "AstroForge").call().map_err(|e| anyhow!("{}", e))?;
        let mut r = resp.into_reader();
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = r.read(&mut buf)?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
            tmp.write_all(&buf[..n])?;
        }
    }
    let sum = format!("{:x}", h.finalize());
    if sum != entry.get("sha256").and_then(|v| v.as_str()).unwrap_or("").to_lowercase() {
        bail!("Somme de contrôle SHA-256 incorrecte : fichier rejeté");
    }
    tmp.persist(&target).map_err(|e| anyhow!("{}", e))?;
    let mut cfg = Map::new();
    for k in ["task", "input_size", "overlap", "scale", "norm", "input_name"] {
        if let Some(v) = entry.get(k) {
            cfg.insert(k.into(), v.clone());
        }
    }
    cfg.insert("source".into(), json!(url));
    cfg.insert("sha256".into(), entry["sha256"].clone());
    cfg.insert("license".into(), entry.get("license").cloned().unwrap_or(json!("")));
    cfg.insert("name".into(), entry["name"].clone());
    std::fs::write(target.with_extension("json"), serde_json::to_string_pretty(&cfg)?)?;
    Ok(target)
}
