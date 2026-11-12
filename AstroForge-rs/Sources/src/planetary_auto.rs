//! Planétaire entièrement automatique : moteur de décision IA local.
//!
//! Le moteur enchaîne, sans réglage manuel, les étapes d'un traitement
//! planétaire complet et choisit chaque paramètre lui-même :
//!
//! 1. **Analyse de la vidéo** : détection du disque (ou mode « surface » pour
//!    la Lune et le Soleil), centrage de chaque frame, rejet des frames
//!    inutilisables (nuage, planète coupée par le bord, saturation).
//! 2. **Tri et classement** : plusieurs mesures de netteté sont calculées ;
//!    leur robustesse au bruit est testée sur la vidéo elle-même et le
//!    classement final fusionne les mesures pondérées par leur fiabilité.
//! 3. **Sélection** : le nombre d'images empilées est choisi en maximisant un
//!    critique de qualité « sans référence » (voir ci-dessous) sur des
//!    empilements d'essai.
//! 4. **Alignement et empilement** : analyse de la turbulence (distorsion
//!    locale), essai des méthodes candidates (globale, multi-points,
//!    multipoint local façon AutoStakkert!) et des méthodes d'empilement
//!    (moyenne, moyenne pondérée, sigma-clipping) ; la meilleure est retenue.
//! 5. **Restauration** : PSF mesurée sur le limbe, ondelettes à trous réglées
//!    par un filtre de Wiener par couche (gain = 1/FTM × RSB²/(RSB²+1)),
//!    débruitage par seuillage doux adaptatif, protection du limbe contre les
//!    halos ; force et débruitage choisis sous contraintes (bruit invisible,
//!    pas d'anneau, pas d'écrêtage).
//! 6. **Finition** : alignement des canaux RVB (dispersion atmosphérique),
//!    équilibre des couleurs partiel, niveaux sans écrêtage (genou doux).
//!
//! **Critique de qualité.** Chaque essai d'empilement est fait en deux
//! moitiés indépendantes (frames paires / impaires du classement). Leur
//! demi-différence est une réalisation pure du bruit (et des erreurs
//! d'alignement) ; leur moyenne est le signal. Par couche d'ondelettes, on
//! obtient ainsi le rapport signal/bruit réel, sans hypothèse sur la caméra,
//! et le score est la « capacité » Σ log2(1 + S²/N). Le même principe mesure
//! le bruit réellement visible après accentuation.
//!
//! **Apprentissage local.** Chaque traitement est mémorisé (caractéristiques
//! de la vidéo + choix retenus) dans `ai/planetary_memory.json` du dossier
//! des données ; les vidéos ressemblantes réutilisent ces choix comme point
//! de départ (k plus proches voisins). Un modèle ONNX de débruitage installé
//! (`models/ai`, option `onnx`) peut être essayé : il n'est retenu que s'il
//! réduit le bruit sans perte de détail mesurable. Rien ne quitte la machine.

use crate::ai;
use crate::frames::Frames;
use crate::image::{mean_images, Image};
use crate::ndimage;
use crate::paths;
use crate::planetary::{self, AlignStats};
use crate::registration::{apply_shift, phase_corr};
use crate::stacking::{self, StackParams};
use crate::stats;
use crate::wavelets::atrous_decompose;
use anyhow::{bail, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Niveaux d'effort de la recherche.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Effort {
    /// règles expertes + mémoire, peu d'essais
    Rapide,
    /// essais des candidats les plus probables
    Equilibre,
    /// essai exhaustif des méthodes et réglages
    Maximal,
}

impl Effort {
    pub const ALL: [Effort; 3] = [Effort::Rapide, Effort::Equilibre, Effort::Maximal];

    pub fn label(&self) -> &'static str {
        match self {
            Effort::Rapide => "Rapide",
            Effort::Equilibre => "Équilibré",
            Effort::Maximal => "Maximal",
        }
    }

    pub fn parse(s: &str) -> Result<Effort> {
        let l = s.to_lowercase();
        if l.starts_with("rap") || l == "fast" {
            Ok(Effort::Rapide)
        } else if l.starts_with("éq") || l.starts_with("eq") || l == "balanced" {
            Ok(Effort::Equilibre)
        } else if l.starts_with("max") {
            Ok(Effort::Maximal)
        } else {
            bail!("Effort inconnu : {} (Rapide, Équilibré, Maximal)", s)
        }
    }
}

/// Options (volontairement réduites : tout le reste est décidé par le moteur).
#[derive(Clone, Debug)]
pub struct AutoOptions {
    pub effort: Effort,
    /// réutiliser les décisions de vidéos ressemblantes
    pub use_memory: bool,
    /// mémoriser ce traitement
    pub learn: bool,
    /// essayer un modèle ONNX de débruitage s'il est installé
    pub onnx: bool,
    /// équilibre automatique des couleurs (images couleur)
    pub color_balance: bool,
    /// interruption demandée par l'utilisateur
    pub cancel: Option<Arc<AtomicBool>>,
    /// fichier de mémoire (défaut : dossier des données)
    pub memory_path: Option<PathBuf>,
    /// plan imposé (traitement uniforme d'une série de vidéos)
    pub lock: Option<AutoLock>,
}

/// Plan verrouillé : mêmes choix pour toutes les vidéos d'une série
/// (time-lapse), pour un rendu identique d'une image à l'autre.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutoLock {
    /// fraction des frames valides empilées
    pub keep_fraction: f64,
    pub align_mode: String,
    pub box_size: usize,
    pub stack_method: String,
    pub strength: f64,
    pub denoise_k: f64,
    /// gains d'ondelettes communs (7 couches)
    pub wavelet_gains: Vec<f64>,
    /// gains de couleur communs (vide : aucun)
    pub color_gains: Vec<f64>,
}

impl AutoLock {
    pub fn from_result(r: &AutoResult) -> AutoLock {
        let valid = (r.features.n_frames - r.features.rejected).max(1);
        AutoLock {
            keep_fraction: (r.plan.keep as f64 / valid as f64).clamp(0.0, 1.0),
            align_mode: r.plan.align_mode.clone(),
            box_size: r.plan.box_size,
            stack_method: r.plan.stack_method.clone(),
            strength: r.plan.strength,
            denoise_k: r.plan.denoise_k,
            wavelet_gains: r.plan.wavelet_gains.clone(),
            color_gains: r.plan.color_gains.clone(),
        }
    }
}

impl Default for AutoOptions {
    fn default() -> Self {
        AutoOptions { effort: Effort::Equilibre, use_memory: true, learn: true, onnx: true, color_balance: true, cancel: None, memory_path: None, lock: None }
    }
}

/// Caractéristiques mesurées de la vidéo.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Features {
    pub n_frames: usize,
    pub width: usize,
    pub height: usize,
    pub color: bool,
    /// « disque » (planète entière) ou « surface » (Lune, Soleil, gros plan)
    pub mode: String,
    pub disc_radius: f64,
    pub crop: usize,
    /// bruit d'une frame (écart-type, échelle 0..1)
    pub noise: f64,
    /// rapport signal/bruit d'une frame (médiane du disque / bruit)
    pub snr: f64,
    pub brightness: f64,
    pub saturation: f64,
    /// dispersion relative de la qualité (variabilité du seeing)
    pub quality_cv: f64,
    /// agitation globale (px, écart-type du centre)
    pub jitter: f64,
    /// distorsion locale résiduelle (px rms) : turbulence
    pub distortion: f64,
    pub rejected: usize,
}

impl Features {
    /// Vecteur normalisé pour la recherche de vidéos ressemblantes.
    pub fn vector(&self) -> Vec<f64> {
        vec![
            (self.n_frames.max(1) as f64).ln() / 3.0,
            (self.disc_radius.max(1.0)).ln() / 1.5,
            (self.snr.max(0.5)).ln() / 1.5,
            self.quality_cv * 4.0,
            (1.0 + self.jitter).ln(),
            (1.0 + 4.0 * self.distortion).ln() * 1.5,
            if self.color { 1.0 } else { 0.0 },
            if self.mode == "surface" { 2.0 } else { 0.0 },
        ]
    }
}

/// Une décision du moteur, avec sa justification.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Decision {
    pub step: String,
    pub choice: String,
    pub reason: String,
}

/// Plan de traitement retenu (exportable en JSON, reproductible).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AutoPlan {
    pub quality_weights: Vec<(String, f64)>,
    pub keep: usize,
    pub keep_percent: f64,
    pub align_mode: String,
    pub box_size: usize,
    pub grid_n: usize,
    pub ap_percent: f64,
    pub stack_method: String,
    pub rgb_align: bool,
    pub rgb_shifts: Vec<(f64, f64)>,
    pub psf_sigma: f64,
    pub onnx_denoise: Option<String>,
    pub strength: f64,
    pub denoise_k: f64,
    pub wavelet_gains: Vec<f64>,
    pub limb_protection: bool,
    pub color_gains: Vec<f64>,
    pub black: f64,
    pub white: f64,
}

/// Indicateurs de qualité avant / après.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Metrics {
    /// score du critique sur l'empilement retenu
    pub stack_score: f64,
    /// RSB de l'empilement (médiane du disque / bruit)
    pub stack_snr: f64,
    /// gain de RSB par rapport à une frame
    pub snr_gain: f64,
    /// bruit visible après traitement (écart-type relatif au disque)
    pub noise_out: f64,
    /// énergie de détail (couches 1-4) après / avant restauration
    pub detail_gain: f64,
    /// fraction de pixels écrêtés
    pub clipped: f64,
    /// halo au limbe (relatif)
    pub ringing: f64,
}

pub struct AutoResult {
    /// image finale (recadrée et centrée sur l'objet)
    pub image: Image,
    /// image restaurée avant les niveaux (linéaire, 32 bits) : base des
    /// séries harmonisées
    pub linear: Image,
    /// disque dans le cadrage de l'image (cx, cy, r), s'il y en a un
    pub disc: Option<(f64, f64, f64)>,
    /// empilement brut (avant restauration)
    pub stack: Image,
    /// meilleure frame (même cadrage)
    pub best_frame: Image,
    pub features: Features,
    pub plan: AutoPlan,
    pub decisions: Vec<Decision>,
    /// qualité fusionnée de chaque frame (NaN : rejetée)
    pub scores: Vec<f64>,
    /// classement (meilleure d'abord, frames valides seulement)
    pub order: Vec<usize>,
    pub rejected: Vec<usize>,
    /// courbe (pourcentage gardé, score du critique)
    pub keep_curve: Vec<(f64, f64)>,
    /// essais d'alignement / empilement : (libellé, score)
    pub trials: Vec<(String, f64)>,
    /// essais de restauration : (force, débruitage, détail, bruit, admissible)
    pub restoration_trials: Vec<(f64, f64, f64, f64, bool)>,
    pub metrics: Metrics,
    pub memory_hits: usize,
    pub seconds: f64,
    /// restauration rejouable (retouche de la force / du débruitage)
    pub finisher: Finisher,
}

impl AutoResult {
    pub fn summary(&self) -> String {
        let f = &self.features;
        let p = &self.plan;
        format!(
            "{} frames ({} rejetées), mode {}, {} gardées ({:.1} %), alignement « {} », empilement « {} », PSF σ = {:.2} px, force {:.2}, débruitage {:.2} — RSB ×{:.1}, détails ×{:.2}, {:.1} s",
            f.n_frames,
            f.rejected,
            f.mode,
            p.keep,
            p.keep_percent,
            p.align_mode,
            p.stack_method,
            p.psf_sigma,
            p.strength,
            p.denoise_k,
            self.metrics.snr_gain,
            self.metrics.detail_gain,
            self.seconds
        )
    }

    /// Rapport JSON complet (plan, décisions, caractéristiques, indicateurs).
    pub fn report_json(&self) -> serde_json::Value {
        serde_json::json!({
            "version": crate::VERSION,
            "features": self.features,
            "plan": self.plan,
            "metrics": self.metrics,
            "decisions": self.decisions,
            "keep_curve": self.keep_curve,
            "trials": self.trials,
            "memory_hits": self.memory_hits,
            "seconds": self.seconds,
        })
    }

    /// Options équivalentes pour l'onglet / la commande planétaire manuelle.
    pub fn manual_options(&self) -> planetary::PlanetOptions {
        planetary::PlanetOptions {
            quality_method: planetary::QUALITY_METHODS[0].into(),
            roi_mode: "Disque auto-détecté".into(),
            n_align: self.plan.keep.max(1),
            align_mode: self.plan.align_mode.clone(),
            grid_n: self.plan.grid_n.max(2),
            box_size: self.plan.box_size,
            manual_points: None,
            stack_method: if self.plan.stack_method == "Sigma-clipping" { "Sigma-clipping".into() } else { "Moyenne".into() },
            stack_percent: if self.plan.align_mode.starts_with("Multi-points local") { self.plan.ap_percent } else { 100.0 },
        }
    }
}

type Progress<'a> = Option<&'a (dyn Fn(f64, &str) + Sync)>;

struct Ctx<'a> {
    opts: &'a AutoOptions,
    progress: Progress<'a>,
    decisions: Vec<Decision>,
}

impl Ctx<'_> {
    fn step(&self, frac: f64, msg: &str) -> Result<()> {
        if let Some(c) = &self.opts.cancel {
            if c.load(Ordering::Relaxed) {
                bail!("Traitement annulé.");
            }
        }
        if let Some(p) = self.progress {
            p(frac.clamp(0.0, 1.0), msg);
        }
        Ok(())
    }

    fn decide(&mut self, step: &str, choice: impl Into<String>, reason: impl Into<String>) {
        self.decisions.push(Decision { step: step.into(), choice: choice.into(), reason: reason.into() });
    }
}

// ------------------------------------------------------------ utilitaires

/// Générateur pseudo-aléatoire déterministe (splitmix64).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.uniform(), self.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn lum(img: &Image) -> Image {
    img.mean_lum()
}

/// Recadrage avec remplissage (valeur `fill`) hors de l'image.
fn crop_padded(img: &Image, x0: i64, y0: i64, cw: usize, ch: usize, fill: f32) -> Image {
    if x0 >= 0 && y0 >= 0 && x0 as usize + cw <= img.w && y0 as usize + ch <= img.h {
        return img.crop(x0 as usize, y0 as usize, cw, ch);
    }
    let c = img.c;
    let mut out = Image::filled(cw, ch, c, fill);
    for y in 0..ch {
        let sy = y0 + y as i64;
        if sy < 0 || sy >= img.h as i64 {
            continue;
        }
        for x in 0..cw {
            let sx = x0 + x as i64;
            if sx < 0 || sx >= img.w as i64 {
                continue;
            }
            let s = (sy as usize * img.w + sx as usize) * c;
            let d = (y * cw + x) * c;
            out.data[d..d + c].copy_from_slice(&img.data[s..s + c]);
        }
    }
    out
}

/// Frames recentrées sur l'objet (recadrage par frame).
pub struct Centered<'a> {
    pub src: &'a dyn Frames,
    pub idx: Vec<usize>,
    pub origin: Vec<(i64, i64)>,
    pub size: usize,
    pub fill: f32,
}

impl Frames for Centered<'_> {
    fn len(&self) -> usize {
        self.idx.len()
    }
    fn get(&self, i: usize) -> Image {
        let (x0, y0) = self.origin[i];
        crop_padded(&self.src.get(self.idx[i]), x0, y0, self.size, self.size, self.fill)
    }
}

/// Réduction 2×2 (moyenne) d'un plan.
fn bin2(p: &Image) -> Image {
    let (w, h) = (p.w / 2, p.h / 2);
    if w == 0 || h == 0 {
        return p.clone();
    }
    Image::from_fn(w, h, |x, y| {
        let (a, b) = (2 * x, 2 * y);
        0.25 * (p.data[b * p.w + a] + p.data[b * p.w + a + 1] + p.data[(b + 1) * p.w + a] + p.data[(b + 1) * p.w + a + 1])
    })
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let rank = |v: &[f64]| -> Vec<f64> {
        let o = stats::argsort_asc(v);
        let mut r = vec![0.0; v.len()];
        for (k, &i) in o.iter().enumerate() {
            r[i] = k as f64;
        }
        r
    };
    let (ra, rb) = (rank(a), rank(b));
    pearson(&ra, &rb)
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 3 {
        return 0.0;
    }
    let ma = a[..n].iter().sum::<f64>() / n as f64;
    let mb = b[..n].iter().sum::<f64>() / n as f64;
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let (x, y) = (a[i] - ma, b[i] - mb);
        sab += x * y;
        saa += x * x;
        sbb += y * y;
    }
    if saa <= 0.0 || sbb <= 0.0 {
        return 0.0;
    }
    sab / (saa * sbb).sqrt()
}

fn noise_of(img: &Image) -> f64 {
    crate::restoration::noise_sigma(img)
}

// ------------------------------------------------------- géométrie / masque

#[derive(Clone, Copy, Debug)]
struct Disc {
    cx: f64,
    cy: f64,
    r: f64,
}

/// Masque d'analyse : intérieur du disque (sans le limbe) ou image entière
/// sans les bords.
fn analysis_mask(w: usize, h: usize, disc: Option<Disc>) -> Vec<bool> {
    let border = 8usize.min(w / 8).min(h / 8);
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            match disc {
                Some(d) => (x - d.cx).hypot(y - d.cy) < d.r - 3.0_f64.max(d.r * 0.04),
                None => {
                    let (xi, yi) = (i % w, i / w);
                    xi >= border && yi >= border && xi + border < w && yi + border < h
                }
            }
        })
        .collect()
}

fn masked_mean_sq(p: &Image, mask: &[bool]) -> f64 {
    let (mut s, mut n) = (0.0f64, 0usize);
    for (v, &m) in p.data.iter().zip(mask) {
        if m {
            s += (*v as f64) * (*v as f64);
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        s / n as f64
    }
}

fn masked_median(p: &Image, mask: &[bool]) -> f64 {
    let v: Vec<f32> = p.data.iter().zip(mask).filter(|(_, &m)| m).map(|(v, _)| *v).collect();
    if v.is_empty() {
        stats::median(&p.data) as f64
    } else {
        stats::median(&v) as f64
    }
}

// ------------------------------------------------------------- le critique

/// Résultat du critique à deux moitiés.
#[derive(Clone, Debug, Default)]
pub struct Critic {
    pub score: f64,
    /// variance du signal par couche (bruit retiré)
    pub signal: [f64; 4],
    /// variance du bruit par couche
    pub noise: [f64; 4],
    /// écart-type du bruit de l'empilement (domaine image)
    pub sigma: f64,
}

const CRITIC_WEIGHTS: [f64; 4] = [1.0, 1.0, 0.8, 0.6];

/// Critique sans référence : `a`, `b` = empilements de deux moitiés
/// indépendantes (luminance). Le signal est (a+b)/2, le bruit (a−b)/2.
pub fn critic(a: &Image, b: &Image, mask: &[bool]) -> Critic {
    let (la, lb) = (lum(a), lum(b));
    let s = la.zip_map(&lb, |x, y| 0.5 * (x + y));
    let d = la.zip_map(&lb, |x, y| 0.5 * (x - y));
    let (ls, ld) = rayon::join(|| atrous_decompose(&s, 4), || atrous_decompose(&d, 4));
    let mut c = Critic::default();
    for j in 0..4 {
        let vs = masked_mean_sq(&ls[j], mask);
        let vn = masked_mean_sq(&ld[j], mask).max(1e-14);
        let sig = (vs - vn).max(0.0);
        c.signal[j] = sig;
        c.noise[j] = vn;
        c.score += CRITIC_WEIGHTS[j] * (1.0 + sig / vn).log2();
    }
    let (mut sum, mut n) = (0.0f64, 0usize);
    for (v, &m) in d.data.iter().zip(mask) {
        if m {
            sum += (*v as f64).powi(2);
            n += 1;
        }
    }
    c.sigma = (sum / n.max(1) as f64).sqrt();
    c
}

// ------------------------------------------------- détection du disque

/// Cercle du limbe par RANSAC sur les bords « clair à l'intérieur » :
/// insensible au croissant d'une éclipse (le bord de la Lune, clair à
/// l'extérieur, est ignoré) et aux phases. `r_hint` : rayon connu (±5 %).
pub fn robust_disc(img: &Image, r_hint: Option<f64>) -> Option<(f64, f64, f64)> {
    let l = ndimage::gaussian(&lum(img), 1.2);
    let (w, h) = (l.w, l.h);
    if w < 16 || h < 16 {
        return None;
    }
    let mx = (l.max_value() as f64).max(1e-9);
    let u8v: Vec<u8> = l.data.iter().map(|&v| ((v as f64 / mx).clamp(0.0, 1.0) * 255.0) as u8).collect();
    let thr = (crate::features::otsu(&u8v) as f64 / 255.0 * mx) as f32;
    let mask: Vec<bool> = l.data.iter().map(|&v| v > thr).collect();
    // plus grand objet, trous bouchés
    let (lab, n) = ndimage::label(&mask, w, h);
    if n == 0 {
        return None;
    }
    let mut sizes = vec![0usize; n + 1];
    for &k in &lab {
        sizes[k as usize] += 1;
    }
    let big = (1..=n).max_by_key(|&k| sizes[k]).unwrap() as u32;
    let m = ndimage::binary_fill_holes(&lab.iter().map(|&k| k == big).collect::<Vec<_>>(), w, h);
    // points de bord avec direction du gradient (vers le clair)
    let mut pts: Vec<(f64, f64, f64, f64)> = Vec::new();
    for y in 2..h - 2 {
        for x in 2..w - 2 {
            let i = y * w + x;
            if !m[i] || (m[i - 1] && m[i + 1] && m[i - w] && m[i + w]) {
                continue;
            }
            let gx = (l.data[i + 1] - l.data[i - 1]) as f64;
            let gy = (l.data[i + w] - l.data[i - w]) as f64;
            let g = gx.hypot(gy);
            if g > 0.0 {
                pts.push((x as f64, y as f64, gx / g, gy / g));
            }
        }
    }
    if pts.len() < 20 {
        return None;
    }
    if pts.len() > 5000 {
        let step = pts.len() / 5000 + 1;
        pts = pts.into_iter().step_by(step).collect();
    }
    let npts = pts.len();
    let fits = |c: (f64, f64, f64), tol: f64| -> Vec<usize> {
        (0..npts)
            .filter(|&k| {
                let (x, y, gx, gy) = pts[k];
                let (dx, dy) = (c.0 - x, c.1 - y);
                let d = dx.hypot(dy);
                (d - c.2).abs() < tol && d > 0.0 && (gx * dx + gy * dy) / d > 0.5
            })
            .collect()
    };
    let rmax = 2.0 * w.max(h) as f64;
    let mut rng = Rng(0xD15C);
    let mut best: (usize, (f64, f64, f64)) = (0, (0.0, 0.0, 0.0));
    for _ in 0..700 {
        let mut cands: Vec<(f64, f64, f64)> = Vec::new();
        let a = pts[(rng.next() as usize) % npts];
        let b = pts[(rng.next() as usize) % npts];
        match r_hint {
            Some(r) => {
                let (mx2, my2) = ((a.0 + b.0) / 2.0, (a.1 + b.1) / 2.0);
                let half = (b.0 - a.0).hypot(b.1 - a.1) / 2.0;
                if half < 2.0 || half >= r {
                    continue;
                }
                let t = (r * r - half * half).sqrt();
                let (ux, uy) = (-(b.1 - a.1) / (2.0 * half), (b.0 - a.0) / (2.0 * half));
                cands.push((mx2 + t * ux, my2 + t * uy, r));
                cands.push((mx2 - t * ux, my2 - t * uy, r));
            }
            None => {
                let c = pts[(rng.next() as usize) % npts];
                let d = 2.0 * (a.0 * (b.1 - c.1) + b.0 * (c.1 - a.1) + c.0 * (a.1 - b.1));
                if d.abs() < 1e-6 {
                    continue;
                }
                let (a2, b2, c2) = (a.0 * a.0 + a.1 * a.1, b.0 * b.0 + b.1 * b.1, c.0 * c.0 + c.1 * c.1);
                let ux = (a2 * (b.1 - c.1) + b2 * (c.1 - a.1) + c2 * (a.1 - b.1)) / d;
                let uy = (a2 * (c.0 - b.0) + b2 * (a.0 - c.0) + c2 * (b.0 - a.0)) / d;
                let r = (a.0 - ux).hypot(a.1 - uy);
                if r < 8.0 || r > rmax {
                    continue;
                }
                cands.push((ux, uy, r));
            }
        }
        for c in cands {
            let tol = 1.5f64.max(0.03 * c.2);
            let k = fits(c, tol).len();
            if k > best.0 {
                best = (k, c);
            }
        }
    }
    let (k, mut c) = best;
    if k < 15 || (k as f64) < 0.08 * std::f64::consts::TAU * c.2 {
        return None;
    }
    // affinage (moindres carrés géométriques, Gauss-Newton)
    for _ in 0..6 {
        let tol = 1.5f64.max(0.05 * c.2);
        let inl = fits(c, tol);
        if inl.len() < 10 {
            break;
        }
        let (mut jtj, mut jtr) = ([[0.0f64; 3]; 3], [0.0f64; 3]);
        for &i in &inl {
            let (x, y, _, _) = pts[i];
            let d = (x - c.0).hypot(y - c.1).max(1e-9);
            let res = d - c.2;
            let jac = [-(x - c.0) / d, -(y - c.1) / d, -1.0];
            let nv = 3;
            for p in 0..nv {
                jtr[p] += jac[p] * res;
                for q in 0..nv {
                    jtj[p][q] += jac[p] * jac[q];
                }
            }
        }
        let nv = 3;
        let mut a = nalgebra::DMatrix::<f64>::zeros(nv, nv);
        let mut bv = nalgebra::DVector::<f64>::zeros(nv);
        for p in 0..nv {
            bv[p] = -jtr[p];
            for q in 0..nv {
                a[(p, q)] = jtj[p][q];
            }
        }
        let Some(delta) = a.lu().solve(&bv) else { break };
        c.0 += delta[0];
        c.1 += delta[1];
        c.2 += delta[2];
        if let Some(rh) = r_hint {
            c.2 = c.2.clamp(0.95 * rh, 1.05 * rh);
        }
        if delta.norm() < 1e-3 {
            break;
        }
    }
    Some(c)
}

// ------------------------------------------------------------- l'analyse

#[derive(Clone, Copy, Debug, Default)]
struct Scan {
    cx: f64,
    cy: f64,
    area: f64,
    mean: f64,
    sat: f64,
}

struct Analysis {
    features: Features,
    view_idx: Vec<usize>,
    origin: Vec<(i64, i64)>,
    size: usize,
    fill: f32,
    rejected: Vec<usize>,
    disc_local: Option<Disc>,
}

fn analyse(frames: &dyn Frames, ctx: &mut Ctx) -> Result<Analysis> {
    let n = frames.len();
    let first = frames.get(0);
    let (w, h, c) = (first.w, first.h, first.c);
    ctx.step(0.01, "Analyse de la vidéo")?;
    let mid = planetary::mean_sample(frames, (n / 30).max(1));
    let midl = ndimage::gaussian(&lum(&mid), 1.5);
    let p50 = stats::percentile(&midl.data, 50.0) as f64;
    let p995 = stats::percentile(&midl.data, 99.5) as f64;
    let thr = (0.5 * (p50 + p995)) as f32;
    let factor = if w * h > 700 * 500 { 2usize } else { 1 };
    let scans: Vec<Scan> = (0..n)
        .into_par_iter()
        .map(|i| {
            let f = frames.get(i);
            let l = lum(&f);
            let sat = f.data.iter().filter(|&&v| v >= 0.985).count() as f64 / f.data.len().max(1) as f64;
            let mean = l.mean();
            let l = if factor == 2 { bin2(&l) } else { l };
            let (mut sx, mut sy, mut sw, mut area) = (0.0f64, 0.0f64, 0.0f64, 0usize);
            for y in 0..l.h {
                for x in 0..l.w {
                    let v = l.data[y * l.w + x];
                    if v > thr {
                        let wv = (v - thr) as f64;
                        sx += wv * x as f64;
                        sy += wv * y as f64;
                        sw += wv;
                        area += 1;
                    }
                }
            }
            let fct = factor as f64;
            let off = if factor == 2 { 0.5 } else { 0.0 };
            if sw > 0.0 {
                Scan { cx: sx / sw * fct + off, cy: sy / sw * fct + off, area: area as f64 * fct * fct, mean, sat }
            } else {
                Scan { cx: w as f64 / 2.0, cy: h as f64 / 2.0, area: 0.0, mean, sat }
            }
        })
        .collect();
    ctx.step(0.08, "Détection de l'objet")?;
    let areas: Vec<f64> = scans.iter().map(|s| s.area).collect();
    let area_med = stats::median_f64(&areas);
    let r_est = (area_med / std::f64::consts::PI).sqrt();
    let means: Vec<f64> = scans.iter().map(|s| s.mean).collect();
    let mean_med = stats::median_f64(&means);
    let mean_mad = 1.4826 * stats::median_f64(&means.iter().map(|m| (m - mean_med).abs()).collect::<Vec<_>>()) + 1e-6;
    let sats: Vec<f64> = scans.iter().map(|s| s.sat).collect();
    let sat_med = stats::median_f64(&sats);
    // centre et étendue de l'objet sur l'image moyenne
    let (mut sx, mut sy, mut sw) = (0.0f64, 0.0f64, 0.0f64);
    for y in 0..h {
        for x in 0..w {
            let v = midl.data[y * w + x];
            if v > thr {
                let wv = (v - thr) as f64;
                sx += wv * x as f64;
                sy += wv * y as f64;
                sw += wv;
            }
        }
    }
    let (mcx, mcy) = if sw > 0.0 { (sx / sw, sy / sw) } else { (w as f64 / 2.0, h as f64 / 2.0) };
    let mut r_box = 0.0f64;
    for y in 0..h {
        for x in 0..w {
            if midl.data[y * w + x] > thr {
                r_box = r_box.max((x as f64 - mcx).hypot(y as f64 - mcy));
            }
        }
    }
    // disque entier visible ? sinon mode surface. Le cercle du limbe est
    // cherché sur les bords « clair à l'intérieur » (croissant d'éclipse,
    // phase) ; repli sur l'ajustement classique.
    let disc_fit = robust_disc(&mid, None).or_else(|| crate::derotation::fit_disc(&mid, None).ok());
    let disc_mode = area_med > 30.0
        && area_med < 0.55 * (w * h) as f64
        && match disc_fit {
            Some((cx, cy, r)) => cx - r > 1.0 && cy - r > 1.0 && cx + r < w as f64 - 1.0 && cy + r < h as f64 - 1.0 && r > 0.8 * r_est && r > 6.0 && r_box < 1.3 * r + 3.0,
            None => false,
        };
    let r_edge = if disc_mode { r_box.max(0.9 * r_est) } else { r_est };
    let mut rejected = Vec::new();
    let mut valid = Vec::new();
    let mut reasons = (0usize, 0usize, 0usize, 0usize);
    for (i, s) in scans.iter().enumerate() {
        let z = (s.mean - mean_med) / mean_mad;
        let bad_bright = z < -6.0 || s.mean < 0.35 * mean_med || s.mean > 2.5 * mean_med.max(1e-6);
        let bad_area = disc_mode && (s.area < 0.8 * area_med || s.area > 1.8 * area_med);
        let bad_edge = disc_mode && (s.cx - r_edge < 1.0 || s.cy - r_edge < 1.0 || s.cx + r_edge > w as f64 - 1.0 || s.cy + r_edge > h as f64 - 1.0);
        let bad_sat = s.sat > (4.0 * sat_med).max(0.02);
        if bad_bright || bad_area || bad_edge || bad_sat {
            rejected.push(i);
            if bad_bright {
                reasons.0 += 1;
            } else if bad_area {
                reasons.1 += 1;
            } else if bad_edge {
                reasons.2 += 1;
            } else {
                reasons.3 += 1;
            }
        } else {
            valid.push(i);
        }
    }
    if valid.len() < 2 {
        // garde-fou : ne jamais tout rejeter
        valid = (0..n).collect();
        rejected.clear();
    }
    let fill = (stats::percentile(&lum(&mid).data, 5.0) as f64).max(0.0) as f32;
    let (size, origin, disc_local, mode) = if disc_mode {
        let (fcx, fcy, r) = disc_fit.unwrap();
        // décalage du centre du cercle par rapport au barycentre : n'est pris
        // en compte que pour un objet nettement dissymétrique (croissant
        // d'éclipse, phase) ; pour un disque entier le barycentre est plus sûr
        let (ox, oy) = if (fcx - mcx).hypot(fcy - mcy) > 0.12 * r { (fcx - mcx, fcy - mcy) } else { (0.0, 0.0) };
        let ext = (r + ox.hypot(oy)).max(r_box);
        let mut size = (2.0 * (ext * 1.3 + 10.0)).ceil() as usize;
        size = size.div_ceil(8) * 8;
        size = size.max(64).min(w.max(h) + 64);
        let origin: Vec<(i64, i64)> = valid.iter().map(|&i| ((scans[i].cx - size as f64 / 2.0).round() as i64, (scans[i].cy - size as f64 / 2.0).round() as i64)).collect();
        (size, origin, Some(Disc { cx: size as f64 / 2.0 + ox, cy: size as f64 / 2.0 + oy, r }), "disque")
    } else {
        // surface : on garde le plus grand carré central
        let size = w.min(h);
        let o = (((w - size) / 2) as i64, ((h - size) / 2) as i64);
        (size, vec![o; valid.len()], None, "surface")
    };
    let cxs: Vec<f64> = valid.iter().map(|&i| scans[i].cx).collect();
    let cys: Vec<f64> = valid.iter().map(|&i| scans[i].cy).collect();
    let jitter = if disc_mode { (stats::std_f64(&cxs).powi(2) + stats::std_f64(&cys).powi(2)).sqrt() } else { 0.0 };
    // bruit et RSB d'une frame (échantillon)
    let samp: Vec<usize> = valid.iter().step_by((valid.len() / 12).max(1)).cloned().take(12).collect();
    let noises: Vec<f64> = samp.par_iter().map(|&i| noise_of(&frames.get(i))).collect();
    let noise = stats::median_f64(&noises).max(1e-5);
    let brightness = if disc_mode {
        let v = Centered { src: frames, idx: vec![valid[0]], origin: vec![origin[0]], size, fill };
        let im = lum(&v.get(0));
        let m = analysis_mask(size, size, disc_local);
        masked_median(&im, &m)
    } else {
        stats::median(&lum(&mid).data) as f64
    };
    let features = Features {
        n_frames: n,
        width: w,
        height: h,
        color: c == 3,
        mode: mode.into(),
        disc_radius: disc_local.map(|d| d.r).unwrap_or(size as f64 / 2.0),
        crop: size,
        noise,
        snr: brightness / noise,
        brightness,
        saturation: sat_med,
        jitter,
        rejected: rejected.len(),
        ..Default::default()
    };
    if disc_mode {
        ctx.decide(
            "Objet",
            format!("Disque de rayon {:.0} px, recadrage {}×{} centré par frame", features.disc_radius, size, size),
            format!("Planète entière dans le champ ; agitation globale {:.1} px rattrapée par recentrage individuel.", jitter),
        );
    } else {
        ctx.decide("Objet", format!("Mode surface ({}×{})", size, size), "Objet plus grand que le champ ou sans disque complet (Lune, Soleil) : analyse sur toute l'image.");
    }
    ctx.decide(
        "Rejet",
        format!("{} frame(s) écartée(s) sur {}", rejected.len(), n),
        format!("luminosité anormale (nuage) : {}, taille de disque anormale : {}, objet coupé par le bord : {}, saturation : {}", reasons.0, reasons.1, reasons.2, reasons.3),
    );
    Ok(Analysis { features, view_idx: valid, origin, size, fill, rejected, disc_local })
}

// ------------------------------------------------------- tri et classement

const METRICS: [&str; 4] = ["Contraste HF (AutoStakkert)", "Netteté (Laplacien)", "Gradient (Sobel)", "Bande passante (DoG)"];

fn metric(a: &Image, m: usize) -> f64 {
    match m {
        0 => planetary::q_hf(a),
        1 => planetary::q_lap(a),
        2 => planetary::q_grad(a),
        _ => {
            let (g1, g2) = rayon::join(|| ndimage::gaussian(a, 1.0), || ndimage::gaussian(a, 2.5));
            g1.data.iter().zip(&g2.data).map(|(x, y)| ((x - y) as f64).powi(2)).sum::<f64>() / a.data.len().max(1) as f64 + 1e-12
        }
    }
}

/// Zone de mesure de la qualité : disque + limbe.
fn quality_patch(img: &Image, disc: Option<Disc>) -> Image {
    let l = lum(img);
    match disc {
        Some(d) => {
            let half = (d.r * 1.08 + 4.0) as i64;
            let (x0, y0) = ((d.cx as i64 - half).max(0) as usize, (d.cy as i64 - half).max(0) as usize);
            l.crop(x0, y0, 2 * half as usize, 2 * half as usize)
        }
        None => l,
    }
}

struct Ranking {
    fused: Vec<f64>,
    order: Vec<usize>,
    weights: Vec<(String, f64)>,
    cv: f64,
}

fn rank_frames(view: &Centered, disc: Option<Disc>, noise: f64, ctx: &mut Ctx) -> Result<Ranking> {
    let n = view.len();
    ctx.step(0.12, "Mesure de la netteté de chaque frame")?;
    let raw: Vec<[f64; 4]> = (0..n)
        .into_par_iter()
        .map(|i| {
            let p = quality_patch(&view.get(i), disc);
            [metric(&p, 0), metric(&p, 1), metric(&p, 2), metric(&p, 3)]
        })
        .collect();
    ctx.step(0.22, "Test de robustesse des mesures au bruit")?;
    // robustesse : mesure recalculée après ajout d'un bruit de même niveau
    let samp: Vec<usize> = (0..n).step_by((n / 60).max(1)).take(60).collect();
    let noisy: Vec<[f64; 4]> = samp
        .par_iter()
        .map(|&i| {
            let mut p = quality_patch(&view.get(i), disc);
            let mut rng = Rng(0xA57F0 + i as u64);
            let s = noise as f32;
            p.data.iter_mut().for_each(|v| *v += s * rng.normal() as f32);
            [metric(&p, 0), metric(&p, 1), metric(&p, 2), metric(&p, 3)]
        })
        .collect();
    let mut weights = [0.0f64; 4];
    let mut rho = [0.0f64; 4];
    let mut spread = [0.0f64; 4];
    for m in 0..4 {
        let a: Vec<f64> = samp.iter().map(|&i| raw[i][m]).collect();
        let b: Vec<f64> = noisy.iter().map(|v| v[m]).collect();
        rho[m] = if samp.len() >= 5 { spearman(&a, &b) } else { 1.0 };
        let all: Vec<f64> = raw.iter().map(|v| v[m]).collect();
        let med = stats::median_f64(&all).max(1e-30);
        let q = |p: f64| {
            let mut s = all.clone();
            s.sort_by(|x, y| x.total_cmp(y));
            s[((s.len() - 1) as f64 * p).round() as usize]
        };
        spread[m] = ((q(0.9) - q(0.1)) / med).max(0.0);
        weights[m] = (rho[m] - 0.3).max(0.0).powi(2) * spread[m].min(1.0).max(0.02);
    }
    if weights.iter().sum::<f64>() <= 1e-12 {
        weights = [0.0, 0.0, 0.0, 1.0];
    }
    let tot: f64 = weights.iter().sum();
    weights.iter_mut().for_each(|w| *w /= tot);
    // fusion des rangs (percentiles)
    let mut fused = vec![0.0f64; n];
    for m in 0..4 {
        if weights[m] <= 0.0 {
            continue;
        }
        let v: Vec<f64> = raw.iter().map(|x| x[m]).collect();
        let o = stats::argsort_asc(&v);
        for (k, &i) in o.iter().enumerate() {
            fused[i] += weights[m] * (k as f64 + 0.5) / n as f64;
        }
    }
    let order = stats::argsort_desc(&fused);
    let best_m = (0..4).max_by(|&a, &b| weights[a].total_cmp(&weights[b])).unwrap();
    let main: Vec<f64> = raw.iter().map(|v| v[best_m]).collect();
    let cv = stats::std_f64(&main) / stats::mean_f64(&main).max(1e-30);
    let wlist: Vec<(String, f64)> = (0..4).map(|m| (METRICS[m].to_string(), (weights[m] * 1000.0).round() / 1000.0)).collect();
    let desc: Vec<String> = (0..4).filter(|&m| weights[m] > 0.0).map(|m| format!("{} {:.0} %", METRICS[m], weights[m] * 100.0)).collect();
    ctx.decide(
        "Tri",
        format!("Classement fusionné : {}", desc.join(", ")),
        format!(
            "Fiabilité face au bruit (corrélation de rang) : HF {:.2}, Laplacien {:.2}, Gradient {:.2}, DoG {:.2} ; les mesures trompées par le bruit sont écartées. Variabilité du seeing : {:.0} %.",
            rho[0],
            rho[1],
            rho[2],
            rho[3],
            cv * 100.0
        ),
    );
    Ok(Ranking { fused, order, weights: wlist, cv })
}

// --------------------------------------------------- alignement global

/// Référence (moyenne des meilleures frames recalées) et décalages globaux
/// des `k` meilleures frames.
fn global_shifts(view: &Centered, order: &[usize], k: usize, ctx: &Ctx) -> Result<(Image, Vec<(f64, f64)>)> {
    let n_ref = (k / 20).clamp(3, 12).min(order.len());
    let best = view.get(order[0]);
    let refl = lum(&best);
    let first: Vec<Image> = order[..n_ref]
        .par_iter()
        .enumerate()
        .map(|(j, &i)| {
            if j == 0 {
                return best.clone();
            }
            let f = view.get(i);
            let (dx, dy, _) = phase_corr(&refl, &lum(&f));
            apply_shift(&f, dx, dy)
        })
        .collect();
    let reference = mean_images(&first.iter().collect::<Vec<_>>());
    let rl = lum(&reference);
    ctx.step(0.3, "Alignement global des meilleures frames")?;
    let shifts: Vec<(f64, f64)> = order[..k]
        .par_iter()
        .map(|&i| {
            let (dx, dy, _) = phase_corr(&rl, &lum(&view.get(i)));
            (dx, dy)
        })
        .collect();
    Ok((reference, shifts))
}

/// Distorsion locale (turbulence) : dispersion des décalages locaux
/// autour du décalage global, sur quelques bonnes frames.
fn local_distortion(view: &Centered, order: &[usize], shifts: &[(f64, f64)], reference: &Image, disc: Option<Disc>, bx: usize) -> f64 {
    let rl = lum(reference);
    let (w, h) = (rl.w, rl.h);
    let pts: Vec<(usize, usize)> = {
        let mut v = Vec::new();
        let step = bx;
        let mut y = bx / 2;
        while y + bx / 2 <= h {
            let mut x = bx / 2;
            while x + bx / 2 <= w {
                let inside = match disc {
                    Some(d) => (x as f64 - d.cx).hypot(y as f64 - d.cy) + bx as f64 * 0.5 < d.r,
                    None => true,
                };
                if inside {
                    v.push((x, y));
                }
                x += step;
            }
            y += step;
        }
        v
    };
    if pts.len() < 3 {
        return 0.0;
    }
    let nf = order.len().min(shifts.len()).min(12);
    let res: Vec<f64> = (1..nf)
        .into_par_iter()
        .map(|k| {
            let f = apply_shift(&lum(&view.get(order[k])), shifts[k].0, shifts[k].1);
            let loc: Vec<(f64, f64)> = pts
                .iter()
                .map(|&(cx, cy)| {
                    let (x0, y0) = (cx - bx / 2, cy - bx / 2);
                    let (dx, dy, _) = phase_corr(&rl.crop(x0, y0, bx, bx), &f.crop(x0, y0, bx, bx));
                    (dx.clamp(-4.0, 4.0), dy.clamp(-4.0, 4.0))
                })
                .collect();
            let mx = loc.iter().map(|v| v.0).sum::<f64>() / loc.len() as f64;
            let my = loc.iter().map(|v| v.1).sum::<f64>() / loc.len() as f64;
            let v: Vec<f64> = loc.iter().map(|v| (v.0 - mx).hypot(v.1 - my)).collect();
            stats::median_f64(&v)
        })
        .collect();
    if res.is_empty() {
        0.0
    } else {
        stats::median_f64(&res)
    }
}

// ------------------------------------------------ empilements d'essai

/// Empilement en deux moitiés (positions paires / impaires du classement)
/// des frames déjà recalées par `align(k)`.
fn halves_mean(len: usize, align: &(dyn Fn(usize) -> Image + Sync), lum_only: bool) -> (Image, Image) {
    let chunk = 64usize;
    let mut acc: [Option<Vec<f64>>; 2] = [None, None];
    let mut cnt = [0usize; 2];
    let mut shape = (0, 0, 0);
    let mut start = 0;
    while start < len {
        let end = (start + chunk).min(len);
        let imgs: Vec<(usize, Image)> = (start..end).into_par_iter().map(|k| (k, if lum_only { lum(&align(k)) } else { align(k) })).collect();
        for (k, im) in imgs {
            let hlf = k % 2;
            shape = (im.w, im.h, im.c);
            let a = acc[hlf].get_or_insert_with(|| vec![0.0; im.data.len()]);
            for (s, v) in a.iter_mut().zip(&im.data) {
                *s += *v as f64;
            }
            cnt[hlf] += 1;
        }
        start = end;
    }
    let mk = |h: usize| -> Image {
        match &acc[h] {
            Some(a) => Image::from_vec(shape.0, shape.1, shape.2, a.iter().map(|v| (v / cnt[h].max(1) as f64) as f32).collect()),
            None => Image::new(shape.0, shape.1, shape.2),
        }
    };
    let (a, b) = (mk(0), mk(1));
    if cnt[1] == 0 {
        return (a.clone(), a);
    }
    (a, b)
}

/// Courbe « nombre de frames gardées → score du critique » (luminance,
/// alignement global, sommes cumulées : un seul passage).
fn keep_curve(view: &Centered, order: &[usize], shifts: &[(f64, f64)], cands: &[usize], mask: &[bool], ctx: &Ctx) -> Result<Vec<(usize, Critic)>> {
    let kmax = *cands.iter().max().unwrap();
    let size = view.size;
    let mut acc = [vec![0.0f64; size * size], vec![0.0f64; size * size]];
    let mut cnt = [0usize; 2];
    let mut out = Vec::new();
    let mut ci = 0;
    let mut sorted: Vec<usize> = cands.to_vec();
    sorted.sort();
    sorted.dedup();
    let chunk = 64;
    let mut start = 0;
    while start < kmax {
        let end = (start + chunk).min(kmax);
        ctx.step(0.32 + 0.18 * start as f64 / kmax as f64, "Choix du nombre d'images à empiler")?;
        let imgs: Vec<Image> = (start..end).into_par_iter().map(|k| apply_shift(&lum(&view.get(order[k])), shifts[k].0, shifts[k].1)).collect();
        for (j, im) in imgs.into_iter().enumerate() {
            let k = start + j;
            let hlf = k % 2;
            for (s, v) in acc[hlf].iter_mut().zip(&im.data) {
                *s += *v as f64;
            }
            cnt[hlf] += 1;
            while ci < sorted.len() && sorted[ci] == k + 1 {
                let mk = |h: usize| Image::from_vec(size, size, 1, acc[h].iter().map(|v| (v / cnt[h].max(1) as f64) as f32).collect());
                let (a, b) = (mk(0), mk(1));
                out.push((k + 1, critic(&a, &b, mask)));
                ci += 1;
            }
        }
        start = end;
    }
    Ok(out)
}

struct Trial {
    label: String,
    mode: String,
    method: String,
    bx: usize,
    score: f64,
}

#[allow(clippy::too_many_arguments)]
fn stack_trial(view: &Centered, sel: &[usize], shifts: &[(f64, f64)], reference: &Image, mode: &str, method: &str, bx: usize, grid_n: usize, ap_percent: f64, disc: Option<Disc>, weights: &[f64]) -> Result<(Image, Image, Image)> {
    let size = view.size;
    if mode.starts_with("Multi-points local") {
        let base_of = |idx: &[usize]| -> Image {
            let nb = ((idx.len() as f64 * ap_percent / 100.0) as usize).clamp(2.min(idx.len()), idx.len()).max(1);
            let imgs: Vec<Image> = idx[..nb].par_iter().map(|&k| apply_shift(&view.get(sel[k]), shifts[k].0, shifts[k].1)).collect();
            mean_images(&imgs.iter().collect::<Vec<_>>())
        };
        let roi = disc.map(|d| {
            let r = d.r * 1.05;
            (((d.cx - r).max(0.0)) as usize, ((d.cy - r).max(0.0)) as usize, ((d.cx + r) as usize).min(size), ((d.cy + r) as usize).min(size))
        });
        let run = |idx: Vec<usize>| -> Image {
            let s: Vec<usize> = idx.iter().map(|&k| sel[k]).collect();
            let sh: Vec<(f64, f64)> = idx.iter().map(|&k| shifts[k]).collect();
            let base = base_of(&idx);
            planetary::local_ap_core(view, &s, &sh, reference, &base, roi, bx, ap_percent, 0.15).0
        };
        let all: Vec<usize> = (0..sel.len()).collect();
        let ev: Vec<usize> = all.iter().cloned().filter(|k| k % 2 == 0).collect();
        let od: Vec<usize> = all.iter().cloned().filter(|k| k % 2 == 1).collect();
        let od = if od.is_empty() { ev.clone() } else { od };
        let a = run(ev);
        let b = run(od);
        let full = run(all);
        return Ok((full, a, b));
    }
    let pts: Option<Vec<(f64, f64)>> = if mode.starts_with("Multi-points") {
        let roi = disc.map(|d| {
            let r = d.r * 0.92;
            (((d.cx - r).max(0.0)) as usize, ((d.cy - r).max(0.0)) as usize, ((d.cx + r) as usize).min(size), ((d.cy + r) as usize).min(size))
        });
        Some(planetary::grid_points(size, size, grid_n.max(2), bx, roi))
    } else {
        None
    };
    let align = |k: usize| -> Image {
        let f = view.get(sel[k]);
        match &pts {
            None => apply_shift(&f, shifts[k].0, shifts[k].1),
            Some(p) => {
                // pré-recalage global puis champ multi-points
                let pre = apply_shift(&f, shifts[k].0, shifts[k].1);
                let (im, _st): (Image, AlignStats) = planetary::align_frame(reference, &pre, Some(p), bx, "Multi-points (grille NxN)");
                im
            }
        }
    };
    let bytes = sel.len() * size * size * reference.c * 4;
    if method == "Moyenne" || bytes > 1_200_000_000 {
        let (a, b) = halves_mean(sel.len(), &align, false);
        let full = a.zip_map(&b, |x, y| 0.5 * (x + y));
        return Ok((full, a, b));
    }
    let aligned: Vec<Image> = (0..sel.len()).into_par_iter().map(align).collect();
    let split = |h: usize| -> (Vec<&Image>, Vec<f64>) {
        let idx: Vec<usize> = (0..aligned.len()).filter(|k| k % 2 == h || aligned.len() < 2).collect();
        (idx.iter().map(|&k| &aligned[k]).collect(), idx.iter().map(|&k| weights[k]).collect())
    };
    let do_stack = |imgs: &[&Image], ws: &[f64]| -> Result<Image> {
        if method == "Moyenne pondérée" {
            stacking::stack(imgs, "Moyenne", &StackParams::default(), Some(ws))
        } else {
            stacking::stack(imgs, method, &StackParams { sigma: 2.5, ..Default::default() }, None)
        }
    };
    let (ia, wa) = split(0);
    let (ib, wb) = split(1);
    let a = do_stack(&ia, &wa)?;
    let b = do_stack(&ib, &wb)?;
    let all: Vec<&Image> = aligned.iter().collect();
    let full = do_stack(&all, &weights[..aligned.len()])?;
    Ok((full, a, b))
}

// ------------------------------------------------------- la restauration

/// FTM de chaque couche d'ondelettes pour une PSF gaussienne σ (mesurée
/// numériquement sur un bruit blanc).
fn layer_mtf(sigma: f64, levels: usize) -> Vec<f64> {
    let n = 128usize;
    let mut rng = Rng(0x5EED);
    let white = Image::from_vec(n, n, 1, (0..n * n).map(|_| rng.normal() as f32).collect());
    let blurred = ndimage::gaussian(&white, sigma.max(0.05));
    let (lw, lb) = (atrous_decompose(&white, levels), atrous_decompose(&blurred, levels));
    let m = vec![true; n * n];
    (0..levels)
        .map(|j| {
            let inner: Vec<bool> = (0..n * n).map(|i| (i % n) >= 16 && (i % n) < n - 16 && (i / n) >= 16 && (i / n) < n - 16).collect();
            let mask = if j < 4 { &inner } else { &m };
            (masked_mean_sq(&lb[j], mask) / masked_mean_sq(&lw[j], mask).max(1e-20)).sqrt().clamp(1e-3, 1.0)
        })
        .collect()
}

/// PSF (σ gaussien) estimée sur le limbe : profils radiaux par secteurs, on
/// garde les limbes les plus nets (le terminateur des phases est ignoré).
fn psf_from_limb(img: &Image, disc: Disc) -> Option<f64> {
    let l = lum(img);
    let (w, h) = (l.w, l.h);
    let sectors = 36usize;
    let (r0, r1) = ((disc.r - 14.0).max(disc.r * 0.6), disc.r + 14.0);
    let nb = ((r1 - r0) * 4.0) as usize;
    let mut sig = Vec::new();
    for s in 0..sectors {
        let a0 = s as f64 / sectors as f64 * std::f64::consts::TAU;
        let mut prof = vec![0.0f64; nb];
        let mut cnt = vec![0usize; nb];
        for t in 0..12 {
            let a = a0 + t as f64 / 12.0 * std::f64::consts::TAU / sectors as f64;
            let (ca, sa) = (a.cos(), a.sin());
            for (b, (p, c)) in prof.iter_mut().zip(cnt.iter_mut()).enumerate() {
                let r = r0 + b as f64 / 4.0;
                let (x, y) = (disc.cx + r * ca, disc.cy + r * sa);
                if x < 1.0 || y < 1.0 || x >= w as f64 - 2.0 || y >= h as f64 - 2.0 {
                    continue;
                }
                let (xi, yi) = (x.floor() as usize, y.floor() as usize);
                let (fx, fy) = (x - xi as f64, y - yi as f64);
                let v = l.data[yi * w + xi] as f64 * (1.0 - fx) * (1.0 - fy) + l.data[yi * w + xi + 1] as f64 * fx * (1.0 - fy) + l.data[(yi + 1) * w + xi] as f64 * (1.0 - fx) * fy + l.data[(yi + 1) * w + xi + 1] as f64 * fx * fy;
                *p += v;
                *c += 1;
            }
        }
        if cnt.iter().any(|&c| c == 0) {
            continue;
        }
        let prof: Vec<f64> = prof.iter().zip(&cnt).map(|(p, c)| p / *c as f64).collect();
        let lsf: Vec<f64> = (0..nb).map(|b| if b == 0 || b + 1 >= nb { 0.0 } else { -(prof[b + 1] - prof[b - 1]) * 2.0 }).collect();
        let (bi, &peak) = lsf.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
        let contrast = prof[..nb / 4].iter().sum::<f64>() / (nb / 4) as f64 - prof[nb * 3 / 4..].iter().sum::<f64>() / (nb - nb * 3 / 4) as f64;
        if peak <= 0.0 || contrast <= 0.05 * prof.iter().cloned().fold(0.0, f64::max) {
            continue;
        }
        let win = 24usize; // ±6 px
        let (lo, hi) = (bi.saturating_sub(win), (bi + win).min(nb - 1));
        let (mut s0, mut s1, mut s2) = (0.0, 0.0, 0.0);
        for (b, &v) in lsf.iter().enumerate().take(hi + 1).skip(lo) {
            let v = (v - 0.1 * peak).max(0.0);
            let r = b as f64 / 4.0;
            s0 += v;
            s1 += v * r;
            s2 += v * r * r;
        }
        if s0 > 0.0 {
            let m = s1 / s0;
            sig.push((s2 / s0 - m * m).max(0.0).sqrt());
        }
    }
    if sig.len() < 6 {
        return None;
    }
    sig.sort_by(|a, b| a.total_cmp(b));
    Some(sig[sig.len() / 4])
}

/// Paramètres d'une restauration.
#[derive(Clone, Debug)]
struct Restore {
    gains: Vec<f64>,
    thresholds: Vec<Vec<f64>>, // [canal][couche]
    limb: Option<Disc>,
}

const LEVELS: usize = 6;

/// Masque d'anneau autour du limbe, largeur adaptée à l'échelle de la couche.
fn limb_ring(w: usize, h: usize, d: Disc, j: usize) -> Vec<f32> {
    let width = 1.5 * (1usize << j) as f64 + 2.0;
    (0..w * h)
        .map(|i| {
            let r = ((i % w) as f64 - d.cx).hypot((i / w) as f64 - d.cy);
            let t = ((r - d.r).abs() / width).min(1.0);
            (1.0 - t * t) as f32
        })
        .collect()
}

fn soft(v: f32, t: f32) -> f32 {
    if t <= 0.0 {
        v
    } else {
        v.signum() * (v.abs() - t).max(0.0)
    }
}

/// Décomposition par canal (mise en cache pour les essais).
struct Decomp {
    w: usize,
    h: usize,
    planes: Vec<Vec<Image>>,
}

fn decompose(img: &Image) -> Decomp {
    let planes: Vec<Vec<Image>> = img.channels().par_iter().map(|p| atrous_decompose(p, LEVELS)).collect();
    Decomp { w: img.w, h: img.h, planes }
}

fn recompose(d: &Decomp, r: &Restore, rings: &[Vec<f32>]) -> Image {
    let planes: Vec<Image> = d
        .planes
        .par_iter()
        .enumerate()
        .map(|(c, layers)| {
            let n = layers.len();
            let mut out = layers[n - 1].clone();
            for j in 0..n - 1 {
                let g = r.gains[j] as f32;
                let t = r.thresholds[c][j] as f32;
                let ring = rings.get(j);
                for (i, (o, &v)) in out.data.iter_mut().zip(&layers[j].data).enumerate() {
                    let ge = match ring {
                        Some(m) if r.limb.is_some() && g > 1.0 => 1.0 + (g - 1.0) * (1.0 - 0.8 * m[i]),
                        _ => g,
                    };
                    *o += ge * soft(v, t);
                }
            }
            out
        })
        .collect();
    let _ = (d.w, d.h);
    Image::from_planes(&planes)
}

/// Gains de Wiener par couche : (1/FTM)^force × RSB²/(RSB²+1).
fn wiener_gains(mtf: &[f64], snr2: &[f64], strength: f64) -> Vec<f64> {
    let mut g: Vec<f64> = (0..LEVELS)
        .map(|j| {
            let s = snr2.get(j).cloned().unwrap_or(1e3);
            let w = s / (s + 1.0);
            let inv = (1.0 / mtf[j]).powf(strength);
            (w * inv).clamp(0.15, 12.0)
        })
        .collect();
    g.push(1.0);
    g
}

/// Tonalité finale sans écrêtage : noir, blanc, genou doux.
pub(crate) fn tone(img: &Image, black: f64, white: f64) -> Image {
    let k = 1.0 / (white - black).max(1e-6);
    img.map(|v| {
        let a = ((v as f64 - black) * k * 0.95).max(0.0);
        let a = if a > 0.9 { 0.9 + 0.1 * ((a - 0.9) / 0.1).tanh() } else { a };
        a.clamp(0.0, 1.0) as f32
    })
}

fn radial_profile(l: &Image, d: Disc, r0: f64, r1: f64, step: f64) -> Vec<f64> {
    let nb = ((r1 - r0) / step).ceil().max(1.0) as usize;
    let mut s = vec![0.0f64; nb];
    let mut c = vec![0usize; nb];
    for y in 0..l.h {
        for x in 0..l.w {
            let r = (x as f64 - d.cx).hypot(y as f64 - d.cy);
            if r >= r0 && r < r1 {
                let b = ((r - r0) / step) as usize;
                if b < nb {
                    s[b] += l.data[y * l.w + x] as f64;
                    c[b] += 1;
                }
            }
        }
    }
    s.iter().zip(&c).map(|(a, n)| if *n > 0 { a / *n as f64 } else { f64::NAN }).collect()
}

/// Halo au limbe : liseré brillant ou anneau sombre ajoutés par
/// l'accentuation (relatif à la médiane du disque).
fn ringing(before: &Image, after: &Image, d: Disc, mask: &[bool]) -> f64 {
    let (lb, la) = (lum(before), lum(after));
    let (mb, ma) = (masked_median(&lb, mask).max(1e-6), masked_median(&la, mask).max(1e-6));
    let r0 = (d.r - 16.0).max(d.r * 0.5);
    let pb = radial_profile(&lb, d, r0, d.r + 6.0, 1.0);
    let pa = radial_profile(&la, d, r0, d.r + 6.0, 1.0);
    let mut worst = 0.0f64;
    for (b, a) in pb.iter().zip(&pa) {
        if b.is_finite() && a.is_finite() {
            worst = worst.max((a / ma - b / mb).abs());
        }
    }
    worst
}


/// Évaluation d'un réglage de restauration.
#[derive(Clone, Debug)]
pub struct RestoreEval {
    pub strength: f64,
    pub denoise_k: f64,
    /// énergie de détail (couches 1-4) relative à l'empilement brut
    pub detail: f64,
    /// bruit après traitement, relatif à la luminosité du disque
    pub noise_rel: f64,
    /// halo au limbe (relatif)
    pub ring: f64,
    pub sigma: f64,
    /// sous les contraintes « sans perte » (bruit invisible, pas de halo)
    pub ok: bool,
}

/// Image restaurée et finie.
#[derive(Clone)]
pub struct Rendered {
    pub image: Image,
    /// avant niveaux (couleurs équilibrées)
    pub linear: Image,
    pub white: f64,
    pub color_gains: Vec<f64>,
}

/// Restauration (ondelettes de Wiener + seuillage + protection du limbe +
/// finition) préparée une fois et rejouable avec d'autres réglages.
pub struct Finisher {
    ds: Decomp,
    da: Decomp,
    db: Decomp,
    mean_halves: Image,
    mtf: Vec<f64>,
    snr2: Vec<f64>,
    noise_layers: Vec<Vec<f64>>,
    rings: Vec<Vec<f32>>,
    disc: Option<Disc>,
    mask: Vec<bool>,
    color_balance: bool,
    fixed_gains: Option<Vec<f64>>,
    fixed_colors: Option<Vec<f64>>,
    pub black: f64,
    pub disc_med: f64,
    pub noise_target: f64,
    pub ring_max: f64,
    sig_ref: f64,
}

impl Finisher {
    fn new(stack: &Image, ha: &Image, hb: &Image, disc: Option<Disc>, psf: f64, color_balance: bool) -> Finisher {
        let size = stack.w;
        let mask = analysis_mask(stack.w, stack.h, disc);
        let mtf = layer_mtf(psf, LEVELS);
        let ((da, db), ds) = rayon::join(|| rayon::join(|| decompose(ha), || decompose(hb)), || decompose(stack));
        let (la, lb) = (lum(ha), lum(hb));
        let s = la.zip_map(&lb, |x, y| 0.5 * (x + y));
        let d = la.zip_map(&lb, |x, y| 0.5 * (x - y));
        let (ls, ld) = (atrous_decompose(&s, LEVELS), atrous_decompose(&d, LEVELS));
        let snr2: Vec<f64> = (0..LEVELS)
            .map(|j| {
                let vn = masked_mean_sq(&ld[j], &mask).max(1e-14);
                (masked_mean_sq(&ls[j], &mask) - vn).max(0.0) / vn
            })
            .collect();
        let noise_layers: Vec<Vec<f64>> = (0..stack.c)
            .map(|c| {
                let d = ha.channel(c).zip_map(&hb.channel(c), |x, y| 0.5 * (x - y));
                let ld = atrous_decompose(&d, LEVELS);
                (0..LEVELS).map(|j| masked_mean_sq(&ld[j], &mask).sqrt()).collect()
            })
            .collect();
        let rings: Vec<Vec<f32>> = match disc {
            Some(d) => (0..LEVELS).map(|j| limb_ring(stack.w, stack.h, d, j)).collect(),
            None => vec![],
        };
        let l_st = lum(stack);
        let bg_mask: Vec<bool> = match disc {
            Some(d) => (0..stack.w * stack.h).map(|i| ((i % size) as f64 - d.cx).hypot((i / size) as f64 - d.cy) > d.r + 6.0).collect(),
            None => vec![false; stack.w * stack.h],
        };
        let black = if bg_mask.iter().filter(|&&b| b).count() > 100 { masked_median(&l_st, &bg_mask) } else { stats::percentile(&l_st.data, 0.5) as f64 };
        let disc_med = masked_median(&l_st, &mask).max(1e-6);
        let c0 = critic(ha, hb, &mask);
        let sig_ref = c0.signal.iter().map(|v| v.sqrt()).sum::<f64>().max(1e-12);
        // bruit visible : au plus 1,5× celui de l'empilement (0,3 % du disque
        // toléré pour une image très propre, jamais plus de 0,75 %)
        let noise_target = (1.5 * c0.sigma).max(0.003 * disc_med).min(0.0075 * disc_med).max(1.05 * c0.sigma);
        let mean_halves = ha.zip_map(hb, |x, y| 0.5 * (x + y));
        Finisher { ds, da, db, mean_halves, mtf, snr2, noise_layers, rings, disc, mask, color_balance, fixed_gains: None, fixed_colors: None, black, disc_med, noise_target, ring_max: 0.05, sig_ref }
    }

    /// Gains d'ondelettes (7 couches) pour une force donnée.
    pub fn gains(&self, strength: f64) -> Vec<f64> {
        if let Some(g) = &self.fixed_gains {
            let mut g = g.clone();
            g.resize(LEVELS + 1, 1.0);
            return g;
        }
        wiener_gains(&self.mtf, &self.snr2, strength)
    }

    fn restore(&self, d: &Decomp, strength: f64, k: f64) -> Image {
        let r = Restore {
            gains: self.gains(strength),
            thresholds: self.noise_layers.iter().map(|nl| nl.iter().enumerate().map(|(j, v)| if j < 3 { k * v } else { 0.0 }).collect()).collect(),
            limb: self.disc,
        };
        recompose(d, &r, &self.rings)
    }

    /// Mesure un réglage sur les deux moitiés (bruit réel après traitement).
    pub fn evaluate(&self, strength: f64, k: f64) -> RestoreEval {
        let (pa, pb) = rayon::join(|| self.restore(&self.da, strength, k), || self.restore(&self.db, strength, k));
        let c = critic(&pa, &pb, &self.mask);
        let detail: f64 = c.signal.iter().map(|v| v.sqrt()).sum::<f64>() / self.sig_ref;
        let ring = match self.disc {
            Some(d) => ringing(&self.mean_halves, &pa.zip_map(&pb, |x, y| 0.5 * (x + y)), d, &self.mask),
            None => 0.0,
        };
        let ok = c.sigma <= self.noise_target && ring <= self.ring_max;
        RestoreEval { strength, denoise_k: k, detail, noise_rel: c.sigma / self.disc_med, ring, sigma: c.sigma, ok }
    }

    /// Meilleur réglage admissible (sinon le moins mauvais).
    pub fn choose(&self, evals: &[RestoreEval]) -> RestoreEval {
        let adm: Vec<&RestoreEval> = evals.iter().filter(|e| e.ok).collect();
        if !adm.is_empty() {
            let maxd = adm.iter().map(|e| e.detail).fold(f64::MIN, f64::max);
            return (*adm.iter().filter(|e| e.detail >= 0.985 * maxd).min_by(|a, b| a.sigma.total_cmp(&b.sigma)).unwrap()).clone();
        }
        let pen = |e: &RestoreEval| (e.sigma / self.noise_target).max(1.0) + (e.ring / self.ring_max).max(1.0) - 0.05 * e.detail;
        evals.iter().min_by(|a, b| pen(a).total_cmp(&pen(b))).unwrap().clone()
    }

    /// Restauration + couleurs + niveaux de l'empilement complet.
    pub fn render(&self, strength: f64, k: f64) -> Rendered {
        let mut img = self.restore(&self.ds, strength, k);
        let mut color_gains = Vec::new();
        if img.is_color() && (self.color_balance || self.fixed_colors.is_some()) {
            color_gains = match &self.fixed_colors {
                Some(g) if g.len() == 3 => g.clone(),
                Some(_) => vec![1.0; 3],
                None => {
                    let meds: Vec<f64> = (0..3).map(|c| masked_median(&img.channel(c), &self.mask).max(1e-6)).collect();
                    let m = (meds[0] + meds[1] + meds[2]) / 3.0;
                    // un tiers du chemin vers le gris : corrige une dominante de
                    // caméra sans éteindre les couleurs propres de la planète
                    meds.iter().map(|v| (m / v).powf(0.35).clamp(0.8, 1.25)).collect()
                }
            };
            let g = color_gains.clone();
            let bl = self.black as f32;
            let c = img.c;
            img.data.par_chunks_mut(c).for_each(|p| {
                for k in 0..3 {
                    p[k] = bl + (p[k] - bl) * g[k] as f32;
                }
            });
        }
        let l = lum(&img);
        let v: Vec<f32> = match self.disc {
            Some(_) => l.data.iter().zip(&self.mask).filter(|(_, &m)| m).map(|(v, _)| *v).collect(),
            None => l.data.clone(),
        };
        let white = if v.is_empty() { l.max_value() as f64 } else { stats::percentile(&v, 99.95) as f64 }.max(self.black + 1e-3);
        Rendered { image: tone(&img, self.black, white), linear: img, white, color_gains }
    }
}

// ------------------------------------------------------------- mémoire

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MemRecord {
    vector: Vec<f64>,
    keep_frac: f64,
    align_mode: String,
    box_ratio: f64,
    stack_method: String,
    strength: f64,
    denoise_k: f64,
    date: String,
}

fn memory_file(o: &AutoOptions) -> PathBuf {
    o.memory_path.clone().unwrap_or_else(|| paths::data_root().join("ai").join("planetary_memory.json"))
}

fn load_memory(o: &AutoOptions) -> Vec<MemRecord> {
    std::fs::read_to_string(memory_file(o)).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
}

fn save_memory(o: &AutoOptions, mut recs: Vec<MemRecord>) -> Result<()> {
    if recs.len() > 500 {
        recs.drain(..recs.len() - 500);
    }
    let p = memory_file(o);
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tmp = p.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(&recs)?)?;
    std::fs::rename(&tmp, &p)?;
    Ok(())
}

fn neighbours<'a>(mem: &'a [MemRecord], v: &[f64]) -> Vec<&'a MemRecord> {
    let mut d: Vec<(f64, &MemRecord)> = mem
        .iter()
        .filter(|r| r.vector.len() == v.len())
        .map(|r| (r.vector.iter().zip(v).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt(), r))
        .filter(|(dist, _)| *dist < 0.8)
        .collect();
    d.sort_by(|a, b| a.0.total_cmp(&b.0));
    d.into_iter().take(3).map(|x| x.1).collect()
}

// ---------------------------------------------------------- point d'entrée

/// Traitement planétaire entièrement automatique.
pub fn run_auto(frames: &dyn Frames, opts: &AutoOptions, progress: Progress) -> Result<AutoResult> {
    let t0 = Instant::now();
    let n = frames.len();
    if n == 0 {
        bail!("Aucune frame chargée");
    }
    let mut ctx = Ctx { opts, progress, decisions: Vec::new() };
    let an = analyse(frames, &mut ctx)?;
    let mut features = an.features.clone();
    let view = Centered { src: frames, idx: an.view_idx.clone(), origin: an.origin.clone(), size: an.size, fill: an.fill };
    let disc = an.disc_local;
    let size = an.size;
    let mask = analysis_mask(size, size, disc);
    let nv = view.len();

    // ---- tri
    let rk = rank_frames(&view, disc, features.noise, &mut ctx)?;
    features.quality_cv = rk.cv;

    // ---- mémoire
    let mem = if opts.use_memory { load_memory(opts) } else { Vec::new() };
    // le vecteur complet dépend de la distorsion (mesurée plus bas) : premier
    // passage sans elle, affiné ensuite
    let mut fv = features.vector();
    let nb0 = neighbours(&mem, &fv);
    let memory_hint = nb0.first().map(|r| (*r).clone());

    // ---- nombre d'images : candidats
    let min_k = if nv >= 10 { 5 } else { nv.min(2).max(1) };
    let mut pcts: Vec<f64> = match opts.effort {
        Effort::Rapide => vec![5.0, 10.0, 20.0, 35.0, 60.0],
        _ => vec![3.0, 5.0, 8.0, 12.0, 18.0, 25.0, 35.0, 50.0, 65.0, 80.0],
    };
    if let Some(h) = &memory_hint {
        pcts.push(h.keep_frac * 100.0);
    }
    let mut cands: Vec<usize> = pcts.iter().map(|p| ((nv as f64 * p / 100.0).round() as usize).clamp(min_k.min(nv), nv)).collect();
    cands.push(nv.min(min_k.max(2)));
    if nv <= 12 {
        cands.push(nv);
    }
    if let Some(l) = &opts.lock {
        cands = vec![((nv as f64 * l.keep_fraction).round() as usize).clamp(min_k.min(nv), nv)];
    }
    cands.sort();
    cands.dedup();
    let kmax = *cands.iter().max().unwrap();
    let (reference, shifts) = global_shifts(&view, &rk.order, kmax, &ctx)?;
    let curve = keep_curve(&view, &rk.order, &shifts, &cands, &mask, &ctx)?;
    // maximum ; à 0,3 % près, le plus petit nombre de frames (les plus nettes)
    let sc: Vec<f64> = curve.iter().map(|c| c.1.score).collect();
    let smax = sc.iter().cloned().fold(f64::MIN, f64::max);
    let bi = (0..sc.len()).find(|&i| sc[i] >= smax - 0.003 * smax.abs()).unwrap_or(0);
    let keep = curve[bi].0;
    let keep_pct = keep as f64 * 100.0 / n as f64;
    ctx.decide(
        "Sélection",
        format!("{} meilleures frames ({:.1} % de la vidéo)", keep, keep_pct),
        format!(
            "Maximum du critique à deux moitiés sur {} empilements d'essai ({} à {} frames) : au-delà, les frames floues coûtent plus de détails que le bruit gagné.",
            curve.len(),
            curve.first().map(|c| c.0).unwrap_or(0),
            curve.last().map(|c| c.0).unwrap_or(0)
        ),
    );
    let keep_curve_out: Vec<(f64, f64)> = curve.iter().map(|(k, c)| (*k as f64 * 100.0 / n as f64, c.score)).collect();

    // ---- turbulence
    ctx.step(0.52, "Analyse de la turbulence")?;
    let r = disc.map(|d| d.r).unwrap_or(size as f64 / 2.0);
    let bx_local = (((r / 2.5) / 8.0).round() as usize * 8).clamp(24, 96).min(size / 2).max(16);
    let distortion = local_distortion(&view, &rk.order, &shifts, &reference, disc, bx_local);
    features.distortion = distortion;
    fv = features.vector();
    let nbs = neighbours(&mem, &fv);
    let memory_hits = nbs.len();

    // ---- alignement / empilement : candidats
    let local_ok = r >= 35.0 && keep >= 6;
    let rule_mode = if !local_ok || distortion < 0.3 {
        planetary::ALIGN_MODES[1]
    } else if distortion > 0.6 && r >= 60.0 {
        planetary::ALIGN_MODES[3]
    } else {
        planetary::ALIGN_MODES[0]
    };
    let bx_grid = 24usize.max(size / 6).min(size / 2);
    let grid_n = ((2.0 * r / bx_grid as f64).round() as usize).clamp(3, 7);
    let ap_percent = 50.0f64;
    let mut modes: Vec<(&str, usize)> = vec![(rule_mode, if rule_mode.starts_with("Multi-points local") { bx_local } else { bx_grid })];
    if let Some(h) = nbs.first() {
        if let Some(m) = planetary::ALIGN_MODES.iter().find(|m| **m == h.align_mode) {
            let b = ((h.box_ratio * r).round() as usize).clamp(16, size / 2);
            modes.push((m, b));
        }
    }
    match opts.effort {
        Effort::Rapide => {}
        Effort::Equilibre => {
            modes.push((planetary::ALIGN_MODES[1], bx_grid));
            if local_ok {
                modes.push((planetary::ALIGN_MODES[3], bx_local));
            }
        }
        Effort::Maximal => {
            modes.push((planetary::ALIGN_MODES[1], bx_grid));
            if size >= 64 {
                modes.push((planetary::ALIGN_MODES[0], bx_grid));
            }
            if local_ok {
                modes.push((planetary::ALIGN_MODES[3], bx_local));
                let b2 = ((bx_local as f64 * 1.4 / 8.0).round() as usize * 8).min(size / 2);
                if b2 != bx_local {
                    modes.push((planetary::ALIGN_MODES[3], b2));
                }
                let b3 = ((bx_local as f64 * 0.7 / 8.0).round() as usize * 8).max(16);
                if b3 != bx_local {
                    modes.push((planetary::ALIGN_MODES[3], b3));
                }
            }
        }
    }
    let mut seen = std::collections::HashSet::new();
    modes.retain(|m| seen.insert((m.0, m.1)));
    if opts.effort == Effort::Rapide {
        modes.truncate(1);
    }
    let mut first_method = "Moyenne";
    if let Some(l) = &opts.lock {
        let m = planetary::ALIGN_MODES.iter().find(|m| **m == l.align_mode).cloned().unwrap_or(planetary::ALIGN_MODES[1]);
        let m = if m.starts_with("Multi-points local") && !local_ok { planetary::ALIGN_MODES[1] } else { m };
        modes = vec![(m, l.box_size.clamp(16, (size / 2).max(16)))];
        first_method = match l.stack_method.as_str() {
            "Sigma-clipping" | "Winsorisé" if keep < 15 => "Moyenne",
            "Sigma-clipping" => "Sigma-clipping",
            "Winsorisé" => "Winsorisé",
            "Moyenne pondérée" => "Moyenne pondérée",
            _ => "Moyenne",
        };
    }
    let sel: Vec<usize> = rk.order[..keep].to_vec();
    let sel_shifts: Vec<(f64, f64)> = shifts[..keep].to_vec();
    // poids de qualité (pour la moyenne pondérée) : rang fusionné
    let qweights: Vec<f64> = sel.iter().map(|&i| 0.25 + rk.fused[i]).collect();
    let mut trials: Vec<Trial> = Vec::new();
    let mut best: Option<(Trial, Image, Image, Image)> = None;
    let nt = modes.len();
    for (ti, (mode, bx)) in modes.iter().enumerate() {
        ctx.step(0.55 + 0.25 * ti as f64 / nt.max(1) as f64, &format!("Essai d'alignement : {}", mode))?;
        let (full, a, b) = stack_trial(&view, &sel, &sel_shifts, &reference, mode, first_method, *bx, grid_n, ap_percent, disc, &qweights)?;
        let c = critic(&a, &b, &mask);
        let label = if mode.starts_with("Global") { mode.to_string() } else { format!("{} (boîtes {} px)", mode, bx) };
        let t = Trial { label, mode: mode.to_string(), method: first_method.into(), bx: *bx, score: c.score };
        if best.as_ref().map(|b| t.score > b.0.score).unwrap_or(true) {
            best = Some((Trial { label: t.label.clone(), mode: t.mode.clone(), method: t.method.clone(), bx: t.bx, score: t.score }, full, a, b));
        }
        trials.push(t);
    }
    let (mut bt, mut stack, mut ha, mut hb) = best.unwrap();
    // méthodes d'empilement (alignements globaux / grille uniquement)
    if !bt.mode.starts_with("Multi-points local") && opts.effort != Effort::Rapide && opts.lock.is_none() {
        let mut methods = vec!["Moyenne pondérée"];
        if keep >= 15 {
            methods.push("Sigma-clipping");
        }
        if opts.effort == Effort::Maximal && keep >= 15 {
            methods.push("Winsorisé");
        }
        for m in methods {
            ctx.step(0.8, &format!("Essai d'empilement : {}", m))?;
            let (full, a, b) = stack_trial(&view, &sel, &sel_shifts, &reference, &bt.mode, m, bt.bx, grid_n, ap_percent, disc, &qweights)?;
            let c = critic(&a, &b, &mask);
            trials.push(Trial { label: format!("{} + {}", bt.label, m), mode: bt.mode.clone(), method: m.into(), bx: bt.bx, score: c.score });
            if c.score > bt.score * 1.002 {
                bt = Trial { label: format!("{} + {}", bt.label, m), mode: bt.mode.clone(), method: m.into(), bx: bt.bx, score: c.score };
                stack = full;
                ha = a;
                hb = b;
            }
        }
    }
    let trials_out: Vec<(String, f64)> = trials.iter().map(|t| (t.label.clone(), t.score)).collect();
    let others: Vec<String> = trials.iter().filter(|t| t.label != bt.label).map(|t| format!("{} : {:.2}", t.label, t.score)).collect();
    ctx.decide(
        "Alignement",
        format!("{}{}", bt.mode, if bt.mode.starts_with("Global") { String::new() } else { format!(", boîtes {} px", bt.bx) }),
        format!(
            "Distorsion locale mesurée {:.2} px (turbulence {}). Score {:.2}{}{}",
            distortion,
            if distortion < 0.3 {
                "faible"
            } else if distortion < 0.6 {
                "modérée"
            } else {
                "forte"
            },
            bt.score,
            if others.is_empty() { String::new() } else { format!(" ; autres essais — {}", others.join(" ; ")) },
            if memory_hits > 0 { format!(" ; {} vidéo(s) ressemblante(s) en mémoire consultée(s).", memory_hits) } else { ".".into() }
        ),
    );
    ctx.decide(
        "Empilement",
        bt.method.clone(),
        match bt.method.as_str() {
            "Sigma-clipping" => "Rejet des pixels aberrants (σ = 2,5) : meilleur score, des défauts ponctuels étaient présents.".to_string(),
            "Moyenne pondérée" => "Les meilleures frames pèsent davantage : meilleur compromis détail / bruit mesuré.".to_string(),
            "Winsorisé" => "Écrêtage winsorisé : meilleur score mesuré.".to_string(),
            _ => "Moyenne simple : aucune autre méthode ne l'a battue sur ces données.".to_string(),
        },
    );

    // ---- canaux RVB
    ctx.step(0.84, "Alignement des canaux couleur")?;
    let mut rgb_shifts = Vec::new();
    if stack.is_color() {
        let g = stack.channel(1);
        let mut out = [stack.clone(), ha.clone(), hb.clone()];
        for ch in [0usize, 2] {
            let (dx, dy, _) = phase_corr(&g, &stack.channel(ch));
            let (dx, dy) = if dx.hypot(dy) > 6.0 { (0.0, 0.0) } else { (dx, dy) };
            rgb_shifts.push((dx, dy));
            if dx.hypot(dy) > 0.05 {
                for o in out.iter_mut() {
                    let sh = apply_shift(&o.channel(ch), dx, dy);
                    o.set_channel(ch, &sh);
                }
            }
        }
        let [s, a, b] = out;
        stack = s;
        ha = a;
        hb = b;
        ctx.decide(
            "Couleurs",
            format!("R décalé de ({:.2}, {:.2}) px, B de ({:.2}, {:.2}) px", rgb_shifts[0].0, rgb_shifts[0].1, rgb_shifts[1].0, rgb_shifts[1].1),
            "Correction de la dispersion atmosphérique par corrélation de phase sur le canal vert.",
        );
    }
    let stack_raw = stack.clone();
    let c_stack = critic(&ha, &hb, &mask);

    // ---- modèle ONNX de débruitage (facultatif)
    let mut onnx_used = None;
    if opts.onnx {
        let models: Vec<ai::Model> = ai::list_models().into_iter().filter(|m| ai::model_supports_task(m, "denoise")).collect();
        if let Some(m) = models.first() {
            ctx.step(0.86, "Essai du modèle IA ONNX de débruitage")?;
            let prov = format!("onnx:{}", m.name);
            match (ai::run("denoise", &stack, &prov, 1.0, false), ai::run("denoise", &ha, &prov, 1.0, false), ai::run("denoise", &hb, &prov, 1.0, false)) {
                (Ok(s2), Ok(a2), Ok(b2)) if s2.same_shape(&stack) => {
                    let c2 = critic(&a2, &b2, &mask);
                    let keep_detail = (1..4).all(|j| c2.signal[j] >= 0.9 * c_stack.signal[j]);
                    if c2.sigma < 0.8 * c_stack.sigma && keep_detail {
                        ctx.decide("IA ONNX", format!("Modèle « {} » appliqué", m.name), format!("Bruit {:.2e} ramené à {:.2e} sans perte de détail mesurable.", c_stack.sigma, c2.sigma));
                        stack = s2;
                        ha = a2;
                        hb = b2;
                        onnx_used = Some(m.name.clone());
                    } else {
                        ctx.decide("IA ONNX", format!("Modèle « {} » refusé", m.name), "Il gommait des détails réels ou ne réduisait pas assez le bruit : débruitage par ondelettes à la place.");
                    }
                }
                (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => ctx.decide("IA ONNX", "Non utilisé", format!("{}", e)),
                _ => ctx.decide("IA ONNX", "Non utilisé", "Sortie du modèle de taille différente."),
            }
        } else {
            ctx.decide("IA ONNX", "Aucun modèle installé", "Débruitage par ondelettes adaptatif (aucun modèle de tâche « denoise » dans le dossier des modèles).");
        }
    }

    // ---- restauration : PSF, ondelettes de Wiener, débruitage
    ctx.step(0.88, "Mesure de la PSF")?;
    let psf = match disc.and_then(|d| psf_from_limb(&stack, d)) {
        Some(s) => {
            let s = s.clamp(0.5, 3.5);
            ctx.decide("PSF", format!("σ = {:.2} px (FWHM {:.2} px)", s, s * 2.3548), "Mesurée sur la pente du limbe (secteurs les plus nets, terminateur ignoré).");
            s
        }
        None => {
            ctx.decide("PSF", "σ = 1,20 px (par défaut)", "Pas de limbe exploitable : valeur typique, la force est ensuite ajustée par mesure.");
            1.2
        }
    };
    let mut fin = Finisher::new(&stack, &ha, &hb, disc, psf, opts.color_balance);
    if let Some(l) = &opts.lock {
        fin.fixed_gains = Some(l.wavelet_gains.clone());
        fin.fixed_colors = Some(l.color_gains.clone());
    }
    let c_stack = critic(&ha, &hb, &fin.mask);
    let (strengths, ks): (Vec<f64>, Vec<f64>) = match opts.effort {
        Effort::Rapide => (vec![0.0, 0.4, 0.7, 1.0], vec![0.0, 1.0, 2.0]),
        Effort::Equilibre => (vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.25], vec![0.0, 0.75, 1.5, 2.5]),
        Effort::Maximal => (vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4], vec![0.0, 0.5, 1.0, 1.5, 2.0, 3.0]),
    };
    let mut strengths = strengths;
    let mut ks = ks;
    if let Some(h) = nbs.first() {
        strengths.push(h.strength);
        ks.push(h.denoise_k);
    }
    ctx.step(0.9, "Réglage automatique des ondelettes et du débruitage")?;
    let mut combos: Vec<(f64, f64)> = strengths.iter().flat_map(|&s| ks.iter().map(move |&k| (s, k))).collect();
    if let Some(l) = &opts.lock {
        combos = vec![(l.strength, l.denoise_k)];
    }
    let evals: Vec<RestoreEval> = combos.par_iter().map(|&(s, k)| fin.evaluate(s, k)).collect();
    let chosen = if opts.lock.is_some() { evals[0].clone() } else { fin.choose(&evals) };
    if opts.lock.is_some() {
        ctx.decide("Plan commun", "Réglages verrouillés de la série", "Même sélection, alignement, empilement, ondelettes, débruitage et couleurs que la vidéo de référence : rendu identique d'une image à l'autre.");
    }
    let (strength, denoise_k) = (chosen.strength, chosen.denoise_k);
    let gains = fin.gains(strength);
    ctx.decide(
        "Ondelettes",
        format!("Gains {}", gains.iter().map(|g| format!("{:.2}", g)).collect::<Vec<_>>().join(" / ")),
        format!(
            "Filtre de Wiener par couche à partir de la PSF et du RSB mesuré de chaque couche ; force {:.2} retenue parmi {} essais (détails ×{:.2}, bruit visible {:.2} % du disque, halo {:.1} %).",
            strength,
            evals.len(),
            chosen.detail,
            chosen.noise_rel * 100.0,
            chosen.ring * 100.0
        ),
    );
    ctx.decide(
        "Débruitage",
        if denoise_k > 0.0 { format!("Seuillage doux {:.2} σ sur les 3 couches fines", denoise_k) } else { "Aucun seuillage nécessaire".to_string() },
        format!("Bruit cible ≤ {:.2} % de la luminosité du disque (sous le seuil de visibilité) ; le seuillage n'agit que là où le signal est sous le bruit.", fin.noise_target / fin.disc_med * 100.0),
    );
    if disc.is_some() {
        ctx.decide("Limbe", "Protection anti-halo active", "Accentuation réduite près du bord, à l'échelle de chaque couche, pour éviter liseré brillant et anneau sombre.");
    }

    // ---- finition : couleurs et niveaux
    ctx.step(0.96, "Finition (couleurs, niveaux)")?;
    let rendered = fin.render(strength, denoise_k);
    let final_img = rendered.image.clone();
    let (black, white, color_gains) = (fin.black, rendered.white, rendered.color_gains.clone());
    if !color_gains.is_empty() {
        ctx.decide(
            "Balance des blancs",
            format!("R ×{:.2}, V ×{:.2}, B ×{:.2}", color_gains[0], color_gains[1], color_gains[2]),
            "Équilibre partiel (environ un tiers) sur le disque : corrige la dominante de la caméra sans neutraliser les couleurs propres de la planète.",
        );
    }
    let clipped = final_img.data.iter().filter(|&&v| v >= 0.9995).count() as f64 / final_img.data.len().max(1) as f64;
    ctx.decide("Niveaux", format!("Noir {:.4}, blanc {:.4}", black, white), "Noir sur le fond du ciel, blanc au 99,95e centile du disque, genou doux au-dessus de 90 % : aucune zone brûlée, dynamique 32 bits conservée jusqu'à l'export 16 bits.");
    let disc_med = fin.disc_med;

    // ---- indicateurs
    let stack_snr = disc_med / c_stack.sigma.max(1e-9);
    let metrics = Metrics {
        stack_score: c_stack.score,
        stack_snr,
        snr_gain: stack_snr / features.snr.max(1e-6),
        noise_out: chosen.noise_rel,
        detail_gain: chosen.detail,
        clipped,
        ringing: chosen.ring,
    };
    let best_frame = {
        let b = view.get(rk.order[0]);
        let (dx, dy) = shifts[0];
        apply_shift(&b, dx, dy)
    };
    let plan = AutoPlan {
        quality_weights: rk.weights.clone(),
        keep,
        keep_percent: keep_pct,
        align_mode: bt.mode.clone(),
        box_size: bt.bx,
        grid_n,
        ap_percent,
        stack_method: bt.method.clone(),
        rgb_align: !rgb_shifts.is_empty(),
        rgb_shifts,
        psf_sigma: psf,
        onnx_denoise: onnx_used,
        strength,
        denoise_k,
        wavelet_gains: gains,
        limb_protection: disc.is_some(),
        color_gains,
        black,
        white,
    };
    // scores par frame vidéo
    let mut scores = vec![f64::NAN; n];
    for (k, &fi) in an.view_idx.iter().enumerate() {
        scores[fi] = rk.fused[k];
    }
    let order: Vec<usize> = rk.order.iter().map(|&k| an.view_idx[k]).collect();
    if opts.learn {
        let mut recs = load_memory(opts);
        recs.push(MemRecord {
            vector: fv,
            keep_frac: keep as f64 / nv as f64,
            align_mode: plan.align_mode.clone(),
            box_ratio: plan.box_size as f64 / r.max(1.0),
            stack_method: plan.stack_method.clone(),
            strength,
            denoise_k,
            date: chrono::Local::now().format("%Y-%m-%d %H:%M").to_string(),
        });
        if let Err(e) = save_memory(opts, recs) {
            ctx.decide("Mémoire", "Non enregistrée", format!("{}", e));
        }
    }
    ctx.step(1.0, "Terminé")?;
    Ok(AutoResult {
        image: final_img,
        linear: rendered.linear,
        disc: disc.map(|d| (d.cx, d.cy, d.r)),
        stack: stack_raw,
        best_frame,
        features,
        plan,
        decisions: ctx.decisions,
        scores,
        order,
        rejected: an.rejected.clone(),
        keep_curve: keep_curve_out,
        trials: trials_out,
        restoration_trials: evals.iter().map(|e| (e.strength, e.denoise_k, e.detail, e.noise_rel, e.ok)).collect(),
        finisher: fin,
        metrics,
        memory_hits,
        seconds: t0.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn critic_prefers_lower_noise() {
        let mut rng = Rng(7);
        let base = Image::from_fn(96, 96, |x, y| (0.4 + 0.2 * ((x as f32) * 0.3).sin() * ((y as f32) * 0.25).cos()) as f32);
        let mk = |s: f32, rng: &mut Rng| {
            let mut o = base.clone();
            o.data.iter_mut().for_each(|v| *v += s * rng.normal() as f32);
            o
        };
        let mask = vec![true; 96 * 96];
        let lo = critic(&mk(0.01, &mut rng), &mk(0.01, &mut rng), &mask);
        let hi = critic(&mk(0.05, &mut rng), &mk(0.05, &mut rng), &mask);
        assert!(lo.score > hi.score, "{} {}", lo.score, hi.score);
        assert!((lo.sigma - 0.01 / 2f64.sqrt()).abs() < 0.002, "{}", lo.sigma);
    }

    #[test]
    fn mtf_decreases_with_blur() {
        let a = layer_mtf(0.8, 4);
        let b = layer_mtf(2.0, 4);
        assert!(a[0] > b[0] && a[1] > b[1]);
        assert!(b[0] < b[3]);
    }

    #[test]
    fn psf_from_sharp_and_blurred_limb() {
        let disc = Image::from_fn(160, 160, |x, y| if ((x as f64 - 80.0).powi(2) + (y as f64 - 80.0).powi(2)).sqrt() < 50.0 { 0.6 } else { 0.02 });
        let d = Disc { cx: 80.0, cy: 80.0, r: 50.0 };
        let s1 = psf_from_limb(&ndimage::gaussian(&disc, 1.0), d).unwrap();
        let s2 = psf_from_limb(&ndimage::gaussian(&disc, 2.5), d).unwrap();
        assert!(s2 > s1 + 0.8, "{} {}", s1, s2);
        assert!((s2 - 2.5).abs() < 0.8, "{}", s2);
    }
}

#[cfg(test)]
mod disc_tests {
    use super::*;

    /// Soleil partiellement éclipsé : le cercle trouvé est celui du Soleil,
    /// pas celui de la Lune (de rayon voisin).
    #[test]
    fn robust_disc_on_eclipse_crescent() {
        let (sx, sy, sr) = (120.3, 95.7, 60.0);
        let (mx, my, mr) = (150.0, 80.0, 62.0);
        let img = Image::from_fn(240, 200, |x, y| {
            let (xf, yf) = (x as f64, y as f64);
            let ds = (xf - sx).hypot(yf - sy);
            let dm = (xf - mx).hypot(yf - my);
            if ds < sr && dm > mr {
                (0.7 * (1.0 - 0.5 * (ds / sr).powi(2))) as f32
            } else {
                0.02
            }
        });
        let img = ndimage::gaussian(&img, 1.0);
        let (cx, cy, r) = robust_disc(&img, None).unwrap();
        assert!((cx - sx).abs() < 0.8 && (cy - sy).abs() < 0.8 && (r - sr).abs() < 1.0, "{} {} {}", cx, cy, r);
        let (cx, cy, _) = robust_disc(&img, Some(sr)).unwrap();
        assert!((cx - sx).abs() < 0.5 && (cy - sy).abs() < 0.5, "{} {}", cx, cy);
    }
}
