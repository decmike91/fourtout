//! Time-lapse multi-vidéos (éclipse, rotation d'une planète, passage
//! d'ombre…) : chaque vidéo donne une image, avec **le même traitement**, **le
//! même cadrage** et **le même rendu**, puis l'ensemble devient une animation.
//!
//! 1. **Référence** : la vidéo la plus propre (meilleur rapport signal/bruit)
//!    est traitée par le moteur automatique complet ; ses choix (sélection,
//!    alignement, empilement, ondelettes, débruitage, couleurs) forment le
//!    *plan commun* ([`AutoLock`]).
//! 2. **Série** : toutes les vidéos sont traitées avec ce plan verrouillé ;
//!    les images restaurées sont gardées en linéaire (32 bits).
//! 3. **Cadrage commun** : mode « disque » — le cercle du limbe est retrouvé
//!    sur chaque image, même partiellement éclipsée (RANSAC sur les bords
//!    clairs à l'intérieur, rayon commun), et placé au centre exact d'un
//!    canevas de taille fixe ; mode « surface » (Lune, Soleil en gros plan) —
//!    recalage en chaîne par corrélation de phase. Option : correction de la
//!    rotation de champ (monture altazimutale).
//! 4. **Rendu harmonisé** (rejouable sans retraitement) : luminosité réelle
//!    (mêmes niveaux pour tout), dé-scintillement (les variations rapides de
//!    transparence sont lissées, la tendance réelle est gardée) ou
//!    normalisée ; noir et blanc communs, genou doux, horodatage facultatif.
//! 5. **Animation** : images intermédiaires en fondu pour la fluidité, export
//!    MP4 / AVI (ffmpeg), GIF, ou série d'images numérotées.

use crate::draw::Canvas;
use crate::frames::Frames;
use crate::geom::{self, Interp};
use crate::image::Image;
use crate::io;
use crate::ndimage::{self, Border};
use crate::planetary_auto::{self as pa, AutoLock, Decision, Effort};
use crate::registration::{self, phase_corr};
use crate::stats;
use anyhow::{bail, Result};
use chrono::NaiveDateTime;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Une vidéo de la série.
#[derive(Clone)]
pub struct SeqInput {
    pub name: String,
    pub frames: Arc<dyn Frames + Send + Sync>,
    /// instant de prise de vue (milieu de la vidéo), si connu
    pub time: Option<NaiveDateTime>,
}

/// Instant médian d'une vidéo SER (horodatages des frames, sinon en-tête).
pub fn ser_time(meta: &io::ser::SerMeta) -> Option<NaiveDateTime> {
    let ts: Vec<String> = meta.timestamps.iter().flatten().cloned().collect();
    if let Some(m) = crate::derotation::mid_time(&ts) {
        if let Ok(t) = crate::timeutil::parse_iso_utc(&m) {
            return Some(t);
        }
    }
    meta.date_utc.as_ref().and_then(|d| crate::timeutil::parse_iso_utc(d).ok())
}

/// Charge des vidéos SER (ou les SER d'un dossier), triées par heure puis
/// par nom.
pub fn load_inputs(paths: &[PathBuf], max_frames: usize) -> Result<Vec<SeqInput>> {
    let mut files: Vec<PathBuf> = Vec::new();
    for p in paths {
        if p.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(p)?.filter_map(|e| e.ok().map(|e| e.path())).filter(|f| io::ext_of(f) == "ser").collect();
            v.sort();
            files.extend(v);
        } else {
            files.push(p.clone());
        }
    }
    if files.is_empty() {
        bail!("Aucune vidéo SER");
    }
    let mut out = Vec::new();
    for f in files {
        if io::ext_of(&f) != "ser" {
            bail!("{} : seules les vidéos SER sont acceptées pour une série", f.display());
        }
        let (frames, meta) = io::ser::load_ser(&f, max_frames)?;
        let name = f.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        out.push(SeqInput { name, time: ser_time(&meta), frames: Arc::new(frames) });
    }
    sort_inputs(&mut out);
    Ok(out)
}

/// Tri chronologique (les vidéos sans heure gardent l'ordre des noms).
pub fn sort_inputs(v: &mut [SeqInput]) {
    v.sort_by(|a, b| match (a.time, b.time) {
        (Some(x), Some(y)) => x.cmp(&y).then(a.name.cmp(&b.name)),
        _ => a.name.cmp(&b.name),
    });
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Framing {
    /// disque si l'objet est un disque entier, sinon surface
    Auto,
    /// centre du limbe fixe (planète, Soleil ou Lune entiers, éclipses)
    Disc,
    /// recalage par corrélation (gros plan, objet plus grand que le champ)
    Surface,
}

impl Framing {
    pub const ALL: [Framing; 3] = [Framing::Auto, Framing::Disc, Framing::Surface];
    pub fn label(&self) -> &'static str {
        match self {
            Framing::Auto => "Automatique",
            Framing::Disc => "Disque centré (limbe)",
            Framing::Surface => "Surface (corrélation)",
        }
    }
    pub fn parse(s: &str) -> Result<Framing> {
        let l = s.to_lowercase();
        Ok(if l.starts_with("auto") {
            Framing::Auto
        } else if l.starts_with("disq") || l.starts_with("disc") {
            Framing::Disc
        } else if l.starts_with("surf") {
            Framing::Surface
        } else {
            bail!("Cadrage inconnu : {} (auto, disque, surface)", s)
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Brightness {
    /// mêmes niveaux pour toutes les images (variations réelles visibles)
    Real,
    /// variations rapides (voile, transparence) lissées, tendance gardée
    Deflicker,
    /// même luminosité pour toutes les images
    Normalized,
}

impl Brightness {
    pub const ALL: [Brightness; 3] = [Brightness::Deflicker, Brightness::Real, Brightness::Normalized];
    pub fn label(&self) -> &'static str {
        match self {
            Brightness::Real => "Réelle (niveaux communs)",
            Brightness::Deflicker => "Dé-scintillement (recommandé)",
            Brightness::Normalized => "Normalisée (constante)",
        }
    }
    pub fn parse(s: &str) -> Result<Brightness> {
        let l = s.to_lowercase();
        Ok(if l.starts_with("r") {
            Brightness::Real
        } else if l.starts_with("d") || l.starts_with("li") {
            Brightness::Deflicker
        } else if l.starts_with("n") || l.starts_with("c") {
            Brightness::Normalized
        } else {
            bail!("Luminosité inconnue : {} (reelle, descintillement, normalisee)", s)
        })
    }
}

#[derive(Clone, Debug)]
pub struct SeqOptions {
    pub effort: Effort,
    /// vidéo de référence (indice dans l'ordre chronologique) ; None = auto
    pub reference: Option<usize>,
    pub framing: Framing,
    /// marge autour du disque (fraction du rayon)
    pub margin: f64,
    /// correction de la rotation de champ (monture altazimutale)
    pub field_rotation: bool,
    pub use_memory: bool,
    pub onnx: bool,
    pub color_balance: bool,
    pub cancel: Option<Arc<AtomicBool>>,
}

impl Default for SeqOptions {
    fn default() -> Self {
        SeqOptions { effort: Effort::Equilibre, reference: None, framing: Framing::Auto, margin: 0.25, field_rotation: false, use_memory: true, onnx: true, color_balance: true, cancel: None }
    }
}

/// Bilan d'une image de la série.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SeqItem {
    pub name: String,
    pub time: Option<String>,
    pub n_frames: usize,
    pub rejected: usize,
    pub keep: usize,
    /// centre de l'objet dans l'image empilée et rayon (mode disque)
    pub center: (f64, f64),
    pub radius: f64,
    /// rotation de champ corrigée (degrés)
    pub rotation: f64,
    /// luminosité de l'objet (linéaire) et fond
    pub level: f64,
    pub background: f64,
    /// RSB de l'empilement
    pub snr: f64,
    pub error: Option<String>,
}

/// Série traitée, cadrée, en linéaire (rendu rejouable par [`render`]).
pub struct SeqProcessed {
    /// images cadrées (taille commune), avant niveaux
    pub frames: Vec<Image>,
    /// bilan, dans le même ordre que `frames`
    pub items: Vec<SeqItem>,
    /// vidéos écartées (erreur)
    pub failed: Vec<SeqItem>,
    pub reference: usize,
    pub lock: AutoLock,
    pub framing: Framing,
    pub disc_radius: f64,
    pub decisions: Vec<Decision>,
    pub reference_decisions: Vec<Decision>,
    pub seconds: f64,
}

type Progress<'a> = Option<&'a (dyn Fn(f64, &str) + Sync)>;

fn lum(img: &Image) -> Image {
    img.mean_lum()
}

fn check_cancel(c: &Option<Arc<AtomicBool>>) -> Result<()> {
    if c.as_ref().map(|c| c.load(Ordering::Relaxed)).unwrap_or(false) {
        bail!("Traitement annulé.");
    }
    Ok(())
}

/// Rapport signal/bruit rapide d'une vidéo (quelques frames).
fn quick_snr(f: &dyn Frames) -> f64 {
    let n = f.len();
    let idx: Vec<usize> = (0..n).step_by((n / 8).max(1)).take(8).collect();
    let v: Vec<f64> = idx
        .par_iter()
        .map(|&i| {
            let im = f.get(i);
            let l = lum(&im);
            let sig = stats::percentile(&l.data, 99.0) as f64 - stats::percentile(&l.data, 5.0) as f64;
            sig / crate::restoration::noise_sigma(&im).max(1e-6)
        })
        .collect();
    stats::median_f64(&v)
}

/// Recadre `img` pour que (cx, cy) tombe au centre d'un canevas `size`.
fn center_on(img: &Image, cx: f64, cy: f64, size: usize, angle_deg: f64) -> Image {
    let half = size as f64 / 2.0;
    let (ca, sa) = (angle_deg.to_radians().cos(), angle_deg.to_radians().sin());
    geom::remap_fn(img, size, size, Interp::Cubic, Border::Nearest, |x, y| {
        let (u, v) = (x as f64 + 0.5 - half, y as f64 + 0.5 - half);
        // rotation de la sortie autour du centre
        let (ru, rv) = (ca * u - sa * v, sa * u + ca * v);
        (cx + ru, cy + rv)
    })
}

/// Traitement complet de la série (sans le rendu final).
pub fn run_timelapse(inputs: &[SeqInput], o: &SeqOptions, progress: Progress) -> Result<SeqProcessed> {
    let t0 = std::time::Instant::now();
    let n = inputs.len();
    if n == 0 {
        bail!("Aucune vidéo dans la série");
    }
    let report = |f: f64, m: &str| {
        if let Some(p) = progress {
            p(f.clamp(0.0, 1.0), m);
        }
    };
    let mut decisions: Vec<Decision> = Vec::new();
    let mut decide = |s: &str, c: String, r: String| decisions.push(Decision { step: s.into(), choice: c, reason: r });

    // ---- référence
    report(0.0, "Choix de la vidéo de référence");
    let reference = match o.reference {
        Some(r) if r < n => {
            decide("Référence", format!("{} (choisie)", inputs[r].name), "Vidéo imposée par l'utilisateur.".into());
            r
        }
        _ => {
            let snrs: Vec<f64> = inputs.iter().map(|i| quick_snr(&*i.frames)).collect();
            let best = (0..n).max_by(|&a, &b| snrs[a].total_cmp(&snrs[b])).unwrap();
            decide("Référence", inputs[best].name.clone(), format!("Meilleur rapport signal/bruit de la série ({:.0}, médiane {:.0}) : c'est elle qui fixe le plan commun.", snrs[best], stats::median_f64(&snrs)));
            best
        }
    };
    check_cancel(&o.cancel)?;
    let per = 1.0 / (n as f64 + 0.6);
    let base = pa::AutoOptions { effort: o.effort, use_memory: o.use_memory, learn: true, onnx: o.onnx, color_balance: o.color_balance, cancel: o.cancel.clone(), memory_path: None, lock: None };
    let sub = |k: usize, label: String| move |f: f64, m: &str| report(per * (k as f64 + f), &format!("{} — {}", label, m));
    let pf = sub(0, format!("Référence {}", inputs[reference].name));
    let ref_res = pa::run_auto(&*inputs[reference].frames, &base, Some(&pf))?;
    let lock = AutoLock::from_result(&ref_res);
    let reference_decisions = ref_res.decisions.clone();
    decide(
        "Plan commun",
        format!("{} · {:.0} % des frames · {} · force {:.2} · débruitage {:.1} σ", lock.align_mode, lock.keep_fraction * 100.0, lock.stack_method, lock.strength, lock.denoise_k),
        "Appliqué à l'identique à toutes les vidéos : mêmes gains d'ondelettes, même seuil de débruitage relatif au bruit, mêmes gains de couleur.".into(),
    );

    // ---- série (plan verrouillé)
    struct Done {
        idx: usize,
        linear: Image,
        disc: Option<(f64, f64, f64)>,
        item: SeqItem,
        mode: String,
    }
    let mut done: Vec<Done> = Vec::new();
    let mut failed = Vec::new();
    let mut k = 1usize;
    for (i, inp) in inputs.iter().enumerate() {
        check_cancel(&o.cancel)?;
        let mut item = SeqItem { name: inp.name.clone(), time: inp.time.map(|t| crate::timeutil::iso_ms(&t)), n_frames: inp.frames.len(), ..Default::default() };
        let res = if i == reference {
            Ok(None)
        } else {
            let pf = sub(k, format!("{}/{} {}", k + 1, n, inp.name));
            k += 1;
            let opts = pa::AutoOptions { learn: false, use_memory: false, lock: Some(lock.clone()), ..base.clone() };
            pa::run_auto(&*inp.frames, &opts, Some(&pf)).map(Some)
        };
        match res {
            Ok(r) => {
                let r = r.as_ref().unwrap_or(&ref_res);
                item.rejected = r.features.rejected;
                item.keep = r.plan.keep;
                item.snr = r.metrics.stack_snr;
                done.push(Done { idx: i, linear: r.linear.clone(), disc: r.disc, item, mode: r.features.mode.clone() });
            }
            Err(e) => {
                if o.cancel.as_ref().map(|c| c.load(Ordering::Relaxed)).unwrap_or(false) {
                    bail!("Traitement annulé.");
                }
                item.error = Some(format!("{:#}", e));
                failed.push(item);
            }
        }
    }
    if done.is_empty() {
        bail!("Aucune vidéo n'a pu être traitée");
    }
    if !failed.is_empty() {
        decide("Vidéos écartées", format!("{}", failed.len()), failed.iter().map(|f| format!("{} : {}", f.name, f.error.clone().unwrap_or_default())).collect::<Vec<_>>().join(" ; "));
    }
    done.sort_by_key(|d| d.idx);
    // toutes en couleur si l'une l'est
    let color = done.iter().any(|d| d.linear.c == 3);
    if color {
        for d in done.iter_mut() {
            if d.linear.c != 3 {
                d.linear = d.linear.to_rgb();
            }
        }
    }

    // ---- cadrage commun
    report(per * (n as f64 + 0.1), "Cadrage commun");
    let ref_pos = done.iter().position(|d| d.idx == reference).unwrap_or(0);
    // heures de prise de vue (minutes) ; à défaut, le rang
    let times: Vec<f64> = if done.iter().all(|d| inputs[d.idx].time.is_some()) {
        let t0 = inputs[done[0].idx].time.unwrap();
        done.iter().map(|d| (inputs[d.idx].time.unwrap() - t0).num_milliseconds() as f64 / 60000.0).collect()
    } else {
        (0..done.len()).map(|k| k as f64).collect()
    };
    let mut framing = o.framing;
    let first_pass: Vec<Option<(f64, f64, f64)>> = done.par_iter().map(|d| pa::robust_disc(&d.linear, None).or(d.disc)).collect();
    if framing == Framing::Auto {
        let ok = first_pass.iter().filter(|c| c.is_some()).count();
        framing = if done[ref_pos].mode == "disque" && ok * 10 >= done.len() * 7 { Framing::Disc } else { Framing::Surface };
    }
    let frames_out: Vec<Image>;
    let mut items: Vec<SeqItem> = Vec::new();
    let mut disc_radius = 0.0;
    if framing == Framing::Disc {
        let radii: Vec<f64> = first_pass.iter().flatten().map(|c| c.2).collect();
        if radii.is_empty() {
            bail!("Cadrage « disque » : aucun limbe détecté ; essayez le cadrage « surface ».");
        }
        let r = stats::median_f64(&radii);
        disc_radius = r;
        // second passage, rayon commun : centres cohérents d'une image à l'autre
        let centers: Vec<Option<(f64, f64, f64)>> = done.par_iter().zip(&first_pass).map(|(d, f)| pa::robust_disc(&d.linear, Some(r)).or(*f)).collect();
        let size = ((2.0 * r * (1.0 + o.margin.clamp(0.0, 2.0))).ceil() as usize).div_ceil(2) * 2;
        let size = size.max(32);
        let mut missing = 0;
        let mut angles = vec![0.0f64; done.len()];
        let centered: Vec<Image> = done
            .par_iter()
            .zip(&centers)
            .map(|(d, c)| {
                let (cx, cy) = c.map(|c| (c.0, c.1)).unwrap_or((d.linear.w as f64 / 2.0, d.linear.h as f64 / 2.0));
                center_on(&d.linear, cx, cy, size, 0.0)
            })
            .collect();
        let centered = if o.field_rotation {
            report(per * (n as f64 + 0.3), "Rotation de champ");
            angles = chain_rotation(&centered, ref_pos, Some(r), &times);
            centered.par_iter().zip(&angles).map(|(im, &a)| if a != 0.0 { registration::apply_shift_rot(im, 0.0, 0.0, a) } else { im.clone() }).collect()
        } else {
            centered
        };
        for (j, d) in done.iter().enumerate() {
            let mut it = d.item.clone();
            match centers[j] {
                Some(c) => {
                    it.center = (c.0, c.1);
                    it.radius = c.2;
                }
                None => missing += 1,
            }
            it.rotation = angles[j];
            items.push(it);
        }
        frames_out = centered;
        decide(
            "Cadrage",
            format!("Disque centré, rayon commun {:.1} px, canevas {}×{}", r, size, size),
            format!(
                "Le limbe est retrouvé sur chaque image (même partiellement éclipsée : seuls les bords clairs à l'intérieur comptent), puis placé au centre exact au dixième de pixel.{}",
                if missing > 0 { format!(" {} image(s) sans limbe exploitable : centrées sur leur cadrage d'origine.", missing) } else { String::new() }
            ),
        );
    } else {
        // surface : recalage en chaîne (les changements lents restent suivis)
        let mw = done.iter().map(|d| d.linear.w).min().unwrap();
        let mh = done.iter().map(|d| d.linear.h).min().unwrap();
        let imgs: Vec<Image> = done.iter().map(|d| d.linear.crop((d.linear.w - mw) / 2, (d.linear.h - mh) / 2, mw, mh)).collect();
        let steps: Vec<(f64, f64)> = (1..imgs.len()).into_par_iter().map(|j| {
            let (dx, dy, _) = phase_corr(&imgs[j - 1], &imgs[j]);
            (dx, dy)
        }).collect();
        let mut shifts = vec![(0.0, 0.0); imgs.len()];
        for j in 1..imgs.len() {
            shifts[j] = (shifts[j - 1].0 + steps[j - 1].0, shifts[j - 1].1 + steps[j - 1].1);
        }
        // référence au centre : on retire le décalage de la référence
        let (rx, ry) = shifts[ref_pos];
        for s in shifts.iter_mut() {
            s.0 -= rx;
            s.1 -= ry;
        }
        let mut angles = vec![0.0; imgs.len()];
        if o.field_rotation {
            let shifted: Vec<Image> = imgs.par_iter().zip(&shifts).map(|(im, s)| registration::apply_shift(im, s.0, s.1)).collect();
            angles = chain_rotation(&shifted, ref_pos, None, &times);
        }
        let (x0, y0, cw, ch) = crate::animation::common_crop(mw, mh, &shifts, &angles, None, None);
        let side = cw.min(ch) / 2 * 2;
        frames_out = imgs
            .par_iter()
            .zip(&shifts)
            .zip(&angles)
            .map(|((im, s), &a)| {
                let cx = x0 as f64 + cw as f64 / 2.0 - s.0;
                let cy = y0 as f64 + ch as f64 / 2.0 - s.1;
                let c = center_on(im, cx, cy, side, 0.0);
                if a != 0.0 {
                    registration::apply_shift_rot(&c, 0.0, 0.0, a)
                } else {
                    c
                }
            })
            .collect();
        for (j, d) in done.iter().enumerate() {
            let mut it = d.item.clone();
            it.center = (mw as f64 / 2.0 - shifts[j].0, mh as f64 / 2.0 - shifts[j].1);
            it.rotation = angles[j];
            items.push(it);
        }
        decide("Cadrage", format!("Surface, recalage en chaîne, cadre commun {}×{}", side, side), "Chaque image est recalée sur la précédente par corrélation de phase ; le cadre final est la zone présente sur toutes les images.".into());
    }
    if o.field_rotation {
        let amax = items.iter().map(|i| i.rotation.abs()).fold(0.0, f64::max);
        decide("Rotation de champ", format!("jusqu'à {:.2}°", amax), "Rotation estimée entre images successives et cumulée (monture altazimutale).".into());
    }

    // ---- mesures de luminosité (pour le rendu)
    report(per * (n as f64 + 0.5), "Mesure des luminosités");
    let dr = if framing == Framing::Disc { Some(disc_radius) } else { None };
    let levels: Vec<(f64, f64)> = frames_out.par_iter().map(|im| object_levels(im, dr)).collect();
    let rel = relative_levels(&frames_out, &levels.iter().map(|l| l.1).collect::<Vec<_>>(), dr, ref_pos);
    let ref_level = levels[ref_pos].0;
    for ((it, (_, bg)), r) in items.iter_mut().zip(levels).zip(rel) {
        it.level = ref_level * r;
        it.background = bg;
    }
    report(1.0, "Terminé");
    Ok(SeqProcessed {
        frames: frames_out,
        items,
        failed,
        reference: ref_pos,
        lock,
        framing,
        disc_radius,
        decisions,
        reference_decisions,
        seconds: t0.elapsed().as_secs_f64(),
    })
}

/// Luminosité de chaque image relative à la référence, par rapports pixel à
/// pixel entre images successives, sur les seules zones éclairées dans les
/// deux : insensible à la forme de l'objet (croissant d'éclipse, ombre,
/// rotation de la planète), sensible aux voiles et à l'exposition.
fn relative_levels(frames: &[Image], bgs: &[f64], disc_r: Option<f64>, ref_pos: usize) -> Vec<f64> {
    let n = frames.len();
    let lums: Vec<Image> = frames.par_iter().zip(bgs).map(|(f, &b)| lum(f).map(|v| v - b as f32)).collect();
    // zones franchement éclairées (érodées : les bords flous d'un croissant
    // fin ne comptent pas)
    let lit: Vec<Vec<bool>> = lums
        .par_iter()
        .map(|l| {
            let t = 0.35 * stats::percentile(&l.data, 99.5);
            let er = ndimage::minimum_filter(l, 5);
            let (w, h) = (l.w, l.h);
            (0..w * h)
                .map(|i| {
                    let inside = match disc_r {
                        Some(rr) => ((i % w) as f64 + 0.5 - w as f64 / 2.0).hypot((i / w) as f64 + 0.5 - h as f64 / 2.0) <= 0.95 * rr,
                        None => true,
                    };
                    inside && er.data[i] > t
                })
                .collect()
        })
        .collect();
    let ratio = |a: usize, b: usize| -> Option<f64> {
        let mut r = Vec::new();
        for i in 0..lums[a].data.len() {
            if lit[a][i] && lit[b][i] {
                r.push((lums[b].data[i] as f64 / lums[a].data[i] as f64).ln() as f32);
            }
        }
        if r.len() >= 150 {
            Some(stats::median(&r) as f64)
        } else {
            None
        }
    };
    let mut acc = vec![0.0f64; n];
    for j in 1..n {
        // image précédente la plus proche ayant assez de zone commune ; sinon
        // aucune information : on garde le niveau précédent
        acc[j] = acc[j - 1];
        for k in (j.saturating_sub(3)..j).rev() {
            if let Some(lr) = ratio(k, j) {
                acc[j] = acc[k] + lr;
                break;
            }
        }
    }
    acc.iter().map(|a| (a - acc[ref_pos]).exp()).collect()
}

/// Rotation (degrés) à appliquer à `cur` (apply_shift_rot) pour le
/// superposer à `prev`. Seuls comptent les détails internes éclairés dans les
/// deux images (taches, reliefs) : le limbe, invariant par rotation, et le
/// bord mobile de la Lune pendant une éclipse sont exclus. Retourne 0 si la
/// mesure n'est pas fiable (None).
pub fn masked_rotation(prev: &Image, cur: &Image, disc_r: Option<f64>, max_deg: f64) -> Option<f64> {
    let (lp, lc) = (lum(prev), lum(cur));
    let (w, h) = (lp.w, lp.h);
    if lc.w != w || lc.h != h {
        return None;
    }
    let hp = |l: &Image| l.zip_map(&ndimage::gaussian(l, 2.0), |a, b| a - b);
    // éclairement à grande échelle (les taches, petites, n'en sortent pas),
    // puis érosion : bords du limbe et de la Lune exclus
    let lit = |l: &Image| -> Vec<bool> {
        let g = ndimage::gaussian(l, 6.0);
        let t = 0.35 * stats::percentile(&g.data, 99.5);
        let m: Vec<bool> = g.data.iter().map(|&v| v > t).collect();
        ndimage::binary_erosion(&m, l.w, l.h, 12)
    };
    let (hpp, mp) = (hp(&lp), lit(&lp));
    let inside: Vec<bool> = (0..w * h)
        .map(|i| match disc_r {
            Some(r) => ((i % w) as f64 + 0.5 - w as f64 / 2.0).hypot((i / w) as f64 + 0.5 - h as f64 / 2.0) < 0.9 * r,
            None => {
                let (x, y) = (i % w, i / w);
                let b = w.min(h) / 8;
                x >= b && y >= b && x + b < w && y + b < h
            }
        })
        .collect();
    // score insensible à une petite translation résiduelle : pic de la
    // corrélation (recherche du décalage) des détails masqués
    let score = |a: f64| -> f64 {
        let rc = if a == 0.0 { lc.clone() } else { registration::apply_shift_rot(&lc, 0.0, 0.0, a) };
        let (hc, mc) = (hp(&rc), lit(&rc));
        let m: Vec<bool> = (0..w * h).map(|i| inside[i] && mp[i] && mc[i]).collect();
        let n = m.iter().filter(|&&b| b).count();
        if n < 400 {
            return f64::NAN;
        }
        let pa = Image::from_vec(w, h, 1, hpp.data.iter().zip(&m).map(|(v, &k)| if k { *v } else { 0.0 }).collect());
        let pb = Image::from_vec(w, h, 1, hc.data.iter().zip(&m).map(|(v, &k)| if k { *v } else { 0.0 }).collect());
        let (dx, dy, _) = phase_corr(&pa, &pb);
        if dx.hypot(dy) > 2.0 {
            return f64::NAN;
        }
        let pb = registration::apply_shift(&pb, dx, dy);
        let (mut sab, mut saa, mut sbb) = (0.0f64, 0.0f64, 0.0f64);
        for i in 0..w * h {
            if m[i] {
                let (x, y) = (pa.data[i] as f64, pb.data[i] as f64);
                sab += x * y;
                saa += x * x;
                sbb += y * y;
            }
        }
        if saa <= 0.0 || sbb <= 0.0 {
            f64::NAN
        } else {
            sab / (saa * sbb).sqrt()
        }
    };
    // recherche grossière puis fine
    let coarse: Vec<f64> = {
        let k = (max_deg / 0.25).round() as i64;
        (-k..=k).map(|i| i as f64 * 0.25).collect()
    };
    let cs: Vec<f64> = coarse.par_iter().map(|&a| score(a)).collect();
    let ci = (0..cs.len()).filter(|&i| cs[i].is_finite()).max_by(|&a, &b| cs[a].total_cmp(&cs[b]))?;
    let fine: Vec<f64> = (-5..=5).map(|i| coarse[ci] + i as f64 * 0.05).collect();
    let fs: Vec<f64> = fine.par_iter().map(|&a| score(a)).collect();
    let fi = (0..fs.len()).filter(|&i| fs[i].is_finite()).max_by(|&a, &b| fs[a].total_cmp(&fs[b]))?;
    let best = fs[fi];
    // pic significatif : net par rapport à ±0,5°
    let side = [score(fine[fi] - 0.5), score(fine[fi] + 0.5)];
    let side: Vec<f64> = side.iter().cloned().filter(|v| v.is_finite()).collect();
    if best < 0.3 || side.is_empty() || best - side.iter().sum::<f64>() / (side.len() as f64) < 0.002 {
        return None;
    }
    let mut a = fine[fi];
    if fi > 0 && fi + 1 < fs.len() && fs[fi - 1].is_finite() && fs[fi + 1].is_finite() {
        let den = fs[fi - 1] - 2.0 * fs[fi] + fs[fi + 1];
        if den.abs() > 1e-12 {
            a += (0.5 * (fs[fi - 1] - fs[fi + 1]) / den).clamp(-0.5, 0.5) * 0.05;
        }
    }
    Some(a)
}

/// Rotation de chaque image par rapport à la référence. Mesures entre images
/// distantes de 1 à 8 rangs (une rotation lente devient mesurable),
/// confirmées aller-retour ; la rotation de champ variant régulièrement,
/// l'angle est modélisé comme une fonction lisse de l'heure de prise de vue
/// (linéaire, ou quadratique sur une longue série) ajustée par moindres
/// carrés robustes : les images sans détail mesurable (phase profonde d'une
/// éclipse) sont couvertes sans dérive inventée.
fn chain_rotation(imgs: &[Image], ref_pos: usize, disc_r: Option<f64>, times: &[f64]) -> Vec<f64> {
    let n = imgs.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let pairs: Vec<(usize, usize)> = (1..=8usize).flat_map(|d| (d..n).map(move |j| (j - d, j))).collect();
    let meas: Vec<(usize, usize, f64)> = pairs
        .par_iter()
        .filter_map(|&(i, j)| {
            let f = masked_rotation(&imgs[i], &imgs[j], disc_r, 6.0)?;
            let b = masked_rotation(&imgs[j], &imgs[i], disc_r, 6.0)?;
            if (f + b).abs() < 0.2 {
                Some((i, j, 0.5 * (f - b)))
            } else {
                None
            }
        })
        .collect();
    if meas.len() < 2 {
        return vec![0.0; n];
    }
    let tr = times[ref_pos];
    let span = times.iter().cloned().fold(f64::MIN, f64::max) - times.iter().cloned().fold(f64::MAX, f64::min);
    // terme quadratique seulement si les mesures couvrent les deux extrémités
    let (tmin, tmax) = (times.iter().cloned().fold(f64::MAX, f64::min), times.iter().cloned().fold(f64::MIN, f64::max));
    let covers = |lo: f64, hi: f64| meas.iter().any(|&(i, j, _)| (times[i] >= lo && times[i] <= hi) || (times[j] >= lo && times[j] <= hi));
    let quad = meas.len() >= 20 && span > 0.0 && covers(tmin, tmin + 0.2 * span) && covers(tmax - 0.2 * span, tmax);
    let np = if quad { 2 } else { 1 };
    let basis = |t: f64| -> [f64; 2] {
        let u = (t - tr) / span.max(1e-9);
        [u, u * u]
    };
    let mut keep = vec![true; meas.len()];
    let mut coef = [0.0f64; 2];
    for _ in 0..3 {
        let idx: Vec<usize> = (0..meas.len()).filter(|&k| keep[k]).collect();
        if idx.len() < np {
            break;
        }
        let mut a = nalgebra::DMatrix::<f64>::zeros(idx.len(), np);
        let mut bv = nalgebra::DVector::<f64>::zeros(idx.len());
        for (r, &k) in idx.iter().enumerate() {
            let (i, j, m) = meas[k];
            let (bi, bj) = (basis(times[i]), basis(times[j]));
            for p in 0..np {
                a[(r, p)] = bj[p] - bi[p];
            }
            bv[r] = m;
        }
        let Ok(sol) = a.svd(true, true).solve(&bv, 1e-12) else { break };
        for p in 0..np {
            coef[p] = sol[p];
        }
        // rejet des mesures aberrantes
        let res: Vec<f64> = meas
            .iter()
            .map(|&(i, j, m)| {
                let (bi, bj) = (basis(times[i]), basis(times[j]));
                ((0..np).map(|p| coef[p] * (bj[p] - bi[p])).sum::<f64>() - m).abs()
            })
            .collect();
        let rk: Vec<f64> = (0..meas.len()).filter(|&k| keep[k]).map(|k| res[k]).collect();
        let lim = (3.0 * 1.4826 * stats::median_f64(&rk)).max(0.15);
        let nk: Vec<bool> = res.iter().map(|&r| r <= lim).collect();
        if nk == keep {
            break;
        }
        keep = nk;
    }
    (0..n)
        .map(|k| {
            let b = basis(times[k]);
            (0..np).map(|p| coef[p] * b[p]).sum::<f64>()
        })
        .collect()
}

/// (niveau de l'objet, fond) d'une image cadrée.
fn object_levels(im: &Image, disc_r: Option<f64>) -> (f64, f64) {
    let l = lum(im);
    let (w, h) = (l.w, l.h);
    let (cx, cy) = (w as f64 / 2.0, h as f64 / 2.0);
    match disc_r {
        Some(r) => {
            let mut inside = Vec::new();
            let mut outside = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    let d = (x as f64 + 0.5 - cx).hypot(y as f64 + 0.5 - cy);
                    let v = l.data[y * w + x];
                    if d < 0.95 * r {
                        inside.push(v);
                    } else if d > r + 4.0 {
                        outside.push(v);
                    }
                }
            }
            let bg = if outside.len() > 50 { stats::median(&outside) as f64 } else { stats::percentile(&l.data, 1.0) as f64 };
            // partie éclairée : haut de la distribution (indépendant du croissant)
            let lv = if inside.is_empty() { stats::percentile(&l.data, 97.0) as f64 } else { stats::percentile(&inside, 97.0) as f64 };
            ((lv - bg).max(1e-6), bg)
        }
        None => {
            let bg = stats::percentile(&l.data, 1.0) as f64;
            ((stats::percentile(&l.data, 97.0) as f64 - bg).max(1e-6), bg)
        }
    }
}

/// Options de rendu (rejouables instantanément).
#[derive(Clone, Debug)]
pub struct RenderOptions {
    pub brightness: Brightness,
    /// fenêtre de lissage du dé-scintillement (images)
    pub smoothing: usize,
    /// horodatage incrusté
    pub timestamp: bool,
    /// légende facultative (en haut à gauche)
    pub caption: String,
}

impl Default for RenderOptions {
    fn default() -> Self {
        RenderOptions { brightness: Brightness::Deflicker, smoothing: 5, timestamp: true, caption: String::new() }
    }
}

/// Gains de luminosité appliqués à chaque image.
pub fn brightness_gains(levels: &[f64], mode: Brightness, smoothing: usize, ref_pos: usize) -> Vec<f64> {
    let n = levels.len();
    match mode {
        Brightness::Real => vec![1.0; n],
        Brightness::Normalized => {
            let r = levels.get(ref_pos).cloned().unwrap_or(1.0);
            levels.iter().map(|l| r / l.max(1e-9)).collect()
        }
        Brightness::Deflicker => {
            let lg: Vec<f64> = levels.iter().map(|l| l.max(1e-9).ln()).collect();
            let half = smoothing.max(1) / 2;
            // médiane glissante puis moyenne à 3 points
            let med: Vec<f64> = (0..n)
                .map(|i| {
                    let (a, b) = (i.saturating_sub(half), (i + half + 1).min(n));
                    stats::median_f64(&lg[a..b])
                })
                .collect();
            let sm: Vec<f64> = (0..n)
                .map(|i| {
                    let (a, b) = (i.saturating_sub(1), (i + 2).min(n));
                    med[a..b].iter().sum::<f64>() / (b - a) as f64
                })
                .collect();
            (0..n).map(|i| (sm[i] - lg[i]).exp()).collect()
        }
    }
}

/// Rendu final de toutes les images : luminosité harmonisée, niveaux
/// communs, horodatage.
pub fn render(p: &SeqProcessed, o: &RenderOptions) -> Vec<Image> {
    let levels: Vec<f64> = p.items.iter().map(|i| i.level).collect();
    let gains = brightness_gains(&levels, o.brightness, o.smoothing, p.reference);
    // blanc commun après gain
    let whites: Vec<f64> = p.items.iter().zip(&gains).map(|(it, g)| it.level * g).collect();
    let white = match o.brightness {
        Brightness::Real => whites.iter().cloned().fold(0.0, f64::max),
        _ => stats::median_f64(&whites),
    }
    .max(1e-6)
        * 1.04;
    p.frames
        .par_iter()
        .zip(&p.items)
        .zip(&gains)
        .map(|((im, it), &g)| {
            let (bg, gf) = (it.background as f32, g as f32);
            let lin = im.map(|v| (v - bg) * gf);
            let mut out = pa::tone(&lin, 0.0, white);
            if o.timestamp || !o.caption.is_empty() {
                out = overlay(&out, if o.timestamp { it.time.as_deref() } else { None }, &o.caption);
            }
            out
        })
        .collect()
}

/// Texte incrusté : heure (en bas à gauche) et légende (en haut à gauche).
pub fn overlay(img: &Image, time: Option<&str>, caption: &str) -> Image {
    let mut cv = Canvas::from_rgb_f32(img);
    let scale = (img.w.min(img.h) as f64 / 260.0).clamp(1.0, 4.0);
    let pad = 6.0 * scale;
    if let Some(t) = time {
        // « 2026-08-12T17:42:10.250 » -> « 2026-08-12 17:42:10 UT »
        let s = format!("{} UT", t.replace('T', " ").split('.').next().unwrap_or(t));
        cv.text(&s, pad, img.h as f64 - pad, scale, [225, 225, 225]);
    }
    if !caption.is_empty() {
        let ascii: String = caption.chars().map(|c| if c.is_ascii() { c } else { deaccent(c) }).collect();
        // la légende tient dans la largeur (avance de 6 px par caractère à l'échelle 1)
        let sc = scale.min((img.w as f64 - 2.0 * pad) / (6.0 * ascii.len().max(1) as f64)).max(1.0);
        let fit = (((img.w as f64 - 2.0 * pad) / (6.0 * sc)) as usize).max(4);
        let txt = if ascii.chars().count() > fit { format!("{}...", ascii.chars().take(fit.saturating_sub(3)).collect::<String>()) } else { ascii };
        cv.text(&txt, pad, pad + 8.0 * sc, sc, [225, 225, 225]);
    }
    let out = cv.to_image();
    if img.c == 1 {
        out.mean_lum()
    } else {
        out
    }
}

fn deaccent(c: char) -> char {
    match c {
        'à' | 'â' | 'ä' => 'a',
        'é' | 'è' | 'ê' | 'ë' => 'e',
        'î' | 'ï' => 'i',
        'ô' | 'ö' => 'o',
        'ù' | 'û' | 'ü' => 'u',
        'ç' => 'c',
        'É' | 'È' => 'E',
        'À' => 'A',
        _ => '?',
    }
}

/// Images intermédiaires en fondu (`between` entre deux images).
pub fn interpolate(frames: &[Image], between: usize) -> Vec<Image> {
    if between == 0 || frames.len() < 2 {
        return frames.to_vec();
    }
    let mut out = Vec::with_capacity(frames.len() + (frames.len() - 1) * between);
    for j in 0..frames.len() - 1 {
        out.push(frames[j].clone());
        for k in 1..=between {
            let t = k as f32 / (between + 1) as f32;
            out.push(frames[j].zip_map(&frames[j + 1], |a, b| a * (1.0 - t) + b * t));
        }
    }
    out.push(frames.last().unwrap().clone());
    out
}

/// Images de l'animation : rendu sans texte, transitions en fondu, puis
/// texte incrusté sur chaque image avec une heure interpolée (l'horloge
/// défile au lieu de se superposer pendant les fondus).
pub fn render_animation(p: &SeqProcessed, o: &RenderOptions, between: usize) -> Vec<Image> {
    let plain = render(p, &RenderOptions { timestamp: false, caption: String::new(), ..o.clone() });
    let mut frames = interpolate(&plain, between);
    if !o.timestamp && o.caption.is_empty() {
        return frames;
    }
    let times: Vec<Option<NaiveDateTime>> = p.items.iter().map(|i| i.time.as_deref().and_then(|t| crate::timeutil::parse_iso_utc(t).ok())).collect();
    let step = between + 1;
    frames.par_iter_mut().enumerate().for_each(|(k, f)| {
        let (j, r) = (k / step, k % step);
        let t = match (times.get(j).cloned().flatten(), times.get(j + 1).cloned().flatten()) {
            (Some(a), Some(b)) if r > 0 => Some(a + (b - a) * r as i32 / step as i32),
            (Some(a), _) => Some(a),
            _ => None,
        };
        let ts = if o.timestamp { t.map(|t| t.format("%Y-%m-%dT%H:%M:%S").to_string()) } else { None };
        *f = overlay(f, ts.as_deref(), &o.caption);
    });
    frames
}

/// Export de l'animation : .mp4 / .avi (ffmpeg) ou .gif.
pub fn export_animation(p: &SeqProcessed, o: &RenderOptions, path: &Path, fps: u32, between: usize) -> Result<()> {
    let all = render_animation(p, o, between);
    crate::animation::export_video(&all, path, fps.max(1), 90)
}

/// Série d'images numérotées (format selon l'extension du modèle, ex.
/// « eclipse.png » -> eclipse_0001.png …). Retourne les fichiers écrits.
pub fn export_frames(frames: &[Image], dir: &Path, stem: &str, ext: &str) -> Result<Vec<PathBuf>> {
    std::fs::create_dir_all(dir)?;
    let ext = ext.trim_start_matches('.').to_lowercase();
    let depth = if ext == "png" { "16" } else { "auto" };
    frames
        .par_iter()
        .enumerate()
        .map(|(i, im)| {
            let p = dir.join(format!("{}_{:04}.{}", stem, i + 1, ext));
            io::save_image(&p, im, depth, None)?;
            Ok(p)
        })
        .collect()
}

/// Rapport JSON de la série.
pub fn report_json(p: &SeqProcessed) -> serde_json::Value {
    serde_json::json!({
        "version": crate::VERSION,
        "framing": p.framing,
        "disc_radius": p.disc_radius,
        "reference": p.items.get(p.reference).map(|i| i.name.clone()),
        "lock": p.lock,
        "decisions": p.decisions,
        "reference_decisions": p.reference_decisions,
        "items": p.items,
        "failed": p.failed,
        "seconds": p.seconds,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deflicker_keeps_trend_removes_flicker() {
        // tendance lente (÷2) + scintillement ±20 %
        let lv: Vec<f64> = (0..30).map(|i| (1.0 - i as f64 / 60.0) * if i % 3 == 1 { 0.8 } else { 1.0 }).collect();
        let g = brightness_gains(&lv, Brightness::Deflicker, 5, 0);
        let out: Vec<f64> = lv.iter().zip(&g).map(|(a, b)| a * b).collect();
        for w in out.windows(2) {
            assert!((w[1] / w[0] - 1.0).abs() < 0.06, "{:?}", out);
        }
        assert!(out[29] < 0.8 * out[0]);
        let gn = brightness_gains(&lv, Brightness::Normalized, 5, 0);
        assert!(lv.iter().zip(&gn).all(|(a, b)| (a * b - lv[0]).abs() < 1e-9));
    }

    #[test]
    fn rotation_ignores_moving_moon_and_finds_real_rotation() {
        let sun = |mx: f64| {
            Image::from_fn(160, 160, |x, y| {
                let (u, v) = ((x as f64 + 0.5 - 80.0) / 60.0, (y as f64 + 0.5 - 80.0) / 60.0);
                let rr = u * u + v * v;
                if rr >= 1.0 || (x as f64 - mx).hypot(y as f64 - 40.0) < 58.0 {
                    return 0.02;
                }
                let mut b = 0.3 + 0.7 * (1.0 - rr).sqrt();
                for &(su, sv) in &[(0.5, 0.45), (-0.6, 0.35), (0.1, 0.75), (-0.45, -0.55), (0.7, -0.2)] {
                    b *= 1.0 - 0.7 * (-((u - su).powi(2) + (v - sv).powi(2)) / 0.004).exp();
                }
                b as f32
            })
        };
        // la Lune avance, aucune rotation : rien à corriger
        let a = masked_rotation(&sun(10.0), &sun(25.0), Some(60.0), 2.0).unwrap_or(0.0);
        assert!(a.abs() < 0.15, "fausse rotation {}", a);
        // vraie rotation de 1,2° : retrouvée (le signe rétablit l'image)
        let base = sun(-40.0);
        let rot = registration::apply_shift_rot(&base, 0.0, 0.0, 1.2);
        let est = masked_rotation(&base, &rot, Some(60.0), 2.0).unwrap();
        assert!((est + 1.2).abs() < 0.15, "{}", est);
    }

    #[test]
    fn interpolation_counts() {
        let f = vec![Image::filled(4, 4, 1, 0.0), Image::filled(4, 4, 1, 1.0)];
        let o = interpolate(&f, 3);
        assert_eq!(o.len(), 5);
        assert!((o[2].data[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn center_on_places_point_at_center() {
        let img = Image::from_fn(60, 50, |x, y| if (x as f64 - 20.3).hypot(y as f64 - 30.6) < 8.0 { 1.0 } else { 0.0 });
        let c = center_on(&img, 20.3, 30.6, 40, 0.0);
        let (mut sx, mut sy, mut s) = (0.0, 0.0, 0.0);
        for y in 0..40 {
            for x in 0..40 {
                let v = c.data[y * 40 + x] as f64;
                sx += v * x as f64;
                sy += v * y as f64;
                s += v;
            }
        }
        assert!((sx / s - 19.5).abs() < 0.2 && (sy / s - 19.5).abs() < 0.2, "{} {}", sx / s, sy / s);
    }
}
