//! Ligne de commande AstroForge : traitement sans interface (serveur,
//! scripts, automatisation).
//!
//!     astroforge sort NUIT/
//!     astroforge analyze lights/*.fits --csv analyse.csv
//!     astroforge stack --session NUIT/ -o m31.fits --method Sigma-clipping --weights Mixte
//!     astroforge stack --lights l/*.fits --darks darks/ --flats flats/ --bayer RGGB --drizzle 2 -o out.fits
//!     astroforge multinight NUIT1/ NUIT2/ -o m31.fits --normalization Locale
//!     astroforge solve image.fits --catalog gaia.csv --ra 10h42m --dec "+41°16'" --write-wcs
//!     astroforge annotate image.fits -o image_annotee.png
//!     astroforge compose --ha ha.fits --oiii oiii.fits --palette HOO -o hoo.tif
//!     astroforge batch sequence.json entree/ sortie/
//!     astroforge live dossier/ --duration 3600 -o live.fits
//!     astroforge planetary video.ser -o jupiter.fits
//!     astroforge planetary-auto video.ser -o jupiter.tif --report rapport.json
//!     astroforge timelapse eclipse/*.ser -o eclipse.mp4 --fps 12 --interp 2
//!     astroforge pipeline processus.json --session NUIT/ -o resultat.fits

use anyhow::{anyhow, bail, Result};
use astroforge::calibration::{self as C, DarkScale, MasterOptions};
use astroforge::image::Image;
use astroforge::io;
use astroforge::{annotate, astrometry, display, drizzle, equipment, livestack, mosaic, multisession, narrowband, normalization, planetary, registration, sequence, session, stacking, stats, subframes, updates, wavelets};
use clap::{Args, Parser, Subcommand};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Parser)]
#[command(name = "astroforge", version = concat!("AstroForge ", env!("CARGO_PKG_VERSION")), about = "AstroForge en ligne de commande")]
struct Cli {
    /// progression détaillée
    #[arg(short, long, global = true)]
    verbose: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

const WEIGHTS: [&str; 4] = ["Aucune", "SNR", "FWHM", "Mixte"];
const NORMS: [&str; 3] = ["Aucune", "Globale", "Locale"];

#[derive(Subcommand)]
enum Cmd {
    /// tri automatique d'une session
    Sort {
        folder: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// analyse des brutes (FWHM, étoiles, poids)
    Analyze {
        #[arg(required = true)]
        files: Vec<String>,
        #[arg(long, default_value = "auto")]
        bayer: String,
        #[arg(long, default_value = "Mixte", value_parser = WEIGHTS)]
        weights: String,
        #[arg(long)]
        csv: Option<PathBuf>,
    },
    /// calibration + alignement + empilement
    Stack(StackArgs),
    /// empilement de plusieurs nuits
    Multinight {
        #[arg(required = true)]
        folders: Vec<PathBuf>,
        #[arg(long, default_value = "Sigma-clipping")]
        method: String,
        #[arg(long, default_value_t = 3.0)]
        sigma: f64,
        #[arg(long, default_value = "Locale", value_parser = NORMS)]
        normalization: String,
        #[arg(long, default_value = "Aucune", value_parser = WEIGHTS)]
        weights: String,
        #[arg(long, default_value = "auto")]
        bayer: String,
        #[arg(short = 'o', long = "out", required = true)]
        out: PathBuf,
    },
    /// astrométrie (plate solving)
    Solve {
        image: PathBuf,
        #[arg(long)]
        catalog: Option<PathBuf>,
        #[arg(long, allow_hyphen_values = true)]
        ra: Option<String>,
        #[arg(long, allow_hyphen_values = true)]
        dec: Option<String>,
        #[arg(long, default_value_t = 1.0)]
        radius: f64,
        #[arg(long)]
        focal: Option<f64>,
        #[arg(long)]
        pixel: Option<f64>,
        /// ne pas utiliser solve-field
        #[arg(long)]
        internal: bool,
        #[arg(long)]
        write_wcs: bool,
    },
    /// annotation Messier/NGC/IC
    Annotate {
        image: PathBuf,
        #[arg(long, default_value_t = 14.0)]
        mag: f64,
        #[arg(short = 'o', long = "out")]
        out: Option<PathBuf>,
    },
    /// composition bande étroite / LRGB
    Compose {
        #[arg(long)]
        ha: Option<PathBuf>,
        #[arg(long)]
        oiii: Option<PathBuf>,
        #[arg(long)]
        sii: Option<PathBuf>,
        #[arg(long)]
        r: Option<PathBuf>,
        #[arg(long)]
        g: Option<PathBuf>,
        #[arg(long)]
        b: Option<PathBuf>,
        #[arg(long)]
        lum: Option<PathBuf>,
        #[arg(long, default_value = "HOO")]
        palette: String,
        #[arg(long)]
        align: bool,
        #[arg(long)]
        no_normalize: bool,
        #[arg(long, default_value_t = 1.0)]
        saturation: f64,
        #[arg(short = 'o', long = "out", required = true)]
        out: PathBuf,
    },
    /// séquence enregistrée appliquée à un dossier
    Batch { sequence: PathBuf, input: PathBuf, output: PathBuf },
    /// empilement en direct d'un dossier
    Live {
        folder: PathBuf,
        /// secondes (0 = fichiers présents)
        #[arg(long, default_value_t = 0.0)]
        duration: f64,
        #[arg(long, default_value_t = 2.0)]
        interval: f64,
        #[arg(long)]
        darks: Option<String>,
        #[arg(long)]
        flats: Option<String>,
        #[arg(long, default_value = "auto")]
        bayer: String,
        #[arg(long, default_value_t = 3.0)]
        kappa: f64,
        #[arg(short = 'o', long = "out")]
        out: Option<PathBuf>,
    },
    /// empilement planétaire d'une vidéo SER (6 étapes)
    Planetary(PlanetArgs),
    /// planétaire entièrement automatique (moteur de décision IA local)
    PlanetaryAuto(AutoArgs),
    /// time-lapse multi-vidéos : traitement, cadrage et rendu uniformes, animation
    Timelapse(TimelapseArgs),
    /// processus de traitement automatique enregistré (JSON)
    Pipeline {
        process: PathBuf,
        #[arg(long)]
        session: Option<PathBuf>,
        #[arg(long, default_value_t = 0)]
        group: usize,
        #[arg(long, num_args = 0..)]
        lights: Vec<String>,
        /// vidéo SER (processus planétaire)
        #[arg(long)]
        ser: Option<PathBuf>,
        #[arg(long, default_value = "auto")]
        bayer: String,
        #[arg(short = 'o', long = "out", required = true)]
        out: PathBuf,
    },
    /// mosaïque de panneaux (WCS dans les en-têtes FITS, sinon étoiles)
    Mosaic {
        #[arg(required = true)]
        panels: Vec<String>,
        #[arg(long, default_value_t = 64.0)]
        feather: f64,
        #[arg(short = 'o', long = "out", required = true)]
        out: PathBuf,
    },
    /// version, calcul, langue, dossier des données
    Info {
        #[arg(long)]
        check_updates: bool,
    },
}

#[derive(Args)]
struct StackArgs {
    /// dossier de session (tri automatique)
    #[arg(long)]
    session: Option<PathBuf>,
    /// groupe de lumières (0 = le plus fourni)
    #[arg(long, default_value_t = 0)]
    group: usize,
    #[arg(long, num_args = 0..)]
    lights: Vec<String>,
    #[arg(long)]
    bias: Option<String>,
    #[arg(long)]
    darks: Option<String>,
    #[arg(long)]
    flats: Option<String>,
    #[arg(long)]
    darkflats: Option<String>,
    #[arg(long, default_value = "auto")]
    bayer: String,
    #[arg(long, default_value = "RGB", value_parser = ["RGB", "Ha", "OIII"])]
    output_kind: String,
    #[arg(long, default_value = "Sigma-clipping")]
    method: String,
    #[arg(long, default_value_t = 3.0)]
    sigma: f64,
    #[arg(long, default_value = "Aucune", value_parser = WEIGHTS)]
    weights: String,
    #[arg(long, default_value = "Aucune", value_parser = NORMS)]
    normalization: String,
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=3))]
    drizzle: u32,
    #[arg(long, default_value_t = 0.7)]
    pixfrac: f64,
    #[arg(short = 'o', long = "out", required = true)]
    out: PathBuf,
}

#[derive(Args)]
struct PlanetArgs {
    video: PathBuf,
    #[arg(long, default_value = "Contraste HF (AutoStakkert)")]
    quality: String,
    /// pourcentage d'images retenues
    #[arg(long, default_value_t = 75.0)]
    keep: f64,
    #[arg(long, default_value = "Multi-points (grille NxN)")]
    align_mode: String,
    #[arg(long, default_value_t = 4)]
    grid: usize,
    #[arg(long = "box", default_value_t = 0)]
    box_size: usize,
    #[arg(long, default_value = "Moyenne")]
    stack_method: String,
    #[arg(long, default_value_t = 100.0)]
    stack_percent: f64,
    /// 0 = toutes les frames
    #[arg(long, default_value_t = 0)]
    max_frames: usize,
    /// gains des 7 couches d'ondelettes, ex. « 1.6,1.3,1,1,1,1,1 »
    #[arg(long)]
    gains: Option<String>,
    #[arg(long)]
    rgb_align: bool,
    /// fausses couleurs (image mono), ex. « Solaire (orange) »
    #[arg(long)]
    palette: Option<String>,
    #[arg(short = 'o', long = "out", required = true)]
    out: PathBuf,
}

#[derive(Args)]
struct AutoArgs {
    /// vidéo SER, ou images (FITS, TIFF, PNG… ; motifs acceptés)
    #[arg(required = true)]
    inputs: Vec<String>,
    /// effort de recherche : Rapide, Équilibré, Maximal
    #[arg(long, default_value = "Équilibré")]
    effort: String,
    /// 0 = toutes les frames
    #[arg(long, default_value_t = 0)]
    max_frames: usize,
    /// ne pas consulter la mémoire des traitements précédents
    #[arg(long)]
    no_memory: bool,
    /// ne pas mémoriser ce traitement
    #[arg(long)]
    no_learn: bool,
    /// ne pas essayer de modèle ONNX de débruitage
    #[arg(long)]
    no_onnx: bool,
    /// pas d'équilibre automatique des couleurs
    #[arg(long)]
    no_color_balance: bool,
    /// rapport JSON (plan, décisions, indicateurs)
    #[arg(long)]
    report: Option<PathBuf>,
    /// enregistre aussi l'empilement brut (avant restauration)
    #[arg(long)]
    stack_out: Option<PathBuf>,
    #[arg(short = 'o', long = "out", required = true)]
    out: PathBuf,
}

#[derive(Args)]
struct TimelapseArgs {
    /// vidéos SER (motifs acceptés) ou dossier(s) de vidéos
    #[arg(required = true)]
    inputs: Vec<String>,
    /// animation : .mp4, .avi (ffmpeg) ou .gif
    #[arg(short = 'o', long = "out")]
    out: Option<PathBuf>,
    /// dossier où écrire les images de la série (numérotées)
    #[arg(long)]
    frames_dir: Option<PathBuf>,
    /// format des images de la série : png (16 bits), tif, fits, jpg
    #[arg(long, default_value = "png")]
    frames_format: String,
    #[arg(long, default_value_t = 10)]
    fps: u32,
    /// images intermédiaires en fondu entre deux images (fluidité)
    #[arg(long, default_value_t = 2)]
    interp: usize,
    /// effort de recherche sur la vidéo de référence
    #[arg(long, default_value = "Équilibré")]
    effort: String,
    /// numéro (1 = première dans l'ordre chronologique) de la vidéo de référence
    #[arg(long)]
    reference: Option<usize>,
    /// cadrage : auto, disque, surface
    #[arg(long, default_value = "auto")]
    framing: String,
    /// marge autour du disque (fraction du rayon)
    #[arg(long, default_value_t = 0.25)]
    margin: f64,
    /// correction de la rotation de champ (monture altazimutale)
    #[arg(long)]
    field_rotation: bool,
    /// luminosité : descintillement, reelle, normalisee
    #[arg(long, default_value = "descintillement")]
    brightness: String,
    /// pas d'horodatage incrusté
    #[arg(long)]
    no_timestamp: bool,
    /// légende incrustée (en haut à gauche)
    #[arg(long, default_value = "")]
    caption: String,
    #[arg(long, default_value_t = 0)]
    max_frames: usize,
    /// rapport JSON de la série
    #[arg(long)]
    report: Option<PathBuf>,
}

/// Progression façon « [n/total] étiquette » (une ligne par étiquette).
struct Progress {
    on: bool,
    last: Mutex<Option<String>>,
}

impl Progress {
    fn new(on: bool) -> Self {
        Progress { on, last: Mutex::new(None) }
    }
    fn call(&self, done: usize, total: usize, label: &str) {
        if !self.on {
            return;
        }
        let mut l = self.last.lock().unwrap();
        if l.as_deref() != Some(label) || done == total {
            eprintln!("  [{}/{}] {}", done, total, label);
            *l = Some(label.to_string());
        }
    }
}

fn expand(items: &[String]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for it in items {
        let mut g: Vec<PathBuf> = glob::glob(it).map(|p| p.filter_map(|x| x.ok()).collect()).unwrap_or_default();
        g.sort();
        if g.is_empty() {
            out.push(PathBuf::from(it));
        } else {
            out.extend(g);
        }
    }
    out
}

fn bayer_opt(v: &str) -> Option<String> {
    let l = v.to_lowercase();
    if v.is_empty() || ["auto", "none", "aucune", "mono"].contains(&l.as_str()) {
        None
    } else {
        Some(v.to_uppercase())
    }
}

fn is_fits(p: &Path) -> bool {
    matches!(io::ext_of(p).as_str(), "fits" | "fit")
}

fn cmd_sort(folder: &Path, json_out: bool) -> Result<i32> {
    let plan = session::plan_session(folder, true, 3.0);
    if json_out {
        let data: Vec<serde_json::Value> = plan
            .groups
            .iter()
            .map(|g| {
                let calib: serde_json::Map<String, serde_json::Value> = ["bias", "dark", "darkflat", "flat"]
                    .iter()
                    .map(|k| (k.to_string(), serde_json::json!(g.paths(k).iter().map(|p| p.to_string_lossy()).collect::<Vec<_>>())))
                    .collect();
                serde_json::json!({
                    "name": g.name(),
                    "lights": g.paths("light").iter().map(|p| p.to_string_lossy()).collect::<Vec<_>>(),
                    "calib": calib,
                    "dark_scale": g.dark_scale,
                    "warnings": g.warnings,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&data)?);
    } else {
        println!("{}", plan.summary());
    }
    Ok(0)
}

fn cmd_analyze(files: &[String], bayer: &str, weights: &str, csv: Option<&Path>, prog: &Progress) -> Result<i32> {
    let paths = expand(files);
    let pf = |d: usize, t: usize, l: &str| prog.call(d, t, l);
    let mut rows = subframes::analyze_paths(&paths, bayer_opt(bayer).as_deref(), Some(&pf));
    subframes::flag_outliers_default(&mut rows);
    let w = subframes::compute_weights(&rows, weights);
    for (r, x) in rows.iter_mut().zip(&w) {
        r.weight = *x;
    }
    println!("{}", subframes::report(&rows));
    if let Some(c) = csv {
        let mut s = String::from("path;stars;fwhm;ecc;background;noise;snr;weight;keep\n");
        for r in &rows {
            s += &format!(
                "{};{};{};{};{};{};{};{};{}\n",
                r.path,
                r.stars,
                r.fwhm,
                r.ecc,
                r.background,
                r.noise,
                r.snr,
                r.weight,
                if r.keep { "True" } else { "False" }
            );
        }
        std::fs::write(c, s)?;
    }
    Ok(0)
}

fn cmd_stack(a: &StackArgs, prog: &Progress) -> Result<i32> {
    let pf = |d: usize, t: usize, l: &str| prog.call(d, t, l);
    let mut calib = None;
    let lights: Vec<PathBuf> = if let Some(sdir) = &a.session {
        let plan = session::plan_session(sdir, true, 3.0);
        if plan.groups.is_empty() {
            eprintln!("Aucune image lumière.");
            return Ok(2);
        }
        let g = plan.groups.get(a.group).ok_or_else(|| anyhow!("Groupe {} inexistant ({} groupe(s))", a.group, plan.groups.len()))?;
        calib = Some(g.calib_paths());
        eprintln!("{}", g.summary());
        g.paths("light")
    } else {
        expand(&a.lights)
    };
    if lights.is_empty() {
        eprintln!("Aucune image à empiler.");
        return Ok(2);
    }
    let masters = C::build_masters(&MasterOptions {
        bias_dir: a.bias.clone().unwrap_or_default(),
        dark_dir: a.darks.clone().unwrap_or_default(),
        flat_dir: a.flats.clone().unwrap_or_default(),
        darkflat_dir: a.darkflats.clone().unwrap_or_default(),
        sigma: 3.0,
        paths: calib,
        ..Default::default()
    })?;
    let imgs = C::calibrate_paths(&lights, &masters, bayer_opt(&a.bayer).as_deref(), DarkScale::Auto, true, None, Some(&pf), &a.output_kind)?;
    let mut weights: Option<Vec<f64>> = None;
    if a.weights != "Aucune" {
        use rayon::prelude::*;
        let mut rows: Vec<subframes::FrameStats> = imgs
            .par_iter()
            .zip(lights.par_iter())
            .map(|(im, p)| {
                let mut r = subframes::analyze_image(im, false, 60);
                r.path = p.to_string_lossy().to_string();
                r
            })
            .collect();
        subframes::flag_outliers_default(&mut rows);
        let w = subframes::compute_weights(&rows, &a.weights);
        eprintln!("Rejetées : {}", w.iter().filter(|&&x| x == 0.0).count());
        weights = Some(w);
    }
    let ri = imgs.len() / 2;
    let sp = stacking::StackParams { sigma: a.sigma, ..Default::default() };
    let out = {
        use rayon::prelude::*;
        if a.drizzle > 1 {
            let r = &imgs[ri];
            let mats: Vec<_> = imgs.par_iter().enumerate().map(|(i, im)| if i == ri { astroforge::geom::IDENTITY } else { registration::estimate_transform(r, im, true) }).collect();
            pf(0, 1, "Alignement");
            let aligned: Vec<Image> = imgs.par_iter().zip(mats.par_iter()).map(|(im, m)| registration::warp(im, m)).collect();
            let reference = stacking::stack(&aligned.iter().collect::<Vec<_>>(), &a.method, &sp, weights.as_deref())?;
            drop(aligned);
            let (o, _) = drizzle::drizzle_stack(&imgs, &mats, a.drizzle as usize, a.pixfrac, weights.as_deref(), Some(&reference), Some(a.sigma), Some(&pf))?;
            o
        } else {
            let r = imgs[ri].clone();
            let n = imgs.len();
            let done = std::sync::atomic::AtomicUsize::new(0);
            let aligned: Vec<Image> = imgs
                .into_par_iter()
                .enumerate()
                .map(|(i, im)| {
                    let o = if i == ri { im } else { registration::warp(&im, &registration::estimate_transform(&r, &im, true)) };
                    let d = done.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
                    pf(d, n, "alignement");
                    o
                })
                .collect();
            let aligned = if a.normalization != "Aucune" { normalization::normalize_frames(aligned, &r, &a.normalization, 128)? } else { aligned };
            pf(0, 1, "empilement");
            stacking::stack(&aligned.iter().collect::<Vec<_>>(), &a.method, &sp, weights.as_deref())?
        }
    };
    let header = if is_fits(&a.out) {
        Some(session::stack_header(&lights, weights.as_deref(), None))
    } else {
        None
    };
    io::save_image(&a.out, &out, "auto", header.as_ref())?;
    println!("Écrit : {} ({}×{})", a.out.display(), out.w, out.h);
    Ok(0)
}

#[allow(clippy::too_many_arguments)]
fn cmd_multinight(folders: &[PathBuf], method: &str, sigma: f64, norm: &str, weights: &str, bayer: &str, out: &Path, prog: &Progress) -> Result<i32> {
    let pf = |d: usize, t: usize, l: &str| prog.call(d, t, l);
    let sess: Vec<multisession::Session> = folders.iter().map(|f| multisession::session_from_folder(f)).collect::<Result<_>>()?;
    let (img, info) = multisession::stack_sessions(&sess, method, sigma, norm, bayer_opt(bayer).as_deref(), "RGB", weights, Some(&pf), None)?;
    io::save(out, &img)?;
    println!("Écrit : {} — {} images ({:?})", out.display(), info.n_frames, info.per_night);
    Ok(0)
}

#[allow(clippy::too_many_arguments)]
fn cmd_solve(image: &Path, catalog: Option<&Path>, ra: Option<&str>, dec: Option<&str>, radius: f64, focal: Option<f64>, pixel: Option<f64>, internal: bool, write_wcs: bool) -> Result<i32> {
    let img = io::load(image)?;
    let ra = match ra {
        Some(r) => equipment::parse_coord(r, true)?,
        None => None,
    };
    let dec = match dec {
        Some(d) => equipment::parse_coord(d, false)?,
        None => None,
    };
    let res = astrometry::solve_image(&img, catalog, ra, dec, radius, focal, pixel, !internal)?;
    println!(
        "Méthode {} — centre RA {:.5}° Dec {:.5}° — {:.3}\"/px — {} étoiles — rms {:.2}\"",
        res.method, res.center_ra, res.center_dec, res.scale_arcsec_pix, res.n_matched, res.rms_arcsec
    );
    if write_wcs {
        if !io::has_ext(image, &["fits", "fit", "fts"]) {
            eprintln!("--write-wcs : FITS uniquement");
            return Ok(2);
        }
        io::fits::update_header(image, &res.wcs.to_header())?;
        println!("WCS écrit dans l'en-tête.");
    }
    Ok(0)
}

fn cmd_annotate(image: &Path, mag: f64, out: Option<&Path>) -> Result<i32> {
    let Some(w) = mosaic::wcs_from_fits(image) else {
        eprintln!("Pas de WCS dans l'en-tête (lancez « solve --write-wcs »).");
        return Ok(2);
    };
    let img = io::load(image)?;
    let base = if stats::median(&img.data) > 0.1 { img } else { display::autostretch(&img) };
    let (o, objs) = annotate::annotate(&base, &w, mag, [1.0, 0.85, 0.2], None, None)?;
    for ob in &objs {
        let m = ob.mag.map(io::header::py_float_repr).unwrap_or_else(|| "—".into());
        println!("{:<10} {:<24} mag {}  {}", ob.label, ob.type_fr, m, ob.common);
    }
    if let Some(p) = out {
        io::save(p, &o)?;
    }
    Ok(0)
}

#[allow(clippy::too_many_arguments)]
fn cmd_compose(chans: Vec<(&str, Option<PathBuf>)>, lum: Option<&Path>, palette: &str, align: bool, no_norm: bool, sat: f64, out: &Path) -> Result<i32> {
    let mut ch: BTreeMap<String, Image> = BTreeMap::new();
    for (k, p) in chans {
        if let Some(p) = p {
            ch.insert(k.to_string(), io::load(&p)?.mean_lum());
        }
    }
    if align && ch.len() > 1 {
        let rk = if ch.contains_key("Ha") { Some("Ha") } else { None };
        ch = narrowband::align_channels(&ch, rk);
    }
    let pal = narrowband::palette(palette)?;
    let mut img = narrowband::compose(&ch, &pal, !no_norm, None)?;
    if let Some(l) = lum {
        let lm = io::load(l)?.mean_lum();
        img = narrowband::lrgb(&lm, &img, sat)?;
    }
    io::save(out, &img)?;
    println!("Écrit : {}", out.display());
    Ok(0)
}

fn cmd_batch(seqp: &Path, input: &Path, output: &Path, prog: &Progress) -> Result<i32> {
    let pf = |d: usize, t: usize, l: &str| prog.call(d, t, l);
    let seq = sequence::load_sequence(seqp)?;
    let res = if seq.get("mode").and_then(|v| v.as_str()) == Some("ser") {
        sequence::run_batch_ser_chain(&seq, input, output, 0, Some(&pf))?
    } else {
        sequence::run_batch_image(&seq, input, output, "auto", Some(&pf))?
    };
    println!("{}", res.summary());
    Ok(if res.errors.is_empty() { 0 } else { 1 })
}

#[allow(clippy::too_many_arguments)]
fn cmd_live(folder: &Path, duration: f64, interval: f64, darks: Option<String>, flats: Option<String>, bayer: &str, kappa: f64, out: Option<PathBuf>) -> Result<i32> {
    let masters = C::build_masters(&MasterOptions { dark_dir: darks.unwrap_or_default(), flat_dir: flats.unwrap_or_default(), sigma: 3.0, ..Default::default() })?;
    let mut st = livestack::LiveStacker::new(masters, bayer_opt(bayer), kappa);
    let mut on = |s: &livestack::LiveStacker| {
        eprintln!("{}", s.status());
        if let (Some(o), Some(img)) = (&out, s.image()) {
            if let Err(e) = io::save(o, &img) {
                eprintln!("Erreur d'écriture : {}", e);
            }
        }
    };
    livestack::run_folder(folder, &mut st, duration, interval, &mut on);
    println!("{}", st.status());
    Ok(0)
}

fn cmd_planetary(a: &PlanetArgs, prog: &Progress) -> Result<i32> {
    let pf = |d: usize, t: usize, l: &str| prog.call(d, t, l);
    let (frames, meta) = io::ser::load_ser(&a.video, a.max_frames)?;
    eprintln!("{} : {} frames {}×{} ({} bits{})", a.video.display(), meta.frames, meta.width, meta.height, meta.bits, meta.bayer.as_ref().map(|b| format!(", Bayer {}", b)).unwrap_or_default());
    let n = astroforge::frames::Frames::len(&frames);
    let opts = planetary::PlanetOptions {
        quality_method: a.quality.clone(),
        n_align: ((n as f64 * a.keep / 100.0).round() as usize).max(1),
        align_mode: a.align_mode.clone(),
        grid_n: a.grid,
        box_size: a.box_size,
        stack_method: a.stack_method.clone(),
        stack_percent: a.stack_percent,
        ..Default::default()
    };
    let r = planetary::run_planetary(&frames, &opts, Some(&pf))?;
    let mut img = r.image;
    if a.rgb_align && img.is_color() {
        img = registration::align_channels(&img, None);
    }
    if let Some(g) = &a.gains {
        let gains: Vec<f64> = g.split(',').map(|x| x.trim().parse::<f64>()).collect::<std::result::Result<_, _>>().map_err(|_| anyhow!("--gains : liste de nombres attendue"))?;
        img = wavelets::registax_enhance(&img, Some(&gains), 0.0, None, None)?;
    }
    if let Some(p) = &a.palette {
        if !img.is_color() {
            img = astroforge::effects::apply_palette(&img, p)?;
        }
    }
    io::save(&a.out, &img)?;
    let s = &r.stats;
    println!(
        "Écrit : {} ({}×{}) — {} frames empilées sur {}, meilleure frame n° {}, {} points d'alignement",
        a.out.display(),
        img.w,
        img.h,
        s.stacked,
        s.n_frames,
        s.best_frame + 1,
        s.n_points
    );
    Ok(0)
}

/// Charge une vidéo SER ou une liste d'images comme séquence de frames.
fn load_frames(inputs: &[String], max_frames: usize) -> Result<Box<dyn astroforge::frames::Frames>> {
    let paths = expand(inputs);
    if paths.len() == 1 && io::ext_of(&paths[0]) == "ser" {
        let (frames, meta) = io::ser::load_ser(&paths[0], max_frames)?;
        eprintln!("{} : {} frames {}×{} ({} bits{})", paths[0].display(), meta.frames, meta.width, meta.height, meta.bits, meta.bayer.as_ref().map(|b| format!(", Bayer {}", b)).unwrap_or_default());
        return Ok(Box::new(frames));
    }
    let mut paths = paths;
    if max_frames > 0 {
        paths.truncate(max_frames);
    }
    if paths.is_empty() {
        bail!("Aucune image");
    }
    let imgs: Vec<Image> = paths.iter().map(io::load).collect::<Result<_>>()?;
    let (w, h, c) = (imgs[0].w, imgs[0].h, imgs[0].c);
    if imgs.iter().any(|i| (i.w, i.h, i.c) != (w, h, c)) {
        bail!("Toutes les images doivent avoir la même taille");
    }
    eprintln!("{} images {}×{}", imgs.len(), w, h);
    Ok(Box::new(imgs))
}

fn cmd_planetary_auto(a: &AutoArgs, verbose: bool) -> Result<i32> {
    use astroforge::planetary_auto as pa;
    let frames = load_frames(&a.inputs, a.max_frames)?;
    let opts = pa::AutoOptions {
        effort: pa::Effort::parse(&a.effort)?,
        use_memory: !a.no_memory,
        learn: !a.no_learn,
        onnx: !a.no_onnx,
        color_balance: !a.no_color_balance,
        ..Default::default()
    };
    let last = Mutex::new(String::new());
    let pf = |f: f64, msg: &str| {
        if verbose {
            let mut l = last.lock().unwrap();
            if *l != msg {
                eprintln!("  [{:3.0} %] {}", f * 100.0, msg);
                *l = msg.to_string();
            }
        }
    };
    let r = pa::run_auto(frames.as_ref(), &opts, Some(&pf))?;
    io::save(&a.out, &r.image)?;
    if let Some(s) = &a.stack_out {
        io::save(s, &r.stack)?;
    }
    if let Some(rep) = &a.report {
        std::fs::write(rep, serde_json::to_string_pretty(&r.report_json())?)?;
    }
    for d in &r.decisions {
        println!("• {} — {}\n    {}", d.step, d.choice, d.reason);
    }
    println!("Écrit : {} ({}×{}) — {}", a.out.display(), r.image.w, r.image.h, r.summary());
    Ok(0)
}

fn cmd_timelapse(a: &TimelapseArgs, verbose: bool) -> Result<i32> {
    use astroforge::timelapse as tl;
    if a.out.is_none() && a.frames_dir.is_none() {
        bail!("Indiquez --out (animation) et/ou --frames-dir (images)");
    }
    let inputs = tl::load_inputs(&expand(&a.inputs), a.max_frames)?;
    eprintln!("{} vidéo(s) :", inputs.len());
    for (i, v) in inputs.iter().enumerate() {
        eprintln!("  {:3}. {} — {} frames{}", i + 1, v.name, v.frames.len(), v.time.map(|t| format!(", {}", t.format("%Y-%m-%d %H:%M:%S"))).unwrap_or_default());
    }
    let opts = tl::SeqOptions {
        effort: astroforge::planetary_auto::Effort::parse(&a.effort)?,
        reference: a.reference.map(|r| r.saturating_sub(1)),
        framing: tl::Framing::parse(&a.framing)?,
        margin: a.margin,
        field_rotation: a.field_rotation,
        ..Default::default()
    };
    let last = Mutex::new(String::new());
    let pf = |f: f64, msg: &str| {
        if verbose {
            let mut l = last.lock().unwrap();
            let head = msg.split(" — ").next().unwrap_or(msg).to_string();
            if *l != head {
                eprintln!("  [{:3.0} %] {}", f * 100.0, head);
                *l = head;
            }
        }
    };
    let p = tl::run_timelapse(&inputs, &opts, Some(&pf))?;
    let ro = tl::RenderOptions { brightness: tl::Brightness::parse(&a.brightness)?, timestamp: !a.no_timestamp, caption: a.caption.clone(), ..Default::default() };
    let frames = tl::render(&p, &ro);
    for d in &p.decisions {
        println!("• {} — {}\n    {}", d.step, d.choice, d.reason);
    }
    if let Some(dir) = &a.frames_dir {
        let w = tl::export_frames(&frames, dir, "image", &a.frames_format)?;
        println!("Images : {} fichiers dans {}", w.len(), dir.display());
    }
    if let Some(o) = &a.out {
        tl::export_animation(&p, &ro, o, a.fps, a.interp)?;
        println!("Animation : {} ({} images, {} intermédiaires, {} i/s)", o.display(), frames.len(), a.interp, a.fps);
    }
    if let Some(r) = &a.report {
        std::fs::write(r, serde_json::to_string_pretty(&tl::report_json(&p))?)?;
    }
    println!("Série : {} image(s), {} écartée(s), cadre {}×{}, {:.1} s", frames.len(), p.failed.len(), frames[0].w, frames[0].h, p.seconds);
    Ok(if p.failed.is_empty() { 0 } else { 1 })
}

#[allow(clippy::too_many_arguments)]
fn cmd_pipeline(process: &Path, sess: Option<&Path>, group: usize, lights: &[String], ser: Option<&Path>, bayer: &str, out: &Path, prog: &Progress) -> Result<i32> {
    let p = astroforge::pipeline::Pipeline::load(process)?;
    let mut ctx = astroforge::pipeline::Ctx { bayer: bayer_opt(bayer), ..Default::default() };
    if let Some(sdir) = sess {
        let plan = session::plan_session(sdir, true, 3.0);
        let g = plan.groups.get(group).ok_or_else(|| anyhow!("Aucun groupe de lumières n° {}", group))?;
        eprintln!("{}", g.summary());
        ctx.light_paths = g.paths("light");
        ctx.calib_paths = Some(g.calib_paths());
    } else {
        ctx.light_paths = expand(lights);
    }
    if let Some(s) = ser {
        let (frames, _) = io::ser::load_ser(s, 0)?;
        ctx.frames = frames.to_vec()?;
    }
    if ctx.light_paths.is_empty() && ctx.frames.is_empty() {
        bail!("Aucune image : indiquez --session, --lights ou --ser");
    }
    let pf = |d: usize, t: usize, l: &str| prog.call(d, t, l);
    p.run(&mut ctx, Some(&pf))?;
    let img = ctx.image.ok_or_else(|| anyhow!("Le processus n'a produit aucune image (étape d'empilement absente ?)"))?;
    io::save(out, &img)?;
    println!("Écrit : {} ({}×{})", out.display(), img.w, img.h);
    Ok(0)
}

fn cmd_mosaic(panels: &[String], feather: f64, out: &Path) -> Result<i32> {
    let paths = expand(panels);
    if paths.len() < 2 {
        bail!("Mosaïque : au moins 2 panneaux");
    }
    let imgs: Vec<Image> = paths.iter().map(io::load).collect::<Result<_>>()?;
    let wcs: Vec<Option<astroforge::wcs::Wcs>> = paths.iter().map(|p| mosaic::wcs_from_fits(p)).collect();
    let img = if wcs.iter().all(|w| w.is_some()) {
        let w: Vec<_> = wcs.into_iter().map(|w| w.unwrap()).collect();
        let (o, wout, _) = mosaic::mosaic_wcs(&imgs, &w, None, feather)?;
        if is_fits(out) {
            io::save_image(out, &o, "auto", Some(&wout.to_header()))?;
            println!("Écrit : {} ({}×{}, WCS inclus)", out.display(), o.w, o.h);
            return Ok(0);
        }
        o
    } else {
        eprintln!("Panneaux sans WCS : assemblage par appariement d'étoiles");
        mosaic::mosaic_features(&imgs, feather)?.0
    };
    io::save(out, &img)?;
    println!("Écrit : {} ({}×{})", out.display(), img.w, img.h);
    Ok(0)
}

fn cmd_info(check: bool) -> Result<i32> {
    println!("AstroForge {}", astroforge::VERSION);
    println!("Données utilisateur : {}", astroforge::paths::data_root().display());
    println!("Calcul : processeur, {} fil(s) d'exécution parallèles (Rust/rayon)", rayon::current_num_threads());
    println!("IA ONNX : {}", if cfg!(feature = "onnx") { "intégrée (tract)" } else { "non compilée (option « onnx »)" });
    println!("Langue : {}", updates::configured_language());
    if check {
        let url = updates::update_url();
        if url.is_empty() {
            println!("Mises à jour : aucune adresse configurée (settings.json « update_url »)");
        } else {
            match updates::check(&url, astroforge::VERSION)? {
                Some(i) => println!("Nouvelle version : {} — {}", i.version, i.url),
                None => println!("Mises à jour : version à jour"),
            }
        }
    }
    Ok(0)
}

fn run(cli: Cli) -> Result<i32> {
    let prog = Progress::new(cli.verbose);
    match cli.cmd {
        Cmd::Sort { folder, json } => cmd_sort(&folder, json),
        Cmd::Analyze { files, bayer, weights, csv } => cmd_analyze(&files, &bayer, &weights, csv.as_deref(), &prog),
        Cmd::Stack(a) => cmd_stack(&a, &prog),
        Cmd::Multinight { folders, method, sigma, normalization, weights, bayer, out } => cmd_multinight(&folders, &method, sigma, &normalization, &weights, &bayer, &out, &prog),
        Cmd::Solve { image, catalog, ra, dec, radius, focal, pixel, internal, write_wcs } => {
            cmd_solve(&image, catalog.as_deref(), ra.as_deref(), dec.as_deref(), radius, focal, pixel, internal, write_wcs)
        }
        Cmd::Annotate { image, mag, out } => cmd_annotate(&image, mag, out.as_deref()),
        Cmd::Compose { ha, oiii, sii, r, g, b, lum, palette, align, no_normalize, saturation, out } => cmd_compose(
            vec![("Ha", ha), ("OIII", oiii), ("SII", sii), ("R", r), ("G", g), ("B", b)],
            lum.as_deref(),
            &palette,
            align,
            no_normalize,
            saturation,
            &out,
        ),
        Cmd::Batch { sequence, input, output } => cmd_batch(&sequence, &input, &output, &prog),
        Cmd::Live { folder, duration, interval, darks, flats, bayer, kappa, out } => cmd_live(&folder, duration, interval, darks, flats, &bayer, kappa, out),
        Cmd::Planetary(a) => cmd_planetary(&a, &prog),
        Cmd::PlanetaryAuto(a) => cmd_planetary_auto(&a, cli.verbose),
        Cmd::Timelapse(a) => cmd_timelapse(&a, cli.verbose),
        Cmd::Pipeline { process, session, group, lights, ser, bayer, out } => cmd_pipeline(&process, session.as_deref(), group, &lights, ser.as_deref(), &bayer, &out, &prog),
        Cmd::Mosaic { panels, feather, out } => cmd_mosaic(&panels, feather, &out),
        Cmd::Info { check_updates } => cmd_info(check_updates),
    }
}

fn main() {
    // Sortie redirigée vers un tube fermé (`astroforge info | head`) :
    // terminer silencieusement comme les outils Unix au lieu de paniquer.
    #[cfg(unix)]
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    let cli = Cli::parse();
    let code = match run(cli) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Erreur : {:#}", e);
            1
        }
    };
    std::process::exit(code);
}
