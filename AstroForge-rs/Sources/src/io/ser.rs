//! Vidéo SER (planétaire) : norme v3 (en-tête fixe de 178 octets —
//! SharpCap, FireCapture, ASIStudio, Siril, AutoStakkert!…), fichiers des
//! anciennes versions d'AstroForge (chaînes préfixées, dialecte ≤ 1.7).
//! Les frames sont projetées en mémoire (memmap) et converties à la demande.

use super::debayer::{debayer, BAYER_PATTERNS};
use crate::frames::Frames;
use crate::image::{f_to_u16, Image};
use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SerType {
    U8,
    U16,
    F32,
}

impl SerType {
    fn size(&self) -> usize {
        match self {
            SerType::U8 => 1,
            SerType::U16 => 2,
            SerType::F32 => 4,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SerMeta {
    pub camera: String,
    pub observer: String,
    pub instrument: String,
    pub telescope: String,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub frames_header: usize,
    pub date_utc: Option<String>,
    pub timestamps: Vec<Option<String>>,
    pub bits: usize,
    pub color_mode: i32,
    pub bayer: Option<String>,
    pub layout: String,
}

#[derive(Clone)]
pub struct SerFrames {
    mmap: Arc<Mmap>,
    offset: usize,
    w: usize,
    h: usize,
    planes: usize,
    dtype: SerType,
    scale: f32,
    bgr: bool,
    bayer: Option<String>,
    start: usize,
    count: usize,
}

impl SerFrames {
    fn frame_bytes(&self) -> usize {
        self.w * self.h * self.planes * self.dtype.size()
    }

    /// Sous-séquence [a, b).
    pub fn slice(&self, a: usize, b: usize) -> SerFrames {
        let a = a.min(self.count);
        let b = b.clamp(a, self.count);
        let mut s = self.clone();
        s.start = self.start + a;
        s.count = b - a;
        s
    }

    pub fn frame_shape(&self) -> (usize, usize, usize) {
        let color = self.planes == 3 || self.bayer.is_some();
        (self.h, self.w, if color { 3 } else { 1 })
    }

    pub fn try_get(&self, i: usize) -> Result<Image> {
        if i >= self.count {
            bail!("Frame {} hors limites ({} frames)", i, self.count);
        }
        let fb = self.frame_bytes();
        let off = self.offset + (self.start + i) * fb;
        let raw = &self.mmap[off..off + fb];
        let n = self.w * self.h * self.planes;
        let mut v = vec![0.0f32; n];
        match self.dtype {
            SerType::U8 => {
                for k in 0..n {
                    v[k] = (raw[k] as f32 / self.scale).clamp(0.0, 1.0);
                }
            }
            SerType::U16 => {
                for k in 0..n {
                    v[k] = (u16::from_le_bytes([raw[2 * k], raw[2 * k + 1]]) as f32 / self.scale).clamp(0.0, 1.0);
                }
            }
            SerType::F32 => {
                for k in 0..n {
                    let x = f32::from_le_bytes(raw[4 * k..4 * k + 4].try_into().unwrap());
                    v[k] = if x.is_finite() { (x / self.scale).clamp(0.0, 1.0) } else { 0.0 };
                }
            }
        }
        let mut img = Image::from_vec(self.w, self.h, self.planes, v);
        if self.planes == 3 && self.bgr {
            for p in img.data.chunks_mut(3) {
                p.swap(0, 2);
            }
        } else if self.planes == 1 {
            if let Some(b) = &self.bayer {
                img = debayer(&img, b)?;
            }
        }
        Ok(img)
    }

    /// Charge toutes les frames en mémoire.
    pub fn to_vec(&self) -> Result<Vec<Image>> {
        (0..self.count).map(|i| self.try_get(i)).collect()
    }
}

impl Frames for SerFrames {
    fn len(&self) -> usize {
        self.count
    }
    fn get(&self, i: usize) -> Image {
        self.try_get(i).expect("lecture de frame SER")
    }
}

fn rd_i32(b: &[u8], p: &mut usize) -> Option<i32> {
    if *p + 4 > b.len() {
        return None;
    }
    let v = i32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
    *p += 4;
    Some(v)
}

fn rd_strings(b: &[u8], p: &mut usize, count: usize) -> Option<Vec<String>> {
    let mut out = Vec::new();
    for _ in 0..count {
        let n = rd_i32(b, p)?;
        if !(0..=4096).contains(&n) {
            return None;
        }
        let n = n as usize;
        if *p + n > b.len() {
            return None;
        }
        let s = String::from_utf8_lossy(&b[*p..*p + n]).trim_end_matches(['\0', ' ']).to_string();
        *p += n;
        out.push(s);
    }
    Some(out)
}

fn ser_dtype(depth: i32) -> Option<(SerType, f32)> {
    match depth {
        1 | 8 => Some((SerType::U8, 255.0)),
        2 | 16 => Some((SerType::U16, 65535.0)),
        4 | 32 => Some((SerType::F32, 1.0)),
        3..=7 => Some((SerType::U8, ((1u32 << depth) - 1) as f32)),
        9..=15 => Some((SerType::U16, ((1u32 << depth) - 1) as f32)),
        _ => None,
    }
}

fn ser_bayer(cm: i32) -> Option<&'static str> {
    match cm {
        8 => Some("RGGB"),
        9 => Some("GRBG"),
        10 => Some("GBRG"),
        11 => Some("BGGR"),
        _ => None,
    }
}

/// (plans, bayer, bgr, rang)
fn variants(cm: i32) -> Vec<(usize, Option<String>, bool, i32)> {
    let mut out = Vec::new();
    if cm == 0 || (16..=19).contains(&cm) {
        out.push((1, None, false, 0));
    } else if let Some(b) = ser_bayer(cm) {
        out.push((1, Some(b.to_string()), false, 0));
    } else if cm == 100 {
        out.push((3, None, false, 0));
    } else if cm == 101 {
        out.push((3, None, true, 0));
    }
    if cm == 8 {
        out.push((3, None, true, 1));
    }
    if cm == 16 {
        out.push((3, None, false, 1));
    }
    out
}

#[derive(Clone, Debug)]
struct Cand {
    w: usize,
    h: usize,
    nframes: usize,
    dtype: SerType,
    scale: f32,
    per: usize,
    colormode: i32,
    bayer: Option<String>,
    bgr: bool,
    variant: i32,
    camera: String,
    observer: String,
    instrument: String,
    telescope: String,
    header_end: usize,
    layout: &'static str,
    dt_utc: i64,
    fb: usize,
    possible: usize,
    consistent: bool,
    exact: bool,
}

fn strip_spec(b: &[u8]) -> String {
    String::from_utf8_lossy(b).trim_matches(|c| c == '?' || c == '-' || c == ' ' || c == '\0').to_string()
}

fn candidates(b: &[u8], layout: &'static str) -> Vec<Cand> {
    let mut p = 14usize;
    let (w, h, nframes, depth, colormode, camera, obs, inst, tel, legacy_bayer);
    if layout == "legacy" {
        let strs = match rd_strings(b, &mut p, 4) {
            Some(s) => s,
            None => return vec![],
        };
        let vals: Vec<i32> = (0..5).filter_map(|_| rd_i32(b, &mut p)).collect();
        if vals.len() < 5 {
            return vec![];
        }
        w = vals[0];
        h = vals[1];
        nframes = vals[2];
        depth = vals[3];
        colormode = vals[4];
        camera = strs[0].clone();
        obs = strs[1].clone();
        inst = strs[2].clone();
        tel = strs[3].clone();
        legacy_bayer = if colormode == 100 {
            if p + 32 > b.len() {
                return vec![];
            }
            let s = String::from_utf8_lossy(&b[p..p + 32]).trim_end_matches(['\0', ' ']).to_string();
            p += 32;
            Some(s)
        } else {
            None
        };
    } else {
        let vals: Vec<i32> = (0..7).filter_map(|_| rd_i32(b, &mut p)).collect();
        if vals.len() < 7 {
            return vec![];
        }
        colormode = vals[1];
        w = vals[3];
        h = vals[4];
        depth = vals[5];
        nframes = vals[6];
        legacy_bayer = None;
        if layout == "spec" {
            if p + 120 > b.len() {
                return vec![];
            }
            obs = strip_spec(&b[p..p + 40]);
            inst = strip_spec(&b[p + 40..p + 80]);
            tel = strip_spec(&b[p + 80..p + 120]);
            camera = inst.clone();
            p += 120;
        } else {
            let s = match rd_strings(b, &mut p, 3) {
                Some(s) => s,
                None => return vec![],
            };
            obs = s[0].clone();
            inst = s[1].clone();
            tel = s[2].clone();
            camera = inst.clone();
        }
    }
    if !((1..=20000).contains(&w) && (1..=20000).contains(&h) && (1..=10_000_000).contains(&nframes)) {
        return vec![];
    }
    let (dtype, scale) = match ser_dtype(depth) {
        Some(x) => x,
        None => return vec![],
    };
    let dt_utc = if p + 16 <= b.len() { i64::from_le_bytes(b[p + 8..p + 16].try_into().unwrap()) } else { 0 };
    p += 16;
    let header_end = p;
    let vars = if layout == "legacy" && colormode == 100 {
        vec![(1usize, legacy_bayer.clone(), false, 0i32)]
    } else {
        variants(colormode)
    };
    vars.into_iter()
        .map(|(per, bayer, bgr, rank)| Cand {
            w: w as usize,
            h: h as usize,
            nframes: nframes as usize,
            dtype,
            scale,
            per,
            colormode,
            bayer,
            bgr,
            variant: rank,
            camera: camera.clone(),
            observer: obs.clone(),
            instrument: inst.clone(),
            telescope: tel.clone(),
            header_end,
            layout,
            dt_utc,
            fb: 0,
            possible: 0,
            consistent: false,
            exact: false,
        })
        .collect()
}

/// Ticks .NET (100 ns depuis 0001-01-01) -> ISO 8601 (millisecondes).
pub fn ticks_to_iso(t: i64) -> Option<String> {
    if t <= 0 || t > 3155378975999999999 {
        return None;
    }
    let base = chrono::NaiveDate::from_ymd_opt(1, 1, 1)?.and_hms_opt(0, 0, 0)?;
    let d = base.checked_add_signed(chrono::Duration::microseconds(t / 10))?;
    use chrono::Datelike;
    if d.year() < 1990 {
        return None;
    }
    Some(d.format("%Y-%m-%dT%H:%M:%S%.3f").to_string())
}

/// ISO 8601 -> ticks .NET (UTC).
pub fn iso_to_ticks(s: &str) -> Result<i64> {
    let d = crate::timeutil::parse_iso_utc(s)?;
    let base = chrono::NaiveDate::from_ymd_opt(1, 1, 1).unwrap().and_hms_opt(0, 0, 0).unwrap();
    let delta = d - base;
    Ok(delta.num_microseconds().unwrap_or(0) * 10)
}

/// Lit un SER : (frames projetées, métadonnées). `max_frames` 0 = toutes.
pub fn load_ser(path: &Path, max_frames: usize) -> Result<(SerFrames, SerMeta)> {
    let f = std::fs::File::open(path).with_context(|| format!("Impossible de lire : {}", path.display()))?;
    let mmap = unsafe { Mmap::map(&f)? };
    let b: &[u8] = &mmap;
    if b.len() < 14 || &b[..14] != b"LUCAM-RECORDER" {
        bail!("Pas un fichier SER : {}", path.display());
    }
    let size = b.len();
    let mut cands: Vec<Cand> = Vec::new();
    for layout in ["spec", "v3", "legacy"] {
        for mut c in candidates(b, layout) {
            c.fb = c.w * c.h * c.per * c.dtype.size();
            c.possible = size.saturating_sub(c.header_end) / c.fb;
            if c.possible < 1 {
                continue;
            }
            let over = c.possible.saturating_sub(c.nframes);
            c.consistent = c.nframes <= c.possible && over * c.fb <= 8 * c.nframes + c.fb;
            let rest = size as i64 - c.header_end as i64 - (c.nframes * c.fb) as i64;
            c.exact = rest == 0 || rest == 8 * c.nframes as i64;
            cands.push(c);
        }
    }
    let rank = |l: &str| match l {
        "spec" => 0,
        "v3" => 1,
        _ => 2,
    };
    let good: Vec<&Cand> = cands.iter().filter(|c| c.consistent).collect();
    let best: Cand = if !good.is_empty() {
        (*good
            .iter()
            .min_by_key(|c| (if c.exact { 0 } else { 1 }, c.possible as i64 - c.nframes as i64, c.variant, rank(c.layout)))
            .unwrap())
        .clone()
    } else if !cands.is_empty() {
        cands
            .iter()
            .max_by_key(|c| (c.possible >= 1, -c.variant, c.possible))
            .unwrap()
            .clone()
    } else {
        bail!("En-tête SER non reconnu — fichier corrompu ou format inconnu (vidéo AVI/MP4 renommée en .ser ?)");
    };
    let mut count = best.nframes.min(best.possible);
    if max_frames > 0 {
        count = count.min(max_frames);
    }
    let trailer = best.header_end + best.nframes * best.fb;
    let mut timestamps: Vec<Option<String>> = Vec::new();
    if size >= trailer + 8 * best.nframes {
        for i in 0..count {
            let o = trailer + 8 * i;
            let t = i64::from_le_bytes(b[o..o + 8].try_into().unwrap());
            timestamps.push(ticks_to_iso(t));
        }
        if !timestamps.iter().any(|t| t.is_some()) {
            timestamps.clear();
        }
    }
    let meta = SerMeta {
        camera: best.camera.clone(),
        observer: best.observer.clone(),
        instrument: best.instrument.clone(),
        telescope: best.telescope.clone(),
        width: best.w,
        height: best.h,
        frames: count,
        frames_header: best.nframes,
        date_utc: ticks_to_iso(best.dt_utc),
        timestamps,
        bits: best.dtype.size() * 8,
        color_mode: best.colormode,
        bayer: best.bayer.clone(),
        layout: best.layout.to_string(),
    };
    let bayer = best.bayer.clone().filter(|b| BAYER_PATTERNS.contains(&b.as_str()));
    let frames = SerFrames {
        offset: best.header_end,
        w: best.w,
        h: best.h,
        planes: best.per,
        dtype: best.dtype,
        scale: best.scale,
        bgr: best.bgr,
        bayer,
        start: 0,
        count,
        mmap: Arc::new(mmap),
    };
    Ok((frames, meta))
}

/// Écrit des frames (mono ou RGB, [0..1]) en SER v3 standard 16 bits.
pub fn write_ser(path: &Path, frames: &[Image], observer: &str, instrument: &str, telescope: &str, timestamps: Option<&[String]>) -> Result<()> {
    if frames.is_empty() {
        bail!("Aucune frame à écrire");
    }
    let (w, h, c) = (frames[0].w, frames[0].h, frames[0].c);
    let colormode: i32 = if c == 1 { 0 } else { 100 };
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"LUCAM-RECORDER")?;
    for v in [0i32, colormode, 0, w as i32, h as i32, 16, frames.len() as i32] {
        f.write_all(&v.to_le_bytes())?;
    }
    let field = |s: &str| -> Vec<u8> {
        let mut b: Vec<u8> = s.chars().map(|c| if c.is_ascii() { c as u8 } else { b'?' }).take(40).collect();
        b.resize(40, 0);
        b
    };
    let inst = if instrument.is_empty() { "AstroForge" } else { instrument };
    f.write_all(&field(observer))?;
    f.write_all(&field(inst))?;
    f.write_all(&field(telescope))?;
    let t0 = match timestamps {
        Some(ts) if !ts.is_empty() => iso_to_ticks(&ts[0])?,
        _ => 0,
    };
    f.write_all(&t0.to_le_bytes())?;
    f.write_all(&t0.to_le_bytes())?;
    for fr in frames {
        if fr.w != w || fr.h != h || fr.c != c {
            bail!("SER : toutes les frames doivent avoir la même taille");
        }
        let mut buf = Vec::with_capacity(fr.data.len() * 2);
        for &v in &fr.data {
            buf.extend_from_slice(&f_to_u16(v).to_le_bytes());
        }
        f.write_all(&buf)?;
    }
    if let Some(ts) = timestamps {
        if !ts.is_empty() {
            for t in ts {
                f.write_all(&iso_to_ticks(t)?.to_le_bytes())?;
            }
        }
    }
    f.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("t.ser");
        let frames: Vec<Image> = (0..5).map(|k| Image::from_fn(12, 8, move |x, y| ((x + y + k) % 7) as f32 / 7.0)).collect();
        let ts: Vec<String> = (0..5).map(|k| format!("2024-03-01T21:00:0{}.250", k)).collect();
        write_ser(&p, &frames, "Moi", "ASI", "C8", Some(&ts)).unwrap();
        let (s, m) = load_ser(&p, 0).unwrap();
        assert_eq!(s.len(), 5);
        assert_eq!(m.layout, "spec");
        assert_eq!(m.timestamps[2].as_deref(), Some("2024-03-01T21:00:02.250"));
        let f2 = s.get(2);
        assert!((f2.get(3, 4, 0) - frames[2].get(3, 4, 0)).abs() < 2e-5);
        let rgb: Vec<Image> = frames.iter().map(|f| f.to_rgb()).collect();
        write_ser(&p, &rgb, "", "", "", None).unwrap();
        let (s, _) = load_ser(&p, 3).unwrap();
        assert_eq!(s.len(), 3);
        assert_eq!(s.get(0).c, 3);
    }
}
