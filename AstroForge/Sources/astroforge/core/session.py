# -*- coding: utf-8 -*-
"""
Sessions d'acquisition : tri automatique des fichiers d'une nuit
(lumières, darks, flats, darks de flats, bias) d'après les en-têtes FITS
(IMAGETYP, EXPTIME, FILTER, GAIN, CCD-TEMP, XBINNING, OBJECT…) ou, à défaut,
d'après les noms de fichiers et de dossiers, puis appariement des images de
calibration avec chaque groupe de lumières (même gain, même binning, même
taille, exposition et température proches).

Usage :
    plan = plan_session("/chemin/vers/nuit")
    for g in plan.groups: print(g.summary())
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .calibration import CALIB_EXTENSIONS

FRAME_TYPES = ("light", "dark", "flat", "darkflat", "bias")

# valeurs IMAGETYP/FRAME rencontrées (NINA, SGP, MaxIm DL, KStars/Ekos,
# ASIAIR, APT, SharpCap…), comparées en minuscules sans espaces ni tirets
_TYPE_WORDS = {
    "light": ("light", "lightframe", "object", "science"),
    "dark": ("dark", "darkframe"),
    "flat": ("flat", "flatfield", "flatframe", "skyflat", "domeflat"),
    "darkflat": ("darkflat", "flatdark", "darkflats", "flatdarks", "dflat"),
    "bias": ("bias", "biasframe", "offset", "offsetframe", "zero"),
}
_NAME_HINTS = [  # ordre important : « darkflat » avant « dark » et « flat »
    ("darkflat", re.compile(r"(dark[\s_-]*flat|flat[\s_-]*dark|dflat)", re.I)),
    ("bias", re.compile(r"(bias|offset)", re.I)),
    ("flat", re.compile(r"flat", re.I)),
    ("dark", re.compile(r"dark", re.I)),
    ("light", re.compile(r"(light|lum|object|brut|target)", re.I)),
]


@dataclass
class FrameInfo:
    path: str
    kind: str = "light"
    exptime: float | None = None
    filter: str = ""
    gain: float | None = None
    offset: float | None = None
    temp: float | None = None
    binning: int = 1
    width: int | None = None
    height: int | None = None
    date_obs: str = ""
    object: str = ""
    bayer: str | None = None
    source: str = "nom"          # « en-tête » ou « nom » : origine du type

    def key(self) -> tuple:
        return (self.object or "?", self.filter or "", _round(self.exptime),
                _round(self.gain), self.binning, self.width, self.height)


def _round(v, nd=2):
    return None if v is None else round(float(v), nd)


def _num(h: dict, *keys) -> float | None:
    for k in keys:
        if k in h:
            try:
                return float(h[k])
            except (TypeError, ValueError):
                continue
    return None


def _norm_word(v) -> str:
    return re.sub(r"[\s_\-]", "", str(v or "")).lower()


def read_header(path: str) -> dict:
    """En-tête FITS (première HDU image) ; {} pour les autres formats."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".fits", ".fit", ".fts", ".fz"):
        return {}
    try:
        from astropy.io import fits
        with fits.open(path, memmap=False) as hdul:
            for hdu in hdul:
                if getattr(hdu, "header", None) is not None and hdu.header.get("NAXIS", 0) >= 2:
                    return {k: hdu.header[k] for k in hdu.header.keys()
                            if k and k not in ("COMMENT", "HISTORY")}
            return {k: hdul[0].header[k] for k in hdul[0].header.keys()
                    if k and k not in ("COMMENT", "HISTORY")}
    except Exception:
        return {}


def classify(path: str, header: dict | None = None) -> tuple[str, str]:
    """Type d'image -> (type, origine) : mot-clé IMAGETYP/FRAME d'abord, puis
    nom du fichier, puis noms des dossiers parents."""
    h = header or {}
    for key in ("IMAGETYP", "FRAME", "FRAMETYP", "IMAGETYPE"):
        if key in h:
            w = _norm_word(h[key])
            for kind, words in _TYPE_WORDS.items():
                if w in words:
                    return kind, "en-tête"
            if "flat" in w and "dark" in w:
                return "darkflat", "en-tête"
            for kind in ("bias", "flat", "dark", "light"):
                if any(x in w for x in _TYPE_WORDS[kind]):
                    return kind, "en-tête"
    parts = [os.path.basename(path)] + list(reversed(os.path.normpath(
        os.path.dirname(path)).split(os.sep)))[:3]
    for part in parts:
        stem = os.path.splitext(part)[0]
        for kind, rx in _NAME_HINTS:
            if rx.search(stem):
                return kind, "nom"
    return "light", "défaut"


_FILTER_ALIASES = {
    "l": "L", "lum": "L", "luminance": "L", "r": "R", "red": "R", "rouge": "R",
    "g": "G", "green": "G", "vert": "G", "b": "B", "blue": "B", "bleu": "B",
    "ha": "Ha", "halpha": "Ha", "h-alpha": "Ha", "oiii": "OIII", "o3": "OIII",
    "sii": "SII", "s2": "SII",
}


def infer_filter(path: str) -> str:
    """Filtre déduit du nom (« flats_R », « M31_Ha_300s.fits ») ; '' sinon."""
    parts = [os.path.splitext(os.path.basename(path))[0],
             os.path.basename(os.path.dirname(path))]
    for part in parts:
        for tok in re.split(r"[\s_\-.()\[\]]+", part):
            alias = _FILTER_ALIASES.get(tok.lower())
            if alias:
                return alias
    return ""


def frame_info(path: str) -> FrameInfo:
    h = read_header(path)
    kind, src = classify(path, h)
    binning = _num(h, "XBINNING", "BINNING")
    info = FrameInfo(
        path=path, kind=kind, source=src,
        exptime=_num(h, "EXPTIME", "EXPOSURE", "EXP"),
        filter=str(h.get("FILTER", "") or "").strip(),
        gain=_num(h, "GAIN", "EGAIN_SET", "ISOSPEED"),
        offset=_num(h, "OFFSET", "BLKLEVEL"),
        temp=_num(h, "CCD-TEMP", "CCD_TEMP", "SENSOR-TEMP", "SET-TEMP"),
        binning=int(binning) if binning else 1,
        width=int(h["NAXIS1"]) if "NAXIS1" in h else None,
        height=int(h["NAXIS2"]) if "NAXIS2" in h else None,
        date_obs=str(h.get("DATE-OBS", "") or ""),
        object=str(h.get("OBJECT", "") or "").strip(),
        bayer=(str(h.get("BAYERPAT", "")).strip().upper() or None),
    )
    if not info.filter:
        info.filter = infer_filter(path)
    if info.exptime is None:  # « 300s », « 120sec » dans le nom
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:s|sec|secs)(?![a-z])", os.path.basename(path), re.I)
        if m:
            info.exptime = float(m.group(1))
    if info.width is None and os.path.splitext(path)[1].lower() not in (".cr2", ".cr3", ".nef",
                                                                       ".arw", ".dng", ".raf",
                                                                       ".orf", ".rw2", ".pef",
                                                                       ".srw"):
        try:  # taille lue dans l'en-tête du fichier, sans décoder l'image
            from PIL import Image
            with Image.open(path) as im:
                info.width, info.height = im.size
        except Exception:
            pass
    return info


def scan_folder(folder: str, recursive: bool = True) -> list[FrameInfo]:
    """Inventaire de tous les fichiers image d'un dossier (et sous-dossiers)."""
    out = []
    if recursive:
        walker = os.walk(folder)
    else:
        walker = [(folder, [], os.listdir(folder))]
    for root, _dirs, files in walker:
        for f in sorted(files):
            if f.lower().endswith(CALIB_EXTENSIONS):
                out.append(frame_info(os.path.join(root, f)))
    return out


@dataclass
class LightGroup:
    key: tuple
    lights: list[FrameInfo] = field(default_factory=list)
    darks: list[FrameInfo] = field(default_factory=list)
    flats: list[FrameInfo] = field(default_factory=list)
    darkflats: list[FrameInfo] = field(default_factory=list)
    biases: list[FrameInfo] = field(default_factory=list)
    dark_scale: float = 1.0
    warnings: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        obj, filt, exp, gain, binning, w, h = self.key
        parts = [obj]
        if filt:
            parts.append(f"filtre {filt}")
        if exp is not None:
            parts.append(f"{exp:g} s")
        if gain is not None:
            parts.append(f"gain {gain:g}")
        if binning and binning != 1:
            parts.append(f"bin {binning}")
        return " · ".join(parts)

    def summary(self) -> str:
        txt = (f"{self.name} : {len(self.lights)} lumières, {len(self.darks)} darks, "
               f"{len(self.flats)} flats, {len(self.darkflats)} darks de flats, "
               f"{len(self.biases)} bias")
        if abs(self.dark_scale - 1.0) > 1e-6:
            txt += f" (dark ×{self.dark_scale:.3f})"
        if self.warnings:
            txt += "\n  ⚠ " + "\n  ⚠ ".join(self.warnings)
        return txt

    def paths(self, kind: str) -> list[str]:
        return [f.path for f in getattr(self, {"light": "lights", "dark": "darks",
                                               "flat": "flats", "darkflat": "darkflats",
                                               "bias": "biases"}[kind])]


@dataclass
class SessionPlan:
    frames: list[FrameInfo]
    groups: list[LightGroup]
    unused: list[FrameInfo]

    def summary(self) -> str:
        counts = {k: sum(1 for f in self.frames if f.kind == k) for k in FRAME_TYPES}
        head = ", ".join(f"{v} {k}" for k, v in counts.items() if v)
        return f"{len(self.frames)} fichiers ({head}) — {len(self.groups)} groupe(s) de lumières\n" + \
            "\n".join(g.summary() for g in self.groups)


def _same_geometry(a: FrameInfo, b: FrameInfo) -> bool:
    if a.binning != b.binning:
        return False
    if None not in (a.width, b.width, a.height, b.height):
        return (a.width, a.height) == (b.width, b.height)
    return True


def _gain_ok(a: FrameInfo, b: FrameInfo) -> bool:
    return a.gain is None or b.gain is None or abs(a.gain - b.gain) < 0.5


def _temp_ok(a: FrameInfo, b: FrameInfo, tol: float) -> bool:
    return a.temp is None or b.temp is None or abs(a.temp - b.temp) <= tol


def match_group(group: LightGroup, frames: list[FrameInfo], temp_tol: float = 3.0,
                exp_tol: float = 0.05) -> LightGroup:
    """Associe à un groupe de lumières ses images de calibration."""
    ref = group.lights[0]
    darks = [f for f in frames if f.kind == "dark" and _same_geometry(f, ref)
             and _gain_ok(f, ref) and _temp_ok(f, ref, temp_tol)]
    biases = [f for f in frames if f.kind == "bias" and _same_geometry(f, ref)
              and _gain_ok(f, ref)]
    if ref.exptime is not None and darks:
        exact = [f for f in darks if f.exptime is not None
                 and abs(f.exptime - ref.exptime) <= exp_tol * max(ref.exptime, 1e-3)]
        if exact:
            darks = exact
        else:  # exposition différente : on prend la plus proche, mise à l'échelle
            with_exp = [f for f in darks if f.exptime]
            if with_exp:
                best = min({f.exptime for f in with_exp}, key=lambda e: abs(e - ref.exptime))
                darks = [f for f in with_exp if f.exptime == best]
                if biases:
                    group.dark_scale = float(ref.exptime / best)
                    group.warnings.append(
                        f"darks de {best:g} s pour des lumières de {ref.exptime:g} s : "
                        f"dark thermique mis à l'échelle ×{group.dark_scale:.3f}")
                else:
                    group.warnings.append(
                        f"darks de {best:g} s pour des lumières de {ref.exptime:g} s et "
                        "aucun bias : mise à l'échelle impossible (dark utilisé tel quel)")
    if darks and ref.temp is not None:
        temps = [f.temp for f in darks if f.temp is not None]
        if temps and min(abs(t - ref.temp) for t in temps) > 1.0:
            group.warnings.append("température des darks différente de plus de 1 °C")
    flats = [f for f in frames if f.kind == "flat" and _same_geometry(f, ref)]
    if ref.filter:
        same = [f for f in flats if f.filter == ref.filter]
        flats = same or [f for f in flats if not f.filter]
    darkflats = []
    if flats:
        fexp = flats[0].exptime
        darkflats = [f for f in frames if f.kind == "darkflat" and _same_geometry(f, ref)
                     and (fexp is None or f.exptime is None
                          or abs(f.exptime - fexp) <= exp_tol * max(fexp, 1e-3) + 0.01)]
    group.darks, group.biases, group.flats, group.darkflats = darks, biases, flats, darkflats
    if not darks:
        group.warnings.append("aucun dark compatible (gain/binning/taille/température)")
    if not flats:
        group.warnings.append("aucun flat compatible (filtre/binning/taille)")
    return group


def plan_session(folder_or_frames, recursive: bool = True, temp_tol: float = 3.0) -> SessionPlan:
    """Tri complet d'une session : groupes de lumières (objet, filtre,
    exposition, gain, binning, taille) avec leurs images de calibration."""
    frames = (scan_folder(folder_or_frames, recursive) if isinstance(folder_or_frames, str)
              else list(folder_or_frames))
    groups: dict[tuple, LightGroup] = {}
    for f in frames:
        if f.kind == "light":
            groups.setdefault(f.key(), LightGroup(f.key())).lights.append(f)
    out = [match_group(g, frames, temp_tol) for g in groups.values()]
    out.sort(key=lambda g: (-len(g.lights), g.name))
    used = {f.path for g in out for k in FRAME_TYPES for f in
            getattr(g, {"light": "lights", "dark": "darks", "flat": "flats",
                        "darkflat": "darkflats", "bias": "biases"}[k])}
    return SessionPlan(frames, out, [f for f in frames if f.path not in used])


def stack_header(light_paths: list[str], weights=None, extra: dict | None = None) -> dict:
    """En-tête FITS d'une image empilée : nombre d'images, pose totale,
    date de la première image, objet, filtre, instrument, logiciel."""
    import datetime as _dt
    from .. import __version__
    infos = [frame_info(p) for p in light_paths]
    used = [i for i, w in zip(infos, weights or [1] * len(infos)) if w > 0]
    h = {"CREATOR": f"AstroForge {__version__}", "SWCREATE": f"AstroForge {__version__}",
         "DATE": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
         "NCOMBINE": len(used), "IMAGETYP": "Master Light"}
    exps = [i.exptime for i in used if i.exptime]
    if exps:
        h["EXPTIME"] = float(sum(exps))
        h["EXPOSURE"] = h["EXPTIME"]
        h["SUBEXP"] = float(exps[0])
    dates = sorted(i.date_obs for i in used if i.date_obs)
    if dates:
        h["DATE-OBS"] = dates[0]
        h["DATE-END"] = dates[-1]
    for key, attr in (("OBJECT", "object"), ("FILTER", "filter")):
        vals = {getattr(i, attr) for i in used if getattr(i, attr)}
        if len(vals) == 1:
            h[key] = vals.pop()
    gains = {i.gain for i in used if i.gain is not None}
    if len(gains) == 1:
        h["GAIN"] = gains.pop()
    first = read_header(light_paths[0]) if light_paths else {}
    for k in ("INSTRUME", "TELESCOP", "FOCALLEN", "XPIXSZ", "YPIXSZ", "OBSERVER", "SITELAT",
              "SITELONG"):
        if k in first:
            h[k] = first[k]
    h.update(extra or {})
    return h
