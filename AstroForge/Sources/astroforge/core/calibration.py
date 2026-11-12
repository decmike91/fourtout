# -*- coding: utf-8 -*-
"""
Calibration : masters (bias, dark, dark-flat, flat) et calibration des images brutes.
Toutes les fonctions travaillent en float32 [0..1] (ou ADU brut FITS).
"""
from __future__ import annotations
import numpy as np

from .stacking import combine_sigma_clip  # réutilisation du rejet sigma


def _stack3d(images: list[np.ndarray]) -> np.ndarray:
    return np.stack([np.asarray(im, np.float32) for im in images], axis=0)


def make_master_bias(bias_frames: list[np.ndarray], sigma: float = 3.0) -> np.ndarray:
    """Master bias = médiane (ou sigma-clip) des offsets."""
    if not bias_frames:
        raise ValueError("Aucun bias fourni")
    return combine_sigma_clip(_stack3d(bias_frames), sigma=sigma)


def make_master_dark(dark_frames: list[np.ndarray], master_bias: np.ndarray | None = None,
                     sigma: float = 3.0) -> np.ndarray:
    """Master dark = médiane des (dark - bias). Le dark thermique seul est retourné."""
    if not dark_frames:
        raise ValueError("Aucune dark fournie")
    d = _stack3d(dark_frames)
    if master_bias is not None:
        d = d - np.asarray(master_bias, np.float32)
    return combine_sigma_clip(d, sigma=sigma)


def make_master_darkflat(darkflat_frames: list[np.ndarray], master_bias: np.ndarray | None = None,
                         sigma: float = 3.0) -> np.ndarray:
    return make_master_dark(darkflat_frames, master_bias, sigma)


def make_master_flat(flat_frames: list[np.ndarray], master_bias: np.ndarray | None = None,
                     master_darkflat: np.ndarray | None = None,
                     sigma: float = 3.0) -> np.ndarray:
    """Master flat : chaque flat est corrigé (bias + dark-flat), normalisé par sa moyenne,
    puis combiné et renormalisé à 1 (vignettage/poussière préservés)."""
    if not flat_frames:
        raise ValueError("Aucune flat fournie")
    corrected = []
    for f in flat_frames:
        a = np.asarray(f, np.float32).copy()
        if master_bias is not None:
            a -= np.asarray(master_bias, np.float32)
        if master_darkflat is not None:
            a -= np.asarray(master_darkflat, np.float32)
        m = np.median(a[a > 0]) if np.any(a > 0) else 1.0
        if m <= 0:
            m = 1.0
        corrected.append(a / m)
    master = combine_sigma_clip(_stack3d(corrected), sigma=sigma)
    master /= max(np.median(master[master > 0]), 1e-9)
    return np.clip(master, 1e-4, None)


def scale_dark(master_dark: np.ndarray, factor: float) -> np.ndarray:
    """Met le dark à l'échelle entre la durée d'exposition des darks et des lumières."""
    return master_dark * np.float32(max(factor, 0.0))


def _match(master: np.ndarray, light: np.ndarray, name: str) -> np.ndarray:
    m = np.asarray(master, np.float32)
    if m.shape == light.shape:
        return m
    if m.ndim == 2 and light.ndim == 3 and m.shape == light.shape[:2]:
        raise ValueError(
            f"Master {name} mono (CFA) et image lumière couleur : calibrez les "
            "lumières AVANT dématriçage (chargement CFA), ou fournissez des "
            "masters de même type.")
    raise ValueError(f"Taille du master {name} {m.shape} différente de l'image "
                     f"{light.shape} (binning, recadrage ou orientation ?)")


def calibrate(light: np.ndarray, master_bias: np.ndarray | None = None,
              master_dark: np.ndarray | None = None,
              master_flat: np.ndarray | None = None,
              dark_scale: float = 1.0,
              cosmetic: bool = True, cosmetic_sigma: float = 5.0,
              cfa: bool = False) -> np.ndarray:
    """Calibre une image lumière : (L − bias − k·dark) / flat, puis correction
    cosmétique des pixels chauds.
    - master_dark est le dark THERMIQUE (bias déjà retiré par
      make_master_dark lorsqu'un bias lui a été fourni) ; si aucun bias n'est
      donné, le dark contient l'offset et suffit seul ;
    - cfa=True : image brute de capteur couleur (avant dématriçage) — la
      correction cosmétique compare chaque pixel à ses voisins DE MÊME COULEUR."""
    L = np.asarray(light, np.float32).copy()
    if master_bias is not None:
        L -= _match(master_bias, L, "bias")
    if master_dark is not None:
        L -= _match(master_dark, L, "dark") * np.float32(dark_scale)
    if master_flat is not None:
        L /= np.maximum(_match(master_flat, L, "flat"), np.float32(1e-4))
    if cosmetic:
        L = remove_hot_pixels(L, cosmetic_sigma, cfa=cfa)
    return np.clip(L, 0, None).astype(np.float32)


def _hot_plane(img: np.ndarray, sigma: float) -> np.ndarray:
    import scipy.ndimage as ndi
    med = ndi.median_filter(img, size=3, mode="reflect")
    diff = img - med
    std = 1.4826 * np.median(np.abs(diff - np.median(diff))) + 1e-9
    bad = diff > sigma * std
    out = img.copy()
    out[bad] = med[bad]
    return out


def remove_hot_pixels(img: np.ndarray, sigma: float = 5.0, cfa: bool = False) -> np.ndarray:
    """Remplace les pixels isolés (cosmiques/chauds) par la médiane locale 3×3,
    calculée canal par canal (couleur) ou par sous-plan de même couleur du
    motif de Bayer 2×2 (cfa=True)."""
    a = np.asarray(img, np.float32)
    if a.ndim == 3:
        return np.stack([_hot_plane(a[..., c], sigma) for c in range(a.shape[2])], -1)
    if cfa and a.shape[0] >= 4 and a.shape[1] >= 4:
        out = a.copy()
        for i in (0, 1):
            for j in (0, 1):
                out[i::2, j::2] = _hot_plane(np.ascontiguousarray(a[i::2, j::2]), sigma)
        return out
    return _hot_plane(a, sigma)


# compatibilité
_remove_hot_pixels = remove_hot_pixels


# ------------------------------------------------------------- session complète
CALIB_EXTENSIONS = (".fits", ".fit", ".fts", ".fz", ".tif", ".tiff", ".png", ".xisf",
                    ".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2",
                    ".pef", ".srw")


def list_frames(directory: str) -> list[str]:
    """Fichiers image d'un dossier de calibration (triés)."""
    import os
    if not directory or not os.path.isdir(directory):
        return []
    return [os.path.join(directory, f) for f in sorted(os.listdir(directory))
            if f.lower().endswith(CALIB_EXTENSIONS)]


def load_frames_cfa(paths: list[str]) -> list[np.ndarray]:
    """Charge des images de calibration SANS dématriçage."""
    from .image_io import load_cfa
    return [load_cfa(p)[0] for p in paths]


def _group_meta(paths: list[str]) -> dict:
    """Métadonnées médianes d'un lot d'images (en-têtes FITS)."""
    from .session import frame_info
    infos = [frame_info(p) for p in paths[:50]]
    out = {}
    for key in ("exptime", "gain", "offset", "temp"):
        vals = [getattr(i, key) for i in infos if getattr(i, key) is not None]
        out[key] = float(np.median(vals)) if vals else None
    out["binning"] = infos[0].binning if infos else 1
    out["filter"] = infos[0].filter if infos else ""
    out["bayer"] = infos[0].bayer if infos else None
    out["ncombine"] = len(paths)
    return out


def build_masters(bias_dir: str = "", dark_dir: str = "", flat_dir: str = "",
                  darkflat_dir: str = "", sigma: float = 3.0, log=None,
                  paths: dict | None = None, light_meta: dict | None = None,
                  use_library: bool = False, save_to_library: bool = False,
                  library: str | None = None) -> dict:
    """Construit les masters (chaque lot n'est lu qu'une fois, en CFA).
    - `paths` : listes explicites {"bias": [...], "dark": [...], ...} (session
      triée automatiquement), prioritaires sur les dossiers ;
    - `use_library` : un master manquant est cherché dans la bibliothèque
      (masters.find_master) d'après `light_meta` ;
    - `save_to_library` : les masters calculés y sont enregistrés.
    Retourne {"bias", "dark", "darkflat", "flat", "meta": {type: métadonnées},
    "sources": {type: "calculé"|"bibliothèque"}}."""
    from . import masters as LIB
    dirs = {"bias": bias_dir, "dark": dark_dir, "darkflat": darkflat_dir, "flat": flat_dir}
    lists = {k: list((paths or {}).get(k) or list_frames(dirs[k])) for k in dirs}
    out = {"bias": None, "dark": None, "darkflat": None, "flat": None,
           "meta": {}, "sources": {}}
    for kind in ("bias", "dark", "darkflat", "flat"):
        files = lists[kind]
        if files:
            frames = load_frames_cfa(files)
            if kind == "bias":
                out[kind] = make_master_bias(frames, sigma)
            elif kind == "dark":
                out[kind] = make_master_dark(frames, out["bias"], sigma)
            elif kind == "darkflat":
                out[kind] = make_master_darkflat(frames, out["bias"], sigma)
            else:
                out[kind] = make_master_flat(frames, out["bias"], out["darkflat"], sigma)
            del frames
            meta = _group_meta(files)
            meta["thermal"] = kind in ("dark", "darkflat") and out["bias"] is not None
            out["meta"][kind] = meta
            out["sources"][kind] = "calculé"
            if save_to_library:
                LIB.save_master(out[kind], kind, meta, library)
        elif use_library and light_meta:
            entry, _scale = LIB.find_master(kind, light_meta, library)
            if entry is not None:
                if kind == "dark" and entry.get("thermal") and out["bias"] is None:
                    # un dark thermique exige un bias : cherché aussi
                    b, _ = LIB.find_master("bias", light_meta, library)
                    if b is None:
                        continue
                    out["bias"] = LIB.load_master(b, library)
                    out["sources"]["bias"] = "bibliothèque"
                    out["meta"]["bias"] = b
                out[kind] = LIB.load_master(entry, library)
                out["meta"][kind] = entry
                out["sources"][kind] = "bibliothèque"
    if log:
        done = [f"{k} ({out['sources'][k]})" for k in ("bias", "dark", "darkflat", "flat")
                if out[k] is not None]
        log("Masters : " + (", ".join(done) if done else "aucun"))
    return out


def auto_dark_scale(light_path: str, masters: dict) -> float:
    """Facteur d'échelle du dark = EXPTIME lumière / EXPTIME dark, si le dark
    est thermique (bias retiré) ; 1 sinon ou si une exposition est inconnue."""
    meta = (masters or {}).get("meta", {}).get("dark") or {}
    if masters.get("dark") is None or not meta.get("thermal") or not meta.get("exptime"):
        return 1.0
    from .session import frame_info
    exp = frame_info(light_path).exptime
    if not exp:
        return 1.0
    return float(exp / meta["exptime"])


OUTPUTS = ("RGB (dématriçage)", "Ha (duo-bande)", "OIII (duo-bande)")


def calibrate_paths(light_paths: list[str], masters: dict | None = None,
                    bayer: str | None = None, dark_scale="auto",
                    cosmetic: bool = True, defects=None, progress=None,
                    output: str = "RGB") -> list[np.ndarray]:
    """Chaîne de pré-traitement dans le bon ordre, image par image :
    lecture CFA → calibration (bias, dark, flat) → cosmétique et défauts sur
    les pixels de même couleur → dématriçage. `bayer` force le motif ; sinon
    celui du fichier (RAW, BAYERPAT FITS) est utilisé. `defects(img, cfa)`
    est un rappel optionnel de correction des pixels défectueux.
    dark_scale="auto" : rapport des temps de pose (EXPTIME) lumière / dark
    thermique, calculé pour chaque image.
    output : « RGB » (dématriçage) ou, pour un capteur couleur avec filtre
    duo-bande, « Ha » / « OIII » (extraction des pixels du CFA calibré)."""
    from .image_io import debayer, load_cfa
    from .narrowband import extract_ha_oiii
    out_kind = "Ha" if str(output).startswith("Ha") else (
        "OIII" if str(output).startswith("OIII") else "RGB")
    m = masters or {}
    out = []
    n = len(light_paths)
    for i, lp in enumerate(light_paths):
        raw, detected = load_cfa(lp)
        pattern = (bayer or detected) if raw.ndim == 2 else None
        k = auto_dark_scale(lp, m) if dark_scale == "auto" else float(dark_scale)
        cal = calibrate(raw, m.get("bias"), m.get("dark"), m.get("flat"),
                        dark_scale=k, cosmetic=cosmetic,
                        cfa=pattern is not None)
        if defects is not None:
            cal = defects(cal, pattern is not None)
        if pattern is not None:
            if out_kind == "RGB":
                cal = debayer(np.clip(cal, 0, 1), pattern)
            else:
                ha, oiii = extract_ha_oiii(cal, pattern, upscale=True)
                cal = ha if out_kind == "Ha" else oiii
        elif out_kind != "RGB":
            raise ValueError("Extraction Ha/OIII : images de capteur couleur (motif de "
                             "Bayer) requises")
        out.append(np.asarray(cal, np.float32))
        if progress:
            progress(i + 1, n, "calibration")
    return out
