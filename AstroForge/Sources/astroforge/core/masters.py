# -*- coding: utf-8 -*-
"""
Bibliothèque de masters réutilisables (bias, darks, darks de flats, flats).

Chaque master est enregistré en FITS float32 dans le dossier des données de
l'utilisateur (masters/), avec ses métadonnées dans l'en-tête et dans un
index JSON : type, exposition, gain, offset, température, binning, taille,
filtre (flats), nombre d'images combinées, date. `find_master` retrouve le
master le plus adapté à une image lumière (même gain/binning/taille,
température et exposition les plus proches) ; un dark d'une autre durée est
utilisable avec un bias (dark thermique mis à l'échelle).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import uuid

import numpy as np

from ..paths import DATA_ROOT

LIBRARY_DIR = os.path.join(DATA_ROOT, "masters")
KINDS = ("bias", "dark", "darkflat", "flat")
_META_KEYS = ("kind", "exptime", "gain", "offset", "temp", "binning", "width",
              "height", "filter", "ncombine", "date", "bayer", "thermal", "camera")


def _index_path(lib: str | None = None) -> str:
    return os.path.join(lib or LIBRARY_DIR, "index.json")


def load_index(lib: str | None = None) -> list[dict]:
    p = _index_path(lib)
    if not os.path.exists(p):
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [d for d in data if isinstance(d, dict) and os.path.exists(
            os.path.join(lib or LIBRARY_DIR, d.get("file", "")))]
    except (OSError, ValueError):
        return []


def _save_index(entries: list[dict], lib: str | None = None) -> None:
    os.makedirs(lib or LIBRARY_DIR, exist_ok=True)
    tmp = _index_path(lib) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    os.replace(tmp, _index_path(lib))


def save_master(img: np.ndarray, kind: str, meta: dict | None = None,
                lib: str | None = None) -> dict:
    """Enregistre un master et retourne son entrée d'index."""
    from astropy.io import fits
    if kind not in KINDS:
        raise ValueError(f"Type de master inconnu : {kind}")
    lib = lib or LIBRARY_DIR
    os.makedirs(lib, exist_ok=True)
    a = np.asarray(img, np.float32)
    m = {k: (meta or {}).get(k) for k in _META_KEYS}
    m.update(kind=kind, height=int(a.shape[0]), width=int(a.shape[1]),
             date=m.get("date") or _dt.datetime.now().isoformat(timespec="seconds"))
    name = (f"master_{kind}_{(m.get('exptime') or 0):g}s_g{(m.get('gain') or 0):g}_"
            f"{(m.get('temp') if m.get('temp') is not None else 'NA')}C_"
            f"{uuid.uuid4().hex[:6]}.fits").replace(" ", "")
    hdr = fits.Header()
    hdr["IMAGETYP"] = {"bias": "Master Bias", "dark": "Master Dark",
                       "darkflat": "Master DarkFlat", "flat": "Master Flat"}[kind]
    for key, fk in (("exptime", "EXPTIME"), ("gain", "GAIN"), ("offset", "OFFSET"),
                    ("temp", "CCD-TEMP"), ("binning", "XBINNING"), ("filter", "FILTER"),
                    ("ncombine", "NCOMBINE"), ("bayer", "BAYERPAT")):
        if m.get(key) not in (None, ""):
            hdr[fk] = m[key]
    hdr["THERMAL"] = (bool(m.get("thermal")), "dark without bias offset (thermal only)")
    hdr["CREATOR"] = "AstroForge"
    data = np.moveaxis(a, -1, 0) if a.ndim == 3 else a
    fits.writeto(os.path.join(lib, name), data, hdr, overwrite=True)
    entry = dict(m, file=name)
    entries = load_index(lib) + [entry]
    _save_index(entries, lib)
    return entry


def load_master(entry: dict, lib: str | None = None) -> np.ndarray:
    from astropy.io import fits
    d = np.asarray(fits.getdata(os.path.join(lib or LIBRARY_DIR, entry["file"])), np.float32)
    return np.moveaxis(d, 0, -1) if d.ndim == 3 else d


def delete_master(entry: dict, lib: str | None = None) -> None:
    lib = lib or LIBRARY_DIR
    try:
        os.remove(os.path.join(lib, entry["file"]))
    except OSError:
        pass
    _save_index([e for e in load_index(lib) if e.get("file") != entry.get("file")], lib)


def find_master(kind: str, light: dict, lib: str | None = None, temp_tol: float = 3.0,
                allow_scaling: bool = True) -> tuple[dict | None, float]:
    """Master le plus adapté aux métadonnées `light` (gain, binning, width,
    height, temp, exptime, filter). Retourne (entrée, facteur d'échelle du
    dark) ; (None, 1.0) si aucun master compatible."""
    best, best_score, scale = None, None, 1.0
    for e in load_index(lib):
        if e.get("kind") != kind:
            continue
        if light.get("width") and e.get("width") and (e["width"], e["height"]) != (
                light["width"], light["height"]):
            continue
        if (e.get("binning") or 1) != (light.get("binning") or 1):
            continue
        if light.get("gain") is not None and e.get("gain") is not None and \
                abs(e["gain"] - light["gain"]) >= 0.5:
            continue
        if kind == "flat" and light.get("filter") and e.get("filter") and \
                e["filter"] != light["filter"]:
            continue
        dt = 0.0
        if light.get("temp") is not None and e.get("temp") is not None:
            dt = abs(e["temp"] - light["temp"])
            if kind in ("dark", "bias", "darkflat") and dt > temp_tol:
                continue
        de = 0.0
        if kind == "dark" and light.get("exptime") and e.get("exptime"):
            de = abs(e["exptime"] - light["exptime"]) / light["exptime"]
            if de > 0.05 and not (allow_scaling and e.get("thermal")):
                continue
        age = 0.0
        try:
            age = (_dt.datetime.now() - _dt.datetime.fromisoformat(e["date"])).days / 365.0
        except Exception:
            pass
        score = de * 10 + dt + age * 0.5
        if best_score is None or score < best_score:
            best, best_score = e, score
            scale = (light["exptime"] / e["exptime"]
                     if kind == "dark" and light.get("exptime") and e.get("exptime")
                     and de > 0.05 else 1.0)
    return best, float(scale)
