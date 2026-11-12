# -*- coding: utf-8 -*-
"""
Annotation des objets du ciel profond (Messier, NGC, IC — sous-ensemble du
catalogue OpenNGC, CC BY-SA 4.0, voir data/OPENNGC_LICENSE.txt) sur une
image résolue (WCS) : liste des objets du champ et image annotée (ellipses
à la taille et à l'orientation des objets, étiquettes).
"""
from __future__ import annotations

import csv
import os

import numpy as np

CATALOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "openngc_lite.csv")
TYPES = {"G": "galaxie", "GPair": "paire de galaxies", "GTrpl": "triplet de galaxies",
         "GGroup": "groupe de galaxies", "OCl": "amas ouvert", "GCl": "amas globulaire",
         "Cl+N": "amas + nébuleuse", "PN": "nébuleuse planétaire", "HII": "région HII",
         "Neb": "nébuleuse", "EmN": "nébuleuse en émission", "RfN": "nébuleuse par réflexion",
         "SNR": "reste de supernova", "*Ass": "association stellaire", "DrkN": "nébuleuse obscure"}


def load_objects(path: str | None = None) -> list[dict]:
    out = []
    with open(path or CATALOG, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for k in ("ra", "dec", "maj_arcmin", "min_arcmin", "pa", "mag"):
                r[k] = float(r[k]) if r.get(k) not in (None, "") else None
            out.append(r)
    return out


def objects_in_field(wcs, shape, mag_limit: float = 14.0, path: str | None = None,
                     margin: float = 0.0) -> list[dict]:
    """Objets dont le centre tombe dans l'image (positions en pixels)."""
    h, w = shape[:2]
    objs = load_objects(path)
    ra = np.array([o["ra"] for o in objs])
    dec = np.array([o["dec"] for o in objs])
    try:
        px, py = wcs.all_world2pix(ra, dec, 0)
    except Exception:
        px, py = wcs.wcs_world2pix(ra, dec, 0)
    # objets du côté visible de la projection uniquement
    c = wcs.pixel_to_world(w / 2, h / 2)
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    sep = SkyCoord(ra * u.deg, dec * u.deg).separation(c).deg
    scale = float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))))  # deg/px
    found = []
    for o, x, y, s in zip(objs, px, py, sep):
        if s > 90 or not np.isfinite(x) or not np.isfinite(y):
            continue
        if -margin <= x < w + margin and -margin <= y < h + margin:
            m = o["mag"]
            if not o["label"].startswith("M") and m is not None and m > mag_limit:
                continue
            d = dict(o, x=float(x), y=float(y))
            d["maj_px"] = (o["maj_arcmin"] or 0) / 60.0 / scale
            d["min_px"] = (o["min_arcmin"] or o["maj_arcmin"] or 0) / 60.0 / scale
            d["type_fr"] = TYPES.get(o["type"], o["type"])
            found.append(d)
    found.sort(key=lambda d: (not d["label"].startswith("M"), d["mag"] if d["mag"] else 99))
    return found


def _north_angle(wcs, x, y) -> float:
    """Angle (degrés image) de la direction du nord céleste au point (x, y)."""
    c = wcs.pixel_to_world(x, y)
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    n = SkyCoord(c.ra, c.dec + 1 * u.arcmin)
    xn, yn = wcs.world_to_pixel(n)
    return float(np.degrees(np.arctan2(yn - y, xn - x)))


def annotate(img, wcs, mag_limit: float = 14.0, color=(1.0, 0.85, 0.2),
             font_scale: float | None = None, path: str | None = None):
    """Image RGB annotée + liste des objets."""
    import cv2
    a = np.asarray(img, np.float32)
    rgb = np.repeat(a[..., None], 3, axis=2) if a.ndim == 2 else a[..., :3].copy()
    h, w = rgb.shape[:2]
    objs = objects_in_field(wcs, (h, w), mag_limit, path)
    u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8).copy()
    col = tuple(int(c * 255) for c in color)
    fs = font_scale or max(0.4, min(h, w) / 1400.0)
    th = max(1, int(round(fs * 2)))
    for o in objs:
        x, y = int(round(o["x"])), int(round(o["y"]))
        a_ax = max(6, int(round(o["maj_px"] / 2)))
        b_ax = max(6, int(round(o["min_px"] / 2)))
        ang = _north_angle(wcs, o["x"], o["y"]) + 90.0 - (o["pa"] or 0.0)
        cv2.ellipse(u8, (x, y), (a_ax, b_ax), ang, 0, 360, col, th, cv2.LINE_AA)
        label = o["label"] + (f" ({o['common']})" if o.get("common") and o["label"].startswith("M") else "")
        label = label.encode("ascii", "replace").decode()
        cv2.putText(u8, label, (x + a_ax + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, col, th,
                    cv2.LINE_AA)
    return u8.astype(np.float32) / 255.0, objs
