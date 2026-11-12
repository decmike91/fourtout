# -*- coding: utf-8 -*-
"""
Courbe de lumière (étoile variable, transit d'exoplanète) par photométrie
différentielle sur une série d'images : chaque image est alignée sur la
référence résolue (WCS) puis mesurée ; magnitude différentielle de la cible
contre un ensemble d'étoiles de comparaison, avec instants en jour julien
(DATE-OBS, milieu de pose) et export CSV.
"""
from __future__ import annotations

import csv

import numpy as np


def julian_date(date_obs: str | None, exptime: float | None = None) -> float | None:
    if not date_obs:
        return None
    from astropy.time import Time, TimeDelta
    try:
        t = Time(str(date_obs), format="isot", scale="utc")
    except ValueError:
        return None
    if exptime:
        t = t + TimeDelta(float(exptime) / 2.0, format="sec")
    return float(t.jd)


def light_curve(paths: list[str], wcs, catalog_path: str, target_ra: float,
                target_dec: float, n_comp: int = 5, align: bool = True, ref_img=None,
                r: float = 6.0, r_in: float = 10.0, r_out: float = 16.0,
                progress=None) -> list[dict]:
    """Retourne une liste de mesures {fichier, jd, dmag, dmag_err, n_comp}.
    `wcs` : astrométrie de l'image de référence `ref_img` (sinon la première
    image) ; les autres images sont alignées dessus."""
    from .astrometry import differential_photometry
    from .image_io import load_image
    from .registration import estimate_transform, warp
    from .session import read_header
    ref = np.asarray(ref_img if ref_img is not None else load_image(paths[0]), np.float32)
    ref_m = ref if ref.ndim == 2 else ref.mean(axis=2)
    rows = []
    for i, p in enumerate(paths):
        rec = {"fichier": p, "jd": None, "dmag": float("nan"), "dmag_err": float("nan"),
               "n_comp": 0, "erreur": ""}
        try:
            img = np.asarray(load_image(p), np.float32)
            mono = img if img.ndim == 2 else img.mean(axis=2)
            if align and mono.shape == ref_m.shape:
                mono = warp(mono, estimate_transform(ref_m, mono))
            tab = differential_photometry(mono, wcs, catalog_path, target_ra, target_dec,
                                          n_comp=n_comp, r=r, r_in=r_in, r_out=r_out)
            h = read_header(p)
            rec.update(jd=julian_date(h.get("DATE-OBS"), h.get("EXPTIME") or h.get("EXPOSURE")),
                       dmag=float(tab.meta["dmag"]), dmag_err=float(tab.meta["dmag_err"]),
                       n_comp=int(tab.meta["n_comp"]))
        except Exception as e:
            rec["erreur"] = str(e)
        rows.append(rec)
        if progress:
            progress(i + 1, len(paths), "Photométrie")
    return rows


def save_csv(rows: list[dict], path: str) -> None:
    keys = ("fichier", "jd", "dmag", "dmag_err", "n_comp", "erreur")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(keys)
        for r in rows:
            w.writerow([r.get(k) for k in keys])
