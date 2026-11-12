# -*- coding: utf-8 -*-
"""
Outils solaires (H-alpha, Ca-K, lumière blanche) :

- détection du limbe (cercle ajusté, voir derotation.fit_disc) ;
- inversion du disque (les filaments sombres deviennent clairs, aspect
  « relief ») et rehaussement des protubérances hors du disque, avec des
  réglages séparés et une transition douce au limbe ;
- assombrissement centre-bord compensé (aplatissement du disque) ;
- mosaïque solaire : voir mosaic.mosaic_features (points d'intérêt).
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

from .derotation import fit_disc


def disc_masks(shape, disc, feather: float = 3.0):
    """(intérieur doux 0→1, distance au centre normalisée r/R)."""
    h, w = shape[:2]
    cx, cy, r = disc
    yy, xx = np.mgrid[0:h, 0:w]
    rr = np.hypot(xx - cx, yy - cy)
    inside = np.clip((r - rr) / max(feather, 0.5) + 0.5, 0, 1).astype(np.float32)
    return inside, (rr / max(r, 1e-6)).astype(np.float32)


def flatten_limb_darkening(img, disc, degree: int = 4) -> np.ndarray:
    """Compense l'assombrissement centre-bord : profil radial médian du disque
    ajusté par un polynôme en r/R, l'intérieur est divisé par ce profil."""
    a = np.asarray(img, np.float32)
    mono = a if a.ndim == 2 else a.mean(axis=2)
    inside, rn = disc_masks(mono.shape, disc, 1.0)
    sel = (inside > 0.99) & (rn < 0.98)
    if sel.sum() < 100:
        return a
    bins = np.linspace(0, 0.98, 40)
    idx = np.digitize(rn[sel], bins)
    vals = mono[sel]
    xs, ys = [], []
    for b in range(1, len(bins)):
        v = vals[idx == b]
        if len(v) > 20:
            xs.append(0.5 * (bins[b - 1] + bins[b]))
            ys.append(np.median(v))
    if len(xs) < degree + 2:
        return a
    c = np.polyfit(xs, ys, degree)
    prof = np.polyval(c, np.clip(rn, 0, 0.999))
    prof = np.maximum(prof, 1e-4) / max(float(np.polyval(c, 0.0)), 1e-4)
    k = np.where(inside > 0, 1.0 / prof, 1.0)
    k = 1 + (k - 1) * inside
    out = a * (k[..., None] if a.ndim == 3 else k)
    return np.clip(out, 0, 1).astype(np.float32)


def prominence_composite(img, disc=None, invert_disc: bool = True, disc_gamma: float = 1.0,
                         prom_gain: float = 4.0, prom_gamma: float = 0.7,
                         feather: float = 3.0, flatten: bool = True) -> tuple[np.ndarray, tuple]:
    """Composite solaire : disque (inversé ou non, gamma) + protubérances
    (gain et gamma propres), raccord progressif au limbe. Retourne
    (image, disque)."""
    a = np.asarray(img, np.float32)
    mono = a if a.ndim == 2 else a.mean(axis=2)
    disc = disc or fit_disc(mono)
    if flatten:
        mono = flatten_limb_darkening(mono, disc)
    inside, rn = disc_masks(mono.shape, disc, feather)
    d = mono.copy()
    vals = d[inside > 0.99]
    lo, hi = (np.percentile(vals, 0.5), np.percentile(vals, 99.5)) if vals.size else (0, 1)
    d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    if invert_disc:
        d = 1.0 - d
    d = np.power(d, 1.0 / max(disc_gamma, 1e-3))
    out_reg = (inside < 0.01) & (rn < 2.0)
    bg = float(np.median(mono[(rn > 1.15) & (rn < 2.0)])) if ((rn > 1.15) & (rn < 2.0)).any() else 0.0
    p = np.clip((mono - bg) * prom_gain, 0, 1)
    p = np.power(p, max(prom_gamma, 1e-3))
    p = ndi.gaussian_filter(p, 0.5) if out_reg.any() else p
    # niveau des protubérances raccordé au bord du disque
    comp = d * inside + p * (1 - inside)
    return np.clip(comp, 0, 1).astype(np.float32), disc
