# -*- coding: utf-8 -*-
"""
Masques d'application (0 = protégé, 1 = traité) et outils d'étoiles :

- masque de luminance (plage [bas, haut] avec transition douce), masque de
  plage, masque d'étoiles (détection multi-échelle + dilatation + flou),
  combinaisons (inversion) ;
- application d'un traitement à travers un masque : out = img·(1−m) + f·m ;
- réduction d'étoiles (érosion morphologique douce limitée au masque
  d'étoiles), extraction des étoiles (image − image sans étoiles) et
  recombinaison image sans étoiles + étoiles (addition ou écran) ;
- débruitage de la chrominance seule (luminance intacte), multi-échelle.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

MASK_TYPES = ("Aucun", "Luminance", "Luminance inversée (fond)", "Étoiles",
              "Sans étoiles (inverse)", "Plage")


def _lum(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, np.float32)
    if a.ndim == 2:
        return a
    return (0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]).astype(np.float32)


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def luminance_mask(img, low: float = 0.0, high: float = 1.0, fuzz: float = 0.1,
                   blur: float = 2.0) -> np.ndarray:
    """Masque de luminance : 1 dans [low, high], transition douce de largeur
    `fuzz` de part et d'autre, flou gaussien `blur` px."""
    L = _lum(img)
    f = max(float(fuzz), 1e-4)
    lo = np.ones_like(L) if low <= 0 else _smoothstep((L - (low - f)) / f)
    hi = np.ones_like(L) if high >= 1 else _smoothstep(((high + f) - L) / f)
    m = lo * hi
    if blur > 0:
        m = ndi.gaussian_filter(m, blur)
    return np.clip(m, 0, 1).astype(np.float32)


def star_mask(img, threshold: float = 4.0, grow: int = 2, blur: float = 1.5,
              max_size: int = 40) -> np.ndarray:
    """Masque d'étoiles : résidu « chapeau haut-de-forme » (image − ouverture
    morphologique) au-dessus de `threshold` σ, composantes compactes
    seulement (≤ max_size px de côté : pas les nébuleuses), dilatées de
    `grow` px puis adoucies."""
    L = _lum(img)
    bg = ndi.grey_opening(L, size=(9, 9))
    th = L - bg
    sig = 1.4826 * float(np.median(np.abs(th - np.median(th)))) + 1e-6
    core = th > threshold * sig
    lab, n = ndi.label(core)
    if n:
        sl = ndi.find_objects(lab)
        keep = np.zeros(n + 1, bool)
        for i, s in enumerate(sl, start=1):
            if s is not None and (s[0].stop - s[0].start) <= max_size and \
                    (s[1].stop - s[1].start) <= max_size:
                keep[i] = True
        core = keep[lab]
    if grow > 0:
        core = ndi.binary_dilation(core, iterations=int(grow))
    m = core.astype(np.float32)
    if blur > 0:
        m = ndi.gaussian_filter(m, blur)
        m = np.clip(m * 1.5, 0, 1)
    return m


def make_mask(img, kind: str = "Aucun", low: float = 0.25, high: float = 1.0,
              fuzz: float = 0.1, blur: float = 2.0) -> np.ndarray | None:
    """Masque nommé (MASK_TYPES) ou None pour « Aucun »."""
    if not kind or kind == "Aucun":
        return None
    if kind == "Luminance":
        return luminance_mask(img, low, high, fuzz, blur)
    if kind.startswith("Luminance inversée"):
        return 1.0 - luminance_mask(img, low, high, fuzz, blur)
    if kind == "Étoiles":
        return star_mask(img, blur=max(blur, 0.5))
    if kind.startswith("Sans étoiles"):
        return 1.0 - star_mask(img, blur=max(blur, 0.5))
    if kind == "Plage":
        L = _lum(img)
        m = ((L >= low) & (L <= high)).astype(np.float32)
        return np.clip(ndi.gaussian_filter(m, blur), 0, 1) if blur > 0 else m
    raise ValueError(f"Masque inconnu : {kind}")


def apply_masked(original, processed, mask) -> np.ndarray:
    """Mélange : 0 → original, 1 → traité."""
    a = np.asarray(original, np.float32)
    b = np.asarray(processed, np.float32)
    if mask is None:
        return b
    m = np.asarray(mask, np.float32)
    if b.shape[:2] != m.shape:
        raise ValueError("Le masque et l'image n'ont pas la même taille")
    if b.ndim == 3:
        m = m[..., None]
    if a.shape != b.shape:  # ex. traitement qui change le nombre de canaux
        return b
    return (a * (1 - m) + b * m).astype(np.float32)


# ------------------------------------------------------------------ étoiles
def reduce_stars(img, amount: float = 0.0, iterations: int = 1,
                 threshold: float = 4.0) -> np.ndarray:
    """Réduction d'étoiles : érosion morphologique (minimum 3×3 répété)
    appliquée seulement dans le masque d'étoiles, mélangée selon `amount`
    (0 = identité, 1 = érosion complète). La nébulosité est préservée."""
    a = np.asarray(img, np.float32)
    k = float(np.clip(amount, 0, 1))
    if k <= 0:
        return np.clip(a, 0, 1).astype(np.float32)
    m = star_mask(a, threshold=threshold, grow=1, blur=1.0) * k
    er = a.copy()
    for _ in range(max(1, int(iterations))):
        if er.ndim == 3:
            er = np.stack([ndi.grey_erosion(er[..., c], size=(3, 3))
                           for c in range(er.shape[2])], -1)
        else:
            er = ndi.grey_erosion(er, size=(3, 3))
    # l'érosion assombrit aussi le fond local : on garde le max(fond, érodé)
    bgk = ndi.median_filter(_lum(a), size=9)
    if a.ndim == 3:
        er = np.maximum(er, (a / np.maximum(_lum(a)[..., None], 1e-6))
                        * np.minimum(bgk, _lum(a))[..., None])
    else:
        er = np.maximum(er, np.minimum(bgk, a))
    return np.clip(apply_masked(a, er, m), 0, 1).astype(np.float32)


def extract_stars(img, starless) -> np.ndarray:
    """Étoiles seules = image − image sans étoiles (≥ 0)."""
    return np.clip(np.asarray(img, np.float32) - np.asarray(starless, np.float32), 0, 1)


def recombine(starless, stars, mode: str = "Écran", stars_gain: float = 1.0) -> np.ndarray:
    """Recombinaison : « Addition » (linéaire) ou « Écran »
    (1 − (1−a)(1−b), sans écrêtage brutal) ; gain des étoiles réglable."""
    a = np.asarray(starless, np.float32)
    b = np.clip(np.asarray(stars, np.float32) * float(stars_gain), 0, 1)
    if a.shape != b.shape:
        raise ValueError("Image sans étoiles et étoiles de tailles différentes")
    if str(mode).startswith("Add"):
        return np.clip(a + b, 0, 1).astype(np.float32)
    return (1.0 - (1.0 - np.clip(a, 0, 1)) * (1.0 - b)).astype(np.float32)


def chroma_denoise(img, strength: float = 0.0, scales: int = 4) -> np.ndarray:
    """Débruitage de la CHROMINANCE seule (luminance intacte) : passage en
    luminance + deux différences de couleur, seuillage doux des couches
    d'ondelettes à trous des composantes de couleur, retour en RGB."""
    a = np.asarray(img, np.float32)
    s = float(np.clip(strength, 0, 1))
    if a.ndim != 3 or s <= 0:
        return np.clip(a, 0, 1).astype(np.float32)
    from .wavelets import atrous_decompose, atrous_recompose
    L = _lum(a)
    c1 = a[..., 0] - L
    c2 = a[..., 2] - L
    out_c = []
    for c in (c1, c2):
        layers = atrous_decompose(c, int(scales))
        new = []
        for i, layer in enumerate(layers[:-1]):
            sig = np.median(np.abs(layer)) / 0.6745 + 1e-9
            t = sig * (3.0 * s) / (1 + 0.5 * i)
            new.append(np.sign(layer) * np.maximum(np.abs(layer) - t, 0))
        new.append(layers[-1])
        out_c.append(atrous_recompose(new))
    r = L + out_c[0]
    b = L + out_c[1]
    g = (L - 0.2126 * r - 0.0722 * b) / 0.7152
    return np.clip(np.stack([r, g, b], -1), 0, 1).astype(np.float32)
