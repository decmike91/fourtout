# -*- coding: utf-8 -*-
"""
Calibration couleur & neutralisation du fond (style Siril) :
- calibration couleur sur le fond du ciel ou sur une référence blanche ;
- neutralisation du fond (le ciel devient gris neutre) ;
- SCNR vert (suppression de la dominante verte des caméras OSC).
"""
from __future__ import annotations
import numpy as np


def _lum(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else img[..., :3].mean(axis=2)


def background_mask(img: np.ndarray, percentile: float = 25.0) -> np.ndarray:
    """Masque du fond : pixels les plus sombres (sous le percentile donné)."""
    lum = _lum(img)
    thr = np.percentile(lum, float(np.clip(percentile, 1, 99)))
    return lum <= thr


def color_calibrate(img: np.ndarray, white: tuple[float, float, float] = (1.0, 1.0, 1.0),
                    use_background: bool = True, percentile: float = 25.0) -> np.ndarray:
    """Calibration couleur : égalise les médianes des canaux, soit sur le fond
    du ciel (use_background), soit globalement, vers les ratios `white` (R,G,B)."""
    a = np.asarray(img, np.float32)
    if a.ndim != 3 or a.shape[2] < 3:
        return a
    out = a.copy()
    if use_background:
        mask = background_mask(a, percentile)
        if mask.sum() < 100:
            mask = np.ones(a.shape[:2], bool)
        m = [float(np.median(a[..., c][mask])) for c in range(3)]
    else:
        m = [float(np.median(a[..., c])) for c in range(3)]
    ref = max(m[1], 1e-9)  # le vert sert de référence
    for c in range(3):
        target = np.float32(white[c] * ref)
        if m[c] > 1e-9:
            out[..., c] = a[..., c] * (target / np.float32(m[c]))
    return np.clip(out, 0, 1).astype(np.float32)


def background_neutralize(img: np.ndarray, percentile: float = 25.0) -> np.ndarray:
    """Neutralisation du fond : soustrait les médianes résiduelles du fond
    pour que le ciel soit strictement neutre (R = G = B sur le fond)."""
    a = np.asarray(img, np.float32)
    if a.ndim != 3 or a.shape[2] < 3:
        return a
    mask = background_mask(a, percentile)
    if mask.sum() < 100:
        mask = np.ones(a.shape[:2], bool)
    out = a.copy()
    m = [float(np.median(a[..., c][mask])) for c in range(3)]
    g = m[1]
    for c in range(3):
        out[..., c] = np.maximum(a[..., c] - (m[c] - g), 0)
    return out.astype(np.float32)


def scnr_green(img: np.ndarray) -> np.ndarray:
    """SCNR « moyenne neutre » : le vert au-dessus de la moyenne (R+B)/2 est ramené
    à cette moyenne (supprime le bruit vert des caméras couleur)."""
    a = np.asarray(img, np.float32)
    if a.ndim != 3 or a.shape[2] < 3:
        return a
    out = a.copy()
    rb = (a[..., 0] + a[..., 2]) * 0.5
    out[..., 1] = np.minimum(a[..., 1], rb)
    return np.clip(out, 0, 1).astype(np.float32)


def correct_chromatic_aberration(img: np.ndarray, k_r: float = 0.02,
                                 k_b: float = 0.02,
                                 center: tuple[float, float] | None = None) -> np.ndarray:
    """Correction d'aberration chromatique latérale : R et B sont remis à
    l'échelle radialement autour du centre (k positif = rétrécit le canal,
    corrige une image « trop grande » sur ce canal). k ∈ [-0.1 .. 0.1]."""
    a = np.asarray(img, np.float32)
    if a.ndim != 3 or a.shape[2] < 3:
        return a
    cv2 = __import__("cv2")
    h, w = a.shape[:2]
    cx, cy = center or ((w - 1) / 2.0, (h - 1) / 2.0)
    out = a.copy()
    for c, k in ((0, k_r), (2, k_b)):
        k = float(np.clip(k, -0.1, 0.1))
        if abs(k) < 1e-4:
            continue
        s = 1.0 - k
        M = np.array([[s, 0.0, cx * (1 - s)], [0.0, s, cy * (1 - s)]], np.float64)
        out[..., c] = cv2.warpAffine(a[..., c], M, (w, h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REFLECT)
    return out.astype(np.float32)


def photometric_color_calibration(img: np.ndarray, wcs, catalog_path: str,
                                  r: float = 6.0, r_in: float = 10.0,
                                  r_out: float = 16.0, n_max: int = 200,
                                  neutralize: bool = True) -> tuple[np.ndarray, dict]:
    """Calibration photométrique automatique des couleurs (style PCC de Siril) :
    les étoiles détectées sont appariées au catalogue (via le WCS) ; les flux
    mesurés par canal (R, G, B) sont comparés aux couleurs du catalogue
    (colonnes BP/RP de Gaia) et les canaux sont corrigés pour que les couleurs
    mesurées retrouvent les couleurs stellaires du catalogue."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    from .astrometry import detect_stars, load_catalog
    a = np.asarray(img, np.float32)
    if a.ndim != 3 or a.shape[2] < 3:
        raise ValueError("PCC : une image couleur (RGB) est requise")
    h, w = a.shape[:2]
    stars = detect_stars(a.mean(axis=2), n_max)
    if len(stars) < 5:
        raise RuntimeError("PCC : moins de 5 étoiles détectées")
    tab = load_catalog(catalog_path)
    cols = tab.colnames
    if not all(c in cols for c in ("bp", "rp")):
        raise RuntimeError("Catalogue sans couleurs BP/RP — régénérez le "
                           "catalogue Gaia (onglet Configuration)")
    ra = np.asarray(tab["ra"], float)
    dec = np.asarray(tab["dec"], float)
    mag = np.asarray(tab["mag"], float)
    bp = np.asarray(tab["bp"], float)
    rp = np.asarray(tab["rp"], float)
    px = wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
    d = np.hypot(px[0][None, :] - stars[:, 0][:, None],
                 px[1][None, :] - stars[:, 1][:, None])
    best = np.argmin(d, axis=1)
    ok = d[np.arange(len(stars)), best] < 4.0
    if ok.sum() < 5:
        raise RuntimeError("PCC : moins de 5 étoiles appariées au catalogue — "
                           "vérifiez le plate solving et le catalogue")
    ratios_b, ratios_r = [], []
    R = int(np.ceil(r_out)) + 1
    for ic, ii in zip(best[ok], np.nonzero(ok)[0], strict=True):
        sx, sy = stars[ii]
        x0, x1 = int(sx) - R, int(sx) + R + 1
        y0, y1 = int(sy) - R, int(sy) + R + 1
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            continue  # étoile trop près du bord : couronne incomplète
        win = a[y0:y1, x0:x1]
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dd = (xx - sx) ** 2 + (yy - sy) ** 2
        ap = dd <= r * r
        an = (dd > r_in * r_in) & (dd <= r_out * r_out)
        if ap.sum() == 0 or an.sum() < 8:
            continue
        vals = win[ap][:, :3]
        peak = vals.max(axis=0)
        if bool(((vals >= 0.995 * peak) & (peak > 0)).sum(axis=0).max() >= 3):
            continue  # sommet plat = étoile saturée : couleur non mesurable
        # fond soustrait PAR CANAL (le fond de ciel est souvent coloré)
        bg = np.median(win[an], axis=0)
        flux = win[ap].sum(axis=0) - bg * float(ap.sum())
        fr, fg, fb = (float(v) for v in flux[:3])
        if fg <= 0 or fb <= 0 or fr <= 0:
            continue
        if not (np.isfinite(bp[ic]) and np.isfinite(rp[ic]) and np.isfinite(mag[ic])):
            continue
        # attendu : flux_c/flux_G = 10^(-0.4·(mag_c − mag_G))
        eb = 10 ** (-0.4 * (bp[ic] - mag[ic]))
        er = 10 ** (-0.4 * (rp[ic] - mag[ic]))
        ratios_b.append(eb / (fb / fg))
        ratios_r.append(er / (fr / fg))
    if len(ratios_b) < 5:
        raise RuntimeError("PCC : pas assez d'étoiles mesurables (saturation ?)")
    kb = float(np.median(ratios_b))
    kr = float(np.median(ratios_r))
    out = a.copy()
    out[..., 0] *= np.float32(kr)
    out[..., 2] *= np.float32(kb)
    if neutralize:
        out = background_neutralize(out)
    return np.clip(out, 0, 1).astype(np.float32), \
        {"n": len(ratios_b), "kR": kr, "kB": kb}
