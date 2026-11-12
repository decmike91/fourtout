# -*- coding: utf-8 -*-
"""
Analyse des brutes (esprit SubframeSelector de PixInsight / « PSF » de Siril) :
pour chaque image, nombre d'étoiles, FWHM et excentricité médianes (moments
adaptatifs pondérés par une gaussienne), niveau et bruit du fond, signal des
étoiles, rapport signal/bruit ; détection des images anormales (nuages,
mise au point, suivi, bruit) par écart robuste à la médiane de la session ;
poids d'empilement.

    rows = analyze_paths(paths)
    rows = flag_outliers(rows)
    w = compute_weights(rows, "Mixte")
"""
from __future__ import annotations

import numpy as np

WEIGHT_MODES = ("Aucune", "SNR", "FWHM", "Mixte")


def _mono(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img, np.float32)
    return a if a.ndim == 2 else a[..., :3].mean(axis=2)


def _bin2(a: np.ndarray) -> np.ndarray:
    """Binning 2×2 (supprime le motif de Bayer d'une image CFA)."""
    h, w = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
    return a[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def star_shape(img: np.ndarray, x: float, y: float, bg: float, s0: float = 1.5,
               iters: int = 20) -> tuple[float, float, float, float] | None:
    """Moments adaptatifs elliptiques (méthode HSM : pondération gaussienne de
    même covariance que l'étoile, itérée) : retourne (fwhm, excentricité, x,
    y) ou None. Pour une étoile gaussienne de covariance C, les moments
    pondérés convergent vers C/2, d'où C = 2·M."""
    h, w = img.shape
    C = np.array([[s0 * s0, 0.0], [0.0, s0 * s0]])
    cx, cy = float(x), float(y)
    M = None
    for _ in range(iters):
        ev = np.linalg.eigvalsh(C)
        if ev.min() <= 0 or ev.max() > 900:
            return None
        r = int(max(4, np.ceil(4 * np.sqrt(ev.max()))))
        x0, x1 = int(round(cx)) - r, int(round(cx)) + r + 1
        y0, y1 = int(round(cy)) - r, int(round(cy)) + r + 1
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return None
        win = img[y0:y1, x0:x1] - bg
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
        dx, dy = xx - cx, yy - cy
        Ci = np.linalg.inv(C)
        q = Ci[0, 0] * dx * dx + 2 * Ci[0, 1] * dx * dy + Ci[1, 1] * dy * dy
        wt = np.exp(-0.5 * q) * win
        tot = float(wt.sum())
        if tot <= 0:
            return None
        ncx = float((wt * xx).sum() / tot)
        ncy = float((wt * yy).sum() / tot)
        dx, dy = xx - ncx, yy - ncy
        M = np.array([[(wt * dx * dx).sum(), (wt * dx * dy).sum()],
                      [(wt * dx * dy).sum(), (wt * dy * dy).sum()]]) / tot
        newC = 2.0 * M
        if np.linalg.eigvalsh(newC).min() <= 0:
            return None
        done = np.abs(newC - C).max() < 1e-4 and abs(ncx - cx) < 1e-3 and abs(ncy - cy) < 1e-3
        C = 0.5 * (C + newC)
        cx, cy = ncx, ncy
        if done:
            C = newC
            break
    l2, l1 = np.linalg.eigvalsh(C)
    if l2 <= 0:
        return None
    sigma = np.sqrt((l1 + l2) / 2.0)
    ecc = float(np.sqrt(max(0.0, 1.0 - l2 / l1)))
    return float(2.3548 * sigma), ecc, cx, cy


def analyze_image(img: np.ndarray, cfa: bool = False, max_stars: int = 60) -> dict:
    """Mesures d'une image (mono, couleur ou CFA si cfa=True)."""
    from .astrometry import detect_stars
    a = _mono(img)
    scale = 1.0
    if cfa:
        a = _bin2(a)
        scale = 2.0
    bg = float(np.median(a))
    noise = float(1.4826 * np.median(np.abs(a - bg))) + 1e-9
    stars = detect_stars(a, n_max=max_stars * 2, fwhm=3.0, threshold_sigma=6.0)
    shapes, peaks = [], []
    sat = float(np.percentile(a, 99.99))
    for x, y in stars:
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= yi < a.shape[0] and 0 <= xi < a.shape[1]):
            continue
        pk = float(a[yi, xi])
        if pk >= 0.98 * max(sat, 1e-6) and pk > 0.9:
            continue  # étoile saturée : forme non mesurable
        r = star_shape(a, x, y, bg)
        if r is None or not (0.5 < r[0] < 40):
            continue
        shapes.append(r)
        peaks.append(pk - bg)
        if len(shapes) >= max_stars:
            break
    n = len(stars)
    if shapes:
        fwhm = float(np.median([s[0] for s in shapes])) * scale
        ecc = float(np.median([s[1] for s in shapes]))
        signal = float(np.median(peaks))
    else:
        fwhm, ecc, signal = float("nan"), float("nan"), 0.0
    return {"stars": int(n), "measured": len(shapes), "fwhm": fwhm, "ecc": ecc,
            "background": bg, "noise": noise, "signal": signal,
            "snr": float(signal / noise) if noise > 0 else 0.0}


def analyze_paths(paths: list[str], progress=None, bayer: str | None = None) -> list[dict]:
    """Analyse une liste de fichiers (lecture CFA : un RAW ou un FITS de
    capteur couleur est mesuré après binning 2×2, sans dématriçage)."""
    from .image_io import load_cfa
    rows = []
    n = len(paths)
    for i, p in enumerate(paths):
        try:
            img, pattern = load_cfa(p)
            cfa = img.ndim == 2 and (pattern is not None or bayer is not None)
            r = analyze_image(img, cfa=cfa)
            r["error"] = ""
        except Exception as e:  # fichier illisible : signalé, jamais bloquant
            r = {"stars": 0, "measured": 0, "fwhm": float("nan"), "ecc": float("nan"),
                 "background": float("nan"), "noise": float("nan"), "signal": 0.0,
                 "snr": 0.0, "error": str(e)}
        r["path"] = p
        rows.append(r)
        if progress:
            progress(i + 1, n, "Analyse des brutes")
    return rows


def _robust(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)], np.float64)
    if not len(v):
        return float("nan"), float("nan")
    med = float(np.median(v))
    return med, float(1.4826 * np.median(np.abs(v - med))) + 1e-9


def flag_outliers(rows: list[dict], k: float = 3.0, max_ecc: float = 0.65,
                  min_star_frac: float = 0.5) -> list[dict]:
    """Ajoute à chaque ligne `keep` (bool) et `reasons` (liste) : FWHM ou
    fond anormalement élevés (k écarts robustes), trop peu d'étoiles (nuages,
    buée), étoiles allongées (suivi, vent), fichier illisible."""
    f_med, f_sd = _robust([r["fwhm"] for r in rows])
    b_med, b_sd = _robust([r["background"] for r in rows])
    n_med = float(np.median([r["stars"] for r in rows])) if rows else 0.0
    for r in rows:
        reasons = []
        if r.get("error"):
            reasons.append("illisible")
        else:
            if np.isfinite(r["fwhm"]) and r["fwhm"] > f_med + k * max(f_sd, 0.05 * f_med):
                reasons.append(f"FWHM {r['fwhm']:.2f} px (médiane {f_med:.2f})")
            if not np.isfinite(r["fwhm"]):
                reasons.append("aucune étoile mesurable")
            if np.isfinite(r["background"]) and r["background"] > b_med + k * max(
                    b_sd, 0.02 * abs(b_med)):
                reasons.append("fond élevé (nuage, lune, lumière)")
            if n_med > 0 and r["stars"] < min_star_frac * n_med:
                reasons.append(f"{r['stars']} étoiles (médiane {n_med:.0f})")
            if np.isfinite(r["ecc"]) and r["ecc"] > max_ecc:
                reasons.append(f"étoiles allongées (excentricité {r['ecc']:.2f})")
        r["reasons"] = reasons
        r["keep"] = not reasons
    return rows


def compute_weights(rows: list[dict], mode: str = "Mixte") -> list[float]:
    """Poids d'empilement normalisés (moyenne 1 sur les images gardées) :
    - SNR : (signal/bruit)² — variance inverse, pénalise nuages et bruit ;
    - FWHM : 1/FWHM² — favorise les images les plus fines ;
    - Mixte : (signal/bruit)² / FWHM²."""
    if mode not in WEIGHT_MODES or mode == "Aucune":
        return [1.0 if r.get("keep", True) else 0.0 for r in rows]
    f_med = _robust([r["fwhm"] for r in rows])[0]
    w = []
    for r in rows:
        if not r.get("keep", True):
            w.append(0.0)
            continue
        snr = max(r.get("snr", 0.0), 1e-6)
        fw = r["fwhm"] if np.isfinite(r["fwhm"]) and r["fwhm"] > 0 else f_med
        fw = fw if np.isfinite(fw) and fw > 0 else 1.0
        if mode == "SNR":
            w.append(snr ** 2)
        elif mode == "FWHM":
            w.append(1.0 / fw ** 2)
        else:
            w.append(snr ** 2 / fw ** 2)
    arr = np.asarray(w, np.float64)
    kept = arr[arr > 0]
    if len(kept):
        arr = arr / kept.mean()
    return [float(x) for x in arr]


def report(rows: list[dict]) -> str:
    """Tableau texte des mesures."""
    import os
    lines = [f"{'Fichier':<32} {'Étoiles':>7} {'FWHM':>6} {'Exc.':>5} {'Fond':>7} "
             f"{'Bruit':>7} {'SNR':>7} {'Poids':>6}  Statut"]
    for r in rows:
        lines.append(
            f"{os.path.basename(r['path'])[:32]:<32} {r['stars']:>7d} {r['fwhm']:>6.2f} "
            f"{r['ecc']:>5.2f} {r['background']:>7.4f} {r['noise']:>7.5f} {r['snr']:>7.1f} "
            f"{r.get('weight', 1.0):>6.2f}  "
            + ("garder" if r.get("keep", True) else "rejeter : " + "; ".join(r.get("reasons", []))))
    return "\n".join(lines)
