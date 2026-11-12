# -*- coding: utf-8 -*-
"""
Pixels/colonnes/lignes défectueux : détection (master dark/flat ou carte
fournie) et correction par interpolation locale. Une « carte de pixels morts »
est une image (FITS/PNG : pixels > 0,5 = défectueux) ou un texte (une ligne
« x,y » par pixel).
"""
from __future__ import annotations
import os

import numpy as np


# ------------------------------------------------------------------ cartes
def load_bad_pixel_map(path: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Charge une carte de défauts -> masque booléen (H, W)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".csv", ".dat"):
        pts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.replace(",", " ").replace(";", " ").split()
                if len(parts) >= 2:
                    try:
                        pts.append((int(float(parts[0])), int(float(parts[1]))))
                    except ValueError:
                        pass
        if not pts:
            raise ValueError(f"Carte de pixels vide : {path}")
        mask = np.zeros(shape if shape else
                        (max(y for _, y in pts) + 2, max(x for x, _ in pts) + 2), bool)
        for x, y in pts:
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                mask[y, x] = True
    else:
        from .image_io import load_image
        m = np.asarray(load_image(path), np.float32)
        mask = m > 0.5
        if mask.ndim == 3:
            mask = mask.any(axis=-1)
        mask = np.asarray(mask, bool)
    if shape is not None and mask.shape != tuple(shape):
        out = np.zeros(shape, bool)
        h = min(shape[0], mask.shape[0])
        w = min(shape[1], mask.shape[1])
        out[:h, :w] = mask[:h, :w]
        mask = out
    return mask


def detect_bad_pixels(reference: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Détection auto sur un master dark (chauds+froids) ou flat : pixels
    s'écartant de plus de `sigma` MAD de la médiane globale."""
    a = np.asarray(reference, np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    med = float(np.median(a))
    mad = 1.4826 * float(np.median(np.abs(a - med))) + 1e-9
    hot = a > med + sigma * mad
    cold = a < med - sigma * mad if med > 1e-6 else np.zeros_like(hot)
    return hot | cold


def detect_bad_columns(reference: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Colonnes anormalement sombres/claires (profils médians par colonne)."""
    a = np.asarray(reference, np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    prof = np.median(a, axis=0)
    med = np.median(prof)
    mad = 1.4826 * np.median(np.abs(prof - med)) + 1e-9
    return np.abs(prof - med) > sigma * mad


def detect_bad_rows(reference: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    a = np.asarray(reference, np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    prof = np.median(a, axis=1)
    med = np.median(prof)
    mad = 1.4826 * np.median(np.abs(prof - med)) + 1e-9
    return np.abs(prof - med) > sigma * mad


# ------------------------------------------------------------------ correction
def fix_defects(img: np.ndarray, bad: np.ndarray | None = None,
                cols: np.ndarray | None = None, rows: np.ndarray | None = None,
                passes: int = 2, cfa: bool = False) -> np.ndarray:
    """Corrige l'image : colonnes/lignes mortes par moyennage des voisines,
    pixels défectueux par médiane des voisins (`passes` pour les amas).
    Mono (H,W) et couleur (H,W,3) ; masque `bad` mono (H,W).
    cfa=True (image de capteur couleur avant dématriçage) : les voisins
    utilisés sont ceux de MÊME COULEUR (pas de 2 pixels)."""
    a = np.asarray(img, np.float32).copy()
    was2d = a.ndim == 2
    if was2d:
        a = a[..., None]
    h, w, ch = a.shape
    step = 2 if (cfa and was2d) else 1
    if cols is not None:
        for c in np.nonzero(np.asarray(cols, bool))[0]:
            c = int(c)
            left, right = c - step, c + step
            if left < 0:
                left = right
            if right >= w:
                right = left
            if 0 <= left < w and 0 <= right < w:
                a[:, c] = (a[:, left] + a[:, right]) * 0.5
    if rows is not None:
        for r in np.nonzero(np.asarray(rows, bool))[0]:
            r = int(r)
            up, down = r - step, r + step
            if up < 0:
                up = down
            if down >= h:
                down = up
            if 0 <= up < h and 0 <= down < h:
                a[r] = (a[up] + a[down]) * 0.5
    if bad is not None and np.any(bad):
        import scipy.ndimage as ndi
        m2 = np.asarray(bad, bool)
        if step == 2:
            plane = a[..., 0]
            for i in (0, 1):
                for j in (0, 1):
                    sub = np.ascontiguousarray(plane[i::2, j::2])
                    msub = m2[i::2, j::2]
                    if not msub.any():
                        continue
                    for _ in range(max(1, passes)):
                        med = ndi.median_filter(sub, size=3)
                        sub = np.where(msub, med, sub)
                    plane[i::2, j::2] = sub
        else:
            for _ in range(max(1, passes)):
                med = np.stack([ndi.median_filter(a[..., c], size=5)
                                for c in range(ch)], axis=-1)
                a = np.where(np.broadcast_to(m2[..., None], a.shape), med, a)
    return (a[..., 0] if was2d else a).astype(np.float32)


def auto_correct(img: np.ndarray, master_dark: np.ndarray | None = None,
                 master_flat: np.ndarray | None = None,
                 bpm_path: str | None = None, sigma: float = 5.0,
                 fix_cols_rows: bool = True) -> tuple[np.ndarray, dict]:
    """Tout-en-un : carte depuis bpm_path sinon détection auto sur dark/flat,
    colonnes/lignes depuis la même référence. Retourne (image corrigée, info)."""
    ref = master_dark if master_dark is not None else master_flat
    shape = np.asarray(img).shape[:2]
    bad = load_bad_pixel_map(bpm_path, shape) if bpm_path else None
    cols = rows = None
    if bad is None and ref is not None:
        bad = detect_bad_pixels(ref, sigma)
        if fix_cols_rows:
            cols = detect_bad_columns(ref, sigma)
            rows = detect_bad_rows(ref, sigma)
    out = fix_defects(img, bad=bad, cols=cols, rows=rows)
    info = {"bad_pixels": int(bad.sum()) if bad is not None else 0,
            "bad_columns": int(np.sum(cols)) if cols is not None else 0,
            "bad_rows": int(np.sum(rows)) if rows is not None else 0}
    return out, info
