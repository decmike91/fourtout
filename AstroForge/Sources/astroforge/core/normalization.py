# -*- coding: utf-8 -*-
"""
Normalisation des images avant empilement — indispensable pour combiner
plusieurs nuits (fond de ciel, transparence et gradients différents) :

- globale : niveau (médiane) et échelle (MAD) ramenés sur la référence ;
- locale (esprit LocalNormalization de PixInsight) : décalage et échelle
  estimés par tuiles sur le fond (étoiles rejetées), lissés puis interpolés
  en cartes continues : les gradients propres à chaque nuit disparaissent
  avant le rejet des pixels.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

NORMALIZATIONS = ("Aucune", "Globale", "Locale")


def _robust(a: np.ndarray):
    med = float(np.median(a))
    return med, 1.4826 * float(np.median(np.abs(a - med))) + 1e-9


def global_normalize(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """img·s + o tel que médiane et MAD de img égalent celles de ref
    (par canal)."""
    a = np.asarray(img, np.float32)
    r = np.asarray(ref, np.float32)
    if a.ndim == 3:
        return np.stack([global_normalize(a[..., c], r[..., c]) for c in range(a.shape[2])], -1)
    ma, sa = _robust(a)
    mr, sr = _robust(r)
    return ((a - ma) * np.float32(sr / sa) + np.float32(mr)).astype(np.float32)


def _tile_stats(a: np.ndarray, tile: int, sigma: float = 2.5):
    h, w = a.shape
    ny, nx = max(1, h // tile), max(1, w // tile)
    med = np.full((ny, nx), np.nan)
    sd = np.full((ny, nx), np.nan)
    ys = np.linspace(0, h, ny + 1).astype(int)
    xs = np.linspace(0, w, nx + 1).astype(int)
    for i in range(ny):
        for j in range(nx):
            blk = a[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].ravel()
            m, s = _robust(blk)
            ok = np.abs(blk - m) <= sigma * s
            if ok.sum() > 0.3 * blk.size:
                med[i, j], sd[i, j] = _robust(blk[ok])
    return med, sd


def _fill_smooth(grid: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2
    g = grid.copy()
    if np.isnan(g).all():
        g[:] = 0
    elif np.isnan(g).any():
        idx = ndi.distance_transform_edt(np.isnan(g), return_distances=False, return_indices=True)
        g = g[tuple(idx)]
    if min(g.shape) >= 3:
        g = ndi.median_filter(g, size=3, mode="nearest")
    return cv2.resize(g.astype(np.float32), (size[1], size[0]), interpolation=cv2.INTER_CUBIC)


def background_map(a: np.ndarray, tile: int = 128) -> np.ndarray:
    """Fond lissé : médianes par tuiles (étoiles rejetées), interpolées."""
    med, _sd = _tile_stats(a, tile)
    return _fill_smooth(med, a.shape)


def signal_scale(det: np.ndarray, det_ref: np.ndarray) -> float:
    """Rapport d'intensité du signal (étoiles, nébuleuse) entre deux images
    dont le fond a été retiré : médiane des rapports sur les pixels nettement
    au-dessus du bruit dans les deux images ; repli sur le rapport des MAD."""
    n_ref = 1.4826 * float(np.median(np.abs(det_ref - np.median(det_ref)))) + 1e-9
    n_det = 1.4826 * float(np.median(np.abs(det - np.median(det)))) + 1e-9
    sel = (det_ref > 8 * n_ref) & (det > 8 * n_det)
    if sel.sum() >= 50:
        return float(np.clip(np.median(det_ref[sel] / det[sel]), 0.05, 20.0))
    return float(np.clip(n_ref / n_det, 0.05, 20.0))


def local_normalize(img: np.ndarray, ref: np.ndarray, tile: int = 128) -> np.ndarray:
    """Normalisation locale : fond de chaque image estimé par tuiles (étoiles
    rejetées) et remplacé par celui de la référence ; intensité du signal
    (transparence) recalée par un facteur unique mesuré sur les pixels de
    signal : out = s·(img − fond_img) + fond_ref."""
    a = np.asarray(img, np.float32)
    r = np.asarray(ref, np.float32)
    if a.shape != r.shape:
        raise ValueError("Normalisation locale : images de même taille requises")
    if a.ndim == 3:
        return np.stack([local_normalize(a[..., c], r[..., c], tile)
                         for c in range(a.shape[2])], -1)
    tile = int(max(16, min(tile, min(a.shape) // 2)))
    bg_a = background_map(a, tile)
    bg_r = background_map(r, tile)
    k = signal_scale(a - bg_a, r - bg_r)
    return (k * (a - bg_a) + bg_r).astype(np.float32)


def normalize_frames(frames: list[np.ndarray], ref: np.ndarray, mode: str = "Locale",
                     tile: int = 128, progress=None) -> list[np.ndarray]:
    out = []
    for i, f in enumerate(frames):
        if mode == "Globale":
            out.append(global_normalize(f, ref))
        elif mode == "Locale":
            out.append(local_normalize(f, ref, tile))
        else:
            out.append(np.asarray(f, np.float32))
        if progress:
            progress(i + 1, len(frames), "Normalisation")
    return out
