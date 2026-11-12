# -*- coding: utf-8 -*-
"""
Drizzle (Fruchter & Hook 2002), noyau carré, pixfrac réglable, échelle ×2/×3.

Chaque pixel d'ENTRÉE (image calibrée, NON rééchantillonnée) est projeté sur
la grille de sortie suréchantillonnée grâce à la transformation de sa frame
(translation, rotation, similitude…) ; sa « goutte » carrée de côté
pixfrac × échelle pixels de sortie est répartie au prorata des surfaces
recouvertes. Le gain de résolution vient du sous-échantillonnage aléatoire
(dithering) entre les frames : il faut donc les images AVANT alignement.

Rejet optionnel : si une image de référence (empilement classique avec
rejet, à la résolution d'entrée) est fournie, les pixels d'entrée qui s'en
écartent de plus de `reject_sigma` × bruit de la frame (satellites, rayons
cosmiques, pixels chauds) ne sont pas déposés.
"""
from __future__ import annotations
import math

import numpy as np


def shift_matrix(dx: float, dy: float, angle_deg: float = 0.0,
                 shape: tuple[int, int] | None = None) -> np.ndarray:
    """Matrice 2×3 frame -> référence, identique à registration.apply_shift
    (rotation autour du centre puis translation (dx, dy))."""
    cv2 = __import__("cv2")
    if angle_deg and shape is not None:
        h, w = shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    else:
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    M = np.asarray(M, np.float64).copy()
    M[0, 2] += dx
    M[1, 2] += dy
    return M


def drizzle_stack(frames, shifts: list[tuple[float, float]] | None = None,
                  angles: list[float] | None = None, scale: int = 2,
                  pixfrac: float = 0.7, weights: list[float] | None = None,
                  progress=None, matrices: list[np.ndarray] | None = None,
                  reference: np.ndarray | None = None,
                  reject_sigma: float | None = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Combine des frames calibrées NON alignées par drizzle.
    - frames : liste de HxW ou HxWx3 (float [0..1]) ;
    - shifts : (dx, dy) par frame tels que retournés par
      registration.phase_corr(ref, frame) (translation qui recale la frame) ;
    - angles : rotation (deg) par frame, même convention que apply_shift ;
    - matrices : alternative à shifts/angles, matrice 2×3 frame -> référence ;
    - reference : image empilée (taille d'entrée) servant au rejet ;
    Retourne (image drizzlée, couverture normalisée)."""
    cv2 = __import__("cv2")
    if frames is None or len(frames) == 0:
        raise ValueError("Drizzle : aucune frame")
    n = len(frames)
    scale = int(max(1, scale))
    pixfrac = float(np.clip(pixfrac, 0.05, 1.0))
    first = np.asarray(frames[0], np.float32)
    h, w = first.shape[:2]
    chans = first.shape[2] if first.ndim == 3 else 1
    if matrices is None:
        shifts = list(shifts) if shifts is not None else [(0.0, 0.0)] * n
        angles = list(angles) if angles is not None else [0.0] * n
        matrices = [shift_matrix(float(s[0]), float(s[1]), float(a), (h, w))
                    for s, a in zip(shifts, angles)]
    weights = list(weights) if weights is not None else [1.0] * n
    oh, ow = h * scale, w * scale
    acc = np.zeros((chans, oh * ow), np.float64)
    cov = np.zeros(oh * ow, np.float64)

    ref_img = None
    if reference is not None and reject_sigma:
        ref_img = np.asarray(reference, np.float32)
        if ref_img.ndim == 2:
            ref_img = ref_img[..., None]

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    for k in range(n):
        img = np.asarray(frames[k], np.float32)
        if img.ndim == 2:
            img = img[..., None]
        M = np.asarray(matrices[k], np.float64)
        u = M[0, 0] * xx + M[0, 1] * yy + M[0, 2]       # coordonnées référence
        v = M[1, 0] * xx + M[1, 1] * yy + M[1, 2]
        lin = math.sqrt(abs(np.linalg.det(M[:, :2]))) or 1.0
        half = pixfrac * scale * lin / 2.0               # demi-goutte (px de sortie)
        cx = (u + 0.5) * scale                            # sortie : pixel j = [j, j+1)
        cy = (v + 0.5) * scale
        valid = np.ones((h, w), bool)
        if ref_img is not None:
            refv = np.stack([cv2.remap(np.ascontiguousarray(ref_img[..., c]),
                                       u.astype(np.float32), v.astype(np.float32),
                                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                             for c in range(ref_img.shape[2])], -1)
            if refv.shape[2] != chans:
                refv = np.repeat(refv.mean(axis=2, keepdims=True), chans, axis=2)
            resid = img - refv
            mad = np.median(np.abs(resid - np.median(resid, axis=(0, 1))), axis=(0, 1))
            sig = 1.4826 * mad + 1e-6
            valid = np.all(np.abs(resid) <= float(reject_sigma) * sig, axis=2)
        x0 = np.floor(cx - half).astype(np.int64)
        y0 = np.floor(cy - half).astype(np.int64)
        kmax = int(math.ceil(2 * half)) + 1
        wk = float(weights[k])
        vals = img.reshape(-1, chans)
        for oy in range(kmax):
            iy = y0 + oy
            ovy = np.minimum(iy + 1, cy + half) - np.maximum(iy, cy - half)
            for ox in range(kmax):
                ix = x0 + ox
                ovx = np.minimum(ix + 1, cx + half) - np.maximum(ix, cx - half)
                wgt = np.clip(ovx, 0, None) * np.clip(ovy, 0, None)
                ok = (wgt > 0) & valid & (ix >= 0) & (ix < ow) & (iy >= 0) & (iy < oh)
                if not ok.any():
                    continue
                idx = (iy[ok] * ow + ix[ok]).ravel()
                wv = (wgt[ok] * wk).ravel()
                cov += np.bincount(idx, weights=wv, minlength=oh * ow)
                for c in range(chans):
                    acc[c] += np.bincount(idx, weights=vals[ok.ravel(), c] * wv,
                                          minlength=oh * ow)
        if progress:
            progress(k + 1, n)

    out = (acc / np.maximum(cov, 1e-12)).reshape(chans, oh, ow)
    out = np.moveaxis(out, 0, -1) if chans > 1 else out[0]
    holes = cov.reshape(oh, ow) <= 1e-9
    if holes.any():  # trous (pixfrac trop petit / peu de frames) : bouchés
        fill_src = reference if reference is not None else np.mean(
            [np.asarray(f, np.float32) for f in frames[:min(n, 8)]], axis=0)
        fill = cv2.resize(np.asarray(fill_src, np.float32), (ow, oh),
                          interpolation=cv2.INTER_CUBIC)
        if fill.ndim == 2 and out.ndim == 3:
            fill = fill[..., None]
        out[holes] = fill[holes]
    coverage = cov.reshape(oh, ow)
    coverage_norm = np.clip(coverage / max(float(coverage.max()), 1e-9), 0, 1)
    return out.astype(np.float32), coverage_norm.astype(np.float32)
