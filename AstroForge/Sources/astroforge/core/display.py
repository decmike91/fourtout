# -*- coding: utf-8 -*-
"""Réglages d'affichage : historique d'étirement (autostretch), histogramme."""
from __future__ import annotations
import numpy as np


def mtf(m: float, x):
    """Fonction de transfert des tons moyens (Midtones Transfer Function) :
    MTF(m, 0) = 0, MTF(m, m) = 0,5, MTF(m, 1) = 1."""
    x = np.asarray(x, np.float32)
    m = float(np.clip(m, 1e-6, 1 - 1e-6))
    return ((m - 1.0) * x) / ((2.0 * m - 1.0) * x - m)


def stf_params(img: np.ndarray, shadows_clip: float = -2.8,
               target_bg: float = 0.25) -> tuple[float, float]:
    """Paramètres d'un étirement automatique type STF : point noir =
    médiane + shadows_clip·σ (σ = 1,4826·MAD), tons moyens m choisis pour que
    la médiane du fond arrive à `target_bg`. Retourne (noir, m)."""
    a = np.asarray(img, np.float32)
    lum = a if a.ndim == 2 else a[..., :3].mean(axis=2)
    sample = lum[::max(1, lum.shape[0] // 512), ::max(1, lum.shape[1] // 512)]
    med = float(np.median(sample))
    sig = 1.4826 * float(np.median(np.abs(sample - med)))
    black = float(np.clip(med + shadows_clip * sig, 0.0, max(med - 1e-6, 0.0)))
    x0 = (med - black) / max(1.0 - black, 1e-6)
    t = float(np.clip(target_bg, 0.01, 0.99))
    if x0 <= 1e-7:
        return black, 0.5
    m = x0 * (1.0 - t) / (x0 - 2.0 * t * x0 + t)
    return black, float(np.clip(m, 1e-6, 1 - 1e-6))


def autostretch(img: np.ndarray, sigma: float = 3.0, shadows_clip: float = -2.8,
                target_bg: float = 0.25) -> np.ndarray:
    """Étirement automatique NON LINÉAIRE (esprit STF de PixInsight / Siril) :
    point noir sous le fond puis fonction de transfert des tons moyens. Le fond
    est amené à ~25 % sans écrêter les hautes lumières (cœurs de galaxies,
    étoiles). `sigma` est conservé pour compatibilité (non utilisé)."""
    a = np.asarray(img, np.float32)
    black, m = stf_params(a, shadows_clip, target_bg)
    x = np.clip((a - black) / max(1.0 - black, 1e-6), 0, 1)
    return np.clip(mtf(m, x), 0, 1).astype(np.float32)


def histogram(img: np.ndarray, bins: int = 256) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(img, np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    hist, edges = np.histogram(np.clip(a, 0, 1), bins=bins, range=(0, 1))
    return hist, edges
