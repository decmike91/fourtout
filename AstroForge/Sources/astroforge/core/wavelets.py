# -*- coding: utf-8 -*-
"""
Ondelettes à trous (stationnaires), contrôles comparables à Registax6,
sans prétendre reproduire son algorithme. Décomposition en détails et
résidu, gains, débruitage et accentuation par couche.
"""
from __future__ import annotations
import numpy as np
import scipy.ndimage as ndi

# Spline B3 (filtre binomial) : [1,8,28,56,70,56,28,8,1]/256
B3 = np.array([1, 8, 28, 56, 70, 56, 28, 8, 1], np.float64) / 256.0


def _kernel_level(level: int) -> np.ndarray:
    """Noyau 1D à trous au niveau L : B3 intercalé de 2^(L-1) - 1 zéros."""
    if level == 0:
        return B3
    step = 2 ** level
    k = np.zeros(9 + (9 - 1) * (step - 1), np.float64)
    k[::step] = B3
    return k


def atrous_decompose(img: np.ndarray, levels: int = 6) -> list[np.ndarray]:
    """Décompose en [détail1, ..., détailN, résidu]."""
    a = np.asarray(img, np.float64)
    layers = []
    current = a.copy()
    for lvl in range(levels):
        k = _kernel_level(lvl)
        smooth = ndi.convolve1d(current, k, axis=0, mode="reflect")
        smooth = ndi.convolve1d(smooth, k, axis=1, mode="reflect")
        layers.append(current - smooth)
        current = smooth
    layers.append(current)
    return layers


def atrous_recompose(layers: list[np.ndarray], gains: list[float] | None = None,
                     thresholds: list[float] | None = None) -> np.ndarray:
    """Recompose avec gain par couche (1.0 = neutre) et seuil doux optionnel
    (valeur absolue en dessous de laquelle le détail est atténué)."""
    n = len(layers)
    gains = list(gains) if gains else [1.0] * n
    thresholds = list(thresholds) if thresholds else [0.0] * n
    out = np.zeros_like(layers[0], np.float64)
    for i in range(n):
        l = layers[i]
        t = thresholds[i]
        if t and i < n - 1:  # seuil doux sur les détails uniquement
            l = np.sign(l) * np.maximum(np.abs(l) - t, 0)
        out += l * np.float64(gains[i])
    return out.astype(np.float32)


def wavelet_denoise(img: np.ndarray, levels: int = 4, strength: float = 0.5) -> np.ndarray:
    """Denoise par seuillage doux des couches de détail (seuil universel)."""
    layers = atrous_decompose(img, levels)
    new = []
    for _i, l in enumerate(layers[:-1]):
        sigma = np.median(np.abs(l)) / 0.6745 + 1e-9
        t = sigma * np.sqrt(2 * np.log(l.size)) * strength
        new.append(np.sign(l) * np.maximum(np.abs(l) - t, 0))
    new.append(layers[-1])
    return atrous_recompose(new)


def enhance_layers(layers, denoise=None, sharpen=None):
    """Seuil doux MAD indépendant puis amplification (1 + netteté) du détail.
    Débruitage 0..1, netteté 0..3 ; résidu inchangé, pas de mélange des canaux.
    Les autres couches ne sont jamais retouchées.
    """
    dn = ([0.0] * 6 if denoise is None else list(denoise))
    sh = ([0.0] * 6 if sharpen is None else list(sharpen))
    if len(dn) != 6 or len(sh) != 6 or not np.isfinite(dn + sh).all():
        raise ValueError("Six valeurs finies requises par contrôle d'ondelettes.")
    out = []
    for i, layer in enumerate(layers[:-1]):
        detail = layer
        strength = float(np.clip(dn[i], 0, 1))
        if strength:
            axes = (0, 1)
            median = np.median(detail, axis=axes, keepdims=True)
            sigma = np.median(np.abs(detail - median), axis=axes,
                              keepdims=True) / .6745
            threshold = sigma * np.sqrt(2 * np.log(max(2, detail.shape[0] * detail.shape[1]))) * strength
            detail = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
        amount = float(np.clip(sh[i], 0, 3))
        out.append(detail * (1 + amount) if amount else detail)
    return out + [layers[-1]]


def registax_enhance(img: np.ndarray, gains: list[float] | None = None,
                     denoise_strength: float = 0.0, *, layer_denoise=None,
                     layer_sharpen=None) -> np.ndarray:
    """Six couches indépendantes. Les trois arguments historiques restent
    compatibles (notamment le débruitage global utilisé en ciel profond).
    Sans nouveaux contrôles, la sortie conserve le comportement historique.
    """
    a = np.asarray(img, np.float32)
    if denoise_strength and denoise_strength > 0:
        a = wavelet_denoise(a, 4, float(np.clip(denoise_strength, 0, 1)))
    layers = atrous_decompose(a, 6)
    if layer_denoise is not None or layer_sharpen is not None:
        layers = enhance_layers(layers, layer_denoise, layer_sharpen)
    g = (list(gains) + [1.0] * 7)[:7] if gains is not None else [1.0] * 7
    return atrous_recompose(layers, g)
