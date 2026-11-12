# -*- coding: utf-8 -*-
"""
Restauration : débruitage (non-local means, bilatéral, médiane, ondelettes),
déconvolution de Richardson-Lucy et masque flou (unsharp).
"""
from __future__ import annotations
import numpy as np
import scipy.ndimage as ndi
import scipy.signal as sig

DENOISE_METHODS = ["Non-local means", "Bilatéral", "Médiane", "Ondelettes"]


def gaussian_psf(sigma: float, size: int | None = None) -> np.ndarray:
    size = size or max(3, int(6 * sigma) | 1)
    ax = np.arange(size) - (size - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k = np.outer(g, g)
    return (k / k.sum()).astype(np.float64)


def _rl_plane(a: np.ndarray, psf: np.ndarray, iterations: int, damping: float) -> np.ndarray:
    psf_m = psf[::-1, ::-1]
    est = a.copy()
    for _ in range(max(1, iterations)):
        conv = sig.fftconvolve(est, psf, mode="same")
        rel = a / np.maximum(conv, 1e-7)
        new = est * sig.fftconvolve(rel, psf_m, mode="same")
        if damping:
            new = (1 - damping) * new + damping * est
        est = np.clip(new, 0, None)
    # conservation du flux (standard RL) : les pics montent, le fond reste stable
    return est * (a.sum() / max(est.sum(), 1e-9))


def richardson_lucy(img: np.ndarray, sigma_psf: float = 1.5, iterations: int = 20,
                    damping: float = 0.2) -> np.ndarray:
    """Déconvolution de Richardson-Lucy avec PSF gaussienne (+ amortissement),
    canal par canal pour une image couleur."""
    a = np.clip(np.asarray(img, np.float64), 0, None)
    psf = gaussian_psf(sigma_psf)
    if a.ndim == 3:
        out = np.stack([_rl_plane(a[..., c], psf, iterations, damping)
                        for c in range(a.shape[2])], axis=-1)
    else:
        out = _rl_plane(a, psf, iterations, damping)
    return out.astype(np.float32)


def noise_sigma(img: np.ndarray) -> float:
    """Écart-type du bruit estimé par MAD du résidu passe-haut (robuste aux
    étoiles et au signal)."""
    a = np.asarray(img, np.float32)
    mono = a if a.ndim == 2 else a.mean(axis=2)
    hp = mono - ndi.median_filter(mono, size=3)
    return float(1.4826 * np.median(np.abs(hp - np.median(hp))) / 0.85 + 1e-6)


def unsharp(img: np.ndarray, amount: float = 0.6, radius: float = 2.0) -> np.ndarray:
    """Masque flou ( netteté classique)."""
    a = np.asarray(img, np.float32)
    blur = ndi.gaussian_filter(a, sigma=radius, axes=(0, 1) if a.ndim == 3 else None)
    out = a + np.float32(amount) * (a - blur)
    return np.clip(out, 0, 1).astype(np.float32) if a.max() <= 1.5 else out.astype(np.float32)


def denoise(img: np.ndarray, method: str = "Non-local means",
            strength: float = 0.5, median_size: int = 3) -> np.ndarray:
    """Débruitage classique en flottant / 16 bits (aucune quantification
    8 bits : les données linéaires faibles gardent leur dynamique).
    strength ∈ [0..1] ; la force est relative au bruit mesuré de l'image."""
    cv2 = __import__("cv2")
    from .image_io import is_color
    a = np.asarray(img, np.float32)
    s = float(np.clip(strength, 0, 1))
    if method == "Médiane":
        k = median_size | 1
        if is_color(a):
            return np.stack([ndi.median_filter(a[..., c], size=k)
                             for c in range(a.shape[2])], -1).astype(np.float32)
        return ndi.median_filter(a, size=k).astype(np.float32)
    if method == "Ondelettes":
        from .wavelets import wavelet_denoise
        if is_color(a):
            return np.stack([wavelet_denoise(a[..., c], 4, s)
                             for c in range(a.shape[2])], -1).astype(np.float32)
        return wavelet_denoise(a, 4, s)
    sigma_n = noise_sigma(a)
    if method == "Bilatéral":
        # filtre bilatéral natif float32 : écart de couleur proportionnel au bruit
        sc = float(sigma_n * (1.0 + 5.0 * s))
        return cv2.bilateralFilter(np.ascontiguousarray(a), 9, sc, 3.0 + 4.0 * s).astype(np.float32)
    # Non-local means en 16 bits (norme L1), canal par canal ; h ∝ bruit mesuré
    hi = float(max(a.max(initial=0), 1e-6))
    scale = 65535.0 / hi
    h = float(np.clip(sigma_n * scale * (0.6 + 2.4 * s), 1.0, 65535.0))

    def nlm(plane):
        u16 = np.clip(plane * scale, 0, 65535).astype(np.uint16)
        d = cv2.fastNlMeansDenoising(u16, h=[h], templateWindowSize=7,
                                     searchWindowSize=21, normType=cv2.NORM_L1)
        return d.astype(np.float32) / np.float32(scale)
    if is_color(a):
        return np.stack([nlm(a[..., c]) for c in range(a.shape[2])], -1).astype(np.float32)
    return nlm(a).astype(np.float32)
