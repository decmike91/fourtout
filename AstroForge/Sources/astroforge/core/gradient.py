# -*- coding: utf-8 -*-
"""
Extraction du fond de ciel / gradients : ajustement polynomial ou interpolation
RBF sur une grille d'échantillons (style ABE/DBE de PixInsight & GraXpert).
"""
from __future__ import annotations
import numpy as np

METHODS = ["Polynôme 2D", "RBF (thin plate)"]


def _background_samples(img: np.ndarray, grid: int = 12, sigma: float = 2.5,
                        min_frac: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Échantillonne le fond : médiane par case avec rejet sigma des étoiles
    (rejet décidé sur la luminance). Retourne (points (N,2), valeurs (N,) en
    mono ou (N,C) en couleur — un échantillon par canal)."""
    a = np.asarray(img, np.float32)
    mono = a if a.ndim == 2 else a.mean(axis=2)
    H, W = mono.shape
    gh = max(3, grid)
    gw = max(3, grid)
    pts, vals = [], []
    edges_y = np.linspace(0, H, gh + 1).astype(int)
    edges_x = np.linspace(0, W, gw + 1).astype(int)
    for i in range(gh):
        for j in range(gw):
            block = mono[edges_y[i]:edges_y[i + 1], edges_x[j]:edges_x[j + 1]]
            if block.size == 0:
                continue
            med = np.median(block)
            mad = 1.4826 * np.median(np.abs(block - med)) + 1e-9
            ok = np.abs(block - med) <= sigma * mad
            if ok.mean() < min_frac:
                continue  # case trop occupée par des étoiles/nébuleuse
            pts.append(((edges_x[j] + edges_x[j + 1]) / 2.0,
                        (edges_y[i] + edges_y[i + 1]) / 2.0))
            if a.ndim == 3:
                cell = a[edges_y[i]:edges_y[i + 1], edges_x[j]:edges_x[j + 1]]
                vals.append(np.median(cell[ok], axis=0))
            else:
                vals.append(np.median(block[ok]))
    if len(pts) < 6:
        raise ValueError("Pas assez d'échantillons de fond — réduisez la grille ou sigma")
    return np.array(pts, np.float64), np.array(vals, np.float64)


def _fit_poly(pts: np.ndarray, vals: np.ndarray, degree: int, H: int, W: int) -> np.ndarray:
    """Ajuste un polynôme 2D (deg ≤ 5) et évalue sur toute l'image."""
    x, y = pts[:, 0], pts[:, 1]
    xn, yn = x / W, y / H
    terms = []
    for dy in range(degree + 1):
        for dx in range(degree + 1 - dy):
            terms.append((xn ** dx) * (yn ** dy))
    A = np.stack(terms, axis=1)
    coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
    yy, xx = np.mgrid[0:H, 0:W]
    xn2, yn2 = xx / W, yy / H
    out = np.zeros((H, W), np.float64)
    idx = 0
    for dy in range(degree + 1):
        for dx in range(degree + 1 - dy):
            out += coef[idx] * (xn2 ** dx) * (yn2 ** dy)
            idx += 1
    return out


def _fit_rbf(pts: np.ndarray, vals: np.ndarray, H: int, W: int, smooth: float = 1.0) -> np.ndarray:
    from scipy.interpolate import RBFInterpolator
    rbf = RBFInterpolator(pts, vals, kernel="thin_plate_spline", smoothing=smooth)
    yy, xx = np.mgrid[0:H, 0:W]
    np.stack([(xx / W).ravel(), (yy / H).ravel()], axis=1)
    # évalue sur grille réduite puis agrandit (économie mémoire)
    step = max(1, int(np.sqrt((H * W) / 4e6)))
    small = rbf(np.stack([(xx[::step, ::step] / W).ravel(),
                          (yy[::step, ::step] / H).ravel()], axis=1))
    small = small.reshape(H // step + (H % step > 0), W // step + (W % step > 0))
    model = cv2resize(small, (W, H))
    return model


def cv2resize(small: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    cv2 = __import__("cv2")
    return cv2.resize(small.astype(np.float32), size, interpolation=cv2.INTER_CUBIC)


def extract_gradient(img: np.ndarray, method: str = "Polynôme 2D", degree: int = 3,
                     grid: int = 12, sigma: float = 2.5,
                     correction: str = "Soustraction") -> tuple[np.ndarray, np.ndarray]:
    """Extrait et retire le fond (gradient). Retourne (image corrigée, modèle
    de fond). En couleur, un modèle est ajusté PAR CANAL (la pollution
    lumineuse est colorée). Correction : Soustraction (fond remplacé par un
    piédestal neutre égal à son niveau moyen) ou Division (vignettage résiduel, ciel très dégradé)."""
    a = np.asarray(img, np.float32)
    H, W = a.shape[:2]
    pts, vals = _background_samples(a, grid, sigma)
    degree = int(np.clip(degree, 1, 5))

    def fit(v):
        if method.startswith("Polynôme"):
            return _fit_poly(pts, v, degree, H, W)
        return _fit_rbf(pts, v, H, W)
    if a.ndim == 3:
        model = np.stack([fit(vals[:, c]) for c in range(a.shape[2])], axis=-1)
    else:
        model = fit(vals)
    model = np.clip(model, 0, None).astype(np.float32)
    level = np.median(model.reshape(-1, model.shape[-1]) if a.ndim == 3 else model,
                      axis=0).astype(np.float32)
    if correction.startswith("Division"):
        out = a / np.maximum(model, 1e-6) * level
    else:
        # piédestal neutre (même valeur sur tous les canaux) : le bruit du fond
        # n'est pas écrêté à 0, sans réintroduire de dominante colorée
        out = a - model + np.float32(np.mean(level))
    return np.clip(out, 0, None).astype(np.float32), model
