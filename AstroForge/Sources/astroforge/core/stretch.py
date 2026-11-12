# -*- coding: utf-8 -*-
"""
Étirements non linéaires avancés :

- GHS (Generalized Hyperbolic Stretch, D. Payne & M. Cranfield) : famille
  d'étirements paramétrée par l'intensité D, le type b (logarithmique,
  exponentiel, hyperbolique, harmonique…), le point de symétrie SP (là où le
  contraste est le plus augmenté), la protection des ombres LP et des hautes
  lumières HP (prolongements linéaires de même pente). Implémentation
  d'après les équations publiées (script GHS pour PixInsight/Siril).
- arcsinh (Lupton et al. 2004) : étirement qui préserve les COULEURS des
  étoiles (chaque pixel est multiplié par le même facteur sur ses 3 canaux).

Toutes les fonctions : float32 [0,1] mono ou RGB, paramètres neutres =
identité.
"""
from __future__ import annotations

import numpy as np


def _base(u: np.ndarray, D: float, b: float) -> np.ndarray:
    """Transformation de base T(u), u ≥ 0 (distance au point de symétrie)."""
    if abs(b + 1.0) < 1e-9:
        return np.log1p(D * u)
    if b < 0:
        return (1.0 - np.power(np.maximum(1.0 - b * D * u, 1e-12), (b + 1.0) / b)) / (D * (b + 1.0))
    if abs(b) < 1e-9:
        return 1.0 - np.exp(-D * u)
    return 1.0 - np.power(1.0 + b * D * u, -1.0 / b)


def _dbase(u: np.ndarray, D: float, b: float) -> np.ndarray:
    """Dérivée de T par rapport à u."""
    if abs(b + 1.0) < 1e-9:
        return D / (1.0 + D * u)
    if b < 0:
        return np.power(np.maximum(1.0 - b * D * u, 1e-12), 1.0 / b)
    if abs(b) < 1e-9:
        return D * np.exp(-D * u)
    return D * np.power(1.0 + b * D * u, -(1.0 + b) / b)


def ghs_curve(x: np.ndarray, D: float = 0.0, b: float = 0.0, SP: float = 0.0,
              LP: float = 0.0, HP: float = 1.0) -> np.ndarray:
    """Courbe GHS appliquée à x ∈ [0,1]. D est l'intensité « utilisateur »
    (0 à 10) : la transformation interne utilise exp(D) − 1, comme le script
    d'origine. Contraintes : 0 ≤ LP ≤ SP ≤ HP ≤ 1."""
    x = np.clip(np.asarray(x, np.float64), 0, 1)
    if D <= 0:
        return x
    Dn = float(np.expm1(D))
    b = float(np.clip(b, -5, 15))
    SP = float(np.clip(SP, 0, 1))
    LP = float(np.clip(LP, 0, SP))
    HP = float(np.clip(HP, SP, 1))

    def T(v):
        v = np.asarray(v, np.float64)
        out = np.empty_like(v)
        hi = v >= SP
        out[hi] = _base(v[hi] - SP, Dn, b)
        out[~hi] = -_base(SP - v[~hi], Dn, b)
        return out

    def dT(v):
        return float(_dbase(np.asarray([abs(v - SP)]), Dn, b)[0])

    y = T(x)
    if LP < SP:
        lo = x < LP
        y[lo] = T(np.array([LP]))[0] + dT(LP) * (x[lo] - LP)
    if HP > SP:
        hi = x > HP
        y[hi] = T(np.array([HP]))[0] + dT(HP) * (x[hi] - HP)
    # normalisation : 0 → 0, 1 → 1 (y monotone croissante)
    y0 = T(np.array([0.0]))[0] if LP <= 0 else T(np.array([LP]))[0] - dT(LP) * LP
    y1 = T(np.array([1.0]))[0] if HP >= 1 else T(np.array([HP]))[0] + dT(HP) * (1 - HP)
    return np.clip((y - y0) / max(y1 - y0, 1e-12), 0, 1)


def ghs(img, D: float = 0.0, b: float = 0.0, SP: float = 0.0, LP: float = 0.0,
        HP: float = 1.0, mode: str = "Canaux") -> np.ndarray:
    """Étirement GHS. mode « Canaux » : chaque canal ; « Luminance » :
    courbe appliquée à la luminance et rapports de couleurs conservés."""
    a = np.asarray(img, np.float32)
    if D <= 0:
        return np.clip(a, 0, 1).astype(np.float32)
    if a.ndim == 3 and str(mode).startswith("Lum"):
        lum = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
        new = ghs_curve(lum, D, b, SP, LP, HP)
        k = new / np.maximum(lum, 1e-6)
        out = a * k[..., None]
        # écrêtage doux : les canaux > 1 sont ramenés en gardant la teinte
        mx = out.max(axis=2, keepdims=True)
        out = np.where(mx > 1, out / mx, out)
        return np.clip(out, 0, 1).astype(np.float32)
    return ghs_curve(a, D, b, SP, LP, HP).astype(np.float32)


def arcsinh_stretch(img, stretch: float = 0.0, black: float = 0.0) -> np.ndarray:
    """Étirement arcsinh préservant les couleurs : facteur commun
    asinh(β·L)/(L·asinh β) appliqué aux trois canaux (L = moyenne RGB).
    β = 0 : identité ; noir soustrait avant étirement."""
    a = np.clip(np.asarray(img, np.float32), 0, None)
    beta = float(max(stretch, 0.0))
    bk = float(np.clip(black, 0.0, 0.5))
    if beta <= 0 and bk <= 0:
        return np.clip(a, 0, 1).astype(np.float32)
    a = np.clip((a - bk) / max(1.0 - bk, 1e-6), 0, None)
    if beta <= 0:
        return np.clip(a, 0, 1).astype(np.float32)
    lum = a.mean(axis=2) if a.ndim == 3 else a
    k = np.arcsinh(beta * lum) / (np.maximum(lum, 1e-9) * np.arcsinh(beta))
    k = np.where(lum > 1e-9, k, beta / np.arcsinh(beta))
    out = a * (k[..., None] if a.ndim == 3 else k)
    if out.ndim == 3:
        mx = out.max(axis=2, keepdims=True)
        out = np.where(mx > 1, out / mx, out)
    return np.clip(out, 0, 1).astype(np.float32)
