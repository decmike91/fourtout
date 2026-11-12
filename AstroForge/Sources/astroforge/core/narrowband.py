# -*- coding: utf-8 -*-
"""
Bande étroite et composition couleur.

- Extraction Ha / OIII d'une image de capteur couleur (CFA) prise avec un
  filtre duo-bande (L-eXtreme, L-eNhance, ALP-T…) : Ha = pixels rouges,
  OIII = moyenne des pixels verts et bleus (les deux voient la raie OIII à
  500,7 nm). Résolution native de chaque plan (moitié de l'image) ou
  agrandie à la taille d'origine.
- Normalisation des canaux (fond et dispersion égalisés sur une référence).
- Palettes : HOO, HOO équilibrée, SHO (Hubble), HSO, OSH, personnalisée
  (matrice 3×3).
- Combinaison LRGB : la luminance L remplace celle de l'image RGB, les
  couleurs (rapports de canaux) sont conservées.
- Alignement des canaux entre eux (images prises à des nuits différentes).
"""
from __future__ import annotations

import numpy as np

PALETTES = {
    # nom : {canal de sortie: {canal source: poids}}
    "HOO": {"R": {"Ha": 1.0}, "G": {"OIII": 1.0}, "B": {"OIII": 1.0}},
    "HOO équilibrée": {"R": {"Ha": 1.0}, "G": {"Ha": 0.3, "OIII": 0.7}, "B": {"OIII": 1.0}},
    "SHO (Hubble)": {"R": {"SII": 1.0}, "G": {"Ha": 1.0}, "B": {"OIII": 1.0}},
    "HSO": {"R": {"Ha": 1.0}, "G": {"SII": 1.0}, "B": {"OIII": 1.0}},
    "OSH": {"R": {"OIII": 1.0}, "G": {"SII": 1.0}, "B": {"Ha": 1.0}},
    "RGB": {"R": {"R": 1.0}, "G": {"G": 1.0}, "B": {"B": 1.0}},
}
OFFSETS = {"RGGB": (0, 0), "BGGR": (1, 1), "GRBG": (0, 1), "GBRG": (1, 0)}


def _red_offset(pattern: str):
    p = pattern.upper()
    if p not in OFFSETS:
        raise ValueError(f"Motif de Bayer inconnu : {pattern}")
    return OFFSETS[p]


def extract_ha_oiii(cfa: np.ndarray, pattern: str = "RGGB",
                    upscale: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """CFA mono (avant dématriçage) -> (Ha, OIII). Sans `upscale`, plans de
    demi-résolution sans aucune interpolation ; avec, agrandis (bicubique) à
    la taille du capteur."""
    a = np.asarray(cfa, np.float32)
    if a.ndim != 2:
        raise ValueError("Extraction Ha/OIII : image CFA mono (non dématricée) attendue")
    h, w = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
    a = a[:h, :w]
    ry, rx = _red_offset(pattern)
    by, bx = 1 - ry, 1 - rx
    ha = a[ry::2, rx::2]
    g1 = a[ry::2, bx::2]
    g2 = a[by::2, rx::2]
    b = a[by::2, bx::2]
    oiii = (g1 + g2 + b) / 3.0
    if upscale:
        import cv2
        ha = cv2.resize(ha, (w, h), interpolation=cv2.INTER_CUBIC)
        oiii = cv2.resize(oiii, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(ha, 0, None).astype(np.float32), np.clip(oiii, 0, None).astype(np.float32)


def extract_from_rgb(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Approximation sur une image déjà dématricée : Ha = R, OIII = (G+B)/2."""
    a = np.asarray(rgb, np.float32)
    if a.ndim != 3:
        raise ValueError("Image couleur attendue")
    return a[..., 0].copy(), ((a[..., 1] + a[..., 2]) * 0.5).astype(np.float32)


def normalize_to(img: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Égalise fond (médiane) et dispersion (MAD) de `img` sur `ref` —
    équilibre les canaux avant composition (linear fit robuste)."""
    a = np.asarray(img, np.float32)
    r = np.asarray(ref, np.float32)
    ma, mr = float(np.median(a)), float(np.median(r))
    sa = float(np.median(np.abs(a - ma))) + 1e-9
    sr = float(np.median(np.abs(r - mr))) + 1e-9
    return np.clip((a - ma) * (sr / sa) + mr, 0, None).astype(np.float32)


def align_channels(channels: dict, ref_key: str | None = None) -> dict:
    """Aligne tous les canaux sur le canal de référence (similitude par
    étoiles, sinon corrélation de phase)."""
    from .registration import estimate_transform, warp
    keys = list(channels)
    if not keys:
        return {}
    ref_key = ref_key or keys[0]
    ref = channels[ref_key]
    out = {ref_key: np.asarray(ref, np.float32)}
    for k in keys:
        if k == ref_key:
            continue
        img = np.asarray(channels[k], np.float32)
        if img.shape[:2] != ref.shape[:2]:
            import cv2
            img = cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)
        out[k] = warp(img, estimate_transform(ref, img, with_rotation=True))
    return out


def compose(channels: dict, palette="HOO", normalize: bool = True,
            ref_key: str | None = None) -> np.ndarray:
    """Compose une image RGB à partir de canaux mono {"Ha": …, "OIII": …,
    "SII": …, "R": …, "G": …, "B": …, "L": …}. `palette` : nom de PALETTES
    ou dictionnaire {sortie: {source: poids}}. Les canaux doivent être
    alignés et de même taille (voir align_channels)."""
    pal = PALETTES[palette] if isinstance(palette, str) else palette
    needed = {src for mix in pal.values() for src in mix}
    missing = sorted(needed - set(channels))
    if missing:
        raise ValueError(f"Canaux manquants pour la palette : {', '.join(missing)}")
    chans = {k: np.asarray(v, np.float32) for k, v in channels.items() if k in needed}
    shapes = {v.shape for v in chans.values()}
    if len(shapes) != 1 or len(next(iter(shapes))) != 2:
        raise ValueError("Les canaux doivent être des images mono de même taille")
    if normalize and len(chans) > 1:
        ref_key = ref_key if ref_key in chans else ("Ha" if "Ha" in chans else next(iter(chans)))
        ref = chans[ref_key]
        chans = {k: (v if k == ref_key else normalize_to(v, ref)) for k, v in chans.items()}
    out = []
    for ch in ("R", "G", "B"):
        mix = pal.get(ch, {})
        tot = sum(mix.values()) or 1.0
        out.append(sum(chans[src] * float(wt) for src, wt in mix.items()) / tot
                   if mix else np.zeros(next(iter(shapes)), np.float32))
    return np.clip(np.stack(out, -1), 0, 1).astype(np.float32)


def lrgb(lum: np.ndarray, rgb: np.ndarray, saturation: float = 1.0) -> np.ndarray:
    """Combinaison LRGB : la luminance de l'image couleur est remplacée par L
    (images alignées, même taille) ; teintes conservées, saturation réglable."""
    L = np.asarray(lum, np.float32)
    C = np.asarray(rgb, np.float32)
    if C.ndim != 3 or L.shape != C.shape[:2]:
        raise ValueError("LRGB : L mono et RGB couleur de même taille requis")
    Y = 0.2126 * C[..., 0] + 0.7152 * C[..., 1] + 0.0722 * C[..., 2]
    chroma = C - Y[..., None]
    out = L[..., None] + chroma * float(saturation) * (L / np.maximum(Y, 1e-4))[..., None]
    mx = out.max(axis=2, keepdims=True)
    out = np.where(mx > 1, out / mx, out)
    return np.clip(out, 0, 1).astype(np.float32)
