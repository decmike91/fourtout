# -*- coding: utf-8 -*-
"""
Empilement de comètes (esprit DeepSkyStacker « étoiles + comète ») :

1. les images sont d'abord alignées sur les ÉTOILES (pipeline habituel) ;
2. la position du noyau est connue sur la première et la dernière image
   (saisie ou détection), interpolée pour chaque image d'après l'heure
   d'acquisition (DATE-OBS) ou, à défaut, son rang ;
3. deux empilements avec rejet : sur les étoiles (la comète mobile est
   rejetée) et sur la comète (les étoiles, qui défilent, sont rejetées) ;
4. composition : maximum des deux (« étoiles + comète ») ou addition du
   signal de la comète au-dessus du fond.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

from . import stacking as S
from .registration import apply_shift

COMPOSITE_MODES = ("Étoiles + comète (maximum)", "Étoiles + comète (addition)",
                   "Comète seule", "Étoiles seules")


def parse_times(dates: list) -> list[float] | None:
    """DATE-OBS (ISO) -> secondes depuis la première image ; None si absent."""
    out = []
    for d in dates:
        if not d:
            return None
        try:
            t = _dt.datetime.fromisoformat(str(d).replace("Z", "+00:00"))
        except ValueError:
            return None
        out.append(t.timestamp())
    return [x - out[0] for x in out]


def interpolate_positions(n: int, first: tuple, last: tuple,
                          times: list[float] | None = None) -> list[tuple[float, float]]:
    """Positions du noyau pour n images, linéaires dans le temps (ou le rang)."""
    t = np.asarray(times if times is not None else np.arange(n), np.float64)
    span = (t[-1] - t[0]) or 1.0
    f = (t - t[0]) / span
    x = first[0] + f * (last[0] - first[0])
    y = first[1] + f * (last[1] - first[1])
    return list(zip(x.tolist(), y.tolist()))


def refine_nucleus(img: np.ndarray, x: float, y: float, radius: int = 12) -> tuple[float, float]:
    """Centre photométrique du noyau autour de (x, y) (fenêtre lissée)."""
    import scipy.ndimage as ndi
    a = np.asarray(img, np.float32)
    mono = a if a.ndim == 2 else a.mean(axis=2)
    h, w = mono.shape
    x0, x1 = max(0, int(x) - radius), min(w, int(x) + radius + 1)
    y0, y1 = max(0, int(y) - radius), min(h, int(y) + radius + 1)
    win = ndi.gaussian_filter(mono[y0:y1, x0:x1], 2.0)
    win = np.clip(win - np.median(win), 0, None)
    if win.sum() <= 0:
        return float(x), float(y)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    return float((win * xx).sum() / win.sum()), float((win * yy).sum() / win.sum())


def comet_stack(frames: list[np.ndarray], positions: list[tuple[float, float]],
                method: str = "Sigma-clipping", sigma: float = 2.5,
                mode: str = COMPOSITE_MODES[0], refine: bool = True,
                progress=None) -> dict:
    """`frames` alignées sur les étoiles ; `positions` du noyau dans chaque
    image. Retourne {"stars", "comet", "composite", "positions"}."""
    if len(frames) != len(positions) or len(frames) < 3:
        raise ValueError("Comète : au moins 3 images et une position par image")
    pos = [refine_nucleus(f, *p) if refine else tuple(map(float, p))
           for f, p in zip(frames, positions)]
    ref = pos[len(pos) // 2]
    stars = S.stack(np.stack([np.asarray(f, np.float32) for f in frames]), method, sigma=sigma)
    shifted = []
    for i, (f, p) in enumerate(zip(frames, pos)):
        shifted.append(apply_shift(np.asarray(f, np.float32), ref[0] - p[0], ref[1] - p[1],
                                   border_mode="reflect"))
        if progress:
            progress(i + 1, len(frames), "Alignement sur la comète")
    comet = S.stack(np.stack(shifted), method, sigma=sigma)
    if mode.startswith("Comète seule"):
        comp = comet
    elif mode.startswith("Étoiles seules"):
        comp = stars
    elif "addition" in mode:
        bg = np.median(comet.reshape(-1, comet.shape[-1]) if comet.ndim == 3 else comet, axis=0)
        comp = stars + np.clip(comet - bg, 0, None)
    else:
        comp = np.maximum(stars, comet)
    return {"stars": stars, "comet": comet, "composite": np.clip(comp, 0, 1).astype(np.float32),
            "positions": pos}
