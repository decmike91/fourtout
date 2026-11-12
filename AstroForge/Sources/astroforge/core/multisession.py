# -*- coding: utf-8 -*-
"""
Empilement multi-nuits : chaque nuit est calibrée avec SES propres masters
(darks, flats de la nuit), puis toutes les images sont alignées sur une
référence commune, normalisées (globalement ou localement : fonds et
gradients différents d'une nuit à l'autre), pondérées et empilées ensemble.
"""
from __future__ import annotations

import numpy as np

from . import calibration as C
from . import stacking as S
from .normalization import normalize_frames
from .registration import estimate_transform, warp


def session_from_folder(folder: str) -> dict:
    """Session triée automatiquement : groupe de lumières le plus fourni."""
    from .session import plan_session
    plan = plan_session(folder)
    if not plan.groups:
        raise ValueError(f"Aucune image lumière dans {folder}")
    g = plan.groups[0]
    return {"name": g.name, "lights": g.paths("light"),
            "calib": {k: g.paths(k) for k in ("bias", "dark", "darkflat", "flat")},
            "summary": g.summary()}


def stack_sessions(sessions: list, method: str = "Sigma-clipping", sigma: float = 3.0,
                   normalization: str = "Locale", bayer: str | None = None,
                   output: str = "RGB", weight_mode: str = "Aucune",
                   progress=None, log=None) -> tuple[np.ndarray, dict]:
    """`sessions` : liste de dossiers ou de dicts {"lights": [...],
    "calib": {"bias": [...], "dark": [...], "flat": [...], "darkflat": [...]}}.
    Retourne (image empilée, infos)."""
    sess = [session_from_folder(s) if isinstance(s, str) else dict(s) for s in sessions]
    if not sess:
        raise ValueError("Aucune session")
    frames, origin = [], []
    for k, s in enumerate(sess):
        masters = C.build_masters(paths=s.get("calib") or {})
        imgs = C.calibrate_paths(s["lights"], masters, bayer=bayer, output=output,
                                 progress=(lambda d, t, l="", k=k: progress(
                                     d, t, f"Nuit {k + 1} : calibration")) if progress else None)
        frames += imgs
        origin += [k] * len(imgs)
        if log:
            log(f"Nuit {k + 1} ({s.get('name', '')}) : {len(imgs)} images calibrées")
    if not frames:
        raise ValueError("Aucune image calibrée")
    first = [i for i, o in enumerate(origin) if o == 0]
    ref = frames[first[len(first) // 2]]
    aligned = []
    for i, f in enumerate(frames):
        aligned.append(f if f is ref else warp(f, estimate_transform(ref, f)))
        if progress:
            progress(i + 1, len(frames), "Alignement commun")
    del frames
    aligned = normalize_frames(aligned, ref, normalization, progress=progress)
    weights = None
    if weight_mode and weight_mode != "Aucune":
        from .subframes import analyze_image, compute_weights, flag_outliers
        rows = [dict(analyze_image(f), path=str(i)) for i, f in enumerate(aligned)]
        rows = flag_outliers(rows)
        weights = compute_weights(rows, weight_mode)
    out = S.stack(np.stack(aligned), method, sigma=sigma, weights=weights)
    info = {"n_frames": len(aligned), "per_night": [origin.count(k) for k in range(len(sess))],
            "weights": weights}
    return out, info
