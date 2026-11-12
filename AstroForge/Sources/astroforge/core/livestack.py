# -*- coding: utf-8 -*-
"""
Empilement en direct (« live stacking », esprit SharpCap / ASIAIR) : chaque
nouvelle image d'un dossier surveillé est calibrée (masters facultatifs),
dématricée, contrôlée (étoiles, FWHM), alignée sur la première image valide
puis ajoutée à une moyenne courante avec rejet kappa-sigma progressif
(moyenne et variance de Welford par pixel : un pixel à plus de κ·σ de la
moyenne courante — satellite, avion, rayon cosmique — n'est pas ajouté).
"""
from __future__ import annotations

import os
import time

import numpy as np

from .calibration import CALIB_EXTENSIONS, calibrate
from .image_io import debayer, load_cfa
from .registration import estimate_transform, warp


class LiveStacker:
    def __init__(self, masters: dict | None = None, bayer: str | None = None,
                 kappa: float = 3.0, min_frames_reject: int = 5,
                 max_fwhm_ratio: float = 1.6, min_stars: int = 5):
        self.masters = masters or {}
        self.bayer = bayer
        self.kappa = float(kappa)
        self.min_frames_reject = int(min_frames_reject)
        self.max_fwhm_ratio = float(max_fwhm_ratio)
        self.min_stars = int(min_stars)
        self.reset()

    def reset(self):
        self.ref = None
        self.mean = None
        self.m2 = None
        self.count = None
        self.n = 0
        self.rejected = []          # (fichier, raison)
        self.accepted = []
        self.fwhms = []

    # ------------------------------------------------------------ étapes
    def prepare(self, path_or_img) -> np.ndarray:
        if isinstance(path_or_img, str):
            raw, pattern = load_cfa(path_or_img)
        else:
            raw, pattern = np.asarray(path_or_img, np.float32), None
        pattern = (self.bayer or pattern) if raw.ndim == 2 else None
        m = self.masters
        cal = calibrate(raw, m.get("bias"), m.get("dark"), m.get("flat"),
                        cosmetic=True, cfa=pattern is not None)
        if pattern is not None:
            cal = debayer(np.clip(cal, 0, 1), pattern)
        return np.asarray(cal, np.float32)

    def add(self, path_or_img, name: str | None = None) -> bool:
        """Ajoute une image ; retourne False si elle est rejetée."""
        from .subframes import analyze_image
        name = name or (path_or_img if isinstance(path_or_img, str) else f"image {self.n + 1}")
        try:
            img = self.prepare(path_or_img)
        except Exception as e:
            self.rejected.append((name, f"illisible : {e}"))
            return False
        q = analyze_image(img)
        if q["stars"] < self.min_stars:
            self.rejected.append((name, f"{q['stars']} étoiles"))
            return False
        if self.fwhms and np.isfinite(q["fwhm"]) and \
                q["fwhm"] > self.max_fwhm_ratio * float(np.median(self.fwhms)):
            self.rejected.append((name, f"FWHM {q['fwhm']:.2f} px"))
            return False
        if self.ref is None:
            self.ref = img
            aligned = img
        else:
            if img.shape != self.ref.shape:
                self.rejected.append((name, "taille différente"))
                return False
            M = estimate_transform(self.ref, img)
            if not np.isfinite(M).all() or abs(np.linalg.det(M[:, :2]) - 1) > 0.05:
                self.rejected.append((name, "alignement impossible"))
                return False
            aligned = warp(img, M)
        if np.isfinite(q["fwhm"]):
            self.fwhms.append(q["fwhm"])
        self._accumulate(aligned)
        self.accepted.append(name)
        return True

    def _accumulate(self, x: np.ndarray):
        x = x.astype(np.float64)
        if self.mean is None:
            self.mean = x.copy()
            self.m2 = np.zeros_like(x)
            self.count = np.ones(x.shape, np.float64)
            self.n = 1
            return
        self.n += 1
        if self.n > self.min_frames_reject:
            sd = np.sqrt(self.m2 / np.maximum(self.count - 1, 1))
            ok = np.abs(x - self.mean) <= self.kappa * np.maximum(sd, 1e-4)
        else:
            ok = np.ones(x.shape, bool)
        c = self.count + ok
        delta = np.where(ok, x - self.mean, 0.0)
        self.mean = self.mean + delta / c
        self.m2 = self.m2 + np.where(ok, delta * (x - self.mean), 0.0)
        self.count = c

    def image(self) -> np.ndarray | None:
        return None if self.mean is None else np.clip(self.mean, 0, 1).astype(np.float32)

    def status(self) -> str:
        return (f"{len(self.accepted)} image(s) empilée(s), {len(self.rejected)} rejetée(s)"
                + (f" — FWHM médiane {np.median(self.fwhms):.2f} px" if self.fwhms else ""))


class FolderWatcher:
    """Détecte les nouveaux fichiers image d'un dossier (taille stable entre
    deux passages : l'écriture par le logiciel d'acquisition est terminée)."""

    def __init__(self, folder: str, include_existing: bool = False):
        self.folder = folder
        self.seen = set() if include_existing else set(self._list())
        self.sizes = {}

    def _list(self):
        try:
            return sorted(os.path.join(self.folder, f) for f in os.listdir(self.folder)
                          if f.lower().endswith(CALIB_EXTENSIONS))
        except OSError:
            return []

    def poll(self) -> list[str]:
        ready = []
        for p in self._list():
            if p in self.seen:
                continue
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            if sz > 0 and self.sizes.get(p) == sz:
                ready.append(p)
                self.seen.add(p)
                self.sizes.pop(p, None)
            else:
                self.sizes[p] = sz
        return ready


def run_folder(folder: str, stacker: LiveStacker, duration_s: float = 0.0,
               interval_s: float = 2.0, stop=None, on_update=None) -> LiveStacker:
    """Boucle de surveillance (scripts / ligne de commande). duration_s = 0 :
    traite les fichiers présents puis s'arrête."""
    watcher = FolderWatcher(folder, include_existing=True)
    t0 = time.time()
    while True:
        new = watcher.poll()
        if not new and duration_s <= 0 and not watcher.sizes:
            break
        for p in new:
            stacker.add(p)
            if on_update:
                on_update(stacker)
        if stop is not None and stop():
            break
        if duration_s > 0 and time.time() - t0 > duration_s:
            break
        time.sleep(interval_s if duration_s > 0 else 0.05)
    return stacker
