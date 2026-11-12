# -*- coding: utf-8 -*-
"""Accélération GPU facultative (CuPy, cartes NVIDIA/CUDA ou AMD/ROCm).

Activée si CuPy est installé (``pip install astroforge[gpu]``, soit
``cupy-cuda12x``) ET demandée : réglage ``"gpu": true`` de settings.json ou
variable ``ASTROFORGE_GPU=1``. En cas d'absence ou d'erreur (mémoire GPU
insuffisante, pilote), le calcul repasse automatiquement sur le processeur.

Utilisé par l'empilement (moyenne, médiane, sigma-clipping) : ces réductions
par pixel sur une pile N×H×W sont limitées par la bande passante mémoire et
gagnent nettement à tourner sur GPU.
"""
from __future__ import annotations

import os

_module = None          # module CuPy (ou substitut injecté par les tests)
_checked = False
_forced: bool | None = None


def _load():
    global _module, _checked
    if not _checked:
        _checked = True
        try:
            import cupy  # type: ignore
            cupy.zeros(1).sum()          # vérifie pilote / périphérique
            _module = cupy
        except Exception:
            _module = None
    return _module


def available() -> bool:
    return _load() is not None


def requested() -> bool:
    if _forced is not None:
        return _forced
    env = os.environ.get("ASTROFORGE_GPU")
    if env is not None:
        return env.strip().lower() in ("1", "true", "yes", "on", "oui")
    try:
        from .equipment import load_settings
        return bool(load_settings().get("gpu", False))
    except Exception:
        return False


def enabled() -> bool:
    return requested() and available()


def xp():
    """Module de calcul actif : CuPy si activé, sinon NumPy."""
    import numpy as np
    return _load() if enabled() else np


def set_enabled(flag: bool | None):
    """Force l'activation (True/False) ou revient au réglage (None)."""
    global _forced
    _forced = flag


def _inject(module):
    """Tests : remplace CuPy par un module compatible (asarray/asnumpy)."""
    global _module, _checked
    _module, _checked = module, True


def to_device(a):
    return _load().asarray(a)


def to_host(a):
    m = _load()
    return m.asnumpy(a) if hasattr(m, "asnumpy") else a


def info() -> str:
    m = _load()
    if m is None:
        return "GPU : CuPy non disponible (calcul sur processeur)"
    try:
        dev = m.cuda.Device()
        name = m.cuda.runtime.getDeviceProperties(dev.id)["name"]
        name = name.decode() if isinstance(name, bytes) else name
    except Exception:
        name = "périphérique CuPy"
    return f"GPU : {name} ({'activé' if enabled() else 'disponible, non activé'})"
