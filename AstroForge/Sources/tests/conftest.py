# -*- coding: utf-8 -*-
"""Configuration pytest : chemin du projet, Qt hors écran, données isolées."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# données utilisateur (réglages, bibliothèque de masters…) isolées du dépôt
if "ASTROFORGE_HOME" not in os.environ:
    import tempfile
    os.environ["ASTROFORGE_HOME"] = tempfile.mkdtemp(prefix="astroforge-tests-")
os.environ.pop("ASTROFORGE_LANG", None)
os.environ.pop("ASTROFORGE_GPU", None)
os.environ.pop("ASTROFORGE_UPDATE_URL", None)

import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def write_fits(tmp_path):
    """Écrit un FITS uint16 (comme une caméra) avec en-tête."""
    from astropy.io import fits

    def _w(name, data01, **header):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        hdr = fits.Header()
        for k, v in header.items():
            hdr[k.replace("_", "-") if k in ("CCD_TEMP", "DATE_OBS") else k] = v
        fits.writeto(p, (np.clip(data01, 0, 1) * 65535).astype(np.uint16), hdr, overwrite=True)
        return str(p)
    return _w


@pytest.fixture
def star_field():
    """Champ d'étoiles gaussiennes décalable (x, y en pixels)."""
    def _f(h=128, w=128, dx=0.0, dy=0.0, n=25, seed=3, fwhm=3.0, bg=0.05, amp=(0.2, 0.8)):
        r = np.random.default_rng(seed)
        ys, xs = r.uniform(10, h - 10, n), r.uniform(10, w - 10, n)
        a = r.uniform(*amp, n)
        yy, xx = np.mgrid[0:h, 0:w]
        s = fwhm / 2.3548
        img = np.full((h, w), bg, np.float32)
        for y, x, k in zip(ys, xs, a):
            img += (k * np.exp(-((xx - x - dx) ** 2 + (yy - y - dy) ** 2) / (2 * s * s))).astype(np.float32)
        return img
    return _f


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])
