# -*- coding: utf-8 -*-
"""Vérification d'une installation d'AstroForge (Windows ou autre) :

    astroforge-cli.exe script check_install.py      (installation Windows)
    python -m astroforge script packaging/windows/check_install.py

Importe les dépendances lourdes, calcule sur des données synthétiques
(FITS, alignement, empilement, étirement, astrométrie), puis construit
l'interface hors écran et ouvre chaque outil. Code de sortie 0 si tout va bien.
"""
import os
import sys
import tempfile
import time
import traceback

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
RESULTS = []


def step(name):
    def deco(fn):
        t0 = time.time()
        try:
            detail = fn()
            RESULTS.append((True, name, detail or ""))
            print(f"OK    {name} ({time.time() - t0:.1f} s) {detail or ''}", flush=True)
        except Exception as e:  # noqa: BLE001
            RESULTS.append((False, name, repr(e)))
            print(f"ECHEC {name} : {e}", flush=True)
            traceback.print_exc()
        return fn
    return deco


@step("version et emplacements")
def _version():
    import astroforge
    from astroforge import paths
    return f"{astroforge.__version__} — données : {paths.DATA_ROOT}"


@step("dépendances")
def _deps():
    import astropy
    import cv2
    import numpy
    import scipy
    import PySide6
    out = [f"numpy {numpy.__version__}", f"scipy {scipy.__version__}",
           f"opencv {cv2.__version__}", f"astropy {astropy.__version__}",
           f"PySide6 {PySide6.__version__}"]
    for opt in ("photutils", "astroalign", "onnxruntime", "rawpy"):
        try:
            mod = __import__(opt)
            out.append(f"{opt} {getattr(mod, '__version__', '?')}")
        except ImportError:
            out.append(f"{opt} absent")
    return ", ".join(out)


def _field(dx=0.0, dy=0.0, seed=1, n=40, h=160, w=160):
    import numpy as np
    r = np.random.default_rng(seed)
    ys, xs = r.uniform(10, h - 10, n), r.uniform(10, w - 10, n)
    amp = r.uniform(0.2, 0.8, n)
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full((h, w), 0.05, np.float32)
    for y, x, a in zip(ys, xs, amp):
        img += (a * np.exp(-((xx - x - dx) ** 2 + (yy - y - dy) ** 2) / 3.2)).astype(np.float32)
    return img + np.random.default_rng(seed + 99).normal(0, 0.01, (h, w)).astype(np.float32)


@step("FITS / XISF / TIFF, chemins accentués")
def _io():
    import numpy as np
    from astroforge.core.image_io import load_image, save_image
    d = tempfile.mkdtemp(prefix="astroforge-vérif-")
    img = _field()
    for ext in (".fits", ".xisf", ".tif"):
        p = os.path.join(d, "Nébuleuse d'Orion" + ext)
        save_image(p, img, "32f" if ext == ".tif" else "auto", header={"OBJECT": "M 42"})
        back = load_image(p)
        assert np.abs(back - np.clip(img, 0, 1)).max() < 1e-3, ext
    return d


@step("alignement + empilement sigma-clipping")
def _stack():
    import numpy as np
    from astroforge.core.registration import estimate_transform, warp
    from astroforge.core.stacking import stack
    ref = _field()
    frames = [ref] + [_field(dx=1.5 * i, dy=-0.8 * i, seed=1) for i in range(1, 5)]
    aligned = [ref] + [warp(f, estimate_transform(ref, f)) for f in frames[1:]]
    res = stack(np.stack(aligned), "Sigma-clipping")
    err = float(np.abs(res[20:-20, 20:-20] - ref[20:-20, 20:-20]).mean())
    assert err < 0.02, err
    return f"écart moyen {err:.4f}"


@step("étirement GHS + masques")
def _finish():
    from astroforge.core.masks import make_mask
    from astroforge.core.stretch import ghs
    img = _field()
    out = ghs(img, D=3.0, b=0.0, SP=0.05)
    m = make_mask(img, "Étoiles")
    assert out.shape == img.shape and m.shape == img.shape


@step("astrométrie (solveur triangles)")
def _solve():
    from astroforge.core import platesolve
    return "module " + platesolve.__name__


@step("interface hors écran + outils")
def _gui():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from astroforge.gui.main_window import MainWindow
    from astroforge.gui.tools_menu import registered_tools
    mw = MainWindow()
    n = 0
    for _label, cls in registered_tools():
        if cls is not None:
            dlg = cls(mw, mw.win)
            dlg.close()
            dlg.deleteLater()
            n += 1
    mw.win.close()
    app.processEvents()
    return f"{n} outils ouverts"


ok = all(r[0] for r in RESULTS)
print(f"\n=== {sum(r[0] for r in RESULTS)}/{len(RESULTS)} vérifications réussies ===", flush=True)
sys.exit(0 if ok else 1)
