# -*- coding: utf-8 -*-
"""Analyse des brutes : FWHM/excentricité, rejet des mauvaises images, poids."""
import numpy as np
import pytest

from astroforge.core import stacking as S
from astroforge.core import subframes as SF


def _frame(rng, fwhm=2.5, ecc_ratio=1.0, bg=0.05, amp=1.0, noise=0.003, n=50, seed=0):
    r = np.random.default_rng(seed)
    h = w = 200
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full((h, w), bg, np.float32)
    sx = fwhm / 2.3548
    sy = sx / ecc_ratio
    for _ in range(n):
        x, y = r.uniform(12, w - 12), r.uniform(12, h - 12)
        img += (amp * r.uniform(0.1, 0.6)
                * np.exp(-((xx - x) ** 2 / (2 * sx * sx) + (yy - y) ** 2 / (2 * sy * sy)))
                ).astype(np.float32)
    return img + rng.normal(0, noise, img.shape).astype(np.float32)


@pytest.mark.parametrize("sx,sy", [(1.0, 1.0), (2.0, 2.0), (2.5, 1.2)])
def test_star_shape_accuracy(sx, sy):
    yy, xx = np.mgrid[0:64, 0:64]
    img = (0.05 + 0.8 * np.exp(-((xx - 31.6) ** 2 / (2 * sx * sx)
                                 + (yy - 32.4) ** 2 / (2 * sy * sy)))).astype(np.float32)
    fwhm, ecc, cx, cy = SF.star_shape(img, 32, 32, 0.05)
    assert abs(fwhm - 2.3548 * np.sqrt((sx * sx + sy * sy) / 2)) < 0.03
    assert abs(ecc - np.sqrt(1 - (min(sx, sy) / max(sx, sy)) ** 2)) < 0.01
    assert abs(cx - 31.6) < 0.01 and abs(cy - 32.4) < 0.01


def test_flags_bad_frames_and_weights(tmp_path, rng):
    from astroforge.core.image_io import save_image
    frames = [_frame(rng, seed=1) for _ in range(6)]
    frames.append(_frame(rng, fwhm=5.0, seed=1))             # mise au point
    frames.append(_frame(rng, amp=0.25, bg=0.12, seed=1))    # nuage
    frames.append(_frame(rng, ecc_ratio=2.5, seed=1))        # suivi
    paths = []
    for i, f in enumerate(frames):
        p = str(tmp_path / f"f{i}.tif")
        save_image(p, f)
        paths.append(p)
    rows = SF.flag_outliers(SF.analyze_paths(paths))
    keep = [r["keep"] for r in rows]
    assert keep == [True] * 6 + [False, False, False], [r["reasons"] for r in rows]
    assert abs(rows[0]["fwhm"] - 2.5) < 0.3 and rows[8]["ecc"] > 0.65
    w = SF.compute_weights(rows, "Mixte")
    assert w[6:] == [0.0, 0.0, 0.0] and abs(np.mean(w[:6]) - 1.0) < 1e-9
    assert "rejeter" in SF.report(rows)


def test_weighted_stack_reduces_noise(rng):
    clean = np.full((64, 64), 0.2, np.float32)
    good = [clean + rng.normal(0, 0.01, clean.shape).astype(np.float32) for _ in range(4)]
    bad = [clean + rng.normal(0, 0.05, clean.shape).astype(np.float32) for _ in range(2)]
    a = np.stack(good + bad)
    w = [1 / 0.01 ** 2] * 4 + [1 / 0.05 ** 2] * 2
    for m in ("Moyenne", "Sigma-clipping", "Winsorisé", "MAD clipping", "Linear fit"):
        assert S.stack(a, m, weights=w).std() < S.stack(a, m).std()
    with pytest.raises(ValueError):
        S.stack(a, "Moyenne", weights=[1, 2])


def test_subframe_dialog(rng, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from astroforge.gui.tools_dialogs import SubframeDialog
    app = QApplication.instance() or QApplication([])
    rows = [{"path": f"/x/f{i}.fits", "stars": 40, "fwhm": 2.5 + 0.1 * i, "ecc": 0.1,
             "background": 0.05, "noise": 0.003, "signal": 0.3, "snr": 100.0,
             "keep": True, "reasons": []} for i in range(4)]
    dlg = SubframeDialog(rows)
    dlg.table.item(1, 0).setCheckState(Qt.Unchecked)
    kept, weights = dlg.result_selection()
    assert kept == ["/x/f0.fits", "/x/f2.fits", "/x/f3.fits"]
    assert weights["/x/f0.fits"] > weights["/x/f3.fits"]  # FWHM plus fine
    dlg.close()
    dlg.deleteLater()
    app.processEvents()
