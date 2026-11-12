# -*- coding: utf-8 -*-
"""Bande étroite : extraction Ha/OIII duo-bande, palettes, LRGB, calibration."""
import numpy as np
import pytest

from astroforge.core import narrowband as NB


def _cfa_from(ha, oiii, pattern="RGGB"):
    h, w = ha.shape
    c = np.zeros((h * 2, w * 2), np.float32)
    ry, rx = NB.OFFSETS[pattern]
    by, bx = 1 - ry, 1 - rx
    c[ry::2, rx::2] = ha
    c[ry::2, bx::2] = oiii
    c[by::2, rx::2] = oiii
    c[by::2, bx::2] = oiii
    return c


@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_extract_ha_oiii_exact(pattern, rng):
    ha = rng.random((20, 30)).astype(np.float32)
    o3 = rng.random((20, 30)).astype(np.float32)
    h2, o2 = NB.extract_ha_oiii(_cfa_from(ha, o3, pattern), pattern, upscale=False)
    assert np.allclose(h2, ha) and np.allclose(o2, o3)
    hu, ou = NB.extract_ha_oiii(_cfa_from(ha, o3, pattern), pattern, upscale=True)
    assert hu.shape == (40, 60)


def test_palettes_and_normalization(rng):
    ha = (0.1 + 0.05 * rng.random((32, 32))).astype(np.float32)
    o3 = (0.02 + 0.01 * rng.random((32, 32))).astype(np.float32)
    s2 = (0.05 + 0.02 * rng.random((32, 32))).astype(np.float32)
    hoo = NB.compose({"Ha": ha, "OIII": o3}, "HOO", normalize=False)
    assert np.allclose(hoo[..., 0], ha) and np.allclose(hoo[..., 1], o3) and np.allclose(hoo[..., 2], o3)
    sho = NB.compose({"Ha": ha, "OIII": o3, "SII": s2}, "SHO (Hubble)", normalize=True)
    assert abs(np.median(sho[..., 2]) - np.median(ha)) < 1e-4    # OIII égalisé sur Ha
    with pytest.raises(ValueError):
        NB.compose({"Ha": ha}, "SHO (Hubble)")


def test_lrgb_keeps_hue_and_takes_luminance(rng):
    rgb = np.stack([np.full((8, 8), v, np.float32) for v in (0.3, 0.2, 0.1)], -1)
    L = np.full((8, 8), 0.4, np.float32)
    out = NB.lrgb(L, rgb)
    Y = 0.2126 * out[..., 0] + 0.7152 * out[..., 1] + 0.0722 * out[..., 2]
    assert np.allclose(Y, 0.4, atol=1e-5)
    assert np.allclose(out[..., 0] / out[..., 2], 3.0, rtol=1e-4)


def test_align_channels(star_field):
    a = star_field(dx=0, dy=0)
    b = star_field(dx=3.2, dy=-1.7)
    out = NB.align_channels({"Ha": a, "OIII": b}, "Ha")
    from astroforge.core.registration import phase_corr
    assert np.hypot(*phase_corr(out["Ha"], out["OIII"])[:2]) < 0.15


def test_calibrate_paths_duo_band(write_fits, rng):
    from astroforge.core.calibration import calibrate_paths
    ha = np.full((16, 16), 0.4, np.float32)
    o3 = np.full((16, 16), 0.1, np.float32)
    p = write_fits("osc.fits", _cfa_from(ha, o3), BAYERPAT="RGGB")
    h = calibrate_paths([p], {}, cosmetic=False, output="Ha (duo-bande)")[0]
    o = calibrate_paths([p], {}, cosmetic=False, output="OIII (duo-bande)")[0]
    assert h.shape == (32, 32) and abs(h.mean() - 0.4) < 1e-3 and abs(o.mean() - 0.1) < 1e-3


def test_compose_dialog_runs(tmp_path, rng):
    from PySide6.QtWidgets import QApplication
    from astroforge.core.image_io import save_image
    from astroforge.gui.tools_menu import ComposeDialog
    app = QApplication.instance() or QApplication([])
    for name, v in (("ha", 0.3), ("o3", 0.1)):
        save_image(str(tmp_path / f"{name}.tif"), np.full((24, 24), v, np.float32)
                   + rng.normal(0, .005, (24, 24)).astype(np.float32))
    dlg = ComposeDialog()
    dlg.rows["Ha"].edit.setText(str(tmp_path / "ha.tif"))
    dlg.rows["OIII"].edit.setText(str(tmp_path / "o3.tif"))
    dlg.cmbPal.setCurrentText("HOO")
    dlg.chkAlign.setChecked(False)
    dlg.chkNorm.setChecked(False)
    res = dlg.run_sync()
    assert res.shape == (24, 24, 3) and dlg.result_image is not None
    assert abs(res[..., 0].mean() - 0.3) < 0.01 and abs(res[..., 2].mean() - 0.1) < 0.01
    dlg.deleteLater()
    app.processEvents()
