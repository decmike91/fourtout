# -*- coding: utf-8 -*-
"""Étirements GHS / arcsinh, masques, réduction et recombinaison d'étoiles,
débruitage de chrominance."""
import numpy as np
import pytest

from astroforge.core import finishing as F
from astroforge.core import masks as MK
from astroforge.core.stretch import arcsinh_stretch, ghs_curve


@pytest.mark.parametrize("b", [-1.0, -0.5, 0.0, 1.0, 4.0])
def test_ghs_monotonic_endpoints_and_symmetry_point(b):
    x = np.linspace(0, 1, 2001)
    y = ghs_curve(x, D=4.0, b=b, SP=0.1)
    assert y[0] == pytest.approx(0, abs=1e-9) and y[-1] == pytest.approx(1, abs=1e-9)
    assert np.all(np.diff(y) >= -1e-12)
    slope = np.diff(y) / np.diff(x)
    assert abs(x[np.argmax(slope)] - 0.1) < 0.01          # contraste max. au point SP


def test_ghs_protection_linear_segments():
    x = np.linspace(0, 1, 1001)
    y = ghs_curve(x, D=5.0, b=0.0, SP=0.3, LP=0.1, HP=0.8)
    lo = np.diff(y[x < 0.1])
    hi = np.diff(y[x > 0.8])
    assert np.allclose(lo, lo[0], rtol=1e-6) and np.allclose(hi, hi[0], rtol=1e-6)
    assert np.allclose(ghs_curve(x, D=0.0), x)


def test_arcsinh_preserves_colour_ratios():
    rgb = np.array([[[0.02, 0.01, 0.005], [0.2, 0.1, 0.05]]], np.float32)
    out = arcsinh_stretch(rgb, stretch=200.0)
    r_in = rgb[..., 0] / rgb[..., 1]
    r_out = out[..., 0] / out[..., 1]
    assert np.allclose(r_in, r_out, rtol=1e-4) and out[0, 0, 0] > rgb[0, 0, 0]
    assert np.allclose(arcsinh_stretch(rgb, 0.0), rgb)


def _nebula_with_stars(rng):
    yy, xx = np.mgrid[0:160, 0:160]
    neb = 0.15 * np.exp(-((xx - 80) ** 2 + (yy - 80) ** 2) / (2 * 40 ** 2))
    stars = np.zeros((160, 160), np.float32)
    pos = rng.uniform(10, 150, (25, 2))
    for y, x in pos:
        stars += 0.7 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.3 ** 2))
    base = (0.05 + neb).astype(np.float32)
    img = np.clip(base + stars + rng.normal(0, 0.002, base.shape), 0, 1).astype(np.float32)
    return img, base, stars, pos


def test_star_mask_targets_stars_not_nebula(rng):
    img, base, stars, pos = _nebula_with_stars(rng)
    m = MK.star_mask(img)
    on = np.mean([m[int(round(y)), int(round(x))] for y, x in pos])
    assert on > 0.8 and float(m[stars < 1e-4].mean()) < 0.05


def test_luminance_mask_ranges():
    ramp = np.tile(np.linspace(0, 1, 101, dtype=np.float32), (4, 1))
    m = MK.luminance_mask(ramp, 0.4, 0.6, fuzz=0.05, blur=0)
    assert m[0, 50] == pytest.approx(1) and m[0, 10] == pytest.approx(0) and m[0, 90] == pytest.approx(0)
    inv = MK.make_mask(ramp, "Luminance inversée (fond)", 0.4, 0.6, 0.05, 0)
    assert np.allclose(inv, 1 - m)


def test_reduce_stars_keeps_nebula(rng):
    img, base, stars, pos = _nebula_with_stars(rng)
    out = F.apply_filter(img, "star_reduction", {"amount": 1.0, "iterations": 2})
    star_flux_before = (img - base)[stars > 0.1].sum()
    star_flux_after = (out - base)[stars > 0.1].sum()
    assert star_flux_after < 0.6 * star_flux_before
    far = MK.star_mask(img, blur=0, grow=4) < 0.01
    assert np.abs(out - img)[far].max() < 1e-3


def test_extract_and_recombine_roundtrip(rng):
    img, base, stars, pos = _nebula_with_stars(rng)
    st = MK.extract_stars(img, base)
    assert np.allclose(MK.recombine(base, st, "Addition"), np.clip(base + st, 0, 1), atol=1e-6)
    scr = MK.recombine(base, st, "Écran")
    assert scr.max() <= 1 and np.all(scr >= base - 1e-6)


def test_chroma_denoise_keeps_luminance(rng):
    flat = np.full((96, 96, 3), 0.3, np.float32)
    noisy = np.clip(flat + rng.normal(0, 0.03, flat.shape).astype(np.float32), 0, 1)
    out = MK.chroma_denoise(noisy, 0.8)
    lum = lambda a: 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    chroma = lambda a: (a[..., 0] - lum(a)).std() + (a[..., 2] - lum(a)).std()
    assert chroma(out) < 0.5 * chroma(noisy)
    assert np.abs(lum(out) - lum(noisy)).max() < 2e-3


def test_masked_filter_and_sequence(rng):
    img, *_ = _nebula_with_stars(rng)
    rgb = np.repeat(img[..., None], 3, axis=2)
    out = F.apply_filter_masked(rgb, "saturation", {"amount": 0.0},
                                {"kind": "Luminance", "low": 0.5, "high": 1.0, "fuzz": 0.01, "blur": 0})
    dark = img < 0.3
    assert np.allclose(out[dark], rgb[dark], atol=1e-6)
    from astroforge.core import sequence as Q
    seq = {"mode": "image", "ops": [{"op": "fin_ghs", "params": {"D": 3.0, "SP": 0.05}},
                                    {"op": "fin_masked", "params": {
                                        "filter": "star_reduction", "params": {"amount": 1.0},
                                        "mask": {"kind": "Aucun"}}}]}
    res = Q.apply_to_image(seq, rgb)
    assert res.shape == rgb.shape and res.mean() > rgb.mean()


def test_finishing_gui_split_merge_and_mask(rng):
    from PySide6.QtWidgets import QApplication
    from astroforge.gui.page_common import PageCommon
    app = QApplication.instance() or QApplication([])
    img, base, *_ = _nebula_with_stars(rng)
    page = PageCommon()
    page.set_image(np.repeat(img[..., None], 3, axis=2))
    page.split_stars(np.repeat(base[..., None], 3, axis=2))
    assert np.allclose(page.image, np.repeat(base[..., None], 3, axis=2))
    page.cmbStarsMode.setCurrentText("Addition")
    page.merge_stars()
    # étoiles = max(image − sans étoiles, 0) : recombinaison = max(image, sans étoiles)
    expected = np.maximum(img, base)
    assert np.allclose(page.image, np.repeat(expected[..., None], 3, axis=2), atol=1e-5)
    page.cmbMask.setCurrentText("Étoiles")
    page.chkMaskShow.setChecked(True)
    p = page._preview_params()
    shown = page._compute(page.image, p)
    assert shown.ndim == 2 and shown.max() > 0.5
    idx = page.cmbFin.findData("ghs")
    assert idx > 0
    page.cmbFin.setCurrentIndex(idx)
    assert page._fin_state()[0] == "ghs"
    page.deleteLater()
    app.processEvents()
