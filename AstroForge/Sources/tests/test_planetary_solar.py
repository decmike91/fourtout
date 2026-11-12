# -*- coding: utf-8 -*-
"""Planétaire : multipoint local, horodatages SER, dérotation, WinJUPOS ;
solaire : limbe, composite protubérances."""
import datetime as dt

import cv2
import numpy as np

from astroforge.core import derotation as DR
from astroforge.core import planetary as PL
from astroforge.core import solar as SO


def _planet_frames(rng, n=40, amp=2.0):
    H = W = 160
    yy, xx = np.mgrid[0:H, 0:W]
    disc = ((xx - 80) ** 2 + (yy - 80) ** 2) < 60 ** 2
    tex = np.zeros((H, W), np.float32)
    for _ in range(60):
        y, x = rng.uniform(25, 135, 2)
        tex += rng.uniform(.05, .2) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * rng.uniform(1, 3) ** 2))
    truth = cv2.GaussianBlur((disc * (0.4 + tex)).astype(np.float32), (0, 0), 0.8)
    frames = []
    for _ in range(n):
        fx = cv2.resize(rng.normal(0, amp, (4, 4)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        fy = cv2.resize(rng.normal(0, amp, (4, 4)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
        gx, gy = rng.uniform(-4, 4, 2)
        f = cv2.remap(truth, (xx + fx + gx).astype(np.float32), (yy + fy + gy).astype(np.float32),
                      cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.5, 1.8))
        frames.append((f + rng.normal(0, .01, f.shape)).astype(np.float32))
    return truth, frames


def _hf_fidelity(img, truth):
    from astroforge.core.registration import apply_shift, phase_corr
    a = apply_shift(img, *phase_corr(truth, img)[:2])
    hp = lambda z: z - cv2.GaussianBlur(z, (0, 0), 2)
    c = slice(30, 130)
    return np.corrcoef(hp(a)[c, c].ravel(), hp(truth)[c, c].ravel())[0, 1]


def test_local_multipoint_beats_global_under_local_turbulence():
    rng = np.random.default_rng(0)
    truth, frames = _planet_frames(rng, n=60)
    kw = dict(quality_method="Contraste HF (AutoStakkert)", n_align=60, grid_n=5,
              box_size=32, stack_percent=50, stack_method="Moyenne (meilleurs %)")
    g = PL.run_planetary(frames, align_mode="Global (une translation)", **kw)
    loc = PL.run_planetary(frames, align_mode="Multi-points local (AS!)", **kw)
    assert loc["stats"]["n_points"] > 10
    assert _hf_fidelity(loc["image"], truth) > _hf_fidelity(g["image"], truth)


def test_ser_timestamps_roundtrip(tmp_path):
    from astroforge.core.image_io import load_ser, write_ser
    t0 = dt.datetime(2026, 10, 15, 21, 30, 0)
    ts = [t0 + dt.timedelta(milliseconds=20 * i) for i in range(5)]
    p = str(tmp_path / "jup.ser")
    write_ser(p, np.random.rand(5, 8, 8).astype(np.float32), timestamps=ts)
    fr, meta = load_ser(p)
    assert meta["timestamps"][4] == ts[4].isoformat(timespec="milliseconds")
    assert meta["date_utc"].startswith("2026-10-15T21:30:00")
    assert DR.mid_time(meta["timestamps"]) == (t0 + dt.timedelta(milliseconds=40)).isoformat(
        timespec="milliseconds")


def _render(t_sec, cx=100.3, cy=99.6, r=80.0, P=9.925, flat=0.0649, pa=20.0, B=3.0):
    yy, xx = np.mgrid[0:200, 0:200].astype(float)
    lon, lat, vis, z = DR._image_to_sphere(xx, yy, cx, cy, r, pa, flat, B)
    tex = 0.4 + 0.15 * np.sin(6 * (lon - 2 * np.pi * t_sec / (P * 3600))) * np.cos(3 * lat) \
        + 0.1 * np.sin(9 * lat)
    return np.where(vis, tex * (0.5 + 0.5 * z), 0.0).astype(np.float32)


def test_derotation_recovers_reference_time():
    img0, img_t = _render(0.0), _render(600.0)
    out, w = DR.derotate_image(img_t, (100.3, 99.6, 80.0), -600.0, 9.925, 20.0, 0.0649, 3.0)
    m = w > 0.4
    assert np.abs(out - img0)[m].mean() < 0.35 * np.abs(img_t - img0)[m].mean()


def test_derotate_series_and_winjupos():
    t0 = dt.datetime(2026, 10, 15, 21, 0, 0)
    imgs = [_render(s) for s in (-300.0, 0.0, 300.0)]
    times = [t0 + dt.timedelta(seconds=s) for s in (-300, 0, 300)]
    res, mid, disc = DR.derotate_series(imgs, times, pa=20.0, B=3.0)
    assert mid == "2026-10-15T21:00:00"
    inner = DR._image_to_sphere(*np.mgrid[0:200, 0:200][::-1].astype(float), 100.3, 99.6, 80.0,
                                20.0, 0.0649, 3.0)[3] > 0.6
    assert np.abs(res - imgs[1])[inner].mean() < 0.4 * np.abs(imgs[0] - imgs[1])[inner].mean()
    name = DR.winjupos_name(dt.datetime(2026, 10, 15, 21, 30, 25), "Obs Name", "R")
    assert name == "2026-10-15-2130_4-ObsName-R.png"
    assert DR.parse_winjupos_name(name) == dt.datetime(2026, 10, 15, 21, 30, 24)


def test_solar_limb_and_composite():
    yy, xx = np.mgrid[0:240, 0:240]
    rr = np.hypot(xx - 118.4, yy - 121.7)
    disc = np.where(rr < 90, 0.7 * (1 - 0.4 * (rr / 90) ** 2), 0.0)
    prom = 0.05 * np.exp(-((xx - 118) ** 2 + (yy - 26) ** 2) / 30.0)  # protubérance au-dessus
    img = (disc + prom + 0.01).astype(np.float32)
    cx, cy, r = SO.fit_disc(img)
    assert abs(cx - 118.4) < 0.6 and abs(cy - 121.7) < 0.6 and abs(r - 90) < 1.5
    flat = SO.flatten_limb_darkening(img, (cx, cy, r))
    inner = rr < 80
    assert flat[inner].std() < 0.3 * img[inner].std()
    comp, _ = SO.prominence_composite(img, (cx, cy, r), prom_gain=10.0)
    assert comp[26, 118] > 0.2 and comp[121, 118] < 0.6   # protubérance visible, disque inversé


def test_planetary_gui_local_mode_and_winjupos_time(tmp_path):
    from PySide6.QtWidgets import QApplication
    from astroforge.gui.page_planetary import PagePlanetary
    from astroforge.gui import tools_extra as TX
    app = QApplication.instance() or QApplication([])
    page = PagePlanetary()
    combo = page._o("cmbAlignKindP")
    assert combo.findText("Multi-points local (AS!)") >= 0
    t0 = dt.datetime(2026, 10, 15, 21, 0, 0)
    page.frame_meta = {"timestamps": [(t0 + dt.timedelta(seconds=i)).isoformat() for i in range(10)]}
    page.selected = [2, 3, 4]
    assert page.winjupos_time().startswith("2026-10-15T21:00:03")
    # dialogue de dérotation : instants lus dans les noms WinJUPOS
    from astroforge.core.image_io import save_image
    names = []
    for s in (-120, 0, 120):
        n = DR.winjupos_name(t0 + dt.timedelta(seconds=s), "AB", "R", "tif")
        save_image(str(tmp_path / n), _render(float(s)))
        names.append(str(tmp_path / n))
    dlg = TX.DerotationDialog()
    dlg.files.edit.setText(";".join(names))
    dlg.spnPA.setValue(20.0)
    dlg.spnB.setValue(3.0)
    img, mid, disc = dlg.run_sync()
    assert mid == "2026-10-15T21:00:00" and img.shape == (200, 200)
    page.deleteLater()
    dlg.deleteLater()
    app.processEvents()
