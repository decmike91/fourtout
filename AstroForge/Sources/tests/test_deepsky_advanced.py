# -*- coding: utf-8 -*-
"""Multi-nuits + normalisation locale, mosaïques, comète, empilement en direct."""

import numpy as np

from astroforge.core import comet as CO
from astroforge.core import mosaic as MO
from astroforge.core.normalization import global_normalize, local_normalize
from astroforge.core.registration import apply_shift, phase_corr


def test_local_normalization_removes_night_gradient(star_field, rng):
    ref = star_field(256, 256, seed=5) + rng.normal(0, 0.002, (256, 256)).astype(np.float32)
    yy, xx = np.mgrid[0:256, 0:256]
    # autre nuit : transparence 0,8 (signal ET bruit), fond plus haut et gradient
    other = (ref * 0.8 + 0.03 + 0.04 * xx / 256).astype(np.float32)
    ln = local_normalize(other, ref, tile=64)
    gn = global_normalize(other, ref)
    import scipy.ndimage as ndi
    # résidu basse fréquence (le bruit indépendant des deux images s'annule)
    r_ln = np.abs(ndi.gaussian_filter(ln - ref, 8))[16:-16, 16:-16]
    r_gn = np.abs(ndi.gaussian_filter(gn - ref, 8))[16:-16, 16:-16]
    assert r_ln.mean() < 0.3 * r_gn.mean() and r_ln.mean() < 0.002


def test_multi_night_stack(tmp_path, write_fits, star_field, rng):
    from astroforge.core.multisession import stack_sessions
    truth = star_field(96, 96, seed=9)
    for night, (gain, offs) in enumerate(((1.0, 0.0), (0.7, 0.04))):
        for i in range(4):
            dx, dy = rng.uniform(-3, 3, 2)
            img = apply_shift(truth, dx, dy) * gain + offs + 0.01  # + dark
            img = img + rng.normal(0, 0.003, img.shape)
            write_fits(f"n{night}/light_{i}.fits", img, IMAGETYP="Light", EXPTIME=60.0)
            write_fits(f"n{night}/dark_{i}.fits", np.full((96, 96), 0.01), IMAGETYP="Dark",
                       EXPTIME=60.0)
    out, info = stack_sessions([str(tmp_path / "n0"), str(tmp_path / "n1")],
                               normalization="Locale")
    assert info["per_night"] == [4, 4] and out.shape == (96, 96)
    d = phase_corr(truth, out)[:2]
    t = apply_shift(truth, -d[0], -d[1]) if np.hypot(*d) > 0.5 else truth
    assert np.corrcoef(out[8:-8, 8:-8].ravel(), t[8:-8, 8:-8].ravel())[0, 1] > 0.98


def _sky(rng, H=300, W=420):
    yy, xx = np.mgrid[0:H, 0:W]
    sky = 0.05 + 0.2 * np.exp(-((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (2 * 80 ** 2))
    for y, x in rng.uniform(5, [H - 5, W - 5], (220, 2)):
        sky += rng.uniform(.1, .7) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.2 ** 2))
    return sky.astype(np.float32)


def test_mosaic_wcs_and_features(rng):
    from astropy.wcs import WCS
    sky = _sky(rng)
    ref = WCS(naxis=2)
    ref.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    ref.wcs.crval = [83.8, -5.4]
    ref.wcs.crpix = [211, 151]
    s = 2 / 3600
    ref.wcs.cd = [[-s, 0], [0, s]]
    panels, wcss = [], []
    for k, (x0, y0) in enumerate([(0, 0), (170, 0), (0, 120), (170, 120)]):
        p = sky[y0:y0 + 180, x0:x0 + 250] + 0.01 * k
        w = ref.deepcopy()
        w.wcs.crpix = [211 - x0, 151 - y0]
        panels.append(p.astype(np.float32))
        wcss.append(w)
    out, wout, cov = MO.mosaic_wcs(panels, wcss)
    assert cov.mean() > 0.95
    px = wout.wcs_world2pix([ref.wcs_pix2world([[0, 0]], 0)[0]], 0)[0]
    x0, y0 = int(round(px[0])), int(round(px[1]))
    part = out[y0 + 10:y0 + 290, x0 + 10:x0 + 410]
    assert np.corrcoef(part.ravel(), sky[10:290, 10:410].ravel())[0, 1] > 0.99
    out2, cov2, mats = MO.mosaic_features(panels)
    offs = [tuple(np.round(m[:, 2]).astype(int)) for m in mats]
    assert offs == [(0, 0), (170, 0), (0, 120), (170, 120)]


def test_comet_stack_separates_stars_and_comet(star_field, rng):
    n = 8
    frames, pos = [], []
    yy, xx = np.mgrid[0:128, 0:128]
    for i in range(n):
        cx, cy = 30 + 8 * i, 40 + 5 * i
        com = 0.5 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0 ** 2))
        frames.append((star_field(seed=4) + com + rng.normal(0, .002, (128, 128))).astype(np.float32))
        pos.append((cx + rng.normal(0, 0.5), cy + rng.normal(0, 0.5)))
    interp = CO.interpolate_positions(n, pos[0], pos[-1])
    res = CO.comet_stack(frames, interp, method="Sigma-clipping", sigma=2.0)
    stars = star_field(seed=4)
    mid = res["positions"][n // 2]
    # étoiles : la comète mobile est rejetée ; comète : nette au centre
    assert np.abs(res["stars"] - stars).mean() < 0.01
    cy, cx = int(round(mid[1])), int(round(mid[0]))
    assert res["comet"][cy, cx] > 0.45 + np.median(res["comet"])
    assert res["composite"][cy, cx] >= res["comet"][cy, cx] - 1e-6
    assert CO.parse_times(["2026-01-01T00:00:00", "2026-01-01T00:01:00"]) == [0.0, 60.0]


def test_live_stacker_rejects_bad_and_trails(tmp_path, star_field, rng):
    from astroforge.core.image_io import save_image
    from astroforge.core.livestack import LiveStacker, run_folder
    base = star_field(128, 128, seed=2)
    for i in range(8):
        img = apply_shift(base, *rng.uniform(-2, 2, 2)) + rng.normal(0, 0.003, base.shape)
        if i == 6:
            img[60:62, :] += 0.5          # satellite
        save_image(str(tmp_path / f"f{i:02d}.tif"), np.clip(img, 0, 1))
    save_image(str(tmp_path / "f99_nuage.tif"), np.full((128, 128), 0.3, np.float32))
    st = run_folder(str(tmp_path), LiveStacker(kappa=3.0))
    assert len(st.accepted) == 8 and len(st.rejected) == 1
    out = st.image()
    assert out[60:62, 20:108].mean() < base[60:62, 20:108].mean() + 0.03


def test_tool_dialogs_build(tmp_path, rng):
    from PySide6.QtWidgets import QApplication
    from astroforge.core.image_io import save_image
    from astroforge.gui import tools_extra as TX
    from astroforge.gui.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    mw = MainWindow()
    labels = [a.text() for a in mw.menu_tools.actions() if a.text()]
    assert any("Mosaïque" in t for t in labels) and any("direct" in t for t in labels)
    sky = _sky(rng, 160, 200)
    for k, (x0, y0) in enumerate([(0, 0), (90, 0)]):
        save_image(str(tmp_path / f"p{k}.tif"), sky[:, x0:x0 + 110])
    dlg = TX.MosaicDialog(mw)
    dlg.files.edit.setText(";".join(str(tmp_path / f"p{k}.tif") for k in range(2)))
    img, how = dlg.run_sync()
    assert how == "points d'intérêt" and img.shape[1] >= 195
    dlg.send_to_finishing()
    assert mw.page_common.image is not None
    mw.win.close()
    mw.win.deleteLater()
    app.processEvents()
