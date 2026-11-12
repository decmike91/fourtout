# -*- coding: utf-8 -*-
"""Solveur par triangles + SIP, annotation OpenNGC, courbe de lumière."""
import numpy as np
import pytest
from astropy.wcs import WCS

from astroforge.core import annotate as AN
from astroforge.core.platesolve import solve_triangles, umeyama


def _field(tmp_path, rng, rot=25.0, mirror=False, k1=0.0, ra0=150.0, dec0=20.0, n=400):
    ras = ra0 + rng.uniform(-.6, .6, n) / np.cos(np.radians(dec0))
    decs = dec0 + rng.uniform(-.6, .6, n)
    mags = rng.uniform(7, 14, n)
    cat = tmp_path / "cat.csv"
    with open(cat, "w") as f:
        f.write("ra,dec,mag\n")
        for a, d, m in zip(ras, decs, mags):
            f.write(f"{a},{d},{m}\n")
    w = WCS(naxis=2)
    w.wcs.crpix = [500, 400]
    w.wcs.crval = [ra0, dec0]
    s, t = 3 / 3600, np.radians(rot)
    cd = np.array([[-s * np.cos(t), s * np.sin(t)], [s * np.sin(t), s * np.cos(t)]])
    if mirror:
        cd = cd @ np.diag([-1, 1])
    w.wcs.cd = cd
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    x, y = w.world_to_pixel_values(ras, decs)
    dx, dy = x - 500, y - 400
    r2 = dx * dx + dy * dy
    x, y = 500 + dx * (1 + k1 * r2), 400 + dy * (1 + k1 * r2)
    img = np.full((800, 1000), 0.05, np.float32)
    yy, xx = np.mgrid[0:800, 0:1000]
    for xi, yi, m in zip(x, y, mags):
        if 8 < xi < 992 and 8 < yi < 792 and m < 13:
            sl = (slice(int(yi) - 8, int(yi) + 9), slice(int(xi) - 8, int(xi) + 9))
            img[sl] += (10 ** (-0.4 * (m - 7))) * 0.9 * np.exp(-((xx[sl] - xi) ** 2 + (yy[sl] - yi) ** 2) / 4.5)
    img += rng.normal(0, 0.002, img.shape).astype(np.float32)
    return img, str(cat), ras, decs, mags, x, y, w


def test_umeyama_with_reflection():
    src = np.array([[0, 0], [1, 0], [0, 2], [3, 1]], float)
    R = np.array([[0, 1], [1, 0]]) * 2.5
    dst = src @ R.T + [4, -1]
    M, t = umeyama(src, dst)
    assert np.allclose(M, R) and np.allclose(t, [4, -1])


@pytest.mark.parametrize("rot,mirror,k1", [(25, False, 0.0), (-70, True, 0.0), (10, False, 5e-8)])
def test_triangle_solver(tmp_path, rng, rot, mirror, k1):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    img, cat, ras, decs, mags, x, y, _w = _field(tmp_path, rng, rot, mirror, k1)
    res = solve_triangles(img, cat, 150.1, 19.92, 1.0)        # échelle inconnue
    sel = (mags < 12) & (x > 20) & (x < 980) & (y > 20) & (y < 780)
    p = res["wcs"].world_to_pixel(SkyCoord(ras[sel] * u.deg, decs[sel] * u.deg))
    err = np.hypot(p[0] - x[sel], p[1] - y[sel])
    assert abs(res["scale_arcsec_pix"] - 3.0) < 0.1
    assert np.median(err) < 0.1 and err.max() < 0.6 and res["n_matched"] >= 25
    if k1:
        assert res["sip"]


def test_solve_image_uses_triangles(tmp_path, rng):
    from astroforge.core.astrometry import solve_image
    img, cat, *_ = _field(tmp_path, rng)
    res = solve_image(img, cat, 150.05, 20.03, 1.0, prefer_solve_field=False)
    assert res["method"].startswith("triangles")


def test_annotation_finds_messier():
    objs = AN.load_objects()
    m42 = next(o for o in objs if o["label"] == "M42")
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [m42["ra"], m42["dec"]]
    w.wcs.crpix = [400, 300]
    s = 4 / 3600
    w.wcs.cd = [[-s, 0], [0, s]]
    found = AN.objects_in_field(w, (600, 800))
    labels = [o["label"] for o in found]
    assert labels[0] in ("M42", "M43") and "M42" in labels and "M43" in labels
    img, objs = AN.annotate(np.full((600, 800), 0.1, np.float32), w)
    assert img.shape == (600, 800, 3) and img.max() > 0.5   # ellipses dessinées
    o = next(x for x in found if x["label"] == "M42")
    assert abs(o["x"] - 399) < 1 and abs(o["y"] - 299) < 1


def test_light_curve_detects_dimming(tmp_path, rng):
    from astropy.io import fits
    from astroforge.core.lightcurve import light_curve, save_csv
    n = 300
    center = (156.25, 19.75)
    xs = rng.uniform(50, n - 50, 8)
    ys = rng.uniform(50, n - 50, 8)
    mags = np.array([8.0] + list(rng.uniform(7.0, 9.0, 7)))
    cat = tmp_path / "c.csv"
    dec_c = center[1] + (n / 2.0 - ys) / 3600.0
    ra_c = center[0] + (xs - n / 2.0) / 3600.0 / np.cos(np.deg2rad(center[1]))
    with open(cat, "w") as f:
        f.write("ra,dec,mag\n")
        for a, d, m in zip(ra_c, dec_c, mags):
            f.write(f"{a:.6f},{d:.6f},{m:.3f}\n")
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = list(center)
    w.wcs.crpix = [n / 2 + 1, n / 2 + 1]
    w.wcs.cd = [[-1 / 3600 / 1.0, 0], [0, -1 / 3600]]
    # vérification : pixels attendus
    px = np.stack(w.world_to_pixel_values(ra_c, dec_c), 1)
    paths = []
    yy, xx = np.mgrid[0:n, 0:n]
    for k, dm in enumerate([0.0, 0.0, 0.1, 0.2, 0.1, 0.0]):   # baisse de 0,2 mag de la cible
        img = 0.02 + rng.normal(0, 0.001, (n, n))
        for i, ((x, y), m) in enumerate(zip(px, mags)):
            fl = 0.8 * 10 ** (-0.4 * (m + (dm if i == 0 else 0) - 7.0))
            img += fl * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 4.5)
        p = tmp_path / f"t{k}.fits"
        hdr = w.to_header()
        hdr["DATE-OBS"] = f"2026-10-15T21:{10 + k:02d}:00"
        hdr["EXPTIME"] = 60.0
        fits.writeto(p, np.clip(img, 0, 1).astype(np.float32), hdr, overwrite=True)
        paths.append(str(p))
    rows = light_curve(paths, w, str(cat), float(ra_c[0]), float(dec_c[0]), n_comp=4, align=False)
    d = np.array([r["dmag"] for r in rows])
    assert all(not r["erreur"] for r in rows)
    assert abs((d[3] - d[0]) - 0.2) < 0.03 and abs(d[5] - d[0]) < 0.02
    assert rows[1]["jd"] - rows[0]["jd"] == pytest.approx(60 / 86400, abs=1e-6)
    save_csv(rows, str(tmp_path / "lc.csv"))
    assert (tmp_path / "lc.csv").read_text(encoding="utf-8").count("\n") == 7
