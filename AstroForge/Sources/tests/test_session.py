# -*- coding: utf-8 -*-
"""Tri automatique de session, bibliothèque de masters, échelle du dark."""
import numpy as np

from astroforge.core import calibration as C
from astroforge.core import masters as M
from astroforge.core.session import classify, plan_session


def _session(write_fits, rng):
    h = w = 32
    for i in range(3):
        write_fits(f"nuit/L_{i}.fits", 0.1 + 0.3 + 0.02 * rng.random((h, w)), IMAGETYP="Light Frame",
                   EXPTIME=300.0, FILTER="L", GAIN=100, CCD_TEMP=-10.0, OBJECT="M31")
        write_fits(f"nuit/Lr_{i}.fits", 0.1 + 0.2 + 0.02 * rng.random((h, w)), IMAGETYP="LIGHT",
                   EXPTIME=300.0, FILTER="R", GAIN=100, CCD_TEMP=-10.0, OBJECT="M31")
        write_fits(f"nuit/D_{i}.fits", np.full((h, w), 0.12), IMAGETYP="Dark Frame",
                   EXPTIME=300.0, GAIN=100, CCD_TEMP=-10.2)
        write_fits(f"nuit/D60_{i}.fits", np.full((h, w), 0.104), IMAGETYP="Dark Frame",
                   EXPTIME=60.0, GAIN=100, CCD_TEMP=-10.0)
        write_fits(f"nuit/Dg_{i}.fits", np.full((h, w), 0.5), IMAGETYP="Dark Frame",
                   EXPTIME=300.0, GAIN=200, CCD_TEMP=-10.0)
        write_fits(f"nuit/F_{i}.fits", np.full((h, w), 0.5), IMAGETYP="Flat Field",
                   EXPTIME=2.0, FILTER="L", GAIN=100)
        write_fits(f"nuit/B_{i}.fits", np.full((h, w), 0.1), IMAGETYP="Bias Frame",
                   EXPTIME=0.0, GAIN=100)
        write_fits(f"nuit/flats_R/sans_entete_{i}.fits", np.full((h, w), 0.4))


def test_classify_header_and_names():
    assert classify("x.fits", {"IMAGETYP": "Flat Field"}) == ("flat", "en-tête")
    assert classify("x.fits", {"IMAGETYP": "Offset"})[0] == "bias"
    assert classify("/d/DarkFlats/a.fits", {})[0] == "darkflat"
    assert classify("/d/flats_R/a.fits", {})[0] == "flat"
    assert classify("/d/lights/a.cr2", {})[0] == "light"


def test_plan_groups_and_matching(tmp_path, write_fits, rng):
    _session(write_fits, rng)
    plan = plan_session(str(tmp_path / "nuit"))
    names = {g.key[1]: g for g in plan.groups}
    assert set(names) == {"L", "R"}
    gl = names["L"]
    assert len(gl.lights) == 3 and len(gl.biases) == 3 and len(gl.flats) == 3
    # darks : même gain et même exposition (pas ceux de 60 s ni de gain 200)
    assert len(gl.darks) == 3 and all(d.exptime == 300 and d.gain == 100 for d in gl.darks)
    assert "M31" in gl.summary()


def test_dark_scaling_when_exposure_differs(tmp_path, write_fits, rng):
    h = w = 16
    for i in range(3):
        write_fits(f"s/L{i}.fits", np.full((h, w), 0.5), IMAGETYP="Light", EXPTIME=120.0, GAIN=0)
        write_fits(f"s/D{i}.fits", np.full((h, w), 0.11), IMAGETYP="Dark", EXPTIME=60.0, GAIN=0)
        write_fits(f"s/B{i}.fits", np.full((h, w), 0.10), IMAGETYP="Bias", GAIN=0)
    g = plan_session(str(tmp_path / "s")).groups[0]
    assert abs(g.dark_scale - 2.0) < 1e-9 and g.warnings
    masters = C.build_masters(paths={"bias": g.paths("bias"), "dark": g.paths("dark")})
    assert masters["meta"]["dark"]["thermal"] and masters["meta"]["dark"]["exptime"] == 60
    cal = C.calibrate_paths(g.paths("light"), masters, cosmetic=False)
    # lumière 0,5 = bias 0,10 + 2 × thermique 0,01 + signal 0,38
    assert abs(float(cal[0].mean()) - 0.38) < 2e-3


def test_master_library_roundtrip(tmp_path):
    lib = str(tmp_path / "lib")
    d = np.full((20, 30), 0.01, np.float32)
    M.save_master(d, "dark", {"exptime": 60.0, "gain": 100, "temp": -10.0, "thermal": True}, lib)
    M.save_master(d * 2, "dark", {"exptime": 300.0, "gain": 100, "temp": -10.0, "thermal": True}, lib)
    M.save_master(d, "dark", {"exptime": 300.0, "gain": 200, "temp": -10.0}, lib)
    M.save_master(np.full((20, 30), 0.1, np.float32), "bias", {"gain": 100}, lib)
    light = {"exptime": 300.0, "gain": 100, "temp": -9.0, "width": 30, "height": 20, "binning": 1}
    e, k = M.find_master("dark", light, lib)
    assert e["exptime"] == 300.0 and e["gain"] == 100 and k == 1.0
    e, k = M.find_master("dark", dict(light, exptime=120.0), lib)
    assert e["exptime"] in (60.0, 300.0) and abs(k * e["exptime"] - 120.0) < 1e-6
    assert M.find_master("dark", dict(light, temp=5.0), lib)[0] is None
    assert np.allclose(M.load_master(e, lib), d * (2 if e["exptime"] == 300 else 1))
    masters = C.build_masters(light_meta=light, use_library=True, library=lib)
    assert masters["sources"] == {"dark": "bibliothèque", "bias": "bibliothèque"}


def test_save_to_library(tmp_path, write_fits):
    lib = str(tmp_path / "lib2")
    paths = [write_fits(f"b/B{i}.fits", np.full((8, 8), 0.1), IMAGETYP="Bias", GAIN=50) for i in range(3)]
    C.build_masters(paths={"bias": paths}, save_to_library=True, library=lib)
    idx = M.load_index(lib)
    assert len(idx) == 1 and idx[0]["kind"] == "bias" and idx[0]["gain"] == 50 and idx[0]["ncombine"] == 3
