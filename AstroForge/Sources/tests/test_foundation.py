# -*- coding: utf-8 -*-
"""Socle : ligne de commande, scripts, statistiques d'aperçu, export avec
en-tête (FITS/XISF/TIFF 32 bits), traduction, rotation log-polaire, GPU
(module substitut), vérification des mises à jour."""
import json
import os
import types

import numpy as np
import pytest


# ------------------------------------------------------------ ligne de commande
def _session(tmp_path, write_fits, star_field, n=5):
    d = tmp_path / "session"
    for i in range(n):
        img = star_field(dx=0.7 * i, dy=-0.4 * i, seed=3) + \
            np.random.default_rng(i).normal(0, 0.01, (128, 128)).astype(np.float32)
        write_fits(f"session/light_{i:02d}.fits", img, IMAGETYP="Light Frame", EXPTIME=60.0,
                   OBJECT="M 42", FILTER="L", DATE_OBS=f"2026-01-10T22:0{i}:00")
    return d


def test_cli_sort_and_stack(tmp_path, write_fits, star_field, capsys):
    from astroforge import cli
    from astropy.io import fits
    d = _session(tmp_path, write_fits, star_field)
    assert cli.main(["sort", str(d), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data) == 1 and len(data[0]["lights"]) == 5
    out = tmp_path / "out.fits"
    assert cli.main(["stack", "--session", str(d), "-o", str(out)]) == 0
    hdr = fits.getheader(out)
    assert hdr["NCOMBINE"] == 5
    assert hdr["EXPTIME"] == pytest.approx(300.0)
    assert hdr["OBJECT"] == "M 42"


def test_cli_info_and_errors(capsys):
    from astroforge import cli, __version__
    assert cli.main(["info"]) == 0
    assert __version__ in capsys.readouterr().out
    assert cli.main(["sort", "/chemin/inexistant/xyz"]) in (0, 1)


def test_cli_script(tmp_path, capsys):
    from astroforge import cli
    p = tmp_path / "s.py"
    p.write_text("import sys\nprint('ARGS', sys.argv[1:])\n", encoding="utf-8")
    assert cli.main(["script", str(p), "a", "b"]) == 0
    assert "ARGS ['a', 'b']" in capsys.readouterr().out


def test_run_script_namespace():
    from astroforge.gui.tools_extra import run_script
    img = np.full((16, 16), 0.2, np.float32)
    out, res = run_script("print(image.mean())\nset_image(af.stretch.arcsinh_stretch(image, 10))",
                          img)
    assert "0.2" in out
    assert res is not None and res.shape == (16, 16) and res.mean() > 0.2


# ------------------------------------------------------------ aperçu
def test_image_stats(rng):
    from astroforge.gui.preview import image_stats
    img = (0.1 + rng.normal(0, 0.02, (300, 300))).astype(np.float32)
    st = image_stats(img)
    assert st["median"] == pytest.approx(0.1, abs=0.005)
    assert st["noise"] == pytest.approx(0.02, rel=0.25)
    assert image_stats(None) is None


# ------------------------------------------------------------ export
def test_export_headers(tmp_path):
    from astroforge.core.image_io import load_image, save_image
    from astroforge.core.xisf import fits_keywords, read_xisf
    img = np.linspace(0, 1, 64 * 48 * 3, dtype=np.float32).reshape(48, 64, 3)
    hdr = {"NCOMBINE": 12, "EXPTIME": 720.5, "OBJECT": "NGC 7000 l'Amérique",
           "SIMPLE": True, "ROWORDER": "TOP-DOWN"}
    x = tmp_path / "a.xisf"
    save_image(str(x), img, header=hdr)
    kw = fits_keywords(str(x))
    assert kw["NCOMBINE"] == 12 and kw["EXPTIME"] == pytest.approx(720.5)
    assert kw["OBJECT"] == "NGC 7000 l'Amérique" and "SIMPLE" not in kw
    np.testing.assert_allclose(read_xisf(str(x)), img, atol=1e-6)
    t = tmp_path / "a.tif"
    save_image(str(t), img, "32f")
    back = load_image(str(t))
    assert back.dtype == np.float32
    np.testing.assert_allclose(back, img, atol=1e-6)


def test_sky_export_full_header(tmp_path, write_fits, star_field, qapp):
    from astropy.io import fits
    from astroforge.gui.page_sky import PageSky
    d = _session(tmp_path, write_fits, star_field, n=3)
    page = PageSky(shared={"image": None})
    page.state["light_paths"] = sorted(str(p) for p in d.iterdir())
    page.state["image"] = star_field()
    out = tmp_path / "res.fits"
    page.export_to(str(out))
    h = fits.getheader(out)
    assert h["NCOMBINE"] == 3 and h["EXPTIME"] == pytest.approx(180.0)
    page.export_to(str(tmp_path / "res.xisf"))
    from astroforge.core.xisf import fits_keywords
    assert fits_keywords(str(tmp_path / "res.xisf"))["NCOMBINE"] == 3
    page.deleteLater()


# ------------------------------------------------------------ traduction
def test_i18n_table_and_tr():
    from astroforge import i18n
    langs = i18n.available_languages()
    assert langs["fr"] == "Français" and "en" in langs
    try:
        i18n.set_language("en")
        assert i18n.tr("Lancer") == "Run"
        assert i18n.tr("Annuler") == "Cancel"
        # ponctuation de fin conservée
        assert i18n.tr("Lancer :") == "Run :"
        assert i18n.tr("Texte inconnu xyz") == "Texte inconnu xyz"
    finally:
        i18n.set_language("fr")
    assert i18n.tr("Lancer") == "Lancer"


def test_i18n_widget_tree_skips_combos(qapp):
    from PySide6.QtWidgets import QComboBox
    from astroforge import i18n
    from astroforge.gui.tools_menu import ComposeDialog
    try:
        i18n.set_language("en")
        dlg = ComposeDialog()
        before = [dlg.cmbPal.itemText(i) for i in range(dlg.cmbPal.count())]
        n = i18n.translate_widget_tree(dlg)
        assert n > 0
        assert dlg.btnRun.text() == "Run"
        assert [dlg.cmbPal.itemText(i) for i in range(dlg.cmbPal.count())] == before
        for cb in dlg.findChildren(QComboBox):
            assert cb.currentText() in before or cb is not dlg.cmbPal
        dlg.deleteLater()
    finally:
        i18n.set_language("fr")


def test_i18n_main_window_english(qapp, monkeypatch):
    """Toute l'interface se construit en anglais ; les listes restent valides."""
    from PySide6.QtWidgets import QTabWidget
    from astroforge import i18n
    monkeypatch.setenv("ASTROFORGE_LANG", "en")
    try:
        from astroforge.gui.main_window import MainWindow
        mw = MainWindow()
        tabs = mw.win.findChild(QTabWidget, "tabs")
        names = [tabs.tabText(i) for i in range(tabs.count())]
        assert any("Deep" in t or "Sky" in t for t in names), names
        assert mw.menu_language is not None
        mw.win.deleteLater()
    finally:
        monkeypatch.delenv("ASTROFORGE_LANG")
        i18n.set_language("fr")


def test_i18n_json_well_formed():
    from astroforge import i18n
    with open(os.path.join(i18n.HERE, "en.json"), encoding="utf-8") as f:
        d = json.load(f)
    assert d["language"] == "English" and len(d["strings"]) > 300
    assert all(isinstance(v, str) and v for v in d["strings"].values())


# ------------------------------------------------------------ rotation
@pytest.mark.parametrize("true", [-2.4, -0.6, 1.3])
def test_logpolar_rotation(true, star_field):
    from astroforge.core.registration import apply_shift, estimate_rotation, logpolar_rotation
    img = star_field(h=192, w=192, n=60, seed=7)
    mov = apply_shift(img, 0, 0, true)
    a, s, _ = logpolar_rotation(img, mov)
    assert a == pytest.approx(-true, abs=0.15)
    assert s == pytest.approx(1.0, abs=0.03)
    assert estimate_rotation(img, mov, 3.0) == pytest.approx(-true, abs=0.08)
    assert estimate_rotation(img, mov, 3.0, method="sweep") == pytest.approx(-true, abs=0.08)


# ------------------------------------------------------------ GPU
class _FakeCupy(types.ModuleType):
    """Substitut de CuPy fondé sur NumPy : vérifie le chemin GPU (transferts,
    code générique xp) sans carte graphique."""

    class ndarray(np.ndarray):
        pass

    def __init__(self):
        super().__init__("fakecupy")
        self.calls = 0
        for k in ("float32", "float64", "median", "abs", "maximum", "sqrt", "where", "ones"):
            setattr(self, k, getattr(np, k))

    def asarray(self, a, dtype=None):
        self.calls += 1
        return np.asarray(a, dtype).view(_FakeCupy.ndarray)

    def asnumpy(self, a):
        return np.asarray(a).view(np.ndarray)


def test_gpu_path_matches_cpu(rng, monkeypatch):
    from astroforge.core import gpu, stacking as S
    a = rng.normal(0.3, 0.05, (9, 40, 30)).astype(np.float32)
    a[2, 5, 5] = 5.0
    w = np.linspace(0.5, 1.5, 9)
    cpu = {m: S.stack(a, m, weights=w if m != "Médiane" else None)
           for m in ("Moyenne", "Médiane", "Sigma-clipping")}
    fake = _FakeCupy()
    monkeypatch.setattr(S, "_xp", lambda x: fake if isinstance(x, _FakeCupy.ndarray) else np)
    gpu._inject(fake)
    gpu.set_enabled(True)
    try:
        for m, ref in cpu.items():
            out = S.stack(a, m, weights=w if m != "Médiane" else None)
            np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
        assert fake.calls > 0
    finally:
        gpu.set_enabled(None)
        gpu._inject(None)
        gpu._checked = False


def test_gpu_falls_back_on_error(rng):
    from astroforge.core import gpu, stacking as S

    class Broken(types.ModuleType):
        def asarray(self, *a, **k):
            raise MemoryError("plus de mémoire GPU")
    a = rng.random((5, 10, 10)).astype(np.float32)
    ref = S.stack(a, "Moyenne")
    gpu._inject(Broken("broken"))
    gpu.set_enabled(True)
    try:
        np.testing.assert_allclose(S.stack(a, "Moyenne"), ref)
    finally:
        gpu.set_enabled(None)
        gpu._inject(None)
        gpu._checked = False


# ------------------------------------------------------------ mises à jour
def test_update_versions_and_manifest(tmp_path):
    from astroforge import updates
    assert updates.is_newer("2.0.1", "2.0.0")
    assert updates.is_newer("2.0.0", "2.0.0rc1")
    assert not updates.is_newer("2.0", "2.0.0")
    assert not updates.is_newer("1.9.9", "2.0.0")
    m = tmp_path / "latest.json"
    m.write_text(json.dumps({"version": "99.0", "url": "https://exemple/af.exe",
                             updates.platform_key(): {"url": "https://exemple/af-plat"},
                             "notes": "N"}), encoding="utf-8")
    info = updates.check(m.as_uri())
    assert info["version"] == "99.0" and info["url"] == "https://exemple/af-plat"
    m.write_text(json.dumps({"version": "0.1"}), encoding="utf-8")
    assert updates.check(m.as_uri()) is None
    assert updates.check("") is None
    with pytest.raises(ValueError):
        updates.fetch_manifest("http://non-securise/latest.json")
