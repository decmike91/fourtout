# -*- coding: utf-8 -*-
"""Interface : chaque outil du menu s'ouvre (français et anglais), import
d'une session dans l'onglet Ciel profond, empilement en direct piloté par le
minuteur, envoi d'un résultat vers la Finition."""
import os
import shutil

import numpy as np
import pytest


@pytest.fixture
def main_window(qapp):
    from astroforge.gui.main_window import MainWindow
    mw = MainWindow()
    yield mw
    for d in getattr(mw, "_tool_dialogs", {}).values():
        d.close()
    mw.win.close()
    mw.win.deleteLater()


def test_every_tool_opens(main_window):
    from astroforge.gui.tools_menu import registered_tools
    tools = [(label, cls) for label, cls in registered_tools() if cls is not None]
    assert len(tools) >= 12
    acts = [a.text() for a in main_window.menu_tools.actions() if a.text()]
    assert len(acts) == len(tools)
    for label, cls in tools:
        dlg = cls(main_window, main_window.win)
        dlg.show()
        assert dlg.windowTitle()
        assert dlg.btnRun is not None
        dlg.close()
        dlg.deleteLater()


def test_every_tool_translates(qapp):
    from astroforge import i18n
    from astroforge.gui.tools_menu import registered_tools
    try:
        i18n.set_language("en")
        for label, cls in registered_tools():
            if cls is None:
                continue
            dlg = cls()
            i18n.translate_widget_tree(dlg)
            assert dlg.btnCancel.text() == "Cancel", cls.__name__
            dlg.deleteLater()
    finally:
        i18n.set_language("fr")


def test_menu_triggers_open_dialog(main_window):
    act = [a for a in main_window.menu_tools.actions() if a.text()][0]
    act.trigger()
    assert len(main_window._tool_dialogs) == 1
    act.trigger()   # réutilise la même fenêtre
    assert len(main_window._tool_dialogs) == 1


def _make_session(tmp_path, write_fits, star_field):
    rng = np.random.default_rng(0)
    for i in range(4):
        img = star_field(dx=i * 0.5, seed=5) + rng.normal(0, 0.01, (128, 128)).astype(np.float32)
        write_fits(f"s/L_{i}.fits", img, IMAGETYP="Light Frame", EXPTIME=120.0,
                   OBJECT="M 31", FILTER="Ha", GAIN=100, CCD_TEMP=-10.0)
    for i in range(3):
        write_fits(f"s/D_{i}.fits", np.full((128, 128), 0.02) + rng.normal(0, 0.003, (128, 128)),
                   IMAGETYP="Dark Frame", EXPTIME=120.0, GAIN=100, CCD_TEMP=-10.0)
        write_fits(f"s/F_{i}.fits", np.full((128, 128), 0.5) + rng.normal(0, 0.01, (128, 128)),
                   IMAGETYP="Flat Field", EXPTIME=1.0, FILTER="Ha", GAIN=100)
    return str(tmp_path / "s")


def test_session_import_into_sky_page(tmp_path, write_fits, star_field, main_window):
    from PySide6.QtWidgets import QListWidget
    from astroforge.core.session import plan_session
    from astroforge.gui.tools_dialogs import SessionDialog
    plan = plan_session(_make_session(tmp_path, write_fits, star_field))
    dlg = SessionDialog(plan, main_window.win)
    g = dlg.selected_group()
    assert g is not None and len(g.paths("light")) == 4
    assert len(g.paths("dark")) == 3 and len(g.paths("flat")) == 3
    sky = main_window.page_sky
    sky.apply_session_group(g)
    lst = sky.ui.findChild(QListWidget, "listLight")
    assert lst.count() == 4
    assert len(sky.state["calib_paths"]["dark"]) == 3
    dlg.deleteLater()


def test_subframe_and_library_dialogs(qapp, star_field):
    from astroforge.core import subframes as SF
    from astroforge.gui.tools_dialogs import MasterLibraryDialog, SubframeDialog
    rows = []
    for i, f in enumerate((2.5, 2.6, 2.4, 5.5)):
        r = SF.analyze_image(star_field(fwhm=f, seed=2))
        r["path"] = f"img{i}.fits"
        rows.append(r)
    rows = SF.flag_outliers(rows)
    dlg = SubframeDialog(rows)
    kept, weights = dlg.result_selection()
    assert "img3.fits" not in kept and len(kept) == 3
    dlg.deleteLater()
    lib = MasterLibraryDialog()
    lib.deleteLater()


def test_live_stack_dialog_tick(tmp_path, write_fits, star_field, qapp):
    from astroforge.gui.tools_extra import LiveStackDialog
    src = _make_session(tmp_path, write_fits, star_field)
    live = tmp_path / "live"
    live.mkdir()
    dlg = LiveStackDialog()
    dlg.folder.edit.setText(str(live))
    dlg.chkExisting.setChecked(True)
    dlg.start()
    assert dlg.timer.isActive()
    for i in range(3):
        shutil.copy(os.path.join(src, f"L_{i}.fits"), live / f"L_{i}.fits")
    dlg._tick()          # 1er passage : tailles relevées
    dlg._tick()          # 2e passage : fichiers stables → empilés
    assert len(dlg.stacker.accepted) == 3, dlg.stacker.rejected
    assert dlg.result_image is not None and dlg.btnSend.isEnabled()
    assert "3 image" in dlg.lblStatus.text()
    dlg.cancel()
    assert not dlg.timer.isActive()
    dlg.deleteLater()


def test_compose_send_to_finishing(tmp_path, main_window, star_field):
    from astroforge.core.image_io import save_image
    from astroforge.gui.tools_menu import ComposeDialog
    ha, o3 = tmp_path / "ha.fits", tmp_path / "o3.fits"
    save_image(str(ha), star_field(seed=4))
    save_image(str(o3), 0.5 * star_field(seed=4))
    dlg = ComposeDialog(main_window, main_window.win)
    dlg.rows["Ha"].edit.setText(str(ha))
    dlg.rows["OIII"].edit.setText(str(o3))
    dlg.cmbPal.setCurrentText("HOO")
    res = dlg.run_sync()
    assert res.shape == (128, 128, 3)
    dlg.send_to_finishing()
    assert main_window.shared["image"] is not None
    assert main_window.shared["image"].shape == (128, 128, 3)
    dlg.deleteLater()


def test_open_file_argument(tmp_path, main_window, star_field):
    """Double-clic / « Ouvrir avec » : le fichier passé en argument s'ouvre
    dans la Finition ; les fichiers absents ou illisibles sont ignorés."""
    from astroforge.core.image_io import save_image
    from astroforge.gui.app import _open_files
    bad = tmp_path / "pas_une_image.fits"
    bad.write_text("x")
    good = tmp_path / "M 42 é.fits"
    save_image(str(good), star_field(seed=9))
    opened = _open_files(main_window, [str(tmp_path / "absent.fits"), str(bad), str(good)])
    assert opened == str(good)
    assert main_window.shared["image"].shape == (128, 128)


def test_prebuilt_ui_used_when_cache_elsewhere(tmp_path, monkeypatch):
    """Installation en lecture seule : les interfaces pré-générées livrées
    avec le programme sont utilisées sans recompilation."""
    import shutil
    from astroforge.gui import loader
    try:
        loader._generate("main_window.ui")          # garantit un fichier à jour
    except Exception:
        pytest.skip("pyside6-uic indisponible dans cet environnement")
    pre = tmp_path / "pre"
    pre.mkdir()
    for f in os.listdir(loader.GEN_DIR):
        if f.endswith(".py"):
            shutil.copy(os.path.join(loader.GEN_DIR, f), pre / f)
    monkeypatch.setattr(loader, "PREBUILT_DIR", str(pre))
    monkeypatch.setattr(loader, "GEN_DIR", str(tmp_path / "cache_vide"))
    py, base, root = loader._generate("main_window.ui")
    assert py == str(pre / "ui_MainWindow.py") and root == "QMainWindow"
    assert not (tmp_path / "cache_vide").exists()
