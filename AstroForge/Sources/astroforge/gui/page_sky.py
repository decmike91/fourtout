# -*- coding: utf-8 -*-
"""Onglet Ciel profond : prétraitement, masters, alignement/drizzle, empilement,
traitement, IA, HDR, processus auto, profils et aperçu live."""
from __future__ import annotations
import os

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QListWidget,
                               QListWidgetItem, QMessageBox, QVBoxLayout,
                               QComboBox, QSlider, QWidget)

from ..processing.pipeline import STEPS, Pipeline
from ..processing import profiles as P
from ..core.image_io import load_image, save_image
from ..core.gradient import extract_gradient
from ..core.restoration import denoise, richardson_lucy
from ..core.effects import merge_hdr
from .loader import load_ui
from .preview import LivePreview, PreviewPane
from .profiles_ui import ProfileBar
from .commands import StepCommands


def _d(**kw):
    return dict(kw)


class PageSky(StepCommands, QWidget):
    def __init__(self, parent=None, shared=None):
        super().__init__(parent)
        self.shared = shared if shared is not None else {}
        self.ui = load_ui("page_sky.ui", self)
        _outer = QVBoxLayout(self)   # la page suit la taille de la fenêtre
        _outer.setContentsMargins(0, 0, 0, 0)
        _outer.addWidget(self.ui)

        self.ui.findChild(QVBoxLayout, "paramsLayout")
        # conteneur d'aperçu
        pv_container = self.ui.findChild(QVBoxLayout, "previewContainer")
        self.pane = PreviewPane()
        pv_container.addWidget(self.pane)
        self.preview_base = None
        self.state = {"light_paths": [], "images": [], "image": None,
                      "background_model": None, "hdr_images": []}
        self.live = LivePreview(self.pane, self._collect_preview_params,
                                self._compute_preview, self)
        from .ai_dialog import populate_ai_tasks
        from ..ai.manager import register_provider_refresh
        populate_ai_tasks(self._cmb("cmbAITask"))
        self._cmb("cmbAITask").currentIndexChanged.connect(self._refresh_ai_providers)
        register_provider_refresh(self._refresh_ai_providers)
        self._build_profiles()
        self._build_session_tools()
        self._init_commands({
            "grpImages": self._load_lights_step,
            "grpCalib": lambda: self._run_sky_steps(["calibration"]),
            "grpAlign": lambda: self._run_sky_steps(["registration"]),
            "grpStack": lambda: self._run_sky_steps(["drizzle"]),
            "grpTrait": lambda: self._run_sky_steps(["gradient", "denoise", "deconvolution"]),
            "grpIA": self._apply_ai_step, "grpHDR": self._do_hdr,
            "grpPipe": self._process_full, "grpColor": self._color_cal,
            "grpAstro": self._plate_solve, "grpDefects": self._apply_defects_step,
            "grpRun": self._process_full,
        }, ("state", "preview_base"))
        self._wire()
        self._build_pipeline_list()
        self._log("AstroForge — mode Ciel profond prêt.")

    # ------------------------------------------------- session & masters
    def _build_session_tools(self):
        """Tri automatique de session + bibliothèque de masters (groupe 2)."""
        from PySide6.QtWidgets import QCheckBox, QGridLayout, QHBoxLayout, QPushButton
        grid = self.ui.findChild(QGridLayout, "gridCalib")
        row = QHBoxLayout()
        b_sess = QPushButton("Importer une session (tri auto)…")
        b_sess.setObjectName("btnSession")
        b_sess.setToolTip("Classe lumières, darks, flats, bias d'après les en-têtes FITS "
                          "et apparie la calibration (gain, binning, pose, température).")
        b_lib = QPushButton("Bibliothèque…")
        b_lib.setObjectName("btnLibrary")
        row.addWidget(b_sess)
        row.addWidget(b_lib)
        grid.addLayout(row, 6, 0, 1, 3)
        self.chkLibUse = QCheckBox("Compléter avec la bibliothèque de masters")
        self.chkLibUse.setObjectName("chkLibUse")
        self.chkLibSave = QCheckBox("Enregistrer les masters calculés dans la bibliothèque")
        self.chkLibSave.setObjectName("chkLibSave")
        grid.addWidget(self.chkLibUse, 7, 0, 1, 3)
        # analyse des brutes (groupe 1 « Images »)
        from PySide6.QtWidgets import QLayout
        grp_img = self.ui.findChild(object, "grpImages")
        b_an = QPushButton("Analyser les brutes (FWHM, étoiles, nuages, poids)…")
        b_an.setObjectName("btnAnalyze")
        lay_img = grp_img.layout()
        if isinstance(lay_img, QGridLayout):
            lay_img.addWidget(b_an, lay_img.rowCount(), 0, 1, max(1, lay_img.columnCount()))
        elif isinstance(lay_img, QLayout):
            lay_img.addWidget(b_an)
        b_an.clicked.connect(self._analyze_subframes)
        from PySide6.QtWidgets import QComboBox, QLabel
        from ..core.calibration import OUTPUTS
        row_out = QHBoxLayout()
        row_out.addWidget(QLabel("Sortie capteur couleur :"))
        self.cmbOutput = QComboBox()
        self.cmbOutput.setObjectName("cmbOscOutput")
        self.cmbOutput.addItems(list(OUTPUTS))
        self.cmbOutput.setToolTip("Filtre duo-bande : empilez séparément Ha puis OIII, "
                                  "puis composez (menu Outils › Composition couleur).")
        row_out.addWidget(self.cmbOutput, 1)
        if isinstance(lay_img, QGridLayout):
            lay_img.addLayout(row_out, lay_img.rowCount(), 0, 1, max(1, lay_img.columnCount()))
        elif isinstance(lay_img, QLayout):
            lay_img.addLayout(row_out)
        self.state["weights"] = {}
        grid.addWidget(self.chkLibSave, 8, 0, 1, 3)
        b_sess.clicked.connect(self._import_session)
        b_lib.clicked.connect(self._open_library)
        self.state["calib_paths"] = None

    def _import_session(self, folder: str | None = None):
        from .tools_dialogs import SessionDialog
        from ..core.session import plan_session
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "Dossier de la session (nuit)")
        if not folder:
            return

        def run(progress=None):
            if progress:
                progress(0, 1, "Lecture des en-têtes…")
            return plan_session(folder)

        def done(plan):
            if not plan.groups:
                QMessageBox.information(self, "Session", "Aucune image lumière trouvée.\n\n"
                                        + plan.summary())
                return
            dlg = SessionDialog(plan, self)
            if dlg.exec() and dlg.selected_group() is not None:
                self.apply_session_group(dlg.selected_group())
        self._start_job(run, done, self.ui)

    def apply_session_group(self, g):
        """Charge un groupe de session : lumières + listes de calibration."""
        self.state["light_paths"] = g.paths("light")
        self.state["calib_paths"] = {k: g.paths(k) for k in ("bias", "dark", "darkflat", "flat")}
        lst = self.ui.findChild(QListWidget, "listLight")
        if lst is not None:
            lst.clear()
            lst.addItems([os.path.basename(p) for p in self.state["light_paths"]])
        bay = g.lights[0].bayer
        if bay and self._cmb("cmbBayer").findText(bay) >= 0:
            self._cmb("cmbBayer").setCurrentText(bay)
        self._invalidate_sky(images=True)
        self._log("Session : " + g.summary().replace("\n", " "))

    def _analyze_subframes(self):
        from .tools_dialogs import SubframeDialog
        from ..core import subframes as SF
        paths = list(self.state["light_paths"])
        if not paths:
            QMessageBox.information(self, "Analyse", "Ajoutez d'abord des images brutes.")
            return
        bayer = self._cmb("cmbBayer").currentText()
        bayer = None if bayer.startswith("Aucune") else bayer

        def run(progress=None):
            return SF.flag_outliers(SF.analyze_paths(paths, progress, bayer=bayer))

        def done(rows):
            dlg = SubframeDialog(rows, self)
            if dlg.exec():
                kept, weights = dlg.result_selection()
                self.apply_subframe_selection(kept, weights)
        self._start_job(run, done, self.ui)

    def apply_subframe_selection(self, kept, weights):
        """Garde les images choisies et mémorise leurs poids d'empilement."""
        removed = len(self.state["light_paths"]) - len(kept)
        self.state["light_paths"] = list(kept)
        self.state["weights"] = dict(weights)
        lw = self.ui.findChild(QListWidget, "listLight")
        lw.clear()
        lw.addItems([os.path.basename(p) for p in kept])
        self._invalidate_sky(images=True)
        self._log(f"Analyse : {len(kept)} images gardées, {removed} rejetées ; "
                  "poids appliqués à l'empilement.")

    def _open_library(self):
        from .tools_dialogs import MasterLibraryDialog
        MasterLibraryDialog(self).exec()

    def _light_meta(self):
        from ..core.session import frame_info
        if not self.state["light_paths"]:
            return None
        fi = frame_info(self.state["light_paths"][0])
        return {"exptime": fi.exptime, "gain": fi.gain, "temp": fi.temp,
                "binning": fi.binning, "width": fi.width, "height": fi.height,
                "filter": fi.filter}

    # ------------------------------------------------------------------ UI
    def _prof_layout(self, name, grid):
        """Layout de profil : retrouvé par nom, sinon recréé dans la grille."""
        from PySide6.QtWidgets import QVBoxLayout
        lay = self.ui.findChild(QVBoxLayout, name)
        if lay is None:
            g = self.ui.findChild(object, grid)
            if g is not None:
                lay = QVBoxLayout()
                lay.setObjectName(name)
                g.addLayout(lay)
        return lay

    def _log(self, msg):
        from PySide6.QtWidgets import QPlainTextEdit
        box = self.ui.findChild(QPlainTextEdit, "log")
        if box:
            from PySide6.QtCore import Q_ARG, QMetaObject
            QMetaObject.invokeMethod(box, "appendPlainText", Qt.AutoConnection, Q_ARG(str, msg))

    def _wire(self):
        self._btn("btnAddLight", self._add_lights)
        self._btn("btnClearLight", self._clear_lights)
        for btn, ed in (("btnBias", "txtBias"), ("btnDark", "txtDark"),
                        ("btnFlat", "txtFlat"), ("btnDarkFlat", "txtDarkFlat")):
            self._btn(btn, lambda _=False, e=ed: self._pick_dir(e))
        self._btn("btnAddHDR", self._add_hdr)
        self._btn("btnDoHDR", self._do_hdr)
        self._btn("btnAIConfig", self._ai_config)
        self._btn("btnProcess", self._process_full)
        self._btn("btnExport", self._export)
        self._btn("btnStepUp", lambda: self._move_step(-1))
        self._btn("btnStepDown", lambda: self._move_step(+1))
        self._btn("btnStepToggle", self._toggle_step)
        self._btn("btnPipeSave", self._save_pipeline)
        self._btn("btnPipeLoad", self._load_pipeline)
        self._btn("btnColorCal", self._color_cal)
        self._btn("btnBgNeutral", self._bg_neutral)
        self._btn("btnSCNR", self._scnr)
        self._btn("btnCatalogBrowse", self._pick_catalog)
        self._btn("btnSolve", self._plate_solve)
        self._btn("btnPhotometry", self._photometry)
        self._btn("btnBPM", self._pick_bpm)
        self._btn("btnPCC", self._pcc)
        self._btn("btnDPH", self._dph)
        # aperçu live : tout changement déclenche le recalcul
        for name in ("cmbStackMethod", "spnSigma", "cmbDrizzle", "spnPixfrac",
                     "cmbGradMethod", "cmbGradCorr", "spnGradDeg", "spnGradGrid",
                     "cmbDenoiseMethod", "spnDeconvIters", "spnDeconvSigma"):
            wid = self.ui.findChild(object, name)
            sig = "currentTextChanged" if isinstance(wid, QComboBox) else "valueChanged"
            getattr(wid, sig).connect(self.live.trigger)
        self._sld("sldDenoiseStrength").valueChanged.connect(self.live.trigger)

    def _btn(self, name, slot):
        groups = {"btnAddLight": "grpImages", "btnClearLight": "grpImages",
                  "btnDoHDR": "grpHDR", "btnProcess": "grpPipe",
                  "btnColorCal": "grpColor", "btnBgNeutral": "grpColor",
                  "btnSCNR": "grpColor", "btnPCC": "grpColor",
                  "btnSolve": "grpAstro", "btnPhotometry": "grpAstro",
                  "btnDPH": "grpAstro"}
        if name in groups:
            self.ui.findChild(object, name).clicked.connect(
                lambda: self._execute(groups[name], slot))
        else:
            self.ui.findChild(object, name).clicked.connect(slot)

    def _sld(self, name):
        return self.ui.findChild(QSlider, name)

    def _pick_dir(self, editor_name):
        d = QFileDialog.getExistingDirectory(self, "Choisir un dossier")
        if d:
            self.ui.findChild(object, editor_name).setText(d)

    def _add_lights(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Images brutes (lights)", "",
            "Images (*.fits *.fit *.fts *.fz *.xisf *.tif *.tiff *.png *.jpg *.cr2 *.cr3 *.nef *.arw *.dng *.raf *.orf *.rw2)")
        if files:
            self._invalidate_sky(images=True)
        lw: QListWidget = self.ui.findChild(QListWidget, "listLight")
        for f in files:
            self.state["light_paths"].append(f)
            lw.addItem(os.path.basename(f))

    def _clear_lights(self):
        self._invalidate_sky(images=True)
        self.state["light_paths"] = []
        self.state["calib_paths"] = None  # listes de calibration de la session
        self.state["weights"] = {}
        self.ui.findChild(QListWidget, "listLight").clear()

    def _add_hdr(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Expositions pour le HDR", "",
                                                "Images (*.fits *.tif *.tiff *.png)")
        lw: QListWidget = self.ui.findChild(QListWidget, "listHDR")
        for f in files:
            self.state["hdr_images"].append(f)
            lw.addItem(os.path.basename(f))

    def _do_hdr(self):
        if len(self.state["hdr_images"]) < 2:
            QMessageBox.information(self, "HDR", "Ajoutez au moins 2 expositions.")
            return
        self._log("Fusion HDR en cours…")
        paths = list(self.state["hdr_images"])
        def run(progress=None):
            images = []
            for i, path in enumerate(paths):
                progress(i, len(paths), "HDR")
                images.append(load_image(path))
            return merge_hdr(images)
        self._start_job(run, self._after_edit, self.ui)

    def _ai_config(self):
        from .ai_dialog import AIToolsDialog
        AIToolsDialog(self).exec()
        self._refresh_ai_providers()

    # ------------------------------------------------------------- profils
    def _build_profiles(self):
        from PySide6.QtWidgets import QVBoxLayout
        # alignement
        self.pb_align = ProfileBar(
            "sky_align", _d(rotation=False, drizzle="Sans", pixfrac=0.7),
            lambda: {"rotation": self._cbx("chkRotation").isChecked(),
                     "drizzle": self._cmb("cmbDrizzle").currentText(),
                     "pixfrac": self._spn("spnPixfrac").value()},
            lambda p: (self._cbx("chkRotation").setChecked(bool(p.get("rotation"))),
                       self._cmb("cmbDrizzle").setCurrentText(p.get("drizzle", "Sans")),
                       self._spn("spnPixfrac").setValue(float(p.get("pixfrac", 0.7)))))
        self.ui.findChild(QVBoxLayout, "profAlign").addWidget(self.pb_align)
        self.pb_stack = ProfileBar(
            "sky_stack", _d(method="Sigma-clipping", sigma=3.0),
            lambda: {"method": self._cmb("cmbStackMethod").currentText(),
                     "sigma": self._spn("spnSigma").value()},
            lambda p: (self._cmb("cmbStackMethod").setCurrentText(p.get("method", "Sigma-clipping")),
                       self._spn("spnSigma").setValue(float(p.get("sigma", 3.0)))))
        self.ui.findChild(QVBoxLayout, "profStack").addWidget(self.pb_stack)
        self.pb_trait = ProfileBar(
            "sky_trait",
            _d(grad_method="Polynôme 2D", grad_corr="Soustraction", grad_deg=3,
               grad_grid=12, denoise_method="Non-local means", denoise_strength=0.5,
               deconv_iters=20, deconv_sigma=1.5),
            self._trait_params,
            lambda p: (self._cmb("cmbGradMethod").setCurrentText(p.get("grad_method", "Polynôme 2D")),
                       self._cmb("cmbGradCorr").setCurrentText(p.get("grad_corr", "Soustraction")),
                       self._spn("spnGradDeg").setValue(int(p.get("grad_deg", 3))),
                       self._spn("spnGradGrid").setValue(int(p.get("grad_grid", 12))),
                       self._cmb("cmbDenoiseMethod").setCurrentText(p.get("denoise_method", "Non-local means")),
                       self._sld("sldDenoiseStrength").setValue(int(p.get("denoise_strength", 0.5) * 100)),
                       self._spn("spnDeconvIters").setValue(int(p.get("deconv_iters", 20))),
                       self._spn("spnDeconvSigma").setValue(float(p.get("deconv_sigma", 1.5)))))
        self.ui.findChild(QVBoxLayout, "profTrait").addWidget(self.pb_trait)
        self.pb_ia = ProfileBar(
            "sky_ia", _d(task="denoise", provider="onnx", strength=0.5),
            self._ia_params,
            self._set_ia_params)
        self.ui.findChild(QVBoxLayout, "profIA").addWidget(self.pb_ia)
        self.pb_defects = ProfileBar(
            "sky_defects", {"enabled": False, "auto": True, "cols": True, "bpm": ""},
            lambda: {"enabled": self._cbx("chkDefects").isChecked(),
                     "auto": self._cbx("chkDefectsAuto").isChecked(),
                     "cols": self._cbx("chkDefectsCols").isChecked(),
                     "bpm": self._ed("txtBPM").text()},
            lambda p: (self._cbx("chkDefects").setChecked(bool(p.get("enabled"))),
                       self._cbx("chkDefectsAuto").setChecked(bool(p.get("auto", True))),
                       self._cbx("chkDefectsCols").setChecked(bool(p.get("cols", True))),
                       self._ed("txtBPM").setText(str(p.get("bpm", "")))))
        self._prof_layout("profDefects", "gridDefects").addWidget(self.pb_defects)
        self._refresh_ai_providers()

    def _cmb(self, name):
        return self.ui.findChild(QComboBox, name)

    def _spn(self, name):
        o = self.ui.findChild(object, name)
        return o

    def _cbx(self, name):
        return self.ui.findChild(QCheckBox, name)

    def _trait_params(self):
        return {"grad_method": self._cmb("cmbGradMethod").currentText(),
                "grad_corr": self._cmb("cmbGradCorr").currentText(),
                "grad_deg": self._spn("spnGradDeg").value(),
                "grad_grid": self._spn("spnGradGrid").value(),
                "denoise_method": self._cmb("cmbDenoiseMethod").currentText(),
                "denoise_strength": self._sld("sldDenoiseStrength").value() / 100.0,
                "deconv_iters": self._spn("spnDeconvIters").value(),
                "deconv_sigma": self._spn("spnDeconvSigma").value()}

    def _set_ia_params(self, params):
        task = self._cmb("cmbAITask")
        index = task.findData(params.get("task", "denoise"))
        if index >= 0:
            task.setCurrentIndex(index)
        self._refresh_ai_providers()
        provider = self._cmb("cmbAIProvider")
        index = provider.findData(params.get("provider", "onnx"))
        if index >= 0:
            provider.setCurrentIndex(index)
        self._sld("sldAIStrength").setValue(int(params.get("strength", 0.5) * 100))

    def _ia_params(self):
        return {"task": self._cmb("cmbAITask").currentData(),
                "provider": self._cmb("cmbAIProvider").currentData() or "onnx",
                "strength": self._sld("sldAIStrength").value() / 100.0}

    def _refresh_ai_providers(self):
        from ..ai.manager import get_ai_manager
        cmb = self._cmb("cmbAIProvider")
        selected = cmb.currentData()
        task = self._cmb("cmbAITask").currentData()
        cmb.clear()
        for prov in get_ai_manager().providers(task):
            cmb.addItem(prov["label"], prov["id"])
        index = cmb.findData(selected)
        if index >= 0:
            cmb.setCurrentIndex(index)

    # ------------------------------------------------------------- pipeline
    def _build_pipeline_list(self):
        lw: QListWidget = self.ui.findChild(QListWidget, "lstSteps")
        lw.clear()
        for name, st in STEPS.items():
            it = QListWidgetItem(f"{st['label']}  [{st['category']}]")
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if name in
                             ("calibration", "registration", "drizzle", "gradient",
                              "denoise") else Qt.Unchecked)
            it.setData(Qt.UserRole, name)
            lw.addItem(it)

    def _pipeline_from_list(self) -> Pipeline:
        lw: QListWidget = self.ui.findChild(QListWidget, "lstSteps")
        steps = []
        for i in range(lw.count()):
            it = lw.item(i)
            steps.append({"name": it.data(Qt.UserRole),
                          "params": {}, "enabled": it.checkState() == Qt.Checked})
        return Pipeline(steps)

    def _move_step(self, delta: int):
        lw: QListWidget = self.ui.findChild(QListWidget, "lstSteps")
        row = lw.currentRow()
        if row < 0:
            return
        it = lw.takeItem(row)
        lw.insertItem(max(0, min(lw.count(), row + delta)), it)
        lw.setCurrentRow(row + delta)

    def _toggle_step(self):
        lw: QListWidget = self.ui.findChild(QListWidget, "lstSteps")
        it = lw.currentItem()
        if it:
            it.setCheckState(Qt.Unchecked if it.checkState() == Qt.Checked
                             else Qt.Checked)

    def _save_pipeline(self):
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer le processus", "",
                                              "Processus (*.json)")
        if path:
            P.save_pipeline(path, self._pipeline_from_list().steps)
            self._log(f"Processus enregistré : {path}")

    def _load_pipeline(self):
        path, _ = QFileDialog.getOpenFileName(self, "Charger un processus", "",
                                              "Processus (*.json)")
        if not path:
            return
        steps = P.load_pipeline(path)
        lw: QListWidget = self.ui.findChild(QListWidget, "lstSteps")
        lw.clear()
        from ..processing.pipeline import STEPS as S
        for s in steps:
            st = S.get(s["name"])
            if not st:
                continue
            it = QListWidgetItem(st["label"])
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if s.get("enabled", True) else Qt.Unchecked)
            it.setData(Qt.UserRole, s["name"])
            lw.addItem(it)
        self._log(f"Processus chargé : {path}")

    # ------------------------------------------------------------- aperçu live
    def _base_preview_image(self):
        if self.preview_base is not None:
            return self.preview_base
        if self.state["image"] is not None:
            src = self.state["image"]
        elif self.state["light_paths"]:
            bayer = self._cmb("cmbBayer").currentText()
            src = load_image(self.state["light_paths"][0],
                             None if bayer.startswith("Aucune") else bayer)
        else:
            return None
        import cv2
        h, w0 = src.shape[:2]
        f = min(1.0, 700.0 / max(h, w0))
        if f < 1.0:
            src = cv2.resize(src, (int(w0 * f), int(h * f)), interpolation=cv2.INTER_AREA)
        src = np.clip(src, 0, None)
        if src.max() > 1.5:
            src = src / np.percentile(src, 99.5)
        self.preview_base = np.clip(src, 0, 1).astype(np.float32)
        return self.preview_base

    def _collect_preview_params(self):
        return self._trait_params()

    def _compute_preview(self, img, p, progress=None):
        out = img
        out, _ = extract_gradient(out, p["grad_method"], p["grad_deg"], p["grad_grid"],
                                  2.5, p["grad_corr"])
        out = denoise(out, p["denoise_method"], p["denoise_strength"])
        if p["deconv_iters"] > 0:
            out = richardson_lucy(out, p["deconv_sigma"], min(p["deconv_iters"], 40))
        return out

    def _invalidate_sky(self, images=False):
        if images:
            self.state["images"] = []
        self.state["image"] = None
        self.state["background_model"] = None
        self.state.pop("wcs", None)
        self.state.pop("pre_align", None)  # images remplacées : plus de drizzle possible
        self.preview_base = None
        self.shared["image"] = None
        self.pane.set_after(None)
        self.ui.findChild(object, "lblAstroResult").setText("—")

    def _load_lights_step(self):
        if not self.state["light_paths"]:
            self._add_lights()
        if not self.state["light_paths"]:
            return
        paths = list(self.state["light_paths"])
        bayer = self._cmb("cmbBayer").currentText()
        def run(progress=None):
            images = []
            for i, path in enumerate(paths):
                progress(i, len(paths), "Chargement")
                images.append(load_image(path, None if bayer.startswith("Aucune") else bayer))
            return images
        def done(images):
            self._invalidate_sky()
            self.state["images"] = images
            self.pane.set_before(images[0])
            self.pane.cmbMode.setCurrentIndex(1)
        self._start_job(run, done, self.ui)

    def _run_sky_steps(self, names):
        if names[0] == "calibration":
            if not self.state["light_paths"]:
                raise ValueError("Ajoutez d'abord des images brutes.")
        elif names[0] in ("registration", "drizzle"):
            if not self.state["images"]:
                raise ValueError("Lancez d'abord le chargement ou la calibration.")
        elif self.state["image"] is None:
            raise ValueError("Lancez d'abord l'empilement.")
        pipeline = Pipeline([{"name": n, "params": {}, "enabled": True} for n in names])
        self._start_job(self._run_pipeline_job(pipeline, reuse=True), self._on_full_done, self.ui)

    def _apply_ai_step(self):
        image = self._current_image()
        if image is None:
            raise ValueError("Chargez d'abord une image.")
        params = self._ia_params()
        def run(progress=None):
            from ..ai.manager import get_ai_manager
            return get_ai_manager().run(params["task"], image, params["provider"],
                                        params["strength"], progress=progress)
        self._start_job(run, self._after_edit, self.ui)

    def _apply_defects_step(self):
        if not self.state["images"]:
            raise ValueError("Lancez d'abord le chargement ou la calibration.")
        from ..core import defects as D
        images = list(self.state["images"])
        path = self._ed("txtBPM").text().strip()
        enabled = self._cbx("chkDefects").isChecked()
        auto = self._cbx("chkDefectsAuto").isChecked()
        cols = self._cbx("chkDefectsCols").isChecked()
        master = self.state.get("master_dark")
        if master is None:
            master = self.state.get("master_flat")
        def run(progress=None):
            mask = D.load_bad_pixel_map(path, images[0].shape[:2]) if path else None
            if enabled and mask is None and auto and master is not None:
                mask = D.detect_bad_pixels(master)
            columns = D.detect_bad_columns(master) if enabled and cols and master is not None else None
            rows = D.detect_bad_rows(master) if enabled and cols and master is not None else None
            output = []
            for i, image in enumerate(images):
                progress(i, len(images), "Pixels défectueux")
                output.append(D.fix_defects(image, bad=mask, cols=columns, rows=rows)
                              if enabled else image)
            return output
        def done(images):
            self._invalidate_sky()
            self.state["images"] = images
            self.pane.set_before(images[0])
            self.pane.cmbMode.setCurrentIndex(1)
        self._start_job(run, done, self.ui)

    # ------------------------------------------------------------- traitements
    def _run_pipeline_job(self, pipeline: Pipeline, progress=None, reuse=False):
        from ..processing.pipeline import STEPS as S
        settings = {
            "dirs": {n: self._ed(n).text() for n in ("txtBias", "txtDark", "txtFlat", "txtDarkFlat")},
            "defects": self._cbx("chkDefects").isChecked(),
            "defects_auto": self._cbx("chkDefectsAuto").isChecked(),
            "defects_cols": self._cbx("chkDefectsCols").isChecked(),
            "bpm": self._ed("txtBPM").text().strip(),
            "reg": self._cmb("cmbRegMethod").currentText(),
            "rotation": self._cbx("chkRotation").isChecked(),
            "drizzle": self._cmb("cmbDrizzle").currentText(),
            "pixfrac": self._spn("spnPixfrac").value(),
            "stack": self._cmb("cmbStackMethod").currentText(),
            "sigma": self._spn("spnSigma").value(),
            "ai": self._ia_params(),
            "calib_paths": self.state.get("calib_paths"),
            "output": self.cmbOutput.currentText(),
            "lib_use": self.chkLibUse.isChecked(),
            "lib_save": self.chkLibSave.isChecked(),
            "light_meta": self._light_meta() if self.chkLibUse.isChecked() else None,
        }
        hdr_paths = list(self.state["hdr_images"])
        wmap = dict(self.state.get("weights") or {})
        ctx = {"light_paths": list(self.state["light_paths"]),
               "weights": ([float(wmap.get(p, 1.0)) for p in self.state["light_paths"]]
                           if wmap else None),
               "images": list(self.state["images"]) if reuse else [],
               "pre_align": (list(self.state.get("pre_align") or [])
                             if reuse else []),
               "image": self.state["image"] if reuse else None,
               "background_model": self.state.get("background_model") if reuse else None,
               "hdr_images": [],
               "bayer": None if self._cmb("cmbBayer").currentText().startswith("Aucune")
               else self._cmb("cmbBayer").currentText(),
               "preview_base": self._base_preview_image(),
               "params_trait": self._trait_params(),
               "progress": progress}
        # injection des étapes GUI (calibration/align/stack/denoise) avec params UI
        def calib_step(ctx, p, progress=None):
            self._log("Chargement + calibration des lumières (CFA puis dématriçage)…")
            from ..core import calibration as C
            from ..core import defects as D
            dirs = settings["dirs"]
            masters = C.build_masters(dirs["txtBias"], dirs["txtDark"],
                                      dirs["txtFlat"], dirs["txtDarkFlat"],
                                      log=self._log, paths=settings["calib_paths"],
                                      light_meta=settings["light_meta"],
                                      use_library=settings["lib_use"],
                                      save_to_library=settings["lib_save"])
            md, mf = masters["dark"], masters["flat"]
            ctx["master_dark"], ctx["master_flat"] = md, mf
            ctx["image"] = ctx["background_model"] = None
            ctx.pop("pre_align", None)
            paths = ctx["light_paths"]
            if not paths:
                raise RuntimeError("Aucune image lumière à traiter")
            fix = None
            if settings["defects"]:
                bpm = settings["bpm"] or None
                ref_d = md if md is not None else mf
                state = {}

                def fix(img, cfa):
                    if "mask" not in state:
                        mask = D.load_bad_pixel_map(bpm, img.shape[:2]) if bpm else None
                        if mask is None and (settings["defects_auto"]
                                             or bpm is None) and ref_d is not None:
                            mask = D.detect_bad_pixels(ref_d)
                        cols = rows = None
                        if settings["defects_cols"] and ref_d is not None:
                            cols = D.detect_bad_columns(ref_d)
                            rows = D.detect_bad_rows(ref_d)
                        state.update(mask=mask, cols=cols, rows=rows)
                        self._log(f"Défauts corrigés : "
                                  f"{int(np.sum(mask)) if mask is not None else 0} px, "
                                  f"{0 if cols is None else int(np.sum(cols))} colonnes, "
                                  f"{0 if rows is None else int(np.sum(rows))} lignes")
                    if state["mask"] is None and state["cols"] is None and state["rows"] is None:
                        return img
                    return D.fix_defects(img, bad=state["mask"], cols=state["cols"],
                                         rows=state["rows"], cfa=cfa)
            ctx["images"] = C.calibrate_paths(paths, masters, bayer=ctx["bayer"],
                                              cosmetic=True, defects=fix,
                                              progress=progress,
                                              output=settings["output"])
            return ctx

        def reg_step(ctx, p, progress=None):
            self._log("Alignement…")
            ctx["image"] = ctx["background_model"] = None
            ctx["pre_align"] = list(ctx["images"])  # pour le drizzle
            if settings["reg"].startswith("Étoiles"):
                from ..core.astrometry import align_by_stars
                imgs = ctx["images"]
                ref = imgs[len(imgs) // 2]
                out = []
                for i, im in enumerate(imgs):
                    out.append(im if im is ref else align_by_stars(ref, im))
                    if progress:
                        progress(i + 1, len(imgs), "alignement étoiles")
                ctx["images"] = out
                return ctx
            if settings["reg"].startswith("Plate solving"):
                from ..core.registration import align_poly
                imgs = ctx["images"]
                ref = imgs[len(imgs) // 2]
                out = []
                for i, im in enumerate(imgs):
                    out.append(im if im is ref else align_poly(ref, im))
                    if progress:
                        progress(i + 1, len(imgs), "plate solving + poly")
                self._log("Alignement par plate solving par image (distorsions "
                          "polynomiales) appliqué.")
                ctx["images"] = out
                return ctx
            p = dict(p, with_rotation=settings["rotation"])
            ctx = S["registration"]["fn"](ctx, p, progress)
            return ctx

        def stack_step(ctx, p, progress=None):
            self._log("Empilement…")
            imgs = ctx["images"]
            mode = settings["drizzle"]
            if mode.startswith("×"):
                from ..processing.pipeline import drizzle_images
                if not ctx.get("pre_align"):
                    self._log("Drizzle : images non alignées par AstroForge — "
                              "transformations mesurées directement.")
                img = drizzle_images(ctx, int(mode[1]), settings["pixfrac"],
                                     settings["stack"], settings["sigma"],
                                     with_rotation=settings["rotation"],
                                     progress=progress, weights=ctx.get("weights"))
            else:
                from ..core.stacking import stack as st
                w = ctx.get("weights")
                img = st(np.stack(imgs), settings["stack"], sigma=settings["sigma"],
                         progress=progress,
                         weights=w if w and len(w) == len(imgs) else None)
            ctx["image"] = img
            ctx["background_model"] = None
            return ctx

        def grad_step(ctx, p, progress=None):
            self._log("Extraction de fond…")
            tp = ctx["params_trait"]
            ctx["image"], ctx["background_model"] = extract_gradient(
                ctx["image"], tp["grad_method"], tp["grad_deg"], tp["grad_grid"],
                2.5, tp["grad_corr"])
            return ctx

        def dn_step(ctx, p, progress=None):
            self._log("Débruitage…")
            tp = ctx["params_trait"]
            ctx["image"] = denoise(ctx["image"], tp["denoise_method"],
                                   tp["denoise_strength"])
            return ctx

        def dc_step(ctx, p, progress=None):
            self._log("Déconvolution…")
            tp = ctx["params_trait"]
            ctx["image"] = richardson_lucy(ctx["image"], tp["deconv_sigma"],
                                           tp["deconv_iters"])
            return ctx

        custom = {"calibration": calib_step, "registration": reg_step,
                  "drizzle": stack_step, "gradient": grad_step,
                  "denoise": dn_step, "deconvolution": dc_step}

        def run(progress=None):
            c = ctx
            c["hdr_images"] = [load_image(path) for path in hdr_paths]
            enabled = [s for s in pipeline.steps if s.get("enabled", True)]
            total = len(enabled)
            for i, s in enumerate(enabled):
                fn = custom.get(s["name"], S[s["name"]]["fn"])
                label = S[s["name"]]["label"]
                if progress:
                    progress(i, total, label)
                params = dict(S[s["name"]]["params"])
                params.update(s.get("params", {}))
                if s["name"] == "ai_tool":
                    params.update(settings["ai"])
                c = fn(c, params, progress)
                if progress:
                    progress(i + 1, total, label)
            return c
        return run

    def _ed(self, name):
        from PySide6.QtWidgets import QLineEdit
        return self.ui.findChild(QLineEdit, name)

    def _process_full(self):
        if not self.state["light_paths"]:
            QMessageBox.information(self, "Traitement",
                                    "Ajoutez d'abord des images brutes.")
            return
        self.preview_base = None
        pipeline = self._pipeline_from_list()
        self._log(f"Traitement de {len(self.state['light_paths'])} images — "
                  f"{sum(1 for s in pipeline.steps if s.get('enabled', True))} étapes…")
        self._start_job(self._run_pipeline_job(pipeline),
                        self._on_full_done, self.ui)

    def _on_full_done(self, ctx):
        self.state.update({k: v for k, v in ctx.items() if k in
                           ("images", "image", "background_model", "master_dark",
                            "master_flat", "pre_align")})
        if "pre_align" not in ctx:
            self.state.pop("pre_align", None)
        self.state.pop("wcs", None)
        self.ui.findChild(object, "lblAstroResult").setText("—")
        self.preview_base = None
        self._base_preview_image()
        self.pane.set_after(self.state["image"])
        if self.state["image"] is not None:
            self.pane.cmbMode.setCurrentIndex(0)
        elif self.state["images"]:
            self.pane.set_before(self.state["images"][0])
            self.pane.cmbMode.setCurrentIndex(1)
        self._log("✔ Traitement terminé — utilisez « Exporter l'image… ».")
        self.shared["image"] = self.state["image"]

    def _start_job(self, fn_factory, done_cb, ui):
        from PySide6.QtWidgets import QProgressBar
        bar = ui.findChild(QProgressBar, "progress")
        self._job = self._new_job(fn_factory)
        self._job.progress.connect(
            lambda d, t, lbl: (bar.setMaximum(t), bar.setValue(d),
                               self._log(f"[{d}/{t}] {lbl}") if lbl else None))
        self._connect_done(self._job, done_cb)
        self._connect_error(self._job, lambda m: (self._log("✖ " + m.splitlines()[-1]),
                                         QMessageBox.critical(self, "Erreur", m[-2000:])))
        self._job.start()

    def export_to(self, path: str, fmt: str = "") -> None:
        """Exporte l'image traitée : FITS ou XISF (en-tête complet : nombre
        d'images, pose totale, dates, objet, filtre, WCS avec distorsions SIP),
        TIFF 16 bits ou 32 bits flottant, PNG 16 bits."""
        from ..core.session import stack_header
        img = self.state["image"]
        low = path.lower()
        if low.endswith((".fits", ".fit", ".fts", ".xisf")):
            hdr = {}
            if self.state["light_paths"]:
                try:
                    w = self.state.get("weights") or {}
                    hdr = stack_header(self.state["light_paths"],
                                       [w.get(p, 1.0) for p in self.state["light_paths"]]
                                       if w else None)
                except Exception as e:
                    self._log(f"En-tête partiel : {e}")
            if self.state.get("wcs") is not None:
                try:
                    wh = self.state["wcs"].to_header(relax=True)
                    hdr.update({k: wh[k] for k in wh.keys()})
                except Exception as e:
                    self._log(f"WCS non inscrit : {e}")
            save_image(path, img, header=hdr)
        elif low.endswith((".tif", ".tiff")):
            save_image(path, img, "32f" if "32" in fmt else "16")
        elif low.endswith(".png"):
            save_image(path, img, "16")
        else:
            save_image(path, img)
        self._log(f"Image exportée : {path}")

    def _export(self):
        if self.state["image"] is None:
            QMessageBox.information(self, "Export", "Aucune image traitée à exporter.")
            return
        path, fmt = QFileDialog.getSaveFileName(
            self, "Exporter l'image", "resultat.fits",
            "FITS 32 bits flottant (*.fits);;XISF (*.xisf);;TIFF 16 bits (*.tif);;"
            "TIFF 32 bits flottant (*.tif);;PNG 16 bits (*.png)")
        if path:
            self.export_to(path, fmt)

    # ------------------------------------------------- couleur & astrométrie
    def _current_image(self):
        img = self.state.get("image")
        return img if img is not None else self._base_preview_image()

    def _after_edit(self, img):
        self.state["image"] = img
        self.state["background_model"] = None
        self.state.pop("wcs", None)
        self.ui.findChild(object, "lblAstroResult").setText("—")
        self.preview_base = None
        self.shared["image"] = img
        self.pane.set_after(img)
        self.pane.cmbMode.setCurrentIndex(0)

    def _color_cal(self):
        img = self._current_image()
        if img is None:
            return
        from ..core.colorcal import color_calibrate
        self._after_edit(color_calibrate(
            img, use_background=True, percentile=self._spn("spnBgPct").value()))
        self._log("Calibration couleur sur le fond du ciel appliquée.")

    def _bg_neutral(self):
        img = self._current_image()
        if img is None:
            return
        from ..core.colorcal import background_neutralize
        self._after_edit(background_neutralize(
            img, self._spn("spnBgPct").value()))
        self._log("Fond neutralisé.")

    def _scnr(self):
        img = self._current_image()
        if img is None:
            return
        from ..core.colorcal import scnr_green
        self._after_edit(scnr_green(img))
        self._log("SCNR vert appliqué.")

    def _pick_catalog(self):
        f, _ = QFileDialog.getOpenFileName(self, "Catalogue local (CSV ra,dec,mag)",
                                           "", "Catalogue (*.csv *.fits)")
        if f:
            self._ed("txtCatalog").setText(f)

    def _pick_bpm(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Carte de pixels défectueux (pixels > 0,5 = défauts)",
            "", "Cartes (*.fits *.fit *.png *.tif *.tiff *.txt *.csv)")
        if f:
            self._ed("txtBPM").setText(f)

    @staticmethod
    def _parse_coord(txt: str, is_ra: bool):
        """Délègue au parseur commun (h/m/s, °/'/\", d/m/s, décimal)."""
        from ..core.equipment import parse_coord
        return parse_coord(txt, is_ra)

    def _plate_solve(self):
        img = self._current_image()
        if img is None:
            QMessageBox.information(self, "Astrométrie", "Chargez d'abord une image.")
            return
        from ..core import astrometry as A
        from ..core.equipment import load_settings
        st = load_settings()
        cat = self._ed("txtCatalog").text().strip() or st.get("gaia_catalog", "")
        ra = self._parse_coord(self._ed("txtRA").text(), True)
        dec = self._parse_coord(self._ed("txtDec").text(), False)

        radius = self._spn("spnRadius").value()
        def run(progress=None):
            return A.solve_image(img, catalog_path=cat or None, center_ra=ra,
                                 center_dec=dec,
                                  radius_deg=radius,
                                 prefer_solve_field=bool(st.get("use_solve_field", True)))
        self._job = self._new_job(run)
        self._connect_done(self._job, self._solve_done)
        self._connect_error(self._job, lambda m: (self._log("✖ " + m.splitlines()[-1]),
                                         QMessageBox.critical(self, "Plate solving",
                                                              m[-1500:])))
        self._job.start()

    def _solve_done(self, res):
        from PySide6.QtWidgets import QLabel
        self.state["wcs"] = res.get("wcs")
        msg = (f"Résolu ({res['method']}) — {res['scale_arcsec_pix']:.2f}\"/px, "
               f"centre {res['center_ra']:.4f}° {res['center_dec']:+.4f}°, "
               f"{res['n_matched']} étoiles, rms {res['rms_arcsec']:.2f}\"")
        lbl = self.ui.findChild(QLabel, "lblAstroResult")
        if lbl:
            lbl.setText(msg)
        self._log("✔ Plate solving : " + msg)

    def _photometry(self):
        img = self._current_image()
        if img is None:
            return
        from ..core import astrometry as A
        cat = self._ed("txtCatalog").text().strip()
        wcs = self.state.get("wcs")

        def run(progress=None):
            return A.aperture_photometry(img, wcs=wcs, catalog_path=cat or None)
        self._job = self._new_job(run)

        def done(tab):
            path, _ = QFileDialog.getSaveFileName(self, "Photométrie (CSV)",
                                                  "photometrie.csv", "CSV (*.csv)")
            if path:
                tab.write(path, format="csv", overwrite=True)
            self._log(f"Photométrie : {len(tab)} étoiles mesurées"
                      f"{' → ' + path if path else ''}")
        self._connect_done(self._job, done)
        self._connect_error(self._job, lambda m: (self._log("✖ " + m.splitlines()[-1]),
                                         QMessageBox.critical(self, "Photométrie",
                                                              m[-1500:])))
        self._job.start()

    def _pcc(self):
        img = self._current_image()
        wcs = self.state.get("wcs")
        if img is None or wcs is None:
            QMessageBox.information(
                self, "PCC",
                "Lancez d'abord le plate solving (groupe 9) : il fournit le WCS "
                "nécessaire à l'appariement des étoiles.")
            return
        from ..core.equipment import load_settings
        cat = self._ed("txtCatalog").text().strip() or \
            load_settings().get("gaia_catalog", "")
        if not cat:
            QMessageBox.information(self, "PCC",
                                    "Indiquez un catalogue (CSV avec BP/RP).")
            return
        from ..core.colorcal import photometric_color_calibration

        def run(progress=None):
            return photometric_color_calibration(img, wcs, cat)
        self._job = self._new_job(run)
        self._connect_done(self._job, lambda r: (self._after_edit(r[0]),
                                        self._log(f"✔ PCC : {r[1]['n']} étoiles — "
                                                  f"kR={r[1]['kR']:.3f} "
                                                  f"kB={r[1]['kB']:.3f}")))
        self._connect_error(self._job, lambda m: (self._log("✖ " + m.splitlines()[-1]),
                                         QMessageBox.critical(self, "PCC", m[-1500:])))
        self._job.start()

    def _dph(self):
        img = self._current_image()
        wcs = self.state.get("wcs")
        if img is None or wcs is None:
            QMessageBox.information(self, "Photométrie différentielle",
                                    "Lancez d'abord le plate solving (groupe 9).")
            return
        from ..core.equipment import load_settings
        cat = self._ed("txtCatalog").text().strip() or \
            load_settings().get("gaia_catalog", "")
        if not cat:
            QMessageBox.information(self, "Photométrie différentielle",
                                    "Indiquez un catalogue (CSV).")
            return
        ra = self._parse_coord(self._ed("txtRA").text(), True)
        dec = self._parse_coord(self._ed("txtDec").text(), False)
        from ..core.astrometry import differential_photometry

        def run(progress=None):
            return differential_photometry(img, wcs, cat, ra, dec)
        self._job = self._new_job(run)
        self._connect_done(self._job, self._dph_done)
        self._connect_error(self._job, lambda m: (self._log("✖ " + m.splitlines()[-1]),
                                         QMessageBox.critical(
                                             self, "Photom. diff.", m[-1500:])))
        self._job.start()

    def _dph_done(self, tab):
        path, _ = QFileDialog.getSaveFileName(
            self, "Photométrie différentielle (CSV)", "photom_diff.csv", "CSV (*.csv)")
        if path:
            tab.write(path, format="csv", overwrite=True)
        t = tab.meta
        self._log(f"✔ Photométrie différentielle : dmag = {t['dmag']:+.3f} "
                  f"± {t['dmag_err']:.3f} ({t['n_comp']} comparaisons, "
                  f"mag réf. {t['mag_ref_ensemble']:.2f})"
                  f"{' → ' + path if path else ''}")

    # API pour la fenêtre principale
    def refresh_all(self):
        self.live.trigger()
