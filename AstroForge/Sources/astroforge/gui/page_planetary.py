# -*- coding: utf-8 -*-
"""Onglet Planétaire — parcours en 6 étapes (esprit Registax6 / AstroSurface /
AutoStakkert!) :
  1. chargement du fichier + navigation image par image ;
  2. méthode de tri, classement des frames de la meilleure à la moins bonne
     (défilement des images triées) ;
  3. nombre d'images à aligner ;
  4. méthode d'alignement (grille NxN de carrés, global, points manuels
     placés au clic sur l'aperçu) ;
  5. alignement de toutes les images sélectionnées + empilement ;
  6. visualisation de l'image finale (puis ondelettes, IA, film, export…).
"""
from __future__ import annotations
import os

import numpy as np
from PySide6.QtWidgets import (QComboBox, QFileDialog, QLabel,
                               QMessageBox, QPlainTextEdit, QProgressBar,
                               QVBoxLayout,
                               QWidget)

from ..core import planetary as PL
from ..core import registration
from ..core.animation import export_video, stabilized_frames
from ..core.effects import apply_palette
from ..core.image_io import is_color, load_image, load_ser, save_image
from ..core.wavelets import registax_enhance
from .loader import load_ui
from .preview import CurveEditor, LivePreview, PreviewPane
from .profiles_ui import ProfileBar
from .commands import StepCommands


class PagePlanetary(StepCommands, QWidget):
    def __init__(self, parent=None, shared=None):
        super().__init__(parent)
        from PySide6.QtCore import Qt as QtC
        self.setFocusPolicy(QtC.StrongFocus)  # Ctrl+Z (annulation) sur la page
        self.shared = shared if shared is not None else {}
        self.ui = load_ui("page_planetary.ui", self)
        _outer = QVBoxLayout(self)   # la page suit la taille de la fenêtre
        _outer.setContentsMargins(0, 0, 0, 0)
        _outer.addWidget(self.ui)

        pv = self.ui.findChild(QVBoxLayout, "previewContainer")
        self.pane = PreviewPane()
        # Barre compacte : conserver un vrai aperçu à 1024×700.
        self.pane.chkStretch.setText("Auto")
        self.pane.chkStretch.setToolTip("Étirement automatique pour l'affichage uniquement")
        pv.addWidget(self.pane)
        self.frames = []
        self.original_frames = ()
        self.applied_tone = PL.tone_settings()
        self.curve = CurveEditor()
        self.ui.findChild(QVBoxLayout, "curveContainer").addWidget(self.curve)
        self.frame_info = ""
        self.frame_meta = {}
        self.result = None
        self.preview_base = None
        self.scores = None          # étape 2 : qualité par frame
        self.order = None           # étape 2 : indices triés (meilleur -> pire)
        self.roi_used = None
        self.manual_points = []     # étape 4 : points placés au clic
        self._placing = False
        self.live = LivePreview(self.pane, self._preview_params, self._wave_preview, self)
        from .ai_dialog import populate_ai_tasks
        from ..ai.manager import register_provider_refresh
        populate_ai_tasks(self._o("cmbAITask"))
        self._o("cmbAITask").currentIndexChanged.connect(self._refresh_ai_providers)
        register_provider_refresh(self._refresh_ai_providers)
        self._build_profiles()
        self.selected = None
        self.align_settings = None
        self.stack_base = None
        self._init_commands({
            "grpSource": self._load_source, "grpRank": self._run_rank,
            "grpNAlign": self._select_frames, "grpAlign": self._apply_alignment,
            "grpStackP": self._run_all, "grpWave": self._apply_waves,
            "grpIA": self._ai_apply, "grpAnim": self._make_movie,
            "grpFalseColor": self._apply_false_color, "grpSeqP": self._batch_ser,
            "grpCA": self._apply_ca, "grpRun": self._show_final,
            "grpTone": self._apply_tone,
        }, ("frames", "original_frames", "applied_tone", "frame_info", "result", "preview_base", "scores", "order",
            "roi_used", "manual_points", "_placing", "selected", "align_settings", "stack_base"))
        # Film et batch n'écrivent que des fichiers : pas d'annulation globale.
        self._no_undo_commands = {"grpAnim", "grpSeqP"}
        self._wire()
        self._sync_rank_navigation()
        self._refresh_ai_providers()
        self._log("AstroForge — mode Planétaire prêt (parcours en 6 étapes).")

    # ------------------------------------------------------------------ UI
    def _log(self, msg):
        box = self.ui.findChild(QPlainTextEdit, "log")
        if box:
            from PySide6.QtCore import Q_ARG, QMetaObject, Qt
            QMetaObject.invokeMethod(box, "appendPlainText", Qt.AutoConnection, Q_ARG(str, msg))

    def _o(self, name):
        return self.ui.findChild(object, name)

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

    def _wire(self):
        self._o("btnLoadSource").clicked.connect(lambda: self._execute("grpSource"))
        self._o("sldBrowse").valueChanged.connect(self._on_browse)
        self._o("btnPrevFrame").clicked.connect(
            lambda: self._o("sldBrowse").setValue(self._o("sldBrowse").value() - 1))
        self._o("btnNextFrame").clicked.connect(
            lambda: self._o("sldBrowse").setValue(self._o("sldBrowse").value() + 1))
        self._o("btnRank").setText("Lancer le tri")
        self._o("btnRank").clicked.connect(lambda: self._execute("grpRank"))
        self._o("sldRankBrowse").valueChanged.connect(self._on_rank_browse)
        self._o("btnPrevRank").clicked.connect(
            lambda: self._o("sldRankBrowse").setValue(self._o("sldRankBrowse").value() - 1))
        self._o("btnNextRank").clicked.connect(
            lambda: self._o("sldRankBrowse").setValue(self._o("sldRankBrowse").value() + 1))
        self._o("spnNAlign").valueChanged.connect(lambda _: self._update_ncut())
        self._o("btnPickPoints").toggled.connect(self._toggle_place)
        self._o("btnClearPoints").clicked.connect(self._clear_points)
        self.pane.canvasBefore.imageClicked.connect(self._on_canvas_click)
        self._o("btnWaveReset").clicked.connect(self._reset_waves)
        self._o("btnAIApply").clicked.connect(lambda: self._execute("grpIA"))
        self._o("btnAIConfig").clicked.connect(self._ai_config)
        self._o("btnRunAll").clicked.connect(lambda: self._execute("grpStackP"))
        self._o("btnExport").clicked.connect(self._export)
        from PySide6.QtWidgets import QPushButton
        b_wj = QPushButton("Exporter pour WinJUPOS…")
        b_wj.setObjectName("btnExportWinJupos")
        b_wj.setToolTip("Nom AAAA-MM-JJ-HHMM_D-Observateur-Filtre (instant médian des "
                        "frames retenues, d'après les horodatages SER)")
        exp = self._o("btnExport")
        lay = exp.parentWidget().layout() if exp is not None and exp.parentWidget() else None
        if lay is not None:
            try:
                lay.addWidget(b_wj)
            except TypeError:
                lay.addWidget(b_wj, lay.rowCount(), 0, 1, max(1, lay.columnCount()))
        b_wj.clicked.connect(self._export_winjupos)
        self._o("btnMakeMovie").clicked.connect(lambda: self._execute("grpAnim"))
        self._o("btnSaveSeqP").clicked.connect(self._save_planet_seq)
        self._o("btnBatchSER").clicked.connect(lambda: self._execute("grpSeqP"))
        self._o("chkCA").toggled.connect(self.live.trigger)
        self._o("sldCAR").valueChanged.connect(self.live.trigger)
        self._o("sldCAB").valueChanged.connect(self.live.trigger)
        self._o("btnTonePreview").clicked.connect(
            lambda: self._execute("grpTone", self._preview_tone, undoable=False))
        self._o("btnToneReset").clicked.connect(
            lambda: self._execute("grpTone", self._reset_tone))
        self._o("btnUndoP").clicked.connect(self._undo)
        self._o("btnAutoCompute").clicked.connect(self._auto_contrast)
        self._o("btnAutoGamma").clicked.connect(self._auto_gamma)
        self._o("spnAutoBlack").valueChanged.connect(self._apply_black_white)
        self._o("spnAutoWhite").valueChanged.connect(self._apply_black_white)
        self._o("spnAutoGamma").valueChanged.connect(self._apply_gamma_field)
        for name in ("spnBrightness", "spnContrast"):
            self._o(name).valueChanged.connect(self._tone_edited)
        self.curve.pointsChanged.connect(self._tone_edited)
        from PySide6.QtCore import Qt as QtC
        from PySide6.QtGui import QKeySequence, QShortcut
        shortcut = QShortcut(QKeySequence.Undo, self)
        shortcut.setContext(QtC.WidgetWithChildrenShortcut)
        shortcut.activated.connect(self._undo)
        self._o("btnLoadSeqP").clicked.connect(self._load_seq_p)
        for i in range(1, 7):
            for name in (f"sldW{i}", f"spnWDn{i}", f"spnWSh{i}"):
                self._o(name).valueChanged.connect(self.live.trigger)

    def _build_profiles(self):
        self.pb_rank = ProfileBar(
            "planet_rank", {"mode": "Contraste HF (AutoStakkert)", "n_align": 50},
            lambda: {"mode": self._o("cmbQualityMode").currentText(),
                     "n_align": self._o("spnNAlign").value()},
            lambda p: (self._o("cmbQualityMode").setCurrentText(
                           p.get("mode", "Contraste HF (AutoStakkert)")),
                       self._o("spnNAlign").setValue(int(p.get("n_align", 50)))))
        self.ui.findChild(QVBoxLayout, "profRank").addWidget(self.pb_rank)
        self.pb_wave = ProfileBar(
            "planet_wavelets", {"gains": [1.0] * 7,
                                "layer_denoise": [0.0] * 6,
                                "layer_sharpen": [0.0] * 6},
            self._wave_profile_params, self._set_wave_controls)
        self.ui.findChild(QVBoxLayout, "profWave").addWidget(self.pb_wave)
        self.pb_stack = ProfileBar(
            "planet_stack", {"method": "Moyenne", "percent": 90.0},
            lambda: {"method": self._o("cmbStackP").currentText(),
                     "percent": self._o("spnStackPercent").value()},
            lambda p: (self._o("cmbStackP").setCurrentText(p.get("method", "Moyenne")),
                       self._o("spnStackPercent").setValue(int(p.get("percent", 90)))))
        self.ui.findChild(QVBoxLayout, "profStackP").addWidget(self.pb_stack)
        self.pb_anim = ProfileBar(
            "planet_anim", {"crop_size": 0, "fps": 24, "fmt": "MP4", "quality": 90},
            lambda: {"crop_size": self._o("spnCropSize").value(),
                     "fps": self._o("spnFps").value(),
                     "fmt": self._o("cmbAnimFmt").currentText(),
                     "quality": self._o("spnAnimQuality").value()},
            lambda p: (self._o("spnCropSize").setValue(int(p.get("crop_size", 0))),
                       self._o("spnFps").setValue(int(p.get("fps", 24))),
                       self._o("cmbAnimFmt").setCurrentText(p.get("fmt", "MP4")),
                       self._o("spnAnimQuality").setValue(int(p.get("quality", 90)))))
        self.ui.findChild(QVBoxLayout, "profAnim").addWidget(self.pb_anim)
        self.pb_ca = ProfileBar(
            "planet_ca", {"enabled": False, "kr": 0.0, "kb": 0.0},
            lambda: {"enabled": self._o("chkCA").isChecked(),
                     "kr": self._o("sldCAR").value() / 100.0,
                     "kb": self._o("sldCAB").value() / 100.0},
            lambda p: (self._o("chkCA").setChecked(bool(p.get("enabled"))),
                       self._o("sldCAR").setValue(int(p.get("kr", 0.0) * 100)),
                       self._o("sldCAB").setValue(int(p.get("kb", 0.0) * 100))))
        self._prof_layout("profCA", "gridCA").addWidget(self.pb_ca)
        self.pb_tone = ProfileBar(
            "planet_tone", {"brightness": 0.0, "contrast": 1.0,
                            "curve": [[0.0, 0.0], [1.0, 1.0]]},
            self._tone_params, self._set_tone_controls)
        lay_tone = self.ui.findChild(QVBoxLayout, "profTone")
        if lay_tone is not None:
            lay_tone.addWidget(self.pb_tone)

    def _wave_params(self):
        return {"gains": [self._o(f"sldW{i}").value() / 100.0 for i in range(1, 7)]
                + [1.0],
                "ca": self._o("chkCA").isChecked(),
                "kr": self._o("sldCAR").value() / 100.0,
                "kb": self._o("sldCAB").value() / 100.0,
                "layer_denoise": [self._o(f"spnWDn{i}").value() for i in range(1, 7)],
                "layer_sharpen": [self._o(f"spnWSh{i}").value() for i in range(1, 7)]}

    def _preview_params(self):
        """Aperçu live : ondelettes + tonalité courante. (L'empilement part
        des frames déjà ajustées — jamais de tonalité appliquée deux fois.)"""
        p = self._wave_params()
        p["tone"] = self._tone_params()
        return p

    def _wave_profile_params(self):
        p = self._wave_params()
        return {k: p[k] for k in ("gains", "layer_denoise", "layer_sharpen")}

    def _set_wave_controls(self, p):
        gains = list(p.get("gains") or [])
        # Migration UI des anciens profils : force globale vers les six couches.
        dn = list(p.get("layer_denoise") or [p.get("wave_denoise", 0.0)] * 6)
        sh = list(p.get("layer_sharpen") or [])
        for i in range(1, 7):
            self._o(f"sldW{i}").setValue(round((gains + [1.0] * 6)[i - 1] * 100))
            self._o(f"spnWDn{i}").setValue((dn + [0.0] * 6)[i - 1])
            self._o(f"spnWSh{i}").setValue((sh + [0.0] * 6)[i - 1])

    def _reset_waves(self):
        self._set_wave_controls({})

    def _tone_params(self):
        return PL.tone_settings(self._o("spnBrightness").value(),
                                self._o("spnContrast").value(),
                                self.curve.points())

    def _set_tone_controls(self, p):
        p = PL.tone_settings(**p)
        for name, value in (("spnBrightness", p["brightness"]),
                            ("spnContrast", p["contrast"])):
            widget = self._o(name)
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self.curve.set_points(p["curve"], silent=True)
        self._tone_edited()

    def _tone_edited(self, *_):
        text = ("Réglages appliqués à toutes les images" if self._tone_params() == self.applied_tone
                else "Modifications non appliquées — cliquer Appliquer")
        self._o("lblToneState").setText(text)
        self.live.trigger()

    # --------------------------------------------- automatiques (esprit Photoshop)
    def _auto_base(self):
        if not self.frames:
            raise ValueError("Chargez d'abord une source.")
        return self._best_frame_small()  # aperçu réduit, calcul unique

    def _auto_contrast(self):
        try:
            base = self._auto_base()
        except ValueError as e:
            QMessageBox.information(self, "Contraste automatique", str(e))
            return
        clip = self._o("spnAutoClip").value() / 100.0
        _params, black, white = PL.auto_contrast_params(base, clip)
        for name, value in (("spnAutoBlack", black), ("spnAutoWhite", white)):
            widget = self._o(name)
            widget.blockSignals(True)
            widget.setValue(value)
            widget.blockSignals(False)
        self._apply_black_white()
        self._o("lblAutoInfo").setText(
            f"Noir {black:.3f} — Blanc {white:.3f} (rognage {clip * 100:.2f} %)")
        self._log(f"Contraste automatique : noir {black:.3f}, blanc {white:.3f} "
                  f"— modifiables, puis « Appliquer ».")

    def _apply_black_white(self):
        black = self._o("spnAutoBlack").value()
        white = self._o("spnAutoWhite").value()
        if white - black < 1e-3:
            return
        k = 1.0 / (white - black)
        self._set_tone_controls(
            PL.tone_settings(brightness=k * (0.5 + black) - 0.5, contrast=k))

    def _auto_gamma(self):
        try:
            base = self._auto_base()
        except ValueError as e:
            QMessageBox.information(self, "Luminosité automatique", str(e))
            return
        _params, gamma = PL.auto_brightness_params(base)
        widget = self._o("spnAutoGamma")
        widget.blockSignals(True)
        widget.setValue(gamma)
        widget.blockSignals(False)
        self._apply_gamma_field()
        self._log(f"Luminosité automatique : gamma {gamma:.2f} — modifiable, "
                  f"puis « Appliquer ».")

    def _apply_gamma_field(self):
        self._set_tone_controls(PL.tone_settings(
            curve=PL.gamma_curve(self._o("spnAutoGamma").value())))

    # ------------------------------------------------------ état courbe (annulation)
    def _extra_state(self):
        return {"curve": self.curve.points()}

    def _restore_extra(self, extra):
        self.curve.set_points(extra.get("curve") or [[0.0, 0.0], [1.0, 1.0]],
                              silent=True)

    def _tone_originals(self):
        if not self.frames:
            raise ValueError("Chargez d'abord une source.")
        if isinstance(self.frames, PL.ToneFrames):
            return self.frames.originals
        # Références uniquement ; également compatible avec une série injectée.
        return tuple(self.frames)

    def _apply_tone(self):
        view = PL.ToneFrames(self._tone_originals() if self.frames else (), self._tone_params())
        self._job = self._new_job(lambda progress=None: view.validate(progress))
        bar = self._o("progress")
        self._job.progress.connect(lambda d, t, l: (bar.setMaximum(t), bar.setValue(d)))
        self._connect_done(self._job, self._commit_tone)
        self._job.start()

    def _preview_tone(self):
        originals, params = self._tone_originals(), self._tone_params()
        index = min(self._o("sldBrowse").value(), len(originals) - 1)
        source = originals[index]
        def run(progress=None):
            if progress:
                progress(0, 1, "Aperçu tonalité")
            import cv2
            h, w = source.shape[:2]
            scale = min(1.0, 700 / max(h, w))
            image = cv2.resize(source, (max(1, round(w * scale)), max(1, round(h * scale)))) if scale < 1 else source
            return image, PL.adjust_tone(image, **params)
        def show(pair):
            self.pane.chkStretch.setChecked(False)
            self.pane.set_before(pair[0])
            self.pane.set_after(pair[1])
            self.pane.cmbMode.setCurrentIndex(2)
            self._o("lblToneState").setText("Aperçu seulement — cliquer Appliquer")
        self._job = self._new_job(run)
        self._connect_done(self._job, show)
        self._job.start()

    def _commit_tone(self, view):
        self.original_frames = view.originals
        self.frames = view
        self.applied_tone = view.params
        self.scores = self.order = self.roi_used = self.selected = self.align_settings = None
        self.manual_points = []
        self._placing = False
        self._o("btnPickPoints").setChecked(False)
        self._o("lblPoints").setText("0 point")
        self.preview_base = None
        self._invalidate_result()
        self.live.source = self._best_frame_small()
        self._o("lblRankInfo").setText("Tri non effectué — tonalité modifiée")
        self._sync_rank_navigation()
        self._update_ncut()
        self._on_browse(self._o("sldBrowse").value())
        self._tone_edited()
        # Éliminer les instantanés descendants : ils ne doivent pas ressusciter
        # un classement ou un empilement calculé avec l'ancienne séquence.
        self._history = {k: v for k, v in self._history.items()
                         if k in ("grpSource", "grpTone")}
        self._log("Tonalité appliquée à toute la vidéo ; tri et résultats invalidés.")

    def _reset_tone(self):
        view = PL.ToneFrames(self._tone_originals())
        self._set_tone_controls({})
        self._commit_tone(view)

    def _invalidate_result(self):
        self.result = None
        self.stack_base = None
        self.shared["image"] = None
        self.pane.set_after(None)

    def _sync_rank_navigation(self):
        ready = self.order is not None and len(self.order) > 0
        for name in ("sldRankBrowse", "btnPrevRank", "btnNextRank"):
            self._o(name).setEnabled(ready)
        self._o("sldRankBrowse").setRange(0, len(self.order) - 1 if ready else 0)

    def _select_frames(self):
        if self.order is None:
            raise ValueError("Lancez d'abord le tri à l'étape 2.")
        self.selected = self.order[:self._o("spnNAlign").value()]
        self.align_settings = None
        self._invalidate_result()
        self._update_ncut()
        self._log(f"[3] {len(self.selected)} meilleures images sélectionnées.")

    def _apply_alignment(self):
        if self.selected is None:
            raise ValueError("Validez la sélection à l'étape 3.")
        settings = self._collect_settings()
        if settings["align_mode"].startswith("Points manuels") and not self.manual_points:
            raise ValueError("Placez au moins un point sur l'aperçu.")
        self.align_settings = {k: settings[k] for k in
                               ("align_mode", "grid_n", "box_size")}
        self.align_settings["manual_points"] = list(self.manual_points)
        self._placing = False
        self._o("btnPickPoints").setChecked(False)
        self._invalidate_result()
        self._log("[4] Méthode et points validés pour l'alignement.")

    def _commit_image(self, image):
        self.result = image
        self.shared["image"] = image
        self.pane.set_after(image)
        self.pane.cmbMode.setCurrentIndex(0)

    def _image_base(self):
        if self.result is not None:
            return self.result
        if self.frames:
            return self.frames[int(self.order[0]) if self.order is not None else 0]
        raise ValueError("Chargez d'abord une source.")

    def _apply_waves(self):
        image, params = self._image_base(), self._wave_params()
        self._job = self._new_job(lambda progress=None: self._wave_preview(image, params, progress))
        self._connect_done(self._job, self._commit_image)
        self._connect_error(self._job, lambda msg: self._log(msg.splitlines()[-1]))
        self._job.start()

    def _apply_ca(self):
        from ..core.colorcal import correct_chromatic_aberration
        image = self._image_base()
        p = self._wave_params()
        self._commit_image(correct_chromatic_aberration(image, p["kr"], p["kb"])
                           if p["ca"] else image)

    def _apply_false_color(self):
        image = self._image_base()
        self._commit_image(apply_palette(image, self._o("cmbPalette").currentText())
                           if self._o("chkFalseColor").isChecked() and not is_color(image)
                           else image)

    def _show_final(self):
        if self.result is None:
            raise ValueError("Lancez l'alignement et l'empilement à l'étape 5.")
        self._commit_image(self.result)

    # ------------------------------------------------------------- 1) source
    def _load_source(self):
        stype = self._o("cmbSourceType").currentText()
        if stype.startswith("Vidéo"):
            path, _ = QFileDialog.getOpenFileName(
                self, "Vidéo planétaire", "",
                "Vidéos (*.ser *.avi *.mp4 *.mpeg *.wmv *.SER);;Tous les fichiers (*)")
            if not path:
                return
            try:
                if path.lower().endswith(".ser"):
                    frames, meta = load_ser(path, lazy=True)  # projeté en mémoire
                    self.frame_meta = meta
                    self.frame_info = (f"{meta['width']}×{meta['height']} — "
                                       f"{meta['frames']} frames — {meta['camera'] or 'SER'}")
                else:
                    frames, self.frame_info = _read_video(path)
                    self.frame_meta = {}
            except Exception as e:
                QMessageBox.critical(self, "Chargement impossible", f"{path}\n\n{e}")
                self._log(f"✘ Chargement : {e}")
                return
            if frames is None or getattr(frames, "size", 0) == 0:
                QMessageBox.critical(self, "Chargement impossible",
                                     f"{path}\n\nAucune frame lisible (fichier tronqué ?)")
                self._log("✘ Chargement : 0 frame lisible")
                return
        else:
            files, _ = QFileDialog.getOpenFileNames(
                self, "Images planétaires", "", "Images (*.fits *.tif *.tiff *.png *.jpg)")
            if not files:
                return
            frames = np.stack([load_image(f) for f in files])
            self.frame_info = f"{len(files)} images"
            self.frame_meta = {}
        from ..core.image_io import SerFrames
        self.original_frames = frames if isinstance(frames, SerFrames) else tuple(frames)
        self.frames = PL.ToneFrames(self.original_frames)
        self.applied_tone = PL.tone_settings()
        self._set_tone_controls({})
        self.selected = None
        self.align_settings = None
        self._invalidate_result()
        self.preview_base = None
        self.result = None
        self.scores = None
        self.order = None
        self.roi_used = None
        self.manual_points = []
        self._placing = False
        self._o("btnPickPoints").setChecked(False)
        self._o("lblPoints").setText("0 point")
        self._o("lblRankInfo").setText("Tri non effectué")
        self._o("sldRankBrowse").setRange(0, 0)
        n = len(self.frames)
        self._o("lblSourceInfo").setText(f"{self.frame_info} — {n} frames chargées")
        self._log(f"[1] Source chargée : {self.frame_info}")
        self._o("spnNAlign").setRange(1, n)
        self._o("spnNAlign").setValue(min(50, n))
        self._o("sldBrowse").setRange(0, n - 1)
        ref = self._best_frame_small()
        self.live.set_source(ref)
        self._o("sldBrowse").setValue(0)
        self._on_browse(0)
        self._sync_rank_navigation()
        self._history = {k: v for k, v in self._history.items() if k == "grpSource"}

    def _best_frame_small(self, maxpx=700):
        if self.preview_base is None:
            import cv2
            idx = int(np.argmax([f.mean() for f in self.frames[:50]])) if self.frames else 0
            f = self.frames[min(idx, len(self.frames) - 1)]
            h, w = f.shape[:2]
            sc = min(1.0, maxpx / max(h, w))
            if sc < 1.0:
                f = cv2.resize(f, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
            self.preview_base = np.clip(f, 0, 1).astype(np.float32)
            self.curve.set_image_histogram(self.preview_base)
        return self.preview_base

    def _on_browse(self, i):
        """Étape 1 : navigation image par image dans la vidéo chargée."""
        if not self.frames:
            return
        i = int(np.clip(int(i), 0, len(self.frames) - 1))
        self.pane.cmbMode.setCurrentIndex(1)
        self.pane.set_before(self.frames[i])
        self._o("lblBrowse").setText(f"Frame {i + 1} / {len(self.frames)}")

    # ------------------------------------------------------- 2) tri qualité
    def _run_rank(self):
        if not self.frames:
            QMessageBox.information(self, "Tri", "Chargez d'abord une source (étape 1).")
            return
        mode = self._o("cmbQualityMode").currentText()
        roi_mode = self._o("cmbRoiMode").currentText()
        frames = self.frames

        def run(progress=None):
            roi = None
            if roi_mode.startswith("Disque"):
                step = max(1, len(frames) // 20)
                mid = PL.mean_sample(frames, step, progress)
                roi = PL.detect_planet_box(mid)
            sc = PL.measure_quality(frames, mode, roi, progress=progress)
            return {"roi": roi, "scores": sc, "order": PL.sort_quality(sc)}

        self._job = self._new_job(run)
        bar: QProgressBar = self._o("progress")
        self._job.progress.connect(lambda d, t, l: (bar.setMaximum(t), bar.setValue(d)))
        self._connect_done(self._job, self._done_rank)
        self._connect_error(self._job, lambda m: (self._log("✖ " + m.splitlines()[-1]),
                                         QMessageBox.critical(self, "Tri", m[-2000:])))
        self._log(f"[2] Mesure de la qualité ({mode}, {roi_mode.lower()})…")
        self._job.start()

    def _done_rank(self, res):
        self._invalidate_result()
        self.selected = None
        self.align_settings = None
        self.scores = res["scores"]
        self.order = res["order"]
        self.roi_used = res["roi"]
        n = len(self.frames)
        self._o("sldRankBrowse").setRange(0, n - 1)
        self._o("sldRankBrowse").setValue(0)
        self._update_ncut()
        self._sync_rank_navigation()
        self._on_rank_browse(0)  # setValue(0) n'émet rien si le curseur est déjà à zéro.
        zone = f"disque {self.roi_used}" if self.roi_used else "image entière"
        self._log(f"[2] ✔ Tri terminé ({zone}) — meilleure frame "
                  f"#{int(self.order[0]) + 1}/{n} ; défilez les images triées "
                  f"avec le curseur.")

    def _on_rank_browse(self, r):
        """Étape 2 : défilement des frames triées de la meilleure à la moins bonne."""
        if self.order is None or not self.frames:
            return
        r = int(np.clip(int(r), 0, len(self.order) - 1))
        fi = int(self.order[r])
        self.pane.cmbMode.setCurrentIndex(1)
        self.pane.set_before(self.frames[fi])
        self._o("lblRankInfo").setText(
            f"Rang {r + 1}/{len(self.order)} — frame #{fi + 1} — "
            f"qualité {float(self.scores[fi]):.3f}")

    def _update_ncut(self):
        n = self._o("spnNAlign").value()
        lbl: QLabel = self._o("lblNCut")
        if self.scores is not None and self.order is not None:
            q = float(self.scores[int(self.order[min(n, len(self.order)) - 1])])
            lbl.setText(f"Les {n} meilleures — qualité de coupure : {q:.3f}")
        else:
            lbl.setText(f"Les {n} meilleures (tri effectué à l'étape 2 pour "
                        f"la qualité de coupure)")

    # ------------------------------------------------- 4) points manuels
    def _toggle_place(self, on):
        self._placing = bool(on)
        btn = self._o("btnPickPoints")
        btn.setText("Cliquez sur l'aperçu pour placer les carrés…"
                    if on else "Placer des points (clic sur l'aperçu)")

    def _on_canvas_click(self, fx, fy):
        if not self._placing or not self.frames:
            return
        h, w = self.frames[0].shape[:2]
        x, y = float(fx) * w, float(fy) * h
        self.manual_points.append((round(x, 1), round(y, 1)))
        shown = self.manual_points[-6:]
        self._o("lblPoints").setText(
            f"{len(self.manual_points)} point(s) : "
            + " ".join(f"({px:.0f},{py:.0f})" for px, py in shown))

    def _clear_points(self):
        self.manual_points = []
        self._o("lblPoints").setText("0 point")

    # ------------------------------------------------------------- ondelettes live
    def _wave_preview(self, img, p, progress=None):
        out = img
        if p.get("tone"):
            out = PL.adjust_tone(out, **p["tone"])
        if p.get("ca"):
            from ..core.colorcal import correct_chromatic_aberration
            out = correct_chromatic_aberration(out, p.get("kr", 0.0),
                                               p.get("kb", 0.0))
        return registax_enhance(out, p["gains"], p.get("wave_dn", 0.0),
                                layer_denoise=p.get("layer_denoise"),
                                layer_sharpen=p.get("layer_sharpen"))

    # ------------------------------------------------------------- IA
    def _ai_config(self):
        from .ai_dialog import AIToolsDialog
        AIToolsDialog(self).exec()
        self._refresh_ai_providers()

    def _refresh_ai_providers(self):
        from ..ai.manager import get_ai_manager
        cmb: QComboBox = self._o("cmbAIProvider")
        task = self._o("cmbAITask").currentData()
        selected = cmb.currentData()
        cmb.clear()
        for prov in get_ai_manager().providers(task):
            cmb.addItem(prov["label"], prov["id"])
        index = cmb.findData(selected)
        if index >= 0:
            cmb.setCurrentIndex(index)

    def _ai_apply(self):
        if not self.frames:
            QMessageBox.information(self, "IA", "Chargez d'abord une source.")
            return
        cmb: QComboBox = self._o("cmbAIProvider")
        provider = cmb.currentData() or "onnx"
        task = self._o("cmbAITask").currentData()
        base = self._image_base()

        def run(progress=None):
            from ..ai.manager import get_ai_manager
            return get_ai_manager().run(task, base, provider, 0.5, progress=progress)
        self._job = self._new_job(run)
        self._connect_done(self._job, lambda img: (self._commit_image(img),
                                          self._log(f"IA {task} appliquée à l'aperçu.")))
        self._connect_error(self._job, lambda m: (self._log("✖ " + m.splitlines()[-1]),
                                         QMessageBox.critical(self, "IA", m[-1500:])))
        self._job.start()

    # ------------------------------------------ 5) alignement + 6) image finale
    def _run_all(self):
        if not self.frames:
            QMessageBox.information(self, "Traitement", "Chargez d'abord une source (étape 1).")
            return
        st = self._collect_settings()
        if self.align_settings is not None:
            st.update(self.align_settings)
        if self.selected is not None:
            st["n_align"] = len(self.selected)
        points = list(st.get("manual_points", self.manual_points))
        waves = self._wave_params()
        ranked_scores, ranked_order, ranked_roi = self.scores, self.order, self.roi_used
        frames0 = self.frames

        def run(progress=None):
            frames = frames0
            if st["derot"]:
                if st["derot_mode"] == "auto":
                    frames = registration.derotate(frames, auto_max_deg=2.0,
                                                   progress=progress)
                else:
                    frames = registration.derotate(
                        frames, rate_deg_per_min=st["derot_rate"],
                        interval_s=st["interval"], progress=progress)
            res = PL.run_planetary(
                frames, quality_method=st["quality_method"],
                roi_mode=st["roi_mode"], n_align=st["n_align"],
                align_mode=st["align_mode"], grid_n=st["grid_n"],
                box_size=st["box_size"], manual_points=points,
                stack_method=st["stack_method"],
                stack_percent=st["stack_percent"], progress=progress,
                ranked_scores=ranked_scores, ranked_order=ranked_order,
                ranked_roi=ranked_roi)
            img = res["image"]
            if st["rgb_align"] and is_color(img):
                img = registration.align_channels(img)
            res["stack_base"] = img
            img = self._wave_preview(img, waves)
            if st["false_color"] and not is_color(img):
                img = apply_palette(img, st["palette"])
            res["image"] = img
            return res

        self._job = self._new_job(run)
        bar: QProgressBar = self._o("progress")
        self._job.progress.connect(lambda d, t, l: (bar.setMaximum(t), bar.setValue(d)))
        self._connect_done(self._job, self._done_all)
        self._connect_error(self._job, lambda m: (self._log("✖ " + m.splitlines()[-1]),
                                         QMessageBox.critical(self, "Erreur", m[-2000:])))
        self._log(f"[5] Alignement de {st['n_align']} images "
                  f"({st['align_mode']}) puis empilement…")
        self._job.start()

    def _done_all(self, res):
        s = res.get("stats", {})
        self.stack_base = res.get("stack_base", res["image"])
        self.scores, self.order = res["scores"], res["order"]
        self.selected = res["selected"]
        self.roi_used = res.get("stats", {}).get("roi")
        self._sync_rank_navigation()
        self._commit_image(res["image"])
        self.pane.set_after(self.result)          # 6) visualisation de l'image finale
        self.shared["image"] = self.result
        self._log(f"[5] ✔ {s.get('n_selected', '?')} frames alignées — "
                  f"{s.get('n_points', '?')} point(s)/carré(s), "
                  f"{s.get('rejected_boxes', 0)} boîte(s) rejetée(s), "
                  f"décalage moyen {s.get('shift_mean', 0.0):.2f} px "
                  f"(max {s.get('shift_max', 0.0):.2f}) ; "
                  f"{s.get('stacked', '?')} empilées.")
        self._log("[6] ✔ Image finale prête (panneau « Après ») — ondelettes/"
                  "IA/export disponibles.")

    # ------------------------------------------------------------- film
    def _make_movie(self):
        if not self.frames:
            QMessageBox.information(self, "Film", "Chargez d'abord une source.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer le film",
            "animation." + self._o("cmbAnimFmt").currentText().split()[0].lower(),
            "Vidéo (*.mp4 *.avi *.gif)")
        if not path:
            return
        self._o("cmbAnimFmt").currentText()
        fps = self._o("spnFps").value()
        quality = self._o("spnAnimQuality").value()
        crop_size = self._o("spnCropSize").value()
        auto_crop = self._o("chkAutoCrop").isChecked()
        frames0 = self.frames

        def run(progress=None):
            frames = frames0
            if auto_crop:
                frames = stabilized_frames(frames0, size=int(crop_size) if crop_size else None,
                                           progress=progress)
                self._log(f"Film stabilisé : recadrage commun {frames.box}")
            export_video(frames, path, fps=fps, quality=quality, progress=progress)
            return path
        self._job = self._new_job(run)
        bar: QProgressBar = self._o("progress")
        self._job.progress.connect(lambda d, t, l: (bar.setMaximum(t), bar.setValue(d)))
        self._connect_done(self._job, lambda p: self._log(f"🎬 Film écrit : {p}"))
        self._connect_error(self._job, lambda m: (self._log("✖ " + m.splitlines()[-1]),
                                         QMessageBox.critical(self, "Film", m[-1500:])))
        self._job.start()

    # ------------------------------------------------- séquence & batch SER
    def _collect_settings(self):
        return {"quality_method": self._o("cmbQualityMode").currentText(),
                "roi_mode": self._o("cmbRoiMode").currentText(),
                "n_align": self._o("spnNAlign").value(),
                "align_mode": self._o("cmbAlignKindP").currentText(),
                "grid_n": self._o("spnGridN").value(),
                "box_size": self._o("spnBoxSize").value(),
                "derot": self._o("chkDerot").isChecked(),
                "derot_mode": self._o("cmbDerotMode").currentText(),
                "derot_rate": self._o("spnDerotRate").value(),
                "interval": self._o("spnInterval").value(),
                "rgb_align": self._o("chkRGBAlign").isChecked(),
                "ca": self._o("chkCA").isChecked(),
                "kr": self._o("sldCAR").value() / 100.0,
                "kb": self._o("sldCAB").value() / 100.0,
                "stack_method": self._o("cmbStackP").currentText(),
                "stack_percent": self._o("spnStackPercent").value(),
                "gains": self._wave_params()["gains"],
                "layer_denoise": self._wave_params()["layer_denoise"],
                "layer_sharpen": self._wave_params()["layer_sharpen"],
                # Seuls les paramètres validés sont rejoués dans les lots SER.
                "tone": self.applied_tone,
                "false_color": self._o("chkFalseColor").isChecked(),
                "palette": self._o("cmbPalette").currentText()}

    def _save_planet_seq(self):
        from ..core.sequence import save_sequence
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer la séquence planétaire", "sequence_planetaire.json",
            "Séquence (*.json)")
        if path:
            save_sequence(path, [{"op": "planet_chain",
                                  "params": self._collect_settings()}], mode="ser")
            self._log(f"Séquence planétaire enregistrée : {path}")

    def _batch_ser(self):
        in_dir = QFileDialog.getExistingDirectory(self, "Dossier de vidéos SER")
        if not in_dir:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Dossier de sortie")
        if not out_dir:
            return
        from ..core import sequence as Q
        maxf = self._o("spnBatchMax").value()
        settings = self._collect_settings()

        def run(progress=None):
            seq = {"mode": "ser",
                   "ops": [{"op": "planet_chain", "params": settings}]}
            return Q.run_batch_ser_chain(seq, in_dir, out_dir, max_frames=maxf,
                                         progress=progress)
        self._job = self._new_job(run)
        bar = self._o("progress")
        self._job.progress.connect(lambda d, t, l: (bar.setMaximum(t), bar.setValue(d)))
        self._connect_done(self._job,
            lambda files: self._batch_done(files, out_dir))
        self._connect_error(self._job, lambda m: (self._log("✖ " + m.splitlines()[-1]),
                                         QMessageBox.critical(self, "Batch", m[-1500:])))
        self._job.start()

    def _batch_done(self, files, out_dir):
        summary = files.summary() if hasattr(files, "summary") else f"{len(files)} empilements"
        self._log(f"✔ Batch SER terminé : {summary}")
        if getattr(files, "errors", None):
            QMessageBox.warning(self, "Batch SER", f"{summary}\n\nDétail : "
                                f"{os.path.join(out_dir, 'rapport_batch.txt')}")

    def _apply_settings(self, st: dict):
        """Applique des réglages enregistrés aux widgets (séquence chargeable)."""
        self._o("cmbQualityMode").setCurrentText(
            st.get("quality_method", st.get("quality_mode",
                                            "Contraste HF (AutoStakkert)")))
        self._o("cmbRoiMode").setCurrentText(st.get("roi_mode", "Disque auto-détecté"))
        self._o("spnNAlign").setValue(int(st.get("n_align", 50)))
        self._o("cmbAlignKindP").setCurrentText(
            st.get("align_mode", "Multi-points (grille NxN)"))
        self._o("spnGridN").setValue(int(st.get("grid_n", 4)))
        self._o("spnBoxSize").setValue(int(st.get("box_size", 0)))
        self._o("chkDerot").setChecked(bool(st.get("derot")))
        self._o("cmbDerotMode").setCurrentText(st.get("derot_mode", "auto"))
        self._o("spnDerotRate").setValue(float(st.get("derot_rate", 0.0)))
        self._o("spnInterval").setValue(float(st.get("interval", 1.0)))
        self._o("chkRGBAlign").setChecked(bool(st.get("rgb_align")))
        self._o("chkCA").setChecked(bool(st.get("ca")))
        self._o("sldCAR").setValue(int(st.get("kr", 0.0) * 100))
        self._o("sldCAB").setValue(int(st.get("kb", 0.0) * 100))
        self._o("cmbStackP").setCurrentText(st.get("stack_method", "Moyenne"))
        self._o("spnStackPercent").setValue(int(st.get("stack_percent", 90)))
        self._set_wave_controls(st)
        self._set_tone_controls(st.get("tone") or {})
        # Charger des réglages ne modifie pas les pixels courants : Appliquer
        # reste explicite avant tri/film/batch de la série chargée.
        self._o("chkFalseColor").setChecked(bool(st.get("false_color")))
        self._o("cmbPalette").setCurrentText(
            st.get("palette", "Solaire (orange)"))

    def _load_seq_p(self):
        """Charge une séquence enregistrée et applique ses réglages aux widgets
        (modifiables puis ré-enregistrables)."""
        from ..core.sequence import load_sequence
        path, _ = QFileDialog.getOpenFileName(self, "Charger une séquence", "",
                                              "Séquence (*.json)")
        if not path:
            return
        seq = load_sequence(path)
        st = seq["ops"][0]["params"] if seq.get("ops") else {}
        self._apply_settings(st)
        self._log(f"Séquence chargée ({path}) — modifiable, puis « Enregistrer »")

    def winjupos_time(self):
        """Instant médian (UT) des frames retenues, d'après les horodatages
        du SER ; à défaut la date d'en-tête."""
        from ..core.derotation import mid_time
        meta = getattr(self, "frame_meta", None) or {}
        ts = meta.get("timestamps") or []
        if ts:
            idx = list(self.selected) if self.selected is not None else range(len(ts))
            sel = [ts[int(i)] for i in idx if int(i) < len(ts) and ts[int(i)]]
            t = mid_time(sel or ts)
            if t:
                return t
        return meta.get("date_utc")

    def _export_winjupos(self):
        from PySide6.QtWidgets import QInputDialog
        from ..core.derotation import winjupos_name
        img = self.result
        if img is None:
            QMessageBox.information(self, "WinJUPOS", "Empilez d'abord une image.")
            return
        when = self.winjupos_time()
        if not when:
            when, ok = QInputDialog.getText(self, "WinJUPOS", "Instant UT (AAAA-MM-JJTHH:MM:SS) :")
            if not ok or not when:
                return
        obs, ok = QInputDialog.getText(self, "WinJUPOS", "Observateur (initiales) :")
        if not ok:
            return
        filt, ok = QInputDialog.getText(self, "WinJUPOS", "Filtre (RGB, R, IR685…) :", text="RGB")
        if not ok:
            return
        folder = QFileDialog.getExistingDirectory(self, "Dossier d'export WinJUPOS")
        if not folder:
            return
        try:
            name = winjupos_name(when, obs, filt, "png")
        except ValueError as e:
            QMessageBox.critical(self, "WinJUPOS", f"Instant invalide : {e}")
            return
        path = os.path.join(folder, name)
        from ..core.image_io import save_image
        save_image(path, img, "16")
        self._log(f"Export WinJUPOS : {path}")

    def _export(self):
        img = self.result if self.result is not None else (
            self.preview_base if self.preview_base is not None else None)
        if img is None:
            QMessageBox.information(self, "Export", "Aucune image à exporter.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exporter", "planete.png",
                                              "PNG (*.png);;FITS (*.fits);;TIFF (*.tif)")
        if path:
            save_image(path, img)
            self._log(f"Image exportée : {path}")


def _read_video(path: str, max_frames: int = 20000):
    """Lit AVI/MP4 (BGR -> RGB). Les frames sont gardées en 8 bits (4 fois
    moins de mémoire qu'en float32) et converties en float [0..1] à la
    demande (SerFrames)."""
    import cv2
    from ..core.image_io import SerFrames
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Impossible d'ouvrir la vidéo : {path}")
    frames = []
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(np.ascontiguousarray(f[:, :, ::-1]) if f.ndim == 3 else f)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    wh = (int(cap.get(3)), int(cap.get(4)))
    cap.release()
    if not frames:
        raise IOError("Aucune frame lue")
    raw = np.stack(frames)
    if raw.dtype != np.uint8:
        raw = raw.astype(np.uint16) if raw.max(initial=0) > 255 else raw.astype(np.uint8)
    scale = 255.0 if raw.dtype == np.uint8 else 65535.0
    info = f"{wh[0]}×{wh[1]} — {len(frames)} frames"
    if total > len(frames):
        info += f" (limitées sur {total})"
    return SerFrames(raw, scale, 0, None), info
