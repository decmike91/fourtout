# -*- coding: utf-8 -*-
"""Onglet commun : finition type GIMP (tonalité, géométrie, netteté),
retouches astro (courbes, filtres de réglages, IA locale), recalcul pour
augmenter la résolution (Lanczos ou IA), export."""
from __future__ import annotations
import os

import numpy as np
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
                               QLabel, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from ..core import finishing as FIN
from ..core import planetary as PL
from ..core.image_io import load_image, save_image
from ..core.restoration import unsharp
from .loader import load_ui
from .preview import CurveEditor, LivePreview, PreviewPane, draw_histogram_pixmap
from .profiles_ui import ProfileBar
from .workers import Job

DEFAULTS = {"bright": 0, "contrast": 0, "sat": 100, "gamma": 1.0, "wr": 1.0,
            "wg": 1.0, "wb": 1.0, "usm_amount": 0.6, "usm_radius": 2.0}


class PageCommon(QWidget):
    def __init__(self, parent=None, shared=None):
        super().__init__(parent)
        self.shared = shared or {}
        self.ui = load_ui("page_common.ui", self)
        _outer = QVBoxLayout(self)   # la page suit la taille de la fenêtre
        _outer.setContentsMargins(0, 0, 0, 0)
        _outer.addWidget(self.ui)

        pv = self.ui.findChild(QVBoxLayout, "previewContainer")
        self.pane = PreviewPane()
        pv.addWidget(self.pane)
        self.image = None   # image de travail pleine résolution
        self.preview = None  # version réduite
        self.seq_ops: list[dict] = []  # séquence batch enregistrée
        # bouton de chargement (non présent dans .ui) :
        btn = QPushButton("Charger une image à finir…")
        right = self.ui.findChild(QVBoxLayout, "rightCol")
        right.insertWidget(0, btn)
        btn.clicked.connect(self._load)
        # Grandes colonnes : largeur minimale des zones numériques plafonnée
        # (le panneau gauche doit tenir dans 1024 px sans barre horizontale).
        for name in ("spnCX", "spnCY", "spnCW", "spnCH",
                     "spnWR", "spnWG", "spnWB"):
            w = self.ui.findChild(object, name)
            if w is not None:
                w.setMinimumWidth(48)
        self._build_retouch()   # courbes + filtres + IA (avant l'aperçu live)
        # la colonne de réglages s'élargit à la largeur minimale de son contenu
        # (filtres dynamiques) : pas de barre de défilement horizontale
        from .commands import FitScrollWidth
        self._fit_params = FitScrollWidth(self._o("scrollParams"))
        self.live = LivePreview(self.pane, self._preview_params, self._compute, self)
        self._build_profiles()
        self._wire()
        self._hist()

    def _o(self, name):
        return self.ui.findChild(object, name)

    def _wire(self):
        for name in ("sldBright", "sldContrast", "sldSat", "spnGamma", "spnWR",
                     "spnWG", "spnWB", "sldUSMAmount", "spnUSMRadius"):
            o = self._o(name)
            sig = "valueChanged"
            getattr(o, sig).connect(self.live.trigger)
        self._o("btnApplyGeo").clicked.connect(self._apply_geo)
        self._o("btnApplySuperRes").clicked.connect(self._apply_superres)
        self._o("btnSave").clicked.connect(self._save)
        self._o("btnSeqAdd").clicked.connect(self._seq_add)
        self._o("btnSeqDel").clicked.connect(self._seq_del)
        self._o("btnSeqSave").clicked.connect(self._seq_save)
        self._o("btnSeqLoad").clicked.connect(self._seq_load)
        self._o("btnSeqBatch").clicked.connect(self._seq_batch)
        self._wire_retouch()

    def _build_profiles(self):
        self.pb = ProfileBar(
            "common_tone", dict(DEFAULTS),
            lambda: {"bright": self._o("sldBright").value(),
                     "contrast": self._o("sldContrast").value(),
                     "sat": self._o("sldSat").value(),
                     "gamma": self._o("spnGamma").value(),
                     "wr": self._o("spnWR").value(), "wg": self._o("spnWG").value(),
                     "wb": self._o("spnWB").value(),
                     "usm_amount": self._o("sldUSMAmount").value() / 100.0,
                     "usm_radius": self._o("spnUSMRadius").value()},
            lambda p: (self._o("sldBright").setValue(int(p.get("bright", 0))),
                       self._o("sldContrast").setValue(int(p.get("contrast", 0))),
                       self._o("sldSat").setValue(int(p.get("sat", 100))),
                       self._o("spnGamma").setValue(float(p.get("gamma", 1.0))),
                       self._o("spnWR").setValue(float(p.get("wr", 1.0))),
                       self._o("spnWG").setValue(float(p.get("wg", 1.0))),
                       self._o("spnWB").setValue(float(p.get("wb", 1.0))),
                       self._o("sldUSMAmount").setValue(int(p.get("usm_amount", 0.6) * 100)),
                       self._o("spnUSMRadius").setValue(float(p.get("usm_radius", 2.0)))))
        self.ui.findChild(QVBoxLayout, "profTone").addWidget(self.pb)
        self.pb_sr = ProfileBar(
            "common_superres", {"method": "Lanczos ×2"},
            lambda: {"method": self._o("cmbSuperRes").currentText()},
            lambda p: self._o("cmbSuperRes").setCurrentText(p.get("method", "Lanczos ×2")))
        self.ui.findChild(QVBoxLayout, "profSuperRes").addWidget(self.pb_sr)
        lay_fin = self.ui.findChild(QVBoxLayout, "profRetouch")
        if lay_fin is not None:
            self.pb_fin = ProfileBar("common_retouch", self._fin_profile_defaults(),
                                     self._fin_profile_get, self._fin_profile_set)
            lay_fin.addWidget(self.pb_fin)

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Charger une image", "",
            "Images (*.fits *.fit *.tif *.tiff *.png *.jpg)")
        if not path:
            return
        self.set_image(load_image(path))

    def set_image(self, img):
        """Reçoit l'image depuis les autres modes ou un fichier."""
        self.image = np.clip(np.asarray(img, np.float32), 0, None)
        if self.image.max() > 1.5:
            self.image = self.image / np.percentile(self.image, 99.5)
        import cv2
        h, w = self.image.shape[:2]
        f = min(1.0, 700.0 / max(h, w))
        small = cv2.resize(self.image, (int(w * f), int(h * f)),
                           interpolation=cv2.INTER_AREA) if f < 1 else self.image
        self.preview = np.clip(small, 0, 1).astype(np.float32)
        self.pane.set_before(self.preview)
        self.live.set_source(self.preview)
        if getattr(self, "curve", None) is not None:
            self.curve.set_image_histogram(self.preview)
        self._hist()

    def _collect(self):
        return {"bright": self._o("sldBright").value() / 100.0,
                "contrast": self._o("sldContrast").value() / 100.0,
                "sat": self._o("sldSat").value() / 100.0,
                "gamma": self._o("spnGamma").value(),
                "wr": self._o("spnWR").value(), "wg": self._o("spnWG").value(),
                "wb": self._o("spnWB").value(),
                "usm_amount": self._o("sldUSMAmount").value() / 100.0,
                "usm_radius": self._o("spnUSMRadius").value()}

    def _compute(self, img, p, progress=None):
        out = np.asarray(img, np.float32).copy()
        # balance des blancs
        if out.ndim == 3:
            out[..., 0] *= p["wr"]
            out[..., 1] *= p["wg"]
            out[..., 2] *= p["wb"]
        # luminosité / contraste
        out = (out - 0.5) * (1 + p["contrast"]) + 0.5 + p["bright"]
        # saturation
        if out.ndim == 3 and abs(p["sat"] - 1.0) > 1e-3:
            gray = out.mean(axis=2, keepdims=True)
            out = gray + (out - gray) * p["sat"]
        # gamma (courbes point milieu)
        out = np.clip(out, 0, 1)
        out = np.power(out, 1.0 / max(p["gamma"], 0.01))
        # retouches astro : courbe (esprit Photoshop) puis filtre courant
        out = self._apply_toolkit(out, p)
        # netteté
        if p["usm_amount"] > 0:
            out = unsharp(out, p["usm_amount"], p["usm_radius"])
        mk = p.get("mask") or {}
        if mk.get("show") and mk.get("kind", "Aucun") != "Aucun":
            from ..core.masks import make_mask
            return make_mask(np.asarray(img, np.float32), mk["kind"], mk["low"], mk["high"],
                             mk["fuzz"], mk["blur"]).astype(np.float32)
        return np.clip(out, 0, 1).astype(np.float32)

    @staticmethod
    def _apply_toolkit(img, p):
        """Courbes tonales (commune + par canal) puis filtre courant —
        toujours recalculés depuis l'image fournie (image de base), jamais
        de cumul sur un état ancien."""
        out = np.asarray(img, np.float32)
        curves = {"commune": p.get("curve"), "r": p.get("curve_r"),
                  "g": p.get("curve_g"), "b": p.get("curve_b")}
        out = FIN.apply_curves(out, curves)
        fid, params = p.get("fin", ("none", {}))
        return FIN.apply_filter_masked(out, fid, params, p.get("mask"))

    def _preview_params(self):
        """Paramètres d'aperçu : réglages GIMP + courbes (par canal) +
        filtre courant."""
        p = self._collect()
        self._sync_curve_channels()
        p["curve"] = self.curve_channels["commune"]
        for k in ("r", "g", "b"):
            p[f"curve_{k}"] = self.curve_channels[k]
        p["fin"] = self._fin_state()
        p["mask"] = self._mask_state()
        return p

    def _apply_geo(self):
        if self.image is None:
            return
        x, y = self._o("spnCX").value(), self._o("spnCY").value()
        cw, ch = self._o("spnCW").value(), self._o("spnCH").value()
        rot = self._o("cmbRotate").currentText()
        import cv2
        img = self.image
        if cw > 0 and ch > 0:
            img = img[y:y + ch, x:x + cw]
        if rot.startswith("90"):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rot.startswith("180"):
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif rot.startswith("270"):
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rot.startswith("Miroir h"):
            img = cv2.flip(img, 1)
        elif rot.startswith("Miroir v"):
            img = cv2.flip(img, 0)
        self.set_image(img)

    def _apply_superres(self):
        if self.image is None:
            QMessageBox.information(self, "Résolution", "Chargez d'abord une image.")
            return
        method = self._o("cmbSuperRes").currentText()
        if method.startswith("IA"):
            def run(progress=None):
                from ..ai.manager import get_ai_manager
                out = get_ai_manager().run("superres", self.image, "onnx", 0.5,
                                           progress=progress)
                return out
            self._job = Job(run)
            self._job.ok.connect(lambda img: (self.set_image(img),
                                              self._log_sr("Super-résolution IA appliquée.")))
            self._job.err.connect(lambda m: QMessageBox.critical(self, "IA", m[-1500:]))
            self._job.start()
            return
        import cv2
        scale = 3 if "×3" in method else 2
        out = cv2.resize(self.image, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_LANCZOS4)
        self.set_image(out)

    def _log_sr(self, msg):
        lbl: QLabel = self._o("lblInfo")
        lbl.setText(msg)

    # ---------------------------------------------------- retouches astro
    @staticmethod
    def _dspin(vmin, vmax, step, value, dec):
        w = QDoubleSpinBox()
        w.setDecimals(dec)
        w.setMinimumWidth(56)
        w.setRange(vmin, vmax)
        w.setSingleStep(step)
        w.setValue(value)
        return w

    def _make_filter_page(self, fid, specs):
        """Page de paramètres d'un filtre (libellés français repliés)."""
        from PySide6.QtWidgets import QFormLayout
        page = QWidget()
        form_f = QFormLayout(page)
        for key, plabel, vmin, vmax, step, default, dec in specs:
            if key == "method":
                w = QComboBox()
                w.addItems(FIN.CHOICES.get(fid, ["Soustraire", "Diviser"]))
                w.setObjectName(f"cmbFin_{fid}_{key}")
            elif dec == 0:
                w = QSpinBox()
                w.setRange(int(vmin), int(vmax))
                w.setValue(int(default))
                w.setObjectName(f"spnFin_{fid}_{key}")
            else:
                w = self._dspin(vmin, vmax, step, default, dec)
                w.setObjectName(f"spnFin_{fid}_{key}")
            w.setMinimumWidth(56)
            self._fin_widgets[(fid, key)] = w
            w.setToolTip(plabel)
            form_f.addRow(plabel, w)
        for lbl in page.findChildren(QLabel):
            lbl.setWordWrap(True)   # largeur minimale bornée (1024 px)
        return page

    def _build_retouch(self):
        """Groupe « Retouches astro » : courbes (esprit Photoshop) +
        automatiques, filtres de réglages (implémentations maison inspirées
        de PixInsight/Siril/AstroPixelProcessor) et IA locale (mêmes outils
        que le planétaire). Le .ui ne fournit que les conteneurs."""
        from PySide6.QtWidgets import QFormLayout, QStackedWidget, QTabWidget
        # Courbes par canal : un seul éditeur, un sélecteur de canal ;
        # « commune » = comportement historique (tous les canaux).
        self._curve_keys = ("commune", "r", "g", "b")
        self.curve_channels = {k: [list(p) for p in PL.IDENTITY_CURVE_POINTS]
                               for k in self._curve_keys}
        self._curve_prev_key = "commune"
        self.curve = CurveEditor()
        self.tabsFin = QTabWidget()
        self.tabsFin.setObjectName("tabsFin")
        self.tabsFin.setStyleSheet("QTabBar::tab { padding-left: 0px; padding-right: 0px; }")

        tab_curve = QWidget()
        v = QVBoxLayout(tab_curve)
        self.cmbCurveChannel = QComboBox()
        self.cmbCurveChannel.setObjectName("cmbCurveChannel")
        self.cmbCurveChannel.addItems(["Commune", "Rouge", "Vert", "Bleu"])
        self.cmbCurveChannel.setMinimumWidth(60)
        row_ch = QHBoxLayout()
        row_ch.addWidget(QLabel("Canal de la courbe :"))
        row_ch.addWidget(self.cmbCurveChannel)
        row_ch.addStretch(1)
        v.addLayout(row_ch)
        v.addWidget(self.curve)
        hint = QLabel("Clic : ajouter/déplacer un point · Suppr : retirer "
                      "(extrémités épinglées) · aperçu live ci-contre.")
        hint.setWordWrap(True)
        v.addWidget(hint)
        self.tabsFin.addTab(tab_curve, "Courbe")

        tab_ac = QWidget()
        form = QFormLayout(tab_ac)
        self.spnAutoClipF = self._dspin(0.0, 0.49, 0.01, 0.1, 2)
        self.spnAutoClipF.setObjectName("spnFinAutoClip")
        self.spnAutoBlackF = self._dspin(0.0, 1.0, 0.001, 0.0, 3)
        self.spnAutoBlackF.setObjectName("spnFinAutoBlack")
        self.spnAutoWhiteF = self._dspin(0.0, 1.0, 0.001, 1.0, 3)
        self.spnAutoWhiteF.setObjectName("spnFinAutoWhite")
        form.addRow("Rognage (part d'histogramme)", self.spnAutoClipF)
        row_bw = QHBoxLayout()
        row_bw.addWidget(QLabel("Noir"))
        row_bw.addWidget(self.spnAutoBlackF)
        row_bw.addWidget(QLabel("Blanc"))
        row_bw.addWidget(self.spnAutoWhiteF)
        form.addRow(row_bw)
        self.btnAutoComputeF = QPushButton("Actualiser (depuis l'aperçu)")
        self.btnAutoComputeF.setObjectName("btnFinAutoCompute")
        self.lblAutoInfoF = QLabel("—")
        self.lblAutoInfoF.setObjectName("lblFinAutoInfo")
        form.addRow(self.btnAutoComputeF, self.lblAutoInfoF)
        self.tabsFin.addTab(tab_ac, "Contraste automatique")

        tab_al = QWidget()
        form_g = QFormLayout(tab_al)
        self.spnAutoGammaF = self._dspin(0.1, 10.0, 0.05, 1.0, 2)
        self.spnAutoGammaF.setObjectName("spnFinAutoGamma")
        self.btnAutoGammaF = QPushButton("Actualiser (depuis l'aperçu)")
        self.btnAutoGammaF.setObjectName("btnFinAutoGamma")
        form_g.addRow("Gamma visé (moyenne vers 0,5)", self.spnAutoGammaF)
        form_g.addRow(self.btnAutoGammaF)
        self.tabsFin.addTab(tab_al, "Luminosité automatique")

        lay: QVBoxLayout = self.ui.findChild(QVBoxLayout, "layRetouch")
        lay.addWidget(self.tabsFin)

        # Filtres de réglages (chaque filtre recalculé depuis l'image de base)
        self.cmbFin = QComboBox()
        self.cmbFin.setObjectName("cmbFinFilter")
        self.stkFin = QStackedWidget()
        self.stkFin.setObjectName("stkFin")
        self._fin_widgets = {}
        self.cmbFin.addItem("Aucun filtre", "none")
        none_page = QWidget()
        none_v = QVBoxLayout(none_page)
        none_hint = QLabel(
            "Choisir un filtre : l'aperçu recalcule toujours depuis l'image "
            "de base (aucun cumul) ; « Appliquer sur l'image » fige le "
            "résultat et remet les réglages au neutre.")
        none_hint.setWordWrap(True)
        none_v.addWidget(none_hint)
        self.stkFin.addWidget(none_page)
        for fid, _label, _opname, specs in FIN.FILTERS_CLASSIC + FIN.FILTERS_ADVANCED:
            self.cmbFin.addItem(FIN.LABELS[fid], fid)
            self.stkFin.addWidget(self._make_filter_page(fid, specs))
        lay.addWidget(self.cmbFin)
        lay.addWidget(self.stkFin)
        self._build_mask_controls(lay)
        row_fin = QHBoxLayout()
        self.btnFinApply = QPushButton("Appliquer sur l'image")
        self.btnFinApply.setObjectName("btnFinApply")
        self.btnFinReset = QPushButton("Réinitialiser les réglages")
        self.btnFinReset.setObjectName("btnFinReset")
        row_fin.addWidget(self.btnFinApply)
        row_fin.addWidget(self.btnFinReset)
        lay.addLayout(row_fin)
        note = QLabel("Implémentations maison inspirées des principes "
                      "Photoshop / PixInsight / Siril / AstroPixelProcessor "
                      "(non portées des algorithmes commerciaux).")
        note.setWordWrap(True)
        lay.addWidget(note)
        prof = QVBoxLayout()
        prof.setObjectName("profRetouch")
        lay.addLayout(prof)

        # Banque « Retouches couleur & détail » (esprit Affinity Photo /
        # PixInsight) : mêmes mécanismes — un filtre actif à la fois
        # (banques exclusives), aperçu recalculé depuis l'image de base,
        # application sans cumul, profils et opérations de séquence.
        lay_c: QVBoxLayout = self.ui.findChild(QVBoxLayout, "layColor")
        self.cmbFin2 = QComboBox()
        self.cmbFin2.setObjectName("cmbColorFilter")
        self.cmbFin2.setMinimumWidth(60)
        self.stkFin2 = QStackedWidget()
        self.stkFin2.setObjectName("stkColor")
        self.cmbFin2.addItem("Aucun filtre", "none")
        none2 = QWidget()
        none2_v = QVBoxLayout(none2)
        hint2 = QLabel("Filtres couleur & détail : niveaux, tons clairs / "
                       "sombres, vibrance, HSL (par gammes), contraste "
                       "local, tonalité divisée, mappage de dégradé.")
        hint2.setWordWrap(True)
        none2_v.addWidget(hint2)
        self.stkFin2.addWidget(none2)
        for fid, _label, _opname, specs in FIN.FILTERS_COLOR:
            self.cmbFin2.addItem(FIN.LABELS[fid], fid)
            self.stkFin2.addWidget(self._make_filter_page(fid, specs))
        lay_c.addWidget(self.cmbFin2)
        lay_c.addWidget(self.stkFin2)

        # IA locale (mêmes outils que le planétaire, sur l'image de finition)
        from .ai_dialog import populate_ai_tasks
        self.cmbFinAITask = QComboBox()
        self.cmbFinAITask.setObjectName("cmbFinAITask")
        populate_ai_tasks(self.cmbFinAITask)
        self.cmbFinAIProv = QComboBox()
        self.cmbFinAIProv.setObjectName("cmbFinAIProvider")
        self.cmbFinAITask.setMinimumWidth(60)
        self.cmbFinAIProv.setMinimumWidth(60)
        self.spnFinAI = self._dspin(0.0, 1.0, 0.05, 0.5, 2)
        self.spnFinAI.setObjectName("spnFinAIStrength")
        ial: QVBoxLayout = self.ui.findChild(QVBoxLayout, "layIAF")
        lbl_ia = QLabel("Tâche IA appliquée à l'image de finition :")
        lbl_ia.setWordWrap(True)
        ial.addWidget(lbl_ia)
        row_ia = QHBoxLayout()
        row_ia.addWidget(self.cmbFinAITask, 1)
        row_ia.addWidget(self.cmbFinAIProv, 1)
        row_ia.addWidget(self.spnFinAI)
        ial.addLayout(row_ia)
        row_ia2 = QHBoxLayout()
        self.btnFinAIRun = QPushButton("Lancer l'IA sur l'image")
        self.btnFinAIRun.setObjectName("btnFinAIRun")
        self.btnFinAIConfig = QPushButton("Configurer les outils IA…")
        self.btnFinAIConfig.setObjectName("btnFinAIConfig")
        row_ia2.addWidget(self.btnFinAIRun)
        row_ia2.addWidget(self.btnFinAIConfig)
        ial.addLayout(row_ia2)

    # ------------------------------------------------ masques & étoiles
    def _build_mask_controls(self, lay):
        """Masque d'application du filtre courant + séparation/recombinaison
        des étoiles."""
        from PySide6.QtWidgets import QCheckBox, QFormLayout, QGroupBox
        from ..core.masks import MASK_TYPES
        grp = QGroupBox("Masque d'application du filtre")
        grp.setObjectName("grpFinMask")
        form = QFormLayout(grp)
        self.cmbMask = QComboBox()
        self.cmbMask.setObjectName("cmbFinMask")
        self.cmbMask.addItems(list(MASK_TYPES))
        self.spnMaskLow = self._dspin(0.0, 1.0, 0.01, 0.25, 2)
        self.spnMaskHigh = self._dspin(0.0, 1.0, 0.01, 1.0, 2)
        self.spnMaskFuzz = self._dspin(0.0, 0.5, 0.01, 0.10, 2)
        self.spnMaskBlur = self._dspin(0.0, 50.0, 0.5, 2.0, 1)
        self.chkMaskShow = QCheckBox("Afficher le masque dans l'aperçu")
        self.chkMaskShow.setObjectName("chkFinMaskShow")
        for w in (self.cmbMask, self.spnMaskLow, self.spnMaskHigh, self.spnMaskFuzz,
                  self.spnMaskBlur):
            w.setMinimumWidth(56)
        form.addRow("Type", self.cmbMask)
        form.addRow("Bas (luminance)", self.spnMaskLow)
        form.addRow("Haut (luminance)", self.spnMaskHigh)
        form.addRow("Transition", self.spnMaskFuzz)
        form.addRow("Flou (px)", self.spnMaskBlur)
        form.addRow(self.chkMaskShow)
        for lbl in grp.findChildren(QLabel):
            lbl.setWordWrap(True)
        lay.addWidget(grp)
        grp2 = QGroupBox("Étoiles : séparation et recombinaison")
        grp2.setObjectName("grpFinStars")
        v = QVBoxLayout(grp2)
        info = QLabel("Séparer : charge l'image SANS étoiles (StarNet, GraXpert…) ; "
                      "les étoiles = image − sans étoiles sont mises de côté. "
                      "Traitez le fond, puis recombinez.")
        info.setWordWrap(True)
        v.addWidget(info)
        row = QHBoxLayout()
        self.btnStarsSplit = QPushButton("Séparer les étoiles…")
        self.btnStarsSplit.setObjectName("btnStarsSplit")
        self.btnStarsMerge = QPushButton("Recombiner")
        self.btnStarsMerge.setObjectName("btnStarsMerge")
        self.btnStarsMerge.setEnabled(False)
        row.addWidget(self.btnStarsSplit)
        row.addWidget(self.btnStarsMerge)
        v.addLayout(row)
        row2 = QHBoxLayout()
        self.cmbStarsMode = QComboBox()
        self.cmbStarsMode.addItems(["Écran", "Addition"])
        self.spnStarsGain = self._dspin(0.0, 2.0, 0.05, 1.0, 2)
        row2.addWidget(QLabel("Mode"))
        row2.addWidget(self.cmbStarsMode)
        row2.addWidget(QLabel("Gain"))
        row2.addWidget(self.spnStarsGain)
        v.addLayout(row2)
        lay.addWidget(grp2)
        self.stars_layer = None

    def _mask_state(self):
        if not hasattr(self, "cmbMask"):
            return None
        return {"kind": self.cmbMask.currentText(), "low": self.spnMaskLow.value(),
                "high": self.spnMaskHigh.value(), "fuzz": self.spnMaskFuzz.value(),
                "blur": self.spnMaskBlur.value(), "show": self.chkMaskShow.isChecked()}

    def split_stars(self, starless):
        """Met les étoiles de côté (image − sans étoiles) ; l'image de travail
        devient l'image sans étoiles."""
        from ..core.masks import extract_stars
        if self.image is None:
            raise ValueError("Chargez d'abord une image.")
        sl = np.asarray(starless, np.float32)
        if sl.shape != self.image.shape:
            raise ValueError(f"Taille de l'image sans étoiles {sl.shape} ≠ {self.image.shape}")
        self.stars_layer = extract_stars(self.image, sl)
        self.set_image(sl)
        self.btnStarsMerge.setEnabled(True)
        self._log_sr("Étoiles mises de côté ; l'image de travail est sans étoiles.")

    def _stars_split(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Image SANS étoiles (même cadrage)", "",
            "Images (*.fits *.fit *.xisf *.tif *.tiff *.png)")
        if not path:
            return
        from ..core.image_io import load_image
        try:
            self.split_stars(load_image(path))
        except Exception as e:
            QMessageBox.critical(self, "Étoiles", str(e))

    def merge_stars(self):
        from ..core.masks import recombine
        if self.stars_layer is None or self.image is None:
            return
        self.set_image(recombine(self.image, self.stars_layer, self.cmbStarsMode.currentText(),
                                 self.spnStarsGain.value()))
        self.stars_layer = None
        self.btnStarsMerge.setEnabled(False)
        self._log_sr("Étoiles recombinées.")

    def _wire_retouch(self):
        self.curve.pointsChanged.connect(self.live.trigger)
        self.tabsFin.currentChanged.connect(self.live.trigger)
        self.btnAutoComputeF.clicked.connect(self._fin_auto_contrast)
        self.btnAutoGammaF.clicked.connect(self._fin_auto_gamma)
        self.spnAutoBlackF.valueChanged.connect(self._fin_apply_bw)
        self.spnAutoWhiteF.valueChanged.connect(self._fin_apply_bw)
        self.spnAutoGammaF.valueChanged.connect(self._fin_apply_gamma)
        self.cmbFin.currentIndexChanged.connect(self._fin_filter_changed)
        self.cmbFin2.currentIndexChanged.connect(self._fin2_filter_changed)
        self.cmbCurveChannel.currentIndexChanged.connect(self._on_curve_channel)
        self.btnFinApply.clicked.connect(self._apply_retouch)
        self.btnFinReset.clicked.connect(self._fin_reset)
        self.btnStarsSplit.clicked.connect(self._stars_split)
        self.btnStarsMerge.clicked.connect(self.merge_stars)
        for w in (self.spnMaskLow, self.spnMaskHigh, self.spnMaskFuzz, self.spnMaskBlur):
            w.valueChanged.connect(self.live.trigger)
        self.cmbMask.currentIndexChanged.connect(self.live.trigger)
        self.chkMaskShow.toggled.connect(self.live.trigger)
        self.btnFinAIRun.clicked.connect(self._fin_ai_run)
        self.btnFinAIConfig.clicked.connect(self._fin_ai_config)
        self.cmbFinAITask.currentIndexChanged.connect(
            self._refresh_ai_providers_fin)
        for (_fid, _key), w in self._fin_widgets.items():
            sig = ("currentTextChanged" if isinstance(w, QComboBox)
                   else "valueChanged")
            getattr(w, sig).connect(self.live.trigger)
        from ..ai.manager import register_provider_refresh
        register_provider_refresh(self._refresh_ai_providers_fin)
        self._refresh_ai_providers_fin()

    # ------------------------------------------------ état filtre / courbe
    def _fin_filter_changed(self, index):
        self.stkFin.setCurrentIndex(index)
        if index > 0 and self.cmbFin2.currentIndex() != 0:
            self.cmbFin2.blockSignals(True)
            self.cmbFin2.setCurrentIndex(0)
            self.cmbFin2.blockSignals(False)
            self.stkFin2.setCurrentIndex(0)
        self.live.trigger()

    def _fin2_filter_changed(self, index):
        self.stkFin2.setCurrentIndex(index)
        if index > 0 and self.cmbFin.currentIndex() != 0:
            self.cmbFin.blockSignals(True)
            self.cmbFin.setCurrentIndex(0)
            self.cmbFin.blockSignals(False)
            self.stkFin.setCurrentIndex(0)
        self.live.trigger()

    def _on_curve_channel(self, index):
        """Changement de canal édité : mémorise les points courants sous le
        canal précédent puis charge ceux du canal choisi."""
        self.curve_channels[self._curve_prev_key] = self.curve.points()
        key = self._curve_keys[index]
        self.curve.set_points(self.curve_channels[key], silent=True)
        self._curve_prev_key = key
        self.live.trigger()

    def _sync_curve_channels(self):
        self.curve_channels[self._curve_prev_key] = self.curve.points()

    def _fin_state(self):
        """(identifiant de filtre, paramètres) — (« none », {}) si aucun."""
        fid = self.cmbFin.currentData() or "none"
        if fid == "none":
            fid = self.cmbFin2.currentData() or "none"
        if fid == "none":
            return "none", {}
        params = {}
        for f_id, _label, _opname, specs in FIN.FILTERS:
            if f_id != fid:
                continue
            for key, *_rest in specs:
                w = self._fin_widgets[(fid, key)]
                params[key] = w.currentText() if key == "method" else w.value()
        return fid, params

    def _fin_set_params(self, fid, params):
        for f_id, _label, _opname, specs in FIN.FILTERS:
            if f_id != fid:
                continue
            for key, *_rest in specs:
                w = self._fin_widgets[(fid, key)]
                value = params.get(key, FIN.neutral_params(fid)[key])
                w.blockSignals(True)
                if key == "method":
                    w.setCurrentText(str(value))
                elif isinstance(w, QSpinBox):
                    w.setValue(int(value))
                else:
                    w.setValue(float(value))
                w.blockSignals(False)
        self.live.trigger()

    def _apply_retouch(self):
        if self.image is None:
            QMessageBox.information(self, "Retouches", "Chargez d'abord une image.")
            return
        out = self._apply_toolkit(self.image, self._preview_params())
        self.set_image(np.clip(out, 0, 1).astype(np.float32))
        # Non cumul : courbes au neutre (tous les canaux) et filtre remis
        # aux défauts.
        for k in self.curve_channels:
            self.curve_channels[k] = [list(p) for p in PL.IDENTITY_CURVE_POINTS]
        self.curve.set_points(PL.IDENTITY_CURVE_POINTS)
        fid, _params = self._fin_state()
        if fid != "none":
            self._fin_set_params(fid, FIN.neutral_params(fid))
        self._log_sr("Retouche appliquée sur l'image ; réglages remis au neutre.")

    def _fin_reset(self):
        for k in self.curve_channels:
            self.curve_channels[k] = [list(p) for p in PL.IDENTITY_CURVE_POINTS]
        self.curve.set_points(PL.IDENTITY_CURVE_POINTS)
        for fid, _label, _opname, _specs in FIN.FILTERS:
            self._fin_set_params(fid, FIN.neutral_params(fid))
        for cmb, stk in ((self.cmbFin, self.stkFin), (self.cmbFin2, self.stkFin2)):
            cmb.blockSignals(True)
            cmb.setCurrentIndex(0)
            cmb.blockSignals(False)
            stk.setCurrentIndex(0)
        self.cmbCurveChannel.blockSignals(True)
        self.cmbCurveChannel.setCurrentIndex(0)
        self.cmbCurveChannel.blockSignals(False)
        self._curve_prev_key = "commune"
        self.live.trigger()

    # --------------------------------------------- automatiques (finition)
    def _fin_auto_base(self):
        if self.preview is None:
            raise ValueError("Chargez d'abord une image.")
        return self.preview   # aperçu réduit : calcul unique, comme au planétaire

    def _fin_auto_contrast(self):
        try:
            base = self._fin_auto_base()
        except ValueError as e:
            QMessageBox.information(self, "Contraste automatique", str(e))
            return
        _params, black, white = PL.auto_contrast_params(
            base, self.spnAutoClipF.value())
        for w, value in ((self.spnAutoBlackF, black),
                         (self.spnAutoWhiteF, white)):
            w.blockSignals(True)
            w.setValue(value)
            w.blockSignals(False)
        self._fin_apply_bw()
        self.lblAutoInfoF.setText(f"Noir {black:.3f} — Blanc {white:.3f}")

    def _fin_apply_bw(self, *_):
        black, white = self.spnAutoBlackF.value(), self.spnAutoWhiteF.value()
        if white - black < 1e-3:
            return
        self.curve.set_points([[black, 0.0], [white, 1.0]])

    def _fin_auto_gamma(self):
        try:
            base = self._fin_auto_base()
        except ValueError as e:
            QMessageBox.information(self, "Luminosité automatique", str(e))
            return
        _params, gamma = PL.auto_brightness_params(base)
        self.spnAutoGammaF.blockSignals(True)
        self.spnAutoGammaF.setValue(gamma)
        self.spnAutoGammaF.blockSignals(False)
        self._fin_apply_gamma()

    def _fin_apply_gamma(self, *_):
        self.curve.set_points(PL.gamma_curve(self.spnAutoGammaF.value()))

    # ------------------------------------------------------------- IA locale
    def _refresh_ai_providers_fin(self):
        from ..ai.manager import get_ai_manager
        task = self.cmbFinAITask.currentData()
        selected = self.cmbFinAIProv.currentData()
        self.cmbFinAIProv.clear()
        for prov in get_ai_manager().providers(task):
            self.cmbFinAIProv.addItem(prov["label"], prov["id"])
        index = self.cmbFinAIProv.findData(selected)
        if index >= 0:
            self.cmbFinAIProv.setCurrentIndex(index)

    def _fin_ai_config(self):
        from .ai_dialog import AIToolsDialog
        AIToolsDialog(self).exec()
        self._refresh_ai_providers_fin()

    def _fin_ai_run(self):
        if self.image is None:
            QMessageBox.information(self, "IA", "Chargez d'abord une image.")
            return
        task = self.cmbFinAITask.currentData()
        provider = self.cmbFinAIProv.currentData() or "onnx"
        strength = self.spnFinAI.value()
        base = self.image

        def run(progress=None):
            from ..ai.manager import get_ai_manager
            return get_ai_manager().run(task, base, provider, strength,
                                        progress=progress)
        self._job = Job(run)
        self._job.ok.connect(
            lambda img: (self.set_image(img),
                         self._log_sr(f"IA « {task} » appliquée sur l'image.")))
        self._job.err.connect(
            lambda m: QMessageBox.critical(self, "IA", m[-1500:]))
        self._job.start()

    # ------------------------------------------------ profils retouches
    def _fin_profile_defaults(self):
        d = {"curve": [[0.0, 0.0], [1.0, 1.0]], "curve_r": None,
             "curve_g": None, "curve_b": None, "filter": "none"}
        for fid, _label, _opname, _specs in FIN.FILTERS:
            for key, value in FIN.neutral_params(fid).items():
                d[f"{fid}.{key}"] = value
        return d

    def _fin_profile_get(self):
        self._sync_curve_channels()
        fid = self.cmbFin.currentData() or "none"
        if fid == "none":
            fid = self.cmbFin2.currentData() or "none"
        p = {"curve": self.curve_channels["commune"],
             "curve_r": self.curve_channels["r"],
             "curve_g": self.curve_channels["g"],
             "curve_b": self.curve_channels["b"],
             "filter": fid}
        for (fid, key), w in self._fin_widgets.items():
            p[f"{fid}.{key}"] = w.currentText() if key == "method" else w.value()
        return p

    def _fin_profile_set(self, p):
        for k, key in (("commune", "curve"), ("r", "curve_r"),
                       ("g", "curve_g"), ("b", "curve_b")):
            pts = p.get(key) or [[0.0, 0.0], [1.0, 1.0]]
            self.curve_channels[k] = [list(pt) for pt in pts]
        self.cmbCurveChannel.blockSignals(True)
        self.cmbCurveChannel.setCurrentIndex(0)
        self.cmbCurveChannel.blockSignals(False)
        self._curve_prev_key = "commune"
        self.curve.set_points(self.curve_channels["commune"])
        fid = p.get("filter", "none")
        i2 = self.cmbFin2.findData(fid) if fid != "none" else -1
        if i2 >= 0:
            self.cmbFin.blockSignals(True)
            self.cmbFin.setCurrentIndex(0)
            self.cmbFin.blockSignals(False)
            self.stkFin.setCurrentIndex(0)
            self.cmbFin2.setCurrentIndex(i2)
        else:
            self.cmbFin2.blockSignals(True)
            self.cmbFin2.setCurrentIndex(0)
            self.cmbFin2.blockSignals(False)
            self.stkFin2.setCurrentIndex(0)
            self.cmbFin.setCurrentIndex(max(0, self.cmbFin.findData(fid)))
        for (f_id, key), w in self._fin_widgets.items():
            value = p.get(f"{f_id}.{key}", FIN.neutral_params(f_id)[key])
            if key == "method":
                w.setCurrentText(str(value))
            elif isinstance(w, QSpinBox):
                w.setValue(int(value))
            else:
                w.setValue(float(value))
        self.live.trigger()

    # ------------------------------------------------------------- séquences
    def _seq_add(self):
        p = self._collect()
        self.seq_ops.append({"op": "gimp_tone", "params": p,
                             "label": f"Étape {len(self.seq_ops) + 1} : tonalité & netteté"})
        fid, fparams = self._fin_state()
        if fid != "none":
            self.seq_ops.append({"op": FIN.OP_FOR[fid], "params": fparams,
                                 "label": f"Étape {len(self.seq_ops) + 1} : {FIN.LABELS[fid]}"})
        self._seq_refresh()

    def _seq_del(self):
        lw = self._o("lstSeqOps")
        row = lw.currentRow()
        if row >= 0:
            self.seq_ops.pop(row)
            self._seq_refresh()

    def _seq_refresh(self):
        lw = self._o("lstSeqOps")
        lw.clear()
        for o in self.seq_ops:
            lw.addItem(o.get("label", o["op"]))

    def _seq_save(self):
        from ..core import sequence as Q
        path, _ = QFileDialog.getSaveFileName(self, "Enregistrer la séquence",
                                              "sequence.json", "Séquence (*.json)")
        if path:
            Q.save_sequence(path, self.seq_ops, mode="image")
            self._log_sr(f"Séquence enregistrée : {path}")

    def _seq_load(self):
        from ..core import sequence as Q
        path, _ = QFileDialog.getOpenFileName(self, "Charger une séquence", "",
                                              "Séquence (*.json)")
        if path:
            self.seq_ops = Q.load_sequence(path).get("ops", [])
            self._seq_refresh()
            self._log_sr(f"Séquence chargée : {len(self.seq_ops)} étapes")

    def _seq_batch(self):
        if not self.seq_ops:
            QMessageBox.information(self, "Séquence",
                                    "Ajoutez d'abord des étapes (« ＋ Ajouter l'étape courante »).")
            return
        in_dir = QFileDialog.getExistingDirectory(self, "Dossier d'images à traiter")
        if not in_dir:
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Dossier de sortie")
        if not out_dir:
            return
        from ..core import sequence as Q
        ops = list(self.seq_ops)

        def run(progress=None):
            return Q.run_batch_image({"mode": "image", "ops": ops}, in_dir, out_dir,
                                     progress=progress)
        self._job = Job(run)
        def done(files):
            summary = files.summary() if hasattr(files, "summary") else f"{len(files)} images"
            self._log_sr(f"✔ Batch terminé : {summary}")
            if getattr(files, "errors", None):
                QMessageBox.warning(self, "Batch", f"{summary}\n\nDétail : "
                                    f"{os.path.join(out_dir, 'rapport_batch.txt')}")
        self._job.ok.connect(done)
        self._job.err.connect(lambda m: QMessageBox.critical(self, "Batch", m[-1500:]))
        self._job.start()

    def _hist(self):
        lbl: QLabel = self._o("histLabel")
        if self.preview is not None:
            lbl.setPixmap(draw_histogram_pixmap(self.preview))
        else:
            lbl.setText("—")

    def _save(self):
        if self.image is None:
            QMessageBox.information(self, "Export", "Aucune image.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer", "finition.fits",
            "FITS (*.fits);;TIFF 16 bits (*.tif);;PNG 8 bits (*.png);;"
            "PNG 16 bits (*.png);;JPEG (*.jpg)")
        if not path:
            return
        fmt = self._o("cmbFmt").currentText()
        depth = "float32" if fmt.startswith("FITS") else \
            ("16" if "16" in fmt else "auto")
        save_image(path, self.image, depth)
