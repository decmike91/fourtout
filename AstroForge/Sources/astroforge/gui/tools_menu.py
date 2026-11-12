# -*- coding: utf-8 -*-
"""Menu « Outils » : composition couleur, mosaïque, comète, empilement en
direct, planétaire (dérotation/WinJUPOS), solaire, astrométrie (annotation,
courbe de lumière), comparaison IA, scripts, bibliothèque de masters.

Chaque outil est une boîte de dialogue autonome ; les calculs longs passent
par un Job (thread) avec barre de progression et annulation, et le résultat
peut être envoyé vers l'onglet Finition ou enregistré."""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

IMG_FILTER = "Images (*.fits *.fit *.fts *.fz *.xisf *.tif *.tiff *.png *.jpg)"


# ------------------------------------------------------------ utilitaires
class PathRow(QWidget):
    """Champ chemin + bouton « … » (fichier, fichiers ou dossier)."""

    def __init__(self, mode="file", filt=IMG_FILTER, parent=None):
        super().__init__(parent)
        self.mode, self.filt = mode, filt
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        btn = QPushButton("…")
        btn.setFixedWidth(32)
        btn.clicked.connect(self._pick)
        lay.addWidget(self.edit, 1)
        lay.addWidget(btn)

    def _pick(self):
        if self.mode == "dir":
            p = QFileDialog.getExistingDirectory(self, "Dossier")
        elif self.mode == "files":
            ps, _ = QFileDialog.getOpenFileNames(self, "Fichiers", "", self.filt)
            p = ";".join(ps)
        elif self.mode == "save":
            p, _ = QFileDialog.getSaveFileName(self, "Enregistrer", "", self.filt)
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Fichier", "", self.filt)
        if p:
            self.edit.setText(p)

    def text(self) -> str:
        return self.edit.text().strip()

    def paths(self) -> list[str]:
        t = self.text()
        if not t:
            return []
        if self.mode == "dir" and os.path.isdir(t):
            from ..core.calibration import list_frames
            return list_frames(t)
        return [p for p in t.split(";") if p]


class ToolDialog(QDialog):
    """Base : formulaire + journal + progression + boutons Lancer/Annuler,
    résultat image envoyable vers la Finition ou enregistrable."""

    title = "Outil"

    def __init__(self, main=None, parent=None):
        super().__init__(parent)
        self.main = main
        self.result_image = None
        self._job = None
        self.setWindowTitle(self.title)
        self.resize(640, 520)
        lay = QVBoxLayout(self)
        self.form = QFormLayout()
        lay.addLayout(self.form)
        # boutons créés AVANT build() (les sous-classes peuvent les adapter)
        self.btnRun = QPushButton("Lancer")
        self.btnCancel = QPushButton("Annuler")
        self.btnCancel.setEnabled(False)
        self.btnSend = QPushButton("Envoyer vers la Finition")
        self.btnSave = QPushButton("Enregistrer…")
        self.btnSend.setEnabled(False)
        self.btnSave.setEnabled(False)
        self.progress = QProgressBar()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.build()
        lay.addWidget(self.progress)
        lay.addWidget(self.log, 1)
        row = QHBoxLayout()
        for b in (self.btnRun, self.btnCancel, self.btnSend, self.btnSave):
            row.addWidget(b)
        lay.addLayout(row)
        self.btnRun.clicked.connect(self.start)
        self.btnCancel.clicked.connect(self.cancel)
        self.btnSend.clicked.connect(self.send_to_finishing)
        self.btnSave.clicked.connect(self.save_result)

    # à surcharger
    def build(self):
        pass

    def make_job(self):
        """Retourne une fonction run(progress) exécutée dans un thread."""
        raise NotImplementedError

    def on_result(self, res):
        if isinstance(res, np.ndarray):
            self.set_result(res)

    # mécanique commune
    def add_log(self, msg: str):
        self.log.appendPlainText(str(msg))

    def set_result(self, img):
        self.result_image = np.asarray(img, np.float32)
        self.btnSend.setEnabled(True)
        self.btnSave.setEnabled(True)
        self.add_log(f"Résultat : {self.result_image.shape}")

    def start(self):
        from .workers import Job
        try:
            fn = self.make_job()
        except Exception as e:
            QMessageBox.warning(self, self.title, str(e))
            return
        if fn is None:
            return
        self._job = Job(fn)
        self._job.progress.connect(self._on_progress)
        self._job.ok.connect(self._on_ok)
        self._job.err.connect(self._on_err)
        self._job.finished.connect(lambda: (self.btnRun.setEnabled(True),
                                            self.btnCancel.setEnabled(False)))
        self.btnRun.setEnabled(False)
        self.btnCancel.setEnabled(True)
        self._job.start()

    def run_sync(self):
        """Exécution synchrone (tests, scripts)."""
        res = self.make_job()(progress=lambda *a: None)
        self.on_result(res)
        return res

    def cancel(self):
        if self._job is not None:
            self._job.cancel()

    def _on_progress(self, d, t, label):
        self.progress.setMaximum(t)
        self.progress.setValue(d)
        if label:
            self.progress.setFormat(f"{label} — %p %")

    def _on_ok(self, res):
        try:
            self.on_result(res)
        except Exception as e:
            self._on_err(str(e))

    def _on_err(self, msg):
        self.add_log("✖ " + msg.strip().splitlines()[-1])
        QMessageBox.critical(self, self.title, msg[-1500:])

    def send_to_finishing(self):
        if self.result_image is None or self.main is None:
            return
        send_to_finishing(self.main, self.result_image)
        self.add_log("Image envoyée vers l'onglet Finition.")

    def save_result(self):
        if self.result_image is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer", "resultat.fits",
            "FITS (*.fits);;XISF (*.xisf);;TIFF 16 bits (*.tif);;PNG (*.png)")
        if path:
            from ..core.image_io import save_image
            save_image(path, self.result_image)
            self.add_log(f"Enregistré : {path}")


def send_to_finishing(main, img):
    """Ouvre l'image dans l'onglet Finition."""
    from PySide6.QtWidgets import QTabWidget
    main.shared["image"] = img
    main.page_common.set_image(np.asarray(img, np.float32))
    tabs = main.win.findChild(QTabWidget, "tabs")
    if tabs is not None:
        tabs.setCurrentIndex(2)


def dspin(lo, hi, step, val, dec=2):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(dec)
    w.setValue(val)
    return w


def ispin(lo, hi, val):
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setValue(val)
    return w


def load_one(path: str):
    from ..core.image_io import load_image
    if not path:
        return None
    return load_image(path)


# ------------------------------------------------------ composition couleur
class ComposeDialog(ToolDialog):
    title = "Composition couleur (bande étroite, LRGB)"
    CHANNELS = ("L", "R", "G", "B", "Ha", "OIII", "SII")

    def build(self):
        from ..core.narrowband import PALETTES
        self.rows = {}
        for ch in self.CHANNELS:
            r = PathRow("file")
            self.rows[ch] = r
            self.form.addRow(f"Canal {ch}", r)
        self.cmbPal = QComboBox()
        self.cmbPal.addItems(list(PALETTES) + ["LRGB (L + image couleur R)"])
        self.form.addRow("Palette", self.cmbPal)
        self.chkAlign = QCheckBox("Aligner les canaux entre eux")
        self.chkAlign.setChecked(True)
        self.chkNorm = QCheckBox("Égaliser fond et dispersion des canaux")
        self.chkNorm.setChecked(True)
        self.spnSat = dspin(0.0, 3.0, 0.05, 1.0)
        self.form.addRow(self.chkAlign)
        self.form.addRow(self.chkNorm)
        self.form.addRow("Saturation (LRGB)", self.spnSat)
        hint = QLabel("LRGB : renseignez L (mono) et, dans « Canal R », l'image COULEUR "
                      "(RGB) ; pour une composition RGB à partir de 3 filtres, choisissez "
                      "la palette « RGB ».")
        hint.setWordWrap(True)
        self.form.addRow(hint)

    def make_job(self):
        from ..core import narrowband as NB
        pal = self.cmbPal.currentText()
        paths = {k: r.text() for k, r in self.rows.items() if r.text()}
        align, norm, sat = self.chkAlign.isChecked(), self.chkNorm.isChecked(), self.spnSat.value()

        def run(progress=None):
            imgs = {k: load_one(p) for k, p in paths.items()}
            if pal.startswith("LRGB"):
                if "L" not in imgs or "R" not in imgs or imgs["R"].ndim != 3:
                    raise ValueError("LRGB : L (mono) et une image couleur dans « Canal R »")
                L = imgs["L"] if imgs["L"].ndim == 2 else imgs["L"].mean(axis=2)
                rgb = imgs["R"]
                if align:
                    from ..core.registration import estimate_transform, warp
                    if rgb.shape[:2] != L.shape:
                        import cv2
                        rgb = cv2.resize(rgb, (L.shape[1], L.shape[0]),
                                         interpolation=cv2.INTER_CUBIC)
                    rgb = warp(rgb, estimate_transform(L, rgb.mean(axis=2)))
                return NB.lrgb(L, rgb, sat)
            mono = {k: (v if v.ndim == 2 else v.mean(axis=2)) for k, v in imgs.items()}
            if align and len(mono) > 1:
                mono = NB.align_channels(mono, "Ha" if "Ha" in mono else None)
            return NB.compose(mono, pal, normalize=norm)
        return run


TOOLS = []  # (libellé, classe de dialogue) — complété plus bas par d'autres modules


def build_tools_menu(main):
    """Ajoute le menu « Outils » à la fenêtre principale."""
    bar = main.win.menuBar()
    menu = bar.addMenu("Outils")
    menu.setObjectName("menuOutils")
    main._tool_dialogs = {}

    def opener(cls):
        def _open():
            dlg = main._tool_dialogs.get(cls)
            if dlg is None:
                dlg = cls(main, main.win)
                main._tool_dialogs[cls] = dlg
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
        return _open
    for label, cls in registered_tools():
        if label is None:
            menu.addSeparator()
            continue
        act = menu.addAction(label)
        act.triggered.connect(opener(cls))
    return menu


def registered_tools():
    """Liste ordonnée des outils (None = séparateur)."""
    tools = [("Composition couleur (bande étroite, LRGB)…", ComposeDialog)]
    try:
        from . import tools_extra
        tools += tools_extra.TOOLS
    except ImportError:
        pass
    return tools


__all__ = ["ToolDialog", "ComposeDialog", "PathRow", "build_tools_menu", "send_to_finishing",
           "dspin", "ispin", "load_one", "Qt"]
