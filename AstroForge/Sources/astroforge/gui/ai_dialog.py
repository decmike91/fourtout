# -*- coding: utf-8 -*-
"""Boîte de dialogue de configuration des outils IA externes (chemins binaires)."""
from __future__ import annotations

from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLineEdit, QPushButton,
                               QVBoxLayout, QLabel)

from ..ai.manager import (load_tool_settings, save_tool_settings, SETTINGS_PATH,
                          TASKS, TASK_LABELS)


def populate_ai_tasks(combo):
    """Identifiants stables en données ; libellés français à l'écran."""
    selected = combo.currentData() or combo.currentText()
    combo.clear()
    for task in TASKS:
        combo.addItem(TASK_LABELS[task], task)
    index = combo.findData(selected)
    combo.setCurrentIndex(index if index >= 0 else 0)


class AIToolsDialog(QDialog):
    """Chemins séparés : GraXpert, Cosmic Clarity, StarNet2 et DeepSNR.
    Les modèles ONNX se déposent simplement dans models/ai/."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Outils IA locaux — chemins")
        self.setMinimumWidth(620)
        self.cfg = load_tool_settings()
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(
            "AstroForge peut appeler des modèles IA installés sur votre machine.\n"
            "• Modèles ONNX : fichier .onnx et .json déclarant la tâche dans models/ai/.\n"
            "• GraXpert : binaire (GraXpert-linux / GraXpert-win64.exe…)\n"
            "• Cosmic Clarity : dossier conservé ; utilisation manuelle uniquement.\n"
            "• StarNet2 : CLI actuelle, suppression des étoiles.\n"
            "• DeepSNR : CLI actuelle, débruitage (minimum 512 × 512)."))
        form = QFormLayout()
        self.ed_graxpert = QLineEdit(self.cfg.get("graxpert", {}).get("binary", ""))
        self.ed_cc = QLineEdit(self.cfg.get("cosmicclarity", {}).get("binary", ""))
        self.ed_starnet = QLineEdit(self.cfg.get("starnet2", {}).get("binary", ""))
        self.ed_deepsnr = QLineEdit(self.cfg.get("deepsnr", {}).get("binary", ""))
        for label, ed in (("Binaire GraXpert", self.ed_graxpert),
                          ("Dossier Cosmic Clarity (manuel)", self.ed_cc),
                          ("Binaire StarNet2 (étoiles)", self.ed_starnet),
                          ("Binaire DeepSNR (débruitage)", self.ed_deepsnr)):
            row = QHBoxLayout()
            row.addWidget(ed)
            btn = QPushButton("…")
            btn.clicked.connect(lambda _=False, e=ed: self._browse(e))
            row.addWidget(btn)
            form.addRow(label, row)
        lay.addLayout(form)
        legacy = self.cfg.get("starnet2", {}).get("legacy_binary")
        if legacy:
            notice = QLabel("Ancien chemin conservé pour vérification : " + legacy +
                            "\nVérifiez les deux champs StarNet2 et DeepSNR avant utilisation.")
            notice.setWordWrap(True)
            lay.addWidget(notice)
        lay.addWidget(QLabel(f"Réglages enregistrés dans : {SETTINGS_PATH}"))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _browse(self, ed: QLineEdit):
        if ed is self.ed_cc:
            path = QFileDialog.getExistingDirectory(self, "Choisir le dossier Cosmic Clarity")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Choisir un binaire")
        if path:
            ed.setText(path)

    def accept(self):
        self.cfg["graxpert"]["binary"] = self.ed_graxpert.text().strip()
        self.cfg["cosmicclarity"]["binary"] = self.ed_cc.text().strip()
        self.cfg["starnet2"]["binary"] = self.ed_starnet.text().strip()
        self.cfg["deepsnr"]["binary"] = self.ed_deepsnr.text().strip()
        save_tool_settings(self.cfg)
        super().accept()
