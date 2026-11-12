# -*- coding: utf-8 -*-
"""Barre de profils réutilisable : [profil ▾] [Enregistrer] [Réinitialiser].
Chaque groupe de paramètres obtient la sienne ; le profil « standard »
contient les valeurs par défaut."""
from __future__ import annotations

from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QInputDialog, QPushButton,
                               QVBoxLayout, QWidget)

from ..processing import profiles as P


class ProfileBar(QWidget):
    def __init__(self, step: str, defaults: dict, get_params, set_params,
                 parent=None):
        super().__init__(parent)
        self.step = step
        self.defaults = dict(defaults)
        self.get_params = get_params
        self.set_params = set_params
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        self.cmb = QComboBox()
        self.btn_save = QPushButton("💾 Enregistrer")
        self.btn_reset = QPushButton("↺ Standard")
        self.btn_save.setToolTip("Enregistrer les paramètres actuels comme profil")
        self.btn_reset.setToolTip("Revenir aux valeurs du profil standard (défauts)")
        lay.addWidget(self.cmb, 1)
        lay.addWidget(self.btn_save)
        lay.addWidget(self.btn_reset)
        self.btn_save.clicked.connect(self._save)
        self.btn_reset.clicked.connect(self._reset)
        self.cmb.currentTextChanged.connect(self._load)
        self.reload_list()

    def reload_list(self):
        self.cmb.blockSignals(True)
        self.cmb.clear()
        self.cmb.addItems(P.list_profiles(self.step))
        self.cmb.blockSignals(False)

    def _save(self):
        name, ok = QInputDialog.getText(self, "Enregistrer le profil",
                                        f"Nom du profil ({self.step}) :")
        if not ok:
            return
        try:
            P.save_profile(self.step, name, self.get_params())
        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Profil", str(e))
            return
        self.reload_list()
        self.cmb.setCurrentText(name.strip() if name.strip() != "standard"
                                else "standard")

    def _load(self, name: str):
        if not name:
            return
        self.set_params(P.load_profile(self.step, name,
                                       {self.step: {"params": self.defaults}}))

    def _reset(self):
        self.set_params(dict(self.defaults))


def add_profile_bar(layout: QVBoxLayout, step: str, defaults: dict,
                    get_params, set_params) -> ProfileBar:
    bar = ProfileBar(step, defaults, get_params, set_params)
    layout.addWidget(bar)
    return bar
