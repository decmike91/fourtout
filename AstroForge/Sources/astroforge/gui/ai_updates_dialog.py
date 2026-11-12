# -*- coding: utf-8 -*-
"""Boîte de dialogue « Rechercher les mises à jour des modèles IA ».

Catalogue limité aux pages OFFICIELLES des outils (aucune autre URL) :
- GraXpert : vérification en ligne possible via l'API GitHub
  (releases/latest), actif d'installation attendu graxpert-windows-amd64.msi ;
- StarNet2, DeepSNR, Cosmic Clarity : PAS d'API de version fiable →
  « à vérifier sur la page officielle » ; « Télécharger » ouvre la page
  officielle (et pour Cosmic Clarity, le dossier Google Drive officiel) ;
- Modèles ONNX locaux : fichiers .onnx fournis par l'utilisateur dans
  models/ai/ — aucun registre en ligne, seule l'ouverture du dossier est
  proposée.

Les vérifications réseau tournent dans un thread (QThread), sont annulables
et ne bloquent jamais l'interface ; tout échec (réseau, délai, JSON) se
dégrade en « vérification impossible » sans plantage.
"""
from __future__ import annotations

import json
import urllib.request

from PySide6.QtCore import QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                               QPushButton, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from ..ai.manager import load_tool_settings
from ..ai.onnx_runner import MODELS_DIR, list_models

# Catégorie officielle UNIQUE — ne pas en inventer d'autres.
CATALOG = [
    {"id": "graxpert", "label": "GraXpert",
     "page": "https://github.com/Steffenhir/GraXpert/releases/latest",
     "api": "https://api.github.com/repos/Steffenhir/GraXpert/releases/latest",
     "asset": "graxpert-windows-amd64.msi", "settings_key": "graxpert"},
    {"id": "starnet2", "label": "StarNet2",
     "page": "https://starnetastro.com/documentation/starnet/command-line-tool/",
     "settings_key": "starnet2"},
    {"id": "deepsnr", "label": "DeepSNR",
     "page": "https://starnetastro.com/documentation/deepsnr/command-line-tool/",
     "settings_key": "deepsnr"},
    {"id": "cosmicclarity", "label": "Cosmic Clarity",
     "page": "https://www.setiastro.com/cosmic-clarity",
     "drive": "https://drive.google.com/drive/folders/1MKsuA4XgIMd0cJshJHEWdN782cXvzR3Y",
     "settings_key": "cosmicclarity"},
]

NOT_INSTALLED = "non installé"
PENDING_TEXT = "à vérifier sur la page officielle"
FAILED_TEXT = "vérification impossible"
ONNX_GUIDANCE = ("Aucun modèle ONNX dans models/ai — déposez-y vos fichiers "
                 ".onnx (+ .json de tâche). Fichiers fournis par "
                 "l'utilisateur : aucun registre en ligne.")


def fetch_latest_release(api_url: str, timeout: float = 10.0,
                         asset_hint: str | None = None) -> dict:
    """Interroge une API « releases/latest » de style GitHub (fonction de
    module, mockable dans les tests). Retourne {"tag", "asset_url"} ; lève
    OSError/ValueError en cas d'échec réseau ou de réponse inattendue."""
    req = urllib.request.Request(
        api_url, headers={"Accept": "application/vnd.github+json",
                          "User-Agent": "AstroForge"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    tag = payload.get("tag_name")
    if not tag:
        raise ValueError("tag_name absent de la réponse GitHub")
    asset_url = ""
    if asset_hint:
        for asset in payload.get("assets") or []:
            if asset.get("name") == asset_hint:
                asset_url = asset.get("browser_download_url", "")
                break
    return {"tag": str(tag), "asset_url": asset_url}


class CheckThread(QThread):
    """Vérifications réseau hors interface, annulation coopérative."""
    rowChecked = Signal(int, object)   # index de ligne, résultat ou exception
    finishedAll = Signal()

    def __init__(self, rows, parent=None):
        super().__init__(parent)
        self._rows = list(rows)
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        for row, api_url, asset_hint in self._rows:
            if self._cancelled:
                break
            try:
                result = fetch_latest_release(api_url, asset_hint=asset_hint)
            except Exception as exc:  # réseau, délai, JSON : dégradation douce
                result = exc
            self.rowChecked.emit(row, result)
        self.finishedAll.emit()


class AIUpdatesDialog(QDialog):
    """Table des outils IA + modèles ONNX locaux, états et actions."""

    COL_TOOL, COL_STATE, COL_VERSION, COL_ACTIONS = 0, 1, 2, 3

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mises à jour des modèles IA")
        self.setMinimumWidth(860)
        self._thread = None
        self._download_urls: dict[str, str] = {}
        lay = QVBoxLayout(self)
        intro = QLabel(
            "Recherche des mises à jour des outils et modèles IA.\n"
            "• Seule la version de GraXpert est interrogeable en ligne (API "
            "GitHub) ; pour StarNet2, DeepSNR et Cosmic Clarity, la page "
            "officielle fait foi.\n"
            "• Les modèles ONNX sont des fichiers fournis par l'utilisateur "
            "dans models/ai/ : aucun registre en ligne.")
        intro.setWordWrap(True)
        lay.addWidget(intro)
        self.tbl = QTableWidget(0, 4, self)
        self.tbl.setObjectName("tblAIUpdates")
        self.tbl.setHorizontalHeaderLabels(
            ["Outil", "État / chemin configuré", "Dernière version en ligne",
             "Actions"])
        self.tbl.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setColumnWidth(self.COL_TOOL, 130)
        self.tbl.setColumnWidth(self.COL_STATE, 230)
        self.tbl.setColumnWidth(self.COL_VERSION, 190)
        self.tbl.setColumnWidth(self.COL_ACTIONS, 300)
        lay.addWidget(self.tbl, 1)
        bar = QHBoxLayout()
        self.btnCheck = QPushButton("Vérifier en ligne")
        self.btnCheck.setObjectName("btnCheckOnline")
        self.btnCancelCheck = QPushButton("Annuler la vérification")
        self.btnCancelCheck.setObjectName("btnCancelCheck")
        self.btnCancelCheck.setEnabled(False)
        self.btnFolder = QPushButton("Ouvrir le dossier modèles")
        self.btnFolder.setObjectName("btnOpenModels")
        self.lblCheck = QLabel("Prêt.")
        self.lblCheck.setObjectName("lblCheckState")
        bar.addWidget(self.btnCheck)
        bar.addWidget(self.btnCancelCheck)
        bar.addWidget(self.btnFolder)
        bar.addStretch(1)
        bar.addWidget(self.lblCheck)
        lay.addLayout(bar)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.btnCheck.clicked.connect(self.start_check)
        self.btnCancelCheck.clicked.connect(self.cancel_check)
        self.btnFolder.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(MODELS_DIR)))
        self.refresh_states()

    # ------------------------------------------------------------- états
    def refresh_states(self):
        """Reconstruit les lignes : outils du catalogue puis modèles ONNX."""
        if self._thread is not None and self._thread.isRunning():
            return
        self.tbl.setRowCount(0)
        self._download_urls = {}
        self._api_rows = []
        cfg = load_tool_settings()
        for entry in CATALOG:
            row = self.tbl.rowCount()
            self.tbl.insertRow(row)
            binary = str(cfg.get(entry["settings_key"], {}).get("binary", ""))
            state = binary if binary else NOT_INSTALLED
            self.tbl.setItem(row, self.COL_TOOL,
                             QTableWidgetItem(entry["label"]))
            self.tbl.setItem(row, self.COL_STATE, QTableWidgetItem(state))
            self.tbl.setItem(row, self.COL_VERSION,
                             QTableWidgetItem(PENDING_TEXT))
            self._download_urls[entry["id"]] = entry["page"]
            self._add_tool_actions(row, entry)
            if entry.get("api"):
                self._api_rows.append((row, entry["api"], entry.get("asset")))
        self._add_onnx_rows()

    def _add_tool_actions(self, row, entry):
        widget = QWidget()
        col = QVBoxLayout(widget)
        col.setContentsMargins(2, 0, 2, 0)
        col.setSpacing(2)
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        download = QPushButton("Télécharger")
        download.setObjectName(f"btnDownload_{entry['id']}")
        download.clicked.connect(
            lambda _=False, tool=entry["id"]: QDesktopServices.openUrl(
                QUrl(self._download_urls.get(tool, ""))))
        bar.addWidget(download)
        configure = QPushButton("Configurer…")
        configure.setObjectName(f"btnConfigure_{entry['id']}")
        configure.clicked.connect(lambda _=False, r=row: self._configure(r))
        bar.addWidget(configure)
        col.addLayout(bar)
        if entry.get("drive"):
            bar2 = QHBoxLayout()
            bar2.setContentsMargins(0, 0, 0, 0)
            drive = QPushButton("Dossier officiel (Drive)")
            drive.setObjectName(f"btnDrive_{entry['id']}")
            drive.clicked.connect(
                lambda _=False: QDesktopServices.openUrl(QUrl(entry["drive"])))
            bar2.addWidget(drive)
            col.addLayout(bar2)
        widget.setStyleSheet("QPushButton{padding:2px 6px;}")
        self.tbl.setCellWidget(row, self.COL_ACTIONS, widget)
        self.tbl.resizeRowToContents(row)

    def _add_onnx_rows(self):
        models = list_models()
        if not models:
            row = self.tbl.rowCount()
            self.tbl.insertRow(row)
            self.tbl.setItem(row, self.COL_TOOL, QTableWidgetItem("Modèles ONNX"))
            self.tbl.setItem(row, self.COL_STATE, QTableWidgetItem(MODELS_DIR))
            self.tbl.setItem(row, self.COL_VERSION,
                             QTableWidgetItem(ONNX_GUIDANCE))
            self._add_folder_button(row)
            return
        for model in models:
            row = self.tbl.rowCount()
            self.tbl.insertRow(row)
            self.tbl.setItem(row, self.COL_TOOL,
                             QTableWidgetItem(f"ONNX : {model['name']}"))
            self.tbl.setItem(row, self.COL_STATE,
                             QTableWidgetItem(model["path"]))
            self.tbl.setItem(row, self.COL_VERSION,
                             QTableWidgetItem("fichier local (utilisateur)"))
            self._add_folder_button(row)

    def _add_folder_button(self, row):
        widget = QWidget()
        bar = QHBoxLayout(widget)
        bar.setContentsMargins(2, 0, 2, 0)
        folder = QPushButton("Ouvrir le dossier modèles")
        folder.setObjectName(f"btnOpenModels_row{row}")
        folder.clicked.connect(
            lambda _=False: QDesktopServices.openUrl(QUrl.fromLocalFile(MODELS_DIR)))
        bar.addWidget(folder)
        widget.setStyleSheet("QPushButton{padding:2px 6px;}")
        self.tbl.setCellWidget(row, self.COL_ACTIONS, widget)
        self.tbl.resizeRowToContents(row)

    def _configure(self, _row):
        from .ai_dialog import AIToolsDialog
        AIToolsDialog(self).exec()
        self.refresh_states()

    # ------------------------------------------------- vérification en ligne
    def _set_busy(self, busy, text=None):
        self.btnCheck.setEnabled(not busy)
        self.btnCancelCheck.setEnabled(busy)
        if text is not None:
            self.lblCheck.setText(text)

    def start_check(self):
        if self._thread is not None and self._thread.isRunning():
            return
        self._set_busy(True, "Vérification en cours…")
        self._thread = CheckThread(self._api_rows, self)
        self._thread.rowChecked.connect(self._on_row_checked)
        self._thread.finishedAll.connect(
            lambda: self._set_busy(False, "Vérification terminée."))
        self._thread.start()

    def cancel_check(self):
        if self._thread is not None and self._thread.isRunning():
            self._thread.cancel()
            self._set_busy(True, "Annulation…")

    def perform_check(self):
        """Vérification synchrone (les tests l'utilisent ; l'interface passe
        par le thread). Mise à jour des cellules, exceptions attrapées."""
        for row, api_url, asset in self._api_rows:
            try:
                self._on_row_checked(
                    row, fetch_latest_release(api_url, asset_hint=asset))
            except Exception as exc:
                self._on_row_checked(row, exc)
        return True

    def _on_row_checked(self, row, result):
        if isinstance(result, Exception):
            self.tbl.setItem(row, self.COL_VERSION,
                             QTableWidgetItem(FAILED_TEXT))
            return
        text = f"{result['tag']} (en ligne)"
        if result.get("asset_url"):
            text += " — installateur détecté"
            self._download_urls["graxpert"] = result["asset_url"]
        self.tbl.setItem(row, self.COL_VERSION, QTableWidgetItem(text))

    def version_text(self, row) -> str:
        item = self.tbl.item(row, self.COL_VERSION)
        return item.text() if item else ""

    def closeEvent(self, event):
        if self._thread is not None and self._thread.isRunning():
            self._thread.cancel()
            self._thread.wait(500)
        super().closeEvent(event)
