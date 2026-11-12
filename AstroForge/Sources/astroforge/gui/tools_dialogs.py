# -*- coding: utf-8 -*-
"""Boîtes de dialogue des outils : session, bibliothèque de masters, analyse
des brutes, et dialogues génériques réutilisés par les autres outils."""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QAbstractItemView, QDialog, QDialogButtonBox,
                               QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QMessageBox, QPlainTextEdit, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout)


class SessionDialog(QDialog):
    """Résultat du tri automatique : choix du groupe de lumières à traiter."""

    def __init__(self, plan, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Session triée automatiquement")
        self.resize(720, 460)
        self.plan = plan
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Groupes de lumières détectés (objet · filtre · pose · gain) "
                             "avec leurs images de calibration :"))
        self.list = QListWidget()
        for i, g in enumerate(plan.groups):
            it = QListWidgetItem(g.summary())
            it.setData(Qt.UserRole, i)
            self.list.addItem(it)
        if plan.groups:
            self.list.setCurrentRow(0)
        lay.addWidget(self.list, 1)
        if plan.unused:
            lay.addWidget(QLabel(f"{len(plan.unused)} fichier(s) non utilisé(s) "
                                 "(calibration sans lumières correspondantes)."))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Utiliser ce groupe")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def selected_group(self):
        it = self.list.currentItem()
        return None if it is None else self.plan.groups[it.data(Qt.UserRole)]


class MasterLibraryDialog(QDialog):
    """Liste et suppression des masters de la bibliothèque."""

    COLS = ("Type", "Pose (s)", "Gain", "Temp. (°C)", "Bin", "Taille", "Filtre",
            "Images", "Date")

    def __init__(self, parent=None, lib=None):
        super().__init__(parent)
        from ..core import masters as M
        self.M, self.lib = M, lib
        self.setWindowTitle("Bibliothèque de masters")
        self.resize(820, 420)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"Dossier : {lib or M.LIBRARY_DIR}"))
        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        b_del = QPushButton("Supprimer la sélection")
        b_del.clicked.connect(self._delete)
        row.addWidget(b_del)
        row.addStretch(1)
        b_close = QPushButton("Fermer")
        b_close.clicked.connect(self.accept)
        row.addWidget(b_close)
        lay.addLayout(row)
        self.refresh()

    def refresh(self):
        self.entries = self.M.load_index(self.lib)
        self.table.setRowCount(len(self.entries))
        for r, e in enumerate(self.entries):
            vals = (e.get("kind"), e.get("exptime"), e.get("gain"), e.get("temp"),
                    e.get("binning") or 1, f"{e.get('width')}×{e.get('height')}",
                    e.get("filter") or "", e.get("ncombine") or "", e.get("date") or "")
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem("" if v is None else str(v)))

    def _delete(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        if QMessageBox.question(self, "Bibliothèque", f"Supprimer {len(rows)} master(s) ?") \
                != QMessageBox.Yes:
            return
        for r in rows:
            self.M.delete_master(self.entries[r], self.lib)
        self.refresh()


class TextReportDialog(QDialog):
    """Affiche un rapport texte (copiable)."""

    def __init__(self, title: str, text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(720, 480)
        lay = QVBoxLayout(self)
        box = QPlainTextEdit(text)
        box.setReadOnly(True)
        lay.addWidget(box)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        bb.accepted.connect(self.accept)
        lay.addWidget(bb)


def short_path(p: str, n: int = 2) -> str:
    parts = os.path.normpath(p).split(os.sep)
    return os.sep.join(parts[-n:])


__all__ = ["SessionDialog", "MasterLibraryDialog", "TextReportDialog", "short_path", "np"]


class SubframeDialog(QDialog):
    """Analyse des brutes : graphique par image, tableau, rejet et poids."""

    COLS = ("Garder", "Fichier", "Étoiles", "FWHM (px)", "Excentricité", "Fond",
            "SNR", "Poids", "Remarques")

    def __init__(self, rows, parent=None, mode="Mixte"):
        super().__init__(parent)
        from PySide6.QtWidgets import QComboBox
        from ..core import subframes as SF
        self.SF = SF
        self.rows = rows
        self.setWindowTitle("Analyse des brutes")
        self.resize(980, 720)
        lay = QVBoxLayout(self)
        self.canvas = None
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self.fig = Figure(figsize=(9, 3.6), tight_layout=True)
            self.canvas = FigureCanvasQTAgg(self.fig)
            self.canvas.setMinimumHeight(260)
            lay.addWidget(self.canvas, 2)
        except Exception as e:  # matplotlib/Qt indisponible : tableau seul
            lay.addWidget(QLabel(f"(graphique indisponible : {e})"))
        top = QHBoxLayout()
        top.addWidget(QLabel("Pondération :"))
        self.cmbMode = QComboBox()
        self.cmbMode.addItems(list(SF.WEIGHT_MODES))
        self.cmbMode.setCurrentText(mode)
        self.cmbMode.currentTextChanged.connect(self._recompute)
        top.addWidget(self.cmbMode)
        top.addStretch(1)
        self.lblSummary = QLabel()
        top.addWidget(self.lblSummary)
        lay.addLayout(top)
        self.table = QTableWidget(len(rows), len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemChanged.connect(self._toggled)
        lay.addWidget(self.table, 3)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Appliquer la sélection et les poids")
        self.btnCsv = bb.addButton("Exporter CSV…", QDialogButtonBox.ActionRole)
        self.btnCsv.clicked.connect(self._export_csv)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._recompute()

    def _recompute(self, *_):
        w = self.SF.compute_weights(self.rows, self.cmbMode.currentText())
        for r, x in zip(self.rows, w):
            r["weight"] = x
        self._fill()
        self._plot()

    def _fill(self):
        self.table.blockSignals(True)
        for i, r in enumerate(self.rows):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Checked if r.get("keep", True) else Qt.Unchecked)
            self.table.setItem(i, 0, chk)
            vals = (short_path(r["path"], 1), r["stars"], f"{r['fwhm']:.2f}",
                    f"{r['ecc']:.2f}", f"{r['background']:.4f}", f"{r['snr']:.1f}",
                    f"{r.get('weight', 1.0):.2f}",
                    r.get("error") or "; ".join(r.get("reasons", [])))
            for c, v in enumerate(vals, start=1):
                self.table.setItem(i, c, QTableWidgetItem(str(v)))
        self.table.blockSignals(False)
        kept = sum(1 for r in self.rows if r.get("keep", True))
        self.lblSummary.setText(f"{kept} / {len(self.rows)} images gardées")

    def _toggled(self, item):
        if item.column() != 0:
            return
        self.rows[item.row()]["keep"] = item.checkState() == Qt.Checked
        self._recompute()

    def _plot(self):
        if self.canvas is None:
            return
        self.fig.clear()
        x = np.arange(1, len(self.rows) + 1)
        keep = np.array([r.get("keep", True) for r in self.rows])
        series = (("FWHM (px)", [r["fwhm"] for r in self.rows]),
                  ("Étoiles", [r["stars"] for r in self.rows]),
                  ("Fond", [r["background"] for r in self.rows]),
                  ("Poids", [r.get("weight", 1.0) for r in self.rows]))
        for k, (title, vals) in enumerate(series):
            ax = self.fig.add_subplot(1, 4, k + 1)
            v = np.asarray(vals, float)
            ax.plot(x, v, color="#4b6cb7", lw=1)
            ax.scatter(x[keep], v[keep], s=12, color="#4b6cb7")
            ax.scatter(x[~keep], v[~keep], s=22, color="#d1495b", marker="x")
            ax.set_title(title, fontsize=9)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("image", fontsize=7)
        self.canvas.draw_idle()

    def _export_csv(self):
        from PySide6.QtWidgets import QFileDialog
        import csv
        path, _ = QFileDialog.getSaveFileName(self, "Exporter l'analyse", "analyse_brutes.csv",
                                              "CSV (*.csv)")
        if not path:
            return
        keys = ("path", "stars", "fwhm", "ecc", "background", "noise", "signal", "snr",
                "weight", "keep")
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f, delimiter=";")
            wr.writerow(keys + ("remarques",))
            for r in self.rows:
                wr.writerow([r.get(k) for k in keys] + ["; ".join(r.get("reasons", []))])

    def result_selection(self):
        """(chemins gardés, {chemin: poids})."""
        kept = [r for r in self.rows if r.get("keep", True)]
        return [r["path"] for r in kept], {r["path"]: float(r.get("weight", 1.0)) for r in kept}
