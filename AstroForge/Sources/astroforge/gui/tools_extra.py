# -*- coding: utf-8 -*-
"""Outils avancés du menu « Outils » (dialogues)."""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QListWidget, QPushButton, QVBoxLayout)

from .tools_menu import PathRow, ToolDialog, dspin, ispin, load_one

BAYERS = ("Auto (en-tête / RAW)", "Aucune (mono)", "RGGB", "BGGR", "GRBG", "GBRG")


def _bayer(text: str):
    return None if text.startswith(("Auto", "Aucune")) else text


class _FolderList(QVBoxLayout):
    """Liste de dossiers avec boutons Ajouter / Retirer."""

    def __init__(self, label: str):
        super().__init__()
        self.list = QListWidget()
        row = QHBoxLayout()
        b_add = QPushButton("Ajouter un dossier…")
        b_del = QPushButton("Retirer")
        row.addWidget(QLabel(label))
        row.addStretch(1)
        row.addWidget(b_add)
        row.addWidget(b_del)
        self.addLayout(row)
        self.addWidget(self.list)
        b_add.clicked.connect(self._add)
        b_del.clicked.connect(lambda: [self.list.takeItem(self.list.row(i))
                                       for i in self.list.selectedItems()])

    def _add(self):
        d = QFileDialog.getExistingDirectory(None, "Dossier")
        if d:
            self.list.addItem(d)

    def folders(self) -> list[str]:
        return [self.list.item(i).text() for i in range(self.list.count())]


# ----------------------------------------------------------- multi-nuits
class MultiNightDialog(ToolDialog):
    title = "Empilement multi-nuits"

    def build(self):
        from ..core.normalization import NORMALIZATIONS
        from ..core.stacking import METHODS
        from ..core.subframes import WEIGHT_MODES
        self.folders = _FolderList("Une nuit par dossier (tri automatique : lumières + "
                                   "calibration de la nuit)")
        self.form.addRow(self.folders)
        self.cmbMethod = QComboBox()
        self.cmbMethod.addItems(METHODS)
        self.cmbNorm = QComboBox()
        self.cmbNorm.addItems(NORMALIZATIONS)
        self.cmbNorm.setCurrentText("Locale")
        self.cmbWeights = QComboBox()
        self.cmbWeights.addItems(WEIGHT_MODES)
        self.cmbBayer = QComboBox()
        self.cmbBayer.addItems(BAYERS)
        self.spnSigma = dspin(1.0, 6.0, 0.1, 3.0, 1)
        self.form.addRow("Empilement", self.cmbMethod)
        self.form.addRow("Sigma", self.spnSigma)
        self.form.addRow("Normalisation", self.cmbNorm)
        self.form.addRow("Pondération", self.cmbWeights)
        self.form.addRow("Matrice de Bayer", self.cmbBayer)

    def make_job(self):
        from ..core.multisession import stack_sessions
        folders = self.folders.folders()
        if len(folders) < 1:
            raise ValueError("Ajoutez au moins un dossier de nuit.")
        args = dict(method=self.cmbMethod.currentText(), sigma=self.spnSigma.value(),
                    normalization=self.cmbNorm.currentText(),
                    weight_mode=self.cmbWeights.currentText(),
                    bayer=_bayer(self.cmbBayer.currentText()))

        def run(progress=None):
            img, info = stack_sessions(folders, progress=progress, **args)
            return img, info
        return run

    def on_result(self, res):
        img, info = res
        self.add_log(f"{info['n_frames']} images ({' + '.join(map(str, info['per_night']))} "
                     "par nuit) empilées.")
        self.set_result(img)


# ------------------------------------------------------------ mosaïque
class MosaicDialog(ToolDialog):
    title = "Mosaïque"

    def build(self):
        self.files = PathRow("files")
        self.form.addRow("Panneaux", self.files)
        self.cmbMode = QComboBox()
        self.cmbMode.addItems(["Automatique (WCS si présent, sinon points d'intérêt)",
                               "Astrométrie (WCS des en-têtes FITS)",
                               "Points d'intérêt (sans astrométrie : Soleil, Lune…)"])
        self.spnFeather = dspin(4, 400, 4, 64, 0)
        self.form.addRow("Méthode", self.cmbMode)
        self.form.addRow("Fondu des raccords (px)", self.spnFeather)

    def make_job(self):
        from ..core import mosaic as MO
        from ..core.image_io import load_image
        paths = self.files.paths()
        if len(paths) < 2:
            raise ValueError("Choisissez au moins deux panneaux.")
        mode, feather = self.cmbMode.currentText(), self.spnFeather.value()

        def run(progress=None):
            imgs = [load_image(p) for p in paths]
            wcss = [MO.wcs_from_fits(p) for p in paths]
            use_wcs = mode.startswith("Astro") or (mode.startswith("Auto") and all(wcss))
            if use_wcs:
                if not all(wcss):
                    raise ValueError("Panneau(x) sans WCS : résolvez-les d'abord (plate solving)")
                out, _w, _cov = MO.mosaic_wcs(imgs, wcss, feather=feather, progress=progress)
                return out, "astrométrie"
            out, _cov, _m = MO.mosaic_features(imgs, feather=feather, progress=progress)
            return out, "points d'intérêt"
        return run

    def on_result(self, res):
        img, how = res
        self.add_log(f"Mosaïque assemblée par {how}.")
        self.set_result(img)


# ---------------------------------------------------------------- comète
class CometDialog(ToolDialog):
    title = "Empilement de comète (étoiles + comète)"

    def build(self):
        from ..core.comet import COMPOSITE_MODES
        from ..core.stacking import METHODS
        self.files = PathRow("files")
        self.form.addRow("Images (déjà alignées sur les étoiles)", self.files)
        hint = QLabel("Position du noyau sur la PREMIÈRE et la DERNIÈRE image (pixels) ; "
                      "les positions intermédiaires sont interpolées d'après DATE-OBS "
                      "(ou le rang des images) puis affinées automatiquement.")
        hint.setWordWrap(True)
        self.form.addRow(hint)
        self.x1, self.y1 = dspin(0, 1e5, 1, 0, 1), dspin(0, 1e5, 1, 0, 1)
        self.x2, self.y2 = dspin(0, 1e5, 1, 0, 1), dspin(0, 1e5, 1, 0, 1)
        r1, r2 = QHBoxLayout(), QHBoxLayout()
        for r, a, b in ((r1, self.x1, self.y1), (r2, self.x2, self.y2)):
            r.addWidget(QLabel("x"))
            r.addWidget(a)
            r.addWidget(QLabel("y"))
            r.addWidget(b)
        self.form.addRow("Noyau, 1re image", r1)
        self.form.addRow("Noyau, dernière image", r2)
        self.cmbMethod = QComboBox()
        self.cmbMethod.addItems([m for m in METHODS if m != "Moyenne"])
        self.cmbMode = QComboBox()
        self.cmbMode.addItems(COMPOSITE_MODES)
        self.form.addRow("Rejet", self.cmbMethod)
        self.form.addRow("Résultat", self.cmbMode)

    def make_job(self):
        from ..core import comet as CO
        from ..core.image_io import load_image
        from ..core.session import read_header
        paths = self.files.paths()
        if len(paths) < 3:
            raise ValueError("Au moins trois images.")
        p1 = (self.x1.value(), self.y1.value())
        p2 = (self.x2.value(), self.y2.value())
        method, mode = self.cmbMethod.currentText(), self.cmbMode.currentText()

        def run(progress=None):
            frames = [load_image(p) for p in paths]
            times = CO.parse_times([read_header(p).get("DATE-OBS", "") for p in paths])
            pos = CO.interpolate_positions(len(frames), p1, p2, times)
            return CO.comet_stack(frames, pos, method=method, mode=mode, progress=progress)
        return run

    def on_result(self, res):
        self.add_log("Positions du noyau : " + ", ".join(f"({x:.1f}, {y:.1f})"
                                                        for x, y in res["positions"][:6]) + " …")
        self.set_result(res["composite"])


# ------------------------------------------------------ empilement en direct
class LiveStackDialog(ToolDialog):
    title = "Empilement en direct (dossier surveillé)"

    def build(self):
        from .preview import PreviewPane
        self.folder = PathRow("dir")
        self.form.addRow("Dossier d'acquisition", self.folder)
        self.calib = PathRow("dir")
        self.form.addRow("Dossier de darks (facultatif)", self.calib)
        self.flats = PathRow("dir")
        self.form.addRow("Dossier de flats (facultatif)", self.flats)
        self.cmbBayer = QComboBox()
        self.cmbBayer.addItems(BAYERS)
        self.form.addRow("Matrice de Bayer", self.cmbBayer)
        self.spnKappa = dspin(1.5, 6.0, 0.1, 3.0, 1)
        self.form.addRow("Rejet κ·σ", self.spnKappa)
        self.chkExisting = QCheckBox("Inclure les images déjà présentes")
        self.chkExisting.setChecked(True)
        self.form.addRow(self.chkExisting)
        self.pane = PreviewPane()
        self.pane.setMinimumHeight(220)
        self.form.addRow(self.pane)
        self.lblStatus = QLabel("—")
        self.form.addRow(self.lblStatus)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.stacker = None
        self.watcher = None

    def start(self):  # pas de Job : la surveillance est cadencée par un QTimer
        from ..core import calibration as C
        from ..core.livestack import FolderWatcher, LiveStacker
        folder = self.folder.text()
        if not os.path.isdir(folder):
            self.add_log("Choisissez le dossier d'acquisition.")
            return
        masters = C.build_masters(dark_dir=self.calib.text(), flat_dir=self.flats.text())
        self.stacker = LiveStacker(masters, _bayer(self.cmbBayer.currentText()),
                                   kappa=self.spnKappa.value())
        self.watcher = FolderWatcher(folder, include_existing=self.chkExisting.isChecked())
        self.timer.start(1500)
        self.btnRun.setEnabled(False)
        self.btnCancel.setEnabled(True)
        self.add_log(f"Surveillance de {folder}…")

    def cancel(self):
        self.timer.stop()
        self.btnRun.setEnabled(True)
        self.btnCancel.setEnabled(False)
        self.add_log("Surveillance arrêtée.")

    def _tick(self):
        if self.watcher is None:
            return
        for p in self.watcher.poll():
            ok = self.stacker.add(p)
            self.add_log(("✔ " if ok else "✖ ") + os.path.basename(p)
                         + ("" if ok else f" ({self.stacker.rejected[-1][1]})"))
            img = self.stacker.image()
            if img is not None:
                self.pane.set_after(img)
                self.pane.show_after()
                self.set_result(img)
        self.lblStatus.setText(self.stacker.status())

    def closeEvent(self, e):
        self.timer.stop()
        super().closeEvent(e)


TOOLS = [
    (None, None),
    ("Empilement multi-nuits…", MultiNightDialog),
    ("Mosaïque…", MosaicDialog),
    ("Comète (étoiles + comète)…", CometDialog),
    ("Empilement en direct…", LiveStackDialog),
]

__all__ = ["TOOLS", "np", "ispin", "QVBoxLayout"]


# ------------------------------------------------ dérotation (WinJUPOS)
class DerotationDialog(ToolDialog):
    title = "Dérotation d'une série d'images planétaires"

    def build(self):
        from ..core.derotation import PLANETS
        self.files = PathRow("files")
        self.form.addRow("Images empilées", self.files)
        hint = QLabel("Instant de chaque image : nom WinJUPOS (AAAA-MM-JJ-HHMM_D…), "
                      "sinon DATE-OBS de l'en-tête FITS. Angle de position du pôle (PA) "
                      "et latitude du centre (B) : éphémérides (WinJUPOS, IMCCE).")
        hint.setWordWrap(True)
        self.form.addRow(hint)
        self.cmbPlanet = QComboBox()
        self.cmbPlanet.addItems(list(PLANETS))
        self.spnPA = dspin(-180, 180, 0.5, 0.0, 1)
        self.spnB = dspin(-30, 30, 0.1, 0.0, 1)
        self.cmbDir = QComboBox()
        self.cmbDir.addItems(["Normal (est → ouest)", "Inversé (image retournée)"])
        self.form.addRow("Planète", self.cmbPlanet)
        self.form.addRow("PA du pôle nord (°)", self.spnPA)
        self.form.addRow("Latitude du centre B (°)", self.spnB)
        self.form.addRow("Sens de rotation", self.cmbDir)

    def make_job(self):
        from ..core import derotation as DR
        from ..core.image_io import load_image
        from ..core.session import read_header
        paths = self.files.paths()
        if len(paths) < 2:
            raise ValueError("Au moins deux images.")
        times = []
        for p in paths:
            t = DR.parse_winjupos_name(p) or read_header(p).get("DATE-OBS")
            if not t:
                raise ValueError(f"Instant inconnu pour {os.path.basename(p)}")
            times.append(t)
        args = dict(planet=self.cmbPlanet.currentText(), pa=self.spnPA.value(),
                    B=self.spnB.value(),
                    direction=1 if self.cmbDir.currentIndex() == 0 else -1)

        def run(progress=None):
            imgs = [load_image(p) for p in paths]
            return DR.derotate_series(imgs, times, progress=progress, **args)
        return run

    def on_result(self, res):
        img, mid, disc = res
        self.add_log(f"Série ramenée à {mid} UT (disque centre {disc[0]:.1f}, "
                     f"{disc[1]:.1f}, rayon {disc[2]:.1f} px).")
        self.mid_time = mid
        self.set_result(img)


# --------------------------------------------------------------- solaire
class SolarDialog(ToolDialog):
    title = "Outils solaires (limbe, protubérances)"

    def build(self):
        self.file = PathRow("file")
        self.form.addRow("Image solaire", self.file)
        self.chkInvert = QCheckBox("Inverser le disque (filaments en relief)")
        self.chkInvert.setChecked(True)
        self.chkFlat = QCheckBox("Compenser l'assombrissement centre-bord")
        self.chkFlat.setChecked(True)
        self.spnDiscGamma = dspin(0.2, 5.0, 0.05, 1.0)
        self.spnGain = dspin(0.5, 50.0, 0.5, 4.0, 1)
        self.spnPromGamma = dspin(0.2, 3.0, 0.05, 0.7)
        self.spnFeather = dspin(0.5, 30.0, 0.5, 3.0, 1)
        self.form.addRow(self.chkInvert)
        self.form.addRow(self.chkFlat)
        self.form.addRow("Gamma du disque", self.spnDiscGamma)
        self.form.addRow("Gain des protubérances", self.spnGain)
        self.form.addRow("Gamma des protubérances", self.spnPromGamma)
        self.form.addRow("Transition au limbe (px)", self.spnFeather)

    def make_job(self):
        from ..core import solar as SO
        path = self.file.text()
        if not path:
            raise ValueError("Choisissez une image.")
        args = dict(invert_disc=self.chkInvert.isChecked(), flatten=self.chkFlat.isChecked(),
                    disc_gamma=self.spnDiscGamma.value(), prom_gain=self.spnGain.value(),
                    prom_gamma=self.spnPromGamma.value(), feather=self.spnFeather.value())

        def run(progress=None):
            from ..core.image_io import load_image
            return SO.prominence_composite(load_image(path), **args)
        return run

    def on_result(self, res):
        img, (cx, cy, r) = res
        self.add_log(f"Limbe : centre ({cx:.1f}, {cy:.1f}), rayon {r:.1f} px.")
        self.set_result(img)


TOOLS += [
    (None, None),
    ("Dérotation de série planétaire (WinJUPOS)…", DerotationDialog),
    ("Outils solaires (limbe, protubérances)…", SolarDialog),
]


# ------------------------------------------------------------ annotation
class AnnotateDialog(ToolDialog):
    title = "Annotation des objets (Messier, NGC, IC)"

    def build(self):
        self.file = PathRow("file")
        self.form.addRow("Image résolue (FITS avec WCS)", self.file)
        self.chkCurrent = QCheckBox("Utiliser l'image et l'astrométrie de l'onglet Ciel profond")
        self.form.addRow(self.chkCurrent)
        self.spnMag = dspin(5, 18, 0.5, 14.0, 1)
        self.form.addRow("Magnitude limite (hors Messier)", self.spnMag)
        note = QLabel("Catalogue : OpenNGC (M. Verga), CC BY-SA 4.0.")
        note.setWordWrap(True)
        self.form.addRow(note)

    def make_job(self):
        from ..core.annotate import annotate
        from ..core.image_io import load_image
        from ..core.mosaic import wcs_from_fits
        mag = self.spnMag.value()
        if self.chkCurrent.isChecked() and self.main is not None:
            st = self.main.page_sky.state
            img, wcs = st.get("image"), st.get("wcs")
            if img is None or wcs is None:
                raise ValueError("Onglet Ciel profond : image empilée ET plate solving requis.")
        else:
            path = self.file.text()
            if not path:
                raise ValueError("Choisissez une image.")
            wcs = wcs_from_fits(path)
            if wcs is None:
                raise ValueError("Pas de WCS dans l'en-tête : résolvez l'image d'abord.")
            img = load_image(path)
        from ..core.display import autostretch

        def run(progress=None):
            base = img if float(np.median(img)) > 0.1 else autostretch(img)
            return annotate(base, wcs, mag)
        return run

    def on_result(self, res):
        img, objs = res
        self.add_log(f"{len(objs)} objet(s) dans le champ :")
        for o in objs[:60]:
            self.add_log(f"  {o['label']:<10} {o['type_fr']:<24} mag {o['mag'] or '—'}"
                         f"  {o.get('common') or ''}")
        self.set_result(img)


# ------------------------------------------------------ courbe de lumière
class LightCurveDialog(ToolDialog):
    title = "Courbe de lumière (photométrie différentielle)"

    def build(self):
        from PySide6.QtWidgets import QLineEdit
        self.files = PathRow("files")
        self.form.addRow("Série d'images", self.files)
        self.ref = PathRow("file")
        self.form.addRow("Référence résolue (WCS) — sinon 1re image", self.ref)
        self.cat = PathRow("file", "Catalogue CSV (*.csv)")
        self.form.addRow("Catalogue (ra, dec, mag)", self.cat)
        self.txtRA, self.txtDec = QLineEdit(), QLineEdit()
        self.txtRA.setPlaceholderText("ex. 20h41m25.9s")
        self.txtDec.setPlaceholderText("ex. +45°16'49\"")
        self.form.addRow("Cible — ascension droite", self.txtRA)
        self.form.addRow("Cible — déclinaison", self.txtDec)
        self.spnComp = ispin(1, 20, 5)
        self.form.addRow("Étoiles de comparaison", self.spnComp)
        self.canvas = None
        try:
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
            from matplotlib.figure import Figure
            self.fig = Figure(figsize=(6, 2.6), tight_layout=True)
            self.canvas = FigureCanvasQTAgg(self.fig)
            self.canvas.setMinimumHeight(200)
            self.form.addRow(self.canvas)
        except Exception:
            pass
        self.rows = []

    def make_job(self):
        from ..core.equipment import parse_coord
        from ..core.lightcurve import light_curve
        from ..core.mosaic import wcs_from_fits
        paths = self.files.paths()
        if len(paths) < 2:
            raise ValueError("Au moins deux images.")
        ref_path = self.ref.text() or paths[0]
        wcs = wcs_from_fits(ref_path)
        if wcs is None:
            raise ValueError("Référence sans WCS : résolvez-la (plate solving) et exportez-la en FITS.")
        cat = self.cat.text()
        if not cat:
            raise ValueError("Catalogue CSV requis (onglet Configuration › catalogue Gaia).")
        ra, dec = parse_coord(self.txtRA.text(), True), parse_coord(self.txtDec.text(), False)
        if ra is None or dec is None:
            raise ValueError("Coordonnées de la cible requises.")
        n_comp = self.spnComp.value()

        def run(progress=None):
            from ..core.image_io import load_image
            return light_curve(paths, wcs, cat, ra, dec, n_comp, ref_img=load_image(ref_path),
                               progress=progress)
        return run

    def on_result(self, rows):
        self.rows = rows
        ok = [r for r in rows if np.isfinite(r["dmag"])]
        self.add_log(f"{len(ok)} / {len(rows)} mesures valides.")
        for r in rows:
            if r["erreur"]:
                self.add_log(f"  ✖ {os.path.basename(r['fichier'])} : {r['erreur']}")
        if self.canvas is not None and ok:
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            x = [r["jd"] if r["jd"] else i for i, r in enumerate(ok)]
            ax.errorbar(x, [r["dmag"] for r in ok], yerr=[r["dmag_err"] for r in ok],
                        fmt="o", ms=3, color="#4b6cb7")
            ax.invert_yaxis()
            ax.set_xlabel("JD" if ok[0]["jd"] else "image")
            ax.set_ylabel("Δmag")
            self.canvas.draw_idle()
        self.btnSave.setEnabled(bool(ok))

    def save_result(self):
        from PySide6.QtWidgets import QFileDialog as FD
        from ..core.lightcurve import save_csv
        path, _ = FD.getSaveFileName(self, "Courbe de lumière", "courbe.csv", "CSV (*.csv)")
        if path:
            save_csv(self.rows, path)
            self.add_log(f"Enregistré : {path}")


TOOLS += [
    (None, None),
    ("Annotation des objets (Messier, NGC, IC)…", AnnotateDialog),
    ("Courbe de lumière…", LightCurveDialog),
]


# ---------------------------------------------------------- modèles IA
class ModelCatalogDialog(ToolDialog):
    title = "Modèles IA téléchargeables (catalogue)"

    def build(self):
        from PySide6.QtWidgets import QAbstractItemView, QLineEdit, QTableWidget
        from ..ai.manager import load_tool_settings
        self.txtUrl = QLineEdit(load_tool_settings().get("model_catalog_url", ""))
        self.txtUrl.setPlaceholderText("https://…/catalogue.json ou fichier local")
        row = QHBoxLayout()
        row.addWidget(self.txtUrl, 1)
        b = QPushButton("Charger")
        b.clicked.connect(self.load)
        row.addWidget(b)
        self.form.addRow("Catalogue", row)
        hint = QLabel("Aucun modèle n'est fourni par défaut : indiquez le catalogue d'une source "
                      "de confiance. Chaque fichier est vérifié par SHA-256 avant installation.")
        hint.setWordWrap(True)
        self.form.addRow(hint)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Modèle", "Tâche", "Licence", "Taille", "État"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.form.addRow(self.table)
        self.btnRun.setText("Installer la sélection")
        self.btnSend.hide()
        self.btnSave.hide()
        self.entries = []

    def load(self):
        from PySide6.QtWidgets import QTableWidgetItem
        from ..ai import catalog as CAT
        try:
            self.entries = CAT.load_catalog(self.txtUrl.text().strip())
        except Exception as e:
            self.add_log(f"✖ Catalogue : {e}")
            return
        self.table.setRowCount(len(self.entries))
        for r, e in enumerate(self.entries):
            size = f"{(e.get('size') or 0) / 1e6:.1f} Mo" if e.get("size") else ""
            for c, v in enumerate((e["name"], e["task"], e.get("license", ""), size,
                                   "installé" if CAT.is_installed(e) else "")):
                self.table.setItem(r, c, QTableWidgetItem(str(v)))
        self.add_log(f"{len(self.entries)} modèle(s) au catalogue.")
        try:  # mémorise l'URL du catalogue
            from ..ai.manager import load_tool_settings, save_tool_settings
            cfg = load_tool_settings()
            cfg["model_catalog_url"] = self.txtUrl.text().strip()
            save_tool_settings(cfg)
        except Exception:
            pass

    def make_job(self):
        from ..ai import catalog as CAT
        rows = sorted({i.row() for i in self.table.selectedIndexes()})
        if not rows:
            raise ValueError("Sélectionnez un ou plusieurs modèles.")
        entries = [self.entries[r] for r in rows]

        def run(progress=None):
            return [CAT.download_model(e, progress=progress) for e in entries]
        return run

    def on_result(self, paths):
        for p in paths:
            self.add_log(f"✔ Installé : {p}")
        from ..ai.manager import get_ai_manager, save_tool_settings, load_tool_settings
        get_ai_manager().reload_settings()
        save_tool_settings(load_tool_settings())  # rafraîchit les listes de fournisseurs
        self.load()


# ---------------------------------------------------- comparaison IA
class AICompareDialog(ToolDialog):
    title = "Comparaison avant / après (IA et algorithmes)"

    def build(self):
        from PySide6.QtWidgets import QGridLayout, QListWidgetItem, QWidget
        from .ai_dialog import populate_ai_tasks
        self.file = PathRow("file")
        self.form.addRow("Image (sinon image de la Finition)", self.file)
        self.cmbTask = QComboBox()
        populate_ai_tasks(self.cmbTask)
        self.cmbTask.currentIndexChanged.connect(self._refresh)
        self.form.addRow("Tâche", self.cmbTask)
        self.lst = QListWidget()
        self.lst.setMaximumHeight(120)
        self.form.addRow("Fournisseurs à comparer", self.lst)
        self.spnCrop = ispin(64, 2048, 384)
        self.form.addRow("Zone centrale (px)", self.spnCrop)
        self.spnStrength = dspin(0, 1, 0.05, 0.5)
        self.form.addRow("Force", self.spnStrength)
        self.gridW = QWidget()
        self.grid = QGridLayout(self.gridW)
        self.form.addRow(self.gridW)
        self._QListWidgetItem = QListWidgetItem
        self._refresh()
        self.resize(900, 760)

    def _refresh(self, *_):
        from PySide6.QtCore import Qt
        from ..ai.manager import get_ai_manager
        task = self.cmbTask.currentData()
        self.lst.clear()
        opts = [("classique", "Algorithme classique (référence)")] if task == "denoise" else []
        opts += [(p["id"], p["label"]) for p in get_ai_manager().providers(task)
                 if not (p["id"] == "onnx" and p["model"] is None)]
        for pid, label in opts:
            it = self._QListWidgetItem(label)
            it.setData(Qt.UserRole, pid)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked)
            self.lst.addItem(it)

    def make_job(self):
        import time
        from PySide6.QtCore import Qt
        from ..ai.manager import get_ai_manager
        from ..core.restoration import denoise
        task = self.cmbTask.currentData()
        provs = [(self.lst.item(i).data(Qt.UserRole), self.lst.item(i).text())
                 for i in range(self.lst.count())
                 if self.lst.item(i).checkState() == Qt.Checked]
        if not provs:
            raise ValueError("Aucun fournisseur disponible pour cette tâche.")
        img = load_one(self.file.text()) if self.file.text() else (
            self.main.page_common.image if self.main is not None else None)
        if img is None:
            raise ValueError("Choisissez une image.")
        n = self.spnCrop.value()
        h, w = img.shape[:2]
        y0, x0 = max(0, h // 2 - n // 2), max(0, w // 2 - n // 2)
        crop = np.asarray(img[y0:y0 + n, x0:x0 + n], np.float32)
        strength = self.spnStrength.value()

        def run(progress=None):
            out = [("Original", crop, 0.0, "")]
            for k, (pid, label) in enumerate(provs):
                t0 = time.time()
                try:
                    res = (denoise(crop, "Non-local means", strength) if pid == "classique"
                           else get_ai_manager().run(task, crop, pid, strength))
                    out.append((label, np.asarray(res, np.float32), time.time() - t0, ""))
                except Exception as e:
                    out.append((label, None, time.time() - t0, str(e)))
                if progress:
                    progress(k + 1, len(provs), label)
            return out
        return run

    def on_result(self, results):
        from PySide6.QtGui import QPixmap
        from .preview import np_to_qimage
        from ..core.display import autostretch
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        cols = 3
        for k, (label, img, dt, err) in enumerate(results):
            box = QVBoxLayout()
            title = QLabel(f"{label}" + (f" — {dt:.1f} s" if dt else ""))
            title.setWordWrap(True)
            pic = QLabel()
            if img is not None:
                pm = QPixmap.fromImage(np_to_qimage(autostretch(img)))
                pic.setPixmap(pm.scaled(260, 260))
            else:
                pic.setText("échec : " + err[:120])
                pic.setWordWrap(True)
            from PySide6.QtWidgets import QWidget
            cell = QWidget()
            cell.setLayout(box)
            box.addWidget(title)
            box.addWidget(pic)
            self.grid.addWidget(cell, k // cols, k % cols)
        self.results = results
        best = [r for r in results[1:] if r[1] is not None]
        if best:
            self.set_result(best[0][1])
        self.add_log(f"{len(results) - 1} résultat(s) comparé(s).")


TOOLS += [
    (None, None),
    ("Modèles IA téléchargeables…", ModelCatalogDialog),
    ("Comparaison avant / après (IA)…", AICompareDialog),
]


# --------------------------------------------------------------- scripts
SCRIPT_TEMPLATE = '''# Script AstroForge — variables disponibles :
#   np           : NumPy
#   af           : modules d'AstroForge (af.stacking, af.finishing, af.stretch,
#                  af.masks, af.narrowband, af.registration, af.image_io…)
#   image        : image de l'onglet Finition (float32 [0,1]) ou None
#   set_image(x) : renvoie une image vers la Finition
#   print(...)   : écrit dans le journal
if image is not None:
    out = af.stretch.ghs(image, D=3.0, b=0.0, SP=0.02)
    out = af.masks.chroma_denoise(out, 0.4)
    set_image(out)
    print("GHS + débruitage de chrominance appliqués", out.shape)
'''


def script_namespace(image=None):
    import importlib
    from types import SimpleNamespace
    mods = {}
    for name in ("image_io", "calibration", "stacking", "registration", "drizzle", "finishing",
                 "stretch", "masks", "narrowband", "normalization", "multisession", "mosaic",
                 "comet", "subframes", "session", "masters", "astrometry", "platesolve",
                 "annotate", "lightcurve", "planetary", "derotation", "solar", "wavelets",
                 "restoration", "gradient", "colorcal", "effects", "display", "sequence"):
        try:
            mods[name] = importlib.import_module(f"astroforge.core.{name}")
        except ImportError:
            pass
    return {"np": np, "af": SimpleNamespace(**mods), "image": image}


def run_script(code: str, image=None, filename: str = "<script>"):
    """Exécute un script utilisateur ; retourne (texte affiché, image ou None)."""
    import contextlib
    import io
    ns = script_namespace(None if image is None else np.asarray(image, np.float32).copy())
    result = {"image": None}
    ns["set_image"] = lambda x: result.__setitem__("image", np.asarray(x, np.float32))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, filename, "exec"), ns)
    return buf.getvalue(), result["image"]


class ScriptDialog(ToolDialog):
    title = "Scripts Python"

    def build(self):
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import QPlainTextEdit
        row = QHBoxLayout()
        b_open = QPushButton("Ouvrir…")
        b_save = QPushButton("Enregistrer…")
        row.addWidget(b_open)
        row.addWidget(b_save)
        row.addStretch(1)
        self.form.addRow(row)
        self.editor = QPlainTextEdit(SCRIPT_TEMPLATE)
        self.editor.setFont(QFont("monospace"))
        self.editor.setMinimumHeight(260)
        self.form.addRow(self.editor)
        b_open.clicked.connect(self._open)
        b_save.clicked.connect(self._save_script)
        self.resize(760, 680)

    def _open(self):
        p, _ = QFileDialog.getOpenFileName(self, "Script", "", "Python (*.py)")
        if p:
            with open(p, encoding="utf-8") as f:
                self.editor.setPlainText(f.read())

    def _save_script(self):
        p, _ = QFileDialog.getSaveFileName(self, "Script", "script.py", "Python (*.py)")
        if p:
            with open(p, "w", encoding="utf-8") as f:
                f.write(self.editor.toPlainText())

    def make_job(self):
        code = self.editor.toPlainText()
        img = self.main.page_common.image if self.main is not None else None

        def run(progress=None):
            return run_script(code, img)
        return run

    def on_result(self, res):
        text, img = res
        if text:
            self.add_log(text.rstrip())
        if img is not None:
            self.set_result(img)
            if self.main is not None:
                self.send_to_finishing()


TOOLS += [
    (None, None),
    ("Scripts Python…", ScriptDialog),
]
