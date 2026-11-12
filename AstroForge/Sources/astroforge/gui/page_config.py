# -*- coding: utf-8 -*-
"""Onglet Configuration : profils d'équipement (caméra, télescope, réducteur,
pollution lumineuse), choix astrométrie par cases à cocher, catalogue Gaia,
détermination automatique de la focale."""
from __future__ import annotations
import os

from PySide6.QtWidgets import (QComboBox, QFileDialog,
                               QInputDialog, QLabel, QMessageBox,
                               QVBoxLayout, QWidget)

from ..core import equipment as EQ
from .loader import load_ui
from .workers import Job


class PageConfig(QWidget):
    def __init__(self, parent=None, shared=None):
        super().__init__(parent)
        self.shared = shared or {}
        self.ui = load_ui("page_config.ui", self)
        _outer = QVBoxLayout(self)   # la page suit la taille de la fenêtre
        _outer.setContentsMargins(0, 0, 0, 0)
        _outer.addWidget(self.ui)

        self.settings = EQ.load_settings()
        self._loading = False
        self._build_perf_group()
        self._wire()
        self._refresh_profiles()
        self._o("cmbProfile").setCurrentText(self.settings.get("active_profile",
                                                              "standard"))

    # ------------------------------------------------------------------ base
    def _o(self, name):
        return self.ui.findChild(object, name)

    def _log(self, msg):
        print(f"[config] {msg}")

    def _wire(self):
        self._o("btnProfSave").clicked.connect(self._save_profile)
        self._o("btnProfDel").clicked.connect(self._delete_profile)
        self._o("btnBrowseImg").clicked.connect(self._browse_img)
        self._o("btnGenCatalog").clicked.connect(self._gen_catalog)
        self._o("btnAutoFocal").clicked.connect(self._auto_focal)
        for name in ("spnPixel", "spnResW", "spnResH", "spnFocal", "spnAperture",
                     "spnReducerFactor", "spnTemp", "spnMagLimit"):
            self._o(name).valueChanged.connect(self._update_optics)
        self._o("chkReducer").toggled.connect(self._update_optics)
        self._o("chkUseSolveField").toggled.connect(self._save_settings)
        self._o("txtCatalogDef").editingFinished.connect(self._save_settings)
        self._o("cmbProfile").currentTextChanged.connect(self._on_profile)

    # ------------------------------------------------------------------ profils
    def _refresh_profiles(self):
        cmb: QComboBox = self._o("cmbProfile")
        cmb.blockSignals(True)
        cmb.clear()
        names = EQ.list_profiles()
        if "standard" not in names:
            names.insert(0, "standard")
        cmb.addItems(names)
        cmb.blockSignals(False)

    def _on_profile(self, name: str):
        if not name:
            return
        self._loading = True
        p = EQ.load_profile(name)
        self._o("txtCamera").setText(p.get("camera", ""))
        self._o("spnPixel").setValue(float(p.get("pixel_um", 3.8)))
        self._o("spnResW").setValue(int(p.get("sensor_w_px", 5000)))
        self._o("spnResH").setValue(int(p.get("sensor_h_px", 3500)))
        self._o("spnTemp").setValue(float(p.get("sensor_temp", -10.0)))
        self._o("cmbBayer").setCurrentText(p.get("bayer", "RGGB"))
        self._o("txtScope").setText(p.get("telescope", ""))
        self._o("spnAperture").setValue(float(p.get("aperture_mm", 100.0)))
        self._o("spnFocal").setValue(float(p.get("focal_mm", 500.0)))
        self._o("chkReducer").setChecked(bool(p.get("reducer", False)))
        self._o("spnReducerFactor").setValue(float(p.get("reducer_factor", 0.8)))
        self._o("cmbMode").setCurrentText(p.get("mode", "Ciel profond"))
        self._o("cmbPollution").setCurrentText(
            p.get("light_pollution", EQ.POLLUTION[1]))
        self._loading = False
        self.settings["active_profile"] = name
        self._save_settings()
        self._update_optics()

    def _collect(self) -> dict:
        return {"camera": self._o("txtCamera").text().strip(),
                "pixel_um": self._o("spnPixel").value(),
                "sensor_w_px": self._o("spnResW").value(),
                "sensor_h_px": self._o("spnResH").value(),
                "sensor_temp": self._o("spnTemp").value(),
                "bayer": self._o("cmbBayer").currentText(),
                "telescope": self._o("txtScope").text().strip(),
                "aperture_mm": self._o("spnAperture").value(),
                "focal_mm": self._o("spnFocal").value(),
                "reducer": self._o("chkReducer").isChecked(),
                "reducer_factor": self._o("spnReducerFactor").value(),
                "mode": self._o("cmbMode").currentText(),
                "light_pollution": self._o("cmbPollution").currentText()}

    def _save_profile(self):
        name, ok = QInputDialog.getText(self, "Enregistrer le profil d'équipement",
                                        "Nom (ex. : « ASI533MC + Newton 200 ») :")
        if not ok:
            return
        try:
            EQ.save_profile(name, self._collect())
        except Exception as e:
            QMessageBox.warning(self, "Profil", str(e))
            return
        self._refresh_profiles()
        self._o("cmbProfile").setCurrentText(name.strip())

    def _delete_profile(self):
        name = self._o("cmbProfile").currentText()
        try:
            EQ.delete_profile(name)
        except Exception as e:
            QMessageBox.warning(self, "Profil", str(e))
            return
        self._refresh_profiles()
        self._o("cmbProfile").setCurrentText("standard")

    # ------------------------------------------------------------------ optique
    def _update_optics(self):
        if self._loading:
            return
        p = self._collect()
        eff = EQ.effective_focal(p)
        scale = EQ.scale_from_focal(p["pixel_um"], eff)
        fov = EQ.field_of_view(p)
        lbl: QLabel = self._o("lblOptics")
        lbl.setText(f"Focale effective : {eff:.0f} mm — échantillonnage "
                    f"{scale:.2f}\"/px — champ {fov['w_deg']:.2f}° × "
                    f"{fov['h_deg']:.2f}° — dénoise conseillé : "
                    f"{EQ.suggestion_denoise(p):.0%}")

    # ------------------------------------------------------------------ réglages
    def _build_perf_group(self):
        """Groupe « Performances & mises à jour » : GPU (CuPy) facultatif,
        adresse du manifeste de mises à jour."""
        from PySide6.QtWidgets import QCheckBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit
        from ..core import gpu
        anchor = self._o("grpAstroCfg")
        lay = anchor.parentWidget().layout() if anchor is not None else None
        grp = QGroupBox("Performances & mises à jour")
        grp.setObjectName("grpPerf")
        row = QHBoxLayout(grp)
        self.chkGpu = QCheckBox("Accélération GPU (CuPy)")
        self.chkGpu.setObjectName("chkGpu")
        self.chkGpu.setChecked(bool(self.settings.get("gpu", False)))
        self.chkGpu.setToolTip(gpu.info())
        if not gpu.available():
            self.chkGpu.setEnabled(False)
        self.txtUpdateUrl = QLineEdit(str(self.settings.get("update_url", "") or ""))
        self.txtUpdateUrl.setObjectName("txtUpdateUrl")
        self.txtUpdateUrl.setPlaceholderText("https://…/latest.json")
        row.addWidget(self.chkGpu)
        row.addWidget(QLabel("Mises à jour :"))
        row.addWidget(self.txtUpdateUrl, 1)
        self.chkGpu.toggled.connect(self._save_settings)
        self.txtUpdateUrl.editingFinished.connect(self._save_settings)
        if lay is not None:
            idx = lay.indexOf(anchor)
            if hasattr(lay, "insertWidget") and idx >= 0:
                lay.insertWidget(idx + 1, grp)
            else:
                lay.addWidget(grp)

    def _save_settings(self):
        # relit le fichier : ne pas écraser les réglages modifiés ailleurs
        # (langue choisie dans le menu, etc.)
        self.settings = EQ.load_settings()
        self.settings["gpu"] = self.chkGpu.isChecked()
        self.settings["update_url"] = self.txtUpdateUrl.text().strip()
        self.settings["use_solve_field"] = self._o("chkUseSolveField").isChecked()
        self.settings["gaia_catalog"] = self._o("txtCatalogDef").text().strip()
        self.settings["active_profile"] = self._o("cmbProfile").currentText()
        EQ.save_settings(self.settings)

    # ------------------------------------------------------------------ Gaia
    def _gen_catalog(self):
        p = self._collect()
        fov = EQ.field_of_view(p)
        radius = max(max(fov["w_deg"], fov["h_deg"]) * 0.75, 0.5)
        ra_s = self._o("txtRaS").text()
        dec_s = self._o("txtDecS").text()
        ra = EQ.parse_coord(ra_s, True) if ra_s.strip() else None
        dec = EQ.parse_coord(dec_s, False) if dec_s.strip() else None
        if ra is None or dec is None:
            QMessageBox.information(self, "Catalogue Gaia",
                                    "Renseignez d'abord RA / Dec du centre (groupe 3).")
            return
        default_dir = os.path.join(os.path.dirname(EQ.EQUIP_DIR), "profiles")
        path, _ = QFileDialog.getSaveFileName(
            self, "Enregistrer le catalogue Gaia",
            os.path.join(default_dir, "gaia_champ.csv"), "Catalogue (*.csv)")
        if not path:
            return
        mag = self._o("spnMagLimit").value()

        def run(progress=None):
            return EQ.make_gaia_catalog(ra, dec, radius, path, mag_limit=mag)
        self._job = Job(run)
        self._job.ok.connect(self._gen_done)
        self._job.err.connect(lambda m: (QMessageBox.critical(
            self, "Catalogue Gaia", m[-1200:]),
            self._o("lblGenResult").setText("Échec : voir message.")))
        self._o("lblGenResult").setText(f"Téléchargement Gaia en cours "
                                        f"(rayon {radius:.2f}°, mag < {mag})…")
        self._job.start()

    def _gen_done(self, path):
        n = sum(1 for _ in open(path, encoding="utf-8")) - 1
        self._o("txtCatalogDef").setText(path)
        self._o("lblGenResult").setText(f"✔ {n} étoiles écrites → {path}")
        self._save_settings()

    # ------------------------------------------------- focale automatique
    def _browse_img(self):
        f, _ = QFileDialog.getOpenFileName(self, "Image à résoudre", "",
                                           "Images (*.fits *.fit *.tif *.tiff *.png *.jpg)")
        if f:
            self._o("txtImgSolve").setText(f)

    def _auto_focal(self):
        img_path = self._o("txtImgSolve").text().strip()
        ra_s = self._o("txtRaS").text()
        dec_s = self._o("txtDecS").text()
        ra = EQ.parse_coord(ra_s, True) if ra_s.strip() else None
        dec = EQ.parse_coord(dec_s, False) if dec_s.strip() else None
        if not img_path:
            QMessageBox.information(self, "Focale",
                                    "Choisissez une image (FITS/TIFF/PNG) de votre matériel.")
            return
        from ..core import astrometry as A
        from ..core.equipment import load_settings
        from ..core.image_io import load_image
        st = load_settings()
        cat = self._o("txtCatalogDef").text().strip() or st.get("gaia_catalog", "")
        self._o("spnPixel").value()

        def run(progress=None):
            img = load_image(img_path)
            res = A.solve_image(img, catalog_path=cat or None, center_ra=ra,
                                center_dec=dec, radius_deg=2.0,
                                prefer_solve_field=bool(st.get("use_solve_field", True)))
            return res
        self._job = Job(run)
        self._job.ok.connect(self._focal_done)
        self._job.err.connect(lambda m: (QMessageBox.critical(self, "Focale", m[-1200:]),
                                         self._o("lblFocalResult").setText("Échec du plate solving.")))
        self._o("lblFocalResult").setText("Plate solving en cours…")
        self._job.start()

    def _focal_done(self, res):
        p = self._collect()
        focal = EQ.focal_from_scale(p["pixel_um"], res["scale_arcsec_pix"])
        self._o("spnFocal").setValue(round(focal, 1))
        self._o("lblFocalResult").setText(
            f"✔ {res['method']} : {res['scale_arcsec_pix']:.3f}\"/px → focale "
            f"{focal:.1f} mm ({res['n_matched']} étoiles, rms {res['rms_arcsec']:.2f}\")")
        self._update_optics()
        self.shared["last_solve"] = res
