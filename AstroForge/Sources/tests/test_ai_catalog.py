# -*- coding: utf-8 -*-
"""Catalogue de modèles IA (téléchargement vérifié) et comparaison."""
import hashlib
import json

import numpy as np
import pytest

from astroforge.ai import catalog as CAT


def _catalog(tmp_path, payload=b"modele-onnx-factice", sha=None):
    src = tmp_path / "srv" / "m.onnx"
    src.parent.mkdir()
    src.write_bytes(payload)
    entry = {"id": "denoise-test", "name": "Débruitage test", "task": "denoise",
             "url": src.as_uri(), "sha256": sha or hashlib.sha256(payload).hexdigest(),
             "size": len(payload), "license": "MIT", "input_size": 128}
    cat = tmp_path / "catalog.json"
    cat.write_text(json.dumps({"models": [entry, {"id": "incomplet"}]}), encoding="utf-8")
    return str(cat)


def test_catalog_download_and_checksum(tmp_path):
    models = tmp_path / "models"
    entries = CAT.load_catalog(_catalog(tmp_path))
    assert [e["id"] for e in entries] == ["denoise-test"]          # entrée incomplète ignorée
    path = CAT.download_model(entries[0], str(models))
    cfg = json.loads((models / "denoise-test.json").read_text(encoding="utf-8"))
    assert cfg["task"] == "denoise" and cfg["license"] == "MIT" and CAT.is_installed(entries[0], str(models))
    assert CAT.sha256_file(path) == entries[0]["sha256"]


def test_catalog_rejects_bad_checksum_and_http(tmp_path):
    entries = CAT.load_catalog(_catalog(tmp_path, sha="0" * 64))
    with pytest.raises(ValueError):
        CAT.download_model(entries[0], str(tmp_path / "m2"))
    assert not list((tmp_path / "m2").glob("*.onnx"))               # rien d'installé
    with pytest.raises(ValueError):
        CAT.download_model(dict(entries[0], url="http://exemple.org/x.onnx"), str(tmp_path / "m3"))


def test_ai_compare_dialog_classic(tmp_path, rng):
    from PySide6.QtWidgets import QApplication
    from astroforge.core.image_io import save_image
    from astroforge.gui.tools_extra import AICompareDialog
    app = QApplication.instance() or QApplication([])
    p = tmp_path / "n.tif"
    save_image(str(p), np.clip(0.2 + rng.normal(0, 0.02, (200, 200)), 0, 1).astype(np.float32))
    dlg = AICompareDialog()
    dlg.file.edit.setText(str(p))
    dlg.cmbTask.setCurrentIndex(dlg.cmbTask.findData("denoise"))
    dlg.spnCrop.setValue(128)
    res = dlg.run_sync()
    assert res[0][0] == "Original" and res[1][1] is not None and res[1][1].std() < res[0][1].std()
    dlg.deleteLater()
    app.processEvents()
