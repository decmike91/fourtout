# -*- coding: utf-8 -*-
"""Chargement des fichiers .ui de Qt Designer.

Voie principale : compilation à la volée avec pyside6-uic (mise en cache dans
ui_generated/ ; régénérée si le .ui est plus récent) — robuste et sans surprise.
Les .ui restent entièrement éditables dans Qt Designer.
Repli : QUiLoader si pyside6-uic est absent.
"""
from __future__ import annotations
import importlib.util
import hashlib
import os
import re
import shutil
import subprocess
import sys

from PySide6.QtWidgets import QWidget


from ..paths import DATA_ROOT, resource_dir

UI_DIR = resource_dir("ui")
GEN_DIR = os.path.join(os.path.dirname(UI_DIR), "ui_generated")
PREBUILT_DIR = GEN_DIR      # interfaces pré-générées livrées avec le programme
if not getattr(sys, "frozen", False) and not os.access(os.path.dirname(UI_DIR), os.W_OK):
    # paquet installé dans un dossier non inscriptible : cache utilisateur
    GEN_DIR = os.path.join(DATA_ROOT, "ui_generated")


def _uic_exe():
    if shutil.which("pyside6-uic"):
        return ["pyside6-uic"]
    return [sys.executable, "-m", "PySide6.scripts.pyside_tool", "uic"]


def _generate(filename: str) -> tuple[str, str, str]:
    """Compile ui/<filename> si besoin. Retourne (py_path, classe_ui, classe_racine)."""
    src = os.path.join(UI_DIR, filename)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Fichier .ui introuvable : {src}")
    xml = open(src, "r", encoding="utf-8").read()
    m_cls = re.search(r"<class>(\w+)</class>", xml)
    m_root = re.search(r'<widget class="(\w+)"', xml)
    base = m_cls.group(1) if m_cls else os.path.splitext(filename)[0]
    root = m_root.group(1) if m_root else "QWidget"
    frozen = getattr(sys, "frozen", False)
    digest = hashlib.sha256(xml.encode("utf-8")).hexdigest()
    marker = "# ui-source-sha256: " + digest
    # version pré-générée à jour (installation en lecture seule) ?
    pre = os.path.join(PREBUILT_DIR, f"ui_{base}.py")
    if PREBUILT_DIR != GEN_DIR and os.path.exists(pre):
        with open(pre, encoding="utf-8") as fh:
            if marker in fh.read():
                return pre, base, root
    os.makedirs(GEN_DIR, exist_ok=True)
    py = os.path.join(GEN_DIR, f"ui_{base}.py")
    current = open(py, encoding="utf-8").read() if os.path.exists(py) else ""
    if marker not in current:
        if frozen:
            # Pas de compilation ni d'interface périmée dans un bundle en lecture seule.
            raise RuntimeError("Interface générée périmée ; chargement direct du .ui")
        subprocess.run(_uic_exe() + [src, "-o", py], check=True,
                       capture_output=True, timeout=120)
        with open(py, "a", encoding="utf-8") as output:
            output.write("\n" + marker + "\n")
    return py, base, root


def _import_module(py: str, name: str):
    spec = importlib.util.spec_from_file_location(name, py)
    mod = importlib.util.module_from_spec(spec)
    # Éviter un .pyc périmé lors de deux générations dans la même seconde.
    with open(py, encoding="utf-8") as source:
        exec(compile(source.read(), py, "exec"), mod.__dict__)
    return mod


def load_ui(filename: str, parent: QWidget | None = None) -> QWidget:
    """Charge ui/<filename> et retourne le widget racine peuplé."""
    try:
        py, base, root = _generate(filename)
        mod = _import_module(py, f"ui_{base}")
        ui_cls = getattr(mod, f"Ui_{base}", None)
        if ui_cls is None:
            raise RuntimeError(f"Classe Ui_{base} absente du fichier généré")
        RootCls = getattr(__import__("PySide6.QtWidgets", fromlist=[root]), root)
        widget = RootCls(parent) if parent is not None else RootCls()
        ui_cls().setupUi(widget)
        return widget
    except Exception:
        # repli : QUiLoader direct
        from PySide6.QtCore import QFile, QIODevice
        from PySide6.QtUiTools import QUiLoader
        loader = QUiLoader()
        f = QFile(os.path.join(UI_DIR, filename))
        if not f.open(QIODevice.ReadOnly):
            raise IOError(f"Ouverture impossible : {filename}")
        widget = loader.load(f, parent)
        f.close()
        if widget is None:
            raise ValueError(f"Chargement .ui échoué : {filename}")
        return widget


def w(ui: QWidget, name: str, cls=None):
    """Raccourci : retrouve un enfant par son objectName."""
    obj = ui.findChild(cls, name) if cls else ui.findChild(object, name)
    if obj is None:
        raise RuntimeError(f"Widget « {name} » introuvable dans {type(ui).__name__}")
    return obj
