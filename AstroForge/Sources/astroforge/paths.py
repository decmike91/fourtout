# -*- coding: utf-8 -*-
"""Emplacements des données de l'application.

Les réglages, profils et modèles IA de l'utilisateur ne doivent pas être
écrits dans le dossier d'installation lorsqu'il n'est pas modifiable
(Program Files, /Applications, exécutable PyInstaller dont les fichiers sont
extraits dans un dossier temporaire). Règle :
- variable d'environnement ASTROFORGE_HOME si définie ;
- sinon le dossier du programme s'il est inscriptible et que l'on n'est pas
  dans un exécutable figé (mode « portable », comportement historique) ;
- sinon le dossier de configuration de l'utilisateur :
  Windows %APPDATA%\\AstroForge, macOS ~/Library/Application Support/AstroForge,
  Linux $XDG_CONFIG_HOME/astroforge (~/.config/astroforge).
"""
from __future__ import annotations

import os
import sys

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# arborescence source (dossier contenant main.py, ui/, help/) — sinon paquet
# installé par pip (site-packages) : ressources dans astroforge/_res/
SOURCE_TREE = os.path.exists(os.path.join(APP_ROOT, "main.py")) and \
    os.path.isdir(os.path.join(APP_ROOT, "ui"))


def resource_dir(name: str) -> str:
    """Dossier de ressources (ui, help) : exécutable PyInstaller, arborescence
    source, ou paquet installé (astroforge/_res/<name>)."""
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
        return os.path.join(base, name)
    src = os.path.join(APP_ROOT, name)
    if os.path.isdir(src):
        return src
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "_res", name)


def _writable(path: str) -> bool:
    probe = path
    while probe and not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return bool(probe) and os.access(probe, os.W_OK)


def _platform_dir() -> str:
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.expanduser("~\\AppData\\Roaming")
        return os.path.join(base, "AstroForge")
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Application Support/AstroForge")
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "astroforge")


def data_root() -> str:
    env = os.environ.get("ASTROFORGE_HOME")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    if (not getattr(sys, "frozen", False) and SOURCE_TREE
            and _writable(os.path.join(APP_ROOT, "profiles"))):
        return APP_ROOT
    return _platform_dir()


DATA_ROOT = data_root()
PROFILES_ROOT = os.path.join(DATA_ROOT, "profiles")
USER_MODELS_DIR = os.path.join(DATA_ROOT, "models", "ai")
APP_MODELS_DIR = os.path.join(APP_ROOT, "models", "ai")
