#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Construit l'exécutable autonome (PyInstaller) pour la plate-forme courante :
dist/AstroForge/ (Windows, Linux) ou dist/AstroForge.app (macOS).

    python packaging/build.py            # interface graphique
Les installeurs sont ensuite produits par :
    Windows : packaging/windows/build_installer.ps1  (Inno Setup)
    macOS   : packaging/macos/build_dmg.sh           (signature, notarisation, dmg)
    Linux   : packaging/linux/build_appimage.sh      (AppImage)
"""
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEP = ";" if os.name == "nt" else ":"


def main() -> int:
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    # pré-compilation des .ui (le chargeur bascule sinon sur QUiLoader)
    os.makedirs("ui_generated", exist_ok=True)
    try:
        from astroforge.gui.loader import _generate
        for f in sorted(os.listdir("ui")):
            if f.endswith(".ui"):
                _generate(f)
    except Exception as e:  # pyside6-uic absent : QUiLoader dans le bundle
        print(f"(pré-compilation des .ui ignorée : {e})")
    icon = {"win32": "packaging/windows/astroforge.ico"}.get(sys.platform,
                                                             "packaging/astroforge.png")
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed",
           "--name", "AstroForge", "--icon", icon,
           "--add-data", f"ui{SEP}ui",
           "--add-data", f"ui_generated{SEP}ui_generated",
           "--add-data", f"help{SEP}help",
           "--collect-submodules", "astroforge",
           "--collect-data", "astroforge",      # catalogue OpenNGC, traductions
           "--osx-bundle-identifier", "org.astroforge.AstroForge",
           "main.py"]
    print(" ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
