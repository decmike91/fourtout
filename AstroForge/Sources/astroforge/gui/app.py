# -*- coding: utf-8 -*-
"""Lancement de l'interface graphique (``astroforge-gui`` / ``python main.py``)."""
from __future__ import annotations

import sys


def _open_files(win, paths):
    """Fichiers passés en argument (double-clic sur un .fits/.xisf/.tif…,
    « Ouvrir avec ») : la première image lisible est ouverte dans l'onglet
    Finition."""
    import os
    for p in paths:
        if not os.path.isfile(p):
            continue
        try:
            from ..core.image_io import load_image
            from .tools_menu import send_to_finishing
            send_to_finishing(win, load_image(p))
            return p
        except Exception as e:  # format non pris en charge : on continue
            print(f"Ouverture impossible : {p} ({e})", file=sys.stderr)
    return None


def main(argv=None) -> int:
    from PySide6.QtWidgets import QApplication
    argv = list(sys.argv if argv is None else argv)
    app = QApplication.instance() or QApplication(argv)
    app.setApplicationName("AstroForge")
    from .main_window import MainWindow
    win = MainWindow()
    if "--selftest" in argv:  # vérification sans boucle d'événements
        print("SELFTEST OK — fenêtre principale construite")
        return 0
    win.show()
    _open_files(win, [a for a in argv[1:] if not a.startswith("-")])
    if "--no-update-check" not in argv:
        try:
            from ..updates import check_in_background
            check_in_background(win.win)
        except Exception:
            pass
    code = app.exec()
    # destruction ordonnée : fenêtres puis application (évite un plantage
    # à la fermeture dû à l'ordre de destruction des objets Qt)
    for w in app.topLevelWidgets():
        w.deleteLater()
    from PySide6.QtCore import QCoreApplication, QEvent
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    del win
    return code


if __name__ == "__main__":
    sys.exit(main())
