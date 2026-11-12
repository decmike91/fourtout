# -*- coding: utf-8 -*-
"""Fenêtre principale : assemble les 3 modes depuis main_window.ui."""
from __future__ import annotations

from PySide6.QtWidgets import QMessageBox

from .. import __version__ as af_version
from .loader import load_ui

# F1 contextuel : onglet actif → page d'aide (sommaire façon .chm)
HELP_TAB_PAGES = ("man-04-4-mode-ciel.html",        # Ciel profond
                  "man-05-5-mode-planetaire.html",  # Planétaire
                  "man-06-6-traitement-par.html",   # Finition (commune)
                  "tuto-03.html")                   # Configuration


class MainWindow:
    def __init__(self):
        from .. import i18n
        i18n.set_language()
        self.win = load_ui("main_window.ui")
        self.shared = {"image": None}
        from .page_sky import PageSky
        from .page_planetary import PagePlanetary
        from .page_common import PageCommon
        from .page_config import PageConfig
        from PySide6.QtWidgets import QVBoxLayout

        self.page_sky = PageSky(shared=self.shared)
        self.page_planet = PagePlanetary(shared=self.shared)
        self.page_common = PageCommon(shared=self.shared)
        self.page_config = PageConfig(shared=self.shared)
        for name, page in (("skyContainer", self.page_sky),
                           ("planetContainer", self.page_planet),
                           ("commonContainer", self.page_common),
                           ("configContainer", self.page_config)):
            lay = self.win.findChild(QVBoxLayout, name)
            lay.addWidget(page)

        # QStackedWidget utilise sinon la taille minimale de TOUS les onglets,
        # même masqués (Configuration peut empêcher 1024×700 en Planétaire).
        from PySide6.QtWidgets import QTabWidget, QSizePolicy
        tabs = self.win.findChild(QTabWidget, "tabs")
        def sync_tab_sizes(index):
            for i in range(tabs.count()):
                policy = QSizePolicy.Preferred if i == index else QSizePolicy.Ignored
                tabs.widget(i).setSizePolicy(policy, policy)
        tabs.currentChanged.connect(sync_tab_sizes)
        sync_tab_sizes(tabs.currentIndex())

        from PySide6.QtGui import QAction
        act_quit = self.win.findChild(QAction, "actionQuitter")
        act_about = self.win.findChild(QAction, "actionApropos")
        act_help = self.win.findChild(QAction, "actionAide")
        act_ia = self.win.findChild(QAction, "actionIAMisesAJour")
        if act_quit:
            act_quit.triggered.connect(self.win.close)
        if act_about:
            act_about.triggered.connect(self._about)
        if act_help:
            act_help.triggered.connect(self._help)
        if act_ia:
            act_ia.triggered.connect(self._ai_updates)
        from .tools_menu import build_tools_menu
        self.menu_tools = build_tools_menu(self)
        self._build_language_menu()
        self.act_updates = QAction("Rechercher une mise à jour…", self.win)
        self.act_updates.triggered.connect(self._check_updates)
        bar = self.win.menuBar()
        for m in bar.actions():
            if m.menu() is not None and act_help in m.menu().actions():
                m.menu().addAction(self.act_updates)
                bar.removeAction(m)      # le menu Aide reste le dernier
                bar.addAction(m)
                break
        from .. import i18n
        from PySide6.QtWidgets import QApplication
        if i18n.current_language() != "fr":
            i18n.install(QApplication.instance())
            i18n.translate_widget_tree(self.win)

    def _check_updates(self):
        from .. import updates
        url = updates.update_url()
        if not url:
            QMessageBox.information(
                self.win, "Mise à jour",
                "Aucune adresse de mise à jour n'est configurée "
                "(Configuration › Performances & mises à jour).")
            return
        try:
            info = updates.check(url)
        except Exception as e:
            QMessageBox.warning(self.win, "Mise à jour", f"Vérification impossible : {e}")
            return
        updates.mark_checked()
        if info:
            updates._propose(self.win, info)
        else:
            QMessageBox.information(self.win, "Mise à jour",
                                    f"AstroForge {af_version} est à jour.")

    def _build_language_menu(self):
        """Menu « Langue » : enregistre le choix ; effet au redémarrage."""
        from PySide6.QtGui import QAction, QActionGroup
        from .. import i18n
        menu = self.win.menuBar().addMenu("Langue / Language")
        menu.setObjectName("menuLangue")
        group = QActionGroup(menu)
        group.setExclusive(True)
        cur = i18n.current_language()
        for code, name in i18n.available_languages().items():
            act = QAction(name, menu, checkable=True)
            act.setChecked(code == cur)
            act.setData(code)
            group.addAction(act)
            menu.addAction(act)
            act.triggered.connect(lambda _=False, c=code: self._set_language(c))
        self.menu_language = menu

    def _set_language(self, code):
        from .. import i18n
        i18n.save_language(code)
        if code != i18n.current_language():
            QMessageBox.information(
                self.win, "AstroForge",
                "La langue sera appliquée au prochain démarrage.\n"
                "The language will be applied at next start.")

    def _ai_updates(self):
        """Menu « Outils IA » : recherche des mises à jour des modèles IA."""
        from .ai_updates_dialog import AIUpdatesDialog
        if not hasattr(self, "_ai_updates_dlg"):
            self._ai_updates_dlg = AIUpdatesDialog(self.win)
        self._ai_updates_dlg.refresh_states()
        self._ai_updates_dlg.show()
        self._ai_updates_dlg.raise_()
        self._ai_updates_dlg.activateWindow()

    def _help(self):
        """F1 contextuel : ouvre l'aide (.chm) sur la page de l'onglet actif."""
        from PySide6.QtWidgets import QTabWidget
        from .help_viewer import HelpDialog

        if not hasattr(self, "_help_dlg"):
            self._help_dlg = HelpDialog(self.win)
        self._help_dlg.show()
        self._help_dlg.raise_()
        self._help_dlg.activateWindow()
        tabs = self.win.findChild(QTabWidget, "tabs")
        i = tabs.currentIndex() if tabs is not None else -1
        if 0 <= i < len(HELP_TAB_PAGES):
            self._help_dlg.open_by_file(HELP_TAB_PAGES[i])

    def _about(self):
        QMessageBox.about(
            self.win, "À propos d'AstroForge",
            f"AstroForge {af_version} — traitement d'images astronomiques.\n\n"
            "Ciel profond : calibration, alignement, drizzle ×2/×3, empilement, "
            "gradient, IA locale (denoise, fond, netteté, super-résolution, "
            "satellites), HDR, processus automatique, profils.\n\n"
            "Planétaire : SER/AVI/MP4, classement qualité, dérotation, ondelettes "
            "type Registax6, IA, film stabilisé, fausses couleurs solaires.\n\n"
            "Commun : finition type GIMP + augmentation de résolution.\n\n"
            "2.0 : sessions triées, bibliothèque de masters, analyse des brutes, "
            "GHS/arcsinh, masques, bande étroite, multi-nuits, mosaïques, comètes, "
            "empilement en direct, WinJUPOS, outils solaires, annotation, courbes "
            "de lumière, ligne de commande, scripts.\n\nLicence MIT. Catalogue "
            "OpenNGC sous licence CC BY-SA 4.0.")

    def show(self):
        self.win.show()
