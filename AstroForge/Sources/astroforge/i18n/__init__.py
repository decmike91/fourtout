# -*- coding: utf-8 -*-
"""Traduction de l'interface.

L'interface est écrite en français ; une langue cible est un fichier JSON
``<code>.json`` de ce dossier : ``{"language": "English", "strings": {fr: en}}``.
La traduction est appliquée à l'exécution sur l'arbre des widgets (libellés,
boutons, cases, titres de groupes, onglets, menus, actions, infobulles,
titres de fenêtres, textes indicatifs).

Les éléments des listes déroulantes (QComboBox) ne sont PAS traduits : le
programme utilise leur texte comme valeur (``currentText()``) — les traduire
changerait le comportement. C'est une limite connue et documentée.

Langue choisie : réglage ``"language"`` de settings.json (``"fr"`` par défaut)
ou variable d'environnement ``ASTROFORGE_LANG`` ; changement pris en compte au
redémarrage.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_TABLE: dict[str, str] = {}
_LANG = "fr"


def available_languages() -> dict[str, str]:
    """{code: nom affiché} — le français (langue source) est toujours présent."""
    langs = {"fr": "Français"}
    for f in sorted(os.listdir(HERE)):
        if f.endswith(".json"):
            try:
                with open(os.path.join(HERE, f), encoding="utf-8") as fh:
                    langs[f[:-5]] = json.load(fh).get("language", f[:-5])
            except (OSError, ValueError):
                pass
    return langs


def load_table(code: str) -> dict[str, str]:
    if code == "fr":
        return {}
    path = os.path.join(HERE, f"{code}.json")
    with open(path, encoding="utf-8") as fh:
        return dict(json.load(fh).get("strings", {}))


def configured_language() -> str:
    env = os.environ.get("ASTROFORGE_LANG")
    if env:
        return env
    try:
        from ..core.equipment import load_settings
        return str(load_settings().get("language", "fr") or "fr")
    except Exception:
        return "fr"


def set_language(code: str | None = None) -> str:
    """Active une langue (None : celle des réglages). Retourne le code actif."""
    global _TABLE, _LANG
    code = code or configured_language()
    try:
        _TABLE = load_table(code)
        _LANG = code
    except (OSError, ValueError):
        _TABLE, _LANG = {}, "fr"
    return _LANG


def current_language() -> str:
    return _LANG


def save_language(code: str) -> None:
    from ..core.equipment import load_settings, save_settings
    s = load_settings()
    s["language"] = code
    save_settings(s)


def tr(text: str) -> str:
    """Traduit une chaîne (identique si absente de la table). Les deux-points,
    points de suspension et espaces de fin sont conservés."""
    if not _TABLE or not text:
        return text
    if text in _TABLE:
        return _TABLE[text]
    core = text.rstrip(" :…").rstrip()
    if core and core != text and core in _TABLE:
        return _TABLE[core] + text[len(core):]
    s = text.strip()
    if s != text and s in _TABLE:
        return text.replace(s, _TABLE[s])
    return text


def translate_widget_tree(root) -> int:
    """Traduit root et tous ses descendants ; retourne le nombre de textes
    modifiés. Les éléments de QComboBox ne sont jamais touchés."""
    if not _TABLE or root is None:
        return 0
    from PySide6.QtWidgets import (QAbstractButton, QGroupBox, QLabel, QLineEdit,
                                   QMenu, QPlainTextEdit, QTabWidget, QWidget,
                                   QAbstractSpinBox)
    from PySide6.QtGui import QAction
    n = 0

    def setter(get, put):
        nonlocal n
        old = get()
        new = tr(old)
        if new != old:
            put(new)
            n += 1

    widgets = [root] + root.findChildren(QWidget)
    for w in widgets:
        if isinstance(w, QLabel):
            setter(w.text, w.setText)
        elif isinstance(w, QAbstractButton):
            setter(w.text, w.setText)
        elif isinstance(w, QGroupBox):
            setter(w.title, w.setTitle)
        elif isinstance(w, QMenu):
            setter(w.title, w.setTitle)
        elif isinstance(w, (QLineEdit, QPlainTextEdit)):
            setter(w.placeholderText, w.setPlaceholderText)
        elif isinstance(w, QAbstractSpinBox):
            if hasattr(w, "suffix"):
                setter(w.suffix, w.setSuffix)
            if hasattr(w, "prefix"):
                setter(w.prefix, w.setPrefix)
        if isinstance(w, QTabWidget):
            for i in range(w.count()):
                setter(lambda i=i: w.tabText(i), lambda t, i=i: w.setTabText(i, t))
        if w.toolTip():
            setter(w.toolTip, w.setToolTip)
        if w.isWindow() or w is root:
            setter(w.windowTitle, w.setWindowTitle)
    for a in root.findChildren(QAction):
        setter(a.text, a.setText)
        if a.toolTip() and a.toolTip() != a.text():
            setter(a.toolTip, a.setToolTip)
    return n


_FILTER = None


def install(app) -> bool:
    """Installe un filtre d'événements : chaque fenêtre (principale, outils,
    dialogues) est traduite au moment où elle s'affiche. Sans effet en
    français."""
    global _FILTER
    if not _TABLE or app is None or _FILTER is not None:
        return False
    from PySide6.QtCore import QEvent, QObject
    from PySide6.QtWidgets import QWidget

    class _Translator(QObject):
        def eventFilter(self, obj, ev):
            if ev.type() == QEvent.Show and isinstance(obj, QWidget) and obj.isWindow():
                try:
                    translate_widget_tree(obj)
                except Exception:
                    pass
            return False

    _FILTER = _Translator(app)
    app.installEventFilter(_FILTER)
    return True


set_language("fr")
