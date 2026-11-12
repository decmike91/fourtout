# -*- coding: utf-8 -*-
"""Aide intégrée façon .chm : sommaire arborescent, index de mots-clés et
recherche plein texte (insensible aux accents et à la casse), navigation
précédent/suivant. Pages compilées dans help/ par tools/build_help.py."""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton, QSplitter, QTabWidget,
                               QTextBrowser, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                               QWidget)


def help_dir() -> Path:
    from ..paths import resource_dir
    return Path(resource_dir("help"))


def load_pages() -> list[dict]:
    """Charge manifest.json + le texte de chaque page (chemins robustes exe/venv)."""
    d = help_dir()
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    pages = []
    for it in manifest["pages"]:
        p = d / it["file"]
        if p.exists():
            pages.append({"title": it["title"], "group": it.get("group", ""),
                          "file": p, "html": p.read_text(encoding="utf-8"),
                          "keywords": it.get("keywords", [])})
    return pages


def normalize(s: str) -> str:
    """Minuscules + suppression des accents (recherche façon .chm)."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.casefold()


def strip_tags(html: str) -> str:
    html = re.sub(r"<[^>]+>", " ", html)
    for a, b in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&#8217;", "'")):
        html = html.replace(a, b)
    return re.sub(r"\s+", " ", html).strip()


def build_index(pages: list[dict]) -> list[dict]:
    """Index plein texte : texte normalisé + fréquences de mots par page."""
    idx = []
    for p in pages:
        ntext = normalize(strip_tags(p["html"]))
        words = Counter(re.findall(r"[a-z0-9]+", ntext))
        idx.append({"page": p, "ntext": ntext, "words": words})
    return idx


_STOP = {"des", "une", "aux", "dans", "son", "ses", "par", "pour", "avec", "les",
         "que", "qui", "cette", "tout", "tous", "sur", "the", "and"}


def build_keywords(pages: list[dict]) -> list[dict]:
    """Index A→Z façon .chm : mots-clés du manifest (nettoyés : pas de nombres,
    stopwords ni fragments « d' »/« l' ») → pages, tri sans accents ; le clic sur
    un mot-clé ouvre la page la plus pertinente."""
    kw: dict[str, dict] = {}
    for i, p in enumerate(pages):
        for raw in p.get("keywords", []):
            lab = raw.strip(" ():,;!?«»—-·.")
            key = re.sub(r"^(d'|l')", "", normalize(lab))
            lab = re.sub(r"^(d'|l')", "", lab, flags=re.IGNORECASE)
            if len(key) < 3 or not re.search(r"[a-z]{2}", key) or key in _STOP:
                continue
            e = kw.setdefault(key, {"key": key, "label": lab, "pages": []})
            if i not in e["pages"]:
                e["pages"].append(i)
            if lab.istitle() and not e["label"].istitle():
                e["label"] = lab
    return sorted(kw.values(), key=lambda k: k["key"])


def search(index: list[dict], query: str) -> list[dict]:
    """Classe les pages : tous les mots doivent être présents, score = fréquences
    (mots du titre/mots-clés pondérés ×5). Renvoie [{page, count, score, snippet}]."""
    tokens = re.findall(r"[a-z0-9]+", normalize(query))
    if not tokens:
        return []
    results = []
    for e in index:
        w = e["words"]
        if not all(t in w for t in tokens):
            continue
        body = sum(w[t] for t in tokens)
        title_k = normalize(e["page"]["title"] + " " + " ".join(e["page"]["keywords"]))
        tw = Counter(re.findall(r"[a-z0-9]+", title_k))
        bonus = sum(tw[t] for t in tokens) * 5
        pos = min((e["ntext"].find(t) for t in tokens if e["ntext"].find(t) != -1), default=0)
        a = max(0, pos - 60)
        snippet = ("…" if a else "") + e["ntext"][a:pos + 140].strip() + "…"
        results.append({"page": e["page"], "count": body, "score": body + bonus,
                        "snippet": snippet, "token": tokens[0]})
    results.sort(key=lambda r: -r["score"])
    return results


class HelpDialog(QDialog):
    """Fenêtre d'aide : onglets Sommaire / Recherche + navigateur de pages."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Aide AstroForge — tutoriel et manuel")
        self.resize(1150, 720)
        self.pages = load_pages()
        self.index = build_index(self.pages)
        self._hist: list[tuple[dict, str | None]] = []
        self._pos = -1
        self._moving = False

        # -- panneau gauche : Sommaire / Recherche --------------------------------
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Sommaire")
        groups: dict[str, QTreeWidgetItem] = {}
        for i, p in enumerate(self.pages):
            if p["group"] not in groups:
                it = QTreeWidgetItem([p["group"]])
                f = it.font(0)
                f.setBold(True)
                it.setFont(0, f)
                self.tree.addTopLevelItem(it)
                groups[p["group"]] = it
            child = QTreeWidgetItem([p["title"]])
            child.setData(0, Qt.UserRole, i)
            groups[p["group"]].addChild(child)
        self.tree.expandAll()
        self.tree.currentItemChanged.connect(self._on_tree)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Rechercher un mot-clé (ex. : drizzle, dark, dérotation)…")
        self.search_edit.setClearButtonEnabled(True)
        self.results = QListWidget()
        self.results.setWordWrap(True)
        self.results.itemClicked.connect(self._on_result)
        sw = QWidget()
        sv = QVBoxLayout(sw)
        sv.setContentsMargins(0, 4, 0, 0)
        sv.addWidget(QLabel("Recherche plein texte (sans accents) :"))
        sv.addWidget(self.search_edit)
        self.hits_label = QLabel("")
        sv.addWidget(self.hits_label)
        sv.addWidget(self.results)
        self.search_edit.textChanged.connect(self._on_search)
        QShortcut(QKeySequence("Ctrl+F"), self, self.search_edit.setFocus)

        # -- onglet Index A→Z (façon .chm) ----------------------------------------
        self.kw_filter = QLineEdit()
        self.kw_filter.setPlaceholderText("Filtrer les mots-clés (A→Z)…")
        self.kw_filter.setClearButtonEnabled(True)
        self.kw_list = QListWidget()
        iw = QWidget()
        iv = QVBoxLayout(iw)
        iv.setContentsMargins(0, 4, 0, 0)
        iv.addWidget(QLabel("Index de mots-clés — clic = ouvrir la page :"))
        iv.addWidget(self.kw_filter)
        iv.addWidget(self.kw_list)
        self.kws = build_keywords(self.pages)
        self._fill_kws("")
        self.kw_filter.textChanged.connect(self._fill_kws)
        self.kw_list.itemClicked.connect(self._on_keyword)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.tree, "Sommaire")
        self.tabs.addTab(iw, "Index")
        self.tabs.addTab(sw, "Recherche")

        # -- navigateur -----------------------------------------------------------
        self.browser = QTextBrowser()
        self.browser.setOpenLinks(True)

        bar = QHBoxLayout()
        b_back = QPushButton("◀ Précédent")
        b_fwd = QPushButton("Suivant ▶")
        b_home = QPushButton("⌂ Accueil")
        b_back.clicked.connect(lambda: self._move(-1))
        b_fwd.clicked.connect(lambda: self._move(1))
        b_home.clicked.connect(self._home)
        self._nav = (b_back, b_fwd)
        bar.addWidget(b_back)
        bar.addWidget(b_fwd)
        bar.addWidget(b_home)
        bar.addStretch(1)
        self.status = QLabel("F1 ou menu Aide — Ctrl+F pour chercher")
        bar.addWidget(self.status)

        split = QSplitter()
        split.addWidget(self.tabs)
        split.addWidget(self.browser)
        split.setSizes([330, 820])

        root = QVBoxLayout(self)
        root.addLayout(bar)
        root.addWidget(split)

        if self.pages:
            self._show(self.pages[0], None)

    # -- navigation ------------------------------------------------------------
    def _show(self, page: dict, anchor: str | None):
        self._moving = True
        if anchor:
            html = page["html"] + f'<p><a name="{anchor}"></a></p>'
            self.browser.setHtml(html)
            cur = self.browser.document().find(page.get("_token", ""), self.browser.textCursor())
            if cur:
                self.browser.setTextCursor(cur)
        else:
            self.browser.setHtml(page["html"])
        self._moving = False
        self._hist = self._hist[: self._pos + 1]
        self._hist.append((page, anchor))
        self._pos = len(self._hist) - 1
        self._nav[0].setEnabled(self._pos > 0)
        self._nav[1].setEnabled(False)

    def open_page(self, page: dict, anchor: str | None = None):
        self._show(page, anchor)
        self.browser.verticalScrollBar().setValue(0)

    def open_first(self):
        if self.pages:
            self.open_page(self.pages[0], None)

    def _move(self, delta: int):
        new = self._pos + delta
        if 0 <= new < len(self._hist):
            self._pos = new
            page, anchor = self._hist[new]
            self._moving = True
            self.browser.setHtml(page["html"] + (f'<p><a name="{anchor}"></a></p>' if anchor else ""))
            self._moving = False
            self._nav[0].setEnabled(self._pos > 0)
            self._nav[1].setEnabled(self._pos < len(self._hist) - 1)

    def _home(self):
        if self.pages:
            self.open_page(self.pages[0], None)

    def _on_tree(self, cur, _prev=None):
        if cur is None or self._moving:
            return
        i = cur.data(0, Qt.UserRole)
        if i is not None:
            self.open_page(self.pages[i], None)

    # -- index A→Z ---------------------------------------------------------------
    def _fill_kws(self, text: str):
        self.kw_list.clear()
        n = normalize(text.strip())
        for k in self.kws:
            if n and n not in k["key"]:
                continue
            np = len(k["pages"])
            it = QListWidgetItem(f"{k['label']}  ({np} page{'s' if np > 1 else ''})")
            it.setData(Qt.UserRole, k)
            self.kw_list.addItem(it)

    def _on_keyword(self, item):
        k = item.data(Qt.UserRole)
        if not k:
            return
        res = search(self.index, k["label"])
        if res:
            i = self.pages.index(res[0]["page"])
            self._select_tree(i)
            self.open_page(self.pages[i], None)
            self.status.setText(f"Index : « {k['label']} » — {len(k['pages'])} page(s)")

    def _select_tree(self, i: int):
        """Sélectionne la page i dans le sommaire (sans re-déclencher _on_tree)."""
        self._moving = True
        for g in range(self.tree.topLevelItemCount()):
            grp = self.tree.topLevelItem(g)
            for c in range(grp.childCount()):
                if grp.child(c).data(0, Qt.UserRole) == i:
                    self.tree.setCurrentItem(grp.child(c))
                    break
        self._moving = False
        self.tabs.setCurrentIndex(0)

    def open_by_file(self, fname: str) -> bool:
        """F1 contextuel : ouvre la page dont le fichier est fname + sommaire."""
        for i, p in enumerate(self.pages):
            if p["file"].name == fname:
                self._select_tree(i)
                self.open_page(self.pages[i], None)
                return True
        return False

    # -- recherche ---------------------------------------------------------------
    def _on_search(self, text: str):
        self.results.clear()
        if len(text.strip()) < 2:
            self.hits_label.setText("")
            return
        res = search(self.index, text)
        self.hits_label.setText(f"{len(res)} page(s) correspondante(s)" if res else "Aucun résultat")
        for r in res[:40]:
            it = QListWidgetItem(f"{r['page']['title']}\n   {r['count']} occurrence(s) — {r['snippet'][:120]}")
            it.setToolTip(r["snippet"])
            it.setData(Qt.UserRole, (self.pages.index(r["page"]), r["token"]))
            self.results.addItem(it)
        if res:
            self.tabs.setCurrentIndex(2)

    def _on_result(self, item: QListWidgetItem):
        data = item.data(Qt.UserRole)
        if not data:
            return
        i, token = data
        page = dict(self.pages[i])
        page["_token"] = token
        self._show(page, "hl")
