#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compile les sources de documentation (docs_src/*.html) en pages d'aide
intégrée (help/) + manifest.json — façon .chm, affichées par le visualiseur
interne (F1). Aucun lecteur PDF requis.

Le visualiseur utilise QTextBrowser (sous-ensemble HTML) : ce compilateur
réécrit donc le contenu avec des styles inline simples (tables bordées,
encadrés colorés, pas de CSS externe ni de JavaScript).

Usage :  python3 tools/build_help.py
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs_src"
OUT = ROOT / "help"

# Mots-clés supplémentaires par page (recherche àla .chm, insensible aux accents)
KEYWORDS = {
    "ch1": ["installation", "python", "pip", "venv", "lancement", "exe", "formats", "bayer", "aperçu live", "interface"],
    "ch2": ["ordre", "workflow", "chaîne", "pipeline", "ordre des opérations"],
    "ch3": ["profil", "équipement", "caméra", "télescope", "bortle", "focale", "échantillonnage", "gaia", "catalogue"],
    "ch4": ["ciel profond", "calibration", "bias", "dark", "flat", "master", "alignement", "drizzle", "empilement", "sigma", "gradient", "déconvolution", "denoise", "défauts", "hdr", "couleur", "scnr"],
    "ch5": ["tuto", "exemple", "m51", "galaxie", "newton"],
    "ch6": ["planétaire", "ser", "avi", "mp4", "classement", "qualité", "dérotation", "ondelettes", "registax", "autostack", "film", "chromatisme", "fausses couleurs"],
    "ch7": ["tuto", "exemple", "jupiter", "barlow", "seeing"],
    "ch8": ["finition", "gimp", "tonalité", "gamma", "recadrage", "masque flou", "export", "fits", "tiff", "png", "jpeg", "résolution"],
    "ch9": ["astrométrie", "plate solving", "wcs", "solve-field", "photométrie", "ouverture", "différentielle", "exoplanète"],
    "ch10": ["batch", "séquence", "json", "script", "python", "automatisation", "nohup"],
    "ch11": ["profils", "dépannage", "problème", "limites", "astuces"],
    "annexe": ["ordre", "checklist", "résumé", "une page"],
}


def slug(title: str, prefix: str, n: int) -> str:
    t = unicodedata.normalize("NFKD", title.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    words = re.findall(r"[a-z0-9]+", t)
    keep = [w for w in words if w not in ("les", "des", "du", "de", "et", "la", "le", "un", "une", "d", "l", "à")][:3]
    return f"{prefix}-{n:02d}-{'-'.join(keep) or 'page'}.html"


def norm_kw(words):
    out = []
    for w in words:
        w2 = unicodedata.normalize("NFKD", w.lower())
        out.append("".join(c for c in w2 if not unicodedata.combining(c)))
    return sorted(set(out))


def transform(s: str) -> str:
    """Réécrit un extrait HTML document → HTML QTextBrowser."""
    # 1) figures/SVG → liste à puces des libellés du schéma (+ légende)
    def svg_repl(m):
        txts = re.findall(r"<text[^>]*>([^<]*)</text>", m.group(0))
        seen, items = set(), []
        for t in txts:
            t = t.strip()
            if t and "·" not in t and len(t) < 46 and t not in seen:
                seen.add(t)
                items.append(t)
        lis = "".join(f"<li>{t}</li>" for t in items) or "<li>schéma du pipeline</li>"
        cap = re.search(r"<figcaption>(.*?)</figcaption>", m.group(0), re.S)
        capt = re.sub(r"<[^>]+>", "", cap.group(1)).strip() if cap else ""
        return f"<ul>{lis}</ul>" + (f"<p><i>{capt}</i></p>" if capt else "")
    s = re.sub(r"<figure[^>]*>.*?</figure>", svg_repl, s, flags=re.S)

    # 2) étapes numérotées → paragraphe avec titre en gras
    def step_repl(m):
        inner = m.group(1)
        inner = re.sub(r'<span class="num">\s*(.*?)\s*</span>', r"\1 ", inner)
        inner = re.sub(r'<span class="t">\s*(.*?)\s*</span>', r"<b>\1</b> ", inner)
        return f"<p>{inner}</p>"
    s = re.sub(r'<div class="step">(.*?)</div>', step_repl, s, flags=re.S)

    # 3) encadrés astuce/alerte → table colorée (marqueur, normalisée plus bas)
    def box_repl(m):
        bg = "#eef4fb" if m.group(1) == "tip" else "#fdf5e8"
        return f"<pbox {bg}>{m.group(2)}</pbox>"
    s = re.sub(r'<div class="(tip|warn)">*(.*?)</div>', box_repl, s, flags=re.S)

    # 4) puces clavier / spans décoratifs
    s = re.sub(r'<span class="kbd">\s*(.*?)\s*</span>', r"<b>[\1]</b>", s, flags=re.S)
    s = re.sub(r"<span[^>]*>", "", s)
    s = s.replace("</span>", "")
    s = re.sub(r"</?div[^>]*>", "", s)

    # 5) tableaux → bordures simples, 1re ligne sur fond gris-bleu
    def table_repl(m):
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(1), flags=re.S)
        out = []
        for i, r in enumerate(rows):
            cells = re.findall(r"<t([hd])[^>]*>(.*?)</t\1>", r, flags=re.S)
            tds = []
            for _h, c in cells:
                if i == 0:
                    tds.append(f'<td bgcolor="#dfe6f2"><b>{c}</b></td>')
                else:
                    tds.append(f"<td>{c}</td>")
            out.append("<tr>" + "".join(tds) + "</tr>")
        return '<table width="100%" border="1" cellspacing="0" cellpadding="4">' + "".join(out) + "</table>"
    s = re.sub(r"<table[^>]*>(.*?)</table>", table_repl, s, flags=re.S)

    # 6) titres descendus d'un niveau + colorés (une seule passe)
    def _head(m):
        lvl = int(m.group(1)) + 1
        col = "#1c3a6e" if lvl == 2 else "#2b4a7e"
        return f'<h{lvl}><font color="{col}">'

    s = re.sub(r"<h([123])[^>]*>", _head, s)
    s = re.sub(r"</h([123])>", lambda m: f"</font></h{int(m.group(1)) + 1}>", s)

    # 7) encadrés → tables colorées finales
    s = re.sub(r'<pbox ([#0-9a-f]+)>(.*?)</pbox>',
               lambda m: (f'<table width="100%" border="0" cellspacing="0" cellpadding="6">'
                          f'<tr><td bgcolor="{m.group(1)}">{m.group(2)}</td></tr></table>'), s, flags=re.S)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def body_of(src: str) -> str:
    b = src.split("<body", 1)[1]
    return b.split(">", 1)[1]


def split_tutorial(src: str):
    body = body_of(src)
    topics = []
    for part in re.split(r'<section id="', body)[1:]:
        sid = part.split('"', 1)[0]
        if sid == "sommaire":
            continue
        chunk = part[: part.rfind("</section>")] if "</section>" in part else part
        chunk = transform(chunk)
        m = re.search(r"<h2[^>]*>(.*?)</h2>", chunk, re.S)  # h1 déjà rebaptisé h2
        title = re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else sid
        topics.append((sid, title, chunk))
    return topics


def split_manual(src: str):
    body = body_of(src)
    i = body.find("<h2")
    parts = re.split(r"(?=<h2)", body[i:]) if i != -1 else []
    topics = []
    for n, part in enumerate(parts, 1):
        m = re.match(r"<h2[^>]*>(.*?)</h2>", part, re.S)
        if not m:
            continue
        title = re.sub(r"<[^>]+>", "", m.group(1)).strip()
        topics.append((f"man{n}", title, transform(part)))
    return topics


def main():
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.html"):
        old.unlink()
    pages = []

    tut = split_tutorial((SRC / "tutoriel.html").read_text(encoding="utf-8"))
    for n, (sid, title, chunk) in enumerate(tut, 1):
        f = f"tuto-{n:02d}.html" if sid != "annexe" else "tuto-annexe.html"
        (OUT / f).write_text(chunk, encoding="utf-8")
        pages.append({"file": f, "title": title, "group": "Tutoriel",
                      "keywords": norm_kw(KEYWORDS.get(sid, []) + title.split())})

    man = split_manual((SRC / "manuel.html").read_text(encoding="utf-8"))
    for n, (sid, title, chunk) in enumerate(man, 1):
        f = slug(title, "man", n)
        (OUT / f).write_text(chunk, encoding="utf-8")
        pages.append({"file": f, "title": title, "group": "Manuel",
                      "keywords": norm_kw(title.split())})

    version = re.search(r'__version__\s*=\s*"([^"]+)"', (ROOT / "astroforge" / "__init__.py").read_text())
    ver = version.group(1) if version else "?"
    accueil = f"""<h2><font color="#1c3a6e">Bienvenue dans l'aide d'AstroForge {ver}</font></h2>
<p><b>Cette aide est intégrée au logiciel</b> (façon .chm) : elle fonctionne hors ligne, sans lecteur PDF. Elle regroupe le <b>tutoriel complet</b> et le <b>manuel utilisateur</b>.</p>
<ul>
<li><b>Sommaire</b> (onglet de gauche) : naviguez dans l'arborescence, cliquez un chapitre pour l'afficher.</li>
<li><b>Recherche</b> : tapez un mot-clé (par ex. <b>drizzle</b>, <b>ondelettes</b>, <b>dark</b>, <b>dérotation</b>) — la recherche est insensible aux accents et à la casse, et cherche dans tout le texte.</li>
<li>Les boutons <b>◀ / ▶</b> reviennent en arrière ou reprennent la lecture, <b>⌂</b> revient à cette page.</li>
</ul>
<p><b>Par où commencer ?</b> Chapitre 2 du tutoriel (« L'ordre des opérations ») explique la logique d'ensemble ; les chapitres 5 et 7 sont deux guides pas à pas (M51 et Jupiter) à suivre la première fois.</p>"""
    (OUT / "accueil.html").write_text(accueil, encoding="utf-8")
    pages.insert(0, {"file": "accueil.html", "title": "Accueil", "group": "Accueil", "keywords": ["aide", "sommaire"]})

    man_path = OUT / "manifest.json"
    man_path.write_text(json.dumps({"pages": pages}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Aide compilée : {len(pages)} pages -> {man_path}")
    for p in pages:
        print(f"  [{p['group']:8s}] {p['title'][:60]}  ({p['file']})")


if __name__ == "__main__":
    main()
