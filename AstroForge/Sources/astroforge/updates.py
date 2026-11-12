# -*- coding: utf-8 -*-
"""Vérification des mises à jour.

Le programme interroge (au démarrage, en arrière-plan, au plus une fois par
jour) un fichier JSON publié avec les versions :

    {"version": "2.0.1",
     "url": "https://…/AstroForge-2.0.1-setup.exe",      # page ou installeur
     "sha256": "…",                                       # facultatif
     "notes": "Corrections …"}

Des entrées par plate-forme sont possibles : {"version": …, "windows": {"url":
…, "sha256": …}, "macos": {…}, "linux": {…}}.

L'adresse se règle dans settings.json (``"update_url"``) ou par la variable
``ASTROFORGE_UPDATE_URL`` ; vide = aucune vérification (défaut : aucune
adresse n'est imposée tant que le projet n'a pas de serveur de publication).
Rien n'est installé sans l'accord de l'utilisateur : le programme propose
d'ouvrir la page / de télécharger l'installeur, dont l'empreinte SHA-256 est
contrôlée lorsqu'elle est fournie.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request

from . import __version__

CHECK_INTERVAL_S = 24 * 3600


def parse_version(v: str) -> tuple:
    """'2.0.10' → (2, 0, 10) ; suffixes (rc, b…) ignorés pour la comparaison
    mais une pré-version est considérée antérieure à la version finale."""
    m = re.match(r"\s*v?(\d+(?:\.\d+)*)(.*)", str(v))
    if not m:
        return (0,)
    nums = tuple(int(x) for x in m.group(1).split("."))
    nums = nums + (0,) * (3 - len(nums))
    return nums + ((0,) if m.group(2).strip() else (1,))


def is_newer(remote: str, local: str = __version__) -> bool:
    return parse_version(remote) > parse_version(local)


def platform_key() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def update_url() -> str:
    env = os.environ.get("ASTROFORGE_UPDATE_URL")
    if env is not None:
        return env.strip()
    try:
        from .core.equipment import load_settings
        return str(load_settings().get("update_url", "") or "").strip()
    except Exception:
        return ""


def fetch_manifest(url: str, timeout: float = 8.0) -> dict:
    if not url.lower().startswith(("https://", "file:")):
        raise ValueError("Adresse de mise à jour : https:// (ou file:) requis")
    req = urllib.request.Request(url, headers={"User-Agent": f"AstroForge/{__version__}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 (schéma contrôlé)
        data = json.loads(r.read(1_000_000).decode("utf-8"))
    if not isinstance(data, dict) or "version" not in data:
        raise ValueError("Manifeste de mise à jour invalide")
    return data


def check(url: str | None = None, local: str = __version__) -> dict | None:
    """Retourne {'version', 'url', 'sha256', 'notes'} si une version plus
    récente existe, sinon None."""
    url = update_url() if url is None else url
    if not url:
        return None
    m = fetch_manifest(url)
    if not is_newer(str(m["version"]), local):
        return None
    plat = m.get(platform_key()) if isinstance(m.get(platform_key()), dict) else {}
    return {"version": str(m["version"]),
            "url": plat.get("url", m.get("url", "")),
            "sha256": plat.get("sha256", m.get("sha256", "")),
            "notes": str(m.get("notes", ""))}


def _stamp_path() -> str:
    from .paths import DATA_ROOT
    return os.path.join(DATA_ROOT, "profiles", "last_update_check")


def due(now: float | None = None) -> bool:
    try:
        with open(_stamp_path(), encoding="utf-8") as f:
            last = float(f.read().strip() or 0)
    except (OSError, ValueError):
        last = 0.0
    return (now or time.time()) - last >= CHECK_INTERVAL_S


def mark_checked():
    try:
        os.makedirs(os.path.dirname(_stamp_path()), exist_ok=True)
        with open(_stamp_path(), "w", encoding="utf-8") as f:
            f.write(str(time.time()))
    except OSError:
        pass


def download(info: dict, dest_dir: str, progress=None) -> str:
    """Télécharge l'installeur proposé et vérifie son SHA-256 s'il est fourni."""
    from .ai.catalog import sha256_file
    url = info["url"]
    if not url.lower().startswith("https://"):
        raise ValueError("Téléchargement : https:// requis")
    os.makedirs(dest_dir, exist_ok=True)
    name = os.path.basename(url.split("?")[0]) or f"AstroForge-{info['version']}"
    path = os.path.join(dest_dir, name)
    req = urllib.request.Request(url, headers={"User-Agent": f"AstroForge/{__version__}"})
    with urllib.request.urlopen(req, timeout=30) as r, open(path + ".part", "wb") as f:  # noqa: S310
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress:
                progress(done, max(total, done), "Téléchargement")
    if info.get("sha256") and sha256_file(path + ".part").lower() != info["sha256"].lower():
        os.remove(path + ".part")
        raise ValueError("Empreinte SHA-256 incorrecte : fichier supprimé")
    os.replace(path + ".part", path)
    return path


def check_in_background(parent=None, force: bool = False):
    """Lance la vérification dans un thread ; propose la mise à jour si une
    version plus récente est publiée. Silencieux en cas d'erreur réseau."""
    url = update_url()
    if not url or (not force and not due()):
        return None
    from .gui.workers import Job

    def run(progress=None):
        return check(url)

    job = Job(run, parent=parent)

    def on_ok(info):
        mark_checked()
        if info:
            _propose(parent, info)
    job.ok.connect(on_ok)
    job.start()
    if parent is not None:
        parent._update_job = job  # garde une référence
    return job


def _propose(parent, info: dict):
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices
    from PySide6.QtWidgets import QMessageBox
    msg = (f"AstroForge {info['version']} est disponible (version installée : "
           f"{__version__}).\n\n{info['notes']}\n\nOuvrir la page de téléchargement ?")
    if QMessageBox.question(parent, "Mise à jour", msg) == QMessageBox.Yes and info["url"]:
        QDesktopServices.openUrl(QUrl(info["url"]))
