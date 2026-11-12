# -*- coding: utf-8 -*-
"""
Catalogue de modèles ONNX téléchargeables.

Un catalogue est un JSON (fichier local ou URL https) :
{"models": [{"id": "…", "name": "…", "task": "denoise|sharpen|superres|background|
  satellite|starless", "url": "https://…/modele.onnx", "sha256": "…",
  "size": 12345678, "license": "…", "description": "…",
  "input_size": 256, "overlap": 32, "scale": 1, "norm": {"scale": 1.0, "offset": 0.0}}]}

Aucun modèle n'est livré ni référencé par défaut (licences propres à chaque
modèle) : l'utilisateur indique l'URL d'un catalogue de confiance dans
« ai_tools.json » (clé « model_catalog_url ») ou dans le dialogue. Chaque
téléchargement est vérifié par somme SHA-256 avant installation dans le
dossier des modèles de l'utilisateur, avec les métadonnées de tâche.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request

from .onnx_runner import MODELS_DIR

REQUIRED = ("id", "name", "task", "url", "sha256")


def load_catalog(source: str, timeout: float = 20.0) -> list[dict]:
    """Charge un catalogue (chemin local, file:// ou https://)."""
    if not source:
        return []
    if os.path.exists(source):
        with open(source, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        if not source.startswith(("https://", "file://")):
            raise ValueError("Catalogue : URL https:// (ou fichier local) requise")
        req = urllib.request.Request(source, headers={"User-Agent": "AstroForge"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    models = data.get("models", []) if isinstance(data, dict) else []
    out = []
    for m in models:
        if isinstance(m, dict) and all(m.get(k) for k in REQUIRED):
            out.append(m)
    return out


def _target(entry: dict, models_dir: str | None = None) -> str:
    name = "".join(c for c in entry["id"] if c.isalnum() or c in "-_.") or "modele"
    return os.path.join(models_dir or MODELS_DIR, name + ".onnx")


def is_installed(entry: dict, models_dir: str | None = None) -> bool:
    return os.path.exists(_target(entry, models_dir))


def download_model(entry: dict, models_dir: str | None = None, progress=None,
                   timeout: float = 60.0) -> str:
    """Télécharge, vérifie (SHA-256) et installe un modèle + son JSON de
    configuration. Retourne le chemin du .onnx installé."""
    dest_dir = models_dir or MODELS_DIR
    os.makedirs(dest_dir, exist_ok=True)
    url = entry["url"]
    if not url.startswith(("https://", "file://")):
        raise ValueError("Téléchargement refusé : URL non https")
    target = _target(entry, dest_dir)
    h = hashlib.sha256()
    fd, tmp = tempfile.mkstemp(prefix=".dl_", suffix=".onnx", dir=dest_dir)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AstroForge"})
        with urllib.request.urlopen(req, timeout=timeout) as r, os.fdopen(fd, "wb") as out:
            total = int(r.headers.get("Content-Length") or entry.get("size") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                h.update(chunk)
                done += len(chunk)
                if progress:
                    progress(done, max(total, done), f"Téléchargement de {entry['name']}")
        if h.hexdigest().lower() != str(entry["sha256"]).lower():
            raise ValueError("Somme de contrôle SHA-256 incorrecte : fichier rejeté")
        os.replace(tmp, target)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    cfg = {k: entry[k] for k in ("task", "input_size", "overlap", "scale", "norm", "input_name")
           if k in entry}
    cfg.update(source=url, sha256=entry["sha256"], license=entry.get("license", ""),
               name=entry["name"])
    with open(target[:-5] + ".json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return target


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
