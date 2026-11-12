#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Produit le manifeste de mise à jour (latest.json) lu par AstroForge
(astroforge/updates.py) à partir des installeurs d'une publication :

    python packaging/make_manifest.py <dossier> <version> <url_de_base> > latest.json
"""
import hashlib
import json
import os
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(folder, version, base_url, notes=""):
    page = base_url.replace("/releases/download/", "/releases/tag/")
    m = {"version": version, "url": page, "notes": notes or f"AstroForge {version}"}
    for f in sorted(os.listdir(folder)):
        p = os.path.join(folder, f)
        kind = ("windows" if f.endswith(".exe") else "macos" if f.endswith(".dmg")
                else "linux" if f.endswith(".AppImage") else None)
        if kind:
            m[kind] = {"url": f"{base_url.rstrip('/')}/{f}", "sha256": sha256(p)}
    json.dump(m, sys.stdout, indent=2, ensure_ascii=False)
    print()


if __name__ == "__main__":
    main(*sys.argv[1:])
