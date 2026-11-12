# -*- coding: utf-8 -*-
"""Allège un site-packages destiné à la distribution Windows et garantit des
chemins COURTS (limite historique de 260 caractères de Windows : sans elle,
l'Explorateur refuse d'extraire la version portable — « noms de fichiers trop
longs » — dès que le dossier de destination est un peu profond).

    python prune_site.py <site-packages> [--max-len 100] [--prefix 34]

Retire ce qui ne sert pas à l'exécution d'AstroForge :
- PySide6 : QML / Qt Quick (non utilisés), en-têtes C++, fichiers .obj de
  compilation, stubs .pyi de PySide6, systèmes de types, documentation ;
- suites de tests embarquées dans les paquets (sous-dossiers « tests » /
  « test » à partir du 2e niveau ; le 1er niveau est conservé car certains
  paquets, comme astropy, l'importent au démarrage) ;
- outils de développement d'onnxruntime.
Puis vérifie que plus aucun chemin (relatif au dossier d'installation)
ne dépasse --max-len caractères ; sinon liste les fautifs et échoue.
--prefix : longueur du préfixe « python/Lib/site-packages/ » + racine.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

REMOVE_DIRS = [
    "PySide6/qml", "PySide6/include", "PySide6/typesystems", "PySide6/glue",
    "PySide6/doc", "PySide6/examples", "PySide6/metatypes", "PySide6/Qt/qml",
    "onnxruntime/tools", "onnxruntime/transformers", "onnxruntime/quantization",
    "onnxruntime/datasets", "onnxruntime/backend",
]
REMOVE_EXT = (".obj", ".lib", ".pdb", ".cmake", ".prl")
# stubs .pyi : retirés seulement là où ils ne servent qu'aux éditeurs de code
# (scikit-image, lui, les lit à l'exécution via lazy_loader)
PYI_ONLY_IN = ("PySide6", "shiboken6")
TEST_DIRS = ("tests", "test", "testing_data")
KEEP_EXT_IN = ("numpy/testing",)       # API publique, pas une suite de tests


def prune(sp: str) -> int:
    freed = 0

    def rm_tree(p):
        nonlocal freed
        for root, _d, files in os.walk(p):
            for f in files:
                try:
                    freed += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        shutil.rmtree(p, ignore_errors=True)

    for rel in REMOVE_DIRS:
        p = os.path.join(sp, *rel.split("/"))
        if os.path.isdir(p):
            rm_tree(p)
    for root, dirs, files in os.walk(sp, topdown=True):
        rel = os.path.relpath(root, sp).replace(os.sep, "/")
        depth = 0 if rel == "." else rel.count("/") + 1
        for d in list(dirs):
            if d in TEST_DIRS and depth >= 2 and not rel.endswith(".dist-info"):
                rm_tree(os.path.join(root, d))
                dirs.remove(d)
        for f in files:
            stub = f.endswith(".pyi") and rel.split("/")[0] in PYI_ONLY_IN
            if stub or (f.endswith(REMOVE_EXT) and not any(rel.startswith(k) for k in KEEP_EXT_IN)):
                p = os.path.join(root, f)
                freed += os.path.getsize(p)
                os.remove(p)
    # dossiers vides
    for root, dirs, files in os.walk(sp, topdown=False):
        if root != sp and not os.listdir(root):
            os.rmdir(root)
    return freed


def long_paths(sp: str, max_len: int, prefix: int):
    """Chemins (vus depuis la racine d'installation) plus longs que max_len,
    en comptant les fichiers .pyc que Python créera dans __pycache__."""
    bad = []
    for root, _dirs, files in os.walk(sp):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), sp)
            n = prefix + len(rel)
            if f.endswith(".py") and "__pycache__" not in rel:
                n += len("__pycache__/") + len(".cpython-312.pyc") - len(".py")
            if n > max_len:
                bad.append((n, rel))
    return sorted(bad, reverse=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("site_packages")
    ap.add_argument("--max-len", type=int, default=100)
    ap.add_argument("--prefix", type=int, default=len("python/Lib/site-packages/"))
    ap.add_argument("--no-prune", action="store_true")
    a = ap.parse_args(argv)
    if not a.no_prune:
        freed = prune(a.site_packages)
        print(f"Allègement : {freed / 1e6:.0f} Mo retirés")
    bad = long_paths(a.site_packages, a.max_len, a.prefix)
    if bad:
        print(f"{len(bad)} chemin(s) de plus de {a.max_len} caractères :", file=sys.stderr)
        for n, rel in bad[:40]:
            print(f"  {n:4d}  {rel}", file=sys.stderr)
        return 1
    print(f"Chemins : tous ≤ {a.max_len} caractères depuis le dossier d'installation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
