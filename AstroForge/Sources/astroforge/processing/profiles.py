# -*- coding: utf-8 -*-
"""
Profils de paramètres : chaque algorithme/étape possède ses profils enregistrés
(JSON dans profiles/), avec un profil « standard » = valeurs par défaut.
"""
from __future__ import annotations
import json
import os

from ..paths import PROFILES_ROOT

PROFILES_DIR = PROFILES_ROOT


def _dir_for(step: str) -> str:
    d = os.path.join(PROFILES_DIR, step)
    os.makedirs(d, exist_ok=True)
    return d


def defaults_for(step: str, registry: dict) -> dict:
    return dict(registry[step]["params"]) if step in registry else {}


def list_profiles(step: str) -> list[str]:
    d = _dir_for(step)
    names = [f[:-5] for f in os.listdir(d) if f.endswith(".json")]
    if "standard" not in names:
        return ["standard"] + names
    return sorted(names, key=lambda n: (n != "standard", n))


def load_profile(step: str, name: str, registry: dict) -> dict:
    if name == "standard":
        return defaults_for(step, registry)
    path = os.path.join(_dir_for(step), name + ".json")
    base = defaults_for(step, registry)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                base.update(json.load(f))
        except Exception:
            pass
    return base


def save_profile(step: str, name: str, params: dict) -> None:
    if not name or not name.strip():
        raise ValueError("Nom de profil vide")
    name = name.strip()
    if name == "standard":
        raise ValueError("Le profil « standard » est protégé — enregistrez sous un autre nom")
    path = os.path.join(_dir_for(step), name + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)


def delete_profile(step: str, name: str) -> None:
    path = os.path.join(_dir_for(step), name + ".json")
    if os.path.exists(path):
        os.remove(path)


def save_pipeline(path: str, steps: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(steps, f, indent=2, ensure_ascii=False)


def load_pipeline(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
