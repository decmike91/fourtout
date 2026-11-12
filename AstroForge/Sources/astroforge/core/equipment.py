# -*- coding: utf-8 -*-
"""
Profils d'équipement et aide à l'astrométrie :
- profils caméra / télescope / réducteur de focale / conditions (pollution
  lumineuse, température capteur, mode ciel profond ou planétaire) ;
- calculs optiques : focale effective, échelle (\"/px), champ de vue ;
- détermination automatique de la focale depuis le plate solving ;
- génération d'un catalogue Gaia local (CSV) couvrant le champ de vue.
Réglages généraux (solve-field activé, catalogue actif, profil actif) dans
profiles/settings.json.
"""
from __future__ import annotations
import json
import os
import re

import numpy as np

from ..paths import PROFILES_ROOT

EQUIP_DIR = os.path.join(PROFILES_ROOT, "equipment")
SETTINGS_PATH = os.path.join(PROFILES_ROOT, "settings.json")

DEFAULT_EQUIPMENT = {
    "name": "standard",
    "camera": "", "pixel_um": 3.8, "sensor_w_px": 5000, "sensor_h_px": 3500,
    "sensor_temp": -10.0, "bayer": "RGGB",
    "telescope": "", "aperture_mm": 100.0, "focal_mm": 500.0,
    "reducer": False, "reducer_factor": 0.8,
    "mode": "Ciel profond", "light_pollution": "Banlieue (Bortle 5-6)",
    "catalog": "",
}

DEFAULT_SETTINGS = {"use_solve_field": True, "gaia_catalog": "",
                    "active_profile": "standard"}

POLLUTION = ["Rural sombre (Bortle 3-4)", "Banlieue (Bortle 5-6)",
             "Urbain (Bortle 7-8)", "Centre-ville (Bortle 9)"]
# force de débruitage suggérée selon la pollution (0..1)
DENOISE_SUGGESTION = {POLLUTION[0]: 0.35, POLLUTION[1]: 0.5,
                      POLLUTION[2]: 0.65, POLLUTION[3]: 0.8}


# ------------------------------------------------------------------ réglages
def load_settings() -> dict:
    s = json.loads(json.dumps(DEFAULT_SETTINGS))
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                s.update(json.load(f))
        except Exception:
            pass
    return s


def save_settings(settings: dict) -> None:
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------ profils
def list_profiles() -> list[str]:
    os.makedirs(EQUIP_DIR, exist_ok=True)
    names = [f[:-5] for f in os.listdir(EQUIP_DIR) if f.endswith(".json")]
    return sorted(names, key=lambda n: (n != "standard", n)) or ["standard"]


def load_profile(name: str) -> dict:
    p = json.loads(json.dumps(DEFAULT_EQUIPMENT))
    p["name"] = name
    if name != "standard":
        path = os.path.join(EQUIP_DIR, name + ".json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    p.update(json.load(f))
            except Exception:
                pass
    return p


def save_profile(name: str, params: dict) -> None:
    if not name or not name.strip():
        raise ValueError("Nom de profil vide")
    name = name.strip()
    if name == "standard":
        raise ValueError("Le profil « standard » est protégé — enregistrez sous "
                         "un autre nom")
    os.makedirs(EQUIP_DIR, exist_ok=True)
    params = dict(params)
    params["name"] = name
    with open(os.path.join(EQUIP_DIR, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, ensure_ascii=False)


def delete_profile(name: str) -> None:
    if name == "standard":
        raise ValueError("Le profil « standard » ne peut pas être supprimé")
    path = os.path.join(EQUIP_DIR, name + ".json")
    if os.path.exists(path):
        os.remove(path)


# ------------------------------------------------------------------ optique
def effective_focal(p: dict) -> float:
    """Focale réelle = focale télescope × facteur du réducteur (si coché)."""
    return float(p["focal_mm"]) * (float(p["reducer_factor"]) if p.get("reducer")
                                   else 1.0)


def scale_from_focal(pixel_um: float, focal_mm: float) -> float:
    """Échantillonnage (arcsec/pixel) = 206,265 × µm / mm."""
    return 206.265 * float(pixel_um) / float(focal_mm)


def focal_from_scale(pixel_um: float, scale_arcsec_pix: float) -> float:
    """Focale (mm) déduite de l'échelle mesurée par plate solving."""
    return 206.265 * float(pixel_um) / float(scale_arcsec_pix)


def field_of_view(p: dict) -> dict:
    """Champ de vue (degrés) couvert par le capteur."""
    scale = scale_from_focal(p["pixel_um"], effective_focal(p))  # "/px
    return {"w_deg": p["sensor_w_px"] * scale / 3600.0,
            "h_deg": p["sensor_h_px"] * scale / 3600.0,
            "scale": scale}


def suggestion_denoise(p: dict) -> float:
    """Force de débruitage conseillée selon la pollution lumineuse."""
    return DENOISE_SUGGESTION.get(p.get("light_pollution", ""), 0.5)


# ------------------------------------------------------------------ coordonnées
def parse_coord(txt: str, is_ra: bool) -> float | None:
    """Convertit une coordonnée en degrés décimaux. Formats acceptés :
    « 10h25m03s », « 10:25:03 », « 10 25 03.5 », « +19°48'30\" »,
    « 19d48m30s », « -0 30 00 », degrés décimaux (« 156,25 » ou « 156.25 »).
    Pour l'ascension droite, la forme sexagésimale est en HEURES (×15) et la
    forme décimale en degrés."""
    t = (txt or "").strip().replace(",", ".").replace("−", "-")
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        pass
    sign = -1.0 if t.startswith("-") else 1.0
    nums = re.findall(r"\d+(?:\.\d*)?|\.\d+", t)
    rest = re.sub(r"[\d.\s:+\-hHmMsSdD°º'\"′″]", "", t)
    if not nums or len(nums) > 3 or rest:
        raise ValueError(f"Coordonnée illisible : « {txt} »")
    vals = [float(v) for v in nums] + [0.0, 0.0]
    a, m, sec = vals[0], vals[1], vals[2]
    if m >= 60 or sec >= 60:
        raise ValueError(f"Minutes/secondes ≥ 60 dans « {txt} »")
    value = sign * (a + m / 60.0 + sec / 3600.0)
    return value * 15.0 if is_ra else value


# ------------------------------------------------------------------ Gaia
def make_gaia_catalog(ra_deg: float, dec_deg: float, radius_deg: float,
                      out_csv: str, mag_limit: float = 15.0) -> str:
    """Télécharge le catalogue Gaia DR3 (étoiles < mag_limit dans le rayon) et
    écrit un CSV « ra,dec,mag » utilisable par le solveur interne et la
    photométrie. Nécessite `pip install astroquery` et une connexion internet."""
    try:
        from astroquery.gaia import Gaia
    except ImportError as e:
        raise RuntimeError("astroquery non installé : pip install astroquery "
                           "(puis relancez la génération du catalogue)") from e
    try:
        # requête ASYNCHRONE : les requêtes synchrones du service Gaia sont
        # limitées à 2 000 lignes (catalogue silencieusement tronqué)
        Gaia.ROW_LIMIT = -1
        job = Gaia.launch_job_async(
            f"SELECT ra, dec, phot_g_mean_mag AS mag, phot_bp_mean_mag AS bp, "
            f"phot_rp_mean_mag AS rp FROM gaiadr3.gaia_source "
            f"WHERE phot_g_mean_mag < {float(mag_limit)} AND "
            f"CONTAINS(POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {float(ra_deg)}, {float(dec_deg)}, "
            f"{float(radius_deg)}))=1")
        tab = job.get_results()
    except Exception as e:  # réseau indisponible, service coupé…
        raise RuntimeError(f"Échec de la requête Gaia : {e}") from e
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
    tab = tab.filled(np.nan) if getattr(tab, "masked", False) else tab
    for col in list(tab.colnames):  # noms en minuscules attendus (ra, dec, mag, bp, rp)
        if col != col.lower():
            tab.rename_column(col, col.lower())
    tab.write(out_csv, format="ascii.csv", overwrite=True)  # sans pandas
    return out_csv
