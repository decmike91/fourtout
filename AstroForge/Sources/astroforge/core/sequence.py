# -*- coding: utf-8 -*-
"""
Séquences batch (esprit Registax : on enregistre une chaîne de traitement et on
la rejoue) :
- séquences d'OPÉRATIONS sur images (finition, calibration couleur, IA…)
  appliquées à tout un dossier ;
- séquences PLANÉTAIRES : dossier de vidéos SER traitées par la même chaîne
  (classement -> alignement -> empilement -> ondelettes -> palette), réglages
  enregistrés dans un JSON.
Format : JSON {"mode": "image"|"ser", "ops": [{"op": ..., "params": {...}}]}.
"""
from __future__ import annotations
import json
import os
import traceback

import numpy as np

# ------------------------------------------------------------ opérations image
OPS: dict[str, dict] = {}


def op(name, label, params):
    def deco(fn):
        OPS[name] = {"name": name, "label": label, "params": params, "fn": fn}
        return fn
    return deco


@op("brightness_contrast", "Luminosité / contraste", {"bright": 0.0, "contrast": 0.0})
def _op_bc(img, p):
    out = (np.asarray(img, np.float32) - 0.5) * (1 + p["contrast"]) + 0.5 + p["bright"]
    return np.clip(out, 0, 1).astype(np.float32)


@op("gamma", "Gamma", {"gamma": 1.0})
def _op_gamma(img, p):
    return np.power(np.clip(np.asarray(img, np.float32), 0, 1),
                    1.0 / max(p["gamma"], 0.01)).astype(np.float32)


@op("saturation", "Saturation", {"sat": 1.0})
def _op_sat(img, p):
    a = np.asarray(img, np.float32)
    if a.ndim != 3:
        return a
    g = a.mean(axis=2, keepdims=True)
    return np.clip(g + (a - g) * p["sat"], 0, 1).astype(np.float32)


@op("white_balance", "Balance des blancs", {"wr": 1.0, "wg": 1.0, "wb": 1.0})
def _op_wb(img, p):
    a = np.asarray(img, np.float32).copy()
    if a.ndim == 3:
        a[..., 0] *= p["wr"]
        a[..., 1] *= p["wg"]
        a[..., 2] *= p["wb"]
    return np.clip(a, 0, 1).astype(np.float32)


@op("color_calibrate", "Calibration couleur (fond ciel)",
    {"use_background": True, "percentile": 25.0})
def _op_cc(img, p):
    from .colorcal import color_calibrate
    return color_calibrate(img, use_background=p["use_background"],
                           percentile=p["percentile"])


@op("background_neutralize", "Neutralisation du fond", {"percentile": 25.0})
def _op_bn(img, p):
    from .colorcal import background_neutralize
    return background_neutralize(img, p["percentile"])


@op("scnr_green", "SCNR vert", {})
def _op_scnr(img, p):
    from .colorcal import scnr_green
    return scnr_green(img)


@op("autostretch", "Étirement automatique (affichage/traitement)", {})
def _op_as(img, p):
    from .display import autostretch
    return autostretch(img)


@op("denoise", "Débruitage classique", {"method": "Non-local means",
                                        "strength": 0.5})
def _op_dn(img, p):
    from .restoration import denoise
    return denoise(img, p["method"], p["strength"])


@op("unsharp", "Masque flou", {"amount": 0.6, "radius": 2.0})
def _op_us(img, p):
    from .restoration import unsharp
    return unsharp(img, p["amount"], p["radius"])


@op("deconv_rl", "Déconvolution Richardson-Lucy", {"sigma": 1.5, "iterations": 20})
def _op_rl(img, p):
    from .restoration import richardson_lucy
    return richardson_lucy(img, p["sigma"], p["iterations"])


@op("remove_satellite_trails", "Suppression traces satellites",
    {"min_len_frac": 0.25, "threshold": 8.0, "width": 5})
def _op_sat_trails(img, p):
    from .effects import remove_satellite_trails
    return remove_satellite_trails(img, **p)


@op("apply_palette", "Fausses couleurs", {"palette": "Solaire (orange)"})
def _op_pal(img, p):
    from .image_io import is_color
    from .effects import apply_palette
    if is_color(img):
        return img
    return apply_palette(img, p["palette"])


@op("superres_lanczos", "Super-résolution Lanczos", {"scale": 2})
def _op_sr(img, p):
    import cv2
    return cv2.resize(np.asarray(img, np.float32), None, fx=p["scale"],
                      fy=p["scale"], interpolation=cv2.INTER_LANCZOS4)


@op("ai_run", "Outil IA local", {"task": "denoise", "provider": "onnx",
                                 "strength": 0.5})
def _op_ai(img, p):
    from ..ai.manager import get_ai_manager
    return get_ai_manager().run(p["task"], img, p["provider"], p["strength"])


@op("gimp_tone", "Finition GIMP (tonalité + netteté)",
    {"bright": 0.0, "contrast": 0.0, "sat": 1.0, "gamma": 1.0,
     "wr": 1.0, "wg": 1.0, "wb": 1.0, "usm_amount": 0.0, "usm_radius": 2.0})
def _op_gimp(img, p):
    from .restoration import unsharp
    a = np.asarray(img, np.float32).copy()
    if a.ndim == 3:
        a[..., 0] *= p["wr"]
        a[..., 1] *= p["wg"]
        a[..., 2] *= p["wb"]
    a = (a - 0.5) * (1 + p["contrast"]) + 0.5 + p["bright"]
    if a.ndim == 3 and abs(p["sat"] - 1.0) > 1e-3:
        g = a.mean(axis=2, keepdims=True)
        a = g + (a - g) * p["sat"]
    a = np.power(np.clip(a, 0, 1), 1.0 / max(p["gamma"], 0.01))
    if p["usm_amount"] > 0:
        a = unsharp(a, p["usm_amount"], p["usm_radius"])
    return np.clip(a, 0, 1).astype(np.float32)


# Finition astronomique (module finishing) — implémentations maison inspirées
# des principes Photoshop/PixInsight/Siril/AstroPixelProcessor, pas des
# portages ; utilisables dans les lots d'images.
@op("fin_saturation", "Finition : saturation", {"amount": 1.0})
def _op_fin_sat(img, p):
    from .finishing import saturation
    return saturation(img, p["amount"])


@op("fin_white_balance", "Finition : balance des blancs",
    {"wr": 1.0, "wg": 1.0, "wb": 1.0})
def _op_fin_wb(img, p):
    from .finishing import white_balance
    return white_balance(img, p["wr"], p["wg"], p["wb"])


@op("fin_background_neutral", "Finition : neutralisation du fond",
    {"percentile": 5.0, "strength": 1.0})
def _op_fin_bn(img, p):
    from .finishing import neutralize_background
    return neutralize_background(img, p["percentile"], p["strength"])


@op("fin_scnr_green", "Finition : suppression du vert (SCNR)", {"strength": 1.0})
def _op_fin_scnr(img, p):
    from .finishing import scnr_green
    return scnr_green(img, p["strength"])


@op("fin_autostretch", "Finition : étirement automatique (médiane/MAD)",
    {"shadows_clip": -2.8, "target_bg": 0.25})
def _op_fin_as(img, p):
    from .finishing import autostretch
    return autostretch(img, p["shadows_clip"], p.get("target_bg", 0.25))


@op("fin_gradient", "Finition : suppression de gradient (extraction de fond)",
    {"kernel": 64, "method": "Soustraire"})
def _op_fin_grad(img, p):
    from .finishing import remove_gradient
    return remove_gradient(img, p["kernel"], p["method"])


@op("fin_median_denoise", "Finition : débruitage médian (répété)",
    {"radius": 1, "iterations": 1})
def _op_fin_dn(img, p):
    from .finishing import denoise_median
    return denoise_median(img, p["radius"], p["iterations"])


@op("fin_unsharp", "Finition : masque flou (netteté)", {"radius": 2.0, "gain": 0.6})
def _op_fin_usm(img, p):
    from .finishing import unsharp_mask
    return unsharp_mask(img, p["radius"], p["gain"])


@op("fin_levels", "Finition : niveaux", {"black": 0.0, "gamma": 1.0, "white": 1.0})
def _op_fin_lv(img, p):
    from .finishing import levels
    return levels(img, p["black"], p["gamma"], p["white"])


@op("fin_shadows_highlights", "Finition : tons clairs / tons sombres",
    {"shadows": 0.0, "highlights": 0.0, "radius": 24})
def _op_fin_sh(img, p):
    from .finishing import shadows_highlights
    return shadows_highlights(img, p["shadows"], p["highlights"], p["radius"])


@op("fin_vibrance", "Finition : vibrance", {"amount": 1.0})
def _op_fin_vib(img, p):
    from .finishing import vibrance
    return vibrance(img, p["amount"])


@op("fin_hsl", "Finition : teinte/saturation/luminosité (HSL)",
    {"hue": 0, "sat": 1.0, "lum": 0.0, "r_hue": 0, "r_sat": 1.0,
     "y_hue": 0, "y_sat": 1.0, "g_hue": 0, "g_sat": 1.0, "c_hue": 0,
     "c_sat": 1.0, "b_hue": 0, "b_sat": 1.0, "m_hue": 0, "m_sat": 1.0})
def _op_fin_hsl(img, p):
    from .finishing import hsl
    return hsl(img, **p)


@op("fin_local_contrast", "Finition : contraste local (clarté)",
    {"radius": 32, "gain": 0.0})
def _op_fin_lc(img, p):
    from .finishing import local_contrast
    return local_contrast(img, p["radius"], p["gain"])


@op("fin_split_tone", "Finition : tonalité divisée",
    {"sh_r": 0.5, "sh_g": 0.5, "sh_b": 0.5, "hl_r": 0.5, "hl_g": 0.5,
     "hl_b": 0.5, "balance": 0.0, "force": 0.0})
def _op_fin_st(img, p):
    from .finishing import split_tone
    return split_tone(img, **p)


@op("fin_gradient_map", "Finition : mappage de dégradé",
    {"r0": 0.0, "g0": 0.0, "b0": 0.0, "r1": 0.5, "g1": 0.5, "b1": 0.5,
     "r2": 1.0, "g2": 1.0, "b2": 1.0, "intensity": 0.0})
def _op_fin_gm(img, p):
    from .finishing import gradient_map
    return gradient_map(img, **p)


@op("fin_ghs", "Finition : étirement GHS",
    {"D": 0.0, "b": 0.0, "SP": 0.0, "LP": 0.0, "HP": 1.0, "method": "Canaux"})
def _op_fin_ghs(img, p):
    from .finishing import apply_filter
    return apply_filter(img, "ghs", p)


@op("fin_arcsinh", "Finition : étirement arcsinh", {"stretch": 0.0, "black": 0.0})
def _op_fin_asinh(img, p):
    from .finishing import apply_filter
    return apply_filter(img, "arcsinh", p)


@op("fin_star_reduction", "Finition : réduction d'étoiles", {"amount": 0.0, "iterations": 1})
def _op_fin_sr(img, p):
    from .finishing import apply_filter
    return apply_filter(img, "star_reduction", p)


@op("fin_chroma_denoise", "Finition : débruitage de la chrominance",
    {"strength": 0.0, "scales": 4})
def _op_fin_cd(img, p):
    from .finishing import apply_filter
    return apply_filter(img, "chroma_denoise", p)


@op("fin_masked", "Finition : filtre à travers un masque",
    {"filter": "none", "params": {}, "mask": {"kind": "Aucun"}})
def _op_fin_masked(img, p):
    from .finishing import apply_filter_masked
    return apply_filter_masked(img, p.get("filter", "none"), p.get("params") or {},
                               p.get("mask") or {})


# ------------------------------------------------------------ rapport de lot
class BatchResult(list):
    """Liste des fichiers produits + `errors` [(fichier, message)]. Un rapport
    texte est écrit dans le dossier de sortie (visible même sans console)."""

    def __init__(self, outs=(), errors=()):
        super().__init__(outs)
        self.errors = list(errors)

    def summary(self) -> str:
        txt = f"{len(self)} fichier(s) produit(s)"
        if self.errors:
            txt += f", {len(self.errors)} échec(s) : " + "; ".join(
                f"{f} ({m.splitlines()[0] if m else '?'})" for f, m in self.errors[:5])
            if len(self.errors) > 5:
                txt += " …"
        return txt


def _finish_batch(out_dir: str, outs: list, errors: list, total: int) -> BatchResult:
    res = BatchResult(outs, errors)
    try:
        with open(os.path.join(out_dir, "rapport_batch.txt"), "w", encoding="utf-8") as f:
            f.write(f"AstroForge — rapport de traitement par lot\n{res.summary()}\n\n")
            for name in outs:
                f.write(f"OK     {name}\n")
            for name, msg in errors:
                f.write(f"ÉCHEC  {name}\n{msg}\n")
    except OSError:
        pass
    if total and not outs:
        detail = "\n".join(f"- {f} : {m.splitlines()[0] if m else '?'}" for f, m in errors[:10])
        raise RuntimeError(f"Aucun fichier traité avec succès sur {total}.\n{detail}")
    return res


# ------------------------------------------------- chaîne planétaire en batch
def run_batch_ser_chain(seq: dict, in_dir: str, out_dir: str, max_frames: int | None = None,
                        progress=None) -> list[str]:
    """Chaîne planétaire complète par SER : classement -> alignement ->
    empilement -> ondelettes -> palette, avec les réglages enregistrés.
    `max_frames` None ou 0 = toutes les frames de chaque vidéo (lues par
    projection mémoire, sans charger la vidéo en RAM)."""
    from .image_io import is_color, load_ser, save_image
    from .effects import apply_palette
    from .registration import align_channels
    from .planetary import run_planetary, ToneFrames
    files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(".ser"))
    if not files:
        raise RuntimeError(f"Aucun fichier SER dans {in_dir}")
    os.makedirs(out_dir, exist_ok=True)
    outs, errors = [], []
    for i, f in enumerate(files):
        try:
            st = seq["ops"][0]["params"] if seq.get("ops") else {}
            frames, _m = load_ser(os.path.join(in_dir, f), max_frames=max_frames or None,
                                  lazy=True)
            # Avant toute mesure de qualité ; vue différée sans copie vidéo.
            frames = ToneFrames(frames, st.get("tone"))
            frames.validate(progress)
            n_all = int(len(frames))
            n_sel = int(st.get("n_align") or max(1, round(
                n_all * float(st.get("keep_percent", 75.0)) / 100.0)))
            res = run_planetary(
                frames,
                quality_method=st.get("quality_method",
                                      st.get("quality_mode",
                                             "Contraste HF (AutoStakkert)")),
                roi_mode=st.get("roi_mode", "Disque auto-détecté"),
                n_align=n_sel,
                align_mode=st.get("align_mode", "Multi-points (grille NxN)"),
                grid_n=int(st.get("grid_n", 4)),
                box_size=int(st.get("box_size", 0)),
                stack_method=st.get("stack_method", "Moyenne"),
                stack_percent=float(st.get("stack_percent", 100.0)),
                progress=progress)
            img = res["image"]
            if st.get("rgb_align") and is_color(img):
                img = align_channels(img)
            if st.get("ca"):
                from .colorcal import correct_chromatic_aberration
                img = correct_chromatic_aberration(img, st.get("kr", 0.0),
                                                   st.get("kb", 0.0))
            from .wavelets import registax_enhance
            img = registax_enhance(img, st.get("gains") or [1.0] * 7,
                                   st.get("wave_denoise", 0.0),
                                   layer_denoise=st.get("layer_denoise"),
                                   layer_sharpen=st.get("layer_sharpen"))
            if st.get("false_color") and not is_color(img):
                img = apply_palette(img, st.get("palette", "Solaire (orange)"))
            name = os.path.splitext(f)[0] + "_stacked.fits"
            save_image(os.path.join(out_dir, name), img)
            outs.append(name)
        except InterruptedError:
            raise
        except Exception as e:
            errors.append((f, f"{e}\n{traceback.format_exc(limit=3)}"))
        if progress:
            progress(i + 1, len(files))
    return _finish_batch(out_dir, outs, errors, len(files))


# ------------------------------------------------------------------ séquences
def save_sequence(path: str, ops: list[dict], mode: str = "image") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"mode": mode, "ops": ops}, f, indent=2, ensure_ascii=False)


def load_sequence(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        seq = json.load(f)
    for o in seq.get("ops", []):
        if o["op"] not in OPS and not (seq.get("mode") == "ser" and o["op"] == "planet_chain"):
            raise ValueError(f"Opération inconnue dans la séquence : {o['op']}")
    return seq


def apply_to_image(seq: dict, img: np.ndarray) -> np.ndarray:
    for o in seq.get("ops", []):
        st = OPS[o["op"]]
        params = dict(st["params"])
        params.update(o.get("params", {}))
        img = st["fn"](img, params)
    return np.asarray(img, np.float32)


IMAGE_EXTS = (".fits", ".fit", ".fts", ".fz", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".xisf")


def run_batch_image(seq: dict, in_dir: str, out_dir: str, bit_depth: str = "auto",
                    progress=None) -> list[str]:
    """Applique la séquence à toutes les images du dossier -> out_dir."""
    from .image_io import load_image, save_image
    files = sorted(f for f in os.listdir(in_dir)
                   if f.lower().endswith(IMAGE_EXTS))
    if not files:
        raise RuntimeError(f"Aucune image dans {in_dir}")
    os.makedirs(out_dir, exist_ok=True)
    outs, errors = [], []
    for i, f in enumerate(files):
        try:
            img = load_image(os.path.join(in_dir, f))
            img = apply_to_image(seq, img)
            name = os.path.splitext(f)[0] + ".tif"
            save_image(os.path.join(out_dir, name), img, bit_depth)
            outs.append(name)
        except InterruptedError:
            raise
        except Exception as e:
            errors.append((f, f"{e}\n{traceback.format_exc(limit=3)}"))
        if progress:
            progress(i + 1, len(files))
    return _finish_batch(out_dir, outs, errors, len(files))


def run_batch_ser(seq: dict, in_dir: str, out_dir: str, max_frames: int | None = None,
                  keep_percent: float = 75.0, progress=None) -> list[str]:
    """Chaîne planétaire sur un dossier de SER : classement -> alignement ->
    empilement -> opérations de la séquence -> FITS empilé par vidéo."""
    from .image_io import load_ser, save_image
    from .registration import align_stack
    from .stacking import rank_frames, stack
    files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(".ser"))
    if not files:
        raise RuntimeError(f"Aucun fichier SER dans {in_dir}")
    os.makedirs(out_dir, exist_ok=True)
    outs, errors = [], []
    for i, f in enumerate(files):
        try:
            frames, _meta = load_ser(os.path.join(in_dir, f), max_frames=max_frames or None)
            arr = np.asarray(frames)
            idx, _ = rank_frames(arr, "Netteté (Registax)", keep_percent)
            frames = [arr[j] for j in idx]
            ref = frames[len(frames) // 2]
            frames = align_stack(ref, frames)
            img = stack(np.stack(frames), "Moyenne")
            img = apply_to_image(seq, img)
            name = os.path.splitext(f)[0] + "_stacked.fits"
            save_image(os.path.join(out_dir, name), img)
            outs.append(name)
        except InterruptedError:
            raise
        except Exception as e:
            errors.append((f, f"{e}\n{traceback.format_exc(limit=3)}"))
        if progress:
            progress(i + 1, len(files))
    return _finish_batch(out_dir, outs, errors, len(files))
