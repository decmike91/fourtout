# -*- coding: utf-8 -*-
"""Pipeline de traitement automatique + registre des étapes."""
from __future__ import annotations
import traceback
import numpy as np

from ..core import calibration, stacking, registration, drizzle
from ..core.restoration import denoise, richardson_lucy, unsharp
from ..core.gradient import extract_gradient
from ..core.effects import (remove_satellite_trails, merge_hdr, apply_palette)
from ..core.wavelets import atrous_decompose, atrous_recompose
from ..ai.manager import get_ai_manager

# ------------------------------------------------------------------ étapes
# Chaque étape : {label, catégorie, params (défauts « standard »), fonction(ctx, params, progress)}
# ctx : dict partagé (paths, images, meta). La fonction retourne un dict de mise à jour du ctx.

STEPS: dict[str, dict] = {}


def step(name, label, category, params):
    def deco(fn):
        STEPS[name] = {"name": name, "label": label, "category": category,
                       "params": params, "fn": fn}
        return fn
    return deco


# ------------------------------------------------------------- ciel profond
@step("calibration", "Calibration (masters)", "Pré-traitement",
      {"bias_dir": "", "dark_dir": "", "flat_dir": "", "darkflat_dir": "",
       "dark_scale": "auto", "cosmetic": True, "use_library": False,
       "save_to_library": False})
def st_calibration(ctx, p, progress=None):
    light_meta = None
    if ctx.get("light_paths") and p.get("use_library"):
        from ..core.session import frame_info
        fi = frame_info(ctx["light_paths"][0])
        light_meta = {"exptime": fi.exptime, "gain": fi.gain, "temp": fi.temp,
                      "binning": fi.binning, "width": fi.width, "height": fi.height,
                      "filter": fi.filter}
    masters = calibration.build_masters(p["bias_dir"], p["dark_dir"],
                                        p["flat_dir"], p["darkflat_dir"],
                                        paths=ctx.get("calib_paths"), light_meta=light_meta,
                                        use_library=bool(p.get("use_library")),
                                        save_to_library=bool(p.get("save_to_library")))
    ctx["master_dark"], ctx["master_flat"] = masters["dark"], masters["flat"]
    ctx["images"] = calibration.calibrate_paths(
        ctx["light_paths"], masters, bayer=ctx.get("bayer"),
        dark_scale=p["dark_scale"], cosmetic=p["cosmetic"], progress=progress)
    ctx.pop("pre_align", None)
    return ctx


@step("registration", "Alignement des images", "Pré-traitement",
      {"with_rotation": False, "max_deg": 2.0})
def st_registration(ctx, p, progress=None):
    imgs = ctx["images"]
    ref = imgs[len(imgs) // 2]
    ctx["pre_align"] = list(imgs)  # images NON rééchantillonnées pour le drizzle
    ctx["images"] = registration.align_stack(ref, imgs, p["with_rotation"],
                                             p["max_deg"], progress=progress)
    return ctx


def drizzle_images(ctx, scale: int, pixfrac: float, method: str = "Sigma-clipping",
                   sigma: float = 3.0, with_rotation: bool = False, progress=None,
                   weights=None):
    """Drizzle sur les images AVANT alignement (sinon aucun gain de
    résolution) : transformation de chaque frame mesurée sur l'image brute
    (étoiles si possible, sinon corrélation de phase), rejet des pixels
    aberrants contre l'empilement classique des images alignées."""
    src = ctx.get("pre_align") or ctx["images"]
    ref_i = len(src) // 2
    ref = src[ref_i]
    matrices = [registration.estimate_transform(ref, im, with_rotation)
                for im in src]
    aligned = ctx["images"] if ctx.get("pre_align") else [
        registration.warp(im, M) for im, M in zip(src, matrices)]
    if weights is not None and len(weights) != len(src):
        weights = None
    reference = stacking.stack(np.stack(aligned), method, sigma=sigma, weights=weights)
    img, _cov = drizzle.drizzle_stack(src, matrices=matrices, scale=int(scale),
                                      pixfrac=pixfrac, reference=reference,
                                      reject_sigma=sigma, progress=progress,
                                      weights=weights)
    return img


@step("drizzle", "Drizzle + empilement", "Empilement",
      {"enabled_drizzle": True, "scale": 2, "pixfrac": 0.7,
       "method": "Moyenne (drizzle)", "stack_method": "Sigma-clipping",
       "sigma": 3.0})
def st_stack(ctx, p, progress=None):
    imgs = ctx["images"]
    if p.get("enabled_drizzle") and p["method"].startswith("Moyenne (drizzle)"):
        img = drizzle_images(ctx, int(p["scale"]), p["pixfrac"], p["stack_method"],
                             p["sigma"], progress=progress, weights=ctx.get("weights"))
    else:
        w = ctx.get("weights")
        img = stacking.stack(np.stack(imgs), p["stack_method"], sigma=p["sigma"],
                             progress=progress,
                             weights=w if w and len(w) == len(imgs) else None)
    ctx["image"] = img
    return ctx


@step("gradient", "Extraction de fond / gradient", "Traitement",
      {"method": "Polynôme 2D", "degree": 3, "grid": 12, "sigma": 2.5,
       "correction": "Soustraction"})
def st_gradient(ctx, p, progress=None):
    ctx["image"], ctx["background_model"] = extract_gradient(
        ctx["image"], p["method"], p["degree"], p["grid"], p["sigma"], p["correction"])
    return ctx


@step("denoise", "Débruitage", "Traitement",
      {"provider": "classique", "method": "Non-local means", "strength": 0.5})
def st_denoise(ctx, p, progress=None):
    if p["provider"] in ("classique", "onnx-classique", "", None):
        ctx["image"] = denoise(ctx["image"], p["method"], p["strength"])
    else:
        ai = get_ai_manager()
        ctx["image"] = ai.run("denoise", ctx["image"], p["provider"],
                              p["strength"], progress=progress)
    return ctx


@step("deconvolution", "Déconvolution (Richardson-Lucy)", "Traitement",
      {"sigma_psf": 1.5, "iterations": 20, "damping": 0.2})
def st_deconv(ctx, p, progress=None):
    ctx["image"] = richardson_lucy(ctx["image"], p["sigma_psf"], p["iterations"],
                                   p["damping"])
    return ctx


@step("sharpen", "Netteté (masque flou)", "Traitement", {"amount": 0.6, "radius": 2.0})
def st_sharpen(ctx, p, progress=None):
    ctx["image"] = unsharp(ctx["image"], p["amount"], p["radius"])
    return ctx


@step("ai_tool", "Outil IA (netteté/super-résolution/étoiles…)", "IA",
      {"task": "sharpen", "provider": "onnx", "strength": 0.5})
def st_ai(ctx, p, progress=None):
    ai = get_ai_manager()
    ctx["image"] = ai.run(p["task"], ctx["image"], p["provider"], p["strength"],
                          progress=progress)
    return ctx


@step("satellite", "Suppression traces satellites", "Traitement",
      {"min_len_frac": 0.25, "threshold": 8.0, "width": 5})
def st_satellite(ctx, p, progress=None):
    ctx["image"] = remove_satellite_trails(ctx["image"], **p)
    return ctx


@step("hdr", "Composition HDR", "Traitement", {"contrast": 1.0, "saturation": 1.0,
                                               "well_exposed": 1.0})
def st_hdr(ctx, p, progress=None):
    imgs = ctx.get("hdr_images") or []
    if len(imgs) >= 2:
        ctx["image"] = merge_hdr(imgs, **p)
    return ctx


@step("falsecolor", "Fausses couleurs", "Finition", {"palette": "Solaire (orange)"})
def st_falsecolor(ctx, p, progress=None):
    from ..core.image_io import is_color
    if not is_color(ctx["image"]):
        ctx["image"] = apply_palette(ctx["image"], p["palette"])
    return ctx


# ------------------------------------------------------------- planétaire
@step("quality_rank", "Classement qualité des frames", "Planétaire",
      {"mode": "Netteté (Registax)", "keep_percent": 75.0})
def st_rank(ctx, p, progress=None):
    from ..core.stacking import rank_frames
    idx, scores = rank_frames(ctx["frames"], p["mode"], p["keep_percent"])
    ctx["kept_indices"] = idx
    ctx["quality_scores"] = scores
    ctx["frames"] = [ctx["frames"][i] for i in idx]
    return ctx


@step("derotation", "Dérotation du champ", "Planétaire",
      {"mode": "auto", "rate_deg_min": 0.0, "interval_s": 0.0, "auto_max_deg": 2.0})
def st_derotation(ctx, p, progress=None):
    frames = ctx["frames"]
    if p["mode"] == "auto":
        ctx["frames"] = registration.derotate(frames, auto_max_deg=p["auto_max_deg"],
                                              progress=progress)
    else:
        ctx["frames"] = registration.derotate(
            frames, rate_deg_per_min=p["rate_deg_min"], interval_s=p["interval_s"],
            progress=progress)
    return ctx


@step("align_planetary", "Alignement fin (planétaire)", "Planétaire",
      {"with_rotation": False, "max_deg": 1.0})
def st_align_p(ctx, p, progress=None):
    frames = ctx["frames"]
    ref = frames[len(frames) // 2]
    ctx["frames"] = registration.align_stack(ref, frames, p["with_rotation"],
                                             p["max_deg"], progress=progress)
    return ctx


@step("stack_planetary", "Empilement planétaire", "Planétaire",
      {"method": "Moyenne", "sigma": 3.0, "percent": 90.0})
def st_stack_p(ctx, p, progress=None):
    frames = ctx["frames"]
    if p["method"] == "Moyenne (meilleurs %)":
        k = max(1, int(len(frames) * p["percent"] / 100.0))
        frames = frames[:k]
    ctx["image"] = stacking.stack(np.stack(frames), "Moyenne" if p["method"].startswith(
        "Moyenne") else p["method"], sigma=p["sigma"], progress=progress)
    return ctx


@step("rgb_align", "Alignement des canaux couleur", "Planétaire", {"max_shift": 12.0})
def st_rgb(ctx, p, progress=None):
    from ..core.image_io import is_color
    if is_color(ctx["image"]):
        ctx["image"] = registration.align_channels(ctx["image"], p["max_shift"])
    return ctx


@step("wavelets", "Ondelettes (style Registax6)", "Planétaire",
      {"levels": 6, "gains": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]})
def st_wavelets(ctx, p, progress=None):
    layers = atrous_decompose(ctx["image"], int(p["levels"]))
    gains = list(p["gains"])[:len(layers)] + [1.0] * (len(layers) - len(p["gains"]))
    ctx["image"] = atrous_recompose(layers, gains)
    return ctx


@step("export_animation", "Film stabilisé (aligner + recadrer)", "Planétaire",
      {"path": "animation.mp4", "fps": 24, "quality": 90, "crop_size": 0})
def st_anim(ctx, p, progress=None):
    from ..core.animation import stabilized_frames, export_video
    frames = ctx["frames"]
    stab = stabilized_frames(frames, size=int(p["crop_size"]) if p["crop_size"] else None)
    export_video(stab, p["path"], int(p["fps"]), int(p["quality"]),
                 progress=progress)
    ctx["animation_path"] = p["path"]
    return ctx


# ------------------------------------------------------------------ moteur
DEFAULT_CHAIN_SKY = ["calibration", "registration", "drizzle", "gradient", "denoise"]
DEFAULT_CHAIN_PLANET = ["quality_rank", "derotation", "align_planetary",
                        "stack_planetary", "wavelets"]


class Pipeline:
    """Liste ordonnée d'étapes {name, params, enabled} exécutée sur un contexte."""

    def __init__(self, steps: list[dict] | None = None):
        self.steps = steps or []

    @classmethod
    def default_for(cls, mode: str) -> "Pipeline":
        names = DEFAULT_CHAIN_SKY if mode == "sky" else DEFAULT_CHAIN_PLANET
        return cls([{"name": n, "params": dict(STEPS[n]["params"]), "enabled": True}
                    for n in names])

    def run(self, ctx: dict, progress=None) -> dict:
        total = len([s for s in self.steps if s.get("enabled", True)])
        done = 0
        for s in self.steps:
            if not s.get("enabled", True):
                continue
            st = STEPS[s["name"]]
            merged = dict(st["params"])
            merged.update(s.get("params", {}))
            def sub(d, t, label="", _l=st["label"], _done=done):
                if progress:
                    progress(_done, total, f"{_l} ({d}/{t})")
            try:
                ctx = st["fn"](ctx, merged, progress=sub)
            except InterruptedError:
                raise
            except Exception as e:
                raise RuntimeError(
                    f"Étape « {st['label']} » a échoué :\n{traceback.format_exc()}") from e
            done += 1
            if progress:
                progress(done, total, st["label"])
        return ctx
