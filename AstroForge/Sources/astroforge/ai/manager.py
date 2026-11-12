# -*- coding: utf-8 -*-
"""
Gestionnaire IA : unifie les modèles ONNX locaux et les outils externes
(GraXpert, Cosmic Clarity / SetiAstro, StarNet2 / DeepSNR) appelés en sous-process.
Chaque tâche (denoise, fond/gradient, netteté/déconvolution, super-résolution,
traces satellites) peut être servie par plusieurs fournisseurs.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import tempfile
import shlex
import weakref
import time

import numpy as np

from .onnx_runner import OnnxRunner, list_models, model_supports_task

TASKS = ["denoise", "background", "sharpen", "superres", "satellite", "starless"]
TASK_LABELS = {"denoise": "Débruitage", "background": "Extraction de fond",
               "sharpen": "Netteté / déconvolution", "superres": "Super-résolution",
               "satellite": "Traces de satellites", "starless": "Suppression des étoiles"}

from ..paths import PROFILES_ROOT

SETTINGS_PATH = os.path.join(PROFILES_ROOT, "ai_tools.json")

_DEFAULT_TOOLS = {
    "graxpert": {"label": "GraXpert", "binary": "", "extra_args": ""},
    "cosmicclarity": {"label": "Cosmic Clarity (SetiAstro)", "binary": "",
                      "denoise": "", "sharpen": "", "superres": "", "satellite": ""},
    "starnet2": {"label": "StarNet2 (suppression des étoiles)", "binary": "", "args": ""},
    "deepsnr": {"label": "DeepSNR (débruitage)", "binary": "", "args": ""},
}


def normalize_tool_settings(raw: dict) -> dict:
    """Complète les sections sans attribuer un ancien chemin ambigu au débruitage."""
    cfg = json.loads(json.dumps(_DEFAULT_TOOLS))
    if not isinstance(raw, dict):
        raw = {}
    for key, value in raw.items():
        if key in cfg:
            if isinstance(value, dict):
                cfg[key].update(value)
        else:
            cfg[key] = value
    version = raw.get("schema_version", 0)
    if not isinstance(version, int) or version < 2:
        old = cfg["starnet2"].get("binary", "")
        if not isinstance(old, str):
            old = ""
            cfg["starnet2"]["binary"] = ""
        name = os.path.basename(old.replace("\\", "/")).lower()
        if old and "deepsnr" in name and "starnet" not in name:
            if not cfg["deepsnr"].get("binary"):
                cfg["deepsnr"]["binary"] = old
            cfg["starnet2"]["legacy_binary"] = old
            cfg["starnet2"]["binary"] = ""
        elif old and ("starnet" not in name or "deepsnr" in name):
            cfg["starnet2"]["legacy_binary"] = old
            cfg["starnet2"]["binary"] = ""
    for key in ("starnet2", "deepsnr"):
        cfg[key]["label"] = _DEFAULT_TOOLS[key]["label"]
    cfg["schema_version"] = 2
    return cfg


def load_tool_settings() -> dict:
    if os.path.exists(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                return normalize_tool_settings(json.load(f))
        except (OSError, ValueError):
            pass
    return normalize_tool_settings({})


_provider_listeners = []


def register_provider_refresh(callback):
    _provider_listeners.append(weakref.WeakMethod(callback))


def save_tool_settings(cfg: dict) -> None:
    os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(normalize_tool_settings(cfg), f, indent=2, ensure_ascii=False)
    if _manager is not None:
        _manager.reload_settings()
    for ref in list(_provider_listeners):
        callback = ref()
        if callback is None:
            _provider_listeners.remove(ref)
        else:
            try:
                callback()
            except RuntimeError:
                # Widget Qt déjà détruit.
                _provider_listeners.remove(ref)


class AIManager:
    """Point d'entrée unique des fonctions IA. Fournisseurs :
    - 'onnx'          : modèle ONNX déposé dans models/ai/
    - 'graxpert'      : binaire GraXpert (denoise + background)
    - 'cosmicclarity' : suite SetiAstro (denoise, sharpen, superres, satellite)
    - 'starnet2'      : StarNet2 CLI (suppression des étoiles)
    - 'deepsnr'       : DeepSNR CLI (débruitage)
    """

    def __init__(self):
        self.settings = load_tool_settings()
        self._onnx_cache: dict[str, OnnxRunner] = {}

    def reload_settings(self):
        self.settings = load_tool_settings()
        self._onnx_cache.clear()

    # ------------------------------------------------------------- disponibilité
    def providers(self, task: str) -> list[dict]:
        """Les métadonnées doivent déclarer explicitement la tâche ONNX."""
        models = [m for m in list_models() if model_supports_task(m, task)]
        out = [{"id": "onnx", "label": "Modèle ONNX local" if models else
                "ONNX : aucun modèle compatible", "model": models[0]["name"] if models else None}]
        out += [{"id": "onnx:" + m["name"], "label": "ONNX : " + m["name"],
                 "model": m["name"]} for m in models]
        st = self.settings
        if st.get("graxpert", {}).get("binary") and task in ("denoise", "background"):
            out.append({"id": "graxpert", "label": "GraXpert (IA)", "model": None})
        # Aucun lancement interactif déguisé en automatisation.
        for key, supported, label in (("starnet2", "starless", "StarNet2 (IA)"),
                                      ("deepsnr", "denoise", "DeepSNR (IA)")):
            if st.get(key, {}).get("binary") and task == supported:
                out.append({"id": key, "label": label, "model": None})
        return out

    # ------------------------------------------------------------- exécution
    def run(self, task: str, img, provider: str = "onnx", strength: float = 0.5,
            progress=None, **kw):
        """Applique la tâche IA. Retourne une image float [0..1]."""
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("Force IA attendue entre 0 et 1.")
        if task not in TASKS:
            raise ValueError(f"Tâche IA inconnue : {task}")
        if provider == "onnx" or provider.startswith("onnx:"):
            models = list_models()
            compatible = [m for m in models if model_supports_task(m, task)]
            name = provider.split(":", 1)[1] if ":" in provider else (
                compatible[0]["name"] if compatible else None)
            model = next((m for m in models if m["name"] == name), None)
            if model is None:
                raise RuntimeError(
                    f"Aucun modèle ONNX disponible pour « {TASK_LABELS[task]} » "
                    "dans models/ai/. Ajoutez un modèle et ses métadonnées de tâche, "
                    "ou configurez un outil externe.")
            if not model_supports_task(model, task):
                raise RuntimeError(f"Modèle ONNX {name} incompatible avec la tâche {task} "
                                   "(métadonnées absentes ou tâche différente).")
            key = model["path"]
            if key not in self._onnx_cache:
                try:
                    self._onnx_cache[key] = OnnxRunner(model["path"], model["config"])
                except Exception as e:
                    raise RuntimeError(f"onnxruntime introuvable ou modèle invalide "
                                       f"({e}) : pip install onnxruntime") from e
            runner = self._onnx_cache[key]
            out = runner.run(img, progress=progress)
            a = np.asarray(img, np.float32)
            if out.shape == a.shape and task != "superres" and strength < 1.0:
                # force : mélange avec l'original (0 = image inchangée, 1 = modèle seul)
                out = (a + np.float32(strength) * (np.asarray(out, np.float32) - a))
            return np.asarray(out, np.float32)
        if provider == "graxpert":
            return self._run_graxpert(task, img, strength, **kw)
        if provider == "cosmicclarity":
            return self._run_cosmicclarity(task, img, strength, **kw)
        if provider in ("starnet2", "deepsnr"):
            return self._run_starnet(task, img, provider=provider, **kw)
        raise ValueError(f"Fournisseur IA inconnu : {provider}")

    # ------------------------------------------------------------- outils externes
    def _with_temp_fits(self, img, fn, ext: str = ".fits"):
        from ..core.image_io import load_image, save_image
        a = np.asarray(img, np.float32)
        if a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[2] not in (1, 3)):
            raise RuntimeError("L'outil externe accepte une image mono ou RGB uniquement.")
        if not a.size or not np.isfinite(a).all():
            raise RuntimeError("Image IA vide ou contenant des valeurs non finies.")
        with tempfile.TemporaryDirectory(prefix="astroforge_ai_") as d:
            inp = os.path.join(d, "in" + ext)
            save_image(inp, np.clip(a, 0, 1).astype(np.float32))
            # Le dossier vient d'être créé : seules les sorties explicitement
            # annoncées par cet adaptateur sont autorisées, jamais in.fits.
            before = {os.path.realpath(os.path.join(d, name)) for name in os.listdir(d)}
            started = time.time_ns()
            outputs = fn(inp, d)
            if isinstance(outputs, (str, os.PathLike)):
                outputs = [outputs]
            for output in outputs or []:
                path = os.path.realpath(output)
                if (os.path.dirname(path) != os.path.realpath(d) or path in before
                        or not os.path.isfile(path) or os.path.getsize(path) == 0
                        or os.path.samefile(path, inp)
                        or os.stat(path).st_mtime_ns < started - 2_000_000_000):
                    continue
                result = load_image(path)
                if not np.asarray(result).size or not np.isfinite(result).all():
                    raise RuntimeError("L'outil externe a produit une image vide ou non finie.")
                return result
            raise RuntimeError("L'outil externe n'a pas produit de sortie nouvelle valide "
                               "(l'entrée et les fichiers anciens sont exclus).")

    @staticmethod
    def _resolve_executable(binary, label):
        path = os.path.expanduser(binary or "")
        resolved = path if os.path.isfile(path) else shutil.which(path) if path else None
        if not resolved:
            raise RuntimeError(f"Binaire {label} introuvable ou non configuré (menu Outils IA).")
        return os.path.abspath(resolved)

    @staticmethod
    def _execute(cmd, timeout):
        try:
            # Les modèles et bibliothèques restent près du binaire, pas dans le
            # dossier temporaire d'images. Aucun dossier runtime n'est supprimé.
            subprocess.run(cmd, check=True, capture_output=True, text=True,
                           timeout=timeout, cwd=os.path.dirname(cmd[0]))
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr or exc.stdout or "aucun diagnostic fourni"
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", errors="replace")
            raise RuntimeError(f"{os.path.basename(cmd[0])} : code {exc.returncode} — "
                               f"{detail[-2000:].strip()}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"{os.path.basename(cmd[0])} : délai dépassé ({timeout} s).") from exc
        except OSError as exc:
            raise RuntimeError(f"Impossible de lancer {cmd[0]} : {exc}") from exc

    def _run_graxpert(self, task, img, strength, **kw):
        if task not in ("denoise", "background"):
            raise RuntimeError(f"GraXpert ne prend pas en charge la tâche {task}.")
        st = self.settings["graxpert"]
        exe = self._resolve_executable(st["binary"], "GraXpert")
        extra = shlex.split(st.get("extra_args", ""))
        reserved = {"-cli", "-cmd", "-output", "-strength", "-smoothing"}
        if any(arg.split("=", 1)[0] in reserved for arg in extra):
            raise RuntimeError("GraXpert : extra_args ne doit pas modifier la tâche, "
                               "le mode CLI, la force ou le chemin de sortie.")

        def fn(inp, d):
            cmd = [exe, inp, "-cli", "-cmd",
                   "denoising" if task == "denoise" else "background-extraction",
                   "-strength" if task == "denoise" else "-smoothing",
                   str(strength), "-output", os.path.join(d, "out")] + extra
            self._execute(cmd, 1800)
            return [os.path.join(d, "out.fits"), os.path.join(d, "out_GraXpert.fits")]
        return self._with_temp_fits(img, fn)

    def _run_cosmicclarity(self, task, img, strength, **kw):
        # Le workflow standalone documenté demande des choix interactifs.
        # Sans contrat CLI vérifié avec chemins isolés, aucune transaction n'est
        # lancée : ni copie, ni lecture, ni suppression dans Input/Output.
        raise RuntimeError(
            "Cosmic Clarity : automatisation non prise en charge pour cette installation. "
            "Le workflow officiel Input/Output est interactif. Utilisez l'outil "
            "manuellement puis importez sa sortie ; aucun fichier utilisateur n'a été touché.")

    def _run_starnet(self, task, img, provider="starnet2", **kw):
        expected = "starless" if provider == "starnet2" else "denoise"
        if task != expected:
            raise RuntimeError(f"{provider} est incompatible avec la tâche {task} ; "
                               f"tâche attendue : {expected}.")
        label = "StarNet2" if provider == "starnet2" else "DeepSNR"
        exe = self._resolve_executable(self.settings[provider]["binary"], label)
        a = np.asarray(img)
        if a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[2] not in (1, 3)):
            raise RuntimeError(f"{label} accepte une image mono ou RGB uniquement.")
        if min(a.shape[:2]) < 512:
            raise RuntimeError(f"{label} : image trop petite, minimum 512 × 512 pixels.")

        # StarNet n'accepte que des TIFF (16 bits) ; DeepSNR lit le FITS
        ext = ".tif" if provider == "starnet2" else ".fits"
        template = self.settings[provider].get("args", "")

        def fn(inp, d):
            outp = os.path.join(d, "out" + ext)
            if template:  # gabarit personnalisé, ex. « {input} {output} 256 »
                cmd = [exe] + [a.replace("{input}", inp).replace("{output}", outp)
                               for a in shlex.split(template)]
            else:
                cmd = [exe, "--input", inp, "--output", outp]
                if provider == "deepsnr":
                    cmd += ["--model", "2"]
            if kw.get("linear", False):
                cmd.append("--linear")
            self._execute(cmd, 1800)
            return outp
        return self._with_temp_fits(img, fn, ext)


_manager = None


def get_ai_manager() -> AIManager:
    global _manager
    if _manager is None:
        _manager = AIManager()
    return _manager
