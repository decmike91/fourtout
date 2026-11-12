# -*- coding: utf-8 -*-
"""
Exécution de modèles ONNX locaux (débruitage, super-résolution, déconvolution…).
Chaque modèle = fichier .onnx dans models/ai/ + fichier .json optionnel du même nom :
{
  "task": "denoise|superres|deconv|generic",
  "input_name": "x", "input_size": 256, "overlap": 32,
  "scale": 4, "norm": {"scale": 1.0, "offset": 0.0}
}
Inférence par tuiles avec recouvrement et fondu (blend) pour les grandes images.
"""
from __future__ import annotations
import json
import os
import numpy as np

from ..paths import APP_MODELS_DIR, USER_MODELS_DIR

# dossier où l'utilisateur dépose ses modèles (inscriptible) ; les modèles
# livrés avec le programme (APP_MODELS_DIR) sont aussi listés
MODELS_DIR = USER_MODELS_DIR


def list_models() -> list[dict]:
    """Liste les modèles ONNX disponibles avec leur configuration."""
    out, seen = [], set()
    dirs = [MODELS_DIR]
    if os.path.abspath(APP_MODELS_DIR) != os.path.abspath(MODELS_DIR):
        dirs.append(APP_MODELS_DIR)
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.lower().endswith(".onnx") or fn in seen:
                continue
            seen.add(fn)
            cfg = {"task": "generic", "input_size": 256, "overlap": 32, "scale": 1,
                   "norm": {"scale": 1.0, "offset": 0.0}}
            jpath = os.path.join(d, fn[:-5] + ".json")
            if os.path.exists(jpath):
                try:
                    with open(jpath, "r", encoding="utf-8") as f:
                        cfg.update(json.load(f))
                except (OSError, ValueError) as e:
                    cfg["config_error"] = str(e)
            out.append({"name": fn, "path": os.path.join(d, fn), "config": cfg})
    return out


def model_supports_task(model: dict, task: str) -> bool:
    """Pas d'inférence de compatibilité à partir du nom ou d'une tâche générique."""
    declared = model["config"].get("task")
    # Ancienne métadonnée documentée : deconv correspond à la tâche netteté.
    return declared == task or (declared == "deconv" and task == "sharpen")


class OnnxRunner:
    """Charge un modèle ONNX et exécute l'inférence par tuiles."""

    def __init__(self, model_path: str, config: dict | None = None):
        import onnxruntime as ort  # import paresseux
        self.path = model_path
        self.cfg = config or {}
        so = ort.SessionOptions()
        providers = ["CPUExecutionProvider"]
        try:
            avail = ort.get_available_providers()
            for p in ("CUDAExecutionProvider", "DmlExecutionProvider",
                      "CoreMLExecutionProvider"):
                if p in avail:
                    providers = [p, "CPUExecutionProvider"]
                    break
        except Exception:
            pass
        self.session = ort.InferenceSession(model_path, so, providers=providers)
        self.input_name = self.cfg.get("input_name") or self.session.get_inputs()[0].name

    # ------------------------------------------------------------- pré/post
    def _pre(self, tile: np.ndarray) -> np.ndarray:
        """HxW ou HxWx3 float [0..1] -> NCHW normalisé."""
        ns = self.cfg.get("norm", {}).get("scale", 1.0)
        no = self.cfg.get("norm", {}).get("offset", 0.0)
        x = np.asarray(tile, np.float32) * np.float32(ns) + np.float32(no)
        if x.ndim == 2:
            x = x[None, None]
        else:
            x = x.transpose(2, 0, 1)[None]
        return np.ascontiguousarray(x, np.float32)

    def _post(self, y: np.ndarray, shape_ref: np.ndarray) -> np.ndarray:
        ns = self.cfg.get("norm", {}).get("scale", 1.0)
        no = self.cfg.get("norm", {}).get("offset", 0.0)
        y = np.asarray(y, np.float32)
        if y.ndim == 4:
            if y.shape[0] != 1:
                raise RuntimeError("ONNX : sortie avec plusieurs images, attendue : une image NCHW.")
            y = y[0]
        if y.ndim == 3:
            y = y.transpose(1, 2, 0)  # CHW -> HWC, singleton conservé jusqu'ici
            if shape_ref.ndim == 2 and y.shape[2] == 1:
                y = y[..., 0]
        if y.ndim not in (2, 3) or y.ndim != shape_ref.ndim or (
                y.ndim == 3 and y.shape[2] != shape_ref.shape[2]):
            raise RuntimeError("ONNX : canaux de sortie incompatibles avec l'image mono/RGB.")
        if not np.isfinite(y).all() or not np.isfinite(ns) or not np.isfinite(no) or ns == 0:
            raise RuntimeError("ONNX : sortie ou normalisation invalide.")
        y = (y - no) / ns
        scale = int(self.cfg.get("scale", 1))
        expected = (shape_ref.shape[0] * scale, shape_ref.shape[1] * scale)
        if y.shape[:2] != expected:
            raise RuntimeError(f"ONNX : taille de sortie {y.shape[:2]} incompatible "
                               f"avec l'échelle déclarée {scale} (attendue {expected}).")
        return np.clip(y, 0, 1).astype(np.float32)

    # ------------------------------------------------------------- inférence
    def run(self, img: np.ndarray, progress=None) -> np.ndarray:
        """Applique le modèle à une image float [0..1] (mono ou RGB), par tuiles."""
        size = int(self.cfg.get("input_size", 256))
        ov = int(self.cfg.get("overlap", 32))
        scale = int(self.cfg.get("scale", 1))
        if size < 1 or ov < 0 or ov >= size or scale < 1:
            raise RuntimeError("ONNX : input_size, overlap ou scale invalide.")
        a = np.asarray(img, np.float32)
        if a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[2] not in (1, 3)):
            raise RuntimeError("ONNX : image mono ou RGB attendue.")
        if not a.size or not np.isfinite(a).all():
            raise RuntimeError("ONNX : image vide ou non finie.")
        h, w = a.shape[:2]
        if h <= size and w <= size:
            y = self.session.run(None, {self.input_name: self._pre(a)})[0]
            out = self._post(y, a)
            return out
        # grille de tuiles avec recouvrement
        ys = list(range(0, max(h - size, 0) + 1, size - ov)) or [0]
        xs = list(range(0, max(w - size, 0) + 1, size - ov)) or [0]
        if ys[-1] + size < h:
            ys.append(h - size)
        if xs[-1] + size < w:
            xs.append(w - size)
        out = np.zeros((h * scale, w * scale) + a.shape[2:], np.float32) if a.ndim == 3 \
            else np.zeros((h * scale, w * scale), np.float32)
        wsum = np.zeros(out.shape[:2], np.float32)
        total = len(ys) * len(xs)
        done = 0
        for ty in ys:
            for tx in xs:
                tile = a[ty:ty + size, tx:tx + size]
                y = self.session.run(None, {self.input_name: self._pre(tile)})[0]
                r = self._post(y, tile)
                oh, ow = r.shape[:2]
                # fondu triangulaire sur le recouvrement
                wy = np.ones(oh, np.float32)
                wx = np.ones(ow, np.float32)
                if ov and min(oh, ow) >= 2:
                    ramp = np.linspace(0.05, 1, min(ov * scale, oh // 2))
                    wy[:len(ramp)] = np.minimum(wy[:len(ramp)], ramp)
                    wy[-len(ramp):] = np.minimum(wy[-len(ramp):], ramp[::-1])
                    rampx = np.linspace(0.05, 1, min(ov * scale, ow // 2))
                    wx[:len(rampx)] = np.minimum(wx[:len(rampx)], rampx)
                    wx[-len(rampx):] = np.minimum(wx[-len(rampx):], rampx[::-1])
                w2 = np.outer(wy, wx)
                oy, ox = ty * scale, tx * scale
                out[oy:oy + oh, ox:ox + ow] += r * w2[..., None] \
                    if r.ndim == 3 else r * w2
                wsum[oy:oy + oh, ox:ox + ow] += w2
                done += 1
                if progress:
                    progress(done, total)
        wsum = np.maximum(wsum, 1e-6)
        out = out / wsum[..., None] if out.ndim == 3 else out / wsum
        return np.clip(out, 0, 1).astype(np.float32)
