# -*- coding: utf-8 -*-
"""
Animation planétaire : recadrage stabilisé (pour un film sans « danse » d'image)
et export vidéo MP4/AVI ou GIF.
"""
from __future__ import annotations
from collections.abc import Sequence

import numpy as np


def common_crop(frames: list[np.ndarray], shifts: list[tuple[float, float]] | None = None,
                angles: list[float] | None = None, size: int | None = None,
                center: tuple[int, int] | None = None) -> tuple[int, int, int, int]:
    """Calcule le rectangle commun à toutes les frames une fois translatées de
    `shifts` (convention apply_shift) : intersection des zones valides (avec
    marge de rotation). Retourne (x0, y0, w, h). `size` force une taille
    carrée autour de `center`."""
    n = len(frames)
    if n == 0:
        raise ValueError("Aucune frame")
    h, w = frames[0].shape[:2]
    shifts = shifts or [(0.0, 0.0)] * n
    angles = angles or [0.0] * n
    margin = 0.0
    if angles and max(abs(a) for a in angles) > 0:
        margin = (1 - np.cos(np.deg2rad(max(abs(a) for a in angles)))) * max(h, w) / 2.0
    xs = [max(0.0, dx) for dx, _ in shifts]
    ys = [max(0.0, dy) for _, dy in shifts]
    x1 = min(w + min(0.0, dx) for dx, _ in shifts)
    y1 = min(h + min(0.0, dy) for _, dy in shifts)
    x0, y0 = max(xs), max(ys)
    x0 += margin
    y0 += margin
    x1 -= margin
    y1 -= margin
    if size is not None:
        cx, cy = center or (int((x0 + x1) / 2), int((y0 + y1) / 2))
        half = size // 2
        x0, y0 = cx - half, cy - half
        x1, y1 = x0 + size, y0 + size
    x0, y0 = int(max(0, x0)), int(max(0, y0))
    x1, y1 = int(min(w, x1)), int(min(h, y1))
    if x1 - x0 < 16 or y1 - y0 < 16:
        x0, y0, x1, y1 = 0, 0, w, h
    return x0, y0, x1 - x0, y1 - y0


def crop_frames(frames: list[np.ndarray], box: tuple[int, int, int, int]) -> list[np.ndarray]:
    x0, y0, cw, ch = box
    return [f[y0:y0 + ch, x0:x0 + cw] for f in frames]


class StabilizedFrames(Sequence):
    """Frames recalées (translation mesurée par phase_corr) puis recadrées sur
    la zone commune : vue différée, une seule image calculée à la fois."""

    def __init__(self, frames, shifts, box):
        self.frames = frames
        self.shifts = list(shifts)
        self.box = box

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        from .registration import apply_shift
        if isinstance(i, slice):
            idx = range(*i.indices(len(self)))
            return [self[j] for j in idx]
        dx, dy = self.shifts[i]
        f = np.asarray(self.frames[i], np.float32)
        if dx or dy:
            f = apply_shift(f, dx, dy, border_mode="constant")
        x0, y0, cw, ch = self.box
        return f[y0:y0 + ch, x0:x0 + cw]


def stabilized_frames(frames, size: int | None = None, ref_index: int | None = None,
                      progress=None) -> StabilizedFrames:
    """Film stabilisé : chaque frame est recalée sur la frame de référence
    (centrale par défaut) puis toutes sont recadrées sur leur zone commune
    (plus de « danse » de l'image)."""
    from .registration import phase_corr
    n = len(frames)
    if n == 0:
        raise ValueError("Aucune frame")
    ri = n // 2 if ref_index is None else int(ref_index)
    ref = frames[ri]
    shifts = []
    for i in range(n):
        if progress:
            progress(i, n, "Mesure du recalage")
        shifts.append((0.0, 0.0) if i == ri else tuple(phase_corr(ref, frames[i])[:2]))
    box = common_crop([ref], shifts, size=size)
    return StabilizedFrames(frames, shifts, box)


def export_video(frames: list[np.ndarray], path: str, fps: int = 24, quality: int = 90,
                 progress=None) -> str:
    """Exporte en MP4 (mp4v), AVI (MJPG) ou GIF. quality ∈ [1..100] (AVI uniquement)."""
    cv2 = __import__("cv2")
    from .image_io import is_color, to_uint8
    if not frames:
        raise ValueError("Aucune frame à exporter")
    f0 = frames[0]
    h, w = f0.shape[:2]
    ext = path.rsplit(".", 1)[-1].lower()
    if ext == "gif":
        return export_gif(frames, path, fps, progress=progress)
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if ext == "mp4" else "MJPG"))
    writer = cv2.VideoWriter(path, fourcc, float(fps), (int(w), int(h)), is_color(f0))
    if not writer.isOpened():
        raise RuntimeError("Impossible d'ouvrir le VideoWriter (codec manquant ?)")
    try:  # qualité de compression (pris en compte par MJPG ; ignoré par mp4v)
        writer.set(cv2.VIDEOWRITER_PROP_QUALITY, float(np.clip(quality, 1, 100)))
    except Exception:
        pass
    try:
        for i, f in enumerate(frames):
            if is_color(f):
                writer.write(cv2.cvtColor(to_uint8(f), cv2.COLOR_RGB2BGR))
            else:
                writer.write(to_uint8(f))
            if progress:
                progress(i + 1, len(frames))
    finally:
        writer.release()
    return path


def export_gif(frames: list[np.ndarray], path: str, fps: int = 12, max_width: int = 800,
               progress=None) -> str:
    from PIL import Image
    from .image_io import to_uint8
    __import__("cv2")
    pil_frames = []
    for i, f in enumerate(frames):
        if progress:
            progress(i, len(frames), "Préparation GIF")
        u8 = to_uint8(f)  # Pillow attend du RGB : aucune conversion BGR
        img = Image.fromarray(u8)
        if img.width > max_width:
            img = img.resize((max_width, int(img.height * max_width / img.width)),
                             Image.LANCZOS)
        pil_frames.append(img)
    if not pil_frames:
        raise ValueError("Aucune frame")
    pil_frames[0].save(path, save_all=True, append_images=pil_frames[1:],
                       duration=int(1000 / max(fps, 1)), loop=0)
    return path
