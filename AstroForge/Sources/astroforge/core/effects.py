# -*- coding: utf-8 -*-
"""
Effets spéciaux : suppression de traces de satellites, fusion HDR (Mertens),
fausses couleurs (solaires).
"""
from __future__ import annotations
import numpy as np


# ------------------------------------------------------ traces de satellites
def detect_trail_mask(img: np.ndarray, min_len_frac: float = 0.25,
                      threshold: float = 8.0, width: int = 5) -> np.ndarray:
    """Détecte les lignes brillantes fines (satellites/avions) -> masque booléen.
    La détection se fait sur une copie ÉTIRÉE (données linéaires faibles
    invisibles en 8 bits brut) ; l'image elle-même n'est pas modifiée."""
    cv2 = __import__("cv2")
    from .display import autostretch
    from .image_io import to_uint8
    mono = img if img.ndim == 2 else img[..., :3].mean(axis=2)
    u8 = to_uint8(autostretch(np.asarray(mono, np.float32)))
    high = cv2.subtract(u8, cv2.medianBlur(u8, 21))
    _, th = cv2.threshold(high, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lines = cv2.HoughLinesP(th, 1, np.pi / 180, threshold=int(threshold),
                            minLineLength=int(min_len_frac * min(u8.shape)),
                            maxLineGap=6)
    mask = np.zeros(u8.shape, np.uint8)
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, int(width))
    return mask > 0


def remove_satellite_trails(img: np.ndarray, mask: np.ndarray | None = None,
                            **kw) -> np.ndarray:
    """Supprime les traces détectées (ou fournies par `mask`) par inpainting
    TELEA en flottant 32 bits, canal par canal. Seuls les pixels du masque
    sont remplacés : le reste de l'image est strictement inchangé."""
    cv2 = __import__("cv2")
    a = np.asarray(img, np.float32)
    m = detect_trail_mask(a, **kw) if mask is None else np.asarray(mask, bool)
    if m.shape != a.shape[:2]:
        raise ValueError(f"Masque {m.shape} ≠ image {a.shape[:2]}")
    if not m.any():
        return a.copy()
    m8 = m.astype(np.uint8) * 255
    out = a.copy()
    planes = [a] if a.ndim == 2 else [a[..., c] for c in range(a.shape[2])]
    for c, plane in enumerate(planes):
        fixed = cv2.inpaint(np.ascontiguousarray(plane), m8, 7, cv2.INPAINT_TELEA)
        if a.ndim == 2:
            out[m] = fixed[m]
        else:
            out[..., c][m] = fixed[m]
    return out


def remove_trails_by_median(aligned_frames: list[np.ndarray]) -> np.ndarray:
    """Méthode multi-frames : la médiane de frames alignées efface les traces mobiles."""
    if not aligned_frames:
        raise ValueError("Aucune frame")
    return np.median(np.stack(aligned_frames), axis=0).astype(np.float32)


# ------------------------------------------------------------------ HDR
def merge_hdr(images: list[np.ndarray], contrast: float = 1.0, saturation: float = 1.0,
              well_exposed: float = 1.0) -> np.ndarray:
    """Fusion d'expositions (Mertens) en flottant 32 bits (aucune
    quantification 8 bits). Les images doivent partager la même taille."""
    cv2 = __import__("cv2")
    if len(images) < 2:
        raise ValueError("HDR : fournir au moins 2 expositions")
    imgs = [np.ascontiguousarray(np.clip(np.asarray(im, np.float32), 0, 1)) for im in images]
    if any(im.shape != imgs[0].shape for im in imgs):
        raise ValueError("HDR : toutes les expositions doivent avoir la même taille")
    mono = imgs[0].ndim == 2
    if mono:
        imgs = [np.repeat(im[..., None], 3, axis=2) for im in imgs]
    try:
        out = cv2.createMergeMertens(contrast, saturation, well_exposed).process(imgs)
    except cv2.error:
        # repli : moyenne pondérée par la « bonne exposition » (gaussienne autour de 0,5)
        ws = [np.exp(-((im - 0.5) ** 2) / (2 * 0.2 ** 2)) + 1e-6 for im in imgs]
        out = sum(w * im for w, im in zip(ws, imgs)) / sum(ws)
    out = np.clip(np.asarray(out, np.float32), 0, 1)
    return out.mean(axis=2) if mono else out


# ------------------------------------------------------------- fausses couleurs
_SOLAR_RAMP = np.array([
    [0, 0, 0], [60, 20, 0], [140, 60, 0], [210, 120, 0],
    [255, 180, 20], [255, 230, 120], [255, 255, 230], [255, 255, 255]
], np.float32) / 255.0


def _lut_from_mpl(name: str) -> np.ndarray:
    """Palette matplotlib (API matplotlib.colormaps ≥ 3.5 ; cm.get_cmap a été
    supprimé en 3.9)."""
    import matplotlib
    key = name.lower()
    try:
        cmap = matplotlib.colormaps[key]
    except (AttributeError, KeyError):
        import matplotlib.cm as cm
        cmap = cm.get_cmap(key)
    return (cmap(np.linspace(0, 1, 256))[:, :3]).astype(np.float32)


PALETTES = ["Solaire (orange)", "Inferno", "Magma", "Plasma", "Viridis", "Turbo", "Jet"]


def apply_palette(mono: np.ndarray, palette: str = "Solaire (orange)") -> np.ndarray:
    """Applique une palette de fausses couleurs à une image mono (ex. Soleil H-alpha)."""
    a = np.asarray(mono, np.float32)
    if a.ndim == 3:
        a = a.mean(axis=2)
    norm = a - a.min()
    norm = norm / max(norm.max(), 1e-9)
    if palette.startswith("Solaire"):
        lut = _SOLAR_RAMP
    else:
        try:
            lut = _lut_from_mpl(palette)
        except (KeyError, ValueError, ImportError) as e:
            raise ValueError(f"Palette inconnue : {palette}") from e
    idx = np.clip((norm * 255).astype(np.int32), 0, 255)
    if len(lut) != 256:
        xs = np.linspace(0, 1, len(lut))
        lut = np.stack([np.interp(np.linspace(0, 1, 256), xs, lut[:, c])
                        for c in range(lut.shape[1])], axis=1).astype(np.float32)
    return lut[idx]
