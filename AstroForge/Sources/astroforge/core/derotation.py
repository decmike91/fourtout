# -*- coding: utf-8 -*-
"""
Planétaire : disque, dérotation de séries d'images (esprit WinJUPOS) et
export au format de noms WinJUPOS.

- `fit_disc` : centre et rayon du disque (contour ajusté par moindres
  carrés sur un cercle, points aberrants rejetés) — sert aussi au limbe
  solaire.
- `derotate_image` : reprojection d'une image prise à l'instant t vers
  l'instant t0 : chaque point visible du disque (sphère aplatie vue en
  projection orthographique, axe polaire incliné de l'angle de position PA,
  latitude du centre B) est retrouvé à la longitude λ − ω·(t0 − t).
- `derotate_series` : moyenne pondérée de plusieurs images ramenées au
  même instant (poids = cos de l'angle d'émission, bords exclus) : plus de
  signal sans le flou dû à la rotation de la planète.
- `winjupos_name` : « AAAA-MM-JJ-HHMM_D-Observateur-Filtre.png » (D =
  dixième de minute), nom reconnu par WinJUPOS pour la date d'acquisition.
"""
from __future__ import annotations

import datetime as _dt

import numpy as np

# période de rotation (heures) et aplatissement
PLANETS = {
    "Jupiter (Système II)": (9.0 + 55.0 / 60 + 40.632 / 3600, 0.06487),
    "Jupiter (Système I)": (9.0 + 50.0 / 60 + 30.0034 / 3600, 0.06487),
    "Saturne": (10.0 + 33.0 / 60 + 38.0 / 3600, 0.09796),
    "Mars": (24.0 + 37.0 / 60 + 22.66 / 3600, 0.00589),
    "Vénus": (-5832.5, 0.0),
    "Lune (libration ignorée)": (655.72, 0.0012),
}


# ------------------------------------------------------------------ disque
def fit_disc(img: np.ndarray, threshold: float | None = None) -> tuple[float, float, float]:
    """(cx, cy, r) du disque : seuil (Otsu par défaut) puis ajustement d'un
    cercle (moindres carrés algébriques de Kåsa) sur les points de contour,
    avec rejet itératif des points aberrants."""
    import cv2
    import scipy.ndimage as ndi
    a = np.asarray(img, np.float32)
    mono = a if a.ndim == 2 else a[..., :3].mean(axis=2)
    sm = ndi.gaussian_filter(mono, 1.5)
    if threshold is None:
        u8 = (np.clip(sm / max(float(sm.max()), 1e-9), 0, 1) * 255).astype(np.uint8)
        t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold = float(t) / 255.0 * float(sm.max())
    mask = sm > threshold
    lab, n = ndi.label(mask)
    if n == 0:
        raise ValueError("Aucun disque détecté")
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    mask = ndi.binary_fill_holes(lab == (int(np.argmax(sizes)) + 1))
    edge = mask & ~ndi.binary_erosion(mask)
    ys, xs = np.nonzero(edge)
    if len(xs) < 10:
        raise ValueError("Contour du disque insuffisant")
    keep = np.ones(len(xs), bool)
    for _ in range(4):
        x, y = xs[keep].astype(float), ys[keep].astype(float)
        A = np.stack([x, y, np.ones_like(x)], 1)
        b = x * x + y * y
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = c[0] / 2, c[1] / 2
        r = float(np.sqrt(max(c[2] + cx * cx + cy * cy, 1e-9)))
        res = np.abs(np.hypot(xs - cx, ys - cy) - r)
        new = res < max(2.0, 3 * 1.4826 * float(np.median(res[keep])))
        if new.sum() < 10 or np.array_equal(new, keep):
            break
        keep = new
    return float(cx), float(cy), float(r)


# --------------------------------------------------------------- géométrie
def _rot(pa_deg):
    t = np.radians(pa_deg)
    return np.cos(t), np.sin(t)


def _image_to_sphere(x, y, cx, cy, r, pa, flat, B):
    """Pixels -> (lon, lat, visible, cos_e) sur le globe à l'instant de
    l'image. PA : angle de position du pôle nord (degrés, compté depuis le
    haut de l'image vers la gauche), B : latitude du centre (degrés)."""
    c, s = _rot(pa)
    dx, dy = (x - cx) / r, (cy - y) / r          # y vers le haut
    u = c * dx - s * dy                          # repère « pôle en haut »
    v = s * dx + c * dy
    v = v / (1.0 - flat)                         # aplatissement (approx. ortho)
    rho2 = u * u + v * v
    vis = rho2 < 1.0
    z = np.sqrt(np.clip(1.0 - rho2, 0, 1))      # vers l'observateur
    b = np.radians(B)
    # rotation autour de l'axe x (latitude du centre)
    yy = v * np.cos(b) + z * np.sin(b)
    zz = -v * np.sin(b) + z * np.cos(b)
    lat = np.arcsin(np.clip(yy, -1, 1))
    lon = np.arctan2(u, zz)
    return lon, lat, vis, z


def _sphere_to_image(lon, lat, cx, cy, r, pa, flat, B):
    u = np.cos(lat) * np.sin(lon)
    yy = np.sin(lat)
    zz = np.cos(lat) * np.cos(lon)
    b = np.radians(B)
    v = yy * np.cos(b) - zz * np.sin(b)
    z = yy * np.sin(b) + zz * np.cos(b)
    v = v * (1.0 - flat)
    c, s = _rot(pa)
    dx = c * u + s * v
    dy = -s * u + c * v
    return cx + dx * r, cy - dy * r, z


def derotate_image(img, disc, dt_seconds: float, period_hours: float, pa: float = 0.0,
                   flat: float = 0.0, B: float = 0.0, direction: int = 1,
                   limb_cut: float = 0.15):
    """Ramène `img` (prise à t) à l'instant t0 = t + dt_seconds. Retourne
    (image, poids) ; poids = cos(angle d'émission) dans l'image source,
    nul près du limbe (cos < limb_cut) et hors du disque."""
    import cv2
    a = np.asarray(img, np.float32)
    h, w = a.shape[:2]
    cx, cy, r = disc
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    lon, lat, vis, z0 = _image_to_sphere(xx, yy, cx, cy, r, pa, flat, B)
    dlon = direction * 2 * np.pi * dt_seconds / (period_hours * 3600.0)
    sx, sy, zs = _sphere_to_image(lon - dlon, lat, cx, cy, r, pa, flat, B)
    ok = vis & (zs > limb_cut) & (z0 > 0)
    out = cv2.remap(a, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REFLECT)
    wgt = np.where(ok, np.clip(zs, 0, 1), 0.0).astype(np.float32)
    # hors disque : l'image telle quelle (fond), poids faible
    outside = ~vis
    if a.ndim == 3:
        out[outside] = a[outside]
    else:
        out[outside] = a[outside]
    wgt[outside] = 1e-3
    return out.astype(np.float32), wgt


def derotate_series(images, times: list, t0=None, planet: str = "Jupiter (Système II)",
                    pa: float = 0.0, B: float = 0.0, direction: int = 1,
                    period_hours: float | None = None, flat: float | None = None,
                    progress=None):
    """Combine des images prises à des instants différents en une seule,
    ramenée à l'instant t0 (milieu de la série par défaut). `times` :
    datetimes ou chaînes ISO. Retourne (image, instant t0 ISO, disque)."""
    if len(images) != len(times) or not images:
        raise ValueError("Une date par image est requise")
    ts = [t if isinstance(t, _dt.datetime) else _dt.datetime.fromisoformat(str(t))
          for t in times]
    ts = [t.replace(tzinfo=None) if t.tzinfo is None else
          t.astimezone(_dt.timezone.utc).replace(tzinfo=None) for t in ts]
    if t0 is None:
        mid = min(ts) + (max(ts) - min(ts)) / 2
    else:
        mid = t0 if isinstance(t0, _dt.datetime) else _dt.datetime.fromisoformat(str(t0))
    P, f0 = PLANETS.get(planet, (9.925, 0.0649))
    P = period_hours if period_hours else P
    f = flat if flat is not None else f0
    ref_i = int(np.argmin([abs((t - mid).total_seconds()) for t in ts]))
    from .registration import apply_shift
    cx_, cy_, r_ = fit_disc(images[ref_i])
    disc_ref = (cx_, cy_, r_ / (1.0 - f / 2.0))  # rayon moyen -> rayon équatorial
    acc = None
    wacc = None
    for i, (img, t) in enumerate(zip(images, ts)):
        a = np.asarray(img, np.float32)
        d = fit_disc(a)
        # centrage sur le disque de référence (dérive de suivi)
        a = apply_shift(a, disc_ref[0] - d[0], disc_ref[1] - d[1])
        out, wgt = derotate_image(a, disc_ref, (mid - t).total_seconds(), P, pa, f, B,
                                  direction)
        wb = wgt[..., None] if out.ndim == 3 else wgt
        acc = out * wb if acc is None else acc + out * wb
        wacc = wgt if wacc is None else wacc + wgt
        if progress:
            progress(i + 1, len(images), "Dérotation")
    res = acc / (np.maximum(wacc, 1e-9)[..., None] if acc.ndim == 3 else np.maximum(wacc, 1e-9))
    return np.clip(res, 0, 1).astype(np.float32), mid.isoformat(timespec="seconds"), disc_ref


# ----------------------------------------------------------------- WinJUPOS
def winjupos_name(when, observer: str = "", filt: str = "", ext: str = "png") -> str:
    """Nom de fichier WinJUPOS : AAAA-MM-JJ-HHMM_D[-Observateur][-Filtre].ext,
    D = dixièmes de minute, heure UT."""
    d = when if isinstance(when, _dt.datetime) else _dt.datetime.fromisoformat(str(when))
    if d.tzinfo is not None:
        d = d.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    tenth = int((d.second + d.microsecond / 1e6) / 6.0)
    parts = [d.strftime("%Y-%m-%d-%H%M") + f"_{tenth}"]
    for extra in (observer, filt):
        clean = "".join(ch for ch in str(extra) if ch.isalnum() or ch in "+._")
        if clean:
            parts.append(clean)
    return "-".join(parts) + "." + ext.lstrip(".")


def parse_winjupos_name(name: str):
    """Instant UT encodé dans un nom WinJUPOS, ou None."""
    import os
    import re
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})-(\d{2})(\d{2})_(\d)", os.path.basename(name))
    if not m:
        return None
    y, mo, d, hh, mi, tenth = (int(g) for g in m.groups())
    return _dt.datetime(y, mo, d, hh, mi) + _dt.timedelta(seconds=6 * tenth)


def mid_time(timestamps: list) -> str | None:
    """Instant médian d'une liste d'horodatages ISO (frames retenues)."""
    ts = [_dt.datetime.fromisoformat(t) for t in timestamps if t]
    if not ts:
        return None
    ts.sort()
    return (ts[0] + (ts[-1] - ts[0]) / 2).isoformat(timespec="milliseconds")
