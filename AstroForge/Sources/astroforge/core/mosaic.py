# -*- coding: utf-8 -*-
"""
Mosaïques : assemblage de panneaux qui se recouvrent.

- `mosaic_wcs` : chaque panneau possède une astrométrie (WCS dans l'en-tête
  FITS, issue d'astrometry.net ou du plate solving d'AstroForge) ; les
  panneaux sont reprojetés sur une projection tangente commune (nord en
  haut), à l'échelle la plus fine.
- `mosaic_features` : sans astrométrie (mosaïque solaire/lunaire, champs
  peu étoilés) : appariement de points caractéristiques (SIFT, ORB en
  repli) et similitude RANSAC, panneau par panneau.

Dans les deux cas : égalisation du fond dans les zones de recouvrement et
fondu progressif (pondération par la distance au bord du panneau) pour des
raccords invisibles.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi


def _feather(valid: np.ndarray, width: float = 64.0) -> np.ndarray:
    """Poids 0→1 croissant depuis le bord de la zone valide."""
    d = ndi.distance_transform_edt(valid)
    return np.clip(d / max(width, 1.0), 0, 1).astype(np.float32) ** 1.5


def _blend_into(acc, wacc, img, valid, match_bg=True, feather=64.0):
    """Ajoute un panneau reprojeté à l'accumulateur (fond égalisé sur la
    zone de recouvrement)."""
    w = _feather(valid, feather)
    a = np.asarray(img, np.float32)
    if match_bg:
        cur = acc / np.maximum(wacc, 1e-9)[..., None] if acc.ndim == 3 else acc / np.maximum(wacc, 1e-9)
        ov = valid & (wacc > 1e-6)
        if ov.sum() > 100:
            if a.ndim == 3:
                for c in range(a.shape[2]):
                    a[..., c] += np.float32(np.median(cur[..., c][ov]) - np.median(a[..., c][ov]))
            else:
                a = a + np.float32(np.median(cur[ov]) - np.median(a[ov]))
    if a.ndim == 3:
        acc += a * w[..., None]
    else:
        acc += a * w
    wacc += w


def _finish(acc, wacc):
    out = acc / (np.maximum(wacc, 1e-9)[..., None] if acc.ndim == 3 else np.maximum(wacc, 1e-9))
    return np.clip(out, 0, 1).astype(np.float32), (wacc > 1e-6)


# ------------------------------------------------------------- par WCS
def _unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def output_wcs(wcs_list, shapes, scale_deg: float | None = None):
    """Projection TAN commune couvrant tous les panneaux. Retourne
    (WCS, (hauteur, largeur))."""
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales
    centers = []
    for w, (h, wd) in zip(wcs_list, shapes):
        ra, dec = w.wcs_pix2world([[wd / 2.0, h / 2.0]], 0)[0]
        centers.append(_unit(ra, dec))
    v = np.mean(centers, axis=0)
    v /= np.linalg.norm(v)
    ra0 = float(np.degrees(np.arctan2(v[1], v[0])) % 360)
    dec0 = float(np.degrees(np.arcsin(np.clip(v[2], -1, 1))))
    if scale_deg is None:
        scale_deg = float(min(np.mean(proj_plane_pixel_scales(w)) for w in wcs_list))
    out = WCS(naxis=2)
    out.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    out.wcs.crval = [ra0, dec0]
    out.wcs.cd = [[-scale_deg, 0.0], [0.0, scale_deg]]
    out.wcs.crpix = [1.0, 1.0]
    xs, ys = [], []
    for w, (h, wd) in zip(wcs_list, shapes):
        corners = np.array([[0, 0], [wd - 1, 0], [0, h - 1], [wd - 1, h - 1],
                            [wd / 2, 0], [wd / 2, h - 1], [0, h / 2], [wd - 1, h / 2]], float)
        world = w.wcs_pix2world(corners, 0)
        px = out.wcs_world2pix(world, 0)
        xs += list(px[:, 0])
        ys += list(px[:, 1])
    x0, y0 = np.floor(min(xs)), np.floor(min(ys))
    width = int(np.ceil(max(xs) - x0)) + 2
    height = int(np.ceil(max(ys) - y0)) + 2
    out.wcs.crpix = [1.0 - x0 + 1, 1.0 - y0 + 1]
    if width * height > 400e6:
        raise ValueError(f"Mosaïque trop grande ({width}×{height}) — vérifiez les WCS")
    return out, (height, width)


def reproject(img, wcs_in, wcs_out, shape_out, rows: int = 256):
    """Reprojection (interpolation bicubique) de img sur la grille de sortie.
    Retourne (image, masque de validité)."""
    import cv2
    a = np.asarray(img, np.float32)
    h, w = shape_out
    out = np.zeros((h, w) + a.shape[2:], np.float32)
    valid = np.zeros((h, w), bool)
    xx = np.arange(w, dtype=np.float64)
    for y0 in range(0, h, rows):
        y1 = min(h, y0 + rows)
        gx, gy = np.meshgrid(xx, np.arange(y0, y1, dtype=np.float64))
        world = wcs_out.wcs_pix2world(np.stack([gx.ravel(), gy.ravel()], 1), 0)
        px = wcs_in.wcs_world2pix(world, 0)
        mx = px[:, 0].reshape(gx.shape).astype(np.float32)
        my = px[:, 1].reshape(gx.shape).astype(np.float32)
        ok = (mx >= 0) & (my >= 0) & (mx <= a.shape[1] - 1) & (my <= a.shape[0] - 1)
        out[y0:y1] = cv2.remap(a, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        valid[y0:y1] = ok
    return out, valid


def mosaic_wcs(images, wcs_list, scale_deg: float | None = None, feather: float = 64.0,
               progress=None):
    """Mosaïque par astrométrie. Retourne (image, WCS de sortie, couverture)."""
    if len(images) != len(wcs_list) or not images:
        raise ValueError("Un WCS par panneau est requis")
    shapes = [np.asarray(i).shape[:2] for i in images]
    wout, shape = output_wcs(wcs_list, shapes, scale_deg)
    chans = np.asarray(images[0]).shape[2:] if np.asarray(images[0]).ndim == 3 else ()
    acc = np.zeros(shape + chans, np.float32)
    wacc = np.zeros(shape, np.float32)
    for i, (img, w) in enumerate(zip(images, wcs_list)):
        rep, valid = reproject(img, w, wout, shape)
        _blend_into(acc, wacc, rep, valid, match_bg=i > 0, feather=feather)
        if progress:
            progress(i + 1, len(images), "Reprojection des panneaux")
    out, cov = _finish(acc, wacc)
    return out, wout, cov


def wcs_from_fits(path: str):
    """WCS céleste de l'en-tête d'un FITS, ou None."""
    from astropy.io import fits
    from astropy.wcs import WCS
    try:
        with fits.open(path, memmap=False) as h:
            for hdu in h:
                if hdu.header.get("NAXIS", 0) >= 2 and "CTYPE1" in hdu.header:
                    w = WCS(hdu.header).celestial
                    return w if w.has_celestial else None
    except Exception:
        return None
    return None


# ---------------------------------------------------- par points d'intérêt
def _u8(img):
    from .display import autostretch
    a = np.asarray(img, np.float32)
    mono = a if a.ndim == 2 else a[..., :3].mean(axis=2)
    return (np.clip(autostretch(mono), 0, 1) * 255).astype(np.uint8)


def _features(u8):
    import cv2
    try:
        det = cv2.SIFT_create(nfeatures=4000)
        norm = cv2.NORM_L2
    except AttributeError:  # OpenCV sans SIFT
        det = cv2.ORB_create(nfeatures=5000)
        norm = cv2.NORM_HAMMING
    kp, des = det.detectAndCompute(u8, None)
    return kp, des, norm


def match_similarity(img_a, img_b, min_matches: int = 12):
    """Similitude (2×3) envoyant les coordonnées de img_b sur img_a, ou None."""
    import cv2
    ka, da, norm = _features(_u8(img_a))
    kb, db, _ = _features(_u8(img_b))
    if da is None or db is None or len(ka) < min_matches or len(kb) < min_matches:
        return None
    m = cv2.BFMatcher(norm).knnMatch(db, da, k=2)
    good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < min_matches:
        return None
    src = np.float32([kb[g.queryIdx].pt for g in good])
    dst = np.float32([ka[g.trainIdx].pt for g in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                         ransacReprojThreshold=2.0, maxIters=5000)
    if M is None or inl is None or int(inl.sum()) < min_matches:
        return None
    return M.astype(np.float64)


def mosaic_features(images, feather: float = 48.0, progress=None):
    """Mosaïque sans astrométrie : chaque panneau est recalé (similitude
    RANSAC sur points SIFT) sur la mosaïque en cours. Retourne (image,
    couverture, matrices)."""
    import cv2
    imgs = [np.asarray(i, np.float32) for i in images]
    if not imgs:
        raise ValueError("Aucun panneau")
    # transformations vers le repère du premier panneau
    mats = [np.array([[1.0, 0, 0], [0, 1.0, 0]])]
    placed = [0]
    pending = list(range(1, len(imgs)))
    while pending:
        progress_made = False
        for j in list(pending):
            for i in placed:
                M = match_similarity(imgs[i], imgs[j])
                if M is None:
                    continue
                Mi = np.vstack([mats[placed.index(i)], [0, 0, 1]])
                mats.append((Mi @ np.vstack([M, [0, 0, 1]]))[:2])
                placed.append(j)
                pending.remove(j)
                progress_made = True
                break
        if not progress_made:
            raise ValueError(f"Panneau(x) sans recouvrement exploitable : "
                             f"{', '.join(str(p + 1) for p in pending)}")
    order = {p: k for k, p in enumerate(placed)}
    mats_ordered = [mats[order[i]] for i in range(len(imgs))]
    corners = []
    for img, M in zip(imgs, mats_ordered):
        h, w = img.shape[:2]
        c = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], float)
        corners.append(c @ M.T)
    allc = np.vstack(corners)
    x0, y0 = np.floor(allc.min(axis=0))
    W = int(np.ceil(allc[:, 0].max() - x0))
    H = int(np.ceil(allc[:, 1].max() - y0))
    if W * H > 400e6:
        raise ValueError("Mosaïque trop grande — recouvrements incohérents ?")
    chans = imgs[0].shape[2:] if imgs[0].ndim == 3 else ()
    acc = np.zeros((H, W) + chans, np.float32)
    wacc = np.zeros((H, W), np.float32)
    for k, (img, M) in enumerate(zip(imgs, mats_ordered)):
        T = M.copy()
        T[0, 2] -= x0
        T[1, 2] -= y0
        rep = cv2.warpAffine(img, T, (W, H), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT)
        valid = cv2.warpAffine(np.ones(img.shape[:2], np.uint8), T, (W, H),
                               flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
        valid = ndi.binary_erosion(valid, iterations=2)
        _blend_into(acc, wacc, rep, valid, match_bg=k > 0, feather=feather)
        if progress:
            progress(k + 1, len(imgs), "Assemblage")
    out, cov = _finish(acc, wacc)
    return out, cov, mats_ordered
