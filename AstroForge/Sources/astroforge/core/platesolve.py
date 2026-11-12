# -*- coding: utf-8 -*-
"""
Solveur astrométrique par triangles (invariants de forme), plus robuste que
l'appariement par paires d'étoiles brillantes :

1. triangles formés avec les étoiles les plus brillantes de l'image et du
   catalogue (projeté sur le plan tangent) ; descripteur invariant par
   translation, rotation, échelle ET symétrie : rapports des côtés
   (b/c, a/c) triés ;
2. recherche des triangles semblables (arbre k-d), chaque correspondance
   donne une similitude (Umeyama, réflexion permise) ; vote par le nombre
   d'étoiles de l'image retrouvées dans le catalogue ;
3. affinement sur toutes les étoiles appariées, puis WCS TAN, avec
   correction de distorsion SIP (astropy) si assez d'étoiles.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


def _triangles(pts: np.ndarray, k: int):
    """Triangles des k premiers points : (descripteurs (T,2), indices (T,3)
    ordonnés par côté opposé croissant)."""
    k = min(k, len(pts))
    desc, idx = [], []
    for i, j, l in combinations(range(k), 3):
        tri = (i, j, l)
        p = pts[list(tri)]
        # côté opposé à chaque sommet
        sides = np.array([np.linalg.norm(p[1] - p[2]), np.linalg.norm(p[0] - p[2]),
                          np.linalg.norm(p[0] - p[1])])
        order = np.argsort(sides)
        a, b, c = sides[order]
        if c < 1e-9 or a / c < 0.05:
            continue  # triangle dégénéré (presque aligné)
        desc.append((a / c, b / c))
        idx.append([tri[o] for o in order])
    return np.asarray(desc, np.float64), np.asarray(idx, np.int64)


def umeyama(src: np.ndarray, dst: np.ndarray, allow_reflection: bool = True):
    """Similitude dst ≈ s·R·src + t (moindres carrés). Retourne (M 2×2, t)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S0, D0 = src - mu_s, dst - mu_d
    U, sv, Vt = np.linalg.svd(D0.T @ S0)
    d = np.ones(2)
    if not allow_reflection and np.linalg.det(U @ Vt) < 0:
        d[-1] = -1
    R = U @ np.diag(d) @ Vt
    var = float((S0 ** 2).sum())
    s = float((sv * d).sum() / max(var, 1e-12))
    M = s * R
    return M, mu_d - M @ mu_s


def _inliers(M, t, stars, sky, tol):
    q = stars @ M.T + t
    d = np.hypot(q[:, None, 0] - sky[None, :, 0], q[:, None, 1] - sky[None, :, 1])
    j = np.argmin(d, axis=1)
    dist = d[np.arange(len(q)), j]
    ok = dist < tol
    # unicité : une étoile du catalogue pour une seule étoile de l'image
    seen = {}
    for i in np.nonzero(ok)[0]:
        if j[i] in seen and dist[seen[j[i]]] <= dist[i]:
            ok[i] = False
        else:
            if j[i] in seen:
                ok[seen[j[i]]] = False
            seen[j[i]] = i
    return ok, j, dist


def solve_triangles(img: np.ndarray, catalog_path: str, center_ra: float, center_dec: float,
                    radius_deg: float = 1.0, scale_hint: float | None = None,
                    n_img: int = 40, n_cat: int = 120, k_img: int = 14, k_cat: int = 24,
                    sip_degree: int | None = 3):
    """Plate solving par triangles. scale_hint : échelle approximative en
    "/px (±50 %), facultative. Retourne le même dict que solve_internal
    (+ "sip")."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from scipy.spatial import cKDTree
    from .astrometry import _tangent_xy, detect_stars, load_catalog
    h, w = img.shape[:2]
    stars = detect_stars(img, n_img)
    if len(stars) < 5:
        raise RuntimeError("Moins de 5 étoiles détectées — image non résoluble")
    tab = load_catalog(catalog_path)
    ra = np.asarray(tab["ra"], float)
    dec = np.asarray(tab["dec"], float)
    mags = np.asarray(tab["mag"], float) if "mag" in tab.colnames else np.zeros(len(ra))
    center = SkyCoord(center_ra * u.deg, center_dec * u.deg)
    ok = center.separation(SkyCoord(ra * u.deg, dec * u.deg)).deg <= radius_deg
    ra, dec, mags = ra[ok], dec[ok], mags[ok]
    if len(ra) < 5:
        raise RuntimeError("Moins de 5 étoiles de catalogue dans le rayon")
    order = np.argsort(np.nan_to_num(mags, nan=99))[:n_cat]
    ra, dec, mags = ra[order], dec[order], mags[order]
    sky = _tangent_xy(ra, dec, center)
    di, ii = _triangles(stars, k_img)
    dc, ic = _triangles(sky, k_cat)
    if not len(di) or not len(dc):
        raise RuntimeError("Pas assez de triangles exploitables")
    tree = cKDTree(dc)
    diag = float(np.hypot(w, h))
    best = None
    for t_i, (d, tri) in enumerate(zip(di, ii)):
        for c_i in tree.query_ball_point(d, r=0.008):
            src = stars[tri]
            dst = sky[ic[c_i]]
            M, t = umeyama(src, dst)
            sc = float(np.sqrt(abs(np.linalg.det(M))))
            if scale_hint and not (0.5 < sc / scale_hint < 1.5):
                continue
            if sc * diag / 3600.0 > 4 * radius_deg + 1:  # champ incompatible
                continue
            okm, _j, _d = _inliers(M, t, stars, sky, max(3.0 * sc, 2.0))
            n = int(okm.sum())
            if best is None or n > best[0]:
                best = (n, M, t)
        if best is not None and best[0] >= max(12, int(0.6 * len(stars))):
            break
    if best is None or best[0] < 5:
        raise RuntimeError("Plate solving par triangles : aucun appariement fiable")
    _, M, t = best
    for _ in range(3):  # affinement Umeyama sur toutes les étoiles appariées
        sc = float(np.sqrt(abs(np.linalg.det(M))))
        okm, j, dist = _inliers(M, t, stars, sky, max(2.5 * sc, 1.5))
        if okm.sum() < 4:
            break
        M, t = umeyama(stars[okm], sky[j[okm]])
    sc = float(np.sqrt(abs(np.linalg.det(M))))
    okm, j, dist = _inliers(M, t, stars, sky, max(2.0 * sc, 1.5))
    from astropy.wcs import WCS
    pc = -(np.linalg.inv(M) @ t)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [float(pc[0]) + 1.0, float(pc[1]) + 1.0]
    wcs.wcs.crval = [float(center.ra.deg), float(center.dec.deg)]
    wcs.wcs.cd = (M / 3600.0).tolist()
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    sip_used = False
    n_in = int(okm.sum())
    pairs = (np.nonzero(okm)[0], j[okm])
    if sip_degree:
        from astropy.wcs.utils import fit_wcs_from_points
        cat_world = SkyCoord(ra * u.deg, dec * u.deg)
        # appariement initial large (distorsion non encore modélisée)
        sc = float(np.sqrt(abs(np.linalg.det(M))))
        ok0, j0, _d0 = _inliers(M, t, stars, sky, max(6.0 * sc, 3.0))
        if ok0.sum() > len(pairs[0]):
            pairs = (np.nonzero(ok0)[0], j0[ok0])
        for deg in range(2, int(sip_degree) + 1):  # degré croissant
            nterm = (deg + 1) * (deg + 2) // 2
            for _ in range(4):  # ajustement SIP ↔ ré-appariement
                ii_, jj_ = pairs
                if len(ii_) < 2 * nterm + 4:
                    break
                try:
                    w2 = fit_wcs_from_points((stars[ii_, 0], stars[ii_, 1]), cat_world[jj_],
                                             projection="TAN", sip_degree=deg)
                except Exception:
                    break
                px = np.stack(w2.world_to_pixel(cat_world), 1)
                d = np.hypot(stars[:, None, 0] - px[None, :, 0],
                             stars[:, None, 1] - px[None, :, 1])
                jn = np.argmin(d, axis=1)
                dn = d[np.arange(len(stars)), jn]
                okn = dn < 2.0
                if okn.sum() >= len(ii_) * 0.8 and \
                        _resid(w2, stars[ii_], cat_world[jj_]) <= _resid(wcs, stars[ii_], cat_world[jj_]):
                    wcs, sip_used = w2, True
                new = (np.nonzero(okn)[0], jn[okn])
                if len(new[0]) < len(ii_) * 0.8:
                    break
                same = len(new[0]) == len(ii_) and np.array_equal(new[0], ii_)
                pairs = new
                if same:
                    break
        n_in = len(pairs[0])
    okm = np.zeros(len(stars), bool)
    okm[pairs[0]] = True
    j = np.zeros(len(stars), int)
    j[pairs[0]] = pairs[1]
    world = SkyCoord(ra[j[okm]] * u.deg, dec[j[okm]] * u.deg)
    rms = _resid(wcs, stars[okm], world)
    c = wcs.pixel_to_world(w / 2.0, h / 2.0)
    return {"wcs": wcs, "scale_arcsec_pix": sc, "n_matched": n_in, "rms_arcsec": rms,
            "center_ra": float(c.ra.deg), "center_dec": float(c.dec.deg),
            "method": "triangles" + (" + SIP" if sip_used else ""), "sip": sip_used}


def _resid(wcs, xy, world) -> float:
    p = wcs.world_to_pixel(world)
    sc = float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600.0
    d = np.hypot(p[0] - xy[:, 0], p[1] - xy[:, 1]) * sc
    return float(np.sqrt(np.mean(d ** 2))) if len(d) else -1.0
