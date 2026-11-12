# -*- coding: utf-8 -*-
"""
Astrométrie locale : plate solving, WCS, alignement par étoiles, photométrie.
Solveurs :
- « solve-field » (astrometry.net) s'il est installé sur la machine (solveur
  local de référence, index files téléchargées une fois pour toutes) ;
- solveur interne : appariement de motifs d'étoiles contre un catalogue local
  (CSV avec colonnes ra, dec, mag — en degrés ; exportable une fois via
  astroquery/Gaia, procédure dans le README).
Photométrie : ouvertures circulaires + couronne de fond, zero point calibré sur
les étoiles du catalogue. Dépendances optionnelles : photutils, astroalign.
"""
from __future__ import annotations
import os
import shutil
import tempfile

import numpy as np

try:
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    import astropy.units as u
    _ASTROPY = True
except Exception:
    _ASTROPY = False


# ------------------------------------------------------------------ détection
def detect_stars(img: np.ndarray, n_max: int = 60, fwhm: float = 3.0,
                 threshold_sigma: float = 8.0) -> np.ndarray:
    """Détecte les étoiles, triées du plus brillant au plus faible -> (N,2) x,y."""
    mono = img if img.ndim == 2 else img[..., :3].mean(axis=2)
    data = np.asarray(mono, np.float64)
    data = data - np.median(data)
    mad = 1.4826 * np.median(np.abs(data)) + 1e-9
    try:
        from photutils.detection import DAOStarFinder
        finder = DAOStarFinder(fwhm=fwhm, threshold=threshold_sigma * mad)
        tab = finder(data)
        if tab is None or len(tab) == 0:
            return np.empty((0, 2))
        order = np.argsort(-np.asarray(tab["flux"]))
        cols = tab.colnames
        xc = "x_centroid" if "x_centroid" in cols else "xcentroid"   # photutils ≥ 3 / < 3
        yc = "y_centroid" if "y_centroid" in cols else "ycentroid"
        return np.stack([np.asarray(tab[xc])[order],
                         np.asarray(tab[yc])[order]], axis=1)[:n_max]
    except ImportError:
        import scipy.ndimage as ndi
        mx = ndi.maximum_filter(data, size=max(3, int(2 * fwhm)))
        peaks = (data == mx) & (data > threshold_sigma * mad)
        ys, xs = np.nonzero(peaks)
        vals = data[ys, xs]
        order = np.argsort(-vals)
        return np.stack([xs[order], ys[order]], axis=1).astype(np.float64)[:n_max]


# ------------------------------------------------- alignement par étoiles
def align_by_stars(ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
    """Aligne `mov` sur `ref` par appariement d'étoiles (astroalign), repli
    automatique sur la corrélation de phase si échec."""
    try:
        import astroalign
        from ..core.image_io import is_color
        r = ref if ref.ndim == 2 else ref[..., :3].mean(axis=2)
        m = mov if mov.ndim == 2 else mov[..., :3].mean(axis=2)
        best = None
        for a_args in ((r, m), (m, r)):
            try:
                cand, mat = astroalign.register(*a_args, max_control_points=60)
                e = float(np.abs(cand - r).mean())
                if best is None or e < best[0]:
                    best = (e, cand, mat)
                if e < 1e-3:
                    break
            except Exception:
                continue
        if best is None:
            raise RuntimeError("astroalign : aucun appariement possible")
        aligned, matrix = best[1], best[2]
        if not is_color(mov):
            return aligned.astype(np.float32)
        cv2 = __import__("cv2")
        h, w = mov.shape[:2]
        M = matrix[:2, :]
        return cv2.warpAffine(mov, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    except Exception:
        pass
    try:
        from .registration import align_similarity
        return align_similarity(ref, mov)
    except Exception:
        pass
    from .registration import phase_corr, apply_shift
    dx, dy, _ = phase_corr(ref, mov)
    return apply_shift(np.asarray(mov, np.float32), dx, dy)


# ------------------------------------------------------------------ solveur
def load_catalog(path: str) -> "Table":
    tab = Table.read(path, format="csv")
    for col in ("ra", "dec"):
        if col not in tab.colnames:
            raise ValueError(f"Catalogue sans colonne « {col} » : {path}")
    return tab


def _tangent_xy(ra_deg: np.ndarray, dec_deg: np.ndarray, center) -> np.ndarray:
    """Projette (ra,dec) sur le plan tangent au centre -> arcsec (x,y)."""
    c = SkyCoord(ra_deg * u.deg, dec_deg * u.deg)
    off = c.transform_to(center.skyoffset_frame())
    return np.stack([off.lon.deg * 3600.0, off.lat.deg * 3600.0], axis=1)


def solve_internal(img: np.ndarray, catalog_path: str, center_ra: float,
                   center_dec: float, radius_deg: float = 1.0,
                   focal_mm: float | None = None, pixel_um: float | None = None,
                   max_stars: int = 30):
    """Solveur interne : appariement de motifs d'étoiles (2 hypothèses : directe
    et miroir, car l'axe Y image est inversé vs la déclinaison), affinement
    Procrustes (réflexion autorisée), puis construction du WCS TAN."""
    if not _ASTROPY:
        raise RuntimeError("astropy requis pour l'astrométrie")
    h, w = img.shape[:2]
    stars = detect_stars(img, max_stars)
    if len(stars) < 4:
        raise RuntimeError("Moins de 4 étoiles détectées — image non résoluble")
    tab = load_catalog(catalog_path)
    ra = np.asarray(tab["ra"], float)
    dec = np.asarray(tab["dec"], float)
    mags = np.asarray(tab["mag"], float) if "mag" in tab.colnames else np.zeros(len(ra))
    center = SkyCoord(center_ra * u.deg, center_dec * u.deg)
    sep = center.separation(SkyCoord(ra * u.deg, dec * u.deg)).deg
    ok = sep <= radius_deg
    ra, dec, mags = ra[ok], dec[ok], mags[ok]
    if len(ra) < 4:
        raise RuntimeError("Moins de 4 étoiles de catalogue dans le rayon — "
                           "élargissez radius_deg ou vérifiez le centre")
    sky = _tangent_xy(ra, dec, center)  # arcsec sur le plan tangent
    idx_cat = np.argsort(mags)[:40]
    sky = sky[idx_cat]

    diag = float(np.hypot(w, h))
    if focal_mm and pixel_um:
        s0 = 206.265 * pixel_um / focal_mm  # arcsec/pixel
    else:
        s0 = radius_deg * 3600.0 / (diag / 2.0) * 0.9
    F = np.array([[1.0, 0.0], [0.0, -1.0]])  # miroir vertical image<->ciel

    def hypothesis(pi, pj, si, sj, flip):
        v_p = (pj - pi) if not flip else (F @ (pj - pi))
        v_s = sj - si
        lp, ls = float(np.hypot(*v_p)), float(np.hypot(*v_s))
        if lp < 1e-6 or ls < 1e-6:
            return None
        sc = ls / lp
        if not (0.2 < sc / max(s0, 1e-9) < 5.0):
            return None
        ang = np.arctan2(v_s[1], v_s[0]) - np.arctan2(v_p[1], v_p[0])
        c_, s_ = np.cos(ang), np.sin(ang)
        Rm = np.array([[c_, -s_], [s_, c_]]) * sc
        M = Rm @ F if flip else Rm
        t = si - M @ pi
        q = (M @ stars.T).T + t
        d = np.hypot(q[:, None, 0] - sky[None, :, 0], q[:, None, 1] - sky[None, :, 1])
        idx = np.argmin(d, axis=1)
        dist = d[np.arange(len(q)), idx]
        inl = dist < max(3.0 * sc, 1.5)
        return (int(inl.sum()), M, t, idx, dist, inl)

    best = None
    ni, ns = min(5, len(stars)), min(12, len(sky))
    for flip in (False, True):
        for i in range(ni):
            for j in range(i + 1, ni):
                for a in range(ns):
                    for b in range(a + 1, ns):
                        r_ = hypothesis(stars[i], stars[j], sky[a], sky[b], flip)
                        if r_ and (best is None or r_[0] > best[0]):
                            best = r_
    if not best or best[0] < 4:
        raise RuntimeError("Plate solving interne échoué : pas d'appariement fiable "
                           "(vérifiez RA/Dec/rayon, ou installez astrometry.net)")
    _, M, t, idx, dist, inl = best
    scale = float(np.sqrt(abs(np.linalg.det(M))))

    def refine(inl_idx):
        p = stars[inl_idx]
        s_ = sky[idx[inl_idx]]
        mu_p, mu_s = p.mean(0), s_.mean(0)
        P0, S0 = p - mu_p, s_ - mu_s
        # Procrustes avec facteur d'échelle (Umeyama) : rotation R = V·Uᵀ
        # (réflexion autorisée : l'axe Y image est inversé vs la déclinaison),
        # échelle = Σ valeurs singulières / variance des points image
        U, Sv, Vt = np.linalg.svd(P0.T @ S0)
        Rm = Vt.T @ U.T
        sc_ = float(Sv.sum() / max(float((P0 ** 2).sum()), 1e-12))
        M2 = sc_ * Rm
        t2 = mu_s - M2 @ mu_p
        q = (M2 @ stars.T).T + t2
        d = np.hypot(q[:, None, 0] - sky[None, :, 0], q[:, None, 1] - sky[None, :, 1])
        idx2 = np.argmin(d, axis=1)
        dist2 = d[np.arange(len(q)), idx2]
        sc2 = float(np.sqrt(abs(np.linalg.det(M2))))
        inl2 = dist2 < max(2.0 * sc2, 1.5)
        if abs(sc2 / max(scale, 1e-9) - 1.0) > 0.2:  # garde-fou : échelle aberrante
            inl2 = np.zeros_like(inl2)
        return M2, t2, idx2, dist2, inl2, sc2

    for _ in range(3):
        sel_ = np.nonzero(inl)[0]
        if len(sel_) < 3:
            break
        M2, t2, idx2, dist2, inl2, sc2 = refine(sel_)
        if int(inl2.sum()) < int(inl.sum()):
            break
        M, t, idx, dist, inl, scale = M2, t2, idx2, dist2, inl2, sc2

    sel = np.nonzero(inl)[0]
    n_in = int(inl.sum())
    resid = float(np.sqrt(np.mean(dist[sel] ** 2))) if len(sel) else -1.0
    from astropy.wcs import WCS as _WCS
    pc = -(np.linalg.inv(M) @ t)  # pixel correspondant au centre tangent
    wcs = _WCS(naxis=2)
    wcs.wcs.crpix = [float(pc[0]) + 1.0, float(pc[1]) + 1.0]
    wcs.wcs.crval = [float(center.ra.deg), float(center.dec.deg)]
    wcs.wcs.cd = [[float(M[0, 0]) / 3600.0, float(M[0, 1]) / 3600.0],
                  [float(M[1, 0]) / 3600.0, float(M[1, 1]) / 3600.0]]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    center_out = wcs.pixel_to_world(w / 2.0, h / 2.0)
    return {"wcs": wcs, "scale_arcsec_pix": float(scale), "n_matched": n_in,
            "rms_arcsec": resid, "center_ra": float(center_out.ra.deg),
            "center_dec": float(center_out.dec.deg), "method": "interne"}


def solve_with_solvefield(fits_path: str, work_dir: str | None = None,
                          hint: tuple[float, float] | None = None,
                          radius_deg: float = 5.0):
    """Plate solving local via astrometry.net (binaire solve-field + index files
    installés localement). Retourne le même dict que solve_internal."""
    exe = shutil.which("solve-field")
    if not exe:
        raise RuntimeError("solve-field (astrometry.net) introuvable — installez-le "
                           "ou utilisez le solveur interne avec un catalogue local")
    d = work_dir or tempfile.mkdtemp(prefix="astroforge_astro_")
    base = os.path.join(d, os.path.splitext(os.path.basename(fits_path))[0])
    cmd = [exe, "--no-plots", "--overwrite", "--downsample", "2", "-D", d, fits_path]
    if hint:
        cmd = [exe, "--no-plots", "--overwrite", "--downsample", "2",
               "--ra", str(hint[0]), "--dec", str(hint[1]),
               "--radius", str(radius_deg), "-D", d, fits_path]
    import subprocess
    subprocess.run(cmd, check=True, capture_output=True, timeout=1200)
    out = base + ".new"
    if not os.path.exists(out):
        cands = [f for f in os.listdir(d) if f.endswith(".new")]
        if not cands:
            raise RuntimeError("solve-field n'a pas résolu le champ")
        out = os.path.join(d, cands[0])
    from astropy.io import fits as pf
    from astropy.wcs import WCS
    with pf.open(out) as hdul:
        wcs = WCS(hdul[0].header)
        h, w = hdul[0].data.shape[-2:]
    c = wcs.pixel_to_world(w / 2.0, h / 2.0)
    scale = float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix)))) * 3600.0
    return {"wcs": wcs, "scale_arcsec_pix": scale, "n_matched": -1,
            "rms_arcsec": -1.0, "center_ra": float(c.ra.deg),
            "center_dec": float(c.dec.deg), "method": "solve-field"}


def solve_image(img: np.ndarray, catalog_path: str | None = None,
                center_ra: float | None = None, center_dec: float | None = None,
                radius_deg: float = 1.0, focal_mm: float | None = None,
                pixel_um: float | None = None, prefer_solve_field: bool = True):
    """Plate solving : solve-field local si dispo, sinon solveur interne.
    `center_ra/dec` acceptent des degrés décimaux."""
    if prefer_solve_field and shutil.which("solve-field"):
        d = tempfile.mkdtemp(prefix="astroforge_solve_")
        from .image_io import save_image
        p = os.path.join(d, "solve_in.fits")
        save_image(p, np.asarray(img, np.float32))
        return solve_with_solvefield(p, d, hint=None if center_ra is None else
                                     (center_ra, center_dec), radius_deg=radius_deg)
    if not catalog_path:
        raise RuntimeError("Fournissez un catalogue local (CSV) ou installez "
                           "astrometry.net (solve-field) pour le plate solving")
    if center_ra is None or center_dec is None:
        raise RuntimeError("Le solveur interne a besoin d'un centre approximatif "
                           "(RA/Dec) et du rayon de recherche")
    from .platesolve import solve_triangles
    hint = 206.265 * pixel_um / focal_mm if focal_mm and pixel_um else None
    try:  # triangles (robuste, distorsions SIP) puis repli sur les paires
        return solve_triangles(img, catalog_path, center_ra, center_dec, radius_deg,
                               scale_hint=hint)
    except RuntimeError:
        return solve_internal(img, catalog_path, center_ra, center_dec, radius_deg,
                              focal_mm, pixel_um)


# ------------------------------------------------------------------ photométrie
def aperture_photometry(img: np.ndarray, wcs=None, catalog_path: str | None = None,
                        r: float = 6.0, r_in: float = 10.0, r_out: float = 16.0,
                        n_max: int = 150):
    """Photométrie d'ouverture des étoiles détectées. Si un catalogue + un WCS
    sont fournis, calibre le zero point et estime les magnitudes.
    Retourne une astropy Table (x, y, flux, mag)."""
    stars = detect_stars(img, n_max)
    if len(stars) == 0:
        raise RuntimeError("Aucune étoile détectée")
    mono = img if img.ndim == 2 else img[..., :3].mean(axis=2)
    data = np.asarray(mono, np.float64)
    try:
        from photutils.aperture import (ApertureStats, CircularAnnulus,
                                        CircularAperture)
        ap = CircularAperture(stars, r=r)
        an = CircularAnnulus(stars, r_in=r_in, r_out=r_out)
        s_ap = ApertureStats(data, ap).sum
        s_an = ApertureStats(data, an).sum
        n_ap = ap.area
        n_an = an.area
        flux = s_ap - s_an / max(n_an, 1) * n_ap
    except ImportError:
        y, x = np.mgrid[0:data.shape[0], 0:data.shape[1]]
        flux = []
        for (sx, sy) in stars:
            d2 = (x - sx) ** 2 + (y - sy) ** 2
            flux.append(data[d2 <= r ** 2].sum() -
                        data[(d2 > r_in ** 2) & (d2 <= r_out ** 2)].mean() *
                        (np.pi * r * r))
        flux = np.array(flux)
    flux = np.maximum(flux, 1e-9)
    cols = {"x": stars[:, 0], "y": stars[:, 1], "flux": flux}
    if catalog_path and wcs is not None:
        tab = load_catalog(catalog_path)
        cat = SkyCoord(np.asarray(tab["ra"], float) * u.deg,
                       np.asarray(tab["dec"], float) * u.deg)
        mags = np.asarray(tab["mag"], float) if "mag" in tab.colnames \
            else np.full(len(cat), np.nan)
        px = wcs.world_to_pixel(cat)
        d = np.hypot(px[0][None, :] - stars[:, 0][:, None],
                     px[1][None, :] - stars[:, 1][:, None])
        best = np.argmin(d, axis=1)
        ok = d[np.arange(len(stars)), best] < 4.0
        if ok.sum() >= 2:
            zp = float(np.median(mags[best[ok]] + 2.5 * np.log10(flux[ok])))
            cols["mag"] = zp - 2.5 * np.log10(flux)
            cols["zp"] = np.full(len(stars), zp)
    out = Table(cols)
    return out


# ------------------------------------------------ photométrie différentielle
def _aperture_fluxes(data: np.ndarray, stars: np.ndarray, r: float,
                     r_in: float, r_out: float) -> np.ndarray:
    try:
        from photutils.aperture import (ApertureStats, CircularAnnulus,
                                        CircularAperture)
        ap = CircularAperture(stars, r=r)
        an = CircularAnnulus(stars, r_in=r_in, r_out=r_out)
        s_ap = ApertureStats(data, ap).sum
        s_an = ApertureStats(data, an).sum
        return np.maximum(s_ap - s_an / max(an.area, 1) * ap.area, 1e-9)
    except ImportError:
        yy, xx = np.mgrid[0:data.shape[0], 0:data.shape[1]]
        flux = []
        for sx, sy in stars:
            d2 = (xx - sx) ** 2 + (yy - sy) ** 2
            flux.append(data[d2 <= r * r].sum()
                        - float(np.median(data[(d2 > r_in ** 2)
                                               & (d2 <= r_out ** 2)])) * np.pi * r * r)
        return np.maximum(np.array(flux), 1e-9)


def differential_photometry(img: np.ndarray, wcs, catalog_path: str,
                            target_ra: float | None = None,
                            target_dec: float | None = None, n_comp: int = 5,
                            r: float = 6.0, r_in: float = 10.0,
                            r_out: float = 16.0):
    """Photométrie différentielle sur étoiles de référence : la magnitude de la
    cible est mesurée par rapport à la moyenne de l'ensemble d'étoiles de
    comparaison du catalogue — le zero point et l'extinction s'annulent.
    Cible : étoile du catalogue la plus proche de (target_ra, target_dec),
    sinon la plus brillante. Comparaisons : mêmes magnitudes ±1,5.
    Retourne une Table (rôle, ra, dec, x, y, flux, mag, dmag, dmag_err)."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from .astrometry import detect_stars, load_catalog
    mono = img if img.ndim == 2 else img.mean(axis=2)
    data = np.asarray(mono, np.float64)
    stars = detect_stars(np.asarray(mono, np.float32), 150)
    if len(stars) == 0:
        raise RuntimeError("Aucune étoile détectée")
    flux = _aperture_fluxes(data, stars, r, r_in, r_out)
    tab = load_catalog(catalog_path)
    ra = np.asarray(tab["ra"], float)
    dec = np.asarray(tab["dec"], float)
    mags = (np.asarray(tab["mag"], float) if "mag" in tab.colnames
            else np.full(len(ra), np.nan))
    px = wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
    d = np.hypot(px[0][None, :] - stars[:, 0][:, None],
                 px[1][None, :] - stars[:, 1][:, None])
    best = np.argmin(d, axis=1)
    ok = d[np.arange(len(stars)), best] < 4.0
    if ok.sum() < 4:
        raise RuntimeError("Photométrie différentielle : < 4 étoiles appariées "
                           "au catalogue (plate solving requis)")
    idx_img = np.nonzero(ok)[0]
    idx_cat = best[ok]
    m_inst = -2.5 * np.log10(flux)
    # cible
    if target_ra is not None and target_dec is not None:
        tc = SkyCoord(target_ra * u.deg, target_dec * u.deg)
        cc = SkyCoord(ra[idx_cat] * u.deg, dec[idx_cat] * u.deg)
        seps = cc.separation(tc).arcsec
        it = int(np.argmin(seps))
        tol = 10.0
        try:  # tolérance : 3 pixels d'après l'échelle du WCS
            from astropy.wcs.utils import proj_plane_pixel_scales
            tol = max(tol, 3.0 * float(np.mean(proj_plane_pixel_scales(wcs))) * 3600.0)
        except Exception:
            pass
        if float(seps[it]) > tol:
            raise RuntimeError(
                f"Cible introuvable : aucune étoile détectée et appariée à moins de "
                f"{tol:.0f}\" de la position demandée (la plus proche : {seps[it]:.0f}\").")
    else:
        it = int(np.argmax(flux[idx_img]))
    mag_t = mags[idx_cat[it]]
    cand = [i for i in range(len(idx_img))
            if i != it and np.isfinite(mags[idx_cat[i]])
            and abs(mags[idx_cat[i]] - mag_t) < 1.5]
    cand.sort(key=lambda i: -flux[idx_img[i]])
    comp = cand[:max(1, n_comp)]
    if not comp:
        raise RuntimeError("Photométrie différentielle : aucune étoile de comparaison "
                           "de magnitude proche (±1,5) de la cible dans le champ.")
    dmag = float(m_inst[idx_img[it]] - np.mean(m_inst[idx_img[comp]]))
    # erreur : bruit du fond dans les ouvertures (approximation prudente) + dispersion des comp.
    h, w = data.shape
    yy, xx = np.mgrid[0:h, 0:w]
    def _snr(ii):
        sx, sy = stars[idx_img[ii]]
        d2 = (xx - sx) ** 2 + (yy - sy) ** 2
        ap = d2 <= r * r
        an = (d2 > r_in ** 2) & (d2 <= r_out ** 2)
        sig = float(np.std(data[an])) + 1e-9
        return flux[idx_img[ii]] / np.sqrt(flux[idx_img[ii]] + ap.sum() * sig ** 2)
    snr_t = _snr(it)
    snr_c = np.mean([_snr(i) for i in comp])
    dmag_err = float(1.0857 * np.sqrt(snr_t ** -2 + snr_c ** -2 / len(comp)))
    m_ref = float(np.mean(mags[idx_cat[comp]]))
    rows = []
    for k, i in enumerate([it] + comp):
        rows.append((("Cible" if k == 0 else "Comp."),
                     float(ra[idx_cat[i]]), float(dec[idx_cat[i]]),
                     float(stars[idx_img[i]][0]), float(stars[idx_img[i]][1]),
                     float(flux[idx_img[i]]), float(mags[idx_cat[i]]),
                     dmag if k == 0 else np.nan,
                     dmag_err if k == 0 else np.nan))
    from astropy.table import Table
    out = Table(rows=rows, names=("role", "ra", "dec", "x", "y", "flux",
                                  "mag_catalogue", "dmag", "dmag_err"),
                meta={"dmag": dmag, "dmag_err": dmag_err,
                      "mag_ref_ensemble": m_ref, "n_comp": len(comp)})
    return out
