# -*- coding: utf-8 -*-
"""
Alignement (recalage) : corrélation de phase (translation + petite rotation),
déroration planétaire et alignement des canaux couleur.
"""
from __future__ import annotations
import numpy as np
import scipy.ndimage as ndi


def _lum(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return img[..., :3].mean(axis=2)
    return img


def phase_corr(ref: np.ndarray, mov: np.ndarray) -> tuple[float, float, float]:
    """Corrélation croisée passe-haut pondérée par la puissance (robuste au fond
    commun et au gradient), précision sous-pixel.

    Convention : si le contenu de `mov` est déplacé de (Dx, Dy) par rapport à
    `ref`, la fonction retourne (dx, dy) = (−Dx, −Dy), c'est-à-dire la
    translation à appliquer à `mov` pour le recaler : apply_shift(mov, dx, dy)
    réaligne `mov` sur `ref`. Troisième valeur : hauteur du pic."""
    ref_l = np.asarray(_lum(ref), np.float64)
    mov_l = np.asarray(_lum(mov), np.float64)
    H, W = ref_l.shape[:2]
    if mov_l.shape != ref_l.shape:  # recadre ou complète à la taille de ref
        tmp = np.full((H, W), float(np.median(mov_l)), np.float64)
        hh, ww = min(H, mov_l.shape[0]), min(W, mov_l.shape[1])
        tmp[:hh, :ww] = mov_l[:hh, :ww]
        mov_l = tmp

    win = np.outer(np.hanning(H), np.hanning(W))

    def prep(x: np.ndarray) -> np.ndarray:
        x = x - ndi.gaussian_filter(x, 4)  # passe-haut : supprime fond/gradient
        return x * win

    Fa = np.fft.rfft2(prep(ref_l))
    Fb = np.fft.rfft2(prep(mov_l))
    corr = np.fft.irfft2(Fa * np.conj(Fb), s=(H, W))
    y0, x0 = np.unravel_index(int(np.argmax(corr)), corr.shape)

    def fold(v, n):
        return v - n if v > n // 2 else v

    c = float(corr[y0, x0])
    dy, dx = float(fold(y0, H)), float(fold(x0, W))
    # raffinement sous-pixel (parabole 3 points) : voisins pris modulo la
    # taille (la corrélation est circulaire), y compris autour de 0
    if H >= 3:
        cm, cp = corr[(y0 - 1) % H, x0], corr[(y0 + 1) % H, x0]
        denom = cm - 2 * c + cp
        if abs(denom) > 1e-12:
            dy += float(np.clip(0.5 * (cm - cp) / denom, -0.5, 0.5))
    if W >= 3:
        cm, cp = corr[y0, (x0 - 1) % W], corr[y0, (x0 + 1) % W]
        denom = cm - 2 * c + cp
        if abs(denom) > 1e-12:
            dx += float(np.clip(0.5 * (cm - cp) / denom, -0.5, 0.5))
    return dx, dy, c


def apply_shift(img: np.ndarray, dx: float, dy: float, angle_deg: float = 0.0,
                border_mode: str = "reflect") -> np.ndarray:
    """Translate l'image de (dx, dy) pixels (après rotation de `angle_deg`
    autour du centre). Avec (dx, dy) retourné par phase_corr(ref, mov),
    apply_shift(mov, dx, dy) recale mov sur ref."""
    cv2 = __import__("cv2")
    a = np.asarray(img, np.float32)
    h, w = a.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    M[0, 2] += dx
    M[1, 2] += dy
    bmode = {"reflect": cv2.BORDER_REFLECT, "constant": cv2.BORDER_CONSTANT}[border_mode]
    return cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=bmode)


def logpolar_rotation(ref: np.ndarray, mov: np.ndarray) -> tuple[float, float, float]:
    """Estimation directe rotation + échelle (Fourier–Mellin) : le module du
    spectre est invariant par translation ; en coordonnées log-polaires une
    rotation devient un décalage angulaire, mesuré par corrélation de phase.
    Retourne (angle_deg, échelle, réponse) avec la convention de
    estimate_rotation : apply_shift(mov, 0, 0, angle) tourne mov vers ref.
    Ambiguïté de 180° (spectre symétrique) : angle ramené dans ]−90°, 90°]."""
    cv2 = __import__("cv2")
    r = np.asarray(_lum(ref), np.float32)
    m = np.asarray(_lum(mov), np.float32)
    h, w = r.shape
    if m.shape != r.shape:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)

    def mag(x):
        x = x - ndi.gaussian_filter(x, 4)
        F = np.fft.fftshift(np.abs(np.fft.fft2(x * win)))
        # filtre passe-haut du spectre (atténue le centre dominant)
        yy, xx = np.ogrid[-0.5:0.5:h * 1j, -0.5:0.5:w * 1j]
        c = np.cos(np.pi * yy) * np.cos(np.pi * xx)
        return (np.log1p(F) * (1.0 - c) * (2.0 - c)).astype(np.float32)

    Ar, Am = mag(r), mag(m)
    n_ang = 720
    radius = min(h, w) / 2.0
    n_rad = int(max(64, radius))
    flags = cv2.WARP_POLAR_LOG | cv2.INTER_LINEAR | cv2.WARP_FILL_OUTLIERS
    Pr = cv2.warpPolar(Ar, (n_rad, n_ang), (w / 2.0, h / 2.0), radius, flags)
    Pm = cv2.warpPolar(Am, (n_rad, n_ang), (w / 2.0, h / 2.0), radius, flags)
    (sx, sy), resp = cv2.phaseCorrelate(Pr.astype(np.float64), Pm.astype(np.float64))
    ang = sy * 360.0 / n_ang
    ang = (ang + 90.0) % 180.0 - 90.0
    scale = float(np.exp(sx * np.log(radius) / n_rad))
    return float(ang), scale, float(resp)


def _rot_score(r_hp: np.ndarray, mov: np.ndarray, ang: float) -> float:
    m = apply_shift(mov.astype(np.float32), 0, 0, angle_deg=float(ang))
    m = _lum(m).astype(np.float32)
    m = m - ndi.gaussian_filter(m, 6)
    return phase_corr(r_hp, m)[2]


def estimate_rotation(ref: np.ndarray, mov: np.ndarray, max_deg: float = 3.0,
                      step: float = 0.25, method: str = "auto") -> float:
    """Rotation (± max_deg) qui recale mov sur ref.

    - « auto » (défaut) : estimation directe log-polaire (Fourier–Mellin),
      puis raffinement local (parabole sur la réponse de corrélation) ;
      repli sur le balayage si l'estimation sort de la plage.
    - « sweep » : balayage par pas de `step` (ancienne méthode, plus lente),
      raffiné de la même manière."""
    r = _lum(ref).astype(np.float32)
    r = r - ndi.gaussian_filter(r, 6)
    cand = None
    if method == "auto":
        try:
            a, _s, _resp = logpolar_rotation(ref, mov)
            if abs(a) <= max_deg + step:
                cand = a
        except Exception:
            cand = None
    if cand is None:
        best, cand = -np.inf, 0.0
        for ang in np.arange(-max_deg, max_deg + 1e-9, step):
            sc = _rot_score(r, mov, ang)
            if sc > best:
                best, cand = sc, float(ang)
    # raffinement : deux passes de parabole sur 3 points
    h = step
    for _ in range(2):
        s0 = _rot_score(r, mov, cand)
        sm = _rot_score(r, mov, cand - h)
        sp = _rot_score(r, mov, cand + h)
        if sm > s0 and sm >= sp:
            cand -= h
            continue
        if sp > s0 and sp > sm:
            cand += h
            continue
        den = sm - 2 * s0 + sp
        if abs(den) > 1e-12:
            cand += float(np.clip(0.5 * (sm - sp) / den, -0.5, 0.5)) * h
        h /= 4.0
    return float(np.clip(cand, -max_deg, max_deg))


def align_stack(ref: np.ndarray, frames: list[np.ndarray], with_rotation: bool = False,
                max_deg: float = 3.0, progress=None) -> list[np.ndarray]:
    """Aligne toutes les frames sur la référence (translation, + rotation optionnelle)."""
    out = []
    n = len(frames)
    for i, f in enumerate(frames):
        f = np.asarray(f, np.float32)
        ang = estimate_rotation(ref, f, max_deg) if with_rotation else 0.0
        rot = apply_shift(f, 0, 0, ang) if ang else f
        dx, dy, _ = phase_corr(ref, rot)
        out.append(apply_shift(rot, dx, dy) if (dx or dy) else rot)
        if progress:
            progress(i + 1, n)
    return out


# ------------------------------------------------------- déroration planétaire
def derotate(frames: list[np.ndarray], rate_deg_per_min: float | None = None,
             interval_s: float | None = None, auto_max_deg: float = 0.0,
             ref_index: int = 0, progress=None) -> list[np.ndarray]:
    """Dérotation du champ (rotation planétaire) :
    - soit taux fixe (deg/minute × temps écoulé),
    - soit estimation automatique par rapport à la frame de référence."""
    n = len(frames)
    ref = frames[ref_index]
    idx = np.arange(n)
    if auto_max_deg > 0 and not (rate_deg_per_min and interval_s is not None):
        # estimation sur un sous-échantillon de frames + interpolation linéaire
        # (la dérotation du champ varie lentement)
        step = max(1, n // 12)
        sample = list(range(0, n, step))
        if n - 1 not in sample:
            sample.append(n - 1)
        angles = {i: estimate_rotation(ref, frames[i], max_deg=auto_max_deg)
                  for i in sample}
        xs = np.array(sorted(angles))
        ys = np.array([angles[i] for i in xs])
        ang_by_i = np.interp(idx, xs, ys)
    else:
        ang_by_i = [rate_deg_per_min * (i - ref_index) * interval_s / 60.0
                    if (rate_deg_per_min and interval_s is not None) else 0.0
                    for i in idx]
    out = []
    for i, f in enumerate(frames):
        out.append(apply_shift(np.asarray(f, np.float32), 0, 0,
                               angle_deg=float(ang_by_i[i])))
        if progress:
            progress(i + 1, n)
    return out


# ------------------------------------------------------- alignement des couleurs
def align_channels(rgb: np.ndarray, max_shift: float | None = None) -> np.ndarray:
    """Recale les canaux R et B sur G (aberration chromatique planétaire)."""
    from .image_io import is_color
    if not is_color(rgb):
        return rgb
    g = rgb[..., 1].astype(np.float64)
    out = rgb.copy()
    for idx in (0, 2):
        dx, dy, _ = phase_corr(g, rgb[..., idx])
        if max_shift is not None and np.hypot(dx, dy) > max_shift:
            continue
        out[..., idx] = apply_shift(rgb[..., idx], dx, dy)
    return out.astype(np.float32)


# --------------------------------- choix d'algorithme d'alignement
ALIGN_METHODS = ["Phase (rapide)", "Limbe (précis)", "Étoiles (large champ)"]


def _edges(img: np.ndarray) -> np.ndarray:
    """Contours (limbe planétaire, bords) pour un recalage insensible au
    contraste doux."""
    cv2 = __import__("cv2")
    from .display import autostretch
    from .image_io import to_uint8
    mono = _lum(img)
    u8 = to_uint8(autostretch(np.asarray(mono, np.float32)))
    e = cv2.Canny(u8, 40, 140)
    return e.astype(np.float32)


def align_frames(frames: list[np.ndarray], method: str = "Phase (rapide)",
                 progress=None) -> list[np.ndarray]:
    """Aligne les frames avec l'algorithme choisi :
    - « Phase (rapide) » : corrélation de phase passe-haut (translation) ;
    - « Limbe (précis) » : corrélation sur les contours (limbe, bords) ;
    - « Étoiles (large champ) » : appariement d'étoiles (astroalign), repli phase."""
    n = len(frames)
    if n == 0:
        return frames
    ref_i = n // 2
    ref = frames[ref_i]
    out: list[np.ndarray] = [None] * n  # type: ignore
    out[ref_i] = np.asarray(ref, np.float32)
    if method.startswith("Étoiles"):
        from .astrometry import align_by_stars
        for i, f in enumerate(frames):
            if out[i] is None:
                out[i] = align_by_stars(ref, f)
            if progress:
                progress(i + 1, n)
        return out
    use_edges = method.startswith("Limbe")
    ref_p = _edges(ref) if use_edges else ref
    for i, f in enumerate(frames):
        if out[i] is None:
            mov_p = _edges(f) if use_edges else f
            dx, dy, _ = phase_corr(ref_p, mov_p)
            out[i] = apply_shift(np.asarray(f, np.float32), dx, dy)
        if progress:
            progress(i + 1, n)
    return out


# --------------------------------------- alignement en similitude (étoiles)
def _similarity_ls(p: np.ndarray, s: np.ndarray, flip: bool = False):
    """Moindres carrés similitude 2D (échelle+rotation+translation, complexes).
    Retourne (M 2×2, t, résidu médian) ou (None, None, inf)."""
    zp = p[:, 0] + 1j * p[:, 1]
    if flip:
        zp = np.conj(zp)
    zs = s[:, 0] + 1j * s[:, 1]
    dp, ds = zp - zp.mean(), zs - zs.mean()
    den = float(np.vdot(dp, dp).real)
    if den < 1e-9:
        return None, None, np.inf
    alpha = np.vdot(dp, ds) / den
    beta = zs.mean() - alpha * zp.mean()
    M = np.array([[alpha.real, -alpha.imag], [alpha.imag, alpha.real]])
    t = np.array([beta.real, beta.imag])
    return M, t, float(np.median(np.abs(alpha * zp + beta - zs)))


def align_similarity(ref: np.ndarray, mov: np.ndarray, max_stars: int = 40) -> np.ndarray:
    """Alignement en similitude (rotation + échelle + translation, SANS
    distorsion polynomiale) par appariement d'étoiles : paires d'étoiles
    candidates, deux hypothèses (direct/miroir), comptage d'inliers puis
    moindres carrés sur les appariements. Repli sur la phase si échec."""
    cv2 = __import__("cv2")
    from itertools import combinations
    from .astrometry import detect_stars
    r_ = ref if ref.ndim == 2 else ref.mean(axis=2)
    m_ = mov if mov.ndim == 2 else mov.mean(axis=2)
    sr = detect_stars(r_, max_stars)
    sm = detect_stars(m_, max_stars)
    if len(sr) < 3 or len(sm) < 3:
        dx, dy, _ = phase_corr(ref, mov)
        return apply_shift(np.asarray(mov, np.float32), dx, dy)
    ni, nj = min(5, len(sm)), min(12, len(sr))
    best = None
    for flip in (False, True):
        for i, j in combinations(range(ni), 2):
            for a, b in combinations(range(nj), 2):
                M, t, _res = _similarity_ls(sm[[i, j]], sr[[a, b]], flip)
                if M is None:
                    continue
                q = (M @ sm.T).T + t
                d = np.hypot(q[:, None, 0] - sr[None, :, 0],
                             q[:, None, 1] - sr[None, :, 1])
                match = np.argmin(d, axis=1)
                dist = d[np.arange(len(q)), match]
                inl = dist < 3.0
                score = (int(inl.sum()),
                         -float(dist[inl].mean()) if inl.any() else 0.0)
                if best is None or score > best[0]:
                    best = (score, match, inl)
    if best is None or best[0][0] < 3:
        dx, dy, _ = phase_corr(ref, mov)
        return apply_shift(np.asarray(mov, np.float32), dx, dy)
    _, match, inl = best
    keep = np.nonzero(inl)[0]
    M, t, _ = _similarity_ls(sm[keep], sr[match[keep]])
    if M is None:
        dx, dy, _ = phase_corr(ref, mov)
        return apply_shift(np.asarray(mov, np.float32), dx, dy)
    M2 = np.hstack([M, t.reshape(2, 1)]).astype(np.float32)
    out = cv2.warpAffine(np.asarray(mov, np.float32), M2,
                         (mov.shape[1], mov.shape[0]), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REFLECT)
    return out.astype(np.float32)


# ---------------- plate solving par image + distorsions polynomiales
def _poly_terms(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    terms = []
    for dy in range(order + 1):
        for dx in range(order + 1 - dy):
            terms.append((x ** dx) * (y ** dy))
    return np.stack(terms, axis=1)


def fit_poly_transform(ref: np.ndarray, mov: np.ndarray, order: int = 2,
                       max_stars: int = 40) -> dict:
    """Plate solving par image : appariement d'étoiles mov->ref (init par
    paires, hypothèses direct/miroir, seuil d'inliers large puis resserré),
    puis ajustement polynomial (degré `order`) des coordonnées — corrige les
    distorsions optiques/atmosphériques variables d'une image à l'autre.
    Retourne {cx, cy, resid, n, order} : coefficients mov -> référence."""
    from itertools import combinations
    from .astrometry import detect_stars
    r_ = ref if ref.ndim == 2 else ref.mean(axis=2)
    m_ = mov if mov.ndim == 2 else mov.mean(axis=2)
    sr = detect_stars(r_, max_stars)
    sm = detect_stars(m_, max_stars)
    if len(sr) < 6 or len(sm) < 6:
        raise RuntimeError("Pas assez d'étoiles pour le plate solving par image")
    ni, nj = min(5, len(sm)), min(12, len(sr))
    best = None
    for flip in (False, True):
        for i, j in combinations(range(ni), 2):
            for a, b in combinations(range(nj), 2):
                M, t, _res = _similarity_ls(sm[[i, j]], sr[[a, b]], flip)
                if M is None:
                    continue
                q = (M @ sm.T).T + t
                d = np.hypot(q[:, None, 0] - sr[None, :, 0],
                             q[:, None, 1] - sr[None, :, 1])
                match = np.argmin(d, axis=1)
                dist = d[np.arange(len(q)), match]
                inl = dist < 8.0  # seuil large : la distorsion n'est pas encore modélisée
                score = (int(inl.sum()),
                         -float(dist[inl].mean()) if inl.any() else 0.0)
                if best is None or score > best[0]:
                    best = (score, match, inl)
    if not best or best[0][0] < 6:
        raise RuntimeError("Appariement insuffisant pour le poly (≥6 inliers requis)")
    _, match, inl = best
    keep_mov = np.nonzero(inl)[0]
    keep_ref = match[keep_mov]
    order_eff = min(order, max(1, int((-3 + np.sqrt(4 * len(keep_mov) + 1)) / 2)))
    cx = cy = None
    for thr in (8.0, 5.0, 3.0):
        nterm = (order_eff + 1) * (order_eff + 2) // 2
        if len(keep_mov) < 2 * nterm:        # assez de points pour ce degré ?
            order_eff = max(1, int(((-1 + np.sqrt(1 + 2 * len(keep_mov))) - 3) / 2))
        p, s = sm[keep_mov], sr[keep_ref]
        lam = 1e-6 * len(p)
        A = _poly_terms(p[:, 0], p[:, 1], order_eff)
        AtA = A.T @ A + lam * np.eye(A.shape[1])
        cx = np.linalg.solve(AtA, A.T @ s[:, 0])
        cy = np.linalg.solve(AtA, A.T @ s[:, 1])
        # rejet interne des paires aberrantes (résidu radial, 3·MAD)
        rrx = _poly_terms(p[:, 0], p[:, 1], order_eff) @ cx - s[:, 0]
        rry = _poly_terms(p[:, 0], p[:, 1], order_eff) @ cy - s[:, 1]
        rr = np.sqrt(rrx ** 2 + rry ** 2)
        med = np.median(rr)
        mad = 1.4826 * np.median(np.abs(rr - med)) + 1e-6
        good = rr <= med + max(3 * mad, 0.5)
        if good.sum() >= 6 and good.sum() < len(keep_mov):
            keep_mov, keep_ref = keep_mov[good], keep_ref[good]
        if thr == 3.0:
            break
        # ré-association mutuelle (ICP-like) : chaque étoile mov -> ref la plus
        # proche du modèle courant, uniquité garantie dans les deux sens
        q = np.stack([_poly_terms(sm[:, 0], sm[:, 1], order_eff) @ cx,
                      _poly_terms(sm[:, 0], sm[:, 1], order_eff) @ cy], axis=1)
        d = np.hypot(q[:, None, 0] - sr[None, :, 0], q[:, None, 1] - sr[None, :, 1])
        cand = np.argmin(d, axis=1)
        dmin = d[np.arange(len(sm)), cand]
        back = np.argmin(d, axis=0)
        pairs = [(i, int(cand[i])) for i in range(len(sm))
                 if dmin[i] < thr and back[cand[i]] == i]
        if len(pairs) >= 6:
            keep_mov = np.array([i for i, _ in pairs])
            keep_ref = np.array([j for _, j in pairs])
        if len(keep_mov) < 6:
            raise RuntimeError("Poly : raffinement insuffisant")
    if cx is None or len(keep_mov) < 6:
        raise RuntimeError("Poly : raffinement insuffisant")
    rx = _poly_terms(sm[keep_mov, 0], sm[keep_mov, 1], order_eff) @ cx - sr[keep_ref, 0]
    ry = _poly_terms(sm[keep_mov, 0], sm[keep_mov, 1], order_eff) @ cy - sr[keep_ref, 1]
    resid = float(np.sqrt(np.median(rx ** 2 + ry ** 2)))
    return {"cx": cx, "cy": cy, "resid": resid, "n": int(len(keep_mov)),
            "order": int(order_eff)}


def align_poly(ref: np.ndarray, mov: np.ndarray, order: int = 3,
               max_stars: int = 40) -> np.ndarray:
    """Aligne `mov` sur `ref` par plate solving par image : distorsions
    polynomiales (degré choisi au mieux entre 1 et `order` d'après le résidu
    médian), warp via cv2.remap. Repli : similitude puis phase."""
    cv2 = __import__("cv2")
    try:
        fits = [fit_poly_transform(ref, mov, o, max_stars) for o in range(1, order + 1)]
    except RuntimeError:
        return align_similarity(ref, mov)
    best = min(fits, key=lambda f: f["resid"])
    h, w = mov.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    # transformation inverse (ref -> mov) nécessaire au remap
    try:
        inv = _poly_inverse(ref, mov, best, max_stars)
    except RuntimeError:
        return align_similarity(ref, mov)
    map_x = (_poly_terms(xx.ravel(), yy.ravel(), inv["order"]) @ inv["cx"]).reshape(h, w)
    map_y = (_poly_terms(xx.ravel(), yy.ravel(), inv["order"]) @ inv["cy"]).reshape(h, w)
    out = cv2.remap(np.asarray(mov, np.float32), map_x.astype(np.float32),
                    map_y.astype(np.float32), interpolation=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REFLECT)
    return out.astype(np.float32)


def _poly_inverse(ref: np.ndarray, mov: np.ndarray, direct: dict,
                  max_stars: int = 40) -> dict:
    """Ajuste le polynôme inverse (coordonnées ref -> mov) par itérations de
    point fixe : applique le direct, mesure l'écart, corrige. Converge en
    quelques itérations pour des distorsions douces."""
    from .astrometry import detect_stars
    order = direct["order"]
    # paires directes réutilisées : on régénère les correspondances par
    # détection + plus proche voisin via le fit direct (précision suffisante)
    r_ = ref if ref.ndim == 2 else ref.mean(axis=2)
    m_ = mov if mov.ndim == 2 else mov.mean(axis=2)
    sr = detect_stars(r_, max_stars)
    sm = detect_stars(m_, max_stars)
    q = np.stack([( _poly_terms(sm[:, 0], sm[:, 1], order) @ direct["cx"]),
                  (_poly_terms(sm[:, 0], sm[:, 1], order) @ direct["cy"])], axis=1)
    d = np.hypot(q[:, None, 0] - sr[None, :, 0], q[:, None, 1] - sr[None, :, 1])
    match = np.argmin(d, axis=1)
    dist = d[np.arange(len(q)), match]
    keep = dist < 3.0
    if keep.sum() < 6:
        raise RuntimeError("poly inverse : appariement insuffisant")
    p, s = sr[match[keep]], sm[keep]          # ref -> mov
    lam = 1e-6 * len(p)
    A = _poly_terms(p[:, 0], p[:, 1], order)
    AtA = A.T @ A + lam * np.eye(A.shape[1])
    cx = np.linalg.solve(AtA, A.T @ s[:, 0])
    cy = np.linalg.solve(AtA, A.T @ s[:, 1])
    rx, ry = A @ cx - s[:, 0], A @ cy - s[:, 1]
    return {"cx": cx, "cy": cy, "order": order,
            "resid": float(np.sqrt(np.median(rx ** 2 + ry ** 2)))}


# ------------------------------------------- transformation globale (drizzle)
def warp(img: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Applique une matrice 2×3 frame -> référence (convention OpenCV)."""
    cv2 = __import__("cv2")
    a = np.asarray(img, np.float32)
    h, w = a.shape[:2]
    return cv2.warpAffine(a, np.asarray(M, np.float64), (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REFLECT)


def estimate_transform(ref: np.ndarray, mov: np.ndarray,
                       with_rotation: bool = True) -> np.ndarray:
    """Matrice 2×3 qui envoie les coordonnées de `mov` sur celles de `ref` :
    similitude par appariement d'étoiles (astroalign si installé), sinon
    corrélation de phase (translation, + rotation estimée si demandé)."""
    if ref is mov:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    r = _lum(np.asarray(ref, np.float32))
    m = _lum(np.asarray(mov, np.float32))
    try:
        import astroalign
        T, _pts = astroalign.find_transform(m, r, max_control_points=60)
        M = np.asarray(T.params, np.float64)[:2]
        if np.isfinite(M).all():
            return M
    except Exception:
        pass
    ang = estimate_rotation(r, m, 2.0) if with_rotation else 0.0
    rot = apply_shift(m, 0, 0, ang) if ang else m
    dx, dy, _ = phase_corr(r, rot)
    cv2 = __import__("cv2")
    h, w = m.shape[:2]
    M = np.asarray(cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ang, 1.0), np.float64)
    M[0, 2] += dx
    M[1, 2] += dy
    return M
