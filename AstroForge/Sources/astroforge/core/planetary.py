# -*- coding: utf-8 -*-
"""
Traitement planétaire en 6 étapes, reprenant les principes des logiciels
Registax 6, AstroSurface et AutoStakkert! :
  1. chargement de la vidéo (frames en mémoire, navigation image par image) ;
  2. mesure de la qualité de chaque frame puis tri (meilleure -> moins bonne) ;
  3. choix du nombre d'images à aligner ;
  4. choix de la méthode d'alignement (globale, multi-points en grille NxN,
     points/carrés placés manuellement) ;
  5. alignement de toutes les images sélectionnées ;
  6. empilement et image finale.

Estimateurs de qualité (mesurés sur le disque si ROI détectée) :
- « Contraste HF (AutoStakkert) » : énergie haute fréquence locale
  (image - médian filtré), l'esprit de l'estimateur d'AS!;
- « Netteté (Laplacien) » : variance du laplacien (netteté style Registax) ;
- « Gradient (Sobel/diff) » : contraste des gradients ;
- « Luminosité » / « Mixte ».
Alignement : champ de décalage affine ajusté par moindres carrés sur les
mesures multipoints (corrélation de phase sous-pixel par carré), rejet des
boîtes aberrantes (MAD) — même logique que le multi-point d'AS!.
"""
from __future__ import annotations
import numpy as np
import scipy.ndimage as ndi
from collections.abc import Sequence


IDENTITY_CURVE = (0.0, 0.25, 0.5, 0.75, 1.0)  # nœuds hérités (v1.9.5)
IDENTITY_CURVE_POINTS = [[0.0, 0.0], [1.0, 1.0]]


def _normalize_curve(curve):
    """Courbe tonale = liste de points [[x, y], ...] (au moins 2), triés par x
    croissant (écart minimal 1e-6 : fonction monotone), y écrêté à [0, 1].

    - None → identité ;
    - liste plate de 5 nombres (héritage v1.9.5) → sorties aux entrées
      fixes 0, .25, .5, .75, 1 ;
    - suite de paires (x, y) quelconque (>= 2) → telle quelle.
    Les paires sont évaluées en interpolation linéaire par morceaux
    (np.interp), ce qui rend l'évaluation héritée identique à 1.9.5.
    """
    if curve is None:
        return [list(p) for p in IDENTITY_CURVE_POINTS]
    a = np.asarray(curve, np.float64)
    if a.ndim == 1:  # héritage : cinq valeurs de sortie
        if a.shape != (5,):
            raise ValueError("Courbe héritée : exactement cinq valeurs.")
        a = np.stack([np.asarray(IDENTITY_CURVE, np.float64), a], axis=1)
    if a.ndim != 2 or a.shape[1] != 2 or len(a) < 2:
        raise ValueError("Courbe : liste d'au moins deux paires (x, y).")
    if not np.isfinite(a).all():
        raise ValueError("Courbe : valeurs finies requises.")
    a = a[np.argsort(a[:, 0], kind="stable")]
    a[:, 0] = np.clip(a[:, 0], 0, 1)
    a[:, 1] = np.clip(a[:, 1], 0, 1)
    xs = [float(a[0, 0])]
    for x in a[1:, 0]:  # x strictement croissants (écart minimal 1e-6)
        xs.append(min(max(float(x), xs[-1] + 1e-6), 1.0))
    a[:, 0] = xs
    return a.tolist()


def is_identity_curve(curve):
    """Vrai si la courbe normalisée est la droite identité."""
    return curve == [list(p) for p in IDENTITY_CURVE_POINTS]


def tone_settings(brightness=0.0, contrast=1.0, curve=None):
    """Réglages tonaux : luminosité -1..1, contraste 0..3 (pivot 0,5) et
    courbe généralisée (voir _normalize_curve). Rétrorapide : la liste
    héritée de cinq valeurs reste acceptée."""
    points = _normalize_curve(curve)
    if not np.isfinite(brightness) or not np.isfinite(contrast):
        raise ValueError("Luminosité et contraste doivent être finis.")
    return {"brightness": float(np.clip(brightness, -1, 1)),
            "contrast": float(np.clip(contrast, 0, 3)),
            "curve": points}


def adjust_tone(img, brightness=0.0, contrast=1.0, curve=None):
    """Mono/RGB float [0,1], mêmes points pour chaque canal, sans mélange RGB.

    Contraste autour de .5 puis luminosité, écrêtage, interpolation de courbe
    (linéaire par morceaux). Ne modifie jamais les pixels source et ne
    normalise pas selon l'image.
    """
    p = tone_settings(brightness, contrast, curve)
    a = np.asarray(img, np.float32)
    if a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[2] not in (1, 3)):
        raise ValueError("Image attendue : mono ou RGB.")
    if p["contrast"] != 1:
        a = (a - .5) * p["contrast"] + .5
    if p["brightness"] != 0:
        a = a + p["brightness"]
    a = np.clip(a, 0, 1)
    if not is_identity_curve(p["curve"]):
        xs = [pt[0] for pt in p["curve"]]
        ys = [pt[1] for pt in p["curve"]]
        a = np.interp(a, xs, ys)
    return a.astype(np.float32)


def gamma_curve(gamma, knots=65):
    """Courbe x -> x**gamma, discrétisée en `knots` points réguliers (0..1) :
    interpolation linéaire par morceaux, déterministe, écart maximal
    ~1e-4 avec le gamma exact pour 65 nœuds."""
    g = float(np.clip(gamma, 0.1, 10))
    xs = np.linspace(0, 1, int(knots))
    return tone_settings(curve=np.stack([xs, xs ** g], axis=1))["curve"]


def auto_contrast_params(img, clip=0.001):
    """« Contraste automatique » (esprit Photoshop) : points noir et blanc au
    centile `clip` de chaque côté de l'HISTOGRAMME DE LUMINANCE
    (np.quantile : statistiques d'ordre, déterministe), puis étirement
    linéaire [noir, blanc] -> [0, 1] exprimé en (luminosité, contraste)
    autour du pivot 0,5 — identique pour toutes les frames.
    Retourne (params, noir, blanc) ; contraste plafonné à 3 par
    tone_settings (histogramme ultra-étroit : saturation assumée)."""
    lum = np.asarray(luminance(img), np.float64)
    black = float(np.quantile(lum, float(np.clip(clip, 0, .49))))
    white = float(np.quantile(lum, 1 - float(np.clip(clip, 0, .49))))
    if white - black < 1e-3:
        return tone_settings(), black, white
    k = 1.0 / (white - black)
    params = tone_settings(brightness=k * (0.5 + black) - 0.5, contrast=k)
    return params, black, white


def auto_brightness_params(img, target=0.5, knots=65):
    """« Luminosité automatique » (esprit Photoshop) : gamma g tel que la
    luminance moyenne vise le gris moyen `target` : g = ln(target)/ln(moyenne)
    (borné à [0,1] pour la moyenne ; g plafonné à [0.1, 10]). Retourne
    (params, gamma) avec params exprimés en COURBE x**gamma (gamma_curve) :
    mêmes réglages pour toutes les frames, jamais de normalisation par frame."""
    m = float(np.clip(np.mean(luminance(img)), 1e-6, 1 - 1e-6))
    gamma = float(np.log(target) / np.log(m))
    params = tone_settings(curve=gamma_curve(gamma, knots))
    return params, float(np.clip(gamma, 0.1, 10))


class ToneFrames(Sequence):
    """Vue immuable et différée : références aux originaux, au plus une image
    ajustée par accès ; aucune deuxième vidéo en mémoire et aucun cumul.
    Les instantanés de commandes peuvent conserver cette vue sans copier.
    """
    def __init__(self, originals, params=None, box=None):
        from .image_io import SerFrames
        # une vidéo projetée en mémoire (SerFrames) reste différée : la
        # convertir en tuple chargerait toute la vidéo en RAM
        self.originals = (originals if isinstance(originals, (tuple, SerFrames))
                          else tuple(originals))
        self._params = tone_settings(**(params or {}))
        self.box = box

    @property
    def params(self):
        return {**self._params, "curve": list(self._params["curve"])}

    def __len__(self):
        return len(self.originals)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return ToneFrames(self.originals[index], self.params, self.box)
        image = self.originals[index]
        p = self._params
        if (p["brightness"] != 0 or p["contrast"] != 1
                or not is_identity_curve(p["curve"])):
            image = adjust_tone(image, **p)
        if self.box is not None:
            x, y, w, h = self.box
            image = image[y:y + h, x:x + w]
        return image

    def validate(self, progress=None):
        """Parcourt toutes les images avant publication, annulation par Job.
        Aucun résultat partiel n'est publié ; les pixels calculés sont libérés.
        """
        for i in range(len(self)):
            if progress:
                progress(i, len(self), "Réglages de toutes les images")
            frame = self[i]
            if not np.isfinite(frame).all():
                raise ValueError(f"Image {i + 1} : pixels non finis.")
            del frame
        if progress:
            progress(len(self), len(self), "Réglages appliqués")
        return self


def mean_sample(frames, stride=1, progress=None):
    """Moyenne échantillonnée en flux, sans matérialiser les images ajustées."""
    indices = range(0, len(frames), stride)
    total = None
    for j, i in enumerate(indices):
        if progress:
            progress(j, len(indices), "Détection du disque")
        frame = frames[i]
        if total is None:
            total = np.array(frame, dtype=np.float32, copy=True)
        else:
            total += frame
    return total / len(indices)

QUALITY_METHODS = ["Contraste HF (AutoStakkert)", "Netteté (Laplacien)",
                   "Gradient (Sobel)", "Luminosité", "Mixte"]
ALIGN_MODES = ["Multi-points (grille NxN)", "Global (une translation)",
               "Points manuels (carrés)", "Multi-points local (AS!)"]


def luminance(img: np.ndarray) -> np.ndarray:
    return img[..., :3].mean(axis=2) if img.ndim == 3 else np.asarray(img)


# ---------------------------------------------------------------- 2) qualité
def detect_planet_box(img: np.ndarray, pad: float = 0.2) -> tuple | None:
    """Boîte englobante du disque planétaire (seuil sur l'image moyenne).
    Retourne (x0, y0, x1, y1) ou None si aucun disque détecté."""
    a = ndi.gaussian_filter(luminance(np.asarray(img, np.float32)), 2)
    thr = 0.5 * (float(np.percentile(a, 50)) + float(np.percentile(a, 99.5)))
    ys, xs = np.nonzero(a > thr)
    if len(xs) < 30:
        return None
    h, w = a.shape
    cx, cy = (float(xs.min()) + float(xs.max())) / 2, (float(ys.min()) + float(ys.max())) / 2
    r = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 * (1 + pad) + 2
    return (max(0, int(cx - r)), max(0, int(cy - r)),
            min(w, int(cx + r)), min(h, int(cy + r)))


def _q_hf(a: np.ndarray) -> float:
    """Contraste haute fréquence (image - médian 5×5), esprit AutoStakkert!."""
    return float(np.abs(a - ndi.median_filter(a, 5)).mean()) + 1e-9


def _q_lap(a: np.ndarray) -> float:
    return float((ndi.laplace(a) ** 2).mean()) + 1e-9


def _q_grad(a: np.ndarray) -> float:
    gx = np.abs(np.diff(a, axis=1)).mean()
    gy = np.abs(np.diff(a, axis=0)).mean()
    return float(gx + gy) + 1e-9


def measure_quality(frames, method: str = "Contraste HF (AutoStakkert)",
                    roi=None, progress=None) -> np.ndarray:
    """Score de qualité de chaque frame (mesuré dans la ROI si fournie)."""
    arr = frames
    n = len(arr)
    scores = np.zeros(n, np.float64)
    for i in range(n):
        a = luminance(arr[i])
        if roi is not None and a.ndim == 2:
            x0, y0, x1, y1 = roi
            sub = a[max(0, y0):max(1, y1), max(0, x0):max(1, x1)]
            if sub.size >= 64:
                a = sub
        m = float(a.mean())
        if method.startswith("Luminosité"):
            scores[i] = max(0.0, 1.0 - abs(m - 0.5) * 2.0)
        elif method.startswith("Netteté"):
            scores[i] = _q_lap(a)
        elif method.startswith("Gradient"):
            scores[i] = _q_grad(a)
        elif method.startswith("Mixte"):
            scores[i] = _q_lap(a) * max(1e-9, 1.0 - abs(m - 0.5) * 2.0)
        else:
            scores[i] = _q_hf(a)
        if progress:
            progress(i + 1, n)
    return scores


def sort_quality(scores) -> np.ndarray:
    """Indices des frames triées de la meilleure à la moins bonne."""
    return np.argsort(-np.asarray(scores, np.float64), kind="stable")


# ------------------------------------------------------- 4) points d'alignement
def grid_points(shape, n_per_axis: int = 4, box: int = 48,
                roi=None) -> list[tuple]:
    """Grille NxN de centres de carrés de mesure (boîtes) sur l'image ou la ROI."""
    h, w = shape[:2]
    box = int(box)
    if roi is not None:
        x0, y0, x1, y1 = roi
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    n = max(1, int(n_per_axis))
    xs = np.linspace(x0 + box / 2, x1 - box / 2, n) if n > 1 else [(x0 + x1) / 2]
    ys = np.linspace(y0 + box / 2, y1 - box / 2, n) if n > 1 else [(y0 + y1) / 2]
    return [(float(cx), float(cy)) for cy in ys for cx in xs]


def _fit_shift_field(pts: np.ndarray, shifts: np.ndarray):
    """Ajuste un champ de décalage affine s = A·p + b : consensus d'abord sur
    les translations mesurées (les carrés sans détail — fond — mesurent
    n'importe quoi et doivent être ignorés), puis moindres carrés avec rejet
    MAD des boîtes aberrantes. Retourne (coeffs 6, n_rejetés, rms)."""
    n = len(pts)
    keep = np.ones(n, bool)
    if n >= 3:  # plus grand groupe de boîtes d'accord mutuel (± 1 px)
        dd = np.linalg.norm(shifts[:, None, :] - shifts[None, :, :], axis=2)
        center = shifts[int(np.argmax((dd <= 1.0).sum(axis=1)))]
        cons = np.linalg.norm(shifts - center, axis=1) <= 1.0
        if cons.sum() >= 3:
            keep = cons
    x, y = pts[:, 0], pts[:, 1]
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    cxr = np.array([0.0, 0.0, float(np.mean(shifts[keep, 0]) if keep.any() else 0.0)])
    cyr = np.array([0.0, 0.0, float(np.mean(shifts[keep, 1]) if keep.any() else 0.0)])
    if keep.sum() >= 3:
        for _ in range(2):
            cxr, *_ = np.linalg.lstsq(A[keep], shifts[keep, 0], rcond=None)
            cyr, *_ = np.linalg.lstsq(A[keep], shifts[keep, 1], rcond=None)
            r = np.hypot(A @ cxr - shifts[:, 0], A @ cyr - shifts[:, 1])
            med = float(np.median(r[keep]))
            mad = 1.4826 * float(np.median(np.abs(r[keep] - med))) + 0.05
            newkeep = r <= med + 3 * mad
            if newkeep.sum() < 3 or np.array_equal(newkeep, keep):
                break
            keep = newkeep
        cxr, *_ = np.linalg.lstsq(A[keep], shifts[keep, 0], rcond=None)
        cyr, *_ = np.linalg.lstsq(A[keep], shifts[keep, 1], rcond=None)
    r = np.hypot(A @ cxr - shifts[:, 0], A @ cyr - shifts[:, 1])
    rms = float(np.sqrt(np.mean(r[keep] ** 2))) if keep.any() else 0.0
    return np.concatenate([cxr, cyr]), int((~keep).sum()), rms


def align_frame(ref: np.ndarray, mov: np.ndarray, points=None, box: int = 48,
                mode: str = "Multi-points (grille NxN)") -> tuple:
    """Aligne `mov` sur `ref` selon la méthode choisie. Retourne (image
    alignée, stats). Le champ de décalage mesuré est lissé/déduit en affine
    (translation globale ou multi-points), puis appliqué par remap cubique."""
    import cv2
    from .registration import phase_corr
    ref_l, mov_l = luminance(ref), luminance(mov)
    h, w = ref_l.shape
    box = max(16, int(box))
    if mode.startswith("Global"):
        dx, dy, resp = phase_corr(ref, mov)
        coeffs = np.array([0.0, 0.0, dx, 0.0, 0.0, dy])
        n_rej, rms, n_pts = 0, 0.0, 1
    else:
        pts = [(float(px), float(py)) for px, py in (points or [])] or \
            grid_points(ref.shape, 4, box)
        n_pts = len(pts)
        P = np.asarray(pts, np.float64)
        shifts = np.zeros((n_pts, 2), np.float64)
        for k in range(n_pts):
            cx, cy = pts[k]
            x0, y0 = int(cx - box / 2), int(cy - box / 2)
            rb = ref_l[y0:y0 + box, x0:x0 + box]
            mb = mov_l[y0:y0 + box, x0:x0 + box]
            dx, dy, _resp = phase_corr(rb, mb)
            shifts[k] = (dx, dy)
        coeffs, n_rej, rms = _fit_shift_field(P, shifts)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    a1, a2, a3, b1, b2, b3 = (float(c) for c in coeffs)
    map_x = xx - (a1 * xx + a2 * yy + a3)
    map_y = yy - (b1 * xx + b2 * yy + b3)
    out = cv2.remap(np.asarray(mov, np.float32), map_x, map_y,
                    interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    stats = {"n_points": n_pts, "rejected": n_rej, "rms": rms,
             "dx": float(a3), "dy": float(b3)}
    return out.astype(np.float32), stats


# --------------------------------------- multipoint LOCAL (esprit AS!)
def _hann2(n: int) -> np.ndarray:
    w = np.hanning(n + 2)[1:-1]
    return np.outer(w, w).astype(np.float32)


def local_multipoint_stack(frames, order, n_sel: int, roi=None, box: int = 64,
                           grid_n: int = 6, ap_percent: float = 50.0,
                           min_contrast: float = 0.15, progress=None) -> dict:
    """Empilement multipoint LOCAL (principe d'AutoStakkert!) :
    1. référence = moyenne des meilleures images alignées globalement ;
    2. grille de points d'alignement (AP) en carrés qui se recouvrent de 50 %,
       limitée aux carrés contenant du détail ;
    3. pour CHAQUE AP : qualité locale de chaque image candidate, sélection
       des meilleures `ap_percent` %, décalage local sous-pixel puis moyenne
       — chaque zone garde ses propres meilleures images (turbulence locale) ;
    4. fondu des AP par fenêtres de Hann ; hors AP : empilement global.
    `frames` : séquence (vue différée acceptée) ; `order` : indices triés par
    qualité ; `n_sel` : nombre d'images candidates."""
    from .registration import apply_shift, phase_corr
    sel = [int(i) for i in order[:max(2, int(n_sel))]]
    first = np.asarray(frames[sel[0]], np.float32)
    h, w = first.shape[:2]
    color = first.ndim == 3
    # 1) alignement global de toutes les candidates sur la meilleure
    shifts = {sel[0]: (0.0, 0.0)}
    ref_l = luminance(first)
    for k, i in enumerate(sel[1:], start=1):
        dx, dy, _ = phase_corr(ref_l, luminance(np.asarray(frames[i], np.float32)))
        shifts[i] = (dx, dy)
        if progress:
            progress(k, len(sel), "Alignement global")
    n_ref = max(2, len(sel) // 10)
    ref = np.mean([apply_shift(np.asarray(frames[i], np.float32), *shifts[i])
                   for i in sel[:n_ref]], axis=0).astype(np.float32)
    base = np.mean([apply_shift(np.asarray(frames[i], np.float32), *shifts[i])
                    for i in sel[:max(n_ref, int(len(sel) * ap_percent / 100))]],
                   axis=0).astype(np.float32)
    ref_lum = luminance(ref)
    # 2) points d'alignement
    box = int(max(16, box))
    x0, y0, x1, y1 = roi if roi is not None else (0, 0, w, h)
    step = max(8, box // 2)
    aps = []
    contrast_all = float(np.std(ref_lum)) + 1e-9
    for cy in range(int(y0 + box // 2), int(y1 - box // 2) + 1, step):
        for cx in range(int(x0 + box // 2), int(x1 - box // 2) + 1, step):
            patch = ref_lum[cy - box // 2:cy + box // 2, cx - box // 2:cx + box // 2]
            if patch.shape != (box, box):
                continue
            if float(np.std(patch)) >= min_contrast * contrast_all:
                aps.append((cx, cy))
    acc = np.zeros(ref.shape, np.float64)
    wacc = np.zeros((h, w), np.float64)
    win = _hann2(box)
    k_ap = max(1, int(round(len(sel) * float(np.clip(ap_percent, 1, 100)) / 100.0)))
    margin = 8
    for a, (cx, cy) in enumerate(aps):
        ys, xs = cy - box // 2, cx - box // 2
        rp = ref_lum[ys:ys + box, xs:xs + box]
        cands = []
        for i in sel:
            dx, dy = shifts[i]
            # zone correspondante dans l'image i (avant alignement global)
            sx, sy = int(round(xs - dx)), int(round(ys - dy))
            if sx - margin < 0 or sy - margin < 0 or sx + box + margin > w or sy + box + margin > h:
                continue
            big = np.asarray(frames[i], np.float32)[sy - margin:sy + box + margin,
                                                     sx - margin:sx + box + margin]
            q = _q_hf(luminance(big)[margin:-margin, margin:-margin])
            cands.append((q, i, sx, sy, big))
        if not cands:
            continue
        cands.sort(key=lambda t: -t[0])
        stack = []
        for q, i, sx, sy, big in cands[:k_ap]:
            ddx, ddy, _ = phase_corr(rp, luminance(big)[margin:-margin, margin:-margin])
            if np.hypot(ddx, ddy) > margin - 1:
                continue
            al = apply_shift(big, ddx, ddy, border_mode="reflect")
            stack.append(al[margin:-margin, margin:-margin])
        if not stack:
            continue
        patch = np.mean(stack, axis=0)
        if color:
            acc[ys:ys + box, xs:xs + box] += patch * win[..., None]
        else:
            acc[ys:ys + box, xs:xs + box] += patch * win
        wacc[ys:ys + box, xs:xs + box] += win
        if progress:
            progress(a + 1, len(aps), "Points d'alignement locaux")
    # 4) fondu avec l'empilement global là où les AP sont peu présents
    wn = np.clip(wacc / max(float(np.percentile(wacc[wacc > 0], 50)) if (wacc > 0).any() else 1.0,
                            1e-9), 0, 1)
    local = acc / np.maximum(wacc, 1e-12)[..., None] if color else acc / np.maximum(wacc, 1e-12)
    wb = wn[..., None] if color else wn
    out = np.where(wb > 0, local * wb + base * (1 - wb), base)
    return {"image": np.clip(out, 0, 1).astype(np.float32), "n_aps": len(aps),
            "frames_per_ap": k_ap, "reference": ref}


# -------------------------------------------------- pipeline complet (1→6)
def run_planetary(frames, quality_method: str = "Contraste HF (AutoStakkert)",
                  roi_mode: str = "Disque auto-détecté", n_align: int = 30,
                  align_mode: str = "Multi-points (grille NxN)", grid_n: int = 4,
                  box_size: int = 0, manual_points=None,
                  stack_method: str = "Moyenne", stack_percent: float = 100.0,
                  progress=None, ranked_scores=None, ranked_order=None,
                  ranked_roi=None) -> dict:
    """Enchaîne les 6 étapes : qualité + tri (2), sélection des N meilleures
    (3), alignement multipoint/global/manuel (4-5), empilement + image
    finale (6). Retourne {image, order, scores, selected, stats}."""
    # Conserver les vues différées ; seuls les N résultats alignés sont stockés.
    arr = frames
    n = len(arr)
    if n == 0:
        raise ValueError("Aucune frame chargée")
    # 2) qualité + tri
    roi = ranked_roi
    if ranked_order is not None and ranked_scores is not None:
        scores, order = ranked_scores, ranked_order
    else:
        if roi_mode.startswith("Disque"):
            stride = max(1, n // 20)
            mid = mean_sample(arr, stride, progress)
            roi = detect_planet_box(mid)
        scores = measure_quality(arr, quality_method, roi, progress=progress)
        order = sort_quality(scores)
    # 3) nombre d'images à aligner
    n_sel = int(np.clip(int(n_align), 1, n))
    sel = order[:n_sel]
    # 4) méthode + points
    if align_mode.startswith("Multi-points local"):
        ref0 = np.asarray(arr[int(sel[0])], np.float32)
        hh, ww = luminance(ref0).shape
        bx = int(box_size) if box_size and int(box_size) >= 16 else max(32, min(hh, ww) // 8)
        loc = local_multipoint_stack(arr, order, n_sel, roi, bx, grid_n,
                                     ap_percent=float(np.clip(stack_percent, 5, 100)),
                                     progress=progress)
        return {"image": loc["image"], "order": order, "scores": scores, "selected": sel,
                "stats": {"n_frames": n, "n_selected": n_sel, "roi": roi, "box": bx,
                          "n_points": loc["n_aps"], "rejected_boxes": 0, "rms_boxes": 0.0,
                          "shift_mean": 0.0, "shift_max": 0.0, "best_frame": int(order[0]),
                          "stacked": loc["frames_per_ap"]}}
    ref = arr[int(sel[0])]
    h, w = luminance(ref).shape
    box = int(box_size) if box_size and int(box_size) >= 16 else max(24, min(h, w) // 6)
    pts = None
    if align_mode.startswith("Points manuels"):
        pts = [(float(x), float(y)) for x, y in
               (manual_points or [(w / 2.0, h / 2.0)])]
    elif align_mode.startswith("Multi-points"):
        pts = grid_points(ref.shape, max(2, int(grid_n)), box, roi)
    # 5) alignement de toutes les images sélectionnées
    aligned = np.empty((n_sel,) + tuple(ref.shape), np.float32)
    tot_rej, shift_list, rms_sum = 0, [], 0.0
    for j, fi in enumerate(sel):
        if j == 0:
            aligned[0] = np.asarray(ref, np.float32)
        else:
            warped, st = align_frame(ref, arr[fi], pts, box, align_mode)
            aligned[j] = warped
            tot_rej += st["rejected"]
            rms_sum += st["rms"]
            shift_list.append(np.hypot(st["dx"], st["dy"]))
        if progress:
            progress(len(scores) + j + 1, len(scores) + n_sel)
    # 6) empilement -> image finale
    from .stacking import stack
    method = stack_method
    if method.startswith("Moyenne (meilleurs"):
        k = max(1, int(n_sel * float(np.clip(stack_percent, 5, 100)) / 100.0))
        aligned = aligned[:k]
        method = "Moyenne"
    img = stack(aligned, method, progress=progress)
    stats = {"n_frames": n, "n_selected": n_sel, "roi": roi, "box": box,
             "n_points": (1 if align_mode.startswith("Global")
                          else (len(pts) if pts else 0)),
             "rejected_boxes": tot_rej,
             "rms_boxes": rms_sum / max(1, n_sel - 1),
             "shift_mean": float(np.mean(shift_list)) if shift_list else 0.0,
             "shift_max": float(np.max(shift_list)) if shift_list else 0.0,
             "best_frame": int(order[0]), "stacked": int(aligned.shape[0])}
    return {"image": img, "order": order, "scores": scores,
            "selected": sel, "stats": stats}
