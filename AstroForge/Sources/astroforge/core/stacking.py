# -*- coding: utf-8 -*-
"""
Empilement : moyenne, médiane, sigma-clipping, winsorisé, rejet min/max
+ classement de qualité des images (planétaire).
Empilement par blocs de lignes pour limiter la mémoire.
"""
from __future__ import annotations
import numpy as np

METHODS = ["Moyenne", "Médiane", "Sigma-clipping", "Winsorisé", "Rejet min/max",
           "Linear fit", "MAD clipping"]


def _chunks(a: np.ndarray, budget: int = 64_000_000):
    """Découpe l'axe des lignes (axe 1) en tranches dont la pile N×lignes×…
    tient dans `budget` octets de travail (float32, ×4 pour les temporaires)."""
    n = a.shape[0]
    h = a.shape[1]
    per_row = n * int(np.prod(a.shape[2:])) * 4 * 4
    step = max(1, int(budget // max(per_row, 1)))
    for y0 in range(0, h, step):
        yield slice(y0, min(h, y0 + step))


def _xp(a):
    """NumPy ou CuPy selon le type du tableau (code commun CPU/GPU)."""
    if isinstance(a, np.ndarray):
        return np
    from . import gpu
    return gpu._load()


GPU_FUNCS = set()   # fonctions de bloc compatibles CuPy (voir core/gpu.py)


def _by_chunks_gpu(fn, a: np.ndarray, *args) -> np.ndarray:
    from . import gpu
    out = np.empty(a.shape[1:], np.float32)
    dargs = [gpu.to_device(np.asarray(x, np.float32)) if isinstance(x, np.ndarray) else x
             for x in args]
    for sl in _chunks(a, budget=256_000_000):
        d = gpu.to_device(np.asarray(a[:, sl], np.float32))
        out[sl] = gpu.to_host(fn(d, *dargs))
    return out


def _by_chunks(fn, a: np.ndarray, *args) -> np.ndarray:
    """Applique `fn` tranche de lignes par tranche, en parallèle sur les cœurs
    (NumPy libère le GIL pendant les tris et réductions) — ou sur GPU (CuPy)
    si activé et si `fn` le permet ; repli processeur en cas d'erreur GPU."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    a = a if a.ndim >= 3 else a[None]
    if fn in GPU_FUNCS:
        from . import gpu
        if gpu.enabled():
            try:
                return _by_chunks_gpu(fn, a, *args)
            except Exception:
                pass    # mémoire GPU insuffisante, pilote… : repli CPU
    out = np.empty(a.shape[1:], np.float32)
    workers = max(1, min(8, (os.cpu_count() or 2)))
    slices = list(_chunks(a, budget=max(8_000_000, 64_000_000 // workers)))

    def job(sl):
        out[sl] = fn(np.asarray(a[:, sl], np.float32), *args)
    if workers == 1 or len(slices) == 1:
        for sl in slices:
            job(sl)
    else:
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(job, slices))
    return out


def _wb(w, a):
    """Poids (N,) mis en forme pour diffuser sur un bloc (N, …)."""
    if w is None:
        return None
    xp = _xp(a)
    return xp.asarray(w, xp.float32).reshape((-1,) + (1,) * (a.ndim - 1))


def _wmean(a, mask, w):
    """Moyenne (pondérée si w) des valeurs conservées par `mask`."""
    xp = _xp(a)
    if w is None:
        cnt = mask.sum(axis=0)
        return (a * mask).sum(axis=0, dtype=xp.float64) / xp.maximum(cnt, 1), cnt
    ww = _wb(w, a) * mask
    tot = ww.sum(axis=0, dtype=xp.float64)
    return (a * ww).sum(axis=0, dtype=xp.float64) / xp.maximum(tot, 1e-12), tot


def _average_chunk(c, w=None):
    return _wmean(c, _xp(c).ones(c.shape, bool), w)[0]


def _median_chunk(c):
    return _xp(c).median(c, axis=0)


def combine_average(stack: np.ndarray, weights=None) -> np.ndarray:
    return _by_chunks(_average_chunk, np.asarray(stack) if not isinstance(stack, np.ndarray)
                      else stack, weights)


def combine_median(stack: np.ndarray) -> np.ndarray:
    return _by_chunks(_median_chunk, np.asarray(stack))


def _robust_center_scale(a: np.ndarray, mask: np.ndarray | None = None):
    """Médiane et écart-type robuste (1,4826 × MAD) par pixel, en ignorant
    les valeurs masquées (False)."""
    if mask is None:
        med = np.median(a, axis=0)
        mad = np.median(np.abs(a - med), axis=0)
    else:
        m = np.where(mask, a, np.nan)
        med = np.nanmedian(m, axis=0)
        mad = np.nanmedian(np.abs(m - med), axis=0)
        med = np.where(np.isnan(med), np.median(a, axis=0), med)
        mad = np.nan_to_num(mad)
    return med, 1.4826 * mad


def _sigma_clip_chunk(a: np.ndarray, sigma: float, iters: int, w=None) -> np.ndarray:
    # 1) pré-rejet robuste : médiane ± max(4, σ+1)·(1,4826·MAD) — un pixel
    #    aberrant ne gonfle pas l'écart qui sert à le juger, ce qui permet de le
    #    rejeter même avec 3 à 10 images (avec moyenne/écart-type classiques,
    #    l'écart maximal possible vaut (n−1)/√n σ < 3 σ pour n ≤ 10) ;
    # 2) itérations classiques moyenne ± σ·écart-type des valeurs conservées
    #    (efficacité statistique proche de la moyenne sur données saines).
    xp = _xp(a)
    med = xp.median(a, axis=0)
    rsd = 1.4826 * xp.median(xp.abs(a - med), axis=0)
    mask = xp.abs(a - med) <= max(4.0, sigma + 1.0) * xp.maximum(rsd, 1e-5)
    for _ in range(max(1, iters)):
        cnt = mask.sum(axis=0)
        c = (a * mask).sum(axis=0) / xp.maximum(cnt, 1)
        var = (((a - c) * mask) ** 2).sum(axis=0) / xp.maximum(cnt - 1, 1)
        sd = xp.sqrt(var)
        c = xp.where(cnt > 0, c, med)
        new = xp.abs(a - c) <= sigma * xp.maximum(sd, 1e-5)
        if bool((new == mask).all()):
            break
        mask = new
    mean, cnt = _wmean(a, mask, w)
    return xp.where(cnt > 0, mean, med).astype(xp.float32)


GPU_FUNCS.update({_average_chunk, _median_chunk, _sigma_clip_chunk})


def combine_sigma_clip(stack: np.ndarray, sigma: float = 3.0, iters: int = 3,
                       weights=None) -> np.ndarray:
    """Moyenne après rejet itératif des valeurs à plus de `sigma` écarts
    robustes (1,4826·MAD) de la médiane. Traitement par tranches de lignes."""
    return _by_chunks(_sigma_clip_chunk, stack if isinstance(stack, np.ndarray)
                      else np.asarray(stack), float(sigma), int(iters), weights)


def _winsor_chunk(a: np.ndarray, sigma: float, iters: int, w=None) -> np.ndarray:
    # σ initial robuste (MAD) puis estimation de Huber : winsorisation à
    # ±1,5 σ et σ = 1,134 × écart-type des données winsorisées, jusqu'à
    # convergence ; enfin écrêtage à médiane ± sigma·σ et moyenne.
    med = np.median(a, axis=0)
    sd = np.maximum(1.4826 * np.median(np.abs(a - med), axis=0), 1e-6)
    for _ in range(max(1, iters) * 3):
        wz = np.clip(a, med - 1.5 * sd, med + 1.5 * sd)
        new = np.maximum(1.134 * wz.std(axis=0), 1e-6)
        med = np.median(wz, axis=0)
        if np.allclose(new, sd, rtol=5e-4, atol=1e-7):
            sd = new
            break
        sd = new
    out = np.clip(a, med - sigma * sd, med + sigma * sd)
    return _wmean(out, np.ones(out.shape, bool), w)[0].astype(np.float32)


def combine_winsorized(stack: np.ndarray, sigma: float = 3.0, iters: int = 3,
                       weights=None) -> np.ndarray:
    """Sigma-clipping winsorisé : les valeurs hors de médiane ± sigma·σ sont
    ramenées à la borne (au lieu d'être rejetées), σ recalculé sur les
    données winsorisées, puis moyenne."""
    return _by_chunks(_winsor_chunk, stack if isinstance(stack, np.ndarray)
                      else np.asarray(stack), float(sigma), int(iters), weights)


def _linear_fit_chunk(c: np.ndarray, norm: np.ndarray, k: float, w=None) -> np.ndarray:
    ref = np.median(c, axis=0)
    dev = c - norm.reshape((-1,) + (1,) * (c.ndim - 1)) * ref[None]
    sig = 1.4826 * np.median(np.abs(dev - np.median(dev, axis=0)), axis=0) + 1e-6
    mask = np.abs(dev) <= k * sig
    mean, cnt = _wmean(c, mask, w)
    return np.where(cnt > 0, mean, ref)


def combine_linear_fit(a: np.ndarray, k: float = 2.5, weights=None) -> np.ndarray:
    """Linear fit clipping : chaque frame est mise à l'échelle sur l'image
    médiane (droite y = s·x), rejet des pixels à plus de k·sigma robuste de la
    droite, moyenne du reste (par tranches de lignes)."""
    a = np.asarray(a)
    if a.ndim == 2:
        a = a[None]
    n = a.shape[0]
    norm = np.array([float(np.median(np.asarray(a[i], np.float32))) for i in range(n)])
    norm = norm / max(float(np.median(norm)), 1e-9)
    return _by_chunks(_linear_fit_chunk, a, norm, float(k), weights)


def _mad_chunk(a: np.ndarray, k: float, w=None) -> np.ndarray:
    med = np.median(a, axis=0)
    mad = 1.4826 * np.median(np.abs(a - med), axis=0) + 1e-6
    mask = np.abs(a - med) <= k * mad
    mean, cnt = _wmean(a, mask, w)
    return np.where(cnt > 0, mean, med)


def combine_mad(a: np.ndarray, k: float = 3.0, weights=None) -> np.ndarray:
    """MAD clipping : rejet par médiane absolue (robuste, peu d'itérations),
    valeur médiane conservée quand tout est rejeté."""
    return _by_chunks(_mad_chunk, a if isinstance(a, np.ndarray) else np.asarray(a),
                      float(k), weights)


def _minmax_chunk(a: np.ndarray, low: float, high: float) -> np.ndarray:
    n = a.shape[0]
    k_lo = int(np.floor(low / 100.0 * n))
    k_hi = int(np.floor(high / 100.0 * n))
    part = np.sort(a, axis=0)
    if 0 < k_lo + k_hi < n:
        part = part[k_lo:n - k_hi]
    return part.mean(axis=0, dtype=np.float64)


def combine_minmax(stack: np.ndarray, low: float = 1.0, high: float = 1.0) -> np.ndarray:
    """Rejette un pourcentage bas/haut de chaque pixel avant la moyenne."""
    return _by_chunks(_minmax_chunk, np.asarray(stack), float(low), float(high))


def stack(stack_array, method: str = "Sigma-clipping",
          sigma: float = 3.0, iters: int = 3, low: float = 1.0, high: float = 1.0,
          progress=None, weights=None) -> np.ndarray:
    """Empile un tableau N×H×W(×3) (ndarray ou np.memmap), par tranches de
    lignes pour limiter la mémoire. `weights` (N,) : poids par image (analyse
    des brutes) — moyenne pondérée des valeurs conservées après rejet ; un
    poids nul exclut l'image. Médiane et rejet min/max ignorent les poids
    (hors exclusion)."""
    a = stack_array if isinstance(stack_array, np.ndarray) else np.asarray(stack_array, np.float32)
    if a.ndim == 2:
        a = a[None, ...]
    w = None
    if weights is not None:
        w = np.asarray(weights, np.float64)
        if len(w) != a.shape[0]:
            raise ValueError(f"{len(w)} poids pour {a.shape[0]} images")
        keep = w > 0
        if not keep.any():
            raise ValueError("Toutes les images ont un poids nul")
        if not keep.all():
            a = a[np.nonzero(keep)[0]]
            w = w[keep]
        if np.allclose(w, w[0]):
            w = None
    if method == "Moyenne":
        return combine_average(a, w)
    if method == "Médiane":
        return combine_median(a)
    if method == "Sigma-clipping":
        return combine_sigma_clip(a, sigma, iters, w)
    if method == "Winsorisé":
        return combine_winsorized(a, sigma, iters, w)
    if method == "Rejet min/max":
        return combine_minmax(a, low, high)
    if method == "Linear fit":
        return combine_linear_fit(a, sigma, w)
    if method == "MAD clipping":
        return combine_mad(a, sigma, w)
    raise ValueError(f"Méthode d'empilement inconnue : {method}")


def stack_paths(paths: list[str], method: str = "Sigma-clipping", sigma: float = 3.0,
                iters: int = 3, chunk: int = 8, progress=None, bayer: str | None = None,
                **kw) -> np.ndarray:
    """Empile des fichiers depuis le disque sans tout garder en RAM : chaque
    image est lue une fois et recopiée dans une pile np.memmap temporaire,
    puis l'empilement est fait par tranches de lignes."""
    import os
    import tempfile
    from .image_io import load_image
    n = len(paths)
    if n == 0:
        raise ValueError("Aucune image à empiler")
    first = np.asarray(load_image(paths[0], bayer), np.float32)
    if method == "Moyenne":
        acc = first.astype(np.float64)
        for i, p in enumerate(paths[1:], start=2):
            im = np.asarray(load_image(p, bayer), np.float32)
            if im.shape != first.shape:
                raise ValueError(f"Taille différente : {p} {im.shape} ≠ {first.shape}")
            acc += im
            if progress:
                progress(i, n)
        return (acc / n).astype(np.float32)
    fd, tmp = tempfile.mkstemp(prefix="astroforge_stack_", suffix=".dat")
    os.close(fd)
    try:
        mm = np.memmap(tmp, dtype=np.float32, mode="w+", shape=(n,) + first.shape)
        mm[0] = first
        for i, p in enumerate(paths[1:], start=1):
            im = np.asarray(load_image(p, bayer), np.float32)
            if im.shape != first.shape:
                raise ValueError(f"Taille différente : {p} {im.shape} ≠ {first.shape}")
            mm[i] = im
            if progress:
                progress(i + 1, n)
        mm.flush()
        out = stack(mm, method, sigma=sigma, iters=iters, **kw)
        del mm
        return out
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


# ------------------------------------------------------------------ qualité
def sharpness_score(img: np.ndarray) -> float:
    """Netteté : moyenne du laplacien absolu (luminance)."""
    cv2 = __import__("cv2")
    from .image_io import is_color, to_uint16
    a = img[:, :, 1] if is_color(img) else img
    lap = cv2.Laplacian(to_uint16(a), cv2.CV_32F)
    return float(np.abs(lap).mean()) + 1e-9


def brightness_score(img: np.ndarray) -> float:
    """Proximité de l'exposition idéale (pic à ~0.5)."""
    from .image_io import is_color
    a = img[:, :, 1] if is_color(img) else img
    m = float(np.mean(a))
    return 1.0 - abs(m - 0.5) * 2.0


def quality_scores(frames: np.ndarray, mode: str = "Netteté (Registax)") -> np.ndarray:
    """Score de qualité de chaque frame (classement type AutoStacker/Registax)."""
    n = frames.shape[0]
    scores = np.zeros(n, np.float64)
    for i in range(n):
        s = sharpness_score(frames[i])
        b = brightness_score(frames[i])
        if mode.startswith("Netteté"):
            scores[i] = s * (0.5 + 0.5 * max(b, 0.0))
        elif mode.startswith("Luminosité"):
            scores[i] = max(b, 1e-9)
        else:  # mixte
            scores[i] = s * max(b, 1e-9)
    return scores


def rank_frames(frames: np.ndarray, mode: str = "Netteté (Registax)",
                keep_percent: float = 75.0) -> tuple[np.ndarray, np.ndarray]:
    """Retourne (indices triés du meilleur au moins bon, scores), en ne gardant
    que les `keep_percent` % meilleures frames."""
    scores = quality_scores(frames, mode)
    order = np.argsort(-scores)
    k = max(1, int(round(len(order) * keep_percent / 100.0)))
    return order[:k], scores


# ------------------------------------------- empilement basé AutoStakkert!
def _detail_score(img: np.ndarray) -> float:
    import scipy.ndimage as ndi
    mono = img if img.ndim == 2 else img.mean(axis=2)
    return float(np.abs(mono - ndi.median_filter(mono, 5)).mean())


def as_quality_scores(frames: np.ndarray) -> np.ndarray:
    """Qualité par frame : contraste local haute fréquence (esprit AS!)."""
    return np.array([_detail_score(f) for f in np.asarray(frames, np.float32)],
                    np.float64)


def _box_shifts(ref: np.ndarray, mov: np.ndarray, grid_n: int, box: int):
    """Alignement multipoint : décalage (phase_corr) + poids (détail) par boîte."""
    from .registration import phase_corr
    h, w = ref.shape[:2]
    ys = np.linspace(box // 2, h - box // 2, grid_n)
    xs = np.linspace(box // 2, w - box // 2, grid_n)
    shifts = np.zeros((grid_n, grid_n, 2), np.float32)
    weights = np.zeros((grid_n, grid_n), np.float32)
    for iy, cy in enumerate(ys):
        for ix, cx in enumerate(xs):
            y0, x0 = int(cy - box / 2), int(cx - box / 2)
            rb = ref[y0:y0 + box, x0:x0 + box]
            mb = mov[y0:y0 + box, x0:x0 + box]
            dx, dy, _ = phase_corr(rb, mb)
            shifts[iy, ix] = (dx, dy)
            weights[iy, ix] = _detail_score(rb) + 1e-9
    return shifts, weights


def autostacker_stack(frames, keep_percent: float = 75.0, grid_n: int = 4,
                      box_frac: float = 0.3, progress=None) -> np.ndarray:
    """Empilement basé AutoStakkert! : qualité par frame, sélection des
    meilleures, alignement multipoint (champ de décalage lissé) puis moyenne."""
    import cv2
    arr = np.stack([np.asarray(f, np.float32) for f in frames])
    if arr.ndim == 2:
        arr = arr[None]
    n = len(arr)
    order = np.argsort(-as_quality_scores(arr))
    k = max(2, int(round(n * float(np.clip(keep_percent, 5, 100)) / 100.0)))
    kept = order[:k]
    ref = arr[kept[0]]
    h, w = ref.shape[:2]
    box = max(32, int(min(h, w) * box_frac))
    acc = np.zeros_like(ref, np.float64)
    for j, fi in enumerate(kept):
        f = arr[fi]
        if j == 0:
            field = np.zeros((grid_n, grid_n, 2), np.float32)
        else:
            field, _wt = _box_shifts(ref, f, grid_n, box)
        maps = [cv2.resize(np.ascontiguousarray(field[..., c], np.float32),
                           (w, h), interpolation=cv2.INTER_CUBIC) for c in range(2)]
        map_x = np.broadcast_to(np.arange(w, dtype=np.float32), (h, w)) - maps[0]
        map_y = np.broadcast_to(np.arange(h, dtype=np.float32)[:, None],
                                (h, w)) - maps[1]
        warped = cv2.remap(f, map_x, map_y, interpolation=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REFLECT)
        acc += warped
        if progress:
            progress(j + 1, k)
    return (acc / len(kept)).astype(np.float32)
