# -*- coding: utf-8 -*-
"""
Filtres de retouche astronomique de l'onglet Finition.

Ces fonctions sont des implémentations ORIGINALES numPy/SciPy, inspirées des
PRINCIPES généraux des logiciels Photoshop, PixInsight, Siril et
AstroPixelProcessor (mélange avec la luminance, neutralisation de canal,
médiane/MAD, extraction de fond lissé) — elles ne sont PAS le portage des
algorithmes de ces logiciels commerciaux.

Contrat commun :
- image mono (H, W) ou RGB (H, W, 3) en flottant, sortie float32 dans [0, 1] ;
- le tableau source n'est jamais modifié (copie défensive) ;
- paramètres neutres = identité numérique (saturation 1, gains 1, gain 0,
  rayon 0, force 0…) sauf pour l'étirement automatique et la suppression de
  gradient qui sont par nature transformants (image constante → copie).
Les mêmes filtres existent en OPÉRATIONS de séquence (core.sequence, noms
fin_*), réutilisables dans les lots d'images.
"""
from __future__ import annotations

import numpy as np
import scipy.ndimage as ndi

# Spécification des filtres pour l'interface : (clé, libellé, min, max, pas,
# défaut, décimales) ; décimales 0 = QSpinBox, sinon QDoubleSpinBox ; la clé
# spéciale « method » donne une liste déroulante (chaîne).
FILTERS_CLASSIC = [
    ("saturation", "Saturation", "fin_saturation",
     [("amount", "Facteur (0 à 2)",
       0.0, 2.0, 0.05, 1.0, 2)]),
    ("white_balance", "Balance des blancs (R/V/B)", "fin_white_balance",
     [("wr", "Gain rouge", 0.0, 2.0, 0.05, 1.0, 2),
      ("wg", "Gain vert", 0.0, 2.0, 0.05, 1.0, 2),
      ("wb", "Gain bleu", 0.0, 2.0, 0.05, 1.0, 2)]),
    ("background_neutral", "Neutralisation du fond", "fin_background_neutral",
     [("percentile", "Centile du fond", 0.5, 25.0, 0.5, 5.0, 1),
      ("strength", "Force", 0.0, 1.0, 0.05, 1.0, 2)]),
    ("scnr_green", "Suppression du vert (SCNR)", "fin_scnr_green",
     [("strength", "Force", 0.0, 1.0, 0.05, 1.0, 2)]),
    ("autostretch", "Étirement automatique (médiane/MAD, tons moyens)", "fin_autostretch",
     [("shadows_clip", "Point noir (σ)", -6.0, 0.0, 0.1, -2.8, 1),
      ("target_bg", "Fond cible (0,05 à 0,5)", 0.05, 0.5, 0.01, 0.25, 2)]),
    ("remove_gradient", "Suppression de gradient (extraction de fond)",
     "fin_gradient",
     [("kernel", "Noyau (px)", 16, 256, 8, 64, 0),
      ("method", "Méthode", 0, 0, 0, 0, -1)]),
    ("denoise_median", "Débruitage médian (répété)", "fin_median_denoise",
     [("radius", "Rayon (px)", 0, 10, 1, 1, 0),
      ("iterations", "Répétitions", 1, 8, 1, 1, 0)]),
    ("unsharp_mask", "Masque flou (netteté)", "fin_unsharp",
     [("radius", "Rayon (σ)", 0.2, 20.0, 0.1, 2.0, 1),
      ("gain", "Gain", 0.0, 2.0, 0.05, 0.6, 2)]),
]

# Filtres « couleur & détail » (esprit Affinity Photo / PixInsight) — mêmes
# contrats (float [0,1] mono/RGB, identité aux paramètres neutres sauf
# mention contraire) ; banque séparée dans l'interface, opérations fin_*.
FILTERS_COLOR = [
    ("levels", "Niveaux", "fin_levels",
     [("black", "Noir (0 à 0,5)", 0.0, 0.5, 0.01, 0.0, 2),
      ("gamma", "Gamma (0,2 à 5)", 0.2, 5.0, 0.05, 1.0, 2),
      ("white", "Blanc (0,5 à 1)", 0.5, 1.0, 0.01, 1.0, 2)]),
    ("shadows_highlights", "Tons clairs / tons sombres", "fin_shadows_highlights",
     [("shadows", "Tons sombres (force −1 à 1)", -1.0, 1.0, 0.05, 0.0, 2),
      ("highlights", "Tons clairs (force −1 à 1)", -1.0, 1.0, 0.05, 0.0, 2),
      ("radius", "Rayon du masque (px)", 2, 200, 2, 24, 0)]),
    ("vibrance", "Vibrance", "fin_vibrance",
     [("amount", "Intensité (1 = neutre)", 0.0, 2.0, 0.05, 1.0, 2)]),
    ("hsl", "Teinte / saturation / luminosité (HSL)", "fin_hsl",
     [("hue", "Teinte globale (°)", -180, 180, 5, 0, 0),
      ("sat", "Saturation globale (1 = neutre)", 0.0, 2.0, 0.05, 1.0, 2),
      ("lum", "Luminosité globale", -1.0, 1.0, 0.05, 0.0, 2),
      ("r_hue", "Rouge — teinte (°)", -30, 30, 1, 0, 0),
      ("r_sat", "Rouge — saturation", 0.0, 2.0, 0.05, 1.0, 2),
      ("y_hue", "Jaune — teinte (°)", -30, 30, 1, 0, 0),
      ("y_sat", "Jaune — saturation", 0.0, 2.0, 0.05, 1.0, 2),
      ("g_hue", "Vert — teinte (°)", -30, 30, 1, 0, 0),
      ("g_sat", "Vert — saturation", 0.0, 2.0, 0.05, 1.0, 2),
      ("c_hue", "Cyan — teinte (°)", -30, 30, 1, 0, 0),
      ("c_sat", "Cyan — saturation", 0.0, 2.0, 0.05, 1.0, 2),
      ("b_hue", "Bleu — teinte (°)", -30, 30, 1, 0, 0),
      ("b_sat", "Bleu — saturation", 0.0, 2.0, 0.05, 1.0, 2),
      ("m_hue", "Magenta — teinte (°)", -30, 30, 1, 0, 0),
      ("m_sat", "Magenta — saturation", 0.0, 2.0, 0.05, 1.0, 2)]),
    ("local_contrast", "Contraste local (clarté, esprit LHE)", "fin_local_contrast",
     [("radius", "Rayon (px, 4 à 200)", 4, 200, 4, 32, 0),
      ("gain", "Gain (0 = neutre)", 0.0, 2.0, 0.05, 0.0, 2)]),
    ("split_tone", "Tonalité divisée", "fin_split_tone",
     [("sh_r", "Ombres — rouge", 0.0, 1.0, 0.05, 0.5, 2),
      ("sh_g", "Ombres — vert", 0.0, 1.0, 0.05, 0.5, 2),
      ("sh_b", "Ombres — bleu", 0.0, 1.0, 0.05, 0.5, 2),
      ("hl_r", "Hautes lumières — rouge", 0.0, 1.0, 0.05, 0.5, 2),
      ("hl_g", "Hautes lumières — vert", 0.0, 1.0, 0.05, 0.5, 2),
      ("hl_b", "Hautes lumières — bleu", 0.0, 1.0, 0.05, 0.5, 2),
      ("balance", "Équilibre (−1 ombres, +1 hauts)", -1.0, 1.0, 0.05, 0.0, 2),
      ("force", "Force", 0.0, 1.0, 0.05, 0.0, 2)]),
    ("gradient_map", "Mappage de dégradé", "fin_gradient_map",
     [("r0", "Arrêt ombres — rouge", 0.0, 1.0, 0.05, 0.0, 2),
      ("g0", "Arrêt ombres — vert", 0.0, 1.0, 0.05, 0.0, 2),
      ("b0", "Arrêt ombres — bleu", 0.0, 1.0, 0.05, 0.0, 2),
      ("r1", "Arrêt milieux — rouge", 0.0, 1.0, 0.05, 0.5, 2),
      ("g1", "Arrêt milieux — vert", 0.0, 1.0, 0.05, 0.5, 2),
      ("b1", "Arrêt milieux — bleu", 0.0, 1.0, 0.05, 0.5, 2),
      ("r2", "Arrêt hauts — rouge", 0.0, 1.0, 0.05, 1.0, 2),
      ("g2", "Arrêt hauts — vert", 0.0, 1.0, 0.05, 1.0, 2),
      ("b2", "Arrêt hauts — bleu", 0.0, 1.0, 0.05, 1.0, 2),
      ("intensity", "Intensité (0 = neutre)", 0.0, 1.0, 0.05, 0.0, 2)]),
]

# Filtres « astro avancés » (banque 1, après les classiques) : étirements
# non linéaires GHS / arcsinh, réduction d'étoiles, débruitage de chrominance.
FILTERS_ADVANCED = [
    ("ghs", "Étirement hyperbolique généralisé (GHS)", "fin_ghs",
     [("D", "Intensité D (0 = neutre, 0 à 10)", 0.0, 10.0, 0.1, 0.0, 2),
      ("b", "Type b (−1 log, 0 exp., 1 harmonique…)", -5.0, 15.0, 0.5, 0.0, 1),
      ("SP", "Point de symétrie (contraste max.)", 0.0, 1.0, 0.005, 0.0, 3),
      ("LP", "Protection des ombres", 0.0, 1.0, 0.005, 0.0, 3),
      ("HP", "Protection des hautes lumières", 0.0, 1.0, 0.005, 1.0, 3),
      ("method", "Application", 0, 0, 0, 0, -1)]),
    ("arcsinh", "Étirement arcsinh (couleurs préservées)", "fin_arcsinh",
     [("stretch", "Intensité β (0 = neutre)", 0.0, 1000.0, 1.0, 0.0, 1),
      ("black", "Point noir", 0.0, 0.5, 0.001, 0.0, 3)]),
    ("star_reduction", "Réduction d'étoiles", "fin_star_reduction",
     [("amount", "Force (0 = neutre)", 0.0, 1.0, 0.05, 0.0, 2),
      ("iterations", "Passes d'érosion", 1, 4, 1, 1, 0)]),
    ("chroma_denoise", "Débruitage de la chrominance (luminance intacte)",
     "fin_chroma_denoise",
     [("strength", "Force (0 = neutre)", 0.0, 1.0, 0.05, 0.0, 2),
      ("scales", "Échelles", 1, 6, 1, 4, 0)]),
]

# choix des listes déroulantes (clé spéciale « method »)
CHOICES = {"remove_gradient": ["Soustraire", "Diviser"],
           "ghs": ["Canaux", "Luminance (couleurs préservées)"]}

FILTERS = FILTERS_CLASSIC + FILTERS_COLOR + FILTERS_ADVANCED

OP_FOR = {fid: opname for fid, _label, opname, _specs in FILTERS}
LABELS = {fid: label for fid, label, _opname, _specs in FILTERS}
_DEFAULTS = {fid: {s[0]: (CHOICES.get(fid, ["Soustraire"])[0] if s[0] == "method" else s[5])
                  for s in specs}
             for fid, _label, _opname, specs in FILTERS}


def neutral_params(filter_id: str) -> dict:
    """Paramètres par défaut (neutres) d'un filtre."""
    return dict(_DEFAULTS[filter_id])


def _as_image(img):
    a = np.asarray(img, np.float32)
    if a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[2] not in (1, 3)):
        raise ValueError("Image attendue : mono ou RGB.")
    if a.size == 0:
        raise ValueError("Image vide.")
    return a


def _out(a):
    return np.clip(a, 0, 1).astype(np.float32)


def luminance(img) -> np.ndarray:
    """Luminance Rec.709 (mono : telle quelle) — base des mesures statistiques."""
    a = _as_image(img)
    if a.ndim == 2:
        return a
    return (0.2126 * a[..., 0] + 0.7152 * a[..., 1]
            + 0.0722 * a[..., 2]).astype(np.float32)


def saturation(img, amount: float = 1.0):
    """Saturation par mélange avec la luminance : out = lum + (img − lum)·a.
    Aucune conversion HSV entière : pas d'artefacts de quantification.
    (Principe général inspiré des réglages de saturation ; implémentation
    originale.) Mono : copie écrêtée (identité)."""
    a = _as_image(img)
    k = float(np.clip(amount, 0.0, 2.0))
    if a.ndim == 2 or k == 1.0:
        return _out(a)
    lum = luminance(a)[..., None]
    return _out(lum + (a - lum) * k)


def white_balance(img, wr: float = 1.0, wg: float = 1.0, wb: float = 1.0):
    """Gains multiplicatifs par canal (0..2) — équilibre chromatique simple.
    Mono : copie écrêtée (aucun canal distinct)."""
    a = _as_image(img).copy()
    if a.ndim == 3:
        for i, g in enumerate((wr, wg, wb)):
            a[..., i] *= float(np.clip(g, 0.0, 2.0))
    return _out(a)


def neutralize_background(img, percentile: float = 5.0, strength: float = 1.0):
    """NEUTRALISATION DU FOND — inspirée du principe BackgroundNeutralization
    de PixInsight, implémentation originale : le fond est estimé au faible
    centile de chaque canal (statistiques d'ordre, robustes aux étoiles) ;
    l'écart de ce fond au gris médian est retranché, ce qui rend le fond
    neutre sans écraser les hautes lumières. `strength` module l'effet.
    Mono : aucune chrominance, copie écrêtée."""
    a = _as_image(img).copy()
    if a.ndim == 2:
        return _out(a)
    p = float(np.clip(percentile, 0.0, 49.0))
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0.0:
        return _out(a)
    bg = [float(np.percentile(a[..., i], p)) for i in range(3)]
    target = float(np.median(bg))
    for i in range(3):
        a[..., i] -= (bg[i] - target) * s
    return _out(a)


def scnr_green(img, strength: float = 1.0):
    """SUPPRESSION DU VERT « moyenne neutre » — inspirée du SCNR Average
    Neutral, implémentation originale : le vert est plafonné par la moyenne
    des canaux rouge et bleu, G' = min(G, (R+B)/2), mélangé par `strength`.
    Mono : copie écrêtée."""
    a = _as_image(img).copy()
    if a.ndim == 2:
        return _out(a)
    s = float(np.clip(strength, 0.0, 1.0))
    if s > 0.0:
        avg = (a[..., 0] + a[..., 2]) * 0.5
        g = np.minimum(a[..., 1], avg)
        a[..., 1] = a[..., 1] * (1.0 - s) + g * s
    return _out(a)


def autostretch(img, shadows_clip: float = -2.8, target_bg: float = 0.25,
                spread: float | None = None):
    """ÉTIREMENT AUTOMATIQUE — inspiré du principe de la fonction d'écran
    (STF) de PixInsight, implémentation originale : point noir = médiane +
    shadows_clip·σ (σ = MAD recalibrée de la LUMINANCE), puis fonction de
    transfert des tons moyens (MTF) qui amène le fond à `target_bg`. Non
    linéaire : les hautes lumières sont compressées, jamais écrêtées.
    Image quasi constante (σ négligeable) : copie inchangée. `spread`
    (ancien point blanc linéaire) est accepté et ignoré."""
    from .display import mtf
    a = _as_image(img)
    lum = np.asarray(luminance(a), np.float64)
    med = float(np.median(lum))
    sigma = float(np.median(np.abs(lum - med))) * 1.4826
    if sigma < 1e-6:
        return _out(a)
    black = float(np.clip(med + float(shadows_clip) * sigma, 0.0, max(med - 1e-6, 0.0)))
    x0 = (med - black) / max(1.0 - black, 1e-6)
    t = float(np.clip(target_bg, 0.05, 0.5))
    if x0 <= 1e-7:
        return _out(a)
    m = float(np.clip(x0 * (1.0 - t) / (x0 - 2.0 * t * x0 + t), 1e-6, 1 - 1e-6))
    return _out(mtf(m, np.clip((a - black) / max(1.0 - black, 1e-6), 0, 1)))


def remove_gradient(img, kernel: int = 64, method: str = "Soustraire"):
    """SUPPRESSION DE GRADIENT PAR EXTRACTION DE FOND — inspirée des
    principes ABE/DBE de PixInsight, DynamicBackgroundExtraction de Siril et
    d'AstroPixelProcessor, implémentation originale : le fond est le lissage
    gaussien très large de l'image (σ = noyau/6, réglable) ; mode
    « Soustraire » : out = img − fond + médiane(fond) (objets préservés) ;
    mode « Diviser » : out = img · médiane(fond) / fond (aplatissement en
    quotient). Mono/RGB : fond lissé canal par canal, niveau recalé sur la
    médiane de la luminance du fond."""
    a = _as_image(img)
    sigma = max(1.0, float(kernel) / 6.0)
    if a.ndim == 3:
        bg = np.stack([ndi.gaussian_filter(a[..., c], sigma)
                       for c in range(a.shape[2])], axis=-1)
    else:
        bg = ndi.gaussian_filter(a, sigma)
    level = float(np.median(luminance(bg)))
    if str(method).startswith("Div"):
        return _out(a * (level / np.maximum(bg, 1e-3)))
    return _out(a - bg + level)


def denoise_median(img, radius: int = 1, iterations: int = 1):
    """DÉBRUITAGE MÉDIAN — filtre médian (scipy.ndimage.median_filter) par
    canal, taille 2·rayon+1, appliqué `iterations` fois. Rayon 0 : copie.
    (Filtre standard du domaine, comme dans tous les éditeurs d'images.)"""
    a = _as_image(img)
    r = int(np.clip(int(radius), 0, 20))
    n = int(np.clip(int(iterations), 1, 8))
    if r == 0:
        return _out(a)
    out = _out(a)
    size = 2 * r + 1
    for _ in range(n):
        if out.ndim == 3:
            for c in range(out.shape[2]):
                out[..., c] = ndi.median_filter(out[..., c], size=size)
        else:
            out = ndi.median_filter(out, size=size)
    return out.astype(np.float32)


def unsharp_mask(img, radius: float = 2.0, gain: float = 0.6):
    """MASQUE FLOU — principe universel des éditeurs (Photoshop unsharp
    mask) : out = img + gain·(img − flou_gaussien(img)), σ = rayon.
    Gain 0 ou rayon nul : copie écrêtée (identité sur image dans [0,1])."""
    a = _as_image(img)
    g = float(np.clip(gain, 0.0, 3.0))
    r = float(radius)
    if g <= 0.0 or r <= 0.0:
        return _out(a)
    sigma = max(0.3, r)
    if a.ndim == 3:
        blur = ndi.gaussian_filter(a, sigma=(sigma, sigma, 0))
    else:
        blur = ndi.gaussian_filter(a, sigma)
    return _out(a + g * (a - blur))


def apply_curves(img, curves: dict) -> np.ndarray:
    """Courbes tonales par canal : « commune » s'applique à tous les canaux
    (comme core.planetary.adjust_tone, mono/RGB), puis « r »/« g »/« b » à
    leur canal (RGB uniquement, ignorées en mono). Chaque valeur est une
    courbe de points [[x, y], …] normalisée comme au planétaire
    (None ou identité : aucun changement)."""
    from .planetary import _normalize_curve, is_identity_curve
    a = _as_image(img)
    norm = {}
    for key in ("commune", "r", "g", "b"):
        curve = curves.get(key)
        if curve:
            pts = _normalize_curve(curve)
            if not is_identity_curve(pts):
                norm[key] = ([pt[0] for pt in pts], [pt[1] for pt in pts])
    if not norm:
        return _out(a)
    out = a.copy()   # la source n'est jamais modifiée
    commune = norm.get("commune")
    if commune:
        out = np.interp(out, commune[0], commune[1])
    if out.ndim == 3:
        for idx, key in ((0, "r"), (1, "g"), (2, "b")):
            chan = norm.get(key)
            if chan:
                out[..., idx] = np.interp(out[..., idx], chan[0], chan[1])
    return _out(out)


def levels(img, black: float = 0.0, gamma: float = 1.0, white: float = 1.0):
    """NIVEAUX — principe des curseurs de niveaux partagé par tous les
    éditeurs (implémentation maison) : l'entrée [noir, blanc] est ramenée
    sur [0, 1] avec écrêtage puis exposée à 1/gamma (gamma 1 = rectiligne).
    Équivalent à une courbe en trois points (noir→0, milieu→0,5^(1/gamma),
    blanc→1) appliquée exactement ; noir ≤ 0,5 ≤ blanc imposé."""
    a = _as_image(img)
    b = float(np.clip(black, 0.0, 0.5))
    w = float(np.clip(white, 0.5, 1.0))
    g = float(np.clip(gamma, 0.2, 5.0))
    if w - b < 1e-4:
        return _out(a)
    t = np.clip((a - b) / (w - b), 0.0, 1.0)
    if abs(g - 1.0) > 1e-9:
        t = np.power(t, 1.0 / g)
    return _out(t)


def shadows_highlights(img, shadows: float = 0.0, highlights: float = 0.0,
                       radius: float = 24.0):
    """TONS CLAIRS / TONS SOMBRES — inspiré du principe ombres/claires
    (implémentation maison) : masque de luminance lissé (gaussienne de
    `radius` px, poids au carré) ; la force des tons sombres (> 0 dévoile,
    < 0 assombrit) agit en zone sombre, celle des tons clairs en zone
    claire. Le réglage est additif sur la luminance puis redistribué aux
    canaux par le rapport L'/L : les rapports de canaux (chromaticité) sont
    conservés — les étoiles ne se désaturent pas — et les hautes lumières
    restent intactes au-delà du flanc du masque."""
    a = _as_image(img).copy()
    f_sh = float(np.clip(shadows, -1.0, 1.0))
    f_hl = float(np.clip(highlights, -1.0, 1.0))
    if f_sh == 0.0 and f_hl == 0.0:
        return _out(a)
    lum = np.asarray(luminance(a), np.float32)
    lum_s = ndi.gaussian_filter(lum, max(0.5, float(radius)))
    w_sh = np.clip(1.0 - lum_s, 0.0, 1.0) ** 2
    w_hl = np.clip(lum_s, 0.0, 1.0) ** 2
    new_lum = np.clip(lum + (f_sh * w_sh + f_hl * w_hl) * 0.4, 0.0, 1.0)
    ratio = new_lum / np.maximum(lum, 1e-3)
    if a.ndim == 3:
        out = a * ratio[..., None]
    else:
        out = a * ratio
    return _out(out)


def vibrance(img, amount: float = 1.0):
    """VIBRANCE — principe commun aux éditeurs modernes (implémentation
    maison) : amplifie d'abord la saturation des pixels PEU saturés :
    out = img + (a−1)·(1−s)·(img − lum), s = chroma relatif par pixel.
    Les pixels gris (img = lum) ne bougent jamais, les pixels saturés
    presque pas ; a = 1 : identité exacte."""
    a = _as_image(img)
    k = float(np.clip(amount, 0.0, 2.0)) - 1.0
    if a.ndim == 2 or k == 0.0:
        return _out(a)
    lum = luminance(a)[..., None]
    mx = a.max(axis=2, keepdims=True)
    mn = a.min(axis=2, keepdims=True)
    s = np.clip((mx - mn) / np.maximum(mx, 1e-3), 0.0, 1.0)
    return _out(a + k * (1.0 - s) * (a - lum))


def _rgb_to_hsv(a):
    """RGB float [0,1] → teinte [0,1), saturation [0,1], valeur [0,1]
    (tout en flottant, aucune quantification entière)."""
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx = a.max(axis=-1)
    mn = a.min(axis=-1)
    d = mx - mn
    safe = np.maximum(d, 1e-12)
    cmax = a.argmax(axis=-1)
    h = np.zeros_like(mx)
    m0, m1, m2 = cmax == 0, cmax == 1, cmax == 2
    h[m0] = ((g - b)[m0] / safe[m0]) % 6
    h[m1] = (b - r)[m1] / safe[m1] + 2
    h[m2] = (r - g)[m2] / safe[m2] + 4
    h = (h / 6) % 1
    s = np.where(mx > 1e-12, d / np.maximum(mx, 1e-12), 0.0)
    return h, s.astype(np.float32), mx


def _hsv_to_rgb(h, s, v):
    i = np.floor(h * 6).astype(np.int32) % 6
    f = h * 6 - np.floor(h * 6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [v, q, p, p, t, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [t, v, v, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5],
                  [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


def hsl(img, hue: float = 0.0, sat: float = 1.0, lum: float = 0.0,
        r_hue: float = 0, r_sat: float = 1.0, y_hue: float = 0,
        y_sat: float = 1.0, g_hue: float = 0, g_sat: float = 1.0,
        c_hue: float = 0, c_sat: float = 1.0, b_hue: float = 0,
        b_sat: float = 1.0, m_hue: float = 0, m_sat: float = 1.0):
    """HSL GLOBAL + PAR GAMMES — teinte (−180..180°), saturation (×0..2),
    luminosité (−1..1) globales, puis décalages de teinte (−30..30°) et de
    saturation (×0..2) pour six gammes (rouge, jaune, vert, cyan, bleu,
    magenta) avec fenêtre cosinus douce (demi-largeur 75°, lissée) — tout
    en flottant, sans quantification HSV entière (implémentation maison).
    Mono : seule la luminosité globale s'applique. Paramètres neutres :
    identité exacte."""
    a = _as_image(img)
    hue = float(np.clip(hue, -180.0, 180.0))
    sat = float(np.clip(sat, 0.0, 2.0))
    lum = float(np.clip(lum, -1.0, 1.0))
    ranges = ((0.0, r_hue, r_sat), (60.0, y_hue, y_sat), (120.0, g_hue, g_sat),
              (180.0, c_hue, c_sat), (240.0, b_hue, b_sat),
              (300.0, m_hue, m_sat))
    neutral = (abs(hue) < 1e-9 and abs(sat - 1.0) < 1e-9 and abs(lum) < 1e-9
               and all(abs(dh) < 1e-9 and abs(ds - 1.0) < 1e-9
                       for _c, dh, ds in ranges))
    if neutral:
        return _out(a)
    if a.ndim == 2:
        return _out(a + lum) if lum else _out(a)
    h, s, v = _rgb_to_hsv(a)
    weights = []
    for center, dh, ds in ranges:
        if abs(dh) < 1e-9 and abs(ds - 1.0) < 1e-9:
            continue
        deg = h * 360.0
        d = np.abs(deg - center)
        d = np.minimum(d, 360.0 - d)
        w = np.clip(1.0 - d / 75.0, 0.0, 1.0)
        weights.append((w * w * (3.0 - 2.0 * w), float(dh), float(ds)))
    if hue:
        h = (h + hue / 360.0) % 1.0
    if weights:
        wsum = np.maximum(sum(w for w, _dh, _ds in weights), 1e-6)
        shift = sum(w * dh for w, dh, _ds in weights)
        h = (h + (shift / wsum) / 360.0) % 1.0
        factor = 1.0 + sum(w * (ds - 1.0) for w, _dh, ds in weights) / wsum
        s = np.clip(s * factor, 0.0, 1.0)
    s = np.clip(s * sat, 0.0, 1.0)
    v = np.clip(v + lum, 0.0, 1.0)
    return _out(_hsv_to_rgb(h, s, v))


def local_contrast(img, radius: float = 32.0, gain: float = 0.0):
    """CONTRASTE LOCAL (clarté ; esprit de l'égalisation locale de
    luminance de PixInsight, implémentation maison) : renforce le contraste
    à GRANDE échelle — out = img + gain·(img − flou_gaussien(img, σ =
    rayon)), rayon 4 à 200 px. Gain 0 : identité exacte."""
    a = _as_image(img)
    g = float(np.clip(gain, 0.0, 2.0))
    if g <= 0.0:
        return _out(a)
    sigma = float(np.clip(radius, 4.0, 200.0))
    if a.ndim == 3:
        blur = ndi.gaussian_filter(a, sigma=(sigma, sigma, 0))
    else:
        blur = ndi.gaussian_filter(a, sigma)
    return _out(a + g * (a - blur))


def split_tone(img, sh_r: float = 0.5, sh_g: float = 0.5, sh_b: float = 0.5,
               hl_r: float = 0.5, hl_g: float = 0.5, hl_b: float = 0.5,
               balance: float = 0.0, force: float = 0.0):
    """TONALITÉ DIVISÉE (esprit « split toning », implémentation maison) :
    teinte les ombres avec la couleur ombres (R/V/B 0–1) et les hautes
    lumières avec la couleur hautes lumières, pondérées par une luminance
    lissée (fondu doux) décalée par `balance` (−1 vers les ombres, +1 vers
    les hautes lumières). Force 0 : identité. Mono : les couleurs sont
    appliquées en luminance (virage monochrome)."""
    a = _as_image(img).copy()
    f = float(np.clip(force, 0.0, 1.0))
    if f <= 0.0:
        return _out(a)
    bal = float(np.clip(balance, -1.0, 1.0))
    col_sh = np.clip(np.array([sh_r, sh_g, sh_b], np.float32), 0.0, 1.0)
    col_hl = np.clip(np.array([hl_r, hl_g, hl_b], np.float32), 0.0, 1.0)
    if a.ndim == 2:
        c_sh = float(0.2126 * col_sh[0] + 0.7152 * col_sh[1]
                     + 0.0722 * col_sh[2])
        c_hl = float(0.2126 * col_hl[0] + 0.7152 * col_hl[1]
                     + 0.0722 * col_hl[2])
        t = np.clip(ndi.gaussian_filter(a, 8.0) + bal, 0.0, 1.0)
        a_sh, a_hl = f * (1.0 - t), f * t
        return _out(a * (1.0 - a_sh - a_hl) + c_sh * a_sh + c_hl * a_hl)
    lum = np.asarray(luminance(a), np.float32)
    t = np.clip(ndi.gaussian_filter(lum, 8.0) + bal, 0.0, 1.0)
    a_sh, a_hl = (f * (1.0 - t))[..., None], (f * t)[..., None]
    return _out(a * (1.0 - a_sh - a_hl) + col_sh * a_sh + col_hl * a_hl)


def gradient_map(img, r0: float = 0.0, g0: float = 0.0, b0: float = 0.0,
                 r1: float = 0.5, g1: float = 0.5, b1: float = 0.5,
                 r2: float = 1.0, g2: float = 1.0, b2: float = 1.0,
                 intensity: float = 0.0):
    """MAPPAGE DE DÉGRADÉ (principe « gradient map », implémentation
    maison) : la luminance L choisit, sur la rampe à trois arrêts (ombres /
    milieux / hauts), la couleur cible ; le détail de l'image est conservé
    en réajoutant l'écart (img − L). Intensité mélange avec l'original ;
    0 : identité. Mono : la rampe est échantillonnée en niveaux de gris."""
    a = _as_image(img)
    i = float(np.clip(intensity, 0.0, 1.0))
    if i <= 0.0:
        return _out(a)
    lum = np.clip(luminance(a), 0.0, 1.0)
    xs = (0.0, 0.5, 1.0)
    if a.ndim == 3:
        stops = ((r0, g0, b0), (r1, g1, b1), (r2, g2, b2))
        mapped = np.stack(
            [np.interp(lum, xs, [s[c] for s in stops]) for c in range(3)],
            axis=-1)
        out = mapped + (a - lum[..., None])
    else:
        gray = [0.2126 * r0 + 0.7152 * g0 + 0.0722 * b0,
                0.2126 * r1 + 0.7152 * g1 + 0.0722 * b1,
                0.2126 * r2 + 0.7152 * g2 + 0.0722 * b2]
        out = np.interp(lum, xs, gray)
    return _out(a * (1.0 - i) + out * i)


def _ghs(img, D=0.0, b=0.0, SP=0.0, LP=0.0, HP=1.0, method="Canaux"):
    from .stretch import ghs
    return _out(ghs(_as_image(img), D, b, SP, LP, HP,
                    "Luminance" if str(method).startswith("Lum") else "Canaux"))


def _arcsinh(img, stretch=0.0, black=0.0):
    from .stretch import arcsinh_stretch
    return _out(arcsinh_stretch(_as_image(img), stretch, black))


def _star_reduction(img, amount=0.0, iterations=1):
    from .masks import reduce_stars
    return _out(reduce_stars(_as_image(img), amount, int(iterations)))


def _chroma_denoise(img, strength=0.0, scales=4):
    from .masks import chroma_denoise
    return _out(chroma_denoise(_as_image(img), strength, int(scales)))


_FUNCTIONS = {
    "saturation": (saturation, ("amount",)),
    "white_balance": (white_balance, ("wr", "wg", "wb")),
    "background_neutral": (neutralize_background, ("percentile", "strength")),
    "scnr_green": (scnr_green, ("strength",)),
    "autostretch": (autostretch, ("shadows_clip", "target_bg")),
    "remove_gradient": (remove_gradient, ("kernel", "method")),
    "denoise_median": (denoise_median, ("radius", "iterations")),
    "unsharp_mask": (unsharp_mask, ("radius", "gain")),
    "levels": (levels, ("black", "gamma", "white")),
    "shadows_highlights": (shadows_highlights,
                           ("shadows", "highlights", "radius")),
    "vibrance": (vibrance, ("amount",)),
    "hsl": (hsl, ("hue", "sat", "lum", "r_hue", "r_sat", "y_hue", "y_sat",
                  "g_hue", "g_sat", "c_hue", "c_sat", "b_hue", "b_sat",
                  "m_hue", "m_sat")),
    "local_contrast": (local_contrast, ("radius", "gain")),
    "split_tone": (split_tone, ("sh_r", "sh_g", "sh_b", "hl_r", "hl_g",
                                "hl_b", "balance", "force")),
    "gradient_map": (gradient_map, ("r0", "g0", "b0", "r1", "g1", "b1",
                                    "r2", "g2", "b2", "intensity")),
    "ghs": (_ghs, ("D", "b", "SP", "LP", "HP", "method")),
    "arcsinh": (_arcsinh, ("stretch", "black")),
    "star_reduction": (_star_reduction, ("amount", "iterations")),
    "chroma_denoise": (_chroma_denoise, ("strength", "scales")),
}


def apply_filter_masked(img, filter_id, params: dict | None = None,
                        mask: dict | None = None) -> np.ndarray:
    """Filtre appliqué à travers un masque {kind, low, high, fuzz, blur}
    (core.masks.MASK_TYPES), calculé sur l'image d'entrée."""
    out = apply_filter(img, filter_id, params)
    if not mask or mask.get("kind", "Aucun") == "Aucun" or filter_id in (None, "", "none"):
        return out
    from .masks import apply_masked, make_mask
    m = make_mask(img, mask.get("kind"), mask.get("low", 0.25), mask.get("high", 1.0),
                  mask.get("fuzz", 0.1), mask.get("blur", 2.0))
    return _out(apply_masked(np.asarray(img, np.float32), out, m))


def apply_filter(img, filter_id, params: dict | None = None) -> np.ndarray:
    """Applique le filtre `filter_id` avec `params` (clés inconnues ignorées,
    défauts du noyau sinon). « none »/inconnu : image inchangée (copie)."""
    a = np.asarray(img, np.float32)
    if filter_id in (None, "", "none"):
        return a
    if filter_id not in _FUNCTIONS:
        raise ValueError(f"Filtre de finition inconnu : {filter_id}")
    fn, keys = _FUNCTIONS[filter_id]
    kw = {k: (params or {}).get(k, _DEFAULTS[filter_id][k]) for k in keys}
    return fn(a, **kw)
