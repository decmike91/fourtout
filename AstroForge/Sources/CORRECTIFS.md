# AstroForge 1.9.9 — correctifs de l'audit du 14/10/2026

Tous les bugs relevés par l'audit sont corrigés. Chaque correctif est couvert
par un test de non-régression à vérité terrain (`tests/smoke.py`, fonction
`test_regressions_audit` et tests mis à jour) : **429 tests sur 429 réussis**.

## Bugs critiques

| # | Problème | Correctif | Fichiers |
|---|---|---|---|
| C1 | L'alignement par phase doublait le décalage au lieu de le corriger | `apply_shift` translate désormais de (+dx, +dy) ; convention documentée ; corrigé partout où elle sert (alignement ciel profond et planétaire, canaux couleur, replis étoiles/similitude, drizzle) | `core/registration.py`, `core/drizzle.py` |
| C2 | Matrices de Bayer RGGB et BGGR : rouge et bleu inversés | Correspondance OpenCV corrigée (RGGB → `BayerBG`, BGGR → `BayerRG`), vérifiée sur OpenCV 4 et 5 | `core/image_io.py` |
| C3 | FITS 16 bits de caméra (BZERO = 32768) illisibles | Lecture `memmap=False`, HDU image recherchée (FITS compressé `.fz` accepté) | `core/image_io.py` |
| C4 | FITS en ADU jamais normalisés (export blanc) | Normalisation selon le type (uint16 ÷ 65535, int ÷ max, flottants ADU ÷ 65535) ; même règle pour TIFF flottant, XISF et RAW | `core/image_io.py`, `core/xisf.py` |
| C5 | Bias non soustrait quand un dark est fourni | `calibrate` retire bias + k·dark thermique ; message clair si les tailles diffèrent | `core/calibration.py` |
| C6 | Calibration OSC impossible (lumières dématricées, masters CFA) | Nouvelle chaîne `build_masters` + `calibrate_paths` : lecture CFA → calibration → cosmétique et pixels défectueux sur les pixels de même couleur → dématriçage ; motif lu dans le RAW ou le mot-clé FITS `BAYERPAT` ; chaque dossier de calibration n'est lu qu'une fois | `core/calibration.py`, `core/defects.py`, `core/image_io.py` (`load_cfa`), `gui/page_sky.py`, `processing/pipeline.py` |

## Bugs majeurs

| # | Problème | Correctif |
|---|---|---|
| M1 | Sigma-clipping inopérant sous 11 images | Pré-rejet robuste médiane ± max(4, σ+1)·MAD puis itérations moyenne/écart-type ; winsorisé à la Huber (σ initial MAD, facteur 1,134) |
| M2 | Drizzle sans gain de résolution | Drizzle sur les images calibrées AVANT alignement, transformation par image (similitude via astroalign, sinon phase + rotation), vrai noyau carré pixfrac × échelle, rejet des pixels aberrants contre l'empilement à rejet, trous bouchés |
| M3 | Sous-pixel ignoré autour de 0 | Voisins pris modulo la taille (corrélation circulaire) ; images de tailles différentes acceptées |
| M4 | Extraction de gradient : plantage en couleur | Un modèle par canal (pollution lumineuse colorée), piédestal neutre : le bruit du fond n'est plus écrêté à 0 |
| M5 | Richardson-Lucy : plantage en couleur | Déconvolution canal par canal |
| M6 | Quantification 8 bits (NL-means, bilatéral, satellites, HDR) | NL-means 16 bits (norme L1), force relative au bruit mesuré ; bilatéral et inpainting en float32 ; seuls les pixels de la trace sont remplacés ; HDR Mertens en float32 |
| M7 | `remove_satellite_trails(mask=…)` plantait | Paramètre `mask` explicite |
| M8 | Palettes Inferno, Viridis… = palette Solaire | API `matplotlib.colormaps` ; une palette inconnue lève une erreur au lieu d'être remplacée en silence |
| M9 | XISF incompatible PixInsight | Réécriture selon la spécification 1.0 : en-tête 16 octets, `attachment:`, `inline`/`embedded`, zlib et zlib+sh (LZ4/Zstandard si modules installés), stockage planaire ou entrelacé, bornes ; écriture conforme avec métadonnées ; anciens fichiers AstroForge toujours lisibles |
| M10 | `stack_paths("Médiane")` plantait, aucune économie mémoire | Pile `np.memmap` temporaire, chaque image lue une fois ; tous les empilements travaillent par tranches de lignes en parallèle (30 × 1500×1500×3 en sigma-clipping : 9 s au lieu de 13 s, mémoire bornée) |
| M11 | Étirement automatique brûlant les hautes lumières | Fonction de transfert des tons moyens (MTF) type STF : fond amené à 25 %, aucun écrêtage ; paramètre « Fond cible » remplace « Point blanc » (l'ancien paramètre est ignoré dans les séquences existantes) ; aperçus et détection de traces utilisent le même étirement |
| M12 | PCC : fond mesuré sur le seul vert | Fond soustrait par canal, fenêtres locales autour des étoiles (plus rapide), étoiles saturées (sommet plat) et couleurs manquantes ignorées |
| M13 | Catalogue Gaia tronqué à 2 000 étoiles, pandas requis | Requête asynchrone sans limite, écriture CSV par astropy, colonnes normalisées |
| M14 | Photométrie différentielle : NaN sans comparaison, cible non contrôlée | Erreur explicite si aucune étoile de comparaison ; cible refusée au-delà de 3 pixels (min. 10″) |
| M15 | Batch SER limité à 150 images, erreurs invisibles | « 0 = toutes » par défaut ; résultat avec liste d'échecs, fichier `rapport_batch.txt` et message dans l'interface ; erreur si aucun fichier n'a réussi |
| M16 | Vidéo SER chargée entièrement en RAM, interface figée | `load_ser(lazy=True)` : projection mémoire (`SerFrames`), conversion à la demande ; AVI/MP4 gardés en 8 bits. Lecture SER conforme à la norme v3 : ColorID 8-11 = capteur Bayer (dématricé), 100 RGB, 101 BGR, profondeurs 9-15 bits ; écriture avec l'en-tête standard de 178 octets |
| M17 | Intégrations IA non opérantes | StarNet2 échangé en TIFF 16 bits avec gabarit d'arguments configurable (`"args"` dans `ai_tools.json`) ; la « force » s'applique aussi aux modèles ONNX ; Cosmic Clarity documenté comme usage manuel ; étape de débruitage par défaut du pipeline en algorithme classique (plus d'échec sans modèle ONNX) |

## Autres corrections

- Saisie de coordonnées : `+19°48'30"`, `19d48m30s`, `10h25m03s`, `10:25:03`, virgule décimale (un seul parseur pour toute l'application).
- Film planétaire « stabilisé » : les images sont réellement recalées avant le recadrage commun (elles n'étaient que recadrées) ; qualité MJPG appliquée.
- RAW : normalisation par le niveau blanc du capteur (images auparavant blanches), motif de Bayer lu sur la zone visible, motif forcé respecté.
- Chemins accentués sous Windows (« Données/été.tif ») pour TIFF/PNG/JPEG.
- Réglages, profils et modèles IA dans le dossier utilisateur quand le dossier du programme n'est pas modifiable (toujours pour l'exécutable) ; `ASTROFORGE_HOME` pour forcer un dossier.
- Solveur interne : affinage Procrustes avec facteur d'échelle (Umeyama), garde-fou d'échelle.
- Compatibilité photutils ≥ 3 (colonnes `x_centroid`/`y_centroid`).
- Progression détaillée de chaque étape du pipeline ; annulation propagée.
- Plantage à la fermeture (destruction des objets Qt) corrigé dans l'application et les tests.
- Interface : boutons jamais écrasés quand la fenêtre est basse ; colonne Finition sans barre horizontale à 1024×700.
- Tests : dossier temporaire portable (plus de `/tmp/opencode`), test du drizzle contre une vérité haute résolution, test AutoStakkert! indépendant de la frame de référence.
- Nettoyage : 51 imports/variables inutilisés, code mort de `align_poly`, commentaire parasite.
- Documentation (README, manuel, aide F1) alignée sur le comportement réel.

## Vérification

```bash
python tests/smoke.py        # 429/429
python main.py --selftest
```
Un essai de bout en bout de l'interface (session OSC synthétique en FITS
16 bits RGGB, darks, flats, pixel chaud, trace de satellite ; calibration →
alignement → drizzle ×2 → gradient → débruitage) produit une image 320×320
aux couleurs d'étoiles justes, fond neutre, trace rejetée, résidu
d'alignement 0,07 px.
