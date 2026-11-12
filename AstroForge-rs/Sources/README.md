# AstroForge 2.0.0 — moteur et ligne de commande en Rust

Portage en Rust du moteur de traitement d'AstroForge (ciel profond et
planétaire), de son interface en ligne de commande, et interface graphique
egui (`astroforge-gui`) dédiée au planétaire, avec un onglet **Planétaire IA**
qui traite une vidéo de bout en bout sans aucun réglage, et un onglet
**Time-lapse** qui transforme une série de vidéos (éclipse, rotation d'une
planète…) en animation au traitement, au cadrage et au rendu uniformes.

## Compilation

### Windows : paquet précompilé (recommandé)

Le paquet `AstroForge-2.0.0-windows-x64.zip` contient `astroforge.exe` déjà
compilé : décompressez-le puis double-cliquez sur `install.bat`. Ni Rust ni
Visual Studio ne sont nécessaires. L'exécutable ne dépend que des
bibliothèques système de Windows 10/11.

### Installation automatique

Les scripts vérifient Rust (≥ 1.88) et proposent de l'installer via rustup
s'il manque, compilent en mode optimisé, installent l'exécutable, l'ajoutent
au PATH puis vérifient qu'il démarre.

```bash
# Linux / macOS
./install.sh                       # dans ~/.local/bin
sudo ./install.sh --system         # dans /usr/local/bin
./install.sh --portable ~/AstroForge   # version portable
./install.sh --onnx                # avec l'IA ONNX locale
./install.sh --no-gui              # sans l'interface graphique
./install.sh --uninstall           # désinstallation (--purge : données aussi)
```

```bat
rem Windows (double-clic possible)
install.bat                        & rem dans %LOCALAPPDATA%\Programs\AstroForge
install.bat -Portable D:\AstroForge
install.bat -Onnx
install.bat -NoGui                 & rem sans l'interface graphique
install.bat -Uninstall
```

Sous Linux, un compilateur C est nécessaire pour l'édition de liens
(`build-essential`, `gcc` ou `base-devel`) ; sous macOS, `xcode-select
--install` ; sous Windows, les « Build Tools for Visual Studio » (charge de
travail C++), que le script propose d'installer (winget ou Visual Studio
Installer) avant de compiler ; `install.bat -FromSource` force la compilation
même à côté d'un `astroforge.exe` précompilé.

### Compilation manuelle

```bash
# Rust ≥ 1.88 (https://rustup.rs)
cargo build --release              # binaire : target/release/astroforge
cargo test --release               # tests unitaires et d'intégration
```

Fonctionnalités Cargo :

| Fonctionnalité | Par défaut | Rôle |
|---|---|---|
| `raw`  | oui | lecture des RAW d'appareils photo (CR2, NEF, ARW, DNG, RAF, ORF, RW2…) via `rawloader` |
| `net`  | oui | vérification des mises à jour, catalogue de modèles IA, requêtes Gaia |
| `onnx` | non | inférence ONNX locale en pur Rust (`tract`) |
| `gui`  | oui | interface graphique `astroforge-gui` (egui) |

```bash
cargo build --release --no-default-features          # sans RAW ni réseau
cargo build --release --features onnx                # avec l'IA ONNX locale
```

Aucune dépendance système : pas d'OpenCV, de Python ni de bibliothèque C.
Les outils externes restent facultatifs, comme dans la version Python :
`solve-field` (astrometry.net), `ffmpeg` (export MP4/AVI), GraXpert,
StarNet2.

## Interface graphique : onglets Planétaire et Planétaire IA

```bash
astroforge-gui                      # ou menu Démarrer / menu des applications
astroforge-gui jupiter.ser          # ouvre directement la vidéo
astroforge-gui --auto jupiter.ser   # et lance aussitôt le traitement automatique
astroforge-gui --timelapse eclipse/ --auto   # série de vidéos -> animation
```

* **Planétaire IA (automatique)** : ouvrez une vidéo SER (ou une série
  d'images, ou déposez le fichier sur la fenêtre) puis « Lancer le traitement
  automatique ». Le moteur décide de tout et explique chaque décision ; la vue
  « Avant / après » compare l'empilement brut et le résultat par un rideau.
  Une retouche facultative (force des détails, débruitage) rejoue la
  restauration instantanément et signale si l'on dépasse les limites « sans
  perte ». Export TIFF 16 bits, PNG 16 bits, FITS 32 bits, JPEG, et rapport
  JSON complet ; « Ouvrir ces réglages dans l'onglet Planétaire » reprend le
  plan de l'IA pour l'ajuster à la main.
* **Planétaire** : les 6 étapes réglées à la main (qualité, % gardé,
  alignement global / grille / points locaux AS!, empilement), puis gains
  d'ondelettes, débruitage, alignement RVB et niveaux.

* **Time-lapse (multi-vidéos)** : ajoutez les vidéos SER de la série (ou un
  dossier ; elles sont triées par heure de prise de vue), puis « Traiter la
  série ». L'animation se lit aussitôt dans l'onglet ; luminosité, horodatage
  et légende se règlent sans retraitement ; export MP4 / AVI (ffmpeg), GIF ou
  images numérotées (PNG 16 bits, TIFF, FITS, JPEG).

### Time-lapse : même traitement, même cadrage, même rendu

1. **Plan commun** : la vidéo la plus propre de la série (ou celle que vous
   désignez) passe par le moteur automatique complet ; ses choix (part des
   frames gardées, alignement, empilement, gains d'ondelettes, débruitage
   relatif au bruit, gains de couleur) sont ensuite **verrouillés** et
   appliqués à l'identique à toutes les autres vidéos.
2. **Cadrage commun** : en mode disque, le limbe est retrouvé sur chaque image
   par un ajustement de cercle robuste qui ne retient que les bords « clairs
   à l'intérieur » — un croissant d'éclipse ou une phase ne le trompent pas —
   avec un rayon commun à la série ; chaque image est placée au centre exact
   (au dixième de pixel) d'un canevas de taille fixe. En mode surface (Lune
   ou Soleil en gros plan), recalage en chaîne par corrélation et cadre
   commun. Le mode automatique choisit.
3. **Rotation de champ** (option, monture altazimutale) : rotations mesurées
   entre images distantes sur les seuls détails internes éclairés dans les
   deux images (le bord mobile de la Lune et le limbe sont exclus), confirmées
   aller-retour, puis l'angle est ajusté comme une fonction régulière de
   l'heure — les images sans détail (phase profonde) sont couvertes sans
   dérive.
4. **Luminosité harmonisée** : la luminosité relative de chaque image est
   mesurée pixel à pixel par rapport aux images voisines, sur les zones
   éclairées dans les deux — insensible à la forme du croissant, sensible aux
   voiles et changements d'exposition. Trois rendus : *dé-scintillement*
   (voiles passagers gommés, évolution réelle gardée — recommandé),
   *réelle* (mêmes niveaux pour tout), *normalisée*. Noir, blanc et genou doux
   communs à toute la série.
5. **Animation** : images de transition en fondu pour la fluidité ;
   l'horodatage est incrusté après interpolation, avec une heure interpolée.

```bash
astroforge -v timelapse eclipse/ -o eclipse.mp4 --fps 12 --interp 3 \
           --frames-dir images/ --caption "Éclipse du 12 août 2026"
astroforge timelapse jup_*.ser --field-rotation --brightness reelle -o jupiter.gif
```

Série de test : `cargo run --release --example synth_eclipse -- eclipse/ 16`
écrit 16 vidéos SER d'une éclipse partielle de Soleil (dérive du suivi,
voiles, seeing variable, horodatages ; `SYN_ROT=0.4` ajoute une rotation de
champ).

### Comment décide le moteur IA local

Tout tourne sur la machine (aucun service en ligne). Le moteur combine des
mesures, une recherche guidée par un critique de qualité et un apprentissage
par l'expérience :

1. **Analyse** : détection du disque (ou mode « surface » pour la Lune et le
   Soleil), recentrage de chaque frame, rejet des frames inutilisables (voile
   nuageux, planète coupée par le bord, saturation).
2. **Tri** : quatre mesures de netteté (contraste HF façon AutoStakkert!,
   laplacien, gradient, bande passante) ; leur fiabilité est testée sur la
   vidéo elle-même en y ajoutant un bruit de même niveau, et le classement
   final fusionne les mesures pondérées par cette fiabilité.
3. **Critique sans référence** : chaque essai est empilé en deux moitiés
   indépendantes ; leur demi-différence est une mesure exacte du bruit (et des
   défauts d'alignement), leur moyenne le signal. Le score est la quantité
   d'information par échelle d'ondelettes, Σ log2(1 + S²/N).
4. **Sélection** : nombre de frames qui maximise ce score (une dizaine
   d'empilements d'essai calculés en un seul passage).
5. **Alignement et empilement** : mesure de la turbulence (distorsion locale)
   puis essai des méthodes candidates — globale, grille multi-points,
   multipoint local AS! (plusieurs tailles de boîtes en mode Maximal) — et des
   empilements (moyenne, moyenne pondérée, sigma-clipping, winsorisé).
6. **Restauration** : PSF mesurée sur le limbe ; gains d'ondelettes calculés
   par un filtre de Wiener par couche (1/FTM × RSB²/(RSB²+1)) ; débruitage par
   seuillage doux à hauteur du bruit mesuré ; protection du limbe contre les
   halos. Force et débruitage sont choisis **sous contraintes** : bruit visible
   au plus 1,5× celui de l'empilement (≤ 0,75 % du disque), halo au limbe
   ≤ 5 %, aucun écrêtage (genou doux). C'est ce qui garantit l'absence de
   perte : sur vidéos synthétiques à vérité connue, le résultat final n'est
   jamais moins fidèle que l'empilement brut.
7. **Finition** : alignement des canaux RVB (dispersion atmosphérique),
   équilibre partiel des couleurs, niveaux sans écrêtage, calcul en 32 bits.
8. **Apprentissage** : chaque traitement est mémorisé
   (`ai/planetary_memory.json` dans le dossier des données) ; une vidéo
   ressemblante réutilise ces choix comme point de départ (plus proches
   voisins). Un modèle ONNX de débruitage installé dans `models/ai`
   (compilation `--features onnx`) est essayé et n'est retenu que s'il réduit
   le bruit sans perte de détail mesurable.

Efforts : *Rapide* (règles expertes et mémoire, un essai d'alignement),
*Équilibré* (recommandé), *Maximal* (tous les essais).

Vidéo de test : `cargo run --release --example synth_planet -- test.ser 400`
produit une vidéo SER synthétique de Jupiter (seeing variable, turbulence,
nuages, dispersion) ; variables `SYN_R`, `SYN_GAIN`, `SYN_TURB`,
`SYN_SEEING`, `SYN_MONO`, `SYN_SURFACE` pour d'autres cas.

## Ligne de commande

| Commande | Équivalent Python | Rôle |
|---|---|---|
| `sort` | `sort` | tri automatique d'une session (lumières, darks, flats, offsets) |
| `analyze` | `analyze` | FWHM, excentricité, étoiles, fond, poids, rejets |
| `stack` | `stack` | calibration, alignement, empilement, drizzle |
| `multinight` | `multinight` | empilement de plusieurs nuits normalisées |
| `solve` | `solve` | astrométrie (solve-field ou solveur interne par triangles + SIP) |
| `annotate` | `annotate` | annotation Messier / NGC / IC (OpenNGC) |
| `compose` | `compose` | bande étroite (HOO, SHO…) et LRGB |
| `batch` | `batch` | séquence enregistrée appliquée à un dossier (images ou SER) |
| `live` | `live` | empilement en direct d'un dossier surveillé |
| `pipeline` | `script` | processus automatique enregistré (JSON) |
| `planetary` | *(nouveau)* | empilement planétaire d'une vidéo SER |
| `planetary-auto` | *(nouveau)* | planétaire entièrement automatique (moteur IA local) |
| `timelapse` | *(nouveau)* | série de vidéos : traitement, cadrage et rendu uniformes, animation |
| `mosaic` | *(nouveau)* | mosaïque par WCS, ou par appariement d'étoiles |
| `info` | `info` | version, cœurs, langue, dossier des données |

Exemples :

```bash
astroforge sort ~/astro/2026-10-15
astroforge stack --session ~/astro/2026-10-15 --weights Mixte \
                 --normalization Locale --drizzle 2 -o m31.fits
astroforge solve m31.fits --ra 10.68 --dec 41.27 --write-wcs
astroforge annotate m31.fits -o m31_annote.png
astroforge compose --ha ha.fits --oiii o3.fits --sii s2.fits \
                   --palette "SHO (Hubble)" --align -o sho.tif
astroforge planetary jupiter.ser --keep 50 \
                   --align-mode "Multi-points local (AS!)" -o jupiter.tif
astroforge -v planetary-auto jupiter.ser -o jupiter.tif --report rapport.json
astroforge planetary-auto lune.ser --effort Maximal --stack-out brut.tif -o lune.tif
astroforge mosaic p1.fits p2.fits p3.fits p4.fits -o mosaique.fits
astroforge -v batch finitions.json entree/ sortie/
```

`astroforge <commande> --help` détaille toutes les options. Les paramètres
de déclinaison négatifs s'écrivent directement (`--dec -5°24'`).

Les fichiers de paramètres (séquences, processus, profils, bibliothèque de
masters, réglages) gardent le format JSON de la version Python et le même
dossier de données : les deux versions peuvent cohabiter.

## Organisation

```
src/
  image.rs, ndimage.rs, geom.rs, fft.rs, stats.rs   primitives (équivalents scipy/OpenCV)
  io/        FITS, XISF, TIFF/PNG/JPEG, SER, RAW, dématriçage
  stars.rs   détection DAOStarFinder (portage exact de photutils)
  registration.rs, starmatch.rs, drizzle.rs          alignement
  calibration.rs, masters.rs, defects.rs, session.rs prétraitement
  stacking.rs, subframes.rs, normalization.rs, multisession.rs, livestack.rs
  wcs.rs, platesolve.rs, astrometry.rs, annotate.rs, mosaic.rs
  stretch.rs, finishing.rs, masks.rs, wavelets.rs, colorcal.rs, gradient.rs,
  restoration.rs, effects.rs                         finitions
  planetary.rs, derotation.rs, solar.rs, animation.rs, comet.rs, lightcurve.rs
  planetary_auto.rs  planétaire entièrement automatique (moteur de décision IA)
  timelapse.rs       time-lapse multi-vidéos (plan commun, cadrage, rendu, animation)
  sequence.rs, pipeline.rs, profiles.rs, equipment.rs, ai.rs, updates.rs
  main.rs    ligne de commande
  bin/astroforge-gui/  interface graphique (Planétaire, Planétaire IA, Time-lapse)
tests/       tests portés de la suite Python
data/        catalogue OpenNGC allégé (licence CC-BY-SA, voir OPENNGC_LICENSE.txt)
```

Les images sont des `f32` entrelacées (H×W×C, C = 1 ou 3, valeurs dans
[0, 1]) ; tous les traitements lourds sont parallélisés avec `rayon`.

## Fidélité au programme Python

Chaque commande a été comparée à la version Python sur les mêmes données :

* primitives (flou gaussien, médian, laplacien, rééchantillonnage,
  transformations affines, étirements) : écarts ≤ 1e-6 ;
* empilement, rejets, tri de session, analyse, annotation, solveur interne,
  WCS de mosaïque, chaîne SER planétaire, inférence ONNX : résultats
  identiques ou à l'arrondi près ;
* alignement des lumières : estimation par étoiles propre au portage
  (au lieu d'astroalign) ; images empilées corrélées à 0,98–0,99 avec celles
  de Python, précision sous 0,1 px sur images calibrées.

Vitesse (2 cœurs) : empilement 3,6 s → 1,3 s ; planétaire 1,9 s → 0,46 s ;
traitement par lots 0,78 s → 0,08 s. Le gain croît avec le nombre de cœurs.

## Différences et limites

* interface graphique limitée au planétaire (onglets Planétaire et
  Planétaire IA) ; le ciel profond reste en ligne de commande ;
* FITS compressés `.fz` et RAW Canon CR3 non lus ;
* export MP4/AVI via `ffmpeg` (GIF natif) ;
* `script` (scripts Python) devient `pipeline` (processus JSON) ;
* transformée de Hough, retouche (inpainting) et débruitage NLM : réimplémentés,
  résultats proches mais non identiques à OpenCV ;
* correction : le retrait de gradient par RBF (TPS) s'évalue désormais en
  coordonnées pixel, comme prévu ; la version Python mélangeait les échelles ;
* mosaïque sans WCS : appariement d'étoiles au lieu de SIFT.

## Licence

MIT (voir `LICENSE`). Catalogue OpenNGC : CC-BY-SA 4.0.
