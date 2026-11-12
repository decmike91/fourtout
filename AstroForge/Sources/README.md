# AstroForge — Traitement d'images astronomiques

Logiciel Python/Qt (fichiers `.ui` Qt Designer) en **2 modes + une partie commune** :

- **Ciel profond** : pré-traitement, masters de calibration, alignement (phase ou
  **par étoiles** via astroalign), **drizzle ×2/×3**, **7 algorithmes d'empilement**
  (moyenne, médiane, sigma-clip, winsorisé, min/max, **linear fit**, **MAD clipping**),
  extraction de fond/gradient, débruitage, déconvolution, **calibration couleur et
  neutralisation du fond**, **plate solving local + astrométrie + photométrie d'ouverture**,
  **IA locale** (denoise, fond, netteté, super-résolution, traces de satellites),
  composition **HDR**, **processus de traitement automatique** enregistrable,
  profils de paramètres par algorithme (défaut « standard »), matrices de Bayer,
  **XISF lu/écrit nativement**, correction des **pixels/colonnes/lignes
  défectueux** (carte fournie FITS/PNG/TXT ou détection auto sur master
  dark/flat), choix de l'**algorithme d'alignement** (phase / étoiles), empilement **basé AutoStakkert!** (qualité par frame, sélection des meilleures, alignement **multipoint** champ de décalage lissé), débruitage **ondelettes Registax** réglable, séquences **chargeables/modifiables** avant rejeu batch.
- **Planétaire** : source vidéo SER/AVI/MP4 ou images APN, **classement qualité**
  (style AutoStacker/Registax), **déroration du champ**, alignement fin, empilement,
  **ondelettes à trous (sliders type Registax6)**, IA locale, alignement des canaux
  couleur, **film stabilisé** (recadrage commun anti-« danse », MP4/AVI/GIF, FPS,
  qualité), **fausses couleurs** pour le Soleil N&B, correction d'**aberration
  chromatique** radiale, choix de l'**algorithme d'alignement** (phase / limbe /
  étoiles).
- **Commun (finition type GIMP)** : luminosité/contraste/saturation/gamma,
  balance des blancs, recadrage/rotations, masque flou, **recalcul pour augmenter
  la résolution** (Lanczos ×2/×3 ou IA ONNX), export FITS/TIFF 16/PNG 8-16/JPEG,
  **séquences batch enregistrables** (appliquées à tout un dossier d'images).

## Nouveautés 2.0

- **Session triée automatiquement** (en-têtes FITS/XISF ou noms de fichiers) et
  **bibliothèque de masters** avec mise à l'échelle automatique des darks.
- **Analyse des brutes** (FWHM, excentricité, étoiles, fond, SNR), rejet des
  images aberrantes et **empilement pondéré** ; **normalisation locale** et
  **empilement multi-nuits**.
- Finition : **GHS** et **arcsinh** avec histogramme, **masques**, réduction
  d'étoiles, séparation/recombinaison des étoiles, débruitage de chrominance.
- **Bande étroite** : Ha/OIII depuis un capteur couleur (duo-bande), palettes
  HOO/SHO/HSO…, **LRGB**.
- Menu **Outils** : mosaïque (WCS ou étoiles), comète, **empilement en direct**,
  dérotation de séries et export **WinJUPOS** (horodatage SER), outils solaires,
  **annotation Messier/NGC/IC**, **courbe de lumière**, catalogue de **modèles IA**
  téléchargeables, comparaison avant/après, **scripts Python**.
- Planétaire : empilement **multipoint local** façon AutoStakkert!.
- Astrométrie : solveur par triangles avec **distorsions SIP**.
- Socle : aperçu avec zoom 1:1, valeur du pixel et statistiques ; export FITS/XISF
  avec **en-tête complet**, TIFF 32 bits flottant ; **ligne de commande**
  `astroforge` ; **interface en anglais** ; **GPU** facultatif (CuPy) ; estimation
  de rotation log-polaire ; tests pytest et intégration continue
  Windows/macOS/Linux ; `pyproject.toml` ; installeurs et vérification des mises à
  jour. Détails : [CHANGELOG.md](CHANGELOG.md).

## Séquences batch (esprit Registax)

Dans chaque mode vous pouvez **enregistrer la chaîne de traitement** et la rejouer :
- Partie commune : `＋ Ajouter l'étape courante` capture les réglages, puis
  `Appliquer au lot…` traite tout un dossier d'images (FITS/TIFF/PNG/XISF).
- Planétaire : `Enregistrer la séquence` sauvegarde tous les réglages (classement,
  dérotation, ondelettes, palette…) dans un JSON, puis `Batch SER…` empile toutes
  les vidéos d'un dossier avec la même chaîne (exemples dans `scripts/`).
- Ciel profond : le « processus automatique » (liste d'étapes) s'enregistre et se
  recharge au format JSON.
- API Python : `astroforge/core/sequence.py` (`save_sequence`, `run_batch_image`,
  `run_batch_ser_chain`).

## Astrométrie locale (plate solving, photométrie)

Deux solveurs **100 % locaux** :
1. **astrometry.net** si installé (`solve-field` + index files téléchargées une
   fois) — le plus robuste, utilisé automatiquement s'il est présent.
2. **Solveur interne** : détection d'étoiles (photutils), appariement de motifs
   contre un catalogue local CSV (colonnes `ra,dec,mag` en degrés), fit WCS TAN
   (astropy). Construisez votre catalogue une fois pour toutes :
   ```python
   from astroquery.gaia import Gaia
   r = Gaia.launch_job("SELECT ra, dec, phot_g_mean_mag AS mag FROM "
       "gaiadr3.gaia_source WHERE phot_g_mean_mag < 14 AND "
       "CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', 156.25, 19.75, 0.5))=1")
   r.get_results().to_pandas().to_csv("gaia_mon_champ.csv", index=False)
   ```
La photométrie d'ouverture (photutils) calibre le zero point sur les étoiles du
catalogue et exporte un CSV (x, y, flux, mag). La **photométrie différentielle** mesure la cible par rapport à un ensemble d'étoiles de comparaison du catalogue (zero point et extinction s'annulent) : dmag ± erreur. Le WCS résolu est inscrit dans
l'en-tête FITS à l'export.

## Page Configuration (profils d'équipement)

4e onglet : créez un profil par matériel (caméra + taille de pixel + température
capteur + matrice de Bayer, télescope + ouverture + focale, **réducteur de
focale** avec facteur, mode ciel profond ou planétaire, **pollution lumineuse**
Bortle 1-9). La focale effective, l'échantillonnage ("//px) et le champ de vue
sont recalculés en direct. Cases à cocher d'astrométrie : solve-field
(astrometry.net) s'il est installé, catalogue CSV par défaut. **« Générer le
catalogue Gaia pour ce champ »** télécharge une fois les étoiles couvrant le
champ de vue de votre profil (astroquery), et **« Déterminer la focale
automatiquement »** résout une image et en déduit la focale
(f = 206 265 × pixel / échelle). Les autres onglets utilisent ce profil
(catalogue, solveur). Le guide complet des scripts est dans
`GUIDE_SCRIPTS.md`, le comparatif avec Siril/Registax/AutoStakkert!/APP dans
`COMPARATIF.md`.

## Installation

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate     Linux/Mac : source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Optionnel : `pip install onnxruntime` (modèles IA ONNX), `pip install rawpy`
(RAW d'appareils photo : CR2/CR3/NEF/ARW/DNG…), `pip install astroquery`
(génération du catalogue Gaia), `pip install photutils astroalign`
(astrométrie, alignement par étoiles).

Installation comme paquet Python (commandes `astroforge` et `astroforge-gui`) :

```bash
pip install ".[astro,ia,raw]"      # options : astro, ia, raw, gpu, dev, all
astroforge-gui                     # interface
astroforge --help                  # ligne de commande
astroforge stack --session D:/M31 --weights Mixte --drizzle 2 -o M31.fits
```

Langue : menu **Langue / Language** (français, anglais ; au redémarrage) ou
variable `ASTROFORGE_LANG=en`. GPU : `pip install ".[gpu]"` (CuPy CUDA 12)
puis **Configuration › Accélération GPU** ou `ASTROFORGE_GPU=1`.

## Lancer et distribuer (exécutable)

```bash
./launch.sh          # Linux/Mac — venv auto + dépendances + lancement
launch.bat           # Windows — idem, double-clic possible
./build_exe.sh       # Linux/Mac — exécutable autonome : dist/AstroForge
build_exe.bat        # Windows — dist\AstroForge.exe
```
Les scripts `build_exe` utilisent PyInstaller et embarquent `ui/` ;
`main.py --selftest` vérifie une installation sans lancer l'interface.

Installeurs (dossier `packaging/`, produits automatiquement par
`.github/workflows/release.yml` à chaque étiquette `vX.Y.Z`) :

| Plate-forme | Commande | Résultat |
|---|---|---|
| toutes | `python packaging/build.py` | exécutable PyInstaller (`dist/`) |
| Windows (depuis Linux/WSL) | `packaging/windows/build_windows.sh [--portable]` | `AstroForge-<v>-windows-setup.exe` (NSIS, Python embarqué, aucune installation de Python requise) et version portable `.zip` ; signé Authenticode si un certificat est fourni |
| Windows (sur Windows) | `python packaging/build.py` puis `packaging/windows/build_installer.ps1` | variante PyInstaller + Inno Setup |
| macOS | `packaging/macos/build_dmg.sh` | `.dmg`, signé (Developer ID) et notarisé si les identifiants Apple sont fournis |
| Linux | `packaging/linux/build_appimage.sh` | AppImage (signature GPG facultative) |

### Installeur Windows

`AstroForge-<version>-windows-setup.exe` (Windows 10 1903 ou plus récent,
64 bits) installe pour tous les utilisateurs (Program Files) ou pour vous
seul (sans droits administrateur) : programme, Python embarqué et toutes les
dépendances (astrométrie, IA ONNX, RAW), menu Démarrer (interface, ligne de
commande, désinstallation), raccourci bureau et « Ouvrir avec » pour
.fits/.xisf facultatifs, désinstallation depuis Paramètres › Applications
(vos réglages dans `%APPDATA%\AstroForge` sont conservés sauf demande).
Installation silencieuse : `setup.exe /S /CurrentUser` (ou `/AllUsers`,
`/D=C:\Dossier`). Vérifier une installation :
`astroforge-cli.exe script "<dossier>\check_install.py"`.
La version portable (`AstroForge-<version>-portable.zip`) se décompresse
dans un dossier quelconque ; lancer `AstroForge.exe`.

**Longueur des chemins** : Windows limite un chemin complet à 260
caractères. Les fichiers d'AstroForge sont construits pour ne jamais
dépasser 140 caractères à l'intérieur de son dossier (QML, tests embarqués
et fichiers de compilation retirés ; contrôle automatique à la construction,
`packaging/windows/prune_site.py`). Le dossier d'installation ou d'extraction
doit donc faire au plus 110 caractères — l'installeur le vérifie. En cas de
message « noms de fichiers trop longs » avec un ancien paquet, extrayez dans
un dossier court (par exemple `C:\AstroForge`).

La signature exige vos propres certificats (secrets du dépôt, voir
l'en-tête de `release.yml`) ; sans eux, les installeurs sont produits non
signés. Mises à jour : la publication génère `latest.json`
(`packaging/make_manifest.py`) ; renseignez son adresse dans
**Configuration › Mises à jour** (ou `update_url` de settings.json) — le
programme le consulte au plus une fois par jour et propose la page de
téléchargement, sans rien installer de lui-même.

## Aide intégrée (F1) — façon .chm

Le menu **Aide → Aide (F1)** ouvre un visualiseur intégré à trois onglets,
façon `.chm` : sommaire arborescent (tutoriel + manuel complets), **Index
A→Z de mots-clés** (filtrable, clic = ouverture de la page la plus
pertinente) et **recherche par mot-clé** en plein texte (insensible aux
accents et à la casse), navigation précédent/suivant. **F1 est
contextuel** : il ouvre directement la page correspondant à l'onglet
actif (Ciel profond, Planétaire, Finition, Configuration). Les pages
(`help/`) sont compilées sans lecteur PDF depuis `docs_src/` par :
`python3 tools/build_help.py`.

## Lancer les tests

```bash
pip install -e ".[astro,dev]"
pytest                    # tests unitaires par module + suite de bout en bout
pytest -m "not slow"      # sans la suite de bout en bout (tests/smoke.py)
python tests/smoke.py     # suite de bout en bout seule
ruff check astroforge tests
```

L'intégration continue (`.github/workflows/tests.yml`) exécute ces tests sur
Windows, macOS et Linux (Python 3.10 et 3.12).

## Structure

```
main.py                    point d'entrée
ui/*.ui                    interfaces Qt Designer (éditables dans Qt Designer)
astroforge/core/           algorithmes : E/S (FITS/SER/RAW), Bayer, calibration,
                           alignement, drizzle, empilement, ondelettes, déconvolution,
                           gradient, satellites, HDR, fausses couleurs, animation
astroforge/ai/             IA locale : modèles ONNX (models/ai/) + outils externes
astroforge/processing/     pipeline automatique + profils (profiles/)
astroforge/gui/            pages des 3 modes, aperçu live, workers, profils, menu Outils
astroforge/i18n/           traductions (en.json)
astroforge/data/           catalogue OpenNGC (Messier/NGC/IC, CC BY-SA 4.0)
astroforge/cli.py          ligne de commande (python -m astroforge)
packaging/                 exécutable, installeurs, manifeste de mises à jour
tests/                     pytest (un fichier par module) + smoke.py
profiles/                  profils enregistrés (JSON) + ai_tools.json (voir ci-dessous)
models/ai/                 déposez ici vos modèles .onnx (+ .json de config)
```

### Emplacement des réglages

Les profils, réglages et modèles IA de l'utilisateur sont écrits dans le
dossier du programme s'il est modifiable (installation « portable »), sinon
dans le dossier de configuration de l'utilisateur : `%APPDATA%\AstroForge`
(Windows), `~/Library/Application Support/AstroForge` (macOS),
`~/.config/astroforge` (Linux) — c'est toujours le cas pour l'exécutable
PyInstaller. La variable d'environnement `ASTROFORGE_HOME` force un autre
dossier.

## L'aperçu live

Chaque mode recalcule l'aperçu (~450 ms après le dernier réglage) sur une copie
réduite de l'image : réglez les paramètres et voyez l'effet immédiatement
(bascule Avant / Après / Côte à côte, étirement auto commutable). La fenêtre est
**entièrement redimensionnable** : les colonnes Paramètres / Aperçu / Exécution
suivent la taille de la fenêtre, l'aperçu occupant tout l'espace libéré.

## Profils de paramètres

Chaque groupe (alignement, empilement, traitement, IA, ondelettes, animation…)
possède une barre `[profil ▾] [💾 Enregistrer] [↺ Standard]`. Le profil
**standard** contient les valeurs par défaut ; créez autant de profils que
voulu (un JSON par profil dans `profiles/`).

## IA locale — comment l'activer

Trois voies, du plus simple au plus intégré :

1. **Outils externes (recommandé, résultats professionnels)** — installez puis
   renseignez les chemins via « Configurer les outils IA… » :
   - [GraXpert](https://www.graxpert.com) — denoise + extraction de fond par IA
     (gratuit ; `pip install graxpert[cpuonly]` ou installateur sur le site).
   - [StarNet2 / DeepSNR](https://starnetastro.com/) — suppression des étoiles
     (StarNet2, échange en TIFF 16 bits) et denoise DeepSNR en ligne de
     commande. Si votre version attend d'autres arguments, renseignez le
     gabarit `"args"` dans `ai_tools.json`, par ex. `"{input} {output} 256"`.
   - [Cosmic Clarity (SetiAstro)](https://www.setiastro.com/cosmic-clarity) :
     **pas d'automatisation** (son flux Input/Output est interactif). Utilisez-le
     à part puis rechargez sa sortie dans AstroForge.
2. **Modèles ONNX** — déposez un modèle `.onnx` dans `models/ai/` (dossier
   des réglages, voir plus haut) avec un JSON
   optionnel du même nom :
   ```json
   {"task": "superres", "input_size": 256, "overlap": 32, "scale": 4,
    "norm": {"scale": 1.0, "offset": 0.0}}
   ```
   Inférence par tuiles avec fondu ; GPU utilisé automatiquement si disponible
   (CUDA / DirectML / CoreML). La « force » mélange le résultat du modèle avec
   l'image d'origine (1 = modèle seul).
3. **Algorithmes classiques intégrés** (aucune IA requise) : non-local means,
   bilatéral, médiane, ondelettes (denoise) ; Richardson-Lucy (déconvolution) ;
   polynôme 2D / RBF (gradient) ; Hough + inpainting (satellites) ; Lanczos
   (super-résolution).

## Formats pris en charge

- Entrée ciel profond : FITS (mono/RGB, entiers 8-32 bits normalisés selon
  leur dynamique, flottants, FITS compressé .fz), TIFF/PNG/JPEG 8-16 bits,
  RAW APN (via rawpy), XISF natif (PixInsight).
- Matrices de Bayer : RGGB, BGGR, GRBG, GBRG (choix manuel, ou auto depuis le
  RAW / le mot-clé FITS `BAYERPAT`). La calibration se fait sur le CFA brut,
  le dématriçage ensuite.
- Entrée planétaire : SER norme v3 (mono, Bayer ColorID 8-11, RGB/BGR,
  8-16 bits, lu par projection mémoire : vidéos de plusieurs Go sans saturer
  la RAM), AVI, MP4, séquences d'images FITS/TIFF/PNG.
- Sortie : FITS float32 et XISF float32 avec en-tête complet (NCOMBINE, pose
  totale, dates, objet, filtre, WCS + SIP), TIFF 16 bits ou 32 bits flottant,
  PNG 8/16 bits, JPEG 96 %, SER v3 standard (avec horodatage), MP4/AVI/GIF.

## Limites connues

- Le drizzle (noyau carré, pixfrac) travaille sur les images calibrées AVANT
  alignement avec une similitude par image (translation, rotation, échelle) ;
  les distorsions polynomiales ne sont pas modélisées pendant le drizzle.
- Le solveur interne d'astrométrie modélise les distorsions (SIP, degré 3)
  mais suppose un centre approximatif à ±radius ; pour de très grands champs,
  préférez astrometry.net.
- Traduction anglaise : libellés, menus, boutons et infobulles ; les éléments
  des listes déroulantes et une partie des messages du journal restent en
  français (ils servent aussi de valeurs internes).
- GPU : chemin CuPy validé par des tests sur un substitut NumPy ; non mesuré
  sur une vraie carte graphique.
- Aucune comparaison chiffrée avec Siril sur des données réelles n'est
  fournie : les tests utilisent des données synthétiques à vérité connue.
- XISF : lecture non compressé, zlib et zlib+sh (LZ4/LZ4HC/Zstandard si les
  modules `lz4` / `zstandard` sont installés) ; écriture Float32 non
  compressée. Seule la première image d'un fichier multi-images est lue.
- Le sigma-clipping reste plus efficace à partir d'une dizaine d'images ; en
  dessous, le pré-rejet robuste (médiane/MAD) élimine déjà pixels chauds et
  traces.

Licence : MIT (voir [LICENSE](LICENSE)). Le catalogue
`astroforge/data/openngc_lite.csv` est extrait d'OpenNGC (Mattia Verga),
sous licence CC BY-SA 4.0.
