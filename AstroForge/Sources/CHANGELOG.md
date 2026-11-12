# Journal des versions

## 2.0.0 — compléments Windows (26 octobre 2026)
- **Installeur Windows autonome** `AstroForge-2.0.0-windows-setup.exe` (NSIS) :
  Python 3.12 embarqué avec toutes les dépendances, installation pour tous
  les utilisateurs ou pour l'utilisateur courant, menu Démarrer, raccourci
  bureau, « Ouvrir avec » pour .fits/.xisf, désinstallation propre ; version
  **portable** .zip. Construit depuis Linux par
  `packaging/windows/build_windows.sh` ; installé et vérifié sur Windows par
  l'intégration continue (`release.yml`).
- Lanceurs natifs `AstroForge.exe` et `astroforge-cli.exe` (icône, version,
  arguments Unicode, code de sortie).
- Un fichier passé en argument (double-clic, « Ouvrir avec ») s'ouvre dans
  l'onglet Finition.
- Interfaces Qt pré-générées utilisées telles quelles dans une installation
  en lecture seule.
- `check_install.py` : vérification d'une installation.
- **Chemins de fichiers raccourcis** : la distribution Windows contenait des
  fichiers (QML de Qt, fichiers .obj de compilation, suites de tests des
  dépendances) dont le chemin dépassait 175 caractères, ce qui provoquait
  l'erreur « noms de fichiers trop longs » à l'extraction de la version
  portable dans un dossier profond. Ils sont retirés (≈ 95 Mo de moins) ;
  aucun chemin ne dépasse désormais 140 caractères dans le dossier du
  programme (contrôlé à la construction) ; l'installeur refuse un dossier de
  plus de 110 caractères ; version portable renommée
  `AstroForge-<v>-portable.zip` (nom de dossier d'extraction plus court).

## 2.0.0 — 15 octobre 2026

### Ciel profond
- **Tri automatique d'une session** (`core/session.py`) : classement des
  lumières, bias, darks, darks de flat et flats d'après les en-têtes
  FITS/XISF (IMAGETYP, EXPTIME, GAIN, température, binning, filtre, objet) ou,
  à défaut, les noms de fichiers et dossiers. Groupes par objet · filtre ·
  pose · gain, chacun avec sa calibration compatible. Bouton « Importer une
  session (tri auto)… ».
- **Bibliothèque de masters** (`core/masters.py`) : enregistrement et
  réutilisation automatique selon pose, gain, température, binning, taille et
  filtre. **Mise à l'échelle automatique du dark** quand la pose diffère.
- **Analyse des brutes** (`core/subframes.py`) : étoiles, FWHM et excentricité
  (moments adaptatifs), fond, SNR ; rejet des images aberrantes ; poids SNR,
  FWHM ou mixte utilisés par l'empilement et le drizzle.
- Sorties **Ha / OIII** depuis un capteur couleur (filtres duo-bande).
- **Normalisation locale** et **empilement multi-nuits**.
- **Mosaïques** par WCS (reprojection, fondu) ou par appariement d'étoiles.
- **Comètes** : double empilement étoiles / noyau, recombinaison.
- **Empilement en direct** d'un dossier d'acquisition (rejet κ-σ progressif).

### Finition
- Étirements **GHS** et **arcsinh** avec histogramme.
- **Masques** (luminance, fond, étoiles, sans étoiles, plage), applicables à
  tout filtre ; **réduction d'étoiles** ; séparation et **recombinaison des
  étoiles** (images sans étoiles StarNet/GraXpert) ; **débruitage de
  chrominance**.
- **Composition couleur** : palettes HOO, HOO équilibrée, SHO, HSO, OSH,
  RGB, et **LRGB**.

### Planétaire et solaire
- Alignement **multipoint local** façon AutoStakkert!.
- Horodatage des vidéos SER (lecture et écriture), **export WinJUPOS**,
  **dérotation de séries** (sphère aplatie, projection orthographique).
- Outils solaires : assombrissement centre-bord, protubérances renforcées /
  disque inversé, mosaïque H-alpha.

### Astrométrie
- Solveur par **triangles invariants** (images miroir comprises) avec
  ajustement des **distorsions SIP**.
- **Annotation** des objets Messier, NGC, IC (catalogue OpenNGC, CC BY-SA 4.0).
- **Courbes de lumière** (photométrie différentielle, export CSV).

### IA
- **Catalogue de modèles ONNX téléchargeables** (https ou fichier, empreinte
  SHA-256 vérifiée) ; aucun modèle tiers n'est référencé par défaut.
- **Comparaison avant/après** de plusieurs résultats.

### Socle
- Aperçu : zoom (ajuster, 50 % à 400 %), molette, déplacement, valeur du
  pixel sous le curseur, médiane, bruit, SNR.
- Export **FITS et XISF avec en-tête complet** (NCOMBINE, pose totale, dates,
  objet, filtre, instrument, WCS + SIP) ; **TIFF 32 bits flottant**.
- **Ligne de commande** `astroforge` (`python -m astroforge`) : sort, analyze,
  stack, multinight, solve, annotate, compose, batch, live, script, info.
- **Scripts Python** intégrés (menu Outils).
- **Interface en anglais** (menu Langue / Language, `ASTROFORGE_LANG`).
- **GPU facultatif** (CuPy) pour moyenne, médiane et sigma-clipping, avec
  repli automatique sur le processeur.
- Estimation de rotation **log-polaire (Fourier–Mellin)** puis raffinement :
  environ quatre fois plus rapide que le balayage.
- **Vérification des mises à jour** (manifeste JSON configurable, au plus une
  fois par jour, rien n'est installé automatiquement).
- `pyproject.toml` (options astro, ia, raw, gpu, dev), commandes
  `astroforge` / `astroforge-gui`, ressources embarquées dans le paquet.
- Licence **MIT** (`LICENSE`).
- **Installeurs** : Windows (Inno Setup), macOS (.dmg, signature et
  notarisation si certificats fournis), Linux (AppImage) ; publication
  automatique et manifeste de mises à jour (`.github/workflows/release.yml`).
- **Tests** : pytest, un fichier par module, plus la suite de bout en bout
  `smoke.py` ; intégration continue Windows/macOS/Linux.
- Aide intégrée : chapitres 10 à 15 (nouveautés).

### Limites connues
- Traduction anglaise : les listes déroulantes et certains messages du journal
  restent en français.
- Chemin GPU testé sur un substitut NumPy, pas sur une vraie carte.
- Pas de jeu de données réel comparé à Siril : les tests reposent sur des
  données synthétiques à vérité connue.
- Signature des installeurs : nécessite vos propres certificats.

## 1.9.9

Correctifs de l'audit (alignement, Bayer, FITS BZERO, drizzle, XISF, MTF,
chemins Unicode, emplacement des données utilisateur…) — voir `CORRECTIFS.md`.
