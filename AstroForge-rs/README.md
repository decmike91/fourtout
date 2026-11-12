# ASTROFORGE (Rust) 2.0.0 - Planétaire, Planétaire IA et Time-lapse

Portage en Rust du moteur d'AstroForge, avec une interface graphique dédiée
au planétaire :

  - Planétaire : traitement d'une vidéo SER réglé à la main (tri,
    alignement global / multi-points / points locaux, empilement,
    ondelettes, alignement RVB).
  - Planétaire IA (automatique) : un moteur de décision local analyse la
    vidéo, trie et classe les images, choisit l'alignement et l'empilement,
    règle les ondelettes et le débruitage, et vérifie qu'aucun artefact
    n'est introduit (bruit visible, halo au limbe, écrêtage).
  - Time-lapse (multi-vidéos) : une image par vidéo (éclipse, rotation d'une
    planète...), même traitement, même cadrage et même rendu pour toutes,
    puis animation fluide (MP4, AVI, GIF ou images numérotées).

CONTENU
  Windows/AstroForge-2.0.0-windows-x64.zip   Programmes Windows 10/11 64 bits
                                             (astroforge-gui.exe, astroforge.exe,
                                             install.bat, LISEZ-MOI.txt)
  Sources/                                   Code source Rust (cargo)

INSTALLATION WINDOWS
  1. Décompressez le .zip et double-cliquez sur astroforge-gui.exe
     (aucune autre installation nécessaire : ni Python, ni DLL).
  2. Ou lancez install.bat : raccourci « AstroForge » dans le menu Démarrer
     et ligne de commande dans le PATH (sans droits administrateur).
  3. Si SmartScreen s'affiche : « Informations complémentaires »,
     puis « Exécuter quand même » (les exécutables ne sont pas signés).
  L'export MP4 / AVI des animations demande ffmpeg (winget install ffmpeg).

COMPILER DEPUIS LES SOURCES
  Windows : Sources\install.bat -FromSource
  Linux / macOS : ./Sources/install.sh
  Ou : cd Sources && cargo build --release
  (Rust 1.88 ou plus ; le détail est dans Sources/README.md)

DIFFÉRENCE AVEC AstroForge/
  AstroForge/ contient la version Python complète (ciel profond, astrométrie,
  interface PySide6...). AstroForge-rs/ est la version Rust : même moteur de
  calcul, exécutables autonomes, et les onglets Planétaire IA et Time-lapse.
