# ASTROFORGE 2.0.0 - Traitement d'images astronomiques

Ciel profond (calibration, alignement, empilement, drizzle, multi-nuits,
mosaïques, comètes, empilement en direct), planétaire et solaire
(multipoint, dérotation, WinJUPOS), astrométrie, finition (GHS, masques,
bande étroite), IA locale, ligne de commande. Interface en français et en
anglais. Licence MIT.

TÉLÉCHARGEMENT (publication « AstroForge 2.0.0 » du dépôt)
  https://github.com/decmike91/fourtout/releases/tag/astroforge-v2.0.0
  AstroForge-2.0.0-windows-setup.exe   Installateur Windows 10/11 64 bits
                                       (Python et dépendances inclus)
  AstroForge-2.0.0-portable.zip        Version sans installation
  AstroForge_2.0.0_sources.zip         Sources (identiques à Sources/)
  Les fichiers dépassent la limite de 100 Mo de GitHub pour un dépôt :
  ils sont donc joints à la publication et non au dossier.

INSTALLATION WINDOWS
  1. Lancez AstroForge-2.0.0-windows-setup.exe : installation pour vous seul
     (sans droits administrateur) ou pour tous les utilisateurs.
  2. Si SmartScreen s'affiche : « Informations complémentaires »,
     puis « Exécuter quand même » (l'exécutable n'est pas signé).
  3. Choisissez un dossier d'installation court (110 caractères au plus).
  Désinstallation : Paramètres > Applications > AstroForge.
  Vérifier l'installation : menu Démarrer > AstroForge (ligne de commande),
  puis  astroforge-cli.exe script "<dossier>\check_install.py"

VERSION PORTABLE
  Décompressez le .zip dans un dossier court (par ex. C:\AstroForge) et
  lancez AstroForge.exe. Le premier démarrage est un peu plus long
  (Python prépare son cache).

SOURCES
  Sources/                      Code Python/Qt complet (voir Sources/README.md)
    python main.py              Lancer depuis les sources (Python 3.10+,
                                pip install -r requirements.txt)
    pip install ".[astro,ia,raw]"   Installer comme paquet (astroforge, astroforge-gui)
    pytest                      Tests
    packaging/windows/build_windows.sh --portable
                                Reconstruire l'installateur Windows (Linux/WSL)
