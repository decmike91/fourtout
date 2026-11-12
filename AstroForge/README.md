# ASTROFORGE 2.0.0 - Traitement d'images astronomiques

Ciel profond (calibration, alignement, empilement, drizzle, multi-nuits,
mosaïques, comètes, empilement en direct), planétaire et solaire
(multipoint, dérotation, WinJUPOS), astrométrie, finition (GHS, masques,
bande étroite), IA locale, ligne de commande. Interface en français et en
anglais. Licence MIT.

CONTENU
  Windows/                       Installateur et version portable, découpés
                                 en morceaux de 95 Mo (GitHub refuse les
                                 fichiers de plus de 100 Mo dans un dépôt)
    reconstituer.bat             Double-clic : recrée les deux fichiers
                                 ci-dessous et vérifie leur SHA-256
    reconstituer.sh              Idem sous Linux / macOS
    -> AstroForge-2.0.0-windows-setup.exe  Installateur Windows 10/11 64 bits
                                           (Python et dépendances inclus)
    -> AstroForge-2.0.0-portable.zip       Version sans installation
  AstroForge_2.0.0_sources.zip   Sources (identiques à Sources/)
  Sources/                       Code source

RÉCUPÉRER LES FICHIERS WINDOWS
  1. Téléchargez le dépôt (bouton « Code » > « Download ZIP ») ou les
     fichiers du dossier Windows/ (.part00, .part01, .part02 et
     reconstituer.bat).
  2. Double-cliquez sur reconstituer.bat : « OK » doit s'afficher pour
     les deux fichiers.

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
