# TRAIT FIN 1.0 - Vectorisation d'image en SVG (tracé fin et lisse)

CONTENU
  Android/Trait-Fin-1.0.apk            Application Android (Android 7 ou plus)
  Windows/Trait-Fin-Setup-1.0.0.exe    Installateur Windows 10/11 64 bits
  Web (hors ligne)/Trait-Fin.html      Version navigateur, sans installation
  Sources/                             Code source et fichiers de compilation

INSTALLATION ANDROID
  1. Copiez l'APK sur le téléphone et ouvrez-le.
  2. Autorisez l'installation depuis cette source si Android le demande
     (l'application ne vient pas du Play Store).

INSTALLATION WINDOWS
  1. Lancez Trait-Fin-Setup-1.0.0.exe (pas de droits administrateur requis).
  2. Si SmartScreen s'affiche : « Informations complémentaires »,
     puis « Exécuter quand même » (l'exécutable n'est pas signé).
  Désinstallation : Paramètres > Applications > Trait Fin.

VERSION WEB
  Ouvrez Trait-Fin.html avec Chrome, Edge ou Firefox. Aucune connexion requise
  (seules les polices de caractères viennent d'Internet si disponible).

UTILISATION
  - Ouvrir une image (ou la glisser / la coller avec Ctrl+V).
  - Mode « Ligne centrale » : une seule ligne au milieu de chaque trait
    (croquis, plans, dessins au trait).
  - Mode « Contour » : ligne fine qui suit le bord des formes (logos, texte).
  - Réglez seuil, nettoyage, lissage et simplification, puis
    « Enregistrer le SVG ».
  Tout le calcul est fait sur l'appareil ; l'image n'est envoyée nulle part.

SOURCES
  Sources/index.html          Application complète (HTML + JavaScript)
  Sources/android/            Projet Apktool (manifeste, smali, ressources)
                              Compiler : java -jar apktool.jar b android
                              puis signer avec uber-apk-signer.
  Sources/windows/            Projet Electron + script NSIS
                              npm install ; npx electron-builder --win --dir ;
                              node seticon.mjs "dist/win-unpacked/Trait Fin.exe" ;
                              makensis installer.nsi
  Sources/icones/             Icône de l'application (PNG, ICO)

AUTRES LOGICIELS DU DÉPÔT
  AstroForge/                  AstroForge 2.0.0 - traitement d'images
                               astronomiques (voir AstroForge/README.md ;
                               installateur Windows et version portable
                               dans AstroForge/Windows/)
  AstroForge-rs/               AstroForge en Rust : interface Planétaire,
                               Planétaire IA (traitement automatique) et
                               Time-lapse multi-vidéos ; programmes Windows
                               dans AstroForge-rs/Windows (voir
                               AstroForge-rs/README.md)
