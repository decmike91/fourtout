#!/bin/sh
# Construit un exécutable autonome avec PyInstaller (Linux/Mac) :
#   ./build_exe.sh  ->  dist/AstroForge
# Les dossiers ui/ et ui_generated/ sont embarqués dans le bundle.
set -e
cd "$(dirname "$0")"
python3 -m pip install --quiet pyinstaller

# Pré-compile les fichiers .ui (le chargeur basculera sinon sur QUiLoader)
mkdir -p ui_generated
for f in ui/*.ui; do
  if command -v pyside6-uic >/dev/null 2>&1; then
    python3 -c 'from astroforge.gui.loader import _generate; import sys; _generate(sys.argv[1])' "$(basename "$f")"
  else
    echo "(pyside6-uic absent — QUiLoader sera utilisé dans le bundle)"
  fi
done

pyinstaller --noconfirm --windowed --name AstroForge \
  --add-data "ui:ui" \
  --add-data "ui_generated:ui_generated" \
  --add-data "help:help" \
  --collect-submodules astroforge \
  --collect-data astroforge \
  main.py

echo "Exécutable prêt : dist/AstroForge/AstroForge (dossier dist/AstroForge)"
