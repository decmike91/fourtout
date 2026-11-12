#!/bin/sh
# Lancement d'AstroForge (Linux/Mac) : crée le venv et installe les dépendances
# au premier lancement, puis démarre l'application.
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Création de l'environnement virtuel (.venv)…"
  python3 -m venv .venv
fi
. .venv/bin/activate
python -c "import PySide6, astropy, cv2, scipy, numpy" 2>/dev/null || \
  python -m pip install --quiet -r requirements.txt
exec python main.py "$@"
