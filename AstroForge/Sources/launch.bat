@echo off
rem Lancement d'AstroForge (Windows) : cree le venv et installe les dependances
rem au premier lancement, puis demarre l'application.
setlocal
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  echo Creation de l'environnement virtuel ^( .venv ^)...
  python -m venv .venv
)
call .venv\Scripts\activate.bat
python -c "import PySide6, astropy, cv2, scipy, numpy" 2>NUL
if errorlevel 1 python -m pip install --quiet -r requirements.txt
python main.py %*
endlocal
