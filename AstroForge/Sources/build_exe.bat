@echo off
rem Construit un executable autonome avec PyInstaller (Windows) :
rem   build_exe.bat  ->  dist\AstroForge.exe
setlocal
cd /d %~dp0
python -m pip install --quiet pyinstaller

if not exist ui_generated mkdir ui_generated
for %%f in (ui\*.ui) do (
  python -c "from astroforge.gui.loader import _generate; import sys; _generate(sys.argv[1])" "%%~nxf"
  if errorlevel 1 exit /b 1
)

pyinstaller --noconfirm --windowed --name AstroForge ^
  --add-data "ui;ui" ^
  --add-data "ui_generated;ui_generated" ^
  --add-data "help;help" ^
  --collect-submodules astroforge ^
  --collect-data astroforge ^
  main.py

echo Executable pret : dist\AstroForge\AstroForge.exe
pause
endlocal
