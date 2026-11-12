@echo off
rem Reconstitue les fichiers AstroForge decoupes en morceaux de 95 Mo
rem (GitHub refuse les fichiers de plus de 100 Mo dans un depot).
rem Double-cliquez sur ce fichier : il cree l'installateur et la version
rem portable dans ce dossier, puis verifie leur empreinte SHA-256.
cd /d "%~dp0"
set "ECHEC="
echo Reconstitution de AstroForge-2.0.0-windows-setup.exe ...
copy /b /y "AstroForge-2.0.0-windows-setup.exe.part00"+"AstroForge-2.0.0-windows-setup.exe.part01" "AstroForge-2.0.0-windows-setup.exe" >nul
echo Reconstitution de AstroForge-2.0.0-portable.zip ...
copy /b /y "AstroForge-2.0.0-portable.zip.part00"+"AstroForge-2.0.0-portable.zip.part01"+"AstroForge-2.0.0-portable.zip.part02" "AstroForge-2.0.0-portable.zip" >nul
echo.
echo Verification des empreintes SHA-256 :
call :verif "AstroForge-2.0.0-windows-setup.exe" 27E9DA2F2F4FFEFCF70D667E3AC04FADE3C049FED24218329D673CBA732BFD9D
call :verif "AstroForge-2.0.0-portable.zip" 8EE1CE84796CDD436742C53445E8C83DCB2052EF2BE84D0B4145E711F0872355
echo.
if defined ECHEC (
  echo Echec : verifiez que tous les morceaux .part00, .part01, .part02 sont presents et complets.
) else (
  echo Termine. Vous pouvez supprimer les fichiers .part00, .part01, .part02.
)
pause
exit /b 0

:verif
if not exist "%~1" (echo   ABSENT  %~1 & set "ECHEC=1" & exit /b 0)
set "H="
for /f "usebackq delims=" %%h in (`powershell -NoProfile -Command "(Get-FileHash -Algorithm SHA256 -LiteralPath '%~1').Hash"`) do set "H=%%h"
if /i "%H%"=="%~2" (echo   OK      %~1) else (echo   ERREUR  %~1 : fichier corrompu & set "ECHEC=1")
exit /b 0
