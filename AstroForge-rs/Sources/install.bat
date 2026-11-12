@echo off
rem Installation d'AstroForge (version Rust) sous Windows.
rem   install.bat                 installation standard
rem   install.bat -Portable D:\AF version portable
rem   install.bat -Uninstall      desinstallation
rem Bloc entre parentheses : lu en entier par cmd, il continue meme si la
rem desinstallation supprime ce fichier.
(
  chcp 65001 >nul
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
  if "%~1"=="" pause
  exit /b
)
