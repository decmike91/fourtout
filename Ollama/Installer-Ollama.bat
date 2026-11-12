@echo off
REM Lance Install-Ollama.ps1 sans modifier la politique d'execution.
REM Les arguments sont transmis au script, ex :
REM   Installer-Ollama.bat -Model llama3.2 -ModelsDir "D:\OllamaModels"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-Ollama.ps1" %*
pause
