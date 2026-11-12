#!/bin/sh
# Compile les lanceurs Windows avec MinGW-w64 (Linux : apt install mingw-w64).
#   build.sh <version> <dossier de sortie>
set -e
cd "$(dirname "$0")"
VER=${1:-0.0.0}
OUT=${2:-.}
MAJ=$(echo "$VER" | cut -d. -f1); MIN=$(echo "$VER" | cut -d. -f2); PAT=$(echo "$VER" | cut -d. -f3)
CC=${CC:-x86_64-w64-mingw32-gcc}
RC=${RC:-x86_64-w64-mingw32-windres}
mkdir -p "$OUT"
$RC -DVER_MAJOR="$MAJ" -DVER_MINOR="$MIN" -DVER_PATCH="$PAT" -DVER_STRING="\\\"$VER\\\"" \
    -O coff -i launcher.rc -o "$OUT/launcher.res"
$CC -O2 -s -municode -mwindows -o "$OUT/AstroForge.exe" launcher.c "$OUT/launcher.res" -lshell32
$CC -O2 -s -municode -DASTROFORGE_CONSOLE -o "$OUT/astroforge-cli.exe" launcher.c "$OUT/launcher.res" -lshell32
rm -f "$OUT/launcher.res"
echo "Lanceurs : $OUT/AstroForge.exe $OUT/astroforge-cli.exe"
