#!/bin/sh
# Reconstitue les fichiers AstroForge decoupes (Linux, macOS, WSL).
set -e
cd "$(dirname "$0")"
cat AstroForge-2.0.0-windows-setup.exe.part0? > AstroForge-2.0.0-windows-setup.exe
cat AstroForge-2.0.0-portable.zip.part0? > AstroForge-2.0.0-portable.zip
sha256sum -c <<SUMS 2>/dev/null || shasum -a 256 -c <<SUMS2
27e9da2f2f4ffefcf70d667e3ac04fade3c049fed24218329d673cba732bfd9d  AstroForge-2.0.0-windows-setup.exe
8ee1ce84796cdd436742c53445e8c83dcb2052ef2be84d0b4145e711f0872355  AstroForge-2.0.0-portable.zip
SUMS
27e9da2f2f4ffefcf70d667e3ac04fade3c049fed24218329d673cba732bfd9d  AstroForge-2.0.0-windows-setup.exe
8ee1ce84796cdd436742c53445e8c83dcb2052ef2be84d0b4145e711f0872355  AstroForge-2.0.0-portable.zip
SUMS2
