#!/bin/bash
# Linux : AppImage autonome à partir du dossier PyInstaller dist/AstroForge.
# Signature GPG facultative (APPIMAGE_GPG_KEY_BASE64), vérifiable avec
# « ./AstroForge*.AppImage --appimage-signature ».
set -euo pipefail
cd "$(dirname "$0")/../.."
VERSION=$(python -c "import astroforge; print(astroforge.__version__)")
OUT="packaging/out"
APPDIR="build/AstroForge.AppDir"
mkdir -p "$OUT"
rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications" \
         "$APPDIR/usr/share/icons/hicolor/512x512/apps"
cp -r dist/AstroForge/* "$APPDIR/usr/bin/"
cp packaging/astroforge.png "$APPDIR/astroforge.png"
cp packaging/astroforge.png "$APPDIR/usr/share/icons/hicolor/512x512/apps/astroforge.png"
cp packaging/linux/astroforge.desktop "$APPDIR/astroforge.desktop"
cp packaging/linux/astroforge.desktop "$APPDIR/usr/share/applications/"
cat > "$APPDIR/AppRun" <<'RUN'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/AstroForge" "$@"
RUN
chmod +x "$APPDIR/AppRun"

TOOL="build/appimagetool-x86_64.AppImage"
if [[ ! -x "$TOOL" ]]; then
  curl -fsSL -o "$TOOL" \
    https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
  chmod +x "$TOOL"
fi
SIGN=()
if [[ -n "${APPIMAGE_GPG_KEY_BASE64:-}" ]]; then
  echo "$APPIMAGE_GPG_KEY_BASE64" | base64 --decode | gpg --batch --import
  SIGN=(--sign)
fi
ARCH=x86_64 APPIMAGE_EXTRACT_AND_RUN=1 "$TOOL" "${SIGN[@]}" "$APPDIR" \
  "$OUT/AstroForge-$VERSION-x86_64.AppImage"
echo "AppImage : $OUT/AstroForge-$VERSION-x86_64.AppImage"
