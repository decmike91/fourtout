#!/bin/bash
# macOS : signature (Developer ID, runtime renforcé), notarisation Apple,
# agrafage du ticket et image disque .dmg. Sans certificat : dmg non signé
# (Gatekeeper demandera un clic droit › Ouvrir au premier lancement).
set -euo pipefail
cd "$(dirname "$0")/../.."
VERSION=$(python -c "import astroforge; print(astroforge.__version__)")
APP="dist/AstroForge.app"
OUT="packaging/out"
DMG="$OUT/AstroForge-$VERSION-macos.dmg"
mkdir -p "$OUT"

if [[ -n "${MACOS_CERT_P12_BASE64:-}" ]]; then
  KC="$RUNNER_TEMP/af.keychain-db"
  security create-keychain -p af "$KC"
  security set-keychain-settings -lut 21600 "$KC"
  security unlock-keychain -p af "$KC"
  echo "$MACOS_CERT_P12_BASE64" | base64 --decode > "$RUNNER_TEMP/cert.p12"
  security import "$RUNNER_TEMP/cert.p12" -k "$KC" -P "$MACOS_CERT_PASSWORD" -T /usr/bin/codesign
  security set-key-partition-list -S apple-tool:,apple: -s -k af "$KC"
  security list-keychains -d user -s "$KC" $(security list-keychains -d user | tr -d '"')
  codesign --force --deep --options runtime --timestamp \
    --entitlements packaging/macos/entitlements.plist \
    --sign "$MACOS_SIGN_IDENTITY" "$APP"
  codesign --verify --deep --strict --verbose=2 "$APP"
fi

hdiutil create -volname "AstroForge $VERSION" -srcfolder "$APP" -ov -format UDZO "$DMG"

if [[ -n "${MACOS_CERT_P12_BASE64:-}" ]]; then
  codesign --force --timestamp --sign "$MACOS_SIGN_IDENTITY" "$DMG"
  if [[ -n "${APPLE_ID:-}" ]]; then
    xcrun notarytool submit "$DMG" --apple-id "$APPLE_ID" --team-id "$APPLE_TEAM_ID" \
      --password "$APPLE_APP_PASSWORD" --wait
    xcrun stapler staple "$DMG"
  fi
else
  echo "(pas de certificat : dmg non signé)"
fi
echo "Image disque : $DMG"
