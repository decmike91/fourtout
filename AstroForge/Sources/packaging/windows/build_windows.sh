#!/bin/bash
# Construit l'installeur Windows d'AstroForge DEPUIS Linux (ou WSL) :
#
#   packaging/windows/build_windows.sh            -> packaging/out/AstroForge-<v>-windows-setup.exe
#   packaging/windows/build_windows.sh --portable -> + AstroForge-<v>-portable.zip
#
# Principe : Python Windows autonome (python-build-standalone) + roues
# (wheels) Windows de toutes les dépendances installées par « uv » pour la
# plate-forme cible, interfaces Qt pré-générées, bytecode pré-compilé,
# lanceurs AstroForge.exe / astroforge-cli.exe (MinGW-w64), puis NSIS.
# Aucun Windows n'est nécessaire pour construire.
#
# Prérequis (Ubuntu/Debian) : apt install nsis mingw-w64 curl zip python3.12
#                             pip install uv
# Variables : PY_VERSION (3.12.15), PBS_TAG (20261021), EXTRAS (astro,ia,raw),
#             PYTHON_HOST (python3.12 : même version mineure que la cible),
#             MAX_REL_PATH (140 : longueur maximale d'un chemin dans l'installation).
# Signature Authenticode facultative (apt install osslsigncode) :
#             WINDOWS_CERT_PFX (fichier .pfx) et WINDOWS_CERT_PASSWORD ;
#             WINDOWS_TIMESTAMP_URL (http://timestamp.digicert.com par défaut).
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$(pwd)

PY_VERSION=${PY_VERSION:-3.12.15}
PBS_TAG=${PBS_TAG:-20261021}
EXTRAS=${EXTRAS:-astro,ia,raw}
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f1,2)
PYTHON_HOST=${PYTHON_HOST:-python$PY_MINOR}
PORTABLE=0
[[ "${1:-}" == "--portable" ]] && PORTABLE=1

VERSION=$(python3 -c "import re;print(re.search(r'__version__ = \"(.+?)\"', open('astroforge/__init__.py').read()).group(1))")
WORK="$ROOT/build/windows"
STAGE="$WORK/stage"
OUT="$ROOT/packaging/out"
SP="$STAGE/python/Lib/site-packages"
mkdir -p "$WORK/cache" "$OUT"

for tool in uv makensis x86_64-w64-mingw32-gcc curl; do
  command -v "$tool" >/dev/null || { echo "Outil manquant : $tool" >&2; exit 1; }
done
echo "== AstroForge $VERSION pour Windows x64 (Python $PY_VERSION)"

# 1. Python Windows autonome -----------------------------------------------
PBS="cpython-${PY_VERSION}+${PBS_TAG}-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"
if [[ ! -s "$WORK/cache/$PBS" ]]; then
  curl -fL --retry 3 -o "$WORK/cache/$PBS.part" \
    "https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_TAG}/${PBS//+/%2B}"
  mv "$WORK/cache/$PBS.part" "$WORK/cache/$PBS"
fi
rm -rf "$STAGE" && mkdir -p "$STAGE"
tar xzf "$WORK/cache/$PBS" -C "$STAGE"
# inutile à l'exécution : en-têtes C, Tk, suites de tests de CPython
( cd "$STAGE/python" && rm -rf include libs tcl Lib/tkinter Lib/idlelib Lib/turtledemo \
    Lib/test Scripts DLLs/_tkinter.pyd DLLs/tcl*.dll DLLs/tk*.dll DLLs/_test*.pyd )

# 2. AstroForge + dépendances (roues Windows) -------------------------------
rm -rf "$WORK/wheel" && mkdir -p "$WORK/wheel"
uv build --wheel --out-dir "$WORK/wheel" "$ROOT" >/dev/null
WHEEL=$(ls "$WORK/wheel"/astroforge-*.whl)
uv pip install --quiet --target "$SP" \
  --python-platform x86_64-pc-windows-msvc --python-version "$PY_MINOR" --only-binary :all: \
  "$WHEEL[$EXTRAS]"
rm -rf "$SP/bin"   # scripts console générés pour l'hôte Linux

# PySide6 : seuls les modules « Essentials » servent (QtWidgets, QtUiTools…) ;
# Addons, WebEngine et Pdf (≈ 450 Mo) sont retirés d'après leurs RECORD.
python3 - "$SP" <<'PY'
import csv, glob, os, shutil, sys
os.chdir(sys.argv[1])
ess = {r[0] for r in csv.reader(open(glob.glob("pyside6_essentials-*.dist-info")[0] + "/RECORD", encoding="utf-8"))}
for pat in ("pyside6_addons-*", "pyside6_pdf-*", "pyside6_webengine-*", "pyside6-[0-9]*"):
    for dist in glob.glob(pat + ".dist-info"):
        for row in csv.reader(open(os.path.join(dist, "RECORD"), encoding="utf-8")):
            p = row[0]
            if p not in ess and not p.startswith(dist) and os.path.isfile(p):
                os.remove(p)
        shutil.rmtree(dist)
for root, dirs, files in os.walk("PySide6", topdown=False):
    if not os.listdir(root):
        os.rmdir(root)
PY

# Allègement (QML, tests embarqués, fichiers de compilation…) et contrôle de
# la longueur des chemins : Windows limite un chemin complet à 260 caractères
# (au-delà, l'Explorateur refuse d'extraire : « noms de fichiers trop longs »).
python3 packaging/windows/prune_site.py "$SP" --max-len "${MAX_REL_PATH:-140}"

# 3. Interfaces Qt pré-générées (pyside6-uic de la même version, hôte Linux)
QTVER=$(ls -d "$SP"/pyside6_essentials-*.dist-info | sed -E 's/.*essentials-([0-9.]+)\.dist-info/\1/')
if [[ ! -x "$WORK/uic/PySide6/Qt/libexec/uic" ]]; then
  uv pip install --quiet --target "$WORK/uic" --python-version "$PY_MINOR" \
    --python-platform x86_64-manylinux_2_34 "pyside6-essentials==$QTVER"
fi
GEN="$SP/astroforge/_res/ui_generated"
mkdir -p "$GEN"
LD_LIBRARY_PATH="$WORK/uic/PySide6/Qt/lib" python3 - "$ROOT/ui" "$GEN" "$WORK/uic/PySide6/Qt/libexec/uic" <<'PY'
import hashlib, os, re, subprocess, sys
src, gen, uic = sys.argv[1:]
for f in sorted(os.listdir(src)):
    if not f.endswith(".ui"):
        continue
    xml = open(os.path.join(src, f), encoding="utf-8").read()
    m = re.search(r"<class>(\w+)</class>", xml)
    py = os.path.join(gen, f"ui_{m.group(1) if m else f[:-3]}.py")
    subprocess.run([uic, "-g", "python", os.path.join(src, f), "-o", py], check=True)
    with open(py, "a", encoding="utf-8") as o:   # même marqueur que gui/loader.py
        o.write("\n# ui-source-sha256: " + hashlib.sha256(xml.encode("utf-8")).hexdigest() + "\n")
PY

# 4. Bytecode pré-compilé (démarrage plus rapide ; Program Files est en
#    lecture seule). Format .pyc identique sous Windows pour une même version.
if command -v "$PYTHON_HOST" >/dev/null && \
   [[ "$("$PYTHON_HOST" -c 'import sys;print("%d.%d"%sys.version_info[:2])')" == "$PY_MINOR" ]]; then
  "$PYTHON_HOST" -m compileall -q -j 0 --invalidation-mode unchecked-hash \
    "$STAGE/python/Lib" >/dev/null || true
else
  echo "(Python $PY_MINOR hôte absent : bytecode non pré-compilé)"
fi

# Contrôle final, bytecode compris : chemin le plus long vu depuis le dossier
# d'installation (ou depuis le dossier où l'on extrait la version portable).
LONGEST=$(cd "$STAGE" && find . -type f | sed 's|^\./|AstroForge/|' | awk '{ if (length($0) > m) { m = length($0); p = $0 } } END { print m " " p }')
echo "Chemin le plus long : $LONGEST"
if (( ${LONGEST%% *} > ${MAX_REL_PATH:-140} + 11 )); then
  echo "Chemin trop long (> ${MAX_REL_PATH:-140} + 11 caractères)" >&2
  exit 1
fi

# 5. Lanceurs (+ signature facultative) --------------------------------------
bash packaging/windows/launcher/build.sh "$VERSION" "$STAGE" >/dev/null
cp packaging/windows/check_install.py "$STAGE/"

sign() {   # sign <fichier.exe>
  [[ -n "${WINDOWS_CERT_PFX:-}" ]] || return 0
  command -v osslsigncode >/dev/null || { echo "osslsigncode requis pour signer" >&2; exit 1; }
  osslsigncode sign -pkcs12 "$WINDOWS_CERT_PFX" -pass "${WINDOWS_CERT_PASSWORD:-}" \
    -h sha256 -n "AstroForge" -t "${WINDOWS_TIMESTAMP_URL:-http://timestamp.digicert.com}" \
    -in "$1" -out "$1.signed"
  mv "$1.signed" "$1"
  echo "Signé : $(basename "$1")"
}
sign "$STAGE/AstroForge.exe"
sign "$STAGE/astroforge-cli.exe"

# 6. Installeur NSIS ---------------------------------------------------------
SETUP="$OUT/AstroForge-$VERSION-windows-setup.exe"
makensis -V2 -DVERSION="$VERSION" -DSTAGE="$STAGE" -DOUTFILE="$SETUP" \
  packaging/windows/astroforge.nsi
sign "$SETUP"
[[ -n "${WINDOWS_CERT_PFX:-}" ]] || echo "(installeur non signé : Windows SmartScreen affichera un avertissement)"
echo "Installeur : $SETUP ($(du -h "$SETUP" | cut -f1))"

if [[ $PORTABLE == 1 ]]; then
  # nom court : l'Explorateur extrait dans un dossier du même nom
  ZIP="$OUT/AstroForge-$VERSION-portable.zip"
  rm -f "$ZIP"
  cp LICENSE "$STAGE/LICENSE.txt"
  ( cd "$WORK" && rm -rf AstroForge && mv stage AstroForge && zip -qr -9 "$ZIP" AstroForge && mv AstroForge stage )
  echo "Version portable : $ZIP ($(du -h "$ZIP" | cut -f1))"
fi
