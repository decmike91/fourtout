#!/bin/sh
# Installation d'AstroForge (version Rust) — Linux et macOS.
#
#   ./install.sh                     installe dans ~/.local/bin
#   sudo ./install.sh --system       installe dans /usr/local/bin
#   ./install.sh --portable DOSSIER  version portable (données à côté de l'exécutable)
#   ./install.sh --uninstall         désinstalle (les données utilisateur sont conservées)
#
# ./install.sh --help pour toutes les options.
set -eu

VERSION="2.0.0"
RUST_MIN="1.88"
# shellcheck disable=SC1007
SRC_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

PREFIX=""
PORTABLE=""
FEATURES=""
WITH_RAW=1
WITH_NET=1
WITH_GUI=1
ASSUME_YES=0
UNINSTALL=0
PURGE=0
INSTALL_RUST=ask
MODIFY_PATH=ask
JOBS=""

# ------------------------------------------------------------------ affichage
if [ -t 1 ]; then
    B=$(printf '\033[1m'); G=$(printf '\033[32m'); Y=$(printf '\033[33m'); R=$(printf '\033[31m'); N=$(printf '\033[0m')
else
    B=""; G=""; Y=""; R=""; N=""
fi
info()  { printf '%s==>%s %s\n' "$B$G" "$N" "$*"; }
warn()  { printf '%sAttention :%s %s\n' "$B$Y" "$N" "$*" >&2; }
die()   { printf '%sErreur :%s %s\n' "$B$R" "$N" "$*" >&2; exit 1; }

ask() { # ask "question" -> 0 = oui
    [ "$ASSUME_YES" = 1 ] && return 0
    [ -t 0 ] || return 1
    printf '%s [O/n] ' "$1"
    read -r rep || return 1
    case "$rep" in [nN]*) return 1 ;; *) return 0 ;; esac
}

usage() {
    cat <<EOF
Installation d'AstroForge $VERSION (moteur et ligne de commande en Rust)

Usage : ./install.sh [options]

Emplacement :
  --prefix DOSSIER     installe dans DOSSIER/bin (défaut : ~/.local, ou
                       /usr/local avec --system ou en tant que root)
  --system             équivaut à --prefix /usr/local (droits administrateur)
  --portable DOSSIER   copie l'exécutable dans DOSSIER avec un sous-dossier
                       profiles/ : réglages et masters restent dans DOSSIER

Compilation :
  --onnx               active l'inférence ONNX locale (tract)
  --no-raw             sans lecture des RAW d'appareils photo
  --no-net             sans fonctions réseau (mises à jour, catalogues)
  --no-gui             sans l'interface graphique (astroforge-gui : onglets
                       Planétaire et Planétaire IA automatique)
  --jobs N             nombre de compilations parallèles

Comportement :
  -y, --yes            répond oui à toutes les questions
  --install-rust       installe Rust (rustup) sans demander s'il manque
  --no-install-rust    n'installe jamais Rust
  --no-path            ne modifie pas le PATH du shell
  --uninstall          supprime les fichiers installés
  --purge              avec --uninstall : supprime aussi les données
                       utilisateur (réglages, profils, bibliothèque de masters)
  -h, --help           affiche cette aide
EOF
}

# ------------------------------------------------------------------ options
while [ $# -gt 0 ]; do
    case "$1" in
        --prefix) [ $# -ge 2 ] || die "--prefix attend un dossier"; PREFIX=$2; shift ;;
        --prefix=*) PREFIX=${1#*=} ;;
        --system) PREFIX=/usr/local ;;
        --portable) [ $# -ge 2 ] || die "--portable attend un dossier"; PORTABLE=$2; shift ;;
        --portable=*) PORTABLE=${1#*=} ;;
        --onnx) FEATURES="$FEATURES onnx" ;;
        --no-raw) WITH_RAW=0 ;;
        --no-net) WITH_NET=0 ;;
        --no-gui) WITH_GUI=0 ;;
        --jobs) [ $# -ge 2 ] || die "--jobs attend un nombre"; JOBS=$2; shift ;;
        --jobs=*) JOBS=${1#*=} ;;
        -y|--yes) ASSUME_YES=1 ;;
        --install-rust) INSTALL_RUST=yes ;;
        --no-install-rust) INSTALL_RUST=no ;;
        --no-path) MODIFY_PATH=no ;;
        --uninstall) UNINSTALL=1 ;;
        --purge) PURGE=1 ;;
        -h|--help) usage; exit 0 ;;
        *) die "option inconnue : $1 (voir --help)" ;;
    esac
    shift
done

if [ -z "$PREFIX" ]; then
    if [ "$(id -u)" = 0 ]; then PREFIX=/usr/local; else PREFIX="$HOME/.local"; fi
fi
case "$PREFIX" in /*) ;; *) PREFIX="$(pwd)/$PREFIX" ;; esac
BIN_DIR="$PREFIX/bin"
SHARE_DIR="$PREFIX/share/astroforge"
DOC_DIR="$PREFIX/share/doc/astroforge"
MANIFEST="$SHARE_DIR/install-manifest.txt"

user_data_dir() {
    if [ -n "${ASTROFORGE_HOME:-}" ]; then echo "$ASTROFORGE_HOME"; return; fi
    case "$(uname -s)" in
        Darwin) echo "$HOME/Library/Application Support/AstroForge" ;;
        *) echo "${XDG_CONFIG_HOME:-$HOME/.config}/astroforge" ;;
    esac
}

# ------------------------------------------------------------------ désinstallation
if [ "$UNINSTALL" = 1 ]; then
    FOUND=1
    if [ -f "$MANIFEST" ]; then
        info "Désinstallation depuis $PREFIX"
        while IFS= read -r f; do
            [ -n "$f" ] && [ -e "$f" ] && rm -f -- "$f" && echo "  supprimé : $f"
        done < "$MANIFEST"
        rm -f -- "$MANIFEST"
        rmdir "$DOC_DIR" "$SHARE_DIR" 2>/dev/null || true
    elif [ -f "$BIN_DIR/astroforge" ]; then
        info "Pas de manifeste : suppression de $BIN_DIR/astroforge"
        rm -f -- "$BIN_DIR/astroforge" "$BIN_DIR/astroforge-gui"
    else
        warn "aucune installation trouvée dans $PREFIX (utilisez --prefix si besoin)"
        FOUND=0
    fi
    # ligne ajoutée au PATH par l'installation
    for rc in "$HOME/.bashrc" "$HOME/.bash_profile" "$HOME/.zshrc" "$HOME/.profile" "$HOME/.config/fish/config.fish"; do
        if [ -f "$rc" ] && grep -qsF "$BIN_DIR" "$rc" && grep -qs '^# AstroForge$' "$rc"; then
            awk -v bin="$BIN_DIR" '
                /^# AstroForge$/ { held = $0; next }
                held != "" { if (index($0, bin) > 0) { held = ""; next } print held; held = "" }
                { print }
                END { if (held != "") print held }' "$rc" > "$rc.astroforge.tmp" && cat "$rc.astroforge.tmp" > "$rc"
            rm -f "$rc.astroforge.tmp"
            echo "  PATH retiré de $rc"
        fi
    done
    if [ "$PURGE" = 1 ]; then
        d=$(user_data_dir)
        if [ -d "$d" ] && ask "Supprimer les données utilisateur ($d) ?"; then
            rm -rf -- "$d"
            echo "  supprimé : $d"
        fi
    else
        echo "Données utilisateur conservées : $(user_data_dir)"
    fi
    [ "$FOUND" = 1 ] && info "AstroForge est désinstallé."
    exit 0
fi

# ------------------------------------------------------------------ vérifications
[ -f "$SRC_DIR/Cargo.toml" ] || die "Cargo.toml introuvable : lancez ce script depuis le dossier des sources"

ver_ge() { # ver_ge 1.90.0 1.88 -> vrai si $1 >= $2
    a1=$(echo "$1" | cut -d. -f1); a2=$(echo "$1" | cut -d. -f2)
    b1=$(echo "$2" | cut -d. -f1); b2=$(echo "$2" | cut -d. -f2)
    [ "$a1" -gt "$b1" ] || { [ "$a1" -eq "$b1" ] && [ "$a2" -ge "$b2" ]; }
}

# cargo installé par rustup mais absent du PATH de ce shell
if ! command -v cargo >/dev/null 2>&1 && [ -x "$HOME/.cargo/bin/cargo" ]; then
    PATH="$HOME/.cargo/bin:$PATH"; export PATH
fi

install_rustup() {
    command -v curl >/dev/null 2>&1 || die "curl est nécessaire pour installer Rust (ou installez-le depuis https://rustup.rs)"
    info "Installation de Rust via rustup (dans ~/.cargo et ~/.rustup)"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
    PATH="$HOME/.cargo/bin:$PATH"; export PATH
}

if command -v cargo >/dev/null 2>&1; then
    RUSTV=$(rustc --version 2>/dev/null | awk '{print $2}')
    if ! ver_ge "${RUSTV:-0.0}" "$RUST_MIN"; then
        if command -v rustup >/dev/null 2>&1; then
            info "Rust $RUSTV est trop ancien (minimum $RUST_MIN) : mise à jour"
            rustup update stable
        else
            die "Rust $RUSTV est trop ancien (minimum $RUST_MIN). Mettez-le à jour ou installez rustup : https://rustup.rs"
        fi
    fi
else
    case "$INSTALL_RUST" in
        no) die "Rust (cargo) est introuvable. Installez-le depuis https://rustup.rs" ;;
        yes) install_rustup ;;
        ask) if ask "Rust (cargo) est introuvable. L'installer maintenant avec rustup ?"; then
                 install_rustup
             else
                 die "Rust est nécessaire pour compiler AstroForge : https://rustup.rs"
             fi ;;
    esac
fi
rustc --version >/dev/null 2>&1 || die "rustc est présent mais ne démarre pas : réparez l'installation de Rust (rustup self update ; rustup default stable)"
info "Rust : $(rustc --version)"

# éditeur de liens (cc) : seul outil système requis
if ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1 && ! command -v clang >/dev/null 2>&1; then
    case "$(uname -s)" in
        Darwin) die "aucun éditeur de liens : exécutez « xcode-select --install » puis relancez" ;;
        *) die "aucun compilateur C (éditeur de liens) : installez par exemple « build-essential » (Debian/Ubuntu), « gcc » (Fedora) ou « base-devel » (Arch)" ;;
    esac
fi

# ------------------------------------------------------------------ compilation
set -- build --release --locked --manifest-path "$SRC_DIR/Cargo.toml"
if [ "$WITH_RAW" = 0 ] || [ "$WITH_NET" = 0 ] || [ "$WITH_GUI" = 0 ]; then
    set -- "$@" --no-default-features
    [ "$WITH_RAW" = 1 ] && FEATURES="$FEATURES raw"
    [ "$WITH_NET" = 1 ] && FEATURES="$FEATURES net"
    [ "$WITH_GUI" = 1 ] && FEATURES="$FEATURES gui"
fi
# shellcheck disable=SC2086
FEATURES=$(echo $FEATURES | tr ' ' ',')
[ -n "$FEATURES" ] && set -- "$@" --features "$FEATURES"
[ -n "$JOBS" ] && set -- "$@" --jobs "$JOBS"

info "Compilation optimisée (quelques minutes la première fois)…"
echo "  cargo $*"
cargo "$@" || die "la compilation a échoué"

TARGET_DIR=$(cargo metadata --format-version 1 --no-deps --manifest-path "$SRC_DIR/Cargo.toml" 2>/dev/null \
    | sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p')
[ -n "$TARGET_DIR" ] || TARGET_DIR="$SRC_DIR/target"
EXE="$TARGET_DIR/release/astroforge"
[ -x "$EXE" ] || die "exécutable introuvable après compilation : $EXE"
GUI_EXE="$TARGET_DIR/release/astroforge-gui"
if [ "$WITH_GUI" = 1 ] && [ ! -x "$GUI_EXE" ]; then
    warn "interface graphique introuvable après compilation : $GUI_EXE"
    WITH_GUI=0
fi

# ------------------------------------------------------------------ installation portable
if [ -n "$PORTABLE" ]; then
    mkdir -p "$PORTABLE/profiles"
    cp "$EXE" "$PORTABLE/astroforge"
    [ "$WITH_GUI" = 1 ] && cp "$GUI_EXE" "$PORTABLE/astroforge-gui"
    for f in README.md LICENSE; do [ -f "$SRC_DIR/$f" ] && cp "$SRC_DIR/$f" "$PORTABLE/"; done
    [ -f "$SRC_DIR/data/OPENNGC_LICENSE.txt" ] && cp "$SRC_DIR/data/OPENNGC_LICENSE.txt" "$PORTABLE/"
    info "Version portable prête : $PORTABLE/astroforge"
    echo "Réglages, profils et masters seront enregistrés dans $PORTABLE"
    "$PORTABLE/astroforge" info || true
    exit 0
fi

# ------------------------------------------------------------------ installation
if ! mkdir -p "$BIN_DIR" "$SHARE_DIR" "$DOC_DIR" 2>/dev/null || [ ! -w "$BIN_DIR" ]; then
    die "pas de droit d'écriture dans $PREFIX : relancez avec sudo, ou choisissez --prefix ~/.local"
fi

info "Installation dans $PREFIX"
: > "$MANIFEST.tmp"
put() { # put source destination mode
    [ -f "$1" ] || return 0
    rm -f -- "$2"     # remplace un exécutable en cours d'utilisation sans erreur
    cp "$1" "$2"
    chmod "$3" "$2"
    echo "$2" >> "$MANIFEST.tmp"
    echo "  $2"
}
put "$EXE" "$BIN_DIR/astroforge" 755
if [ "$WITH_GUI" = 1 ]; then
    put "$GUI_EXE" "$BIN_DIR/astroforge-gui" 755
    if [ "$(uname -s)" != Darwin ]; then
        # raccourci du menu des applications (Linux)
        APP_DIR="$PREFIX/share/applications"
        mkdir -p "$APP_DIR"
        cat > "$APP_DIR/astroforge.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=AstroForge
GenericName=Traitement planétaire
Comment=Planétaire et Planétaire IA (traitement automatique de vidéos SER)
Exec=$BIN_DIR/astroforge-gui %f
Terminal=false
Categories=Graphics;Science;Astronomy;
MimeType=application/octet-stream;
DESKTOP
        echo "$APP_DIR/astroforge.desktop" >> "$MANIFEST.tmp"
        echo "  $APP_DIR/astroforge.desktop"
    fi
fi
put "$SRC_DIR/README.md" "$DOC_DIR/README.md" 644
put "$SRC_DIR/LICENSE" "$DOC_DIR/LICENSE" 644
put "$SRC_DIR/data/OPENNGC_LICENSE.txt" "$DOC_DIR/OPENNGC_LICENSE.txt" 644
put "$SRC_DIR/install.sh" "$SHARE_DIR/install.sh" 755
mv "$MANIFEST.tmp" "$MANIFEST"

# ------------------------------------------------------------------ PATH
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        LINE="export PATH=\"$BIN_DIR:\$PATH\""
        case "$(basename "${SHELL:-sh}")" in
            zsh) RC="$HOME/.zshrc" ;;
            bash) if [ "$(uname -s)" = Darwin ]; then RC="$HOME/.bash_profile"; else RC="$HOME/.bashrc"; fi ;;
            fish) RC="$HOME/.config/fish/config.fish"; LINE="fish_add_path \"$BIN_DIR\"" ;;
            *) RC="$HOME/.profile" ;;
        esac
        if [ "$MODIFY_PATH" != no ] && [ "$(id -u)" != 0 ] && ask "$BIN_DIR n'est pas dans le PATH. L'ajouter à $RC ?"; then
            if ! grep -qsF "$LINE" "$RC"; then
                mkdir -p "$(dirname "$RC")"
                printf '\n# AstroForge\n%s\n' "$LINE" >> "$RC"
            fi
            echo "PATH mis à jour : ouvrez un nouveau terminal (ou « . $RC »)."
        else
            warn "$BIN_DIR n'est pas dans le PATH ; ajoutez : $LINE"
        fi
        ;;
esac

# ------------------------------------------------------------------ vérification
info "Vérification"
"$BIN_DIR/astroforge" info || die "l'exécutable installé ne démarre pas"
echo
info "AstroForge $VERSION est installé : $BIN_DIR/astroforge"
[ "$WITH_GUI" = 1 ] && echo "Interface graphique : astroforge-gui (onglets Planétaire et Planétaire IA)"
echo "Aide : astroforge --help — désinstallation : $SHARE_DIR/install.sh --uninstall${PREFIX:+ --prefix $PREFIX}"
