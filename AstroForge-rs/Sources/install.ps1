# Installation d'AstroForge (version Rust) — Windows 10/11.
#
#   install.bat                         installe dans %LOCALAPPDATA%\Programs\AstroForge
#   install.bat -Portable D:\AstroForge version portable (données à côté de l'exécutable)
#   install.bat -Uninstall              désinstalle (données utilisateur conservées)
#
# Ou directement : powershell -ExecutionPolicy Bypass -File install.ps1 [options]
# Get-Help .\install.ps1 -Detailed pour toutes les options.

<#
.SYNOPSIS
Compile et installe AstroForge (moteur et ligne de commande en Rust).
.PARAMETER Prefix
Dossier d'installation (défaut : %LOCALAPPDATA%\Programs\AstroForge).
.PARAMETER Portable
Copie l'exécutable dans ce dossier avec un sous-dossier profiles\ : réglages et masters y restent.
.PARAMETER Onnx
Active l'inférence ONNX locale (tract).
.PARAMETER NoRaw
Sans lecture des RAW d'appareils photo.
.PARAMETER NoNet
Sans fonctions réseau.
.PARAMETER NoGui
Sans l'interface graphique (astroforge-gui.exe : onglets Planétaire et Planétaire IA automatique).
.PARAMETER Yes
Répond oui à toutes les questions.
.PARAMETER NoPath
Ne modifie pas le PATH de l'utilisateur.
.PARAMETER Uninstall
Supprime l'installation.
.PARAMETER FromSource
Compile depuis les sources même si un astroforge.exe précompilé est présent.
.PARAMETER Purge
Avec -Uninstall : supprime aussi les données utilisateur (%APPDATA%\AstroForge).
#>
[CmdletBinding()]
param(
    [string]$Prefix = "",
    [string]$Portable = "",
    [switch]$Onnx,
    [switch]$NoRaw,
    [switch]$NoNet,
    [switch]$NoGui,
    [int]$Jobs = 0,
    [switch]$Yes,
    [switch]$NoPath,
    [switch]$Uninstall,
    [switch]$Purge,
    [switch]$FromSource
)

$ErrorActionPreference = "Stop"
$Version = "2.0.0"
$RustMin = [version]"1.88"
$SrcDir = $PSScriptRoot
if (-not $Prefix) { $Prefix = Join-Path $env:LOCALAPPDATA "Programs\AstroForge" }
$Prefix = [IO.Path]::GetFullPath($Prefix)
$Manifest = Join-Path $Prefix "install-manifest.txt"
$ExeName = if ($env:OS -eq "Windows_NT") { "astroforge.exe" } else { "astroforge" }
$GuiName = if ($env:OS -eq "Windows_NT") { "astroforge-gui.exe" } else { "astroforge-gui" }
$Shortcut = if ($env:APPDATA) { Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\AstroForge.lnk" } else { "" }

function Info($m) { Write-Host "==> $m" -ForegroundColor Green }
function Warn($m) { Write-Host "Attention : $m" -ForegroundColor Yellow }
function Die($m)  { Write-Host "Erreur : $m" -ForegroundColor Red; exit 1 }
function Ask($q) {
    if ($Yes) { return $true }
    $r = Read-Host "$q [O/n]"
    return -not ($r -match '^[nN]')
}

function Get-UserPath { [Environment]::GetEnvironmentVariable("Path", "User") }
function Set-UserPath($v) { [Environment]::SetEnvironmentVariable("Path", $v, "User") }

# ------------------------------------------------------------------ désinstallation
if ($Uninstall) {
    $found = $true
    if (Test-Path $Manifest) {
        Info "Désinstallation depuis $Prefix"
        foreach ($f in Get-Content $Manifest) {
            if ($f -and (Test-Path $f)) { Remove-Item -Force $f; Write-Host "  supprimé : $f" }
        }
        if (Test-Path $Manifest) { Remove-Item -Force $Manifest }
    } elseif (Test-Path (Join-Path $Prefix $ExeName)) {
        Remove-Item -Force (Join-Path $Prefix $ExeName)
        if (Test-Path (Join-Path $Prefix $GuiName)) { Remove-Item -Force (Join-Path $Prefix $GuiName) }
        if ($Shortcut -and (Test-Path $Shortcut)) { Remove-Item -Force $Shortcut }
    } else {
        Warn "aucune installation trouvée dans $Prefix (utilisez -Prefix si besoin)"
        $found = $false
    }
    if ((Test-Path $Prefix) -and -not (Get-ChildItem $Prefix -Force)) { Remove-Item $Prefix }
    $p = Get-UserPath
    if ($p) {
        $parts = $p.Split(';') | Where-Object { $_ -and ($_.TrimEnd('\') -ne $Prefix.TrimEnd('\')) }
        Set-UserPath ($parts -join ';')
    }
    $data = if ($env:ASTROFORGE_HOME) { $env:ASTROFORGE_HOME } else { Join-Path $env:APPDATA "AstroForge" }
    if ($Purge -and (Test-Path $data) -and (Ask "Supprimer les données utilisateur ($data) ?")) {
        Remove-Item -Recurse -Force $data
        Write-Host "  supprimé : $data"
    } else {
        Write-Host "Données utilisateur conservées : $data"
    }
    if ($found) { Info "AstroForge est désinstallé." }
    exit 0
}

# ------------------------------------------------------------------ exécutable précompilé ?
# Le paquet Windows contient astroforge.exe à côté de ce script : rien à compiler.
$prebuilt = Join-Path $SrcDir $ExeName
$hasSources = Test-Path (Join-Path $SrcDir "Cargo.toml")
if ((Test-Path $prebuilt) -and -not $FromSource) {
    if ($Onnx -or $NoRaw -or $NoNet -or $NoGui) {
        Warn "options de compilation ignorées : exécutable précompilé (utilisez -FromSource depuis les sources)"
    }
    Info "Exécutable précompilé trouvé : aucune compilation nécessaire"
    $exe = $prebuilt
    $guiExe = Join-Path $SrcDir $GuiName
    if ($NoGui -or -not (Test-Path $guiExe)) { $guiExe = $null }
} elseif (-not $hasSources) {
    Die "ni astroforge.exe ni Cargo.toml à côté de ce script : lancez-le depuis le paquet AstroForge"
} else {

# ------------------------------------------------------------------ vérifications (compilation)
$cargoHome = Join-Path $env:USERPROFILE ".cargo\bin"
if (-not (Get-Command cargo -ErrorAction SilentlyContinue) -and (Test-Path (Join-Path $cargoHome "cargo.exe"))) {
    $env:Path = "$cargoHome;$env:Path"
}

function Install-Rust {
    Info "Installation de Rust via rustup"
    $tmp = Join-Path $env:TEMP "rustup-init.exe"
    $arch = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "aarch64" } else { "x86_64" }
    Invoke-WebRequest -UseBasicParsing "https://static.rust-lang.org/rustup/dist/$arch-pc-windows-msvc/rustup-init.exe" -OutFile $tmp
    & $tmp -y --profile minimal
    if ($LASTEXITCODE -ne 0) { Die "l'installation de Rust a échoué" }
    $env:Path = "$cargoHome;$env:Path"
}

if (Get-Command cargo -ErrorAction SilentlyContinue) {
    $rv = [version](((rustc --version) -split ' ')[1] -replace '-.*$', '')
    if ($rv -lt $RustMin) {
        if (Get-Command rustup -ErrorAction SilentlyContinue) {
            Info "Rust $rv est trop ancien (minimum $RustMin) : mise à jour"
            rustup update stable
        } else {
            Die "Rust $rv est trop ancien (minimum $RustMin) : https://rustup.rs"
        }
    }
} elseif (Ask "Rust (cargo) est introuvable. L'installer maintenant avec rustup ?") {
    Install-Rust
} else {
    Die "Rust est nécessaire pour compiler AstroForge : https://rustup.rs"
}
Info ("Rust : " + (rustc --version))

# Rust sous Windows (cible MSVC) a besoin de l'éditeur de liens link.exe
# fourni par les « Build Tools for Visual Studio » (composant C++).
$host_triple = ((rustc -vV) | Select-String '^host:').ToString().Split(' ')[1]
if ($host_triple -like "*msvc*") {
    $vsInstallerDir = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer"
    $vswhere = Join-Path $vsInstallerDir "vswhere.exe"
    function Test-VcTools {
        if (Get-Command link.exe -ErrorAction SilentlyContinue) { return $true }
        if (-not (Test-Path $vswhere)) { return $false }
        $p = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        return [bool]$p
    }
    if (-not (Test-VcTools)) {
        Warn "l'éditeur de liens de Microsoft (link.exe) est absent : il fait partie des « Build Tools for Visual Studio » (composant C++)."
        $vsPath = if (Test-Path $vswhere) { & $vswhere -latest -products * -property installationPath } else { $null }
        $components = "--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621"
        if ($vsPath) {
            # Visual Studio présent mais sans le C++ : on ajoute le composant
            if (-not (Ask "Ajouter les outils C++ à l'installation existante ($vsPath) ? (environ 2 Go, droits administrateur)")) {
                Die "outils C++ requis. Ouvrez « Visual Studio Installer » > Modifier > cochez « Développement Desktop en C++ », puis relancez install.bat"
            }
            Start-Process -FilePath (Join-Path $vsInstallerDir "setup.exe") -Verb RunAs -Wait `
                -ArgumentList "modify --installPath `"$vsPath`" $components --passive --norestart"
        } elseif (Get-Command winget -ErrorAction SilentlyContinue) {
            if (-not (Ask "Installer les Build Tools for Visual Studio avec winget ? (environ 2 Go, droits administrateur)")) {
                Die "outils C++ requis : https://visualstudio.microsoft.com/fr/visual-cpp-build-tools/ (cochez « Développement Desktop en C++ »), puis relancez install.bat"
            }
            winget install --id Microsoft.VisualStudio.2022.BuildTools -e --accept-package-agreements --accept-source-agreements `
                --override "--passive --wait --norestart --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
        } else {
            Die "installez les Build Tools for Visual Studio : https://visualstudio.microsoft.com/fr/visual-cpp-build-tools/ (cochez « Développement Desktop en C++ »), puis relancez install.bat"
        }
        if (-not (Test-VcTools)) {
            Die "les outils C++ ne sont toujours pas détectés. Terminez leur installation (un redémarrage peut être nécessaire) puis relancez install.bat"
        }
        Info "Outils C++ de Visual Studio installés"
    }
}

# ------------------------------------------------------------------ compilation
$cargoArgs = @("build", "--release", "--locked", "--manifest-path", (Join-Path $SrcDir "Cargo.toml"))
$features = @()
if ($Onnx) { $features += "onnx" }
if ($NoRaw -or $NoNet -or $NoGui) {
    $cargoArgs += "--no-default-features"
    if (-not $NoRaw) { $features += "raw" }
    if (-not $NoNet) { $features += "net" }
    if (-not $NoGui) { $features += "gui" }
}
if ($features.Count) { $cargoArgs += @("--features", ($features -join ",")) }
if ($Jobs -gt 0) { $cargoArgs += @("--jobs", "$Jobs") }

Info "Compilation optimisée (quelques minutes la première fois)…"
Write-Host "  cargo $($cargoArgs -join ' ')"
& cargo @cargoArgs
if ($LASTEXITCODE -ne 0) { Die "la compilation a échoué" }

$meta = cargo metadata --format-version 1 --no-deps --manifest-path (Join-Path $SrcDir "Cargo.toml") | ConvertFrom-Json
$exe = Join-Path (Join-Path $meta.target_directory "release") $ExeName
if (-not (Test-Path $exe)) { Die "exécutable introuvable après compilation : $exe" }
$guiExe = $null
if (-not $NoGui) {
    $guiExe = Join-Path (Join-Path $meta.target_directory "release") $GuiName
    if (-not (Test-Path $guiExe)) { Warn "interface graphique introuvable après compilation : $guiExe"; $guiExe = $null }
}
}

$docs = @(@("README.md", "LICENSE", "OPENNGC_LICENSE.txt", (Join-Path "data" "OPENNGC_LICENSE.txt")) | ForEach-Object { Join-Path $SrcDir $_ } | Where-Object { Test-Path $_ })

# ------------------------------------------------------------------ version portable
if ($Portable) {
    $Portable = [IO.Path]::GetFullPath($Portable)
    New-Item -ItemType Directory -Force -Path (Join-Path $Portable "profiles") | Out-Null
    Copy-Item -Force $exe $Portable
    if ($guiExe) { Copy-Item -Force $guiExe $Portable }
    foreach ($d in $docs) { Copy-Item -Force $d $Portable }
    Info "Version portable prête : $(Join-Path $Portable $ExeName)"
    Write-Host "Réglages, profils et masters seront enregistrés dans $Portable"
    & (Join-Path $Portable $ExeName) info
    exit 0
}

# ------------------------------------------------------------------ installation
Info "Installation dans $Prefix"
New-Item -ItemType Directory -Force -Path $Prefix | Out-Null
$installed = @()
$bins = @($exe)
if ($guiExe) { $bins += $guiExe }
foreach ($f in $bins + $docs + @((Join-Path $SrcDir "install.ps1"), (Join-Path $SrcDir "install.bat"))) {
    if (-not (Test-Path $f)) { continue }
    $dst = Join-Path $Prefix (Split-Path $f -Leaf)
    if ([IO.Path]::GetFullPath($f) -eq [IO.Path]::GetFullPath($dst)) { $installed += $dst; continue }
    try { Copy-Item -Force $f $dst }
    catch { Die "impossible d'écrire $dst (AstroForge est-il en cours d'exécution ?)" }
    $installed += $dst
    Write-Host "  $dst"
}
# raccourci du menu Démarrer vers l'interface graphique
if ($guiExe -and $Shortcut) {
    try {
        $ws = New-Object -ComObject WScript.Shell
        $lnk = $ws.CreateShortcut($Shortcut)
        $lnk.TargetPath = Join-Path $Prefix $GuiName
        $lnk.WorkingDirectory = $Prefix
        $lnk.Description = "AstroForge — Planétaire et Planétaire IA"
        $lnk.Save()
        $installed += $Shortcut
        Write-Host "  $Shortcut"
    } catch { Warn "raccourci du menu Démarrer non créé : $_" }
}
$installed += $Manifest
$installed | Set-Content -Encoding UTF8 $Manifest

# ------------------------------------------------------------------ PATH
$userPath = Get-UserPath
$inPath = $userPath -and ($userPath.Split(';') | Where-Object { $_.TrimEnd('\') -eq $Prefix.TrimEnd('\') })
if (-not $inPath) {
    if (-not $NoPath -and (Ask "Ajouter $Prefix au PATH de l'utilisateur ?")) {
        Set-UserPath ((@($userPath, $Prefix) | Where-Object { $_ }) -join ';')
        $env:Path = "$env:Path;$Prefix"
        Write-Host "PATH mis à jour : ouvrez un nouveau terminal pour l'utiliser."
    } else {
        Warn "$Prefix n'est pas dans le PATH."
    }
}

# ------------------------------------------------------------------ vérification
Info "Vérification"
& (Join-Path $Prefix $ExeName) info
if ($LASTEXITCODE -ne 0) { Die "l'exécutable installé ne démarre pas" }
Write-Host ""
Info "AstroForge $Version est installé : $(Join-Path $Prefix $ExeName)"
if ($guiExe) { Write-Host "Interface graphique : menu Démarrer > AstroForge (ou astroforge-gui), onglets Planétaire et Planétaire IA" }
Write-Host "Aide : astroforge --help — désinstallation : $(Join-Path $Prefix 'install.bat') -Uninstall"
