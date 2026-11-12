<#
.SYNOPSIS
    Télécharge et installe Ollama sous Windows (installation silencieuse).

.DESCRIPTION
    - Télécharge le dernier OllamaSetup.exe depuis ollama.com
    - Vérifie la signature numérique de l'installeur
    - Installe en mode silencieux (pas besoin des droits administrateur)
    - Option : dossier d'installation et dossier des modèles personnalisés
    - Vérifie que le serveur Ollama répond, puis télécharge un modèle si demandé

.PARAMETER InstallDir
    Dossier d'installation. Par défaut : %LOCALAPPDATA%\Programs\Ollama

.PARAMETER ModelsDir
    Dossier de stockage des modèles (définit OLLAMA_MODELS pour l'utilisateur).

.PARAMETER Model
    Modèle à télécharger après l'installation (ex : llama3.2, mistral, qwen3).

.PARAMETER Force
    Réinstalle même si Ollama est déjà présent (utile pour mettre à jour).

.EXAMPLE
    .\Install-Ollama.ps1
.EXAMPLE
    .\Install-Ollama.ps1 -InstallDir "D:\Ollama" -ModelsDir "D:\OllamaModels" -Model llama3.2
#>

[CmdletBinding()]
param(
    [string]$InstallDir,
    [string]$ModelsDir,
    [string]$Model,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$DownloadUrl = 'https://ollama.com/download/OllamaSetup.exe'
$ApiUrl      = 'http://127.0.0.1:11434/api/version'

function Write-Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    $msg" -ForegroundColor Green }
function Write-Warn2($msg){ Write-Host "    $msg" -ForegroundColor Yellow }

function Update-SessionPath {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user    = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user"
}

function Test-OllamaApi {
    try { Invoke-RestMethod -Uri $ApiUrl -TimeoutSec 3 } catch { $null }
}

# --- 0. Prérequis ----------------------------------------------------------
if ([Environment]::OSVersion.Version.Major -lt 10) {
    throw "Ollama nécessite Windows 10 ou plus récent."
}
if (-not [Environment]::Is64BitOperatingSystem) {
    throw "Ollama nécessite un Windows 64 bits."
}
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# --- 1. Déjà installé ? ----------------------------------------------------
Update-SessionPath
$existing = Get-Command ollama -ErrorAction SilentlyContinue
if ($existing -and -not $Force) {
    $ver = (& ollama --version 2>$null) -join ' '
    Write-Ok "Ollama est déjà installé ($ver) : $($existing.Source)"
    Write-Ok "Relancez avec -Force pour réinstaller / mettre à jour."
}
else {
    # --- 2. Téléchargement -------------------------------------------------
    $workDir   = Join-Path $env:TEMP ("ollama-install-" + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $workDir | Out-Null
    $installer = Join-Path $workDir 'OllamaSetup.exe'

    Write-Step "Téléchargement de $DownloadUrl"
    try {
        Start-BitsTransfer -Source $DownloadUrl -Destination $installer -DisplayName 'Ollama'
    } catch {
        Write-Warn2 "BITS indisponible, utilisation d'Invoke-WebRequest..."
        $old = $ProgressPreference; $ProgressPreference = 'SilentlyContinue'  # bien plus rapide
        Invoke-WebRequest -Uri $DownloadUrl -OutFile $installer -UseBasicParsing
        $ProgressPreference = $old
    }
    $sizeMb = [math]::Round((Get-Item $installer).Length / 1MB, 1)
    Write-Ok "Téléchargé ($sizeMb Mo)"

    # --- 3. Vérification de la signature ----------------------------------
    Write-Step "Vérification de la signature numérique"
    $sig = Get-AuthenticodeSignature -FilePath $installer
    if ($sig.Status -ne 'Valid') {
        throw "Signature invalide ($($sig.Status)). Installation annulée."
    }
    if ($sig.SignerCertificate.Subject -notmatch 'Ollama') {
        throw "Signataire inattendu : $($sig.SignerCertificate.Subject). Installation annulée."
    }
    Write-Ok "Signature valide : $($sig.SignerCertificate.Subject)"

    # --- 4. Arrêt d'une instance en cours ---------------------------------
    Get-Process -Name 'ollama', 'ollama app' -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue

    # --- 5. Installation silencieuse --------------------------------------
    Write-Step "Installation d'Ollama (silencieuse)"
    $setupArgs = @('/VERYSILENT', '/SUPPRESSMSGBOXES', '/NORESTART', '/SP-')
    if ($InstallDir) {
        New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
        $setupArgs += "/DIR=`"$InstallDir`""
        Write-Ok "Dossier d'installation : $InstallDir"
    }
    $proc = Start-Process -FilePath $installer -ArgumentList $setupArgs -Wait -PassThru
    if ($proc.ExitCode -ne 0) {
        throw "L'installeur a échoué (code de sortie $($proc.ExitCode))."
    }
    Write-Ok "Installation terminée"
    Remove-Item $workDir -Recurse -Force -ErrorAction SilentlyContinue
    Update-SessionPath
}

# --- 6. Dossier des modèles -----------------------------------------------
if ($ModelsDir) {
    Write-Step "Configuration du dossier des modèles : $ModelsDir"
    New-Item -ItemType Directory -Path $ModelsDir -Force | Out-Null
    [Environment]::SetEnvironmentVariable('OLLAMA_MODELS', $ModelsDir, 'User')
    $env:OLLAMA_MODELS = $ModelsDir
    # Redémarre Ollama pour qu'il prenne en compte la variable
    Get-Process -Name 'ollama', 'ollama app' -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Write-Ok "OLLAMA_MODELS défini pour l'utilisateur"
}

# --- 7. Démarrage et vérification ------------------------------------------
$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollamaCmd) {
    $guess = if ($InstallDir) { $InstallDir } else { Join-Path $env:LOCALAPPDATA 'Programs\Ollama' }
    $exe = Join-Path $guess 'ollama.exe'
    if (Test-Path $exe) { $env:Path += ";$guess"; $ollamaCmd = Get-Command $exe }
    else { throw "ollama.exe introuvable après l'installation." }
}

Write-Step "Démarrage du serveur Ollama"
if (-not (Test-OllamaApi)) {
    $appExe = Join-Path (Split-Path $ollamaCmd.Source) 'ollama app.exe'
    if (Test-Path $appExe) { Start-Process -FilePath $appExe }
    else { Start-Process -FilePath $ollamaCmd.Source -ArgumentList 'serve' -WindowStyle Hidden }
}

$info = $null
for ($i = 0; $i -lt 30 -and -not $info; $i++) {
    Start-Sleep -Seconds 1
    $info = Test-OllamaApi
}
if ($info) {
    Write-Ok "Serveur actif sur http://localhost:11434 (version $($info.version))"
} else {
    Write-Warn2 "Le serveur ne répond pas encore. Lancez « ollama serve » manuellement si besoin."
}

# --- 8. Téléchargement d'un modèle (optionnel) -----------------------------
if ($Model -and $info) {
    Write-Step "Téléchargement du modèle « $Model »"
    & $ollamaCmd.Source pull $Model
    if ($LASTEXITCODE -eq 0) { Write-Ok "Modèle prêt. Essayez : ollama run $Model" }
    else { Write-Warn2 "Échec du téléchargement du modèle « $Model »." }
}

Write-Host ""
Write-Host "Terminé ! Ouvrez un NOUVEAU terminal pour utiliser la commande « ollama »." -ForegroundColor Green
