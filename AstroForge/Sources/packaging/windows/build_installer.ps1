# Installeur Windows : signe l'exécutable (si un certificat est fourni),
# compile l'installeur Inno Setup, puis signe l'installeur.
#   Variables : CERT (PFX en base64) et CERT_PASSWORD — facultatives.
$ErrorActionPreference = "Stop"
$root = Resolve-Path "$PSScriptRoot\..\.."
Set-Location $root
$version = (python -c "import astroforge; print(astroforge.__version__)").Trim()
New-Item -ItemType Directory -Force -Path packaging\out | Out-Null

function Find-SignTool {
  $kits = Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\bin" -Recurse -Filter signtool.exe -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -match "x64" } | Sort-Object FullName -Descending
  if ($kits) { return $kits[0].FullName } else { return $null }
}

function Sign-File($path) {
  if (-not $env:CERT) { Write-Host "(pas de certificat : $path non signé)"; return }
  $pfx = Join-Path $env:RUNNER_TEMP "astroforge.pfx"
  [IO.File]::WriteAllBytes($pfx, [Convert]::FromBase64String($env:CERT))
  $st = Find-SignTool
  & $st sign /f $pfx /p $env:CERT_PASSWORD /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 $path
  if ($LASTEXITCODE -ne 0) { throw "Échec de la signature de $path" }
  Remove-Item $pfx
}

Sign-File "dist\AstroForge\AstroForge.exe"

$iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
if (-not (Test-Path $iscc)) {
  choco install innosetup --no-progress -y | Out-Null
}
& $iscc "/DAppVersion=$version" packaging\windows\astroforge.iss
if ($LASTEXITCODE -ne 0) { throw "Échec Inno Setup" }

$setup = "packaging\out\AstroForge-$version-windows-setup.exe"
Sign-File $setup
Write-Host "Installeur : $setup"
