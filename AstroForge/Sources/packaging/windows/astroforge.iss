; Installeur Windows (Inno Setup 6) — compilé par build_installer.ps1 :
;   ISCC /DAppVersion=2.0.0 packaging\windows\astroforge.iss
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{6B1E2D7C-4F2A-4E55-9C1B-AF0A57F0F0A1}
AppName=AstroForge
AppVersion={#AppVersion}
AppPublisher=Contributeurs AstroForge
DefaultDirName={autopf}\AstroForge
DefaultGroupName=AstroForge
DisableProgramGroupPage=yes
LicenseFile=..\..\LICENSE
OutputDir=..\out
OutputBaseFilename=AstroForge-{#AppVersion}-windows-setup
SetupIconFile=astroforge.ico
UninstallDisplayIcon={app}\AstroForge.exe
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog
WizardStyle=modern

[Languages]
Name: "french"; MessagesFile: "compiler:Languages\French.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "assocfits"; Description: "Ouvrir les fichiers .fits/.xisf/.ser avec AstroForge"; Flags: unchecked

[Files]
Source: "..\..\dist\AstroForge\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\AstroForge"; Filename: "{app}\AstroForge.exe"
Name: "{group}\{cm:UninstallProgram,AstroForge}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\AstroForge"; Filename: "{app}\AstroForge.exe"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\.fits\OpenWithProgids"; ValueType: string; ValueName: "AstroForge.Image"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assocfits
Root: HKA; Subkey: "Software\Classes\.xisf\OpenWithProgids"; ValueType: string; ValueName: "AstroForge.Image"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assocfits
Root: HKA; Subkey: "Software\Classes\.ser\OpenWithProgids"; ValueType: string; ValueName: "AstroForge.Image"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assocfits
Root: HKA; Subkey: "Software\Classes\AstroForge.Image"; ValueType: string; ValueName: ""; ValueData: "Image astronomique"; Flags: uninsdeletekey; Tasks: assocfits
Root: HKA; Subkey: "Software\Classes\AstroForge.Image\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\AstroForge.exe"" ""%1"""; Tasks: assocfits

[Run]
Filename: "{app}\AstroForge.exe"; Description: "{cm:LaunchProgram,AstroForge}"; Flags: nowait postinstall skipifsilent
