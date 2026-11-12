Unicode true
SetCompressor /SOLID lzma
!include "MUI2.nsh"

!define APP "Trait Fin"
!define VER "1.0.0"
!define UNKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\TraitFin"

Name "${APP}"
OutFile "dist\Trait-Fin-Setup-${VER}.exe"
InstallDir "$LOCALAPPDATA\Programs\Trait Fin"
InstallDirRegKey HKCU "Software\TraitFin" "InstallDir"
RequestExecutionLevel user
BrandingText "${APP} ${VER}"

VIProductVersion "1.0.0.0"
VIAddVersionKey /LANG=1036 "ProductName" "${APP}"
VIAddVersionKey /LANG=1036 "FileDescription" "Installation de ${APP}"
VIAddVersionKey /LANG=1036 "FileVersion" "${VER}"
VIAddVersionKey /LANG=1036 "ProductVersion" "${VER}"
VIAddVersionKey /LANG=1036 "LegalCopyright" ""

!define MUI_ICON "icon.ico"
!define MUI_UNICON "icon.ico"
!define MUI_ABORTWARNING
!define MUI_FINISHPAGE_RUN "$INSTDIR\Trait Fin.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Lancer Trait Fin"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "French"

Section "Install"
  SetOutPath "$INSTDIR"
  File /r "dist\win-unpacked\*.*"
  File "icon.ico"
  WriteUninstaller "$INSTDIR\Desinstaller.exe"
  CreateShortcut "$SMPROGRAMS\Trait Fin.lnk" "$INSTDIR\Trait Fin.exe" "" "$INSTDIR\icon.ico"
  CreateShortcut "$DESKTOP\Trait Fin.lnk" "$INSTDIR\Trait Fin.exe" "" "$INSTDIR\icon.ico"
  WriteRegStr HKCU "Software\TraitFin" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "${UNKEY}" "DisplayName" "${APP}"
  WriteRegStr HKCU "${UNKEY}" "DisplayVersion" "${VER}"
  WriteRegStr HKCU "${UNKEY}" "Publisher" "${APP}"
  WriteRegStr HKCU "${UNKEY}" "DisplayIcon" "$INSTDIR\icon.ico"
  WriteRegStr HKCU "${UNKEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNKEY}" "UninstallString" '"$INSTDIR\Desinstaller.exe"'
  WriteRegDWORD HKCU "${UNKEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNKEY}" "NoRepair" 1
SectionEnd

Section "Uninstall"
  Delete "$SMPROGRAMS\Trait Fin.lnk"
  Delete "$DESKTOP\Trait Fin.lnk"
  RMDir /r "$INSTDIR"
  DeleteRegKey HKCU "${UNKEY}"
  DeleteRegKey HKCU "Software\TraitFin"
SectionEnd
