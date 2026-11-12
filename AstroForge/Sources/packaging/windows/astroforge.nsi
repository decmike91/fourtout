; Installeur Windows d'AstroForge (NSIS 3) — produit par build_windows.sh :
;   makensis -DVERSION=2.0.0 -DSTAGE=<dossier préparé> -DOUTFILE=<setup.exe> astroforge.nsi
;
; Contenu installé : lanceurs AstroForge.exe / astroforge-cli.exe, Python
; embarqué (python\) avec AstroForge et ses dépendances — aucun Python à
; installer au préalable.
; Installation pour tous les utilisateurs (Program Files, droits
; administrateur) ou pour l'utilisateur courant seulement (sans droits).
; Ligne de commande : /S (silencieux), /AllUsers ou /CurrentUser, /D=<dossier>.

Unicode true
ManifestDPIAware true
SetCompressor /SOLID lzma
SetCompressorDictSize 64

!ifndef VERSION
  !define VERSION "0.0.0"
!endif
!ifndef STAGE
  !error "Définir -DSTAGE=<dossier préparé>"
!endif
!ifndef OUTFILE
  !define OUTFILE "AstroForge-${VERSION}-windows-setup.exe"
!endif

!define APPNAME "AstroForge"
!define PUBLISHER "Contributeurs AstroForge"
!define UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\AstroForge"
!define PROGID "AstroForge.Image"

Name "${APPNAME} ${VERSION}"
OutFile "${OUTFILE}"
BrandingText "${APPNAME} ${VERSION}"

; ---------------------------------------------------------------- multi-utilisateur
!define MULTIUSER_EXECUTIONLEVEL Highest
!define MULTIUSER_MUI
!define MULTIUSER_INSTALLMODE_COMMANDLINE
!define MULTIUSER_USE_PROGRAMFILES64
!define MULTIUSER_INSTALLMODE_INSTDIR "AstroForge"
!define MULTIUSER_INSTALLMODE_INSTDIR_REGISTRY_KEY "${UNINST_KEY}"
!define MULTIUSER_INSTALLMODE_INSTDIR_REGISTRY_VALUENAME "InstallLocation"
!define MULTIUSER_INSTALLMODE_DEFAULT_REGISTRY_KEY "${UNINST_KEY}"
!define MULTIUSER_INSTALLMODE_DEFAULT_REGISTRY_VALUENAME "InstallMode"
!include MultiUser.nsh
!include MUI2.nsh
!include LogicLib.nsh
!include FileFunc.nsh
!include x64.nsh
!include WinVer.nsh

; ---------------------------------------------------------------- interface
!define MUI_ICON "astroforge.ico"
!define MUI_UNICON "astroforge.ico"
!define MUI_ABORTWARNING
!define MUI_FINISHPAGE_RUN "$INSTDIR\AstroForge.exe"
!define MUI_FINISHPAGE_RUN_TEXT "$(RunText)"
!define MUI_COMPONENTSPAGE_SMALLDESC

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "..\..\LICENSE"
!insertmacro MULTIUSER_PAGE_INSTALLMODE
!insertmacro MUI_PAGE_COMPONENTS
!define MUI_DIRECTORYPAGE_TEXT_TOP "$(DirHint)"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "French"
!insertmacro MUI_LANGUAGE "English"

LangString RunText ${LANG_FRENCH} "Lancer AstroForge"
LangString RunText ${LANG_ENGLISH} "Run AstroForge"
LangString SecMain ${LANG_FRENCH} "AstroForge (requis)"
LangString SecMain ${LANG_ENGLISH} "AstroForge (required)"
LangString SecDesk ${LANG_FRENCH} "Raccourci sur le bureau"
LangString SecDesk ${LANG_ENGLISH} "Desktop shortcut"
LangString SecAssoc ${LANG_FRENCH} "Proposer AstroForge pour ouvrir les .fits / .xisf"
LangString SecAssoc ${LANG_ENGLISH} "Offer AstroForge to open .fits / .xisf files"
LangString DescMain ${LANG_FRENCH} "Programme, Python embarqué et dépendances (aucune installation de Python nécessaire)."
LangString DescMain ${LANG_ENGLISH} "Program, embedded Python and dependencies (no Python installation needed)."
LangString DescDesk ${LANG_FRENCH} "Ajoute une icône AstroForge sur le bureau."
LangString DescDesk ${LANG_ENGLISH} "Adds an AstroForge icon on the desktop."
LangString DescAssoc ${LANG_FRENCH} "Ajoute AstroForge au menu « Ouvrir avec » des images FITS et XISF (sans changer le programme par défaut)."
LangString DescAssoc ${LANG_ENGLISH} "Adds AstroForge to the 'Open with' menu of FITS and XISF images (default program unchanged)."
LangString CliName ${LANG_FRENCH} "AstroForge (ligne de commande)"
LangString CliName ${LANG_ENGLISH} "AstroForge (command line)"
LangString NeedWin10 ${LANG_FRENCH} "AstroForge nécessite Windows 10 (version 1903) ou plus récent, 64 bits."
LangString NeedWin10 ${LANG_ENGLISH} "AstroForge requires 64-bit Windows 10 (version 1903) or later."
LangString AskData ${LANG_FRENCH} "Supprimer aussi vos réglages, profils, bibliothèque de masters et modèles IA ($APPDATA\AstroForge) ?"
LangString AskData ${LANG_ENGLISH} "Also delete your settings, profiles, master library and AI models ($APPDATA\AstroForge)?"
LangString AppRunning ${LANG_FRENCH} "AstroForge est en cours d'exécution. Fermez-le puis cliquez sur OK."
LangString AppRunning ${LANG_ENGLISH} "AstroForge is running. Close it, then click OK."
; Windows limite un chemin complet à 260 caractères ; les fichiers les plus
; profonds d'AstroForge en occupent jusqu'à 140 dans le dossier d'installation.
!define MAX_INSTDIR_LEN 110
LangString DirHint ${LANG_FRENCH} "Choisissez un dossier dont le chemin fait au plus ${MAX_INSTDIR_LEN} caractères (limite de longueur des chemins de Windows)."
LangString DirHint ${LANG_ENGLISH} "Choose a folder whose path is at most ${MAX_INSTDIR_LEN} characters long (Windows path length limit)."
LangString DirTooLong ${LANG_FRENCH} "Le chemin du dossier d'installation est trop long (plus de ${MAX_INSTDIR_LEN} caractères) : Windows ne pourrait pas créer certains fichiers. Choisissez un dossier plus court, par exemple C:\AstroForge."
LangString DirTooLong ${LANG_ENGLISH} "The installation folder path is too long (over ${MAX_INSTDIR_LEN} characters): Windows could not create some files. Choose a shorter folder, e.g. C:\AstroForge."

VIProductVersion "${VERSION}.0"
VIAddVersionKey /LANG=${LANG_FRENCH} "ProductName" "${APPNAME}"
VIAddVersionKey /LANG=${LANG_FRENCH} "CompanyName" "${PUBLISHER}"
VIAddVersionKey /LANG=${LANG_FRENCH} "FileDescription" "Installation d'AstroForge"
VIAddVersionKey /LANG=${LANG_FRENCH} "FileVersion" "${VERSION}"
VIAddVersionKey /LANG=${LANG_FRENCH} "ProductVersion" "${VERSION}"
VIAddVersionKey /LANG=${LANG_FRENCH} "LegalCopyright" "Licence MIT"

; ---------------------------------------------------------------- utilitaires
; Attend la fermeture d'AstroForge (fichiers verrouillés) : tente de
; renommer pythonw.exe sur lui-même (impossible tant qu'il s'exécute).
!macro DefWaitAppClosed PREFIX
Function ${PREFIX}WaitAppClosed
  StrCpy $R9 0
  ${If} ${FileExists} "$INSTDIR\python\pythonw.exe"
    retry:
    ClearErrors
    Rename "$INSTDIR\python\pythonw.exe" "$INSTDIR\python\pythonw.exe.lock"
    ${If} ${Errors}
      IntOp $R9 $R9 + 1
      ${If} ${Silent}
        IntCmp $R9 20 0 +2 0
        Abort
        Sleep 3000
        Goto retry
      ${EndIf}
      MessageBox MB_OKCANCEL|MB_ICONEXCLAMATION "$(AppRunning)" IDOK retry
      Abort
    ${EndIf}
    Rename "$INSTDIR\python\pythonw.exe.lock" "$INSTDIR\python\pythonw.exe"
  ${EndIf}
FunctionEnd
!macroend
!insertmacro DefWaitAppClosed ""
!insertmacro DefWaitAppClosed "un."

Function .onVerifyInstDir
  StrLen $0 "$INSTDIR"
  IntCmp $0 ${MAX_INSTDIR_LEN} ok ok 0
  Abort          ; bouton Suivant désactivé tant que le chemin est trop long
  ok:
FunctionEnd

Function .onInit
  ${IfNot} ${RunningX64}
  ${OrIfNot} ${AtLeastWin10}
    MessageBox MB_OK|MB_ICONSTOP "$(NeedWin10)" /SD IDOK
    Abort
  ${EndIf}
  SetRegView 64
  ; /D=<dossier> (ligne de commande) : MultiUser.nsh écraserait $INSTDIR
  StrCpy $R8 "$INSTDIR"
  !insertmacro MULTIUSER_INIT
  StrCmp $R8 "" +2
  StrCpy $INSTDIR "$R8"
FunctionEnd

Function un.onInit
  SetRegView 64
  !insertmacro MULTIUSER_UNINIT
FunctionEnd

; ---------------------------------------------------------------- sections
Section "!$(SecMain)" SEC_MAIN
  SectionIn RO
  StrLen $0 "$INSTDIR"
  ${If} $0 > ${MAX_INSTDIR_LEN}
    MessageBox MB_OK|MB_ICONSTOP "$(DirTooLong)" /SD IDOK
    Abort
  ${EndIf}
  SetOutPath "$INSTDIR"
  Call WaitAppClosed
  ; mise à jour : on repart d'un Python propre (pas de fichiers périmés)
  RMDir /r "$INSTDIR\python"
  File "${STAGE}\AstroForge.exe"
  File "${STAGE}\astroforge-cli.exe"
  File "/oname=LICENSE.txt" "..\..\LICENSE"
  File "check_install.py"
  File /r "${STAGE}\python"

  ; menu Démarrer
  CreateDirectory "$SMPROGRAMS\AstroForge"
  CreateShortcut "$SMPROGRAMS\AstroForge\AstroForge.lnk" "$INSTDIR\AstroForge.exe" "" "$INSTDIR\AstroForge.exe" 0
  CreateShortcut "$SMPROGRAMS\AstroForge\$(CliName).lnk" "$SYSDIR\cmd.exe" \
    '/K "cd /d "%USERPROFILE%" && "$INSTDIR\astroforge-cli.exe" --help"' "$INSTDIR\astroforge-cli.exe" 0
  CreateShortcut "$SMPROGRAMS\AstroForge\Désinstaller AstroForge.lnk" "$INSTDIR\Uninstall.exe"

  ; désinstallation (Paramètres › Applications)
  WriteUninstaller "$INSTDIR\Uninstall.exe"
  WriteRegStr SHCTX "${UNINST_KEY}" "DisplayName" "${APPNAME} ${VERSION}"
  WriteRegStr SHCTX "${UNINST_KEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr SHCTX "${UNINST_KEY}" "Publisher" "${PUBLISHER}"
  WriteRegStr SHCTX "${UNINST_KEY}" "DisplayIcon" "$INSTDIR\AstroForge.exe,0"
  WriteRegStr SHCTX "${UNINST_KEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr SHCTX "${UNINST_KEY}" "InstallMode" "$MultiUser.InstallMode"
  WriteRegStr SHCTX "${UNINST_KEY}" "UninstallString" '"$INSTDIR\Uninstall.exe" /$MultiUser.InstallMode'
  WriteRegStr SHCTX "${UNINST_KEY}" "QuietUninstallString" '"$INSTDIR\Uninstall.exe" /$MultiUser.InstallMode /S'
  WriteRegDWORD SHCTX "${UNINST_KEY}" "NoModify" 1
  WriteRegDWORD SHCTX "${UNINST_KEY}" "NoRepair" 1
  ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
  IntFmt $0 "0x%08X" $0
  WriteRegDWORD SHCTX "${UNINST_KEY}" "EstimatedSize" "$0"
SectionEnd

Section "$(SecDesk)" SEC_DESK
  CreateShortcut "$DESKTOP\AstroForge.lnk" "$INSTDIR\AstroForge.exe" "" "$INSTDIR\AstroForge.exe" 0
SectionEnd

!macro _AF_OpenWith EXT
  WriteRegStr SHCTX "Software\Classes\${EXT}\OpenWithProgids" "${PROGID}" ""
!macroend

Section "$(SecAssoc)" SEC_ASSOC
  WriteRegStr SHCTX "Software\Classes\${PROGID}" "" "Image astronomique (AstroForge)"
  WriteRegStr SHCTX "Software\Classes\${PROGID}\DefaultIcon" "" "$INSTDIR\AstroForge.exe,0"
  WriteRegStr SHCTX "Software\Classes\${PROGID}\shell\open\command" "" '"$INSTDIR\AstroForge.exe" "%1"'
  !insertmacro _AF_OpenWith ".fits"
  !insertmacro _AF_OpenWith ".fit"
  !insertmacro _AF_OpenWith ".fts"
  !insertmacro _AF_OpenWith ".xisf"
  WriteRegStr SHCTX "Software\Classes\Applications\AstroForge.exe\shell\open\command" "" '"$INSTDIR\AstroForge.exe" "%1"'
  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'
SectionEnd

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_MAIN} "$(DescMain)"
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_DESK} "$(DescDesk)"
  !insertmacro MUI_DESCRIPTION_TEXT ${SEC_ASSOC} "$(DescAssoc)"
!insertmacro MUI_FUNCTION_DESCRIPTION_END

; ---------------------------------------------------------------- désinstallation
Section "Uninstall"
  Call un.WaitAppClosed
  Delete "$DESKTOP\AstroForge.lnk"
  RMDir /r "$SMPROGRAMS\AstroForge"
  DeleteRegKey SHCTX "Software\Classes\${PROGID}"
  DeleteRegKey SHCTX "Software\Classes\Applications\AstroForge.exe"
  DeleteRegValue SHCTX "Software\Classes\.fits\OpenWithProgids" "${PROGID}"
  DeleteRegValue SHCTX "Software\Classes\.fit\OpenWithProgids" "${PROGID}"
  DeleteRegValue SHCTX "Software\Classes\.fts\OpenWithProgids" "${PROGID}"
  DeleteRegValue SHCTX "Software\Classes\.xisf\OpenWithProgids" "${PROGID}"
  DeleteRegKey SHCTX "${UNINST_KEY}"
  System::Call 'shell32::SHChangeNotify(i 0x08000000, i 0, p 0, p 0)'

  RMDir /r "$INSTDIR\python"
  Delete "$INSTDIR\AstroForge.exe"
  Delete "$INSTDIR\astroforge-cli.exe"
  Delete "$INSTDIR\LICENSE.txt"
  Delete "$INSTDIR\check_install.py"
  Delete "$INSTDIR\Uninstall.exe"
  RMDir "$INSTDIR"

  ; données de l'utilisateur : conservées par défaut
  ${IfNot} ${Silent}
    MessageBox MB_YESNO|MB_ICONQUESTION|MB_DEFBUTTON2 "$(AskData)" IDNO keep_data
    SetShellVarContext current
    RMDir /r "$APPDATA\AstroForge"
    keep_data:
  ${EndIf}
SectionEnd
