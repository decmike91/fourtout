/*
 * Lanceur Windows d'AstroForge (AstroForge.exe / astroforge-cli.exe).
 *
 * Démarre l'interpréteur Python embarqué (<dossier>\python\) sur le
 * paquet astroforge en transmettant les arguments (fichier à ouvrir,
 * options de la ligne de commande) et renvoie son code de sortie.
 *   - GUI (AstroForge.exe)        : pythonw.exe -m astroforge.gui.app ...
 *   - CONSOLE (astroforge-cli.exe): python.exe  -m astroforge ...
 * Compilation (MinGW-w64) : voir build.sh.
 */
#define WIN32_LEAN_AND_MEAN
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#include <windows.h>
#include <shellapi.h>
#include <stdio.h>
#include <wchar.h>

#ifdef ASTROFORGE_CONSOLE
#define PY_EXE L"python.exe"
#define PY_ARGS L"-m astroforge"
#else
#define PY_EXE L"pythonw.exe"
#define PY_ARGS L"-m astroforge.gui.app"
#endif

static void fail(const wchar_t *msg)
{
#ifdef ASTROFORGE_CONSOLE
    fwprintf(stderr, L"AstroForge : %ls\n", msg);
#else
    MessageBoxW(NULL, msg, L"AstroForge", MB_ICONERROR | MB_OK);
#endif
}

/* Ajoute un argument à la ligne de commande avec les règles de
 * guillemets de CommandLineToArgvW (barres obliques inverses doublées
 * devant un guillemet). */
static void append_arg(wchar_t *dst, size_t cap, const wchar_t *arg)
{
    size_t n = wcslen(dst);
    if (n + 4 >= cap) return;
    dst[n++] = L' ';
    dst[n++] = L'"';
    for (const wchar_t *p = arg; *p && n + 4 < cap; ++p) {
        size_t bs = 0;
        while (*p == L'\\') { ++bs; ++p; }
        if (*p == 0) {               /* barres en fin : doublées avant le " final */
            for (size_t i = 0; i < bs * 2 && n + 3 < cap; ++i) dst[n++] = L'\\';
            break;
        }
        if (*p == L'"') {
            for (size_t i = 0; i < bs * 2 + 1 && n + 3 < cap; ++i) dst[n++] = L'\\';
        } else {
            for (size_t i = 0; i < bs && n + 3 < cap; ++i) dst[n++] = L'\\';
        }
        dst[n++] = *p;
    }
    dst[n++] = L'"';
    dst[n] = 0;
}

int wmain_impl(void)
{
    wchar_t self[MAX_PATH * 4], dir[MAX_PATH * 4], py[MAX_PATH * 4];
    DWORD len = GetModuleFileNameW(NULL, self, (DWORD)(sizeof(self) / sizeof(self[0])));
    if (len == 0 || len >= sizeof(self) / sizeof(self[0])) {
        fail(L"Chemin du programme introuvable.");
        return 1;
    }
    wcscpy(dir, self);
    wchar_t *slash = wcsrchr(dir, L'\\');
    if (slash) *slash = 0;
    _snwprintf(py, sizeof(py) / sizeof(py[0]), L"%ls\\python\\%ls", dir, PY_EXE);
    py[sizeof(py) / sizeof(py[0]) - 1] = 0;
    if (GetFileAttributesW(py) == INVALID_FILE_ATTRIBUTES) {
        fail(L"Python embarqué introuvable (dossier « python » à côté du programme). "
             L"Réinstallez AstroForge.");
        return 1;
    }

    size_t cap = 32768;
    wchar_t *cmd = (wchar_t *)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, cap * sizeof(wchar_t));
    if (!cmd) return 1;
    /* -I : mode isolé (ni dossier courant, ni variables PYTHON*, ni paquets
     * de l'utilisateur dans sys.path) ; -X utf8 : E/S en UTF-8 */
    _snwprintf(cmd, cap, L"\"%ls\" -I -X utf8 %ls", py, PY_ARGS);

    int argc = 0;
    wchar_t **argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    for (int i = 1; argv && i < argc; ++i) append_arg(cmd, cap, argv[i]);
    if (argv) LocalFree(argv);

    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));
    /* dossier courant hérité : les chemins relatifs restent valides */
    if (!CreateProcessW(py, cmd, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
        fail(L"Impossible de démarrer Python.");
        HeapFree(GetProcessHeap(), 0, cmd);
        return 1;
    }
    HeapFree(GetProcessHeap(), 0, cmd);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return (int)code;
}

#ifdef ASTROFORGE_CONSOLE
int wmain(void) { return wmain_impl(); }
#else
int WINAPI wWinMain(HINSTANCE h, HINSTANCE p, LPWSTR c, int s)
{
    (void)h; (void)p; (void)c; (void)s;
    return wmain_impl();
}
#endif
