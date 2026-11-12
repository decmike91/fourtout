# -*- coding: utf-8 -*-
"""
E/S images : FITS, TIFF, PNG, JPEG, RAW (APN), vidéo SER.
Toutes les images sont converties en float32 [0..1] (mono HxW ou couleur HxWx3 RGB).
"""
from __future__ import annotations
import os
import struct
from collections.abc import Sequence

import numpy as np

BAYER_PATTERNS = ["RGGB", "BGGR", "GRBG", "GBRG"]

try:
    from astropy.io import fits as _pfits
except Exception:
    _pfits = None

try:
    import rawpy  # optionnel :lecture des RAW d'appareils photo
except Exception:
    rawpy = None

IMG_EXTENSIONS = {".fits", ".fit", ".fts", ".fz", ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".xisf"}
RAW_EXTENSIONS = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw"}


# ---------------------------------------------------------------- conversions
def _normalize_float(a: np.ndarray) -> np.ndarray:
    """Ramène des données flottantes dans [0..1] : déjà normalisées (max ≤ 1,5)
    → simple écrêtage ; valeurs ADU 16 bits (≤ 65535) → ÷ 65535 ; au-delà →
    ÷ maximum. Les NaN/inf deviennent 0."""
    a = np.nan_to_num(np.asarray(a, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    m = float(a.max(initial=0))
    if m > 1.5:
        a = a / np.float32(65535.0 if m <= 65535.0 * 1.001 else m)
    return np.clip(a, 0, 1).astype(np.float32)


def _normalize_int(a: np.ndarray) -> np.ndarray:
    """Entiers → float32 [0..1] selon la dynamique du type. Les entiers plus
    larges que 16 bits contenant des valeurs ≤ 65535 (données 16 bits
    stockées en 32 bits, fréquent en FITS) sont divisés par 65535."""
    a = np.asarray(a)
    info = np.iinfo(a.dtype)
    scale = float(info.max)
    if a.dtype.itemsize > 2 and float(a.max(initial=0)) <= 65535:
        scale = 65535.0
    return np.clip(a.astype(np.float32) / np.float32(scale), 0, 1).astype(np.float32)


def to_float(img: np.ndarray) -> np.ndarray:
    """Convertit n'importe quel dtype en float32 [0..1]."""
    a = np.asarray(img)
    if a.dtype.kind == "f":
        return _normalize_float(a)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535.0
    if a.dtype.kind in "iu":
        return _normalize_int(a)
    if a.dtype == np.bool_:
        return a.astype(np.float32)
    raise TypeError(f"Type de données non pris en charge : {a.dtype}")


def to_uint16(img: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(img, np.float64), 0, 1).__mul__(65535.0).astype(np.uint16)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(img, np.float64), 0, 1).__mul__(255.0).astype(np.uint8)


def is_color(img: np.ndarray) -> bool:
    return img.ndim == 3 and img.shape[2] >= 3


# ---------------------------------------------------------------- chargement
def load_image(path: str, bayer: str | None = None) -> np.ndarray:
    """Charge une image (FITS/TIFF/PNG/JPEG/RAW). Débayerise si `bayer` est fourni
    et que l'image mono vient d'un capteur couleur."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".fits", ".fit", ".fts", ".fz"):
        img, header = read_fits(path)
        if bayer is not None and img.ndim == 2:
            img = debayer(img, bayer)
        return img
    if ext in RAW_EXTENSIONS:
        cfa, pattern = _read_raw(path)
        return debayer(cfa, bayer or pattern)  # motif forcé ou celui du capteur
    if ext == ".xisf":
        from .xisf import read_xisf
        return read_xisf(path)
    # TIFF / PNG / JPEG / BMP via OpenCV (16 bits conservés). Lecture par
    # imdecode : cv2.imread échoue sous Windows sur les chemins accentués.
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError as e:
        raise IOError(f"Impossible de lire : {path} ({e})") from e
    data = _cv2().imdecode(buf, _cv2().IMREAD_UNCHANGED) if buf.size else None
    if data is None:
        raise IOError(f"Impossible de lire : {path}")
    if data.ndim == 3 and data.shape[2] >= 3:  # BGR -> RGB
        data = data[:, :, :3][:, :, ::-1]
    img = to_float(data)
    if bayer is not None and not is_color(img) and img.ndim == 2:
        img = debayer(img, bayer)
    return img


def _fits_image_hdu(hdul):
    """Première HDU contenant une image (primaire, extension ou FITS
    compressé .fz)."""
    for hdu in hdul:
        data = getattr(hdu, "data", None)
        if data is not None and np.ndim(data) >= 2:
            return hdu
    return None


def read_fits(path: str) -> tuple[np.ndarray, dict]:
    """Lit un FITS -> (float32 [0..1] HxW ou HxWx3, en-tête sous forme de dict).
    memmap=False : obligatoire pour les FITS 16 bits non signés (BZERO = 32768),
    format de toutes les caméras astro. Les entiers sont normalisés selon leur
    dynamique (uint16 ÷ 65535), les flottants ADU ramenés dans [0..1]."""
    if _pfits is None:
        raise RuntimeError("astropy est requis pour les FITS : pip install astropy")
    with _pfits.open(path, memmap=False) as hdul:
        hdu = _fits_image_hdu(hdul)
        if hdu is None:
            raise ValueError(f"FITS sans données image : {path}")
        raw = np.asarray(hdu.data)
        header = {k: hdu.header[k] for k in hdu.header.keys()
                  if k and k not in ("COMMENT", "HISTORY")}
    d = to_float(raw)
    if d.ndim == 3:  # plans FITS (C,H,W) -> canaux (H,W,C)
        if d.shape[0] in (1, 3) and d.shape[0] < d.shape[2]:
            d = np.moveaxis(d, 0, -1)
        elif d.shape[-1] not in (1, 3):
            d = d[0] if d.shape[0] == 1 else d.transpose(1, 2, 0)
    if d.ndim == 3 and d.shape[2] == 1:
        d = d[:, :, 0]
    return np.ascontiguousarray(d, dtype=np.float32), header


def load_fits(path: str) -> np.ndarray:
    return read_fits(path)[0]


def fits_bayer_pattern(header: dict) -> str | None:
    """Motif de Bayer déclaré dans l'en-tête FITS (BAYERPAT / COLORTYP)."""
    for key in ("BAYERPAT", "COLORTYP", "CFA-PAT"):
        v = str(header.get(key, "")).strip().upper().replace(" ", "")
        if v in BAYER_PATTERNS:
            return v
    return None


def _read_raw(path: str) -> tuple[np.ndarray, str]:
    """RAW APN non dématricé -> (CFA float32 [0..1], motif de Bayer réel de la
    zone visible)."""
    if rawpy is None:
        raise RuntimeError("rawpy est requis pour les RAW APN : pip install rawpy")
    with rawpy.imread(path) as raw:
        data = np.array(raw.raw_image_visible, dtype=np.float32)
        white = float(getattr(raw, "white_level", 0) or 0) or float(data.max(initial=1))
        pattern = None
        try:
            desc = raw.color_desc.decode("ascii", "replace")
            cols = np.asarray(raw.raw_colors_visible)[:2, :2].ravel()
            pattern = "".join(desc[int(c)] for c in cols).replace("G2", "G")
            pattern = "".join("G" if ch not in "RB" else ch for ch in pattern)
        except Exception:
            pattern = _rawpattern_to_str(getattr(raw, "raw_pattern", None))
    img = np.clip(data / np.float32(max(white, 1.0)), 0, 1).astype(np.float32)
    if pattern not in BAYER_PATTERNS:
        pattern = "RGGB"
    return img, pattern


def load_raw(path: str, bayer: str | None = None, cfa: bool = False) -> np.ndarray:
    """RAW APN dématricé (motif `bayer` forcé, sinon motif réel du capteur) ;
    cfa=True retourne le CFA brut non dématricé (pour la calibration)."""
    img, pattern = _read_raw(path)
    return img if cfa else debayer(img, bayer or pattern)


def load_cfa(path: str) -> tuple[np.ndarray, str | None]:
    """Charge une image SANS la dématricer, pour la calibrer en CFA :
    retourne (image, motif de Bayer détecté ou None). RAW : CFA + motif du
    capteur ; FITS : données + BAYERPAT éventuel ; autres : image telle quelle."""
    ext = os.path.splitext(path)[1].lower()
    if ext in RAW_EXTENSIONS:
        return _read_raw(path)
    if ext in (".fits", ".fit", ".fts", ".fz"):
        img, header = read_fits(path)
        return img, (fits_bayer_pattern(header) if img.ndim == 2 else None)
    return load_image(path), None


def _rawpattern_to_str(p) -> str | None:
    if p is None:
        return None
    m = {0: "R", 1: "G", 2: "B"}
    try:
        return "".join(m[c] for c in p.ravel()[:4])
    except Exception:
        return None


def _cv2():
    import cv2
    return cv2


# ---------------------------------------------------------------- débayerisation
def debayer(cfa: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Débayerise une image CFA mono (float [0..1]) selon la matrice de Bayer choisie."""
    cv2 = _cv2()
    p = pattern.upper().replace(" ", "")
    if p not in BAYER_PATTERNS:
        raise ValueError(f"Matrice de Bayer inconnue : {pattern} (choix: {BAYER_PATTERNS})")
    # Convention OpenCV : le code nomme les 2e et 3e pixels de la 2e ligne du
    # motif (et non le coin haut-gauche). Capteur RGGB -> BayerBG, BGGR ->
    # BayerRG, GRBG -> BayerGB, GBRG -> BayerGR (vérifié sur OpenCV 4 et 5).
    code_map = {
        "RGGB": cv2.COLOR_BayerBG2RGB, "BGGR": cv2.COLOR_BayerRG2RGB,
        "GRBG": cv2.COLOR_BayerGB2RGB, "GBRG": cv2.COLOR_BayerGR2RGB,
    }
    u16 = to_uint16(cfa)
    rgb = cv2.cvtColor(u16, code_map[p])
    return to_float(rgb)


# ---------------------------------------------------------------- sauvegarde
def _imwrite(path: str, arr: np.ndarray, params=None) -> bool:
    """cv2.imwrite compatible chemins non ASCII (Windows) : encodage en
    mémoire puis écriture binaire."""
    ext = os.path.splitext(path)[1].lower() or ".png"
    ok, buf = _cv2().imencode(ext, arr, params or [])
    if not ok:
        raise IOError(f"Encodage impossible : {path}")
    buf.tofile(path)
    return True


def save_image(path: str, img: np.ndarray, bit_depth: str = "auto",
               header: dict | None = None) -> np.ndarray:
    """Sauve en FITS (float32, en-tête `header` + WCS éventuel) / XISF /
    TIFF (16 bits, ou 32 bits flottant avec bit_depth="32f") / PNG / JPEG.
    Retourne les données écrites."""
    ext = os.path.splitext(path)[1].lower()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if ext in (".fits", ".fit", ".fts"):
        if _pfits is None:
            raise RuntimeError("astropy requis pour écrire du FITS")
        out = np.clip(img, 0, 1).astype(np.float32) if bit_depth in ("auto", "float32") else img
        if out.ndim == 3:
            out = np.moveaxis(out, -1, 0)  # (3,H,W)
        hdr = _pfits.Header()
        for k, v in (header or {}).items():
            if v is None or k.upper() in ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2",
                                          "NAXIS3", "EXTEND", "BZERO", "BSCALE"):
                continue
            try:
                hdr[k[:8].upper()] = v
            except (ValueError, KeyError):
                continue
        _pfits.writeto(path, out, hdr, overwrite=True)
        return out
    if ext == ".xisf":
        from .xisf import write_xisf
        out = np.clip(img, 0, 1).astype(np.float32)
        write_xisf(path, out, header=header)
        return out
    if ext in (".tif", ".tiff"):
        if bit_depth in ("32f", "float32"):
            arr = np.ascontiguousarray(np.clip(img, 0, 1).astype(np.float32))
            if arr.ndim == 3:
                arr = arr[:, :, ::-1]
            return _imwrite(path, arr)
        arr = to_uint16(img) if bit_depth in ("auto", "16") else to_uint8(img)
        if arr.ndim == 3:
            arr = arr[:, :, ::-1]  # RGB -> BGR
        return _imwrite(path, arr)
    if ext == ".png":
        arr = to_uint16(img) if bit_depth == "16" else to_uint8(img)
        if arr.ndim == 3:
            arr = arr[:, :, ::-1]
        return _imwrite(path, arr)
    # JPEG / BMP : 8 bits
    arr = to_uint8(img)
    if arr.ndim == 3:
        arr = arr[:, :, ::-1]
    return _imwrite(path, arr, [int(_cv2().IMWRITE_JPEG_QUALITY), 96])


# ---------------------------------------------------------------- SER (vidéo planétaire)
def _ser_read_strings(f, count: int):
    """Chaînes préfixées par leur longueur (norme SER v3). None si une
    longueur est absurde (dialecte sans préfixes ou fichier corrompu)."""
    out = []
    for _ in range(count):
        (n,) = struct.unpack("<i", f.read(4))
        if n < 0 or n > 4096:
            return None
        out.append(f.read(n).decode("ascii", "replace").rstrip("\x00 "))
    return out


def _ser_dtype(depth: int):
    """Champ PixelDepthPerPlane : en bits selon la spec (1..8 → 1 octet,
    9..16 → 2 octets ; 12 bits = valeurs 0..4095), mais en octets (1/2/4)
    chez certains logiciels → accepter les deux. Retourne (dtype
    little-endian, échelle [0..1]) ou None."""
    if depth in (1, 8):
        return np.dtype("<u1"), 255.0
    if depth in (2, 16):
        return np.dtype("<u2"), 65535.0
    if depth in (4, 32):
        return np.dtype("<f4"), 1.0
    if 3 <= depth <= 7:
        return np.dtype("<u1"), float((1 << depth) - 1)
    if 9 <= depth <= 15:
        return np.dtype("<u2"), float((1 << depth) - 1)
    return None


# ColorID de la spécification SER v3 (G. Hahn) : 0 mono, 8-11 Bayer
# (RGGB, GRBG, GBRG, BGGR), 16-19 CYYM…, 100 RGB, 101 BGR.
SER_BAYER = {8: "RGGB", 9: "GRBG", 10: "GBRG", 11: "BGGR"}


def _ser_variants(colormode: int, layout: str):
    """Interprétations possibles du ColorID -> [(planes, bayer, bgr, rang)].
    La norme d'abord ; puis les conventions des anciennes versions
    d'AstroForge (8 = BGR 3 plans, 16 = RGB 3 plans), départagées par la
    taille du fichier."""
    out = []
    if colormode == 0 or 16 <= colormode <= 19:
        out.append((1, None, False, 0))
    elif colormode in SER_BAYER:
        out.append((1, SER_BAYER[colormode], False, 0))
    elif colormode == 100:
        out.append((3, None, False, 0))
    elif colormode == 101:
        out.append((3, None, True, 0))
    if colormode == 8:
        out.append((3, None, True, 1))
    if colormode == 16:
        out.append((3, None, False, 1))
    return out


def _ser_candidate(f, layout: str):
    """Tente de lire l'en-tête selon la disposition indiquée :
    - « spec » : norme SER v3 — 3 champs FIXES de 40 octets (Observer,
      Instrument, Telescope) puis 2 dates de 8 octets : en-tête de 178 octets
      (SharpCap, FireCapture, ASIStudio, Siril, AutoStakkert!…) ;
    - « v3 » : chaînes préfixées par leur longueur (fichiers écrits par
      AstroForge ≤ 2.x) ;
    - « legacy » : ancien dialecte AstroForge ≤ 1.7.
    Retourne une liste d'interprétations cohérentes (vide si aucune)."""
    try:
        legacy_bayer = None
        if layout == "legacy":
            strs = _ser_read_strings(f, 4)
            if strs is None:
                return []
            w, h, nframes, depth, colormode = struct.unpack("<iiiii", f.read(20))
            camera, obs, inst, tel = strs[0], strs[1], strs[2], strs[3]
            if colormode == 100:
                legacy_bayer = f.read(32).decode("ascii", "replace").rstrip("\x00 ")
        else:
            _lu, color_id, _end = struct.unpack("<iii", f.read(12))
            w, h, depth, nframes = struct.unpack("<iiii", f.read(16))
            colormode = color_id
            if layout == "spec":
                raw = f.read(120)
                obs = raw[0:40].decode("ascii", "replace").strip("?- \x00")
                inst = raw[40:80].decode("ascii", "replace").strip("?- \x00")
                tel = raw[80:120].decode("ascii", "replace").strip("?- \x00")
                camera = inst
            else:
                strs = _ser_read_strings(f, 3)
                if strs is None:
                    return []
                obs, inst, tel = strs
                camera = inst
        if not (1 <= w <= 20000 and 1 <= h <= 20000 and 1 <= nframes <= 10**7):
            return []
        dts = _ser_dtype(depth)
        if dts is None:
            return []
        dt, scale = dts
        dt_raw = f.read(16)  # DateTime (local) + DateTimeUTC : ticks .NET (100 ns)
        try:
            dt_local, dt_utc = struct.unpack("<qq", dt_raw)
        except struct.error:
            dt_local = dt_utc = 0
        end = f.tell()
        variants = ([(1, legacy_bayer, False, 0)] if layout == "legacy"
                    and colormode == 100 else _ser_variants(colormode, layout))
        return [{"w": w, "h": h, "nframes": nframes, "dt": dt, "scale": scale,
                 "per": per, "colormode": colormode, "bayer": bayer, "bgr": bgr,
                 "variant": rank, "camera": camera, "observer": obs,
                 "instrument": inst, "telescope": tel, "header_end": end,
                 "layout": layout, "dt_utc": dt_utc, "dt_local": dt_local}
                for per, bayer, bgr, rank in variants]
    except (struct.error, OSError):
        return []


class SerFrames(Sequence):
    """Vidéo SER projetée en mémoire (np.memmap) : les frames restent sur le
    disque et sont converties en float32 [0..1] (BGR→RGB, dématriçage) à la
    demande. Une vidéo de 5 000 frames n'occupe ainsi presque pas de RAM."""

    def __init__(self, raw, scale: float, colormode: int, bayer: str | None):
        self._raw = raw
        self._scale = float(scale)
        self._colormode = colormode
        self._bayer = bayer if bayer in BAYER_PATTERNS else None

    def _convert(self, fr) -> np.ndarray:
        a = np.asarray(fr, np.float32) / np.float32(self._scale)
        np.clip(a, 0, 1, out=a)
        if a.ndim == 3 and self._colormode == 8:
            a = a[:, :, ::-1]
        elif a.ndim == 2 and self._bayer:
            a = debayer(a, self._bayer)
        return np.ascontiguousarray(a, dtype=np.float32)

    def __len__(self):
        return int(self._raw.shape[0])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return SerFrames(self._raw[index], self._scale, self._colormode, self._bayer)
        if isinstance(index, (list, tuple, np.ndarray)):
            return np.stack([self[int(i)] for i in index])
        return self._convert(self._raw[int(index)])

    @property
    def frame_shape(self):
        h, w = self._raw.shape[1:3]
        color = self._raw.ndim == 4 or self._bayer is not None
        return (h, w, 3) if color else (h, w)

    @property
    def shape(self):
        return (len(self),) + self.frame_shape

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    dtype = np.dtype(np.float32)

    def __array__(self, dtype=None, copy=None):
        out = np.stack([self[i] for i in range(len(self))]) if len(self) else \
            np.empty((0,) + self.frame_shape, np.float32)
        return out.astype(dtype) if dtype is not None else out


def ser_ticks_to_iso(ticks: int) -> str | None:
    """Horodatage SER (ticks .NET de 100 ns depuis 0001-01-01) -> ISO 8601."""
    import datetime as _dt
    try:
        t = int(ticks)
        if t <= 0 or t > 3155378975999999999:
            return None
        d = _dt.datetime(1, 1, 1) + _dt.timedelta(microseconds=t // 10)
        if d.year < 1990:
            return None
        return d.isoformat(timespec="milliseconds")
    except (OverflowError, ValueError):
        return None


def _ser_check(c: dict, size: int, cands: list) -> None:
    """Évalue une interprétation d'en-tête contre la taille réelle du fichier."""
    fb = c["w"] * c["h"] * c["per"] * c["dt"].itemsize
    c["fb"] = fb
    c["possible"] = (size - c["header_end"]) // fb
    if c["possible"] < 1:
        return
    over = max(0, c["possible"] - c["nframes"])
    # un éventuel bloc d'horodatage (8 octets/frame) peut suivre les images
    c["consistent"] = (c["nframes"] <= c["possible"]
                       and over * fb <= 8 * c["nframes"] + fb)
    c["exact"] = (size - c["header_end"] - c["nframes"] * fb) in (0, 8 * c["nframes"])
    cands.append(c)


def load_ser(path: str, max_frames: int | None = None,
             lazy: bool = False) -> tuple[np.ndarray, dict]:
    """Lit un fichier SER : norme v3 (SharpCap, FireCapture, ASIStudio,
    Siril…), fichiers des anciennes versions d'AstroForge (chaînes préfixées,
    dialecte ≤ 1.7). ColorID selon la norme : 8-11 = capteur Bayer (dématricé
    à la lecture), 100 = RGB, 101 = BGR. Chaque interprétation est validée par
    cohérence avec la taille du fichier.
    Retourne (frames float32 [0..1] N×H×W ou N×H×W×3, meta). Avec
    `lazy=True`, les frames sont un SerFrames projeté en mémoire (aucune
    copie de la vidéo en RAM). `max_frames` None ou ≤ 0 = toutes les frames."""
    with open(path, "rb") as f:
        if f.read(14) != b"LUCAM-RECORDER":
            raise ValueError(f"Pas un fichier SER : {path}")
        f.seek(0, 2)
        size = f.tell()
        cands: list[dict] = []
        for layout in ("spec", "v3", "legacy"):
            f.seek(14)
            for c in _ser_candidate(f, layout):
                _ser_check(c, size, cands)
    good = [c for c in cands if c["consistent"]]
    rank = {"spec": 0, "v3": 1, "legacy": 2}
    if good:
        best = min(good, key=lambda c: (0 if c["exact"] else 1,
                                        c["possible"] - c["nframes"],
                                        c["variant"], rank[c["layout"]]))
    elif cands:  # en-tête douteux mais borné : lire ce qui existe réellement
        best = max(cands, key=lambda c: (c["possible"] >= 1, -c["variant"],
                                         c["possible"]))
    else:
        raise ValueError("En-tête SER non reconnu — fichier corrompu ou "
                         "format inconnu (vidéo AVI/MP4 renommée en .ser ?)")
    count = min(best["nframes"], best["possible"])
    if max_frames is not None and max_frames > 0:
        count = min(count, int(max_frames))
    timestamps = []
    trailer = best["header_end"] + best["nframes"] * best["fb"]
    if size >= trailer + 8 * best["nframes"]:
        with open(path, "rb") as f:
            f.seek(trailer)
            ticks = np.frombuffer(f.read(8 * best["nframes"]), "<i8")
        timestamps = [ser_ticks_to_iso(t) for t in ticks[:count]]
        if not any(timestamps):
            timestamps = []
    meta = {"camera": best["camera"], "observer": best["observer"],
            "instrument": best["instrument"], "telescope": best["telescope"],
            "width": best["w"], "height": best["h"], "frames": int(count),
            "frames_header": best["nframes"],
            "date_utc": ser_ticks_to_iso(best.get("dt_utc", 0)),
            "timestamps": timestamps,
            "bits": best["dt"].itemsize * 8, "colorMode": best["colormode"],
            "bayer": best["bayer"], "layout": best["layout"]}
    shape = ((count, best["h"], best["w"]) if best["per"] == 1
             else (count, best["h"], best["w"], best["per"]))
    raw = np.memmap(path, dtype=best["dt"], mode="r",
                    offset=best["header_end"], shape=shape)
    seq = SerFrames(raw, best["scale"], 8 if best["bgr"] else 0, best["bayer"])
    if lazy:
        return seq, meta
    frames = np.asarray(seq)
    del raw, seq
    return frames, meta


def iso_to_ser_ticks(value) -> int:
    """datetime ou chaîne ISO -> ticks .NET (100 ns depuis 0001-01-01)."""
    import datetime as _dt
    d = value if isinstance(value, _dt.datetime) else _dt.datetime.fromisoformat(str(value))
    if d.tzinfo is not None:
        d = d.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    delta = d - _dt.datetime(1, 1, 1)
    return (delta.days * 86400 + delta.seconds) * 10_000_000 + delta.microseconds * 10


def write_ser(path: str, frames: np.ndarray, fps: float = 30.0, camera: str = "AstroForge",
              observer: str = "", instrument: str = "", telescope: str = "",
              timestamps=None) -> None:
    """Écrit des frames mono/couleur (float [0..1]) dans un fichier SER v3
    standard (en-tête fixe de 178 octets, ColorID 0 = mono, 100 = RGB),
    lisible par AutoStakkert!, Registax, Siril, PIPP…"""
    frames = np.asarray(frames)
    if frames.ndim == 3:
        n, h, w = frames.shape
        colormode = 0
    else:
        n, h, w, _ = frames.shape
        colormode = 100  # RGB
    data = to_uint16(frames)

    def field(txt: str) -> bytes:
        return txt.encode("ascii", "replace")[:40].ljust(40, b"\x00")

    with open(path, "wb") as f:
        f.write(b"LUCAM-RECORDER")
        f.write(struct.pack("<iii", 0, colormode, 0))   # LuID, ColorID, LittleEndian
        f.write(struct.pack("<iiii", w, h, 16, n))      # W, H, PixelDepthPerPlane, FrameCount
        f.write(field(observer) + field(instrument or camera) + field(telescope))
        t0 = iso_to_ser_ticks(timestamps[0]) if timestamps else 0
        f.write(struct.pack("<qq", t0, t0))             # DateTime / DateTimeUTC
        f.write(data.astype("<u2").tobytes())
        if timestamps:  # bloc d'horodatage par frame (UTC)
            f.write(np.asarray([iso_to_ser_ticks(t) for t in timestamps], "<i8").tobytes())


def stretch_display(img: np.ndarray, black: float = 0.0, gamma: float = 1.0,
                    max_val: float | None = None) -> np.ndarray:
    """Étire une image linéaire pour l'affichage (autoscaling type zscale/asinh)."""
    a = np.asarray(img, np.float32)
    if max_val is None:
        med = np.median(a)
        max_val = float(med + 3.0 * a.std()) if med > 0 else float(a.max())
        max_val = max(max_val, 1e-6)
    a = np.clip((a - black) / (max_val - black), 0, 1)
    if gamma != 1.0:
        a = np.power(a, 1.0 / gamma)
    return a
