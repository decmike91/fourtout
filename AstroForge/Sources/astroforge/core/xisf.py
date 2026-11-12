# -*- coding: utf-8 -*-
"""
XISF natif (PixInsight), spécification XISF 1.0 :
- lecture : fichiers monolithiques conformes (signature « XISF0100 »,
  longueur d'en-tête sur 4 octets + 4 octets réservés, blocs
  « attachment:position:taille » ou « inline:base64 »), stockage planaire ou
  entrelacé, compression zlib / zlib+sh (et lz4, lz4hc, zstd si les modules
  lz4 / zstandard sont installés) ; fichiers écrits par AstroForge ≤ 2.x
  (ancien format sans longueur d'en-tête) toujours lisibles ;
- écriture : Float32 planaire non compressé, avec bornes [0, 1] et
  métadonnées obligatoires (ouvrable dans PixInsight et Siril).
"""
from __future__ import annotations
import base64
import datetime as _dt
import os
import re
import struct
import xml.etree.ElementTree as ET
import zlib

import numpy as np

_DTYPE = {
    "UInt8": np.dtype("<u1"), "UInt16": np.dtype("<u2"), "UInt32": np.dtype("<u4"),
    "Float32": np.dtype("<f4"), "Float64": np.dtype("<f8"),
}
_SIGNATURE = b"XISF0100"


def _decompress(raw: bytes, comp: str, itemsize: int) -> bytes:
    """compression = « codec:taille-décompressée[:taille-élément] »."""
    parts = comp.split(":")
    codec = parts[0].strip().lower()
    shuffled = codec.endswith("+sh")
    base = codec[:-3] if shuffled else codec
    item = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else itemsize
    if base == "zlib":
        out = zlib.decompress(raw)
    elif base in ("lz4", "lz4hc"):
        try:
            import lz4.block
        except ImportError as e:
            raise ValueError("XISF compressé LZ4 : installez le module lz4 "
                             "(pip install lz4)") from e
        out = lz4.block.decompress(raw, uncompressed_size=int(parts[1]))
    elif base == "zstd":
        try:
            import zstandard
        except ImportError as e:
            raise ValueError("XISF compressé Zstandard : installez le module "
                             "zstandard (pip install zstandard)") from e
        out = zstandard.ZstdDecompressor().decompress(raw, max_output_size=int(parts[1]))
    else:
        raise ValueError(f"XISF : compression « {codec} » non prise en charge")
    if shuffled and item > 1:  # défait le byte-shuffle
        b = np.frombuffer(out, np.uint8)
        n = len(b) // item
        head = b[:n * item].reshape(item, n).T.tobytes()
        out = head + b[n * item:].tobytes()
    return out


def _parse(blob: bytes):
    """Retourne (racine XML, décalage de base des positions)."""
    if not blob.startswith(b"XISF"):
        raise ValueError("Pas un fichier XISF")
    xml = None
    if len(blob) >= 16:
        (hlen,) = struct.unpack("<I", blob[8:12])
        if 16 + hlen <= len(blob) and blob[16:16 + hlen].lstrip()[:5] in (b"<?xml", b"<xisf"):
            xml = blob[16:16 + hlen]
    if xml is None:  # ancien format AstroForge : XML juste après la signature
        end = blob.find(b"</xisf>")
        if end < 0:
            raise ValueError("XISF corrompu (en-tête XML absent)")
        xml = blob[8:end + 7]
    text = xml.decode("utf-8", "replace").rstrip("\x00 \r\n\t")
    return ET.fromstring(text)


def _strip(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_xisf(path: str) -> np.ndarray:
    """Lit la première image d'un fichier XISF -> float32 [0..1]
    (HxW ou HxWx3)."""
    from .image_io import to_float
    with open(path, "rb") as f:
        blob = f.read()
    try:
        root = _parse(blob)
    except ET.ParseError as e:
        raise ValueError(f"XISF illisible ({e}) : {path}") from e
    img = next((el for el in root.iter() if _strip(el.tag) == "Image"), None)
    if img is None:
        raise ValueError(f"XISF sans image : {path}")
    geom = [int(v) for v in img.get("geometry").split(":")]  # w:h[:c]
    w, h = geom[0], geom[1]
    c = geom[2] if len(geom) > 2 else 1
    fmt = img.get("sampleFormat", "Float32")
    if fmt not in _DTYPE:
        raise ValueError(f"XISF : format d'échantillon {fmt} non pris en charge")
    dt = _DTYPE[fmt]
    loc = img.get("location", "")
    m_att = re.match(r"attachment:(\d+):(\d+)", loc)
    m_old = re.match(r"payload:@(\d+):(\d+)", loc)
    if m_att or m_old:
        m = m_att or m_old
        off, length = int(m.group(1)), int(m.group(2))
        raw = blob[off:off + length]
    elif loc.startswith("inline:"):
        enc = loc.split(":", 1)[1]
        txt = (img.text or "").strip()
        raw = base64.b64decode(txt) if enc == "base64" else bytes.fromhex(txt)
    elif loc == "embedded":
        data_el = next((el for el in img if _strip(el.tag) == "Data"), None)
        if data_el is None:
            raise ValueError("XISF : bloc embarqué sans élément Data")
        raw = base64.b64decode((data_el.text or "").strip())
    else:
        raise ValueError(f"XISF : emplacement de données non pris en charge « {loc} »")
    comp = img.get("compression", "")
    if comp:
        raw = _decompress(raw, comp, dt.itemsize)
    if img.get("byteOrder", "little") == "big":
        dt = dt.newbyteorder(">")
    data = np.frombuffer(raw, dtype=dt, count=w * h * c)
    if c == 1:
        arr = data.reshape(h, w)
    elif img.get("pixelStorage", "Planar").lower() == "normal":
        arr = data.reshape(h, w, c)
    else:
        arr = np.moveaxis(data.reshape(c, h, w), 0, -1)  # planaire -> HWC
    if c > 3:
        arr = arr[..., :3]
    if arr.dtype.kind == "f":
        bounds = img.get("bounds")
        if bounds and ":" in bounds:
            lo, hi = (float(v) for v in bounds.split(":")[:2])
            if hi > lo and (lo, hi) != (0.0, 1.0):
                arr = (arr.astype(np.float32) - lo) / np.float32(hi - lo)
    return np.ascontiguousarray(to_float(np.asarray(arr).astype(arr.dtype.newbyteorder("="))))


def _fits_value(v) -> str:
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return repr(float(v)).upper().replace("INF", "Inf")
    return "'" + str(v).replace("'", "''") + "'"


def _xml_attr(v: str) -> str:
    return (str(v).replace("&", "&amp;").replace('"', "&quot;")
            .replace("<", "&lt;").replace(">", "&gt;"))


def fits_keywords(path: str) -> dict:
    """Mots-clés FITS (éléments FITSKeyword) d'un fichier XISF."""
    import xml.etree.ElementTree as ET
    with open(path, "rb") as f:
        head = f.read(16)
        if head[:8] != _SIGNATURE:
            raise ValueError("Pas un fichier XISF")
        n = struct.unpack("<I", head[8:12])[0]
        root = ET.fromstring(f.read(n))
    out = {}
    for el in root.iter():
        if _strip(el.tag) == "FITSKeyword":
            v = el.get("value", "").strip()
            if v.startswith("'") and v.endswith("'"):
                v = v[1:-1].replace("''", "'").rstrip()
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        v = {"T": True, "F": False}.get(v, v)
            out[el.get("name", "")] = v
    return out


def write_xisf(path: str, img: np.ndarray, header: dict | None = None) -> None:
    """Écrit un XISF 1.0 monolithique : Float32 planaire non compressé,
    bornes 0:1, métadonnées de création ; `header` : mots-clés FITS
    (éléments FITSKeyword, lus par PixInsight/Siril)."""
    a = np.clip(np.nan_to_num(np.asarray(img, np.float32)), 0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if a.ndim == 2:
        c, color = 1, "Gray"
        data = a.astype("<f4").tobytes()
    else:
        c = 3 if a.shape[2] >= 3 else 1
        color = "RGB" if c == 3 else "Gray"
        data = np.moveaxis(a[..., :c], -1, 0).astype("<f4").tobytes()  # planaire
    h, w = a.shape[:2]
    now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    skip = {"SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "EXTEND",
            "BZERO", "BSCALE"}
    kw = "".join(
        f'<FITSKeyword name="{_xml_attr(str(k)[:8].upper())}" '
        f'value="{_xml_attr(_fits_value(v))}" comment=""/>'
        for k, v in (header or {}).items()
        if v is not None and str(k).upper() not in skip and str(k).strip())

    def header(off: int) -> bytes:
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<xisf version="1.0" xmlns="http://www.pixinsight.com/xisf" '
            'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
            'xsi:schemaLocation="http://www.pixinsight.com/xisf '
            'http://pixinsight.com/xisf/xisf-1.0.xsd">'
            f'<Image geometry="{w}:{h}:{c}" sampleFormat="Float32" bounds="0:1" '
            f'colorSpace="{color}" location="attachment:{off}:{len(data)}">'
            f'{kw}</Image>'
            '<Metadata>'
            f'<Property id="XISF:CreationTime" type="TimePoint" value="{now}"/>'
            '<Property id="XISF:CreatorApplication" type="String">AstroForge</Property>'
            '</Metadata></xisf>').encode("utf-8")

    off = 0
    for _ in range(4):  # point fixe : l'offset dépend de la taille de l'en-tête
        xml = header(off)
        new = 16 + len(xml)
        new = (new + 4095) // 4096 * 4096  # bloc aligné sur 4 Kio
        if new == off:
            break
        off = new
    xml = header(off)
    with open(path, "wb") as f:
        f.write(_SIGNATURE + struct.pack("<II", len(xml), 0) + xml)
        f.write(b"\x00" * (off - 16 - len(xml)))
        f.write(data)
