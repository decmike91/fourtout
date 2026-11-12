# -*- coding: utf-8 -*-
"""
Ligne de commande AstroForge : traitement sans interface (serveur, scripts,
automatisation).

    astroforge sort NUIT/                       # tri automatique d'une session
    astroforge analyze lights/*.fits --csv analyse.csv
    astroforge stack --session NUIT/ -o m31.fits --method Sigma-clipping --weights Mixte
    astroforge stack --lights l/*.fits --darks darks/ --flats flats/ --bayer RGGB --drizzle 2 -o out.fits
    astroforge multinight NUIT1/ NUIT2/ -o m31.fits --normalization Locale
    astroforge solve image.fits --catalog gaia.csv --ra 10h42m --dec "+41°16'" --write-wcs
    astroforge annotate image.fits -o image_annotee.png
    astroforge compose --ha ha.fits --oiii oiii.fits --palette HOO -o hoo.tif
    astroforge batch sequence.json entree/ sortie/
    astroforge live dossier/ --duration 3600 -o live.fits
    astroforge script mon_script.py [args…]
(« python -m astroforge … » est équivalent.)
"""
from __future__ import annotations

import argparse
import glob
import json
import runpy
import sys

import numpy as np

from . import __version__


def _expand(items):
    out = []
    for it in items or []:
        g = sorted(glob.glob(it))
        out += g if g else [it]
    return out


def _progress(verbose):
    if not verbose:
        return None
    last = {"label": None}

    def p(done, total, label=""):
        if label != last["label"] or done == total:
            print(f"  [{done}/{total}] {label}", file=sys.stderr)
            last["label"] = label
    return p


def _bayer(v):
    return None if not v or v.lower() in ("auto", "none", "aucune", "mono") else v.upper()


def cmd_sort(a):
    from .core.session import plan_session
    plan = plan_session(a.folder)
    if a.json:
        data = [{"name": g.name, "lights": g.paths("light"),
                 "calib": {k: g.paths(k) for k in ("bias", "dark", "darkflat", "flat")},
                 "dark_scale": g.dark_scale, "warnings": g.warnings} for g in plan.groups]
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(plan.summary())
    return 0


def cmd_analyze(a):
    from .core import subframes as SF
    rows = SF.flag_outliers(SF.analyze_paths(_expand(a.files), _progress(a.verbose),
                                             bayer=_bayer(a.bayer)))
    w = SF.compute_weights(rows, a.weights)
    for r, x in zip(rows, w):
        r["weight"] = x
    print(SF.report(rows))
    if a.csv:
        import csv
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f, delimiter=";")
            keys = ("path", "stars", "fwhm", "ecc", "background", "noise", "snr", "weight", "keep")
            wr.writerow(keys)
            for r in rows:
                wr.writerow([r.get(k) for k in keys])
    return 0


def cmd_stack(a):
    from .core import calibration as C
    from .core import stacking as S
    from .core.image_io import save_image
    from .core.registration import estimate_transform, warp
    from .core.session import plan_session, stack_header
    prog = _progress(a.verbose)
    calib_paths = None
    if a.session:
        plan = plan_session(a.session)
        if not plan.groups:
            print("Aucune image lumière.", file=sys.stderr)
            return 2
        g = plan.groups[a.group]
        lights = g.paths("light")
        calib_paths = {k: g.paths(k) for k in ("bias", "dark", "darkflat", "flat")}
        print(g.summary(), file=sys.stderr)
    else:
        lights = _expand(a.lights)
    if not lights:
        print("Aucune image à empiler.", file=sys.stderr)
        return 2
    masters = C.build_masters(a.bias or "", a.darks or "", a.flats or "", a.darkflats or "",
                              paths=calib_paths)
    imgs = C.calibrate_paths(lights, masters, bayer=_bayer(a.bayer), progress=prog,
                             output=a.output_kind)
    weights = None
    if a.weights != "Aucune":
        from .core import subframes as SF
        rows = SF.flag_outliers([dict(SF.analyze_image(im), path=p)
                                 for im, p in zip(imgs, lights)])
        weights = SF.compute_weights(rows, a.weights)
        print(f"Rejetées : {sum(1 for x in weights if x == 0)}", file=sys.stderr)
    ref = imgs[len(imgs) // 2]
    if a.drizzle > 1:
        from .core.drizzle import drizzle_stack
        mats = [estimate_transform(ref, im) for im in imgs]
        aligned = [warp(im, M) for im, M in zip(imgs, mats)]
        reference = S.stack(np.stack(aligned), a.method, sigma=a.sigma, weights=weights)
        out, _ = drizzle_stack(imgs, matrices=mats, scale=a.drizzle, pixfrac=a.pixfrac,
                               reference=reference, reject_sigma=a.sigma, weights=weights,
                               progress=prog)
    else:
        aligned = [im if im is ref else warp(im, estimate_transform(ref, im)) for im in imgs]
        if a.normalization != "Aucune":
            from .core.normalization import normalize_frames
            aligned = normalize_frames(aligned, ref, a.normalization)
        out = S.stack(np.stack(aligned), a.method, sigma=a.sigma, weights=weights)
    save_image(a.out, out, header=stack_header(lights, weights)
               if a.out.lower().endswith((".fits", ".fit")) else None)
    print(f"Écrit : {a.out} ({out.shape[1]}×{out.shape[0]})")
    return 0


def cmd_multinight(a):
    from .core.image_io import save_image
    from .core.multisession import stack_sessions
    out, info = stack_sessions(a.folders, a.method, a.sigma, a.normalization,
                               _bayer(a.bayer), weight_mode=a.weights,
                               progress=_progress(a.verbose))
    save_image(a.out, out)
    print(f"Écrit : {a.out} — {info['n_frames']} images ({info['per_night']})")
    return 0


def cmd_solve(a):
    from .core.astrometry import solve_image
    from .core.equipment import parse_coord
    from .core.image_io import load_image
    img = load_image(a.image)
    res = solve_image(img, a.catalog, parse_coord(a.ra, True) if a.ra else None,
                      parse_coord(a.dec, False) if a.dec else None, a.radius,
                      a.focal, a.pixel, prefer_solve_field=not a.internal)
    print(f"Méthode {res['method']} — centre RA {res['center_ra']:.5f}° Dec "
          f"{res['center_dec']:.5f}° — {res['scale_arcsec_pix']:.3f}\"/px — "
          f"{res['n_matched']} étoiles — rms {res['rms_arcsec']:.2f}\"")
    if a.write_wcs:
        if not a.image.lower().endswith((".fits", ".fit", ".fts")):
            print("--write-wcs : FITS uniquement", file=sys.stderr)
            return 2
        from astropy.io import fits
        with fits.open(a.image, mode="update") as h:
            h[0].header.update(res["wcs"].to_header(relax=True))
            h.flush()
        print("WCS écrit dans l'en-tête.")
    return 0


def cmd_annotate(a):
    from .core.annotate import annotate
    from .core.display import autostretch
    from .core.image_io import load_image, save_image
    from .core.mosaic import wcs_from_fits
    wcs = wcs_from_fits(a.image)
    if wcs is None:
        print("Pas de WCS dans l'en-tête (lancez « solve --write-wcs »).", file=sys.stderr)
        return 2
    img = load_image(a.image)
    base = img if float(np.median(img)) > 0.1 else autostretch(img)
    out, objs = annotate(base, wcs, a.mag)
    for o in objs:
        print(f"{o['label']:<10} {o['type_fr']:<24} mag {o['mag'] or '—'}  {o.get('common') or ''}")
    if a.out:
        save_image(a.out, out)
    return 0


def cmd_compose(a):
    from .core import narrowband as NB
    from .core.image_io import load_image, save_image
    ch = {}
    for key, path in (("Ha", a.ha), ("OIII", a.oiii), ("SII", a.sii), ("R", a.r),
                      ("G", a.g), ("B", a.b)):
        if path:
            im = load_image(path)
            ch[key] = im if im.ndim == 2 else im.mean(axis=2)
    if a.align and len(ch) > 1:
        ch = NB.align_channels(ch, "Ha" if "Ha" in ch else None)
    out = NB.compose(ch, a.palette, normalize=not a.no_normalize)
    if a.lum:
        L = load_image(a.lum)
        out = NB.lrgb(L if L.ndim == 2 else L.mean(axis=2), out, a.saturation)
    save_image(a.out, out)
    print(f"Écrit : {a.out}")
    return 0


def cmd_batch(a):
    from .core import sequence as Q
    seq = Q.load_sequence(a.sequence)
    if seq.get("mode") == "ser":
        res = Q.run_batch_ser_chain(seq, a.input, a.output, progress=_progress(a.verbose))
    else:
        res = Q.run_batch_image(seq, a.input, a.output, progress=_progress(a.verbose))
    print(res.summary() if hasattr(res, "summary") else f"{len(res)} fichiers")
    return 0 if not getattr(res, "errors", None) else 1


def cmd_live(a):
    from .core import calibration as C
    from .core.image_io import save_image
    from .core.livestack import LiveStacker, run_folder
    masters = C.build_masters(dark_dir=a.darks or "", flat_dir=a.flats or "")
    st = LiveStacker(masters, _bayer(a.bayer), kappa=a.kappa)

    def upd(s):
        print(s.status(), file=sys.stderr)
        if a.out:
            save_image(a.out, s.image())
    run_folder(a.folder, st, duration_s=a.duration, interval_s=a.interval, on_update=upd)
    print(st.status())
    return 0


def cmd_script(a):
    """Exécute un script Python avec l'API AstroForge importable."""
    sys.argv = [a.file] + list(a.args)
    runpy.run_path(a.file, run_name="__main__")
    return 0


def cmd_info(a):
    from . import i18n, paths
    from .core import gpu
    print(f"AstroForge {__version__}")
    print(f"Données utilisateur : {paths.DATA_ROOT}")
    print(gpu.info())
    print(f"Langue : {i18n.configured_language()}")
    if a.check_updates:
        from . import updates
        url = updates.update_url()
        if not url:
            print("Mises à jour : aucune adresse configurée (settings.json « update_url »)")
        else:
            info = updates.check(url)
            print(f"Nouvelle version : {info['version']} — {info['url']}" if info
                  else "Mises à jour : version à jour")
    return 0


def gui_main():
    """Point d'entrée « astroforge-gui »."""
    from .gui.app import main as _gui
    return _gui()


def build_parser():
    p = argparse.ArgumentParser(prog="astroforge", description="AstroForge en ligne de commande")
    p.add_argument("--version", action="version", version=f"AstroForge {__version__}")
    p.add_argument("-v", "--verbose", action="store_true", help="progression détaillée")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sort", help="tri automatique d'une session")
    s.add_argument("folder")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_sort)

    s = sub.add_parser("analyze", help="analyse des brutes (FWHM, étoiles, poids)")
    s.add_argument("files", nargs="+")
    s.add_argument("--bayer", default="auto")
    s.add_argument("--weights", default="Mixte", choices=["Aucune", "SNR", "FWHM", "Mixte"])
    s.add_argument("--csv")
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("stack", help="calibration + alignement + empilement")
    s.add_argument("--session", help="dossier de session (tri automatique)")
    s.add_argument("--group", type=int, default=0, help="groupe de lumières (0 = le plus fourni)")
    s.add_argument("--lights", nargs="*")
    for k in ("bias", "darks", "flats", "darkflats"):
        s.add_argument(f"--{k}")
    s.add_argument("--bayer", default="auto")
    s.add_argument("--output-kind", default="RGB", choices=["RGB", "Ha", "OIII"])
    s.add_argument("--method", default="Sigma-clipping")
    s.add_argument("--sigma", type=float, default=3.0)
    s.add_argument("--weights", default="Aucune", choices=["Aucune", "SNR", "FWHM", "Mixte"])
    s.add_argument("--normalization", default="Aucune", choices=["Aucune", "Globale", "Locale"])
    s.add_argument("--drizzle", type=int, default=1, choices=[1, 2, 3])
    s.add_argument("--pixfrac", type=float, default=0.7)
    s.add_argument("-o", "--out", required=True)
    s.set_defaults(fn=cmd_stack)

    s = sub.add_parser("multinight", help="empilement de plusieurs nuits")
    s.add_argument("folders", nargs="+")
    s.add_argument("--method", default="Sigma-clipping")
    s.add_argument("--sigma", type=float, default=3.0)
    s.add_argument("--normalization", default="Locale", choices=["Aucune", "Globale", "Locale"])
    s.add_argument("--weights", default="Aucune", choices=["Aucune", "SNR", "FWHM", "Mixte"])
    s.add_argument("--bayer", default="auto")
    s.add_argument("-o", "--out", required=True)
    s.set_defaults(fn=cmd_multinight)

    s = sub.add_parser("solve", help="astrométrie (plate solving)")
    s.add_argument("image")
    s.add_argument("--catalog")
    s.add_argument("--ra")
    s.add_argument("--dec")
    s.add_argument("--radius", type=float, default=1.0)
    s.add_argument("--focal", type=float)
    s.add_argument("--pixel", type=float)
    s.add_argument("--internal", action="store_true", help="ne pas utiliser solve-field")
    s.add_argument("--write-wcs", action="store_true")
    s.set_defaults(fn=cmd_solve)

    s = sub.add_parser("annotate", help="annotation Messier/NGC/IC")
    s.add_argument("image")
    s.add_argument("--mag", type=float, default=14.0)
    s.add_argument("-o", "--out")
    s.set_defaults(fn=cmd_annotate)

    s = sub.add_parser("compose", help="composition bande étroite / LRGB")
    for k in ("ha", "oiii", "sii", "r", "g", "b", "lum"):
        s.add_argument(f"--{k}")
    s.add_argument("--palette", default="HOO")
    s.add_argument("--align", action="store_true")
    s.add_argument("--no-normalize", action="store_true")
    s.add_argument("--saturation", type=float, default=1.0)
    s.add_argument("-o", "--out", required=True)
    s.set_defaults(fn=cmd_compose)

    s = sub.add_parser("batch", help="séquence enregistrée appliquée à un dossier")
    s.add_argument("sequence")
    s.add_argument("input")
    s.add_argument("output")
    s.set_defaults(fn=cmd_batch)

    s = sub.add_parser("live", help="empilement en direct d'un dossier")
    s.add_argument("folder")
    s.add_argument("--duration", type=float, default=0.0, help="secondes (0 = fichiers présents)")
    s.add_argument("--interval", type=float, default=2.0)
    s.add_argument("--darks")
    s.add_argument("--flats")
    s.add_argument("--bayer", default="auto")
    s.add_argument("--kappa", type=float, default=3.0)
    s.add_argument("-o", "--out")
    s.set_defaults(fn=cmd_live)

    s = sub.add_parser("info", help="version, GPU, langue, dossier des données")
    s.add_argument("--check-updates", action="store_true")
    s.set_defaults(fn=cmd_info)

    s = sub.add_parser("script", help="exécuter un script Python utilisant l'API")
    s.add_argument("file")
    s.add_argument("args", nargs=argparse.REMAINDER)
    s.set_defaults(fn=cmd_script)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except (ValueError, RuntimeError, OSError) as e:
        print(f"Erreur : {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
