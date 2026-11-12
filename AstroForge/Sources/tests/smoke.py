# -*- coding: utf-8 -*-
"""Tests automatiques AstroForge : algorithmes + chargement headless de l'interface.
Lancer :  python tests/smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import shutil
import tempfile

# dossier de travail des tests (portable : Windows, macOS, Linux)
TMP = tempfile.mkdtemp(prefix="astroforge_tests_")

import astroforge.core.calibration as C
import astroforge.core.effects as E
import astroforge.core.gradient as G
import astroforge.core.registration as R
import astroforge.core.restoration as RT
import astroforge.core.stacking as S
import astroforge.core.wavelets as W
from astroforge.core.drizzle import drizzle_stack
from astroforge.core.image_io import load_ser, write_ser

PASS = []


def check(name, cond):
    PASS.append((name, bool(cond)))
    print(("✔" if cond else "✖"), name)


def synth_frame(shift=(0, 0), angle=0.0, noise=0.01, seed=0, trail=False):
    rng = np.random.default_rng(seed)
    n = 128
    y, x = np.mgrid[0:n, 0:n]
    img = np.zeros((n, n), np.float32)
    # "étoiles"
    for (cy, cx) in [(40, 60), (80, 30), (64, 100), (20, 20)]:
        img += 0.8 * np.exp(-((x - cx - shift[0]) ** 2 + (y - cy - shift[1]) ** 2) / 6.0)
    img += 0.05  # fond + gradient
    img += (x / n) * 0.05
    if angle:
        import cv2
        img = cv2.warpAffine(img, cv2.getRotationMatrix2D((n / 2, n / 2), angle, 1.0),
                             (n, n))
    if trail:
        img[100:102, 10:120] += 0.5  # trace de satellite
    img += rng.normal(0, noise, (n, n)).astype(np.float32)
    return np.clip(img, 0, None).astype(np.float32)


def test_registration():
    ref = synth_frame(seed=1)
    mov = synth_frame(shift=(4, -7), seed=1)
    dx, dy, resp = R.phase_corr(ref, mov)
    check("phase_corr retrouve le décalage", abs(dx + 4) < 0.6 and abs(dy - 7) < 0.6)
    aligned = R.apply_shift(mov, dx, dy)
    err = np.abs(aligned - ref).mean()
    check(f"alignement résiduel faible ({err:.4f})", err < 0.02)


def test_calibration_stacking():
    bias = np.full((128, 128), 0.02, np.float32)
    darks = [bias + 0.05 + np.random.default_rng(i).normal(0, .002, (128, 128)).astype(np.float32) for i in range(5)]
    flats = [np.full((128, 128), 0.5, np.float32) * (1 + 0.1 * np.arange(128)[None, :] / 128) for _ in range(5)]
    mb = C.make_master_bias(darks)
    md = C.make_master_dark(darks, mb)
    mf = C.make_master_flat(flats, mb)
    light = synth_frame() * 0.4 + 0.05 + md
    cal = C.calibrate(light, mb, md, mf)
    check("calibration : sortie bornée et non vide", cal.shape == light.shape and cal.min() >= 0)
    frames = [synth_frame(seed=i, noise=0.03) for i in range(9)]
    st = np.stack(frames)
    for m in S.METHODS:
        out = S.stack(st, m)
        check(f"empilement {m}", out.shape == (128, 128) and np.isfinite(out).all())
    patch = (slice(100, 120), slice(0, 20))  # zone de fond sans étoile
    n_in = float(frames[0][patch].std())
    out = S.stack(st, "Sigma-clipping")
    n_out = float(out[patch].std())
    check(f"empilement réduit le bruit de fond ({n_in:.3f} → {n_out:.3f})",
          n_out < n_in * 0.8)


def test_drizzle():
    """Vérité terrain : image haute résolution, frames basse résolution
    décalées d'un demi-pixel (dithering) ; le drizzle doit mieux restituer la
    vérité que la moyenne alignée puis agrandie."""
    import cv2
    rng = np.random.default_rng(0)
    H = 192
    yy, xx = np.mgrid[0:H, 0:H]
    hi = np.full((H, H), 0.02, np.float32)
    for cy, cx in rng.uniform(16, H - 16, (30, 2)):
        hi += 0.8 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 1.2 ** 2))
    frames = []
    for _ in range(12):
        ox, oy = rng.integers(0, 2, 2)
        sx, sy = rng.integers(-3, 4, 2)
        sh = np.roll(np.roll(hi, 2 * sy + oy, 0), 2 * sx + ox, 1)
        lo = sh.reshape(H // 2, 2, H // 2, 2).mean((1, 3))
        frames.append((lo + rng.normal(0, 0.002, lo.shape)).astype(np.float32))
    shifts = [R.phase_corr(frames[0], f)[:2] for f in frames]
    img, cov = drizzle_stack(frames, shifts, scale=2, pixfrac=0.6)
    check("drizzle ×2 : taille et couverture complète",
          img.shape == (H, H) and cov.min() > 0)
    aligned = [R.apply_shift(f, *s_) for f, s_ in zip(frames, shifts)]
    up = cv2.resize(S.stack(np.stack(aligned), "Moyenne"), (H, H),
                    interpolation=cv2.INTER_CUBIC)

    def fidelity(x):
        dx, dy, _ = R.phase_corr(x, hi)
        t = R.apply_shift(hi, dx, dy)
        c = slice(16, -16)
        return float(np.corrcoef(x[c, c].ravel(), t[c, c].ravel())[0, 1])
    f_dz, f_up = fidelity(img), fidelity(up)
    check(f"drizzle plus fidèle que moyenne agrandie ({f_dz:.4f} > {f_up:.4f})",
          f_dz > f_up)


def test_wavelets_restoration():
    img = synth_frame(seed=4, noise=0.06)
    dn = RT.denoise(img, "Ondelettes", 0.8)
    check("denoise ondelettes réduit le bruit", dn.std() < img.std())
    dn2 = RT.denoise(img, "Non-local means", 0.5)
    check("denoise NL-means", dn2.shape == img.shape)
    dv = RT.richardson_lucy(synth_frame(seed=4), 1.2, 25, damping=0.1)
    check("déconvolution RL accentue la netteté",
          S.sharpness_score(dv) > S.sharpness_score(synth_frame(seed=4)))
    layers = W.atrous_decompose(img, 5)
    rec = W.atrous_recompose(layers, [2, 1, 1, 1, 1, 1])
    check("ondelettes à trous : recomposition à gain neutre ≈ identité",
          np.abs(W.atrous_recompose(layers, [1.0] * 6) - img).mean() < 1e-3)


def test_gradient():
    base = synth_frame(seed=5)
    img = np.zeros((256, 256), np.float32)
    for i in range(2):
        for j in range(2):
            img[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = base
    x = np.arange(256)[None, :].repeat(256, 0)
    img += (x / 256.0 * 0.15).astype(np.float32)
    fixed, model = G.extract_gradient(img, "Polynôme 2D", 3, 12, 2.5, "Soustraction")
    slope_before = abs(float(np.diff(img.mean(axis=0)).mean()))
    slope_after = abs(float(np.diff(fixed.mean(axis=0)).mean()))
    check(f"extraction de gradient réduit la pente ({slope_before:.4f} → {slope_after:.4f})",
          slope_after < slope_before)


def test_effects():
    img = synth_frame(seed=6, trail=True)
    out = E.remove_satellite_trails(img)
    check("trace satellite atténuée",
          out[100:102, 10:120].mean() < img[100:102, 10:120].mean())
    exp1 = np.clip(synth_frame(seed=7) * 0.5, 0, 1)
    exp2 = np.clip(synth_frame(seed=7) * 1.5, 0, 1)
    hdr = E.merge_hdr([exp1, exp2])
    check("fusion HDR", hdr.shape == exp1.shape and np.isfinite(hdr).all())
    fc = E.apply_palette(synth_frame(seed=8), "Solaire (orange)")
    check("fausses couleurs 3 canaux", fc.ndim == 3 and fc.shape[2] == 3)


def test_ser():
    frames = np.stack([synth_frame(seed=i) for i in range(3)])
    p = TMP + "/test.ser"
    write_ser(p, frames)
    back, meta = load_ser(p)
    check("SER : écriture/lecture aller-retour",
          back.shape == frames.shape and abs(float(back.mean() - frames.mean())) < 1e-3)

    import struct as st

    def _ser_file(pf, depth_field, color_id, n, w, h, body_u16, nstrings=3):
        hdr = (b"LUCAM-RECORDER" + st.pack("<iii", 0, color_id, 1)
               + st.pack("<iiii", w, h, depth_field, n))
        for _ in range(nstrings):
            hdr += st.pack("<i", 1) + b"\x00"
        hdr += st.pack("<qq", 0, 0)
        open(pf, "wb").write(hdr + body_u16)

    w, h, n = 24, 16, 4
    body = np.zeros((n, h, w), np.float32)
    body[2, 5, 7] = 0.75
    # a) v3 mono, profondeur en BITS (16) — spec officielle
    _ser_file(TMP + "/ser_bits.ser", 16, 0, n, w, h,
              (np.clip(body, 0, 1) * 65535).astype("<u2").tobytes())
    fr, m = load_ser(TMP + "/ser_bits.ser")
    check(f"SER v3 profondeur en bits : {m['width']}×{m['height']}, {m['frames']} frames",
          fr.shape == (n, h, w) and abs(float(fr[2, 5, 7]) - 0.75) < 1e-3)
    # b) v3 mono, profondeur en OCTETS (2) — variante FireCapture/SharpCap
    _ser_file(TMP + "/ser_bytes.ser", 2, 0, n, w, h,
              (np.clip(body, 0, 1) * 65535).astype("<u2").tobytes())
    fr, _m = load_ser(TMP + "/ser_bytes.ser")
    check("SER v3 profondeur en octets acceptée",
          fr.shape == (n, h, w) and abs(float(fr[2, 5, 7]) - 0.75) < 1e-3)
    # c) couleur BGR (ColorID 8) → convertie RGB
    px = np.zeros((2, h, w, 3), np.float32)
    px[0, 3, 4] = (0.1, 0.5, 0.9)  # stocké BGR → lu RGB (0.9, 0.5, 0.1)
    _ser_file(TMP + "/ser_bgr.ser", 16, 8, 2, w, h,
              (np.clip(px, 0, 1) * 65535).astype("<u2").tobytes())
    fr3, _m3 = load_ser(TMP + "/ser_bgr.ser")
    check("SER couleur BGR (ColorID 8) → RGB",
          fr3.shape == (2, h, w, 3) and abs(float(fr3[0, 3, 4, 0]) - 0.9) < 1e-3
          and abs(float(fr3[0, 3, 4, 2]) - 0.1) < 1e-3)
    # d) fichier tronqué : 100 frames annoncées, 2 présentes → 2 lues, sans erreur
    _ser_file(TMP + "/ser_trunc.ser", 16, 0, 100, w, h,
              (np.full((2, h, w), 0.5, np.float32) * 65535).astype("<u2").tobytes())
    frt, mt = load_ser(TMP + "/ser_trunc.ser")
    check(f"SER tronqué : {frt.shape[0]} frames lues (pas d'erreur)",
          frt.shape == (2, h, w))
    # e) ancien dialecte AstroForge ≤ 1.7 toujours lisible
    leg = b"LUCAM-RECORDER"
    for sname in ("AstroForge", "obs", "inst", "tel"):
        b = sname.encode() + b"\x00"
        leg += st.pack("<i", len(b)) + b
    leg += (st.pack("<iiiii", w, h, 2, 16, 0) + st.pack("<qq", 0, 0)
            + (np.full((2, h, w), 0.5, np.float32) * 65535).astype("<u2").tobytes())
    open(TMP + "/ser_legacy.ser", "wb").write(leg)
    frl, _ml = load_ser(TMP + "/ser_legacy.ser")
    check("SER ancien format AstroForge ≤1.7 encore lisible",
          frl.shape == (2, h, w) and abs(float(frl.mean()) - 0.5) < 1e-3)
    # f) en-tête absurde → message clair (jamais « negative dimensions »)
    open(TMP + "/ser_bad.ser", "wb").write(
        b"LUCAM-RECORDER" + bytes(range(256)))
    try:
        load_ser(TMP + "/ser_bad.ser")
        ok_bad = False
    except ValueError as e:
        ok_bad = "negative" not in str(e) and "En-tête" in str(e)
    check("SER corrompu → message clair (pas d'exception numpy)", ok_bad)
    # g) Norme SER v3 (FireCapture, SharpCap…) : champs fixes de 40 octets SANS préfixe
    # (reproduit l'entête réelle : « ? » + tirets, dates ASCII « 01011900 »)
    fire = (b"LUCAM-RECORDER" + st.pack("<iii", 0, 0, 0)
            + st.pack("<iiii", 41, 39, 16, 6)
            + (b"?-" * 20) * 3
            + b"01011900" + b"00000000")
    bodyf = np.zeros((6, 39, 41), np.float32)
    bodyf[2, 5, 7] = 0.75
    open(TMP + "/ser_fire.ser", "wb").write(
        fire + (np.clip(bodyf, 0, 1) * 65535).astype("<u2").tobytes())
    frf, mf = load_ser(TMP + "/ser_fire.ser")
    check(f"SER FireCapture (chaînes fixes 40 o) : {mf['width']}×{mf['height']}, "
          f"{mf['frames']} frames",
          frf.shape == (6, 39, 41) and abs(float(frf[2, 5, 7]) - 0.75) < 1e-3
          and mf["layout"] == "spec")


def test_gui():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    from astroforge.gui.main_window import MainWindow
    mw = MainWindow()
    check("fenêtre principale + 3 modes chargés (.ui)", mw.win is not None)
    # aperçu live ciel profond : calcul direct
    page = mw.page_sky
    img = synth_frame(seed=9)
    page.preview_base = img
    out = page._compute_preview(img, page._trait_params())
    check("aperçu live ciel profond (gradient+denoise+deconv)", out.shape == img.shape)
    page_p = mw.page_planet
    outw = page_p._wave_preview(img, {"gains": [1.5, 1, 1, 1, 1, 1, 1]})
    check("aperçu live ondelettes planétaire", outw.shape == img.shape)
    pc = mw.page_config
    check("page configuration chargée (profils équipement)", pc is not None
          and pc._o("cmbProfile") is not None)
    pc._update_optics()  # ne doit pas lever
    check("workflow planétaire 6 étapes (widgets présents)",
          page_p._o("cmbAlignKindP") is not None
          and page_p._o("sldBrowse") is not None
          and page_p._o("btnRank") is not None
          and page_p._o("spnNAlign") is not None
          and page_p._o("sldRankBrowse") is not None)
    check("groupe pixels défectueux ciel profond",
          page._cbx("chkDefects") is not None)
    check("contrôles aberration chromatique planétaire",
          page_p._o("chkCA") is not None and page_p._o("sldCAR") is not None)
    out_ca = page_p._wave_preview(img, {"gains": [1.0] * 7, "ca": True,
                                        "kr": 0.04, "kb": 0.04})
    check("aperçu live avec correction CA", out_ca.shape == img.shape)
    from astroforge.gui.preview import np_to_qimage
    w_, h_ = 1795, 40
    af = ((np.arange(h_ * w_ * 3).reshape(h_, w_, 3) % 251) / 255.0).astype(np.float32)
    q = np_to_qimage(af)
    raw = np.frombuffer(q.constBits(), np.uint8)[:q.bytesPerLine() * h_].reshape(
        h_, q.bytesPerLine())
    exp = (np.clip(af, 0, 1) * 255).astype(np.uint8).reshape(h_, 3 * w_)
    check("affichage : lignes exactes largeur impaire couleur (ex. 1792×1795)",
          q.bytesPerLine() == (3 * w_ + 3) // 4 * 4
          and np.array_equal(raw[:, :3 * w_], exp))
    am = ((np.arange(h_ * w_).reshape(h_, w_) % 251) / 255.0).astype(np.float32)
    qm = np_to_qimage(am)
    rawm = np.frombuffer(qm.constBits(), np.uint8)[:qm.bytesPerLine() * h_].reshape(
        h_, qm.bytesPerLine())
    expm = (np.clip(am, 0, 1) * 255).astype(np.uint8).reshape(h_, w_)
    check("affichage : lignes exactes largeur impaire N&B",
          qm.bytesPerLine() == (w_ + 3) // 4 * 4
          and np.array_equal(rawm[:, :w_], expm))


def test_stacking_rejection():
    ref = synth_frame(seed=11)
    clean = [ref + np.random.default_rng(100 + i).normal(0, .01, ref.shape).astype(np.float32)
             for i in range(8)]
    bad = ref.copy()
    bad[60:70, 60:70] += 0.5  # artefact type satellite/cosmique
    st = np.stack(clean + [bad])
    out_avg = S.stack(st, "Moyenne")
    patch = (slice(60, 70), slice(60, 70))
    err_avg = abs(float(out_avg[patch].mean() - ref[patch].mean()))
    for name in ("Linear fit", "MAD clipping"):
        out = S.stack(st, name)
        err = abs(float(out[patch].mean() - ref[patch].mean()))
        check(f"{name} élimine l'artefact ({err:.3f} vs moyenne {err_avg:.3f})",
              err < err_avg * 0.7)


def test_colorcal():
    rng = np.random.default_rng(5)
    img = rng.random((100, 100, 3)).astype(np.float32) * 0.05
    img[..., 0] += 0.25
    img[..., 1] += 0.15
    img[..., 2] += 0.10  # fond teinté
    from astroforge.core.colorcal import background_neutralize, scnr_green
    out = background_neutralize(img, 40)
    lum = out.mean(axis=2)
    bg = lum <= np.percentile(lum, 20)
    meds = [float(np.median(out[..., c][bg])) for c in range(3)]
    check(f"neutralisation du fond (écart canaux {max(meds) - min(meds):.4f})",
          max(meds) - min(meds) < 0.03)
    out2 = scnr_green(img)
    check("SCNR vert réduit G", float(out2[..., 1].mean()) <= float(img[..., 1].mean()) + 1e-6)


def test_xisf():
    from astroforge.core.xisf import read_xisf, write_xisf
    rng = np.random.default_rng(3)
    a = rng.random((40, 50)).astype(np.float32)
    col = rng.random((30, 40, 3)).astype(np.float32)
    p1, p2 = TMP + "/t1.xisf", TMP + "/t2.xisf"
    write_xisf(p1, a)
    b = read_xisf(p1)
    check("XISF mono aller-retour", b.shape == a.shape and np.abs(b - a).max() < 1e-5)
    write_xisf(p2, col)
    b2 = read_xisf(p2)
    check("XISF couleur aller-retour", b2.shape == col.shape and np.abs(b2 - col).max() < 1e-5)


def test_astrometry():
    import astroforge.core.astrometry as A
    rng = np.random.default_rng(7)
    n = 400
    yy, xx = np.mgrid[0:n, 0:n]
    img = rng.random((n, n)).astype(np.float32) * 0.02
    xs = rng.uniform(40, n - 40, 12)
    ys = rng.uniform(40, n - 40, 12)
    for x, y in zip(xs, ys):
        img += 0.9 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 4.5).astype(np.float32)
    # catalogue : échelle exacte 1.0"/pix, nord vers le haut, centre (156.25°, 19.75°)
    center = (156.25, 19.75)
    scale = 1.0
    dec = center[1] + (n / 2.0 - ys) * scale / 3600.0
    ra = center[0] + (xs - n / 2.0) * scale / 3600.0 / np.cos(np.deg2rad(center[1]))
    cat = TMP + "/cat.csv"
    with open(cat, "w") as f:
        f.write("ra,dec,mag\n")
        for r_, d_, m_ in zip(ra, dec, rng.uniform(5, 9, len(ra))):
            f.write(f"{r_:.6f},{d_:.6f},{m_:.2f}\n")
    res = A.solve_internal(img, cat, center[0], center[1], radius_deg=0.5,
                           focal_mm=1000.0, pixel_um=4.86)
    check(f"plate solving interne : échelle {res['scale_arcsec_pix']:.3f}\"/px, "
          f"{res['n_matched']} étoiles",
          abs(res["scale_arcsec_pix"] - scale) < 0.05 * scale and res["n_matched"] >= 8)
    ref2 = synth_frame(seed=13)
    aligned = A.align_by_stars(ref2, synth_frame(shift=(3, -2), seed=14))
    check("alignement par étoiles s'exécute", aligned.shape == ref2.shape)


def test_sequence():
    from astroforge.core import sequence as Q
    from astroforge.core.image_io import load_image, save_image
    d, o = TMP + "/seq_in", TMP + "/seq_out"
    os.makedirs(d, exist_ok=True)
    base = synth_frame(seed=20)
    save_image(os.path.join(d, "a.fits"), base)
    save_image(os.path.join(d, "b.fits"), base * 0.8)
    seq = {"mode": "image",
           "ops": [{"op": "brightness_contrast", "params": {"bright": 0.2}}]}
    files = Q.run_batch_image(seq, d, o)
    check("batch séquence : 2 images traitées", len(files) == 2)
    out = load_image(os.path.join(o, "a.tif"))
    expected = float(np.clip(base + 0.2, 0, 1).mean())
    check("séquence appliquée (luminosité)", abs(float(out.mean()) - expected) < 0.02)
    Q.save_sequence(TMP + "/seq.json", seq["ops"], mode="image")
    back = Q.load_sequence(TMP + "/seq.json")
    check("séquence enregistrée/chargée", back["ops"][0]["op"] == "brightness_contrast")


def test_equipment():
    from astroforge.core import equipment as EQ
    p = dict(EQ.DEFAULT_EQUIPMENT)
    p.update(focal_mm=500.0, pixel_um=3.8, reducer=True, reducer_factor=0.8)
    EQ.save_profile("test_eq", p)
    back = EQ.load_profile("test_eq")
    check("profil équipement aller-retour", back["focal_mm"] == 500.0
          and back["reducer"] is True)
    check("focale effective avec réducteur", abs(EQ.effective_focal(back) - 400.0) < 1e-6)
    sc = EQ.scale_from_focal(3.8, 500.0)
    check("échelle depuis focale (206,265·µm/mm)", abs(sc - 206.265 * 3.8 / 500.0) < 1e-6)
    check("focale depuis échelle (aller-retour)",
          abs(EQ.focal_from_scale(3.8, sc) - 500.0) < 1e-6)
    fov = EQ.field_of_view(back)
    check("champ de vue calculé", fov["w_deg"] > 0 and fov["h_deg"] > 0)
    check("parse_coord sexagésimal",
          abs(EQ.parse_coord("10:25:03", True) - (10 + 25 / 60 + 3 / 3600) * 15) < 1e-6
          and abs(EQ.parse_coord("+19:48", False) - (19 + 48 / 60)) < 1e-6)
    EQ.delete_profile("test_eq")
    try:
        import astroquery  # noqa: F401
        check("astroquery présent (test réseau ignoré)", True)
    except ImportError:
        try:
            EQ.make_gaia_catalog(10.0, 10.0, 1.0, TMP + "/g.csv")
            check("catalogue Gaia sans astroquery : aurait dû échouer", False)
        except RuntimeError:
            check("catalogue Gaia : message d'aide clair sans astroquery", True)


def test_defects():
    from astroforge.core import defects as D
    dark = np.full((100, 100), 0.05, np.float32)
    dark[10, 10] = 0.5
    dark[:, 40] = 0.0
    dark[80, :] = 0.0
    frame = np.full((100, 100), 0.3, np.float32)
    frame[10, 10] = 0.9
    frame[:, 40] = 0.0
    frame[80, :] = 0.0
    fixed, info = D.auto_correct(frame, master_dark=dark)
    check(f"pixel chaud corrigé ({float(fixed[10, 10]):.3f})",
          0.2 < float(fixed[10, 10]) < 0.4)
    check("colonne morte restaurée", float(fixed[:, 40].mean()) > 0.2)
    check("ligne morte restaurée", float(fixed[80, :].mean()) > 0.2)
    from astroforge.core.image_io import save_image
    bpm = np.zeros((100, 100), np.float32)
    bpm[10, 10] = 1.0
    save_image(TMP + "/bpm.fits", bpm)
    m = D.load_bad_pixel_map(TMP + "/bpm.fits", (100, 100))
    check("carte de pixels défectueux chargée (1 px)", int(m.sum()) == 1)


def test_ca():
    from astroforge.core.colorcal import correct_chromatic_aberration
    ref = synth_frame(seed=30)
    rgb = np.stack([ref, ref, ref], axis=-1)
    cv2 = __import__("cv2")
    h, w = ref.shape
    s = 1.04
    M = np.array([[s, 0, (w - 1) / 2 * (1 - s)], [0, s, (h - 1) / 2 * (1 - s)]],
                 np.float64)
    rgb[..., 0] = cv2.warpAffine(rgb[..., 0], M, (w, h),
                                 borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    from astroforge.core.astrometry import detect_stars
    c = np.array([(w - 1) / 2.0, (h - 1) / 2.0])

    def mean_radius(ch):
        st = detect_stars(ch, threshold_sigma=6)
        return float(np.hypot(*(st - c).T).mean()) if len(st) else 99.0

    d0 = abs(mean_radius(rgb[..., 0]) - mean_radius(rgb[..., 1]))
    fixed = correct_chromatic_aberration(rgb, k_r=0.04, k_b=0.0)
    d1 = abs(mean_radius(fixed[..., 0]) - mean_radius(fixed[..., 1]))
    check(f"aberration chromatique corrigée (Δrayon {d0:.2f}px → {d1:.2f}px)",
          d1 < d0 * 0.6)


def test_align_methods():
    import astroforge.core.registration as R
    ref = synth_frame(seed=31)
    mov = synth_frame(shift=(3, -2), seed=31)
    for m in R.ALIGN_METHODS:
        out = R.align_frames([ref, mov], m)
        err = float(np.abs(out[1] - out[0]).mean())
        check(f"alignement {m} (résidu {err:.3f})",
              bool(np.isfinite(out).all()) and err < 0.03)


def test_pcc():
    from astroforge.core import astrometry as A
    from astroforge.core.colorcal import photometric_color_calibration
    rng = np.random.default_rng(9)
    n = 300
    yy, xx = np.mgrid[0:n, 0:n]
    xs = rng.uniform(40, n - 40, 10)
    ys = rng.uniform(40, n - 40, 10)
    mags = rng.uniform(7, 10, 10)
    colors = rng.uniform(-0.5, 1.2, 10)   # BP-mag ; RP-mag ≈ -0.8×couleur
    bp, rp = mags + colors, mags - 0.8 * colors
    cat = TMP + "/cat_pcc.csv"
    center = (156.25, 19.75)
    with open(cat, "w") as f:
        f.write("ra,dec,mag,bp,rp\n")
        dec_c = center[1] + (n / 2.0 - ys) / 3600.0
        ra_c = center[0] + (xs - n / 2.0) / 3600.0 / np.cos(np.deg2rad(center[1]))
        for r_, d_, m_, b_, p_ in zip(ra_c, dec_c, mags, bp, rp):
            f.write(f"{r_:.6f},{d_:.6f},{m_:.3f},{b_:.3f},{p_:.3f}\n")
    def chan(amp):
        img = rng.random((n, n)).astype(np.float32) * 0.02
        for x, y, a in zip(xs, ys, amp):
            img += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 4.5).astype(np.float32)
        return np.clip(img, 0, None)
    rgb = np.stack([chan(10 ** (-0.4 * (rp - mags))), chan(np.ones(10)),
                    chan(10 ** (-0.4 * (bp - mags)))], axis=-1)
    rgb[..., 0] *= 1.25   # dominante rouge
    rgb[..., 2] *= 0.8    # déficit de bleu
    res = A.solve_internal(rgb.mean(axis=2), cat, center[0], center[1],
                           radius_deg=0.5, focal_mm=1000.0, pixel_um=4.86)
    out, info = photometric_color_calibration(rgb, res["wcs"], cat)
    check(f"PCC : kR={info['kR']:.2f} kB={info['kB']:.2f} ({info['n']} étoiles)",
          abs(info["kR"] - 1 / 1.25) < 0.25 and abs(info["kB"] - 1 / 0.8) < 0.25
          and info["n"] >= 6)
    # M12 : fond de ciel coloré (pollution lumineuse) soustrait par canal
    out2, info2 = photometric_color_calibration(
        rgb + np.array([0.06, 0.0, 0.03], np.float32), res["wcs"], cat)
    check(f"PCC fond coloré : kR={info2['kR']:.2f} kB={info2['kB']:.2f}",
          abs(info2["kR"] - info["kR"]) < 0.05 and abs(info2["kB"] - info["kB"]) < 0.05)


def test_similarity():
    import astroforge.core.registration as R
    cv2 = __import__("cv2")
    ref = synth_frame(seed=40)
    h, w = ref.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 2.0, 1.04)  # rotation + échelle
    mov = cv2.warpAffine(ref, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    out = R.align_similarity(ref, mov)
    err = float(np.abs(out - ref).mean())
    check(f"alignement en similitude (échelle 1.04, rot 2°) résidu {err:.4f}",
          err < 0.03)


def test_autostacker():
    from astroforge.core.stacking import autostacker_stack
    ref = synth_frame(seed=41)
    shifts = [(2.5, 1.5), (-2.0, 3.0), (3.0, -2.5), (-1.5, -3.0), (0.0, 2.0)]
    frames = [ref] + [R.apply_shift(ref, -dx, -dy) for dx, dy in shifts]
    rng = np.random.default_rng(3)
    noisy = [f + rng.normal(0, .02, f.shape).astype(np.float32) for f in frames]
    out = autostacker_stack(noisy, keep_percent=100, grid_n=3, box_frac=0.3)
    plain = np.mean(noisy, axis=0)
    # l'empilement est aligné sur la MEILLEURE frame (comme AS!) : on le
    # recale globalement sur ref avant de comparer les formes
    out = R.apply_shift(out, *R.phase_corr(ref, out)[:2])
    plain = R.apply_shift(plain, *R.phase_corr(ref, plain)[:2])
    c = slice(8, -8)
    e_as = float(np.abs(out - ref)[c, c].mean())
    e_pl = float(np.abs(plain - ref)[c, c].mean())
    check(f"AutoStack-style aligne le champ ({e_as:.4f} vs {e_pl:.4f})",
          e_as < e_pl * 0.8)


def test_planetary_flow():
    """Flux planétaire 6 étapes : qualité/tri, ROI disque, alignement
    multipoint/global/manuel, pipeline complet et image finale."""
    import scipy.ndimage as ndi2
    import astroforge.core.planetary as PL
    rng = np.random.default_rng(7)
    n = 96
    yy, xx = np.mgrid[0:n, 0:n]
    tex = rng.random((n, n)).astype(np.float32)
    planet = np.exp(-((xx - 48.0) ** 2 + (yy - 48.0) ** 2) / (2 * 14 ** 2)) * 0.9
    frames = []
    for i in range(24):
        sig = 0 if i % 6 == 0 else 0.5 + 0.4 * (i % 5)
        f = planet + 0.03 * tex
        if sig:
            f = ndi2.gaussian_filter(f, sig)
        f = f + rng.normal(0, 0.008, (n, n)).astype(np.float32)
        dx, dy = int(round(0.4 * i)), int(round(0.25 * i))
        frames.append(np.roll(np.roll(f, dy, axis=0), dx, axis=1).astype(np.float32))
    arr = np.stack(frames)
    # étape 2 : qualité + tri (la frame la moins floue doit arriver 1re)
    sc = PL.measure_quality(arr, "Contraste HF (AutoStakkert)")
    order = PL.sort_quality(sc)
    check("tri qualité : frame la plus nette classée 1re", int(order[0]) % 6 == 0)
    roi = PL.detect_planet_box(arr.mean(axis=0))
    check(f"disque auto-détecté {roi}",
          roi is not None and roi[0] < 40 and roi[2] > 55)
    # étape 4/5 : alignement multipoint / global / manuel
    ref = frames[0]

    def peak(img):
        sm = ndi2.gaussian_filter(img, 3)
        y, x = np.unravel_index(int(np.argmax(sm)), sm.shape)
        return float(x), float(y)

    pr = peak(ref)
    pts = PL.grid_points(ref.shape, 3, 24)
    check(f"grille multipoint : {len(pts)} points", len(pts) == 9)
    wm, _st = PL.align_frame(ref, frames[10], pts, 24,
                             "Multi-points (grille NxN)")
    pw = peak(wm)
    check(f"alignement multipoint : écart ({pw[0] - pr[0]:.1f}, {pw[1] - pr[1]:.1f}) px",
          abs(pw[0] - pr[0]) <= 2 and abs(pw[1] - pr[1]) <= 2)
    wg, _st2 = PL.align_frame(ref, frames[10], None, 24,
                              "Global (une translation)")
    pg = peak(wg)
    check("alignement global (une translation)",
          abs(pg[0] - pr[0]) <= 2 and abs(pg[1] - pr[1]) <= 2)
    wman, _st3 = PL.align_frame(ref, frames[10], [(48, 30), (30, 60), (66, 60)],
                                24, "Points manuels (carrés)")
    pm = peak(wman)
    check("alignement points manuels (carrés)",
          abs(pm[0] - pr[0]) <= 2 and abs(pm[1] - pr[1]) <= 2)
    # étapes 3 + 6 : pipeline complet
    res = PL.run_planetary(frames, n_align=10, grid_n=3, box_size=24,
                           stack_method="Moyenne")
    stt = res["stats"]
    check(f"pipeline 6 étapes : {stt['n_selected']} frames, {stt['n_points']} points, "
          f"{stt['rejected_boxes']} boîte(s) rejetée(s)",
          stt["n_selected"] == 10 and stt["n_points"] == 9)
    img = res["image"]
    disk = float(img[40:56, 40:56].mean())
    corner = float(img[:10, :10].mean())
    check(f"image finale : disque brillant ({disk:.3f} vs coin {corner:.3f})",
          img.shape == (n, n) and disk > 2 * corner)


def test_registax_enhance():
    from astroforge.core.wavelets import registax_enhance
    img = synth_frame(seed=42, noise=0.05)
    ident = registax_enhance(img, [1.0] * 7, 0.0)
    check("registax_enhance neutre", np.abs(ident - img).mean() < 1e-3)
    dn = registax_enhance(img, [1.0] * 7, 0.6)
    check("registax denoise réduit le bruit", float(dn.std()) < float(img.std()))
    sh = registax_enhance(img, [1.6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 0.0)
    check("registax gains accentuent", S.sharpness_score(sh) > S.sharpness_score(img))


def test_batch_ser():
    from astroforge.core import sequence as Q
    from astroforge.core.image_io import load_image, write_ser
    import shutil
    d, o = TMP + "/ser_batch", TMP + "/ser_out"
    shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(o, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    ref = synth_frame(seed=43)
    frames = np.stack([R.apply_shift(ref, -dx, -dy) for dx, dy in
                       [(0, 0), (2, 1), (-2, 2), (1, -2)]])
    write_ser(os.path.join(d, "jupiter.ser"), frames)
    seq = {"mode": "ser", "ops": [{"op": "planet_chain", "params": {
        "quality_mode": "Netteté (Registax)", "keep_percent": 100.0,
        "derot": False, "align_method": "Phase (rapide)", "rgb_align": False,
        "ca": False, "kr": 0.0, "kb": 0.0,
        "stack_method": "Moyenne", "stack_percent": 100.0,
        "gains": [1.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "wave_denoise": 0.2,
        "false_color": False, "palette": ""}}]}
    outs = Q.run_batch_ser_chain(seq, d, o, max_frames=10)
    img_dn = load_image(os.path.join(o, "jupiter_stacked.fits"))
    ok1 = len(outs) == 1 and img_dn.size > 0
    seq["ops"][0]["params"]["wave_denoise"] = 0.0
    outs2 = Q.run_batch_ser_chain(seq, d, o, max_frames=10)
    img_raw = load_image(os.path.join(o, "jupiter_stacked.fits"))
    ok2 = abs(float(img_dn.mean() - img_raw.mean())) > 1e-5
    check("batch SER complet (AutoStack + Registax, denoise actif)",
          ok1 and ok2)


def test_align_poly():
    import astroforge.core.registration as R
    cv2 = __import__("cv2")
    rng = np.random.default_rng(50)
    n = 300
    yy, xx = np.mgrid[0:n, 0:n]
    img = rng.random((n, n)).astype(np.float32) * 0.01
    base = np.linspace(45, 255, 5)  # grille 5x5 = 25 étoiles (20 requises pour deg3)
    xs = np.tile(base, 5) + rng.uniform(-8, 8, 25)
    ys = np.repeat(base, 5) + rng.uniform(-8, 8, 25)
    for x, y in zip(xs, ys):
        img += 0.9 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 5.0).astype(np.float32)
    cx0 = cy0 = (n - 1) / 2.0
    dx, dy = xx - cx0, yy - cy0
    map_x = (xx + 0.0000005 * dx * (dx ** 2 + dy ** 2)).astype(np.float32)
    map_y = (yy + 0.0000005 * dy * (dx ** 2 + dy ** 2)).astype(np.float32)
    mov = cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    r1 = R.fit_poly_transform(img, mov, order=1)
    r3 = R.fit_poly_transform(img, mov, order=3)
    check(f"distorsion poly : résidu {r1['resid']:.2f}px (deg1) → {r3['resid']:.2f}px (deg3)",
          r3["resid"] < r1["resid"] * 0.7 and r3["n"] >= 8)
    out = R.align_poly(img, mov, order=3)
    sim = R.align_similarity(img, mov)
    e_poly = float(np.abs(out - img).mean())
    e_sim = float(np.abs(sim - img).mean())
    check(f"align_poly ≥ similitude ({e_poly:.4f} vs {e_sim:.4f})", e_poly < e_sim)


def test_differential():
    import astroforge.core.astrometry as AA
    rng = np.random.default_rng(61)
    n = 300
    center = (156.25, 19.75)
    xs = rng.uniform(50, n - 50, 7)
    ys = rng.uniform(50, n - 50, 7)
    mags = np.array([8.0] + list(rng.uniform(7.0, 9.0, 6)))
    cat = TMP + "/cat_dph.csv"
    dec_c = center[1] + (n / 2.0 - ys) / 3600.0
    ra_c = center[0] + (xs - n / 2.0) / 3600.0 / np.cos(np.deg2rad(center[1]))
    with open(cat, "w") as f:
        f.write("ra,dec,mag,bp,rp\n")
        for r_, d_, m_ in zip(ra_c, dec_c, mags):
            f.write(f"{r_:.6f},{d_:.6f},{m_:.3f},{m_ + 0.3:.3f},{m_ - 0.3:.3f}\n")

    def build(transp, seed):
        rng2 = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:n, 0:n]
        img = rng2.random((n, n)).astype(np.float32) * 0.02
        for x, y, m in zip(xs, ys, mags):
            fl = 0.8 * transp * 10 ** (-0.4 * (m - 7.0))  # pic ~0.8 pour m=7
            img += fl * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / 4.5).astype(np.float32)
        return img
    img1 = build(1.0, 1)
    res = AA.solve_internal(img1, cat, center[0], center[1], radius_deg=0.5,
                            focal_mm=1000.0, pixel_um=4.86)
    t1 = AA.differential_photometry(img1, res["wcs"], cat,
                                    target_ra=float(ra_c[0]),
                                    target_dec=float(dec_c[0]), n_comp=5)
    img2 = build(0.7, 1)
    t2 = AA.differential_photometry(img2, res["wcs"], cat,
                                    target_ra=float(ra_c[0]),
                                    target_dec=float(dec_c[0]), n_comp=5)
    d1, d2 = t1.meta["dmag"], t2.meta["dmag"]
    dm_truth = float(mags[0] - np.mean(sorted(mags[1:])[:5]))  # 5 comp. les plus brillantes
    check(f"photom. diff. : dmag {d1:+.3f} (vérité {dm_truth:+.3f}), stable ({d2:+.3f})",
          abs(d1 - dm_truth) < 0.35 and abs(d1 - d2) < 0.05 and t1.meta["n_comp"] >= 4)
    try:  # cible hors champ : erreur explicite, jamais une étoile au hasard
        AA.differential_photometry(img1, res["wcs"], cat, target_ra=center[0] + 0.2,
                                   target_dec=center[1] + 0.2)
        far = False
    except RuntimeError as e:
        far = "introuvable" in str(e)
    check("photom. diff. : cible hors champ refusée", far)




def test_regressions_audit():
    """Non-régression des bugs de l'audit du 14/10/2026 (vérité terrain)."""
    import struct
    import zlib
    from astropy.io import fits
    import astroforge.core.image_io as IO
    import astroforge.core.calibration as CAL
    import astroforge.core.finishing as FIN
    from astroforge.core import display, xisf
    from astroforge.core.equipment import parse_coord
    yy, xx = np.mgrid[0:128, 0:128]

    def field(x, y):
        return sum(np.exp(-((xx - x - ox) ** 2 + (yy - y - oy) ** 2) / 4.0)
                   for ox, oy in [(0, 0), (20, 30), (-30, 10), (15, -25)]).astype(np.float32)

    # C1 + M3 : sens et précision sous-pixel de l'alignement par phase
    ref = field(60, 60)
    errs = []
    for sx, sy in [(4, 0), (0, -7), (0.3, 0), (2.5, 1.5), (-1.2, 0.4)]:
        mov = field(60 + sx, 60 + sy)
        dx, dy, _ = R.phase_corr(ref, mov)
        errs.append(np.hypot(dx + sx, dy + sy))
        al = R.apply_shift(mov, dx, dy)
        errs.append(np.hypot(*R.phase_corr(ref, al)[:2]))
    check(f"C1/M3 alignement phase : sens et sous-pixel (erreur max {max(errs):.3f} px)",
          max(errs) < 0.06)
    rng = np.random.default_rng(0)
    frames = [field(60 + dx, 60 + dy) for dx, dy in rng.uniform(-3, 3, (8, 2))]
    stk = S.stack(np.stack(R.align_stack(frames[4], frames)), "Moyenne")
    check(f"C1 empilement aligné net (pic {stk.max():.2f} / {frames[4].max():.2f})",
          stk.max() > 0.9 * frames[4].max())
    rgb = np.stack([field(62, 60), field(60, 60), field(60, 60)], -1)
    fixed = R.align_channels(rgb)
    check("C1 alignement des canaux couleur corrige le décalage R",
          np.hypot(*R.phase_corr(fixed[..., 1], fixed[..., 0])[:2]) < 0.1)

    # C2 : matrices de Bayer (scène rouge pure)
    ok = True
    for pat, (ry, rx) in {"RGGB": (0, 0), "BGGR": (1, 1), "GRBG": (0, 1), "GBRG": (1, 0)}.items():
        cfa = np.zeros((32, 32), np.float32)
        cfa[ry::2, rx::2] = 1.0
        m = IO.debayer(cfa, pat)[4:-4, 4:-4].mean((0, 1))
        ok &= m[0] > 0.9 and m[2] < 0.1
    check("C2 dématriçage : rouge reste rouge pour les 4 motifs", ok)

    # C3/C4 : FITS entiers de caméra (uint16 BZERO), int16, flottants ADU
    d16 = (np.arange(64 * 64).reshape(64, 64) % 4000 + 100).astype(np.uint16)
    p16 = os.path.join(TMP, "cam16.fits")
    fits.writeto(p16, d16, overwrite=True)
    a16 = IO.load_image(p16)
    pf = os.path.join(TMP, "adu.fits")
    fits.writeto(pf, d16.astype(np.float32), overwrite=True)
    af = IO.load_image(pf)
    out = os.path.join(TMP, "cam16_out.fits")
    IO.save_image(out, a16)
    back = fits.getdata(out)
    check("C3/C4 FITS uint16 (BZERO) lu et normalisé ÷65535, export non saturé",
          np.allclose(a16, d16 / 65535.0, atol=1e-6) and np.allclose(af, a16, atol=1e-6)
          and float(back.max()) < 0.1)
    fits.writeto(p16, d16, header=fits.Header([("BAYERPAT", "RGGB")]), overwrite=True)
    check("C6 motif de Bayer lu dans l'en-tête FITS", IO.load_cfa(p16)[1] == "RGGB")

    # C5 : bias + dark thermique
    bias = np.full((32, 32), 0.10, np.float32)
    mb = CAL.make_master_bias([bias] * 5)
    md = CAL.make_master_dark([bias + 0.02] * 5, mb)
    cal = CAL.calibrate(bias + 0.02 + 0.30, mb, md, None, cosmetic=False)
    check(f"C5 calibration bias + dark : {cal.mean():.3f} (attendu 0,300)",
          abs(float(cal.mean()) - 0.30) < 1e-4)

    # C6 : session OSC complète (CFA → calibration → dématriçage)
    sess = os.path.join(TMP, "session")
    for sub in ("lights", "darks", "flats"):
        os.makedirs(os.path.join(sess, sub), exist_ok=True)
    h = w = 48
    scene = np.zeros((h, w), np.float32)
    scene[0::2, 0::2] = 0.30  # rouge (RGGB)
    vign = np.ones((h, w), np.float32) * np.linspace(0.8, 1.0, w)[None]
    dark = np.full((h, w), 0.05, np.float32)
    for i in range(3):
        lt = scene * vign + dark
        lt[10, 10] = 0.95  # pixel chaud
        fits.writeto(os.path.join(sess, "lights", f"l{i}.fits"),
                     (np.clip(lt, 0, 1) * 65535).astype(np.uint16), overwrite=True)
        fits.writeto(os.path.join(sess, "darks", f"d{i}.fits"),
                     (dark * 65535).astype(np.uint16), overwrite=True)
        fits.writeto(os.path.join(sess, "flats", f"f{i}.fits"),
                     (0.5 * vign * 65535).astype(np.uint16), overwrite=True)
    masters = CAL.build_masters("", os.path.join(sess, "darks"), os.path.join(sess, "flats"))
    lights = sorted(os.path.join(sess, "lights", f) for f in os.listdir(os.path.join(sess, "lights")))
    imgs = CAL.calibrate_paths(lights, masters, bayer="RGGB")
    m = imgs[0][6:-6, 6:-6].mean((0, 1))
    check(f"C6 session OSC calibrée en CFA puis dématricée (R,G,B = {np.round(m, 3)})",
          imgs[0].shape == (h, w, 3) and m[0] > 0.2 and m[1] < 0.02 and m[2] < 0.02
          and float(np.ptp(imgs[0][6:-6, 6:-6, 0])) < 0.05)

    # M1 : rejet avec peu d'images
    a = np.full((5, 8, 8), 0.1, np.float32) + rng.normal(0, 0.005, (5, 8, 8)).astype(np.float32)
    a[0, 4, 4] = 1.0
    check("M1 sigma-clipping/winsorisé rejettent un pixel chaud sur 5 images",
          abs(float(S.stack(a, "Sigma-clipping")[4, 4]) - 0.1) < 0.02
          and abs(float(S.stack(a, "Winsorisé")[4, 4]) - 0.1) < 0.03)

    # M4/M5 : gradient et déconvolution en couleur
    y, x = np.mgrid[0:120, 0:160]
    col = np.stack([0.1 + x / 1600, 0.1 + y / 1600, 0.1 + 0 * x], -1).astype(np.float32)
    flat_, model = G.extract_gradient(col)
    bg = flat_.reshape(-1, 3).mean(0)
    check("M4 gradient coloré retiré canal par canal, fond neutre",
          model.shape == col.shape and max(float(np.ptp(flat_[..., c])) for c in range(3)) < 0.002
          and float(np.ptp(bg)) < 0.002)
    check("M5 Richardson-Lucy en couleur",
          RT.richardson_lucy(np.random.rand(32, 32, 3).astype(np.float32), 1.5, 3).shape == (32, 32, 3))

    # M6/M7 : pas de quantification 8 bits ; traces uniquement dans le masque
    faint = (0.02 + rng.normal(0, 0.003, (160, 160))).astype(np.float32)
    faint[80:82, 10:150] += 0.05
    clean = E.remove_satellite_trails(faint)
    mask = E.detect_trail_mask(faint)
    check("M6 satellites : trace atténuée, reste de l'image identique",
          float(clean[80:82, 30:130].mean()) < 0.035
          and np.array_equal(clean[~mask], faint[~mask]))
    m2 = np.zeros((160, 160), bool)
    m2[40, :] = True
    check("M7 suppression de traces avec masque fourni",
          E.remove_satellite_trails(faint, mask=m2).shape == faint.shape)
    dn = RT.denoise(faint, "Non-local means", 0.5)
    check(f"M6 NL-means 16 bits ({len(np.unique(dn))} niveaux distincts)",
          len(np.unique(dn)) > 1000 and dn.std() < faint.std())

    # M8 : palettes matplotlib
    g = np.linspace(0, 1, 256, dtype=np.float32)[None].repeat(4, 0)
    check("M8 palettes Inferno/Viridis distinctes de Solaire",
          not np.allclose(E.apply_palette(g, "Inferno"), E.apply_palette(g, "Solaire (orange)"))
          and not np.allclose(E.apply_palette(g, "Viridis"), E.apply_palette(g, "Inferno")))

    # M9 : XISF conforme (en-tête 16 octets, attachment, zlib+sh)
    data = rng.random((20, 30)).astype(np.float32)
    raw = data.tobytes()
    shuffled = np.frombuffer(raw, np.uint8).reshape(-1, 4).T.tobytes()
    comp = zlib.compress(shuffled)

    def hdr(off):
        return ('<?xml version="1.0" encoding="UTF-8"?><xisf version="1.0" '
                'xmlns="http://www.pixinsight.com/xisf"><Image geometry="30:20:1" '
                'sampleFormat="Float32" bounds="0:1" colorSpace="Gray" '
                f'location="attachment:{off}:{len(comp)}" '
                f'compression="zlib+sh:{len(raw)}:4"/></xisf>').encode()
    off = 4096
    xml = hdr(off)
    blob = b"XISF0100" + struct.pack("<II", len(xml), 0) + xml
    blob += b"\0" * (off - len(blob)) + comp
    px = os.path.join(TMP, "pi.xisf")
    open(px, "wb").write(blob)
    rd = xisf.read_xisf(px)
    pw = os.path.join(TMP, "af.xisf")
    xisf.write_xisf(pw, data)
    head = open(pw, "rb").read(64)
    (hl,) = struct.unpack("<I", head[8:12])
    check("M9 XISF PixInsight lu (zlib+sh) et écrit au format 1.0",
          np.allclose(rd, data, atol=1e-6) and head[16:21] == b"<?xml" and hl > 0
          and b"attachment:" in open(pw, "rb").read(16 + hl)
          and np.allclose(xisf.read_xisf(pw), data, atol=1e-6))

    # M10 : empilement depuis le disque (médiane, rejet) = empilement en mémoire
    paths = []
    stack_in = []
    for i in range(5):
        im = (0.2 + rng.normal(0, 0.01, (24, 24))).astype(np.float32)
        if i == 2:
            im[5, 5] = 1.0
        pth = os.path.join(TMP, f"sp{i}.tif")
        IO.save_image(pth, im)
        stack_in.append(IO.load_image(pth))
        paths.append(pth)
    ok = all(np.allclose(S.stack_paths(paths, m), S.stack(np.stack(stack_in), m), atol=1e-6)
             for m in ("Médiane", "Sigma-clipping", "Moyenne"))
    check("M10 stack_paths (memmap) identique à l'empilement en mémoire", ok)

    # M11 : étirement automatique non linéaire, sans écrêtage
    gal = (0.05 + 0.4 * np.exp(-((xx - 64) ** 2 + (yy - 64) ** 2) / 300)
           + rng.normal(0, 0.003, (128, 128))).astype(np.float32)
    st_f = FIN.autostretch(gal)
    st_d = display.autostretch(gal)
    check(f"M11 étirement auto : fond {np.median(st_d):.2f}, aucun pixel brûlé",
          float((st_f >= 0.999).mean()) == 0.0 and abs(float(np.median(st_d)) - 0.25) < 0.05
          and st_d.max() > 0.9)

    # M15/M16 : SER norme v3 (Bayer ColorID 8), projection mémoire, batch complet
    n_fr = 40
    body = np.zeros((n_fr, 16, 16), np.uint16)
    body[:, 0::2, 0::2] = 60000  # capteur RGGB, scène rouge
    hdr_ser = (b"LUCAM-RECORDER" + struct.pack("<iii", 0, 8, 0)
               + struct.pack("<iiii", 16, 16, 16, n_fr) + b"\0" * 120 + struct.pack("<qq", 0, 0))
    ps = os.path.join(TMP, "osc.ser")
    open(ps, "wb").write(hdr_ser + body.astype("<u2").tobytes())
    lazy, meta = IO.load_ser(ps, lazy=True)
    eager, _ = IO.load_ser(ps)
    rgb0 = lazy[0][4:-4, 4:-4].mean((0, 1))
    check("M16 SER Bayer (ColorID 8) dématricé, lecture projetée = lecture complète",
          isinstance(lazy, IO.SerFrames) and lazy.shape == (n_fr, 16, 16, 3)
          and meta["bayer"] == "RGGB" and rgb0[0] > 0.8 and rgb0[2] < 0.1
          and np.allclose(np.asarray(lazy), eager))
    pw_ser = os.path.join(TMP, "std.ser")
    IO.write_ser(pw_ser, eager[:3])
    sz = os.path.getsize(pw_ser)
    back, mb_ = IO.load_ser(pw_ser)
    check("SER écrit avec l'en-tête standard de 178 octets (lisible AS!/Registax)",
          sz == 178 + 3 * 16 * 16 * 3 * 2 and mb_["layout"] == "spec"
          and np.allclose(back, eager[:3], atol=1e-4))
    from astroforge.core import sequence as Q
    bdir, odir = os.path.join(TMP, "bimg"), os.path.join(TMP, "bout")
    os.makedirs(bdir, exist_ok=True)
    IO.save_image(os.path.join(bdir, "ok.tif"), gal)
    open(os.path.join(bdir, "casse.tif"), "wb").write(b"pas une image")
    res_b = Q.run_batch_image({"mode": "image", "ops": [{"op": "fin_saturation",
                                                         "params": {"amount": 1.0}}]},
                              bdir, odir)
    check("M15 batch : échecs rapportés (liste + rapport_batch.txt)",
          len(res_b) == 1 and len(res_b.errors) == 1
          and os.path.isfile(os.path.join(odir, "rapport_batch.txt")))

    # coordonnées, film stabilisé, chemins accentués
    check("saisie de coordonnées tolérante (°'\" / hms / d m s / décimal virgule)",
          abs(parse_coord("+19°48'30\"", False) - 19.808333) < 1e-5
          and abs(parse_coord("10h25m03s", True) - 156.2625) < 1e-6
          and abs(parse_coord("19d48m30s", False) - 19.808333) < 1e-5
          and abs(parse_coord("156,25", True) - 156.25) < 1e-9)
    from astroforge.core.animation import stabilized_frames
    jitter = [field(60 + dx, 60 + dy) for dx, dy in rng.uniform(-3, 3, (6, 2))]
    stab = stabilized_frames(jitter)
    resid = max(np.hypot(*R.phase_corr(stab[3], stab[i])[:2]) for i in range(len(stab)))
    check(f"film stabilisé : images recalées (résidu {resid:.3f} px)", resid < 0.1)
    acc = os.path.join(TMP, "Données été")
    os.makedirs(acc, exist_ok=True)
    pa = os.path.join(acc, "étoile.tif")
    IO.save_image(pa, gal)
    check("TIFF : chemin accentué en écriture/lecture",
          np.allclose(IO.load_image(pa), gal, atol=2e-5))


def test_step_commands():
    """Vrais clics Qt, références mémoire, annulation et visibilité hors écran."""
    import time
    import tempfile
    from unittest.mock import patch
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QPushButton, QTabWidget
    from astroforge.gui.main_window import MainWindow
    from astroforge.gui.workers import Job
    from astroforge.gui import loader
    from astroforge.core.image_io import save_image
    app = QApplication.instance() or QApplication([])
    mw = MainWindow()
    mw.show()
    tabs = mw.win.findChild(QTabWidget, "tabs")
    p, sky = mw.page_planet, mw.page_sky

    def drain(page):
        deadline = time.monotonic() + 15
        while page._jobs or page.live._jobs:
            app.processEvents()
            QTest.qWait(5)
            if time.monotonic() > deadline:
                raise AssertionError("Délai de traitement dépassé")
        app.processEvents()

    def click(page, name, cancel=False):
        btn = page._command_buttons[name][int(cancel)]
        page.ui.findChild(object, "scrollParams").ensureWidgetVisible(btn)
        app.processEvents()
        QTest.mouseClick(btn, Qt.LeftButton)
        drain(page)

    for index, page in ((1, p), (0, sky)):
        tabs.setCurrentIndex(index)
        for width, height in ((1280, 820), (1024, 700)):
            mw.win.resize(width, height)
            app.processEvents()
            scroll = page.ui.findChild(object, "scrollParams")
            reachable = True
            for name, pair in page._command_buttons.items():
                for btn in pair:
                    if scroll.isAncestorOf(btn):
                        scroll.ensureWidgetVisible(btn, 0, 0)
                        app.processEvents()
                        pos = btn.mapTo(scroll.viewport(), QPoint(0, 0))
                        reachable &= (btn.isVisible() and btn.height() >= 25
                                      and pos.x() >= 0 and pos.y() >= 0
                                      and pos.x() + btn.width() <= scroll.viewport().width()
                                      and pos.y() + btn.height() <= scroll.viewport().height())
                    else:
                        reachable &= btn.isVisible() and btn.height() >= 25
            check(f"commandes {index} accessibles à {width}×{height}", reachable)
        check(f"annulations {index} désactivées sans historique",
              all(not pair[1].isEnabled() for pair in page._command_buttons.values()))

    tabs.setCurrentIndex(1)
    original = synth_frame(seed=70)
    frames = [original, synth_frame(shift=(2, -1), seed=71),
              synth_frame(shift=(-1, 2), seed=72)]
    p.frames = frames
    p._o("spnNAlign").setRange(1, 3)
    p._o("spnNAlign").setValue(2)
    p._o("cmbRoiMode").setCurrentText("Image entière")
    p._o("cmbQualityMode").setCurrentText("Luminosité")
    p._o("cmbAlignKindP").setCurrentText("Global (une translation)")
    p.ui.findChild(object, "scrollParams").ensureWidgetVisible(p._o("btnRank"))
    app.processEvents()
    check("bouton historique Lancer le tri visible et non écrasé",
          p._o("btnRank").isVisible() and p._o("btnRank").height() >= 25)
    snapshot = p._snapshot()
    check("snapshot planétaire : pixels conservés par référence",
          snapshot["attrs"]["frames"] is not frames
          and snapshot["attrs"]["frames"][0] is original)
    click(p, "grpRank")
    check("Lancer tri : scores et classement calculés",
          p.order is not None and len(p.order) == 3 and p.scores.shape == (3,))
    check("tri : première image visible malgré curseur initial zéro",
          p.pane._before is p.frames[int(p.order[0])]
          and p.pane.canvasBefore.isVisible() and p.pane.cmbMode.currentIndex() == 1
          and p._o("sldRankBrowse").isEnabled())
    QTest.mouseClick(p._o("btnNextRank"), Qt.LeftButton)
    check("navigation image suivante triée",
          p._o("sldRankBrowse").value() == 1
          and p.pane._before is p.frames[int(p.order[1])])
    QTest.mouseClick(p._o("btnPrevRank"), Qt.LeftButton)
    check("navigation image précédente triée", p._o("sldRankBrowse").value() == 0)
    click(p, "grpNAlign")
    check("sélection N meilleures appliquée", np.array_equal(p.selected, p.order[:2]))
    click(p, "grpAlign")
    check("méthode alignement validée", p.align_settings["align_mode"].startswith("Global"))
    with patch("astroforge.core.planetary.measure_quality",
               side_effect=AssertionError("Le tri validé doit être réutilisé")):
        click(p, "grpStackP")
    result = p.result
    check("empilement réutilise le tri et publie pleine résolution",
          result is not None and result.shape == original.shape and p.shared["image"] is result)
    click(p, "grpRun")
    check("étape finale : panneau Après visible", p.pane.canvasAfter.isVisible())
    p._o("sldW1").setValue(160)
    click(p, "grpWave")
    check("ondelettes Lancer modifie le résultat", not np.allclose(p.result, result))
    click(p, "grpWave", cancel=True)
    check("ondelettes Annuler restaure la même référence", p.result is result)
    click(p, "grpRank")
    check("nouveau tri invalide empilement et sélection",
          p.result is None and p.selected is None and p.shared["image"] is None)
    click(p, "grpRank", cancel=True)
    check("Annuler nouveau tri restaure résultat précédent", p.result is result)
    click(p, "grpAlign", cancel=True)
    check("Annuler alignement restaure sélection et invalide descendants",
          p.align_settings is None and p.selected is not None and p.result is None
          and "grpStackP" not in p._history)
    click(p, "grpNAlign", cancel=True)
    # Une seule dernière opération par étape : le second tri a remplacé son historique.
    p.order = p.scores = None
    p._sync_rank_navigation()
    click(p, "grpRank")
    click(p, "grpRank", cancel=True)
    check("Annuler tri initial désactive navigation et conserve source",
          p.order is None and p.scores is None and not p._o("sldRankBrowse").isEnabled()
          and p.frames[0] is original)

    # Annulation pendant un vrai QThread : aucun résultat tardif ne doit être publié.
    def delayed(progress=None):
        time.sleep(.08)
        return np.zeros_like(original)
    def launch_delayed():
        p._job = p._new_job(delayed)
        p._connect_done(p._job, p._commit_image)
        p._job.start()
    p._commit_image(original)
    p._execute("grpWave", launch_delayed)
    check("traitement actif : Lancer verrouillé, Annuler disponible",
          not p._command_buttons["grpWave"][0].isEnabled()
          and p._command_buttons["grpWave"][1].isEnabled())
    QTest.mouseClick(p._command_buttons["grpWave"][1], Qt.LeftButton)
    drain(p)
    check("annulation active : résultat tardif rejeté et boutons réactivés",
          p.result is original and p.shared["image"] is original
          and p._command_buttons["grpWave"][0].isEnabled())
    p._execute("grpWave", lambda: (_ for _ in ()).throw(ValueError("échec attendu")))
    check("erreur synchrone : état restauré", p.result is original)

    tabs.setCurrentIndex(0)
    with tempfile.TemporaryDirectory(prefix="astroforge-commandes-") as directory:
        paths = [os.path.join(directory, f"light{i}.fits") for i in range(3)]
        for path, image in zip(paths, frames):
            save_image(path, image)
        sky.state["light_paths"] = paths
        sky.ui.findChild(object, "listLight").addItems(paths)
        sky._cmb("cmbStackMethod").setCurrentText("Moyenne")
        sky._spn("spnDeconvIters").setValue(0)
        click(sky, "grpImages")
        loaded = sky.state["images"]
        check("Ciel profond chargement indépendant", len(loaded) == 3 and sky.pane._before is loaded[0])
        ss = sky._snapshot()
        check("snapshot ciel profond : tableaux non copiés", ss["attrs"]["state"]["images"][0] is loaded[0])
        click(sky, "grpCalib")
        calibrated = sky.state["images"]
        check("calibration indépendante : images produites", len(calibrated) == 3)
        click(sky, "grpAlign")
        aligned = sky.state["images"]
        check("alignement indépendant : séries produites", len(aligned) == 3)
        click(sky, "grpStack")
        stacked = sky.state["image"]
        check("empilement indépendant : image publiée", stacked is not None and sky.shared["image"] is stacked)
        click(sky, "grpTrait")
        check("traitement indépendant fond et débruitage", sky.state["image"] is not stacked and sky.state["background_model"] is not None)
        click(sky, "grpTrait", cancel=True)
        check("Annuler traitement restaure image empilée", sky.state["image"] is stacked)
        click(sky, "grpAlign", cancel=True)
        check("Annuler alignement invalide empilement et traitement",
              sky.state["images"][0] is calibrated[0] and sky.state["image"] is None
              and "grpStack" not in sky._history)
        click(sky, "grpCalib", cancel=True)
        check("Annuler calibration restaure images chargées", sky.state["images"][0] is loaded[0])
        sky._cbx("chkDefects").setChecked(True)
        click(sky, "grpDefects")
        check("correction défauts indépendante conserve série", len(sky.state["images"]) == 3)
        click(sky, "grpDefects", cancel=True)
        check("Annuler défauts restaure références", sky.state["images"][0] is loaded[0])
        sky._after_edit(np.stack([original, original * .8, original * .6], axis=-1))
        rgb = sky.state["image"]
        click(sky, "grpColor")
        check("calibration couleur Lancer appliquée", sky.state["image"] is not rgb)
        click(sky, "grpColor", cancel=True)
        check("calibration couleur Annuler restaurée", sky.state["image"] is rgb)
        sky.state["hdr_images"] = paths[:2]
        click(sky, "grpHDR")
        check("HDR Lancer fusionne les expositions", sky.state["image"].shape == original.shape)
        click(sky, "grpHDR", cancel=True)
        check("HDR Annuler restaure image précédente", sky.state["image"] is rgb)
        with patch("astroforge.ai.manager.get_ai_manager") as manager:
            manager.return_value.run.return_value = original * .5
            click(sky, "grpIA")
            check("IA Lancer appelle le fournisseur sélectionné", manager.return_value.run.call_count == 1)
        click(sky, "grpIA", cancel=True)
        check("IA Annuler restaure image précédente", sky.state["image"] is rgb)
        click(sky, "grpPipe")
        check("processus complet avec paramètres par défaut fusionnés", sky.state["image"] is not None and len(sky.state["images"]) == 3)
        click(sky, "grpPipe", cancel=True)
        check("processus Annuler restaure état précédent", sky.state["image"] is rgb)
        sky._execute("grpImages", sky._clear_lights)
        check("vider sources invalide image et séries", not sky.state["light_paths"] and sky.state["image"] is None and not sky.state["images"])
        click(sky, "grpImages", cancel=True)
        check("Annuler vider restaure chemins, listes et résultat",
              sky.state["light_paths"] == paths and sky.state["image"] is rgb
              and sky.ui.findChild(object, "listLight").count() == 3)

    # Commandes restantes : connexion et restauration, sans dépendance réseau/modèles.
    tabs.setCurrentIndex(1)
    with tempfile.TemporaryDirectory(prefix="astroforge-source-") as directory:
        source_path = os.path.join(directory, "source.ser")
        write_ser(source_path, np.stack(frames))
        old_frames = p.frames
        old_result = p.result
        p._o("cmbSourceType").setCurrentIndex(0)
        with patch("astroforge.gui.page_planetary.QFileDialog.getOpenFileName",
                   return_value=(source_path, "")):
            click(p, "grpSource")
        check("source Lancer : SER chargé et résultat invalidé",
              len(p.frames) == 3 and p.result is None and p.order is None)
        click(p, "grpSource", cancel=True)
        check("source Annuler : série et résultat précédents restaurés",
              p.frames[0] is old_frames[0] and p.result is old_result)
    p._commit_image(original)
    p._o("chkFalseColor").setChecked(True)
    click(p, "grpFalseColor")
    check("fausses couleurs Lancer : trois canaux", p.result.shape == original.shape + (3,))
    click(p, "grpFalseColor", cancel=True)
    check("fausses couleurs Annuler : image mono restaurée", p.result is original)
    rgb = np.stack([original, original, original], axis=-1)
    p._commit_image(rgb)
    p._o("chkCA").setChecked(True)
    p._o("sldCAR").setValue(4)
    click(p, "grpCA")
    check("correction CA Lancer appliquée", p.result is not rgb)
    click(p, "grpCA", cancel=True)
    check("correction CA Annuler restaurée", p.result is rgb)
    with patch("astroforge.ai.manager.get_ai_manager") as manager:
        manager.return_value.run.return_value = rgb * .8
        click(p, "grpIA")
        check("IA planétaire Lancer travaille sur résultat courant",
              manager.return_value.run.call_args.args[1] is rgb)
    click(p, "grpIA", cancel=True)
    check("IA planétaire Annuler restaure résultat", p.result is rgb)
    with patch("astroforge.gui.page_planetary.QFileDialog.getSaveFileName",
               return_value=(TMP + "/test-command-film.mp4", "")), \
         patch("astroforge.gui.page_planetary.export_video") as exporter:
        click(p, "grpAnim")
        check("film Lancer appelle exporteur", exporter.call_count == 1)
    click(p, "grpAnim", cancel=True)
    check("film Annuler restaure état en mémoire", p.result is rgb)
    with patch("astroforge.gui.page_planetary.QFileDialog.getExistingDirectory",
               side_effect=[TMP + "", TMP + ""]), \
         patch("astroforge.core.sequence.run_batch_ser_chain", return_value=[]) as batch:
        click(p, "grpSeqP")
        check("batch SER Lancer appelle moteur existant", batch.call_count == 1)
    click(p, "grpSeqP", cancel=True)
    check("batch SER Annuler restaure état en mémoire", p.result is rgb)
    tabs.setCurrentIndex(0)
    sky._after_edit(original)
    solution = {"wcs": object(), "method": "test", "scale_arcsec_pix": 1.,
                "center_ra": 10., "center_dec": 20., "n_matched": 12, "rms_arcsec": .1}
    with patch("astroforge.core.astrometry.solve_image", return_value=solution) as solver:
        click(sky, "grpAstro")
        check("astrométrie Lancer publie WCS", solver.call_count == 1 and sky.state["wcs"] is solution["wcs"])
    click(sky, "grpAstro", cancel=True)
    check("astrométrie Annuler supprime WCS calculé", "wcs" not in sky.state and sky.state["image"] is original)
    with patch("astroforge.gui.page_sky.QMessageBox.critical"):
        def failing(progress=None):
            raise ValueError("échec asynchrone attendu")
        sky._execute("grpTrait", lambda: sky._start_job(failing, sky._after_edit, sky.ui))
        drain(sky)
    check("erreur asynchrone : état et boutons restaurés",
          sky.state["image"] is original and sky._command_buttons["grpTrait"][0].isEnabled())
    progress_seen = []
    def cooperative(progress=None):
        for i in range(100):
            time.sleep(.005)
            progress(i, 100, "test annulation")
    worker = Job(cooperative)
    worker.progress.connect(lambda *args: progress_seen.append(args))
    worker.start()
    QTest.qWait(30)
    worker.cancel()
    worker.wait(2000)
    app.processEvents()
    check("annulation coopérative interrompt boucle de progression",
          worker.cancelled and not worker.isRunning() and len(progress_seen) < 100)
    with patch.object(loader, "_generate", side_effect=RuntimeError("repli forcé")):
        from astroforge.gui.page_planetary import PagePlanetary
        from astroforge.gui.page_sky import PageSky
        fallback_p, fallback_s = PagePlanetary(), PageSky()
        check("QUiLoader direct : toutes commandes des deux onglets présentes",
              len(fallback_p._command_buttons) == 13
              and len(fallback_s._command_buttons) == 12
              and fallback_p._o("btnCancel_grpRank") is not None)

    # Cache périmé même si sa date est plus récente que le XML ; mode gelé = repli .ui.
    with tempfile.TemporaryDirectory(prefix="astroforge-ui-") as directory:
        ui_dir, gen_dir = os.path.join(directory, "ui"), os.path.join(directory, "generated")
        os.makedirs(ui_dir)
        os.makedirs(gen_dir)
        source = os.path.join(ui_dir, "test.ui")
        with open(source, "w") as output:
            output.write('<ui version="4.0"><class>CacheTest</class><widget class="QWidget" name="CacheTest"><layout class="QVBoxLayout" name="layout"><item><widget class="QPushButton" name="fresh"><property name="text"><string>Lancer</string></property></widget></item></layout></widget></ui>')
        generated = os.path.join(gen_dir, "ui_CacheTest.py")
        with open(generated, "w") as output:
            output.write("# périmé\n")
        os.utime(generated, (time.time() + 600, time.time() + 600))
        with patch.object(loader, "UI_DIR", ui_dir), patch.object(loader, "GEN_DIR", gen_dir):
            widget = loader.load_ui("test.ui")
            check("cache généré périmé détecté indépendamment des dates", widget.findChild(QPushButton, "fresh") is not None)
            with open(generated, "w") as output:
                output.write("# périmé dans bundle\n")
            with patch.object(sys, "frozen", True, create=True):
                widget = loader.load_ui("test.ui")
                check("bundle périmé : repli .ui fonctionnel", widget.findChild(QPushButton, "fresh") is not None)
    mw.win.close()
    drain(p)
    drain(sky)


def test_planetary_tone_layers():
    """Régressions 1.9.5 : nombres, flux différé, vrais widgets et commandes."""
    import time
    import tempfile
    from unittest.mock import patch
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QTabWidget
    from astroforge.core import planetary as PL, sequence as Q
    from astroforge.processing import profiles as P
    from astroforge.gui.main_window import MainWindow
    from astroforge import __version__
    check("version 2.0.0", __version__ == "2.0.0")
    a = np.linspace(0, 1, 49, dtype=np.float32).reshape(7, 7)
    check("tonalité neutre mono : identité numérique", np.array_equal(PL.adjust_tone(a), a))
    check("luminosité : ajout et écrêtage", np.allclose(PL.adjust_tone(a, .2), np.clip(a + .2, 0, 1)))
    check("contraste : pivot 0,5 et écrêtage", np.allclose(PL.adjust_tone(a, contrast=2), np.clip((a - .5) * 2 + .5, 0, 1)))
    curve = [0, .1, .8, .9, 1]
    rgb = np.array([[[-.1, .125, .375], [.625, .875, 1.2]]], np.float32)
    out = PL.adjust_tone(rgb, curve=curve)
    check("courbe RGB : interpolation, canaux conservés, bornes", out.dtype == np.float32
          and out.shape == rgb.shape and np.allclose(out, [[[0, .05, .45], [.85, .95, 1]]]))
    check("tonalité RGB neutre : identité", np.array_equal(PL.adjust_tone(np.clip(rgb, 0, 1)), np.clip(rgb, 0, 1)))
    check("courbe inversée éditable (pas seulement gamma)", np.allclose(PL.adjust_tone(a, curve=[1, .75, .5, .25, 0]), 1 - a))
    try:
        PL.tone_settings(curve=[0, 1])
        valid = False
    except ValueError:
        valid = True
    check("courbe malformée rejetée", valid)
    originals = [a, a * .8]
    view = PL.ToneFrames(originals, {"brightness": .2})
    progress = []
    check("validation de toutes les frames sans stockage de pixels", view.validate(lambda *args: progress.append(args)) is view
          and progress[-1][:2] == (2, 2) and view.originals[0] is a
          and not any(isinstance(v, np.ndarray) for v in vars(view).values()))
    check("vue : index/slice/crop restent cohérents", np.allclose(view[1], originals[1] + .2)
          and np.allclose(view[1:][0], view[1])
          and np.allclose(PL.ToneFrames(originals, view.params, (1, 2, 3, 2))[0], view[0][2:4, 1:4]))
    check("moyenne différée en flux", np.allclose(PL.mean_sample(view), np.mean([view[0], view[1]], axis=0)))
    def cancel(done, total, label=""):
        if done == 1:
            raise InterruptedError("test")
    try:
        view.validate(cancel)
        cancelled = False
    except InterruptedError:
        cancelled = True
    check("validation annulation coopérative", cancelled and np.array_equal(a, originals[0]))

    img = synth_frame(seed=85, noise=.05)
    layers = W.atrous_decompose(img, 6)
    check("API ondelettes defaults compatible, gains omis acceptés", np.allclose(W.registax_enhance(img), img, atol=1e-7)
          and np.array_equal(W.registax_enhance(img, [1.] * 7),
                             W.registax_enhance(img, [1.] * 7, layer_denoise=[0.] * 6, layer_sharpen=[0.] * 6)))
    legacy = W.atrous_recompose(W.atrous_decompose(W.wavelet_denoise(img, 4, .4), 6), [1.] * 7)
    check("API global historique préservée ciel profond", np.array_equal(W.registax_enhance(img, [1.] * 7, .4), legacy))
    for i in range(6):
        sh = [0.] * 6
        sh[i] = .7
        changed = W.enhance_layers(layers, sharpen=sh)
        result = W.registax_enhance(img, layer_sharpen=sh)
        check(f"netteté couche {i + 1} : seul détail ciblé amplifié", np.allclose(changed[i], layers[i] * 1.7)
              and all(changed[j] is layers[j] for j in range(7) if j != i)
              and np.max(np.abs(result - img)) > 1e-7
              and np.allclose(result - img, layers[i] * .7, atol=1e-7))
        dn = [0.] * 6
        dn[i] = .5
        changed = W.enhance_layers(layers, denoise=dn)
        check(f"débruitage couche {i + 1} : seul détail ciblé atténué", np.abs(changed[i]).sum() < np.abs(layers[i]).sum()
              and all(changed[j] is layers[j] for j in range(7) if j != i))
    flat_noise = .5 + np.random.default_rng(44).normal(0, .04, (64, 64)).astype(np.float32)
    dn = W.registax_enhance(flat_noise, layer_denoise=[.6] * 6)
    check("débruitage par couches réduit bruit et conserve moyenne", dn.std() < flat_noise.std() * .5
          and abs(float(dn.mean() - flat_noise.mean())) < .002)
    color = np.stack([img, img * .7, img * .4], axis=-1)
    combined = W.registax_enhance(color, layer_denoise=[.3] * 6, layer_sharpen=[.5] * 6)
    separate = np.stack([W.registax_enhance(color[..., c], layer_denoise=[.3] * 6,
                                          layer_sharpen=[.5] * 6) for c in range(3)], axis=-1)
    check("ondelettes RGB : absence de mélange entre canaux", np.allclose(combined, separate, atol=1e-7))

    app = QApplication.instance() or QApplication([])
    mw = MainWindow()
    mw.show()
    mw.win.findChild(QTabWidget, "tabs").setCurrentIndex(1)
    p = mw.page_planet
    def drain():
        deadline = time.monotonic() + 15
        while p._jobs or p.live._jobs:
            app.processEvents()
            QTest.qWait(5)
            assert time.monotonic() < deadline
        app.processEvents()
    def apply():
        QTest.mouseClick(p._o("btnLaunch_grpTone"), Qt.LeftButton)
        drain()
    frames = [img, img * .8, img * .6]
    p.frames = PL.ToneFrames(frames)
    p.original_frames = p.frames.originals
    p._o("sldBrowse").setRange(0, 2)
    for width, height in ((1280, 820), (1024, 700)):
        mw.win.resize(width, height)
        QTest.qWait(30)
        app.processEvents()
        scroll_r = p._o("scrollRightP")
        group = p._o("grpTone")
        pos = group.mapTo(mw.win, QPoint(0, 0))
        right = p.pane.mapTo(mw.win, QPoint(p.pane.width(), 0)).x()
        scroll_r.ensureWidgetVisible(group, 10, 10)
        app.processEvents()
        gpos = group.mapTo(scroll_r.viewport(), QPoint(0, 0))
        check(f"module tonalité à droite et défilable à {width}×{height}",
              mw.win.width() == width and mw.win.height() == height
              and pos.x() >= right
              and gpos.x() >= 0 and gpos.y() >= 0
              and gpos.x() + group.width() <= scroll_r.viewport().width()
              and gpos.y() + group.height() <= scroll_r.viewport().height())
        tabs_tone = p._o("tabsTone")
        check(f"onglets tonalité au même endroit à {width}×{height}",
              tabs_tone.count() == 3 and tabs_tone.parentWidget() is group
              and tabs_tone.tabText(0) == "Courbe"
              and tabs_tone.tabText(1) == "Contraste automatique"
              and tabs_tone.tabText(2) == "Luminosité automatique")
        visible = all(p._o(n).isVisible() for n in
                      ("spnBrightness", "spnContrast", "btnTonePreview",
                       "btnToneReset", "btnLaunch_grpTone", "btnCancel_grpTone",
                       "btnUndoP"))
        for index, names in ((0, ("toneCurve",)),
                             (1, ("spnAutoClip", "spnAutoBlack", "spnAutoWhite",
                                  "btnAutoCompute", "lblAutoInfo")),
                             (2, ("spnAutoGamma", "btnAutoGamma"))):
            tabs_tone.setCurrentIndex(index)
            app.processEvents()
            visible &= all(p._o(n).isVisible() and p._o(n).width() > 20
                           for n in names)
        tabs_tone.setCurrentIndex(0)
        app.processEvents()
        check(f"onglets tonalité : widgets visibles à {width}×{height}", visible)
        scroll = p._o("scrollParams")
        reachable = True
        for i in range(1, 7):
            for n in (f"spnWDn{i}", f"spnWSh{i}"):
                widget = p._o(n)
                # Qt peut viser le microFocusRect de la zone de saisie plutôt
                # que le widget entier : marge suffisante pour ses bordures.
                scroll.ensureWidgetVisible(widget, 24, 80)
                QTest.qWait(10)
                app.processEvents()
                q = widget.mapTo(scroll.viewport(), QPoint(0, 0))
                inside = (widget.isVisible() and q.x() >= 0 and q.y() >= 0
                          and q.x() + widget.width() <= scroll.viewport().width()
                          and q.y() + widget.height() <= scroll.viewport().height())
                if not inside:
                    print("Géométrie inaccessible :", n, q, widget.size(), scroll.viewport().size())
                reachable &= inside
        check(f"douze contrôles ondelettes accessibles à {width}×{height}", reachable and p._o("sldWDn") is None)
        screenshots = os.environ.get("ASTROFORGE_UI_SCREENSHOTS")
        if screenshots:
            os.makedirs(screenshots, exist_ok=True)
            assert mw.win.grab().save(os.path.join(screenshots, f"planetary-{width}x{height}.png"))
    p._o("spnBrightness").setValue(.15)
    p._execute("grpTone", p._preview_tone)
    drain()
    check("aperçu non destructif et tonalité visible sans étirement", p.frames[0] is img
          and p.result is None and not p.pane.chkStretch.isChecked()
          and np.allclose(p.pane._after, PL.adjust_tone(img, .15)))
    p.scores = np.ones(3)
    p.order = p.selected = np.arange(3)
    p.align_settings = {"align_mode": "Global"}
    p.roi_used = (0, 0, 128, 128)
    p._commit_image(img)
    ranked_order, ranked_scores = p.order, p.scores
    apply()
    p._cancel_command("grpTone")
    drain()
    check("annuler application restaure classement et résultat par références", p.order is ranked_order
          and p.scores is ranked_scores and p.result is img and p.frames[0] is img)
    apply()
    check("application toute séquence et originaux par références", isinstance(p.frames, PL.ToneFrames)
          and p.original_frames[0] is img and all(np.allclose(p.frames[i], PL.adjust_tone(f, .15)) for i, f in enumerate(frames)))
    check("tonalité invalide tri sélection alignement résultat et partage", all(getattr(p, n) is None for n in
          ("scores", "order", "selected", "align_settings", "roi_used", "result", "stack_base")) and p.shared["image"] is None)
    p._o("spnNAlign").setValue(3)
    p._o("cmbRoiMode").setCurrentText("Image entière")
    p._o("cmbAlignKindP").setCurrentText("Global (une translation)")
    with patch("astroforge.core.planetary.measure_quality", wraps=PL.measure_quality) as measure:
        p._execute("grpStackP")
        drain()
        check("empilement reclasse la séquence ajustée et résultat changé", measure.call_count == 1
              and isinstance(measure.call_args.args[0], PL.ToneFrames)
              and p.result is not None and p.result.mean() > img.mean() + .05)
    apply()
    check("réapplication non cumulative", np.allclose(p.frames[0], PL.adjust_tone(img, .15)))
    QTest.mouseClick(p._o("btnToneReset"), Qt.LeftButton)
    drain()
    check("réinitialisation restitue tous originaux et contrôles", all(p.frames[i] is f for i, f in enumerate(frames))
          and p._tone_params() == PL.tone_settings() and p.applied_tone == PL.tone_settings())
    QTest.mouseClick(p._o("btnCancel_grpTone"), Qt.LeftButton)
    drain()
    check("annuler reset restaure séquence ajustée", np.allclose(p.frames[0], PL.adjust_tone(img, .15)) and p.applied_tone["brightness"] == .15)
    before = p.frames
    p._o("spnBrightness").setValue(.25)
    real_validate = PL.ToneFrames.validate
    def slow(view, progress=None):
        time.sleep(.08)
        return real_validate(view, progress)
    with patch.object(PL.ToneFrames, "validate", slow):
        p._execute("grpTone")
        check("tonalité occupée : contrôles synchronisés", not p._o("spnBrightness").isEnabled()
              and not p._o("toneCurve").isEnabled() and not p._o("spnWDn1").isEnabled()
              and not p._o("btnToneReset").isEnabled() and p._o("btnCancel_grpTone").isEnabled())
        QTest.mouseClick(p._o("btnCancel_grpTone"), Qt.LeftButton)
        drain()
    check("annulation tonalité active : vue précédente sans résultat tardif", p.frames is before
          and p.applied_tone["brightness"] == .15 and p._o("spnBrightness").isEnabled())
    # Les lots rejouent le validé, pas les valeurs encore en édition.
    check("collecte batch : tonalité appliquée, pas brouillon", p._collect_settings()["tone"]["brightness"] == .15
          and p._tone_params()["brightness"] == .25)
    with patch("astroforge.gui.page_planetary.QFileDialog.getSaveFileName", return_value=(TMP + "/tone-film.mp4", "")), \
         patch("astroforge.gui.page_planetary.export_video") as exporter:
        p._o("chkAutoCrop").setChecked(False)
        p._execute("grpAnim")
        drain()
        exported = exporter.call_args.args[0]
        check("film reçoit toutes images ajustées", isinstance(exported, PL.ToneFrames)
              and all(np.allclose(exported[i], PL.adjust_tone(f, .15)) for i, f in enumerate(frames)))
    with patch("astroforge.gui.page_planetary.QFileDialog.getSaveFileName", return_value=(TMP + "/tone-crop.mp4", "")), \
         patch("astroforge.core.animation.common_crop", return_value=(8, 9, 64, 63)), \
         patch("astroforge.gui.page_planetary.export_video") as exporter:
        from astroforge.core.animation import StabilizedFrames
        import astroforge.core.registration as REG
        p._o("chkAutoCrop").setChecked(True)
        p._execute("grpAnim")
        drain()
        exported = exporter.call_args.args[0]
        dx1, dy1 = exported.shifts[1]
        expected = REG.apply_shift(PL.adjust_tone(frames[1], .15), dx1, dy1,
                                   border_mode="constant")[9:72, 8:72]
        check("film stabilisé : recalé + recadré, vue différée, tonalité préservée",
              isinstance(exported, StabilizedFrames)
              and isinstance(exported.frames, PL.ToneFrames)
              and np.allclose(exported[1], expected, atol=1e-5))
    from astroforge.core.animation import export_video
    try:
        export_video(view, TMP + "/tone-cancel.gif", progress=cancel)
        gif_cancelled = False
    except InterruptedError:
        gif_cancelled = True
    check("film GIF : progression annulation propagée", gif_cancelled)
    controls = {"gains": [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.],
                "layer_denoise": [.1, .2, .3, .4, .5, .6], "layer_sharpen": [.2, .4, .6, .8, 1., 1.2]}
    with tempfile.TemporaryDirectory() as directory, patch.object(P, "PROFILES_DIR", directory):
        p._set_wave_controls(controls)
        P.save_profile("planet_wavelets", "six", p.pb_wave.get_params())
        p._reset_waves()
        check("reset ondelettes : six gains/débruitages/nettetés", p._wave_profile_params() == p.pb_wave.defaults)
        p.pb_wave._load("six")
        check("profil six couches aller-retour complet", p._wave_profile_params() == controls)
        path = os.path.join(directory, "sequence.json")
        st = p._collect_settings()
        Q.save_sequence(path, [{"op": "planet_chain", "params": st}], "ser")
        back = Q.load_sequence(path)
        p._reset_waves()
        p._apply_settings(back["ops"][0]["params"])
        check("séquence SER chargeable et tous contrôles restaurés", p._wave_profile_params() == controls
              and p._tone_params() == st["tone"])
        with patch.object(Q.os, "listdir", return_value=["test.ser"]), \
             patch("astroforge.core.image_io.load_ser", return_value=(np.stack(frames), {})), \
             patch("astroforge.core.planetary.run_planetary", return_value={"image": img}) as chain, \
             patch("astroforge.core.wavelets.registax_enhance", return_value=img) as wave, \
             patch("astroforge.core.image_io.save_image"):
            Q.run_batch_ser_chain(back, directory, directory)
            batch_frames = chain.call_args.args[0]
            check("batch tonalité avant classement et six couches propagées", isinstance(batch_frames, PL.ToneFrames)
                  and np.allclose(batch_frames[0], PL.adjust_tone(img, .15))
                  and wave.call_args.kwargs["layer_denoise"] == controls["layer_denoise"]
                  and wave.call_args.kwargs["layer_sharpen"] == controls["layer_sharpen"])
            try:
                Q.run_batch_ser_chain(back, directory, directory, progress=cancel)
                interrupted = False
            except InterruptedError:
                interrupted = True
            check("batch annulation non avalée", interrupted)
    with tempfile.TemporaryDirectory() as directory:
        source = os.path.join(directory, "tone.ser")
        write_ser(source, np.stack([img, img]))
        loaded, _ = load_ser(source)
        tone = {"brightness": .1, "contrast": 1.1, "curve": [0, .15, .6, .85, 1]}
        ops = [{"op": "planet_chain", "params": {"tone": tone, "n_align": 1,
                "roi_mode": "Image entière", "align_mode": "Global (une translation)",
                "layer_denoise": [0.] * 6, "layer_sharpen": [0.] * 6}}]
        outputs = Q.run_batch_ser_chain({"mode": "ser", "ops": ops}, directory, directory)
        from astroforge.core.image_io import load_image
        actual = load_image(os.path.join(directory, outputs[0]))
        check("batch SER réel : luminosité contraste courbe numériques avant empilement",
              np.allclose(actual, PL.adjust_tone(loaded[0], **tone), atol=2e-7))
    p._set_wave_controls({"gains": [1.5], "wave_denoise": .4})
    migrated = p._wave_profile_params()
    check("ancien profil UI : migration et couches manquantes neutres",
          migrated["gains"] == [1.5, 1., 1., 1., 1., 1., 1.]
          and migrated["layer_denoise"] == [.4] * 6
          and migrated["layer_sharpen"] == [0.] * 6)

    # ---------------- v1.9.6 : courbes façon Photoshop, automatiques, annulation
    tabs_tone.setCurrentIndex(0)
    check("onglets tonalité : courbe + deux automatiques au même endroit",
          tabs_tone.count() == 3 and tabs_tone.parentWidget() is p._o("grpTone")
          and tabs_tone.tabText(0) == "Courbe"
          and tabs_tone.tabText(1) == "Contraste automatique"
          and tabs_tone.tabText(2) == "Luminosité automatique")
    legacy5 = [0.0, 0.3, 0.5, 0.8, 1.0]
    knots5 = [[0.0, 0.0], [0.25, 0.3], [0.5, 0.5], [0.75, 0.8], [1.0, 1.0]]
    check("courbe généralisée : évaluation identique à l'héritage cinq points",
          np.array_equal(PL.adjust_tone(a, curve=legacy5), PL.adjust_tone(a, curve=knots5))
          and PL.tone_settings(curve=legacy5)["curve"] == knots5)
    check("courbe : x triés, y écrêtés, identité reconnue",
          PL.tone_settings(curve=[[0.9, 1.4], [0.1, -0.5], [1.0, 1.0]])["curve"]
          == [[0.1, 0.0], [0.9, 1.0], [1.0, 1.0]]
          and PL.is_identity_curve(PL.tone_settings()["curve"])
          and not PL.is_identity_curve(knots5))
    ok_bad = True
    for bad in ([0.5], [[0.5, 0.5]], [[0.5, np.nan], [1.0, 1.0]]):
        try:
            PL.tone_settings(curve=bad)
            ok_bad = False
        except ValueError:
            ok_bad = ok_bad and True
    check("courbes malformées rejetées (un point, vide, non fini)", ok_bad)
    editor = p.curve
    editor.setFixedSize(240, 150)
    app.processEvents()
    editor.set_points([[0.0, 0.0], [1.0, 1.0]])
    emitted = []
    editor.pointsChanged.connect(lambda: emitted.append(1))
    rect = editor._plot_rect()
    def px(x, y):
        return QPoint(int(rect.left() + x * rect.width()),
                      int(rect.bottom() - y * rect.height()))
    QTest.mouseClick(editor, Qt.LeftButton, pos=px(.5, .25))
    check("éditeur courbe : clic ajoute un point", len(editor.points()) == 3
          and abs(editor.points()[1][0] - .5) < .02 and abs(editor.points()[1][1] - .25) < .02)
    QTest.mousePress(editor, Qt.LeftButton, pos=px(.5, .25))
    QTest.mouseMove(editor, px(.7, .8))
    QTest.mouseRelease(editor, Qt.LeftButton, pos=px(.7, .8))
    pts = editor.points()
    check("éditeur courbe : glissement déplace, x monotone", len(pts) == 3
          and abs(pts[1][0] - .7) < .02 and abs(pts[1][1] - .8) < .02
          and pts[0][0] < pts[1][0] < pts[2][0])
    editor.set_points([[0.0, 0.0], [0.7, 0.8], [1.0, 1.0]])
    QTest.mouseClick(editor, Qt.LeftButton, pos=px(.7, .8))
    QTest.keyClick(editor, Qt.Key_Delete)
    check("éditeur courbe : Suppr retire le point sélectionné", len(editor.points()) == 2)
    editor.set_points([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    QTest.mouseDClick(editor, Qt.LeftButton, pos=px(.5, .5))
    check("éditeur courbe : double-clic retire un point intérieur", len(editor.points()) == 2)
    editor.set_points([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    QTest.mouseClick(editor, Qt.LeftButton, pos=px(.5, .5))
    QTest.mouseClick(editor, Qt.RightButton, pos=px(.5, .5))
    check("éditeur courbe : clic droit retire un point intérieur", len(editor.points()) == 2)
    editor.set_points([[0.0, 0.0], [1.0, 1.0]])
    QTest.mousePress(editor, Qt.LeftButton, pos=px(0, 0))
    QTest.mouseMove(editor, px(.4, .6))
    QTest.mouseRelease(editor, Qt.LeftButton, pos=px(.4, .6))
    check("éditeur courbe : extrémité x=0 épinglée, y réglable",
          editor.points()[0][0] == 0.0 and abs(editor.points()[0][1] - .6) < .02)
    QTest.mousePress(editor, Qt.LeftButton, pos=px(1, 1))
    QTest.mouseMove(editor, px(.4, .3))
    QTest.mouseRelease(editor, Qt.LeftButton, pos=px(.4, .3))
    check("éditeur courbe : extrémité x=1 épinglée, y réglable",
          editor.points()[-1][0] == 1.0 and abs(editor.points()[-1][1] - .3) < .02)
    QTest.mouseClick(editor, Qt.LeftButton, pos=px(0, .6))
    QTest.keyClick(editor, Qt.Key_Delete)
    check("éditeur courbe : extrémité x=0 non supprimable", len(editor.points()) == 2)
    QTest.mouseClick(editor, Qt.LeftButton, pos=px(1, .3))
    QTest.keyClick(editor, Qt.Key_Delete)
    check("éditeur courbe : extrémité x=1 non supprimable", len(editor.points()) == 2)
    editor.set_points([[0.0, 0.0], [1.0, 1.0]])
    QTest.mouseClick(editor, Qt.LeftButton, pos=px(.5, .5))
    QTest.mousePress(editor, Qt.LeftButton, pos=px(.5, .5))
    QTest.mouseMove(editor, px(1, 1))
    QTest.mouseRelease(editor, Qt.LeftButton, pos=px(1, 1))
    pts = editor.points()
    check("éditeur courbe : x bloqué entre voisins (monotonie)", len(pts) == 3
          and pts[1][0] <= 1 - editor.GAP and pts[0][0] < pts[1][0] < pts[2][0])
    check("éditeur courbe : pointsChanged émis", len(emitted) >= 4)
    editor.set_points([[0.0, 0.0], [1.0, 1.0]])

    # Contraste automatique : centiles exacts sur une rampe déterministe
    ramp = np.linspace(0, 1, 128 * 128, dtype=np.float32).reshape(128, 128)
    params, black, white = PL.auto_contrast_params(ramp, 0.001)
    k_expected = 1.0 / 0.998
    check("contraste automatique : noir/blanc aux centiles 0,1 %",
          abs(black - 0.001) < 1e-5 and abs(white - 0.999) < 1e-5
          and abs(params["contrast"] - k_expected) < 1e-5
          and abs(params["brightness"] - (k_expected * 0.501 - 0.5)) < 1e-5
          and PL.is_identity_curve(params["curve"]))
    dark = ramp * 0.25
    view = PL.ToneFrames([ramp, dark], params)
    own, _b2, _w2 = PL.auto_contrast_params(dark, 0.001)
    check("contraste automatique : mêmes réglages pour toutes les frames",
          np.allclose(view[0], PL.adjust_tone(ramp, **params))
          and np.allclose(view[1], PL.adjust_tone(dark, **params))
          and view.params["curve"] == params["curve"]
          and abs(own["contrast"] - params["contrast"]) > 1e-3)
    gparams, gamma = PL.auto_brightness_params(np.full((64, 64), 0.25, np.float32))
    check("luminosité automatique : gamma exact vers gris moyen 0,5",
          abs(gamma - 0.5) < 1e-6
          and abs(float(np.mean(PL.adjust_tone(
              np.full((64, 64), 0.25, np.float32), **gparams))) - 0.5) < 1e-6)
    rparams, rgamma = PL.auto_brightness_params(ramp)
    check("luminosité automatique : rampe à 0,5 → gamma 1 identité",
          abs(rgamma - 1.0) < 1e-6
          and np.allclose(PL.adjust_tone(ramp, **rparams), ramp, atol=1e-6))

    # Flux GUI : calcul une fois sur Actualiser, champs éditables, Appliquer
    p.frames = PL.ToneFrames([ramp, ramp * .8, ramp * .6])
    p.original_frames = p.frames.originals
    p.preview_base = None
    p._o("sldBrowse").setRange(0, 2)
    p._o("spnAutoClip").setValue(0.1)
    QTest.mouseClick(p._o("btnAutoCompute"), Qt.LeftButton)
    app.processEvents()
    computed = p._tone_params()
    check("contraste auto GUI : calculé une fois, champs éditables",
          abs(computed["brightness"] - (k_expected * 0.501 - 0.5)) < 1e-3
          and p._o("spnAutoBlack").isEnabled() and p._o("spnAutoWhite").isEnabled())
    before_clip = p._o("spnBrightness").value()
    p._o("spnAutoClip").setValue(1.0)
    check("rognage modifié sans recalcul (seul Actualiser mesure)",
          p._o("spnBrightness").value() == before_clip)
    p._o("spnAutoWhite").setValue(0.90)
    edited = p._tone_params()
    check("noir/blanc éditables avant application",
          abs(edited["contrast"] - 1.0 / (0.90 - black)) < 1e-3)
    apply()
    check("contraste auto appliqué à toutes les frames (même chemin ToneFrames)",
          np.allclose(p.frames[0], PL.adjust_tone(ramp, **edited))
          and np.allclose(p.frames[2], PL.adjust_tone(ramp * .6, **edited)))
    flat = np.full((64, 64), 0.25, np.float32)
    p.frames = PL.ToneFrames([flat, flat * .8])
    p.original_frames = p.frames.originals
    p.preview_base = None
    QTest.mouseClick(p._o("btnAutoGamma"), Qt.LeftButton)
    app.processEvents()
    check("luminosité auto GUI : gamma calculé, champ éditable",
          abs(p._o("spnAutoGamma").value() - 0.5) < 1e-6
          and p._o("spnAutoGamma").isEnabled())
    p._o("spnAutoGamma").setValue(0.7)
    apply()
    check("luminosité auto appliquée uniformément en courbe gamma",
          np.allclose(p.frames[0], PL.adjust_tone(flat, curve=PL.gamma_curve(0.7)))
          and np.allclose(p.frames[1], PL.adjust_tone(flat * .8, curve=PL.gamma_curve(0.7))))

    # Profils tonalité : courbe arbitraire aller-retour + migration cinq valeurs
    with tempfile.TemporaryDirectory() as directory, patch.object(P, "PROFILES_DIR", directory):
        curve_pts = [[0.0, 0.0], [0.2, 0.8], [0.55, 0.35], [1.0, 1.0]]
        p._set_tone_controls({"brightness": 0.05, "contrast": 1.1, "curve": curve_pts})
        P.save_profile("planet_tone", "ma courbe", p.pb_tone.get_params())
        p._set_tone_controls({})
        check("profil tonalité : défauts = identité",
              p._tone_params() == PL.tone_settings())
        p.pb_tone._load("ma courbe")
        back = p._tone_params()
        check("profil tonalité : courbe arbitraire aller-retour",
              back["curve"] == curve_pts and back["brightness"] == 0.05
              and back["contrast"] == 1.1)
        P.save_profile("planet_tone", "ancien", {"brightness": 0.1, "contrast": 1.2,
                                                 "curve": [0.0, 0.2, 0.6, 0.9, 1.0]})
        p.pb_tone._load("ancien")
        check("profil tonalité ancien format : cinq valeurs migrées en points",
              p._tone_params()["curve"] == [[0.0, 0.0], [0.25, 0.2], [0.5, 0.6],
                                            [0.75, 0.9], [1.0, 1.0]]
              and p._tone_params()["brightness"] == 0.1
              and p._tone_params()["contrast"] == 1.2)
        p._set_tone_controls({})

    # Annulation globale : bouton en haut, dix actions, Ctrl+Z
    btn_undo = p._o("btnUndoP")
    check("bouton Annuler en haut du module planétaire",
          btn_undo is not None and btn_undo.isVisible()
          and btn_undo.parentWidget() is p.ui
          and btn_undo.mapTo(p.ui, QPoint(0, 0)).y() < 40)
    for _ in range(15):
        if not btn_undo.isEnabled():
            break
        QTest.mouseClick(btn_undo, Qt.LeftButton)
        drain()
    check("annulation : désactivée et sans texte de compteur à vide",
          btn_undo.text() == "Annuler" and not btn_undo.isEnabled()
          and not p._undo_stack)
    for i in range(1, 13):
        p._o("spnBrightness").setValue(round(.05 * i, 2))
        apply()
    check("annulation : dix actions plafonnées", btn_undo.text() == "Annuler (10)"
          and btn_undo.isEnabled() and len(p._undo_stack) == 10)
    seen = []
    for expected in (.55, .50, .45, .40, .35, .30, .25, .20, .15, .10):
        QTest.mouseClick(btn_undo, Qt.LeftButton)
        drain()
        seen.append(round(p.applied_tone["brightness"], 6))
    check("annulation : dix actions restaurées dans l'ordre",
          seen == [.55, .5, .45, .4, .35, .3, .25, .2, .15, .1]
          and not btn_undo.isEnabled() and btn_undo.text() == "Annuler")
    p._execute("grpTone", p._reset_tone)
    drain()
    check("retour identité avant le test Ctrl+Z", p.applied_tone == PL.tone_settings())
    raw0, raw1 = p.original_frames[0], p.original_frames[1]
    p._o("spnBrightness").setValue(.15)
    apply()
    check("application : vue ajustée, originaux intacts", p.frames[0] is not raw0)
    spin = p._o("spnBrightness")
    p._o("spnBrightness").setValue(.20)
    apply()
    p.setFocus()  # un champ texte consinerait sinon Ctrl+Z pour son propre undo
    QTest.keyClick(p, Qt.Key_Z, Qt.ControlModifier)
    drain()
    check("Ctrl+Z : dernière action annulée",
          abs(p.applied_tone["brightness"] - .15) < 1e-9)
    QTest.mouseClick(btn_undo, Qt.LeftButton)
    drain()
    check("annulation tonalité : références originales restaurées",
          p.frames[0] is raw0 and p.frames[1] is raw1
          and p.applied_tone == PL.tone_settings())
    count_before = len(p._undo_stack)
    p._execute("grpWave", lambda: (_ for _ in ()).throw(ValueError("échec attendu")))
    check("commande en échec : aucune entrée d'annulation",
          len(p._undo_stack) == count_before)
    p.live.invalidate()
    drain()
    mw.win.close()
    app.processEvents()


def test_finishing_ai_updates():
    """1.9.7 : filtres de finition (noyau + GUI), menu « Outils IA »,
    dialogue des mises à jour, modèles ONNX."""
    import tempfile
    from unittest.mock import patch
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QAction
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QMenu, QTabWidget
    from astroforge import __version__
    from astroforge.core import finishing as F
    from astroforge.core import planetary as PL
    from astroforge.core import sequence as Q
    from astroforge.gui import ai_updates_dialog as AU
    from astroforge.gui.main_window import MainWindow
    check("version 2.0.0", __version__ == "2.0.0")

    # ------------------------------------------------- noyau finishing
    rng = np.random.default_rng(7)
    mono = rng.random((32, 32), dtype=np.float32) * .8
    rgb = np.clip(np.stack([mono, mono * .6, mono * .2], axis=-1), 0, 1)
    check("saturation neutre : identité mono et RGB", np.array_equal(F.saturation(mono), mono)
          and np.array_equal(F.saturation(rgb, 1.0), rgb))
    gray = np.full((8, 8, 3), .4, np.float32)
    check("saturation : gris invariant et 0 = luminance", np.allclose(F.saturation(gray, 1.7), gray)
          and np.allclose(F.saturation(rgb, 0.0), F.luminance(rgb)[..., None].repeat(3, axis=2)))
    out2 = F.saturation(rgb, 2.0)
    lum = F.luminance(rgb)
    check("saturation 2 écarte davantage de la luminance (écrêtée)",
          np.abs(out2 - lum[..., None]).sum() >= np.abs(rgb - lum[..., None]).sum()
          and out2.max() <= 1 and out2.min() >= 0)
    gains = F.white_balance(rgb, .5, 1.0, 2.0)
    check("balance des blancs : gains par canal", np.allclose(gains[..., 0], rgb[..., 0] * .5)
          and np.allclose(gains[..., 1], rgb[..., 1])
          and np.allclose(gains[..., 2], np.clip(rgb[..., 2] * 2, 0, 1)))
    check("balance des blancs : mono inchangé", np.array_equal(F.white_balance(mono), mono))
    cast = np.full((16, 16, 3), .1, np.float32) + rng.random((16, 16, 3)).astype(np.float32) * .02
    cast[..., 1] += .1
    neu = F.neutralize_background(cast, percentile=5, strength=1.0)
    lows = [float(np.percentile(neu[..., i], 5)) for i in range(3)]
    check("neutralisation du fond : fond rendu gris", max(lows) - min(lows) < .02)
    check("neutralisation force 0 : identité", np.allclose(F.neutralize_background(cast, 5, 0), cast))
    green = np.zeros((4, 4, 3), np.float32) + np.array([.2, .5, .2], np.float32)
    out = F.scnr_green(green)
    check("SCNR : vert plafonné à la moyenne R/B", np.all(out[..., 1] <= (out[..., 0] + out[..., 2]) / 2 + 1e-6)
          and np.allclose(F.scnr_green(green, 0), green))
    check("SCNR : mono copie", np.array_equal(F.scnr_green(mono), mono))
    ramp = np.linspace(.49, .51, 4096, dtype=np.float32).reshape(64, 64)
    stretched = F.autostretch(ramp)
    check("autostretch : rampe faible-contraste élargie", (stretched.max() - stretched.min()) > .15
          and stretched.min() >= 0 and stretched.max() <= 1)
    flat = np.full((16, 16), .3, np.float32)
    check("autostretch : image constante inchangée", np.array_equal(F.autostretch(flat), flat))
    grad = np.tile(np.linspace(.1, .3, 64, dtype=np.float32), (64, 1)) \
        + rng.random((64, 64)).astype(np.float32) * .01
    sub = F.remove_gradient(grad, kernel=48, method="Soustraire")
    div = F.remove_gradient(grad, kernel=48, method="Diviser")
    check("suppression de gradient : soustraire aplatit le fond",
          float(sub.std()) < float(grad.std()) / 5)
    check("suppression de gradient : diviser aplatit le fond",
          float(div.std()) < float(grad.std()) / 5)
    check("gradient : noyaux différents acceptés", F.remove_gradient(grad, 16).shape == grad.shape
          and F.remove_gradient(grad, 128, "Diviser").shape == grad.shape)
    noisy = np.full((32, 32), .5, np.float32)
    noisy[10, 10] = 1.0
    dn = F.denoise_median(noisy, radius=1)
    check("débruitage médian : impulsion supprimée, fond intact", abs(float(dn[10, 10]) - .5) < 1e-6
          and np.allclose(dn[5, 5], .5) and np.array_equal(F.denoise_median(noisy, 0), noisy))
    edge = np.zeros((32, 32), np.float32)
    edge[:, :16] = .4

    def hf(img):
        import scipy.ndimage as ndi
        return float(((img - ndi.gaussian_filter(img, 2)) ** 2).sum())
    usm = F.unsharp_mask(edge, radius=1.5, gain=.8)
    check("masque flou : énergie haute fréquence accrue", hf(usm) > hf(edge))
    check("masque flou : gain 0 identité et image plate inchangée",
          np.array_equal(F.unsharp_mask(edge, 2, 0), edge)
          and np.array_equal(F.unsharp_mask(flat, 3, 1.5), flat))
    ok_mono = True
    for fid, *_rest in F.FILTERS:
        out_m, out_c = F.apply_filter(mono, fid), F.apply_filter(rgb, fid)
        ok_mono &= (out_m.shape == mono.shape and out_c.shape == rgb.shape
                    and out_m.dtype == np.float32 and np.isfinite(out_m).all()
                    and out_m.min() >= 0 and out_m.max() <= 1)
    check("tous les filtres mono/RGB sûrs, float32 borné", ok_mono)
    check("filtre « none » : identité", np.array_equal(F.apply_filter(rgb, "none"), rgb))
    try:
        F.apply_filter(rgb, "inconnu")
        rejected = False
    except ValueError:
        rejected = True
    check("filtre inconnu rejeté", rejected)
    names = ("fin_saturation", "fin_white_balance", "fin_background_neutral",
             "fin_scnr_green", "fin_autostretch", "fin_gradient",
             "fin_median_denoise", "fin_unsharp")
    check("opérations de séquence fin_* enregistrées", all(n in Q.OPS for n in names))
    chain = Q.apply_to_image({"mode": "image", "ops": [
        {"op": "fin_saturation"}, {"op": "fin_white_balance"},
        {"op": "fin_scnr_green", "params": {"strength": 0}},
        {"op": "fin_median_denoise", "params": {"radius": 0}}]}, rgb)
    check("chaîne de séquence fin_* avec défauts neutres identique", np.allclose(chain, rgb))
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "seq.json")
        Q.save_sequence(path, [{"op": "fin_unsharp", "params": {"radius": 1.0, "gain": .3}}])
        seq = Q.load_sequence(path)
    check("séquence fin_* enregistrée et rechargée", seq["ops"][0]["op"] == "fin_unsharp")

    # ------------------------------------------------- menu « Outils IA »
    app = QApplication.instance() or QApplication([])
    mw = MainWindow()
    mw.show()
    app.processEvents()
    menu_ia = mw.win.findChild(QMenu, "menuOutilsIA")
    check("menu « Outils IA » présent avec action mises à jour",
          menu_ia is not None and menu_ia.title() == "Outils IA"
          and mw.win.findChild(QAction, "actionIAMisesAJour").text()
          == "Rechercher les mises à jour des modèles IA…")

    # --------------------------------- dialogue des mises à jour (hors ligne)
    empty_cfg = {"graxpert": {"binary": ""}, "starnet2": {"binary": ""},
                 "deepsnr": {"binary": ""}, "cosmicclarity": {"binary": ""}}
    with patch.object(AU, "load_tool_settings", return_value=empty_cfg):
        dlg = AU.AIUpdatesDialog()
        dlg.show()
        app.processEvents()
        n_rows = 4 + max(1, len(AU.list_models()))
        check("dialogue : lignes outils + modèles/guidance ONNX",
              dlg.tbl.rowCount() == n_rows)
        check("dialogue : outil non configuré « non installé »",
              all(dlg.tbl.item(r, 1).text() == "non installé" for r in range(4)))
        check("dialogue : versions sans API → page officielle",
              all(dlg.tbl.item(r, 2).text() == "à vérifier sur la page officielle"
                  for r in range(1, 4)))
        with patch.object(AU, "fetch_latest_release",
                          return_value={"tag": "v2.5.1", "asset_url":
                          "https://github.com/Steffenhir/GraXpert/releases/download/v2.5.1/graxpert-windows-amd64.msi"}):
            dlg.perform_check()
        check("dialogue : JSON GitHub fictif analysé (GraXpert)",
              "v2.5.1" in dlg.version_text(0) and "installateur" in dlg.version_text(0))
        with patch.object(AU, "fetch_latest_release", side_effect=OSError("réseau")):
            dlg.perform_check()
        check("dialogue : échec réseau dégradé sans plantage",
              dlg.version_text(0) == "vérification impossible"
              and dlg.btnCheck.isEnabled())
        dlg.close()
        app.processEvents()
    fake_models = [{"name": "test.onnx", "path": "/tmp/models/test.onnx", "config": {}}]
    with patch.object(AU, "load_tool_settings", return_value=empty_cfg), \
         patch.object(AU, "list_models", return_value=fake_models):
        dlg2 = AU.AIUpdatesDialog()
        check("dialogue : modèles ONNX listés avec ouverture du dossier",
              dlg2.tbl.rowCount() == 5 and "test.onnx" in dlg2.tbl.item(4, 0).text()
              and dlg2.tbl.cellWidget(4, 3) is not None)
        dlg2.close()

    # --------------------------------- finition : tonalité, filtres, IA
    page = mw.page_common
    check("finition : courbes et 3 onglets comme au planétaire",
          page.curve is not None and page.tabsFin.count() == 3
          and page.tabsFin.tabText(0) == "Courbe"
          and page.tabsFin.tabText(1) == "Contraste automatique"
          and page.tabsFin.tabText(2) == "Luminosité automatique")
    page.set_image(rgb)
    page._o("sldUSMAmount").setValue(0)
    page.curve.set_points([[0.0, .2], [1.0, .8]])
    out = page._compute(page.preview, page._preview_params())
    check("finition : courbe appliquée au calcul d'aperçu (sans cumul)",
          np.allclose(out, PL.adjust_tone(page.preview, curve=[[0.0, .2], [1.0, .8]]), atol=1e-6))
    page.curve.set_points(PL.IDENTITY_CURVE_POINTS)
    dim = np.tile(np.linspace(.30, .34, 64, dtype=np.float32), (64, 1))
    dim = np.stack([dim] * 3, axis=-1)
    page.set_image(dim)
    QTest.mouseClick(page._o("btnFinAutoCompute"), Qt.LeftButton)
    app.processEvents()
    black, white = page._o("spnFinAutoBlack").value(), page._o("spnFinAutoWhite").value()
    pts = page.curve.points()
    check("finition : contraste automatique remplit noir/blanc et courbe",
          0 < black < white < 1 and len(pts) == 2
          and abs(pts[0][0] - black) < 1e-6 and abs(pts[1][0] - white) < 1e-6
          and pts[0][1] == 0 and pts[1][1] == 1)
    QTest.mouseClick(page._o("btnFinAutoGamma"), Qt.LeftButton)
    app.processEvents()
    gamma = page._o("spnFinAutoGamma").value()
    check("finition : luminosité automatique régénère la courbe gamma",
          abs(gamma - PL.auto_brightness_params(page.preview)[1]) < 0.05
          and len(page.curve.points()) == 65)
    page.set_image(rgb)
    page._o("sldUSMAmount").setValue(0)
    page.curve.set_points(PL.IDENTITY_CURVE_POINTS)
    page._o("cmbFinFilter").setCurrentIndex(1)
    page._fin_widgets[("saturation", "amount")].setValue(0.0)
    base_before = page.image
    QTest.mouseClick(page._o("btnFinApply"), Qt.LeftButton)
    app.processEvents()
    grayed = np.stack([F.luminance(base_before)] * 3, -1)
    check("finition : saturation 0 appliquée sur l'image",
          not np.array_equal(page.image, base_before)
          and np.allclose(page.image, np.clip(grayed, 0, 1), atol=1e-5))
    check("finition : réglages remis au neutre après application",
          page._fin_widgets[("saturation", "amount")].value() == 1.0
          and page.curve.points() == [[0.0, 0.0], [1.0, 1.0]])
    out = page._compute(page.preview, page._preview_params())
    check("finition : aucun cumul après application (aperçu = base)",
          np.allclose(out, page.preview, atol=1e-5))
    from astroforge.ai.manager import TASKS
    check("finition : IA locale câblée (tâches, fournisseurs, boutons)",
          page._o("cmbFinAITask").count() == len(TASKS)
          and page._o("cmbFinAIProvider").count() >= 1
          and page._o("btnFinAIRun").isEnabled())
    page._fin_widgets[("saturation", "amount")].setValue(0.7)
    page._seq_add()
    last = page.seq_ops[-1]
    check("finition : ajout de séquence inclut le filtre courant",
          last["op"] == "fin_saturation" and last["params"]["amount"] == 0.7)

    # --------------------------------- captures d'écran (finition + dialogue)
    shots = os.environ.get("ASTROFORGE_UI_SCREENSHOTS")
    if shots:
        os.makedirs(shots, exist_ok=True)
        mw.win.findChild(QTabWidget, "tabs").setCurrentIndex(2)
        app.processEvents()
        for width, height in ((1280, 820), (1024, 700)):
            mw.win.resize(width, height)
            QTest.qWait(30)
            app.processEvents()
            assert mw.win.grab().save(os.path.join(shots, f"finition-{width}x{height}.png"))
        dlg3 = AU.AIUpdatesDialog()
        dlg3.show()
        app.processEvents()
        assert dlg3.grab().save(os.path.join(shots, "ai-updates-dialog.png"))
        dlg3.close()
    mw.win.close()
    app.processEvents()


def test_color_detail():
    """1.9.8 : niveaux, tons clairs/sombres, vibrance, HSL par gammes,
    contraste local, tonalité divisée, mappage de dégradé, courbes par
    canal — noyau, interface, profils, séquences, géométrie."""
    import tempfile
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QTabWidget
    from astroforge import __version__
    from astroforge.core import finishing as F
    from astroforge.core import planetary as PL
    from astroforge.core import sequence as Q
    from astroforge.gui.main_window import MainWindow
    check("version 2.0.0", __version__ == "2.0.0")

    rng = np.random.default_rng(11)
    mono = rng.random((32, 32), dtype=np.float32) * .8
    rgb = np.clip(np.stack([mono, mono * .6, mono * .2], axis=-1), 0, 1)
    ramp2 = np.linspace(0, 1, 256, dtype=np.float32).reshape(16, 16)

    # ---------------------------------------------------------- niveaux
    check("niveaux : identité aux paramètres neutres", np.allclose(F.levels(ramp2), ramp2, atol=1e-6))
    lv = F.levels(ramp2, .2, 1.0, .8).reshape(-1)
    check("niveaux : noir/blanc exact sur rampe", float(lv[0]) == 0 and float(lv[-1]) == 1
          and abs(float(lv[128]) - (float(ramp2.reshape(-1)[128]) - .2) / .6) < 1e-6
          and float(lv[40]) == 0 and float(lv[220]) == 1)
    lv2 = F.levels(ramp2, .2, 2.0, .8).reshape(-1)
    t128 = (float(ramp2.reshape(-1)[128]) - .2) / .6
    check("niveaux : gamma 2 au point milieu", abs(float(lv2[128]) - t128 ** .5) < 1e-5)

    # -------------------------------------------------- courbes par canal
    out = F.apply_curves(rgb, {"r": [[0.0, 0.0], [1.0, 0.5]]})
    check("courbe par canal : seul le rouge change", np.allclose(out[..., 0], rgb[..., 0] * .5, atol=1e-6)
          and np.array_equal(out[..., 1], rgb[..., 1]) and np.array_equal(out[..., 2], rgb[..., 2]))
    check("courbe par canal : mono ignorée", np.array_equal(F.apply_curves(mono, {"r": [[0.0, 0.0], [1.0, 0.5]]}), mono))
    both = F.apply_curves(rgb, {"commune": [[0.0, 0.0], [1.0, 0.5]],
                                "g": [[0.0, 0.0], [1.0, 1.0]]})
    check("courbe commune + canal combinées", np.allclose(both, rgb * .5, atol=1e-6))
    try:
        F.apply_curves(rgb, {"r": [0, 1]})
        bad = False
    except ValueError:
        bad = True
    check("courbe par canal malformée rejetée", bad)

    # --------------------------------------------- tons clairs / sombres
    img2 = np.full((64, 64, 3), .1, np.float32)
    img2[32:] = .9
    sh = F.shadows_highlights(img2, shadows=.6, radius=6)
    check("tons sombres : zones sombres dévoilées", float(sh[:32].mean()) > float(img2[:32].mean()) + .05)
    check("tons clairs : hautes lumières intactes (tolérance rayon)", abs(float(sh[32:].mean()) - .9) < .02)
    hl = F.shadows_highlights(img2, highlights=.5, radius=6)
    check("tons clairs : zones claires renforcées, sombres intactes",
          float(hl[32:].mean()) > .9 and abs(float(hl[:32].mean()) - .1) < .02)
    colored = np.zeros((16, 16, 3), np.float32) + np.array([.3, .15, .1], np.float32)
    lift = F.shadows_highlights(colored, shadows=.4, radius=6)
    r0 = float(colored[..., 0].mean()) / float(colored[..., 1].mean())
    r1 = float(lift[..., 0].mean()) / float(lift[..., 1].mean())
    check("tons sombres : chromaticité conservée (pas de désaturation)", abs(r1 - r0) < .02
          and float(lift.mean()) > float(colored.mean()))
    check("tons clairs/sombres : identité à force nulle", np.allclose(F.shadows_highlights(rgb, 0, 0, 24), rgb))

    # ---------------------------------------------------------- vibrance
    check("vibrance : identité à 1", np.allclose(F.vibrance(rgb, 1.0), rgb))
    gray = np.full((8, 8, 3), .5, np.float32)
    check("vibrance : pixels gris invariants", np.allclose(F.vibrance(gray, 1.8), gray))
    low = np.array([.4, .38, .36], np.float32)
    high = np.array([.9, .2, .2], np.float32)
    probe = np.stack([low, high]).reshape(2, 1, 3)
    outv = F.vibrance(probe, 1.8)
    def sat_of(px):
        return float(px.max() - px.min())
    gain_low = (sat_of(outv[0, 0]) - sat_of(low)) / sat_of(low)
    gain_high = (sat_of(outv[1, 0]) - sat_of(high)) / sat_of(high)
    check("vibrance : faible saturation boostée davantage (relatif)", gain_low > gain_high > 0)

    # --------------------------------------------------------------- HSL
    check("HSL : identité exacte aux paramètres neutres", np.array_equal(F.hsl(rgb), rgb))
    red = np.array([[[.8, .1, .1]]], np.float32)
    comp = F.hsl(red, hue=180)
    check("HSL : teinte globale 180° renverse rouge → cyan", float(comp[0, 0, 0]) < .2
          and float(comp[0, 0, 1]) > .7 and float(comp[0, 0, 2]) > .7)
    grayed = F.hsl(rgb, sat=0)
    check("HSL : saturation 0 → gris", np.allclose(grayed[..., 0], grayed[..., 2], atol=1e-6))
    check("HSL : luminosité globale ajoutée (mono)", np.allclose(F.hsl(mono, lum=.2), np.clip(mono + .2, 0, 1), atol=1e-6))
    green_px = np.array([[[.1, .8, .1]]], np.float32)
    outg = F.hsl(green_px, g_hue=30)
    check("HSL gamme verte : teinte décalée vers le cyan", float(outg[0, 0, 2]) > .1)
    outr = F.hsl(red, g_hue=30, g_sat=2.0)
    check("HSL gamme verte : pixel rouge intact", np.allclose(outr, red, atol=2e-6))

    # ---------------------------------------------------- contraste local
    blob = np.full((64, 64), .3, np.float32)
    blob[24:40, 24:40] = .5
    check("contraste local : identité à gain nul", np.allclose(F.local_contrast(blob, 32, 0), blob))
    import scipy.ndimage as ndi
    def detail(a):
        return float(((a - ndi.gaussian_filter(a, 8)) ** 2).sum())
    lc = F.local_contrast(blob, radius=12, gain=.8)
    check("contraste local : contraste d'un bloc accru", detail(lc) > detail(blob) * 1.5
          and float(lc.min()) >= 0 and float(lc.max()) <= 1)

    # -------------------------------------------------- tonalité divisée
    check("tonalité divisée : identité à force nulle", np.allclose(F.split_tone(rgb, force=0), rgb))
    grad = np.linspace(.05, .95, 256, dtype=np.float32).reshape(16, 16, 1).repeat(3, 2)
    st = F.split_tone(grad, 0, 0, 1, 1, 0, 0, balance=0, force=1)
    check("tonalité divisée : ombres bleutées, hautes lumières rougeâtres",
          float(st[0, 0, 2]) > float(st[0, 0, 0]) and float(st[-1, -1, 0]) > float(st[-1, -1, 2]))
    stm = F.split_tone(np.linspace(.05, .95, 64, dtype=np.float32).reshape(8, 8),
                       0, 0, 1, 1, 0, 0, 0, 1)
    check("tonalité divisée : mono sûre", stm.shape == (8, 8) and stm.dtype == np.float32
          and float(stm.min()) >= 0 and float(stm.max()) <= 1)

    # ------------------------------------------------- mappage de dégradé
    check("mappage de dégradé : identité à intensité nulle", np.allclose(F.gradient_map(rgb, intensity=0), rgb))
    gm = F.gradient_map(np.tile(ramp2, (1, 8)), 0, 0, 0, .5, .5, .5, 1, 1, 1, 1.0)
    prof = gm[:, 0]
    check("mappage de dégradé : rampe monotone", np.all(np.diff(prof) >= -1e-6)
          and abs(float(prof[0])) < 1e-6
          and abs(float(prof[-1]) - float(ramp2.reshape(-1)[240])) < 1e-6)
    lumc = F.luminance(rgb)
    mapped = np.stack([np.interp(lumc, [0, .5, 1], [.8, .5, .2]),
                       np.interp(lumc, [0, .5, 1], [0, .5, 0]),
                       np.interp(lumc, [0, .5, 1], [0, .5, 0])], axis=-1)
    gmc = F.gradient_map(rgb, .8, 0, 0, .5, .5, .5, .2, 0, 0, 1.0)
    check("mappage de dégradé : détail conservé (rampe + écart à la luminance)",
          np.allclose(gmc, mapped + (rgb - lumc[..., None]), atol=1e-5))

    # --------------------------------------- sûreté + séquences + profils
    ok_all = True
    for fid, *_rest in F.FILTERS:
        o1, o2 = F.apply_filter(mono, fid), F.apply_filter(rgb, fid)
        ok_all &= (o1.shape == mono.shape and o2.shape == rgb.shape
                   and o1.dtype == np.float32 and np.isfinite(o1).all()
                   and float(o1.min()) >= 0 and float(o1.max()) <= 1)
    check("les quinze filtres mono/RGB sûrs", ok_all)
    names = ("fin_levels", "fin_shadows_highlights", "fin_vibrance", "fin_hsl",
             "fin_local_contrast", "fin_split_tone", "fin_gradient_map")
    check("opérations de séquence 1.9.8 enregistrées", all(n in Q.OPS for n in names))
    chain = Q.apply_to_image({"mode": "image", "ops": [
        {"op": "fin_levels"}, {"op": "fin_hsl"}, {"op": "fin_gradient_map"},
        {"op": "fin_vibrance", "params": {"amount": 1.5}}]}, rgb)
    check("chaîne de séquence 1.9.8 : neutres identité puis vibrance effective",
          not np.allclose(chain, rgb) and chain.shape == rgb.shape)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "seq8.json")
        Q.save_sequence(path, [{"op": "fin_vibrance", "params": {"amount": 1.3}}])
        seq = Q.load_sequence(path)
    check("séquence 1.9.8 enregistrée et rechargée", seq["ops"][0]["op"] == "fin_vibrance")

    # ------------------------------------------------- interface finition
    app = QApplication.instance() or QApplication([])
    mw = MainWindow()
    mw.show()
    mw.win.findChild(QTabWidget, "tabs").setCurrentIndex(2)
    app.processEvents()
    page = mw.page_common
    check("finition : sélecteur de canal de courbe présent",
          page._o("cmbCurveChannel").count() == 4
          and page._o("cmbCurveChannel").itemText(0) == "Commune")
    check("finition : banque couleur/détail avec sept filtres",
          page._o("cmbColorFilter").count() == 8
          and page._o("cmbColorFilter").itemData(1) == "levels"
          and page._o("cmbColorFilter").itemData(7) == "gradient_map")
    page.set_image(rgb)
    page._o("sldUSMAmount").setValue(0)
    combo_ch = page._o("cmbCurveChannel")
    combo_ch.setCurrentIndex(1)   # Rouge
    page.curve.set_points([[0.0, 0.0], [1.0, 0.5]])
    combo_ch.setCurrentIndex(0)   # Commune
    out = page._compute(page.preview, page._preview_params())
    check("finition : courbe rouge appliquée au canal R uniquement",
          np.allclose(out[..., 0], np.interp(page.preview[..., 0], [0, 1], [0, .5]), atol=1e-6)
          and np.allclose(out[..., 1], page.preview[..., 1], atol=1e-6)
          and np.allclose(out[..., 2], page.preview[..., 2], atol=1e-6))
    saved = page._fin_profile_get()
    check("finition : profil contient les courbes par canal", saved["curve_r"] == [[0.0, 0.0], [1.0, 0.5]])
    page._fin_reset()
    check("finition : réinitialisation remet tous les canaux", page.curve.points() == [[0.0, 0.0], [1.0, 1.0]]
          and page.curve_channels["r"] == [[0.0, 0.0], [1.0, 1.0]])
    page._fin_profile_set(saved)
    check("finition : profil rechargé retrouve la courbe rouge", page.curve_channels["r"] == [[0.0, 0.0], [1.0, 0.5]])
    page.curve_channels["r"] = [list(p) for p in PL.IDENTITY_CURVE_POINTS]
    page.curve.set_points(PL.IDENTITY_CURVE_POINTS)
    page._o("cmbFinFilter").setCurrentIndex(1)   # saturation (banque 1)
    page._o("cmbColorFilter").setCurrentIndex(
        page._o("cmbColorFilter").findData("shadows_highlights"))
    check("finition : banques exclusives et état courant",
          page._o("cmbFinFilter").currentIndex() == 0
          and page._fin_state()[0] == "shadows_highlights")
    page._o("cmbColorFilter").setCurrentIndex(
        page._o("cmbColorFilter").findData("vibrance"))
    page._fin_widgets[("vibrance", "amount")].setValue(1.8)
    base0 = page.image
    QTest.mouseClick(page._o("btnFinApply"), Qt.LeftButton)
    app.processEvents()
    check("finition : vibrance appliquée et réglages remis au neutre",
          not np.array_equal(page.image, base0)
          and page._fin_widgets[("vibrance", "amount")].value() == 1.0)
    out = page._compute(page.preview, page._preview_params())
    check("finition : après application l'aperçu ne recumule pas", np.allclose(out, page.preview, atol=1e-5))
    shots = os.environ.get("ASTROFORGE_UI_SCREENSHOTS")
    sp = page._o("scrollParams")
    for width, height in ((1280, 820), (1024, 700)):
        mw.win.resize(width, height)
        QTest.qWait(30)
        app.processEvents()
        tb = page.tabsFin.tabBar()
        check(f"finition : panneau sans défilement horizontal à {width}×{height}",
              sp.isVisible() and not sp.horizontalScrollBar().isVisible()
              and page._o("grpColorDetail").isVisible())
        check(f"onglets tonalité entiers à {width}×{height}",
              tb.sizeHint().width() <= tb.width() + 1)
        if shots:
            os.makedirs(shots, exist_ok=True)
            assert mw.win.grab().save(os.path.join(shots, f"finition-{width}x{height}.png"))
    mw.win.close()
    app.processEvents()


def test_help():
    """Aide intégrée : pages compilées, index, recherche, visualiseur."""
    from astroforge.gui import help_viewer as H
    pages = H.load_pages()
    check(f"aide : {len(pages)} pages compilées chargées", len(pages) >= 15)
    idx = H.build_index(pages)
    r1 = H.search(idx, "drizzle")
    r2 = H.search(idx, "DEROTATION")           # casse + accent
    r3 = H.search(idx, "ondelettes registax")  # plusieurs mots
    check(f"aide : recherche drizzle({len(r1)}) derotation({len(r2)}) ondelettes({len(r3)})",
          bool(r1) and bool(r2) and bool(r3))
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    dlg = H.HelpDialog()
    dlg.open_first()
    check(f"aide : arbre {dlg.tree.topLevelItemCount()} groupes, navigateur {len(dlg.browser.toPlainText())} car.",
          dlg.tree.topLevelItemCount() >= 2 and "AstroForge" in dlg.browser.toPlainText())
    dlg.search_edit.setText("drizzle")
    check("aide : recherche 'drizzle' → résultats listés", dlg.results.count() >= 1)
    kws = H.build_keywords(pages)
    keys = [k["key"] for k in kws]
    check(f"aide : index {len(kws)} mots-clés A→Z triés (sans bruit)",
          len(kws) >= 40 and keys == sorted(keys)
          and all(len(x) >= 3 and x not in ("des", "les") for x in keys))
    check("aide : onglets Sommaire/Index/Recherche présents", dlg.tabs.count() == 3)
    dlg2 = H.HelpDialog()
    dlg2.kw_filter.setText("drizz")
    it0 = dlg2.kw_list.item(0)
    check("aide : index filtré 'drizz' → entrée drizzle",
          it0 is not None and "drizzle" in it0.text())
    if it0 is not None:
        dlg2._on_keyword(it0)
    check("aide : clic index → page drizzle affichée",
          "drizzle" in dlg2.browser.toPlainText().lower())
    from astroforge.gui.main_window import HELP_TAB_PAGES
    check("aide : F1 contextuel ciel profond → man-04 + sommaire synchronisé",
          HELP_TAB_PAGES[0] == "man-04-4-mode-ciel.html"
          and dlg2.open_by_file("man-04-4-mode-ciel.html")
          and dlg2.tabs.currentIndex() == 0
          and "ciel" in dlg2.browser.toPlainText().lower())
    # destruction explicite tant que QApplication existe (sinon plantage
    # « Bus error » à la sortie de l'interpréteur, ordre de destruction aléatoire)
    for d in (dlg, dlg2):
        d.close()
        d.deleteLater()
    app.processEvents()
    from PySide6.QtCore import QCoreApplication, QEvent
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_ai_contracts():
    """Contrats IA simulés : aucun poids ni binaire externe réellement exécuté."""
    import shutil
    import subprocess
    import tempfile
    from unittest.mock import patch
    from astroforge.ai import manager as M
    from astroforge.ai.onnx_runner import OnnxRunner
    from astroforge.core.image_io import save_image

    def rejects(name, action, fragment):
        try:
            action()
        except (RuntimeError, ValueError) as exc:
            check(name, fragment in str(exc))
        else:
            check(name, False)

    with tempfile.TemporaryDirectory(prefix="astroforge IA espaces ") as directory:
        exe = os.path.join(directory, "outil CLI.exe")
        with open(exe, "wb") as stream:
            stream.write(b"binaire simule")
        runtime = os.path.join(directory, "lib")
        os.mkdir(runtime)
        marker = os.path.join(runtime, "modele.onnx")
        with open(marker, "wb") as stream:
            stream.write(b"ressource simulee")
        ai = M.AIManager()
        ai.settings = M.normalize_tool_settings({"schema_version": 2,
            "graxpert": {"binary": exe}, "starnet2": {"binary": exe},
            "deepsnr": {"binary": exe}, "cosmicclarity": {"binary": directory}})
        mono = np.full((512, 512), .2, np.float32)
        rgb = np.repeat(mono[..., None], 3, axis=2)
        calls = []

        def external(cmd, **options):
            calls.append((cmd, options))
            if "-output" in cmd:
                output = cmd[cmd.index("-output") + 1] + ".fits"
                source = cmd[1]
            else:
                output = cmd[cmd.index("--output") + 1]
                source = cmd[cmd.index("--input") + 1]
            from astroforge.core.image_io import load_image
            assert os.path.isfile(source)
            save_image(output, load_image(source) * .5)
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch.object(M.subprocess, "run", side_effect=external), \
             patch.object(tempfile, "tempdir", directory):
            out = ai.run("denoise", mono, "graxpert", .4)
            cmd, options = calls[-1]
            check("IA GraXpert denoise : -cli/-cmd/-strength, vraie sortie out.fits",
                  cmd[:5] == [exe, cmd[1], "-cli", "-cmd", "denoising"]
                  and "-strength" in cmd and "-smoothing" not in cmd
                  and np.allclose(out, .1))
            check("IA chemins espaces et cwd des ressources runtime",
                  options["cwd"] == directory and cmd[0] == exe
                  and " " in cmd[1] and options["check"] and options["capture_output"])
            check("IA nettoyage du seul dossier temporaire, runtime préservé",
                  not os.path.exists(os.path.dirname(cmd[1])) and os.path.isfile(marker))
            ai.run("background", mono, "graxpert", .3)
            cmd = calls[-1][0]
            check("IA GraXpert fond : background-extraction/-smoothing sans -strength",
                  cmd[4] == "background-extraction" and "-smoothing" in cmd
                  and "-strength" not in cmd)
            for image in (mono, rgb):
                out = ai.run("denoise", image, "deepsnr", linear=True)
                cmd = calls[-1][0]
                check("IA DeepSNR mono/RGB : --input/--output/--model 2/--linear",
                      cmd[1] == "--input" and cmd[3] == "--output"
                      and cmd[-3:] == ["--model", "2", "--linear"]
                      and out.shape == image.shape and np.allclose(out, .1))
            ai.run("starless", rgb, "starnet2")
            cmd = calls[-1][0]
            check("IA StarNet2 suppression étoiles : --input/--output séparés",
                  cmd[1] == "--input" and cmd[3] == "--output" and len(cmd) == 5)
            ai.settings["graxpert"]["extra_args"] = '-preferences_file "prefs avec espaces.json"'
            ai.run("background", mono, "graxpert")
            check("IA arguments additionnels cités restent un argument",
                  calls[-1][0][-1] == "prefs avec espaces.json")
            ai.settings["graxpert"]["extra_args"] = "-output ailleurs"
            rejects("IA sortie réservée non remplaçable par extra_args",
                    lambda: ai.run("denoise", mono, "graxpert"), "extra_args")
            ai.settings["graxpert"]["extra_args"] = ""
        rejects("IA StarNet jamais débruiteur",
                lambda: ai.run("denoise", mono, "starnet2"), "incompatible")
        rejects("IA DeepSNR jamais suppression étoiles",
                lambda: ai.run("starless", mono, "deepsnr"), "incompatible")
        rejects("IA DeepSNR minimum 512 pixels",
                lambda: ai.run("denoise", mono[:511], "deepsnr"), "512")
        rejects("IA GraXpert tâche non supportée rejetée",
                lambda: ai.run("superres", mono, "graxpert"), "ne prend pas")

        def suffix_output(cmd, **options):
            save_image(cmd[cmd.index("-output") + 1] + "_GraXpert.fits", mono * .25)
        with patch.object(M.subprocess, "run", side_effect=suffix_output):
            check("IA variante out_GraXpert.fits acceptée",
                  np.allclose(ai.run("denoise", mono, "graxpert"), .05))
        with patch.object(M.subprocess, "run", return_value=subprocess.CompletedProcess([], 0)):
            rejects("IA code zéro sans sortie : jamais l'entrée comme résultat",
                    lambda: ai.run("denoise", mono, "graxpert"), "sortie nouvelle")
        rejects("IA adaptateur retournant explicitement in.fits rejeté",
                lambda: ai._with_temp_fits(mono, lambda inp, d: inp), "sortie nouvelle")
        old_output = os.path.join(directory, "ancienne.fits")
        save_image(old_output, mono)
        rejects("IA sortie ancienne hors transaction rejetée",
                lambda: ai._with_temp_fits(mono, lambda inp, d: old_output), "sortie nouvelle")
        def old_copy(inp, d):
            path = os.path.join(d, "out.fits")
            shutil.copy2(old_output, path)
            os.utime(path, (1, 1))
            return path
        rejects("IA sortie recopiée avec date ancienne rejetée",
                lambda: ai._with_temp_fits(mono, old_copy), "sortie nouvelle")
        def alias(inp, d):
            path = os.path.join(d, "out.fits")
            os.link(inp, path)
            return path
        rejects("IA lien dur vers l'entrée rejeté",
                lambda: ai._with_temp_fits(mono, alias), "sortie nouvelle")
        def empty_output(inp, d):
            path = os.path.join(d, "out.fits")
            with open(path, "wb"):
                pass
            return path
        rejects("IA sortie vide rejetée",
                lambda: ai._with_temp_fits(mono, empty_output), "sortie nouvelle")
        rejects("IA entrée non finie rejetée avant lancement",
                lambda: ai._with_temp_fits(np.full((2, 2), np.nan), lambda inp, d: inp), "non finies")
        with patch.object(M.subprocess, "run", side_effect=subprocess.CalledProcessError(
                7, [exe], stderr="poids du modèle introuvables")):
            rejects("IA erreur utile avec code et stderr",
                    lambda: ai.run("denoise", mono, "graxpert"), "code 7 — poids")
        with patch.object(M.subprocess, "run", side_effect=subprocess.TimeoutExpired([exe], 1800)):
            rejects("IA délai dépassé explicite",
                    lambda: ai.run("denoise", mono, "graxpert"), "délai dépassé")
        ai.settings["deepsnr"]["binary"] = os.path.join(directory, "absent.exe")
        rejects("IA binaire absent : erreur explicite",
                lambda: ai.run("denoise", mono, "deepsnr"), "introuvable")

        for folder in ("Input", "Output"):
            os.mkdir(os.path.join(directory, folder))
            with open(os.path.join(directory, folder, "utilisateur.fits"), "wb") as stream:
                stream.write(b"fichier utilisateur preexistant")
        with patch.object(M.subprocess, "run") as process:
            for task in ("denoise", "sharpen", "superres", "satellite"):
                rejects("IA Cosmic Clarity refuse lancement interactif " + task,
                        lambda t=task: ai.run(t, mono, "cosmicclarity"), "automatisation non")
            check("IA Cosmic Clarity : aucun sous-processus ni suppression utilisateur",
                  process.call_count == 0 and all(
                      open(os.path.join(directory, f, "utilisateur.fits"), "rb").read()
                      == b"fichier utilisateur preexistant" for f in ("Input", "Output")))
        check("IA Cosmic Clarity non proposé comme automatisé",
              all(p["id"] != "cosmicclarity" for p in ai.providers("denoise")))

        for binary, star, deep in (("C:\\Program Files\\StarNet2.exe", True, False),
                                    ("C:\\Program Files\\DeepSNR.exe", False, True),
                                    ("C:\\outils\\outil.exe", False, False)):
            cfg = M.normalize_tool_settings({"starnet2": {"binary": binary},
                                             "cosmicclarity": {"denoise": "custom.exe"},
                                             "personnalise": {"x": 1}})
            check("IA migration prudente : " + os.path.basename(binary.replace("\\", "/")),
                  bool(cfg["starnet2"]["binary"]) == star
                  and bool(cfg["deepsnr"]["binary"]) == deep
                  and cfg["cosmicclarity"]["denoise"] == "custom.exe"
                  and cfg["personnalise"] == {"x": 1}
                  and M.normalize_tool_settings(cfg) == cfg)
        check("IA sections partielles/invalides complétées",
              M.normalize_tool_settings({"starnet2": None})["deepsnr"]["binary"] == "")
        ai.settings["deepsnr"]["binary"] = exe
        check("IA fournisseurs StarNet/DeepSNR séparés par tâche",
              "starnet2" not in [p["id"] for p in ai.providers("denoise")]
              and "deepsnr" in [p["id"] for p in ai.providers("denoise")]
              and "starnet2" in [p["id"] for p in ai.providers("starless")]
              and "deepsnr" not in [p["id"] for p in ai.providers("starless")])

        models = [{"name": name, "path": name, "config": {"task": task}}
                  for name, task in (("a-super.onnx", "superres"), ("b-generic.onnx", "generic"),
                                     ("c-denoise.onnx", "denoise"), ("d-deconv.onnx", "deconv"))]
        with patch.object(M, "list_models", return_value=models), patch.object(M, "OnnxRunner") as runner:
            runner.return_value.run.return_value = mono
            providers = ai.providers("denoise")
            check("IA ONNX liste réellement filtrée par tâche",
                  [p["id"] for p in providers if p["id"].startswith("onnx:")]
                  == ["onnx:c-denoise.onnx"])
            ai.run("denoise", mono)
            check("IA ONNX défaut choisit premier compatible, pas premier fichier",
                  runner.call_args.args[0] == "c-denoise.onnx")
            rejects("IA ONNX sélection explicite incompatible rejetée",
                    lambda: ai.run("denoise", mono, "onnx:a-super.onnx"), "incompatible")
            rejects("IA ONNX générique non présumé débruiteur",
                    lambda: ai.run("denoise", mono, "onnx:b-generic.onnx"), "incompatible")
            rejects("IA ONNX aucun modèle pour tâche : message explicite",
                    lambda: ai.run("starless", mono), "Aucun modèle")
            check("IA métadonnée historique deconv conservée pour sharpen",
                  any(p["id"] == "onnx:d-deconv.onnx" for p in ai.providers("sharpen")))
        with patch.object(M, "list_models", return_value=[]):
            rejects("IA ONNX dossier vide : message explicite",
                    lambda: ai.run("denoise", mono), "Aucun modèle")

        # Session factice, aucune dépendance à onnxruntime ni aux poids.
        class Session:
            def __init__(self, scale):
                self.scale = scale
            def run(self, outputs, inputs):
                x = next(iter(inputs.values()))
                return [np.repeat(np.repeat(x, self.scale, axis=2), self.scale, axis=3)]
        for scale in (1, 2):
            for shape in ((3, 5), (9, 11), (9, 11, 1), (9, 11, 3), (1, 9)):
                image = np.random.default_rng(4).random(shape).astype(np.float32)
                runner = OnnxRunner.__new__(OnnxRunner)
                runner.cfg = {"input_size": 6, "overlap": 2, "scale": scale}
                runner.input_name, runner.session = "x", Session(scale)
                expected = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
                result = runner.run(image)
                check(f"IA ONNX mono/RGB singleton et tuiles ×{scale} {shape}",
                      result.shape == expected.shape and np.allclose(result, expected, atol=1e-6))
        runner.cfg["overlap"] = 6
        rejects("IA ONNX recouvrement invalide rejeté", lambda: runner.run(mono), "invalide")
        runner.cfg = {"input_size": 6, "overlap": 2, "scale": 2}
        runner.session = Session(1)
        rejects("IA ONNX échelle déclarée incohérente non redimensionnée silencieusement",
                lambda: runner.run(mono[:3, :3]), "échelle déclarée")
        from astroforge.processing.pipeline import Pipeline, STEPS
        with patch("astroforge.processing.pipeline.get_ai_manager", return_value=ai), \
             patch.object(ai, "run", return_value=mono * .5) as run:
            result = Pipeline([{"name": "ai_tool", "params": {
                "task": "starless", "provider": "starnet2"}}]).run({"image": mono})
            check("IA suppression étoiles intégrée à l'étape ai_tool sans retirer les étapes",
                  run.call_args.args[0] == "starless" and np.allclose(result["image"], .1)
                  and all(name in STEPS for name in ("denoise", "satellite", "ai_tool", "wavelets")))


def test_ai_settings_gui():
    import tempfile
    from unittest.mock import patch
    from PySide6.QtWidgets import QApplication
    from astroforge.ai import manager as M
    from astroforge.gui.ai_dialog import AIToolsDialog
    from astroforge.gui.main_window import MainWindow
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory(prefix="astroforge reglages ") as directory, \
         patch.object(M, "SETTINGS_PATH", os.path.join(directory, "ai_tools.json")), \
         patch.object(M, "_manager", None), patch.object(M, "_provider_listeners", []):
        mw = MainWindow()
        manager = M.get_ai_manager()
        manager._onnx_cache["ancien"] = object()
        dialog = AIToolsDialog()
        with patch("astroforge.gui.ai_dialog.QFileDialog.getExistingDirectory",
                   return_value=directory) as pick_dir, \
             patch("astroforge.gui.ai_dialog.QFileDialog.getOpenFileName") as pick_file:
            dialog._browse(dialog.ed_cc)
            check("IA dialogue CC : sélection répertoire seule, sans déballage tuple",
                  dialog.ed_cc.text() == directory and pick_dir.call_count == 1
                  and pick_file.call_count == 0)
        dialog.ed_starnet.setText(os.path.join(directory, "StarNet2.exe"))
        dialog.ed_deepsnr.setText(os.path.join(directory, "DeepSNR.exe"))
        dialog.accept()
        check("IA sauvegarde recharge le même singleton sans redémarrage et vide cache",
              M.get_ai_manager() is manager and not manager._onnx_cache
              and manager.settings["deepsnr"]["binary"] == dialog.ed_deepsnr.text())
        for page in (mw.page_sky, mw.page_planet):
            task = page.ui.findChild(object, "cmbAITask")
            provider = page.ui.findChild(object, "cmbAIProvider")
            check("IA deux pages : TASKS/TASK_LABELS et tâches existantes préservées",
                  [task.itemData(i) for i in range(task.count())] == M.TASKS
                  and [task.itemText(i) for i in range(task.count())]
                  == [M.TASK_LABELS[t] for t in M.TASKS])
            check("IA sauvegarde rafraîchit fournisseurs des deux pages ouvertes",
                  provider.findData("deepsnr") >= 0 and provider.findData("starnet2") < 0)
            task.setCurrentIndex(task.findData("starless"))
            check("IA changement tâche rafraîchit immédiatement fournisseurs",
                  provider.findData("starnet2") >= 0 and provider.findData("deepsnr") < 0)
        mw.page_sky._set_ia_params({"task": "starless", "provider": "starnet2", "strength": .7})
        check("IA profil recharge identifiants tâche/fournisseur et force",
              mw.page_sky._ia_params() == {"task": "starless", "provider": "starnet2", "strength": .7})
        with patch("astroforge.gui.ai_dialog.QFileDialog.getOpenFileName",
                   return_value=("", "")), \
             patch("astroforge.gui.ai_dialog.QFileDialog.getExistingDirectory") as pick_dir:
            dialog._browse(dialog.ed_starnet)
            check("IA annuler choix binaire ne lance pas un choix répertoire",
                  pick_dir.call_count == 0)
        mw.win.close()
        app.processEvents()


if __name__ == "__main__":
    test_ai_contracts()
    test_ai_settings_gui()
    test_registration()
    test_calibration_stacking()
    test_drizzle()
    test_wavelets_restoration()
    test_gradient()
    test_effects()
    test_ser()
    test_stacking_rejection()
    test_colorcal()
    test_xisf()
    test_astrometry()
    test_sequence()
    test_equipment()
    test_defects()
    test_pcc()
    test_similarity()
    test_autostacker()
    test_registax_enhance()
    test_align_poly()
    test_differential()
    test_batch_ser()
    test_planetary_flow()
    test_ca()
    test_align_methods()
    test_regressions_audit()
    test_gui()
    test_step_commands()
    test_planetary_tone_layers()
    test_finishing_ai_updates()
    test_color_detail()
    test_help()
    n_ok = sum(1 for _, ok in PASS if ok)
    print(f"\n=== {n_ok}/{len(PASS)} tests réussis ===")
    shutil.rmtree(TMP, ignore_errors=True)
    # fermeture propre de Qt avant la sortie de l'interpréteur
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication
    _app = QApplication.instance()
    if _app is not None:
        for w in _app.topLevelWidgets():
            try:
                w.close()
                w.deleteLater()
            except RuntimeError:  # déjà détruit côté C++
                pass
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        _app.processEvents()
    sys.exit(0 if n_ok == len(PASS) else 1)
