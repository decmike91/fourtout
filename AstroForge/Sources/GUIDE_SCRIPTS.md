# Guide de scripts AstroForge

Tout le noyau d'AstroForge est utilisable en Python pur, sans interface — pour
automatiser vos traitements (plan d'action `scripts/`, crontab, etc.).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # + onnxruntime, photutils, astroalign, rawpy
cd <dossier d'AstroForge>              # les scripts supposent être lancés d'ici
export PYTHONPATH=.
```

## 1. Séquence batch sur un dossier d'images (finition)

```python
from astroforge.core import sequence as Q

seq = {"mode": "image", "ops": [
    {"op": "gimp_tone", "params": {"contrast": 0.1, "sat": 1.1,
                                   "usm_amount": 0.5, "usm_radius": 1.5}},
    {"op": "denoise", "params": {"method": "Non-local means", "strength": 0.4}},
]}
Q.run_batch_image(seq, "entree/", "sortie/")          # -> sortie/*.tif
# ou rejouer une séquence enregistrée depuis l'interface :
seq = Q.load_sequence("scripts/batch_finition.json")
```

Opérations disponibles (`Q.OPS`) : `brightness_contrast`, `gamma`, `saturation`,
`white_balance`, `color_calibrate`, `background_neutralize`, `scnr_green`,
`autostretch`, `denoise`, `unsharp`, `deconv_rl`, `remove_satellite_trails`,
`apply_palette`, `superres_lanczos`, `ai_run`, `gimp_tone`.

## 2. Batch planétaire : toutes les vidéos SER d'un dossier

```python
from astroforge.core import sequence as Q

seq = Q.load_sequence("scripts/batch_planetaire_ser.json")  # réglages Registax
Q.run_batch_ser_chain(seq, "videos/", "empiles/", max_frames=200)
# chaque SER -> classement qualité, alignement, empilement, ondelettes,
# (palette solaire si demandée) -> empiles/<nom>_stacked.fits
```

## 3. Pipeline ciel profond scripté (calibration → drizzle → gradient)

```python
import numpy as np
from astroforge.core.image_io import load_image, save_image
from astroforge.core import calibration as C, registration as R
from astroforge.core.stacking import stack
from astroforge.core.drizzle import drizzle_stack
from astroforge.core.gradient import extract_gradient
from astroforge.core.restoration import denoise

lights = ["lights/" + f for f in __import__("os").listdir("lights") if f.endswith(".fits")]
mb = C.make_master_bias([load_image(p) for p in lights_bias])
md = C.make_master_dark([load_image(p) for p in lights_dark], mb)
mf = C.make_master_flat([load_image(p) for p in lights_flat], mb)
cal = [C.calibrate(load_image(p), mb, md, mf) for p in lights]

ref = cal[len(cal) // 2]
shifts = [R.phase_corr(ref, im)[:2] for im in cal]
img, _ = drizzle_stack(cal, shifts, scale=2, pixfrac=0.7)   # drizzle ×2

img, fond = extract_gradient(img, "Polynôme 2D", 3, 12, 2.5, "Soustraction")
img = denoise(img, "Non-local means", 0.5)
save_image("resultat.fits", img)
```

## 4. Astrométrie : catalogue Gaia, plate solving, photométrie

```python
from astroforge.core import equipment as EQ
from astroforge.core import astrometry as A
from astroforge.core.image_io import load_image

# a) catalogue Gaia une seule fois (internet + pip install astroquery)
EQ.make_gaia_catalog(156.25, 19.75, 0.6, "profiles/gaia_champ.csv", mag_limit=15)

# b) focale automatique depuis une image
img = load_image("m51.fits")
res = A.solve_image(img, catalog_path="profiles/gaia_champ.csv",
                    center_ra=156.25, center_dec=19.75)
focal = EQ.focal_from_scale(pixel_um=3.8, scale_arcsec_pix=res["scale_arcsec_pix"])
print(f"échelle {res['scale_arcsec_pix']:.3f}\"/px → focale {focal:.1f} mm")

# c) photométrie d'ouverture calibrée -> CSV
tab = A.aperture_photometry(img, wcs=res["wcs"], catalog_path="profiles/gaia_champ.csv")
tab.write("photometrie.csv", format="csv")
```

## 5. Ondelettes à la Registax + fausses couleurs (Soleil)

```python
from astroforge.core.wavelets import atrous_decompose, atrous_recompose
from astroforge.core.effects import apply_palette

layers = atrous_decompose(img, 6)          # 6 détails + résidu
gains  = [1.4, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0]
img = atrous_recompose(layers, gains)
img = apply_palette(img, "Solaire (orange)")   # images N&B (H-alpha…)
```

## 6. IA locale

```python
from astroforge.ai.manager import get_ai_manager

ai = get_ai_manager()
ai.settings["graxpert"]["binary"] = "/opt/graxpert/GraXpert-linux"  # une fois
img = ai.run("denoise", img, "graxpert", 0.5)        # ou provider "onnx"
img = ai.run("superres", img, "onnx")                # modèle dans models/ai/
```

## 7. Exécuter en tâche de fond / enchaîner

```bash
nohup python3 mon_script.py > log.txt 2>&1 &
```
Chaque fonction lourde accepte `progress=ma_fonction(fait, total)` pour
l'affichage d'avancement ; toutes travaillent en float32 [0..1].
