# Comparatif — AstroForge / Siril / Registax / AutoStakkert! / AstroPixelProcessor

Caractérisation d'ensemble au 22/09/2026 — les outils évoluent, vérifiez les
sites officiels pour les dernières versions.

| Domaine | **AstroForge** | **Siril** | **Registax 6** | **AutoStakkert!** | **AstroPixelProcessor** |
|---|---|---|---|---|---|
| Licence / prix | Libre, gratuit | Libre, gratuit | Gratuit (Windows) | Gratuit (Windows) | Payant (licence unique, essai) |
| Ciel profond : pré-traitement & masters | ✔ | ✔ (référence) | ✘ | ✘ | ✔ |
| Empilement à rejet | 7 algorithmes (dont linear fit, MAD) | 7+ (linear fit, MAD, percentile…) | — (planétaire) | — | ✔ (multi-nuits) |
| Drizzle | ✔ ×2/×3 (noyau carré) | ✔ (intégré, distorsions) | ✘ | ✘ | ✔ |
| Extraction de gradient | ✔ poly 2D / RBF auto | ✔ RBF + points manuels, GraXpert intégré | ✘ | ✘ | ✔ (analytique) |
| Déconvolution / netteté | ✔ Richardson-Lucy | ✔ multi-alg., PSF sur étoiles | ✔ (ondelettes) | ✘ | limité |
| Denoise | ✔ 4 classiques + IA externe | ✔ avancé + IA (GraXpert, Starnet) | ✘ | ✘ | ✔ (simple) |
| IA locale (denoise, super-rés., satellites) | ✔ (ONNX + GraXpert/Cosmic Clarity/DeepSNR) | ✔ (GraXpert, StarNet intégrés) | ✘ | ✘ | ✘ |
| Suppression traces satellites | ✔ Hough + inpainting + médiane | ✔ | ✘ | ✘ | ✔ (rejet) |
| HDR | ✔ (Mertens) | ✔ | ✘ | ✘ | ✔ |
| Astrométrie (plate solving) & photométrie | ✔ local (solve-field ou interne + Gaia) | ✔ (résoud & co + scripts) | ✘ | ✘ | ✔ (mosaïques) |
| Planétaire : SER/AVI, classement qualité | ✔ | ✔ (base) | ✔ (référence) | ✔ (référence qualité) | ✘ |
| Ondelettes (sliders) | ✔ 6 couches live | ✔ (multi-échelle) | ✔ (référence) | ✘ | ✘ |
| Dérotation planétaire | ✔ (auto ou °/min) | ✔ (récemment) | ✘ (WinJUPOS) | ✘ | ✘ |
| Film stabilisé (recadrage commun) | ✔ | ✘ | partiel | ✘ | ✘ |
| Fausses couleurs (Soleil) | ✔ palettes | ✔ (script) | ✔ | ✘ | ✘ |
| Profils de paramètres par outil | ✔ (JSON, standard inclus) | ✔ (réglages) | partiel | partiel | ✔ |
| Scripts / batch | ✔ (JSON + API Python + guide) | ✔ (puissant) | ✘ | ✘ | limité |
| XISF natif | ✔ (lecture/écriture) | ✔ | ✘ | ✘ | ✔ |
| Plateformes | Win / Mac / Linux | Win / Mac / Linux | Windows | Windows | Win / Mac / Linux |

**En résumé** : Siril reste la référence gratuite du ciel profond ; Registax et
AutoStakkert! dominent l'empilement/ondelettes planétaires (Windows) ;
AstroPixelProcessor est fort sur multi-nuits/mosaïques (payant). AstroForge
couvre les deux mondes dans un seul outil scriptable, avec profils, aperçu live
et IA locale — à l'état de v1.x : moins raffiné que chacun sur son terrain.
