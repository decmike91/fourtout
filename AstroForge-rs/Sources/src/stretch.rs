//! Étirements non linéaires : GHS (Generalized Hyperbolic Stretch, Payne &
//! Cranfield) et arcsinh (Lupton et al., couleurs préservées).

use crate::image::Image;

fn base(u: f64, d: f64, b: f64) -> f64 {
    if (b + 1.0).abs() < 1e-9 {
        return (d * u).ln_1p();
    }
    if b < 0.0 {
        return (1.0 - (1.0 - b * d * u).max(1e-12).powf((b + 1.0) / b)) / (d * (b + 1.0));
    }
    if b.abs() < 1e-9 {
        return 1.0 - (-d * u).exp();
    }
    1.0 - (1.0 + b * d * u).powf(-1.0 / b)
}

fn dbase(u: f64, d: f64, b: f64) -> f64 {
    if (b + 1.0).abs() < 1e-9 {
        return d / (1.0 + d * u);
    }
    if b < 0.0 {
        return (1.0 - b * d * u).max(1e-12).powf(1.0 / b);
    }
    if b.abs() < 1e-9 {
        return d * (-d * u).exp();
    }
    d * (1.0 + b * d * u).powf(-(1.0 + b) / b)
}

/// Courbe GHS (x ∈ [0,1]) ; D « utilisateur » 0..10 (exp(D) − 1 interne).
pub struct Ghs {
    dn: f64,
    b: f64,
    sp: f64,
    lp: f64,
    hp: f64,
    y0: f64,
    y1: f64,
    active: bool,
}

impl Ghs {
    pub fn new(d: f64, b: f64, sp: f64, lp: f64, hp: f64) -> Ghs {
        if d <= 0.0 {
            return Ghs { dn: 0.0, b: 0.0, sp: 0.0, lp: 0.0, hp: 1.0, y0: 0.0, y1: 1.0, active: false };
        }
        let dn = d.exp_m1();
        let b = b.clamp(-5.0, 15.0);
        let sp = sp.clamp(0.0, 1.0);
        let lp = lp.clamp(0.0, sp);
        let hp = hp.clamp(sp, 1.0);
        let mut g = Ghs { dn, b, sp, lp, hp, y0: 0.0, y1: 1.0, active: true };
        g.y0 = if lp <= 0.0 { g.t(0.0) } else { g.t(lp) - g.dt(lp) * lp };
        g.y1 = if hp >= 1.0 { g.t(1.0) } else { g.t(hp) + g.dt(hp) * (1.0 - hp) };
        g
    }
    fn t(&self, v: f64) -> f64 {
        if v >= self.sp {
            base(v - self.sp, self.dn, self.b)
        } else {
            -base(self.sp - v, self.dn, self.b)
        }
    }
    fn dt(&self, v: f64) -> f64 {
        dbase((v - self.sp).abs(), self.dn, self.b)
    }
    pub fn eval(&self, x: f64) -> f64 {
        let x = x.clamp(0.0, 1.0);
        if !self.active {
            return x;
        }
        let mut y = self.t(x);
        if self.lp < self.sp && x < self.lp {
            y = self.t(self.lp) + self.dt(self.lp) * (x - self.lp);
        }
        if self.hp > self.sp && x > self.hp {
            y = self.t(self.hp) + self.dt(self.hp) * (x - self.hp);
        }
        ((y - self.y0) / (self.y1 - self.y0).max(1e-12)).clamp(0.0, 1.0)
    }
}

pub fn ghs_curve(x: f64, d: f64, b: f64, sp: f64, lp: f64, hp: f64) -> f64 {
    Ghs::new(d, b, sp, lp, hp).eval(x)
}

/// Étirement GHS : « Canaux » (chaque canal) ou « Luminance » (teintes conservées).
pub fn ghs(img: &Image, d: f64, b: f64, sp: f64, lp: f64, hp: f64, mode: &str) -> Image {
    if d <= 0.0 {
        return img.clip01();
    }
    let g = Ghs::new(d, b, sp, lp, hp);
    if img.c == 3 && mode.starts_with("Lum") {
        let mut out = img.clone();
        for p in out.data.chunks_mut(3) {
            let lum = 0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64;
            let k = g.eval(lum) / lum.max(1e-6);
            let mut v = [p[0] as f64 * k, p[1] as f64 * k, p[2] as f64 * k];
            let mx = v[0].max(v[1]).max(v[2]);
            if mx > 1.0 {
                v.iter_mut().for_each(|x| *x /= mx);
            }
            for i in 0..3 {
                p[i] = v[i].clamp(0.0, 1.0) as f32;
            }
        }
        return out;
    }
    img.map(|v| g.eval(v as f64) as f32)
}

/// Étirement arcsinh préservant les couleurs.
pub fn arcsinh_stretch(img: &Image, stretch: f64, black: f64) -> Image {
    let a = img.clip_min(0.0);
    let beta = stretch.max(0.0);
    let bk = black.clamp(0.0, 0.5);
    if beta <= 0.0 && bk <= 0.0 {
        return a.clip01();
    }
    let a = a.map(|v| ((v as f64 - bk) / (1.0 - bk).max(1e-6)).max(0.0) as f32);
    if beta <= 0.0 {
        return a.clip01();
    }
    let ab = beta.asinh();
    let c = a.c;
    let mut out = a.clone();
    for p in out.data.chunks_mut(c) {
        let lum = if c == 3 { (p[0] + p[1] + p[2]) as f64 / 3.0 } else { p[0] as f64 };
        let k = if lum > 1e-9 { (beta * lum).asinh() / (lum.max(1e-9) * ab) } else { beta / ab };
        for v in p.iter_mut() {
            *v = (*v as f64 * k) as f32;
        }
        if c == 3 {
            let mx = p[0].max(p[1]).max(p[2]);
            if mx > 1.0 {
                p.iter_mut().for_each(|x| *x /= mx);
            }
        }
        for v in p.iter_mut() {
            *v = v.clamp(0.0, 1.0);
        }
    }
    out
}
