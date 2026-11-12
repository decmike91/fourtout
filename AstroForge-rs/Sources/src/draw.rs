//! Dessin sur image RGB 8 bits : segments épais, ellipses orientées,
//! texte (police bitmap 8×8 mise à l'échelle).

pub struct Canvas {
    pub w: usize,
    pub h: usize,
    pub data: Vec<u8>, // RGB
}

impl Canvas {
    pub fn from_rgb_f32(img: &crate::image::Image) -> Canvas {
        let rgb = img.to_rgb();
        Canvas { w: rgb.w, h: rgb.h, data: rgb.data.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0) as u8).collect() }
    }

    pub fn to_image(&self) -> crate::image::Image {
        crate::image::Image::from_vec(self.w, self.h, 3, self.data.iter().map(|&v| v as f32 / 255.0).collect())
    }

    #[inline]
    fn blend(&mut self, x: isize, y: isize, col: [u8; 3], a: f64) {
        if x < 0 || y < 0 || x >= self.w as isize || y >= self.h as isize || a <= 0.0 {
            return;
        }
        let i = (y as usize * self.w + x as usize) * 3;
        let a = a.min(1.0);
        for k in 0..3 {
            self.data[i + k] = (self.data[i + k] as f64 * (1.0 - a) + col[k] as f64 * a).round() as u8;
        }
    }

    /// Segment épais anticrénelé.
    pub fn line(&mut self, p1: (f64, f64), p2: (f64, f64), col: [u8; 3], thickness: f64) {
        let r = (thickness / 2.0).max(0.5);
        let (x0, y0, x1, y1) = (p1.0, p1.1, p2.0, p2.1);
        let (minx, maxx) = ((x0.min(x1) - r - 1.0).floor() as isize, (x0.max(x1) + r + 1.0).ceil() as isize);
        let (miny, maxy) = ((y0.min(y1) - r - 1.0).floor() as isize, (y0.max(y1) + r + 1.0).ceil() as isize);
        let (vx, vy) = (x1 - x0, y1 - y0);
        let l2 = (vx * vx + vy * vy).max(1e-12);
        for y in miny.max(0)..=maxy.min(self.h as isize - 1) {
            for x in minx.max(0)..=maxx.min(self.w as isize - 1) {
                let t = (((x as f64 - x0) * vx + (y as f64 - y0) * vy) / l2).clamp(0.0, 1.0);
                let d = ((x as f64 - (x0 + t * vx)).powi(2) + (y as f64 - (y0 + t * vy)).powi(2)).sqrt();
                let a = (r + 0.5 - d).clamp(0.0, 1.0);
                self.blend(x, y, col, a);
            }
        }
    }

    /// Ellipse (demi-axes a, b ; angle en degrés, sens d'OpenCV).
    pub fn ellipse(&mut self, cx: f64, cy: f64, a: f64, b: f64, angle_deg: f64, col: [u8; 3], thickness: f64) {
        let n = ((a.max(b) * 2.0 * std::f64::consts::PI) / 3.0).ceil().clamp(24.0, 720.0) as usize;
        let (ca, sa) = (angle_deg.to_radians().cos(), angle_deg.to_radians().sin());
        let pt = |k: usize| {
            let t = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            let (x, y) = (a * t.cos(), b * t.sin());
            (cx + x * ca - y * sa, cy + x * sa + y * ca)
        };
        let mut prev = pt(0);
        for k in 1..=n {
            let p = pt(k);
            self.line(prev, p, col, thickness);
            prev = p;
        }
    }

    /// Texte ASCII (police 8×8) ; `scale` = taille en multiples de 8 px ; (x, y) = coin bas-gauche.
    pub fn text(&mut self, s: &str, x: f64, y: f64, scale: f64, col: [u8; 3]) {
        use font8x8::legacy::BASIC_LEGACY;
        let cell = (8.0 * scale).max(6.0);
        let px = cell / 8.0;
        let top = y - cell;
        for (i, ch) in s.chars().enumerate() {
            let code = ch as usize;
            let glyph = if code < 128 { BASIC_LEGACY[code] } else { BASIC_LEGACY[b'?' as usize] };
            let ox = x + i as f64 * cell * 0.75;
            for (gy, row) in glyph.iter().enumerate() {
                for gx in 0..8 {
                    if row & (1 << gx) == 0 {
                        continue;
                    }
                    let (x0, y0) = (ox + gx as f64 * px, top + gy as f64 * px);
                    for yy in (y0.floor() as isize)..((y0 + px).ceil() as isize) {
                        for xx in (x0.floor() as isize)..((x0 + px).ceil() as isize) {
                            self.blend(xx, yy, col, 1.0);
                        }
                    }
                }
            }
        }
    }
}
