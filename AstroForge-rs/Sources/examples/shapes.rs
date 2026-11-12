fn main() {
    let p = std::env::args().nth(1).unwrap();
    let (img, _) = astroforge::io::load_cfa(std::path::Path::new(&p)).unwrap();
    let a = img.mean_lum();
    let (w, h) = (a.w / 2, a.h / 2);
    let mut b = astroforge::Image::new(w, h, 1);
    for y in 0..h { for x in 0..w {
        b.data[y*w+x] = (a.data[2*y*a.w+2*x] + a.data[2*y*a.w+2*x+1] + a.data[(2*y+1)*a.w+2*x] + a.data[(2*y+1)*a.w+2*x+1]) / 4.0;
    }}
    let bg = astroforge::stats::median(&b.data) as f64;
    for (x, y) in astroforge::stars::detect_stars_opts(&b, 120, 3.0, 6.0) {
        match astroforge::subframes::star_shape(&b, x, y, bg, 1.5, 20) {
            Some(r) => println!("{:.3} {:.3} {:.4} {:.4}", x, y, r.0, r.1),
            None => println!("{:.3} {:.3} none", x, y),
        }
    }
}
