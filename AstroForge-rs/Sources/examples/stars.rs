fn main() {
    let p = std::env::args().nth(1).unwrap();
    let img = astroforge::io::load(&p).unwrap();
    let mono = img.mean_lum();
    for s in astroforge::stars::detect(&mono, 1000, 3.0, 8.0) {
        println!("{:.4} {:.4} {:.6}", s.x, s.y, s.flux);
    }
}
