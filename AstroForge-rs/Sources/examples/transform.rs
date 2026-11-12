fn main() {
    let a: Vec<String> = std::env::args().collect();
    let r = astroforge::io::load_image(std::path::Path::new(&a[1]), None).unwrap();
    for p in &a[2..] {
        let m = astroforge::io::load_image(std::path::Path::new(p), None).unwrap();
        let t = astroforge::registration::estimate_transform(&r, &m, true);
        println!("{:.5} {:.5} {:.4} {:.5} {:.5} {:.4}", t[0][0], t[0][1], t[0][2], t[1][0], t[1][1], t[1][2]);
    }
}
