use astroforge::{geom, io, ndimage, registration, display, stacking};
use std::path::Path;
fn main() -> anyhow::Result<()> {
    let d = std::env::args().nth(1).unwrap();
    let p = |n: &str| Path::new(&d).join(n);
    let (a, _) = io::fits::read_fits(&p("a.fits"))?;
    let (b, _) = io::fits::read_fits(&p("b.fits"))?;
    let (c, _) = io::fits::read_fits(&p("c.fits"))?;
    io::fits::write_fits(&p("r_gauss.fits"), &ndimage::gaussian(&a, 2.5), None, false)?;
    io::fits::write_fits(&p("r_med5.fits"), &ndimage::median_filter(&a, 5), None, false)?;
    let m = geom::shift_matrix(a.w, a.h, 3.3, -2.2, 7.0);
    io::fits::write_fits(&p("r_warp.fits"), &geom::warp(&a, &m), None, false)?;
    io::fits::write_fits(&p("r_resize.fits"), &geom::resize(&a, 217, 151, geom::Interp::Cubic), None, false)?;
    io::fits::write_fits(&p("r_area.fits"), &geom::resize(&a, 50, 40, geom::Interp::Area), None, false)?;
    io::fits::write_fits(&p("r_stretch.fits"), &display::autostretch(&a), None, false)?;
    io::fits::write_fits(&p("r_lap.fits"), &ndimage::laplace(&a), None, false)?;
    let (dx, dy, pk) = registration::phase_corr(&a, &b);
    println!("phase {:.4} {:.4} {:.6e}", dx, dy, pk);
    let lp = registration::logpolar_rotation(&a, &c);
    println!("logpolar {:.4} {:.4} {:.6e}", lp.0, lp.1, lp.2);
    println!("rot {:.4}", registration::estimate_rotation(&a, &c, 3.0, 0.25, "auto"));
    let fr: Vec<_> = (0..7).map(|k| io::fits::read_fits(&p(&format!("s{}.fits", k))).unwrap().0).collect();
    for m in stacking::METHODS {
        let o = stacking::stack_owned(&fr, m, 3.0, None)?;
        io::fits::write_fits(&p(&format!("r_stack_{}.fits", m.replace('/', "_").replace(' ', "_"))), &o, None, false)?;
    }
    let cfa = io::fits::read_fits(&p("cfa.fits"))?.0;
    io::fits::write_fits(&p("r_debayer.fits"), &io::debayer(&cfa, "RGGB")?, None, false)?;
    Ok(())
}
