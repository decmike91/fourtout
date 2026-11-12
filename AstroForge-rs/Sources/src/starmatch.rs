//! Appariement d'étoiles par triangles (invariants de forme) et similitudes
//! de Umeyama — base de l'alignement par étoiles (équivalent d'astroalign)
//! et du solveur astrométrique.

use nalgebra::{Matrix2, Vector2};

pub type Pt = (f64, f64);

/// Similitude : q = M·p + t.
#[derive(Clone, Copy, Debug)]
pub struct Sim {
    pub m: [[f64; 2]; 2],
    pub t: [f64; 2],
}

impl Sim {
    #[inline]
    pub fn apply(&self, p: Pt) -> Pt {
        (self.m[0][0] * p.0 + self.m[0][1] * p.1 + self.t[0], self.m[1][0] * p.0 + self.m[1][1] * p.1 + self.t[1])
    }
    pub fn det(&self) -> f64 {
        self.m[0][0] * self.m[1][1] - self.m[0][1] * self.m[1][0]
    }
    pub fn scale(&self) -> f64 {
        self.det().abs().sqrt()
    }
    pub fn to_affine(&self) -> crate::geom::Affine {
        [[self.m[0][0], self.m[0][1], self.t[0]], [self.m[1][0], self.m[1][1], self.t[1]]]
    }
    pub fn is_finite(&self) -> bool {
        self.m.iter().flatten().all(|v| v.is_finite()) && self.t.iter().all(|v| v.is_finite())
    }
}

/// Umeyama : dst ≈ s·R·src + t (moindres carrés) ; réflexion permise ou non.
pub fn umeyama(src: &[Pt], dst: &[Pt], allow_reflection: bool) -> Sim {
    let n = src.len().max(1) as f64;
    let ms = (src.iter().map(|p| p.0).sum::<f64>() / n, src.iter().map(|p| p.1).sum::<f64>() / n);
    let md = (dst.iter().map(|p| p.0).sum::<f64>() / n, dst.iter().map(|p| p.1).sum::<f64>() / n);
    let mut cov = Matrix2::<f64>::zeros(); // D0ᵀ·S0
    let mut var = 0.0;
    for (s, d) in src.iter().zip(dst) {
        let s0 = Vector2::new(s.0 - ms.0, s.1 - ms.1);
        let d0 = Vector2::new(d.0 - md.0, d.1 - md.1);
        cov += d0 * s0.transpose();
        var += s0.norm_squared();
    }
    let svd = cov.svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let sv = svd.singular_values;
    let mut d = [1.0f64, 1.0];
    if !allow_reflection && (u * vt).determinant() < 0.0 {
        d[1] = -1.0;
    }
    let dm = Matrix2::new(d[0], 0.0, 0.0, d[1]);
    let r = u * dm * vt;
    let s = (sv[0] * d[0] + sv[1] * d[1]) / var.max(1e-12);
    let m = r * s;
    let t = (md.0 - (m[(0, 0)] * ms.0 + m[(0, 1)] * ms.1), md.1 - (m[(1, 0)] * ms.0 + m[(1, 1)] * ms.1));
    Sim { m: [[m[(0, 0)], m[(0, 1)]], [m[(1, 0)], m[(1, 1)]]], t: [t.0, t.1] }
}

/// Triangles formés par les k premiers points : (descripteurs, sommets
/// ordonnés par côté opposé croissant).
pub fn triangles_all(pts: &[Pt], k: usize) -> (Vec<(f64, f64)>, Vec<[usize; 3]>) {
    let k = k.min(pts.len());
    let mut desc = Vec::new();
    let mut idx = Vec::new();
    for i in 0..k {
        for j in i + 1..k {
            for l in j + 1..k {
                if let Some((d, t)) = triangle(pts, [i, j, l]) {
                    desc.push(d);
                    idx.push(t);
                }
            }
        }
    }
    (desc, idx)
}

fn dist(a: Pt, b: Pt) -> f64 {
    ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
}

fn triangle(pts: &[Pt], tri: [usize; 3]) -> Option<((f64, f64), [usize; 3])> {
    let p = [pts[tri[0]], pts[tri[1]], pts[tri[2]]];
    let sides = [dist(p[1], p[2]), dist(p[0], p[2]), dist(p[0], p[1])];
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| sides[a].total_cmp(&sides[b]));
    let (a, b, c) = (sides[order[0]], sides[order[1]], sides[order[2]]);
    if c < 1e-9 || a / c < 0.05 {
        return None;
    }
    Some(((a / c, b / c), [tri[order[0]], tri[order[1]], tri[order[2]]]))
}

/// Triangles locaux : chaque point et ses 4 plus proches voisins (robuste
/// aux champs partiellement recouvrants, comme astroalign).
pub fn triangles_local(pts: &[Pt], k_nn: usize) -> (Vec<(f64, f64)>, Vec<[usize; 3]>) {
    let mut seen = std::collections::HashSet::new();
    let mut desc = Vec::new();
    let mut idx = Vec::new();
    for i in 0..pts.len() {
        let mut nb: Vec<(f64, usize)> = (0..pts.len()).filter(|&j| j != i).map(|j| (dist(pts[i], pts[j]), j)).collect();
        nb.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut group: Vec<usize> = vec![i];
        group.extend(nb.iter().take(k_nn).map(|x| x.1));
        let g = group.len();
        for a in 0..g {
            for b in a + 1..g {
                for c in b + 1..g {
                    let mut key = [group[a], group[b], group[c]];
                    key.sort();
                    if !seen.insert(key) {
                        continue;
                    }
                    if let Some((d, t)) = triangle(pts, key) {
                        desc.push(d);
                        idx.push(t);
                    }
                }
            }
        }
    }
    (desc, idx)
}

/// Inliers : étoiles `src` transformées à moins de `tol` d'une étoile `dst`,
/// appariement unique. Retourne (masque, indice dst, distance).
pub fn inliers(sim: &Sim, src: &[Pt], dst: &[Pt], tol: f64) -> (Vec<bool>, Vec<usize>, Vec<f64>) {
    let n = src.len();
    let mut ok = vec![false; n];
    let mut j = vec![0usize; n];
    let mut d = vec![f64::INFINITY; n];
    for i in 0..n {
        let q = sim.apply(src[i]);
        for (k, p) in dst.iter().enumerate() {
            let dd = dist(q, *p);
            if dd < d[i] {
                d[i] = dd;
                j[i] = k;
            }
        }
        ok[i] = d[i] < tol;
    }
    let mut seen: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for i in 0..n {
        if !ok[i] {
            continue;
        }
        match seen.get(&j[i]).copied() {
            Some(prev) if d[prev] <= d[i] => ok[i] = false,
            Some(prev) => {
                ok[prev] = false;
                seen.insert(j[i], i);
            }
            None => {
                seen.insert(j[i], i);
            }
        }
    }
    (ok, j, d)
}

/// Résultat d'un appariement.
#[derive(Clone, Debug)]
pub struct Match {
    pub sim: Sim,
    pub n_inliers: usize,
    pub pairs: Vec<(usize, usize)>,
}

/// Similitude envoyant `src` sur `dst` par triangles semblables + vote.
/// `allow_reflection` : miroir permis (astrométrie). `scale_range` : bornes
/// facultatives de l'échelle.
pub fn match_points(src: &[Pt], dst: &[Pt], allow_reflection: bool, tol: f64, scale_range: Option<(f64, f64)>, local: bool) -> Option<Match> {
    if src.len() < 3 || dst.len() < 3 {
        return None;
    }
    let (ds, is) = if local { triangles_local(src, 4) } else { triangles_all(src, 14) };
    let (dd, id) = if local { triangles_local(dst, 4) } else { triangles_all(dst, 24) };
    if ds.is_empty() || dd.is_empty() {
        return None;
    }
    // index grossier des descripteurs (grille 0,01)
    let cell = 0.01;
    let mut grid: std::collections::HashMap<(i32, i32), Vec<usize>> = std::collections::HashMap::new();
    for (k, d) in dd.iter().enumerate() {
        grid.entry(((d.0 / cell) as i32, (d.1 / cell) as i32)).or_default().push(k);
    }
    let mut best: Option<(usize, Sim)> = None;
    let target = 12usize.max((0.6 * src.len() as f64) as usize);
    'outer: for (ti, d) in ds.iter().enumerate() {
        let (gx, gy) = ((d.0 / cell) as i32, (d.1 / cell) as i32);
        for ox in -1..=1 {
            for oy in -1..=1 {
                let Some(list) = grid.get(&(gx + ox, gy + oy)) else { continue };
                for &ci in list {
                    let e = dd[ci];
                    if ((e.0 - d.0).powi(2) + (e.1 - d.1).powi(2)).sqrt() > 0.008 {
                        continue;
                    }
                    let s3: Vec<Pt> = is[ti].iter().map(|&k| src[k]).collect();
                    let d3: Vec<Pt> = id[ci].iter().map(|&k| dst[k]).collect();
                    let sim = umeyama(&s3, &d3, allow_reflection);
                    if !sim.is_finite() {
                        continue;
                    }
                    let sc = sim.scale();
                    if let Some((lo, hi)) = scale_range {
                        if sc < lo || sc > hi {
                            continue;
                        }
                    }
                    let t = (tol * sc).max(tol.min(2.0));
                    let (ok, _, _) = inliers(&sim, src, dst, t);
                    let n = ok.iter().filter(|&&b| b).count();
                    if best.as_ref().map(|b| n > b.0).unwrap_or(true) {
                        best = Some((n, sim));
                    }
                }
            }
        }
        if best.as_ref().map(|b| b.0 >= target).unwrap_or(false) {
            break 'outer;
        }
    }
    let (_, mut sim) = best?;
    // affinement sur tous les appariements
    for _ in 0..3 {
        let t = (tol * sim.scale()).max(tol.min(1.5));
        let (ok, j, _) = inliers(&sim, src, dst, t);
        let s: Vec<Pt> = (0..src.len()).filter(|&i| ok[i]).map(|i| src[i]).collect();
        let d: Vec<Pt> = (0..src.len()).filter(|&i| ok[i]).map(|i| dst[j[i]]).collect();
        if s.len() < 3 {
            break;
        }
        let ns = umeyama(&s, &d, allow_reflection);
        if !ns.is_finite() {
            break;
        }
        sim = ns;
    }
    let t = (tol * sim.scale()).max(tol.min(1.5));
    let (ok, j, _) = inliers(&sim, src, dst, t);
    let pairs: Vec<(usize, usize)> = (0..src.len()).filter(|&i| ok[i]).map(|i| (i, j[i])).collect();
    Some(Match { sim, n_inliers: pairs.len(), pairs })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn recovers_similarity() {
        let src: Vec<Pt> = (0..25).map(|i| ((i * 37 % 101) as f64 * 1.3, (i * 53 % 97) as f64 * 1.1)).collect();
        let a = 0.3f64;
        let true_sim = Sim { m: [[1.02 * a.cos(), -1.02 * a.sin()], [1.02 * a.sin(), 1.02 * a.cos()]], t: [5.0, -3.0] };
        let dst: Vec<Pt> = src.iter().map(|&p| true_sim.apply(p)).collect();
        let m = match_points(&src, &dst, false, 2.0, None, true).unwrap();
        assert!(m.n_inliers >= 20);
        assert!((m.sim.t[0] - 5.0).abs() < 1e-6);
    }
}
