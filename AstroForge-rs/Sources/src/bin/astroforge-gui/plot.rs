//! Petits graphiques dessinés directement (courbes et barres).

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Sense, Shape, Stroke, Vec2};

pub const ACCENT: Color32 = Color32::from_rgb(240, 170, 80);
pub const SERIES: Color32 = Color32::from_rgb(120, 170, 235);
const GRID: Color32 = Color32::from_rgb(55, 58, 66);
const TEXT: Color32 = Color32::from_rgb(160, 164, 172);

fn frame(ui: &mut egui::Ui, height: f32) -> (Rect, Rect, egui::Painter, egui::Response) {
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), height), Sense::hover());
    let painter = ui.painter_at(rect);
    let plot = Rect::from_min_max(rect.min + Vec2::new(8.0, 6.0), rect.max - Vec2::new(8.0, 18.0));
    painter.line_segment([plot.left_bottom(), plot.right_bottom()], Stroke::new(1.0_f32, GRID));
    (rect, plot, painter, resp)
}

/// Courbe y(x) ; `mark` : abscisse mise en évidence ; `cut` : trait vertical.
pub fn line(ui: &mut egui::Ui, xs: &[f64], ys: &[f64], mark: Option<f64>, cut: Option<f64>, x_label: &str, fmt_x: &dyn Fn(f64) -> String, height: f32) {
    let pts: Vec<(f64, f64)> = xs.iter().zip(ys).filter(|(x, y)| x.is_finite() && y.is_finite()).map(|(x, y)| (*x, *y)).collect();
    let (rect, plot, painter, resp) = frame(ui, height);
    if pts.len() < 2 {
        return;
    }
    let (x0, x1) = pts.iter().fold((f64::MAX, f64::MIN), |a, p| (a.0.min(p.0), a.1.max(p.0)));
    let (y0, y1) = pts.iter().fold((f64::MAX, f64::MIN), |a, p| (a.0.min(p.1), a.1.max(p.1)));
    let (y0, y1) = if y1 - y0 < 1e-12 { (y0 - 1.0, y1 + 1.0) } else { (y0 - 0.05 * (y1 - y0), y1 + 0.05 * (y1 - y0)) };
    let map = |x: f64, y: f64| Pos2::new(plot.left() + ((x - x0) / (x1 - x0).max(1e-12)) as f32 * plot.width(), plot.bottom() - ((y - y0) / (y1 - y0)) as f32 * plot.height());
    if let Some(c) = cut {
        let p = map(c, y0);
        painter.line_segment([Pos2::new(p.x, plot.top()), Pos2::new(p.x, plot.bottom())], Stroke::new(1.0_f32, ACCENT.gamma_multiply(0.7)));
    }
    painter.add(Shape::line(pts.iter().map(|&(x, y)| map(x, y)).collect(), Stroke::new(1.6_f32, SERIES)));
    if pts.len() <= 24 {
        for &(x, y) in &pts {
            painter.circle_filled(map(x, y), 2.5, SERIES);
        }
    }
    if let Some(m) = mark {
        if let Some(&(x, y)) = pts.iter().min_by(|a, b| (a.0 - m).abs().total_cmp(&(b.0 - m).abs())) {
            painter.circle_filled(map(x, y), 5.0, ACCENT);
        }
    }
    let f = FontId::proportional(11.0);
    painter.text(Pos2::new(plot.left(), rect.bottom() - 2.0), Align2::LEFT_BOTTOM, fmt_x(x0), f.clone(), TEXT);
    painter.text(Pos2::new(plot.right(), rect.bottom() - 2.0), Align2::RIGHT_BOTTOM, fmt_x(x1), f.clone(), TEXT);
    painter.text(Pos2::new(plot.center().x, rect.bottom() - 2.0), Align2::CENTER_BOTTOM, x_label, f, TEXT);
    if let Some(h) = resp.hover_pos() {
        if plot.contains(h) {
            let xv = x0 + (h.x - plot.left()) as f64 / plot.width() as f64 * (x1 - x0);
            if let Some(&(x, y)) = pts.iter().min_by(|a, b| (a.0 - xv).abs().total_cmp(&(b.0 - xv).abs())) {
                let p = map(x, y);
                painter.circle_stroke(p, 5.0, Stroke::new(1.5_f32, Color32::WHITE));
                painter.text(p + Vec2::new(6.0, -6.0), Align2::LEFT_BOTTOM, format!("{} : {:.2}", fmt_x(x), y), FontId::proportional(11.0), Color32::WHITE);
            }
        }
    }
}

/// Barres horizontales (libellé, valeur) ; `best` mis en évidence.
pub fn bars(ui: &mut egui::Ui, items: &[(String, f64)], best: Option<usize>) {
    if items.is_empty() {
        return;
    }
    let (lo, hi) = items.iter().fold((f64::MAX, f64::MIN), |a, i| (a.0.min(i.1), a.1.max(i.1)));
    let lo = lo - 0.15 * (hi - lo).max(1e-6);
    for (k, (label, v)) in items.iter().enumerate() {
        let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 18.0), Sense::hover());
        let p = ui.painter_at(rect);
        let frac = ((v - lo) / (hi - lo).max(1e-9)).clamp(0.02, 1.0) as f32;
        let bar = Rect::from_min_size(rect.min + Vec2::new(0.0, 3.0), Vec2::new(rect.width() * 0.22 * frac, 12.0));
        let col = if Some(k) == best { ACCENT } else { SERIES.gamma_multiply(0.6) };
        p.rect_filled(bar, 2.0, col);
        p.text(Pos2::new(rect.left() + rect.width() * 0.24, rect.center().y), Align2::LEFT_CENTER, format!("{:.2}  {}", v, label), FontId::proportional(11.0), if Some(k) == best { Color32::WHITE } else { TEXT });
    }
}
