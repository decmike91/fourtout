//! Visionneuse d'images : ajustement, zoom à la molette, déplacement,
//! comparaison avant / après par rideau.

use astroforge::image::Image;
use astroforge::stats;
use eframe::egui::{self, Color32, Pos2, Rect, Sense, Stroke, TextureHandle, TextureOptions, Vec2};

/// Conversion pour l'affichage ; `stretch` : niveaux automatiques (aperçu).
pub fn to_color_image(img: &Image, stretch: bool) -> egui::ColorImage {
    let (lo, hi) = if stretch {
        let l = img.mean_lum();
        let step = (l.data.len() / 200_000).max(1);
        let s: Vec<f32> = l.data.iter().step_by(step).cloned().collect();
        let lo = stats::percentile(&s, 0.05);
        let hi = stats::percentile(&s, 99.95).max(lo + 1e-4);
        (lo, hi)
    } else {
        (0.0, 1.0)
    };
    let k = 255.0 / (hi - lo);
    let px = |v: f32| (((v - lo) * k).clamp(0.0, 255.0) + 0.5) as u8;
    let mut rgba = Vec::with_capacity(img.w * img.h * 4);
    for i in 0..img.w * img.h {
        if img.c >= 3 {
            let p = &img.data[i * img.c..i * img.c + 3];
            rgba.extend_from_slice(&[px(p[0]), px(p[1]), px(p[2]), 255]);
        } else {
            let v = px(img.data[i * img.c]);
            rgba.extend_from_slice(&[v, v, v, 255]);
        }
    }
    egui::ColorImage::from_rgba_unmultiplied([img.w, img.h], &rgba)
}

struct Tex {
    key: u64,
    handle: TextureHandle,
    size: [usize; 2],
}

#[derive(Default)]
pub struct Viewer {
    a: Option<Tex>,
    b: Option<Tex>,
    /// 0 = ajuster à la fenêtre
    zoom: f32,
    offset: Vec2,
    split: f32,
    dragging_split: bool,
}

fn ensure(ctx: &egui::Context, slot: &mut Option<Tex>, name: &str, img: &Image, key: u64, stretch: bool) {
    if slot.as_ref().map(|t| t.key != key || t.size != [img.w, img.h]).unwrap_or(true) {
        let ci = to_color_image(img, stretch);
        match slot {
            Some(t) if t.size == [img.w, img.h] => {
                t.handle.set(ci, TextureOptions::LINEAR);
                t.key = key;
            }
            _ => {
                *slot = Some(Tex { key, handle: ctx.load_texture(name, ci, TextureOptions::LINEAR), size: [img.w, img.h] });
            }
        }
    }
}

impl Viewer {
    pub fn toolbar(&mut self, ui: &mut egui::Ui) {
        if ui.selectable_label(self.zoom == 0.0, "Ajuster").clicked() {
            self.zoom = 0.0;
            self.offset = Vec2::ZERO;
        }
        for (z, l) in [(1.0, "100 %"), (2.0, "200 %"), (3.0, "300 %")] {
            if ui.selectable_label(self.zoom == z, l).clicked() {
                self.zoom = z;
                self.offset = Vec2::ZERO;
            }
        }
    }

    fn layout(&mut self, rect: Rect, size: [usize; 2]) -> (Rect, f32) {
        let (w, h) = (size[0] as f32, size[1] as f32);
        let fit = (rect.width() / w).min(rect.height() / h).max(0.01);
        let scale = if self.zoom == 0.0 { fit } else { self.zoom };
        let center = rect.center() + self.offset;
        (Rect::from_center_size(center, Vec2::new(w * scale, h * scale)), scale)
    }

    fn interact(&mut self, ui: &egui::Ui, resp: &egui::Response, rect: Rect, size: [usize; 2]) {
        if resp.hovered() {
            let (scroll, zoomd) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            let factor = if zoomd != 1.0 { zoomd } else { (scroll / 400.0).exp() };
            if (factor - 1.0).abs() > 1e-4 {
                let (_, cur) = self.layout(rect, size);
                let nz = (cur * factor).clamp(0.05, 16.0);
                if let Some(p) = resp.hover_pos() {
                    // zoom centré sur le curseur
                    let c = rect.center() + self.offset;
                    let rel = p - c;
                    self.offset += rel - rel * (nz / cur);
                }
                self.zoom = nz;
            }
        }
        if resp.dragged() && !self.dragging_split {
            self.offset += resp.drag_delta();
            if self.zoom == 0.0 {
                let (_, cur) = self.layout(rect, size);
                self.zoom = cur;
            }
        }
        if resp.double_clicked() {
            self.zoom = 0.0;
            self.offset = Vec2::ZERO;
        }
    }

    /// Affiche une image (clé : change quand l'image change).
    pub fn show(&mut self, ui: &mut egui::Ui, img: &Image, key: u64, stretch: bool) {
        let ctx = ui.ctx().clone();
        ensure(&ctx, &mut self.a, "vue", img, key, stretch);
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let size = self.a.as_ref().unwrap().size;
        self.interact(ui, &resp, rect, size);
        let (irect, scale) = self.layout(rect, size);
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::from_rgb(8, 9, 12));
        painter.image(self.a.as_ref().unwrap().handle.id(), irect, Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)), Color32::WHITE);
        self.caption(&painter, rect, &format!("{}×{} · {:.0} %", img.w, img.h, scale * 100.0));
    }

    /// Comparaison avant / après : rideau déplaçable.
    pub fn show_compare(&mut self, ui: &mut egui::Ui, before: &Image, kb: u64, after: &Image, ka: u64) {
        let ctx = ui.ctx().clone();
        ensure(&ctx, &mut self.a, "avant", before, kb, true);
        ensure(&ctx, &mut self.b, "apres", after, ka, false);
        if self.split <= 0.0 || self.split >= 1.0 {
            self.split = 0.5;
        }
        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        let size = self.b.as_ref().unwrap().size;
        let (irect0, _) = self.layout(rect, size);
        let sx = irect0.left() + self.split * irect0.width();
        if resp.drag_started() {
            if let Some(p) = resp.interact_pointer_pos() {
                self.dragging_split = (p.x - sx).abs() < 12.0;
            }
        }
        if resp.drag_stopped() {
            self.dragging_split = false;
        }
        if self.dragging_split {
            if let Some(p) = resp.interact_pointer_pos() {
                self.split = ((p.x - irect0.left()) / irect0.width()).clamp(0.01, 0.99);
            }
        }
        self.interact(ui, &resp, rect, size);
        let (irect, scale) = self.layout(rect, size);
        let sx = irect.left() + self.split * irect.width();
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, Color32::from_rgb(8, 9, 12));
        let uv_split = self.split;
        let left = Rect::from_min_max(irect.min, Pos2::new(sx, irect.max.y));
        let right = Rect::from_min_max(Pos2::new(sx, irect.min.y), irect.max);
        painter.image(self.a.as_ref().unwrap().handle.id(), left, Rect::from_min_max(Pos2::ZERO, Pos2::new(uv_split, 1.0)), Color32::WHITE);
        painter.image(self.b.as_ref().unwrap().handle.id(), right, Rect::from_min_max(Pos2::new(uv_split, 0.0), Pos2::new(1.0, 1.0)), Color32::WHITE);
        let accent = Color32::from_rgb(240, 170, 80);
        painter.line_segment([Pos2::new(sx, rect.top()), Pos2::new(sx, rect.bottom())], Stroke::new(2.0_f32, accent));
        painter.circle_filled(Pos2::new(sx, rect.center().y), 7.0, accent);
        let f = egui::FontId::proportional(13.0);
        painter.text(Pos2::new(sx - 10.0, rect.top() + 10.0), egui::Align2::RIGHT_TOP, "Empilement brut", f.clone(), Color32::from_gray(220));
        painter.text(Pos2::new(sx + 10.0, rect.top() + 10.0), egui::Align2::LEFT_TOP, "Résultat IA", f, Color32::from_gray(220));
        if resp.hovered() && ui.input(|i| i.pointer.hover_pos()).map(|p| (p.x - sx).abs() < 12.0).unwrap_or(false) {
            ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeHorizontal);
        }
        self.caption(&painter, rect, &format!("{}×{} · {:.0} % · faites glisser le rideau", after.w, after.h, scale * 100.0));
    }

    fn caption(&self, painter: &egui::Painter, rect: Rect, text: &str) {
        painter.text(rect.left_bottom() + Vec2::new(8.0, -6.0), egui::Align2::LEFT_BOTTOM, text, egui::FontId::proportional(12.0), Color32::from_gray(150));
    }
}
