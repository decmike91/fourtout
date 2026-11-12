//! AstroForge — interface graphique : onglets « Planétaire » (réglages
//! manuels), « Planétaire IA » (traitement entièrement automatique) et
//! « Time-lapse » (série de vidéos -> animation uniforme).
//!
//!     astroforge-gui [vidéo.ser]
//!     astroforge-gui --auto vidéo.ser    (lance aussitôt le traitement automatique)
//!     astroforge-gui --timelapse DOSSIER [--auto]   (série de vidéos SER)

#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

mod job;
mod plot;
mod tab_auto;
mod tab_manual;
mod tab_timelapse;
mod video;
mod viewer;

use eframe::egui::{self, Color32, RichText};
use std::path::PathBuf;

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Manual,
    Auto,
    Timelapse,
}

struct App {
    tab: Tab,
    video: video::VideoSlot,
    manual: tab_manual::ManualTab,
    auto: tab_auto::AutoTab,
    timelapse: tab_timelapse::TimelapseTab,
    autorun: bool,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>, open: Option<PathBuf>, autorun: bool, timelapse: bool) -> App {
        let mut v = egui::Visuals::dark();
        v.selection.bg_fill = Color32::from_rgb(150, 100, 40);
        v.selection.stroke.color = Color32::from_rgb(250, 200, 130);
        v.hyperlink_color = plot::ACCENT;
        v.panel_fill = Color32::from_rgb(24, 26, 31);
        v.window_fill = Color32::from_rgb(28, 30, 36);
        v.extreme_bg_color = Color32::from_rgb(14, 15, 19);
        // thème sombre imposé (sinon Windows et macOS appliquent le thème du système)
        cc.egui_ctx.set_theme(egui::ThemePreference::Dark);
        cc.egui_ctx.set_visuals_of(egui::Theme::Dark, v);
        let mut app = App { tab: Tab::Auto, video: video::VideoSlot::default(), manual: Default::default(), auto: Default::default(), timelapse: Default::default(), autorun };
        if timelapse {
            app.tab = Tab::Timelapse;
            app.autorun = false;
            if let Some(p) = open {
                app.timelapse.preload(&cc.egui_ctx, vec![p], autorun);
            }
        } else if let Some(p) = open {
            app.video.open(&cc.egui_ctx, vec![p]);
        }
        app
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.video.poll(ctx, self.tab != Tab::Timelapse);
        if self.autorun && self.video.video.is_some() && !self.video.is_loading() {
            self.autorun = false;
            self.tab = Tab::Auto;
            self.auto.request_start();
        }
        egui::TopBottomPanel::top("tabs").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new("AstroForge").strong().size(17.0).color(plot::ACCENT));
                ui.label(RichText::new(astroforge::VERSION).small().weak());
                ui.add_space(18.0);
                let busy = self.auto.is_busy() || self.timelapse.is_busy();
                for (t, l) in [(Tab::Manual, "Planétaire"), (Tab::Auto, "Planétaire IA (automatique)"), (Tab::Timelapse, "Time-lapse (multi-vidéos)")] {
                    let txt = RichText::new(l).size(15.0);
                    if ui.add_enabled(!busy || t == self.tab, egui::Button::selectable(self.tab == t, txt)).clicked() {
                        self.tab = t;
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if let (Some(v), false) = (&self.video.video, self.tab == Tab::Timelapse) {
                        ui.label(RichText::new(format!("{} · {}", v.name, v.info)).small().weak());
                    }
                });
            });
            ui.add_space(2.0);
        });
        match self.tab {
            Tab::Timelapse => self.timelapse.ui(ctx),
            Tab::Manual => self.manual.ui(ctx, &mut self.video),
            Tab::Auto => {
                if let tab_auto::Action::SendToManual(o, gains) = self.auto.ui(ctx, &mut self.video) {
                    let n = self.video.video.as_ref().map(|v| v.n).unwrap_or(1);
                    self.manual.apply_from_auto(o, gains, n);
                    self.tab = Tab::Manual;
                }
            }
        }
    }
}

fn main() -> eframe::Result<()> {
    let args: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
    let autorun = args.iter().any(|a| a == "--auto");
    let timelapse = args.iter().any(|a| a == "--timelapse");
    let open = args.iter().find(|a| !a.to_string_lossy().starts_with("--")).map(PathBuf::from);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1440.0, 900.0]).with_min_inner_size([980.0, 620.0]).with_title(format!("AstroForge {} — Planétaire", astroforge::VERSION)).with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native("AstroForge", options, Box::new(move |cc| Ok(Box::new(App::new(cc, open, autorun, timelapse)))))
}
