//! Tâches de fond : calcul dans un fil séparé, progression et annulation.

use eframe::egui;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Transmet la progression d'une tâche à l'interface.
#[derive(Clone)]
pub struct Reporter {
    shared: Arc<Mutex<(f32, String)>>,
    ctx: egui::Context,
    pub cancel: Arc<AtomicBool>,
}

impl Reporter {
    pub fn report(&self, frac: f64, msg: &str) {
        if let Ok(mut s) = self.shared.lock() {
            *s = (frac as f32, msg.to_string());
        }
        self.ctx.request_repaint();
    }
}

pub struct Job<T> {
    shared: Arc<Mutex<(f32, String)>>,
    rx: Receiver<Result<T, String>>,
    pub cancel: Arc<AtomicBool>,
    pub started: Instant,
}

impl<T: Send + 'static> Job<T> {
    pub fn spawn(ctx: &egui::Context, f: impl FnOnce(Reporter) -> anyhow::Result<T> + Send + 'static) -> Job<T> {
        let shared = Arc::new(Mutex::new((0.0f32, String::from("Démarrage…"))));
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = channel();
        let rep = Reporter { shared: shared.clone(), ctx: ctx.clone(), cancel: cancel.clone() };
        let ctx2 = ctx.clone();
        std::thread::spawn(move || {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(rep)));
            let out = match r {
                Ok(Ok(v)) => Ok(v),
                Ok(Err(e)) => Err(format!("{:#}", e)),
                Err(_) => Err("Erreur interne pendant le calcul.".to_string()),
            };
            let _ = tx.send(out);
            ctx2.request_repaint();
        });
        Job { shared, rx, cancel, started: Instant::now() }
    }

    pub fn progress(&self) -> (f32, String) {
        self.shared.lock().map(|s| s.clone()).unwrap_or((0.0, String::new()))
    }

    pub fn request_cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    /// Résultat si la tâche est terminée.
    pub fn poll(&mut self) -> Option<Result<T, String>> {
        match self.rx.try_recv() {
            Ok(r) => Some(r),
            Err(TryRecvError::Empty) => None,
            Err(TryRecvError::Disconnected) => Some(Err("Tâche interrompue.".into())),
        }
    }
}
