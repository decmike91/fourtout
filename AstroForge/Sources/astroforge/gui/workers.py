# -*- coding: utf-8 -*-
"""Workers : exécution des traitements en thread pour garder l'interface réactive."""
from __future__ import annotations
import traceback
from threading import Event

from PySide6.QtCore import QThread, Signal


class Job(QThread):
    """Exécute fn(*args) en arrière-plan. Signaux : progress, ok, err."""
    progress = Signal(int, int, str)
    ok = Signal(object)
    err = Signal(str)

    def __init__(self, fn, *args, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.args = args
        self._cancelled = Event()

    @property
    def cancelled(self):
        return self._cancelled.is_set()

    def cancel(self):
        self._cancelled.set()
        self.requestInterruption()

    def run(self):
        try:
            self.check_cancelled()
            res = self.fn(*self.args, progress=self._prog)
            self.check_cancelled()
            self.ok.emit(res)
        except InterruptedError:
            pass
        except Exception:
            if not self.cancelled:
                self.err.emit(traceback.format_exc())

    def check_cancelled(self):
        if self.cancelled:
            raise InterruptedError("Traitement annulé")

    def _prog(self, done, total, label=""):
        self.check_cancelled()
        self.progress.emit(int(done), max(int(total), 1), str(label))
