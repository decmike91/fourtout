"""Commandes réversibles : références aux tableaux, jamais de copie des pixels."""
from __future__ import annotations

from collections import deque

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout,
                              QHBoxLayout, QLabel, QLayout, QLineEdit,
                              QListWidget, QPushButton, QSpinBox, QSlider, QWidget)
from .preview import CurveEditor
from .workers import Job


def shallow(value):
    if isinstance(value, dict):
        return {k: shallow(v) for k, v in value.items()}
    if isinstance(value, list):
        return list(value)
    return value


class StepCommands:
    """Un seul calcul actif ; annulation coopérative et rejet des signaux tardifs."""
    def _init_commands(self, actions, attributes):
        self._command_attributes = attributes
        self._history = {}
        self._serial = 0
        self._active_command = None
        self._jobs = set()
        self._command_buttons = {}
        self._busy_widgets = None
        self._command_actions = actions
        # Pile d'annulation globale (v1.9.6) : dix actions, références only.
        self._undo_stack = deque(maxlen=10)
        self._pending_undo = None
        self._no_undo_commands = set()
        scroll = self.ui.findChild(object, "scrollParams")
        scroll.setMinimumWidth(380)
        scroll.setMaximumWidth(560)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.ui.findChild(QLayout, "paramsLayout").setSizeConstraint(QLayout.SetMinAndMaxSize)
        for name in actions:
            group = self.ui.findChild(object, name)
            layout = group.layout()
            launch = self.ui.findChild(QPushButton, "btnLaunch_" + name)
            cancel = self.ui.findChild(QPushButton, "btnCancel_" + name)
            if launch is None:
                row = QHBoxLayout()
                launch = QPushButton("Lancer", group)
                launch.setObjectName("btnLaunch_" + name)
                cancel = QPushButton("Annuler", group)
                cancel.setObjectName("btnCancel_" + name)
                row.addWidget(launch)
                row.addWidget(cancel)
                if isinstance(layout, QGridLayout):
                    layout.addLayout(row, layout.rowCount(), 0, 1, layout.columnCount())
                else:
                    layout.addLayout(row)
            launch.setMinimumHeight(30)
            cancel.setMinimumHeight(30)
            launch.clicked.connect(lambda checked=False, n=name: self._execute(n))
            cancel.clicked.connect(lambda checked=False, n=name: self._cancel_command(n))
            self._command_buttons[name] = (launch, cancel)
        # aucun bouton écrasé quand la fenêtre est basse : la zone défile
        for btn in scroll.widget().findChildren(QPushButton):
            if btn.minimumHeight() < 28:
                btn.setMinimumHeight(28)
        for label in self.ui.findChildren(QLabel):
            label.setWordWrap(True)
        self._refresh_commands()

    def _snapshot(self):
        controls = []
        for widget in self.ui.findChildren(QWidget):
            if not isinstance(widget, (QCheckBox, QComboBox, QDoubleSpinBox,
                                       QSpinBox, QSlider, QLineEdit)):
                continue
            if isinstance(widget, QCheckBox):
                value = widget.isChecked()
            elif isinstance(widget, QComboBox):
                value = widget.currentIndex()
            elif isinstance(widget, QLineEdit):
                value = widget.text()
            else:
                value = widget.value()
            limits = (widget.minimum(), widget.maximum()) if isinstance(
                widget, (QDoubleSpinBox, QSpinBox, QSlider)) else None
            controls.append((widget, value, limits))
        return {"attrs": {n: shallow(getattr(self, n)) for n in self._command_attributes},
                "controls": controls, "before": self.pane._before,
                "after": self.pane._after, "mode": self.pane.cmbMode.currentIndex(),
                "source": self.live.source,
                "labels": [(w, w.text()) for w in self.ui.findChildren(QLabel)],
                "shared": self.shared.get("image"),
                "extra": self._extra_state() if hasattr(self, "_extra_state") else None}

    def _restore(self, snapshot):
        for name, value in snapshot["attrs"].items():
            setattr(self, name, shallow(value))
        for widget, value, limits in snapshot["controls"]:
            widget.blockSignals(True)
            if limits is not None:
                widget.setRange(*limits)
            if isinstance(widget, QCheckBox):
                widget.setChecked(value)
            elif isinstance(widget, QComboBox):
                widget.setCurrentIndex(value)
            elif isinstance(widget, QLineEdit):
                widget.setText(value)
            else:
                widget.setValue(value)
            widget.blockSignals(False)
        if hasattr(self, "_toggle_place"):
            self._toggle_place(self._placing)
        for widget, text in snapshot["labels"]:
            widget.setText(text)
        self.live.source = snapshot["source"]
        self.pane.set_before(snapshot["before"])
        self.pane.set_after(snapshot["after"])
        self.pane.cmbMode.setCurrentIndex(snapshot["mode"])
        self.shared["image"] = snapshot["shared"]
        if hasattr(self, "state"):
            for name, key in (("listLight", "light_paths"), ("listHDR", "hdr_images")):
                widget = self.ui.findChild(QListWidget, name)
                widget.clear()
                widget.addItems(self.state[key])
        if hasattr(self, "_sync_rank_navigation"):
            self._sync_rank_navigation()
        extra = snapshot.get("extra")
        if extra is not None and hasattr(self, "_restore_extra"):
            self._restore_extra(extra)
        if hasattr(self, "curve"):
            self.curve.update()

    def _execute(self, name, action=None, undoable=True):
        if self._active_command is not None:
            self._log("Un traitement est déjà actif ; annulez-le ou attendez sa fin.")
            return
        self.live.invalidate()
        self.live.suspended = True
        self._serial += 1
        serial = self._serial
        snapshot = self._snapshot()
        previous = self._history.get(name)
        self._history[name] = (serial, snapshot)
        if undoable and name not in self._no_undo_commands:
            # Pile d'annulation globale : l'état AVANT l'action ; conservé
            # seulement si l'action aboutit (références, jamais de pixels).
            self._undo_stack.append((serial, name, snapshot))
            self._pending_undo = serial
        self._active_command = name
        self._job = None
        try:
            (action or self._command_actions[name])()
        except Exception as exc:
            self._restore(snapshot)
            if previous is None:
                self._history.pop(name, None)
            else:
                self._history[name] = previous
            self._log(f"Traitement impossible : {exc}")
            self._drop_pending_undo()
        finally:
            if self._job is None:
                self._active_command = None
                if self._pending_undo == serial:  # action synchrone terminée
                    self._pending_undo = None
            self._refresh_commands()

    def _new_job(self, fn):
        job = Job(fn, parent=self)
        job.command_serial = self._serial
        self._jobs.add(job)
        job.err.connect(lambda message, j=job: self._command_failed(j, message))
        job.finished.connect(lambda j=job: self._job_finished(j))
        return job

    def _command_failed(self, job, message):
        self._drop_pending_undo()
        if job.command_serial == self._serial and self._active_command in self._history:
            _, snapshot = self._history.pop(self._active_command)
            self.live.invalidate()
            self._restore(snapshot)
        self._log(message.splitlines()[-1])

    def _connect_done(self, job, callback):
        def done(result):
            if job.cancelled or job.command_serial != self._serial:
                return
            try:
                callback(result)
                if self._pending_undo == job.command_serial:
                    self._pending_undo = None  # action terminée : entry conservée
                    self._update_undo_button()
            except Exception as exc:
                self._command_failed(job, str(exc))
        job.ok.connect(done)

    def _connect_error(self, job, callback):
        job.err.connect(lambda message: callback(message)
                        if not job.cancelled and job.command_serial == self._serial else None)

    def _job_finished(self, job):
        self._jobs.discard(job)
        if getattr(self, "_job", None) is job:
            self._active_command = None
        self._refresh_commands()
        job.deleteLater()
        if getattr(self, "_job", None) is job:
            self._job = None

    def _cancel_command(self, name):
        if self._active_command is not None and self._active_command != name:
            return
        entry = self._history.get(name)
        if entry is None:
            return
        serial, snapshot = entry
        self._serial += 1
        self._drop_pending_undo()
        self.live.invalidate()
        for job in self._jobs:
            job.cancel()
        self._restore(snapshot)
        # Tout résultat postérieur dépend potentiellement de l'opération annulée.
        self._history = {k: v for k, v in self._history.items() if v[0] < serial}
        self._log("Opération annulée ; état précédent restauré.")
        self._refresh_commands()

    # ------------------------------------------------ annulation globale (1.9.6)
    def _drop_pending_undo(self):
        """La commande courante n'aboutit pas : son instantané est retiré."""
        serial = self._pending_undo
        self._pending_undo = None
        if serial is not None and any(e[0] == serial for e in self._undo_stack):
            self._undo_stack = deque(
                (e for e in self._undo_stack if e[0] != serial), maxlen=10)
        self._update_undo_button()

    def _undo(self):
        """Annule la dernière action terminée (jusqu'à dix, en ordre inverse)."""
        if self._active_command is not None or not self._undo_stack:
            return
        serial, _name, snapshot = self._undo_stack.pop()
        self._serial += 1
        self._pending_undo = None
        self.live.invalidate()
        self._restore(snapshot)
        # Tout résultat postérieur à l'action annulée est invalidé.
        self._history = {k: v for k, v in self._history.items() if v[0] < serial}
        self._log("Action annulée (annulation globale).")
        self._refresh_commands()

    def _update_undo_button(self):
        btn = self.ui.findChild(QPushButton, "btnUndoP") if hasattr(self, "ui") else None
        if btn is not None:
            n = len(self._undo_stack) - (1 if self._pending_undo is not None else 0)
            btn.setText(f"Annuler ({n})" if n > 0 else "Annuler")
            btn.setEnabled(n > 0 and self._active_command is None)

    def _refresh_commands(self):
        busy = self._active_command is not None
        self.live.suspended = busy
        protected = [w for w in self.ui.findChildren(QWidget) if isinstance(
            w, (QPushButton, QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox,
                QSlider, QLineEdit, QListWidget, CurveEditor))]
        if busy and self._busy_widgets is None:
            self._busy_widgets = [(w, w.isEnabled()) for w in protected]
            for widget, _ in self._busy_widgets:
                widget.setEnabled(False)
        elif not busy and self._busy_widgets is not None:
            for widget, enabled in self._busy_widgets:
                widget.setEnabled(enabled)
            self._busy_widgets = None
            if hasattr(self, "_sync_rank_navigation"):
                self._sync_rank_navigation()
        for name, (launch, cancel) in self._command_buttons.items():
            launch.setEnabled(not busy)
            cancel.setEnabled(name in self._history and
                              (not busy or self._active_command == name))
        self._update_undo_button()


class FitScrollWidth(QObject):
    """Ajuste la largeur minimale d'une zone de défilement à la largeur
    minimale de son contenu (+ barre verticale), à chaque changement de mise
    en page (filtres dynamiques, onglets…)."""

    def __init__(self, scroll):
        super().__init__(scroll)
        self.scroll = scroll
        scroll.widget().installEventFilter(self)
        self.fit()

    def fit(self):
        try:
            self._fit()
        except RuntimeError:  # widget déjà détruit
            pass

    def _fit(self):
        sc = self.scroll
        content = sc.widget()
        if content is None:
            return
        need = (content.minimumSizeHint().width()
                + sc.verticalScrollBar().sizeHint().width() + 2 * sc.frameWidth() + 2)
        need = min(max(need, 380), 640)
        if sc.minimumWidth() < need:
            sc.setMinimumWidth(need)
            if sc.maximumWidth() < need:
                sc.setMaximumWidth(need)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.LayoutRequest:
            QTimer.singleShot(0, self.fit)
        return False
