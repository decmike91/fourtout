# -*- coding: utf-8 -*-
"""Aperçu : canvas d'image, panneau avant/après, contrôleur d'aperçu live."""
from __future__ import annotations
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel,
                               QSplitter, QVBoxLayout, QWidget)

from ..core.display import autostretch, histogram
from ..core.image_io import is_color, to_uint8


def np_to_qimage(img: np.ndarray) -> QImage:
    """float [0..1] (HxW ou HxWx3 RGB) -> QImage 8 bits (copie profonde).
    Chaque ligne est complétée pour que le pas mémoire (bytesPerLine) soit
    un multiple de 4 : sinon Qt décale les lignes des images de largeur
    impaire (disque déchiré / rebouclé, image « pixelisée »)."""
    a = to_uint8(np.clip(img, 0, 1))
    if a.ndim == 2:
        h, w = a.shape
        bpl = (w + 3) & ~3
        if bpl != w:
            pad = np.zeros((h, bpl), np.uint8)
            pad[:, :w] = a
            a = pad
        return QImage(a.data, w, h, bpl, QImage.Format_Grayscale8).copy()
    h, w, _ = a.shape
    row = 3 * w
    bpl = (row + 3) & ~3
    if bpl != row:
        pad = np.zeros((h, bpl), np.uint8)
        pad[:, :row] = a.reshape(h, row)
        a = pad
    return QImage(a.data, w, h, bpl, QImage.Format_RGB888).copy()


def image_stats(img: np.ndarray | None) -> dict | None:
    """Statistiques robustes d'une image : médiane, bruit (MAD du résidu
    passe-haut), RSB des hautes lumières (99,5e centile), min/max."""
    if img is None:
        return None
    a = np.asarray(img, np.float32)
    mono = a if a.ndim == 2 else a[..., :3].mean(axis=2)
    step = max(1, int(np.sqrt(mono.size / 1.5e6)))
    m = mono[::step, ::step]
    med = float(np.median(m))
    hp = m[1:, 1:] - 0.25 * (m[1:, 1:] + m[:-1, 1:] + m[1:, :-1] + m[:-1, :-1])
    noise = float(1.4826 * np.median(np.abs(hp - np.median(hp))) / 0.75) + 1e-9
    hi = float(np.percentile(m, 99.5))
    return {"median": med, "noise": noise, "snr": (hi - med) / noise,
            "min": float(m.min()), "max": float(m.max())}


ZOOMS = ("Ajuster", "50 %", "100 %", "200 %", "400 %")


class ImageCanvas(QWidget):
    """Affiche une image ajustée à la taille du widget, ou zoomée (50 % à
    400 %, déplacement à la souris, molette) ; émet imageClicked(fx, fy)
    (coordonnées relatives 0..1) au clic et pixelHovered(x, y) au survol."""

    imageClicked = Signal(float, float)
    pixelHovered = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pm = QPixmap()
        self._geom = None
        self.zoom = 0.0          # 0 = ajuster à la fenêtre
        self.center = None       # centre de la vue (pixels image) en mode zoom
        self._drag = None
        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)

    def set_image(self, img: np.ndarray | None):
        if img is None:
            self._pm = QPixmap()
        else:
            self._pm = QPixmap.fromImage(np_to_qimage(img))
        self.update()

    def set_zoom(self, z: float):
        self.zoom = float(z)
        if self.zoom > 0 and self.center is None and not self._pm.isNull():
            self.center = [self._pm.width() / 2.0, self._pm.height() / 2.0]
        self.update()

    def paintEvent(self, ev):
        from PySide6.QtGui import QPainter
        p = QPainter(self)
        p.fillRect(self.rect(), self.palette().window())
        self._geom = None
        if not self._pm.isNull():
            if self.zoom <= 0:
                scaled = self._pm.scaled(self.size(), Qt.KeepAspectRatio,
                                         Qt.SmoothTransformation)
                x = (self.width() - scaled.width()) // 2
                y = (self.height() - scaled.height()) // 2
                self._geom = (float(x), float(y), float(scaled.width()),
                              float(scaled.height()))
                p.drawPixmap(x, y, scaled)
            else:
                z = self.zoom
                if self.center is None:
                    self.center = [self._pm.width() / 2.0, self._pm.height() / 2.0]
                cx, cy = self.center
                sw, sh = self._pm.width() * z, self._pm.height() * z
                x = self.width() / 2.0 - cx * z
                y = self.height() / 2.0 - cy * z
                self._geom = (x, y, sw, sh)
                from PySide6.QtCore import QRectF
                p.setRenderHint(QPainter.SmoothPixmapTransform, z < 1)
                p.drawPixmap(QRectF(x, y, sw, sh), self._pm, QRectF(self._pm.rect()))
        p.end()

    def _to_image(self, pos):
        if self._pm.isNull() or not self._geom:
            return None
        gx, gy, sw, sh = self._geom
        fx = (pos.x() - gx) / max(1.0, sw)
        fy = (pos.y() - gy) / max(1.0, sh)
        return fx, fy

    def mousePressEvent(self, ev):
        r = self._to_image(ev.position())
        if r is None:
            return
        if self.zoom > 0 and ev.button() == Qt.RightButton or (
                self.zoom > 0 and ev.modifiers() & Qt.ShiftModifier):
            self._drag = (ev.position().x(), ev.position().y(), list(self.center))
            return
        fx, fy = r
        if 0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0:
            self.imageClicked.emit(fx, fy)
        if self.zoom > 0:
            self._drag = (ev.position().x(), ev.position().y(), list(self.center))

    def mouseMoveEvent(self, ev):
        if self._drag is not None and self.zoom > 0:
            x0, y0, c0 = self._drag
            self.center = [c0[0] - (ev.position().x() - x0) / self.zoom,
                           c0[1] - (ev.position().y() - y0) / self.zoom]
            self.update()
        r = self._to_image(ev.position())
        if r is not None:
            fx, fy = r
            if 0.0 <= fx < 1.0 and 0.0 <= fy < 1.0:
                self.pixelHovered.emit(int(fx * self._pm.width()), int(fy * self._pm.height()))

    def mouseReleaseEvent(self, ev):
        self._drag = None

    def wheelEvent(self, ev):
        levels = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        cur = self.zoom if self.zoom > 0 else 0.0
        if ev.angleDelta().y() > 0:
            nxt = next((z for z in levels if z > cur + 1e-9), levels[-1])
        else:
            smaller = [z for z in levels if z < cur - 1e-9]
            nxt = smaller[-1] if smaller else 0.0
        r = self._to_image(ev.position())
        if r is not None and nxt > 0:
            self.center = [r[0] * self._pm.width(), r[1] * self._pm.height()]
        self.set_zoom(nxt)


class PreviewPane(QWidget):
    """Aperçu avec bascule Avant / Après / Côte à côte + étirement auto,
    zoom (ajusté, 50 à 400 %), valeur du pixel survolé et statistiques."""

    paramsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        bar = QHBoxLayout()
        self.cmbMode = QComboBox()
        self.cmbMode.addItems(["Après", "Avant", "Côte à côte"])
        self.chkStretch = QCheckBox("Étirement auto")
        self.chkStretch.setToolTip("Étirement automatique pour l'affichage seulement")
        self.chkStretch.setChecked(True)
        self.cmbZoom = QComboBox()
        self.cmbZoom.setObjectName("cmbPreviewZoom")
        self.cmbZoom.addItems(list(ZOOMS))
        self.cmbZoom.setToolTip("Zoom (molette) ; glisser pour déplacer la vue")
        self.cmbMode.setToolTip("Aperçu : après, avant ou côte à côte")
        bar.addWidget(self.cmbMode)
        bar.addWidget(self.chkStretch)
        bar.addWidget(self.cmbZoom)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.split = QSplitter(Qt.Horizontal)
        self.canvasBefore = ImageCanvas()
        self.canvasAfter = ImageCanvas()
        self.split.addWidget(self.canvasBefore)
        self.split.addWidget(self.canvasAfter)
        lay.addWidget(self.split, 1)
        self.lblInfo = QLabel("")
        self.lblInfo.setObjectName("lblPreviewInfo")
        self.lblInfo.setTextInteractionFlags(Qt.TextSelectableByMouse)
        from PySide6.QtWidgets import QSizePolicy
        # le texte ne doit jamais imposer de largeur minimale à la fenêtre
        self.lblInfo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.cmbZoom.setMinimumWidth(40)
        self.cmbZoom.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cmbZoom.setMinimumContentsLength(4)
        lay.addWidget(self.lblInfo)
        self.cmbMode.currentIndexChanged.connect(self._update_layout)
        self.chkStretch.toggled.connect(self.paramsChanged)
        self.cmbZoom.currentIndexChanged.connect(self._zoom_changed)
        self.canvasBefore.pixelHovered.connect(lambda x, y: self._hover(self._before, x, y))
        self.canvasAfter.pixelHovered.connect(lambda x, y: self._hover(self._after, x, y))
        self._before = None
        self._after = None
        self._stats = None
        self._update_layout()

    def set_before(self, img: np.ndarray | None):
        self._before = img
        self.show_before()

    def set_after(self, img: np.ndarray | None):
        self._after = img
        self._stats = image_stats(img)
        self.show_after()
        self._show_stats()

    def _prep(self, img):
        if img is None:
            return None
        return autostretch(img) if self.chkStretch.isChecked() else np.clip(img, 0, 1)

    def show_before(self):
        self.canvasBefore.set_image(self._prep(self._before))

    def show_after(self):
        self.canvasAfter.set_image(self._prep(self._after))

    def _zoom_changed(self, i):
        z = {0: 0.0, 1: 0.5, 2: 1.0, 3: 2.0, 4: 4.0}.get(i, 0.0)
        for c in (self.canvasBefore, self.canvasAfter):
            c.set_zoom(z)

    def _show_stats(self, extra: str = ""):
        st = self._stats
        if not st:
            self.lblInfo.setText(extra)
            return
        txt = (f"médiane {st['median']:.4f} · bruit {st['noise']:.5f} · RSB {st['snr']:.0f}"
               f" · min {st['min']:.4f} · max {st['max']:.4f}")
        self.lblInfo.setText(txt + (f"   |   {extra}" if extra else ""))

    def _hover(self, img, x, y):
        if img is None:
            return
        a = np.asarray(img)
        if not (0 <= y < a.shape[0] and 0 <= x < a.shape[1]):
            return
        v = a[y, x]
        val = (f"R {v[0]:.4f} V {v[1]:.4f} B {v[2]:.4f}" if np.ndim(v) else f"{float(v):.5f}")
        self._show_stats(f"x {x}, y {y} : {val}")

    def _update_layout(self):
        mode = self.cmbMode.currentIndex()
        self.canvasBefore.setVisible(mode in (1, 2))
        self.canvasAfter.setVisible(mode in (0, 2))


class CurveEditor(QWidget):
    """Éditeur de courbes (principe Photoshop) : carré avec grille et
    diagonale, histogramme de fond, points déplaçables.
    - clic gauche hors point : ajoute un point (drag immédiat) ;
    - clic gauche sur un point : sélection + déplacement ;
    - Suppr/Backspace sur la sélection, double-clic ou clic droit sur un
      point intérieur : suppression ;
    - extrémités épinglées : x = 0 pour la première, x = 1 pour la dernière
      (seul y est réglable) ; x strictement croissants, y dans [0, 1].
    pointsChanged émet à chaque modification (aperçu live à l'appui).
    """
    pointsChanged = Signal()

    MARGIN = 8.0
    GRAB = 8.0        # distance px de saisie d'un point
    GAP = 0.005       # écart minimal en x lors d'un déplacement

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("toneCurve")
        self._points = [[0.0, 0.0], [1.0, 1.0]]
        self._histogram = None
        self._selected = None
        self._dragging = False
        self.setFixedHeight(150)
        self.setMinimumWidth(120)
        self.setFocusPolicy(Qt.ClickFocus)

    # ------------------------------------------------------------- état
    def points(self):
        return [[float(x), float(y)] for x, y in self._points]

    def set_points(self, pts, silent=False):
        from ..core.planetary import _normalize_curve
        self._points = _normalize_curve(pts)
        self._selected = None
        if not silent:
            self.pointsChanged.emit()
        self.update()

    def set_image_histogram(self, img, bins=64):
        """Histogramme de luminance (0..1) en fond, calculé une fois par
        image de prévisualisation — jamais pendant le glissement."""
        from ..core.planetary import luminance
        if img is None:
            self._histogram = None
        else:
            values = np.asarray(luminance(img), np.float64)
            values = values[np.isfinite(values)]
            hist, _ = np.histogram(values, bins, (0, 1)) if values.size else \
                (np.zeros(bins), None)
            peak = float(hist.max()) if hist.size else 0.0
            self._histogram = (hist / peak).tolist() if peak > 0 else None
        self.update()

    # ------------------------------------------------------------- géométrie
    def _plot_rect(self):
        from PySide6.QtCore import QRectF
        side = max(40.0, min(self.width(), self.height()) - 2 * self.MARGIN)
        x = (self.width() - side) / 2
        y = (self.height() - side) / 2
        return QRectF(x, y, side, side)

    def _point_to_px(self, rect, x, y):
        from PySide6.QtCore import QPointF
        return QPointF(rect.left() + x * rect.width(),
                       rect.bottom() - y * rect.height())

    def _px_to_point(self, rect, pos):
        fx = (pos.x() - rect.left()) / max(1e-9, rect.width())
        fy = (rect.bottom() - pos.y()) / max(1e-9, rect.height())
        return float(np.clip(fx, 0, 1)), float(np.clip(fy, 0, 1))

    def _hit(self, rect, pos):
        for i, (x, y) in enumerate(self._points):
            p = self._point_to_px(rect, x, y)
            if abs(p.x() - pos.x()) <= self.GRAB and abs(p.y() - pos.y()) <= self.GRAB:
                return i
        return None

    # ------------------------------------------------------------- interactions
    def mousePressEvent(self, ev):
        rect = self._plot_rect()
        idx = self._hit(rect, ev.position())
        if ev.button() == Qt.RightButton:
            if idx is not None and 0 < idx < len(self._points) - 1:
                self._remove_point(idx)
            return
        if ev.button() != Qt.LeftButton:
            return
        if idx is not None:
            self._selected = idx
        else:
            fx, fy = self._px_to_point(rect, ev.position())
            x = float(np.clip(fx, self._points[0][0] + self.GAP,
                              self._points[-1][0] - self.GAP))
            i = 0
            while i < len(self._points) and self._points[i][0] < x:
                i += 1
            self._points.insert(i, [x, float(fy)])
            self._selected = i
            self._dragging = True
            self.pointsChanged.emit()
        self._dragging = True
        self.setFocus()
        self.update()

    def mouseMoveEvent(self, ev):
        if not self._dragging or self._selected is None:
            return
        rect = self._plot_rect()
        i = self._selected
        lo = self._points[i - 1][0] + self.GAP if i > 0 else 0.0
        hi = self._points[i + 1][0] - self.GAP if i < len(self._points) - 1 else 1.0
        pinned = (i == 0 or i == len(self._points) - 1)
        if pinned:
            self._points[i][1] = float(self._px_to_point(rect, ev.position())[1])
        else:
            fx, fy = self._px_to_point(rect, ev.position())
            self._points[i] = [float(np.clip(fx, lo, hi)), float(fy)]
        self.pointsChanged.emit()
        self.update()

    def mouseReleaseEvent(self, ev):
        self._dragging = False

    def mouseDoubleClickEvent(self, ev):
        idx = self._hit(self._plot_rect(), ev.position())
        if idx is not None:
            self._remove_point(idx)

    def keyPressEvent(self, ev):
        if ev.key() in (Qt.Key_Delete, Qt.Key_Backspace) and self._selected is not None:
            if self._remove_point(self._selected):
                return
        super().keyPressEvent(ev)

    def _remove_point(self, idx):
        if idx is None or not (0 < idx < len(self._points) - 1):
            return False     # extrémités épinglées
        del self._points[idx]
        self._selected = None
        self.pointsChanged.emit()
        self.update()
        return True

    # ------------------------------------------------------------- rendu
    def paintEvent(self, event):
        from PySide6.QtCore import QPointF, QRectF
        from PySide6.QtGui import QColor, QPainter, QPen
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self._plot_rect()
        painter.fillRect(rect, QColor(30, 30, 34) if self.palette().window()
                         .color().lightness() < 128 else QColor(250, 250, 250))
        if self._histogram:  # fond : histogramme de luminance
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(128, 128, 128, 70))
            n = len(self._histogram)
            for i, v in enumerate(self._histogram):
                h = float(v) * rect.height()
                painter.drawRect(QRectF(rect.left() + i * rect.width() / n,
                                        rect.bottom() - h, rect.width() / n, h))
            painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(self.palette().mid().color(), 1))
        painter.drawRect(rect)
        for k in (1, 2, 3):  # grille en quarts
            painter.setPen(QPen(self.palette().mid().color(), 1))
            painter.drawLine(QPointF(rect.left() + k * rect.width() / 4, rect.top()),
                             QPointF(rect.left() + k * rect.width() / 4, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), rect.top() + k * rect.height() / 4),
                             QPointF(rect.right(), rect.top() + k * rect.height() / 4))
        painter.setPen(QPen(self.palette().mid().color(), 1))
        painter.drawLine(rect.bottomLeft(), rect.topRight())  # diagonale
        pts = [self._point_to_px(rect, x, y) for x, y in self._points]
        painter.setPen(QPen(self.palette().highlight().color(), 2))
        for a, b in zip(pts, pts[1:]):
            painter.drawLine(a, b)
        for i, p in enumerate(pts):
            painter.setPen(QPen(self.palette().text().color(), 1)
                           if i == self._selected else Qt.NoPen)
            painter.setBrush(self.palette().highlight())
            painter.drawEllipse(p, 4 if i == self._selected else 3,
                                4 if i == self._selected else 3)
            painter.setBrush(Qt.NoBrush)
        painter.end()


class LivePreview:
    """Debounce + thread : recalcule l'aperçu quand les paramètres changent.
    compute(preview_image, params, progress) -> image résultat."""

    def __init__(self, pane: PreviewPane, get_params, compute, parent=None):
        from .workers import Job
        self.Job = Job
        self.pane = pane
        self.get_params = get_params
        self.compute = compute
        self.job = None
        self.generation = 0
        self.suspended = False
        self._jobs = set()
        self.source = None  # image linéaire réduite
        self.timer = QTimer(parent)
        self.timer.setSingleShot(True)
        self.timer.setInterval(450)
        self.timer.timeout.connect(self.refresh)

    def set_source(self, img: np.ndarray | None):
        """Image de travail (déjà réduite) pour l'aperçu."""
        self.source = img
        self.pane.set_before(img)
        self.refresh()

    def trigger(self):
        if not self.suspended:
            self.timer.start()

    def refresh(self):
        if self.suspended or self.source is None or self.job is not None:
            return
        params = self.get_params()
        source = self.source
        generation = self.generation
        job = self.Job(lambda progress=None: self.compute(source, params, progress),
                       parent=self.timer.parent())
        self.job = job
        self._jobs.add(job)
        job.ok.connect(lambda img: self._done(img) if generation == self.generation else None)
        job.err.connect(self._fail)
        job.finished.connect(lambda: self._finished(job))
        job.start()

    def invalidate(self):
        self.timer.stop()
        self.generation += 1
        for job in self._jobs:
            job.cancel()
        self.job = None

    def _finished(self, job):
        self._jobs.discard(job)
        if self.job is job:
            self.job = None
        job.deleteLater()

    def _run(self, params, progress=None):
        return self.compute(self.source, params, progress)

    def _done(self, img):
        if img is not None:
            self.pane.set_after(img)

    def _fail(self, msg):
        print("[aperçu] erreur :", msg.splitlines()[-1] if msg else msg)


def draw_histogram_pixmap(img: np.ndarray, w=240, h=110) -> QPixmap:
    """Mini-histogramme R/G/B sous forme de QPixmap."""
    from PySide6.QtGui import QColor, QPainter, QPen, QPixmap as QPM
    pm = QPM(w, h)
    pm.fill(QColor(25, 25, 30))
    p = QPainter(pm)
    colors = [(QPen(QColor(200, 60, 60)), 0), (QPen(QColor(60, 200, 60)), 1),
              (QPen(QColor(80, 110, 220)), 2)]
    chans = [img] if not is_color(img) else [img[..., i] for i in range(3)]
    if is_color(img):
        pairs = colors
    else:
        pairs = [(QPen(QColor(220, 220, 220)), 0)]
    for pen, i in pairs:
        hist, _ = histogram(chans[i] if i < len(chans) else chans[0])
        hist = hist.astype(np.float64)
        hist = hist / max(hist.max(), 1)
        p.setPen(pen)
        prev = h
        for x in range(w):
            v = hist[int(x / w * len(hist))]
            y = int(h - v * (h - 4))
            p.drawLine(x, prev, x, y)
            prev = y
    p.end()
    return pm
