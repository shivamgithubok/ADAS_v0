"""Full-bleed ADAS cockpit: warning log, speed, limit sign, scrollable view cards."""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import (
    QEasingCurve,
    QEvent,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QSize,
    Qt,
    QVariantAnimation,
    QPropertyAnimation,
    Signal,
)
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

ICE = QColor("#F4F7FB")
GREEN = QColor("#3DDC84")
CYAN = QColor("#4FD2FF")
WINDOW_W, WINDOW_H = 1280, 720

FEATURES = (
    ("adas", "ADAS View"),
    ("live", "Live View"),
)
_MPS_TO_MPH = 2.2369362920544


def _set_text(label, text):
    if label.text() != text:
        label.setText(text)


class WarningCard(QWidget):
    """Left-side warning log. Green only while that warning is active."""

    def __init__(self, kind, parent=None):
        super().__init__(parent)
        self._kind = kind
        self._active = False
        self.setFixedSize(96, 108)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def set_trigger(self, on, title=None, body=""):
        del title, body
        active = bool(on)
        if active == self._active:
            return
        self._active = active
        self.update()

    def paintEvent(self, event):
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        card = QRectF(2, 2, self.width() - 4, self.height() - 4)
        border = GREEN if self._active else QColor("#3A4553")
        fill = QColor(14, 36, 28, 210) if self._active else QColor(16, 20, 28, 185)
        icon = GREEN if self._active else QColor("#8E98A4")
        p.setPen(QPen(border, 2.0))
        p.setBrush(fill)
        p.drawRoundedRect(card, 16, 16)
        self._draw_icon(p, QRectF(card.left(), card.top() + 8, card.width(), 58), icon)
        p.setPen(icon if self._active else QColor("#C5CDD6"))
        p.setFont(QFont("Segoe UI", 11, QFont.DemiBold))
        p.drawText(QRectF(card.left(), card.bottom() - 30, card.width(), 22), Qt.AlignCenter, self._kind)
        p.end()

    def _draw_icon(self, p, area, color):
        p.save()
        p.setPen(QPen(color, 2.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.setBrush(Qt.NoBrush)
        cx = area.center().x()
        top = area.top() + 6
        bot = area.bottom() - 2
        if self._kind == "LDW":
            p.drawLine(QPointF(cx, top), QPointF(cx, bot))
            p.drawLine(QPointF(cx - 8, top + 10), QPointF(cx - 16, bot))
            p.drawLine(QPointF(cx + 8, top + 10), QPointF(cx + 16, bot))
        elif self._kind == "FCW":
            body = QRectF(cx - 16, top + 16, 32, 16)
            p.drawRoundedRect(body, 4, 4)
            p.drawLine(QPointF(cx - 10, top + 16), QPointF(cx - 6, top + 8))
            p.drawLine(QPointF(cx + 10, top + 16), QPointF(cx + 6, top + 8))
            p.drawLine(QPointF(cx - 6, top + 8), QPointF(cx + 6, top + 8))
            p.setBrush(color)
            p.drawEllipse(QPointF(cx - 8, top + 34), 3.2, 3.2)
            p.drawEllipse(QPointF(cx + 8, top + 34), 3.2, 3.2)
        else:
            body = QRectF(cx - 4, top + 16, 22, 14)
            p.drawRoundedRect(body, 3, 3)
            p.drawArc(QRectF(cx - 28, top + 10, 22, 22), 40 * 16, 110 * 16)
            p.drawArc(QRectF(cx - 22, top + 16, 14, 14), 40 * 16, 110 * 16)
        p.restore()


class SpeedReadout(QWidget):
    """Large mph figure on the right."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._shown = 0
        self.setFixedSize(150, 118)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def set_mps(self, mps):
        shown = 0 if mps is None else int(round(max(0.0, float(mps)) * _MPS_TO_MPH))
        if shown == self._shown:
            return
        self._shown = shown
        self.update()

    def paintEvent(self, event):
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(ICE)
        p.setFont(QFont("Segoe UI", 62, QFont.Normal))
        p.drawText(QRect(0, 0, self.width(), 78), Qt.AlignCenter, str(self._shown))
        p.setPen(QColor("#D5DDE6"))
        p.setFont(QFont("Segoe UI", 16, QFont.DemiBold))
        p.drawText(QRect(0, 76, self.width(), 28), Qt.AlignHCenter | Qt.AlignTop, "MPH")
        p.end()


class LimitSign(QWidget):
    """US speed-limit sign. text() is the live posted number."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = None
        self.setFixedSize(124, 156)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def set_limit(self, value):
        shown = None if value is None else int(round(float(value)))
        if shown == self._value:
            return
        self._value = shown
        self.update()

    def text(self):
        return "—" if self._value is None else str(self._value)

    def setText(self, text):
        digits = "".join(ch for ch in str(text) if ch.isdigit())
        self.set_limit(int(digits) if digits else None)

    def paintEvent(self, event):
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        board = QRectF(3, 2, self.width() - 6, self.height() - 5)
        p.setPen(QPen(QColor("#111111"), 3))
        p.setBrush(QColor("#F7F7F7"))
        p.drawRoundedRect(board, 2, 2)
        p.setPen(QColor("#111111"))
        p.setFont(QFont("Segoe UI", 11, QFont.Bold))
        p.drawText(QRectF(board.left(), board.top() + 8, board.width(), 16), Qt.AlignCenter, "SPEED")
        p.drawText(QRectF(board.left(), board.top() + 24, board.width(), 16), Qt.AlignCenter, "LIMIT")
        p.setFont(QFont("Segoe UI", 34, QFont.Black))
        p.drawText(QRectF(board.left(), board.top() + 42, board.width(), 52), Qt.AlignCenter, self.text())
        p.setFont(QFont("Segoe UI", 10, QFont.Bold))
        p.drawText(QRectF(board.left(), board.bottom() - 42, board.width(), 16), Qt.AlignCenter, "HIGHWAY")
        p.drawText(QRectF(board.left(), board.bottom() - 26, board.width(), 16), Qt.AlignCenter, "ZONE")
        p.end()


class GearButton(QWidget):
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._on = False
        self.setFixedSize(54, 54)
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def set_active(self, on):
        active = bool(on)
        if active == self._on:
            return
        self._on = active
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)

    def paintEvent(self, event):
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        disc = QRectF(3, 3, 48, 48)
        p.setPen(QPen(CYAN if self._on else QColor("#8E98A3"), 1.6))
        p.setBrush(QColor(18, 22, 30, 210))
        p.drawEllipse(disc)
        color = CYAN if self._on else QColor("#C5CDD6")
        center = disc.center()
        p.setPen(Qt.NoPen)
        p.setBrush(color)
        p.save()
        p.translate(center)
        for i in range(8):
            p.save()
            p.rotate(i * 45)
            p.drawRoundedRect(QRectF(-3.2, -19, 6.4, 8), 1.5, 1.5)
            p.restore()
        p.restore()
        ring = QPainterPath()
        ring.addEllipse(center, 11, 11)
        hole = QPainterPath()
        hole.addEllipse(center, 5.2, 5.2)
        p.drawPath(ring.subtracted(hole))
        p.end()


class StatusGlyphs(QWidget):
    """Battery, signal, and wifi marks beside the clock."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(78, 22)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def paintEvent(self, event):
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        color = QColor("#E8EEF4")
        p.setPen(QPen(color, 1.4))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(1, 6, 18, 10), 2, 2)
        p.setBrush(color)
        p.drawRect(QRectF(19, 9, 2, 4))
        p.drawRect(QRectF(4, 8, 12, 6))
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(color, 1.6, Qt.SolidLine, Qt.RoundCap))
        for i, h in enumerate((5, 8, 11, 14)):
            x = 30 + i * 5
            p.drawLine(QPointF(x, 18), QPointF(x, 18 - h))
        p.setPen(QPen(color, 1.5))
        p.drawArc(QRectF(54, 8, 16, 12), 20 * 16, 140 * 16)
        p.drawArc(QRectF(57, 11, 10, 8), 20 * 16, 140 * 16)
        p.setBrush(color)
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(62, 17), 1.6, 1.6)
        p.end()


class FeatureDeck(QWidget):
    """Two wide cards. Each covers half of the next. Scroll brings the next forward."""

    confirmed = Signal(str)
    scrolled = Signal(str)

    _W = 500
    _H = 196
    _CARD_W = 440
    _CARD_H = 88
    _HALF = 44

    def __init__(self, parent=None):
        super().__init__(parent)
        self._index = 0
        self._pos = 0.0
        self._closing = False
        self._geom_anim = None
        self._pos_anim = None
        self.setFixedSize(self._W, self._H)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.WheelFocus)
        self.hide()

    def sync(self, index):
        self._stop_pos()
        self._index = int(index) % 2
        self._pos = float(self._index)
        self.update()

    def target_rect(self):
        parent = self.parentWidget()
        w = parent.width() if parent is not None else WINDOW_W
        h = parent.height() if parent is not None else WINDOW_H
        return QRect((w - self._W) // 2, (h - self._H) // 2 + 24, self._W, self._H)

    def open_menu(self):
        self._closing = False
        self._stop_geom()
        end = self.target_rect().topLeft()
        parent_h = self.parentWidget().height() if self.parentWidget() else WINDOW_H
        self.move(end.x(), parent_h + 12)
        self.show()
        self.raise_()
        self.setFocus()
        anim = QPropertyAnimation(self, b"pos", self)
        anim.setDuration(320)
        anim.setStartValue(self.pos())
        anim.setEndValue(end)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start()
        self._geom_anim = anim

    def close_menu(self):
        if not self.isVisible():
            return
        self._closing = True
        self._stop_geom()
        parent_h = self.parentWidget().height() if self.parentWidget() else WINDOW_H
        anim = QPropertyAnimation(self, b"pos", self)
        anim.setDuration(240)
        anim.setStartValue(self.pos())
        anim.setEndValue(QPoint(self.x(), parent_h + 12))
        anim.setEasingCurve(QEasingCurve.InCubic)
        anim.finished.connect(self._finish_close)
        anim.start()
        self._geom_anim = anim

    def recenter(self):
        if self.isVisible() and not self._closing and not self._geom_busy():
            self.move(self.target_rect().topLeft())

    def apply_wheel(self, delta_y):
        if delta_y == 0 or self._pos_busy():
            return
        step = 1 if delta_y > 0 else -1
        self._animate_to((self._index + step) % 2, step)

    def wheelEvent(self, event):
        self.apply_wheel(event.angleDelta().y())
        event.accept()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        hit = self._hit(event.position())
        if hit is None:
            return
        rel, slot = hit
        if abs(rel) < 0.55:
            self.confirmed.emit(FEATURES[slot][0])
        else:
            step = 1 if slot != self._index else 0
            if step:
                self._animate_to(slot, 1 if (slot - self._index) % 2 == 1 else -1)
        super().mousePressEvent(event)

    def paintEvent(self, event):
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        cards = sorted(self._cards(), key=lambda item: item[0], reverse=True)
        for rel, slot, rect in cards:
            if rect.bottom() < -8 or rect.top() > self.height():
                continue
            self._paint_card(p, rect, slot, abs(rel) < 0.45)
        p.setPen(QColor("#9AA6B2"))
        p.setFont(QFont("Segoe UI", 10, QFont.DemiBold))
        p.drawText(QRectF(0, self._H - 28, self._W, 18), Qt.AlignHCenter | Qt.AlignBottom, "SCROLL FOR NEXT")
        p.setPen(QPen(QColor("#B7C2CC"), 1.8, Qt.SolidLine, Qt.RoundCap))
        cx = self._W / 2
        p.drawLine(QPointF(cx - 7, self._H - 34), QPointF(cx, self._H - 42))
        p.drawLine(QPointF(cx + 7, self._H - 34), QPointF(cx, self._H - 42))
        p.end()

    def _cards(self):
        rows = []
        for slot in (0, 1):
            rel = (slot - self._pos) % 2
            if rel > 1:
                rel -= 2
            rect = QRectF(30, 6 + rel * self._HALF, self._CARD_W, self._CARD_H)
            rows.append((rel, slot, rect))
        return rows

    def _hit(self, point):
        chosen = None
        for rel, slot, rect in self._cards():
            if rect.contains(point) and (chosen is None or abs(rel) < abs(chosen[0])):
                chosen = (rel, slot)
        return chosen

    def _paint_card(self, p, rect, slot, front):
        key, label = FEATURES[slot]
        p.save()
        p.setOpacity(1.0)
        border = CYAN if front else QColor(255, 255, 255, 80)
        fill = QColor(8, 16, 26, 118) if front else QColor(10, 14, 20, 96)
        p.setPen(QPen(border, 2.0 if front else 1.2))
        p.setBrush(fill)
        p.drawRoundedRect(rect, 18, 18)
        icon_color = GREEN if key == "adas" else QColor("#F2F5F8")
        icon = QRectF(rect.left() + 28, rect.center().y() - 16, 54, 32)
        self._paint_icon(p, icon, key, icon_color)
        p.setPen(ICE)
        p.setFont(QFont("Segoe UI", 18, QFont.DemiBold))
        p.drawText(QRectF(icon.right() + 8, rect.top(), rect.right() - icon.right() - 16, rect.height()), Qt.AlignVCenter | Qt.AlignLeft, label)
        p.restore()

    def _paint_icon(self, p, area, key, color):
        p.setPen(QPen(color, 2.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.setBrush(Qt.NoBrush)
        if key == "adas":
            cx = area.center().x()
            top = area.top()
            bot = area.bottom()
            p.drawLine(QPointF(cx, top), QPointF(cx, bot))
            p.drawLine(QPointF(cx - 7, top + 6), QPointF(cx - 16, bot))
            p.drawLine(QPointF(cx + 7, top + 6), QPointF(cx + 16, bot))
            return
        body = QRectF(area.center().x() - 16, area.center().y() - 11, 32, 22)
        p.drawRoundedRect(body, 4, 4)
        p.drawEllipse(body.center(), 6, 6)
        p.drawRoundedRect(QRectF(body.right() - 2, body.center().y() - 3, 7, 6), 1, 1)

    def _animate_to(self, target, step):
        self._stop_pos()
        start = float(self._index)
        end = start + float(step)
        anim = QVariantAnimation(self)
        anim.setStartValue(start)
        anim.setEndValue(end)
        anim.setDuration(260)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.valueChanged.connect(self._on_pos)
        anim.finished.connect(lambda t=target: self._finish_pos(t))
        anim.start()
        self._pos_anim = anim

    def _on_pos(self, value):
        self._pos = float(value)
        self.update()

    def _finish_pos(self, target):
        self._index = int(target) % 2
        self._pos = float(self._index)
        self.update()
        self.scrolled.emit(FEATURES[self._index][0])

    def _pos_busy(self):
        return self._pos_anim is not None and self._pos_anim.state() == QVariantAnimation.Running

    def _geom_busy(self):
        return self._geom_anim is not None and self._geom_anim.state() == QPropertyAnimation.Running

    def _stop_pos(self):
        anim = self._pos_anim
        self._pos_anim = None
        if anim is not None:
            anim.blockSignals(True)
            anim.stop()

    def _stop_geom(self):
        anim = self._geom_anim
        self._geom_anim = None
        if anim is not None:
            anim.blockSignals(True)
            anim.stop()

    def _finish_close(self):
        if self._closing:
            self.hide()
            self._closing = False


class LaneDrawToggle(QPushButton):
    """Checkable LANES pill. Off = corridor only; on = draw lane strokes."""

    def __init__(self, parent=None):
        super().__init__("LANES", parent)
        self.setCheckable(True)
        self.setChecked(False)
        self.setFixedSize(88, 32)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)
        self.setStyleSheet(
            "QPushButton {"
            "  background: rgba(16,20,28,200);"
            "  color: #C5CDD6;"
            "  border: 1.5px solid #3A4553;"
            "  border-radius: 16px;"
            "  font-size: 12px;"
            "  font-weight: 700;"
            "}"
            "QPushButton:checked {"
            "  background: rgba(40,32,8,220);"
            "  color: #F4D03F;"
            "  border: 1.5px solid #F4D03F;"
            "}"
        )


class HexCockpit(QWidget):
    view_changed = Signal(str)
    lane_draw_toggled = Signal(bool)

    def __init__(self, bev_widget, parent=None):
        super().__init__(parent)
        self.setObjectName("hex_cockpit")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("#hex_cockpit{background:#05070C;}")
        self.setMouseTracking(True)
        self._view = "bev"
        self._mode = "adas"
        self._full_view = False
        self._minute = ""
        self._fps = None

        self.stack = QStackedWidget(self)
        self.bev_widget = bev_widget
        self.stack.addWidget(bev_widget)
        self.lbl_camera = QLabel("Live view")
        self.lbl_camera.setAlignment(Qt.AlignCenter)
        self.lbl_camera.setStyleSheet("background:#000;color:#7A93A8;")
        self.lbl_camera.setScaledContents(True)
        self.stack.addWidget(self.lbl_camera)

        self.banner_ldw = WarningCard("LDW", self)
        self.banner_fcw = WarningCard("FCW", self)
        self.card_obj = WarningCard("OBJ", self)
        self.speed = SpeedReadout(self)
        self.lbl_limit = LimitSign(self)
        self.gear = GearButton(self)
        self.btn_settings = self.gear
        self.glyphs = StatusGlyphs(self)
        self.btn_lanes = LaneDrawToggle(self)
        self.btn_lanes.toggled.connect(self.lane_draw_toggled.emit)

        self.topbar = QFrame(self)
        self.topbar.setStyleSheet("background:transparent;")
        top = QHBoxLayout(self.topbar)
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(14)
        self.lbl_clock = QLabel("—")
        self.lbl_temp = QLabel("72°F")
        for label in (self.lbl_clock, self.lbl_temp):
            label.setStyleSheet("color:#E8EEF4;font-size:15px;font-weight:600;background:transparent;")
        top.addStretch()
        top.addWidget(self.lbl_clock)
        top.addWidget(self.lbl_temp)
        top.addWidget(self.glyphs)
        self.lbl_fps = QLabel("FPS —")
        self.lbl_fps.setStyleSheet(
            "color:#E8EEF4;font-size:15px;font-weight:700;background:transparent;"
        )
        top.addWidget(self.lbl_fps)

        self.settings_wheel = FeatureDeck(self)
        self.settings_wheel.confirmed.connect(self._on_menu_choice)
        self.settings_wheel.scrolled.connect(self.set_mode)
        self.gear.clicked.connect(self._toggle_settings)

        self.mode_panel = QFrame(self)
        self.mode_panel.hide()
        QVBoxLayout(self.mode_panel)
        self.btn_play = QPushButton("Pause", self)
        self.btn_open = QPushButton("Open", self)
        self.btn_full = self.btn_play
        self.btn_play.hide()
        self.btn_open.hide()

        for widget in (self.stack, self.bev_widget, self.lbl_camera):
            widget.installEventFilter(self)

        self.set_mode("adas")

    def sizeHint(self):
        return QSize(WINDOW_W, WINDOW_H)

    def eventFilter(self, watched, event):
        if self.settings_wheel.isVisible() and event.type() == QEvent.Wheel:
            self.settings_wheel.apply_wheel(event.angleDelta().y())
            return True
        return super().eventFilter(watched, event)

    def wheelEvent(self, event):
        if self.settings_wheel.isVisible():
            self.settings_wheel.apply_wheel(event.angleDelta().y())
            event.accept()
            return
        super().wheelEvent(event)

    def set_view(self, mode):
        self._view = "live" if mode == "live" else "bev"
        self.stack.setCurrentIndex(1 if self._view == "live" else 0)
        if hasattr(self.bev_widget, "setUpdatesEnabled"):
            self.bev_widget.setUpdatesEnabled(self._view == "bev")
        self.view_changed.emit(self._view)

    def set_mode(self, mode):
        self._mode = "live" if mode in ("live", "front") else "adas"
        if self._mode == "live":
            self.set_view("live")
        else:
            self.set_view("bev")
            if hasattr(self.bev_widget, "set_dashboard_camera"):
                self.bev_widget.set_dashboard_camera("chase")

    def _toggle_settings(self):
        if self.settings_wheel.isVisible():
            self.settings_wheel.close_menu()
            self.gear.set_active(False)
        else:
            self.settings_wheel.sync(1 if self._mode == "live" else 0)
            self.settings_wheel.open_menu()
            self.gear.set_active(True)
            self.gear.raise_()

    def _on_menu_choice(self, key):
        if key in ("live", "front"):
            self.set_mode("live")
        elif key in ("adas", "bev"):
            self.set_mode("adas")
        self.settings_wheel.close_menu()
        self.gear.set_active(False)

    def _raise_chrome(self):
        for widget in (
            self.banner_ldw, self.banner_fcw, self.card_obj,
            self.speed, self.lbl_limit, self.topbar,
            self.settings_wheel, self.gear, self.btn_lanes,
        ):
            widget.raise_()

    def resizeEvent(self, event):
        w, h = self.width(), self.height()
        self.stack.setGeometry(0, 0, w, h)
        self.topbar.setGeometry(w - 460, 14, 436, 28)
        gear_x = w - self.gear.width() - 64
        gear_y = 88
        self.gear.move(gear_x, gear_y)
        gear_cx = gear_x + self.gear.width() / 2
        self.speed.move(int(gear_cx - self.speed.width() / 2), gear_y + self.gear.height() + 10)
        self.lbl_limit.move(
            int(gear_cx - self.lbl_limit.width() / 2),
            self.speed.geometry().bottom() + 8,
        )
        self.banner_ldw.move(28, 72)
        self.banner_fcw.move(28, 192)
        self.card_obj.move(28, 312)
        self.btn_lanes.move(28, h - self.btn_lanes.height() - 24)
        if self.settings_wheel.isVisible():
            self.settings_wheel.recenter()
        self._raise_chrome()
        super().resizeEvent(event)

    def update_hud(self, *, speed_mps=None, cipo_obj=None, cipo_status="SAFE",
                   alerts=None, fps=None, lane_ok=False, objects=None):
        del cipo_obj, cipo_status, lane_ok, objects
        alerts = alerts or {}
        self.speed.set_mps(speed_mps)

        isa = alerts.get("isa") or {}
        posted = isa.get("posted_mph")
        cand = isa.get("candidate_mph")
        self.lbl_limit.set_limit(posted if posted is not None else cand)

        fcw = str(alerts.get("fcw") or "OFF")
        fcw_on = fcw in ("FCW", "FCW+")
        self.banner_fcw.set_trigger(fcw_on)

        ldw = str(alerts.get("ldw") or "OFF")
        side = str(alerts.get("ldw_side") or "")
        ldw_on = ldw in ("LEFT", "RIGHT") or side in ("LEFT", "RIGHT")
        self.banner_ldw.set_trigger(ldw_on)
        self.card_obj.set_trigger(bool(alerts.get("obj")))

        now = datetime.now()
        minute = now.strftime("%I:%M %p").lstrip("0")
        if minute != self._minute:
            self._minute = minute
            _set_text(self.lbl_clock, minute)
        if fps is not None:
            fps_i = int(round(fps))
            if fps_i != self._fps:
                self._fps = fps_i
                _set_text(self.lbl_fps, f"FPS {fps_i}")
