"""Pitch + height for *object ranging* on the current camera.

Does not retune OpenLane P (lanes). Reset returns to the clip's measured /
generic GroundCalibration, not −3° / 1.5 m.
"""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QSlider, QPushButton,
)


class CalibrationPanel(QWidget):
    calibration_changed = Signal(float, float)

    def __init__(self, parent=None, pitch_deg=4.0, height_m=1.35):
        super().__init__(parent)
        self.pitch_deg = float(pitch_deg)
        self.height_m = float(height_m)
        self._default_pitch = self.pitch_deg
        self._default_height = self.height_m
        self.setMaximumHeight(36)
        self.init_ui()

    def init_ui(self):
        row = QHBoxLayout(self)
        row.setContentsMargins(4, 0, 4, 0)
        row.setSpacing(8)

        self.setStyleSheet("""
            QLabel { color: #8B949E; font-size: 10px; }
            QLabel#val { color: #58A6FF; font-weight: bold; font-size: 10px; min-width: 42px; }
            QSlider::groove:horizontal {
                height: 3px; background: #2D3440; border-radius: 1px;
            }
            QSlider::handle:horizontal {
                background: #58A6FF; width: 10px; height: 10px;
                margin: -4px 0; border-radius: 5px;
            }
            QPushButton {
                background-color: #21262D; color: #C9D1D9;
                border: 1px solid #30363D; border-radius: 3px;
                padding: 2px 8px; font-size: 10px;
            }
            QPushButton:hover { border-color: #58A6FF; }
        """)

        row.addWidget(QLabel("Pitch"))
        self.slider_pitch = QSlider(Qt.Horizontal)
        self.slider_pitch.setRange(-150, 150)
        self.slider_pitch.setValue(int(round(self.pitch_deg * 10)))
        self.slider_pitch.setMaximumWidth(140)
        self.slider_pitch.setToolTip("Object horizon (this camera). Does not move the lane net.")
        self.lbl_pitch = QLabel(f"{self.pitch_deg:.1f}°")
        self.lbl_pitch.setObjectName("val")
        self.slider_pitch.valueChanged.connect(self.on_slider_changed)
        row.addWidget(self.slider_pitch)
        row.addWidget(self.lbl_pitch)

        row.addWidget(QLabel("H"))
        self.slider_height = QSlider(Qt.Horizontal)
        self.slider_height.setRange(8, 30)
        self.slider_height.setValue(int(round(self.height_m * 10)))
        self.slider_height.setMaximumWidth(100)
        self.slider_height.setToolTip("Camera height over the road (object range / lateral).")
        self.lbl_height = QLabel(f"{self.height_m:.2f}m")
        self.lbl_height.setObjectName("val")
        self.slider_height.valueChanged.connect(self.on_slider_changed)
        row.addWidget(self.slider_height)
        row.addWidget(self.lbl_height)

        btn_reset = QPushButton("Reset")
        btn_reset.setToolTip("Reset to this clip's measured / generic calib.")
        btn_reset.clicked.connect(self.reset_defaults)
        row.addWidget(btn_reset)
        row.addStretch()

    def set_from_calib(self, calib):
        """Load slider defaults from a GroundCalibration."""
        import math
        pitch = math.degrees(float(calib.pitch_rad))
        height = float(calib.cam_height_m)
        self._default_pitch = pitch
        self._default_height = height
        self.blockSignals(True)
        self.slider_pitch.blockSignals(True)
        self.slider_height.blockSignals(True)
        self.slider_pitch.setValue(int(round(pitch * 10)))
        self.slider_height.setValue(int(round(min(3.0, max(0.8, height)) * 10)))
        self.pitch_deg = self.slider_pitch.value() / 10.0
        self.height_m = self.slider_height.value() / 10.0
        self.lbl_pitch.setText(f"{self.pitch_deg:.1f}°")
        self.lbl_height.setText(f"{self.height_m:.2f}m")
        self.slider_pitch.blockSignals(False)
        self.slider_height.blockSignals(False)
        self.blockSignals(False)

    def on_slider_changed(self, _=None):
        self.pitch_deg = self.slider_pitch.value() / 10.0
        self.height_m = self.slider_height.value() / 10.0
        self.lbl_pitch.setText(f"{self.pitch_deg:.1f}°")
        self.lbl_height.setText(f"{self.height_m:.2f}m")
        self.calibration_changed.emit(self.pitch_deg, self.height_m)

    def reset_defaults(self):
        self.slider_pitch.setValue(int(round(self._default_pitch * 10)))
        self.slider_height.setValue(int(round(self._default_height * 10)))
