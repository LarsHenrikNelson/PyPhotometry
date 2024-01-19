from collections import namedtuple

import numpy as np
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QTabWidget,
)
from PyQt5.QtGui import QIntValidator, QDrag, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QSize, QMimeData
import pyqtgraph as pg

from .qtwidgets import LineEdit


Settings = namedtuple(
    "Analysis_settings",
    [
        "exp",
        "acq",
        "analysis_method",
        "sec_array",
        "time_array",
        "smooth",
        "find_peaks",
        "find_tau",
        "peak_threshold",
        "peak_direction",
        "z_score",
        "z_threshold",
        "filter",
        "filter_type",
        "input_1",
        "input_2",
    ],
)


class AnalysisWidget(QFrame):
    analysis_sig = pyqtSignal(object)

    def __init__(self):
        super().__init__()

        self.setObjectName("frame")

        self.setFrameShape(QFrame.Box)
        # self.setLineWidth(3)
        # self.setMidLineWidth(2)
        # self.setFrameStyle(QFrame.Box | QFrame.Plain)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self)

        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.selection_layout = QFormLayout()
        self.layout.addLayout(self.selection_layout)
        self.sheet_label = QLabel()
        self.selection_layout.addRow(self.sheet_label)

        self.acq_label = QLabel()
        self.selection_layout.addRow(self.acq_label)

        self.method_sel = QComboBox()
        methods = [
            "quantile",
            "cubic_spline",
            "nat_spline",
            "min_peak",
            "plexxon",
            "isobestic",
        ]
        self.method_sel.addItems(methods)
        self.method_label = QLabel("Analysis method")
        self.method_sel.currentTextChanged.connect(self.activateSubs)
        self.selection_layout.addRow(self.method_label, self.method_sel)

        self.isobestic_box = QComboBox()
        self.isobestic_label = QLabel("Isobestic")
        self.selection_layout.addRow(self.isobestic_label, self.isobestic_box)
        self.isobestic_box.setEnabled(False)

        self.time_box = QComboBox()
        self.time_label = QLabel("Time array")
        self.selection_layout.addRow(self.time_label, self.time_box)

        self.smooth_box = LineEdit()
        self.smooth_box.setEnabled(True)
        self.smooth_label = QLabel("Smooth value")
        self.smooth_box.setValidator(QIntValidator())
        self.selection_layout.addRow(self.smooth_label, self.smooth_box)

        self.additional_analysis = QFormLayout()
        self.layout.addLayout(self.additional_analysis)

        self.find_peaks_label = QLabel("Find peaks")
        self.find_peaks = QCheckBox()
        self.find_peaks.setTristate(False)
        self.find_peaks.setChecked(True)
        self.additional_analysis.addRow(self.find_peaks_label, self.find_peaks)

        self.tau_label = QLabel("Find tau")
        self.tau_box = QCheckBox()
        self.tau_box.setChecked(True)
        self.additional_analysis.addRow(self.tau_label, self.tau_box)

        self.peak_dir_label = QLabel("Peak direction")
        self.peak_dir = QComboBox()
        self.peak_dir.addItems(["positive", "negative"])
        self.peak_dir.setCurrentText("positive")
        self.additional_analysis.addRow(self.peak_dir_label, self.peak_dir)

        self.threshold_label = QLabel("Peak threshold")
        self.threshold = LineEdit()
        self.threshold.setText(str(80))
        self.additional_analysis.addRow(self.threshold_label, self.threshold)

        self.z_score_label = QLabel("Z score acq")
        self.z_score = QCheckBox()
        self.z_score.setTristate(False)
        self.z_score.setChecked(True)
        self.additional_analysis.addRow(self.z_score_label, self.z_score)

        self.z_threshold_label = QLabel("Z threshold")
        self.z_threshold = LineEdit()
        self.z_threshold.setText(str(80))
        self.z_threshold.setValidator(QIntValidator())
        self.additional_analysis.addRow(self.z_threshold_label, self.z_threshold)

        self.filter_layout = QFormLayout()
        self.layout.addLayout(self.filter_layout)

        self.filter_label = QLabel("Filter array")
        self.filter_box = QComboBox()
        self.filter_box.addItems(["None", "df_f", "z_array"])
        self.filter_layout.addRow(self.filter_label, self.filter_box)

        self.filter_choice_label = QLabel("Filter array")
        self.filter_type = QComboBox()
        self.filter_type.addItems(["ewma", "savgol"])
        self.filter_type.currentTextChanged.connect(self.filterSelection)
        self.filter_layout.addRow(self.filter_choice_label, self.filter_type)

        self.input1_label = QLabel("Window size")
        self.input1 = LineEdit()
        self.input1.setText(str(10))
        self.input1.setValidator(QIntValidator())
        self.filter_layout.addRow(self.input1_label, self.input1)

        self.input2_label = QLabel("Sample proportion")
        self.input2 = LineEdit()
        self.input2.setText("0.85")
        self.filter_layout.addRow(self.input2_label, self.input2)

        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.analyze)
        self.filter_layout.addRow(self.analyze_btn)

        self.plot_controls = QFormLayout()
        self.zoom_label = QLabel("Zoom")
        self.plot_controls.addRow(self.zoom_label)
        self.zoom_slider = QSlider()
        self.zoom_slider.setOrientation(Qt.Horizontal)
        self.zoom_slider.setMinimum(10)
        self.zoom_slider.setMaximum(100)
        self.zoom_slider.setValue(100)
        self.plot_controls.addRow(self.zoom_slider)
        self.scroll_label = QLabel("Scroll")
        self.plot_controls.addRow(self.scroll_label)
        self.scroll_slider = QSlider()
        self.plot_controls.addRow(self.scroll_slider)
        self.scroll_slider.setOrientation(Qt.Horizontal)
        self.scroll_slider.valueChanged.connect(self.scrollPosition)

        self.plot_tabs = QTabWidget()
        self.layout.addWidget(self.plot_tabs, 1)
        self.plot_1 = pg.PlotWidget()
        self.plot_1.setMinimumSize(400, 200)
        self.plot_tabs.addTab(self.plot_1, "Raw + Baseline")
        self.plot_2 = pg.PlotWidget()
        self.plot_2.setMinimumSize(400, 200)
        self.plot_tabs.addTab(self.plot_2, "Baselined")
        self.plot_3 = pg.PlotWidget()
        self.plot_tabs.addTab(self.plot_3, "Z array")
        self.plot_3.setMinimumSize(400, 200)

    def addData(self, tab, name, acq_list):
        acq_list.insert(0, "None")
        self.sheet_label.setText(tab)
        self.acq_label.setText(name)
        self.isobestic_box.addItems(acq_list)
        self.time_box.addItems(acq_list)

    def activateSubs(self, text):
        if text in ["cubic_spline", "nat_spline", "min_peak", "min_spline"]:
            self.smooth_box.setEnabled(True)
            self.isobestic_box.setEnabled(False)
        elif text == "quantile":
            self.smooth_box.setEnabled(True)
            self.isobestic_box.setEnabled(False)
        else:
            self.smooth_box.setEnabled(False)
            self.isobestic_box.setEnabled(True)
            self.isobestic_label.setText("Isobestic")

    def filterSelection(self, text):
        if text == "savgol":
            self.input2_label.setText("Polyorder")
            self.input2.setText("3")
        else:
            self.input2_label.setText("Sum proportion (0-1)")
            self.input2.setText("0.85")

    def analyze(self):
        temp = self.smooth_box.toText().replace(" ", "").split(",")
        if len(temp) == 1:
            smooth = int(temp[0])
        elif len(temp) == 2:
            smooth = (int(i) for i in temp)
        elif len(temp) == 3:
            smooth = (int(i) for i in temp)
        self.settings = Settings(
            self.sheet_label.text(),
            self.acq_label.text(),
            self.method_sel.currentText(),
            self.isobestic_box.currentText(),
            self.time_box.currentText(),
            smooth,
            self.find_peaks.checkState(),
            self.tau_box.checkState(),
            self.threshold.toFloat(),
            self.peak_dir.currentText(),
            self.z_score.checkState(),
            self.z_threshold.toInt(),
            self.filter_box.currentText(),
            self.filter_type.currentText(),
            self.input1.toInt(),
            self.input2.toFloat(),
        )
        self.analysis_sig.emit(self.settings)

    def plotData(self, atuple):
        self.min_x = atuple.timestamp[0]
        self.max_x = atuple.timestamp[-1]
        self.zoom_slider.valueChanged.connect(self.zoomSlider)
        self.plot_1.clear()
        self.plot_2.clear()
        self.plot_3.clear()

        self.plot_1.plot(x=atuple.timestamp, y=atuple.raw_array, antialias=True)
        self.plot_1.plot(
            x=atuple.timestamp,
            y=atuple.baseline,
            pen=pg.mkPen("m", width=3),
            antialias=True,
        )

        self.plot_2.plot(x=atuple.timestamp, y=atuple.df_f, antialias=True)

        if np.mean(atuple.z_array.size) != 0:
            self.plot_3.plot(x=atuple.timestamp, y=atuple.z_array)

        if atuple.peaks_x.size > 0 and np.mean(atuple.z_array) != 0:
            self.plot_3.plot(
                x=atuple.timestamp[atuple.peaks_x],
                y=atuple.z_array[atuple.peaks_x],
                pen=None,
                symbolBrush="g",
            )
        elif atuple.peaks_x.size > 0 and np.mean(atuple.z_array) == 0:
            self.plot_2.plot(
                x=atuple.timestamp[atuple.peaks_x],
                y=atuple.df_f[atuple.peaks_x],
                pen=None,
                symbolBrush="g",
            )

    def sizeHint(self):
        return QSize(600, 300)

    def zoomSlider(self, value):
        axX = self.p1.getAxis("bottom")
        percent = value / 100
        data_range = (self.max_x - self.min_x) * percent
        plt_range = axX.range[0] - axX.range[1]
        plt_x_max = axX.range[0] + data_range

    def scrollPosition(self, value):
        if self.zoomSlider.value() < 100:
            pass
        else:
            pass

    def update(self):
        """
        This functions is used for the draggable region.
        See PyQtGraphs documentation for more information.
        """
        axX = self.p1.getAxis("bottom")
        self.p1.setXRange(axX.range[0], axX.range[1], padding=0)

    def mouseMoveEvent(self, e):
        if e.buttons() == Qt.LeftButton:
            drag = QDrag(self)
            mime = QMimeData()
            drag.setMimeData(mime)

            pixmap = QPixmap(self.size())
            self.render(pixmap)
            drag.setPixmap(pixmap)

            drag.exec_(Qt.MoveAction)
