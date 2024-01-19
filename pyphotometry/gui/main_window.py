#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 09:40:14 2022

@author: Lars
"""
from PyQt5.QtWidgets import (
    QMainWindow,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

from .analysis_widget import AnalysisWidget
from .dragdropwidget import DragDropWidget
from .tabbed_table import PlotAcquireWidget
from ..acquisition.photometry import ExpManager


class MainWindow(QMainWindow):
    def __init__(self):
        self.exp_manager = ExpManager()
        self.plot_items = {}
        self.legend_items = {}
        self.widget_dict = {}

        pg.setConfigOptions(antialias=True)

        super().__init__()

        self.setStyleSheet(
            """QTabWidget::tab-bar 
                                          {alignment: left;}"""
        )

        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        self.tab1_scroll = QScrollArea()
        self.tab1_scroll.setWidgetResizable(True)
        self.tab1 = QWidget()
        self.tab1_scroll.setWidget(self.tab1)
        self.layout = QVBoxLayout()
        self.tab1.setLayout(self.layout)
        self.tab_widget.addTab(self.tab1_scroll, "Raw data")

        self.table = PlotAcquireWidget()
        self.table.dataFrameAdded.connect(self.updateLists)
        self.table.setManager(self.exp_manager)
        self.table.setUsesScrollButtons(True)
        self.table.setTabsClosable(True)
        self.table.setMinimumHeight(300)
        self.table.setMinimumWidth(600)
        self.layout.addWidget(self.table)

        self.raw_plot = pg.PlotWidget()
        self.raw_plot.setMinimumSize(300, 300)
        self.layout.addWidget(self.raw_plot)
        # self.raw_plot = self.plot_layout.addPlot(row=0, col=0)
        # self.v1 = self.plot_layout.addViewBox(row=0, col=1)
        # self.v1.setMaximumWidth(150)
        # self.v1.setMinimumWidth(150)
        # self.legend = pg.LegendItem()
        # self.v1.addItem(self.legend)
        # self.legend.setParentItem(self.v1)
        # self.legend.anchor((0, 0), (0, 0))
        # self.vlayout.addWidget(self.plot_layout)

        self.tab2 = DragDropWidget()
        self.tab2_scroll = QScrollArea()
        self.tab2_scroll.setWidgetResizable(True)
        self.tab2_scroll.setWidget(self.tab2)
        self.tab2_layout = QVBoxLayout()
        self.tab2.setLayout(self.tab2_layout)
        self.tab_widget.addTab(self.tab2_scroll, "Analysis settings")

        self.tab3 = QWidget()
        self.tab3_scroll = QScrollArea()
        self.tab3_scroll.setWidgetResizable(True)

        # self.delegate = ItemDelegate()

        # Variables to keep track of
        self.counter = 0

    def updateLists(self, view):
        view.clicked.connect(self.updatePlotAnalysisList)

    def updatePlotAnalysisList(self, index):
        if index.column() == 1:
            self.plotRawData(index)
        elif index.column() == 2:
            self.itemToAnalyze(index)

    def plotRawData(self, index):
        list_widget = self.table.currentWidget()
        tab_index = self.table.currentIndex()
        tab_name = self.table.tabText(tab_index)
        status = list_widget.model().df.iloc[index.row(), index.column()]
        name = list_widget.model().df.iloc[index.row(), 0]
        if not status:
            list_widget.model().df.iloc[index.row(), index.column()] = True
            list_widget.model().layoutChanged.emit()
            data = self.exp_manager.get_raw_data(tab_name, name)
            pencil = pg.mkPen(color=pg.intColor(self.counter))
            self.counter += 1
            plot_item = pg.PlotDataItem(data, pen=pencil)
            self.plot_items[name] = plot_item
            self.raw_plot.addItem(plot_item, name=(f"{tab_name}_{name}"))
            # self.legend.addItem(plot_item, name=(f"{tab_name}_{name}"))
        else:
            list_widget.model().df.iloc[index.row(), index.column()] = False
            list_widget.model().layoutChanged.emit()
            self.raw_plot.removeItem(self.plot_items.get(name))
            # self.legend.removeItem(self.plot_items.get(name))

    def itemToAnalyze(self, index):
        list_widget = self.table.currentWidget()
        tab_index = self.table.currentIndex()
        tab_name = self.table.tabText(tab_index)
        status = list_widget.model().df.iloc[index.row(), index.column()]
        name = list_widget.model().df.iloc[index.row(), 0]
        if not status:
            list_widget.model().df.iloc[index.row(), index.column()] = True
            list_widget.model().layoutChanged.emit()
            widget = AnalysisWidget()
            widget.analysis_sig.connect(self.analyze_acq)
            acq_list = self.exp_manager.get_acq_list(tab_name)
            widget.addData(tab_name, name, acq_list)
            self.tab2_layout.addWidget(widget)
            self.widget_dict[f"{tab_name}_{name}"] = widget
        else:
            list_widget.model().df.iloc[index.row(), index.column()] = False
            list_widget.model().layoutChanged.emit()
            widget = self.widget_dict[f"{tab_name}_{name}"]
            self.tab2_layout.removeWidget(widget)
            widget.hide()
            widget.deleteLater()

    def analyze_acq(self, ntuple):
        # The * before ntuple is to unpack the tuple into
        # the arguments that the exp_manager analyze_ acq method
        # expects.
        self.exp_manager.analyze_acq(*ntuple)
        atuple = self.exp_manager.get_analyzed_data(ntuple.exp, ntuple.acq)
        self.widget_dict[f"{ntuple.exp}_{ntuple.acq}"].plotData(atuple)

    def save_as(self):
        pass
