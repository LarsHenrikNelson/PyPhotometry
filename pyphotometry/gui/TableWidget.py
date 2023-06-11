#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 11:22:38 2022

@author: Lars
"""
from pathlib import PurePath

import pandas as pd
from PyQt5.QtWidgets import (
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QLabel,
    QFormLayout,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QMessageBox,
    QTabWidget,
    QTableView,
    QScrollArea,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal

from .custom_tables import TableModel


class TableWidget(QTabWidget):
    dataFrameAdded = pyqtSignal(object, object)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)

    def addData(self):
        if self.exp_manager.new_data:
            for i in self.exp_manager.new_data:
                model = TableModel(self.exp_manager.exp_dict[i].raw_df)
                view = QTableView()
                view.setModel(model)
                self.addTab(view, i)
                self.dataFrameAdded.emit(
                    i, list(self.exp_manager.exp_dict[i].raw_df.columns)
                )

    def currentModel(self):
        index = self.currentIndex()
        return self.models[index]

    def dragEnterEvent(self, e):
        """
                This function will detect the drag enter event from the mouse on the
        main window
        """
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dragMoveEvent(self, e):
        """
        This function will detect the drag move event on the main window
        """
        if e.mimeData().hasUrls:
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        """
        This function will enable the drop file directly on to the
        main window. The file location will be stored in the self.filename
        """
        if e.mimeData().hasUrls:
            e.setDropAction(Qt.CopyAction)
            e.accept()
            self.exp_manager.load_data(
                [str(i.toLocalFile()) for i in e.mimeData().urls()]
            )
            self.addData()
        else:
            e.ignore()  # just like above functions

    def setManager(self, exp_manager):
        self.exp_manager = exp_manager
