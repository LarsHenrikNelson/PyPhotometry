#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 13:51:30 2022

@author: Lars
"""
import pandas as pd
from PyQt5.QtWidgets import QTableView, QStyledItemDelegate
from PyQt5.QtCore import Qt, QAbstractTableModel, QPoint
from PyQt5.QtGui import QColor


# class TableModel(QtCore.QAbstractTableModel):
#     def __init__(self, df):
#         super().__init__()
#         self.df = df

#     def data(self, index, role):
#         if role == Qt.ItemDataRole.DisplayRole:
#             value = self.df.iloc[index.row(), index.column()]
#             return str(value)

#     def rowCount(self, index):
#         return self.df.shape[0]

#     def columnCount(self, index):
#         return self.df.shape[1]

#     def headerData(self, section, orientation, role):
#         if role == Qt.ItemDataRole.DisplayRole:
#             if orientation == Qt.Orientation.Horizontal:
#                 return str(self.df.columns[section])

#             if orientation == Qt.Orientation.Vertical:
#                 return str(self.df.index[section])


class TableModel2(QAbstractTableModel):
    def __init__(self, items):
        super().__init__()
        self.df = self.addItems(items)

    def data(self, index, role):
        if index.isValid():
            value = self.df.iloc[index.row(), index.column()]
        # if role == Qt.DisplayRole:
        #     if index.column() == 0:
        #         x = str(value)
        #     else:
        #         x = ""
        #     return x

        if role == Qt.DisplayRole:
            return str(value)

        # if role == Qt.BackgroundRole and value and index.column() > 0:
        #     return QColor("#ff0000")

    def rowCount(self, index):
        return self.df.shape[0]

    def columnCount(self, index):
        return self.df.shape[1]

    def headerData(self, section, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self.df.columns[section])

            if orientation == Qt.Orientation.Vertical:
                return str(self.df.index[section])

    def addItems(self, items):
        column1 = pd.Series([False for i in items])
        column2 = pd.Series([False for i in items])
        items = pd.Series(items)
        df = pd.DataFrame({"Data": items, "Plot": column1, "Analyze": column2})
        return df

    def flags(self, index):
        return Qt.NoItemFlags


class ItemDelegate(QStyledItemDelegate):
    def __init__(self):
        super().__init__()

    def paint(self, painter, option, index):
        value = index.model().data(index, Qt.ItemDataRole.DisplayRole)
        if value == "True":
            width = option.rect.width()
            height = option.rect.height()
            # x = option.rect.x()
            # y = option.rect.y()
            center = QPoint(option.rect.center().x(), option.rect.center().y() + 1)
            # y_top = center.y() - height * 0.3
            # x_left = center.x() - (height * 0.6) / 2
            painter.setPen(QColor("#ff0000"))
            painter.setBrush(QColor("#ff0000"))
            painter.drawEllipse(center, (height * 0.6) / 2, (height * 0.6) / 2)
        else:
            pass

    # def sizeHint(self, option, index):
    #     return QSize(100, 200)


class TableView(QTableView):
    def __init__(self):
        super().__init__()

    def setWidth(self):
        self.setColumnWidth(1, 50)
        self.setColumnWidth(2, 50)

    def addDelegates(self, delegate):
        self.setItemDelegateForColumn(1, delegate)
        self.setItemDelegateForColumn(2, delegate)
