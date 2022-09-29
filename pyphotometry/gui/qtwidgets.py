from PyQt5.QtWidgets import (
    QLineEdit,
    QSpinBox,
)
from PyQt5.QtCore import (
    QObject,
    pyqtSignal,
    Qt,
    QAbstractListModel,
)
from PyQt5.QtGui import QColor


class LineEdit(QLineEdit):
    """
    This is a subclass of QLineEdit that returns values that are usable for
    Python.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

    def toInt(self):
        if len(self.text()) == 0:
            x = None
        else:
            x = int(float(self.text()))
        return x

    def toText(self):
        if len(self.text()) == 0:
            x = None
        else:
            x = self.text()
        return x

    def toFloat(self):
        if len(self.text()) == 0:
            x = None
        else:
            x = float(self.text())
        return x


class WorkerSignals(QObject):
    """
    This is general 'worker' that provides feedback from events to the window.
    The 'worker' also provides a 'runner' to the main GUI thread to prevent it
    from freezing when there are long running events.
    """

    dictionary = pyqtSignal(dict)
    progress = pyqtSignal(int)
    finished = pyqtSignal(str)
    path = pyqtSignal(object)


class ListModel(QAbstractListModel):
    """
    The model contains all the load data for the list view. Qt works using a
    model-view-controller framework so the view should not contain any data
    analysis or loading, it just facilities the transfer of the data to the model.
    The model is used to add, remove, and modify the data through the use of a
    controller which in Qt is often built into the models.
    """

    def __init__(self, names=None):
        super().__init__()
        self.names = names or []

    def data(self, index, role):
        if role == Qt.ItemDataRole.DisplayRole:
            status, text = self.names[index.row()]
            return text

        if role == Qt.ItemDataRole.DecorationRole:
            status, text = self.names[index.row()]
            if status:
                return QColor("#ff0000")

    def rowCount(self, index):
        return len(self.names)

    def addItems(self, items):
        self.names.extend([(False, i) for i in items])
        self.layoutChanged.emit()


class StringBox(QSpinBox):
    def __init__(self, parent=None):
        super(StringBox, self).__init__(parent)
        strings = []
        self.setStrings(strings)

    def setStrings(self, strings):
        strings = list(strings)
        self._strings = tuple(strings)
        self._values = dict(zip(strings, range(len(strings))))
        self.setRange(0, len(strings) - 1)

    def textFromValue(self, value):

        # returning string from index
        # _string = tuple
        return self._strings[value]
