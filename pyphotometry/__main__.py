import os
import sys

from PyQt5.QtWidgets import QApplication
import qdarkstyle
from qdarkstyle.dark.palette import DarkPalette

from .gui.main_window import MainWindow


def main():
    os.environ["QT_API"] = "pyqt5"
    app = QApplication([])
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyqt5", palette=DarkPalette))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
