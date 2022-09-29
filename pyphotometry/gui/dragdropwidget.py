from PyQt5.QtWidgets import QWidget


class DragDropWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        e.accept()

    def dropEvent(self, e):
        pos = e.pos()
        widget = e.source()

        for n in range(self.layout().count()):
            w = self.layout().itemAt(n).widget()
            drop_here = pos.y() < w.y() + w.size().height() // 2
            if drop_here:
                self.layout().insertWidget(n, widget)
                break
                # self.orderChanged.emit(self.get_item_data())
        e.accept()

    # def get_item_data(self):
    #     data = []
    #     for n in range(self.layout().count()):
    #         w = self.layout().itemAt(n).widget()
    #         data.append(w.data)
