from PyQt5.QtCore import Qt
from InterfaceUi import *
from PyQt5.QtWidgets import QApplication,QMainWindow
import sys

class InterfaceWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_InterfaceWindow()
        self.ui.setupUi(self)
        # 隐藏多余的窗体
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.show()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = InterfaceWindow()
    sys.exit(app.exec_())