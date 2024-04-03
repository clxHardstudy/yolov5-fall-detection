from PyQt5.QtCore import Qt
from InterfaceUi import *
from PyQt5.QtWidgets import QApplication,QMainWindow

import sys

class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_InterfaceWindow()
        self.ui.setupUi(self)
        # 隐藏多余的窗体
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # 添加阴影
        # self.shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        # self.shadow.setOffset(0,0)
        # self.shadow.setBlurRadius(15)
        # self.shadow.setColor(Qt.black)
        # self.ui.frame.setGraphicsEffect(self.shadow)
        self.show()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = LoginWindow()
    sys.exit(app.exec_())