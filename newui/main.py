from PyQt5.QtCore import Qt

from LoginUi import *

from PyQt5.QtWidgets import QApplication,QMainWindow

import sys

class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_LoginWindow()
        self.ui.setupUi(self)
        # 隐藏多余的窗体
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 添加阴影
        self.shadow = QtWidgets.QGraphicsDropShadowEffect(self)
        self.shadow.setOffset(0,0)
        self.shadow.setBlurRadius(15)
        self.shadow.setColor(Qt.black)
        self.ui.frame.setGraphicsEffect(self.shadow)
        # 登录点击展示窗口
        self.ui.pushButton_Login.clicked.connect(lambda:self.ui.stackedWidget_2.setCurrentIndex(1))
        # 注册点击展示窗口
        self.ui.pushButton_Register.clicked.connect(lambda:self.ui.stackedWidget_2.setCurrentIndex(0))
        self.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = LoginWindow()
    sys.exit(app.exec_())