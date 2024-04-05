import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow

from newui import res_rc


class Ui_HomeWindow(QMainWindow):
    def setupUi(self, HomeWindow):
        HomeWindow.setObjectName("HomeWindow")
        HomeWindow.resize(1041, 700)
        self.centralwidget = QtWidgets.QWidget(HomeWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(19, 19, 1001, 661))
        self.frame.setMinimumSize(QtCore.QSize(1001, 661))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                   "border-top-left-radius:20px;\n"
                                   "border-top-right-radius:20px;\n"
                                   "")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame_4 = QtWidgets.QFrame(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_4.sizePolicy().hasHeightForWidth())
        self.frame_4.setSizePolicy(sizePolicy)
        self.frame_4.setStyleSheet("border:none;\n"
                                   "font: 12pt \"微软雅黑\";")
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_4)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton = QtWidgets.QPushButton(self.frame_4)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/跌倒.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton.setIcon(icon)
        self.pushButton.setIconSize(QtCore.QSize(35, 35))
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton, 0, QtCore.Qt.AlignLeft)
        self.horizontalLayout.addWidget(self.frame_4)
        self.frame_5 = QtWidgets.QFrame(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_5.sizePolicy().hasHeightForWidth())
        self.frame_5.setSizePolicy(sizePolicy)
        self.frame_5.setStyleSheet("QPushButton{\n"
                                   "    border:none;\n"
                                   "}\n"
                                   "QPushButton:hover{\n"
                                   "    padding-bottom:5px;\n"
                                   "}")
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_5)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.pushButton_2 = QtWidgets.QPushButton(self.frame_5)
        self.pushButton_2.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/退出.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_2.setIcon(icon1)
        self.pushButton_2.setIconSize(QtCore.QSize(30, 30))
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout_3.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.frame_5)
        self.pushButton_3.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/缩小.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_3.setIcon(icon2)
        self.pushButton_3.setIconSize(QtCore.QSize(30, 30))
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_3.addWidget(self.pushButton_3)
        self.pushButton_4 = QtWidgets.QPushButton(self.frame_5)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/关闭.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_4.setIcon(icon3)
        self.pushButton_4.setIconSize(QtCore.QSize(30, 30))
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout_3.addWidget(self.pushButton_4)
        self.horizontalLayout.addWidget(self.frame_5, 0, QtCore.Qt.AlignRight)
        self.verticalLayout.addWidget(self.frame_2)
        self.frame_3 = QtWidgets.QFrame(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(10)
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.frame_6 = QtWidgets.QFrame(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(3)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_6.sizePolicy().hasHeightForWidth())
        self.frame_6.setSizePolicy(sizePolicy)
        self.frame_6.setMinimumSize(QtCore.QSize(230, 597))
        self.frame_6.setStyleSheet("#frame_6{\n"
                                   "    background-color: qlineargradient(spread:pad, x1:0.179, y1:0.982955, x2:1, y2:0, stop:0 rgba(153, 34, 57, 254), stop:1 rgba(255, 255, 255, 255));\n"
                                   "    border-bottom-left-radius:20px;\n"
                                   "}\n"
                                   "QPushButton{\n"
                                   "    border:none;\n"
                                   "    font-size:20px;        \n"
                                   "}\n"
                                   "QPushButton:pressed{\n"
                                   "    padding-top:4px;\n"
                                   "    padding-left:5px;\n"
                                   "\n"
                                   "}\n"
                                   "QPushButton:hover{\n"
                                   "    padding-bottom:5px;\n"
                                   "}")
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton_Fall_Detection = QtWidgets.QPushButton(self.frame_6)
        self.pushButton_Fall_Detection.setStyleSheet("")
        self.pushButton_Fall_Detection.setIcon(icon)
        self.pushButton_Fall_Detection.setIconSize(QtCore.QSize(30, 30))
        self.pushButton_Fall_Detection.setObjectName("pushButton_Fall_Detection")
        self.verticalLayout_2.addWidget(self.pushButton_Fall_Detection)
        self.pushButton_Fall_Images = QtWidgets.QPushButton(self.frame_6)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icons/图片.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_Fall_Images.setIcon(icon4)
        self.pushButton_Fall_Images.setIconSize(QtCore.QSize(30, 30))
        self.pushButton_Fall_Images.setObjectName("pushButton_Fall_Images")
        self.verticalLayout_2.addWidget(self.pushButton_Fall_Images)
        self.horizontalLayout_4.addWidget(self.frame_6)
        self.frame_7 = QtWidgets.QFrame(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(10)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_7.sizePolicy().hasHeightForWidth())
        self.frame_7.setSizePolicy(sizePolicy)
        self.frame_7.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)

        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.frame_7)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.Home_label = QtWidgets.QLabel(self.frame_7)
        self.Home_label.setText("")
        self.Home_label.setObjectName("Home_label")
        self.horizontalLayout_5.addWidget(self.Home_label)
        self.horizontalLayout_4.addWidget(self.frame_7)
        self.verticalLayout.addWidget(self.frame_3)
        HomeWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(HomeWindow)
        self.pushButton_4.clicked.connect(HomeWindow.close)
        self.pushButton_3.clicked.connect(HomeWindow.showMinimized)
        QtCore.QMetaObject.connectSlotsByName(HomeWindow)

    def retranslateUi(self, HomeWindow):
        _translate = QtCore.QCoreApplication.translate
        HomeWindow.setWindowTitle(_translate("HomeWindow", "MainWindow"))
        self.pushButton.setText(_translate("HomeWindow", "Home"))
        self.pushButton_Fall_Detection.setText(_translate("HomeWindow", "Fall-Detection"))
        self.pushButton_Fall_Images.setText(_translate("HomeWindow", "Fall-Images"))


class HomeWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(HomeWindow, self).__init__()
        self.ui = Ui_HomeWindow()
        self.ui.setupUi(self)
        self.ui.pushButton_Fall_Detection.installEventFilter(self)
        self.ui.pushButton_Fall_Images.installEventFilter(self)
        self.current_image_path = ""
        self.show_image("newui/images/yolo.png")

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Enter:
            if obj == self.ui.pushButton_Fall_Detection:
                self.show_image("newui/images/Fall-Detection.png")
            elif obj == self.ui.pushButton_Fall_Images:
                self.show_image("newui/images/images.png")
        elif event.type() == QtCore.QEvent.Leave:
            self.show_image("newui/images/yolo.png")
        return super().eventFilter(obj, event)

    def show_image(self, image_path):
        pixmap = QPixmap(image_path)
        self.ui.Home_label.setPixmap(pixmap)
        self.ui.Home_label.setAlignment(QtCore.Qt.AlignCenter)
        self.ui.Home_label.setScaledContents(True)  # 让 QLabel 自动缩放以适应图片大小
        self.ui.Home_label.setMinimumSize(1, 1)  # 设置 QLabel 的最小尺寸以避免折叠


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWindow = HomeWindow()
    mainWindow.show()
    sys.exit(app.exec_())
