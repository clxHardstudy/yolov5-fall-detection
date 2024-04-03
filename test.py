import sys

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QListWidget, QListWidgetItem, QVBoxLayout, QWidget, QLabel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Folder Viewer")
        self.setGeometry(100, 100, 400, 300)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()
        self.central_widget.setLayout(layout)

        self.folder_list = QListWidget()
        self.folder_list.itemClicked.connect(self.show_images_in_folder)
        layout.addWidget(self.folder_list)

        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        # 添加示例文件夹和图片
        self.add_folder("Folder 1", ["image1.jpg", "image2.jpg"])
        self.add_folder("Folder 2", ["image3.jpg", "image4.jpg"])

    def add_folder(self, folder_name, images):
        folder_item = QListWidgetItem(folder_name)
        folder_item.setData(1, images)  # 存储文件夹内图片的数据
        self.folder_list.addItem(folder_item)

    def show_images_in_folder(self, item):
        images = item.data(1)  # 获取文件夹内图片的数据
        if images:
            # 显示文件夹内的第一张图片
            if images:
                image_path = images[0]  # 获取第一张图片的路径
                pixmap = QPixmap(image_path)
                self.image_label.setPixmap(pixmap)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
