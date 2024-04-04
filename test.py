import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QScrollArea, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
import requests


class MainWindow(QMainWindow):
    def __init__(self, image_urls):
        super().__init__()

        self.setWindowTitle("Image Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout(self.central_widget)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget(scroll_area)
        scroll_layout = QVBoxLayout(scroll_content)

        hbox_layout = QHBoxLayout()
        hbox_layout.setSpacing(10)

        images_in_row = 0
        for url in image_urls:
            pixmap = self.load_image_from_url(url)
            if pixmap is not None:
                label = QLabel()
                label.setPixmap(pixmap)
                label.setMaximumSize(200, 200)  # 设置最大尺寸为200x200px，等比例缩放
                label.setStyleSheet("border: 1px solid black;")  # 设置边框样式
                hbox_layout.addWidget(label)
                images_in_row += 1
                if images_in_row >= 3:  # 3 images per row
                    scroll_layout.addLayout(hbox_layout)
                    hbox_layout = QHBoxLayout()
                    hbox_layout.setSpacing(10)
                    images_in_row = 0

        scroll_content.setLayout(scroll_layout)
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

    def load_image_from_url(self, url):
        response = requests.get(url)
        if response.status_code == 200:
            image = QImage()
            image.loadFromData(response.content)
            pixmap = QPixmap.fromImage(image)
            return pixmap.scaled(200, 200, aspectRatioMode=Qt.KeepAspectRatio)  # 等比例缩放至指定大小
        else:
            print("Failed to load image from URL:", url)
            return None


if __name__ == "__main__":
    image_urls = [
        "http://127.0.0.1:8000/file//file/minio/202404/04/040909187098.jpg/uri",
        "http://127.0.0.1:8000/file//file/minio/202404/04/040909187098.jpg/uri",
        "http://127.0.0.1:8000/file//file/minio/202404/04/040909187098.jpg/uri",
        "http://127.0.0.1:8000/file//file/minio/202404/04/040909187098.jpg/uri",
        "http://127.0.0.1:8000/file//file/minio/202404/04/040909187098.jpg/uri",
        "http://127.0.0.1:8000/file//file/minio/202404/04/040909187098.jpg/uri",
        "http://127.0.0.1:8000/file/%2Ffile%2Fminio%2F202404%2F04%2F1712191995.jpg/uri",
        "http://127.0.0.1:8000/file/%2Ffile%2Fminio%2F202404%2F04%2F1712191995.jpg/uri",
        "http://127.0.0.1:8000/file/%2Ffile%2Fminio%2F202404%2F04%2F1712191995.jpg/uri",
        "http://127.0.0.1:8000/file/%2Ffile%2Fminio%2F202404%2F04%2F1712191995.jpg/uri",
        "http://127.0.0.1:8000/file/%2Ffile%2Fminio%2F202404%2F04%2F1712191995.jpg/uri",
        "http://127.0.0.1:8000/file/%2Ffile%2Fminio%2F202404%2F04%2F1712191995.jpg/uri",
        "http://127.0.0.1:8000/file/%2Ffile%2Fminio%2F202404%2F04%2F1712191995.jpg/uri",
        "http://127.0.0.1:8000/file/%2Ffile%2Fminio%2F202404%2F04%2F1712191995.jpg/uri",
        # Add more image URLs here
    ]
    app = QApplication(sys.argv)
    window = MainWindow(image_urls)
    window.show()
    sys.exit(app.exec_())
