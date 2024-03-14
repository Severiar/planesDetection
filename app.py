import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image

class ImageSelectorApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Image Selector")
        self.setGeometry(100, 100, 400, 300)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)  # Центрирование по горизонтали и вертикали

        self.select_button = QPushButton("Select Image", self)
        self.select_button.clicked.connect(self.select_image)

        self.get_pixels_button = QPushButton("Get Pixels", self)
        self.get_pixels_button.clicked.connect(self.get_image_pixels)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.select_button)
        layout.addWidget(self.get_pixels_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.image_pixels = None  # Переменная для хранения пикселей изображения

    def select_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        file_dialog.setViewMode(QFileDialog.Detail)

        if file_dialog.exec_():
            selected_file = file_dialog.selectedFiles()[0]
            pixmap = QPixmap(selected_file)
            pixmap = pixmap.scaled(1000, 1000, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)  # Масштабируем содержимое QLabel
            self.image_label.adjustSize()  # Автоматическое подстраивание размеров QLabel под размеры изображения
            # Открытие изображения с использованием PIL
            image = Image.open(selected_file)
            self.image_pixels = list(image.getdata())  # Получаем массив пикселей

    def get_image_pixels(self):
        if self.image_pixels is not None:
            print("Pixels of the image:")
            print(self.image_pixels)
        else:
            QMessageBox.warning(self, "Warning", "Please select an image first.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageSelectorApp()
    window.show()
    sys.exit(app.exec_())
