import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QGridLayout, QWidget, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QColor, QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QPoint
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

        self.process_image_button = QPushButton("Binarize Image", self)
        self.process_image_button.clicked.connect(self.process_image)

        self.erode_button = QPushButton("Erode Image", self)
        self.erode_button.clicked.connect(self.erode_image)

        self.open_button = QPushButton("Open Image", self)
        self.open_button.clicked.connect(self.open_image)

        self.close_button = QPushButton("Close Image", self)
        self.close_button.clicked.connect(self.close_image)

        self.components_button = QPushButton("Connected components", self)
        self.close_button.clicked.connect(self.find_connected_components)

        self.add_circles_button = QPushButton("Add Circles", self)
        self.add_circles_button.clicked.connect(self.add_circles)

        layout = QGridLayout()
        layout.addWidget(self.image_label, 0, 0, 1, 3)
        layout.addWidget(self.select_button, 1, 0)
        layout.addWidget(self.process_image_button, 1, 1)
        layout.addWidget(self.erode_button, 1, 2)
        layout.addWidget(self.open_button, 2, 0)
        layout.addWidget(self.close_button, 2, 1)
        layout.addWidget(self.components_button, 2, 2)
        layout.addWidget(self.add_circles_button, 3, 0)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        self.setWindowIcon(QIcon('icon.png'))

        self.image_pixels = None  # Переменная для хранения пикселей изображения

    def select_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        file_dialog.setViewMode(QFileDialog.Detail)

        if file_dialog.exec_():
            selected_file = file_dialog.selectedFiles()[0]
            pixmap = QPixmap(selected_file)
            pixmap = pixmap.scaled(1200, 1200, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)  # Масштабируем содержимое QLabel
            self.image_label.adjustSize()  # Автоматическое подстраивание размеров QLabel под размеры изображения
            # Открытие изображения с использованием PIL
            image = Image.open(selected_file)
            self.image_pixels = image.convert("RGB")  # Получаем объект изображения в формате RGB

    def process_image(self):
        if self.image_pixels is not None:
            width, height = self.image_pixels.size
            for x in range(width):
                for y in range(height):
                    r, g, b = self.image_pixels.getpixel((x, y))
                    brightness = int(0.299 * r + 0.587 * g + 0.114 * b)
                    # Проверяем является ли пиксель голубым
                    if r < 150 and g > 150 and b > 150 or brightness > 200 and r < b or r + 10 < b:
                        self.image_pixels.putpixel((x, y), (255, 255, 255))  # Голубые пиксели - белый цвет
                    else:
                        self.image_pixels.putpixel((x, y), (0, 0, 0))  # Остальные пиксели - черный цвет

            # Обновляем изображение в QLabel
            image_data = self.image_pixels.tobytes("raw", "RGB")
            q_image = QImage(image_data, width, height, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image).scaled(1200, 1200, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
        else:
            QMessageBox.warning(self, "Warning", "Please select an image first.")

    def erode_image(self):
        if self.image_label:
            image = self.image_label.pixmap().toImage()
            image = image.convertToFormat(QImage.Format_RGB888)
            width, height = image.width(), image.height()
            ptr = image.bits()
            ptr.setsize(image.byteCount())
            arr = np.array(ptr).reshape(height, width, 3)  # Преобразование изображения в массив numpy
            gray_image = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            kernel = np.ones((2, 2), np.uint8)
            eroded_image = cv2.erode(gray_image, kernel, iterations=1)
            q_image = QImage(eroded_image, width, height, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)
            cv2.destroyAllWindows()
    
    def open_image(self):
        if self.image_label:
            image = self.image_label.pixmap().toImage()
            image = image.convertToFormat(QImage.Format_RGB888)
            width, height = image.width(), image.height()
            ptr = image.bits()
            ptr.setsize(image.byteCount())
            arr = np.array(ptr).reshape(height, width, 3)  # Преобразование изображения в массив numpy
            gray_image = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            kernel = np.ones((2, 2), np.uint8)
            opened_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
            q_image = QImage(opened_image, width, height, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)
            cv2.destroyAllWindows()
    
    def close_image(self):
        if self.image_label:
            image = self.image_label.pixmap().toImage()
            image = image.convertToFormat(QImage.Format_RGB888)
            width, height = image.width(), image.height()
            ptr = image.bits()
            ptr.setsize(image.byteCount())
            arr = np.array(ptr).reshape(height, width, 3)  # Преобразование изображения в массив numpy
            gray_image = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            kernel = np.ones((8, 8), np.uint8)
            closed_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
            q_image = QImage(closed_image, width, height, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)
            cv2.destroyAllWindows()
    
    def find_connected_components(self):
        image = self.image_label.pixmap().toImage()
        image = image.convertToFormat(QImage.Format_Grayscale8)
        width, height = image.width(), image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        binary_image = np.array(ptr).reshape(height, width, 1)  # Преобразование изображения в массив numpy
        # Выполняем выделение связных компонент
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

        # Формируем список для хранения информации о компонентах
        components_info = []

        # Проходим по каждой метке связной компоненты (начинаем с 1, так как 0 - фон)
        for label in range(1, num_labels):
            # Извлекаем статистику для компоненты
            left = stats[label, cv2.CC_STAT_LEFT]
            top = stats[label, cv2.CC_STAT_TOP]
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            area = stats[label, cv2.CC_STAT_AREA]
            centroid_x, centroid_y = centroids[label]

            # Создаем маску для текущей компоненты
            mask = np.zeros_like(binary_image)
            mask[labels == label] = 255

            # Добавляем информацию о компоненте в список
            component_info = {
                'label': label,
                'left': left,
                'top': top,
                'width': width,
                'height': height,
                'area': area,
                'centroid': (centroid_x, centroid_y),
                'mask': mask
            }
            components_info.append(component_info)
        
        return components_info

    def add_circles(self):
        self.connected_components = self.find_connected_components()
        if self.image_label:
            painter = QPainter(self.image_label.pixmap())
            pen = QPen(Qt.red)
            pen.setWidth(2)
            painter.setPen(pen)
            # Добавляем кружки в точки (100, 100) и (200, 200)
            for elem in self.connected_components:
                if elem['area'] > 20:
                    painter.drawRect(elem['left'], elem['top'], elem['width'], elem['height'])
            painter.end()
            self.image_label.setPixmap(self.image_label.pixmap().copy())
            QMessageBox.information(self, "Hello", f"Самолетов обнаружено: {len(self.connected_components)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageSelectorApp()
    window.show()
    sys.exit(app.exec_())
