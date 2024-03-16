import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, 
    QGridLayout, QWidget, QFileDialog, QMessageBox, 
    QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage, QIcon, QPainter, QPen
from PyQt5.QtCore import Qt
from PIL import Image

class PlanesDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Установка заголовка и размеров окна
        self.setWindowTitle("Распознавание самолётов")
        self.setGeometry(100, 100, 400, 300)

        # Создание виджетов и кнопок для интерфейса
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.select_button = QPushButton("Загрузить с компьютера", self)
        self.reset_button = QPushButton("Сбросить преобразования", self)
        self.process_image_button = QPushButton("Применить бинаризацию", self)
        self.erode_button = QPushButton("Применить эрозию", self)
        self.dilate_button = QPushButton("Применить дилатацию", self)
        self.open_button = QPushButton("Применить открытие", self)
        self.close_button = QPushButton("Применить замыкание", self)
        self.add_circles_button = QPushButton("Выделить и посчитать самолёты", self)

        self.area_input = QLineEdit(self)
        self.area_input.setPlaceholderText("Введите минимальный размер самолета (пикс.)")
        self.open_kernel_size_input = QLineEdit(self)
        self.close_kernel_size_input = QLineEdit(self)
        self.dilate_kernel_size_input = QLineEdit(self)
        self.erose_kernel_size_input = QLineEdit(self)
        for input_field in (self.open_kernel_size_input, self.close_kernel_size_input, self.erose_kernel_size_input,
                            self.dilate_kernel_size_input):
            input_field.setPlaceholderText("Введите размер ядра (одно число)")

        self.info_button = QPushButton("?", self)
        self.info_button.setFixedSize(25, 25)
        self.info_button.setToolTip("При бинаризации используется формула: (brightness > 200 and R < B) or (R + 10 < B)")
        self.kernel_info_buttons = [QPushButton("?", self) for _ in range(4)]

        # Установка текста и выравнивания для некоторых QLabel
        self.settings_label = QLabel("Выбор изображения", self)
        self.settings_label.setAlignment(Qt.AlignCenter)
        self.process_label = QLabel("Преобразования изображения", self)
        self.process_label.setAlignment(Qt.AlignCenter)
        self.detection_label = QLabel("Детекция самолётов", self)
        self.detection_label.setAlignment(Qt.AlignCenter)
        self.empty_label = QLabel("", self)
        self.empty_label.setAlignment(Qt.AlignCenter)

        # Расположение виджетов с помощью сетки
        layout = QGridLayout()
        layout.addWidget(self.image_label, 0, 2, 30, 1)
        for index, widget in enumerate((self.empty_label, self.settings_label, self.select_button, self.reset_button, self.empty_label, self.process_label,
                                        self.process_image_button, self.empty_label, self.erode_button, 
                                        self.erose_kernel_size_input, self.empty_label, self.dilate_button, 
                                        self.dilate_kernel_size_input, self.empty_label, self.open_button, 
                                        self.open_kernel_size_input, self.empty_label, self.close_button, self.close_kernel_size_input, 
                                        self.empty_label, self.detection_label, self.add_circles_button, self.area_input)):
            layout.addWidget(widget, index, 0)
        layout.addWidget(self.info_button, 6, 1)
        for i, button in enumerate(self.kernel_info_buttons):
            button.setFixedSize(25, 25)
            layout.addWidget(button, 8 + i*3, 1)  # Расположение кнопок с информацией о ядре

        # Создание центрального виджета и установка его для главного окна
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Установка иконки окна
        self.setWindowIcon(QIcon('icon.png'))

        # Инициализация переменных
        self.image_pixels = None  # Переменная для хранения пикселей изображения
        self.binary = False  # Флаг для проверки наличия бинарного изображения

        # Привязка действий к кнопкам
        self.select_button.clicked.connect(self.select_image)
        self.reset_button.clicked.connect(self.reset_image)
        self.process_image_button.clicked.connect(self.binarize_image)
        self.erode_button.clicked.connect(self.erode_image)
        self.dilate_button.clicked.connect(self.dilate_image)
        self.open_button.clicked.connect(self.open_image)
        self.close_button.clicked.connect(self.close_image)
        self.add_circles_button.clicked.connect(self.count_and_highlight_objects)
        self.info_button.clicked.connect(lambda x: QMessageBox.information(self, "О бинаризации", "При бинаризации используется формула: (R < 150 and G > 150 and B > 150) or (brightness > 200 and R < B) or (R + 10 < B)"))
        for button in self.kernel_info_buttons:
            button.setToolTip("В качестве примитива используется квадратное ядро (матрица из единиц)")
            button.clicked.connect(lambda x: QMessageBox.information(self, "О бинаризации", "В качестве примитива используется квадратное ядро (матрица из единиц)"))


    def select_image(self):
        # Создаем диалоговое окно для выбора изображения
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp)")
        file_dialog.setViewMode(QFileDialog.Detail)

        # Открываем диалоговое окно и проверяем, был ли выбран файл
        if file_dialog.exec_():
            # Получаем путь к выбранному файлу
            self.selected_file = file_dialog.selectedFiles()[0]

            # Создаем QPixmap из выбранного файла и масштабируем его
            pixmap = QPixmap(self.selected_file)
            pixmap = pixmap.scaled(1400, 1400, Qt.KeepAspectRatio)

            # Устанавливаем масштабированный QPixmap в QLabel
            self.image_label.setPixmap(pixmap)
            # Масштабируем содержимое QLabel
            self.image_label.setScaledContents(True)
            # Автоматически подстраиваем размеры QLabel под размеры изображения
            self.image_label.adjustSize()

            # Открываем изображение с помощью библиотеки PIL
            image = Image.open(self.selected_file)
            # Преобразуем изображение в формат RGB и сохраняем для дальнейшей обработки
            self.image_pixels = image.convert("RGB")

        
    
    def reset_image(self):
        # Проверяем, было ли выбрано изображение
        if not self.image_pixels:
            QMessageBox.warning(self, "Warning", "Сначала выберите изображение.")
            return
        
        # Сброс флага бинаризации
        self.binary = False

        # Создаем QPixmap из выбранного файла и масштабируем его
        pixmap = QPixmap(self.selected_file)
        pixmap = pixmap.scaled(1400, 1400, Qt.KeepAspectRatio)

        # Устанавливаем масштабированный QPixmap в QLabel
        self.image_label.setPixmap(pixmap)
        # Масштабируем содержимое QLabel
        self.image_label.setScaledContents(True)
        # Автоматически подстраиваем размеры QLabel под размеры изображения
        self.image_label.adjustSize()

        # Открываем изображение с помощью библиотеки PIL
        image = Image.open(self.selected_file)
        # Преобразуем изображение в формат RGB и сохраняем для дальнейшей обработки
        self.image_pixels = image.convert("RGB")


    def binarize_image(self):
        # Проверяем, было ли выбрано изображение
        if not self.image_pixels:
            QMessageBox.warning(self, "Warning", "Сначала выберите изображение.")
            return
        
        width, height = self.image_pixels.size
        # Проходим по каждому пикселю изображения
        for x in range(width):
            for y in range(height):
                r, g, b = self.image_pixels.getpixel((x, y))
                # Вычисляем яркость пикселя по формуле
                brightness = int(0.299 * r + 0.587 * g + 0.114 * b)
                # Проверяем условия для бинаризации пикселя
                if brightness > 200 and r < b or r + 10 < b:
                    # Если пиксель удовлетворяет условиям, делаем его белым
                    self.image_pixels.putpixel((x, y), (255, 255, 255))
                else:
                    # Иначе делаем пиксель черным
                    self.image_pixels.putpixel((x, y), (0, 0, 0))

        # Обновляем изображение в QLabel после бинаризации
        image_data = self.image_pixels.tobytes("raw", "RGB")
        q_image = QImage(image_data, width, height, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image).scaled(1400, 1400, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)
        # Устанавливаем флаг бинаризации в True
        self.binary = True


    def erode_image(self):
        # Проверяем, было ли выбрано изображение
        if not self.image_pixels:
            QMessageBox.warning(self, "Warning", "Сначала выберите изображение.")
            return
        
        # Проверяем, была ли выполнена бинаризация
        if not self.binary:
            QMessageBox.warning(self, "Warning", "Сначала примените бинаризацию.")
            return
        
        # Получаем изображение из QLabel и конвертируем его в формат RGB888
        image = self.image_label.pixmap().toImage()
        image = image.convertToFormat(QImage.Format_RGB888)
        width, height = image.width(), image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(height, width, 3)  # Преобразование изображения в массив numpy

        # Конвертируем изображение в оттенки серого для использования с OpenCV
        gray_image = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # Получаем размер ядра для эрозии из пользовательского ввода или устанавливаем по умолчанию
        kernel_size = 3 if not self.erose_kernel_size_input.text() else int(self.erose_kernel_size_input.text())
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Применяем эрозию с помощью OpenCV
        eroded_image = cv2.erode(gray_image, kernel, iterations=1)

        # Преобразуем результат эрозии в QImage и отображаем в QLabel
        q_image = QImage(eroded_image, width, height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)


    def dilate_image(self):
        # Проверяем, было ли выбрано изображение
        if not self.image_pixels:
            QMessageBox.warning(self, "Warning", "Сначала выберите изображение.")
            return
        
        # Проверяем, была ли выполнена бинаризация
        if not self.binary:
            QMessageBox.warning(self, "Warning", "Сначала примените бинаризацию.")
            return
        
        # Получаем изображение из QLabel и конвертируем его в формат RGB888
        image = self.image_label.pixmap().toImage()
        image = image.convertToFormat(QImage.Format_RGB888)
        width, height = image.width(), image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(height, width, 3)  # Преобразование изображения в массив numpy

        # Конвертируем изображение в оттенки серого для использования с OpenCV
        gray_image = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # Создаем ядро для дилатации (2x2 квадратное ядро)
        kernel = np.ones((2, 2), np.uint8)

        # Применяем дилатацию с помощью OpenCV
        dilated_image = cv2.dilate(gray_image, kernel, iterations=1)

        # Преобразуем результат дилатации в QImage и отображаем в QLabel
        q_image = QImage(dilated_image, width, height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    
    def open_image(self):
        # Проверяем, было ли выбрано изображение
        if not self.image_pixels:
            QMessageBox.warning(self, "Warning", "Сначала выберите изображение.")
            return
        
        # Проверяем, была ли выполнена бинаризация
        if not self.binary:
            QMessageBox.warning(self, "Warning", "Сначала примените бинаризацию.")
            return
        
        # Получаем изображение из QLabel и конвертируем его в формат RGB888
        image = self.image_label.pixmap().toImage()
        image = image.convertToFormat(QImage.Format_RGB888)
        width, height = image.width(), image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(height, width, 3)  # Преобразование изображения в массив numpy

        # Конвертируем изображение в оттенки серого для использования с OpenCV
        gray_image = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # Получаем размер ядра для операции открытия из пользовательского ввода или устанавливаем по умолчанию
        kernel_size = 3 if not self.open_kernel_size_input.text() else int(self.open_kernel_size_input.text())
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Применяем операцию открытия с помощью OpenCV
        opened_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)

        # Преобразуем результат операции открытия в QImage и отображаем в QLabel
        q_image = QImage(opened_image, width, height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

    
    def close_image(self):
        # Проверяем, было ли выбрано изображение
        if not self.image_pixels:
            QMessageBox.warning(self, "Warning", "Сначала выберите изображение.")
            return
        
        # Проверяем, была ли выполнена бинаризация
        if not self.binary:
            QMessageBox.warning(self, "Warning", "Сначала примените бинаризацию.")
            return
        
        # Получаем изображение из QLabel и конвертируем его в формат RGB888
        image = self.image_label.pixmap().toImage()
        image = image.convertToFormat(QImage.Format_RGB888)
        width, height = image.width(), image.height()
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        arr = np.array(ptr).reshape(height, width, 3)  # Преобразование изображения в массив numpy

        # Конвертируем изображение в оттенки серого для использования с OpenCV
        gray_image = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # Получаем размер ядра для операции закрытия из пользовательского ввода или устанавливаем по умолчанию
        kernel_size = 4 if not self.close_kernel_size_input.text() else int(self.close_kernel_size_input.text())
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        # Применяем операцию закрытия с помощью OpenCV
        closed_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)

        # Преобразуем результат операции закрытия в QImage и отображаем в QLabel
        q_image = QImage(closed_image, width, height, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)

        # Закрываем все окна OpenCV
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


    def count_and_highlight_objects(self):
        # Проверяем, было ли выбрано изображение
        if not self.image_pixels:
            QMessageBox.warning(self, "Warning", "Сначала выберите изображение.")
            return
        
        # Проверяем, была ли выполнена бинаризация
        if not self.binary:
            QMessageBox.warning(self, "Warning", "Сначала примените бинаризацию.")
            return

        # Получаем связные компоненты
        self.connected_components = self.find_connected_components()

        # Создаем QPainter для рисования на изображении в QLabel
        painter = QPainter(self.image_label.pixmap())
        pen = QPen(Qt.red)
        pen.setWidth(2)
        painter.setPen(pen)

        # Счетчик для подсчета обнаруженных объектов
        counter = 0
        # Задаем порог для площади объектов (из пользовательского ввода или устанавливаем по умолчанию)
        limit = 10 if not self.area_input.text() else int(self.area_input.text())
        
        # Проходим по каждой связной компоненте
        for elem in self.connected_components:
            if elem['area'] > limit:
                # Рисуем прямоугольник вокруг объекта
                painter.drawRect(elem['left'], elem['top'], elem['width'], elem['height'])
                counter += 1

        # Завершаем рисование
        painter.end()

        # Обновляем изображение в QLabel
        self.image_label.setPixmap(self.image_label.pixmap().copy())

        # Выводим информационное сообщение с количеством обнаруженных объектов
        QMessageBox.information(self, "Результат детекции", f"Самолетов обнаружено: {counter}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlanesDetectionApp()
    window.show()
    sys.exit(app.exec_())
