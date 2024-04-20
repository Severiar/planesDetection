import sys
import cv2
import math
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
        self.setWindowTitle("Детекция и классификация фишек")
        self.setGeometry(50, 50, 400, 300)
        self.active = "None"

        # Создание виджетов и кнопок для интерфейса
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)

        self.select_button = QPushButton("Загрузить с компьютера", self)
        self.reset_button = QPushButton("Сбросить преобразования", self)

        self.canny_operator_button = QPushButton("Применить оператор Кэнни", self)
        self.gradient_lower_bound_input = QLineEdit(self)
        self.gradient_lower_bound_input.setPlaceholderText("Введите нижнее пороговое значение градиента")
        self.gradient_upper_bound_input = QLineEdit(self)
        self.gradient_upper_bound_input.setPlaceholderText("Введите верхнее пороговое значение градиента")
        self.aperture_size_input = QLineEdit(self)
        self.aperture_size_input.setPlaceholderText("Введите размер диафрагмы фильтра Собеля")

        self.hough_transform_button = QPushButton("Применить преобразование Хафа", self)
        self.votes_lower_bound_input = QLineEdit(self)
        self.votes_lower_bound_input.setPlaceholderText("Введите нижнее пороговое значение голосов")

        self.detect_figures_button = QPushButton("Выделить фишки", self)
        #self.min_length_input = QLineEdit(self)
        #self.min_length_input.setPlaceholderText("Введите минимальную длину прямой линии")

        self.detect_dots_button = QPushButton("Классифицировать фишки", self)
        #self.min_length_input = QLineEdit(self)
        #self.min_length_input.setPlaceholderText("Введите минимальную длину прямой линии")

        self.visualize_button = QPushButton("Бонус! Визуализация фишек", self)

        # Установка текста и выравнивания для некоторых QLabel
        self.settings_label = QLabel("Выбор изображения", self)
        self.settings_label.setAlignment(Qt.AlignCenter)
        self.process_label = QLabel("Преобразования изображения", self)
        self.process_label.setAlignment(Qt.AlignCenter)
        self.detection_label = QLabel("Детекция и классификация фишек", self)
        self.detection_label.setAlignment(Qt.AlignCenter)
        self.empty_label = QLabel("", self)
        self.empty_label.setAlignment(Qt.AlignCenter)

        # Расположение виджетов с помощью сетки
        layout = QGridLayout()
        layout.addWidget(self.image_label, 0, 2, 30, 1)
        for index, widget in enumerate((self.empty_label, self.settings_label, self.select_button, self.reset_button, self.empty_label, self.process_label,
                                        self.canny_operator_button, self.gradient_lower_bound_input,
                                        self.gradient_upper_bound_input, self.aperture_size_input, self.empty_label,  
                                        self.hough_transform_button, self.votes_lower_bound_input,
                                        self.empty_label, self.detection_label, self.detect_figures_button,
                                        self.detect_dots_button,
                                        self.visualize_button)):
            layout.addWidget(widget, index, 0)

        # Создание центрального виджета и установка его для главного окна
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Инициализация переменных
        self.image_pixels = None  # Переменная для хранения пикселей изображения
        self.binary = False  # Флаг для проверки наличия бинарного изображения

        # Привязка действий к кнопкам
        self.select_button.clicked.connect(self.select_image)
        self.reset_button.clicked.connect(self.reset_image)
        self.canny_operator_button.clicked.connect(self.apply_canny_operator)
        self.hough_transform_button.clicked.connect(self.hough)
        self.detect_figures_button.clicked.connect(self.detect_triangles)
        self.detect_dots_button.clicked.connect(self.detect_dots)

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
            #pixmap = pixmap.scaled(1400, 1000, Qt.KeepAspectRatio)

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
        self.active = "canny"

        
    
    def reset_image(self):
        # Проверяем, было ли выбрано изображение
        if not self.image_pixels:
            QMessageBox.warning(self, "Warning", "Сначала выберите изображение.")
            return
        
        # Сброс флага бинаризации
        self.binary = False

        # Создаем QPixmap из выбранного файла и масштабируем его
        pixmap = QPixmap(self.selected_file)
        #pixmap = pixmap.scaled(1400, 1000, Qt.KeepAspectRatio)

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
        self.active = "source"

    def apply_canny_operator(self):
        if not self.active == "source":
            QMessageBox.warning(self, "Warning", "Сначала выберите изображение.")
            return
        gray = cv2.cvtColor(cv2.imread(self.selected_file), cv2.COLOR_BGR2GRAY)
        gradient_lower_bound = 50 if not self.gradient_lower_bound_input.text() else int(self.gradient_lower_bound_input.text())
        gradient_upper_bound = 150 if not self.gradient_upper_bound_input.text() else int(self.gradient_upper_bound_input.text())
        aperture_size = 3 if not self.aperture_size_input.text() else int(self.aperture_size_input.text())
        self.edges = cv2.Canny(gray, gradient_lower_bound, gradient_upper_bound, apertureSize=aperture_size)
        cv2.imwrite('edges.bmp', self.edges)
        pixmap = QPixmap('edges.bmp')
        self.image_label.setPixmap(pixmap)
        self.image_label.show()
        self.active = "canny"


    def hough(self):
        if not self.active == "canny":
            QMessageBox.warning(self, "Warning", "Сначала примените оператор Кэнни.")
            return
        votes_lower_bound = 100 if not self.votes_lower_bound_input.text() else int(self.votes_lower_bound_input.text())
        lines = cv2.HoughLines(self.edges, 1, np.pi / 180, votes_lower_bound)
        self.image_with_lines = cv2.imread('edges.bmp')
        if lines is not None:
            for rho, theta in lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                cv2.line(self.image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.imwrite('lines.jpg', self.image_with_lines)
        pixmap = QPixmap('lines.jpg')
        self.image_label.setPixmap(pixmap)
        self.lines = lines
        self.image_label.show()
        self.active = "hough"
    
    def detect_triangles(self):
        if not self.active == "hough":
            QMessageBox.warning(self, "Warning", "Сначала примените преобразование Хафа.")
            return
        self.image_with_edges = cv2.imread('edges.bmp')
        min_line_length = 40
        source_image = cv2.imread(self.selected_file)
        norm = np.zeros_like(source_image)
        source_image = cv2.normalize(source_image,  norm, 0, 255, cv2.NORM_MINMAX)
        source_image = norm
        
        lines_counts = []
        triangles_centers = []
        triangles_squares = []
        triangles_dots = []
        for rho, theta in self.lines[:, 0]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = x0 + 1000 * (-b)
                y1 = y0 + 1000 * (a)
                x2 = x0 - 1000 * (-b)
                y2 = y0 - 1000 * (a)
                #print((x1, y1), (x2, y2))
                count = 0
                best_count = 0
                for i in range(1, 999):
                    xi = x1 + (x2 - x1) / 1000 * i
                    yi = y1 + (y2 - y1) / 1000 * i
                    if 0 < yi < self.image_with_edges.shape[0] - 1 and \
                       0 < xi < self.image_with_edges.shape[1] - 1:
                        #print(int(xi), int(yi))
                        summ = 0
                        for i1 in (-1, 0, 1):
                            for i2 in (-1, 0, 1):
                                summ += np.sum(self.image_with_edges[int(yi)+i1][int(xi) + i2])
                        if summ > 700:
                            #print(int(xi), int(yi), summ)
                            count += 1
                        else:
                            best_count = max(best_count, count)
                            if count >= min_line_length:
                                xmid, ymid = (xi + x_start) / 2, (yi + y_start) / 2
                                xinner, yinner = xi, y_start
                                xouter, youter = x_start, yi
                                norm = (xinner - xmid, yinner - ymid)
                                xinner, yinner = xinner - norm[0] * 0.8, yinner - norm[1] * 0.8
                                norm = (xouter - xmid, youter - ymid)
                                xouter, youter = xouter - norm[0] * 0.8, youter - norm[1] * 0.8
                                brightness_inner = np.sum(source_image[int(yinner)][int(xinner)] * [0.299, 0.587, 0.114])
                                brightness_outer = np.sum(source_image[int(youter)][int(xouter)] * [0.299, 0.587, 0.114])
                                if brightness_inner > brightness_outer:
                                    xinner, xouter = xouter, xinner
                                    yinner, youter = youter, yinner
                                #cv2.circle(source_image, (int(xinner), int(yinner)), radius=3, color=(255, 0, 0), thickness=-1)
                                if yinner < ymid:
                                    pt3 = self.find_third_point((x_start, y_start), (xi, yi), -90)
                                else:
                                    pt3 = self.find_third_point((x_start, y_start), (xi, yi), 90)
                                if self.image_with_edges.shape[1] > pt3[0] >= 0 and self.image_with_edges.shape[0] > pt3[1] >= 0:

                                    #print([(int(x_start), int(y_start)), (int(xi), int(yi)), pt3])
                                    triangle_center = np.array(((int(x_start) + int(xi) + pt3[0]) // 3, (int(y_start) + int(yi) + pt3[1]) // 3))
                                    a = np.linalg.norm((x_start - xi, y_start - yi))
                                    for i in range(len(triangles_centers)):
                                        if np.linalg.norm(triangles_centers[i] - triangle_center) < 50:
                                            if triangles_squares[i] < 3 ** 0.5 * a ** 2 / 4:
                                                triangles_centers[i] = triangle_center.copy()
                                                triangles_squares[i] = 3 ** 0.5 * a ** 2 / 4
                                                triangles_dots[i] = ((x_start, y_start), (xi, yi), pt3)
                                            break
                                    else:
                                        triangles_centers.append(triangle_center)
                                        triangles_squares.append(3 ** 0.5 * a ** 2 / 4)
                                        triangles_dots.append(((x_start, y_start), (xi, yi), pt3))
                                            
                                    
                            count = 0
                            x_start, y_start = xi, yi
                lines_counts.append(best_count)

        for i in range(len(triangles_dots)):
            x_start, y_start = triangles_dots[i][0]
            xi, yi = triangles_dots[i][1]
            pt3 = triangles_dots[i][2]
            cv2.line(source_image, (int(x_start), int(y_start)), (int(xi), int(yi)), (0, 255, 255), 2)
            cv2.line(source_image, (int(xi), int(yi)), pt3, (0, 255, 255), 2)
            cv2.line(source_image, pt3, (int(x_start), int(y_start)), (0, 255, 255), 2)
            cv2.circle(source_image, triangles_centers[i], radius=5, color=(255, 255, 0), thickness=-1)
        self.triangles = triangles_dots
        self.centers = triangles_centers

        cv2.imwrite('lines.jpg', source_image)
        pixmap = QPixmap('lines.jpg')
        self.image_label.setPixmap(pixmap)
        self.image_label.show()
        print(lines_counts)

    def area(self, x1, y1, x2, y2, x3, y3):
        return abs((x1 * (y2 - y3) + x2 * (y3 - y1) 
                + x3 * (y1 - y2)) / 2.0)
 
    def isInside(self, x1, y1, x2, y2, x3, y3, x, y):
    
        # Calculate area of triangle ABC
        A = self.area(x1, y1, x2, y2, x3, y3)
    
        # Calculate area of triangle PBC 
        A1 = self.area(x, y, x2, y2, x3, y3)
        
        # Calculate area of triangle PAC 
        A2 = self.area(x1, y1, x, y, x3, y3)
        
        # Calculate area of triangle PAB 
        A3 = self.area(x1, y1, x2, y2, x, y)
        
        # Check if sum of A1, A2 and A3 
        # is same as A
        if(A == A1 + A2 + A3):
            return True
        else:
            return False
    
    def detect_dots(self):
        if not self.active == "hough":
            QMessageBox.warning(self, "Warning", "Сначала выделите фишки.")
            return
        gray = cv2.cvtColor(cv2.imread(self.selected_file), cv2.COLOR_BGR2GRAY)
        img = cv2.imread(self.selected_file)
        triangles_colors = [[0 for i in range(5)] for j in range(len(self.triangles))]
        for x in range(img.shape[1]):
            for y in range(img.shape[0]):
                for i in range(len(self.triangles)):
                    x_start, y_start = self.triangles[i][0]
                    xi, yi = self.triangles[i][1]
                    pt3 = self.triangles[i][2]
                    if self.isInside(int(x_start), int(y_start), int(xi), int(yi), int(pt3[0]), int(pt3[1]), x, y):
                        brightness = img[y][x][2] * 0.299 + img[y][x][1] * 0.587 + img[y][x][0] * 0.114
                        is_white = brightness > 150 or (brightness > 130 and img[y][x][2] / img[y][x][1] < 1.4 and img[y][x][1] / img[y][x][0] < 1.4 and img[y][x][0] / img[y][x][1] < 1.4 and img[y][x][0] / img[y][x][2] < 1.4)
                        is_green = img[y][x][1] > img[y][x][0] and img[y][x][1] + 5 > img[y][x][2] and 20 < brightness < 80
                        is_yellow_1 = img[y][x][0] < 40 and img[y][x][2] > 130 and img[y][x][1] > 80
                        is_yellow_2 = img[y][x][2] > 7 * img[y][x][0] and img[y][x][1] > 50
                        is_blue = img[y][x][0] > img[y][x][1] + 5 and img[y][x][0] > img[y][x][2] and brightness > 40
                        is_red = img[y][x][2] > 2.7 * img[y][x][1]
                        if prev_border:
                            img[y][x] = (46, 67, 105)
                            if not (is_white or is_green or is_yellow_1 or is_yellow_2 or is_blue or is_red):
                                prev_border = False
                        elif is_white: # white
                            triangles_colors[i][0] += 1
                            img[y][x] = (255, 255, 255)
                        elif is_green: # green
                            triangles_colors[i][1] += 1
                            img[y][x] = (0, 255, 0)
                        elif is_yellow_1 or is_yellow_2: # yellow
                            triangles_colors[i][2] += 1
                            img[y][x] = (0, 255, 255)
                        elif is_blue: # blue
                            triangles_colors[i][3] += 1
                            img[y][x] = (255, 0, 0)
                        elif is_red: # red
                            triangles_colors[i][4] += 1
                            img[y][x] = (0, 0, 255)
                        else:
                            img[y][x] = (46, 67, 105)
                        break
                else:
                    img[y][x] = (0, 0 ,0)
                    gray[y][x] = 0
                    prev_border = True
        corners_dots = [[[0 for i in range(5)] for j in range(3)] for k in range(len(self.triangles))]
        for x in range(1, img.shape[1] - 1):
            for y in range(1, img.shape[0] - 1):
                if not (img[y][x][0] == 46 and img[y][x][1] == 67 and img[y][x][0] == 105) and sum(img[y][x]) != 0:
                    inner = True
                    for i in range(-2, 3):
                        for j in range(-2, 3):
                            if sum(img[y + i][x + j]) == 0:
                                img[y][x] = (46, 67, 105)
                                inner = False
                    if inner:
                        dists = np.array([np.linalg.norm(np.array(self.triangles[i // 3][i % 3]) - np.array([x, y])) for i in range(3 * len(self.triangles))])
                        closest = np.argsort(dists)[0]
                        if img[y][x][0] == 255 and img[y][x][1] == 255 and img[y][x][2] == 255:
                            corners_dots[closest // 3][closest % 3][0] += 1
                        elif img[y][x][0] == 0 and img[y][x][1] == 255 and img[y][x][2] == 0:
                            corners_dots[closest // 3][closest % 3][1] += 1
                        elif img[y][x][0] == 0 and img[y][x][1] == 255 and img[y][x][2] == 255:
                            corners_dots[closest // 3][closest % 3][2] += 1
                        elif img[y][x][0] == 255 and img[y][x][1] == 0 and img[y][x][2] == 0:
                            corners_dots[closest // 3][closest % 3][3] += 1
                        elif img[y][x][0] == 0 and img[y][x][1] == 0 and img[y][x][2] == 255:
                            corners_dots[closest // 3][closest % 3][4] += 1
                            
        triangles_colors[i][0] /= 22
        triangles_colors[i][1] /= 44
        triangles_colors[i][2] /= 66
        triangles_colors[i][3] /= 88
        triangles_colors[i][4] /= 35 * 5
        for i in range(len(self.triangles)):
            print(self.centers[i], triangles_colors[i])
        cv2.imwrite('lines.bmp', img)
        pixmap = QPixmap('lines.bmp')
        self.image_label.setPixmap(pixmap)
        print(corners_dots)
        with open('results.txt', 'w') as file:
            for i in np.argsort(np.array(self.centers)[:, 0]):
                file.write(f'{self.centers[i][0]},{self.centers[i][1]}; {np.argmax(corners_dots[i][0]) + 1 if corners_dots[i][0][np.argmax(corners_dots[i][0])] > 7 else 0}, {np.argmax(corners_dots[i][1]) + 1 if corners_dots[i][1][np.argmax(corners_dots[i][1])] > 7 else 0}, {np.argmax(corners_dots[i][2]) + 1 if corners_dots[i][2][np.argmax(corners_dots[i][2])] > 7 else 0}\n')
        self.image_label.show()

    def find_third_point(self, p1, p2, bal):
        # Находим расстояние между двумя известными точками
        dist = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        # Находим координаты третьей точки
        # Для правильного треугольника высота равна половине стороны, а основание - сторона треугольника
        height = dist * math.sqrt(3) / 2
        midpoint_x = (p1[0] + p2[0]) / 2
        midpoint_y = (p1[1] + p2[1]) / 2
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0]) + math.radians(bal)  # угол поворота на 60 градусов
        third_x = midpoint_x + height * math.cos(angle)
        third_y = midpoint_y + height * math.sin(angle)
        
        return (int(third_x), int(third_y))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlanesDetectionApp()
    window.show()
    sys.exit(app.exec_())
