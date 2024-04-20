import sys
import subprocess
import math
from PyQt5.QtWidgets import QMessageBox, QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QSlider, QLineEdit, QInputDialog
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont
from PyQt5.QtCore import Qt
from PIL import Image
import numpy as np
from PyQt5.QtGui import QDoubleValidator, QIntValidator
import matplotlib.pyplot as plt
import cv2



class ImageUploader(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Сегментация фишек')
        desktop = QApplication.desktop()
        screen_rect = desktop.screenGeometry()
        screen_width, screen_height = screen_rect.width(), screen_rect.height()
        self.resize(int(screen_width * 1), int(screen_height * 1))


        self.label = QLabel(self)
        self.label.setText("Исходное изображение")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFixedSize(int(screen_width * 0.42), int(screen_height * 0.8))
        self.label.setStyleSheet("background-color: white; border: 2px solid black;")

        

        self.par1_input = QLineEdit()
        self.par2_input = QLineEdit()
        self.par3_input = QLineEdit()
        self.par4_input = QLineEdit()
        validator_i = QIntValidator()
        validator_i.setRange(3, 1000)
        self.par1_input.setValidator(validator_i)
        self.par2_input.setValidator(validator_i)
        self.par3_input.setValidator(validator_i)
        self.par4_input.setValidator(validator_i)
        self.par1_input.setText("50")
        self.par2_input.setText("150")
        self.par3_input.setText("3")
        self.par4_input.setText("100")
        self.par1_label = QLabel('lower_bound')
        self.par1_label.setAlignment(Qt.AlignCenter)
        self.par1_label.setFont(QFont("Arial", 12))
        self.par2_label = QLabel('upper_bound')
        self.par2_label.setAlignment(Qt.AlignCenter)
        self.par2_label.setFont(QFont("Arial", 12))
        self.par3_label = QLabel('apertureSize')
        self.par3_label.setAlignment(Qt.AlignCenter)
        self.par3_label.setFont(QFont("Arial", 12))
        self.par4_label = QLabel('vote')
        self.par4_label.setAlignment(Qt.AlignCenter)
        self.par4_label.setFont(QFont("Arial", 12))



        self.result_image_label = QLabel(self)
        self.result_image_label.setText("Преобразование")
        self.result_image_label.setAlignment(Qt.AlignCenter)
        self.result_image_label.setFixedSize(int(screen_width * 0.42), int(screen_height * 0.8))
        self.result_image_label.setStyleSheet("background-color: white; border: 2px solid black;")

        self.button = QPushButton('Загрузить изображение', self)
        self.button.setFont(QFont("Arial", 12))
        self.button.clicked.connect(self.loadImage)

        self.ret_button = QPushButton('Сбросить преобр.', self)
        self.ret_button.setFont(QFont("Arial", 12))
        self.ret_button.clicked.connect(self.ret)

        self.canny_button = QPushButton('Применить Canny.', self)
        self.canny_button.setFont(QFont("Arial", 12))
        self.canny_button.clicked.connect(self.apply_canny)

        self.pol_button = QPushButton('Показать полутоновое', self)
        self.pol_button.setFont(QFont("Arial", 12))
        self.pol_button.clicked.connect(self.convertPolutone)


        self.orig_button = QPushButton('Показать исходное', self)
        self.orig_button.setFont(QFont("Arial", 12))
        self.orig_button.clicked.connect(self.Orig)

        self.hough_button = QPushButton('Применить Хафа', self)
        self.hough_button.setFont(QFont("Arial", 12))
        self.hough_button.clicked.connect(self.hough)


        button_layout = QVBoxLayout()
        button_layout.setSpacing(5)
        button_layout.addWidget(self.button)
        button_layout.addWidget(self.ret_button)
        button_layout.addWidget(self.pol_button)
        button_layout.addWidget(self.orig_button)
        button_layout.addWidget(self.par1_input)
        button_layout.addWidget(self.par1_label)
        button_layout.addWidget(self.par2_input)
        button_layout.addWidget(self.par2_label)
        button_layout.addWidget(self.par3_input)
        button_layout.addWidget(self.par3_label)
        button_layout.addWidget(self.canny_button)
        button_layout.addWidget(self.par4_input)
        button_layout.addWidget(self.par4_label)
        button_layout.addWidget(self.hough_button)
        button_layout.addStretch()

        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(button_layout)
        layout.addWidget(self.result_image_label)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.image_path = ""
        self.canny_path = "canny_image.jpg"
        self.lines_path = "lines.jpg"
        self.grayscale_image = None
        #self.result_image_path = "linear_image.jpg"
        self.was_done = False
        self.segmentation_may_be_done = False

    def hough(self):
        pass


    def loadImage(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", "All Files (*);;Image Files (*.jpg *.png *.jpeg)", options=options)
        if file_name:
            pixmap = QPixmap(file_name)
            self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio))
            self.image_path = file_name

    def ret(self):
        if self.image_path:
            self.was_done = False
            pixmap = QPixmap(self.image_path)
            self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio))
        elif not self.image_path:
            er = QMessageBox()
            er.setWindowTitle('Ошибка')
            er.setText('Загрузите изображение')
            er.setIcon(QMessageBox.Warning)
            er.setStandardButtons(QMessageBox.Ok)
            er.exec_()
            return

    def convertPolutone(self):
        if self.image_path:
            image = cv2.imread(self.image_path)
            converted_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_ = cv2.cvtColor(converted_image, cv2.COLOR_GRAY2RGB)
            h, w, ch = gray_.shape
            bytes_per_line = ch * w
            pixmap = QPixmap.fromImage(QImage(gray_.data, w, h, bytes_per_line, QImage.Format_RGB888))
            self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio))
        else:
            er = QMessageBox()
            er.setWindowTitle('Ошибка')
            er.setText('Загрузите изображение')
            er.setIcon(QMessageBox.Warning)
            er.setStandardButtons(QMessageBox.Ok)
            er.exec_()
            return

    def Orig(self):
        if self.image_path:
            pixmap = QPixmap(self.image_path)
            self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio))
        else:
            er = QMessageBox()
            er.setWindowTitle('Ошибка')
            er.setText('Загрузите изображение')
            er.setIcon(QMessageBox.Warning)
            er.setStandardButtons(QMessageBox.Ok)
            er.exec_()
            return
    
    def apply_canny(self):
        gray = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2GRAY)
        self.edges = cv2.Canny(gray, int(self.par1_input.text()), int(self.par2_input.text()), apertureSize=int(self.par3_input.text()))
        cv2.imwrite(self.canny_path, self.edges)
        pixmap = QPixmap(self.canny_path)
        self.result_image_label.setPixmap(pixmap.scaled(self.result_image_label.size(), Qt.KeepAspectRatio))
        self.result_image_label.show()
        self.was_done = True
    
    def hough(self):
        if not self.was_done:
            er = QMessageBox()
            er.setWindowTitle('Ошибка')
            er.setText('Изображение не было преобразовано')
            er.setIcon(QMessageBox.Warning)
            er.setStandardButtons(QMessageBox.Ok)
            er.exec_()
            return
        lines = cv2.HoughLines(self.edges, 1, np.pi / 180, int(self.par4_input.text()))
        self.image_with_lines = cv2.imread(self.image_path)
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
                cv2.line(self.image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(self.lines_path, self.image_with_lines)
        pixmap = QPixmap(self.lines_path)
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio))
        self.label.show()




    def convertToGrayscale(self):
        if self.was_lin:
            path = self.lin_path
        else:
            path = self.image_path
        if path:
            image = QImage(path)
            width = image.width()
            height = image.height()
            self.grayscale_image = QImage(width, height, QImage.Format_Grayscale8)

            threshold = self.threshold_slider.value()

            for x in range(width):
                for y in range(height):
                    color = QColor(image.pixel(x, y))
                    grayscale_value = color.red() * 0.299 + color.green() * 0.587 + color.blue() * 0.114
                    if grayscale_value < threshold:
                        grayscale_value = 0
                    else:
                        grayscale_value = 255
                    self.grayscale_image.setPixel(x, y, QColor(grayscale_value, grayscale_value, grayscale_value).rgb())

            pixmap = QPixmap.fromImage(self.grayscale_image)
            self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio))
            self.segmentation_may_be_done = True
        else:
            er = QMessageBox()
            er.setWindowTitle('Ошибка')
            er.setText('Загрузите изображение')
            er.setIcon(QMessageBox.Warning)
            er.setStandardButtons(QMessageBox.Ok)
            er.exec_()
            return

    def convertToGrayscale2(self):
        try:
            result = eval(self.condition_input.text(),
                          {'r': 0, 'b': 0, 'g': 0, 'br': 0})
        except:
            er = QMessageBox()
            er.setWindowTitle('Ошибка')
            er.setText('Формула некорректна')
            er.setIcon(QMessageBox.Warning)
            er.setStandardButtons(QMessageBox.Ok)
            er.exec_()
            return

        if self.was_lin:
            path = self.lin_path
        else:
            path = self.image_path
        if path:
            image = QImage(path)
            width = image.width()
            height = image.height()
            self.grayscale_image = QImage(width, height, QImage.Format_Grayscale8)

            threshold = self.threshold_slider.value()

            for x in range(width):
                for y in range(height):
                    color = QColor(image.pixel(x, y))
                    grayscale_value = color.red() * 0.299 + color.green() * 0.587 + color.blue() * 0.114
                    r = color.red()
                    b = color.blue()
                    g = color.green()
                    result = eval(self.condition_input.text(), {'r': r, 'b': b, 'g': g, 'br': grayscale_value})
                    if result:
                        grayscale_value = 255
                    else:
                        grayscale_value = 0
                    self.grayscale_image.setPixel(x, y, QColor(grayscale_value, grayscale_value, grayscale_value).rgb())

            pixmap = QPixmap.fromImage(self.grayscale_image)
            self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio))
            self.segmentation_may_be_done = True

        else:
            er = QMessageBox()
            er.setWindowTitle('Ошибка')
            er.setText('Формула некорректна')
            er.setIcon(QMessageBox.Warning)
            er.setStandardButtons(QMessageBox.Ok)
            er.exec_()
            return

    def updateThresholdLabel(self):
        threshold = self.threshold_slider.value()
        self.threshold_label.setText(f'Порог бинаризации: {threshold}')

    def applyDilation(self):
        if self.segmentation_may_be_done == False:
            er = QMessageBox()
            er.setWindowTitle('Ошибка')
            er.setText('Изображение не было бинеризовано')
            er.setIcon(QMessageBox.Warning)
            er.setStandardButtons(QMessageBox.Ok)
            er.exec_()
            return
        kernel_size, ok = QInputDialog.getInt(self, "Размер ядра", "Введите размер ядра для дилатации:", 1, 1, 15)
        if ok:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            # Преобразование QImage в формат Grayscale8
            grayscale_image = self.grayscale_image.convertToFormat(
                QImage.Format_Grayscale8)

            # Получение данных пикселей в виде массива numpy.ndarray
            ptr = grayscale_image.constBits()
            ptr.setsize(grayscale_image.byteCount())
            grayscale_array = np.array(ptr).reshape(grayscale_image.height(),
                                                    grayscale_image.width())

            # Преобразование в uint8
            grayscale_array = grayscale_array.astype(np.uint8)

            # Применение дилатации
            dilated_array = cv2.dilate(grayscale_array, kernel, iterations=1)

            dilated_image = QImage(dilated_array.data, dilated_array.shape[1], dilated_array.shape[0], QImage.Format_Grayscale8)
            self.grayscale_image = dilated_image
            pixmap = QPixmap.fromImage(dilated_image)
            self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio))
            self.label.show()

    def applyOpening(self):
        if self.segmentation_may_be_done == False:
            er = QMessageBox()
            er.setWindowTitle('Ошибка')
            er.setText('Изображение не было бинеризовано')
            er.setIcon(QMessageBox.Warning)
            er.setStandardButtons(QMessageBox.Ok)
            er.exec_()
            return
        kernel_size, ok = QInputDialog.getInt(self, "Размер ядра", "Введите размер ядра для операции открытия:", 1, 1, 15)
        if ok:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            # Преобразование QImage в формат Grayscale8
            grayscale_image = self.grayscale_image.convertToFormat(
                QImage.Format_Grayscale8)

            # Получение данных пикселей в виде массива numpy.ndarray
            ptr = grayscale_image.constBits()
            ptr.setsize(grayscale_image.byteCount())
            grayscale_array = np.array(ptr).reshape(grayscale_image.height(),
                                                    grayscale_image.width())

            # Преобразование в uint8
            grayscale_array = grayscale_array.astype(np.uint8)

            # Применение операции открытия
            opened_array = cv2.morphologyEx(grayscale_array, cv2.MORPH_OPEN, kernel)

            opened_image = QImage(opened_array.data, opened_array.shape[1], opened_array.shape[0], QImage.Format_Grayscale8)
            self.grayscale_image = opened_image
            pixmap = QPixmap.fromImage(opened_image)
            self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio))
            self.label.show()

    def applyClosing(self):
        if self.segmentation_may_be_done == False:
            er = QMessageBox()
            er.setWindowTitle('Ошибка')
            er.setText('Изображение не было бинеризовано')
            er.setIcon(QMessageBox.Warning)
            er.setStandardButtons(QMessageBox.Ok)
            er.exec_()
            return
        kernel_size, ok = QInputDialog.getInt(self, "Размер ядра", "Введите размер ядра для операции закрытия:", 1, 1, 15)
        if ok:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            # Преобразование QImage в формат Grayscale8
            grayscale_image = self.grayscale_image.convertToFormat(
                QImage.Format_Grayscale8)

            # Получение данных пикселей в виде массива numpy.ndarray
            ptr = grayscale_image.constBits()
            ptr.setsize(grayscale_image.byteCount())
            grayscale_array = np.array(ptr).reshape(grayscale_image.height(),
                                                    grayscale_image.width())

            # Преобразование в uint8
            grayscale_array = grayscale_array.astype(np.uint8)

            # Применение операции закрытия
            closed_array = cv2.morphologyEx(grayscale_array, cv2.MORPH_CLOSE, kernel)

            closed_image = QImage(closed_array.data, closed_array.shape[1], closed_array.shape[0], QImage.Format_Grayscale8)
            self.grayscale_image = closed_image
            pixmap = QPixmap.fromImage(closed_image)
            self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio))
            self.label.show()

    def applyErosion(self):
        if self.segmentation_may_be_done == False:
            er = QMessageBox()
            er.setWindowTitle('Ошибка')
            er.setText('Изображение не было бинеризовано')
            er.setIcon(QMessageBox.Warning)
            er.setStandardButtons(QMessageBox.Ok)
            er.exec_()
            return
        kernel_size, ok = QInputDialog.getInt(self, "Размер ядра", "Введите размер ядра для эрозии:", 1, 1, 15)
        if ok:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            # Преобразование QImage в формат Grayscale8
            grayscale_image = self.grayscale_image.convertToFormat(
                QImage.Format_Grayscale8)

            # Получение данных пикселей в виде массива numpy.ndarray
            ptr = grayscale_image.constBits()
            ptr.setsize(grayscale_image.byteCount())
            grayscale_array = np.array(ptr).reshape(grayscale_image.height(),
                                                    grayscale_image.width())

            # Преобразование в uint8
            grayscale_array = grayscale_array.astype(np.uint8)

            # Применение эрозии
            eroded_array = cv2.erode(grayscale_array, kernel, iterations=1)

            eroded_image = QImage(eroded_array.data, eroded_array.shape[1], eroded_array.shape[0], QImage.Format_Grayscale8)
            self.grayscale_image = eroded_image
            pixmap = QPixmap.fromImage(eroded_image)
            self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio))
            self.label.show()

    def runExternalApplication(self):
        if self.segmentation_may_be_done == False:
            er = QMessageBox()
            er.setWindowTitle('Ошибка')
            er.setText('Изображение не было бинеризовано')
            er.setIcon(QMessageBox.Warning)
            er.setStandardButtons(QMessageBox.Ok)
            er.exec_()
            return
        save_path = "to_application.jpg"
        self.grayscale_image.save(save_path)
        image = Image.open("to_application.jpg")
        save_path = "to_application.bmp"
        image = image.convert('1')
        inverted_image = Image.eval(image, lambda px: 255 - px)
        inverted_image.save(save_path, format='BMP')
        subprocess.call(["./MedialRep_Server_par.exe", save_path, "2", self.square_input.text()])
        count_figures = self.print("to_application.txt", "result.jpg")
        self.displayResultImage("result.jpg")
        self.figures_label.setText(f'Количество фигур: {count_figures}')

    def displayResultImage(self, file_path):
        pixmap = QPixmap(file_path)
        self.result_image_label.setPixmap(pixmap.scaled(self.result_image_label.size(), Qt.KeepAspectRatio))
        self.result_image_label.show()

    def print(self, file_path, result):
        answers = []
        count_for_fig = -1
        with open(file_path, 'r') as file:
            i = 0
            count_figures = 0
            for line in file:
                s = line.strip().split()
                if i == 0:
                    count_figures = int(s[3])
                else:
                    if s[0] == "Number" and s[1] == "of" and s[2] == "vertices":
                        count_for_fig = int(s[3])
                        answers.append([])
                    elif count_for_fig > 0:
                        answers[-1].append((int(s[1]), int(s[2])))
                        count_for_fig -= 1
                i += 1

        plt.figure()

        for figure in answers:
            x_coords = [coord[0] for coord in figure]
            y_coords = [coord[1] for coord in figure]
            plt.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], linewidth=0.5)

        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.grid(True)
        plt.axis(False)
        plt.savefig(result)
        plt.close()
        return count_figures

    def info(self):
        instruction_text = """
        В формуле могут использоваться обозначения r, g, b - значения цветовых каналов RGB, br - интенсивность яркости полутонового изображения и знаки логических операций, поддерживаемые языком Python.
        """
        instruction_box = QMessageBox()
        instruction_box.setWindowTitle("Инструкция")
        instruction_box.setText(instruction_text)
        instruction_box.exec_()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageUploader()
    ex.show()
    sys.exit(app.exec_())

