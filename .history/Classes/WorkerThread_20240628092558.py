from PyQt5.QtCore import QThread, pyqtSignal, QObject
from scipy.signal import convolve2d
import numpy as np


class WorkerSignals(QObject):
    finished = pyqtSignal()
    update = pyqtSignal(np.ndarray)
    calc_area_perimeter = pyqtSignal(float, float)


class WorkerThread(QThread):
    def __init__(self, input_image, contour_points, epochs=100, edges_img=None):
        super(WorkerThread, self).__init__()
        self.signals = WorkerSignals()
        self.input_image = input_image
        self.contour_points = np.array(contour_points).astype(int)
        self.epochs = epochs
        self.edges_img = edges_img
        self.contour = None

    def run(self):
        img = np.array(self.input_image)

        if self.edges_img is None:
            edges = np.array(get_img_edges(img))
        else:
            edges = np.array(self.edges_img)

        edges = add_border(edges)

        self.contour = Contour(self.contour_points)

        # # Add a ton more points
        self.contour.insert_points()
        self.contour.insert_points()

        # Create series of images fitting contour

        for i in range(self.epochs):
            updated_img_con = np.copy(edges)
            img_copy = np.copy(img)

            self.contour.calc_energies(updated_img_con)
            self.contour.update_points()

            self.contour.draw_contour(img_copy)
            self.signals.update.emit(img_copy)

            points = list(zip(self.contour.contour_r, self.contour.contour_c))
            area, perimeter = self.calculate_area(points), self.calculate_perimeter(points)
            self.signals.calc_area_perimeter.emit(area, perimeter)

        self.signals.finished.emit()

    def calculate_perimeter(self, points):
        perimeter = 0
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            perimeter += np.sqrt(dx ** 2 + dy ** 2)
        return perimeter

    def calculate_area(self, points):
        area = 0
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            area += p1[0] * p2[1] - p1[1] * p2[0]
        return abs(area) / 2

