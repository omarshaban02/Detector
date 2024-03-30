import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from home import Ui_MainWindow
import pyqtgraph as pg
from classes import Image, WorkerThread
import cv2

from PyQt5.uic import loadUiType

ui, _ = loadUiType("home.ui")


class Application(QMainWindow, ui):
    def __init__(self):
        super(QMainWindow, self).__init__()
        self.setupUi(self)

        self.scatter_item = pg.ScatterPlotItem(x=(1, 1), y=(1, 1), pen=None, symbol="x", size=20)
        self.wgt_contour_output.addItem(self.scatter_item)

        self.actionOpen_Image.triggered.connect(self.open_image)

        self.plotwidget_set = [self.wgt_hough_input, self.wgt_hough_edges, self.wgt_hough_output,
                               self.wgt_canny_input, self.wgt_canny_output,
                               self.wgt_contour_input, self.wgt_contour_output]

        # Create an image item for each plot-widget
        self.image_item_set = [self.item_hough_input, self.item_hough_edges, self.item_hough_output,
                               self.item_canny_input, self.item_canny_output,
                               self.item_contour_input, self.item_contour_output
                               ] = [pg.ImageItem() for _ in range(7)]

        self.setup_plotwidgets()

        self.btn_start_contour.clicked.connect(self.process_image)
        self.gray_scale_image = None
        self.contour_thread = None

    def update_contour_image(self, image):
        self.display_image(self.item_contour_output, image)

    def processing_finished(self):
        print("Processing Finished")

    def process_image(self):
        self.contour_thread = WorkerThread(self.gray_scale_image)
        self.contour_thread.signals.update.connect(self.update_contour_image)
        self.contour_thread.signals.finished.connect(self.processing_finished)
        self.contour_thread.start()

    # ############################### Misc Functions ################################

    @staticmethod
    def display_image(image_item, image):
        image_item.setImage(image)
        image_item.getViewBox().autoRange()

    def load_img_file(self, image_path):
        # Loads the image using imread, converts it to RGB, then rotates it 90 degrees clockwise
        self.loaded_image = cv2.rotate(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), cv2.ROTATE_90_CLOCKWISE)
        self.img_obj = Image(self.loaded_image)
        self.gray_scale_image = self.img_obj.gray_scale_image
        for item in [self.item_hough_input, self.item_canny_input, self.item_contour_input]:
            self.display_image(item, self.img_obj.gray_scale_image)

    def open_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.bmp *.tif *.jpeg)")
        file_dialog.setWindowTitle("Open Image File")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_() == QFileDialog.Accepted:
            selected_file = file_dialog.selectedFiles()[0]
            self.load_img_file(selected_file)

    def setup_plotwidgets(self):
        for plotwidget in self.findChildren(pg.PlotWidget):
            # Removes Axes and Padding from all plotwidgets intended to display an image
            plotwidget.showAxis('left', False)
            plotwidget.showAxis('bottom', False)
            plotwidget.setBackground((25, 30, 40))
            plotitem = plotwidget.getPlotItem()
            plotitem.getViewBox().setDefaultPadding(0)

        # Adds the image items to their corresponding plot widgets, so they can be used later to display images
        for plotwidget, imgItem in zip(self.plotwidget_set, self.image_item_set):
            plotwidget.addItem(imgItem)


app = QApplication(sys.argv)
win = Application()  # Change to class name
win.show()
app.exec()
