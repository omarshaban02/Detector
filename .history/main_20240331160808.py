import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QShortcut
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from home import Ui_MainWindow
import pyqtgraph as pg
from classes import Image, WorkerThread
import cv2

from PyQt5.uic import loadUiType

ui, _ = loadUiType("home.ui")


class Application(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(QMainWindow, self).__init__()
        self.setupUi(self)

        self.scatter_item = pg.ScatterPlotItem(pen="lime", brush = "lime", symbol="x", size=20)
        self.contour_line_item = pg.PlotDataItem(pen ={'color':"r", 'width': 2} )
        self.contour_line_item.setZValue(-1)
        
        self.points = []
        
        self.wgt_contour_input.addItem(self.scatter_item)
        self.wgt_contour_input.addItem(self.contour_line_item)

        self.actionOpen_Image.triggered.connect(self.open_image)
        
        # # Undo Shortcut
        # self.undo_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Control + Qt.Key.Key_Z), self)
        # self.undo_shortcut.activatedAmbiguously.connect(self.undo_last_point)
        
        # List containing all plotwidgets for ease of access
        self.plotwidget_set = [self.wgt_hough_input, self.wgt_hough_edges, self.wgt_hough_output,
                               self.wgt_canny_input, self.wgt_canny_output,
                               self.wgt_contour_input, self.wgt_contour_output]

        # Create an image item for each plot-widget
        self.image_item_set = [self.item_hough_input, self.item_hough_edges, self.item_hough_output,
                               self.item_canny_input, self.item_canny_output,
                               self.item_contour_input, self.item_contour_output
                               ] = [pg.ImageItem() for _ in range(7)]

        # Initializes all plotwidgets with their items
        self.setup_plotwidgets()

        self.btn_start_contour.clicked.connect(self.process_image)
        self.gray_scale_image = None
        self.contour_thread = None
        self.wgt_contour_input.scene().sigMouseClicked.connect(self.on_mouse_click)
         
        self.undo_shortcut = QApplication.instance().installEventFilter(self)
    
    def eventFilter(self, source, event):
        if event.type() == event.KeyPress and event.key() == Qt.Key_Z and QApplication.keyboardModifiers() == Qt.ControlModifier:
            self.undo_last_point()
            return True
        return super().eventFilter(source, event)
        
    def on_mouse_click(self, event):
        modifiers = QApplication.keyboardModifiers()
        
        if event.button() == 1:
            clicked_point = self.wgt_contour_input.plotItem.vb.mapSceneToView(event.scenePos())
            print(f"Mouse clicked at {clicked_point}")

            point_x = clicked_point.x()
            point_y = clicked_point.y()

            self.points.append((point_x, point_y))
            # self.scatter_item.addPoints(x=[ev.scenePos().x()], y=[ev.scenePos().y()])
            self.scatter_item.addPoints(x=[clicked_point.x()], y=[clicked_point.y()])
            self.contour_line_item.setData(x=[p[0] for p in self.points + [self.points[0]]],
                                        y = [p[1] for p in self.points + [self.points[0]]])
            
            if modifiers == Qt.ControlModifier:
                self.clear_points()
    
    def undo_last_point(self):
        
        print("Ctrl Z triggered")
        if self.points:
            self.points.pop()
            
            # Update the scatter item
            self.scatter_item.setData(x=[p[0] for p in self.points],
                                      y = [p[1] for p in self.points])
            
            # Update the line item
            if len(self.points) > 1:
                self.contour_line_item.setData(x=[p[0] for p in self.points + [self.points[0]]]
                                               ,y= [p[1] for p in self.points + [self.points[0]]])
            else:
                self.contour_line_item.clear()
            
        
    
    def clear_points(self):
        self.points = []
        # Clear scatter plot
        self.scatter_item.clear()
        # Clear line plot
        self.contour_line_item.clear()


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
        image_item.setZValue(-2)
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
            plotitem.showGrid(True)

        # Adds the image items to their corresponding plot widgets, so they can be used later to display images
        for plotwidget, imgItem in zip(self.plotwidget_set, self.image_item_set):
            plotwidget.addItem(imgItem)


app = QApplication(sys.argv)
win = Application()  # Change to class name
win.show()
app.exec()