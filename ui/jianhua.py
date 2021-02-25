from PyQt5.QtWidgets import QMainWindow
from os import getcwd
from PyQt5.QtCore import QTimer, QSize
from PyQt5.QtGui import QFont, QMovie


class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.class_name = ['hat', 'person']
        self.image = None

        self.path = getcwd()
        self.timer_camera = QTimer()

        self.setupUi()
        self.retranslateUi()
        self.slot_init()

        self.ft = QFont()
        self.ft.setPointSize(16)

        gif = QMovie('./images_ui/scan.gif')
        gif.setScaledSize(QSize(420, 280 ))

        self.initUI()

    def initUI(self):
        self.setWindowTitle('高铁站房损伤识别系统')

