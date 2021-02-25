from sys import argv, exit
from PyQt5.QtWidgets import QApplication, QMainWindow
from ui.hat_recognization import Ui_MainWindow

if __name__ == '__main__':
    app = QApplication(argv)

    window = QMainWindow()
    ui = Ui_MainWindow(window)

    window.show()
    exit(app.exec_())
