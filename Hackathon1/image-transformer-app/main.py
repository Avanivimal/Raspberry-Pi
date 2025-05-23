import sys
from PyQt5 import QtWidgets
from gui import Ui_MainWindow
from logic import ImageProcessor

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    with open("style.qss", "r") as f:
        style = f.read()
        app.setStyleSheet(style)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)

    processor = ImageProcessor(ui)

    MainWindow.show()
    sys.exit(app.exec_())
