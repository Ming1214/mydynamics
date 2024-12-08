
import imp, sys
import numpy as np
from sklearn.externals import joblib
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QIcon
from Ui import *

Forest = joblib.load("Forest.m")


class MyWin(QMainWindow, Ui_MainWindow):

	def __init__(self):
		super().__init__()
		self.setupUi(self)
		self.setWindowIcon(QIcon("logo.png"))
		self.predict.clicked.connect(self.advice)

	def advice(self):
		x = np.array([[self.kinds.value(), self.processes.value(), self.agvs.value(), self.parts.value()]])
		self.output.setText(Forest.predict(x)[0])


if __name__ == "__main__":
	app = QApplication(sys.argv)
	win = MyWin()
	win.show()
	sys.exit(app.exec())



