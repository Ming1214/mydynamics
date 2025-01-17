# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(281, 260)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily(" 汉仪良品线简")
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.kinds = QtWidgets.QSpinBox(self.centralwidget)
        self.kinds.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.kinds.setMinimum(1)
        self.kinds.setMaximum(20)
        self.kinds.setObjectName("kinds")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.kinds)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily(" 汉仪良品线简")
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.processes = QtWidgets.QSpinBox(self.centralwidget)
        self.processes.setMinimum(1)
        self.processes.setMaximum(20)
        self.processes.setObjectName("processes")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.processes)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily(" 汉仪良品线简")
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.agvs = QtWidgets.QSpinBox(self.centralwidget)
        self.agvs.setMinimum(1)
        self.agvs.setMaximum(20)
        self.agvs.setProperty("value", 1)
        self.agvs.setObjectName("agvs")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.agvs)
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily(" 汉仪良品线简")
        font.setPointSize(14)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.parts = QtWidgets.QSpinBox(self.centralwidget)
        self.parts.setMinimum(1)
        self.parts.setMaximum(200)
        self.parts.setObjectName("parts")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.parts)
        self.verticalLayout.addLayout(self.formLayout)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.predict = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily(" 汉仪良品线简")
        font.setPointSize(14)
        self.predict.setFont(font)
        self.predict.setObjectName("predict")
        self.verticalLayout.addWidget(self.predict)
        self.output = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily(" 汉仪良品线简")
        font.setPointSize(14)
        self.output.setFont(font)
        self.output.setAlignment(QtCore.Qt.AlignCenter)
        self.output.setReadOnly(True)
        self.output.setObjectName("output")
        self.verticalLayout.addWidget(self.output)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 281, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "调度算法推荐"))
        self.label.setText(_translate("MainWindow", "零件种类"))
        self.label_2.setText(_translate("MainWindow", "工艺数目"))
        self.label_3.setText(_translate("MainWindow", "小车数量"))
        self.label_4.setText(_translate("MainWindow", "零件数量"))
        self.predict.setText(_translate("MainWindow", "推    荐"))

