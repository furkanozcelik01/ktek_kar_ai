# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainIoQKly.ui'
##
## Created by: Qt User Interface Compiler version 6.5.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QFrame, QLabel, QMainWindow,
    QPushButton, QSizePolicy, QTextEdit, QWidget)


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1210, 671)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.arkaplan = QLabel(self.centralwidget)
        self.arkaplan.setObjectName(u"arkaplan")
        self.arkaplan.setGeometry(QRect(-10, 0, 1221, 671))
        sizePolicy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.arkaplan.sizePolicy().hasHeightForWidth())
        self.arkaplan.setSizePolicy(sizePolicy)
        self.arkaplan.setTabletTracking(False)
        self.arkaplan.setAcceptDrops(False)
        self.arkaplan.setLayoutDirection(Qt.LeftToRight)
        self.arkaplan.setAutoFillBackground(True)
        self.arkaplan.setStyleSheet(u"")
        self.arkaplan.setFrameShape(QFrame.NoFrame)
        self.arkaplan.setFrameShadow(QFrame.Raised)
        self.arkaplan.setTextFormat(Qt.AutoText)
        self.arkaplan.setPixmap(QPixmap(u":/res/res/images/arkaplan.png"))
        self.arkaplan.setScaledContents(True)
        self.arkaplan.setWordWrap(True)
        self.arkaplan.setOpenExternalLinks(True)
        self.ekran = QFrame(self.centralwidget)
        self.ekran.setObjectName(u"ekran")
        self.ekran.setGeometry(QRect(670, 130, 411, 241))
        self.ekran.setFrameShape(QFrame.StyledPanel)
        self.ekran.setFrameShadow(QFrame.Raised)
        self.hiz_label_deger = QTextEdit(self.centralwidget)
        self.hiz_label_deger.setObjectName(u"hiz_label_deger")
        self.hiz_label_deger.setGeometry(QRect(290, 180, 111, 31))
        self.hiz_label = QLabel(self.centralwidget)
        self.hiz_label.setObjectName(u"hiz_label")
        self.hiz_label.setGeometry(QRect(130, 180, 141, 31))
        self.hiz_label.setStyleSheet(u"font: 700 28pt \"Segoe UI\";")
        self.durum_label = QLabel(self.centralwidget)
        self.durum_label.setObjectName(u"durum_label")
        self.durum_label.setGeometry(QRect(130, 260, 141, 31))
        self.durum_label.setStyleSheet(u"font: 700 28pt \"Segoe UI\";")
        self.skor_label = QLabel(self.centralwidget)
        self.skor_label.setObjectName(u"skor_label")
        self.skor_label.setGeometry(QRect(130, 220, 141, 31))
        self.skor_label.setStyleSheet(u"font: 700 28pt \"Segoe UI\";")
        self.skor_label_deger = QLabel(self.centralwidget)
        self.skor_label_deger.setObjectName(u"skor_label_deger")
        self.skor_label_deger.setGeometry(QRect(290, 220, 351, 31))
        self.skor_label_deger.setStyleSheet(u"font: 26pt \"Segoe UI\";")
        self.durum_label_deger = QLabel(self.centralwidget)
        self.durum_label_deger.setObjectName(u"durum_label_deger")
        self.durum_label_deger.setGeometry(QRect(290, 260, 351, 31))
        self.durum_label_deger.setStyleSheet(u"font: 26pt \"Segoe UI\";")
        self.info_icon = QLabel(self.centralwidget)
        self.info_icon.setObjectName(u"info_icon")
        self.info_icon.setGeometry(QRect(140, 320, 61, 61))
        self.info_icon.setStyleSheet(u"")
        self.info_icon.setPixmap(QPixmap(u":/res/res/images/star.png"))
        self.info_icon.setScaledContents(True)
        self.pushButton = QPushButton(self.centralwidget)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(770, 410, 191, 161))
        self.pushButton.setStyleSheet(u"background-image: url(:/res/res/images/logo.png);")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.arkaplan.setText("")
        self.hiz_label.setText(QCoreApplication.translate("MainWindow", u"HIZ:", None))
        self.durum_label.setText(QCoreApplication.translate("MainWindow", u"DURUM:", None))
        self.skor_label.setText(QCoreApplication.translate("MainWindow", u"SKOR:", None))
        self.skor_label_deger.setText(QCoreApplication.translate("MainWindow", u"-", None))
        self.durum_label_deger.setText(QCoreApplication.translate("MainWindow", u"-", None))
        self.info_icon.setText("")
        self.pushButton.setText("")
    # retranslateUi

