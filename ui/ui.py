from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class Ui_Form(object):

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1250, 670)

        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(20, 60, 97, 27))
        self.pushButton_2.setObjectName("pushButton_2")

        self.pushButton_6 = QtWidgets.QPushButton(Form)
        self.pushButton_6.setGeometry(QtCore.QRect(20, 100, 97, 27))
        self.pushButton_6.setObjectName("pushButton_6")

        # Generate Parsing
        self.pushButton_0 = QtWidgets.QPushButton(Form)
        self.pushButton_0.setGeometry(QtCore.QRect(126, 60, 150, 27))
        self.pushButton_0.setObjectName("pushButton_0")

        # Generate Human
        self.pushButton_1 = QtWidgets.QPushButton(Form)
        self.pushButton_1.setGeometry(QtCore.QRect(126, 100, 150, 27))
        self.pushButton_1.setObjectName("pushButton_1")

        # shape text box
        self.label_heading_1 = QtWidgets.QLabel(Form)
        self.label_heading_1.setText('Describe the shape.')
        self.label_heading_1.setObjectName("label_heading_1")
        self.label_heading_1.setGeometry(QtCore.QRect(320, 20, 200, 20))

        self.message_box_1 = QtWidgets.QLineEdit(Form)
        self.message_box_1.setGeometry(QtCore.QRect(320, 50, 256, 80))
        self.message_box_1.setObjectName("message_box_1")
        self.message_box_1.setAlignment(Qt.AlignTop)

        # texture text box
        self.label_heading_2 = QtWidgets.QLabel(Form)
        self.label_heading_2.setText('Describe the textures.')
        self.label_heading_2.setObjectName("label_heading_2")
        self.label_heading_2.setGeometry(QtCore.QRect(620, 20, 200, 20))

        self.message_box_2 = QtWidgets.QLineEdit(Form)
        self.message_box_2.setGeometry(QtCore.QRect(620, 50, 256, 80))
        self.message_box_2.setObjectName("message_box_2")
        self.message_box_2.setAlignment(Qt.AlignTop)

        # title icon
        self.title_icon = QtWidgets.QLabel(Form)
        self.title_icon.setGeometry(QtCore.QRect(30, 10, 200, 50))
        self.title_icon.setPixmap(
            QtGui.QPixmap('./ui/icons/icon_title.png').scaledToWidth(200))

        # palette icon
        self.palette_icon = QtWidgets.QLabel(Form)
        self.palette_icon.setGeometry(QtCore.QRect(950, 10, 256, 128))
        self.palette_icon.setPixmap(
            QtGui.QPixmap('./ui/icons/icon_palette.png').scaledToWidth(256))

        # top
        self.pushButton_8 = QtWidgets.QPushButton('   top', Form)
        self.pushButton_8.setGeometry(QtCore.QRect(940, 120, 120, 27))
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_8.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_8.setIcon(QIcon('./ui/color_blocks/class_top.png'))
        # skin
        self.pushButton_9 = QtWidgets.QPushButton('   skin', Form)
        self.pushButton_9.setGeometry(QtCore.QRect(940, 165, 120, 27))
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_9.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_9.setIcon(QIcon('./ui/color_blocks/class_skin.png'))
        # outer
        self.pushButton_10 = QtWidgets.QPushButton('   outer', Form)
        self.pushButton_10.setGeometry(QtCore.QRect(940, 210, 120, 27))
        self.pushButton_10.setObjectName("pushButton_10")
        self.pushButton_10.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_10.setIcon(QIcon('./ui/color_blocks/class_outer.png'))
        # face
        self.pushButton_11 = QtWidgets.QPushButton('   face', Form)
        self.pushButton_11.setGeometry(QtCore.QRect(940, 255, 120, 27))
        self.pushButton_11.setObjectName("pushButton_11")
        self.pushButton_11.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_11.setIcon(QIcon('./ui/color_blocks/class_face.png'))
        # skirt
        self.pushButton_12 = QtWidgets.QPushButton('   skirt', Form)
        self.pushButton_12.setGeometry(QtCore.QRect(940, 300, 120, 27))
        self.pushButton_12.setObjectName("pushButton_12")
        self.pushButton_12.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_12.setIcon(QIcon('./ui/color_blocks/class_skirt.png'))
        # hair
        self.pushButton_13 = QtWidgets.QPushButton('   hair', Form)
        self.pushButton_13.setGeometry(QtCore.QRect(940, 345, 120, 27))
        self.pushButton_13.setObjectName("pushButton_13")
        self.pushButton_13.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_13.setIcon(QIcon('./ui/color_blocks/class_hair.png'))
        # dress
        self.pushButton_14 = QtWidgets.QPushButton('   dress', Form)
        self.pushButton_14.setGeometry(QtCore.QRect(940, 390, 120, 27))
        self.pushButton_14.setObjectName("pushButton_14")
        self.pushButton_14.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_14.setIcon(QIcon('./ui/color_blocks/class_dress.png'))
        # headwear
        self.pushButton_15 = QtWidgets.QPushButton('   headwear', Form)
        self.pushButton_15.setGeometry(QtCore.QRect(940, 435, 120, 27))
        self.pushButton_15.setObjectName("pushButton_15")
        self.pushButton_15.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_15.setIcon(
            QIcon('./ui/color_blocks/class_headwear.png'))
        # pants
        self.pushButton_16 = QtWidgets.QPushButton('   pants', Form)
        self.pushButton_16.setGeometry(QtCore.QRect(940, 480, 120, 27))
        self.pushButton_16.setObjectName("pushButton_16")
        self.pushButton_16.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_16.setIcon(QIcon('./ui/color_blocks/class_pants.png'))
        # eyeglasses
        self.pushButton_17 = QtWidgets.QPushButton('   eyeglass', Form)
        self.pushButton_17.setGeometry(QtCore.QRect(940, 525, 120, 27))
        self.pushButton_17.setObjectName("pushButton_17")
        self.pushButton_17.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_17.setIcon(
            QIcon('./ui/color_blocks/class_eyeglass.png'))
        # rompers
        self.pushButton_18 = QtWidgets.QPushButton('   rompers', Form)
        self.pushButton_18.setGeometry(QtCore.QRect(940, 570, 120, 27))
        self.pushButton_18.setObjectName("pushButton_18")
        self.pushButton_18.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_18.setIcon(
            QIcon('./ui/color_blocks/class_rompers.png'))
        # footwear
        self.pushButton_19 = QtWidgets.QPushButton('   footwear', Form)
        self.pushButton_19.setGeometry(QtCore.QRect(940, 615, 120, 27))
        self.pushButton_19.setObjectName("pushButton_19")
        self.pushButton_19.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_19.setIcon(
            QIcon('./ui/color_blocks/class_footwear.png'))

        # leggings
        self.pushButton_20 = QtWidgets.QPushButton('   leggings', Form)
        self.pushButton_20.setGeometry(QtCore.QRect(1100, 120, 120, 27))
        self.pushButton_20.setObjectName("pushButton_10")
        self.pushButton_20.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_20.setIcon(
            QIcon('./ui/color_blocks/class_leggings.png'))

        # ring
        self.pushButton_21 = QtWidgets.QPushButton('   ring', Form)
        self.pushButton_21.setGeometry(QtCore.QRect(1100, 165, 120, 27))
        self.pushButton_21.setObjectName("pushButton_2`0`")
        self.pushButton_21.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_21.setIcon(QIcon('./ui/color_blocks/class_ring.png'))

        # belt
        self.pushButton_22 = QtWidgets.QPushButton('   belt', Form)
        self.pushButton_22.setGeometry(QtCore.QRect(1100, 210, 120, 27))
        self.pushButton_22.setObjectName("pushButton_2`0`")
        self.pushButton_22.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_22.setIcon(QIcon('./ui/color_blocks/class_belt.png'))

        # neckwear
        self.pushButton_23 = QtWidgets.QPushButton('   neckwear', Form)
        self.pushButton_23.setGeometry(QtCore.QRect(1100, 255, 120, 27))
        self.pushButton_23.setObjectName("pushButton_2`0`")
        self.pushButton_23.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_23.setIcon(
            QIcon('./ui/color_blocks/class_neckwear.png'))

        # wrist
        self.pushButton_24 = QtWidgets.QPushButton('   wrist', Form)
        self.pushButton_24.setGeometry(QtCore.QRect(1100, 300, 120, 27))
        self.pushButton_24.setObjectName("pushButton_2`0`")
        self.pushButton_24.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_24.setIcon(QIcon('./ui/color_blocks/class_wrist.png'))

        # socks
        self.pushButton_25 = QtWidgets.QPushButton('   socks', Form)
        self.pushButton_25.setGeometry(QtCore.QRect(1100, 345, 120, 27))
        self.pushButton_25.setObjectName("pushButton_2`0`")
        self.pushButton_25.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_25.setIcon(QIcon('./ui/color_blocks/class_socks.png'))

        # tie
        self.pushButton_26 = QtWidgets.QPushButton('   tie', Form)
        self.pushButton_26.setGeometry(QtCore.QRect(1100, 390, 120, 27))
        self.pushButton_26.setObjectName("pushButton_2`0`")
        self.pushButton_26.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_26.setIcon(QIcon('./ui/color_blocks/class_tie.png'))

        # earstuds
        self.pushButton_27 = QtWidgets.QPushButton('   necklace', Form)
        self.pushButton_27.setGeometry(QtCore.QRect(1100, 435, 120, 27))
        self.pushButton_27.setObjectName("pushButton_2`0`")
        self.pushButton_27.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_27.setIcon(
            QIcon('./ui/color_blocks/class_necklace.png'))

        # necklace
        self.pushButton_28 = QtWidgets.QPushButton('   earstuds', Form)
        self.pushButton_28.setGeometry(QtCore.QRect(1100, 480, 120, 27))
        self.pushButton_28.setObjectName("pushButton_2`0`")
        self.pushButton_28.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_28.setIcon(
            QIcon('./ui/color_blocks/class_earstuds.png'))

        # bag
        self.pushButton_29 = QtWidgets.QPushButton('   bag', Form)
        self.pushButton_29.setGeometry(QtCore.QRect(1100, 525, 120, 27))
        self.pushButton_29.setObjectName("pushButton_2`0`")
        self.pushButton_29.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_29.setIcon(QIcon('./ui/color_blocks/class_bag.png'))

        # glove
        self.pushButton_30 = QtWidgets.QPushButton('   glove', Form)
        self.pushButton_30.setGeometry(QtCore.QRect(1100, 570, 120, 27))
        self.pushButton_30.setObjectName("pushButton_2`0`")
        self.pushButton_30.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_30.setIcon(QIcon('./ui/color_blocks/class_glove.png'))

        # background
        self.pushButton_31 = QtWidgets.QPushButton('   background', Form)
        self.pushButton_31.setGeometry(QtCore.QRect(1100, 615, 120, 27))
        self.pushButton_31.setObjectName("pushButton_2`0`")
        self.pushButton_31.setStyleSheet(
            "text-align: left; padding-left: 10px;")
        self.pushButton_31.setIcon(QIcon('./ui/color_blocks/class_bg.png'))

        self.graphicsView = QtWidgets.QGraphicsView(Form)
        self.graphicsView.setGeometry(QtCore.QRect(20, 140, 256, 512))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = QtWidgets.QGraphicsView(Form)
        self.graphicsView_2.setGeometry(QtCore.QRect(320, 140, 256, 512))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.graphicsView_3 = QtWidgets.QGraphicsView(Form)
        self.graphicsView_3.setGeometry(QtCore.QRect(620, 140, 256, 512))
        self.graphicsView_3.setObjectName("graphicsView_3")

        self.retranslateUi(Form)
        self.pushButton_2.clicked.connect(Form.open_densepose)
        self.pushButton_6.clicked.connect(Form.save_img)
        self.pushButton_8.clicked.connect(Form.top_mode)
        self.pushButton_9.clicked.connect(Form.skin_mode)
        self.pushButton_10.clicked.connect(Form.outer_mode)
        self.pushButton_11.clicked.connect(Form.face_mode)
        self.pushButton_12.clicked.connect(Form.skirt_mode)
        self.pushButton_13.clicked.connect(Form.hair_mode)
        self.pushButton_14.clicked.connect(Form.dress_mode)
        self.pushButton_15.clicked.connect(Form.headwear_mode)
        self.pushButton_16.clicked.connect(Form.pants_mode)
        self.pushButton_17.clicked.connect(Form.eyeglass_mode)
        self.pushButton_18.clicked.connect(Form.rompers_mode)
        self.pushButton_19.clicked.connect(Form.footwear_mode)
        self.pushButton_20.clicked.connect(Form.leggings_mode)
        self.pushButton_21.clicked.connect(Form.ring_mode)
        self.pushButton_22.clicked.connect(Form.belt_mode)
        self.pushButton_23.clicked.connect(Form.neckwear_mode)
        self.pushButton_24.clicked.connect(Form.wrist_mode)
        self.pushButton_25.clicked.connect(Form.socks_mode)
        self.pushButton_26.clicked.connect(Form.tie_mode)
        self.pushButton_27.clicked.connect(Form.earstuds_mode)
        self.pushButton_28.clicked.connect(Form.necklace_mode)
        self.pushButton_29.clicked.connect(Form.bag_mode)
        self.pushButton_30.clicked.connect(Form.glove_mode)
        self.pushButton_31.clicked.connect(Form.background_mode)
        self.pushButton_0.clicked.connect(Form.generate_parsing)
        self.pushButton_1.clicked.connect(Form.generate_human)

        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Text2Human"))
        self.pushButton_2.setText(_translate("Form", "Load Pose"))
        self.pushButton_6.setText(_translate("Form", "Save Image"))

        self.pushButton_0.setText(_translate("Form", "Generate Parsing"))
        self.pushButton_1.setText(_translate("Form", "Generate Human"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
