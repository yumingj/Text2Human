import sys

import cv2
import numpy as np
import torch
from PIL import Image
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from models.sample_model import SampleFromPoseModel
from ui.mouse_event import GraphicsScene
from ui.ui import Ui_Form
from utils.language_utils import (generate_shape_attributes,
                                  generate_texture_attributes)
from utils.options import dict_to_nonedict, parse

color_list = [(0, 0, 0), (255, 250, 250), (220, 220, 220), (250, 235, 215),
              (255, 250, 205), (211, 211, 211), (70, 130, 180),
              (127, 255, 212), (0, 100, 0), (50, 205, 50), (255, 255, 0),
              (245, 222, 179), (255, 140, 0), (255, 0, 0), (16, 78, 139),
              (144, 238, 144), (50, 205, 174), (50, 155, 250), (160, 140, 88),
              (213, 140, 88), (90, 140, 90), (185, 210, 205), (130, 165, 180),
              (225, 141, 151)]


class Ex(QWidget, Ui_Form):

    def __init__(self, opt):
        super(Ex, self).__init__()
        self.setupUi(self)
        self.show()

        self.output_img = None

        self.mat_img = None

        self.mode = 0
        self.size = 6
        self.mask = None
        self.mask_m = None
        self.img = None

        # about UI
        self.mouse_clicked = False
        self.scene = QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.ref_scene = GraphicsScene(self.mode, self.size)
        self.graphicsView_2.setScene(self.ref_scene)
        self.graphicsView_2.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.result_scene = QGraphicsScene()
        self.graphicsView_3.setScene(self.result_scene)
        self.graphicsView_3.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.graphicsView_3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphicsView_3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.dlg = QColorDialog(self.graphicsView)
        self.color = None

        self.sample_model = SampleFromPoseModel(opt)

    def open_densepose(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open File",
                                                  QDir.currentPath())
        if fileName:
            image = QPixmap(fileName)
            mat_img = Image.open(fileName)
            self.pose_img = mat_img.copy()
            if image.isNull():
                QMessageBox.information(self, "Image Viewer",
                                        "Cannot load %s." % fileName)
                return
            image = image.scaled(self.graphicsView.size(),
                                 Qt.IgnoreAspectRatio)

            if len(self.scene.items()) > 0:
                self.scene.removeItem(self.scene.items()[-1])
            self.scene.addPixmap(image)

            self.ref_scene.clear()
            self.result_scene.clear()

            # load pose to model
            self.pose_img = np.array(
                self.pose_img.resize(
                    size=(256, 512),
                    resample=Image.LANCZOS))[:, :, 2:].transpose(
                        2, 0, 1).astype(np.float32)
            self.pose_img = self.pose_img / 12. - 1

            self.pose_img = torch.from_numpy(self.pose_img).unsqueeze(1)

            self.sample_model.feed_pose_data(self.pose_img)

    def generate_parsing(self):
        self.ref_scene.reset_items()
        self.ref_scene.reset()

        shape_texts = self.message_box_1.text()

        shape_attributes = generate_shape_attributes(shape_texts)
        shape_attributes = torch.LongTensor(shape_attributes).unsqueeze(0)
        self.sample_model.feed_shape_attributes(shape_attributes)

        self.sample_model.generate_parsing_map()
        self.sample_model.generate_quantized_segm()

        self.colored_segm = self.sample_model.palette_result(
            self.sample_model.segm[0].cpu())

        self.mask_m = cv2.cvtColor(
            cv2.cvtColor(self.colored_segm, cv2.COLOR_RGB2BGR),
            cv2.COLOR_BGR2RGB)

        qim = QImage(self.colored_segm.data.tobytes(),
                     self.colored_segm.shape[1], self.colored_segm.shape[0],
                     QImage.Format_RGB888)

        image = QPixmap.fromImage(qim)

        image = image.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)

        if len(self.ref_scene.items()) > 0:
            self.ref_scene.removeItem(self.ref_scene.items()[-1])
        self.ref_scene.addPixmap(image)

        self.result_scene.clear()

    def generate_human(self):
        for i in range(24):
            self.mask_m = self.make_mask(self.mask_m,
                                         self.ref_scene.mask_points[i],
                                         self.ref_scene.size_points[i],
                                         color_list[i])

        seg_map = np.full(self.mask_m.shape[:-1], -1)

        # convert rgb to num
        for index, color in enumerate(color_list):
            seg_map[np.sum(self.mask_m == color, axis=2) == 3] = index
        assert (seg_map != -1).all()

        self.sample_model.segm = torch.from_numpy(seg_map).unsqueeze(
            0).unsqueeze(0).to(self.sample_model.device)
        self.sample_model.generate_quantized_segm()

        texture_texts = self.message_box_2.text()
        texture_attributes = generate_texture_attributes(texture_texts)

        texture_attributes = torch.LongTensor(texture_attributes)

        self.sample_model.feed_texture_attributes(texture_attributes)

        self.sample_model.generate_texture_map()
        result = self.sample_model.sample_and_refine()
        result = result.permute(0, 2, 3, 1)
        result = result.detach().cpu().numpy()
        result = result * 255

        result = np.asarray(result[0, :, :, :], dtype=np.uint8)

        self.output_img = result

        qim = QImage(result.data.tobytes(), result.shape[1], result.shape[0],
                     QImage.Format_RGB888)
        image = QPixmap.fromImage(qim)

        image = image.scaled(self.graphicsView.size(), Qt.IgnoreAspectRatio)

        if len(self.result_scene.items()) > 0:
            self.result_scene.removeItem(self.result_scene.items()[-1])
        self.result_scene.addPixmap(image)

    def top_mode(self):
        self.ref_scene.mode = 1

    def skin_mode(self):
        self.ref_scene.mode = 15

    def outer_mode(self):
        self.ref_scene.mode = 2

    def face_mode(self):
        self.ref_scene.mode = 14

    def skirt_mode(self):
        self.ref_scene.mode = 3

    def hair_mode(self):
        self.ref_scene.mode = 13

    def dress_mode(self):
        self.ref_scene.mode = 4

    def headwear_mode(self):
        self.ref_scene.mode = 7

    def pants_mode(self):
        self.ref_scene.mode = 5

    def eyeglass_mode(self):
        self.ref_scene.mode = 8

    def rompers_mode(self):
        self.ref_scene.mode = 21

    def footwear_mode(self):
        self.ref_scene.mode = 11

    def leggings_mode(self):
        self.ref_scene.mode = 6

    def ring_mode(self):
        self.ref_scene.mode = 16

    def belt_mode(self):
        self.ref_scene.mode = 10

    def neckwear_mode(self):
        self.ref_scene.mode = 9

    def wrist_mode(self):
        self.ref_scene.mode = 17

    def socks_mode(self):
        self.ref_scene.mode = 18

    def tie_mode(self):
        self.ref_scene.mode = 23

    def earstuds_mode(self):
        self.ref_scene.mode = 22

    def necklace_mode(self):
        self.ref_scene.mode = 20

    def bag_mode(self):
        self.ref_scene.mode = 12

    def glove_mode(self):
        self.ref_scene.mode = 19

    def background_mode(self):
        self.ref_scene.mode = 0

    def make_mask(self, mask, pts, sizes, color):
        if len(pts) > 0:
            for idx, pt in enumerate(pts):
                cv2.line(mask, pt['prev'], pt['curr'], color, sizes[idx])
        return mask

    def save_img(self):
        if type(self.output_img):
            fileName, _ = QFileDialog.getSaveFileName(self, "Save File",
                                                      QDir.currentPath())
            cv2.imwrite(fileName + '.png', self.output_img[:, :, ::-1])

    def undo(self):
        self.scene.undo()

    def clear(self):

        self.ref_scene.reset_items()
        self.ref_scene.reset()

        self.ref_scene.clear()

        self.result_scene.clear()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    opt = './configs/sample_from_pose.yml'
    opt = parse(opt, is_train=False)
    opt = dict_to_nonedict(opt)
    ex = Ex(opt)
    sys.exit(app.exec_())
