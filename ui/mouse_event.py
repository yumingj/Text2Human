# -*- coding: utf-8 -*-

import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

color_list = [
    QColor(0, 0, 0),
    QColor(255, 250, 250),
    QColor(220, 220, 220),
    QColor(250, 235, 215),
    QColor(255, 250, 205),
    QColor(211, 211, 211),
    QColor(70, 130, 180),
    QColor(127, 255, 212),
    QColor(0, 100, 0),
    QColor(50, 205, 50),
    QColor(255, 255, 0),
    QColor(245, 222, 179),
    QColor(255, 140, 0),
    QColor(255, 0, 0),
    QColor(16, 78, 139),
    QColor(144, 238, 144),
    QColor(50, 205, 174),
    QColor(50, 155, 250),
    QColor(160, 140, 88),
    QColor(213, 140, 88),
    QColor(90, 140, 90),
    QColor(185, 210, 205),
    QColor(130, 165, 180),
    QColor(225, 141, 151)
]


class GraphicsScene(QGraphicsScene):

    def __init__(self, mode, size, parent=None):
        QGraphicsScene.__init__(self, parent)
        self.mode = mode
        self.size = size
        self.mouse_clicked = False
        self.prev_pt = None

        # self.masked_image = None

        # save the points
        self.mask_points = []
        for i in range(len(color_list)):
            self.mask_points.append([])

        # save the size of points
        self.size_points = []
        for i in range(len(color_list)):
            self.size_points.append([])

        # save the history of edit
        self.history = []

    def reset(self):
        # save the points
        self.mask_points = []
        for i in range(len(color_list)):
            self.mask_points.append([])
        # save the size of points
        self.size_points = []
        for i in range(len(color_list)):
            self.size_points.append([])
        # save the history of edit
        self.history = []

        self.mode = 0
        self.prev_pt = None

    def mousePressEvent(self, event):
        self.mouse_clicked = True

    def mouseReleaseEvent(self, event):
        self.prev_pt = None
        self.mouse_clicked = False

    def mouseMoveEvent(self, event):  # drawing
        if self.mouse_clicked:
            if self.prev_pt:
                self.drawMask(self.prev_pt, event.scenePos(),
                              color_list[self.mode], self.size)
                pts = {}
                pts['prev'] = (int(self.prev_pt.x()), int(self.prev_pt.y()))
                pts['curr'] = (int(event.scenePos().x()),
                               int(event.scenePos().y()))

                self.size_points[self.mode].append(self.size)
                self.mask_points[self.mode].append(pts)
                self.history.append(self.mode)
                self.prev_pt = event.scenePos()
            else:
                self.prev_pt = event.scenePos()

    def drawMask(self, prev_pt, curr_pt, color, size):
        lineItem = QGraphicsLineItem(QLineF(prev_pt, curr_pt))
        lineItem.setPen(QPen(color, size, Qt.SolidLine))  # rect
        self.addItem(lineItem)

    def erase_prev_pt(self):
        self.prev_pt = None

    def reset_items(self):
        for i in range(len(self.items())):
            item = self.items()[0]
            self.removeItem(item)

    def undo(self):
        if len(self.items()) > 1:
            if len(self.items()) >= 9:
                for i in range(8):
                    item = self.items()[0]
                    self.removeItem(item)
                    if self.history[-1] == self.mode:
                        self.mask_points[self.mode].pop()
                        self.size_points[self.mode].pop()
                        self.history.pop()
            else:
                for i in range(len(self.items()) - 1):
                    item = self.items()[0]
                    self.removeItem(item)
                    if self.history[-1] == self.mode:
                        self.mask_points[self.mode].pop()
                        self.size_points[self.mode].pop()
                        self.history.pop()
