# author: Kaan Eraslan
# license: see, LICENSE
# No warranties use at your own risk

from pointdetector.ui.models import PointModel, RegionModel
from PySide2.QtWidgets import QGraphicsRectItem, QGraphicsPolygonItem
from PySide2 import QtCore, QtGui


class PointItem(QGraphicsRectItem):
    "Point model -> graphics scene item"

    def __init__(self, pmodel: PointModel,
                 parent=None,
                 colors={
                     "red": QtCore.Qt.red,
                     "black": QtCore.Qt.black,
                     "green": QtCore.Qt.green,
                     "yellow": QtCore.Qt.yellow,
                     "cyan": QtCore.Qt.cyan,
                     "blue": QtCore.Qt.blue,
                     "gray": QtCore.Qt.gray,
                     "magenta": QtCore.Qt.magenta,
                     "white": QtCore.Qt.white
                 }):
        super().__init__(parent)
        self.pmodel = pmodel
        self.colors = colors

    def renderPoint(self):
        "render point"
        x = self.pmodel['x']
        y = self.pmodel['y']
        size = self.pmodel['size']
        color = self.pmodel['color']
        color = self.colors[color]
        colorRgb = color.rgb()
        qcolor = QtGui.QColor(colorRgb)
        brushColor = qcolor.copy().setAlphaF(0.1)
        penColor = qcolor.copy().setAlphaF(1.0)

        brush = QtGui.QBrush(brushColor)
        pen = QtGui.QPen(penColor)
        self.setBrush(brush)
        self.setPen(pen)
        self.setRect(x, y, size, size)


class RegionItem(QGraphicsPolygonItem):
    "Region model -> polygon item"

    def __init__(self, rmodel: RegionModel,
                 parent=None,
                 colors={
                     "red": QtCore.Qt.red,
                     "black": QtCore.Qt.black,
                     "green": QtCore.Qt.green,
                     "yellow": QtCore.Qt.yellow,
                     "cyan": QtCore.Qt.cyan,
                     "blue": QtCore.Qt.blue,
                     "gray": QtCore.Qt.gray,
                     "magenta": QtCore.Qt.magenta,
                     "white": QtCore.Qt.white
                 }):
        super().__init__(parent)
        self.rmodel = rmodel
        self.colors = colors
        self.isClosed = None

    def popPoint(self, point: QtCore.QPointF):
        "add new point to region"
        pdict = self.convertPoint2PointDict(point)
        points = self.rmodel['points']
        newlist = [p for p in points if p != pdict]
        self.rmodel['points'] = newlist
        self.renderRegion()

    def popPointByIndex(self, index):
        "pop point by index"
        self.rmodel['points'].pop(index)
        self.renderRegion()

    def popPointsByIndices(self, indices: [int]):
        "pop points using indices"
        maxind = max(indices)
        minind = min(indices)
        assert minind >= 0
        assert maxind < len(self.rmodel['points'])
        for i in indices:
            self.rmodel['points'].pop(i)
        #
        self.renderRegion()

    def insertPointDict(self, pdict: {}, index=-1):
        "Inser point dict to model ensuring all points are unique"
        points = self.rmodel['points']
        pointorder = [tuple(p['x'], p['y']) for p in points]
        newkey = tuple(pdict['x'], pdict['y'])
        if newkey in pointorder:
            return
        if index == -1:
            self.rmodel['points'].append(pdict)
        else:
            assert index <= len(points)
            self.rmodel['points'].insert(index, pdict)

    def popPoints(self, plist: [QtCore.QPointF]):
        "pop points from region"
        newlist = [self.convertPoint2PointDict(p) for p in plist]
        points = self.rmodel['points']
        points = [p for p in points if p not in newlist]
        self.rmodel['points'] = points
        self.renderRegion()

    def convertPoint2PointDict(self, point: QtCore.QPointF):
        "convert point to point dict"
        x = int(point.x())
        y = int(point.y())
        return {"x": x, "y": y}

    def insertPoint2Region(self,
                           point: QtCore.QPointF,
                           index: int):
        "inser point to region"
        pdict = self.convertPoint2PointDict(point)
        self.insertPointDict(pdict, index)

        self.renderRegion()

    def appendPoint(self, point: QtCore.QPointF):
        "append point to region"
        pdict = self.convertPoint2PointDict(point)
        self.insertPointDict(pdict)
        self.renderRegion()

    def appendPoints(self, plist: [QtCore.QPointF]):
        "append points to region"
        for point in plist:
            pdict = self.convertPoint2PointDict(point)
            self.insertPointDict(pdict)
        self.renderRegion()

    def renderRegion(self):
        "render region"
        points = self.rmodel['points']
        polygon_f = QtGui.QPolygonF()
        for point in points:
            px = point['x']
            py = point['y']
            p_f = QtCore.QPointF()
            p_f.setX(float(px))
            p_f.setY(float(py))
            polygon_f.append(p_f)
        #
        color = self.rmodel['color']
        color = self.colors[color]
        colorRgb = color.rgb()
        qcolor = QtGui.QColor(colorRgb)
        brushColor = qcolor.copy().setAlphaF(0.1)
        penColor = qcolor.copy().setAlphaF(1.0)

        brush = QtGui.QBrush(brushColor)
        pen = QtGui.QPen(penColor)
        self.setBrush(brush)
        self.setPen(pen)
        self.isClosed = polygon_f.isClosed()

        self.setPolygon(polygon_f)
