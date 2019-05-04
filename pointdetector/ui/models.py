# author: Kaan Eraslan
# license: see, LICENSE
# No warranties use at your own risk

from pointdetector.seammarker.seammarker.utils import assertCond

class PointModel:
    "Points data"

    def __init__(self, point: {}):
        self.data = {}
        pointData = {
            "coordinates": (0, 0),
            "x": 0,
            "y": 0,
            "size": 0,
            "direction": 0.0,
            "threshold": 0,
            "color": "red"
        }

    def setX(self, xval: int):
        assertCond(xval, isinstance(xval, int))
        self.data['x'] = xval

    def setY(self, yval: int):
        assertCond(yval, isinstance(yval, int))
        self.data['y'] = yval

    def setCoordinates(self, xy: dict):
        assertCond(xy, isinstance(xy, dict))
        x = xy['x']
        y = xy['y']
        self.data['coordinates'] = (x, y)
        self.setX(x)
        self.setY(y)

    def setSize(self, sizeval: int):
        assertCond(sizeval, isinstance(sizeval, int))
        self.data['size'] = sizeval

    def setThreshold(self, thresh: int):
        assertCond(thresh, isinstance(thresh, int))
        self.data['threshold'] = thresh

    def setDirection(self, direction: float):
        assertCond(direction, isinstance(direction, float))
        self.data['direction'] = direction

    def setColor(self, color: str):
        assertCond(color, isinstance(color, str))
        self.data['color'] = color

    def setData(self, data: dict):
        assertCond(data, isinstance(data, dict))
        assert "color" in data
        assert "size" in data
        assert "x" in data
        assert "y" in data
        assert "direction" in data
        assert "threshold" in data
        self.setSize(data['size'])
        self.setColor(data['color'])
        self.setDirection(data['direction'])
        self.setThreshold(data['threshold'])
        self.setCoordinates(xy={'x': data['x'], 'y': data['y']})


class RegionModel:
    "Region data"

    def __init__(self, region: dict):
        self.data = region
        regionData = {
            "points": [
                {"x": 0,
                 "y": 0},
                {"x": 0,
                 "y": 0},
            ],
            "color": "",
            "regionType": "textRegion",
            "boundingRect": {"x": 0, "y": 0, "width": 0, "height": 0}
        }

    def setColor(self, color: str):
        assertCond(color, isinstance(color, str))
        self.data['color'] = color

    def setRegionType(self, regionType: str):
        assertCond(regionType, isinstance(regionType, str))
        self.data['regionType'] = regionType

    def setPoints(self, points: list):
        "Set points to region data"
        assertCond(points, isinstance(points, list))
        assert all([True for point in points if 'x' in point and 'y' in point])
        [assertCond(point['x'],
                    isinstance(point['x'], int)) for point in points]
        [assertCond(point['y'],
                    isinstance(point['y'], int)) for point in points]
        self.data['points'] = []
        minval = float('inf')
        maxval = float('-inf')
        (minx, miny,
         maxx, maxy) = (None, None, None, None)
        for point in points:
            regionPoint = {}

            if point['x'] < minval:
                minx = point['x']
            if point['x'] > maxval:
                maxx = point['x']
            if point['y'] < minval:
                miny = point['y']
            if point['y'] > maxval:
                maxy = point['y']

            regionPoint['x'] = point['x']
            regionPoint['y'] = point['y']
            self.data['points'].append(regionPoint)
        #
        brect = {"x": minx,
                 "y": miny,
                 "width": maxx - minx,
                 "height": maxy - miny}
        self.data['boundingRect'] = brect

    def setData(self, data):
        "Set data to region"
        assert "color" in data
        assert "regionType" in data
        assert "points" in data
        self.setColor(data['color'])
        self.setRegionType(data['regionType'])
        self.setPoints(data['points'])

