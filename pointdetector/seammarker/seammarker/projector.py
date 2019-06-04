# mark seams in texts
# author: Kaan Eraslan
# purpose: Project circular or elliptical image coordinates along
# a line, so that seam marking can work on it.

import numpy as np
from PIL import Image, ImageDraw


def sliceImageWithMask(image: np.ndarray, mask: np.ndarray):
    "Slice image using a boolean mask"
    return np.where(mask, image, 255)


def sliceShapeFromImage(image: np.ndarray,
                        fn: lambda x: x,
                        kwargs):
    mask = np.zeros_like(image, dtype=np.uint8)
    img = Image.fromarray(mask)
    fn(img, **kwargs)
    imgarr = np.array(img)
    mask_bool = imgarr == 255
    return sliceImageWithMask(image, mask_bool)


def drawEllipseWithBbox(image: np.ndarray,
                        bbox: {"x1": int, "y1": int, "x2": int, "y2": int}):
    "Slice an ellipse from an image"
    imdraw = ImageDraw.Draw(image)
    imdraw.ellipse([(bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2'])],
                   fill="white")


def getEllipseCoordsInBbox(bbox: {"x1": int, "y1": int,
                                  "x2": int, "y2": int}):
    """
    Get ellipse coordinates from bbox no rotation is supposed

    Implements the following equation:
        ((x * cos(alpha) + y * sin(alpha)) / x_radius) ** 2 +
        ((x * sin(alpha) - y * cos(alpha)) / y_radius) ** 2 = 1
    """
    minX = min(bbox['x1'], bbox['x2'])
    maxX = max(bbox['x1'], bbox['x2'])
    minY = min(bbox['y1'], bbox['y2'])
    maxY = max(bbox['y1'], bbox['y2'])
    topLeft = {"x": minX, "y": minY}
    topRight = {"x": maxX, "y": minY}
    bottomLeft = {"x": minX, "y": maxY}
    bottomRight = {"x": maxX, "y": maxY}

    midX = minX + ((maxX - minX) // 2)
    midY = minY + ((maxY - minY) // 2)
    center = {"x": midX, "y": midY}
    coords = np.array([
        [[k, i] for k in range(minY, maxY+1)] for i in range(minX, maxX+1)
    ], dtype=np.float32)
    coords = np.unique(coords.reshape((-1, 2)), axis=0)
    radius_x = maxX - minX
    radius_y = maxY - minY
    sin_t, cos_t = np.sin(0), np.cos(0)
    distFirstTerm = ((coords[:, 1] * cos_t) + (coords[:, 0] * sin_t) / radius_y) ** 2
    distSecTerm = ((coords[:, 1] * sin_t) - (coords[:, 0] * cos_t) / radius_x) ** 2
    distance = distFirstTerm + distSecTerm
    indices = np.nonzero(distance < 1)
    indices = indices[0]
    return indices, coords


def sliceEllipseFromImage(image: np.ndarray,
                          bbox: {"x1": int, "y1": int, "x2": int, "y2": int}):
    params = {"bbox": bbox}
    return sliceShapeFromImage(image,
                               drawEllipseWithBbox,
                               params)


def drawThickLineWithBbox(image: np.ndarray,
                          bbox: {"x1": int, "y1": int, "x2": int, "y2": int},
                          width: int):
    "Draw line with bbox"
    imdraw = ImageDraw.Draw(image)
    imdraw.line([(bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2'])],
                   fill="white", width=width)


def sliceThickLineFromImage(image: np.ndarray,
                            bbox: {"x1": int, "y1": int,
                                   "x2": int, "y2": int},
                            line_width=3):
    "Slice thick line from image"
    assert line_width > 1
    params = {"bbox": bbox, "width": line_width}
    return sliceShapeFromImage(image, drawThickLineWithBbox, params)


def drawPolygonWithBbox(image: np.ndarray,
                        coords: [{"x": int, "y": int},
                               {"x": int, "y": int}]):
    "Draw polygon with bbox"
    imdraw = ImageDraw.Draw(image)
    coordlist = [(coord['x'], coord['y']) for coord in coords]
    imdraw.polygon(coordlist, fill="white")


def slicePolygonFromImage(image: np.ndarray,
                          coords: [{"x": int, "y": int},
                               {"x": int, "y": int}]):
    "Slice polygon defined by bbox"
    params = {"coords": coords}
    return sliceShapeFromImage(image, drawPolygonWithBbox, params)
