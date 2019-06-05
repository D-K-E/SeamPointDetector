# mark seams in texts
# author: Kaan Eraslan
# purpose: Project circular or elliptical image coordinates along
# a line, so that seam marking can work on it.

import numpy as np
from PIL import Image, ImageDraw


def sliceImageWithMask(image: np.ndarray, mask: np.ndarray):
    "Slice image using a boolean mask"
    return np.where(mask, image, 255)


def sliceCoordinatesFromMask(mask: np.ndarray):
    "Get coordinates from mask"
    return np.argwhere(mask == True)  # operator is == not 'is'


def sliceShapeFromImage(image: np.ndarray,
                        fn: lambda x: x,
                        kwargs,
                        withCoordinate=False):
    mask = np.zeros_like(image, dtype=np.uint8)
    img = Image.fromarray(mask)
    fn(img, **kwargs)
    imgarr = np.array(img)
    mask_bool = imgarr == 255
    if withCoordinate is False:
        return sliceImageWithMask(image, mask_bool)
    else:
        return (sliceImageWithMask(image, mask_bool),
                sliceCoordinatesFromMask(mask_bool))


def drawEllipseWithBbox(image: np.ndarray,
                        bbox: {"x1": int, "y1": int, "x2": int, "y2": int}):
    "Slice an ellipse from an image"
    imdraw = ImageDraw.Draw(image)
    imdraw.ellipse([(bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2'])],
                   fill="white")


def sliceEllipseFromImage(image: np.ndarray,
                          bbox: {"x1": int, "y1": int,
                                 "x2": int, "y2": int},
                          withCoord=False):
    params = {"bbox": bbox}
    return sliceShapeFromImage(image,
                               drawEllipseWithBbox,
                               params, withCoord)


def cutEllipse2Half(image: np.ndarray,
                    bbox: {"x1": int, "y1": int,
                           "x2": int, "y2": int},
                    withMask=False):
    """
    Cut ellipse in half vertically

    For first half we paint the second half in white.
    For the second half we paint the first half in white
    """
    ellipseImage, ellipseCoords = sliceEllipseFromImage(image,
                                                        bbox,
                                                        withCoord=True)
    rownb, colnb, pixelnb = ellipseImage.shape
    xindx = ellipseCoords[:, 1]
    centerX = (bbox['x1'] + bbox['x2']) // 2
    firstHalfEllipseCoords = ellipseCoords[xindx <= centerX]
    secondHalfEllipseCoords = ellipseCoords[xindx > centerX]
    yindx1 = firstHalfEllipseCoords[:, 0]
    yindx2 = secondHalfEllipseCoords[:, 0]
    xindx1 = firstHalfEllipseCoords[:, 1]
    xindx2 = secondHalfEllipseCoords[:, 1]
    firstHalfEllipse = ellipseImage.copy()
    secondHalfEllipse = ellipseImage.copy()
    firstHalfEllipse[yindx2, xindx2, :] = 255
    secondHalfEllipse[yindx1, xindx1, :] = 255
    if not withMask:
        return firstHalfEllipse, secondHalfEllipse
    else:
        immask = np.zeros_like(ellipseImage, dtype=np.uint8)
        firstHalfMask = immask.copy()
        secondHalfMask = immask.copy()
        firstHalfMask[yindx1, xindx1, :] = 255
        secondHalfMask[yindx2, xindx2, :] = 255
        return (firstHalfEllipse, firstHalfMask,
                secondHalfEllipse, secondHalfMask)


def joinSecondHalf2FirstHalf(image: np.ndarray,
                             bbox: {"x1": int, "y1": int,
                                    "x2": int, "y2": int}):
    "Join second half to first half"
    (firstHalf, firstHalfMask,
     secondHalf, secondHalfMask) = cutEllipse2Half(image, bbox, withMask=True)
    rownb, colnb, pixelnb = firstHalf.shape
    mask = np.zeros((rownb * 2, colnb, pixelnb), dtype=np.uint8)


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
                            line_width=3, withCoord=False):
    "Slice thick line from image"
    assert line_width > 1
    params = {"bbox": bbox, "width": line_width}
    return sliceShapeFromImage(image, drawThickLineWithBbox,
                               params, withCoord)


def drawPolygonWithBbox(image: np.ndarray,
                        coords: [{"x": int, "y": int},
                                 {"x": int, "y": int}]):
    "Draw polygon with bbox"
    imdraw = ImageDraw.Draw(image)
    coordlist = [(coord['x'], coord['y']) for coord in coords]
    imdraw.polygon(coordlist, fill="white")


def slicePolygonFromImage(image: np.ndarray,
                          coords: [{"x": int, "y": int},
                                   {"x": int, "y": int}],
                          withCoord=False):
    "Slice polygon defined by bbox"
    params = {"coords": coords}
    return sliceShapeFromImage(image, drawPolygonWithBbox,
                               params, withCoord)
