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


def sliceEllipseFromImage(image: np.ndarray,
                          bbox: {"x1": int, "y1": int, "x2": int, "y2": int}):
    params = {"bbox": bbox}
    return sliceShapeFromImage(image,
                               drawEllipseWithBbox,
                               params)


def sliceThickLineFromImage(image: np.ndarray,
                            bbox: {"x1": int, "y1": int,
                                   "x2": int, "y2": int},
                            width=3):
    "Slice thick line from image"
    assert width > 1
    mask=np.zeros_like(image, dtype=np.uint8)
    img=Image.fromarray(mask)
    imdraw=ImageDraw.Draw(img)
    imdraw.line([(bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2'])],
                fill="white", width=width)
    imgarr=np.array(img)
    mask_bool=imgarr == 255
    return sliceImageWithMask(image, mask_bool)
