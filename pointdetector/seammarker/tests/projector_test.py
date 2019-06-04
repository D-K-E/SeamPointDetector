# author: Kaan Eraslan
# License: see, LICENSE
# No warranties, see LICENSE
# test projector.py

from seammarker import projector as pj

import unittest
import os
from PIL import Image, ImageOps
import numpy as np
import json
import pdb


class ProjectorTest(unittest.TestCase):
    "Test projector"

    def setUp(self):
        "Set up test files"
        currentdir = str(__file__)
        currentdir = os.path.join(currentdir, os.pardir)
        currentdir = os.path.join(currentdir, os.pardir)
        currentdir = os.path.abspath(currentdir)
        testdir = os.path.join(currentdir, "tests")
        assetdir = os.path.join(testdir, 'assets')
        self.assetdir = assetdir
        self.imagedir = os.path.join(assetdir, 'images')
        self.utilsImageDir = os.path.join(self.imagedir, 'utils')
        self.projectorImageDir = os.path.join(self.imagedir, 'projector')
        jsondir = os.path.join(assetdir, 'jsonfiles')
        self.npdir = os.path.join(assetdir, 'numpyfiles')
        self.image_col_path = os.path.join(self.imagedir, 'vietHard.jpg')

        # checks
        self.GEN_SLICE_IMAGE_WITH_MASK = False
        self.slice_img_path = os.path.join(self.projectorImageDir,
                                           "sliceImageWithMask.png")
        self.GEN_SLICE_ELLIPSE_IMAGE_MASK = False
        self.slice_img_ellipse_path = os.path.join(self.projectorImageDir,
                                                   "sliceEllipseFromImage.png")
        self.GEN_SLICE_THICKLINE_IMAGE_MASK = False
        self.slice_img_thickline_path = os.path.join(
            self.projectorImageDir,
            "sliceThickLineFromImage.png"
        )
        self.GEN_SLICE_POLYGON_IMAGE = False
        self.slice_img_polygon_path = os.path.join(self.projectorImageDir,
                                                   "slicePolygonFromImage.png")

    def compareArrays(self, arr1, arr2, message):
        "Compare arrays for equality"
        result = arr1 == arr2
        result = result.all()
        self.assertTrue(result, message)

    def generateSliceImageWithMask(self):
        if self.GEN_SLICE_IMAGE_WITH_MASK:
            vietImg = np.array(Image.open(self.image_col_path).copy())
            rownb, colnb = vietImg.shape[:2]
            width = 20
            height = 30
            halfrow = rownb // 2
            halfcol = colnb // 2
            y1 = halfrow - height
            y2 = halfrow + height
            x1 = halfcol - width
            x2 = halfcol + width
            mask = np.zeros((rownb, colnb), dtype=np.bool)
            mask[y1:y2+1, x1:x2+1] = True
            newimg = pj.sliceImageWithMask(vietImg.copy(), mask)
            newimg = Image.fromarray(newimg)
            newimg.save(self.slice_img_path)

    def generateSliceEllipseFromImage(self):
        if self.GEN_SLICE_ELLIPSE_IMAGE_MASK:
            vietImg = np.array(Image.open(self.image_col_path).copy())
            rownb, colnb = vietImg.shape[:2]
            width = 200
            height = 300
            halfrow = rownb // 2
            halfcol = colnb // 2
            y1 = halfrow - height
            y2 = halfrow + height
            x1 = halfcol - width
            x2 = halfcol + width
            newimg = pj.sliceEllipseFromImage(vietImg.copy(),
                                              bbox={"x1": x1,
                                                    "x2": x2,
                                                    "y1": y1,
                                                    "y2": y2})
            newimg = Image.fromarray(newimg)
            newimg.save(self.slice_img_ellipse_path)

    def generateSliceThickLineFromImage(self):
        if self.GEN_SLICE_THICKLINE_IMAGE_MASK:
            vietImg = np.array(Image.open(self.image_col_path).copy())
            rownb, colnb = vietImg.shape[:2]
            width = 200
            height = 300
            halfrow = rownb // 2
            halfcol = colnb // 2
            thickness = 6
            y1 = halfrow - height
            y2 = halfrow + height
            x1 = halfcol - width
            x2 = halfcol + width
            newimg = pj.sliceThickLineFromImage(vietImg.copy(),
                                                bbox={"x1": x1,
                                                      "x2": x2,
                                                      "y1": y1,
                                                      "y2": y2},
                                                line_width=thickness)
            newimg = Image.fromarray(newimg)
            newimg.save(self.slice_img_thickline_path)

    def generateSlicePolygonFromImage(self):
        if self.GEN_SLICE_THICKLINE_IMAGE_MASK:
            vietImg = np.array(Image.open(self.image_col_path).copy())
            rownb, colnb = vietImg.shape[:2]
            width1 = 200
            height1 = 300
            halfrow = rownb // 2
            halfcol = colnb // 2
            p1 = {"x": halfcol - width1, "y": halfrow - height1}
            p2 = {"x": p1['x'] + width1 // 2, "y": p1['y']}
            p3 = {"x": halfcol, "y": halfrow}
            p4 = {"x": halfcol + width1, "y": halfrow + height1}
            p5 = {"x": halfcol + width1 // 3, "y": halfrow + height1 // 2}
            coords = [p1, p2, p3, p4, p5]
            newimg = pj.slicePolygonFromImage(vietImg.copy(), coords)
            newimg = Image.fromarray(newimg)
            newimg.save(self.slice_img_polygon_path)

    def test_sliceImageWithMask(self):
        "test slice image with mask works for a rectangle"
        vietImg = np.array(Image.open(self.image_col_path).copy())
        rownb, colnb = vietImg.shape[:2]
        width = 20
        height = 30
        halfrow = rownb // 2
        halfcol = colnb // 2
        y1 = halfrow - height
        y2 = halfrow + height
        x1 = halfcol - width
        x2 = halfcol + width
        mask = np.zeros_like(vietImg, dtype=np.bool)
        mask[y1:y2+1, x1:x2+1] = True
        newimg = pj.sliceImageWithMask(vietImg.copy(), mask)
        compimg = Image.open(self.slice_img_path)
        comparr = np.array(compimg)
        self.compareArrays(comparr, newimg,
                           "Image Slice not same with expected image")

    def test_sliceEllipseFromImage(self):
        vietImg = np.array(Image.open(self.image_col_path).copy())
        rownb, colnb = vietImg.shape[:2]
        width = 200
        height = 300
        halfrow = rownb // 2
        halfcol = colnb // 2
        y1 = halfrow - height
        y2 = halfrow + height
        x1 = halfcol - width
        x2 = halfcol + width
        newimg = pj.sliceEllipseFromImage(vietImg.copy(),
                                          bbox={"x1": x1,
                                                "x2": x2,
                                                "y1": y1,
                                                "y2": y2})
        compimg = Image.open(self.slice_img_ellipse_path)
        comparr = np.array(compimg)
        self.compareArrays(comparr, newimg,
                           "Image Slice not same with expected image")

    def test_sliceThickLineFromImage(self):
        vietImg = np.array(Image.open(self.image_col_path).copy())
        rownb, colnb = vietImg.shape[:2]
        width = 200
        height = 300
        halfrow = rownb // 2
        halfcol = colnb // 2
        thickness = 6
        y1 = halfrow - height
        y2 = halfrow + height
        x1 = halfcol - width
        x2 = halfcol + width
        newimg = pj.sliceThickLineFromImage(vietImg.copy(),
                                            bbox={"x1": x1,
                                                  "x2": x2,
                                                  "y1": y1,
                                                  "y2": y2},
                                            line_width=thickness)
        compimg = Image.open(self.slice_img_thickline_path)
        comparr = np.array(compimg)
        self.compareArrays(comparr, newimg,
                           "Image Slice not same with expected image")

    def test_slicePolygonFromImage(self):
        "Test if we can slice a polygon from image"
        vietImg = np.array(Image.open(self.image_col_path).copy())
        rownb, colnb = vietImg.shape[:2]
        width1 = 200
        height1 = 300
        halfrow = rownb // 2
        halfcol = colnb // 2
        p1 = {"x": halfcol - width1, "y": halfrow - height1}
        p2 = {"x": p1['x'] + width1 // 2, "y": p1['y']}
        p3 = {"x": halfcol, "y": halfrow}
        p4 = {"x": halfcol + width1, "y": halfrow + height1}
        p5 = {"x": halfcol + width1 // 3, "y": halfrow + height1 // 2}
        coords = [p1, p2, p3, p4, p5]
        newimg = pj.slicePolygonFromImage(vietImg.copy(), coords)
        compimg = Image.open(self.slice_img_polygon_path)
        comparr = np.array(compimg)
        self.compareArrays(comparr, newimg,
                           "Image Slice not same with expected image")

    def test_getEllipseCoordsInBbox(self):
        "test obtaining ellipse coordinates in bbox"
        vietImg = np.array(Image.open(self.image_col_path).copy())
        rownb, colnb = vietImg.shape[:2]
        width = 200
        height = 300
        halfrow = rownb // 2
        halfcol = colnb // 2
        y1 = halfrow - height
        y2 = halfrow + height
        x1 = halfcol - width
        x2 = halfcol + width
        bbox = {"x1": x1,
                "x2": x2,
                "y1": y1,
                "y2": y2}
        pdb.set_trace()
        coords = pj.getEllipseCoordsInBbox(bbox)


if __name__ == "__main__":
    unittest.main()
