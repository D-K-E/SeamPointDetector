from PIL import Image
import numpy as np
import json

# From stack overflow


def getConsecutive1D(data: np.ndarray,
                     stepsize=1,
                     only_index=False):
    "Get consecutive values from 1d array"
    assert data.ndim == 1
    indices = np.argwhere((data[1:] - data[:-1]) != stepsize)
    indices = indices.T[0] + 1  # include the range
    if only_index is False:
        subarrays = np.split(data, indices, axis=0)
        return subarrays, indices
    else:
        return indices


def getDiffDirection(stepsize: int,
                     direction: str):
    "create 2d diff vec using stepsize"
    direction = direction.lower()
    assert direction in ["vertical", "horizontal",
                         "diagonal-l", "diagonal-r"]
    if direction == "vertical":
        rowdiff, coldiff = stepsize, 0
    elif direction == "horizontal":
        rowdiff, coldiff = 0, stepsize
    elif direction == "diagonal-l":
        rowdiff, coldiff = stepsize, stepsize
    elif direction == "diagonal-r":
        rowdiff, coldiff = stepsize, -stepsize
    return [rowdiff, coldiff]


def computeSubtractionTable(data: np.ndarray,
                            diffval: [int, int]):
    """Obtain subtraction table from data

    Our idea is pretty simple
    the maximum column value marks a change on row values in coordinates
    Ex. matrix like [[3,8], [0,9]] whose coordinate matrix would be
    [[0, 0], [0, 1], [1, 0], [1, 1]], notice that once we arrive to the
    column number 1, we need to change the row number.
    Thus the maximum column number is like a pivot for row number
    In a split coordinate array we want to make sure that each coordinate
    has a row from above and below. Thus we need a split size that is 3 times
    larger than a normal pivot value, when it makes sense.

    TODO
    I can group the elements per element basis. That is I find
    group of elements that have the specified difference.
    I am not able to constitute a single array which satisfies
    the diffval from these groups
    one approach is to group the first and second element of groups
    that satisfy the diffval
    for example [[[1,0],[0,0]], ..., [[2,0], [1,0]], ... ]
    notice that [1, 0] is in first position at first in second position
    at second
    """
    splitsize = 0
    drows = data.shape[0]
    maxcol = data[:, 1].max()
    rowPivotVal = maxcol
    rowAmount = rowPivotVal * 3
    if drows > rowAmount:
        splitsize = drows // rowAmount
        dataList = np.array_split(data, splitsize, axis=0)
        lastData = [dataList[-2], dataList[-1]]
        lastData = np.concatenate(lastData, axis=0)
        dataList[-2] = lastData
    else:
        if drows > 50 and drows < 500:
            splitsize = 100
        elif drows > 500 and drows < 1000:
            splitsize = 800
        elif drows > 1000 and drows < 2000:
            splitsize = 1500
        elif drows > 2000:
            splitsize = 2000
        dataList = np.array_split(data, splitsize, axis=0)
        lastData = [dataList[-2], dataList[-1]]
        lastData = np.concatenate(lastData, axis=0)
        dataList[-2] = lastData
    diffGroups = []
    groupIndices = []
    offset = 0
    for subData in dataList:
        subrowSize = subData.shape[0]
        subData[:, 0] = subData[:, 0]
        subTable = subData[:, np.newaxis, :] - subData
        cond = (
            subTable[:, :, 0] != diffval[0]
        ) & (subTable[:, :, 1] != diffval[1])
        indices = np.argwhere(cond)
        diffGroup = np.take(subData, indices, axis=0)
        diffGroups.append(diffGroup)
        updatedIndices = indices + offset
        groupIndices.append(updatedIndices)
        offset += subrowSize
    diffGroups = np.concatenate(diffGroups, axis=0)
    groupIndices = np.concatenate(groupIndices, axis=0)
    return diffGroups, groupIndices



def getConsecutive2D(data: np.ndarray,
                     direction: str,
                     stepsize=1,
                     only_index=False):
    "Get consecutive values in horizontal vertical and diagonal directions"
    assert len(data.shape) == 2
    assert data.shape[1] == 2
    diffval=getDiffDirection(stepsize, direction)
    diffarr=data[1:] - data[:-1]
    indices=np.argwhere(diffarr != diffval)
    indices=indices.T
    indices=indices[0] + 1
    if only_index:
        return indices
    else:
        splitdata=np.split(data, indices, axis=0)
        splitdata=[
            data for data in splitdata if data.size > 0 and data.shape[0] > 1
        ]
        return splitdata, indices

# End stack overflow


def saveJson(path, obj):
    "Save json"
    with open(path, 'w',
              encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def stripExt(str1: str, ext_delimiter='.') -> [str, str]:
    "Strip extension"
    strsplit=str1.split(ext_delimiter)
    ext=strsplit.pop()
    newstr=ext_delimiter.join(strsplit)
    return (newstr, ext)


def readImage(path: str) -> np.ndarray:
    "Read image from the path"
    pilim=Image.open(path)
    return np.array(pilim)


def shapeCoordinate(coord: np.ndarray):
    "Reshape coordinate to have [[y, x]] structure"
    cshape=coord.shape
    assert 2 in cshape or 3 in cshape
    if cshape[0] == 2:
        coord=coord.T
    elif cshape[1] == 2:
        pass
    elif cshape[0] == 3:
        coord=coord.T
        coord=coord[:, :2]
    elif cshape[1] == 3:
        coord=coord[:, :2]
    # obtain unique coords
    uni1, index=np.unique(coord, return_index=True, axis=0)
    uni1=coord[np.sort(index), :]
    return uni1


def assertCond(var, cond: bool, printType=True):
    "Assert condition print message"
    if printType:
        assert cond, 'variable value: {0}\nits type: {1}'.format(var,
                                                                 type(var))
    else:
        assert cond, 'variable value: {0}'.format(var)


def normalizeImageVals(img: np.ndarray):
    ""
    r, c=img.shape[:2]
    flatim=img.reshape((-1))
    #
    normImg=np.interp(flatim,
                        (flatim.min(), flatim.max()),
                        (0, 255),
                        )
    normImg=normImg.astype(np.uint8)
    normImg=normImg.reshape((r, c))
    return normImg

# Debug related


def drawMark2Image(image: np.ndarray,
                   coord: np.ndarray,
                   imstr: str):
    zeroimg=np.zeros_like(image, dtype=np.uint8)
    imcp=image.copy()
    assert coord.shape[1] == 2
    for i in range(coord.shape[0]):
        yx=coord[i, :]
        imcp[yx[0], yx[1], :]=255
        zeroimg[yx[0], yx[1], :]=255
    #
    zeroname=imstr + "-zero.png"
    name=imstr + ".png"
    Image.fromarray(imcp).save(name)
    Image.fromarray(zeroimg).save(zeroname)
    return imcp, zeroimg
