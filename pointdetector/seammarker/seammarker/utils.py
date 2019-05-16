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


def getConsecutive2D(data: np.ndarray,
                     direction: str,
                     stepsize=1,
                     only_index=False):
    "Get consecutive values in horizontal vertical and diagonal directions"
    assert data.shape[1] == 2
    diffval = getDiffDirection(stepsize, direction)
    diffarr = data[1:] - data[:-1]
    indices = np.argwhere(diffarr != diffval)
    indices = indices.T
    indices = indices[0] + 1
    if only_index:
        return indices
    else:
        splitdata = np.split(data, indices, axis=0)
        splitdata = [
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
    strsplit = str1.split(ext_delimiter)
    ext = strsplit.pop()
    newstr = ext_delimiter.join(strsplit)
    return (newstr, ext)


def readImage(path: str) -> np.ndarray:
    "Read image from the path"
    pilim = Image.open(path)
    return np.array(pilim)


def shapeCoordinate(coord: np.ndarray):
    "Reshape coordinate to have [[y, x]] structure"
    cshape = coord.shape
    assert 2 in cshape or 3 in cshape
    if cshape[0] == 2:
        coord = coord.T
    elif cshape[1] == 2:
        pass
    elif cshape[0] == 3:
        coord = coord.T
        coord = coord[:, :2]
    elif cshape[1] == 3:
        coord = coord[:, :2]
    # obtain unique coords
    uni1, index = np.unique(coord, return_index=True, axis=0)
    uni1 = coord[np.sort(index), :]
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
    r, c = img.shape[:2]
    flatim = img.reshape((-1))
    #
    normImg = np.interp(flatim,
                        (flatim.min(), flatim.max()),
                        (0, 255),
                        )
    normImg = normImg.astype(np.uint8)
    normImg = normImg.reshape((r, c))
    return normImg

# Debug related


def drawMark2Image(image: np.ndarray,
                   coord: np.ndarray,
                   imstr: str):
    zeroimg = np.zeros_like(image, dtype=np.uint8)
    imcp = image.copy()
    assert coord.shape[1] == 2
    for i in range(coord.shape[0]):
        yx = coord[i, :]
        imcp[yx[0], yx[1], :] = 255
        zeroimg[yx[0], yx[1], :] = 255
    #
    zeroname = imstr + "-zero.png"
    name = imstr + ".png"
    Image.fromarray(imcp).save(name)
    Image.fromarray(zeroimg).save(zeroname)
    return imcp, zeroimg
