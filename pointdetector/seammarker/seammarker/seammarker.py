# mark seams in texts
# author: Kaan Eraslan
# Implementation in part taken from the link:
# https://karthikkaranth.me/blog/implementing-seam-carving-with-python/):

# Author in the link: Karthik Karanth

import numpy as np  # array/matrix manipulation
import scipy.ndimage as nd  # operate easily on image matrices
import operator
import pdb
from seammarker.utils import getConsecutive1D
from seammarker.utils import getConsecutive2D
from seammarker import projector as pjr


class SeamFuncs:
    def __init__(self):
        self.mark_color = [0, 255, 0]
        self.ops = {'>': operator.gt,
                    '<': operator.lt,
                    '>=': operator.ge,
                    '<=': operator.le,
                    '==': operator.eq}

    def calc_energy(self,
                    img: np.ndarray([], dtype=np.uint8)):
        filter_du = np.array([
            [1.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, -2.0, -1.0],
        ])
        # This converts it from a 2D filter to a 3D filter, replicating the same
        # filter for each channel: R, G, B
        filter_du = np.stack([filter_du] * 3, axis=2)

        filter_dv = np.array([
            [1.0, 0.0, -1.0],
            [2.0, 0.0, -2.0],
            [1.0, 0.0, -1.0],
        ])
        # This converts it from a 2D filter to a 3D filter, replicating the same
        # filter for each channel: R, G, B
        filter_dv = np.stack([filter_dv] * 3, axis=2)

        img = img.astype('float32')
        convolved = np.absolute(nd.filters.convolve(
            img, filter_du)) + np.absolute(
                nd.filters.convolve(img, filter_dv))

        # We sum the energies in the red, green, and blue channels
        energy_map = convolved.sum(axis=2)

        return energy_map

    def minimum_seam(self, img: np.ndarray([], dtype=np.uint8),
                     emap=None):
        r, c, _ = img.shape

        # if the energy map is already calculated
        if emap is not None:
            energy_map = emap
        else:
            energy_map = self.calc_energy(img)

        M = energy_map.copy()
        backtrack = np.zeros_like(M, dtype=np.int)

        for i in range(1, r):
            for j in range(0, c):
                # Handle the left edge of the image, to ensure 
                # we don't index -1
                if j == 0:
                    offset_value = j
                    colrange = j + 2
                    maprow = M[i - 1, offset_value:colrange]
                    min_energy_indx_in_row = np.argmin(maprow)
                    min_energy_indx = min_energy_indx_in_row + offset_value
                    backtrack[i, j] = min_energy_indx
                    min_energy = M[i - 1, min_energy_indx]
                else:
                    offset_value = j - 1
                    colrange = j + 2
                    maprow = M[i - 1, offset_value:colrange]
                    min_energy_indx_in_row = np.argmin(maprow)
                    min_energy_indx = min_energy_indx_in_row + offset_value
                    backtrack[i, j] = min_energy_indx
                    min_energy = M[i - 1, min_energy_indx]

                M[i, j] += min_energy

        return M, backtrack

    def mark_column(self,
                    img: np.ndarray([], dtype=np.uint8),
                    emap=None,
                    mark_color=[250, 120, 120]  # yellow
                    ):
        r, c, _ = img.shape
        imcopy = img.copy()

        M, backtrack = self.minimum_seam(img, emap)

        # Create a (r, c) matrix filled with the value True
        # We'll be marking all pixels from the image which
        # have False later
        mask = np.zeros((r, c), dtype=np.bool)

        # Find the position of the smallest element in the
        # last row of M
        j = np.argmin(M[-1])

        for i in reversed(range(r)):
            # Mark the pixels
            # and save mark positions for later use
            mask[i, j] = True
            j = backtrack[i, j]

        # Since the image has 3 channels, we convert our
        # mask to 3D
        mask = np.stack([mask] * 3, axis=2)

        # mark the pixels with the given mark color
        imcopy = np.where(mask, mark_color, img)

        return imcopy, mask

    def mark_row(self,
                 img: np.ndarray([], dtype=np.uint8),
                 mark_color=[250, 120, 120]):
        img = np.rot90(img, 1, (0, 1))
        img, mask = self.mark_column(img, mark_color=mark_color)
        img = np.rot90(img, 3, (0, 1))
        mask = np.rot90(mask, 3, (0, 1))
        return img, mask


class SeamFuncsAI(SeamFuncs):
    "Seam funcs ai"

    def __init__(self):
        super().__init__()
        self.pathCost = 0
        self.path = []
        self.frontiers = []
        self.moves = ["right", "left", "up", "down", "leftUp",
                      "leftDown", "rightDown", "rightUp"
                      ]
        self.moveOps = {
            "right": self.goRight, "left": self.goLeft, "down": self.goDown,
            "up": self.goUp, "leftDown": self.goLeftDown,
            "leftUp": self.goLeftUp, "rightDown": self.goRightDown,
            "rightUp": self.goRightUp
        }
        self.moveBackOps = {
            "left": self.goRight, "right": self.goLeft,
            "up": self.goDown, "down": self.goUp,
            "rightUp": self.goLeftDown, "rightDown": self.goLeftUp,
            "leftUp": self.goRightDown, "leftDown": self.goRightUp
        }

    def goUp(self, currentRow: int,
             currentColumn: int):
        "Go up"
        return currentRow - 1, currentColumn

    def goDown(self, currentRow: int, currentColumn: int):
        "go down"
        return currentRow + 1, currentColumn

    def goLeft(self, currentRow: int, currentColumn: int):
        "go left"
        return currentRow, currentColumn - 1

    def goRight(self, currentRow: int, currentColumn: int):
        "go left"
        return currentRow, currentColumn + 1

    def goLeftUp(self, currentRow: int, currentColumn: int):
        "go left up"
        return currentRow - 1, currentColumn - 1

    def goRightUp(self, currentRow: int, currentColumn: int):
        "go left up"
        return currentRow - 1, currentColumn + 1

    def goLeftDown(self, currentRow: int,
                   currentColumn: int):
        "go left up"
        return currentRow + 1, currentColumn - 1

    def goRightDown(self, currentRow: int,
                    currentColumn: int):
        "go left up"
        return currentRow + 1, currentColumn + 1

    def getZeroZones(self, img: np.ndarray):
        "Get the indices of zeros in image"
        return self.getValZones(img, 0, "==")

    def getInferior2MeanZones(self, img: np.ndarray):
        "Get the indices of values inferior to mean value in image"
        meanval = img.mean()
        meanval = int(meanval)
        return self.getValZones(img, meanval, '<')

    def getImageCoordinateArray(self, img: np.ndarray) -> np.ndarray:
        "Get zone coordinates array"
        row, col = img.shape[:2]
        coords = [[[r, c] for c in range(col)] for r in range(row)]
        coordarr = np.array(coords, dtype=np.int)
        coordarr = coordarr.reshape((-1, 2))
        coordarr = np.unique(coordarr, axis=0)
        return coordarr

    def getValZones(self, img: np.ndarray, val: int, op: str):
        "Get indices of val in image"
        coordarr = self.getImageCoordinateArray(img)

        if len(img.shape) > 2:
            raise ValueError("Image has more than one color channel, change it"
                             "to gray scale")
        zones = np.empty((img.shape[0], img.shape[1], 3), dtype=np.int)
        zones[:, :, 0] = img.copy()
        zones[:, :, 1:] = coordarr
        zones = zones.reshape((-1, 3))
        zerovals = self.ops[op](zones[:, 0], val)
        zones = zones[zerovals, :]
        return zones

    def testGoalCoordinate(self,
                           currentRow: int,
                           currentColumn: int,
                           goals: [(int, int)],
                           ):
        "check whether the given current row is in goal range"
        return (currentRow, currentColumn) in goals

    def checkLimit(self,
                   currentRow: int,
                   currentColumn: int,
                   rownb: int,
                   colnb: int) -> bool:
        "check limit"
        inRow = currentRow >= 0 and currentRow < rownb
        inCol = currentColumn >= 0 and currentColumn < colnb
        return inRow and inCol

    def getStepCost(self,
                    currentRow: int,
                    currentColumn: int,
                    nextRow: int,
                    nextColumn: int,
                    img: np.ndarray):
        "Get step cost"
        current_pixel_val = img[currentRow, currentColumn]
        next_pixel_val = img[nextRow, nextColumn]
        next_pixel_sum = np.sum(next_pixel_val, dtype=np.int)
        current_pixel_sum = np.sum(current_pixel_val, dtype=np.int)
        stepcost = 1
        pixelsum = next_pixel_sum + current_pixel_sum
        return stepcost + pixelsum

    def checkInFrontier(self,
                        currentRow: int,
                        currentColumn: int,
                        frontier: list):
        "check if current row and column is in frontier"
        for row, col, step_cost, path in frontier:
            if (row, col) == (currentRow, currentColumn):
                return True
                # print('in frontier')
        #
        return False

    def getUpperBound(self, img: np.ndarray,
                      path: [(int, int)]):
        """
        Get upper bound from path

        The idea is to compute the energy of
        the hypothenus of the extreme points.
        column extreme and row extreme
        """
        col_extrema = sorted(path, key=lambda x: x[1], reverse=True)
        col_extrema = col_extrema.pop()
        col_extrema = np.array([255 for i in range(col_extrema[1])])
        colsum = np.sum(col_extrema, dtype=np.int)
        row_extrema = sorted(path, key=lambda x: x[0], reverse=True)
        row_extrema = row_extrema.pop()
        row_extrema = np.array([255 for i in range(row_extrema[0])])
        rowsum = np.sum(row_extrema)
        return colsum + rowsum

    def getPathCost(self, img: np.ndarray,
                    path: [(int, int)]):
        ""
        path_energy = 0
        for p in path:
            pixelval = np.sum(img[p[0], p[1]], dtype=np.int)
            path_energy += pixelval
        #
        pathCost = path_energy + len(path)
        return pathCost

    def checkAvailable(self, currentRow: int,
                       currentColumn: int,
                       path: [(int, int)],
                       possible_energy: int,
                       img: np.ndarray):
        ""
        pathCost = self.getPathCost(img=img, path=path)
        pathUpperBound = self.getUpperBound(img, path)
        return pathCost < pathUpperBound and pathCost < possible_energy

    def moveGreedy(self, moveDirection: str,
                   fromRow: int, fromColumn: int,
                   img: np.ndarray):
        "move greedy in given direction until one hits a point with energy"
        pixelval = img[fromRow, fromColumn]
        pixelval = np.sum(pixelval, dtype=np.int)
        row = fromRow
        col = fromColumn
        rownb = img.shape[0]
        colnb = img.shape[1]
        stepval = 0
        while self.checkLimit(currentRow=row,
                              currentColumn=col,
                              rownb=rownb,
                              colnb=colnb) and pixelval == 0:
            row, col = self.moveFromCoordinate(moveDirection,
                                               currentRow=row,
                                               currentColumn=col)
            pixelval = img[row, col]
            pixelval = np.sum(pixelval, dtype=np.int)
            stepval += 1
        #
        return row, col, stepval

    def moveGreedy_proc(self, moveDirection: str, img: np.ndarray,
                        row: int, col: int):
        "Move greedy in given direction until one hits a point with energy"
        return self.moveGreedy(moveDirection, img, row, col)

    def jump2PointWithCost(self, moveDirection: str, img: np.ndarray,
                           fromRow: int, fromColumn: int,
                           toRow: int, toColumn: int):
        "Jump in the direction given"
        row, col = fromRow, fromColumn
        stepcost = 0
        path = []
        while row != toRow and col != toColumn:
            oldrow, oldcol = row, col
            row, col = self.moveOps[moveDirection](oldrow, oldcol)
            scost = self.getStepCost(currentRow=oldrow,
                                     currentColumn=oldcol,
                                     nextRow=row,
                                     nextColumn=col,
                                     img=img)
            path.append((oldrow, oldcol))
            stepcost += scost
        #
        return row, col, stepcost, path

    def getMoveOrientation(self, moveDirection: str) -> str:
        "Decide if the move is horizontal/vertical/diagonal"
        if moveDirection == "left":
            return "horizontal"
        elif moveDirection == "right":
            return "horizontal"
        elif moveDirection == "up":
            return "vertical"
        elif moveDirection == "down":
            return "vertical"
        elif moveDirection == "leftDown":
            return "diagonal-r"
        elif moveDirection == "leftUp":
            return "diagonal-l"
        elif moveDirection == "rightDown":
            return "diagonal-l"
        elif moveDirection == "rightUp":
            return "diagonal-r"

    def getVHDMoveZone(self,
                       zoneCoordinates: np.ndarray,
                       moveDirection: str) -> []:
        """
        Group zone coordinates for moving vertical/horizontal/diagonal

        Our approach is pretty simple.
        Vertical movement means moving in the same column,
        and incrementing row values.
        Horizontal movement means moving in the same row,
        and incrementing column values.
        Diagonal movement means moving in row and column
        at the same time

        First we find unique column values in the zone.
        If we can move vertically than we should have multiple
        row values attested for the same column.

        Once we have the unique column values and the count for
        each of the values we skip those that have a count less than 2

        Then for each unique column value we stock row values associated
        with it. Then we take the difference between these row values
        to see whether they are consecutive or not.

        """
        direction = self.getMoveOrientation(moveDirection)
        consecutiveVals = getConsecutive2D(data=zoneCoordinates,
                                           direction=direction,
                                           only_index=False)
        return consecutiveVals

    def getMoveZonesFromValZone(self, valZone: np.ndarray) -> dict:
        "Get move zone for image"
        assert valZone.shape[1] == 2
        zones = {key: [] for key in self.moves}
        for moveDir in self.moves:
            moveZone = self.getVHDMoveZone(valZone,
                                           moveDir)
            zoneMat = np.concatenate(moveZone, axis=0)
            zones[moveDir] = {"zoneArray": moveZone, "zoneMatrix": zoneMat}
        return zones

    def getLeastDistancePointFromZone(self,
                                      zone: np.ndarray,
                                      finalRow: int,
                                      finalColumn: int):
        "Get the point that is closest to goal row/column in the zone"
        zoneRow = zone[:, 0]
        zoneCol = zone[:, 1]
        rowsqr = (zoneRow - finalRow) ** 2
        colsqr = (zoneCol - finalColumn) ** 2
        distance = np.sqrt(rowsqr + colsqr)
        minDistanceArg = np.argmin(distance)
        minDistance = distance.min()
        row, col = zone[minDistanceArg]
        return row, col, minDistance

    def getLeastDistance2GoalsPointFromZone(self,
                                            zone: np.ndarray,
                                            goals: [(int, int)]):
        "Apply least distance point from zone to goal points"
        zonePoints = []
        for goal in goals:
            row, col = goal
            zoneRow, zoneCol, distance = self.getLeastDistancePointFromZone(
                zone=zone,
                finalRow=row,
                finalColumn=col
            )
            zonePoints.append((zoneRow, zoneCol, distance, goal))
        #
        zonePoints.sort(key=lambda x: x[2])
        return zonePoints[0]

    def getLeastDistanceMoveDirection(self,
                                      fromRow: int,
                                      fromColumn: int,
                                      toRow: int,
                                      toColumn: int) -> str:
        "Get move direction from row/column to row/column"
        rowdiff = toRow - fromRow
        coldiff = toColumn - fromColumn
        if rowdiff > 0 and coldiff == 0:
            return "down"
        elif rowdiff < 0 and coldiff == 0:
            return "up"
        elif rowdiff == 0 and coldiff > 0:
            return "right"
        elif rowdiff == 0 and coldiff < 0:
            return "left"
        elif rowdiff > 0 and coldiff > 0:
            return "rightDown"
        elif rowdiff > 0 and coldiff < 0:
            return "leftDown"
        elif rowdiff < 0 and coldiff > 0:
            return "rightUp"
        elif rowdiff < 0 and coldiff < 0:
            return "leftUp"

    def checkPointInCoordZone(self, coordzone: np.ndarray,
                              currentRow: int, currentColumn: int):
        "Check if a coordinate zone contain the point"
        arr = [currentRow, currentColumn]
        return any(np.equal(coordzone, arr).all(axis=1))

    def populateMinimalMoveZone(self, moveZones: dict,
                                currentRow: int,
                                currentColumn: int,
                                goals: [(int, int)]):
        "populate minimal move zone list"
        minimalZoneMove = []
        for move, zoneMap in moveZones.items():
            zoneMat = zoneMap['zoneMatrix']
            if self.checkPointInCoordZone(coordzone=zoneMat,
                                          currentRow=currentRow,
                                          currentColumn=currentColumn):
                continue
            zoneArr = zoneMap['zoneArray']
            containingZone = self.getPointZone(zoneArr, currentRow=currentRow,
                                               currentColumn=currentColumn)
            (closestRow,
             closestColumn,
             distance,
             goal) = self.getLeastDistance2GoalsPointFromZone(
                 zone=containingZone,
                 goals=goals
            )
            minimalZoneMove.append((closestRow, closestColumn, move,
                                    distance, goal))
        #
        return minimalZoneMove

    def sortFrontier_proc(self, frontier: list, compareFn):
        "Sort frontier using the compare function"
        frontier.sort(key=compareFn, reverse=True)

    def addNextStateWithCost2Frontier_proc(self, currentRow: int,
                                           currentColumn: int,
                                           nextRow: int,
                                           nextColumn: int,
                                           path: list,
                                           img: np.ndarray,
                                           compareFn,
                                           oldStepCost: int,
                                           frontier: list) -> None:
        "Add next state to frontier"
        newStepCost = self.getStepCost(currentRow=currentRow,
                                       currentColumn=currentColumn,
                                       nextRow=nextRow,
                                       nextColumn=nextColumn,
                                       img=img)
        newPathCost = self.getPathCost(img, path)
        step_cost = newStepCost + oldStepCost + newPathCost
        self.addNode2Frontier_proc(nextRow,
                                   nextColumn,
                                   step_cost,
                                   path,
                                   frontier,
                                   compareFn)

    def addNode2Frontier_proc(self, nextRow: int,
                              nextColumn: int,
                              step_cost: int,
                              path: list,
                              frontier: list,
                              compareFn):
        "Add node to frontier then sort the frontier"
        frontier.append((nextRow, nextColumn, step_cost, path))
        self.sortFrontier_proc(
            frontier,
            compareFn=compareFn
        )

    def pruneFrontier(self, frontier: list, possible_steps: int,
                      possible_energy: int):
        "Prune nodes in frontier using possible steps and energy"
        maxStepCost = possible_energy + possible_steps
        frontier = [
            node for node in frontier if (node[2] < maxStepCost and
                                          len(node[3]) < possible_steps
                                          )
        ]
        return frontier

    def checkNextStateAvailability(self,
                                   nextRow: int,
                                   nextColumn: int,
                                   explored: set,
                                   img: np.ndarray,
                                   possible_energy: int,
                                   path: list,
                                   frontier: list):
        "Check if next state is available for adding to frontier later on"
        if self.checkLimit(currentRow=nextRow,
                           currentColumn=nextColumn,
                           rownb=img.shape[0],
                           colnb=img.shape[1]) is False:
            return False
        if self.checkAvailable(currentRow=nextRow,
                               currentColumn=nextColumn,
                               img=img,
                               possible_energy=possible_energy,
                               path=path) is False:
            return False
        if (nextRow, nextColumn) in explored:
            return False
        if self.checkInFrontier(currentRow=nextRow,
                                currentColumn=nextColumn,
                                frontier=frontier):
            return False
        return True

    def generateNextStates4Frontier_proc(self, currentRow: int,
                                         currentColumn: int,
                                         img: np.ndarray,
                                         path: [],
                                         explored: set,
                                         possible_energy: int,
                                         frontier: list,
                                         oldStepCost: int,
                                         compareFn=lambda x: (x[2], len(x[3]))
                                         ):
        "generate next states and add it to frontier"
        for act in self.moves:
            nextRow, nextCol = self.moveOps[act](currentRow=currentRow,
                                                 currentColumn=currentColumn)
            if self.checkNextStateAvailability(nextRow=nextRow,
                                               nextColumn=nextCol,
                                               explored=explored,
                                               img=img,
                                               path=path,
                                               possible_energy=possible_energy,
                                               frontier=frontier):
                self.addNextStateWithCost2Frontier_proc(
                    currentRow=currentRow,
                    currentColumn=currentColumn,
                    nextRow=nextRow,
                    nextColumn=nextCol,
                    path=path,
                    img=img,
                    frontier=frontier,
                    compareFn=compareFn,
                    oldStepCost=oldStepCost)

    def filterMinZoneMoveWithCenterGoalDist(self, goalDist: dict,
                                            minimalZoneMove: list):
        "filter minimal zone move list with distance to goal in center"
        minimalZoneMove = [
            (r, c, m, d, g, goalDist[g]) for r, c, m, d, g in minimalZoneMove
        ]
        minimalZoneMove.sort(key=lambda x: x[5])
        min2centerGoal = minimalZoneMove[0][5]
        minimalZoneMove = [
            zm for zm in minimalZoneMove if zm[5] == min2centerGoal
        ]
        return minimalZoneMove

    def getPointZone(self, zoneArray: [np.ndarray],
                     currentRow: int,
                     currentColumn: int):
        "Get the zone that includes the currentColumn and row"
        assert isinstance(zoneArray, list)
        arr = [currentRow, currentColumn]
        pointZones = []
        for zone in zoneArray:
            hasArray = any(np.equal(zone, arr).all(axis=1))
            if hasArray:
                pointZones.append(zone)
        return pointZones

    def generateGoalStatesDistances(self, img: np.ndarray):
        "Generate valid goal states, central state, and distances to center"
        goals = [(img.shape[0]-1, col) for col in range(img.shape[1])]  # 7
        center_goal = (img.shape[0]-1, img.shape[1]//2)
        goalDistance = {goal: abs(goal[1] - center_goal[1]) for goal in goals}
        return goals, center_goal, goalDistance

    def filterMinimalZoneMove(self, minimalZoneMove: list,
                              goalDistance: dict):
        "Arrange minimal zone move to have a single element"
        minimalZoneMove.sort(key=lambda x: x[3])
        dist = minimalZoneMove[0][3]
        minimalZoneMove = [
            el for el in minimalZoneMove if el[3] == dist
        ]
        if len(minimalZoneMove) > 1:
            minimalZoneMove = self.filterMinZoneMoveWithCenterGoalDist(
                goalDist=goalDistance,
                minimalZoneMove=minimalZoneMove)
            if len(minimalZoneMove) > 1:
                minimalZoneMove = [minimalZoneMove[0][:5]]
        return minimalZoneMove

    def search_best_path(self, img: np.ndarray,
                         fromRow: int,
                         fromColumn: int,
                         ):
        """
        Search best path for marking the seam

        The algorithm is mix between a*star path search
        with upper bound pruning implemented in availability function

        Here is the algorithm steps:
        1. Create an explored set containing explored states
        2. Create a frontier containing explored nodes
        3. Create pruning criterias for frontier
        4. Obtain the indices of zero energy zones from the image
        5. Regroup zero energy indices into moves so that each move
        is associated to a zone array.
            5.1 Create zone array
        A zone array contains 1 or multiple zones where a move can be done
        with zero energy.
        For example for any diagonal left to right movement we would have
        zone array like:
        [np.array([[0,1], [1, 2]]), np.array([[53, 54], [54, 55], [55, 56]])]
        the array has 2 zones in it and it can be associated to moves
        leftUp, rightDown
            5.2 Create a zones matrix, which contains all the coordinates
            of all the zones in a zone array

        6. Obtain an initial state after advancing with a greedy move
            the initial state contains, row, column, 0 cost and empty path
        7. Create goal states, which is basically arriving at the end of the
        image
        8. Add initial state to frontier
        9. Loop until and empty frontier or until finding a path that attains
        to any of the goal state.
        Inside the loop:
        10. Pop the last added state with its cost and path from the frontier
        11. Add the state to its path
        12. Add the state to explored states
        13. Check if the current path attains a goal state, if it does
        terminate the loop and return the path
        14. Check if the last state is in zero energy zone:
            if it does
            14.1 Check if zone matrix of a move contains the state
                if not:
                    quit from branch and proceed to step 15
            14.2 Create minimalZoneMove list
            14.3 for all zone matrices associated to a move
                14.3.1 Obtain the zone that contains the state
                14.3.2 Compute the closest point and its distance in
                       the zone to any of the goal states.
                14.3.3 add the zone, closest point, goal state and move
                       to minimalZoneMove list
            14.4 Sort the minimalZoneMove list from shortest distance to
                 longer distance
            14.5 Filter the minimalZoneMove list for elements that contain
            a distance longer than the shortest distance
            14.6 Check if minimalZoneMove list contains more than 1 element
                if it does:
                14.6.1 create minimalZoneMove2 list
                14.6.2 for each element of minimalZoneMove list
                14.6.3 compute its goal state's distance with goal state in
                       center
                14.6.4 add new distance value along with element
                       to minimalZoneMove2
                14.6.5 filter minimalZoneMove2 list using the shortest new
                       distance
                14.6.6 check if minimalZoneMove2 list contains more than 1
                       element
                       if it does:
                       use an external criteria to filter out all but a single
                       value in it
                14.6.7 remove newly computed distance from the element
                14.6.8 assign minimalZoneMove2 to minimalZoneMove
            14.7 Obtain from the only element of minimalZoneMove: zone, move,
                 closest point
            14.8 Compute the step cost from the current state to the closest
                 point of the move within the zone
            14.9 Add points between the current state and the closest point
                to the path, and increment the overall step cost
            14.10 Add closest point, new step cost and path to frontier.
            14.11 sort frontier by distance to goal states and overall step
            cost of paths
        15. Forall moves that are doable:
            15.1 Obtain next state
            15.2 Check whether the given state is available
            15.3 If available state is not in explored set and frontier
            15.4 Obtain the step cost for the new state
            15.5 Add the new state, step cost and path to frontier
            15.6 Sort frontier by distance to goal states and overall step
            costs of paths
        """
        explored = set()  # step 1
        frontier = []  # step 2

        # 3 pruning criteria
        possible_energy = 255 * img.shape[0]
        possible_steps = max(img.shape) * 2
        def compareFn(x): return (x[2], len(x[3]))
        # 4 zero energy zones
        valzone = self.getInferior2MeanZones(img)
        # 5 zero indice regroup
        moveZones = self.getMoveZonesFromValZone(valZone=valzone)
        initial_state = [0, 0, 0, []]  # 6
        (goals,
         center_goal,
         goalDistance) = self.generateGoalStatesDistances(img)  # 7
        frontier.append(initial_state)  # 8
        while frontier:  # 9
            row, col, step_cost, path = frontier.pop()  # 10
            path_copy = path.copy()
            path_copy.append((row, col))  # 11
            explored.add((row, col))  # 12
            self.path = path_copy
            if self.testGoalCoordinate(currentRow=row, currentColumn=col,
                                       goals=goals):  # 13
                return path_copy
            #
            if self.checkPointInCoordZone(coordzone=valzone, currentRow=row,
                                          currentColumn=col):
                #
                minimalZoneMove = self.populateMinimalMoveZone(
                    moveZones, currentRow=row, currentColumn=col,
                    goals=goals)
                if len(minimalZoneMove) == 0:
                    self.generateNextStates4Frontier_proc(
                        currentRow=row,
                        currentColumn=col,
                        img=img,
                        explored=explored,
                        frontier=frontier,
                        path=path_copy,
                        possible_energy=possible_energy,
                        oldStepCost=step_cost,
                        compareFn=compareFn
                    )
                    self.pruneFrontier(frontier,
                                       possible_steps=possible_steps,
                                       possible_energy=possible_energy)
                    continue
                #
                minimalZoneMove = self.filterMinimalZoneMove(
                    minimalZoneMove,
                    goalDistance
                )
                (closestRow, closestCol, zoneMove,
                 dist, goal) = minimalZoneMove.pop()
                (nextRow, nextCol,
                 nextCost, nextPath) = self.jump2PointWithCost(
                    img=img,
                    fromRow=row,
                    fromColumn=col,
                    toRow=closestRow,
                    toColumn=closestCol,
                    moveDirection=zoneMove
                )
                if self.checkNextStateAvailability(
                        nextRow=nextRow,
                        nextColumn=nextCol,
                        explored=explored,
                        img=img,
                        possible_energy=possible_energy,
                        path=path_copy,
                        frontier=frontier):
                    nextPath = path_copy + nextPath
                    nextPathCost = self.getPathCost(img, nextPath)
                    nextCost += step_cost + nextPathCost
                    self.addNode2Frontier_proc(nextRow=nextRow,
                                               nextColumn=nextCol,
                                               step_cost=nextCost,
                                               frontier=frontier,
                                               compareFn=compareFn)
                    self.pruneFrontier(frontier,
                                       possible_steps=possible_steps,
                                       possible_energy=possible_energy)
            else:
                self.generateNextStates4Frontier_proc(
                    currentRow=row,
                    currentColumn=col,
                    img=img,
                    explored=explored,
                    frontier=frontier,
                    possible_energy=possible_energy,
                    oldStepCost=step_cost,
                    path=path_copy)
                self.pruneFrontier(frontier,
                                   possible_steps=possible_steps,
                                   possible_energy=possible_energy)

    def point2pointSearch_full(self, img: np.ndarray,
                               fromRow: int,
                               fromColumn: int,
                               toRow: int,
                               toColumn: int):
        "Search the least cost path from a start point to an end point"
        explored = set()  # step 1
        frontier = []  # step 2
        # 3
        possible_energy = 255 * (toRow - fromRow)
        possible_steps = max(toRow, toColumn) * 2

        initial_state = [fromRow, fromColumn, 0, []]  # 6
        goals = [(toRow, toColumn)]
        frontier.append(initial_state)  # 8
        while frontier:
            row, col, step_cost, path = frontier.pop()  # 10
            path_copy = path.copy()
            path_copy.append((row, col))  # 11
            explored.add((row, col))  # 12
            if self.testGoalCoordinate(row, col, goals):
                return path_copy
            self.generateNextStates4Frontier_proc(
                currentRow=row,
                currentColumn=col,
                img=img,
                explored=explored,
                frontier=frontier,
                possible_energy=possible_energy,
                oldStepCost=step_cost,
                path=path_copy)
            self.pruneFrontier(frontier,
                               possible_steps=possible_steps,
                               possible_energy=possible_energy)

    def point2pointsSearch_full(self, img: np.ndarray,
                                fromRow: int,
                                fromColumn: int,
                                endPoints: list):
        "point2point search adapted to multiple points"
        explored = set()  # step 1
        frontier = []  # step 2
        # 3
        maxGoalRow = max([p[0] for p in endPoints])
        maxGoalCol = max([p[1] for p in endPoints])
        possible_energy = 255 * (maxGoalRow - fromRow)
        possible_steps = max(maxGoalRow, maxGoalCol) * 2

        initial_state = [fromRow, fromColumn, 0, []]  # 6
        goals = endPoints
        frontier.append(initial_state)  # 8
        while frontier:
            row, col, step_cost, path = frontier.pop()  # 10
            path_copy = path.copy()
            path_copy.append((row, col))  # 11
            explored.add((row, col))  # 12
            if self.testGoalCoordinate(row, col, goals):
                return path_copy
            self.generateNextStates4Frontier_proc(
                currentRow=row,
                currentColumn=col,
                img=img,
                explored=explored,
                frontier=frontier,
                possible_energy=possible_energy,
                oldStepCost=step_cost,
                path=path_copy)
            self.pruneFrontier(frontier,
                               possible_steps=possible_steps,
                               possible_energy=possible_energy)

    def findSeamCandidate(self, img: np.ndarray):
        """
        Find a seam candidate in the image

        Algorithm is the following:

        - Compute greedy zone (0 energy zone)
        - compute inferior to mean zone ( low energy zone )
        - regroup both zone types into move directions
        - perform greedy move inside 0 energy zone
        - perform point2point search inside inferior zone
        - perform point1points search outside of these zones
        """
        pass

    def min_seam_with_cumsum(self,
                             img: np.ndarray):
        r, c = img.shape[:2]
        M = img.copy()
        backtrack = np.zeros_like(M, dtype=np.int)
        M = np.cumsum(M, axis=0)

        for i in range(1, r):
            for j in range(0, c):
                # Handle the left edge of the image,
                # to ensure we don't index -1
                if j == 0:
                    offset_value = j
                    colrange = j + 2
                    maprow = M[i - 1, offset_value:colrange]
                    min_energy_indx_in_row = np.argmin(maprow)
                    min_energy_indx = min_energy_indx_in_row + offset_value
                    backtrack[i, j] = min_energy_indx
                    min_energy = M[i - 1, min_energy_indx]
                else:
                    offset_value = j - 1
                    colrange = j + 2
                    maprow = M[i - 1, offset_value:colrange]
                    min_energy_indx_in_row = np.argmin(maprow)
                    min_energy_indx = min_energy_indx_in_row + offset_value
                    backtrack[i, j] = min_energy_indx
                    min_energy = M[i - 1, min_energy_indx]

                M[i, j] += min_energy

        return M, backtrack

    def minimum_seam(self,
                     img: np.ndarray([], dtype=np.uint8),
                     emap=None):
        "Compute the minimum seam"
        r, c, _ = img.shape

        # if the energy map is already calculated
        if emap is not None:
            energy_map = emap
        else:
            energy_map = self.calc_energy(img)
        #
        coordinate_path = self.search_best_path(img=energy_map)
        return coordinate_path

    def mark_column(self, img: np.ndarray,
                    emap=None,
                    mark_color=[255, 120, 120]):
        "mark column implemented with best search path"
        imcp = img.copy()
        mask = np.zeros_like(imcp)
        coordpath = self.minimum_seam(imcp, emap)
        for coord in coordpath:
            row, col = coord
            imcp[row, col] = mark_color
            mask[row, col] = mark_color
        return imcp, mask


class SeamMarker(SeamFuncs):
    def __init__(self,
                 img: np.ndarray([], dtype=np.uint8),
                 plist=[],
                 thresh=10,
                 direction='down'):
        super().__init__()
        self.img = img
        self.plist = plist
        self.direction = direction
        self.thresh = thresh
        self.mark_color = [0, 255, 0]

    def expandPointCoordinate(self,
                              ubound: int,
                              coord: int,
                              thresh: int):
        "Expand the coordinate with ubound using given threshold"
        assert thresh <= 100 and thresh > 0
        assert isinstance(thresh, int) is True
        sliceAmount = ubound * thresh // 100
        sliceHalf = sliceAmount // 2
        coordBefore = coord - sliceHalf
        coordAfter = coord + sliceHalf
        if coordBefore < 0:
            coordBefore = 0
        if coordAfter >= ubound:
            coordAfter = ubound - 1
        return coordBefore, coordAfter

    def getColumnSliceOnPoint(self,
                              point: (int, int),
                              img: np.ndarray,
                              isUpTo: bool,
                              thresh: int):
        "Get column slice on point"
        col_nb = img.shape[1]
        pointCol = point[1]
        colBefore, colAfter = self.expandPointCoordinate(ubound=col_nb,
                                                         coord=pointCol,
                                                         thresh=thresh)
        if isUpTo is False:
            imgSlice = img[point[0]:, colBefore:colAfter]
        else:
            imgSlice = img[:point[0], colBefore:colAfter]
        #
        return imgSlice, (colBefore, colAfter)

    def getRowSliceOnPoint(self,
                           point: (int, int),
                           img: np.ndarray,
                           isUpTo: bool,
                           thresh: int):
        "Get row slice on point"
        row_nb = img.shape[0]
        pointRow = point[0]
        rowBefore, rowAfter = self.expandPointCoordinate(ubound=row_nb,
                                                         coord=pointRow,
                                                         thresh=thresh)
        if isUpTo is False:
            imgSlice = img[rowBefore:rowAfter, point[1]:]
        else:
            imgSlice = img[rowBefore:rowAfter, :point[1]]
        #
        return imgSlice, (rowBefore, rowAfter)

    def sliceOnPoint(self, img: np.ndarray([], dtype=np.uint8),
                     point: (int, int),
                     isUpTo=False,
                     colSlice=False,
                     thresh=3) -> np.ndarray([], dtype=np.uint8):
        """
        Slice based on the point with given threshold as percent

        Description
        ------------
        We take the point as a center. Then we calculate
        the amount of slicing based on the threshold.
        Threshold is a percent value. Thus the sliced amount
        is relative to the image shape.
        Once the slicing amount is computed, we slice the given
        amount from the image. The point should be at the center
        of a side of the sliced area.

        Parameters
        ------------
        img: np.ndarray([], dtype=np.uint8)

        point: (int, int)
            coordinate of row, column for the point


        isUpTo: boolean
            determines whether we slice up to the point or from point
            onwards

        thresh: int
            threshold value for the amount of slicing. It should be
            between 0 - 100.

        Return
        -------

        imgSlice: np.ndarray(dtype=np.uint8)
        """
        # make sure threshold is a percent and an integer
        assert thresh <= 100 and thresh > 0
        assert isinstance(thresh, int) is True

        if colSlice is True:
            imgSlice, (before, after) = self.getColumnSliceOnPoint(point,
                                                                   img,
                                                                   isUpTo,
                                                                   thresh)
        else:
            imgSlice, (before, after) = self.getRowSliceOnPoint(point,
                                                                img,
                                                                isUpTo,
                                                                thresh)

        return imgSlice, (before, after)

    def addColumnSlice2Image(self, image: np.ndarray,
                             point: (int, int), beforeAfterCoord: (int, int),
                             imgSlice: np.ndarray, isUpTo: bool):
        "Add column slice 2 image"
        imcp = image.copy()
        before, after = beforeAfterCoord
        if isUpTo is False:
            imcp[point[0]:, before:after] = imgSlice
        else:
            imcp[:point[0], before:after] = imgSlice
        return imcp

    def addRowSlice2Image(self, image: np.ndarray,
                          point: (int, int), beforeAfterCoord: (int, int),
                          imgSlice: np.ndarray, isUpTo: bool):
        "Row slice 2 image"
        imcp = image.copy()
        before, after = beforeAfterCoord
        if isUpTo is False:
            imcp[before:after, point[1]:] = imgSlice
        else:
            imcp[before:after, :point[1]] = imgSlice
        return imcp

    def addPointSlice2Image(self,
                            img: np.ndarray([], dtype=np.uint8),
                            point: (int, int),  # y, x
                            beforeAfterCoord: (int, int),
                            colSlice: bool,
                            imgSlice: np.ndarray([], dtype=np.uint8),
                            isUpTo: bool,
                            ):
        "Add sliced zone back to image"
        # pdb.set_trace()
        if colSlice is True:
            imcp = self.addColumnSlice2Image(image=img,
                                             point=point,
                                             beforeAfterCoord=beforeAfterCoord,
                                             imgSlice=imgSlice,
                                             isUpTo=isUpTo)
        else:
            imcp = self.addRowSlice2Image(image=img,
                                          point=point,
                                          beforeAfterCoord=beforeAfterCoord,
                                          imgSlice=imgSlice,
                                          isUpTo=isUpTo)
        #
        return imcp

    def getMarkCoordinates(self,
                           markedMask: np.ndarray([], dtype=np.bool)):
        "Get marked coordinates from the mask"
        indexArray = markedMask.nonzero()
        indexArray = np.array(indexArray)
        indexArray = indexArray.T
        # indexArray[0] == [rowPosition, colPosition, colorPosition]
        return indexArray

    def sortCoords4Matching(self, coord1, coord2,
                            colSlice: bool):
        "Sort coordinates with respect to column slicing"
        if colSlice is True:
            # sort by y val
            ind1 = np.argsort(coord1[:, 0])
            coord1 = coord1[ind1]
            ind2 = np.argsort(coord2[:, 0])
            coord2 = coord2[ind2]
        elif colSlice is False:
            # sort by x val
            ind1 = np.argsort(coord1[:, 1])
            coord1 = coord1[ind1]
            ind2 = np.argsort(coord2[:, 1])
            coord2 = coord2[ind2]
        return coord1, coord2

    def getKeepAndLimitValues(self, coord1,
                              coord2, colSlice: bool,
                              isUpTo: bool):
        "Get keep value and limit values to generate fill values later on"
        if isUpTo is False:
            axval = 0
        else:
            axval = -1
        if colSlice is True:
            keepval = 1
            rangeval = 0
        else:
            keepval = 0
            rangeval = 1
        #
        COORD1KEEP = coord1[axval, keepval]
        COORD2KEEP = coord2[axval, keepval]
        coord1val = coord1[axval, rangeval]
        coord2val = coord2[axval, rangeval]
        if isUpTo is False:
            # so their first values should match
            # make sure coord1 is the one with smaller value
            # we'll prepend values to coord2 later on
            if coord1val >= coord2val:
                coord1, coord2 = coord2, coord1
                coord1val, coord2val = coord2val, coord1val
                COORD1KEEP, COORD2KEEP = COORD2KEEP, COORD1KEEP
        else:
            # their last values should match
            # make sure coord2 is the one with smaller value
            # making it shorter
            # we'll append values to coord2 later on
            if coord2val >= coord1val:
                coord1, coord2 = coord2, coord1
                coord1val, coord2val = coord2val, coord1val
                COORD1KEEP, COORD2KEEP = COORD2KEEP, COORD1KEEP
        #
        return (coord1, coord2,
                COORD1KEEP, COORD2KEEP,
                coord1val, coord2val)

    def prepCoords2Matching(self, coord1, coord2,
                            colSlice: bool,
                            isUpTo: bool):
        "Prepare coordinates to matching"
        assert isinstance(colSlice, bool)
        coord1, coord2 = self.sortCoords4Matching(coord1,
                                                  coord2,
                                                  colSlice)
        assert isinstance(isUpTo, bool)
        (coord1, coord2,
         COORD1KEEP, COORD2KEEP,
         coord1val, coord2val) = self.getKeepAndLimitValues(coord1,
                                                            coord2, colSlice,
                                                            isUpTo)
        if isUpTo is False:
            fillvals = [i for i in range(coord2val-1,  # since we prepend
                                         # this array later on
                                         coord1val-1, -1)]
        else:
            fillvals = [i for i in range(coord2val+1, coord1val+1, 1)]
        #
        return (coord1, coord2,
                COORD1KEEP, COORD2KEEP,
                coord1val, coord2val,
                fillvals)

    def matchMarkCoordPairLength(self, coord1, coord2,
                                 colSlice: bool,
                                 isUpTo: bool):
        """Match mark coordinate pairs

        Purpose
        ---------

        Matches the coordinate pairs that start from different points.

        Description
        ------------

        The logic is simple. If we are dealing with a column slice, then
        the y values should match, since we need to have equal column length
        to fill the column mask later on. That's why we sort the coordinates
        by y value at first, to make sure that their top points are closest
        to each other.

        If we are dealing with a row slice, then the x values match since
        we need to have equal line length to fill the line mask later on

        We simply keep the last first value of the unmatched axis of
        the coordinate array. You can think of matching axes as drawing a
        parallel line from the point where a coordinate array falls short,
        up until it matches the other coordinate array's limit.

        """
        assert coord1.shape[1] == 2 and coord2.shape[1] == 2
        # col slice determines the axis of match
        (coord1, coord2,
         COORD1KEEP, COORD2KEEP,
         coord1val, coord2val,
         fillvals) = self.prepCoords2Matching(
             coord1, coord2, colSlice, isUpTo)
        for i in fillvals:
            if colSlice is True:  # column slice
                if isUpTo is False:
                    coord2 = np.insert(coord2, 0, [i, COORD2KEEP], axis=0)
                else:
                    coord2 = np.insert(coord2, coord2.shape[0],
                                       [i, COORD2KEEP], axis=0)
            else:
                if isUpTo is False:
                    coord2 = np.insert(coord2, 0, [COORD2KEEP, i], axis=0)
                else:
                    coord2 = np.insert(coord2, coord2.shape[0],
                                       [COORD2KEEP, i], axis=0)
        return coord1, coord2

    def swapAndSliceMarkCoordPair(self, markCoord1, markCoord2,
                                  image, colSlice: bool) -> np.ndarray:
        "Slice image using mark coordinate pair"
        imcp = np.copy(image)
        mask = np.zeros_like(imcp)
        if colSlice is True:
            # then comparing x values
            fsum = np.sum(markCoord1[:, 1] - markCoord2[:, 1], dtype=np.int)
            if fsum >= 0:
                markCoord1, markCoord2 = markCoord2, markCoord1
            #
            # pdb.set_trace()
            for i in range(markCoord1.shape[0]):
                startx = markCoord1[i, 1]
                yval = markCoord1[i, 0]
                endx = markCoord2[i, 1]
                mask[yval, startx:endx] = imcp[yval, startx:endx]
        else:
            # pdb.set_trace()
            fsum = np.sum(markCoord1[:, 0] - markCoord2[:, 0], dtype=np.int)
            if fsum >= 0:
                markCoord1, markCoord2 = markCoord2, markCoord1
            #
            for i in range(markCoord1.shape[0]):
                xval = markCoord1[i, 1]
                starty = markCoord1[i, 0]
                endy = markCoord2[i, 0]
                mask[starty:endy, xval] = imcp[starty:endy, xval]
        #
        # pdb.set_trace()
        imslice = self.crop_zeros(mask)
        return imslice

    def sliceImageWithMarkCoordPair(self, image: np.ndarray,
                                    markCoord1: np.ndarray,
                                    markCoord2: np.ndarray,
                                    colSlice: bool,
                                    isUpTo: bool) -> np.ndarray:
        "Slice image with mark coordinate pair"
        imcp = np.copy(image)
        # pdb.set_trace()
        assert markCoord1.shape[1] == 2  # [y,x], [y2,x2], etc
        assert markCoord2.shape[1] == 2
        if markCoord1.shape[0] != markCoord2.shape[0]:
            # pdb.set_trace()
            markCoord1, markCoord2 = self.matchMarkCoordPairLength(markCoord1,
                                                                   markCoord2,
                                                                   colSlice,
                                                                   isUpTo)
        #
        imcp = self.swapAndSliceMarkCoordPair(markCoord1, markCoord2,
                                              imcp, colSlice)
        return imcp

    def crop_zeros(self, img: np.ndarray([], dtype=np.uint8)):
        "Crop out zeros from image sides"
        img_cp = img.copy()
        #
        image_col = img_cp.shape[1]
        image_row = img_cp.shape[0]
        #
        delete_list = []
        for col in range(image_col):
            if np.sum(img_cp[:, col], dtype="uint32") == 0:
                delete_list.append(col)
            #
        #
        img_cp = np.delete(arr=img_cp,
                           obj=delete_list,
                           axis=1)
        #
        delete_list = []
        #
        for row in range(image_row):
            if np.sum(img_cp[row, :], dtype="int32") == 0:
                delete_list.append(row)
            #
        img_cp = np.delete(arr=img_cp,
                           obj=delete_list,
                           axis=0)
        #
        return img_cp

    def getMarkCoordinates4Point(self, img: np.ndarray([], dtype=np.uint8),
                                 point1: (int, int),
                                 isUpTo: bool,
                                 colSlice: bool,
                                 thresh: int, mark_color: (int, int, int)):
        "Obtain mark coordinates from image"
        markedImage, mask, sliceImage, beforeAfter = self._markSeam4Point(
            img=img, point1=point1, isUpTo=isUpTo, colSlice=colSlice,
            thresh=thresh, mark_color=mark_color)

        maskImage = np.zeros_like(img, dtype=np.bool)
        maskImage1 = self.addPointSlice2Image(img=maskImage, point=point1,
                                              beforeAfterCoord=beforeAfter,
                                              imgSlice=mask,
                                              colSlice=colSlice,
                                              isUpTo=isUpTo)

        # obtaining mark coordinates from image mask
        m1index = self.getMarkCoordinates(maskImage1)
        if colSlice is False:
            m1index = np.rot90(m1index, 3, (0, 1))
        return m1index

    def getMarkImageWithCoordinates(self, img, point1: (int, int),
                                    isUpTo: bool,
                                    colSlice: bool,
                                    thresh: int,
                                    mark_color: (int, int, int)):
        "Obtain mark coordinates and marked image"
        markedImage, mask, sliceImage, beforeAfter = self._markSeam4Point(
            img=img, point1=point1, isUpTo=isUpTo, colSlice=colSlice,
            thresh=thresh, mark_color=mark_color)
        maskImage = np.zeros_like(img, dtype=np.bool)
        maskImage1 = self.addPointSlice2Image(img=maskImage, point=point1,
                                              beforeAfterCoord=beforeAfter,
                                              imgSlice=mask,
                                              colSlice=colSlice,
                                              isUpTo=isUpTo)
        markedFullImage = self.addPointSlice2Image(
            img=img.copy(),
            point=point1,
            beforeAfterCoord=beforeAfter,
            imgSlice=markedImage,
            colSlice=colSlice,
            isUpTo=isUpTo)

        # obtaining mark coordinates from image mask
        m1index = self.getMarkCoordinates(maskImage1)
        if colSlice is False:
            m1index = np.rot90(m1index, 3, (0, 1))
        return m1index, markedFullImage

    def _markSeam4Point(self, img: np.ndarray([], dtype=np.uint8),
                        point1: (int, int),
                        isUpTo: bool,
                        colSlice: bool,
                        thresh: int,
                        mark_color: (int, int, int)) -> np.ndarray:
        """
        Mark the seam for a given point

        Description
        ------------
        Simple strategy. We slice the image from the given point using
        a threshold value for the sliced area.
        Then mark the seam on that area.
        """
        imcp = img.copy()
        slice1 = self.sliceOnPoint(imcp, point1,
                                   thresh=thresh,
                                   colSlice=colSlice,
                                   isUpTo=isUpTo)
        sl1 = slice1[0]  # image slice
        ba1 = slice1[1]  # before, after coord
        if colSlice is True:
            m1, mask1 = self.mark_column(sl1, mark_color=mark_color)
        else:
            m1, mask1 = self.mark_row(sl1, mark_color=mark_color)
        # m1 == marked image
        return m1, mask1, sl1, ba1
        # adding marked masks back to the image mask

    def markSeam4Point(self, img: np.ndarray([], dtype=np.uint8),
                       point1: (int, int),
                       isUpTo: bool,
                       thresh: int,
                       colSlice: bool,
                       mark_color: (int, int, int)):
        "Mark seam for point"
        markedSlice, mask, sliceImage, beforeAfter = self._markSeam4Point(
            img=img, point1=point1, isUpTo=isUpTo, colSlice=colSlice,
            thresh=thresh, mark_color=mark_color)
        markedImage = self.addPointSlice2Image(img=img, point=point1,
                                               beforeAfterCoord=beforeAfter,
                                               imgSlice=markedSlice,
                                               colSlice=colSlice,
                                               isUpTo=isUpTo)
        return markedImage

    def makePairsFromPoints(self, plist: list,
                            colSlice: bool,
                            isXFirst=False):
        "Make pairs from points by ordering them according to x or y"
        if colSlice is True:
            if isXFirst is False:
                plist.sort(key=lambda p: p[1])
            else:
                plist.sort(key=lambda p: p[0])
        else:
            if isXFirst is False:
                plist.sort(key=lambda p: p[0])
            else:
                plist.sort(key=lambda p: p[1])
        #
        pairs = []
        for i in range(len(plist)):
            if i+1 < len(plist):
                p1 = plist[i]
                p2 = plist[i+1]
                pairs.append((p1, p2))
        return pairs

    def prepDirection(self, direction: str):
        "Prepare direction"
        colSlice = True
        isUpTo = False
        if direction == "down":
            colSlice = True
            isUpTo = False
        elif direction == "up":
            colSlice = True
            isUpTo = True
        elif direction == "right":
            colSlice = False
            isUpTo = False
        elif direction == "left":
            colSlice = False
            isUpTo = True

        return colSlice, isUpTo

    def prepImageWithParams(self, img, plist,
                            direction):
        "Prepare image and point list with respect to the direction"
        imcp = img.copy()
        if direction == 'right':
            # rotate points and image
            imcp = np.rot90(imcp, 1, (0, 1))
            plist = np.rot90(plist, 1, (0, 1))
            plist = [tuple(k) for k in plist.T.tolist()]
        elif direction == 'left':
            imcp = np.rot90(imcp, 1, (0, 1))
            plist = np.rot90(plist, 1, (0, 1))
            plist = [tuple(k) for k in plist.T.tolist()]

        colSlice, isUpTo = self.prepDirection(direction)
        return imcp, plist, isUpTo, colSlice

    def markPointSeam(self, img, point, direction="down",
                      mark_color=[0, 255, 0],
                      thresh=2):
        "Mark seam passes around the point region"
        colSlice, isUpTo = self.prepDirection(direction)
        markedImage = self.markSeam4Point(img.copy(), point, isUpTo, thresh,
                                          colSlice, mark_color)
        return markedImage

    def markPointSeamWithCoordinate(self, img, point, direction='down',
                                    mark_color=[0, 255, 0],
                                    thresh=2):
        "Get mark and coordinate"
        colSlice, isUpTo = self.prepDirection(direction)
        coord, markedImage = self.getMarkImageWithCoordinates(
            img, point, isUpTo, colSlice, thresh, mark_color)
        return markedImage, coord

    def markPointListSeam(self, img, plist: dict, mark_color=[0, 255, 0]):
        """
        Mark seam that passes through the regions of each point

        Description
        -------------

        We assume that each key value pair of the plist, contains
        the following pairs in their values:
        'threshold': int,
        'coordinates': (int, int),
        'x': int,
        'y': int,
        'direction': str
        'color': [int, int, int]
        """
        imcp = img.copy()
        for i, point in plist.items():
            direction = point['direction']
            thresh = point['threshold']
            point_coord = (point['y'], point['x'])
            imcp = self.markPointSeam(imcp,
                                      point_coord,
                                      direction=direction,
                                      thresh=thresh,
                                      mark_color=mark_color)
        #
        return imcp

    def getPointSeamCoordinate(self, img, point,
                               direction="down",
                               thresh=2, mark_color=[0, 255, 0]):
        "Get coordinates of the mark that passes around point region"
        colSlice, isUpTo = self.prepDirection(direction)
        coords = self.getMarkCoordinates4Point(img.copy(), point, isUpTo,
                                               colSlice, thresh, mark_color)
        return coords

    def getPointListSeamCoordinate(self, img, plist,
                                   direction="down",
                                   thresh=2, mark_color=[0, 255, 0]):
        "Get mark coordinates associated to each point region"
        imcp, plist, isUpTo, colSlice = self.prepImageWithParams(img,
                                                                 plist,
                                                                 direction)
        coords = []
        for point in plist:
            coord = self.getMarkCoordinates4Point(img, point, isUpTo,
                                                  colSlice, thresh, mark_color)
            coords.append({"point": point,
                           "markCoordinates": coord})

        return coords

    def makeCoordGroups(self, pointDataCoords: dict):
        "make coordinate groups based on carve directions"
        groups = {"up": [], "down": [], "left": [], "right": []}
        for i, pointData in pointDataCoords.items():
            groups[pointData['direction']].append(pointData)
        return groups

    def segmentImageWithPointListSeamCoordinate(self,
                                                coords: dict,
                                                image):
        "Segment the image using mark coordinates of a point list"
        groups = self.makeCoordGroups(coords)
        segment_groups = {"up": [], "down": [], "left": [], "right": []}
        for groupDirection, pointDataCoords in groups.items():
            # pdb.set_trace()
            colSlice, isUpTo = self.prepDirection(groupDirection)
            plist = [(pointData['y'],
                      pointData['x']) for pointData in pointDataCoords]
            pointCoordMap = {
                (pointData['y'],
                 pointData['x']
                 ): pointData['seamCoordinates'] for pointData in pointDataCoords
            }
            pairs = self.makePairsFromPoints(plist, colSlice,
                                             isXFirst=False)
            segments = []
            for pair in pairs:
                point1 = pair[0]
                point2 = pair[1]
                coord1 = pointCoordMap[point1]
                coord2 = pointCoordMap[point2]
                segment = self.sliceImageWithMarkCoordPair(image, coord1,
                                                           coord2, colSlice,
                                                           isUpTo)
                segments.append(segment)
            #
            segment_groups[groupDirection] = segments
        #
        return segment_groups

    def segmentPageWithPoints(self, img: np.ndarray([], dtype=np.uint8),
                              plist: [],
                              direction='down',  # allowed values
                              # down/up/left/right
                              mark_color=[0, 255, 0],
                              thresh=2):
        assert thresh >= 0 and thresh <= 100

        if (direction != 'down' and
            direction != 'up' and
            direction != 'left' and
                direction != 'right'):
            raise ValueError(
                'unknown direction, list of known directions {0}'.format(
                    str(['down', 'up', 'left', 'right'])
                )
            )
        #
        imcp, plist, isUpTo, colSlice = self.prepImageWithParams(img,
                                                                 plist,
                                                                 direction)
        # let's make the point pairs
        pairs = self.makePairsFromPoints(plist, colSlice)
        segments = [
            self.getSegmentFromPoints(imcp,
                                      isUpTo=isUpTo,
                                      thresh=thresh,
                                      mark_color=mark_color,
                                      colSlice=colSlice,
                                      point1=pair[0],
                                      point2=pair[1])
            for pair in pairs
        ]
        return segments

    def segmentWithPoints(self):
        return self.segmentPageWithPoints(img=self.img,
                                          plist=self.plist,
                                          thresh=self.thresh,
                                          mark_color=self.mark_color,
                                          direction=self.direction)
