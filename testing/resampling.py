'''
Created on Nov 21, 2012

@author: sebastian
'''
import numpy as np


def Main():
    firstVec = np.arange(9)
    origin = firstVec.reshape(3, 3)
    factor = 3
    originShape = origin.shape
    print origin

    targetShape = tuple(i * factor for i in originShape)
    target = np.zeros(targetShape)

    runRow = 0
    for row in origin:
        runCol = 0
        for cell in row:
            # get to new matrix
            target[runRow * factor:runRow * factor + factor,
                   runCol * factor: runCol * factor + factor] = cell

            runCol += 1
        runRow += 1

    print target


if __name__ == '__main__':
    Main()
    pass
