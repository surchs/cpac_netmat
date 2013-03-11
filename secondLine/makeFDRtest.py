'''
Created on Mar 4, 2013

@author: surchs
'''
import numpy as np


def loadNumpyTextFile(pathToNumpyTextFile):
    numpyTextFile = np.loadtxt(pathToNumpyTextFile)

    return numpyTextFile


def getUniqueMatrixElements(squareMatrix):
    # When passed a square similarity matrix, returns the lower triangle of
    # said matrix as a vector of appended rows
    if not squareMatrix.shape[0] == squareMatrix.shape[1]:
        print('Your matrix of shape ' + str(squareMatrix.shape)
              + ' is not symmetrical')
        uniqueElements = squareMatrix.flatten()
    else:
        # Make a mask for the lower triangle of the matrix
        mask = np.ones_like(squareMatrix)
        mask = np.tril(mask, -1)
        # Mask the matrix to retrieve only the lower triangle
        uniqueElements = squareMatrix[mask == 1]

    return uniqueElements


def computeFDR(pValueVector, alpha):
    # This returns a new thresholded p value
    # Sort the p values by size, beginning with the smallest
    sortedP = np.sort(pValueVector)
    # Reverse sort the p-values, so the first one is the biggest
    reverseP = sortedP[::-1]
    # Get the number of p-values
    numP = float(len(reverseP))
    # Create a vector designating the position of the reverse sorted p-values
    # in the sorted p vector (e.g. the first reverse sorted p-value will have
    # the index numP because it would be the last entry in the sorted vector)
    indexP = np.arange(numP, 0, -1)
    # Create test vector of (index of p value / number of p values) * alpha
    test = indexP / numP * alpha
    # Check where p-value <= test
    testIndex = np.where(reverseP <= test)
    if testIndex[0].size == 0:
        print('None of you p values pass FDR correction')
        pFDR = 0
    else:
        # Get the first p value that passes the criterion
        pFDR = reverseP[np.min(testIndex)]
        print('FDR corrected p value for alpha of ' + str(alpha) + ' is '
              + str(pFDR)
              + '\n' + str(testIndex[0].size) + ' out of '
              + str(int(numP)) + ' p-values pass this threshold')

    return pFDR


def thresholdMatrix(pValueMatrix, pThresh):
    thresholdMatrix = np.zeros_like(pValueMatrix, dtype=int)
    threshPMatrix = np.zeros_like(pValueMatrix, dtype=int)
    threshPMatrix[pValueMatrix <= pThresh] = pValueMatrix[pValueMatrix <= pThresh]
    thresholdMatrix[pValueMatrix <= pThresh] = 1

    return thresholdMatrix, threshPMatrix


def quickThreshold(datavector, alpha):
    index = datavector < alpha
    print(index)

    return index


def saveNumpyTextFile(outputFilePath, outputMatrix):
    np.savetxt(outputFilePath, outputMatrix, fmt='%.12f')


def Main():
    # Define inputs
    pathToPvaluesMatrix = '/home2/surchs/secondLine/correlation/pvalue_matrix.txt'
    pathToFDRuniqueP = '/home2/surchs/secondLine/correlation/fdr_unique_p.txt'
    alpha = 0.05

    # Define Outputs
    pathToThresholdedPMatrix = '/home2/surchs/secondLine/correlation/pThreshMat.txt'
    pathToMaskMatrix = '/home2/surchs/secondLine/correlation/PMaskMatrix.txt'
    pathToUniquePValues = '/home2/surchs/secondLine/correlation/uniqePValues.txt'

    # Get the matrix
    pValueMatrix = loadNumpyTextFile(pathToPvaluesMatrix)
    fdrUniqueP = loadNumpyTextFile(pathToFDRuniqueP)

    # Get the pvalues
    pvalues = getUniqueMatrixElements(pValueMatrix)

    # Compute the FDR cutoff
    pThresh = computeFDR(pvalues, alpha)

    # Short shoutout to the people out there
    print('I got a p threshold of: ' + str(pThresh) + ' for alpha: '
          + str(alpha)
          + '\nThe number of p-values is ' + str(len(pvalues)))

    # Threshold the pvalue Matrix and return a mask matrix
    (threshMatrix, threshPMatrix) = thresholdMatrix(pValueMatrix, pThresh)


    print(str(len(pvalues)) + ' ' + str(len(fdrUniqueP)))
    # Compare the results
    # First threshold based on my FDR
    myFDRVector = quickThreshold(pvalues, pThresh)
    # Next, use the fdr corrected values
    RFDRVector = quickThreshold(fdrUniqueP, alpha)
    # Get the lengths for comparison
    myFDRlen = len(myFDRVector)
    rFDRlen = len(RFDRVector)

    # Shoutout
    print('myFDR: ' + str(myFDRlen) + ' / '
          + str(np.sum(myFDRVector)) + ' : ' + str(pvalues[myFDRVector]))
    print('rFDR: ' + str(rFDRlen) + ' / '
          + str(np.sum(RFDRVector)) + ' : ' + str(pvalues[RFDRVector]))

'''
    # Save the outputs
    saveNumpyTextFile(pathToThresholdedPMatrix, threshPMatrix)
    saveNumpyTextFile(pathToMaskMatrix, threshMatrix)
    saveNumpyTextFile(pathToUniquePValues, pvalues)
'''


if __name__ == '__main__':
    Main()
