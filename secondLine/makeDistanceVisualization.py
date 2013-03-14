'''
Created on Feb 25, 2013

@author: surchs
'''
import numpy as np
from matplotlib import pyplot as plt


def loadNumpyTextFile(pathToNumpyTextFile):
    numpyTextFile = np.loadtxt(pathToNumpyTextFile)

    return numpyTextFile


def getUniqueMatrixElements(squareMatrix):
    # When passed a square similarity matrix, returns the lower triangle of
    # said matrix as a vector of appended rows
    if not squareMatrix.shape[0] == squareMatrix.shape[1]:
        raise Exception('Your matrix of shape ' + str(squareMatrix.shape)
                        + ' is not symmetrical')
    # Make a mask for the lower triangle of the matrix
    mask = np.ones_like(squareMatrix)
    mask = np.tril(mask, -1)
    # Mask the matrix to retrieve only the lower triangle
    uniqueElements = squareMatrix[mask == 1]

    return uniqueElements


def getConnectivityDistances(connectivityAge, distances):
    '''
    Returns the distances of positive connections and negative connections
    as a vector
    '''
    # First normalize the connectivity
    # connectivityAge = np.arctanh(connectivityAge)
    # Find connections greater than 0
    positiveConnectivityIndex = connectivityAge > 0
    # Find connections less than 0
    negativeConnectivityIndex = connectivityAge < 0
    # Slice distances for positive connections
    positiveEffectDistances = distances[positiveConnectivityIndex]
    # Slice distances for negative connections
    negativeEffectDistances = distances[negativeConnectivityIndex]
    if (len(positiveEffectDistances) == 0 or len(negativeEffectDistances) == 0):
        raise Exception('No distances passed the thresholding')
    else:
        print('# positive distances ' + str(len(positiveEffectDistances)))
        print('# negative distances ' + str(len(negativeEffectDistances)))

    return positiveEffectDistances, negativeEffectDistances


def plotDistances(posDistances, negDistances, title):
    plt.hist(posDistances, bins=20,
             color='b', label='age pos')
    plt.hist(negDistances, bins=20,
             color='r', alpha=0.5, label='age neg')
    plt.title(title)
    plt.xlabel('distance in volume elements')
    plt.ylabel('frequency of distance')
    plt.legend()
    plt.show()


def saveNumpyTextFile(outputFilePath, outputMatrix):
    np.savetxt(outputFilePath, outputMatrix, fmt='%.12f')
    status = 'cool'

    return status


def Main():
    # Define inputs
    pathToAgeConnectivtyMatrix = '/home2/surchs/secondLine/correlation/abide/dos160/glm_thresholded_matrix_glob_c.txt'
    pathToDistancesMatrix = '/home2/surchs/secondLine/roiDistances/dos160wave_distances.txt'

    # Define Outputs
    pathToPositiveAgeDistances = '/home2/surchs/secondLine/correlation/group_distances_age_pos_dos.txt'
    pathToNegativeAgeDistances = '/home2/surchs/secondLine/correlation/group_distances_age_neg_dos.txt'

    # Read inputs
    ageConnectivityMatrix = loadNumpyTextFile(pathToAgeConnectivtyMatrix)
    distancesMatrix = loadNumpyTextFile(pathToDistancesMatrix)

    # Get the unique elements in the connectivity and distance matrix
    uniqueConnections = getUniqueMatrixElements(ageConnectivityMatrix)
    uniqueDistances = getUniqueMatrixElements(distancesMatrix)

    # Get the distances
    (positiveEffectDistances,
     negativeEffectDistances) = getConnectivityDistances(uniqueConnections,
                                                         uniqueDistances)

    # Plot the raw, unthresholded results as histograms
    plotDistances(positiveEffectDistances, negativeEffectDistances, 'distances')
    # Save the results
    status = saveNumpyTextFile(pathToPositiveAgeDistances,
                               positiveEffectDistances)
    print('positive distances say ' + status)
    status = saveNumpyTextFile(pathToNegativeAgeDistances,
                               negativeEffectDistances)
    print('negative distances say ' + status)


if __name__ == '__main__':
    Main()
