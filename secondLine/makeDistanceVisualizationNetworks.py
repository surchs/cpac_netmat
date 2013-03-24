'''
Created on Feb 25, 2013

@author: surchs
'''
import gzip
import cPickle
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt


def loadNumpyTextFile(pathToNumpyTextFile):
    numpyTextFile = np.loadtxt(pathToNumpyTextFile)

    return numpyTextFile


def loadNiftiImage(pathToNiftiFile):
    image = nib.load(pathToNiftiFile)
    data = image.get_data()

    return image, data


def loadArchive(pathToArchive):
    f = gzip.open(pathToArchive)
    archive = cPickle.load(f)

    return archive


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


def threshold(inputMatrix, thresh=0):
    # Prepare empty matrix
    emptyMatrix = np.zeros_like(inputMatrix, dtype=int)
    # Set empty matrix to one where input passes threshold
    emptyMatrix[inputMatrix < thresh] = 1
    thresholdedMatrix = emptyMatrix == 1

    return thresholdedMatrix


def getConnectivityDistances(connectivityAge, distances, thresholded):
    '''
    Returns the distances of positive connections and negative connections
    as a vector
    '''
    # Define threshold
    thresh = 0
    # First mask the connectivity and distances with the threshold
    thresholdedConnectivity = connectivityAge[thresholded]
    thresholdedDistances = distances[thresholded]
    # Find connections greater than 0
    positiveConnectivityIndex = thresholdedConnectivity > thresh
    # Find connections less than 0
    negativeConnectivityIndex = thresholdedConnectivity < thresh
    # Slice distances for positive connections
    positiveEffectDistances = thresholdedDistances[positiveConnectivityIndex]
    # Slice distances for negative connections
    negativeEffectDistances = thresholdedDistances[negativeConnectivityIndex]
    if (len(positiveEffectDistances) == 0 or len(negativeEffectDistances) == 0):
        print('No distances passed the thresholding')
        pass
    else:
        print('# positive distances ' + str(len(positiveEffectDistances)))
        print('# negative distances ' + str(len(negativeEffectDistances)))

    return positiveEffectDistances, negativeEffectDistances


def getNetworkDistance(networkIndex, connectivityMatrix, distanceMatrix,
                       thresholdMatrix):
    # Make inverse index for between network connections
    betweenIndex = networkIndex != True
    connectivityColumns = connectivityMatrix[networkIndex, ...]
    distanceColumns = distanceMatrix[networkIndex, ...]
    thresholdColums = thresholdMatrix[networkIndex, ...]
    connectivityWithin = connectivityColumns[..., networkIndex]
    distanceWithin = distanceColumns[..., networkIndex]
    thresholdWithin = thresholdColums[..., networkIndex]
    connectivityBetween = connectivityColumns[..., betweenIndex]
    distanceBetween = distanceColumns[..., betweenIndex]
    thresholdBetween = thresholdColums[..., betweenIndex]

    return (connectivityWithin, distanceWithin, thresholdWithin,
            connectivityBetween, distanceBetween, thresholdBetween)


def plotDistances(posDistances, negDistances, title):
    plt.hist(posDistances, bins=10,
             color='b', label='age pos')
    plt.hist(negDistances, bins=10,
             color='r', alpha=0.5, label='age neg')
    plt.title(title)
    plt.xlabel('distance in volume elements')
    plt.ylabel('frequency of distance')
    plt.legend()
    plt.show()
    raw_input("Press Enter to continue...")
    plt.close()


def plotNumbers(posDistances, negDistances, title):
    plt.plot(posDistances)
    pass


def dualPlotDistances(posWithin, negWithin, posBetween, negBetween, title):

    # Check number of significant correlations/distances
    nPosW = len(posWithin)
    nNegW = len(negWithin)
    nPosB = len(posBetween)
    nNegB = len(negBetween)

    # Print some info
    print('Plotting ' + title)
    print('posWithin ' + str(nPosW) + ' negWithin ' + str(nNegW))
    print('posBetween ' + str(nPosB) + ' negBetween ' + str(nNegB))

    fig, (within, between) = plt.subplots(1, 2, sharex=True, sharey=False)
    within.set_title('within network')
    between.set_title('between network')

    if nPosW > 0:
        within.hist(posWithin, bins=10, color='b',
                    label='age pos')
    else:
        print('Not enough values in positive within')

    if nNegW > 0:
        within.hist(negWithin, bins=10, color='r',
                    alpha=0.5, label='age neg')
    else:
        print('Not enough values in negative within')
    within.set_xlabel('distance in volume elements')
    within.set_ylabel('frequency of distance')
    within.legend()


    if nPosB > 0:
        between.hist(posBetween, bins=10, color='b',
                     label='age pos')
    else:
        print('Not enough values in positive between')

    if nNegB > 0:
        between.hist(negBetween, bins=10, color='r',
                     alpha=0.5, label='age neg')
    else:
        print('Not enough values in negative between')

    between.set_xlabel('distance in volume elements')
    between.set_ylabel('frequency of distance')
    between.legend()

    fig.suptitle(title)
    plt.show()
    raw_input("Press Enter to continue...")
    plt.close()


def saveNumpyTextFile(outputFilePath, outputMatrix):
    np.savetxt(outputFilePath, outputMatrix, fmt='%.12f')
    status = 'cool'

    return status


def Main():
    # Define inputs
    pathToAgeConnectivtyMatrix = '/home2/surchs/secondLine/correlation/testing/dos160/glm_matrix_glob_corr.txt'
    pathToDistancesMatrix = '/home2/surchs/secondLine/roiDistances/dos160abide246_3mm_distances.txt'
    pathToPvaluesMatrix = '/home2/surchs/secondLine/correlation/testing/dos160/glm_pvalue_matrix_glob_corr.txt'
    pathToNetworkNodes = '/home2/surchs/secondLine/configs/networkNodes_dosenbach.dict'
    pathToRoiMask = '/home2/surchs/secondLine/masks/dos160_abide_246_3mm.nii.gz'
    pathToSubjectList = '/home2/surchs/secondLine/configs/wave/wave_subjectList.csv'

    # Define Outputs
    pathToPositiveAgeDistances = '/home2/surchs/secondLine/correlation/group_distances_age_pos_plevel.txt'
    pathToNegativeAgeDistances = '/home2/surchs/secondLine/correlation/group_distances_age_neg_plevel.txt'

    # Define parameters
    thresh = 0.0001

    # Read inputs
    ageConnectivityMatrix = loadNumpyTextFile(pathToAgeConnectivtyMatrix)
    distancesMatrix = loadNumpyTextFile(pathToDistancesMatrix)
    pvaluesMatrix = loadNumpyTextFile(pathToPvaluesMatrix)
    networkNodes = loadArchive(pathToNetworkNodes)
    # Read subject list
    subjectListFile = open(pathToSubjectList, 'rb')
    subjectList = subjectListFile.readlines()
    numSubjects = len(subjectList)

    # Load the ROI mask
    roiImage, roiData = loadNiftiImage(pathToRoiMask)
    # get the unique nonzero elements in the ROImask
    uniqueRoi = np.unique(roiData[roiData != 0])

    # Generate the threshold matrix
    thresholdMatrix = threshold(pvaluesMatrix, thresh=thresh)
    # Check the pvalues
    uniquePValues = getUniqueMatrixElements(pvaluesMatrix)
    uniquePValues = uniquePValues[uniquePValues != 0]
    plt.hist(-np.log10(uniquePValues[uniquePValues < thresh]))
    plt.title('pvalues < ' + str(thresh))
    plt.show()
    plt.close()

    # #
    # Plot the results for the network analysis
    # #

    # Loop through all the networks and get within and between connectivity
    # separately
    '''
    for network in networkNodes.keys():
        print('plotting network ' + network)
        netNodes = networkNodes[network]
        # Make an index of the Rois in the current network
        networkIndex = np.in1d(uniqueRoi, netNodes)
        (connectivityWithin,
         distanceWithin,
         thresholdWithin,
         connectivityBetween,
         distanceBetween,
         thresholdBetween) = getNetworkDistance(networkIndex,
                                                ageConnectivityMatrix,
                                                distancesMatrix,
                                                thresholdMatrix)

        # Get unique elements
        uConnWithin = getUniqueMatrixElements(connectivityWithin)
        uDistWithin = getUniqueMatrixElements(distanceWithin)
        uThreshWithin = getUniqueMatrixElements(thresholdWithin)
        uConnBetween = getUniqueMatrixElements(connectivityBetween)
        uDistBetween = getUniqueMatrixElements(distanceBetween)
        uThreshBetween = getUniqueMatrixElements(thresholdBetween)

        # Plot effects within
        tableString = 'dir\loc    within        between\n'
        formatString = '________________________________\n'
        tableString = tableString + formatString
        (positiveEffectWithin,
         negativeEffectWithin) = getConnectivityDistances(uConnWithin,
                                                          uDistWithin,
                                                          uThreshWithin)
        totalWithin = float(len(positiveEffectWithin) + len(negativeEffectWithin))

        (positiveEffectBetween,
         negativeEffectBetween) = getConnectivityDistances(uConnBetween,
                                                          uDistBetween,
                                                          uThreshBetween)
        totalBetween = float(len(positiveEffectBetween) + len(negativeEffectBetween))
        posWperc = len(positiveEffectWithin) / totalWithin
        posBperc = len(positiveEffectBetween) / totalBetween
        posString = (' pos        ' + str(round(posWperc, 2)) + 'p'
                     + '         ' + str(round(posBperc, 2)) + 'p\n')

        negWperc = len(negativeEffectWithin) / totalWithin
        negBperc = len(negativeEffectBetween) / totalBetween
        negString = (' neg        ' + str(round(negWperc, 2)) + 'p'
                     + '         ' + str(round(negBperc, 2)) + 'p\n')

        tableString = tableString + posString + negString
        print('For network ' + network)
        print(tableString)

        dualPlotDistances(positiveEffectWithin, negativeEffectWithin,
                          positiveEffectBetween, negativeEffectBetween,
                          network)
    '''

    # #
    # Plot the results for the global analysis
    # #

    # Get the unique elements in the connectivity and distance matrix
    uniqueConnections = getUniqueMatrixElements(ageConnectivityMatrix)
    uniqueDistances = getUniqueMatrixElements(distancesMatrix)
    uniqueThresholds = getUniqueMatrixElements(thresholdMatrix)

    print('\n\nThis many thresholds survive: '
         + str(np.sum(uniqueThresholds)) + '\n\n')

    # Get the distances
    (positiveEffectDistances,
     negativeEffectDistances) = getConnectivityDistances(uniqueConnections,
                                                         uniqueDistances,
                                                         uniqueThresholds)

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
