'''
Created on Mar 4, 2013

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


def plotDistribution(values, title):
    plt.hist(values, bins=20,
             color='b', label='age pos')
    plt.title(title)
    plt.xlabel('p values')
    plt.ylabel('frequency of distance')
    plt.legend()
    plt.show()
    raw_input("Press Enter to continue...")
    plt.close()


def Main():
    # Define inputs
    pathToPvaluesMatrix = '/home2/surchs/secondLine/correlation/pvalue_matrix.txt'

    # Get the matrix
    pMatrix = loadNumpyTextFile(pathToPvaluesMatrix)

    # Get the pvalues
    pvalues = getUniqueMatrixElements(pMatrix)

    # - log(10) the pvalues
    pvalues = -1 * np.log10(pvalues)

    # Get only pvalues under 0.1
    pvalues1 = pvalues[pvalues > 1]
    pvalues05 = pvalues[pvalues > 1.3]

    # Plot the distribution
    plotDistribution(pvalues, 'pvalues')
    plotDistribution(pvalues1, 'pvalues < 0.1')
    plotDistribution(pvalues05, 'pvalues < 0.05')

    # Get the 200 smalles p-values
    pvaluesSorted = np.sort(pvalues)
    pMin200 = pvaluesSorted[-200:]
    pMin500 = pvaluesSorted[-700:-200]
    # and plot them
    plotDistribution(pMin200, 'minimum 200 p values')
    plotDistribution(pMin500, 'next 500 p values')

    # Print some descriptive statistics
    print('\n\nAll p-values:')
    print('    min: ' + str(pvalues.min()))
    print('    max: ' + str(pvalues.max()))
    print('    mean: ' + str(np.mean(pvalues)))
    print('    median: ' + str(np.median(pvalues)))
    print('\n\np-values < 0.1:')
    print('    min: ' + str(pvalues1.min()))
    print('    max: ' + str(pvalues1.max()))
    print('    mean: ' + str(np.mean(pvalues1)))
    print('    median: ' + str(np.median(pvalues1)))
    print('\n\np-values < 0.05:')
    print('    min: ' + str(pvalues05.min()))
    print('    max: ' + str(pvalues05.max()))
    print('    mean: ' + str(np.mean(pvalues05)))
    print('    median: ' + str(np.median(pvalues05)))
    print('\n\nMinimum 200 p-values:')
    print('    min: ' + str(pMin200.min()))
    print('    max: ' + str(pMin200.max()))
    print('    mean: ' + str(np.mean(pMin200)))
    print('    median: ' + str(np.median(pMin200)))
    print('\n\nNext 500 p-values:')
    print('    min: ' + str(pMin500.min()))
    print('    max: ' + str(pMin500.max()))
    print('    mean: ' + str(np.mean(pMin500)))
    print('    median: ' + str(np.median(pMin500)))

if __name__ == '__main__':
    Main()
