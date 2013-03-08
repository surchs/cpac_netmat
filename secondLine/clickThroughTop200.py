'''
Created on Mar 5, 2013

@author: surchs

Click through the top 200 connections that are correlated with age
'''
import numpy as np
import pandas as pa
import os
import pandas as pa
from scipy import stats as st
from matplotlib import pyplot as plt
import time
from cpac_netmat.tools import meisterlein as mm

def loadNumpyTextFile(pathToNumpyTextFile):
    numpyTextFile = np.loadtxt(pathToNumpyTextFile)

    return numpyTextFile


def loadPhenotypicFile(pathToPhenotypicFile):
    pheno = pa.read_csv(pathToPhenotypicFile)

    return pheno


def getTopCorrMask(squareMatrix):
    matrix = np.abs(squareMatrix)
    mask = np.ones_like(matrix)
    outmask = np.zeros_like(matrix, dtype=int)
    mask = np.tril(mask, -1)
    # Make everything 0 outside lower triangle
    matrix[mask == 0] = 0
    # get values
    values = np.sort(matrix[mask == 1])
    cutOff = values[-200]
    passers = matrix[matrix > cutOff]
    passOrder = np.argsort(passers)
    # add 1 to it so the lowest is 1 and we can differentiate from 0
    passRank = np.argsort(passOrder) + 1
    outmask[matrix > cutOff] = passRank

    print('The 200th highest correlation is ' + str(cutOff))
    print('The highest correlation is ' + str(np.max(values)))

    return outmask


def stackConnectome(connectomeStack, connectome):
    # See if stack is empty, if so, then initialize
    if connectomeStack.size == 0:
        connectomeStack = connectome[..., None]
    else:
        connectomeStack = np.concatenate((connectomeStack,
                                         connectome[..., None]),
                                        axis=2)

    return connectomeStack


def stackAges(ageStack, age):
    ageStack = np.append(ageStack, age)

    return ageStack


def correlateConnectomeWithAge(connectomeStack, ageStack):
    # First I flatten the connectomeStack
    stackShape = connectomeStack.shape
    # Get the number of connections to correlate
    numberOfConnections = stackShape[0] * stackShape[1]
    numberOfTimepoints = stackShape[2]
    # Reshape the stack to number of correlations by number of timepoints
    flatConnectomeStack = np.reshape(connectomeStack,
                                     (numberOfConnections,
                                      numberOfTimepoints))

    # Prepare container variables for correlation and p values for each
    # connection
    correlationVector = np.array([])
    pValueVector = np.array([])

    # Iterate over the elements in the stack and correlate them to the age
    # stack one by one
    for connectionIndex in np.arange(numberOfConnections):
        # Get the vector of connection values across subjects for current
        # connection
        connectionVector = flatConnectomeStack[connectionIndex, :]
        # Correlate the vector to age
        corr, p = st.pearsonr(connectionVector, ageStack)
        # Append correlation and p values to their respective container
        # variables
        correlationVector = np.append(correlationVector, corr)
        pValueVector = np.append(pValueVector, p)

    # Reshape the results of the correlation back into the shape of the
    # original connectivity matrix
    correlationMatrix = np.reshape(correlationVector,
                                   (stackShape[0], stackShape[1]))
    pValueMatrix = np.reshape(pValueVector,
                              (stackShape[0], stackShape[1]))

    return correlationMatrix, pValueMatrix


def Main():
    # Define the inputs
    pathToConnectomeDir = '/home2/surchs/secondLine/connectomes'
    pathToPhenotypicFile = '/home2/surchs/secondLine/configs/pheno81_uniform.csv'
    pathToSubjectList = '/home2/surchs/secondLine/configs/subjectList.csv'
    pathToCorrelationMatrixAges = '/home2/surchs/secondLine/correlation/correlation_matrix.txt'

    connectomeSuffix = '_connectome.txt'

    # Define parameters
    minAge = 6
    maxAge = 18

    # Read subject list
    subjectListFile = open(pathToSubjectList, 'rb')
    subjectList = subjectListFile.readlines()

    # Read the phenotypic file
    pheno = loadPhenotypicFile(pathToPhenotypicFile)
    phenoSubjects = pheno['subject'].tolist()
    phenoAges = pheno['age'].tolist()

    # Read the correlation matrix with age
    # connAgeCorr = loadNumpyTextFile(pathToCorrelationMatrixAges)


    # Prepare container variables for the connectome and for age for each of
    # the three age groups - not currently used
    limitConnectomeStack = np.array([])
    limitAgeStack = np.array([])
    fullconnectomeStack = np.array([])
    fullageStack = np.array([])

    # Loop through the subjects
    for i, subject in enumerate(subjectList):
        subject = subject.strip()
        phenoSubject = phenoSubjects[i]

        if not subject == phenoSubject:
            raise Exception('The Phenofile returned a different subject name '
                            + 'than the subject list:\n'
                            + 'pheno: ' + phenoSubject + ' subjectList '
                            + subject)

        # Get the age of the subject from the pheno file
        phenoAge = phenoAges[i]

        # Now continue with the full stacks
        # Construct the path to the connectome file of the subject
        pathToConnectomeFile = os.path.join(pathToConnectomeDir,
                                            (subject + connectomeSuffix))
        # Load the connectome for the subject
        connectome = loadNumpyTextFile(pathToConnectomeFile)

        # Make a selection of age here
        if phenoAge > minAge and phenoAge < maxAge:
            # include subject
            # Stack the connectome
            limitConnectomeStack = stackConnectome(limitConnectomeStack,
                                                   connectome)
            # Stack ages
            limitAgeStack = stackAges(limitAgeStack, phenoAge)

        elif phenoAge > maxAge:
            # drop subject
            print(subject + ' is too old with age = ' + str(phenoAge))
            pass

        elif phenoAge < minAge:
            print(subject + ' is too young with age = ' + str(phenoAge))
            pass

        # Continue with the full stacks

        # Stack the connectome
        fullconnectomeStack = stackConnectome(fullconnectomeStack, connectome)
        # print(connectomeStack.shape)
        # Stack ages
        fullageStack = stackAges(fullageStack, phenoAge)

    # FULL: Correlate age with connections
    fullCorrMat, fullPMat = correlateConnectomeWithAge(fullconnectomeStack,
                                                       fullageStack)
    # LIMITED: Correlate age with connections
    limCorrMat, limPMat = correlateConnectomeWithAge(limitConnectomeStack,
                                                       limitAgeStack)
    # Get mask for top 200
    top200mask = getTopCorrMask(limCorrMat)
    # make a list of values beginning with the lowest rank (which corresponds
    # to the highest correlation)
    top200ranks = np.arange(np.max(top200mask), 0, -1)
    # topCoords = np.argwhere(top200mask)

    print('\n\nVisualizing')
    # Now start plotting:
    plt.ion()
    run = np.max(top200mask)
    while run > 0:
        coord = np.argwhere(top200mask == run).flatten()
        print(coord)
        print(limCorrMat.shape)
        corrVal = limCorrMat[coord[0], coord[1]]
        fullPVal = fullPMat[coord[0], coord[1]]
        fullConnVector = fullconnectomeStack[coord[0], coord[1], :]
        limPVal = limPMat[coord[0], coord[1]]
        limConnVector = limitConnectomeStack[coord[0], coord[1], :]

        # Fit the curves
        lin = np.polyfit(limitAgeStack, limConnVector, deg=1)
        quad = np.polyfit(limitAgeStack, limConnVector, deg=2)
        cube = np.polyfit(limitAgeStack, limConnVector, deg=3)
        # Make the curves
        plotX = np.arange(limitAgeStack.min(), limitAgeStack.max() + 1, 0.1)
        plotLIN = np.polyval(lin, plotX)
        plotQUAD = np.polyval(quad, plotX)
        plotCUBE = np.polyval(cube, plotX)

        print('\nYou are looking at this now')
        print('The rank is: ' + str(run))
        print('coordinates are: ' + str(i))
        print('correlation is: ' + str(corrVal))
        print('connvec' + str(fullConnVector.shape) + ' '
              + str(fullageStack.shape))
        plt.plot(fullageStack, fullConnVector, 'k.', label='connection')
        plt.plot(plotX, plotLIN, color='g', label='linear fit')
        plt.plot(plotX, plotQUAD, color='r', label='quadratic fit')
        plt.plot(plotX, plotCUBE, color='b', label='cubic fit')

        plt.xlabel('age')
        plt.ylabel('connection strength')
        # plt.legend()
        plt.title('Connection #' + str(run) + ' r =: '
                  + str(np.round(corrVal, 2))
                  + ' p = ' + str(np.round(limPVal, 5)))
        plt.draw()
        plt.show()
        inPut = raw_input('To Continue, give new connection number and '
                          + 'Enter or just Enter...\n')
        if mm.isNumber(inPut):
            print('Got a new number ' + inPut)
            run = int(inPut)
        else:
            run -= 1

        plt.close()


if __name__ == '__main__':
    Main()
