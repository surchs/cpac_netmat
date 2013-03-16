'''
Created on Mar 15, 2013

@author: sebastian
'''
import os
import copy
import random
import numpy as np
from matplotlib import pyplot as plt


def saveConnectome(connectome, pathToSaveFile):
    '''
    Save that stuff, diagonal is bullshit obviously
    '''
    np.savetxt(pathToSaveFile, connectome, fmt='%.12f')


def binarize(value):
    '''
    Return -1 if negative and +1 if positive
    '''
    if value > 0:
        binarized = 1
    elif value <= 0:
        binarized = -1

    return binarized


def makeConnectome(pathToOutputDir, pathToSubjectPheno):
    '''
    just make a quick range of subjects and plot one of the 'connection'
    '''
    # Make a template for the connectome
    numROIs = 147
    template = np.zeros((numROIs, numROIs), dtype=float)
    # Make a mask
    mask = np.ones_like(template)
    empty = np.zeros_like(mask)
    # Get the lower and upper triangle
    upper = np.triu(mask, 1)
    lower = np.tril(mask, -1)
    # Get the number of connections I have to produce
    numConns = lower.sum()

    # Make random base connectome
    connArray = np.random.normal(0, 0.2, numConns)
    # And make the matching age effects
    ageEffect = np.random.normal(0, 0.01, numConns)
    # Make it range from -1 to +1
    connArray = connArray - 1

    # Prepare container for the ages and connections of the subjects
    ageArray = np.array([])
    connectomeStack = np.array([])

    print('Creating connections now')
    # Loop through the subjects
    run = 0
    while run < 100:
        # Make the subjects age
        subAge = np.random.uniform(6, 20)
        ageArray = np.append(ageArray, subAge)

        # Create noise
        connNoise = np.random.normal(0, 0.05, numConns)
        ageNoise = np.random.normal(0, 0.01, numConns)

        # Make the subjects connectome with noise
        subjectConn = (connArray + connNoise) + (subAge * ageEffect + ageNoise)

        # Turn it into a connectivity matrix
        template = copy.deepcopy(empty)
        template[lower == 1] = subjectConn
        template.T[upper.T == 1] = subjectConn

        # create subject name
        subName = ('sub_' + str(run))
        fileName = (subName + '.txt')
        filePath = os.path.join(pathToOutputDir, fileName)
        # save the connectome
        saveConnectome(template, filePath)

        # Store the subjects connectome
        if connectomeStack.size == 0:
            connectomeStack = template[..., None]
        else:
            connectomeStack = np.concatenate((connectomeStack,
                                              template[..., None]),
                                             axis=2)

        run += 1

    print('Done creating connections')
    print('Connectome: ' + str(connectomeStack.shape))

    # Make a sanity check of connections with age
    conn1 = connectomeStack[10, 14, :]
    conn2 = connectomeStack[100, 13, :]
    conn3 = connectomeStack[24, 11, :]
    plt.plot(ageArray, conn1, 'g.')
    plt.plot(ageArray, conn2, 'r.')
    plt.plot(ageArray, conn3, 'b.')
    plt.show()
    raw_input('Hallo...')
    plt.close()

def Main():
    # Define outputs
    pathToOutputDir = '/home2/surchs/secondLine/connectomes/testing'
    pathToSubjectPheno = '/home2/surchs/secondLine/configs/testing/sub100pheno.csv'


    # Run the connectome generator
    connectomeStack, ageStack = makeConnectome(pathToOutputDir,
                                               pathToSubjectPheno)

    pass


if __name__ == '__main__':
    makeConnectome()
    pass
