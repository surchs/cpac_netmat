'''
Created on Mar 15, 2013

@author: sebastian
'''
import random
import numpy as np
from matplotlib import pyplot as plt


def binarize(value):
    '''
    Return -1 if negative and +1 if positive
    '''
    if value > 0:
        binarized = 1
    elif value <= 0:
        binarized = -1

    return binarized


def makeConnectome():
    '''
    just make a quick range of subjects and plot one of the 'connection'
    '''
    # Make a template for the connectome
    numROIs = 147
    template = np.zeros((numROIs, numROIs), dtype=float)
    # Make a mask
    mask = np.ones_like(template)
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

        # Store the subjects connectome
        if connectomeStack.size == 0:
            connectomeStack = subjectConn[None, ...]
        else:
            connectomeStack = np.concatenate((connectomeStack,
                                              subjectConn[None, ...]),
                                             axis=0)

        run += 1

    print('Done creating connections')
    print('Connectome: ' + str(connectomeStack.shape))

    # Make a sanity check of connections with age
    conn1 = connectomeStack[:, 18]
    conn2 = connectomeStack[:, 800]
    conn3 = connectomeStack[:, 1082]
    plt.plot(ageArray, conn1, 'g.')
    plt.plot(ageArray, conn2, 'r.')
    plt.plot(ageArray, conn3, 'b.')
    plt.show()
    raw_input('Hallo...')
    plt.close()

def Main():

    pass


if __name__ == '__main__':
    makeConnectome()
    pass