'''
Created on Jan 16, 2013

@author: surchs
'''
import sys
import gzip
import cPickle
import numpy as np


def Main(studyFile):
    # load the fucking file
    print('Got something, now loading')
    f = gzip.open(studyFile)
    study = cPickle.load(f)
    print('Done loading stuff, now doing stuff')
    # now get one analysis (just the goddamn first one)
    analysis = study.analyses.values()[0]
    # and get one of the masks out
    # mask = analysis.masks.values()[0]
    mask = analysis.mask
    # store the numbers of the networks so I know what I am fucking entering
    networkNumbers = {}
    run = 0.0
    for network in mask.networkNodes.keys():
        networkNumbers[network] = run
        run += 1
        
    # now recreate that connectivity matrix and enter the matrices
    maskNodes = float(len(mask.nodes))
    indexMat = np.zeros((maskNodes, maskNodes))
    print(indexMat.shape)
    # prepare the feature index for the different networks
    netFeatInd = {}
    # and now do what you did to the connectivity matrix just this time enter
    #
    # first pass: get those numbers in there
    for network in mask.networkIndices.keys():
        # grab the network index from the mask
        tempInd = mask.networkIndices[network]
        # and get the network number
        tempNumber = networkNumbers[network]
        # grab the correct rows and then put the correct numbers in there
        indexMat[tempInd, ...] = tempNumber
        
    # second pass: get the numbers out again and store them in vectors
    for network in mask.networkIndices.keys():
        # grab the network index from the mask
        tempInd = mask.networkIndices[network]
        # first get the rows belonging to the network
        tempNet = indexMat[tempInd, ...]
        # then get the matrix belonging to the within features
        tempWithinNet = tempNet[..., tempInd]
        # and now only take the lower triangle
        tempMask = np.ones_like(tempWithinNet)
        tempMask = np.tril(tempMask, -1)
        # and put it in the variable
        tempWithin = tempWithinNet[tempMask == 1]
        
        # now for between - delete the within rows
        tempBetweenNet = np.delete(tempNet, tempInd, 1)
        # now stretch it out
        tempBetween = np.reshape(tempBetweenNet,
                                 tempBetweenNet.size)
        
        # and lastly for the whole connectivity
        tempWhole = np.append(tempWithin, tempBetween)
        
        netFeatInd[network] = tempWhole
    
    # and print some stuff
    print(netFeatInd.keys())
    

if __name__ == '__main__':
    studyFile = sys.argv[1]
    Main(studyFile)
    pass