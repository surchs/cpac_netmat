'''
Created on Jan 28, 2013

@author: surchs
'''
import sys
import numpy as np
from sklearn.svm import SVR
from matplotlib import pyplot as plt


def Main(inFile):
    '''
    Load the file, cut it into pieces and print the last line
    '''
    # open and read file
    loadFile = open(inFile, 'rb')
    fileLines = loadFile.readlines()
    
    # prepare storage variables for features and labels
    featMat = np.array([])
    labVec = np.array([])

    # loop through the lines of the file
    for line in fileLines:
        # strip the line ending and split it into columns by the delimiter
        useLine = line.strip().split()
        # line runtime variable to identify the column (col 14 is the label)
        run = 1
        # temporary storage for the features
        tempFeat = np.array([])
        for word in useLine:
            if run == 14:
                tempPheno = float(word)
            else:
                tempFeat = np.append(tempFeat, float(word))

            run += 1
        
        # add the feature and the label to the storage variables
        if featMat.size == 0:
            featMat = tempFeat[None, ...]
        else:
            featMat = np.concatenate((featMat, tempFeat[None, ...]), axis=0)
        
        labVec = np.append(labVec, tempPheno)
        
    # prepare the data for the model
    nCases = len(labVec)
    nTrain = np.floor(nCases / 2)
    trainX = featMat[:nTrain]
    trainY = labVec[:nTrain]
    testX  = featMat[nTrain:]
    testY = labVec[nTrain:]

    # finished reading the file, prepare the SVR model. Running with default
    # parameters here
    svrModel = SVR()
    
    # dump all the features and labels here
    # stick the labels to the right of the feature matrix:
    fullMatrix = np.concatenate((featMat, labVec[..., None]), axis=1)
    # and write out both the full matrix, features and labels
    np.savetxt('bld_full.csv', fullMatrix, fmt='%10.5f', delimiter=',')
    np.savetxt('bld_feature.csv', featMat, fmt='%10.5f', delimiter=',')
    np.savetxt('bld_label.csv', labVec, fmt='%10.5f', delimiter=',')
    # now also dump the train and test sets
    np.savetxt('bld_trainX.csv', trainX, fmt='%10.5f', delimiter=',')
    np.savetxt('bld_trainY.csv', trainY, fmt='%10.5f', delimiter=',')
    np.savetxt('bld_testX.csv', testX, fmt='%10.5f', delimiter=',')
    np.savetxt('bld_testY.csv', testY, fmt='%10.5f', delimiter=',')
    
    # everything is dumped, train the model
    svrModel.fit(trainX, trainY)
    # and predict the test labels from the test features
    predY = svrModel.predict(testX)
    
    # now dump the predicted Y
    np.savetxt('bld_predY.csv', predY, fmt='%10.5f', delimiter=',')
    
    # and plot everything
    plt.plot(testY, testY, label='true data')
    plt.plot(testY, predY, 'co', label='SVR')
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        inFile = 'housing.data'
    else:
        inFile = sys.argv[1]
    Main(inFile)
    pass