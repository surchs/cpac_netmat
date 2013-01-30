'''
Created on Jan 25, 2013

@author: sebastian

small script to load the Boston Housing Data reference dataset

also turn this into a wrapper to actually run the Network afterwards
'''
import sys
import gzip
import cPickle
import numpy as np
from matplotlib import pyplot as plt
import cpac_netmat.analysis.base as an
import cpac_netmat.preprocessing.base as pp
from sklearn.linear_model import LinearRegression
from matplotlib.backends.backend_pdf import PdfPages as pdf


def compareLogReg(feature, label):
    '''
    quick hack to compare the results of a default parameter logistic regression
    '''
    nCases = len(label)
    nTrain = np.floor(nCases / 2.0)
    trainX = feature[:nTrain]
    trainY = label[:nTrain]
    testX = feature[nTrain:]
    testY = label[nTrain:]
    
    logModel = LinearRegression()
    logModel.fit(trainX, trainY)
    predY = logModel.predict(testX)
    
    return predY, testY


def Main(inFile, outFile, pdfFile):
    '''
    Load the file, cut it into pieces and print the last line
    '''
    loadFile = open(inFile, 'rb')
    fileLines = loadFile.readlines()
    subDir = {}
    # storage for comparing other models
    featMat = np.array([])
    labVec = np.array([])

    subCount = 1
    for line in fileLines:
        useLine = line.strip().split()
        run = 1
        # make a new subject
        subName = ('case_' + str(subCount))
        tempSub = pp.Subject(subName, 'test')

        tempFeat = np.array([])
        for word in useLine:
            if run == 4 or run == 9:
                pass
            elif run == 14:
                tempPheno = float(word)
            else:
                tempFeat = np.append(tempFeat, float(word))

            run += 1
        tempSub.pheno = {}
        tempSub.pheno['houseprice'] = tempPheno
        tempSub.feature = tempFeat
        subDir[subName] = tempSub
        subCount += 1
        
        # and add them also to the storage vars
        if featMat.size == 0:
            featMat = tempFeat[None, ...]
        else:
            featMat = np.concatenate((featMat, tempFeat[None, ...]), axis=0)
        
        labVec = np.append(labVec, tempPheno)
        print(str(featMat.shape) + '/' + str(labVec.shape))

    # now make a network of it and run that stuff
    numberSubjects = len(subDir.keys())
    print(numberSubjects)
    # make a crossvalidation object
    cvObject = an.cv.KFold(numberSubjects, 10, shuffle=True)

    testNetwork = an.Network('test', cvObject)
    testNetwork.subjects = subDir
    testNetwork.pheno = 'houseprice'
    testNetwork.featureSelect = 'None'
    testNetwork.cValue = 1000
    testNetwork.gridCv = 5
    testNetwork.gridCores = 1
    testNetwork.eValue = 0.001
    testNetwork.kernel = 'rbf'
    # set number of parallel processes in Network
    testNetwork.numberCores = 10
    # make the runs
    print(len(testNetwork.subjects.keys()))
    print(len(testNetwork.cvObject))
    testNetwork.makeRuns()
    # now run the runs
    testNetwork.executeRuns()
    # and also run the other model quickly
    (modelPred, modelTrue) = compareLogReg(featMat, labVec)
    print('\nGot here')    
    # now save the result
    outF = gzip.open(outFile, 'wb')
    cPickle.dump(testNetwork, outF, protocol=2)
    
    # and display the rest
    pPheno = testNetwork.predictedPheno
    tPheno = testNetwork.truePheno
    
    fig4 = plt.figure(4, figsize=(8.5, 11), dpi=150)
    fig4.suptitle('predicted over true age')
    
    tSP4 = fig4.add_subplot(111, title=testNetwork.name)
    tSP4.plot(tPheno, tPheno)
    tSP4.plot(tPheno, pPheno, 'co')
    tSP4.plot(modelTrue, modelPred, 'mo')
    
    fig4.subplots_adjust(hspace=0.5, wspace=0.5)
    
    pd = pdf(pdfFile)
    pd.savefig(fig4)
    pd.close()
    plt.close(4)
    print('Just created ' + pdfFile + '\nAll done here!')


if __name__ == '__main__':
    inFile = sys.argv[1]
    outFile = sys.argv[2]
    pdfFile = sys.argv[3]
    Main(inFile, outFile, pdfFile)
    pass
