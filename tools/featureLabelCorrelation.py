'''
Created on Jan 10, 2013

@author: surchs
'''
import sys
import gzip
import glob
import cPickle
import numpy as np
from matplotlib import pyplot as plt


def Main(folder, mask):
    '''
    method to calculate and display the correlation between the features and 
    the labels
    '''
    # import the subjects
    subjectPaths = glob.glob((folder + '*/*.sub'))
    subDict = {}
    
    for subPath in subjectPaths:
        subFile = gzip.open(subPath)
        subject = cPickle.load(subFile)
        subDict[subject.name] = subject
        print(subject.name)
        
    # done loading
    print('Done loading subjects')
    
    # prepare the big correlation matrix
    featMat = np.array([])
    ageMat = np.array([])
    
    for subject in subDict.keys():
        tempSub = subDict[subject]
        tempMask = tempSub.derivativeMasks[mask]
        tempFeat = tempMask['functional_connectivity'].feature
        indexMask = np.ones_like(tempFeat)
        indexMask = np.tril(indexMask, -1)
        
        featureVec = tempFeat[indexMask==1]
        
        if featMat.size == 0:
            featMat = featureVec[None, ...]
        else:
            featMat = np.concatenate((featMat, featureVec[None, ...]), axis=0)
            
        tempLabel = tempSub.pheno['age']
        ageMat = np.append(ageMat, float(tempLabel))
        
    # done with that, get the correlation:
    print(featMat.shape)
    meanVec = np.average(featMat, axis=0)
    print(meanVec.shape)
    stdVec = np.std(featMat, axis=0)
    print(ageMat.shape)
    print(ageMat)
    meanAge = np.average(ageMat)
    colInd = np.arange(featMat.shape[1])
    corrVec = np.ones_like(colInd, dtype=float)
    
    
    for col in colInd:
        tempCol = featMat[..., col]
        fishCol = np.arctanh(tempCol)
        colMean = meanVec[col]
        colStd = stdVec[col]
        demeanAge = ageMat - meanAge
        zCol = (fishCol - colMean) / colStd
        corr = np.min(np.corrcoef(demeanAge, zCol))
        corrVec[col] = corr
    tempAbsCorr = np.absolute(corrVec)
    tempGreater = tempAbsCorr[tempAbsCorr >= 0.22]
    nGreater = len(tempGreater)
    nTotal = len(tempAbsCorr)
    print(str(str(nGreater) + ' features out of ' + str(nTotal)) + ' correlate'
          + ' with age higher than 0.22')
        
    # now we have it
    print('Done calculating correlation')
    print('Max correlation: ' + str(np.max(corrVec)) + ' and Min correlation '
          + str(np.min(corrVec))) 
    
    plt.hist(corrVec, bins=15)
    plt.show()
        
    
if __name__ == '__main__':
    folder = sys.argv[1]
    mask = sys.argv[2]
    Main(folder, mask)
    pass