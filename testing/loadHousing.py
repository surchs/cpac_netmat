'''
Created on Jan 25, 2013

@author: sebastian

small script to load the Boston Housing Data reference dataset

also turn this into a wrapper to actually run the Network afterwards
'''
import sys
import numpy as np
import cpac_netmat.preprocessing.base as pp
import cpac_netmat.analysis.base as an


def Main(inFile):
    '''
    Load the file, cut it into pieces and print the last line
    '''
    loadFile = open(inFile, 'rb')
    fileLines = loadFile.readlines()
    subDir = {}

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

    # now make a network of it and run that stuff
    numberSubjects = len(subDir.keys())
    print(numberSubjects)
    # make a crossvalidation object
    cvObject = an.cv.KFold(numberSubjects, 10, shuffle=True)

    testNetwork = an.Network('test', cvObject)
    testNetwork.subjects = subDir
    testNetwork.pheno = 'houseprice'
    testNetwork.featureSelect = 'None'
    # set number of parallel processes in Network
    testNetwork.numberCores = 10
    # make the runs
    print(len(testNetwork.subjects.keys()))
    print(len(testNetwork.cvObject))
    testNetwork.makeRuns()
    # now run the runs
    testNetwork.executeRuns()
    print('\nGot here')


if __name__ == '__main__':
    inFile = sys.argv[1]
    Main(inFile)
    pass
