'''
Created on Jan 9, 2013

@author: surchs

Module to sample ages from a list of subjects in order to create an even
age distribution
'''
import sys
import random
import numpy as np


def Main(phenoFile, outFile):
    # load the stuff
    phenoText = open(phenoFile, 'rb')
    # set the file counter to 0
    phenoText.seek(0)
    phenoLine = phenoText.readline()
    phenoLines = phenoText.readlines()

    # define the age bins
    minAge = 6
    maxAge = 34
    binWidth = 2.8
    binVector = np.arange(minAge, maxAge, binWidth)
    lowerVector = np.arange(minAge, maxAge - binWidth, binWidth)
    upperVector = np.arange(minAge + binWidth, maxAge, binWidth)
    print(binVector)
    print(lowerVector)
    print(upperVector)
    lenVector = np.arange(len(lowerVector))
    ageDict = {}
    useId = []

    for line in phenoLines:
        useLine = line.strip().split(',')
        subId = useLine[0]
        age = float(useLine[1])
        # print(subId + ' is ' + str(age) + ' years old')
        # store the ages
        for ind in lenVector:
            low = lowerVector[ind]
            up = upperVector[ind]
            if age >= low and age < up:
                # print(subId + ' is between ' + str(low) + ' and ' + str(up)
                #       + ' years old (' + str(age) + ')')
                if not str(low) in ageDict.keys():
                    ageDict[str(low)] = []
                ageDict[str(low)].append(subId)

    numberVector = np.array([])
    for key in ageDict.keys():
        nSubs = len(ageDict[key])
        print(key + ' has ' + str(nSubs) + ' subjects')
        numberVector = np.append(numberVector, nSubs)

    print(numberVector)
    nDabei = np.min(numberVector)
    nMitmach = np.min(numberVector) * len(numberVector)
    nAlle = np.sum(numberVector)
    print(nMitmach)
    print(nAlle)

    for key in ageDict.keys():
        nSubs = len(ageDict[key])
        if nSubs > nDabei:
            subInd = np.arange(nSubs)
            random.shuffle(subInd)
            includeInd = subInd[:nDabei]

            for ind in includeInd:
                useId.append(ageDict[key][ind])
        else:
            for sub in ageDict[key]:
                useId.append(sub)

    print(useId)
    print(len(useId))

    f = open(outFile, 'wb')
    for sub in useId:
        f.write(sub + '\n')
    f.close()


if __name__ == '__main__':
    phenoFile = sys.argv[1]
    outFile = sys.argv[2]
    Main(phenoFile, outFile)
    pass
