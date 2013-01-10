'''
Created on Jan 10, 2013

@author: surchs

script to make a new phenotypic file for just the subjects in a subject list
goal is to reduce N of subjects in phenofile for preprocessing
'''
import sys


def Main(phenoIn, subjectIn, phenoOut):
    '''
    short method to do the task
    '''
    pIn = open(phenoIn, 'rb')
    sIn = open(subjectIn, 'rb')
    pOut = open(phenoOut, 'wb')

    inFirstLine = pIn.readline()
    pInLines = pIn.readlines()
    pOutLines = []
    pOutLines.append(inFirstLine)
    sInLines = sIn.readlines()
    sLines = []
    for sub in sInLines:
        sId = sub.strip()
        sLines.append(sId)
    print(sInLines)

    for pLine in pInLines:
        useLine = pLine.strip().split(',')
        subId = useLine[0]
        # print(subId)
        if subId in sLines:
            # take it in
            pOutLines.append(pLine)
            continue
        else:
            # print(subId + ' ain\'t in!')
            continue

    # write it out
    pOut.writelines(pOutLines)
    pOut.close()
    print('Done')

if __name__ == '__main__':
    phenoIn = sys.argv[1]
    subjectIn = sys.argv[2]
    phenoOut = sys.argv[3]
    Main(phenoIn, subjectIn, phenoOut)
    pass
