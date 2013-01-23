'''
Created on Dec 6, 2012

@author: sebastian

Method to call and run the analysis classes. Alpha version.
'''
import sys
import gzip
import time
import cPickle
from cpac_netmat import analysis as an


def Main(studyName, dataPath, subjectList, numberCores, configFile, outFile):
    '''
    Wrapper Method for the analysis.
    '''
    study = an.base.Study(studyName, dataPath, subjectList=subjectList,
                                numberCores=numberCores)
    study.makeSubjectPaths()
    study.getSubjects()
    # in the current setting, this will also run the analysis
    study.getAnalyses(configFile)
    print('Study is essentially done, folks. Go home')
    start = time.time()
    f = gzip.open(outFile, 'wb')
    # WORKAROUND Delete subjects to save data
    study.maskedSubjects = None
    print('Start saving study')
    cPickle.dump(study, f, 2)
    stop = time.time()
    elapsed = stop - start
    print('Saving took ' + str(elapsed) + ' seconds')


if __name__ == '__main__':
    studyName = sys.argv[1]
    dataPath = sys.argv[2]
    subjectList = sys.argv[3]
    numberCores = int(sys.argv[4])
    configFile = sys.argv[5]
    outFile = sys.argv[6]
    Main(studyName, dataPath, subjectList, numberCores, configFile, outFile)
    pass
