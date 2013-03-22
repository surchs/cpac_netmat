'''
Created on Feb 21, 2013

@author: surchs
'''
import os
import numpy as np


def testCorrelation():
    subject = 'sub3310'
    pathToTimeSeries = '/home2/surchs/secondLine/timeseries/sub3310_timeseries.txt'
    outputFilePath = '/home2/surchs/secondLine/connectomes/test.txt'
    timeSeriesMatrix = loadTimeSeriesMatrix(pathToTimeSeries)
    correlationMatrix = computeCorrelation(timeSeriesMatrix)
    status = saveCorrelationMatrix(correlationMatrix, outputFilePath)
    print(status)

    return

def loadTimeSeriesMatrix(pathToTimeSeries):
    timeSeriesMatrix = np.loadtxt(pathToTimeSeries)

    return timeSeriesMatrix


def computeCorrelation(timeSeriesMatrix):
    correlationMatrix = np.corrcoef(timeSeriesMatrix)

    return correlationMatrix


def saveCorrelationMatrix(correlationMatrix, outputFilePath):
    np.savetxt(outputFilePath, correlationMatrix, fmt='%.12f')
    return 'cool'


def Main():
    # Define the inputs
    pathToSubjectList = '/home2/surchs/secondLine/configs/wave/wave_subjectList.csv'
    pathToTimeSeriesDir = '/home2/surchs/secondLine/timeseries/wave/dos160'
    # Define input timeseries suffix
    timeSeriesSuffix = '_timeseries_glob.txt'

    # Define output directory
    pathToOutputDir = '/home2/surchs/secondLine/connectomes/wave/dos160'
    # Define output file suffix
    outFileSuffix = '_connectome_glob.txt'

    # Read subject list
    subjectListFile = open(pathToSubjectList, 'rb')
    subjectList = subjectListFile.readlines()

    for subject in subjectList:
        subject = subject.strip()
        pathToTimeSeries = os.path.join(pathToTimeSeriesDir,
                                        (subject + timeSeriesSuffix))

        timeSeriesMatrix = loadTimeSeriesMatrix(pathToTimeSeries)
        correlationMatrix = computeCorrelation(timeSeriesMatrix)
        # Define the file name for the connectome output
        outputFilePath = os.path.join(pathToOutputDir,
                                      (subject + outFileSuffix))
        # save the connectome matrix in the specified output file
        status = saveCorrelationMatrix(correlationMatrix, outputFilePath)
        print(subject + ' says ' + status)


if __name__ == '__main__':
    Main()
