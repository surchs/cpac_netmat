'''
Created on Feb 21, 2013

@author: surchs
'''
import re
import os
import numpy as np
import nibabel as nib
import multiprocessing as mp


def getTimeSeries(args):
    # Get back the arguments
    funcPath, roiData, subject = args
    # Create a numpy container to store the timeseries for the subject in
    funcImage = nib.load(funcPath)
    funcData = funcImage.get_data()

    if not funcData[..., 0].shape == roiData.shape:
        print('func and roi not same shape: func=' + str(funcData.shape)
              + ' roi=' + str(roiData.shape))
    else:
        print('All is well with ' + subject)

    # Get the sorted list of nonzero elements in the ROI file
    roiElements = np.unique(roiData[roiData != 0])

    timeSeriesMatrix = np.array([])

    # Get average ROI timeseries
    for roi in roiElements:
        # Extract the timeseries from the func file
        timeSeries = funcData[roiData == roi]
        print('ts shape: ' + str(timeSeries.shape))
        # Average the extracted timeseries across voxels (axis 0)
        averageTimeSeries = np.average(timeSeries, axis=0)
        print('avg ts shape: ' + str(averageTimeSeries.shape))
        # Append current ROIs timeseries to timeseries matrix
        if timeSeriesMatrix.size == 0:
            # First set up container for first use
            timeSeriesMatrix = averageTimeSeries[None, ...]
        else:
            timeSeriesMatrix = np.concatenate((timeSeriesMatrix,
                                               averageTimeSeries[None, ...]),
                                              axis=0)

    return timeSeriesMatrix, subject


def Main():
    # Define input files
    pathToSubjectFile = '/home2/surchs/secondLine/configs/wave/wave_subjectList.csv'
    pathToFuncFile = '/home2/surchs/secondLine/configs/wave/pathsToFuncFiles_wave.csv'
    pathToRoiFile = '/home2/surchs/secondLine/masks/dos160_wave_81_3mm.nii.gz'

    # Define parameters
    nProcs = 15

    # Define output files
    outputDirectory = '/home2/surchs/secondLine/tests/timeseries/mystuff'
    subjectTimeSeriesSuffix = '_timeseries_glob.txt'

    # Read the input files
    roiImage = nib.load(pathToRoiFile)
    roiData = roiImage.get_data()

    funcPathFile = open(pathToFuncFile, 'rb')
    funcPathList = funcPathFile.readlines()

    subjectFile = open(pathToSubjectFile, 'rb')
    subjectList = subjectFile.readlines()

    # Get the sorted list of nonzero elements in the ROI file
    roiElements = np.unique(roiData[roiData != 0])
    print('I got this many unique ROI elements: ' + str(len(roiElements)))

    # Prepare list for parameter tuples to run in parallel
    runList = []

    # Loop through the subjects functional files
    for i, funcPath in enumerate(funcPathList):
        # load the functional file
        funcPath = funcPath.strip()

        # check if the subject is in the funcPath
        subject = subjectList[i]
        subject = subject.strip()
        if not subject in funcPath:
            raise Exception('This is not the funcPath you are looking for '
                            + subject + '\n' + funcPath)

        # Prepare parameters
        tempRunParameters = (funcPath, roiData, subject)
        runList.append(tempRunParameters)

    # Now run this in parallel
    print('prepared to run multicore')
    pool = mp.Pool(processes=nProcs)
    resultList = pool.map(getTimeSeries, runList)
    print('ran multicore')

    # Now loop through the results
    for result in resultList:
        timeSeriesMatrix, subject = result

        # Save subjects ROI timeseries matrix
        timeSeriesFile = (subject + subjectTimeSeriesSuffix)
        timeSeriesPath = os.path.join(outputDirectory, timeSeriesFile)
        print('I save this now: ' + timeSeriesPath)
        np.savetxt(timeSeriesPath, timeSeriesMatrix, fmt='%.12f')

    # Done
    print('Done')


if __name__ == '__main__':
    Main()
