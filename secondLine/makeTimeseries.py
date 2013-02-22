'''
Created on Feb 21, 2013

@author: surchs
'''
import re
import os
import numpy as np
import nibabel as nib

# Define input files
pathToSubjectFile = '/home2/surchs/secondLine/configs/subjectList.csv'
pathToFuncFile = '/home2/surchs/secondLine/configs/pathsToFuncFiles.csv'
pathToRoiFile = '/home2/surchs/masks/ROIs/craddock200wave_p1l.nii.gz'

# Define output files
outputDirectory = '/home2/surchs/secondLine/timeseries'
subjectTimeSeriesSuffix = '_timeseries.txt'

# Read the input files
roiImage = nib.load(pathToRoiFile)
roiData = roiImage.get_data()

funcPathFile = open(pathToFuncFile, 'rb')
funcPathList = funcPathFile.readlines()

subjectFile = open(pathToSubjectFile, 'rb')
subjectList = subjectFile.readlines()

# Get the sorted list of nonzero elements in the ROI file
roiElements = np.unique(roiData[roiData!=0])
print('I got this many unique ROI elements: ' + str(len(roiElements)))

# Loop through the subjects functional files
for i, funcPath in enumerate(funcPathList):
    # load the functional file
    funcPath = funcPath.strip()
    funcImage = nib.load(funcPath)
    funcData = funcImage.get_data()
    
    # check if the subject is in the funcPath
    subject = subjectList[i]
    subject = subject.strip()
    if not subject in funcPath:
        raise Exception('This is not the funcPath you are looking for ' 
                        + subject + '\n' + funcPath)
        
    # Create a numpy container to store the timeseries in
    timeSeriesMatrix = np.array([])
    
    # Get average ROI timeseries
    for roi in roiElements:
        # Extract the timeseries from the func file
        timeSeries = funcData[roiData == roi]
        # Average the extracted timeseries across voxels (axis 0)
        averageTimeSeries = np.average(timeSeries, axis=0)
        # Append current ROIs timeseries to timeseries matrix
        if timeSeriesMatrix.size == 0:
            # First set up container for first use
            timeSeriesMatrix = averageTimeSeries[None, ...]
        else:
            timeSeriesMatrix = np.concatenate((timeSeriesMatrix, 
                                               averageTimeSeries[None, ...]),
                                              axis = 0)
    
    # Save subjects ROI timeseries matrix
    timeSeriesFile = (subject + subjectTimeSeriesSuffix)
    timeSeriesPath = os.path.join(outputDirectory, timeSeriesFile)
    print('I save this now: ' + timeSeriesPath)
    np.savetxt(timeSeriesPath, timeSeriesMatrix, fmt='%.12f')
    