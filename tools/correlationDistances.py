'''
Created on Feb 5, 2013

@author: surchs

Calculate significant connectivity-age correlation and plot its histogram over
distance for positive and negative correlation
'''
import os
import re
import sys
import glob
import copy
import numpy as np
import nibabel as nib
from scipy import ndimage


def MaskCoordinates(maskFile):
    # get the mask
    mF = nib.load(maskFile)
    maskData = mF.get_data()
    
    maskCoords = {}
    maskValues = np.unique(maskData[maskData!=0])
    for roi in maskValues:
        roiCoord = ndimage.measurements.center_of_mass(maskData==roi)
        maskCoords[str(roi)] = roiCoord
        
    # now make a roi by roi matrix of distances
    distMat = np.array({})
    for rowRoi in maskValues:
        rowCoord = np.array(maskCoords[str(rowRoi)])
        distRow = np.array({})
        for colRoi in maskValues:
            colCoord = np.array(maskCoords[str(colRoi)])
            
            print(str(rowRoi) + '/' + str(colRoi))
            print(str(rowCoord) + '/' + str(colCoord) + '\n')
            
            if rowRoi == colRoi:
                dist = 0
            else:
                dist = np.linalg.norm(rowCoord, colCoord)
            distRow = np.append(distRow, dist)
        # stack it up, yo
        if distMat.size == 0:
            distMat = distRow[None, ...]
        else:
            distMat = np.concatenate((distMat, distRow[None, ...]), axis=0)
    
    return maskData, distMat


def Main(maskFile, searchDir, phenoFile, outDir):
    searchPattern = '/scan_func_rest/func/bandpass_freqs_0.009.0.1/functional_mni.nii.gz'

    maskData, distMat = MaskCoordinates(maskFile)
    print(distMat)
    '''
    phenoIndex = {}
    phenoLine = open(phenoFile, 'rb').readline()
    phenoLoop = phenoLine.strip().split(',')
    
    run = 0
    for pheno in phenoLoop:
        # check if it is the subject designator which is not a phenotypic info
        if not pheno == 'subject':
            phenoIndex[pheno] = run

        run += 1
    
    for pipeline in os.listdir(searchDir):
        subjectDir = os.path.abspath(os.path.join(searchDir, pipeline))
        # for each pipeline
        listOfFuncs = []
        listOfAge = []
        
        for subject in os.listdir(subjectDir):
            findSubBase = re.search(r'[a-zA-Z]*[0-9]*(?=_)', subject)
            subBase = findSubBase.group()
            
            if (subBase in open(phenoFile).read()):
                # all is good
                # print('found ' + subBase + ' in pheno and nuisance file. Ya!')
                for line in open(phenoFile, 'rb'):
                    # search for the correct line
                    # currently requires the subject name to be given at
                    # the start
                    if line.startswith(subBase):
                        subLine = line.strip().split(',')
                        
            else:
                # no phenotypic information, presently we skip these subjects
                print('didn\'t find ' + subBase + ' in phenofile or nFile.'
                      + ' skipping for now...')
                continue
            
            # get the phenotypic information
            subAge = subLine[phenoIndex['age']]
            
            
            # get funcFile
            searchString = (subjectDir + '/' + subject + searchPattern)
            a = glob.glob(searchString)
            if len(a) == 0:
                    print(subjectDir + '/' + subject + searchPattern
                          + ' doesn\'t exist')
            else:
                tempScaPath = a[0]
                listOfFuncs.append(tempScaPath)
                listOfAge.append(subAge)
                
        # done with searching - Run
        print(str(len(listOfFuncs)) + ' ' + str(len(listOfAge)))
        # prepare Matrix of Subject features and vector of ages
        subFeatMat = np.array([])
        subAgeVec = np.array([])
        # loop through the subjects and make a full correlation matrix
        for funcPath in listOfFuncs:
            tFuncF = nib.load(funcPath)
            tFuncData = tFuncF.get_data()
            
            tempTimeseries = np.array([])
            # now lets get the timeseries
            for roi in maskValues:
                rawTimeseries = tFuncData[mask == roi]
                timeseries = np.average(rawTimeseries, axis=0)
                if tempTimeseries.size == 0:
                    # first time customer...
                    tempTimeseries = timeseries[np.newaxis, ...]
                else:
                    tempTimeseries = np.concatenate((tempTimeseries,
                                                timeseries[np.newaxis, ...]),
                                                axis=0)
            
            # all ROIs in and ordered, run connectivity analysis
            tempConnMat = np.corrcoef(tempTimeseries)
            # now flatten the lower triangle of the matrix
            tempBrainData = copy.deepcopy(tempConnMat)
            brainMask = np.ones_like(tempBrainData)
            brainMask = np.tril(brainMask, -1)
            flatConnMat = tempBrainData[brainMask == 1]
            # stack this up
            if subFeatMat.size == 0:
                subFeatMat = flatConnMat[None, ...]
            else:
                subFeatMat = np.concatenate((subFeatMat,
                                             flatConnMat[None, ...]), axis=0)
            
        # and ages
        for age in listOfAge:
            subAgeVec = np.append(subAgeVec, age)
                
        # done stacking things up - correlate
        '''
                    

if __name__ == '__main__':
    maskFile = sys.argv[1]
    searchDir = sys.argv[2]
    phenoFile = sys.argv[3]
    outDir = sys.argv[4]
    Main(maskFile, searchDir, phenoFile, outDir)
    pass