'''
Created on Mar 1, 2013

@author: surchs

Script to handle the contrast files from group analysis
'''
import os
import numpy as np


def returnListOfSeeds(searchDir):
    listOfSeeds = os.listdir(searchDir)

    return listOfSeeds


def makeBinary(pathToInputFile):
    inFileName = os.path.basename(pathToInputFile)
    inFileDir = os.path.dirname(os.path.abspath(pathToInputFile))
    outFileName = ('bin_' + inFileName)
    pathToOutputFile = os.path.join(inFileDir, outFileName)
    commandString = ('fslmaths ' + pathToInputFile + ' -bin '
                     + pathToOutputFile)
    print('\nI will do this now:\n'
          + '  ' + commandString)
    os.system(commandString)

    return pathToOutputFile


def mergeInverse(pathToPositiveFile, pathToNegativeFile, pathToOutFile):
    commandString = ('3dcalc -a ' + pathToPositiveFile + ' -b '
                     + pathToNegativeFile + ' -expr \'a - b\' -prefix '
                     + pathToOutFile)
    os.system(commandString)
    print('\nI will do this now:\n'
          + '  ' + commandString)

    return pathToOutFile


def mergeWithinBetween(mergedWithinFile, betweenFile, pathToOutFile):
    commandString = ('3dcalc -a ' + mergedWithinFile + ' -b ' + betweenFile
                     + ' -expr \'a + a*b\' -prefix ' + pathToOutFile)
    print('\nI will do this now:\n'
          + '  ' + commandString)
    os.system(commandString)

    return pathToOutFile


def Main():
    # Define inputs
    pathToSearchDir = '/home2/surchs/testing/modelWorkingDir/trifold_nmfd_mni_pcc'
    withinSubjectContrasts = ['thresh_zstat1.nii.gz', 'thresh_zstat2.nii.gz',
                              'thresh_zstat3.nii.gz', 'thresh_zstat4.nii.gz',
                              'thresh_zstat5.nii.gz', 'thresh_zstat6.nii.gz']
    withinSubjectMerged = ['merge_z1z2.nii.gz', 'merge_z3z4.nii.gz',
                           'merge_z5z6.nii.gz']
    withinSubjectBinMerged = ['bin_merge_z1z2.nii.gz', 'bin_merge_z3z4.nii.gz',
                           'bin_merge_z5z6.nii.gz']
    betweenSubjectContrasts = ['thresh_zstat7.nii.gz', 'thresh_zstat8.nii.gz',
                               'thresh_zstat9.nii.gz', 'thresh_zstat10.nii.gz']
    betweenSubjectMerged = ['increase_z1z2.nii.gz', 'decrease_z1z2.nii.gz',
                            'increase_z3z4.nii.gz', 'decrease_z3z4.nii.gz',
                            'increase_z5z6.nii.gz', 'decrease_z5z6.nii.gz']

    relativePathToFiles = 'thresholded'

    # Get the list of seed directories in the model directory
    seedList = returnListOfSeeds(pathToSearchDir)
    for seed in seedList:
        print('\n\nrunning ' + seed)
        # Prepare path containers
        binaryWithinContrasts = []
        binaryBetweenContrasts = []
        mergedBinWithinFiles = []
        mergedWithinFiles = []

        pathToSeedDir = os.path.join(pathToSearchDir, seed)
        pathToContrasts = os.path.join(pathToSeedDir, relativePathToFiles)
        # Binarize all the contrasts
        for contrast in withinSubjectContrasts:
            print('\nBinarizing contrast ' + contrast + ' in seed ' + seed)
            pathToContrast = os.path.join(pathToContrasts, contrast)
            binaryContrast = makeBinary(pathToContrast)
            binaryWithinContrasts.append(binaryContrast)

        for contrast in betweenSubjectContrasts:
            print('\nBinarizing contrast ' + contrast + ' in seed ' + seed)
            pathToContrast = os.path.join(pathToContrasts, contrast)
            binaryContrast = makeBinary(pathToContrast)
            binaryBetweenContrasts.append(binaryContrast)

        # Merge the within contrasts with their respective partner (+1)
        for run, i  in enumerate(np.arange(0, 6, 2)):
            # Binary contrasts
            posFile = binaryWithinContrasts[i]
            negFile = binaryWithinContrasts[i + 1]
            print('\nMerging positive contrast ' + posFile + ' and negative '
                  + ' contrast ' + negFile + ' in seed ' + seed)
            mergeFile = withinSubjectBinMerged[run]
            mergePath = os.path.join(pathToContrasts, mergeFile)
            # Now merge the files
            mergedBinFile = mergeInverse(posFile, negFile, mergePath)
            mergedBinWithinFiles.append(mergedBinFile)

            # non-binary files
            posFile = withinSubjectContrasts[i]
            negFile = withinSubjectContrasts[i + 1]
            posPath = os.path.join(pathToContrasts, posFile)
            negPath = os.path.join(pathToContrasts, negFile)
            mergeFile = withinSubjectMerged[run]
            mergePath = os.path.join(pathToContrasts, mergeFile)
            mergedFile = mergeInverse(posPath, negPath, mergePath)
            mergedWithinFiles.append(mergedFile)

        # Merge merged whithin contrasts with between contrasts
        for run, i  in enumerate(np.arange(0, 6, 2)):
            contrast = mergedBinWithinFiles[run]
            increaseContrast = binaryBetweenContrasts[2]
            decreaseContrast = binaryBetweenContrasts[3]
            print('merging within subject contrast ' + contrast + '\n'
                  + ' with ' + increaseContrast + ' for seed ' + seed)
            increasingOutputFile = betweenSubjectMerged[i]
            increasingOutputPath = os.path.join(pathToContrasts,
                                                increasingOutputFile)
            print('merging within subject contrast ' + contrast + '\n'
                  + ' with ' + decreaseContrast + ' for seed ' + seed)
            decreasingOutputFile = betweenSubjectMerged[i + 1]
            decreasingOutputPath = os.path.join(pathToContrasts,
                                                decreasingOutputFile)
            # make increasing contrast merge
            mergeWithinBetween(contrast, increaseContrast, increasingOutputPath)
            # make decreasing contrast merge
            mergeWithinBetween(contrast, decreaseContrast, decreasingOutputPath)


if __name__ == '__main__':
    Main()
