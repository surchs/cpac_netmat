'''
Created on Nov 26, 2012

@author: sebastian

script to generate a group mask of all voxels that have nonzero variance

'''
import os
import sys
import glob
import numpy as np
import nibabel as nib
import multiprocessing as mp


def MakeSubjectMask(args):
    '''
    run the subject
    '''
    (subjectDir, funcPath, funcFileName) = args
    funcSearch = glob.glob((subjectDir + funcPath + funcFileName))
    if not funcSearch:
        print('none for ' + subjectDir)
    else:
        funcFile = funcSearch[0]

    funcImg = nib.load(funcFile)
    funcData = funcImg.get_data()

    # this will probably take some time, but we need to get the mask
    maskData = np.zeros_like(funcData[..., 0], dtype=int)
    stdMask = np.std(funcData, axis=3)
    maskData[stdMask != 0] = 1
    if len(np.unique(maskData)) == 1:
        print subjectDir, 'has empty funcmatrix'
    return maskData


def Main(templateFile, tempMask, nProc):
    '''
    get the subjects, make individual masks of nonzero variance across time
    and then make a groupmask and output it.
    '''
    pathFile = open(templateFile)
    pathLines = pathFile.readlines()

    for line in pathLines:
        useLine = line.strip().split()
        pipeName = useLine[0]
        subDir = useLine[1]
        funcPath = useLine[2]
        funcFileName = useLine[3]

        tempImg = nib.load(tempMask)
        subList = []

        print('Running pipeline ' + pipeName)
        print('\nRunning subjects in path ' + subDir)
        print('\nThe path to the functional is ' + funcPath)
        print('\nThe filename I am looking for is ' + funcFileName)

        # second level loop for the pipelines
        for subject in os.listdir(subDir):
            # got the subject, run inside it
            subjectDir = os.path.abspath(os.path.join(subDir, subject))
            subList.append((subjectDir, funcPath, funcFileName))

        print('prepared to run multicore')
        pool = mp.Pool(processes=nProc)
        resultList = pool.map(MakeSubjectMask, subList)
        print('ran multicore')

        print('\nDone with subject level, going on group level now')
        groupMask = np.array([])
        for mask in resultList:
            if groupMask.size == 0:
                groupMask = mask
            else:
                groupMask = groupMask * mask

        groupImg = nib.Nifti1Image(groupMask, tempImg.get_affine(),
                                   tempImg.get_header())
        outName = (pipeName + '_groupmask_wave1.nii.gz')
        print('Done with the groupmask, saving to disk as ' + outName)
        nib.save(groupImg, outName)

if __name__ == '__main__':
    templateFile = sys.argv[1]
    tempMask = sys.argv[2]
    nProc = int(sys.argv[3])
    Main(templateFile, tempMask, nProc)
    pass
