'''
Created on Nov 26, 2012

@author: sebastian

script to generate a group mask of all voxels that have nonzero variance

'''
import os
import numpy as np
import nibabel as nib
import multiprocessing as mp


def MakeSubjectMask(args):
    '''
    run the subject
    '''
    (subjectDir, funcPath, funcFileName) = args
    searchString = os.path.join(subjectDir, funcPath, funcFileName)
    if not os.path.isfile(searchString):
        print('none for ' + searchString)
    else:
        funcFile = searchString

    funcImg = nib.load(funcFile)
    funcData = funcImg.get_data()

    # this will probably take some time, but we need to get the mask
    maskData = np.zeros_like(funcData[..., 0], dtype=int)
    stdMask = np.std(funcData, axis=3)
    maskData[stdMask != 0] = 1
    if len(np.unique(maskData)) == 1:
        print subjectDir, 'has empty funcmatrix'
    return maskData


def Main():
    '''
    get the subjects, make individual masks of nonzero variance across time
    and then make a groupmask and output it.
    '''

    # Define inputs
    pathToTemplateMask = '/home2/surchs/Templates/MNI152_T1_3mm_brain_mask.nii.gz'
    pathToSubDir = '/home2/data/Projects/netmat/outputdirs/abide_246/sym_links/pipeline_HackettCity/linear1.wm1.global1.motion1.csf1_CSF_0.98_GM_0.7_WM_0.98/'
    pathToFuncDir = 'scan_rest_1_rest/func/bandpass_freqs_0.009.0.08'
    funcFileName = 'functional_mni.nii.gz'
    pathToSubjectList = '/home2/surchs/secondLine/configs/abide/combined_across_within_test.txt'
    subjectSuffix = '_session_1'

    # Define parameters
    nProcs = 20

    # Define outputs
    pathToOutFile = '/home2/surchs/secondLine/masks/abide_across_246_groupmask_3mm.nii.gz'

    # Get the subject list
    subjectListFile = open(pathToSubjectList, 'rb')
    subjectList = subjectListFile.readlines()

    # Prepare list of subjects functional files
    subjectFuncPathList = []

    # Get the template mask
    tempImg = nib.load(pathToTemplateMask)

    # Tell some stuff
    print('\nRunning subjects in path ' + pathToSubDir)
    print('The path to the functional is ' + pathToFuncDir)
    print('The filename I am looking for is ' + funcFileName)

    # Get the contents of the subject directory
    subjectDirs = os.listdir(pathToSubDir)

    # Now loop through the subjects in the subject list
    for subject in subjectList:
        subject = subject.strip()
        testSubject = (subject + subjectSuffix)

        # Check if subject exists in the subject directory
        if testSubject in subjectDirs:
            print('I found ' + subject + ' in the subject dir')
            subjectDir = os.path.abspath(os.path.join(pathToSubDir,
                                                      testSubject))
        else:
            print('PROBLEM: I did not find ' + subject + ' in subject dir')
            continue

        # got the subject, run inside it
        subjectFuncPathList.append((subjectDir, pathToFuncDir, funcFileName))

    print('prepared to run multicore')
    pool = mp.Pool(processes=nProcs)
    resultList = pool.map(MakeSubjectMask, subjectFuncPathList)
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
    print('Done with the groupmask, saving to disk as ' + pathToOutFile)
    nib.save(groupImg, pathToOutFile)


if __name__ == '__main__':
    Main()
    pass
