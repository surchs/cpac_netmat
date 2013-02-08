'''
Created on Feb 8, 2013

@author: sebastian

corr tester
'''
import os
import sys
import commands
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt


def Main(subFile, basePath, seed, seedOut, fileMask):
    # get the subjects
    subF = open(subFile, 'rb')
    sublist = subF.readlines()

    outDir = '/home2/surchs/testing/scaTest/'

    roiTimePath = 'roi_timeseries/_scan_func_rest/_csf_threshold_0.98/_gm_threshold_0.7/_wm_threshold_0.98/_compcor_ncomponents_5_selector_pc10.linear1.wm1.global1.motion1.quadratic0.gm0.compcor0.csf1/_bandpass_freqs_0.009.0.1/_roi_rois_1mm'
    roiTimeFile = 'roi_rois_1mm.1D'

    functionalMNI = 'functional_mni/_scan_func_rest/_csf_threshold_0.98/_gm_threshold_0.7/_wm_threshold_0.98/_compcor_ncomponents_5_selector_pc10.linear1.wm1.global1.motion1.quadratic0.gm0.compcor0.csf1/_bandpass_freqs_0.009.0.1'
    funcMniFile = 'bandpassed_demeaned_filtered_warp.nii.gz'

    cpacZPath = 'sca_roi_Z/_scan_func_rest/_csf_threshold_0.98/_gm_threshold_0.7/_wm_threshold_0.98/_compcor_ncomponents_5_selector_pc10.linear1.wm1.global1.motion1.quadratic0.gm0.compcor0.csf1/_bandpass_freqs_0.009.0.1/_roi_rois_1mm/'
    cpacZFile = 'z_score_ROI_number_'
    for sub in sublist:
        # run the correlation for this subject
        subPath = os.path.join(basePath, sub)
        roiPath = os.path.join(subPath, roiTimePath, roiTimeFile)
        funcPath = os.path.join(subPath, functionalMNI, funcMniFile)

        outFile = (outDir + sub + '.nii.gz')

        # run the correlation in MNI space
        commands.getoutput('3dTcorr1D  -pearson -prefix %s  %s  %s'
                           % (outFile, funcPath, roiPath))

        seedIndex = {}
        seedString = ''
        roiF = open(roiPath, 'rb')
        seedList = roiF.readline().strip().split('\t')
        seedRange = np.arange(len(seedList))
        for i in seedRange:
            seedNum = seedList[i].replace('#', '')
            seedIndex[seedNum] = i
            seedString = (seedString + str(i) + ', ' + seedNum + '\n')

    f = open(seedOut, 'wb')
    f.writelines(seedString)

    # so now we have the files, print them out
    print('Done with running the files through')
    # now loop through the subjects again and compare CPAC to my stuff
    seedVolume = seedIndex[seed]

    for sub in sublist:
        print('plotting ' + sub)
        cpacPath = os.path.join(subPath, cpacZPath)
        cpacFile = (cpacPath + cpacZFile + seed + '.nii.gz')
        outFile = (outDir + sub + '.nii.gz')

        cFile = nib.load(cpacPath)
        cData = cFile.get_data()

        mFile = nib.load(outFile)
        mData = mFile.get_data()
        mSeedData = mData[..., seedVolume]
        mSeedData = mSeedData.reshape(mSeedData.shape[:3])

        Mask = nib.load(fileMask)
        maskData = Mask.get_data()

        # this is already flattened out, hopefully in the same way
        cMaskData = cData[Mask == 1]
        mMaskData = mSeedData[Mask == 1]

        plt.plot(mMaskData, cMaskData, 'ko')
        plt.title((sub + ' w seed: ' + seed))
        plt.xlabel('mni2mni')
        plt.ylabel('mni2native')
        plt.show()
        inp = raw_input("Press to continue")
        plt.close()


if __name__ == '__main__':
    subFile = sys.argv[1]
    basePath = sys.argv[2]
    seed = sys.argv[3]
    seedOut = sys.argv[4]
    fileMask = sys.argv[5]
    Main(subFile, basePath)
    pass