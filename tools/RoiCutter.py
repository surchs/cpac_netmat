'''
Created on Nov 26, 2012

@author: sebastian

script to take a ROI mask and reduce it according to a group mask
the difference to simply multiplying it is that a minimum threshold of
in-brain voxels is applied to every ROI to be included in the final mask

'''
import os
import sys
import numpy as np
import nibabel as nib


def Main(roiMask, groupMask, outMask):
    '''
    Main method
    '''
    roi = nib.load(roiMask)
    group = nib.load(groupMask)

    roiData = roi.get_data()
    groupData = group.get_data()
    if len(np.unique(groupData)) > 2:
        print('something wrong with groupData:'
              + '\nUnique values are: ' + np.unique(groupData))
    maskedData = roiData * groupData
    saveMask = np.zeros_like(groupData, dtype=int)

    rois = np.unique(roiData[roiData != 0])
    # now loop through the rois and determine if they can still be in the mask
    for roi in rois:
        nVoxels = len(roiData[roiData == roi])
        maskedVoxels = len(maskedData[maskedData == roi])
        if maskedVoxels < 0.3 * nVoxels:
            print('roi ' + str(roi) + ' is out!')
        else:
            saveMask[maskedData == roi] = roi

    print('Gone through all rois, preparing outmask')
    outImg = nib.Nifti1Image(saveMask, group.get_affine(), group.get_header())
    nib.save(outImg, outMask)
    print('Done')


if __name__ == '__main__':
    roiMask = sys.argv[1]
    groupMask = sys.argv[2]
    outMask = sys.argv[3]
    Main(roiMask, groupMask, outMask)
    pass
