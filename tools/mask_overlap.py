'''
Created on Nov 21, 2012

@author: sebastian
'''
import sys
import numpy as np
import nibabel as nib


def Main(mask1file, mask2file, saveFile):
    '''
    check if the two files are of the same dimensions and if they are, output
    a nifit file containing their exclusive values
    '''
    m1 = nib.load(mask1file)
    mask1 = m1.get_data()
    m2 = nib.load(mask2file)
    mask2 = m2.get_data()

    if mask1.shape == mask2.shape:
        print('mask1 and mask2 have the same shape. good job!')
        mask1mask = np.zeros_like(mask1)
        mask2mask = np.zeros_like(mask2)

        mask1mask[mask1 != 0] = 1
        mask2mask[mask2 != 0] = 2

        outmask = mask1mask + mask2mask
        # get rid of the overlap, we don't care too much about that
        outmask[outmask == 3] = 0

        outImg = nib.Nifti1Image(outmask, m1.get_affine(), m1.get_header())
        nib.save(outImg, saveFile)

    else:
        print('mask1 and mask2 have different shapes:\n'
              + '    mask1: ' + mask1.shape + '\n'
              + '    mask2: ' + mask2.shape)


if __name__ == '__main__':
    mask1file = sys.argv[1]
    mask2file = sys.argv[2]
    saveFile = sys.argv[3]
    Main(mask1file, mask2file, saveFile)
    pass
