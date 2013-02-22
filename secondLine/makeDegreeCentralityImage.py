'''
Created on Feb 22, 2013

@author: surchs
'''
import numpy as np
import nibabel as nib


def testMakeDegreeCentralityImage():
    pathToRoiFile = '/home2/surchs/masks/ROIs/craddock200wave_p1l.nii.gz'
    pathToDegreeCentrality = '/home2/surchs/secondLine/degree_centrality/group_degree_centrality.txt'
    
    pathToOutputFile = '/home2/surchs/secondLine/degree_centrality/degree_centrality_volumes/group.nii.gz'
    
    roiImage, roiData = loadNiftiImage(pathToRoiFile)
    degreeCentrality = loadDegreeCentrality(pathToDegreeCentrality)
    # Quickfix for the group level degree centrality
    groupDegreeCentrality = np.average(degreeCentrality, axis=0)
    degreeCentralityVolume = mapDegreeCentrality(groupDegreeCentrality, roiData)
    status = saveDegreeCentralityVolume(degreeCentralityVolume, pathToOutputFile, roiImage)
    print(status)

    
def loadNiftiImage(pathToNiftiFile):
    image = nib.load(pathToNiftiFile)
    data = image.get_data()
    
    return image, data

def loadDegreeCentrality(pathToDegreeCentrality):
    degreeCentrality = np.loadtxt(pathToDegreeCentrality)
    
    return degreeCentrality

def mapDegreeCentrality(degreeCentrality, roiData):
    # Get the list of unique nonzero elements in the ROI data
    uniqueElements = np.unique(roiData[roiData!=0])
    # Make an empty copy of the roiData to store the degree centrality values
    # in
    degreeCentralityVolume = np.zeros_like(roiData, dtype='float64')
    # Make sure that the number of unique ROI elements is equal to the number
    # of degree centrality values
    if not len(uniqueElements) == len(degreeCentrality):
        raise Exception('The number of ROIs and values of degree centrality '
                        + 'doesn\'t match! ROIs: '
                        + str(len(uniqueElements)) + ' DC: '
                        + str(len(degreeCentrality)))
    else:
        for i, roi in enumerate(uniqueElements):
            # Assign degree centrality value to ROI voxels in the volume
            degreeCentralityVolume[roiData == roi] = degreeCentrality[i]
    
    return degreeCentralityVolume


def saveDegreeCentralityVolume(degreeCentralityVolume, pathToOutputFile, template):
    outNifti = nib.Nifti1Image(degreeCentralityVolume, 
                               template.get_affine(), 
                               template.get_header())
    nib.nifti1.save(outNifti, pathToOutputFile)
    
    return 'cool'
    

if __name__ == '__main__': 
    testMakeDegreeCentralityImage()
            


