'''
Created on Feb 22, 2013

@author: surchs
'''
import numpy as np
import nibabel as nib
from scipy import ndimage


def testMakeDistances():
    # Define inputs
    pathToInputImage = '/home2/surchs/secondLine/masks/dos160_abide_246_3mm.nii.gz'

    # Define outputs
    pathToOutputFile = ''

    # Get data
    image, data = loadNiftiImage(pathToInputImage)
    uniqueValues = np.unique(data[data != 0])
    # Compute distance for two fixed ROIs
    roi1 = uniqueValues[0]
    roi2 = uniqueValues[1]
    print('Testing with roi1 = ' + str(roi1) + ' and roi2 = ' + str(roi2))
    coordRoi1 = computeRoiCenterOfMass(data, roi1)
    coordRoi2 = computeRoiCenterOfMass(data, roi1)
    # Compute distance
    distance = computeDistance(coordRoi1, coordRoi2)
    print('Distance between roi1 (' + str(roi1) + '/' + str(coordRoi1) + ') '
          + 'and roi2 (' + str(roi2) + '/' + str(coordRoi2) + ') '
          + 'is ' + str(distance))


def loadNiftiImage(pathToNiftiFile):
    image = nib.load(pathToNiftiFile)
    data = image.get_data()

    return image, data

def computeRoiCenterOfMass(data, roi):
    # Compute the center of mass for a given ROI
    coordinateTuple = ndimage.measurements.center_of_mass(data == roi)
    # Transform the coordinate tuple to a numpy array for easier usage
    coordinate = np.array(coordinateTuple)

    return coordinate

def computeDistance(a, b):
    # Points have to be in 3D coordinates for now - check here
    if not len(a) == 3 or not len(b) == 3:
        raise Exception('Provided points are not 3D coordinates\n'
                        + 'They are: a ' + str(a) + ' b ' + str(b))

    # Compute offset between points on each axis
    offset = a - b
    distX = offset[0]
    distY = offset[1]
    distZ = offset[2]
    # Compute distance between points
    distance = np.sqrt(np.square(distX) + np.square(distY) + np.square(distZ))

    return distance


def stackRoiDistances(roiDistances, distance):
    # If this is the first time that the stack is used, initiate
    if roiDistances.size == 0:
        roiDistances = distance[None, ...]
    else:
        roiDistances = np.concatenate((roiDistances,
                                       distance[None, ...]),
                                      axis=0)

    return roiDistances


def saveRoiDistances(outputFilePath, roiDistances):
    np.savetxt(outputFilePath, roiDistances, fmt='%.12f')
    status = 'cool'

    return status


def Main():
    # Define inputs
    pathToInputImage = '/home2/surchs/secondLine/masks/dos160_abide_246_3mm.nii.gz'

    # Define outputs
    pathToOutputFile = '/home2/surchs/secondLine/roiDistances/dos160abide246_3mm_distances.txt'

    # Load the ROI Image
    image, data = loadNiftiImage(pathToInputImage)
    uniqueValues = np.unique(data[data != 0])

    # Prepare a container matrix to store the distances for all ROI combinations
    roiDistances = np.array([])

    # Loop through the ROIs inside the image
    for baseRoi in uniqueValues:
        # Get the coordinates of the center of mass of the current ROI that
        # I want to measure the distance from
        baseRoiCoords = computeRoiCenterOfMass(data, baseRoi)

        # Print message so I don't get bored watching
        print('Running roi ' + str(baseRoi) + ' at ' + str(baseRoiCoords))

        # Prepare a container variable for the vector of distances for the
        # current ROI
        baseRoiDistances = np.array([])

        for compareRoi in uniqueValues:
            # Get the coordinates of the center of mass of the ROI that I
            # want to get the distance to
            compareRoiCoords = computeRoiCenterOfMass(data, compareRoi)

            # Compute the distance between the base Roi and the compare Roi
            distance = computeDistance(baseRoiCoords, compareRoiCoords)

            # Append the distance to the base Roi distance vector
            baseRoiDistances = np.append(baseRoiDistances, distance)

        # Stack the current base ROI to the roi distance matrix
        roiDistances = stackRoiDistances(roiDistances, baseRoiDistances)

    # Now save the roiDistance matrix to a txt file
    status = saveRoiDistances(pathToOutputFile, roiDistances)
    print('Status says ' + status)


if __name__ == '__main__':
    Main()
