'''
Created on Feb 22, 2013

@author: surchs
'''
import os
import gzip
import cPickle
import numpy as np
import pandas as pa
import nibabel as nib
import statsmodels.api as sm
from scipy import stats as st
from matplotlib import pyplot as plt


def loadPhenotypicFile(pathToPhenotypicFile):
    pheno = pa.read_csv(pathToPhenotypicFile)

    return pheno


def loadConnectome(pathToConnectomeFile):
    connectome = np.loadtxt(pathToConnectomeFile)

    return connectome


def loadArchive(pathToArchive):
    f = gzip.open(pathToArchive)
    archive = cPickle.load(f)

    return archive


def loadNiftiImage(pathToNiftiFile):
    image = nib.load(pathToNiftiFile)
    data = image.get_data()

    return image, data


def stackConnectome(connectomeStack, connectome):
    # See if stack is empty, if so, then initialize
    if connectomeStack.size == 0:
        connectomeStack = connectome[..., None]
    else:
        connectomeStack = np.concatenate((connectomeStack,
                                         connectome[..., None]),
                                        axis=2)

    return connectomeStack


def stackAges(ageStack, age):
    ageStack = np.append(ageStack, age)

    return ageStack


def getUniqueMatrixElements(squareMatrix):
    # When passed a square similarity matrix, returns the lower triangle of
    # said matrix as a vector of appended rows
    if not squareMatrix.shape[0] == squareMatrix.shape[1]:
        print('Your matrix of shape ' + str(squareMatrix.shape)
              + ' is not symmetrical')
        uniqueElements = squareMatrix.flatten()
    else:
        # Make a mask for the lower triangle of the matrix
        mask = np.ones_like(squareMatrix)
        mask = np.tril(mask, -1)
        # Mask the matrix to retrieve only the lower triangle
        uniqueElements = squareMatrix[mask == 1]

    return uniqueElements


def fisherZ(connectome):
    normalizedConnectome = np.arctanh(connectome)

    return normalizedConnectome


def dualPlot(age, meanWithin, meanBetween, title):

    fig, (within, between) = plt.subplots(1, 2, sharex=True, sharey=False)

    # fitshit
    wP = np.polyfit(age, meanWithin, 1)
    bP = np.polyfit(age, meanBetween, 1)
    xnew = np.arange(age.min() - 1, age.max() + 1, 0.1)
    wFit = np.polyval(wP, xnew)
    bFit = np.polyval(bP, xnew)

    within.set_title('within network')
    between.set_title('between network')

    withinCorr, withinP = st.pearsonr(age, meanWithin)
    within.plot(age, meanWithin, 'k.')
    within.plot(xnew, wFit, 'r', label=(str(np.round(withinCorr, 2))
                                        + ' '
                                        + str(np.round(withinP, 4))))
    within.set_xlabel('mean connectivity')
    within.set_ylabel('age')
    within.legend()


    betweenCorr, betweenP = st.pearsonr(age, meanBetween)
    between.plot(age, meanBetween, 'k.')
    between.plot(xnew, bFit, 'b', label=(str(np.round(betweenCorr, 2))
                                         + ' '
                                         + str(np.round(betweenP, 4))))
    between.set_xlabel('mean connectivity')
    between.set_ylabel('age')
    between.legend()

    fig.suptitle(title)
    plt.show()
    raw_input("Press Enter to continue...")
    plt.close()


def saveOutput(outputFilePath, outputMatrix):
    np.savetxt(outputFilePath, outputMatrix, fmt='%.12f')
    status = 'cool'

    return status


def Main():
    # Define the inputs
    pathToConnectomeDir = '/home2/surchs/secondLine/connectomes/abide/dos160'
    pathToPhenotypicFile = '/home2/surchs/secondLine/configs/abide/abide_across_236_pheno.csv'
    pathToSubjectList = '/home2/surchs/secondLine/configs/abide/abide_across_236_subjects.csv'

    pathToNetworkNodes = '/home2/surchs/secondLine/configs/networkNodes_dosenbach.dict'
    pathToRoiMask = '/home2/surchs/secondLine/masks/dos160_abide_246_3mm.nii.gz'

    connectomeSuffix = '_connectome_glob.txt'

    runwhat = 'glm'
    doPlot = False

    # Define parameters
    alpha = 0.1
    childmax = 12.0
    adolescentmax = 18.0

    # Read subject list
    subjectListFile = open(pathToSubjectList, 'rb')
    subjectList = subjectListFile.readlines()

    # Read the phenotypic file
    pheno = loadPhenotypicFile(pathToPhenotypicFile)
    phenoSubjects = pheno['SubID'].tolist()
    phenoAges = pheno['age'].tolist()

    # Read network nodes
    networkNodes = loadArchive(pathToNetworkNodes)
    # Load the ROI mask
    roiImage, roiData = loadNiftiImage(pathToRoiMask)
    # get the unique nonzero elements in the ROImask
    uniqueRoi = np.unique(roiData[roiData != 0])

    # Prepare the containers
    connectomeStack = np.array([])
    ageStack = np.array([])
    meanConnStack = np.array([])

    # Loop through the subjects
    for i, subject in enumerate(subjectList):
        subject = subject.strip()
        phenoSubject = phenoSubjects[i]
        # Workaround for dumb ass pandas
        phenoSubject = ('00' + str(phenoSubject))

        if not subject == phenoSubject:
            raise Exception('The Phenofile returned a different subject name '
                            + 'than the subject list:\n'
                            + 'pheno: ' + phenoSubject + ' subjectList '
                            + subject)

        # Get the age of the subject from the pheno file
        phenoAge = phenoAges[i]
        # Construct the path to the connectome file of the subject
        pathToConnectomeFile = os.path.join(pathToConnectomeDir,
                                            (subject + connectomeSuffix))
        # Load the connectome for the subject
        connectome = loadConnectome(pathToConnectomeFile)
        normalizedConnectome = fisherZ(connectome)

        # Get the mean connectivity
        uniqueConnections = getUniqueMatrixElements(normalizedConnectome)
        meanConn = np.mean(uniqueConnections)
        meanConnStack = np.append(meanConnStack, meanConn)

        # Stack the connectome
        connectomeStack = stackConnectome(connectomeStack, normalizedConnectome)
        print('connectomeStack: ' + str(connectomeStack.shape))
        # Stack ages
        ageStack = stackAges(ageStack, phenoAge)

    # Now we have the connectome stack
    # Let's loop through the networks again
    for network in networkNodes.keys():
        print('plotting network ' + network)
        netNodes = networkNodes[network]
        # get boolean index of within network nodes
        networkIndex = np.in1d(uniqueRoi, netNodes)
        # make a boolean index for within and between
        betweenIndex = networkIndex != True

        # Get the network connections
        networkStack = connectomeStack[networkIndex, ...]

        # Get the within network connections
        withinStack = networkStack[:, networkIndex, ...]
        # Get the lower triangle of these
        withinMask = np.ones_like(withinStack[..., 0])
        withinMask = np.tril(withinMask, -1)
        withinMatrix = withinStack[withinMask == 1]
        # Get mean connectivity within
        meanWithin = np.average(withinMatrix, axis=0)

        # Get the between network connections
        betweenStack = networkStack[:, betweenIndex, ...]
        betweenRows, betweenCols, betweenSubs = betweenStack.shape
        # Also flatten this stuff out
        betweenMatrix = np.reshape(betweenStack, (betweenRows * betweenCols,
                                                  betweenSubs))
        # Get mean connectivity between
        meanBetween = np.average(betweenMatrix, axis=0)

        # Plot both
        dualPlot(ageStack, meanWithin, meanBetween, network)


if __name__ == '__main__':
    Main()
