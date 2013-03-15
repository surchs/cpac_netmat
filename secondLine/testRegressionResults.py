'''
Created on Mar 15, 2013

@author: surchs
'''
import os
import sys
import time
import numpy as np
import pandas as pa
import statsmodels.api as sm
from scipy import stats as st
from sklearn import linear_model
from matplotlib import pyplot as plt



def loadConnectome(pathToConnectomeFile):
    connectome = np.loadtxt(pathToConnectomeFile)

    return connectome


def loadPhenotypicFile(pathToPhenotypicFile):
    pheno = pa.read_csv(pathToPhenotypicFile)

    return pheno


def stackConnectome(connectomeStack, connectome):
    # See if stack is empty, if so, then initialize
    if connectomeStack.size == 0:
        connectomeStack = connectome[..., None]
    else:
        connectomeStack = np.concatenate((connectomeStack,
                                         connectome[..., None]),
                                        axis=2)

    return connectomeStack


def splitConnectome(connectomeStack, index):
    '''
    Method to get the individual connectome out of the stack again
    '''
    # The stack is ordered (roi X roi X subjects)
    connectome = connectomeStack[..., index]

    return connectome


def stackCov(covStack, cov):
    covStack = np.append(covStack, cov)

    return covStack


def fisherZ(connectome):
    normalizedConnectome = np.arctanh(connectome)

    return normalizedConnectome


def makeSiteRegressors(siteStack):
    '''
    Method that generates one regressor per site
    '''
    # Get the number of sites
    sites = np.unique(siteStack)
    numSubs = len(siteStack)
    print('I found ' + str(len(sites)) + ' Sites for ' + str(numSubs)
          + ' subjects:\n' + str(sites))

    siteRegressorMatrix = np.array([])

    # Fill Regressor matrix
    for siteID in sites:
        tempSiteVector = np.zeros(numSubs)
        # Add 1 for subjects that are from the current session
        tempSiteVector[siteStack == siteID] = 1

        # Add vector to Matrix
        if siteRegressorMatrix.size == 0:
            siteRegressorMatrix = tempSiteVector[..., None]
        else:
            siteRegressorMatrix = np.concatenate((siteRegressorMatrix,
                                                  tempSiteVector[..., None]),
                                                 axis=1)

        # Print out how many subjects are member of this site
        print('Found ' + str(np.sum(tempSiteVector)) + ' subjects with site '
              + str(siteID))

    # Show shape of regressorMatrix
    print('Regressor shape: ' + str(siteRegressorMatrix.shape))

    return siteRegressorMatrix


def runGLM(dataVector, predMat):
    # run a glm with only one factor
    model = sm.OLS(dataVector, predMat)
    results = model.fit()
    residuals = results.resid

    return residuals


def alternativeGLM(dataVector, predMat):
    # run a glm with only one factor
    model = linear_model.LinearRegression()
    # Fit the model
    results = model.fit(predMat, dataVector, n_jobs=1)
    # Get the residuals
    residuals = dataVector - model.predict(predMat)


    return residuals


def runConnections(connectomeStack, regressorMatrix):
    '''
    Loop through all the connections while doing the GLM
    '''
    # First get the lower triangle of the connectome (the unique connections)
    mask = np.ones_like(connectomeStack[..., 0])
    lower = np.tril(mask, -1)
    upper = np.triu(mask, 1)

    # Get the flat connectome in (nConnections by nSubjects)
    flatConnectome = connectomeStack[lower == 1]
    print('The flat connectome has shape: ' + str(flatConnectome.shape))
    print('\nRunning regression now!')

    flatConnections, flatSubjects = flatConnectome.shape

    # Prepare Containers for output (two, for comparison)
    flatOutConnectome = np.zeros_like(flatConnectome)
    flatOutStack = np.array([])
    outConnectome = np.zeros_like(connectomeStack)

    # Now start iterating across all connections and run the GLM for each of
    # them

    tookTime = np.array([])
    for i, connVec in enumerate(flatConnectome):
        # This gets us the index and the timeseries for the current connection
        start = time.time()
        percComplete = np.round(float(i + 1) / (flatConnections) * 100, 1)
        remaining = flatConnections - (i + 1)

        # Short sanity check
        if not len(connVec) == flatSubjects:
            print('\nConnVec has the wrong length!\n'
                  + '    got: ' + str(len(connVec)) + '\n'
                  + '    expected: ' + str(flatSubjects) + '\n'
                  + ' BREAKING!\n\n')
            break
        else:
            # All is good, run the model, it returns only the residuals
            # result = runGLM(connVec, regressorMatrix)
            # newConnVec = result.resid

            # Run the alternative to cut computing bloat
            # newConnVec = runGLM(connVec, regressorMatrix)
            newConnVec = alternativeGLM(connVec, regressorMatrix)

            # Now get the result into the output matrix
            flatOutConnectome[i, :] = newConnVec
            # And for testing, the other one too
            if flatOutStack.size == 0:
                flatOutStack = newConnVec[None, ...]
            else:
                flatOutStack = np.concatenate((flatOutStack,
                                               newConnVec[None, ...]),
                                              axis=0)

        # My little time gadget...
        stop = time.time()
        took = stop - start
        tookTime = np.append(tookTime, took)
        avgTook = np.average(tookTime)
        remTime = np.round((avgTook * remaining), 2)
        sys.stdout.write('\r' + str(percComplete) + '% done. ' + str(remTime)
                         + ' more seconds to go...        ')
        sys.stdout.flush()

    # Done looping, let's see about these results
    if np.array_equal(flatOutStack, flatOutConnectome):
        print('\n\nHorray! All wen\'t well with the regression')
    else:
        print('\n\nOh no! Somehow the two outfiles are different!\n'
              + '    tome: ' + str(flatOutConnectome.shape) + '\n'
              + '    stack: ' + str(flatOutStack.shape))


    # So, we are done, map that shit back!
    # Now since there is a time component involved, this is somewhat more
    # tricky than usual
    for index in np.arange(flatSubjects):
        # Get the vector of lower triangle connections for the current
        # subject
        connVec = flatOutConnectome[:, index]
        # put in the lower triangle
        outConnectome[..., index][lower == 1] = connVec
        # put in the upper triangle
        outConnectome[..., index].T[upper.T == 1] = connVec

    return outConnectome


def saveConnectome(connectome, outputFilePath):
    np.savetxt(outputFilePath, connectome, fmt='%.12f')
    return 'cool'


def Main():
    # Define inputs
    pathToConnectomeDir = '/home2/surchs/secondLine/connectomes/abide/dos160'
    pathToPhenotypicFile = '/home2/surchs/secondLine/configs/abide/abide_across_236_combined_pheno.csv'
    pathToSubjectList = '/home2/surchs/secondLine/configs/abide/abide_across_236_subjects.csv'

    connectomeSuffix = '_connectome_glob.txt'
    corrConnSuffix = '_connectome_glob_corr.txt'

    # Define parameters
    doDebug = False

    # Define outputs
    pathToConnectomeOutDir = '/home2/surchs/secondLine/connectomes/abide/dos160'
    outFileSuffix = '_connectome_glob_corr.txt'

    # Read subject list
    subjectListFile = open(pathToSubjectList, 'rb')
    subjectList = subjectListFile.readlines()

    # Read the phenotypic file
    pheno = loadPhenotypicFile(pathToPhenotypicFile)
    phenoSubjects = pheno['SubID'].tolist()
    phenoAges = pheno['age'].tolist()
    meanFD = pheno['MeanFD'].tolist()
    site = pheno['Site']

    # Prepare container variables
    connectomeStack = np.array([])
    corrConnStack = np.array([])
    ageStack = np.array([])
    meanFdStack = np.array([])
    siteStack = np.array([])

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
        # Get covariates
        phenoAge = phenoAges[i]
        phenoMeanFd = meanFD[i]
        phenoSite = site[i]

        # Stack the covariates
        ageStack = stackCov(ageStack, phenoAge)
        siteStack = stackCov(siteStack, phenoSite)
        meanFdStack = stackCov(meanFdStack, phenoMeanFd)

        # Construct the path to the connectome file of the subject
        pathToConnectomeFile = os.path.join(pathToConnectomeDir,
                                            (subject + connectomeSuffix))
        pathToCorrectedConnectome = os.path.join(pathToConnectomeDir,
                                                 (subject + corrConnSuffix))

        # Load the connectome for the subject
        connectome = loadConnectome(pathToConnectomeFile)
        corrConn = loadConnectome(pathToCorrectedConnectome)
        # Normalize the connectome
        normalizedConnectome = fisherZ(connectome)

        # Stack the connectome
        connectomeStack = stackConnectome(connectomeStack, normalizedConnectome)
        corrConnStack = stackConnectome(corrConnStack, corrConn)
        print('connectome: ' + str(connectomeStack.shape))
        print('corrected: ' + str(corrConnStack.shape))


    # Done stacking this stuff up
    # Create the Site regressors
    # First, find the number of different sites
    siteRegressorMatrix = makeSiteRegressors(siteStack)
    # Create the regressor matrix
    regressorMatrix = np.concatenate((meanFdStack[..., None],
                                      siteRegressorMatrix),
                                     axis=1)
    print('All regressors in. This is their shape: '
          + str(regressorMatrix.shape))

    # Now plot the vectors again
    connVec = connectomeStack[10, 14, :]
    corrVec = corrConnStack[10, 14, :]
    plt.plot(ageStack, connVec, 'ko')
    plt.plot(ageStack, corrVec, 'b.')
    plt.show()
    raw_input('Enter to go on...')
    plt.close()

if __name__ == '__main__':
    Main()
    pass
