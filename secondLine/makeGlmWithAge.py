'''
Created on Feb 22, 2013

@author: surchs
'''
import os
import sys
import time
import copy
import numpy as np
import pandas as pa
import statsmodels.api as sm
from scipy import stats as st
from matplotlib import pyplot as plt


def loadPhenotypicFile(pathToPhenotypicFile):
    pheno = pa.read_csv(pathToPhenotypicFile)

    return pheno


def loadConnectome(pathToConnectomeFile):
    connectome = np.loadtxt(pathToConnectomeFile)

    return connectome


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


def normalize(connectome):
    '''
    Method that normalizes the connectome
    Should be run after fisher-r transform
    '''
    mean = np.mean(connectome, axis=2)
    sd = np.std(connectome, axis=2)
    numSubs = connectome.shape[2]
    normConnectome = copy.deepcopy(connectome)
    # Now loop through and do this for every subject
    for i in np.arange(numSubs):
        normConnectome[..., i] = (normConnectome[..., i] - mean) / sd

    return normConnectome


def runConnections(connectomeStack, regressorStack, runwhat):
    # First I flatten the connectomeStack
    stackShape = connectomeStack.shape
    # Get the number of connections to correlate
    numberOfConnections = stackShape[0] * stackShape[1]
    numberOfTimepoints = stackShape[2]
    # Reshape the stack to number of correlations by number of timepoints
    flatConnectomeStack = np.reshape(connectomeStack,
                                     (numberOfConnections,
                                      numberOfTimepoints))

    # Prepare containers for age
    regressorVector = np.array([])
    pValueVector = np.array([])
    flatConnections, flatSubjects = flatConnectomeStack.shape

    print('I got this many timepoints: ' + str(numberOfTimepoints)
          + ' and this many connections ' + str(numberOfConnections))

    tookTime = np.array([])
    for i, connectionIndex in enumerate(np.arange(numberOfConnections)):
        # Get the vector of connection values across subjects for current
        # connection
        start = time.time()
        percComplete = np.round(float(i + 1) / (flatConnections) * 100, 1)
        remaining = flatConnections - (i + 1)
        connectionVector = flatConnectomeStack[connectionIndex, :]
        # What to run here:
        if runwhat == 'corr':
            if len(regressorStack.shape) > 1:
                print('regressor stack bullshit for correlation '
                      + str(regressorStack.shape))

            corr, p = st.pearsonr(connectionVector, regressorStack)
            regressorVector = np.append(regressorVector, corr)
            pValueVector = np.append(pValueVector, p)

        elif runwhat == 'glm':
            ageTValue, pValues = runGLM(connectionVector, regressorStack)
            regressorVector = np.append(regressorVector, ageTValue)
            pValueVector = np.append(pValueVector, pValues)

        elif runwhat == 'ttest':
            # run the ttest model
            tValue, pValue = runTtest(connectionVector, regressorStack)
            regressorVector = np.append(regressorVector, tValue)
            pValueVector = np.append(pValueVector, pValue)


        else:
            print('don\'t know what to do with ' + runwhat)

        # My little time gadget...
        stop = time.time()
        took = stop - start
        tookTime = np.append(tookTime, took)
        avgTook = np.average(tookTime)
        remTime = np.round((avgTook * remaining), 2)
        sys.stdout.write('\r' + str(percComplete) + '% done. ' + str(remTime)
                         + ' more seconds to go...        ')
        sys.stdout.flush()



    # Done with looping
    # Check if there are any t-values that are positive
    posCount = np.sum(regressorVector > 0)
    negCount = np.sum(regressorVector < 0)
    print('Number of connections:\n'
          + '    pos: ' + str(posCount) + '\n'
          + '    neg: ' + str(negCount))
    # Reshape the results of the correlation back into the shape of the
    # original connectivity matrix
    regressorMatrix = np.reshape(regressorVector,
                                   (stackShape[0], stackShape[1]))
    pValueMatrix = np.reshape(pValueVector,
                              (stackShape[0], stackShape[1]))

    return regressorMatrix, pValueMatrix


def correlateConnectomeWithAge(connectomeStack, ageStack):
    # First I flatten the connectomeStack
    stackShape = connectomeStack.shape
    # Get the number of connections to correlate
    numberOfConnections = stackShape[0] * stackShape[1]
    numberOfTimepoints = stackShape[2]
    # Reshape the stack to number of correlations by number of timepoints
    flatConnectomeStack = np.reshape(connectomeStack,
                                     (numberOfConnections,
                                      numberOfTimepoints))

    # Prepare container variables for correlation and p values for each
    # connection
    correlationVector = np.array([])
    pValueVector = np.array([])

    # Iterate over the elements in the stack and correlate them to the age
    # stack one by one
    for connectionIndex in np.arange(numberOfConnections):
        # Get the vector of connection values across subjects for current
        # connection
        connectionVector = flatConnectomeStack[connectionIndex, :]
        # Correlate the vector to age
        corr, p = st.pearsonr(connectionVector, ageStack)
        # Append correlation and p values to their respective container
        # variables
        correlationVector = np.append(correlationVector, corr)
        pValueVector = np.append(pValueVector, p)

    # Reshape the results of the correlation back into the shape of the
    # original connectivity matrix
    correlationMatrix = np.reshape(correlationVector,
                                   (stackShape[0], stackShape[1]))
    pValueMatrix = np.reshape(pValueVector,
                              (stackShape[0], stackShape[1]))

    return correlationMatrix, pValueMatrix


def fitRobust(dataVec, predMat):
    '''
    Method to fit a vector of data by a vector of predictions
    '''
    robust = sm.RLM(dataVec, predMat, M=sm.robust.norms.HuberT())
    results = robust.fit()

    return results


def fitGLM(dataVector, predMat):
    # run a glm with only one factor
    model = sm.OLS(dataVector, predMat)
    results = model.fit()

    return results


def runGLM(dataVector, predMat):
    # run a glm with only one factor
    model = sm.OLS(dataVector, predMat)
    results = model.fit()
    # make sure that age is in fact the first predictor
    ageTValue = results.tvalues[0]
    ageSlope = results.params[0]

    # Sanity check:
    if (ageTValue < 0 and ageSlope > 0):
        print('T-value and slope don\'t match:\n'
              + '    tvalue: ' + str(ageTValue)
              + '    slope: ' + str(ageSlope))

    elif (ageTValue > 0 and ageSlope < 0):
        print('T-value and. slope don\'t match:\n'
              + '    tvalue: ' + str(ageTValue)
              + '    slope: ' + str(ageSlope))

    # get p-values on absolute t-values (use this)
    pValues = st.t.sf(np.abs(ageTValue), results.df_resid)

    # Divide p-values up into positive and negative (not now)
    posAgePValue = st.t.sf(ageTValue, results.df_resid)
    negAgePValue = st.t.sf(ageTValue * -1, results.df_resid)

    # return posAgePValue, negAgePValue
    return ageTValue, pValues


def runTtest(connVec, labelVec):
    '''
    Method to compare two groups of subjects with respect to their connectivity
    distributions
    '''
    # Check if only two labels
    if not len(np.unique(labelVec)) == 2:
        print('More than two labels here:\n'
              + str(np.unique(labelVec)))
    # Split up the groups
    labelOne = np.unique(labelVec)[0]
    labelTwo = np.unique(labelVec)[1]
    indexOne = labelVec == labelOne
    indexTwo = labelVec == labelTwo

    # Get the values corresponding to the indices
    valuesOne = connVec[indexOne]
    valuesTwo = connVec[indexTwo]

    # Run the t-test
    t, p = st.ttest_ind(valuesOne, valuesTwo)

    return t, p


def prepareFDR(pValueMatrix):
    # This returns a vector of p-values
    aidMask = np.ones_like(pValueMatrix)
    lowerTriangle = np.tril(aidMask, -1)
    independentPValues = pValueMatrix[lowerTriangle == 1]

    return independentPValues


def computeFDR(pValueVector, alpha):
    # This returns a new thresholded p value
    # Sort the p values by size, beginning with the smallest
    sortedP = np.sort(pValueVector)
    # Reverse sort the p-values, so the first one is the biggest
    reverseP = sortedP[::-1]
    # Get the number of p-values
    numP = float(len(reverseP))
    # Create a vector designating the position of the reverse sorted p-values
    # in the sorted p vector (e.g. the first reverse sorted p-value will have
    # the index numP because it would be the last entry in the sorted vector)
    indexP = np.arange(numP, 0, -1)
    # Create test vector of (index of p value / number of p values) * alpha
    test = indexP / numP * alpha
    # Check where p-value <= test
    testIndex = np.where(reverseP <= test)
    if testIndex[0].size == 0:
        print('None of you p values pass FDR correction')
        pFDR = 0
    else:
        # Get the first p value that passes the criterion
        pFDR = reverseP[np.min(testIndex)]
        print('FDR corrected p value for alpha of ' + str(alpha) + ' is '
              + str(pFDR)
              + '\n' + str(testIndex[0].size) + ' out of '
              + str(int(numP)) + ' p-values pass this threshold')

    return pFDR


def thresholdRegressorMatrix(regressorMatrix, pValueMatrix, pThresh):
    threshRegressorMatrix = np.zeros_like(regressorMatrix)
    threshRegressorMatrix[pValueMatrix <= pThresh] = regressorMatrix[pValueMatrix <= pThresh]

    return threshRegressorMatrix


def saveOutput(outputFilePath, outputMatrix):
    np.savetxt(outputFilePath, outputMatrix, fmt='%.12f')
    status = ('Saving to ' + outputFilePath)

    return status


def Main():
    # Define the inputs
    pathToConnectomeDir = '/home2/surchs/secondLine/connectomes/wave/dos160'
    pathToPhenotypicFile = '/home2/surchs/secondLine/configs/wave/wave_pheno81_uniform.csv'
    pathToSubjectList = '/home2/surchs/secondLine/configs/wave/wave_subjectList.csv'

    connectomeSuffix = '_connectome_glob_corr.txt'

    runwhat = 'glm'
    doPlot = False

    # Define parameters
    alpha = 0.05
    childmax = 12.0
    adolescentmax = 18.0
    doClasses = False
    doFDR = True
    doNorm = False
    which = 'wave'

    stratStr = (str(runwhat) + '_' + str(doFDR) + '_' + str(alpha)
                + '_' + str(doClasses))

    # Define the outputs
    pathToDumpDir = '/home2/surchs/secondLine/GLM/wave/dos160/corrected'
    outPath = os.path.join(pathToDumpDir, stratStr)
    # Check if it is there
    if not os.path.isdir(outPath):
        print('Making ' + outPath + ' now')
        os.makedirs(outPath)

    suffix = '_matrix_glob_ai_corr.txt'
    correlationFileName = (runwhat + suffix)
    pValueFileName = (runwhat + '_pvalue' + suffix)
    thresholdFileName = (runwhat + '_thresholded' + suffix)

    pathToCorrelationMatrix = os.path.join(outPath, correlationFileName)
    pathToPValueMatrix = os.path.join(outPath, pValueFileName)
    pathToThresholdedMatrix = os.path.join(outPath, thresholdFileName)
    # Check the fucking paths
    if  (not which in pathToConnectomeDir or
         not which in  pathToPhenotypicFile or
         not which in pathToSubjectList or
         not which in outPath):
        message = 'Your paths are bad!'
        raise Exception(message)
    else:
        print('Your paths are ok.')

    # Read subject list
    subjectListFile = open(pathToSubjectList, 'rb')
    subjectList = subjectListFile.readlines()

    # Read the phenotypic file
    pheno = loadPhenotypicFile(pathToPhenotypicFile)
    phenoSubjects = pheno['subject'].tolist()
    phenoAges = pheno['age'].tolist()

    # Prepare container variables
    connectomeStack = np.array([])
    ageStack = np.array([])
    meanConnStack = np.array([])
    labelStack = np.array([], dtype=int)

    # Loop through the subjects
    for i, subject in enumerate(subjectList):
        subject = subject.strip()
        phenoSubject = phenoSubjects[i]
        # Workaround for dumb ass pandas
        # phenoSubject = ('00' + str(phenoSubject))

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
        # Normalize the connectome
        # normalizedConnectome = fisherZ(connectome)
        normalizedConnectome = connectome
        if np.isnan(normalizedConnectome).any():
            message = (subject + ' has nan in the connectome post norm!')
            raise Exception(message)

        if doClasses:
            # Get the class assignment for the subject
            if phenoAge <= childmax:
                print(subject + ' --> child (' + str(phenoAge) + ')')
                label = 0

            elif phenoAge > childmax and phenoAge <= adolescentmax:
                print(subject + ' --> adolescent (' + str(phenoAge) + ')')
                label = 1
                # Don't use adolescents
                continue

            else:
                print(subject + ' --> adult (' + str(phenoAge) + ')')
                label = 2

            labelStack = np.append(labelStack, label)

        # Get the mean connectivity
        uniqueConnections = getUniqueMatrixElements(normalizedConnectome)
        meanConn = np.mean(uniqueConnections)
        meanConnStack = np.append(meanConnStack, meanConn)

        # Stack the connectome
        connectomeStack = stackConnectome(connectomeStack, normalizedConnectome)
        print('connectomeStack: ' + str(connectomeStack.shape))
        # Stack ages
        ageStack = stackAges(ageStack, phenoAge)

    if doClasses:
        # Force balance the classes, first get the number of classes
        uniqueLabels = np.unique(labelStack)
        labelString = ''
        numLabels = len(labelStack)
        numContainer = np.array([])
        for name in uniqueLabels:
            numCases = np.sum(labelStack == name)
            ratioCases = float(numCases) / numLabels
            labelString = (labelString + str(name) + ': ' + str(numCases)
                           + ' (' + str(np.round(ratioCases, 3)) + ')\n')
            numContainer = np.append(numContainer, numCases)

        if len(np.unique(numContainer)) == 1:
            # All good, just one number of classes
            message = '\nAll classes have the same number of cases:\n'
            print(message + labelString)
        else:
            # Something is off
            message = '\nThe classes are unbalanced, forcing balance!\n'
            print(message + labelString)
            # Now live up to the promise
            minCases = numContainer.min()
            deleteDex = np.array([])
            newString = '\nThe new distribution, balanced sample is:\n'

            for name in uniqueLabels:
                tempNdex = np.where(labelStack == name)[0]
                # get the last number of subjects
                popDex = tempNdex[minCases - 1:]
                # add them to the deleteDex
                deleteDex = np.append(deleteDex, popDex)

            # Now remove the stuff from the stacks:
            newLabelStack = np.delete(labelStack, deleteDex)
            newAgeStack = np.delete(ageStack, deleteDex)
            newConnectomeStack = np.delete(connectomeStack, deleteDex, axis=2)

            # Check the ratio again
            numLabels = len(newLabelStack)
            for name in uniqueLabels:
                numCases = np.sum(newLabelStack == name)
                ratioCases = float(numCases) / numLabels
                newString = (newString + str(name) + ': ' + str(numCases)
                             + ' (' + str(np.round(ratioCases, 3)) + ')\n')

            # Print the new distribution:
            print(newString)
            labelStack = newLabelStack
            ageStack = newAgeStack
            connectomeStack = newConnectomeStack

            print('label: ' + str(labelStack.shape) + '\n'
                  + 'age: ' + str(ageStack.shape) + '\n'
                  + 'connectome: ' + str(connectomeStack.shape))

    # Now we have the connectome stack
    # Normalize is (z-transform)
    if doNorm:
        connectomeStack = normalize(connectomeStack)
    # Prepare vector of ones for intercept
    ones = np.ones_like(ageStack)

    # Make the regressor matrix
    if runwhat == 'glm':
        regressorStack = np.concatenate((ageStack[..., None],
                                         ones[..., None]),
                                        axis=1)
        # regressorStack = ageStack

    elif runwhat == 'corr':
        regressorStack = ageStack

    elif runwhat == 'ttest':
        # run the ttest
        regressorStack = labelStack

    else:
        print('You don\'t know what you are doing...')

    # Check the shapes of age and connectivity stacks
    print('ageStack.shape: ' + str(ageStack.shape) + ' connStack.shape: '
          + str(connectomeStack.shape))

    if doPlot:
        # Plot mean Connectivity with age
        # Make fit of meanConnStack and Age
        fitStack = np.concatenate((ageStack[..., None],
                                   np.ones_like(ageStack)[..., None]),
                                  axis=1)
        results = fitRobust(meanConnStack, fitStack)
        slope = results.params[0]
        intercept = results.params[1]

        xnew = np.arange(ageStack.min() - 1, ageStack.max() + 1, 0.1)
        meanFit = slope * xnew + intercept
        showSlope = np.round(slope, 2)

        plt.plot(ageStack, meanConnStack, '.k', label='mean connectivity')
        plt.plot(xnew, meanFit, 'r', label='robust fit ' + str(showSlope))
        plt.xlabel('age')
        plt.ylabel('mean connectivity')
        plt.title('mean connectivity across age')
        plt.legend()
        plt.show()
        raw_input('Enter to continue...')
        plt.close()

    if doPlot:
        # Plot an exemplary connection across age
        connVec = connectomeStack[10, 14, :]

        # Fit it
        fitStack = np.concatenate((ageStack[..., None],
                                   np.ones_like(ageStack)[..., None]),
                                  axis=1)

        results = fitRobust(connVec, fitStack)
        slope = results.params[0]
        intercept = results.params[1]

        glmFit = fitGLM(connVec, ageStack)
        glmSlope = np.round(glmFit.params[0], 2)
        glmT = glmFit.tvalues[0]

        xnew = np.arange(ageStack.min() - 1, ageStack.max() + 1, 0.1)
        meanFit = slope * xnew + intercept
        showSlope = np.round(slope, 2)

        glmShow = glmSlope * xnew
        plt.plot(ageStack, connVec, '.k')
        plt.plot(xnew, meanFit, 'r', label='robust fit ' + str(showSlope))
        plt.plot(xnew, glmShow, 'g', label=('glm fit ' + str(glmT) + ' '
                                            + str(glmSlope)))
        plt.xlabel('age')
        plt.ylabel('connectivity')
        plt.title('connectivity across age')
        plt.legend()
        plt.show()
        raw_input('Enter to continue...')
        plt.close()

    if doPlot:
        # Plot the distribution of SD for all the timeseries
        mask = np.ones_like(connectomeStack[..., 0])
        mask = np.tril(mask, -1)
        flatConnectome = connectomeStack[mask == 1]
        deviation = np.std(flatConnectome, axis=1)
        plt.hist(deviation)
        plt.xlabel('standard deviation of connectivity across subjects')
        plt.title('SD of connectivity across subjects')
        plt.show()
        raw_input('Enter to continue...')
        plt.close()

    # Now run through the model - only here btw
    (regressorMatrix, pValueMatrix) = runConnections(connectomeStack,
                                                     regressorStack,
                                                     runwhat)

    # Compute the correlations with age
    # correlationMatrix, pValueMatrix = correlateConnectomeWithAge(connectomeStack,
    #                                                              ageStack)

    # Prepare FDR by pulling out the independent p values from the matrix
    independentPValues = prepareFDR(pValueMatrix)
    print('Minimal pvalue: ' + str(independentPValues.min()))

    if doFDR:
        # Compute threshold p value with FDR
        pThresh = computeFDR(independentPValues, alpha)
    else:
        # Don't do FDR
        pThresh = alpha

    # Threshold the pValueMatrix with FDR
    thresholdedRegressorMatrix = thresholdRegressorMatrix(regressorMatrix,
                                                          pValueMatrix,
                                                          pThresh)

    survivorCount = np.sum(getUniqueMatrixElements(thresholdedRegressorMatrix != 0))
    numFeatures = len(getUniqueMatrixElements(thresholdedRegressorMatrix != 0))
    # Show how many survived
    print('survivors: ' + str(survivorCount) + ' / ' + str(numFeatures))

    status = saveOutput(pathToCorrelationMatrix, regressorMatrix)
    print('correlation matrix ' + status)
    status = saveOutput(pathToPValueMatrix, pValueMatrix)
    print('p value matrix ' + status)
    status = saveOutput(pathToThresholdedMatrix, thresholdedRegressorMatrix)
    print('thresholded matrix ' + status)


if __name__ == '__main__':
    Main()
