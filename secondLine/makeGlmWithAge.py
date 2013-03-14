'''
Created on Feb 22, 2013

@author: surchs
'''
import os
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

    print('I got this many timepoints: ' + str(numberOfTimepoints)
          + ' and this many connections ' + str(numberOfConnections))

    for connectionIndex in np.arange(numberOfConnections):
        # Get the vector of connection values across subjects for current
        # connection
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

        else:
            print('don\'t know what to do with ' + runwhat)



    # Done with looping
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

    # get p-values on absolute t-values (use this)
    pValues = st.t.sf(np.abs(ageTValue), results.df_resid)

    # Divide p-values up into positive and negative (not now)
    posAgePValue = st.t.sf(ageTValue, results.df_resid)
    negAgePValue = st.t.sf(ageTValue * -1, results.df_resid)

    # return posAgePValue, negAgePValue
    return ageTValue, pValues


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
    status = 'cool'

    return status


def Main():
    # Define the inputs
    pathToConnectomeDir = '/home2/surchs/secondLine/connectomes/abide/dos160'
    pathToPhenotypicFile = '/home2/surchs/secondLine/configs/abide/abide_across_236_pheno.csv'
    pathToSubjectList = '/home2/surchs/secondLine/configs/abide/abide_across_236_subjects.csv'

    connectomeSuffix = '_connectome_glob.txt'

    debugConnStack = '/home2/surchs/secondLine/debug/ConnStack.npy'
    debugCorrStack = '/home2/surchs/secondLine/debug/CorrStack.npy'
    debugAgeStack = '/home2/surchs/secondLine/debug/AgeStack.npy'

    runwhat = 'glm'
    doPlot = False

    # Define parameters
    alpha = 0.05

    # Define the outputs
    outPath = '/home2/surchs/secondLine/correlation/abide/dos160/'
    correlationFileName = (runwhat + '_matrix_glob_c.txt')
    pValueFileName = (runwhat + '_pvalue_matrix_glob_c.txt')
    thresholdFileName = (runwhat + '_thresholded_matrix_glob_c.txt')

    pathToCorrelationMatrix = os.path.join(outPath, correlationFileName)
    pathToPValueMatrix = os.path.join(outPath, pValueFileName)
    pathToThresholdedMatrix = os.path.join(outPath, thresholdFileName)
    # This path gets appended by the name of the age group (child, adolescent,
    # adult)
    # pathToCorrelationMatrixAges = '/home2/surchs/secondLine/correlation/correlation_matrix_'


    # Read subject list
    subjectListFile = open(pathToSubjectList, 'rb')
    subjectList = subjectListFile.readlines()

    # Read the phenotypic file
    pheno = loadPhenotypicFile(pathToPhenotypicFile)
    phenoSubjects = pheno['SubID'].tolist()
    phenoAges = pheno['age'].tolist()

    # Prepare container variables
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
        print('connectome: ' + str(connectome.shape))
        # Normalize the connectome
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


    # Make the regressor matrix
    if runwhat == 'glm':
        # regressorStack = np.concatenate((ageStack[..., None],
        #                                  meanConnStack[..., None]),
        #                                 axis=1)
        regressorStack = ageStack
    elif runwhat == 'corr':
        regressorStack = ageStack

    else:
        print('You don\'t know what you are doing...')

    # Check the shapes of age and connectivity stacks
    print('ageStack.shape: ' + str(ageStack.shape) + ' connStack.shape: '
          + str(connectomeStack.shape))

    if doPlot:
        # Plot mean Connectivity with age
        plt.plot(ageStack, meanConnStack, '.k', label='mean connectivity')
        plt.xlabel('age')
        plt.ylabel('mean connectivity')
        plt.title('mean connectivity across age')
        plt.legend()
        plt.show()
        raw_input('Enter to continue...')
        plt.close()

    # Now run through the model
    (regressorMatrix,
     pValueMatrix) = runConnections(connectomeStack,
                                    regressorStack,
                                    runwhat)

    # Compute the correlations with age
    # correlationMatrix, pValueMatrix = correlateConnectomeWithAge(connectomeStack,
    #                                                              ageStack)

    # DEBUG CALL
    '''
    np.save(debugConnStack, connectomeStack)
    np.save(debugCorrStack, regressorMatrix)
    np.save(debugAgeStack, ageStack)
    '''

    # Prepare FDR by pulling out the independent p values from the matrix
    independentPValues = prepareFDR(pValueMatrix)

    print('Minimal pvalue: ' + str(independentPValues.min()))

    # Compute threshold p value with FDR
    pThresh = computeFDR(independentPValues, alpha)

    # Threshold the pValueMatrix with FDR
    thresholdedRegressorMatrix = thresholdRegressorMatrix(regressorMatrix,
                                                            pValueMatrix,
                                                            pThresh)

    # Here, we could re-run the analysis for the different age groups but I
    # am currently not doing this because Damien Fair just did it for the whole
    # group at once


    # save the outputs
    status = 'debug'
    # status = saveOutput(pathToCorrelationMatrix, regressorMatrix)
    print('correlation matrix says ' + status)
    # status = saveOutput(pathToPValueMatrix, pValueMatrix)
    print('p value matrix says ' + status)
    # status = saveOutput(pathToThresholdedMatrix, thresholdedRegressorMatrix)
    print('thresholded matrix says ' + status)



if __name__ == '__main__':
    Main()
