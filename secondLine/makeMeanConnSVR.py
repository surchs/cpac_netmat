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
from sklearn import svm
import statsmodels.api as sm
from scipy import stats as st
import sklearn.grid_search as gs
from matplotlib import pyplot as plt
import sklearn.cross_validation as cv

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


def makeFolds(feature, age, crossVal):
    '''
    generate crossvalidations based on the validation object
    '''
    crossValDict = {}

    run = 1
    for cvInstance in crossVal:
        trainIndex = cvInstance[0]
        testIndex = cvInstance[1]

        trainFeat = feature[trainIndex, :]
        testFeat = feature[testIndex, :]

        trainAge = age[trainIndex]
        testAge = age[testIndex]

        # short sanity check:
        if (not trainFeat.shape[0] == trainAge.shape[0]
            or not testFeat.shape[0] == testAge.shape[0]):
            print('The features and ages in run ' + str(run)
                  + 'don\'t match up. Please check!\n'
                  + '    trainFeat: ' + str(trainFeat.shape) + '\n'
                  + '    trainAge: ' + str(trainAge.shape) + '\n'
                  + '    testFeat: ' + str(testFeat.shape) + '\n'
                  + '    testAge: ' + str(testAge.shape) + '\n')

        trainTuple = (trainFeat, trainAge)
        testTuple = (testFeat, testAge)

        crossValDict[str(run)] = (trainTuple, testTuple)
        run += 1

    return crossValDict


def findParameters(trainFeature, trainAge, kernel, nCors):
    '''
    method to find optimal parameters for the chosen SVR model
    '''
    # Make parameters for the first pass
    cExpOne = np.arange(-4, 4, 1)
    cBaseOne = np.ones_like(cExpOne, dtype='float32') * 10
    cParamOne = np.power(cBaseOne, cExpOne).tolist()
    eExpOne = np.arange(-8, 0, 1)
    eBaseOne = np.ones_like(eExpOne, dtype='float32') * 10
    eParamOne = np.power(eBaseOne, eExpOne).tolist()

    firstParameters = {'C': cParamOne, 'epsilon': eParamOne}
    gridModel = svm.SVR(kernel=kernel, degree=2)

    # Train first pass model
    firstTrainModel = gs.GridSearchCV(gridModel,
                                      firstParameters,
                                      cv=2,
                                      n_jobs=nCors,
                                      verbose=0)
    firstTrainModel.fit(trainFeature, trainAge)

    # First pass best parameters
    firstPassC = firstTrainModel.best_estimator_.C
    firstPassE = firstTrainModel.best_estimator_.epsilon

    # Make the parameters for the second run
    firstExpC = np.log10(firstPassC)
    cExpTwo = np.arange(firstExpC - 1, firstExpC + 1.1, 0.1)
    cBaseTwo = np.ones_like(cExpTwo, dtype='float32') * 10
    cParamTwo = np.power(cBaseTwo, cExpTwo).tolist()

    firstExpE = np.log10(firstPassE)
    eExpTwo = np.arange(firstExpE - 1, firstExpE + 1.1, 0.1)
    eBaseTwo = np.ones_like(eExpTwo, dtype='float32') * 10
    eParamTwo = np.power(eBaseTwo, eExpTwo).tolist()

    secondParameters = {'C': cParamTwo, 'epsilon': eParamTwo}

    secondTrainModel = gs.GridSearchCV(gridModel,
                                       secondParameters,
                                       cv=2,
                                       n_jobs=nCors,
                                       verbose=0)
    secondTrainModel.fit(trainFeature, trainAge)

    # Final best parameters
    bestC = secondTrainModel.best_estimator_.C
    bestE = secondTrainModel.best_estimator_.epsilon


    return bestC, bestE


def trainModel(trainFeature, trainAge, kernel, C, E):
    '''
    module to train the model on the data
    '''
    trainModel = svm.SVR(kernel=kernel, C=C, epsilon=E)

    trainModel.fit(trainFeature, trainAge)

    return trainModel


def testModel(model, testFeature):
    '''
    Method to test the model that was trained beforehand
    '''
    predictedAge = model.predict(testFeature)

    return predictedAge



def mainSVR(feature, age, crossVal, kernel, nCors):
    '''
    short method to handle all the steps in the SVR
    '''
    crossValDict = makeFolds(feature, age, crossVal)
    # outputDict = {}
    testAgeVec = np.array([])
    predAgeVec = np.array([])
    runParamEst = True

    for i, run in enumerate(crossValDict.keys()):
        # Alert on running
        print('Running fold ' + str(i))
        # Get the training and test tuples
        trainTuple, testTuple = crossValDict[run]
        # unpack tuples
        trainFeature, trainAge = trainTuple
        testFeature, testAge = testTuple

        # Get the best parameters for this training set
        if runParamEst:
            bestC, bestE = findParameters(trainFeature, trainAge, kernel, nCors)
        else:
            bestC = 1.0
            bestE = 0.1

        # Train model on train data
        model = trainModel(trainFeature, trainAge, kernel, bestC, bestE)
        # Test model on test data
        predictedAge = testModel(model, testFeature)

        # Store predicted and true age in the output directory
        testAgeVec = np.append(testAgeVec, testAge)
        predAgeVec = np.append(predAgeVec, predictedAge)

        # outTuple = (testAge, predictedAge)
        # outputDict[run] = outTuple
        if not testAgeVec.shape == predAgeVec.shape:
            print('true and predicted age don\'t match in run ' + str(i) + ':\n'
                  + '    true: ' + str(testAgeVec.shape) + '\n'
                  + '    pred: ' + str(predAgeVec.shape))
        else:
            print('Run ' + str(i) + ':\n'
                  + '    true: ' + str(testAgeVec.shape) + '\n'
                  + '    pred: ' + str(predAgeVec.shape))
        print('    bestC: ' + str(bestC) + '\n'
              + '    bestE: ' + str(bestE))

    # Done, stack the output together (true age first, then predicted)
    outputMatrix = np.concatenate((testAgeVec[..., None],
                                   predAgeVec[..., None]),
                                  axis=1)

    return outputMatrix
    # Done, return the output dictionary
    # return outputDict


def dualPlot(resultWithin, resultBetween, title):
    '''
    method to plot network results side by side
    '''
    # Unpack the results first
    wTrue = resultWithin[:, 0]
    wPred = resultWithin[:, 1]
    bTrue = resultBetween[:, 0]
    bPred = resultBetween[:, 1]

    # Sanity check
    wSorted = np.sort(wTrue)
    bSorted = np.sort(bTrue)
    # simplest level
    if not np.array_equal(wSorted, bSorted):
        print('Something is very wrong with your results. The true ages don\'t'
              + ' match')

    # more specific
    elif not np.array_equal(wTrue, bTrue):
        print('Your ages for within and between are not in the same order.'
              + ' You think you are using the same crossval, but you are not.')

    # If they pass, take one as the reference age
    refAge = wTrue

    fig, (within, between) = plt.subplots(1, 2, sharex=True, sharey=False)

    # fit ages
    wP = np.polyfit(wTrue, wPred, 1)
    bP = np.polyfit(bTrue, bPred, 1)
    xnew = np.arange(refAge.min() - 1, refAge.max() + 1, 0.1)
    wFit = np.polyval(wP, xnew)
    bFit = np.polyval(bP, xnew)


    within.set_title('within network')
    between.set_title('between network')

    withinCorr, withinP = st.pearsonr(wTrue, wPred)
    within.plot(wTrue, wPred, 'k.')
    within.plot(xnew, wFit, 'r', label=(str(np.round(withinCorr, 2))
                                        + ' '
                                        + str(np.round(withinP, 4))))
    within.set_xlabel('true age')
    within.set_ylabel('predicted age')
    within.legend()


    betweenCorr, betweenP = st.pearsonr(bTrue, bPred)
    between.plot(bTrue, bPred, 'k.')
    between.plot(xnew, bFit, 'b', label=(str(np.round(betweenCorr, 2))
                                         + ' '
                                         + str(np.round(betweenP, 4))))
    between.set_xlabel('true age')
    between.set_ylabel('predicted age')
    between.legend()

    fig.suptitle(title)
    plt.show()
    raw_input("Press Enter to continue...")
    plt.close()


def networkPlot(networkResults):
    '''
    Method to visualize the network level results
    '''
    for network in networkResults.keys():
        print('Plotting network ' + network + ' now.')
        (withinResult, betweenResult) = networkResults[network]
        dualPlot(withinResult, betweenResult, network)


def saveOutput(outputFilePath, output):
    f = gzip.open(outputFilePath, 'wb')
    cPickle.dump(output, f)
    f.close()
    status = ('Saved to ' + outputFilePath)

    return status


def Main():
    # Define the inputs
    pathToConnectomeDir = '/home2/surchs/secondLine/connectomes/abide/dos160'
    pathToPhenotypicFile = '/home2/surchs/secondLine/configs/abide/abide_across_236_pheno.csv'
    pathToSubjectList = '/home2/surchs/secondLine/configs/abide/abide_across_236_subjects.csv'

    pathToNetworkNodes = '/home2/surchs/secondLine/configs/networkNodes_dosenbach.dict'
    pathToRoiMask = '/home2/surchs/secondLine/masks/dos160_abide_246_3mm.nii.gz'

    connectomeSuffix = '_connectome_glob.txt'

    pathToOutputFile = '/home2/surchs/secondLine/SVR/abide/dos/meanConn.dict'

    runwhat = 'glm'
    doPlot = False

    # Define parameters
    alpha = 0.1
    kfold = 10
    nCors = 20
    kernel = 'linear'
    takeFeat = 'conn'

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
    networkResults = {}
    withinFeature = np.array([])
    betweenFeature = np.array([])

    # Prepare the crossvalidation object
    crossVal = cv.KFold(len(phenoSubjects),
                        kfold,
                        shuffle=True)
    nFolds = crossVal.k

    # quick sanity check
    if not nFolds == kfold:
        print('\nThe intended and actual crossvalidation is different:\n'
              + '    kfold: ' + str(kfold) + '\n'
              + '    nFold: ' + str(nFolds) + '\n'
              + 'with ' + str(len(phenoSubjects)) + ' subjects\n')

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
    for i, network in enumerate(networkNodes.keys()):
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

        # Make feature for SVR
        if takeFeat == 'conn':
            withinFeature = withinMatrix.T
            betweenFeature = betweenMatrix.T
        else:
            if withinFeature.size == 0:
                withinFeature = meanWithin[..., None]
            else:
                withinFeature = np.concatenate((withinFeature,
                                                meanWithin[..., None]),
                                               axis=1)

            if betweenFeature.size == 0:
                betweenFeature = meanBetween[..., None]
            else:
                betweenFeature = np.concatenate((betweenFeature,
                                                 meanBetween[..., None]),
                                                axis=1)

        if takeFeat == 'conn':
            # Run SVR
            print('\nRunning within ' + network + ' mean connectivity SVR ('
                  + str(i) + '/' + str(len(networkNodes.keys())) + ')')
            withinResult = mainSVR(withinFeature, ageStack, crossVal, kernel,
                                   nCors)
            print('\nRunning between ' + network + ' mean connectivity SVR ('
                  + str(i) + '/' + str(len(networkNodes.keys())) + ')')
            betweenResult = mainSVR(betweenFeature, ageStack, crossVal, kernel,
                                    nCors)

            # Store the output in the output Dictionary for networks
            result = (withinResult, betweenResult)
            networkResults[network] = result

    if not takeFeat == 'conn':
        # do the real thing
        withinResult = mainSVR(withinFeature, ageStack, crossVal, kernel)
        betweenResult = mainSVR(betweenFeature, ageStack, crossVal, kernel)
        print('Plotting what we actually wanted...')
        dualPlot(withinResult, betweenResult, 'within and between connectivit'
                 + ' predicting age')

    # Save the networkResults
    status = saveOutput(pathToOutputFile, networkResults)
    print(status)

    # Done with running analysis. Plotting by network now
    networkPlot(networkResults)


if __name__ == '__main__':
    Main()
