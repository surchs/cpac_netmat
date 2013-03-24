'''
Created on Feb 22, 2013

@author: surchs
'''
import os
import sys
import gzip
import time
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
import sklearn.feature_selection as fs
from sklearn.metrics import mean_squared_error


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


def corrFeature(trainFeature, trainAge):
    '''
    Method to select features that are significantly correlated with age
    '''
    # Get the number of features
    numberOfFeatures = trainFeature.shape[1]

    rVector = np.array([])
    pVector = np.array([])

    # Iterate over the elements in the stack and correlate them to the age
    # stack one by one
    for connectionIndex in np.arange(numberOfFeatures):
        # Get the vector of connection values across subjects for current
        # connection
        connectionVector = trainFeature[:, connectionIndex]
        # Correlate the vector to age
        r, p = st.pearsonr(connectionVector, trainAge)
        # Append correlation and p values to their respective container
        # variables
        rVector = np.append(rVector, r)
        pVector = np.append(pVector, p)

    # Return the pearson's r and the corresponding p value as a vector
    return rVector, pVector


def glmFeature(trainFeature, trainAge):
    '''
    Method to select the best feature from a GLM
    '''
    # Get the number of features
    numberOfFeatures = trainFeature.shape[1]
    # Generate the regressor matrix
    regressor = np.concatenate((trainAge[..., None],
                                np.ones_like(trainAge)[..., None]),
                               axis=1)

    tVector = np.array([])
    pVector = np.array([])

    # Iterate over the elements in the stack and correlate them to the age
    # stack one by one
    for connectionIndex in np.arange(numberOfFeatures):
        # Get the vector of connection values across subjects for current
        # connection
        connectionVector = trainFeature[:, connectionIndex]
        # Correlate the vector to age
        t, p = runGLM(connectionVector, trainAge)
        # Append correlation and p values to their respective container
        # variables
        tVector = np.append(tVector, t)
        pVector = np.append(pVector, p)

    # Return the pearson's r and the corresponding p value as a vector
    return tVector, pVector


def ttestFeature(trainFeature, trainLabel):
    '''
    Method to find features that significantly differentiate between two groups
    '''
    # Get the number of features
    numberOfFeatures = trainFeature.shape[1]

    tVector = np.array([])
    pVector = np.array([])

    # Iterate over the elements in the stack and correlate them to the age
    # stack one by one
    for connectionIndex in np.arange(numberOfFeatures):
        # Get the vector of connection values across subjects for current
        # connection
        connectionVector = trainFeature[:, connectionIndex]
        # Correlate the vector to age
        t, p = runTtest(connectionVector, trainLabel)
        # Append correlation and p values to their respective container
        # variables
        tVector = np.append(tVector, t)
        pVector = np.append(pVector, p)

    # Return the pearson's r and the corresponding p value as a vector
    return tVector, pVector


def rfeFeature(trainFeature, trainAge, maxFeat, kernel='linear'):
    '''
    Method to get features from recursive feature eleminiation
    '''
    # Get the number of features
    numberOfFeatures = trainFeature.shape[1]
    # get the estimator
    svrEstimator = svm.SVR(kernel=kernel)

    # just get the number of features and go home
    rfeObject = fs.RFE(estimator=svrEstimator,
                       n_features_to_select=maxFeat,
                       step=0.1)
    rfeObject.fit(trainFeature, trainAge)
    # temporary index of selected features
    tempRfeIndex = rfeObject.support_
    # rfe index
    rfeIndex = np.where(tempRfeIndex)[0]

    # prepare a binary and a boolean index
    featIndex = np.zeros(numberOfFeatures, dtype=int)
    featIndex[rfeIndex] = 1
    boolIndex = featIndex == 1

    return boolIndex


def rfecvFeature(trainFeature, trainAge, kernel='linear'):
    '''
    Method to get features using rfe with crossvalidation
    '''
    # Get the number of features
    numberOfFeatures = trainFeature.shape[1]
    # get the estimator
    svrEstimator = svm.SVR(kernel=kernel)
    # Get the model
    rfecvObject = fs.RFECV(estimator=svrEstimator,
                           step=0.01,
                           cv=2,
                           loss_func=mean_squared_error)
    rfecvObject.fit(trainFeature, trainAge)
    tempRfeCvIndex = rfecvObject.support_
    rfeCvIndex = np.where(tempRfeCvIndex)[0]
    # prepare a binary and a boolean index
    featIndex = np.zeros(numberOfFeatures, dtype=int)
    featIndex[rfeCvIndex] = 1
    boolIndex = featIndex == 1

    return boolIndex


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
    print('fdron')
    test = indexP / numP * alpha
    print('fdroff')
    # Check where p-value <= test
    testIndex = np.where(reverseP <= test)
    if testIndex[0].size == 0:
        # print('None of you p values pass FDR correction')
        pFDR = 0
    else:
        # Get the first p value that passes the criterion
        pFDR = reverseP[np.min(testIndex)]
        '''
        print('FDR corrected p value for alpha of ' + str(alpha) + ' is '
              + str(pFDR)
              + '\n' + str(testIndex[0].size) + ' out of '
              + str(int(numP)) + ' p-values pass this threshold')
        '''

    return pFDR


def threshold(dataVector, thresh):
    '''
    Module to get a boolean vector of values in the data-vetor that are less
    than the threshold
    '''
    boolDex = dataVector < thresh

    return boolDex


def findFeatures(trainFeature, trainAge, strat, kernel='linear', numFeat=200,
                 alpha=0.05):
    '''
    Method to do feature selection
    strategies are:
        'corr'    - correlation, passing FDR
        'glm'     - glm with age (and intercept), passing FDR
        'ttest'   - t-test between two groups (trainAge will be label then)
        'rfe'     - recursive feature eleminiation on the kernel model
        'rfecv'   - rfe with cross validation, have to define a max feature
    '''
    numberOfFeatures = trainFeature.shape[1]
    maxFeatRFE = 2000
    pThresh = alpha

    if str(strat) == 'corr':
        rVector, pVector = corrFeature(trainFeature, trainAge)
        # Get the FDR p-cutoff
        # pThresh = computeFDR(pVector, alpha)
        pIndex = threshold(pVector, pThresh)
        # turn the index into a feature index
        featIndex = pIndex

    elif str(strat) == 'glm':
        print('start glm')
        tVector, pVector = glmFeature(trainFeature, trainAge)
        print('done glm')
        # Get the FDR p-cutoff
        # pThresh = computeFDR(pVector, alpha)
        pIndex = threshold(pVector, pThresh)
        # turn the index into a feature index
        featIndex = pIndex

    elif str(strat) == 'ttest':
        tVector, pVector = ttestFeature(trainFeature, trainAge)
        # Get the FDR p-cutoff
        # pThresh = computeFDR(pVector, alpha)
        pIndex = threshold(pVector, pThresh)
        # turn the index into a feature index
        featIndex = pIndex

    elif str(strat) == 'rfe':
        featIndex = rfeFeature(trainFeature, trainAge, numFeat, kernel=kernel)

    elif str(strat) == 'rfecv':
        if numberOfFeatures > maxFeatRFE:
            # First bring the number of features down
            firstIndex = rfeFeature(trainFeature, trainAge,
                                    maxFeatRFE, kernel=kernel)
            # Cut the features down
            firstFeature = trainFeature[:, firstIndex]
            secondIndex = rfecvFeature(firstFeature, trainAge, kernel=kernel)
            # Now we have to get the second index to the length of the first
            tempIndex = np.zeros_like(firstIndex, dtype=int)
            tempIndex[firstIndex[secondIndex]] = 1
            # And now turn it into a boolean index
            featIndex = tempIndex == 1
        else:
            featIndex = rfecvFeature(trainFeature, trainAge, kernel='linear')

    elif str(strat) == 'None':
        print('Not doing any feature selection')
        firstIndex = np.ones(numberOfFeatures)
        featIndex = firstIndex == 1

    else:
        message = ('Your strategy (' + str(strat)
                + ') is not implemented.')
        raise Exception(message)

    # Check if anything comes through at all
    remainingFeatures = np.sum(featIndex)
    if remainingFeatures == 0:
        message('All features have been removed! This is uncool.')
        raise Exception(message)

    return featIndex


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


def mainSVR(feature, age, crossVal, kernel, nCors, runParamEst, alpha=0.05,
            strat=None, numFeat=200):
    '''
    short method to handle all the steps in the SVR
    '''
    crossValDict = makeFolds(feature, age, crossVal)
    # outputDict = {}
    trainDict = {}
    testAgeVec = np.array([])
    predAgeVec = np.array([])
    tempFeatVec = np.zeros_like(feature[0, :])

    for i, run in enumerate(crossValDict.keys()):
        start = time.time()
        if not doPermute:
            # Alert on running
            print('Running fold ' + str(i))
        # Get the training and test tuples
        trainTuple, testTuple = crossValDict[run]
        # unpack tuples
        trainFeature, trainAge = trainTuple
        testFeature, testAge = testTuple

        # Get the number of features
        numberFeat = trainFeature.shape[1]

        # First run feature selection
        featIndex = findFeatures(trainFeature, trainAge,
                                 strat, kernel='linear',
                                 numFeat=numFeat, alpha=alpha)

        tempFeatVec[featIndex] += 1

        # Get number of retained Features
        keptFeat = np.sum(featIndex)

        # Now reduce the features with this index
        selectTrainFeat = trainFeature[:, featIndex]
        selectTestFeat = testFeature[:, featIndex]

        # Get the best parameters for this training set
        if runParamEst:
            bestC, bestE = findParameters(selectTrainFeat, trainAge,
                                          kernel, nCors)
        else:
            bestC = 20.0
            bestE = 0.001

        paramStop = time.time()

        # Train model on train data
        modelstart = time.time()
        model = trainModel(selectTrainFeat, trainAge, kernel, bestC, bestE)
        modelstop = time.time()
        # Test model on test data
        predictedAge = testModel(model, selectTestFeat)
        # Test model on train data - for sanity check
        trainPredict = testModel(model, selectTrainFeat)
        trainOut = np.concatenate((trainAge[..., None],
                                   trainPredict[..., None]),
                                  axis=1)
        trainDict[run] = trainOut

        # Store predicted and true age in the output directory
        testAgeVec = np.append(testAgeVec, testAge)
        predAgeVec = np.append(predAgeVec, predictedAge)

        # Take time
        stop = time.time()
        elapsedParam = np.round(paramStop - start, 2)
        elapsedModel = np.round(modelstop - modelstart, 2)
        elapsedFull = np.round(stop - start, 2)

        # outTuple = (testAge, predictedAge)
        # outputDict[run] = outTuple
        if not testAgeVec.shape == predAgeVec.shape:
            print('true and predicted age don\'t match in run ' + str(i) + ':\n'
                  + '    true: ' + str(testAgeVec.shape) + '\n'
                  + '    pred: ' + str(predAgeVec.shape))
        else:
            if not doPermute:
                print('Run ' + str(i) + ':\n'
                      + '    true: ' + str(testAgeVec.shape) + '\n'
                      + '    pred: ' + str(predAgeVec.shape))
        if not doPermute:
            print('    bestC: ' + str(bestC) + '\n'
                  + '    bestE: ' + str(bestE) + '\n'
                  + '    feat: ' + str(keptFeat) + ' / ' + str(numberFeat) + '\n'
                  + 'parameter selection took: ' + str(elapsedParam) + ' s\n'
                  + 'model fitting took: ' + str(elapsedModel) + ' s\n'
                  + 'in total took: ' + str(elapsedFull) + ' s')

    # Done, stack the output together (true age first, then predicted)
    if not doPermute:
        print(str(np.max(tempFeatVec)) + ' consensus')
    outputMatrix = np.concatenate((testAgeVec[..., None],
                                   predAgeVec[..., None]),
                                  axis=1)

    return outputMatrix, trainDict
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

    fig, (within, between) = plt.subplots(1, 2, sharex=False, sharey=False)

    # fit ages
    predMat = np.concatenate((refAge[..., None], np.ones_like(refAge)[..., None]),
                             axis=1)
    WrobustResult = fitRobust(wPred, predMat)
    WglmResult = fitGLM(wPred, predMat)
    BrobustResult = fitRobust(bPred, predMat)
    BglmResult = fitGLM(bPred, predMat)

    WrobustSlope = WrobustResult.params[0]
    WrobustIntercept = WrobustResult.params[1]
    WglmSlope = WglmResult.params[0]
    WglmIntercept = WglmResult.params[1]

    BrobustSlope = BrobustResult.params[0]
    BrobustIntercept = BrobustResult.params[1]
    BglmSlope = BglmResult.params[0]
    BglmIntercept = BglmResult.params[1]

    WrobustT = WrobustResult.tvalues[0]
    WrobustP = st.t.sf(np.abs(WrobustT), WrobustResult.df_resid)
    WglmT = WglmResult.tvalues[0]
    WglmP = st.t.sf(np.abs(WglmT), WglmResult.df_resid)

    BrobustT = BrobustResult.tvalues[0]
    BrobustP = st.t.sf(np.abs(BrobustT), BrobustResult.df_resid)
    BglmT = BglmResult.tvalues[0]
    BglmP = st.t.sf(np.abs(BglmT), BglmResult.df_resid)

    xnew = np.arange(refAge.min() - 1, refAge.max() + 1, 0.1)
    WrobustFit = WrobustSlope * xnew + WrobustIntercept
    WglmFit = WglmSlope * xnew + WglmIntercept

    BrobustFit = BrobustSlope * xnew + BrobustIntercept
    BglmFit = BglmSlope * xnew + BglmIntercept

    wP = np.polyfit(wTrue, wPred, 1)
    bP = np.polyfit(bTrue, bPred, 1)

    wFit = np.polyval(wP, xnew)
    bFit = np.polyval(bP, xnew)

    withinCorr, withinP = st.pearsonr(wTrue, wPred)
    wMSE = np.mean(np.square(wTrue - wPred))
    within.plot(wTrue, wPred, 'k.')
    within.plot(xnew, WrobustFit, 'r', label=('robust '
                                              + str(np.round(WrobustSlope, 2))
                                              + ' ' + str(np.round(WrobustP, 3))))
    within.plot(xnew, WglmFit, 'b', label=('glm '
                                           + str(np.round(WglmSlope, 2))
                                           + ' ' + str(np.round(WglmP, 3))))
    within.plot(wTrue, wTrue, 'g', label='true')

    within.set_xlabel('true age')
    within.set_ylabel('predicted age')
    within.legend()

    betweenCorr, betweenP = st.pearsonr(bTrue, bPred)
    bMSE = np.mean(np.square(bTrue - bPred))
    between.plot(bTrue, bPred, 'k.')
    between.plot(xnew, BrobustFit, 'r', label=('robust '
                                               + str(np.round(BrobustSlope, 2))
                                               + ' ' + str(np.round(BrobustP, 3))))
    between.plot(xnew, BglmFit, 'b', label=('glm '
                                            + str(np.round(BglmSlope, 2))
                                            + ' ' + str(np.round(BglmP, 3))))
    between.plot(bTrue, bTrue, 'g', label='true')
    between.set_xlabel('true age')
    between.set_ylabel('predicted age')
    between.legend()

    within.set_title('within ('
                     + str(np.round(withinCorr, 2)) + ', '
                     + str(np.round(withinP, 3)) + ') '
                     + str(np.round(wMSE, 2)))
    between.set_title('between ('
                      + str(np.round(betweenCorr, 2)) + ', '
                      + str(np.round(betweenP, 3)) + ') '
                      + str(np.round(bMSE, 2)))

    fig.suptitle(title)
    plt.show()
    userIn = raw_input("Press Enter or break...\n")
    plt.close()

    return userIn

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


def singlePlot(result, title):
    true = result[:, 0]
    pred = result[:, 1]

    # fit ages
    predMat = np.concatenate((true[..., None], np.ones_like(true)[..., None]),
                             axis=1)
    robustResult = fitRobust(pred, predMat)
    glmResult = fitGLM(pred, predMat)

    # Plot the ages again
    robustSlope = robustResult.params[0]
    robustIntercept = robustResult.params[1]
    glmSlope = glmResult.params[0]
    glmIntercept = glmResult.params[1]

    robustT = robustResult.tvalues[0]
    robustP = st.t.sf(np.abs(robustT), robustResult.df_resid)
    glmT = glmResult.tvalues[0]
    glmP = st.t.sf(np.abs(glmT), glmResult.df_resid)

    # prepare
    corr, p = st.pearsonr(true, pred)
    mse = np.mean(np.square(true - pred))
    xnew = np.arange(true.min() - 1, true.max() + 1, 0.1)
    robustFit = robustSlope * xnew + robustIntercept
    glmFit = glmSlope * xnew + glmIntercept

    # Plot shit
    plt.plot(true, pred, 'k.')
    plt.plot(true, true, 'g', label='perfect')

    plt.plot(xnew, robustFit, 'r', label=('robust '
                                          + str(np.round(robustSlope, 2))
                                          + ' ' + str(np.round(robustP, 3))))
    plt.plot(xnew, glmFit, 'b', label=('glm '
                                       + str(np.round(glmSlope, 2))
                                       + ' ' + str(np.round(glmP, 3))))
    plt.legend()
    plt.title(title + ' ('
              + str(np.round(corr, 2)) + ', '
              + str(np.round(p, 3)) + ') '
              + str(np.round(mse, 2)))
    plt.show()
    userIn = raw_input("Press Enter or break...\n")
    plt.close()

    return userIn


def networkPlot(networkResults):
    '''
    Method to visualize the network level results
    '''
    for network in networkResults.keys():
        print('Plotting network ' + network + ' now.')
        (withinResult, betweenResult) = networkResults[network]
        dualPlot(withinResult, betweenResult, network)


def trainPlot(withinDict, betweenDict=None):
    '''
    Method to visualize the network level results on training data (aka for
    each cross validation loop)
    '''
    for run in withinDict.keys():
        print('Plotting fold ' + run + ' now.')
        withinResult = withinDict[run]
        # Plot the stuff
        if betweenDict:
            betweenResult = betweenDict[run]
            userIn = dualPlot(withinResult, betweenResult, run)
        else:
            userIn = singlePlot(withinResult, run)

        if userIn == 'break':
            print('breaking')
            break


def runMean(connectomeStack, ageStack, networkNodes, uniqueRoi, crossVal):
    '''
    Method that runs on the mean connectivity, separated by within and between
    '''
    # Prepare containers
    withinFeature = np.array([])
    betweenFeature = np.array([])

    for i, network in enumerate(networkNodes.keys()):
        if not doPermute:
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

        # Make the features
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

    if not doPermute:
        print('Doing the mean!')
        # Check if the features are ok
        print('Age: ' + str(ageStack.shape))
        print('Within: ' + str(withinFeature.shape))
        print('Between: ' + str(betweenFeature.shape))
        print(stratStr)

    if np.isnan(withinFeature).any():
        howMany = len(np.where(np.isnan(withinFeature))[0])
        message = ('within matrix contains nan at ' + str(howMany)
                   + ' locations')
        raise Exception(message)

    if np.isnan(betweenFeature).any():
        howMany = len(np.where(np.isnan(betweenFeature))[0])
        message = ('between matrix contains nan ' + str(howMany)
                   + ' locations')
        raise Exception(message)

    # do the real thing
    withinResult, withinTrainDict = mainSVR(withinFeature,
                                            ageStack,
                                            crossVal,
                                            kernel,
                                            nCors,
                                            runParamEst,
                                            alpha=alpha,
                                            strat=featureSelection,
                                            numFeat=desFeat)
    betweenResult, betweenTrainDict = mainSVR(betweenFeature,
                                              ageStack,
                                              crossVal,
                                              kernel,
                                              nCors,
                                              runParamEst,
                                              alpha=alpha,
                                              strat=featureSelection,
                                              numFeat=desFeat)

    # Done running, plotting now
    if doPlot:
        print('Plotting what we actually wanted...\n'
              + stratStr)
        # Plot mean connectivity across age
        plt.plot(ageStack, withinFeature, '.')
        plt.title('within mean connectivity')
        plt.show()
        raw_input('Enter...')
        plt.close()

        plt.plot(ageStack, betweenFeature, 'g.')
        plt.title('between mean connectivity')
        plt.show()
        raw_input('Enter...')
        plt.close()

        dualPlot(withinResult, betweenResult, 'within and between connectivity'
                 + ' predicting age')
        trainPlot(withinTrainDict, betweenTrainDict)

    testSaveTuple = (withinResult, betweenResult)
    trainSaveTuple = (withinTrainDict, betweenTrainDict)
    if doSave:
        status = saveOutput(pathToPredictionOutputFile, testSaveTuple)
        print(status)
        status = saveOutput(pathToTrainOutputFile, trainSaveTuple)
        print(status)

    return testSaveTuple


def runBrain(connectomeStack, ageStack, crossVal):
    '''
    Method that runs on whole brain connectivity
    '''
    if not doPermute:
        print('Lets do the brain!\n'
              + stratStr)
    mask = np.ones_like(connectomeStack[..., 0])
    mask = np.tril(mask, -1)
    feature = connectomeStack[mask == 1].T
    result, trainDict = mainSVR(feature,
                                ageStack,
                                crossVal,
                                kernel,
                                nCors,
                                runParamEst,
                                alpha=alpha,
                                strat=featureSelection,
                                numFeat=desFeat)

    if doPlot:
        singlePlot(result, 'whole brain SVR plot')
        trainPlot(trainDict)

    if doSave:
        status = saveOutput(pathToPredictionOutputFile, result)
        print(status)
        status = saveOutput(pathToTrainOutputFile, trainDict)
        print(status)

    return result


def runNetwork(connectomeStack, ageStack, networkNodes, uniqueRoi, crossVal):
    '''
    Method that runs on network based connectivity
    '''
    # Prepare containers
    networkResults = {}
    networkTrainResults = {}

    for i, network in enumerate(networkNodes.keys()):
        if not doPermute:
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

        # Get the between network connections
        betweenStack = networkStack[:, betweenIndex, ...]
        betweenRows, betweenCols, betweenSubs = betweenStack.shape
        # Also flatten this stuff out
        betweenMatrix = np.reshape(betweenStack, (betweenRows * betweenCols,
                                                  betweenSubs))

        # Get the network features
        withinFeature = withinMatrix.T
        betweenFeature = betweenMatrix.T

        # Run SVR
        if not doPermute:
            print('\nRunning within ' + network + ' connectivity SVR ('
                  + str(i) + '/' + str(len(networkNodes.keys())) + ')')
        withinResult, withinTrainDict = mainSVR(withinFeature,
                                                ageStack,
                                                crossVal,
                                                kernel,
                                                nCors,
                                                runParamEst,
                                                alpha=alpha,
                                                strat=featureSelection,
                                                numFeat=desFeat)
        if not doPermute:
            print('\nRunning between ' + network + ' connectivity SVR ('
                  + str(i) + '/' + str(len(networkNodes.keys())) + ')')
        betweenResult, betweenTrainDict = mainSVR(betweenFeature,
                                                  ageStack,
                                                  crossVal,
                                                  kernel,
                                                  nCors,
                                                  runParamEst,
                                                  alpha=alpha,
                                                  strat=featureSelection,
                                                  numFeat=desFeat)

        # Store the output in the output Dictionary for networks
        result = (withinResult, betweenResult)
        networkResults[network] = result
        trainResult = (withinTrainDict, betweenTrainDict)
        networkTrainResults[network] = trainResult

    if doSave:
        status = saveOutput(pathToPredictionOutputFile, networkResults)
        print(status)
        status = saveOutput(pathToTrainOutputFile, networkTrainResults)
        print(status)

    if doPlot:
        # Done with running analysis. Plotting by network now
        networkPlot(networkResults)
        for network in networkTrainResults:
            print('Plotting training on ' + network + '\n'
                  + stratStr)
            (withinTrainDict, betweenTrainDict) = networkTrainResults[network]
            trainPlot(withinTrainDict, betweenTrainDict)

    return networkResults


def runPermute(runwhat, numPermute, connectomeStack, ageStack, networkNodes,
              uniqueRoi, crossVal):
    '''
    Method to do permutation testing. Return a matrix of:
        2 (true, predicted)
        BY
        numSubjects
        BY
        numPermute
    dimensions for calculation of accuracy.
    If there are within AND between results, the between results get appended
    along the 1-Axis (column 3 and 4).
    '''
    # Prepare the storage container
    permutationDict = {}
    # What are we running
    if runwhat == 'network':
        for permutation in np.arange(numPermute):
            # Shuffle the labels
            np.random.shuffle(ageStack)
            networkResults = runNetwork(connectomeStack, ageStack,
                                        networkNodes, uniqueRoi,
                                        crossVal)
            # Get the network results out again
            for network in networkResults.keys():
                # Generate the matrix of results for this network and
                # permutation
                (withinResult, betweenResult) = networkResults[network]
                resultMat = np.concatenate((withinResult, betweenResult),
                                           axis=1)
                resultMat = resultMat[..., None]
                # Now stack it along the Axis-2
                if not network in permutationDict:
                    # This is the first time, initialize
                    permutationDict[network] = resultMat
                else:
                    tempMat = permutationDict[network]
                    tempMat = np.concatenate((tempMat,
                                              resultMat),
                                             axis=2)
                    permutationDict[network] = tempMat
            # Done with reading the networks for this permutation
            sys.stdout.write('\r' + stratStr + ': '
                             + str(permutation) + ' / ' + str(numPermute))
            sys.stdout.flush()
        # Done with permuting

    elif runwhat == 'brain':
        for permutation in np.arange(numPermute):
            # Shuffle the labels
            np.random.shuffle(ageStack)
            result = runBrain(connectomeStack, ageStack, crossVal)
            # Generate result matrix for this permutation by adding one
            # dimension
            resultMat = result[..., None]
            # Get the MSE just for fun
            tempMSE = np.mean(np.square(result[:, 0] - result[:, 1]))
            # Now stack along axis-2 to the dict
            if not runwhat in permutationDict:
                # This is the first time, initialize
                permutationDict[runwhat] = resultMat
            else:
                tempMat = permutationDict[runwhat]
                tempMat = np.concatenate((tempMat,
                                          resultMat),
                                         axis=2)
                permutationDict[runwhat] = tempMat
            # Done with stacking result
            sys.stdout.write('\r' + stratStr + ': '
                             + str(permutation) + ' / ' + str(numPermute)
                             + ' (' + str(tempMSE) + ')')
            sys.stdout.flush()
        # Done with permuting

    elif runwhat == 'mean':
        for permutation in np.arange(numPermute):
            # Shuffle the labels
            np.random.shuffle(ageStack)
            (withinResult, betweenResult) = runMean(connectomeStack, ageStack,
                                                    networkNodes, uniqueRoi,
                                                    crossVal)
            # Generate the matrix of results for this network and
            # permutation
            resultMat = np.concatenate((withinResult, betweenResult),
                                       axis=1)
            resultMat = resultMat[..., None]
            # Get the MSE just for fun
            tempMSE = np.mean(np.square(result[:, 0] - result[:, 1]))
            # Now stack along axis-2 to the dict
            if not runwhat in permutationDict:
                # This is the first time, initialize
                permutationDict[runwhat] = resultMat
            else:
                tempMat = permutationDict[runwhat]
                tempMat = np.concatenate((tempMat,
                                          resultMat),
                                         axis=2)
                permutationDict[runwhat] = tempMat
            # Done with stacking result
            sys.stdout.write('\r' + stratStr + ': '
                             + str(permutation) + ' / ' + str(numPermute))
            sys.stdout.flush()
        # Done with permuting

    else:
        message = ('Don\'t know what to do here. You selected ' + runwhat)
        raise Exception(message)

    return permutationDict


def getWeights(weightVector):
    '''
    Method to get the weights back into the connectome matrix and save them
    '''


    pass


def saveOutput(outputFilePath, output):
    f = gzip.open(outputFilePath, 'wb')
    cPickle.dump(output, f)
    f.close()
    status = ('Saved to ' + outputFilePath)

    return status


def Main():
    # Define the inputs
    pathToConnectomeDir = '/home2/surchs/secondLine/connectomes/wave/dos160'
    pathToPhenotypicFile = '/home2/surchs/secondLine/configs/wave/wave_pheno81_uniform.csv'
    pathToSubjectList = '/home2/surchs/secondLine/configs/wave/wave_subjectList.csv'

    pathToNetworkNodes = '/home2/surchs/secondLine/configs/networkNodes_dosenbach.dict'
    pathToRoiMask = '/home2/surchs/secondLine/masks/dos160_wave_81_3mm.nii.gz'

    connectomeSuffix = '_connectome_glob.txt'

    # Define global variables
    global doCV
    global kfold
    global nCors
    global kernel
    global runParamEst
    global doPlot
    global doSave
    global featureSelection
    global alpha
    global desFeat
    global doPermute
    doCV = 'kfold'
    kfold = 10
    nCors = 5
    kernel = 'linear'
    runParamEst = True
    doPlot = False
    doSave = False
    featureSelection = 'rfe'
    alpha = 0.2
    desFeat = 200
    doPermute = True

    global pathToTrainOutputFile
    global pathToPredictionOutputFile
    global pathToPermutationOutputFile
    global stratStr

    # Define local variables
    runwhat = 'brain'
    numPermute = 100
    which = 'wave'

    stratStr = (doCV
                + '_' + str(kfold)
                + '_' + kernel
                + '_' + str(runParamEst)
                + '_' + str(featureSelection)
                + '_' + str(runwhat)
                + '_' + os.path.splitext(connectomeSuffix)[0])

    print(stratStr)

    pathToOutputFile = ('/home2/surchs/secondLine/SVM/wave/dos160/emp_'
                        + stratStr + '_SVR')
    pathToTrainOutputFile = pathToOutputFile + '.train'
    pathToPredictionOutputFile = pathToOutputFile + '.pred'
    pathToPermutationOutputFile = pathToOutputFile + '.permut'

    # Check the fucking paths
    if  (not which in pathToConnectomeDir or
         not which in  pathToPhenotypicFile or
         not which in pathToSubjectList or
         not which in pathToRoiMask or
         not which in pathToOutputFile):
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

    if doCV == 'loocv':
        crossVal = cv.LeaveOneOut(len(phenoSubjects))
        nFolds = crossVal.n

    elif doCV == 'kfold':
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
        # Check if nan in there
        if np.isnan(connectome).any():
            print(subject + ' has nan in the connectome!')

        normalizedConnectome = fisherZ(connectome)
        # normalizedConnectome = connectome
        # Get the mean connectivity
        uniqueConnections = getUniqueMatrixElements(normalizedConnectome)
        meanConn = np.mean(uniqueConnections)
        meanConnStack = np.append(meanConnStack, meanConn)

        # Stack the connectome
        connectomeStack = stackConnectome(connectomeStack, normalizedConnectome)
        print('connectomeStack: ' + str(connectomeStack.shape))
        # Stack ages
        ageStack = stackAges(ageStack, phenoAge)

    # Done preparing, now run whatever we want
    # First check if we are permuting
    if doPermute:
        # Yes, we are permuting
        permutationDict = runPermute(runwhat, numPermute,
                                     connectomeStack, ageStack,
                                     networkNodes, uniqueRoi,
                                     crossVal)
        status = saveOutput(pathToPermutationOutputFile, permutationDict)
        print(status)

    else:
        # No, we are not permuting
        if runwhat == 'network':
            runNetwork(connectomeStack, ageStack,
                       networkNodes, uniqueRoi,
                       crossVal)
            pass
        elif runwhat == 'brain':
            runBrain(connectomeStack, ageStack,
                     crossVal)
            pass
        elif runwhat == 'mean':
            runMean(connectomeStack, ageStack,
                    networkNodes, uniqueRoi,
                    crossVal)
            pass
        else:
            print('Don\'t know what to do here. You selected ' + runwhat)


if __name__ == '__main__':
    Main()
