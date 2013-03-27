'''
Created on Feb 22, 2013

@author: surchs
'''
import os
import sys
import gzip
import time
import glob
import cPickle
import numpy as np
import pandas as pa
import nibabel as nib
from sklearn import svm
import statsmodels.api as sm
from scipy import stats as st
from sklearn.metrics import auc
import sklearn.grid_search as gs
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


def mainSVR(feature, age, crossVal, kernel, nCors, runParamEst):
    '''
    short method to handle all the steps in the SVR
    '''
    crossValDict = makeFolds(feature, age, crossVal)
    # outputDict = {}
    trainDict = {}
    testAgeVec = np.array([])
    predAgeVec = np.array([])

    for i, run in enumerate(crossValDict.keys()):
        start = time.time()
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

        paramStop = time.time()

        # Train model on train data
        model = trainModel(trainFeature, trainAge, kernel, bestC, bestE)
        # Test model on test data
        predictedAge = testModel(model, testFeature)
        # Test model on train data - for sanity check
        trainPredict = testModel(model, trainFeature)
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
        elapsedFull = np.round(stop - start, 2)

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
              + '    bestE: ' + str(bestE) + '\n'
              + 'parameter selection took: ' + str(elapsedParam) + ' s\n'
              + 'in total took: ' + str(elapsedFull) + ' s')

    # Done, stack the output together (true age first, then predicted)
    outputMatrix = np.concatenate((testAgeVec[..., None],
                                   predAgeVec[..., None]),
                                  axis=1)

    return outputMatrix, trainDict
    # Done, return the output dictionary
    # return outputDict


def dualPlot(resultWithin, resultBetween, title, outDir, trainPlot=False,
             perm=None):
    '''
    method to plot network results side by side
    '''
    # Unpack the results first
    wTrue = resultWithin[:, 0]
    wPred = resultWithin[:, 1]
    bTrue = resultBetween[:, 0]
    bPred = resultBetween[:, 1]

    if doPermut:
        # unpack these aswell
        (withinPerm, betweenPerm) = perm
        (wEmpMSE, wPerMSE, wPValue) = withinPerm
        (bEmpMSE, bPerMSE, bPValue) = betweenPerm

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
    # Set size
    fig.set_size_inches(20.5, 10.5)
    fileName = (title + '.png')
    filePath = os.path.join(outDir, fileName)

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

    # Get the correlation between predicted and true age
    wCorr, wCorrP = st.pearsonr(wTrue, wPred)
    bCorr, bCorrP = st.pearsonr(bTrue, bPred)

    wTitle = ('w ('
              + str(np.round(wCorr, 2)) + ', '
              + str(np.round(wCorrP, 3)) + ')')
    bTitle = ('b (' + str(np.round(bCorr, 2)) + ', '
              + str(np.round(bCorrP, 4)) + ')')

    if doPermut:
        wTitle = (wTitle
                  + ' - ' + str(np.round(wEmpMSE, 2))
                  + ', ', str(np.round(wPerMSE, 2))
                  + ' (' + str(np.round(wPValue, 3)) + ')')
        bTitle = (bTitle
                  + ' - ' + str(np.round(bEmpMSE, 2))
                  + ', ', str(np.round(bPerMSE, 2))
                  + ' (' + str(np.round(bPValue, 3)) + ')')
    within.set_title(wTitle)
    between.set_title(bTitle)

    # withinCorr, withinP = st.pearsonr(wTrue, wPred)
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

    # betweenCorr, betweenP = st.pearsonr(bTrue, bPred)
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
    fig.suptitle(title)

    if not trainPlot:
        print('Saving to ' + filePath)
        fig.savefig(filePath, dpi=150)
        plt.close()

    else:
        plt.show()
        userIn = raw_input("Enter, break, save\n")
        if userIn == 'save':
            # print('Saving to ' + filePath)
            # fig.savefig(filePath, dpi=150)
            pass
        else:
            return userIn
        plt.close()


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


def calcPredAcc(trueLabel, predLabel):
    '''
    returns ratio of correct labels
    '''
    # Sanity check
    if not len(trueLabel) == len(predLabel):
        print('number of true and pred labels doesn\'t match')
        print('    true: ' + str(len(trueLabel)) + '\n'
              + '    pred: ' + str(len(predLabel)))

    else:
        numLabels = len(trueLabel)
        corrIndex = trueLabel == predLabel
        numCorr = float(np.sum(corrIndex))
        ratio = numCorr / numLabels

        return ratio


def singlePlot(predAcc, title):
    # Plot shit
    '''
    Columns:
        1) testLabel
        2) predLabel
        3) testAge
        4) probVec for label testLabel = 1
        5) meanFPR
        6) meanTPR
    '''
    true = predAcc[:, 0]
    pred = predAcc[:, 1]
    age = predAcc[:, 2]
    meanFPR = predAcc[:, 4]
    meanTPR = predAcc[:, 5]
    ratio = calcPredAcc(true, pred)
    print('Prediction accuracy was: ' + str(ratio))

    plt.bar(1, ratio, align='center')
    plt.title(title)
    plt.show()
    userIn = raw_input("Press Enter or break...\n")
    plt.close()

    # offset for better display
    true += 0.1
    ymax = true.max() + 1
    ymin = true.min() - 1
    xmax = age.max() - 0.1
    xmin = age.min() + 0.1
    plt.plot(age, pred, 'r.', label='pred')
    plt.plot(age, true, 'g.', label='true')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.title(title + ' classification')
    plt.legend()
    plt.show()
    userIn = raw_input("Press Enter or break...\n")
    plt.close()

    # Now plot the ROC
    meanAUC = auc(meanFPR, meanTPR)
    plt.plot(meanFPR, meanTPR, 'r--',
             label='Mean ROC (area = %0.2f)' % meanAUC, lw=2)
    plt.plot([0, 1], [0, 1], '--', c=(0.6, 0.6, 0.6),
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operator curve for %s' % title)
    plt.legend()
    plt.show()
    userIn = raw_input("Press Enter or break...\n")
    plt.close()

    return userIn


def networkPlot(networkResults):
    '''
    Method to visualize the network level results
    '''
    withinStack = np.array([])
    betweenStack = np.array([])
    netNames = []
    for network in networkResults.keys():
        print('Plotting network ' + network + ' now.')
        (withinResult, betweenResult) = networkResults[network]
        # Get it out
        '''
        Columns:
            1) testLabel
            2) predLabel
            3) testAge
            4) probVec for label testLabel = 1
            5) meanFPR
            6) meanTPR
        '''
        wTrue = withinResult[:, 0]
        wPred = withinResult[:, 1]
        bTrue = betweenResult[:, 0]
        bPred = betweenResult[:, 1]

        withinRatio = calcPredAcc(wTrue, wPred)
        betweenRatio = calcPredAcc(bTrue, bPred)
        withinStack = np.append(withinStack, withinRatio)
        betweenStack = np.append(betweenStack, betweenRatio)
        netNames.append(network)
        print('    within: ' + str(withinRatio) + '\n'
              '    between: ' + str(betweenRatio))

    # Done, plot
    index = np.arange(len(withinStack)) + 1

    fig, (within, between) = plt.subplots(1, 2, sharex=False, sharey=False)
    within.bar(index, withinStack, align='center')
    between.bar(index, betweenStack, align='center')

    within.set_title('within')
    between.set_title('between')

    within.set_ylabel('% correct')
    within.set_xlabel('network')
    within.set_xticks(index)
    within.set_xticklabels(netNames)

    between.set_ylabel('% correct')
    between.set_xlabel('network')
    between.set_xticks(index)
    between.set_xticklabels(netNames)

    fig.suptitle('Network level classification results')
    fig.autofmt_xdate()
    plt.show()
    raw_input("Press Enter...\n")
    plt.close()


def trainPrep(withinDict, betweenDict=None):
    '''
    Method to get all the fit slopes across folds
    '''
    inDex = np.array([])
    withinGLM = np.array([])
    withinROB = np.array([])
    betweenGLM = np.array([])
    betweenROB = np.array([])

    numRuns = len(withinDict.keys())
    print('Quickfitting now')
    for i, run in enumerate(withinDict.keys()):
        sys.stdout.write('\r' + str(i) + '/' + str(numRuns) + ' done. ')
        sys.stdout.flush()

        withinResult = withinDict[run]
        # inDex
        inDex = np.append(inDex, int(run))

        wTrue = withinResult[:, 0]
        wPred = withinResult[:, 1]
        predMat = np.concatenate((wTrue[..., None],
                                  np.ones_like(wTrue)[..., None]),
                                 axis=1)

        WglmFit = fitGLM(wPred, predMat)
        WrobustFit = fitRobust(wPred, predMat)

        WglmSlope = WglmFit.params[0]
        WrobustSlope = WrobustFit.params[0]

        withinGLM = np.append(withinGLM, WglmSlope)
        withinROB = np.append(withinROB, WrobustSlope)

        if betweenDict:
            betweenResult = betweenDict[run]
            bTrue = betweenResult[:, 0]
            bPred = betweenResult[:, 1]
            predMat = np.concatenate((bTrue[..., None],
                                      np.ones_like(bTrue)[..., None]),
                                     axis=1)

            BglmFit = fitGLM(bPred, predMat)
            BrobustFit = fitRobust(bPred, predMat)

            BglmSlope = BglmFit.params[0]
            BrobustSlope = BrobustFit.params[0]

            betweenGLM = np.append(betweenGLM, BglmSlope)
            betweenROB = np.append(betweenROB, BrobustSlope)

    # Done with all the folds
    within = (withinGLM, withinROB)
    between = (betweenGLM, betweenROB)

    if betweenDict:
        return (inDex, within, between)
    else:
        return (inDex, within)


def trainSlopePlot(index, within, between=None, title='train slopes',
                   outDir='./'):
    '''
    Method to quickly plot the slopes across folds
    '''
    withinGLM, withinROB = within
    fileName = (title + '.png')
    filePath = os.path.join(outDir, fileName)

    withinSlopes = []
    withinSlopes.append(withinGLM)
    withinSlopes.append(withinROB)
    wTest = np.append(withinGLM, withinROB)

    if between:
        betweenGLM, betweenROB = between
        bTest = np.append(betweenGLM, betweenROB)

        fig, (within, between) = plt.subplots(1, 2, sharex=True, sharey=False)
        fig.set_size_inches(15, 8)

        betweenSlopes = []
        betweenSlopes.append(betweenGLM)
        betweenSlopes.append(betweenROB)

        within.boxplot(withinSlopes)
        within.set_title('within')
        within.set_ylim([wTest.min() - 0.1, wTest.max() + 0.1])

        between.boxplot(betweenSlopes)
        between.set_title('between')
        between.set_ylim([bTest.min() - 0.1, bTest.max() + 0.1])

        fig.suptitle(title)

    else:
        fig = plt.figure()
        fig.set_size_inches(20.5, 10.5)
        within = fig.add_subplot(111)
        within.plot(index, withinGLM, 'g.', label='glm slope')
        within.plot(index, withinROB, 'r.', label='robust slope')
        within.set_title('within')
        within.set_xlabel('folds')
        within.set_ylabel('slope')
        within.legend()
        fig.suptitle(title)

    # print('Saving ' + filePath)
    fig.savefig(filePath, dpi=150)
    # plt.show()
    # userIn = raw_input("Enter, break, save\n")
    plt.close()


def trainPlot(withinDict, betweenDict=None):
    '''
    Method to visualize the network level results on training data (aka for
    each cross validation loop)
    '''
    predWithin = np.array([])
    predBetween = np.array([])
    for run in withinDict.keys():
        print('Plotting fold ' + run + ' now.')
        withinResult = withinDict[run]
        wTrue = withinResult[:, 0]
        wPred = withinResult[:, 1]
        wAge = withinResult[:, 2]

        # Get prediction accuracy
        withinRatio = calcPredAcc(wTrue, wPred)
        predWithin = np.append(predWithin, withinRatio)
        wTrue += 0.1

        if betweenDict:
            betweenResult = betweenDict[run]
            bTrue = betweenResult[:, 0]
            bPred = betweenResult[:, 1]
            bAge = betweenResult[:, 2]
            betweenRatio = calcPredAcc(bTrue, bPred)
            predBetween = np.append(predBetween, betweenRatio)
            bTrue += 0.1

            # Plot dual shit
            fig, (within, between) = plt.subplots(1, 2,
                                                  sharex=False,
                                                  sharey=False)

            within.set_title('within network')
            within.set_xlim(wAge.min() - 0.1, wAge.max() + 0.1)
            within.set_ylim(wTrue.min() - 1, wTrue.max() + 1)
            within.plot(wAge, wTrue, 'g.', label='true')
            within.plot(wAge, wPred, 'r.', label=('pred '
                                                  + str(np.round(withinRatio))))
            within.legend()

            between.set_title('between network')
            between.set_xlim(bAge.min() - 0.1, bAge.max() + 0.1)
            between.set_xlim(bTrue.min() - 0.1, bTrue.max() + 0.1)
            between.plot(bAge, bTrue, 'g.', label='true')
            between.plot(bAge, bPred, 'r.', label=('pred '
                                                   + str(np.round(betweenRatio, 3))))
            between.legend()

            fig.suptitle('Run ' + str(run))

        else:
            # just one plot
            fig = plt.figure()
            within = fig.add_subplot(111)
            within.set_title('within network')
            within.set_xlim(wAge.min() - 0.1, wAge.max() + 0.1)
            within.set_ylim(wTrue.min() - 1, wTrue.max() + 1)
            within.plot(wAge, wTrue, 'g.', label='true')
            within.plot(wAge, wPred, 'r.', label=('pred '
                                                  + str(np.round(withinRatio, 3))))
            within.legend()
            fig.suptitle('Run ' + str(run))
        # Plot the shit
        plt.show()
        userIn = raw_input("Press Enter or break...\n")
        plt.close()
        if userIn == 'break':
            print('breaking')
            break

    # Done iterating
    if betweenDict:
        return predWithin, predBetween
    else:
        return predWithin


def testPermutation(testResult, permutationResult):
    '''
    Method that essentially runs a t-test to determine if the test result
    has significantly better error than the permutation set
    '''
    # test data
    trueTest = testResult[:, 0]
    predTest = testResult[:, 1]
    # permutation data
    truePermut = permutationResult[:, 0, :]
    predPermut = permutationResult[:, 1, :]

    # Get the MSE for the empirical values
    empMSE = np.mean(np.square(trueTest - predTest))
    # Now make a distribution of MSE from the permutations
    distMSE = np.mean(np.square(truePermut - predPermut), axis=0)
    # Run a one sample t-test on this thing - if significant, then our
    # empirical MSE is significantly different from the permuted one
    tValue, pValue = st.ttest_1samp(distMSE, empMSE)

    return empMSE, distMSE, tValue, pValue


def saveOutput(outputFilePath, output):
    f = gzip.open(outputFilePath, 'wb')
    cPickle.dump(output, f)
    f.close()
    status = ('Saved to ' + outputFilePath)

    return status


def Main():
    # Define the inputs
    pathToFiles = '/home2/surchs/secondLine/SVC/wave/dos160/kfold_10_linear_True_network__connectome_glob'

    print('\nLooking for files')
    pred = glob.glob(pathToFiles + '/*.pred')
    if pred:
        if len(pred) > 1:
            message = ('More than one pred file!\n' + str(pred))
            raise Exception(message)
        pathToNetworkResults = pred[0]
    else:
        print('No prediction file at ' + str(pathToFiles))

    train = glob.glob(pathToFiles + '/*.train')
    if train:
        if len(train) > 1:
            message = ('More than one pred file!\n' + str(train))
            raise Exception(message)
        pathToTrainingResults = train[0]
    else:
        print('No training file at ' + str(pathToFiles))

    perm = glob.glob(pathToFiles + '/*.permut')
    if perm:
        if len(perm) > 1:
            message = ('More than one pred file!\n' + str(perm))
            raise Exception(message)
        pathToPermutationResults = perm[0]
    else:
        print('No permutation file at ' + str(pathToFiles))
    print('\n\n')


    '''
    pathToNetworkResults = '/home2/surchs/secondLine/SVM/wave/dos160/emp_kfold_10_linear_True_corr_brain__connectome_glob_SVR.pred'
    pathToTrainingResults = '/home2/surchs/secondLine/SVM/wave/dos160/emp_kfold_10_linear_True_corr_brain__connectome_glob_SVR.train'
    pathToPermutationResults = ''
    '''

    pathToOutputDir = '/home2/surchs/secondLine/images/SVC/empirical/wave'
    stratName = os.path.splitext(os.path.basename(pathToNetworkResults))[0]
    imageDir = os.path.join(pathToOutputDir, stratName)

    if not os.path.isdir(imageDir):
        print('Making ' + imageDir + ' now')
        os.makedirs(imageDir)

    # Define parameters
    global doPermut
    doPermut = False
    doNet = True
    doTrain = False

    doPlot = False
    doSave = True

    which = 'network'

    # Read input files
    netDict = loadArchive(pathToNetworkResults)
    trainDict = loadArchive(pathToTrainingResults)

    # trainDict = loadArchive(pathToTrainingResults)
    if doPermut:
        permutDict = loadArchive(pathToPermutationResults)

    ####################################
    # We do have the results, now plot #
    ####################################
    if which == 'mean':
        # Unpack the results
        (withinResult, betweenResult) = netDict
        (withinTrainDict, betweenTrainDict) = trainDict

        dualPlot(withinResult, betweenResult, 'within and between connectivity'
                 + ' predicting age')
        trainPlot(withinTrainDict, betweenTrainDict)

    # Plot whole brain connectivity results
    if which == 'brain':
        # unpack results
        result = netDict
        trainDict = trainDict

        singlePlot(result, 'whole brain SVC plot')
        # accBrain = trainPlot(trainDict)

    # Plot network connectivity results
    if which == 'network':
        # Unpack the network results
        networkResults = netDict
        networkTrainResults = trainDict

        networkPlot(networkResults)
        '''
        for network in networkTrainResults:
            print('Plotting training on ' + network)
            (withinTrainDict, betweenTrainDict) = networkTrainResults[network]
            accWithin, accBetween = trainPlot(withinTrainDict, betweenTrainDict)
        '''

if __name__ == '__main__':
    Main()
