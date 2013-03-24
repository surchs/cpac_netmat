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


def singlePlot(result, title, outDir, doPlot=False, doSave=False):
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
    xnew = np.arange(true.min() - 1, true.max() + 1, 0.1)
    robustFit = robustSlope * xnew + robustIntercept
    glmFit = glmSlope * xnew + glmIntercept

    # Plot shit
    fileName = (title + '.png')
    filePath = os.path.join(outDir, fileName)

    fig = plt.figure(1, figsize=(8, 8), dpi=150)
    subPlot = fig.add_subplot(111)
    subPlot.plot(true, pred, 'k.')
    subPlot.plot(true, true, 'g', label='perfect')
    subPlot.plot(xnew, robustFit, 'r', label=('robust '
                                              + str(np.round(robustSlope, 2))
                                              + ' ' + str(np.round(robustP, 3))))
    subPlot.plot(xnew, glmFit, 'b', label=('glm '
                                           + str(np.round(glmSlope, 2))
                                           + ' ' + str(np.round(glmP, 3))))
    subPlot.legend()
    fig.suptitle(title + ' ('
                 + str(np.round(corr, 2)) + ', '
                 + str(np.round(p, 3)) + ')')
    if doSave:
        print('Saving to ' + filePath)
        fig.savefig(filePath)
    if doPlot:
        plt.show()
        raw_input("Enter\n")

        plt.close()


def networkPlot(networkResults, imageDir, netPerm=None):
    '''
    Method to visualize the network level results
    '''
    for network in networkResults.keys():
        print('Plotting network ' + network + ' now.')
        (withinResult, betweenResult) = networkResults[network]
        # Now get the respective results
        if doPermut:
            # First get the permutation results
            permutResults = netPerm[network]
            # Split them up into within and between
            withinPermut = permutResults[:, :2, :]
            betweenPermut = permutResults[:, 2:4, :]
            wEmpMSE, wDistMSE, wTValue, wPValue = testPermutation(withinResult,
                                                                  withinPermut)
            bEmpMSE, bDistMSE, bTValue, bPValue = testPermutation(betweenResult,
                                                                  betweenPermut)
            # Get the mean of the permutation MSE
            wPerMSE = np.mean(wDistMSE)
            bPerMSE = np.mean(bDistMSE)
            # Stack the permutation results up
            withinPerm = (wEmpMSE, wPerMSE, wPValue)
            betweenPerm = (bEmpMSE, bPerMSE, bPValue)
            permTuple = (withinPerm, betweenPerm)

            dualPlot(withinResult, betweenResult, network, imageDir,
                     perm=permTuple)
        else:
            dualPlot(withinResult, betweenResult, network, imageDir)


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


def trainPlot(withinDict, betweenDict=None, netName='', outDir='./'):
    '''
    Method to visualize the network level results on training data (aka for
    each cross validation loop)
    '''
    for run in withinDict.keys():
        print('Plotting fold ' + run + ' now.')
        withinResult = withinDict[run]
        # Name for the plot
        plotName = (netName + '_' + str(run))
        # Plot the stuff
        if betweenDict:
            betweenResult = betweenDict[run]
            userIn = dualPlot(withinResult, betweenResult, plotName, outDir,
                              trainPlot=True)
        else:
            userIn = singlePlot(withinResult, plotName, outDir)

        if userIn == 'break':
            print('breaking')
            break


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
    pathToFiles = ''

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


    '''
    pathToNetworkResults = '/home2/surchs/secondLine/SVM/wave/dos160/emp_kfold_10_linear_True_corr_brain__connectome_glob_SVR.pred'
    pathToTrainingResults = '/home2/surchs/secondLine/SVM/wave/dos160/emp_kfold_10_linear_True_corr_brain__connectome_glob_SVR.train'
    pathToPermutationResults = ''
    '''

    pathToOutputDir = '/home2/surchs/secondLine/images/SVR/empirical/wave'
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

    doConn = False
    doBrain = True
    doMean = False

    # Read input files
    netDict = loadArchive(pathToNetworkResults)

    # trainDict = loadArchive(pathToTrainingResults)
    if doPermut:
        permutDict = loadArchive(pathToPermutationResults)

    # Plot the mean connectivity
    if doMean:
        print('Plotting what we actually wanted...\n')
        (withinResult, betweenResult) = netDict
        # (withinTrainDict, betweenTrainDict) = trainDict

        dualPlot(withinResult, betweenResult, 'within and between connectivit'
                 + ' predicting age')

        '''
        No longer interested

        index, within, between = trainPrep(withinTrainDict,
                                           betweenTrainDict)
        trainSlopePlot(index, within, between=between, title='mean_slope_train',
                       outDir=imageDir)



        if doTrain:
            trainPlot(withinTrainDict, betweenTrainDict,
                      netName='Mean', outDir='./')

        '''

    # Plot whole brain connectivity results
    if doBrain:
        result = netDict

        title = 'whole brain SVR'
        if doPermut:
            permutResult = permutDict['brain']
            # Check the permutation result
            empMSE, distMSE, tValue, pValue = testPermutation(result,
                                                              permutResult)
            mseStr = ('MSE: ' + str(empMSE) + ' / ' + str(np.mean(distMSE))
                      + ' (' + str(np.round(pValue, 4)) + ')')
            title = (title + ' ' + mseStr)

        singlePlot(result, title, imageDir,
                   doPlot=doPlot, doSave=doSave)

        '''
        No longer interested in this

        index, brain = trainPrep(trainDict)
        trainSlopePlot(index, brain, title='whole_brain_train',
                           outDir=imageDir)



        if doTrain:
            trainPlot(trainDict, netName='whole_brain', outDir=imageDir)
        '''

    # Plot network connectivity results
    if doConn:
        # Done with running analysis. Plotting by network now
        networkResults = netDict
        # networkTrainResults = trainDict

        if doPermut:
            networkPermut = permutDict
            networkPlot(networkResults, imageDir, netPerm=networkPermut)
        else:
            networkPlot(networkResults, imageDir)
        '''
        We are no longer interested in plotting training
        for network in networkTrainResults:
            print('Plotting training on ' + network)
            (withinTrainDict, betweenTrainDict) = networkTrainResults[network]
            index, within, between = trainPrep(withinTrainDict,
                                               betweenTrainDict)
            plotName = (network + '_train')
            trainSlopePlot(index, within, between=between, title=plotName,
                           outDir=imageDir)
            if doTrain:
                trainPlot(withinTrainDict, betweenTrainDict,
                          netName=network, outDir=imageDir)
        '''


if __name__ == '__main__':
    Main()
