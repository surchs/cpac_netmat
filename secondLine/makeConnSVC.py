'''
Created on Feb 22, 2013

@author: surchs

Classify
'''
import os
import gzip
import time
import cPickle
import numpy as np
import pandas as pa
import nibabel as nib
from sklearn import svm
from scipy import interp
import statsmodels.api as sm
from scipy import stats as st
from sklearn.metrics import auc
import sklearn.grid_search as gs
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
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


def makeFolds(feature, label, age, crossVal):
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

        trainLabel = label[trainIndex]
        testLabel = label[testIndex]

        # short sanity check:
        if (not trainFeat.shape[0] == trainAge.shape[0]
            or not testFeat.shape[0] == testAge.shape[0]):
            print('The features and ages in run ' + str(run)
                  + 'don\'t match up. Please check!\n'
                  + '    trainFeat: ' + str(trainFeat.shape) + '\n'
                  + '    trainAge: ' + str(trainAge.shape) + '\n'
                  + '    testFeat: ' + str(testFeat.shape) + '\n'
                  + '    testAge: ' + str(testAge.shape) + '\n')

        trainTuple = (trainFeat, trainLabel, trainAge)
        testTuple = (testFeat, testLabel, testAge)

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

    firstParameters = {'C': cParamOne}
    gridModel = svm.SVC(kernel=kernel)

    # Train first pass model
    firstTrainModel = gs.GridSearchCV(gridModel,
                                      firstParameters,
                                      cv=2,
                                      n_jobs=nCors,
                                      verbose=0)
    firstTrainModel.fit(trainFeature, trainAge)

    # First pass best parameters
    firstPassC = firstTrainModel.best_estimator_.C

    # Make the parameters for the second run
    firstExpC = np.log10(firstPassC)
    cExpTwo = np.arange(firstExpC - 1, firstExpC + 1.1, 0.1)
    cBaseTwo = np.ones_like(cExpTwo, dtype='float32') * 10
    cParamTwo = np.power(cBaseTwo, cExpTwo).tolist()

    secondParameters = {'C': cParamTwo}

    secondTrainModel = gs.GridSearchCV(gridModel,
                                       secondParameters,
                                       cv=2,
                                       n_jobs=nCors,
                                       verbose=0)
    secondTrainModel.fit(trainFeature, trainAge)

    # Final best parameters
    bestC = secondTrainModel.best_estimator_.C

    return bestC


def trainModel(trainFeature, trainLabel, kernel, C):
    '''
    module to train the model on the data
    '''
    trainModel = svm.SVC(kernel=kernel, C=C, probability=True)

    trainModel.fit(trainFeature, trainLabel)

    return trainModel


def testModel(model, testFeature):
    '''
    Method to test the model that was trained beforehand
    '''
    predictedAge = model.predict(testFeature)

    return predictedAge


def getProbability(model, testFeature, testLabel):
    '''
    Method to prepare the ROC
    '''
    probas = model.predict_proba(testFeature)
    # The index for the probability vector corresponds to the larger label
    # (most likely 1). If 1 is not the largest label, the label has to be
    # set in the roc function
    fpr, tpr, thresh = roc_curve(testLabel, probas[:, 1])

    return probas, fpr, tpr


def mainSVC(feature, label, age, crossVal, kernel, nCors, runParamEst):
    '''
    short method to handle all the steps in the SVC
    '''
    crossValDict = makeFolds(feature, label, age, crossVal)
    # outputDict = {}
    trainDict = {}
    testLabelVec = np.array([])
    predLabelVec = np.array([])
    testAgeVec = np.array([])
    trainAgeVec = np.array([])
    # Container for the ROC
    probVec = np.array([])
    rocDict = {}
    keyList = []

    for i, run in enumerate(crossValDict.keys()):
        start = time.time()
        # Alert on running
        print('Running fold ' + str(i))
        # Get the training and test tuples
        trainTuple, testTuple = crossValDict[run]
        # unpack tuples
        (trainFeature, trainLabel, trainAge) = trainTuple
        (testFeature, testLabel, testAge) = testTuple

        # Also store age for comparison
        testAgeVec = np.append(testAgeVec, testAge)
        trainAgeVec = np.append(trainAgeVec, trainAge)

        # Get the best parameters for this training set
        if runParamEst:
            bestC = findParameters(trainFeature, trainLabel, kernel, nCors)
        else:
            bestC = 1.0

        paramStop = time.time()

        # Train model on train data
        print('trainshape: ' + str(trainFeature.shape))
        model = trainModel(trainFeature, trainLabel, kernel, bestC)
        # Get the probability for the ROC
        probas, fpr, tpr = getProbability(model, testFeature, testLabel)
        # Stack fpr and tpr for later use
        rocTuple = (fpr, tpr)
        rocDict[str(run)] = rocTuple
        keyList.append(str(run))

        # Test model on test data
        predictedLabel = testModel(model, testFeature)
        # Test model on train data - for sanity check
        trainPredict = testModel(model, trainFeature)
        trainOut = np.concatenate((trainLabel[..., None],
                                   trainPredict[..., None],
                                   trainAge[..., None]),
                                  axis=1)
        trainDict[run] = trainOut
        print('label: ' + str(len(np.unique(predictedLabel))))

        # Store predicted and true label in the output directory
        testLabelVec = np.append(testLabelVec, testLabel)
        predLabelVec = np.append(predLabelVec, predictedLabel)
        # Also store the probability
        probVec = np.append(probVec, probas)

        # Take time
        stop = time.time()
        elapsedParam = np.round(paramStop - start, 2)
        elapsedFull = np.round(stop - start, 2)

        # outTuple = (testAge, predictedAge)
        # outputDict[run] = outTuple
        if not testLabelVec.shape == predLabelVec.shape:
            print('true and predicted age don\'t match in run ' + str(i) + ':\n'
                  + '    true: ' + str(testLabelVec.shape) + '\n'
                  + '    pred: ' + str(predLabelVec.shape))
        else:
            print('Run ' + str(i) + ':\n'
                  + '    true: ' + str(testLabelVec.shape) + '\n'
                  + '    pred: ' + str(predLabelVec.shape))
        print('    bestC: ' + str(bestC) + '\n'
              + 'parameter selection took: ' + str(elapsedParam) + ' s\n'
              + 'in total took: ' + str(elapsedFull) + ' s')

    # Done with CV, compute mean_tpr
    # Now we have to do an ugly trick to actually fit this. The trick is that
    # the fpr and tpr vector have the be of the same length as the testAgeVec
    # (or any other final vector - actually just nSubs).
    desLength = len(testLabelVec)
    meanFpr = np.linspace(0, 1, desLength)
    meanTpr = 0.0
    numCV = len(keyList)

    for key in keyList:
        (fpr, tpr) = rocDict[key]
        meanTpr += interp(meanFpr, fpr, tpr)
        meanTpr[0] = 0.0

    # divide by number of cv to adjust
    meanTpr /= numCV

    # Done, stack the output together (true label first, then predicted)
    '''
    Columns:
        1) testLabel
        2) predLabel
        3) testAge
        4) probVec for label testLabel = 1
        5) meanFPR
        6) meanTPR
    '''
    outputMatrix = np.concatenate((testLabelVec[..., None],
                                   predLabelVec[..., None],
                                   testAgeVec[..., None],
                                   probVec[..., None],
                                   meanFpr[..., None],
                                   meanTpr[..., None]),
                                  axis=1)

    return outputMatrix, trainDict
    # Done, return the output dictionary
    # return outputDict


def dualPlot(accWithin, accBetween, title):
    '''
    method to plot network results side by side
    '''
    fig, (within, between) = plt.subplots(1, 2, sharex=False, sharey=False)

    within.set_title('within network')
    between.set_title('between network')

    within.hist(accWithin)
    within.set_ylabel('count')
    within.set_xlabel('% correct')

    # betweenCorr, betweenP = st.pearsonr(bTrue, bPred)
    between.hist(accBetween)
    between.set_ylabel('count')
    between.set_xlabel('% correct')

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
    true = predAcc[:, 0]
    pred = predAcc[:, 1]
    age = predAcc[:, 2]
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
    plt.title(title)
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

    connectomeSuffix = '_connectome_glob_corr.txt'

    # Define parameters
    doCV = 'kfold'
    kfold = 10
    nCors = 5
    kernel = 'linear'
    runParamEst = True
    which = 'brain'
    doPlot = True
    what = 'wave'

    childmax = 12.0
    adolescentmax = 18.0

    stratStr = (doCV
                + '_' + str(kfold)
                + '_' + kernel
                + '_' + str(runParamEst)
                + '_' + which
                + '_' + os.path.splitext(connectomeSuffix)[0])

    print(stratStr)

    pathToDumpDir = '/home2/surchs/secondLine/SVC/wave/dos160'
    pathToOutputDir = os.path.join(pathToDumpDir, stratStr)
    # Check if it is there
    if not os.path.isdir(pathToOutputDir):
        print('Making ' + pathToOutputDir + ' now')
        os.makedirs(pathToOutputDir)

    # Basename
    outputBaseName = (stratStr + '_SVC')
    pathToPredictionOutputFile = os.path.join(pathToOutputDir,
                                              (outputBaseName + '.pred'))
    pathToTrainOutputFile = os.path.join(pathToOutputDir,
                                         (outputBaseName + '.train'))

    # Check the fucking paths
    if  (not what in pathToConnectomeDir or
         not what in pathToPhenotypicFile or
         not what in pathToSubjectList or
         not what in pathToRoiMask or
         not what in pathToDumpDir):
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
    labelStack = np.array([], dtype=int)
    meanConnStack = np.array([])
    networkResults = {}
    networkTrainResults = {}
    withinFeature = np.array([])
    betweenFeature = np.array([])

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
            print(subject + ' has nan in the connectome prior norm!')

        # Get the class assignment for the subject
        if phenoAge <= childmax:
            print(subject + ' --> child (' + str(phenoAge) + ')')
            label = 0

        elif phenoAge > childmax and phenoAge <= adolescentmax:
            print(subject + ' --> adolescent (' + str(phenoAge) + ')')
            label = 99
            # Don't use adolescents
            continue

        else:
            print(subject + ' --> adult (' + str(phenoAge) + ')')
            label = 1

        labelStack = np.append(labelStack, label)

        # normalizedConnectome = fisherZ(connectome)
        normalizedConnectome = connectome
        # Check if nan in there
        if np.isnan(normalizedConnectome).any():
            message = (subject + ' has nan in the connectome post norm!')
            raise Exception(message)

        # Get the mean connectivity
        uniqueConnections = getUniqueMatrixElements(normalizedConnectome)
        meanConn = np.mean(uniqueConnections)
        meanConnStack = np.append(meanConnStack, meanConn)

        # Stack the connectome
        connectomeStack = stackConnectome(connectomeStack, normalizedConnectome)
        print('connectomeStack: ' + str(connectomeStack.shape))
        # Stack ages
        ageStack = stackAges(ageStack, phenoAge)

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

    if doCV == 'loocv':
        crossVal = cv.LeaveOneOut(len(labelStack))
        nFolds = crossVal.n

    elif doCV == 'kfold':
        # Prepare the crossvalidation object
        crossVal = cv.StratifiedKFold(labelStack,
                                      kfold)
        nFolds = kfold

        # quick sanity check
        if not nFolds == kfold:
            print('\nThe intended and actual crossvalidation is different:\n'
                  + '    kfold: ' + str(kfold) + '\n'
                  + '    nFold: ' + str(nFolds) + '\n'
                  + 'with ' + str(len(phenoSubjects)) + ' subjects\n')

    # Now we have the connectome stack
    # Let's loop through the networks again
    for i, network in enumerate(networkNodes.keys()):
        if which == 'brain':
            # Leave it alone
            print('Looking for whole brain here...')
            continue

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
        if which == 'network':
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

        if which == 'network':
            # Run SVR
            print('\nRunning within ' + network + ' connectivity SVC ('
                  + str(i) + '/' + str(len(networkNodes.keys())) + ')')
            withinResult, withinTrainDict = mainSVC(withinFeature,
                                                    labelStack,
                                                    ageStack,
                                                    crossVal,
                                                    kernel,
                                                    nCors,
                                                    runParamEst)
            print('\nRunning between ' + network + ' connectivity SVC ('
                  + str(i) + '/' + str(len(networkNodes.keys())) + ')')
            betweenResult, betweenTrainDict = mainSVC(betweenFeature,
                                                      labelStack,
                                                      ageStack,
                                                      crossVal,
                                                      kernel,
                                                      nCors,
                                                      runParamEst)

            # Store the output in the output Dictionary for networks
            result = (withinResult, betweenResult)
            networkResults[network] = result
            trainResult = (withinTrainDict, betweenTrainDict)
            networkTrainResults[network] = trainResult

    if which == 'mean':
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
        withinResult, withinTrainDict = mainSVC(withinFeature,
                                                labelStack,
                                                ageStack,
                                                crossVal,
                                                kernel,
                                                nCors,
                                                runParamEst)
        betweenResult, betweenTrainDict = mainSVC(betweenFeature,
                                                  labelStack,
                                                  ageStack,
                                                  crossVal,
                                                  kernel,
                                                  nCors,
                                                  runParamEst)

        testSaveTuple = (withinResult, betweenResult)
        trainSaveTuple = (withinTrainDict, betweenTrainDict)
        status = saveOutput(pathToPredictionOutputFile, testSaveTuple)
        status = saveOutput(pathToTrainOutputFile, trainSaveTuple)

        if doPlot:
            print('Plotting what we actually wanted...\n'
                  + stratStr)
            # Plot mean connectivity across age
            plt.plot(ageStack, withinFeature, 'g.')
            plt.title('within mean connectivity')
            plt.show()
            raw_input('hallo...')
            plt.close()

            plt.plot(ageStack, betweenFeature, 'g.')
            plt.title('between mean connectivity')
            plt.show()
            raw_input('hallo...')
            plt.close()

            dualPlot(withinResult, betweenResult, 'within and between connectivit'
                     + ' predicting age')
            trainPlot(withinTrainDict, betweenTrainDict)

    if which == 'brain':
        print('Lets do the brain!\n'
              + stratStr)
        mask = np.ones_like(connectomeStack[..., 0])
        mask = np.tril(mask, -1)
        feature = connectomeStack[mask == 1].T
        print('Feature: ' + str(feature.shape))
        result, trainDict = mainSVC(feature,
                                    labelStack,
                                    ageStack,
                                    crossVal,
                                    kernel,
                                    nCors,
                                    runParamEst)

        status = saveOutput(pathToPredictionOutputFile, result)
        print(status)
        status = saveOutput(pathToTrainOutputFile, trainDict)
        print(status)

        if doPlot:
            singlePlot(result, 'whole brain SVC plot')
            accBrain = trainPlot(trainDict)
            print('Prediction accuracy for the whole brain on training:\n'
                  + '    acc: ' + str(np.average(accBrain)))

    elif which == 'network':

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
                accWithin, accBetween = trainPlot(withinTrainDict, betweenTrainDict)

                '''
                print('Prediction accuracy for ' + network + ':\n'
                      + '    within: ' + str(accWithin) + '\n'
                      + '    between: ' + str(accBetween))
                '''
                # dualPlot(accWithin, accBetween, network)


if __name__ == '__main__':
    Main()
