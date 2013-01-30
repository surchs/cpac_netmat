'''
Created on Jan 25, 2013

@author: sebastian

compared to loadHousing, this runs on the dataset that ships with sklearn
I don't think it will make any difference but I can use it to test if there is
anything going wrong with the way I create the subjects

also turn this into a wrapper to actually run the Network afterwards
'''
import sys
import gzip
import random
import cPickle
import numpy as np
from sklearn import svm
import sklearn.grid_search as gs
from matplotlib import pyplot as plt
import cpac_netmat.analysis.base as an
from sklearn.datasets import load_boston
from sklearn.datasets import load_diabetes
import cpac_netmat.preprocessing.base as pp
from matplotlib.backends.backend_pdf import PdfPages as pdf


def runModel(trainFeatures, trainLabels, testFeatures, testLabels, kernel,
             gridCv, epsilon):
    '''
    Method that runs the model for the other functions
    '''
    expArrayOne = np.arange(-6, 6, 1)
    epsExp = np.arange(-10, 0, 1)
    baseArray = np.ones_like(expArrayOne, dtype='float32') * 10
    epsBase = np.ones_like(epsExp, dtype='float32') * 10
    parameterOne = np.power(baseArray, expArrayOne).tolist()
    epsParameterOne = np.power(epsBase, epsExp).tolist()

    # if for some reason this doesn't work, just paste directly
    parameters = {'C': parameterOne, 'epsilon': epsParameterOne}
    gridModel = svm.SVR(kernel=kernel, epsilon=epsilon, degree=2)
    firstTrainModel = gs.GridSearchCV(gridModel,
                                      parameters,
                                      cv=gridCv,
                                      n_jobs=1,
                                      verbose=0)
    
    firstTrainModel.fit(trainFeatures, trainLabels)

    firstPassC = firstTrainModel.best_estimator_.C
    firstPassE = firstTrainModel.best_estimator_.epsilon

    expFirstC = np.log10(firstPassC)
    expArrayTwo = np.arange(expFirstC - 1, expFirstC + 1.1, 0.1)
    baseArrayTwo = np.ones_like(expArrayTwo, dtype='float32') * 10
    expFirstE = np.log10(firstPassE)
    expArrayTwoE = np.arange(expFirstE - 1, expFirstE + 1.1, 0.1)
    baseArrayTwoE = np.ones_like(expArrayTwoE, dtype='float32') * 10
    
    parameterTwo = np.power(baseArrayTwo, expArrayTwo).tolist()
    parameterTwoE = np.power(baseArrayTwoE, expArrayTwoE).tolist()

    # in case this causes trouble, paste directly
    parameters = {'C': parameterTwo, 'epsilon':parameterTwoE}

    secondTrainModel = gs.GridSearchCV(gridModel,
                                       parameters,
                                       cv=gridCv,
                                       n_jobs=1,
                                       verbose=0)
    
    secondTrainModel.fit(trainFeatures, trainLabels)
    bestC = secondTrainModel.best_estimator_.C
    bestE = secondTrainModel.best_estimator_.epsilon

    gridC = bestC
    gridE = bestE

    # now train the model
    trainModel = svm.SVR(kernel=kernel, C=gridC, epsilon=gridE)
    trainModel.fit(trainFeatures, trainLabels)
    # and predict the labels from the test features
    # tempPredLabels = trainModel.predict(testFeatures)
    tempPredLabels = trainModel.predict(testFeatures)
    # get the errors
    tempErrors = tempPredLabels - testLabels

    return (tempPredLabels, testLabels, tempErrors, gridC)


def runShitTheOldWay(features, labels, cvObject, cDict):
    '''
    Method to run stuff in my old classes
    '''

    numSubs = len(labels)
    subDir = {}

    subCount = 1
    for subCount in np.arange(numSubs):
        # make a new subject
        subName = ('case_' + str(subCount))
        tempSub = pp.Subject(subName, '_test')
        tempSub.pheno = {}
        tempSub.pheno['houseprice'] = labels[subCount]
        tempSub.feature = features[subCount, ...]
        subDir[subName] = tempSub
        subCount += 1

    # now make a network of it and run that stuff
    testNetwork = an.Network('test', cvObject)
    testNetwork.subjects = subDir

    testNetwork.pheno = cDict['pheno']
    testNetwork.featureSelect = cDict['fs']
    testNetwork.cValue = cDict['cValue']
    testNetwork.eValue = cDict['eValue']
    testNetwork.kernel = cDict['kernel']
    testNetwork.numberCores = cDict['numberCores']
    testNetwork.gridCv = cDict['gridCv']
    testNetwork.gridCores = 1

    testNetwork.makeRuns()
    # now run the runs
    testNetwork.executeRuns()

    predictedPheno = testNetwork.predictedPheno
    truePheno = testNetwork.truePheno

    return (predictedPheno, truePheno)


def runShitButNotAll(features, labels, cvObject, cDict):
    '''
    just prepare everything in my classes, then run here
    '''
    print('Dudedudedude')

    numSubs = len(labels)
    subDir = {}

    subCount = 1
    for subCount in np.arange(numSubs):
        # make a new subject
        subName = ('case_' + str(subCount))
        tempSub = pp.Subject(subName, '_test')
        tempSub.pheno = {}
        tempSub.pheno['houseprice'] = labels[subCount]
        tempSub.feature = features[subCount, ...]
        subDir[subName] = tempSub
        subCount += 1

    # now make a network of it and run that stuff
    testNetwork = an.Network('test', cvObject)
    testNetwork.subjects = subDir

    testNetwork.pheno = cDict['pheno']
    testNetwork.featureSelect = cDict['fs']
    testNetwork.cValue = cDict['cValue']
    testNetwork.eValue = cDict['eValue']
    testNetwork.kernel = cDict['kernel']
    testNetwork.numberCores = cDict['numberCores']
    testNetwork.gridCv = cDict['gridCv']
    testNetwork.gridCores = 1
    
    # and now prepare the network
    testNetwork.makeRuns()

    # now instead of running it inside the classes, run here
    trueLabels = np.array([])
    predictedLabels = np.array([])
    cValues = np.array([])
    errors = np.array([])
    for run in testNetwork.runs.values():
        trainFeatures = run.trainFeature
        testFeatures = run.testFeature
        trainLabels = run.trainPheno
        testLabels = run.testPheno

        # now run a grid search to find the optimal parameters
        kernel = testNetwork.kernel
        epsilon = testNetwork.eValue
        gridCv = testNetwork.gridCv

        (tempPredLabels,
         testLabels,
         tempErrors,
         gridC) = runModel(trainFeatures, trainLabels, testFeatures,
                           testLabels, kernel, gridCv, epsilon)
        # and append the whole stuff to the mats
        predictedLabels = np.append(predictedLabels, tempPredLabels)
        trueLabels = np.append(trueLabels, testLabels)
        errors = np.append(errors, tempErrors)
        cValues = np.append(cValues, gridC)

    return (predictedLabels, trueLabels, errors, cValues)


def runShitHereYourself(features, labels, cDict):
    '''
    replicate the classes function in here to see what happens
    '''
    index = np.arange(len(labels))
    random.shuffle(index)
    lastSubs = 0
    first = 0

    # load parameters
    epsilon = cDict['eValue']
    kernel = cDict['kernel']
    gridCv = cDict['gridCv']

    # prepare storage
    trueLabels = np.array([])
    predictedLabels = np.array([])
    cValues = np.array([])
    errors = np.array([])

    for i in range(10):
        if not i == 9:
            tNumSubs = np.floor(len(labels) / 10)
            minSubs = lastSubs + 1 * first
            maxSubs = minSubs + tNumSubs
            lastSubs = maxSubs
            first = 1
        else:
            minSubs = lastSubs
            maxSubs = len(labels)
        testIndex = index[minSubs:maxSubs]
        trainIndex = index[:minSubs]
        trainIndex = np.append(trainIndex, index[maxSubs:])
        print(str(i) + ' ' + str(minSubs) + '-' + str(maxSubs) + ' : '
              + str(len(testIndex)) + '/' + str(len(trainIndex)))

        # now extract features and labels and get cooking
        trainFeatures = features[trainIndex, ...]
        trainLabels = labels[trainIndex]
        testFeatures = features[testIndex, ...]
        testLabels = labels[testIndex]
        # and run the model
        (tempPredLabels,
         testLabels,
         tempErrors,
         gridC) = runModel(trainFeatures, trainLabels, testFeatures,
                           testLabels, kernel, gridCv, epsilon)

        # and append the whole stuff to the mats
        predictedLabels = np.append(predictedLabels, tempPredLabels)
        trueLabels = np.append(trueLabels, testLabels)
        errors = np.append(errors, tempErrors)
        cValues = np.append(cValues, gridC)

    return (predictedLabels, trueLabels, errors, cValues)


def runShitOnCv(features, labels, cvObject, cDict):
    '''
    method to use the CV object to get training and testing data but otherwise
    run my own shit
    '''
    # load parameters
    epsilon = cDict['eValue']
    kernel = cDict['kernel']
    gridCv = cDict['gridCv']

    # prepare storage
    trueLabels = np.array([])
    predictedLabels = np.array([])
    cValues = np.array([])
    errors = np.array([])

    for cvInstance in cvObject:
        # create a new instance of the run class
        train = cvInstance[0]
        test = cvInstance[1]

        # make this into features
        trainFeatures = features[train, ...]
        trainLabels = labels[train]
        testFeatures = features[test, ...]
        testLabels = labels[test]

        (tempPredLabels,
         testLabels,
         tempErrors,
         gridC) = runModel(trainFeatures, trainLabels, testFeatures,
                           testLabels, kernel, gridCv, epsilon)

        # and append the whole stuff to the mats
        predictedLabels = np.append(predictedLabels, tempPredLabels)
        print('runshitoncv: ' + str(predictedLabels.shape))
        trueLabels = np.append(trueLabels, testLabels)
        errors = np.append(errors, tempErrors)
        cValues = np.append(cValues, gridC)
   
    return (predictedLabels, trueLabels, errors, cValues)


def runShitNoCv(features, labels, cDict):
    '''
    replicate the classes function in here to see what happens
    '''
    index = np.arange(len(labels))
    random.shuffle(index)
    trainSubs = np.floor(len(labels) / 2)
    train = index[:trainSubs]
    test = index[trainSubs:]
    
    print('numTrain: ' + str(len(train)) + ' numTest: ' + str(len(test)))

    # load parameters
    epsilon = cDict['eValue']
    kernel = cDict['kernel']
    gridCv = cDict['gridCv']

    # now extract features and labels and get cooking
    trainFeatures = features[train, ...]
    trainLabels = labels[train]
    testFeatures = features[test, ...]
    testLabels = labels[test]
    # and run the model
    (predictedLabels,
     trueLabels,
     errors,
     gridC) = runModel(trainFeatures, trainLabels, testFeatures,
                       testLabels, kernel, gridCv, epsilon)

    return (predictedLabels, trueLabels, errors, gridC)


def runShitClean(features, labels):
    '''
    module that simply runs a split-half model with standard parameters
    '''
    index = np.arange(len(labels))
    random.shuffle(index)
    trainSubs = np.floor(len(labels) / 2)
    train = index[:trainSubs]
    test = index[trainSubs:]
    
    # now extract features and labels and get cooking
    trainFeatures = features[train, ...]
    trainLabels = labels[train]
    testFeatures = features[test, ...]
    testLabels = labels[test]
    
    model = svm.SVR(kernel='linear', C=1, epsilon=0.2)
    model.fit(trainFeatures, trainLabels)
    predictedLabels = model.predict(testFeatures)
    errors = predictedLabels - testLabels
    dummy = 1
    
    return (predictedLabels, testLabels, errors, dummy)


def Main(outFile, pdfFile, strategy):
    '''
    Get the data from sklearn

    there are now three strategies:
        1) do everything with my classes and see the result
        2) do everything with my classes but run the runs separately
        3) replicate what my classes do without my classes
    '''
    print('\n\nHello there, welcome to testing things. These are our params:'
          + '\nstrategy:' + str(strategy) + ' / outFile:' + outFile + ' / pdfFile:'
          + pdfFile)
    print('Not happy with it? Probably your fault! Enjoy!\n')
    dataset = load_boston()
    # dataset = load_diabetes()
    features = dataset.data
    labels = dataset.target
    numberCases = len(labels)
    stSt = {}

    cDict = {}
    cDict['pheno'] = 'houseprice'
    cDict['fs'] = 'None'
    cDict['cValue'] = 1000
    cDict['eValue'] = 0.001
    cDict['kernel'] = 'rbf'
    cDict['numberCores'] = 10
    cDict['gridCv'] = 5

    cvObject = an.cv.KFold(numberCases, 10, shuffle=True)

    # now see if we only run one or multiple
    if strategy == None:
        stSt['oldway'] = runShitTheOldWay(features, labels, cvObject, cDict)
        stSt['ownTrain'] = runShitButNotAll(features, labels, cvObject, cDict)
        stSt['manualCv'] = runShitHereYourself(features, labels, cDict)
        stSt['CvOwnTrain'] = runShitOnCv(features, labels, cvObject, cDict)
        stSt['noCv'] = runShitNoCv(features, labels, cDict)
        stSt['clean'] = runShitClean(features, labels)
        # now save the result
        outF = gzip.open(outFile, 'wb')
        cPickle.dump(stSt, outF, protocol=2)

        # and show the results
        for result in stSt.keys():
            if not result == 'oldway':
                (pPheno,
                 tPheno,
                 errors,
                 cValues) = stSt[result]
            else:
                (pPheno,
                 tPheno) = stSt[result]
                 
            # tell a bit about the data
            print('Plotting ' + result)
            #print('    pPheno: ' + str(pPheno.shape) + '/' + str(np.max(pPheno)))
            print('    tPheno ' + str(tPheno.shape))
            print('    tPheno ' + str(np.max(tPheno)))
            fig4 = plt.figure(4, figsize=(8.5, 11), dpi=150)
            fig4.suptitle('predicted over true age')

            tSP4 = fig4.add_subplot(111, title=result)
            tSP4.plot(tPheno, tPheno)
            tSP4.plot(tPheno, pPheno, 'co')

            fig4.subplots_adjust(hspace=0.5, wspace=0.5)
            pdfFileName = (pdfFile + '_' + result + '.pdf')
            pd = pdf(pdfFileName)
            pd.savefig(fig4)
            pd.close()
            plt.close(4)
            print('Just created ' + pdfFile + '\nAll done here!')
    else:
        if strategy == 'old':
            (pPheno,
             tPheno) = runShitTheOldWay(features, labels, cvObject, cDict)
        elif strategy == 'own':
            (pPheno,
             tPheno,
             errors,
             cValues) = runShitButNotAll(features, labels, cvObject, cDict)
        elif strategy == 'manual':
            (pPheno,
             tPheno,
             errors,
             cValues) = runShitHereYourself(features, labels, cDict)
        elif strategy == 'cv':
            (pPheno,
             tPheno,
             errors,
             cValues) = runShitOnCv(features, labels, cvObject, cDict)
        elif strategy == 'nocv':
            (pPheno,
             tPheno,
             errors,
             cValues) = runShitNoCv(features, labels, cDict)
        elif strategy == 'clean':
            (pPheno,
             tPheno,
             errors,
             cValues) = runShitClean(features, labels)
        else:
            print('Bullshit arguments!')

        # tell a bit about the data
        print('Plotting ' + strategy)
        print('    pPheno: ' + str(pPheno.shape) + '/' + str(np.max(pPheno)))
        print('    tPheno' + str(tPheno.shape) + '/' + str(np.max(tPheno)))
        
        # and now display the stuff
        fig4 = plt.figure(4, figsize=(8.5, 11), dpi=150)
        fig4.suptitle('predicted over true age')

        tSP4 = fig4.add_subplot(111, title=strategy)
        tSP4.plot(tPheno, tPheno)
        tSP4.plot(tPheno, pPheno, 'co')

        fig4.subplots_adjust(hspace=0.5, wspace=0.5)
        pdfFileName = (pdfFile + '_strategy_' + strategy + '.pdf')
        pd = pdf(pdfFileName)
        pd.savefig(fig4)
        pd.close()
        plt.close(4)
        print('Just created ' + pdfFile + '\nAll done here!')

if __name__ == '__main__':
    outFile = sys.argv[1]
    pdfFile = sys.argv[2]
    if len(sys.argv) < 4:
        strategy = None
    else:
        strategy = sys.argv[3]
    Main(outFile, pdfFile, strategy)
    pass
