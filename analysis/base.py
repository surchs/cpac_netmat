'''
Created on Nov 7, 2012

@author: sebastian

classes for the analysis
'''
import os
import re
import sys
import time
import glob
import gzip
import copy
import cPickle
import numpy as np
from sklearn import svm
import multiprocessing as mp
import sklearn.grid_search as gs
import sklearn.cross_validation as cv
import sklearn.feature_selection as fs
from sklearn.metrics import mean_squared_error
from cpac_netmat.tools import meisterlein as mm


class Study(object):
    '''
    class to contain a full set of analyses to investigate. parameters, logs
    and results are stored here so the whole object can be saved to disk for
    later use
    '''

    def __init__(self, name, dataPath, subjectList=None, numberCores=8):
        '''
        Constructor method
        '''
        self.name = name
        self.dataPath = dataPath
        self.subjectList = subjectList
        self.numberCores = numberCores
        # parameters to be assigned later
        self.subjectPaths = []
        self.analyses = {}
        self.masks = {}
        self.maskedSubjects = {}

    def makeSubjectPaths(self):
        '''
        Method to generate the paths to the subject files.
        This either reads in all the directory/subjectfiles.sub in a given
        directory (self.dataPath) or only those given by a subject list
        (self.subjectList)

        The method then checks what masks a subject is associated with and
        then loads both the subject and the mask
        '''

        # check the parameters first
        if not os.path.isdir(self.dataPath):
            print('subjectPath ' + self.dataPath + ' doesn\'t exist')

        if self.subjectList == None:
            # no subjects specified, take all you can get in the dir
            print('Using all subjects in ' + os.path.abspath(self.dataPath))
            # here we can set the template for the subject directory
            if self.dataPath.endswith('/'):
                self.subjectPaths = glob.glob((self.dataPath + '*/*.sub'))
            else:
                self.subjectPaths = glob.glob((self.dataPath + '/*/*.sub'))

        else:
            # subject file specified, take only the specified ones
            # check whether the specified sb
            subjectFile = open(self.subjectList, 'rb')
            subjectFileList = subjectFile.readlines()

            for line in subjectFileList:
                # currently only one column expected that contains the subject
                # name
                tempSubName = line.strip()
                tempSubPath = os.path.join(os.path.abspath(self.dataPath),
                                           tempSubName)
                # check if the subject path exists
                if not os.path.isdir(tempSubPath):
                    print(tempSubName + ' is not in ' + self.dataPath)
                    continue

                tempSubjects = glob.glob((tempSubPath + '/*.sub'))
                # loop through the results
                for tempSubject in tempSubjects:
                    self.subjectPaths.append(os.path.abspath(tempSubject))

        print('Finished loading ' + str(len(self.subjectPaths)) + ' subjects')

    def getSubjects(self):
        '''
        Method to get the subjects that are listed in the subject paths.

        Now that we can store multiple masks inside each subject it is no
        longer necessary to check for the masks
        '''
        problemString = 'These were the subjects that caused problems:'
        problemList = []
        run = 0
        for subjectPath in self.subjectPaths:
            # open the file
            tempSubFile = gzip.open(subjectPath, 'rb')
            tempSubject = cPickle.load(tempSubFile)
            tempSubName = tempSubject.name

            # TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP #
            # this is a temporary solution to change the type of the pheno
            # TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP #
            for pheno in tempSubject.pheno.keys():
                tempPheno = tempSubject.pheno[pheno]
                if mm.isNumber(tempPheno):
                    tempPheno = float(tempPheno)

                tempSubject.pheno[pheno] = tempPheno
            # TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP #
            # this was a temporary solution to change the type of the pheno
            # TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP #

            # get the masks in the subject
            for tempSubMask in tempSubject.masks.values():
                tempMaskName = tempSubMask.name
                # check if the mask name already exists in the saved masks
                if tempMaskName in self.masks.keys():
                    # it is already in, compare it to the saved mask
                    if not self.masks[tempMaskName].sameSame(tempSubMask):
                        # this mask is different from the one we already saved
                        # and this is a problem - alert and skip the subject
                        # (not optimal, I know - its an order thing...
                        # fix later)
                        print('The mask: ' + tempMaskName + ' of subject '
                              + tempSubName + ' is different from the saved'
                              + ' mask in our repository')

                        if not tempSubName in problemList:
                            problemList.append(tempSubName)
                            problemString = (problemString
                                         + '\n' + tempSubName)
                        continue

                else:
                    # the mask is not saved yet, so do this now
                    self.masks[tempMaskName] = tempSubMask

                # now we can continue processing the subjects
                # check if there is already a list of subjects for this mask
                if not tempMaskName in self.maskedSubjects.keys():
                    # create the entry and make it a dictionary
                    self.maskedSubjects[tempMaskName] = {}

                # now get rid of the mask and append the subject to the
                # correct list
                tempSubject.mask = None
                self.maskedSubjects[tempMaskName][tempSubName] = tempSubject

            # done with the subject, print a quick notice
            run += 1
            sys.stdout.write('\rDone loading subject ' + tempSubName
                             + ' : ' + str(run) + '/'
                             + str(len(self.subjectPaths)) + '           ')
            sys.stdout.flush()

        # done with the processing, tell the world about it and give a summary
        print('\n\nDone with fetching subjects'
              + '\nwe have ' + str(len(self.maskedSubjects.keys())) + ' masks')
        maskString = 'These are the masks we have:'
        for mask in self.maskedSubjects.keys():
            maskString = (maskString
                          + '\n    ' + mask + ' : '
                          + str(len(self.maskedSubjects[mask].keys()))
                          + ' subjects')
        print(maskString)
        if len(problemList) > 0:
            print('There were ' + str(len(problemString)) + ' subjects with'
                  + ' problems')
            print(problemString)
        else:
            print('No subjects had any problems')

    def getAnalyses(self, configFile):
        '''
        Method that loads a config file line by line according to a fixed
        pattern and creates analyses on the fly for each line in the file.

        Currently the parameters are:
            1) analysis name
            2) derivative to run on
            3) mask to run on
            4) mode of feature selection
            5) mode of crossvalidation over subjects
                1  - LOOCV
                >1 - k-Fold according to number
            6) what feature set to use
                - within: only connections within the network
                - between: only connections of network nodes with outside nodes
                - whole: all connections of network nodes (within + between)
            7) what kernel to use
            8) C value to begin with in grid search for SVM model
            9) E value to use in SVM model
            10) number of crossvalidation for grid search in run
            11) number of cores to use per run
            12) max features to use
        '''

        self.configFile = open(configFile, 'rb')
        # set the file counter to 0
        self.configFile.seek(0)
        tempConfigLines = self.configFile.readlines()

        # loop over the lines
        for line in tempConfigLines:
            configLine = line.strip().split(',')
            if '#' in line or re.match('^\n', line):
                # this is a comment, keep going
                continue

            analysisName = configLine[1]
            derivative = configLine[2]
            mask = configLine[3]
            pheno = configLine[4]
            featureSelect = configLine[5]
            crossvalidate = int(configLine[6])
            featureFocus = configLine[7]
            kernel = configLine[8]
            cValue = float(configLine[9])
            eValue = float(configLine[10])
            gridCv = int(configLine[11])
            runCores = int(configLine[12])
            maxFeat = int(configLine[13])

            # check if the desired mask is present in the Study Object
            if not mask in self.masks.keys():
                # the mask is not here, alert
                print('\nThe mask ' + mask + ' for analysis ' + analysisName
                      + ' is not available in the study object ' + self.name)

            # get the mask ready
            analysisMask = self.masks[mask]

            # make an analysis object and set parameters
            tempAnalysis = Analysis(analysisName, analysisMask)

            tempAnalysis.derivative = derivative
            tempAnalysis.featureSelect = featureSelect
            tempAnalysis.pheno = pheno
            tempAnalysis.crossvalidate = crossvalidate
            tempAnalysis.featureFocus = featureFocus
            tempAnalysis.kernel = kernel
            tempAnalysis.cValue = cValue
            tempAnalysis.eValue = eValue
            tempAnalysis.gridCv = gridCv
            tempAnalysis.numberCores = self.numberCores
            tempAnalysis.runCores = runCores
            tempAnalysis.maxFeat = maxFeat
            # get the subjects and see if any don't have the derivative
            tempSubs = {}
            for subject in self.maskedSubjects[mask].keys():
                # copy the original to make it independent from changes
                tempSub = copy.copy(self.maskedSubjects[mask][subject])
                tempDerivatives = tempSub.derivativeMasks[mask]
                if not derivative in tempDerivatives.keys():
                    # something is wrong with this subject
                    print(subject + ' doesn\'t have derivative ' + derivative)
                    print('it would have these:')
                    for derivative in tempDerivatives.keys():
                        print(derivative)
                    continue
                else:
                    # nothing wrong with this one - assign the correct
                    # derivative and delete the rest
                    tempSub.derivative = tempDerivatives[derivative]
                    tempSub.derivativeMasks[mask] = None
                    tempSubs[subject] = tempSub

                    tempAnalysis.subjects[subject] = tempSub
            # and store the object in the dictionary
            self.analyses[tempAnalysis.name] = tempAnalysis
        # Done creating analyses
        print('Done creating all the analyses')
        aString = 'Here are all the analyses we presently have in the study:'
        for analysis in self.analyses.keys():
            tempAnalysis = self.analyses[analysis]
            aString = (aString +
                       '\n    ' + tempAnalysis.name + ' : '
                       + str(len(tempAnalysis.subjects.keys())) + ' subjects')
        print(aString)


class Analysis(object):
    '''
    class to contain one full analysis. in the current design, one analysis is
    one combination of subjects, parameters and derivative only. So different
    sets of subjects, derivatives or parameters will be represented as
    different analyses
    '''

    def __init__(self, name, mask):
        self.name = name
        # parameters to be determined later
        self.derivative = None
        self.featureSelect = None
        self.pheno = None
        self.crossvalidate = None
        self.featureFocus = None
        self.cvObject = None
        self.kernel = None
        self.subjects = {}
        self.mask = mask
        self.cValue = None
        self.eValue = None
        self.gridCv = None
        self.numberCores = None
        self.runCores = None
        self.maxFeat = None
        self.networks = {}
        self.cvObject = None

    def makeCrossvalidate(self):
        if self.crossvalidate == 1:
            # make a LOOCV object
            self.cvObject = cv.LeaveOneOut(len(self.subjects.keys()))
        elif self.crossvalidate > 1:
            if self.crossvalidate <= len(self.subjects.keys()) - 1:
                # make a KFold object
                self.cvObject = cv.KFold(len(self.subjects.keys()),
                                         self.crossvalidate,
                                         shuffle=True)
            else:
                print('You requested more folds (' + str(self.crossvalidate)
                      + ') than is the number of subjects ('
                      + str(len(self.subjects.keys()))
                      + ') and this won\'t work')
        else:
            print('Something about your crossvalidation option ('
                  + str(self.crossvalidate) + ') is not right. Please check!')

    def prepareNetworks(self):
        '''
        Method to loop through the network/s and get the appropriate
        feature vector of the derivative for each one of them.

        currently gives me three variables:
            - tempWithin - features inside the network nodes
            - tempBetween - features outside of the network (but still with
                            respect to the current network for connectivity)
            - tempWhole - tempWithin + tempBetween

        choosing which of the two to use can be done in the analysis script or
        maybe we just run both by default, don't know
        '''
        # loop through the networkNodes inside the mask
        for network in self.mask.networkNodes.keys():
            # create a network object
            tempNetwork = Network(network, self.cvObject)
            # and now loop through the subjects
            for subject in self.subjects.keys():
                tempSub = self.subjects[subject]
                tempDer = tempSub.derivative.feature
                tempInd = self.mask.networkIndices[network]
                tempFeat = {}
                # see if it is a matrix or vector
                if len(tempDer.shape) == 1:
                    # it's a vector we have to cut it up into pieces
                    tempWithin = tempDer.feature[tempInd]
                    tempBetween = np.delete(tempDer.feature[network], tempInd)
                    tempWhole = tempDer.feature

                elif len(tempDer.shape) == 2:
                    # it's a matrix - this shit is more difficult
                    # first get the rows belonging to the network
                    tempNet = tempDer[tempInd, ...]
                    # then get the matrix belonging to the within features
                    tempWithinNet = tempNet[..., tempInd]
                    # and now only take the lower triangle
                    tempMask = np.ones_like(tempWithinNet)
                    tempMask = np.tril(tempMask, -1)
                    # and put it in the variable
                    tempWithin = tempWithinNet[tempMask == 1]

                    # now for between - delete indices along 1st axis
                    tempBetweenNet = np.delete(tempNet, tempInd, 1)
                    # now stretch it out
                    tempBetween = np.reshape(tempBetweenNet,
                                             tempBetweenNet.size)
                    # and lastly for the whole connectivity
                    tempWhole = np.append(tempWithin, tempBetween)

                else:
                    # something wrong, better don't use this subject
                    print('There is something wrong in analysis '
                          + self.name + ' with subject ' + subject)
                    continue

                tempFeat['within'] = tempWithin
                tempFeat['between'] = tempBetween
                tempFeat['whole'] = tempWhole
                # write the result back into the subject
                # use a new attribute - self.feature
                # depending on the feature focus
                if self.featureFocus:
                    tempSub.feature = tempFeat[self.featureFocus]
                else:
                    tempSub.feature = tempFeat['whole']

                # and write the stuff to the network
                tempNetwork.subjects[subject] = tempSub

            # edit the network parameters
            tempNetwork.gridCv = self.gridCv
            tempNetwork.maxFeat = self.maxFeat
            tempNetwork.numberCores = self.numberCores
            tempNetwork.runCores = self.runCores
            tempNetwork.kernel = self.kernel
            tempNetwork.C = self.cValue
            tempNetwork.E = self.eValue
            tempNetwork.pheno = self.pheno

            # and save the network object to the analysis
            self.networks[network] = tempNetwork


class Network(object):
    '''
    class to contain one sub-analyses of a specific network. if no networkNodes
    are used in the analysis, then this is a unique object. otherwise, each
    inspected network is represented as a network object
    '''

    def __init__(self, name, cvObject):
        self.name = name
        self.cvObject = cvObject
        self.subjects = {}
        self.runs = {}
        # parameters for the runs
        self.E = None
        self.C = None
        self.pheno = None
        self.gridCv = None
        self.maxFeat = None
        self.numberCores = None
        self.runCores = None
        self.kernel = None
        self.cValue = None
        self.eValue = None

    def makeRuns(self):
        '''
        Method to create and store run objects for later execution

        since this takes some time, I will put a print option out there
        '''
        runID = 1
        nFolds = self.cvObject.k
        for cvInstance in self.cvObject:
            # create a new instance of the fold class
            run = Run(str(runID))
            run.kernel = self.kernel
            train = cvInstance[0]
            test = cvInstance[1]
            # now we loop over these
            tempTrainSubs = {}
            for subId in train:
                # get the correct subjects and put them in a temporary dict
                subName = self.subjects.keys()[subId]
                subject = self.subjects[subName]
                # store the feature in the feature array of the fold
                tempTrainSubs[subName] = subject

            tempTestSubs = {}
            for subId in test:
                # get the correct subjects and put them in a temporary dict
                subName = self.subjects.keys()[subId]
                subject = self.subjects[subName]
                # store the feature in the feature array of the fold
                tempTestSubs[subName] = subject

            # hand over the information to the run
            run.train = tempTrainSubs
            run.test = tempTestSubs

            # set the necessary parameters
            run.maxFeat = self.maxFeat
            run.runCores = self.runCores
            run.runCv = self.gridCv

            # model parameters
            run.kernel = self.kernel
            run.C = self.cValue
            run.E = self.eValue
            run.pheno = self.pheno

            # run the prepare run method
            run.prepareRun()
            # store the run in the Network object
            self.runs[run.number] = run

            # print that we are done with the run
            sys.stdout.write('\rDone creating run ' + str(runID) + '/'
                             + str(nFolds) + ' for network ' + self.name)
            sys.stdout.flush()
            # +1 on the run ID
            runID += 1

    def runRun(self, run):
        '''
        Method to run an individual run
        '''
        run.selectFeatures()
        run.selectParameters()
        run.trainModel()
        run.testModel()
        run.getError()

        return run

    def executeRuns(self):
        '''
        Method to execute the previously generated runs in a parallel fashion
        '''
        # first see how many runs we have and how many cores we may use so
        # we don't exceed with the cores per run
        parallelRuns = np.floor(self.numberCores / self.runCores)

        start = time.time()
        pool = mp.Pool(processes=parallelRuns)
        resultList = pool.map(self.runRun, self.runs.values())
        stop = time.time()
        elapsed = stop - start

        print('Running Network ' + self.name + ' is done. This took '
              + str(elapsed) + ' seconds')

        # map back the results
        for run in resultList:
            # loop through the shit
            self.runs[run.name] = run


class Run(object):
    '''
    class to contain one single run of crossvalidation. depending on parameters
    different strategies of feature selection, grid search, training and
    error calculation can be exectued.
    '''

    def __init__(self, number):
        '''
        Constructor method.
        Variables that have to be set:
            - self.kernel - kernel
            - self.C - C parameter for SVM model
            - self.E - epsilon parameter for SVM model
            - self.cv - number of crossvalidation runs
            - feature selection
            - grid search start parameters
            - self.maxFeat - max features for rcrfe
            - self.pheno - phenotypic information to run on
            - self.runCores - number of Cores to run in gridSearch per run
        '''
        self.number = number
        # have to be set before running
        # input data
        self.train = None
        self.test = None

        # processing parameters
        self.maxFeat = 2000
        self.runCores = 2
        self.runCv = 5
        self.pheno = None

        # model parameters
        self.kernel = None
        self.C = None
        self.E = 0.000001

        # are set by running prepareRun()
        # data generated on the run
        self.trainSubs = None
        self.testSubs = None
        self.trainFeature = None
        self.testFeature = None
        self.trainPheno = None
        self.testPheno = None

        # are result of running the model
        # output data
        self.predictPheno = None
        self.featureIndex = None
        self.error = None

    def prepareRun(self):
        '''
        Method to generate the appropriate training and testing matrices
        '''
        # prepare general list of subjects
        self.trainSubs = self.train.keys()
        self.trainSubs.sort()
        self.testSubs = self.test.keys()
        self.testSubs.sort()

        # train first, prepare storage
        for subject in self.trainSubs:
            tempSub = self.train[subject]
            tempFeature = tempSub.feature
            tempPheno = tempSub.pheno[self.pheno]
            # the feature is 2-Dimensional and has to be appended along axis 0
            if self.trainFeature == None:
                self.trainFeature = tempFeature[None, ...]
            else:
                self.trainFeature = np.concatenate((self.trainFeature,
                                                    tempFeature[None, ...]),
                                                   axis=0)

            # the pheno is 1-Dimensional and can be appended like this
            self.trainPheno = np.append(self.trainPheno, tempPheno)

        # now the same for test set\
        for subject in self.testSubs:
            tempSub = self.test[subject]
            tempFeature = tempSub.feature
            tempPheno = tempSub.pheno[self.pheno]
            # the feature is 2-Dimensional and has to be appended along axis 0
            if self.testFeature == None:
                self.testFeature = tempFeature[None, ...]
            else:
                self.testFeature = np.concatenate((self.testFeature,
                                                    tempFeature[None, ...]),
                                                   axis=0)

            # the pheno is 1-Dimensional and can be appended like this
            self.testPheno = np.append(self.testPheno, tempPheno)

    def selectFeatures(self):
        '''
        Method that implements different feature selection strategies

        presently I still have to manually check that I don't use RFE on rbf
        kernels. later this could be automated - nope, I am not stupid

        For the future, I would like to add the following functionality:
        - other types of feature reduction (like iCA or correlation)
        '''
        # see that both groups have the same number of features
        if not self.trainFeature.shape[1] == self.testFeature.shape[1]:
            print('The training and test set of run ' + str(self.number)
                  + ' don\'t have the same number of features')

        numberFeatures = self.trainFeature.shape[1]

        # prepare estimator object for feature selection
        svrEstimator = svm.SVR(kernel=self.kernel)
        # now run the feature selection
        if numberFeatures >= self.maxFeat:
            # more than the number of features we want
            rfeObject = fs.RFE(estimator=svrEstimator,
                               n_features_to_select=self.maxFeat,
                               step=0.1)
            # fit object
            rfeObject.fit(self.trainFeature, self.trainPheno)
            # temporary index of selected features
            tempRfeIndex = rfeObject.support_
            # better rfe index
            rfeIndex = np.where(tempRfeIndex)[0]
            # reduce a temporary copy of the features to the selection
            tempTrainFeatures = self.trainFeature[..., rfeIndex]

            # now run the crossvalidated feature selection on the data
            rfecvObject = fs.RFECV(estimator=svrEstimator,
                                   step=0.01,
                                   cv=2,
                                   loss_func=mean_squared_error)
            rfecvObject.fit(tempTrainFeatures, self.trainPheno)
            tempRfeCvIndex = rfecvObject.support_
            rfeCvIndex = np.where(tempRfeCvIndex)[0]
            # apply the index only to the original features so it can be
            # mapped back better and also used for the testing set
            #
            # create an integer index

            tempIndex = np.zeros(numberFeatures, dtype=int)
            tempIndex[rfeIndex[rfeCvIndex]] = 1
            featureIndex = tempIndex

        else:
            # no need for the initial shrinking, otherwise the same
            rfecvObject = fs.RFECV(estimator=svrEstimator,
                                   step=0.01,
                                   cv=2,
                                   loss_func=mean_squared_error)
            rfecvObject.fit(self.trainFeature, self.trainPheno)
            tempRfeCvIndex = rfecvObject.support_
            rfeCvIndex = np.where(tempRfeCvIndex)[0]

            tempIndex = np.zeros(numberFeatures, dtype=int)
            tempIndex[rfeCvIndex] = 1
            featureIndex = tempIndex

        # done, now regardless of operation, we have the feature index
        # assign the values back to the object
        bestTrainFeatures = self.trainFeature[..., featureIndex == 1]
        self.trainFeature = bestTrainFeatures
        self.featureIndex = featureIndex
        # and do the same to the test features
        bestTestFeatures = self.testFeature[..., self.featureIndex == 1]
        self.testFeature = bestTestFeatures

    def selectParameters(self):
        '''
        Method to select parameters for the mvpa algorithm

        the number of crossvalidation runs is determined by self.cv
        '''
        # check if the number of crossvalidations is higher than the number
        # of phenotypic information
        if len(self.trainPheno) < self.cv:
            self.cv = len(self.Pheno)

        # provide the parameters for the first, coarse pass
        # set of parameters
        # here we use an exponential series to cover more parameters
        expArrayOne = np.arange(-4, 4, 1)
        baseArray = np.ones_like(expArrayOne, dtype='float32') * 10
        parameterOne = np.power(baseArray, expArrayOne).tolist()

        # if for some reason this doesn't work, just paste directly
        parameters = {'C': parameterOne}
        gridModel = svm.SVR(kernel=self.kernel, epsilon=self.E)
        firstTrainModel = gs.GridSearchCV(gridModel,
                                          parameters,
                                          cv=self.cv,
                                          n_jobs=self.numberCores,
                                          verbose=0)

        firstTrainModel.fit(self.trainFeature, self.trainPheno)
        firstPassC = firstTrainModel.best_estimator_.C

        expFirstC = np.log10(firstPassC)
        expArrayTwo = np.arange(expFirstC - 1, expFirstC + 1.1, 0.1)
        baseArrayTwo = np.ones_like(expArrayTwo, dtype='float32') * 10

        parameterTwo = np.power(baseArrayTwo, expArrayTwo).tolist()

        # in case this causes trouble, paste directly
        parameters = {'C': parameterTwo}

        secondTrainModel = gs.GridSearchCV(gridModel,
                                           parameters,
                                           cv=self.cv,
                                           n_jobs=self.runCores,
                                           verbose=0)

        secondTrainModel.fit(self.trainFeature, self.trainPheno)
        bestC = secondTrainModel.best_estimator_.C

        self.C = bestC
        pass

    def trainModel(self):
        '''
        Method to train the model
        '''
        trainModel = svm.SVR(kernel=self.kernel, C=self.C, epsilon=self.E)
        trainModel.fit(self.trainFeature, self.trainPheno)
        self.model = trainModel
        pass

    def testModel(self):
        '''
        Method to test the model that was trained beforehand
        '''
        self.predictedPheno = self.model.predict(self.testFeature)
        pass

    def getError(self):
        '''
        Method to get errors from test and train session
        '''
        self.error = self.predictedPheno - self.testPheno
        pass

    pass

if __name__ == '__main__':
    pass
