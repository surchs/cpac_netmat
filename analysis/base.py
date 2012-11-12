'''
Created on Nov 7, 2012

@author: sebastian

classes for the analysis
'''
import os
import re
import glob
import gzip
import copy
import cPickle
import numpy as np
from sklearn import svm
import sklearn.grid_search as gs
import sklearn.cross_validation as cv
import sklearn.feature_selection as fs
from sklearn.metrics import mean_squared_error


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

    def getSubjectPaths(self):
        '''
        Method to get the subject file paths
        Checks what mask a subject is associated with and then loads both the
        subject and the mask

        since the subject folders contain one file per mask, we always loop
        through the subject files (*.sub) inside any selected subject directory
        '''

        # check the parameters first
        if not os.path.isdir(self.subjectPath):
            print('subjectPath ' + self.subjectPath + ' doesn\'t exist')

        if self.subjectList == None:
            # no subjects specified, take all you can get in the dir
            print('Using all subjects in ' + os.path.abspath(self.subjectPath))
            # here we can set the template for the subject directory
            if self.subjectPath.endswith('/'):
                self.subjectPaths = os.path.abspath(glob.glob((self.subjectPath
                                                               + '*/*.sub')))
            else:
                self.subjectPaths = os.path.abspath(glob.glob((self.subjectPath
                                                               + '/*/*.sub')))

        else:
            # subject file specified, take only the specified ones
            subjectFile = open(self.subjectList, 'rb')
            tempSubjectFileList = subjectFile.readlines()

            for line in tempSubjectFileList:
                # currently only one column expected that contains the subject
                # name
                tempSubName = line.strip()
                tempSubPath = os.path.join(os.path.abspath(self.subjectPath),
                                           tempSubName)
                tempSubjects = glob.glob((tempSubPath + '/*.sub'))
                # loop through the results
                for tempSubject in tempSubjects:
                    self.subjectPaths.append(os.path.abspath(tempSubject))

        print(self.subjectPaths)

    def getSubjects(self):
        '''
        Method to get the subjects that are listed in the subject paths, check
        what mask they belong to, store them accordingly and store the mask if
        it hasn't been stored yet.

        If it has been stored, then check if it is identical to the mask
        '''
        for subjectPath in self.subjectPaths:
            # open the file
            tempSubFile = gzip.open(subjectPath, 'rb')
            tempSubject = cPickle.load(tempSubFile)
            tempSubName = tempSubject.name
            # get the mask in the subject
            tempSubMask = copy.copy(tempSubject.mask)
            tempMaskName = tempSubMask.name
            # check if the mask name already exists in the saved masks
            if tempMaskName in self.masks.keys():
                # it is already in, compare it to the saved mask
                if not self.masks[tempMaskName].sameSame(tempSubMask):
                    # this mask is different from the one we already saved and
                    # this is a problem - alert and skip the subject
                    # (not optimal, I know - its an order thing... fix later)
                    print('The mask: ' + tempMaskName + ' of subject '
                          + tempSubName + ' is different from the saved'
                          + ' mask in our repository')
                    continue
            else:
                # the mask is not saved yet, so do this now
                self.masks[tempMaskName] = tempSubMask

            # now we can continue processing the subjects

            # check if the mask exists already for the maskedSubjects dict
            if not tempMaskName in self.maskedSubjects.keys():
                # create the entry and make it another dictionary
                self.maskedSubjects[tempMaskName] = {}

            # now get rid of the mask
            tempSubject.mask = None
            self.maskedSubjects[tempMaskName][tempSubName] = tempSubject

        # done with the processing, tell the world about it and give a summary
        print('Done with fetching subjects, this is what it looks like')
        for mask in self.maskedSubjects.keys():
            print('    ' + mask + ' : '
                  + str(len(self.maskedSubjects[mask].keys())) + ' subjects')

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
            6) what kernel to use
            7) C value to begin with in grid search for SVM model
            8) E value to use in SVM model
            9) number of crossvalidation for grid search in run
            10) number of cores to use per run
            11) max features to use
        '''

        self.configFile = open(configFile, 'rb')
        # set the file counter to 0
        self.configFile.seek(0)
        tempConfigLines = self.configFile.readlines()

        # loop over the lines
        for line in tempConfigLines:
            configLine = line.strip().split()
            if '#' in line or re.match('^\n', line):
                # this is a comment, keep going
                continue

            analysisName = configLine[1]
            derivative = configLine[2]
            mask = configLine[3]
            featureSelect = configLine[4]
            crossvalidate = configLine[5]
            kernel = configLine[6]
            cValue = int(configLine[7])
            eValue = int(configLine[8])
            gridCv = int(configLine[9])
            runCores = int(configLine[10])
            maxFeat = int(configLine[11])

            # check if the desired mask is present in the Study Object
            if not mask in self.masks.keys():
                # the mask is not here, alert
                print('\nThe mask ' + mask + ' for analysis ' + analysisName
                      + ' is not available in the study object ' + self.name)
            else:
                # get the mask ready
                analysisMask = self.masks[mask]

            # make an analysis object and set parameters
            tempAnalysis = Analysis(analysisName, analysisMask)

            tempAnalysis.derivative = derivative
            tempAnalysis.featureSelect = featureSelect
            tempAnalysis.crossvalidate = crossvalidate
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
                if not derivative in tempSub.derivatives.keys():
                    # something is wrong with this subject
                    print(subject + ' doesn\'t have derivative ' + derivative)
                else:
                    # nothing wrong with this one - assign the correct
                    # derivative and delete the rest
                    tempSub.derivative = tempSub.derivatives[derivative]
                    tempSub.derivatives = None
                    tempSubs[subject] = tempSub

            tempAnalysis.subjects = tempSub
            # and store the object in the dictionary
            self.analyses[tempAnalysis.name] = tempAnalysis
        print(self.analyses.keys())


class Analysis(object):
    '''
    class to contain one full analysis. in the current design, one analysis is
    one combination of subjects, parameters and derivative only. So different
    sets of subjects, derivatives or parameters will be represented as
    different analyses
    '''

    def __init__(self, name):
        self.name = name
        # parameters to be determined later
        self.derivative = None
        self.featureSelect = None
        self.crossvalidate = None
        self.cvObject = None
        self.kernel = None
        self.subjects = None
        self.mask = None
        self.cValue = None
        self.eValue = None
        self.gridCv = None
        self.numberCores = None
        self.runCores = None
        self.maxFeat = None
        self.networks = {}

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
                tempDer = tempSub.derivative
                tempInd = self.mask.networkIndices[network]
                # see if it is a matrix or vector
                if len(tempDer.shape) == 1:
                    # it's a vector we have to cut it up into pieces
                    tempWithin = tempDer.feature[tempInd]
                    tempBetween = np.delete(tempDer.feature[network], tempInd)
                    tempWhole = tempDer.feature

                elif len(tempDer.shape) == 2:
                    # it's a matrix - this shit is more difficult
                    # first get the rows belonging to the network
                    tempNet = tempDer.feature[tempInd, ...]
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

                # write the result back into the subject
                tempSub.derivative['within'] = tempWithin
                tempSub.derivative['between'] = tempBetween
                tempSub.derivative['whole'] = tempWhole

                # and write the stuff to the network
                tempNetwork.subjects[subject] = tempSub

            # edit the network parameters
            tempNetwork.gridCv = self.gridCv

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
        self.gridCv = None
        self.maxFeat = None
        self.numberCores = None
        self.runCores = None

    def makeRuns(self):
        '''
        Method to create and store run objects for later execution
        '''

        runID = 1
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
            run.runCv = self.runCv

            # model parameters
            run.kernel = None
            run.C = None
            run.E = 0.000001

            # run the prepare run method
            run.prepareRun()
            # and set the necessary parameters


            # store the run in the Network object
            self.runs[run.number] = run

            # +1 on the run ID
            runID += 1


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
        self.pheno = None
        self.maxFeat = 2000
        self.runCores = 2
        self.runCv = 5

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
            tempFeature = tempSub.derivative
            tempPheno = tempSub.pheno[self.pheno]
            if not self.trainFeature:
                # self.trainFeature doesn't exist, make it happen
                self.trainFeature = tempFeature[None, ...]
            else:
                self.trainFeature = np.concatenate((self.trainFeature,
                                                    tempFeature[None, ...]),
                                                   axis=0)
            if not self.trainPheno:
                # create self.trainPheno
                self.trainPheno = tempPheno[None, ...]
            else:
                self.trainPheno = np.concatenate((self.trainPheno,
                                                  tempPheno[None, ...]),
                                                 axis=0)

        # now the same for test set\
        for subject in self.testSubs:
            tempSub = self.test[subject]
            tempFeature = tempSub.derivative
            tempPheno = tempSub.pheno[self.pheno]
            if not self.testFeature:
                # self.testFeature doesn't exist, make it happen
                self.testFeature = tempFeature[None, ...]
            else:
                self.testFeature = np.concatenate((self.testFeature,
                                                   tempFeature[None, ...]),
                                                  axis=0)
            if not self.testPheno:
                # create self.testPheno
                self.testPheno = tempPheno[None, ...]
            else:
                self.testPheno = np.concatenate((self.testPheno,
                                                 tempPheno[None, ...]),
                                                axis=0)

    def selectFeatures(self):
        '''
        Method that implements different feature selection strategies

        presently I still have to manually check that I don't use RFE on rbf
        kernels. later this could be automated - nope, I am not stupid

        For the future, I would like to add the following functionality:
        - other types of feature reduction (like iCA or correlation)
        '''
        # see that both groups have the same number of features
        if not len(self.train.shape[1]) == len(self.test.shape[1]):
            print('The training and test set of run ' + str(self.number)
                  + ' don\'t have the same number of features')

        numberFeatures = len(self.train.shape[1])

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
