'''
Created on Oct 18, 2012

@author: sebastian

A class directory for the beautiful classes I will need in preprocessing
'''
import os
import gzip
import copy
import cPickle
import numpy as np
import nibabel as nib


class Subject(object):
    '''
    An object that contains a dynamic number of derivatives and gets stored
    under some informative path and name
    '''

    def __init__(self, name, pipeline):
        '''
        Constructor method
        '''
        self.name = name
        self.pipeline = pipeline
        self.derivativePath = {}
        self.maskDerivativePath = {}
        self.pheno = {}
        self.masks = {}
        self.derivativeMasks = {}
        # attribute used during the analysis
        self.derivative = {}
        self.feature = None
        self.network = None
        self.analysisDerivative = None

    def addDerivativePath(self, derivativeName, derivativePath, mask=None):
        '''
        Method to get and store the template paths for the derivative
        data generated by CPAC

        if mask is defined then the path is assigned to a specific mask
        '''
        if mask:
            if not mask in self.maskDerivativePath.keys():
                self.maskDerivativePath[mask] = {}

            if not mask in self.masks.keys():
                print('The mask you specified (' + mask + ') is not yet saved '
                      + 'for subject ' + self.name)

            if not derivativeName in self.derivativePath[mask].keys():
                self.maskderivativePath[mask][derivativeName] = derivativePath
                # print derivativeName, 'added to', self.name, 'archive'
            else:
                print 'There is already a path saved for', derivativeName

        else:

            if not derivativeName in self.derivativePath.keys():
                self.derivativePath[derivativeName] = derivativePath
                # print derivativeName, 'added to', self.name, 'archive'
            else:
                print 'There is already a path saved for', derivativeName

    def makeDerivative(self, derivative=None):
        '''
        Method to add another derivative to the dictionary. See if it already
        exists.
        If it does, give a warning and don't save.
        Maybe a switch option for overwriting would be nice
        '''
        if derivative == None:
            # no selection by user, run all of them
            selection = self.derivativePath.keys()
        else:
            # user has entered something, check if it exists
            if derivative in self.derivativePath.keys():
                selection = derivative
            else:
                print derivative, 'is not known to the archive', self.name
                selection = []

        # loop through the masks that have been entered so far
        for mask in self.masks.keys():
            tempMask = self.masks[mask]
            # check if the current mask is already represented in the
            # masked derivative storage
            if not mask in self.derivativeMasks.keys():
                self.derivativeMasks[mask] = {}

            print 'running', self.pipeline, tempMask.name, self.name
            # now loop through the derivative paths and make them happen for
            # the current mask
            for derivative in selection:
                filePath = self.derivativePath[derivative]
                tempDerivative = Derivative(derivative)
                tempDerivative.makeFeature(filePath, self.masks[mask])
                derName = tempDerivative.name

                self.derivativeMasks[mask][derName] = tempDerivative

        # now loop through the mask specific derivatives
        for mask in self.maskDerivativePath.keys():
            tempMask = self.maskDerivativePath[mask]
            for derivative in tempMask.keys():
                filePath = tempMask[derivative]
                tempDerivative = Derivative(derivative)
                tempDerivative.makeFeature(filePath, self.masks[mask])
                derName = tempDerivative.name

                # and save it in the same structure
                self.derivativeMasks[mask][derName] = tempDerivative

    def loadMask(self, maskPath):
        '''
        load the mask object and keep it in store to generate the derivatives
        then clear it out again
        '''
        maskFile = gzip.open(maskPath, 'rb')
        tempMask = cPickle.load(maskFile)
        maskName = tempMask.name
        self.masks[maskName] = tempMask


class Derivative(object):
    '''
    An object that contains the fully preprocessed feature vector for a
    derivative.
    It is designed in a way that accomodates all possible derivatives

    Depending on whether the saved derivative is functional connectivity or
    anything else, the features get stored in a matrix or as a vector
    '''
    def __init__(self, name):
        '''
        Constructor method
        '''
        self.name = name
        self.feature = np.array([])

    def addFeature(self, feature):
        '''
        method to add a feature vector
        '''
        self.feature = feature

    def makeFeature(self, filePath, mask):
        '''
        This method loads the nifti path handed over by the mother subject
        object. I could implement different methods for 4D and 3D preprocessed
        files (aka Fcon and everything else) but I'll try to make it work with
        a dynamic switch here instead
        '''
        tempImg = nib.load(filePath)
        tempImgShape = tempImg.shape
        featData = tempImg.get_data()

        # now for the switch
        if len(tempImgShape) == 4:
            # we have ourselves a 4D file - so prepare a temporary timeseries
            # container
            tempTimeseries = np.array([])
            # now lets get the timeseries
            for node in mask.nodes:
                rawTimeseries = featData[mask.mask == node]
                timeseries = np.average(rawTimeseries, axis=0)
                if tempTimeseries.size == 0:
                    # first time customer...
                    tempTimeseries = timeseries[np.newaxis, ...]
                else:
                    tempTimeseries = np.concatenate((tempTimeseries,
                                                timeseries[np.newaxis, ...]),
                                                axis=0)
            # so we have the timeseries in a nice array of ordered nodes - lets
            # make a matrix of it
            self.feature = np.corrcoef(tempTimeseries)

            pass
        elif len(tempImgShape) == 3:
            # we have ourselves a 3D file
            for node in mask.nodes:
                rawNodeFeature = featData[mask.mask == node]
                nodeFeature = np.average(rawNodeFeature, axis=0)
                self.feature = np.append(self.feature, nodeFeature)
            pass
        else:
            # we have some messed up file we don't like
            print 'Nah, this is not a nice file. find someone else to read it'
            pass


class Mask(object):
    '''
    An object that stores a nifti node-mask in a more accessible format than
    the nifti + textfile of the nodes inside each network.
    It contains a list of all nodes inside a network for each network.
    '''
    def __init__(self, name):
        '''
        Constructor method. Simply takes the name of the mask
        '''
        self.name = name
        self.networkNodes = {}
        self.networkIndices = {}
        self.nodes = np.array([])
        self.unassigned = np.array([])
        # attributes that will be used later
        self.mask = None
        self.maskHeader = None
        self.maskAffine = None

    def loadMask(self, maskFile):
        '''
        Method to load a nifti mask into this object
        '''
        tempImg = nib.load(maskFile)
        self.maskHeader = tempImg.get_header()
        self.maskAffine = tempImg.get_affine()
        self.mask = tempImg.get_data()
        self.nodes = np.unique(self.mask[self.mask != 0])

    def addNetwork(self, networkName, nodeList):
        '''
        Method to add an index list of nodes inside the network and assign the
        remaining networkNodes to the network 'unassigned' which is not
        included in the analysis
        '''
        # check if the nodes in nodeList are part of the mask or already in
        # other networks
        goodNodes = np.array([])
        for node in nodeList:
            if not node in self.nodes:
                print('The node ' + str(node) + ' is not in this mask')
                continue

            elif (len(self.networkNodes.keys()) == 0 and
                  not node in goodNodes):
                # no networks assigned so far, just check against goodNodes
                goodNodes = np.append(goodNodes, node)
                continue

            else:
                for network in self.networkNodes.keys():
                    if node in self.networkNodes[network]:
                        print('Node ' + str(node) + ' already in '
                              + network + ' network')
                        break
                    else:
                        goodNodes = np.append(goodNodes, node)
                        break

        # check if anything is left in the nodelist
        if len(goodNodes) == 0:
            print('\nThere are no nodes stored for ' + networkName
                  + ' network')
        else:
            print('\n' + str(len(goodNodes)) + ' nodes will be stored for '
                  + networkName + ' network')
            self.networkNodes[networkName] = goodNodes

        # check if the remaining nodes are all stored inside other networkNodes
        tempNodeArray = np.array([])
        for network in self.networkNodes.keys():
            tempNodes = self.networkNodes[network]
            # make the loop independent of array or list type
            for node in tempNodes:
                tempNodeArray = np.append(tempNodeArray, node)

        # clear out self.unassigned
        self.unassigned = np.array([])
        # now see if any one of the nodes is not inside the networkNodes
        for node in self.nodes:
            if not node in tempNodeArray:
                self.unassigned = np.append(self.unassigned, node)

    def infoAbout(self):
        '''
        Method that returns the values and keys in the network dictionary as
        well as all the (nonzero) nodes that are unassigned at the moment
        '''
        return (self.networkNodes, self.unassigned)

    def makeIndices(self, network=None):
        '''
        Method to get indices of the network nodes in the node vector. If no
        network is specified then just run it on all of them
        '''
        if network == None:
            networkList = self.networkNodes.keys()
        elif not network in self.networkNodes.keys():
            print('Bullshit, your network \'' + network + '\' is wrong!')
            networkList = []
        else:
            networkList = network

        # loop through the networkNodes in the goddamn fucking shitbin of a
        # mask
        for network in networkList:
            tempNodes = self.networkNodes[network]
            tempIndices = np.array([])
            # loop through the goddamn network Nodes and find them in the mask
            # nodes
            for node in tempNodes:
                # get the index
                tempInd = np.where(self.nodes == node)
                # and append it to the goddamn network-node array
                tempIndices = np.append(tempIndices, tempInd)

            # put the indices into the network index file
            self.networkIndices[network] = tempIndices.astype(int)

    def makeMask(self, outFile):
        '''
        Method to create a nifti file containing all the nodes in the self.mask
        matrix
        '''
        outNifti = nib.Nifti1Image(self.mask, self.maskAffine, self.maskHeader)
        nib.nifti1.save(outNifti, outFile)
        print 'Just generated the full mask inside', outFile

    def makeNetwork(self, networks=None, outFile=None):
        '''
        Method that creates a Nifti file containing all the nodes from one or
        multiple networkNodes
        '''
        if networks == None:
            # if no network is given, use all
            networks = self.networkNodes.keys()

        if outFile == None:
            outFile = (self.name + '_network_output.nii.gz')

        showNodes = np.array([])
        for networkName in networks:
            # check if it exists
            if not networkName in self.networkNodes.keys():
                print networkName, 'is not a network here yet'
                continue

            netNodes = self.networkNodes[networkName]
            showNodes = np.append(showNodes, netNodes)

        # make a copy of the mask matrix and set all node values that are not
        # in the showNodes array to zero
        maskCopy = copy.copy(self.mask)
        for node in self.nodes:
            if not node in showNodes:
                maskCopy[maskCopy == node] = 0

        # make it into a file
        outNifti = nib.Nifti1Image(maskCopy, self.maskAffine, self.maskHeader)
        nib.nifti1.save(outNifti, outFile)
        print 'Just generated a mask of networkNodes inside', outFile

    def sameSame(self, compareMask):
        '''
        Method to compare the mask object to another mask object from the same
        class
        returns:
            True if the same
            False if not
            None if not a Mask class object
        '''
        # first, check if the compareMask is from the same class
        sameStatus = False
        if isinstance(compareMask, Mask):
            # OK, this is from the same class
            if (self.mask.all() == compareMask.mask.all() and
                self.networkNodes.keys() == compareMask.networkNodes.keys() and
                self.name == compareMask.name):
                # they are the same mask
                sameStatus = True
        else:
            # this compareMask thing is not a Mask object
            print('the mask you are trying to compare is bullshit, try again')
            sameStatus = None

        return sameStatus

    def saveYourself(self, fileName=None, filePath=None):
        '''
        Method to store the mask object on disk. The default filename is the
        name of the mask itself plus the standard file-ending .mask
        '''
        if filePath == None:
            if fileName == None:
                outPath = (self.name + '_maskobject.mask')
            else:
                outPath = fileName
        else:
            if fileName == None:
                outName = (self.name + '_maskobject.mask')
                outPath = os.path.join(filePath, outName)
            else:
                outPath = os.path.join(filePath, fileName)

        dumpFile = gzip.open(outPath, 'wb')
        cPickle.dump(self, dumpFile, 2)
