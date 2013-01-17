'''
Created on Dec 6, 2012

@author: sebastian

Script to visualize the results of one analysis (for now)
'''
import sys
import gzip
import cPickle
import numpy as np
from scipy import stats as st
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages as pdf
from matplotlib import ticker as tc

def FeatureIndex(analysis):
    # got an analysis, get one of the masks out
    # mask = analysis.masks.values()[0]
    mask = analysis.mask
    # store the numbers of the networks so I know what I am fucking entering
    networkNumbers = {}
    run = 0.0
    for network in mask.networkNodes.keys():
        networkNumbers[network] = run
        run += 1
        
    # now recreate that connectivity matrix and enter the matrices
    maskNodes = float(len(mask.nodes))
    indexMat = np.zeros((maskNodes, maskNodes))
    print(indexMat.shape)
    # prepare the feature index for the different networks
    netFeatInd = {}
    # and now do what you did to the connectivity matrix just this time enter
    #
    # first pass: get those numbers in there
    for network in mask.networkIndices.keys():
        # grab the network index from the mask
        tempInd = mask.networkIndices[network]
        # and get the network number
        tempNumber = networkNumbers[network]
        # grab the correct columns and then put the correct numbers in there
        indexMat[..., tempInd] = tempNumber
        
    # second pass: get the numbers out again and store them in vectors
    for network in mask.networkIndices.keys():
        # grab the network index from the mask
        tempInd = mask.networkIndices[network]
        # first get the rows belonging to the network
        tempNet = indexMat[tempInd, ...]
        # then get the matrix belonging to the within features
        tempWithinNet = tempNet[..., tempInd]
        # and now only take the lower triangle
        tempMask = np.ones_like(tempWithinNet)
        tempMask = np.tril(tempMask, -1)
        # and put it in the variable
        tempWithin = tempWithinNet[tempMask == 1]
        
        # now for between - delete the within rows
        tempBetweenNet = np.delete(tempNet, tempInd, 1)
        # now stretch it out
        tempBetween = np.reshape(tempBetweenNet,
                                 tempBetweenNet.size)
        
        # and lastly for the whole connectivity
        tempWhole = np.append(tempWithin, tempBetween)
        
        netFeatInd[network] = tempWhole
    
    # and print some stuff
    return netFeatInd, networkNumbers


def NetworkFeatures(network):
    # get the different feature weights from the different runs
    weightMat = np.array([])
    for run in network.runs.keys():
        tempRun = network.runs[run]
        tempModel = tempRun.model
        tempWeights = tempModel.coef_
        # get the shit in the matrix
        if weightMat.size == 0:
            weightMat = tempWeights[None, ...]
        else:
            weightMat = np.concatenate((weightMat, tempWeights[None, ...]),
                                       axis=0)
        
    # now get the goddamn mean
    meanFeatures = np.mean(weightMat, axis=0)
    return meanFeatures


def Main(studyFile, analysis):
    f = gzip.open(studyFile)
    study = cPickle.load(f)
    print('Loaded study')
    # just loop through all analyses and run this shit
    if analysis == None:
        print('Running all the analyses in here!')
        for analysis in study.analyses.keys():
            Visualize(study, analysis)
    else:
        Visualize(study, analysis)


def Visualize(study, analysis):
    print('Fetching analysis ' + analysis + ' now. Hold on to your heads!')
    tempAnalysis = study.analyses[analysis]
    networkNames = tempAnalysis.networks.keys()
    networkNames.sort()
    numberNetworks = float(len(networkNames))
    tempNet = tempAnalysis.networks.values()[0]
    numberSubjects = float(len(tempNet.truePheno))
    aName = analysis
    # netFeatInd, networkNumbers = FeatureIndex(tempAnalysis)

    valueDict = {}
    shappStore = np.array([])
    errList = []
    maeList = []
    normCount = 0
    # a matrix to store networks by subjects prediction-errors for crosscorr
    kendallMat = np.array([])
    netErrMat = np.array([])
    netAbsMat = np.array([])

    for network in networkNames:
        tempNetwork = tempAnalysis.networks[network]
        tempDict = {}
        tempTrue = tempNetwork.truePheno
        tempPred = tempNetwork.predictedPheno
        tempErr = tempPred - tempTrue
        # append error to errorlist for ANOVA
        errList.append(tempErr)
        tempAbs = np.absolute(tempErr)
        tempMae = np.mean(tempAbs)
        # now rank those ages and store the ranks in the matrix to calculate
        # Kendall's W
        # must be in the same order for all networks
        tempRanks = np.argsort(tempPred)
        ranks = np.empty(len(tempRanks), int)
        ranks[tempRanks] = np.arange(len(tempRanks))
        ranks += 1
        if kendallMat.size == 0:
            kendallMat = ranks[None, ...]
        else:
            kendallMat = np.concatenate((kendallMat, ranks[None, ...]),
                                        axis=0)
            
        # now get the features for this network
        '''
        meanFeatures = NetworkFeatures(tempNetwork)
        # store the features under the name of the network they connect to
        netInd = netFeatInd[network]
        netInd = netInd[None, ...]
        print('meanFeat ' + str(meanFeatures.shape))
        print('netInd ' + str(netInd.shape))
            
        tempFeatStore = {}
        for netNum in networkNumbers.keys():
            netNumber = networkNumbers[netNum]
            # store this stuff
            tempFeatStore[netNum] = meanFeatures[netInd==netNumber]
        '''

        if netErrMat.size == 0:
            # first entry, populate
            netErrMat = tempErr[None, ...]
        else:
            # concatenate any further values
            netErrMat = np.concatenate((netErrMat, tempErr[None, ...]), axis=0)

        # append absolute error to netAbs Matrix for cross correlation
        if netAbsMat.size == 0:
            # first entry, populate
            netAbsMat = tempAbs[None, ...]
        else:
            # concatenate any further values
            netAbsMat = np.concatenate((netAbsMat, tempAbs[None, ...]), axis=0)

        # append mae to maelist for display
        maeList.append(tempMae)
        tempStd = np.std(tempErr)
        # get the p value of the shapiro-wilk test
        tempShapp = st.shapiro(tempErr)[1]
        if tempShapp >= 0.05:
            normCount += 1
        shappStore = np.append(shappStore, tempShapp)
        # assign these values to the DICT
        tempDict['true'] = tempTrue
        tempDict['pred'] = tempPred
        tempDict['error'] = tempErr
        tempDict['abs'] = tempAbs
        tempDict['std'] = tempStd
        tempDict['shapp'] = tempShapp
        tempDict['mae'] = tempMae
        # tempDict['weights'] = tempFeatStore
        # put the dictionary in the valueDict
        valueDict[network] = tempDict

    # now run the tests to determine if we can run the ANOVA
    if shappStore.max() >= 0.05:
        print 'All networks are nicely normally distributed'
        # now run the ANOVA thing - right now, we run just everything
        anova = st.f_oneway(*errList)
        print '\nANOVA has run'
        print ('Behold the amazing F of '
               + str(round(anova[0], 4))
               + ' and p '
               + str(round(anova[1], 4)))

    else:
        print 'not all networks are normally distributed'
        print (str(normCount)
               + ' out of '
               + str(numberNetworks)
               + ' networks are normally distributed')
        anova = (999, 999)

    # now do the fancy Kendall's W business
    # first get the vector of summed total ranks across all networks (cols)
    
    print('Kendalls')
    print('nNet = ' + str(numberNetworks) + ' nSub = ' + str(numberSubjects))
    print(kendallMat.shape)
    sumRankVec = np.sum(kendallMat, axis=0)
    print(sumRankVec)
    meanRank = 1.0 / 2.0 * numberNetworks * (numberSubjects + 1)
    print(meanRank)
    sumSquaredDevs = np.sum((sumRankVec - meanRank) ** 2)
    print(sumSquaredDevs)
    kendallsW = 12.0 * sumSquaredDevs / ((numberNetworks ** 2.0) * 
                                         ((numberSubjects ** 3)
                                           - numberSubjects))
    txtKendallsW = ('Kendall\'s W = ' + str(kendallsW))
    print('Kendall\'s W = ' + str(kendallsW))

    # now cols are hardcoded and rows depend on them
    cols = 2.0
    rows = np.ceil(numberNetworks / cols)

    # figure for text displays
    fig0 = plt.figure(0, figsize=(8.5, 11), dpi=150)
    fig0.suptitle(aName)

    fig1 = plt.figure(1)
    fig1.suptitle('boxplots of error variance')
    # fig1.tight_layout()

    fig2 = plt.figure(2, figsize=(8.5, 11), dpi=150)
    fig2.suptitle('error over true age')
    # fig2.tight_layout()

    fig3 = plt.figure(3, figsize=(8.5, 11), dpi=150)
    fig3.suptitle('absolute error over true age')
    # fig3.tight_layout()

    fig4 = plt.figure(4, figsize=(8.5, 11), dpi=150)
    fig4.suptitle('predicted over true age')
    # fig4.tight_layout()

    fig5 = plt.figure(5)
    fig5.suptitle('mean absolute error of the networks')

    fig6 = plt.figure(6)
    fig6.suptitle('correlation of errors between networks')

    fig7 = plt.figure(7)
    fig7.suptitle('correlation of absolute errors between networks')

    # another figure for text displays
    fig8 = plt.figure(0, figsize=(8.5, 11), dpi=150)
    fig8.suptitle(aName)

    loc = 1

    txtMae = ''
    # txtRmse = ''
    # txtNodes = ''
    # txtFeat = ''
    txtCorr = ''
    txtParm = ''

    errorVarList = []
    errorNameList = []
    numberFolds = None
    nitFigList = []
    trueAge = None
    loopFigId = 99
    figIds = []
    # now loop over the networks and get the data
    for network in networkNames:
        # first get the values from the dict
        tD = valueDict[network]

        # then start with the texts
        txtMae = (txtMae + 'MAE of ' + network
                  + ' = ' + str(np.round(tD['mae'], 3)) + '\n')
        # txtRmse = (txtRmse + 'RMSE of ' + networkName
        #           + ' = ' + str(tD['rmse']) + '\n')
        # read out temporary network file
        tempNet = tempAnalysis.networks[network]

        tpCorr = st.pearsonr(tempNet.truePheno,
                             tempNet.predictedPheno)[0]
        txtCorr = (txtCorr + 'Pearson\'s r for ' + network
                   + ' = ' + str(np.round(tpCorr, 3)) + '\n')
        txtParm = (txtParm + 'Parameters for ' + network
                   + ': C = ' + str(np.round(tempNet.cValue, 3)) + ' E = '
                   + str(np.round(tempNet.eValue, 6)) + '\n')

        numberFolds = len(tempNet.cvObject)
        trueAge = tempNet.truePheno       
        
        errorVarList.append(tD['error'])
        errorNameList.append(network)

        tSP2 = fig2.add_subplot(rows, cols, loc, title=network)
        tSP2.plot(tD['true'], tD['error'], 'co')

        tSP3 = fig3.add_subplot(rows, cols, loc, title=network)
        tSP3.plot(tD['true'], tD['abs'], 'co')

        tSP4 = fig4.add_subplot(rows, cols, loc, title=network)
        tSP4.plot(tD['true'], tD['true'])
        tSP4.plot(tD['true'], tD['pred'], 'co')
        
        
        # make the loop for the network boxplot figures
        # for the boxplots, we have to append the data to a list
        # first get the current list of networks
        '''
        weightDict = tD['weights']
        
        netWeightList = []
        for netName in weightDict.keys():
            netWeightList.append(weightDict[netName])
            print(network + ' ' + netName + ' ' + str(len(weightDict[netName])))
            
        print(network + ' netweightlength ' + str(len(netWeightList)))
            
        # got all the weight vectors in here, now create a figure and 
        # use loopFigId as index
        tempFigure = plt.figure(loopFigId)
        tempSubPlot = tempFigure.add_subplot(111)
        # boxIndex = np.arange(len(netWeightList))
        tempSubPlot.boxplot(netWeightList)
        tempSubPlot.set_ylabel('weight distribution for network ' + network)
        # tempSubPlot.set_xticks(boxIndex)
        # tempSubPlot.set_xticklabels(networkNames)
        plt.setp(tempSubPlot, xticklabels=networkNames)
        tempFigure.autofmt_xdate()
        # now store figure in list
        nitFigList.append(tempFigure)

        figIds.append(loopFigId)
        loopFigId += 1
        '''
        
        loc += 1

    # now create the text for the whole study
    txtName = ('The name of the current analysis is ' + aName)
    txtKernel = ('Here, a ' + tempAnalysis.kernel + ' kernel was used')
    txtFeat = ('The feature selection was ' + str(tempAnalysis.featureSelect))
    # txtConn = ('The connectivity trained on was ' + analysis.connType)
    txtFolds = (str(numberFolds) + ' folds were run while estimating age')
    txtAnova = ('ANOVA of Network effect on prediction error returned:\nF = '
                + str(np.round(anova[0], 3)) + ' p = '
                + str(np.round(anova[1], 3)))
    txtAge = ('Their ages ranged from ' + str(np.round(trueAge.min(), 2))
              + ' to ' + str(np.round(trueAge.max(), 2))
              + ' years of age (SD = '
              + str(np.round(np.std(trueAge), 2)) + ')')

    statString = (txtName + '\n' + txtKernel + '\n' + txtFeat
                  + '\n' + txtFolds + '\n' + txtAnova + '\n' + txtAge + '\n'
                  + txtKendallsW)
    # + txtRmse + '\n\n'
    dynString = (txtMae + '\n\n' + txtCorr + '\n\n'
                 + txtParm)

    fullString = (statString + '\n\n\n' + dynString)

    # let's build the text
    fig0.text(0.1, 0.2, fullString)

    # now we can build figure 1
    tSP1 = fig1.add_subplot(111)
    tSP1.boxplot(errorVarList)
    plt.setp(tSP1, xticklabels=errorNameList)
    fig1.autofmt_xdate()

    # and now we build figure 5
    tSP5 = fig5.add_subplot(111)
    indMae = range(len(maeList))
    tSP5.bar(indMae, maeList, facecolor='#99CCFF', align='center')
    tSP5.set_ylabel('MAE for network')
    tSP5.set_xticks(indMae)
    # set x-labels to the network names
    tSP5.set_xticklabels(networkNames)
    fig5.autofmt_xdate()

    # and lastly figure 6 with the crosscorrelations
    '''
    tSP6 = fig6.add_subplot(111)
    # run correlation analysis
    netCorrErr = np.corrcoef(netErrMat)
    tSP6.pcolor(ageMat)
    tSP6.pcolor(netCorrErr)
    for y in range(netCorrErr.shape[0]):
        for x in range(netCorrErr.shape[1]):
            tSP6.text(x + 0.5, y + 0.5, '%.2f' % netCorrErr[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',
                     )
    '''

    # and then the same thing for absolute errors
    tSP7 = fig7.add_subplot(111)
    # run correlation analysis
    netCorrAbs = np.corrcoef(netAbsMat)
    tSP7.pcolor(kendallMat)
    '''tSP7.pcolor(netCorrAbs)
    for y in range(netCorrAbs.shape[0]):
        for x in range(netCorrAbs.shape[1]):
            tSP7.text(x + 0.5, y + 0.5, '%.2f' % netCorrAbs[y, x],
                     horizontalalignment='center',
                     verticalalignment='center',
                     )
'''
    # adjust the images
    fig1.subplots_adjust(hspace=0.5, wspace=0.5)
    fig2.subplots_adjust(hspace=0.5, wspace=0.5)
    fig3.subplots_adjust(hspace=0.5, wspace=0.5)
    fig4.subplots_adjust(hspace=0.5, wspace=0.5)
    fig5.subplots_adjust(hspace=0.5, wspace=0.5)
    fig6.subplots_adjust(hspace=0.5, wspace=0.5)
    fig7.subplots_adjust(hspace=0.5, wspace=0.5)

    # now save all that to a pdf
    pp = pdf((aName + '_results.pdf'))
    pp.savefig(fig0)
    pp.savefig(fig1)
    pp.savefig(fig2)
    pp.savefig(fig3)
    pp.savefig(fig4)
    pp.savefig(fig5)
    pp.savefig(fig6)
    pp.savefig(fig7)
    for figure in nitFigList:
        pp.savefig(figure)
    pp.close()
    
    
    plt.close(1)
    plt.close(2)
    plt.close(3)
    plt.close(4)
    plt.close(5)
    plt.close(6)
    plt.close(7)
    for figId in figIds:
        plt.close(figId)
    

    print '\nDone saving. Have a nice day.'


if __name__ == '__main__':
    studyFile = sys.argv[1]
    if len(sys.argv) > 2:
        analysis = sys.argv[2]
    else:
        analysis = None
    Main(studyFile, analysis)
    pass
