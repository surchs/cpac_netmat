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
    networkName = tempAnalysis.networks.keys()[0]
    if len(tempAnalysis.networks.keys()) > 1:
        print('more than one network in there smartass')

    
    # begin with single network analysis
    network = tempAnalysis.networks[networkName]
    tempTrue = network.truePheno
    tempPred = network.predictedPheno
    tempErr = tempPred - tempTrue
    tempAbs = np.absolute(tempErr)
    tempMae = np.mean(tempAbs)
    errorVarList = [tempErr]


    # figure for text displays
    fig0 = plt.figure(0, figsize=(8.5, 11), dpi=150)
    fig0.suptitle(analysis)

    fig1 = plt.figure(1)
    fig1.suptitle('boxplots of error variance')
    # fig1.tight_layout()
    tSP1 = fig1.add_subplot(111)
    tSP1.boxplot(errorVarList)


    fig2 = plt.figure(2, figsize=(8.5, 11), dpi=150)
    fig2.suptitle('error over true age')
    # fig2.tight_layout()
    tSP2 = fig2.add_subplot(111, title=networkName)
    tSP2.plot(tempTrue, tempErr, 'co')
    
    fig3 = plt.figure(3, figsize=(8.5, 11), dpi=150)
    fig3.suptitle('absolute error over true age')
    # fig3.tight_layout()
    tSP3 = fig3.add_subplot(111, title=networkName)
    tSP3.plot(tempTrue, tempAbs, 'co')
    
    fig4 = plt.figure(4, figsize=(8.5, 11), dpi=150)
    fig4.suptitle('predicted over true age')
    # fig4.tight_layout()
    tSP4 = fig4.add_subplot(111, title=networkName)
    tSP4.plot(tempTrue, tempTrue)
    tSP4.plot(tempTrue, tempPred, 'co')

    errorVarList = []

    # then start with the texts
    txtMae = ('MAE of ' + networkName + ' = ' + str(np.round(tempMae, 3)) 
              + '\n')
    tpCorr = st.pearsonr(tempTrue, tempPred)[0]
    txtCorr = ('Pearson\'s r for ' + networkName
               + ' = ' + str(np.round(tpCorr, 3)) + '\n')
    txtParm = ('Parameters for ' + networkName
               + ': C = ' + str(np.round(network.cValue, 3)) + ' E = '
               + str(np.round(network.eValue, 6)) + '\n')

    numberFolds = len(network.cvObject)
    trueAge = tempTrue
    

    # now create the text for the whole study
    txtName = ('The name of the current analysis is ' + analysis)
    txtKernel = ('Here, a ' + tempAnalysis.kernel + ' kernel was used')
    txtFeat = ('The feature selection was ' + str(tempAnalysis.featureSelect))
    # txtConn = ('The connectivity trained on was ' + analysis.connType)
    txtFolds = (str(numberFolds) + ' folds were run while estimating age')
    txtAge = ('Their ages ranged from ' + str(np.round(trueAge.min(), 2))
              + ' to ' + str(np.round(trueAge.max(), 2))
              + ' years of age (SD = '
              + str(np.round(np.std(trueAge), 2)) + ')')
    
    statString = (txtName + '\n' + txtKernel + '\n' + txtFeat
                  + '\n' + txtFolds + '\n' + '\n' + txtAge)
    dynString = (txtMae + '\n\n' + txtCorr + '\n\n' + txtParm)

    fullString = (statString + '\n\n\n' + dynString)

    # let's build the text
    fig0.text(0.1, 0.2, fullString)

    # now save all that to a pdf
    pp = pdf((analysis + '_results.pdf'))
    pp.savefig(fig0)
    pp.savefig(fig1)
    pp.savefig(fig2)
    pp.savefig(fig3)
    pp.savefig(fig4)

    pp.close()

    print '\nDone saving. Have a nice day.'


if __name__ == '__main__':
    studyFile = sys.argv[1]
    analysis = sys.argv[2]
    Main(studyFile, analysis)
    pass
