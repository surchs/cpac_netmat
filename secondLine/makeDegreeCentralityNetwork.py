'''
Created on Feb 25, 2013

@author: surchs
'''
import gzip
import cPickle
import numpy as np
import nibabel as nib


def loadNiftiImage(pathToNiftiFile):
    image = nib.load(pathToNiftiFile)
    data = image.get_data()
    
    return image, data


def loadArchive(pathToArchive):
    f = gzip.open(pathToArchive)
    archive = cPickle.load(f)
    
    return archive


def Main():
    # Define inputs
    pathToNegDegreeCentralityImage = '/home2/surchs/secondLine/degree_centrality/degree_centrality_volumes/group_z_age_neg.nii.gz'
    pathToPosDegreeCentralityImage = '/home2/surchs/secondLine/degree_centrality/degree_centrality_volumes/group_z_age_pos.nii.gz'
    pathToNetworkNodes = '/home2/surchs/secondLine/configs/networkNodes.dict'
    pathToRoiMask = '/home2/surchs/masks/ROIs/craddock200wave_p1l.nii.gz'
    
    # Define outputs
    pathToOutputFile = '/home2/surchs/secondLine/degree_centrality/degree_centrality_volumes/network_results.txt'
    
    # Read inputs
    posDCImage, posDCData = loadNiftiImage(pathToPosDegreeCentralityImage)
    negDCImage, negDCData = loadNiftiImage(pathToNegDegreeCentralityImage)
    roiImage, roiData = loadNiftiImage(pathToRoiMask)
    networkNodes = loadArchive(pathToNetworkNodes)
    
    # Add network container
    networkDict = {}
    
    # Now loop through the networks and search for significant positive and
    # negative parameter estimates of age for DC
    totalPos = 0
    totalNeg = 0
    for network in networkNodes.keys():
        nodes = networkNodes[network]
        # Check if the values in the negative map are above threshold
        for roi in nodes:
            posDC = np.average(posDCData[roiData==roi])
            negDC = np.average(negDCData[roiData==roi])
            if posDC > 1.9:
                
                print(str(roi) + ' in ' + network + ' sig positive ' + str(posDC))
                if not network in networkDict.keys():
                    networkDict[network] = {}
                if not 'pos' in networkDict[network].keys():    
                    networkDict[network]['pos'] = []
                    
                networkDict[network]['pos'].append(posDC)
                totalPos += 1
                
            if  negDC > 1.9:
                print(str(roi) + ' in ' + network + ' sig negative ' + str(negDC))
                if not network in networkDict.keys():
                    networkDict[network] = {}
                if not 'neg' in networkDict[network].keys():    
                    networkDict[network]['neg'] = []
                    
                networkDict[network]['neg'].append(negDC)
                totalNeg += 1
            
    # Now store the information for each network in a textfile
    outString = ''
    for network in networkDict.keys():
        print(network)
        outString = (outString + network + '\n')
        if 'pos' in networkDict[network].keys():
            posPE = networkDict[network]['pos']
            posString = ''
            for pos in posPE:
                posString = (posString + ' ' + str(pos))
            
            outString = (outString + 'signficiant_pos_PE\n' + posString + '\n')
        
        if 'neg' in networkDict[network].keys():
            negPE = networkDict[network]['neg']
            negString = ''
            for neg in negPE:
                negString = (negString + ' ' + str(neg))
                
            outString = (outString + 'signficiant_neg_PE\n' + negString + '\n')
        
    # Save the outstring
    f = open(pathToOutputFile, 'wb')
    f.writelines(outString)
    f.close()
    print('totalPos ' + str(totalPos))
    print('totalNeg ' + str(totalNeg))
    print('Done')
    
if __name__ == '__main__': 
    Main()