'''
Created on Feb 22, 2013

@author: surchs
'''
import os
import numpy as np
import pandas as pa
from scipy import stats as st


def loadPhenotypicFile(pathToPhenotypicFile):
    pheno = pa.read_csv(pathToPhenotypicFile)
    
    return pheno


def loadConnectome(pathToConnectomeFile):
    connectome = np.loadtxt(pathToConnectomeFile)
    
    return connectome


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

def correlateConnectomeWithAge(connectomeStack, ageStack):
    # First I flatten the connectomeStack
    stackShape = connectomeStack.shape
    # Get the number of connections to correlate
    numberOfConnections = stackShape[0] * stackShape[1]
    numberOfTimepoints = stackShape[2]
    # Reshape the stack to number of correlations by number of timepoints
    flatConnectomeStack = np.reshape(connectomeStack,
                                     (numberOfConnections, 
                                      numberOfTimepoints))
    
    # Prepare container variables for correlation and p values for each 
    # connection
    correlationVector = np.array([])
    pValueVector = np.array([])
    
    # Iterate over the elements in the stack and correlate them to the age
    # stack one by one
    for connectionIndex in np.arange(numberOfConnections):
        # Get the vector of connection values across subjects for current
        # connection
        connectionVector = flatConnectomeStack[connectionIndex, :]
        # Correlate the vector to age
        corr, p = st.pearsonr(connectionVector, ageStack)
        # Append correlation and p values to their respective container
        # variables
        correlationVector = np.append(correlationVector, corr)
        pValueVector = np.append(pValueVector, p)
        
    # Reshape the results of the correlation back into the shape of the 
    # original connectivity matrix
    correlationMatrix = np.reshape(correlationVector, 
                                   (stackShape[0], stackShape[1]))
    pValueMatrix = np.reshape(pValueVector, 
                              (stackShape[0], stackShape[1]))
    
    return correlationMatrix, pValueMatrix


def prepareFDR(pValueMatrix):
    # This returns a vector of p-values
    aidMask = np.ones_like(pValueMatrix)
    lowerTriangle = np.tril(aidMask, -1)
    independentPValues = pValueMatrix[lowerTriangle==1]
    
    return independentPValues

        
def computeFDR(pValueVector, alpha):
    # This returns a new thresholded p value
    # Sort the p values by size, beginning with the smallest
    sortedP = np.sort(pValueVector)
    # Reverse sort the p-values, so the first one is the biggest
    reverseP = sortedP[::-1]
    # Get the number of p-values
    numP = float(len(reverseP))
    # Create a vector designating the position of the reverse sorted p-values
    # in the sorted p vector (e.g. the first reverse sorted p-value will have
    # the index numP because it would be the last entry in the sorted vector)
    indexP = np.arange(numP, 0, -1)
    # Create test vector of (index of p value / number of p values) * alpha
    test = indexP/numP * alpha
    # Check where p-value <= test
    testIndex = np.where(reverseP <= test)
    if testIndex[0].size == 0:
        raise Exception('None of you p values pass FDR correction')
    # Get the first p value that passes the criterion
    pFDR = reverseP[np.min(testIndex)]
    
    return pFDR
    
    
    
    
    # now return the p vector
    pOutVector = np.ones_like(pValueVector)
    pOutVector[:cutK] = pValueVector[:cutK]
    
    print(pOutVector)
    return pOutVector
    

def saveOutput(outputFilePath, outputMatrix):
    np.savetxt(outputFilePath, outputMatrix, fmt='%.12f')
    status = 'cool'
    
    return status


def Main():
    # Define the inputs
    pathToConnectomeDir = '/home2/surchs/secondLine/connectomes'
    pathToPhenotypicFile = '/home2/surchs/secondLine/configs/pheno81_uniform.csv'
    pathToSubjectList = '/home2/surchs/secondLine/configs/subjectList.csv'
    
    connectomeSuffix = '_connectome.txt'
    
    # Define parameters
    alpha = 0.9
    
    # Define the outputs
    pathToCorrelationMatrix = '/home2/surchs/secondLine/correlation/correlation_matrix.txt'
    pathToPValueMatrix = '/home2/surchs/secondLine/correlation/pvalue_matrix.txt'
    pathToThresholdedMatrix = '/home2/surchs/secondLine/correlation/thresholded_matrix.txt'
    
    # Read subject list
    subjectListFile = open(pathToSubjectList, 'rb')
    subjectList = subjectListFile.readlines()
    
    # Read the phenotypic file
    pheno = loadPhenotypicFile(pathToPhenotypicFile)
    phenoSubjects = pheno['subject'].tolist()
    phenoAges = pheno['age'].tolist()
    
    # Prepare container variables for the connectome and for age
    connectomeStack = np.array([])
    ageStack = np.array([])
    
    # Loop through the subjects
    for i, subject in enumerate(subjectList):
        subject = subject.strip()
        phenoSubject = phenoSubjects[i]
        
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
        print('connectome: ' + str(connectome.shape))
        
        # Stack the connectome
        connectomeStack = stackConnectome(connectomeStack, connectome)
        print('connectomeStack: ' + str(connectomeStack.shape))
        # Stack ages
        ageStack = stackAges(ageStack, phenoAge)
        
    # Check the shapes of age and connectivity stacks
    print('ageStack.shape: ' + str(ageStack.shape) + ' connStack.shape: '
          + str(connectomeStack.shape))
    # Compute the correlations with age
    correlationMatrix, pValueMatrix = correlateConnectomeWithAge(connectomeStack,
                                                                 ageStack)
    
    # Threshold the pValueMatrix with FDR
    thresholdedMatrix = computeFDR(pValueMatrix, alpha)
    
    # save the outputs
    status = saveOutput(pathToCorrelationMatrix, correlationMatrix)
    print('correlation matrix says ' + status)
    status = saveOutput(pathToPValueMatrix, pValueMatrix)
    print('p value matrix says ' + status)
    status = saveOutput(pathToThresholdedMatrix, thresholdedMatrix)
    print('thresholded matrix says ' + status)    
        
if __name__ == '__main__': 
    Main()    
    