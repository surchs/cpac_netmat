'''
Created on Feb 21, 2013

@author: surchs
'''
import os
import numpy as np


def testDegreeConnectivity():    
    pathToConnectome = '/home2/surchs/secondLine/connectomes/sub3119_connectome.txt'
    # Load the connectome
    connectome = loadConnectome(pathToConnectome)
    # Compute degree centrality
    degreeCentrality = computeDegreeCentrality(connectome)
    pathToOutputFile = '/home2/surchs/secondLine/degree_centrality/test_degree.txt'
    
    status = saveDegreeCentrality(pathToOutputFile, degreeCentrality)
    print('test says ' + status)
    

def loadConnectome(pathToConnectome):
    connectome = np.loadtxt(pathToConnectome)
    
    return connectome

def computeDegreeCentrality(connectome):
    # Take out the diagonal by subtracting 1
    degreeCentrality = np.sum(connectome, axis=0) -1
    
    return degreeCentrality

def stackDegreeCentrality(degreeCentralityStack, degreeCentrality):
    # If this is the first time that the stack is used, initiate
    if degreeCentralityStack.size == 0:
        degreeCentralityStack = degreeCentrality[None, ...]
    else:
        degreeCentralityStack = np.concatenate((degreeCentralityStack,
                                                degreeCentrality[None, ...]),
                                               axis=0)
    
    return degreeCentralityStack

def saveDegreeCentrality(outputFilePath, degreeCentrality):
    np.savetxt(outputFilePath, degreeCentrality, fmt='%.12f')
    status = 'cool'
    
    return status


def Main():
    # Define inputs
    pathToSubjectList = '/home2/surchs/secondLine/configs/subjectList.csv'
    pathToConnectomeDir = '/home2/surchs/secondLine/connectomes'
    # Define input connectome suffix
    connectomeSuffix = '_connectome.txt'
    
    # Define output directory
    pathToOutputDir = '/home2/surchs/secondLine/degree_centrality'
    # Define output file suffix
    outFileSuffix = '_degree_centrality.txt'
    # define group level output file
    groupOutFile = 'group_degree_centrality.txt'
    
    # Read subject list
    subjectListFile = open(pathToSubjectList, 'rb')
    subjectList = subjectListFile.readlines()
    
    
    # Prepare container variable to stack the degree centrality vectors for
    # all subjects (in the same order as in the subject list - along 0 axis)
    degreeCentralityStack = np.array([])
    
    # Loop through the subjects
    for subject in subjectList:
        subject = subject.strip()
        
        # Combine the path to the connectome for the subject
        pathToConnectome = os.path.join(pathToConnectomeDir, (subject 
                                                              + connectomeSuffix))
        # Load the connectome
        connectome = loadConnectome(pathToConnectome)
        # Compute degree centrality
        degreeCentrality = computeDegreeCentrality(connectome)
        # Stack degree centrality in the container variable
        degreeCentralityStack = stackDegreeCentrality(degreeCentralityStack,
                                                      degreeCentrality)
        # Generate output path for degree centrality
        pathToOutputFile = os.path.join(pathToOutputDir, (subject 
                                                          + outFileSuffix))
        status = saveDegreeCentrality(pathToOutputFile, degreeCentrality)
        print('subject ' + subject + ' says ' + status)
        
    # Generate output path for the group level degree centrality stack
    pathToGroupOutputFile = os.path.join(pathToOutputDir, groupOutFile)
    status = saveDegreeCentrality(pathToGroupOutputFile, degreeCentralityStack)
    print('Group output file says ' + status)

    
if __name__ == '__main__': 
    Main()
    