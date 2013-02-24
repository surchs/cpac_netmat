'''
Created on Feb 23, 2013

@author: surchs
'''
import os
import numpy as np
import pandas as pa
from sklearn import linear_model as lin


def loadPhenotypicFile(pathToPhenotypicFile):
    pheno = pa.read_csv(pathToPhenotypicFile)
    
    return pheno


def loadDegreeCentralityStack(pathToDegreeCentralityStack):
    degreeCentralityStack = np.loadtxt(pathToDegreeCentralityStack)
    
    return degreeCentralityStack


def stackAges(ageStack, age):
    ageStack = np.append(ageStack, age)
    
    return ageStack


def runGLM(roiVector, phenoVector):
    # run a glm with only one factor
    glm = lin.LinearRegression()
    glm.fit(roiVector, phenoVector)
    

def Main():
    # Define inputs
    pathToSubjectList = '/home2/surchs/secondLine/configs/subjectList.csv'
    pathToGroupCentralityFile = '/home2/surchs/secondLine/degree_centrality/group_degree_centrality.txt'
    pathToPhenotypicFile = '/home2/surchs/secondLine/configs/pheno81_uniform.csv'

    # Define output directory
    pathToOutputDir = '/home2/surchs/secondLine/degree_centrality/model/'
    # Define output file suffix
    outFileSuffix = '_degree_centrality_model.txt'
    # Define group level output file
    groupOutFile = 'group_degree_centrality.txt'

    # Read subject list
    subjectListFile = open(pathToSubjectList, 'rb')
    subjectList = subjectListFile.readlines()
    
    # Load the degree centrality stack
    degreeCentralityStack = loadDegreeCentralityStack(pathToGroupCentralityFile)
    
    # Check if the number of elements in the DC stack and the subject list 
    # match up
    numDcROIs = degreeCentralityStack.shape[1]
    numDcSubjects = degreeCentralityStack.shape[0]
    numSubjects = len(subjectList)
    if not numDcSubjects == numSubjects:
        raise Exception('The number of subjects in the subject list and the '
                        + 'dc stack are different\n'
                        + 'subList: ' + str(numSubjects) 
                        + ' dcStack: ' + str(numDcSubjects))
        
    # Read the phenotypic file
    pheno = loadPhenotypicFile(pathToPhenotypicFile)
    phenoSubjects = pheno['subject'].tolist()
    phenoAges = pheno['age'].tolist()
     
    # Prepare container variables for age
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
        # Stack ages
        ageStack = stackAges(ageStack, phenoAge)
        
    # Now run a glm for each ROI/Voxel
    for roiIndex in np.arange(numDcROIs):
        roiVector = degreeCentralityStack[..., roiIndex]
        runGLM(roiVector, ageStack)
    
if __name__ == '__main__':
    pass