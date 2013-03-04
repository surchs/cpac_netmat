'''
Created on Feb 23, 2013

@author: surchs
'''
import os
import numpy as np
import pandas as pa
import nibabel as nib
from scipy import stats as st
import statsmodels.api as sm


def readCsvFile(pathToCsvFile):
    csvFile = pa.read_csv(pathToCsvFile)
    
    return csvFile


def loadDegreeCentralityStack(pathToDegreeCentralityStack):
    degreeCentralityStack = np.loadtxt(pathToDegreeCentralityStack)
    
    return degreeCentralityStack


def loadNiftiImage(pathToNiftiFile):
    image = nib.load(pathToNiftiFile)
    data = image.get_data()
    
    return image, data


def stackAges(ageStack, age):
    ageStack = np.append(ageStack, age)
    
    return ageStack


def runGLM(roiVector, predictorMatrix):
    # run a glm with only one factor
    model = sm.OLS(roiVector, predictorMatrix)
    results = model.fit()
    ageTValue = results.tvalues[0]
    posAgePValue = st.t.sf(ageTValue, results.df_resid)
    negAgePValue = st.t.sf(ageTValue * -1, results.df_resid)
    print('happy')
    return posAgePValue, negAgePValue
    

def drawOnROI(roiData, pVector):
    newRoiData = np.zeros_like(roiData, dtype=float)
    uniqueValues = np.unique(roiData[roiData!=0])
    print(str(len(pVector)) + ' ' + str(len(uniqueValues)))
    for i, roi in enumerate(uniqueValues):
        newRoiData[roiData==roi] = pVector[i]
        print('New value for roi ' + str(roi) + ' is ' + str(pVector[i]))
        
    return newRoiData


def saveNiftiImage(outputData, pathToOutputFile, template):
    outNifti = nib.Nifti1Image(outputData, template.get_affine(),
                               template.get_header())
    nib.nifti1.save(outNifti, pathToOutputFile)
    
    return 'alles save'
        

def Main():
    # Define inputs
    pathToSubjectList = '/home2/surchs/secondLine/configs/subjectList.csv'
    pathToGroupCentralityFile = '/home2/surchs/secondLine/degree_centrality/group_degree_centrality.txt'
    pathToPhenotypicFile = '/home2/surchs/secondLine/configs/pheno81_uniform.csv'
    pathToNuisanceFile = '/home2/surchs/secondLine/configs/nuisance_file.csv'
    pathToRoiFile = '/home2/surchs/masks/ROIs/craddock200wave_p1l.nii.gz'
    pathToMatrix = '/home2/surchs/secondLine/supermatrix.txt'

    # Define output directory
    positivePMap = '/home2/surchs/secondLine/degree_centrality/degree_centrality_volumes/group_p_age_pos.nii.gz'
    negativePMap = '/home2/surchs/secondLine/degree_centrality/degree_centrality_volumes/group_p_age_neg.nii.gz'
    
    # Read subject list
    subjectListFile = open(pathToSubjectList, 'rb')
    subjectList = subjectListFile.readlines()

    # Load the degree centrality stack
    degreeCentralityStack = loadDegreeCentralityStack(pathToGroupCentralityFile)
    
    # Load roi image
    roiImage, roiData = loadNiftiImage(pathToRoiFile)

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
    pheno = readCsvFile(pathToPhenotypicFile)
    phenoSubjects = pheno['subject'].tolist()
    phenoAges = pheno['age'].tolist()
    
    # Read the nuisance file
    nuisance = readCsvFile(pathToNuisanceFile)
    # For some reason, the subject names are treated as the index in the 
    # nuisance file
    nuisanceSubjects = nuisance['Subject'].tolist()
     
    # Prepare container variables for age and meanFD
    ageStack = np.array([])
    mfdStack = np.array([])

    # Loop through the subjects
    for i, subject in enumerate(subjectList):
        subject = subject.strip()
        phenoSubject = phenoSubjects[i]
        print('Reading in subject ' + subject)
        
        if not subject == phenoSubject:
            raise Exception('The Phenofile returned a different subject name '
                            + 'than the subject list:\n'
                            + 'pheno: ' + phenoSubject + ' subjectList ' 
                            + subject)
            
        # Get the age of the subject from the pheno file
        phenoAge = phenoAges[i]
        # Stack ages
        ageStack = stackAges(ageStack, phenoAge)
        
        # Get meanFD for the current subject by looping through the indices
        # of the nuisance file and looking for the current subject
        if any(subject in sub for sub in nuisanceSubjects):
            nuisanceSubject = (subject + '_session_1')
            meanFD = nuisance.xs(nuisanceSubject)['MeanFD']
            print(str(meanFD))
            mfdStack = np.append(mfdStack, meanFD)
        else:
            raise Exception(subject + ' is not in the nuisance file')
            
    # Concatenate the age and the meanFD vectors into a matrix
    predictorMatrix = np.concatenate((ageStack[..., None], mfdStack[..., None]),
                                     axis=1)
    # quickly save the predictor matrix (age, nuisance)
    np.savetxt(pathToMatrix, predictorMatrix)
    
    # Prepare container for the positive and negative p values
    posAgePValues = np.array([])
    negAgePValues = np.array([])
    
    # Now run a glm for each ROI/Voxel
    for roiIndex in np.arange(numDcROIs):
        roiVector = degreeCentralityStack[..., roiIndex]
        print('roiVec.shape ' + str(roiVector.shape) + ' ageVec.shape '
              + str(ageStack.shape))
        posAgePValue, negAgePValue = runGLM(roiVector, predictorMatrix)
        posAgePValues = np.append(posAgePValues, posAgePValue)
        negAgePValues = np.append(negAgePValues, negAgePValue)
        
        
    # Draw on the ROI maps and save them out
    positiveROImap = drawOnROI(roiData, posAgePValues)
    negativeROImap = drawOnROI(roiData, negAgePValues)
    
    # Save the new ROI maps
    status = saveNiftiImage(positiveROImap, positivePMap, roiImage)
    print('Positive save says ' + status)
    status = saveNiftiImage(negativeROImap, negativePMap, roiImage)
    print('Negative save says ' + status)
    
    
if __name__ == '__main__':
    Main()
    pass