'''
Created on Feb 21, 2013

@author: surchs
'''
import os
import pandas as pa

# Base path to preprocessed functional files (pathToPipeline)
# using
#    - non scrubbing
#    - non compcor
pathToSymlinks = '/home2/data/Projects/netmat/'
pipelineDir = 'workingdirs/'
strategyDir = 'abide_246/'
# pathToPipeline = os.path.join(pathToSymlinks, pipelineDir, strategyDir)
pathToPipeline = '/home2/data/Projects/netmat/outputdirs/abide_246/sym_links/pipeline_HackettCity/linear1.wm1.global1.motion1.csf1_CSF_0.98_GM_0.7_WM_0.95/'

debugPrefix = 'resting_preproc_'

# Relative path to functional file for each subject
subjectSuffix = '_session_1'
funcDir = 'scan_rest_1_rest/func/bandpass_freqs_0.009.0.08'
funcFile = 'functional_mni.nii.gz'

# Input phenotypic file
pathToPheno = '/home2/surchs/secondLine/configs/abide/abide_across_236_pheno.csv'
# Output functional path file
funcPathsFile = '/home2/surchs/secondLine/configs/abide/pathsToFuncFiles_abide_global.csv'

# Reading phenotypic file and creating list of subjects
pheno = pa.read_csv(pathToPheno)
pheno.astype(str)
subjectList = pheno['SubID'].tolist()

# Create container for functional paths
funcFileList = []

# loop through subjects and look if funcfile is where we expect it to be
# and then append path to functional file to funcFileList
for subject in subjectList:
    subject = ('00' + str(subject))

    # DEBUG DEBUG
    # subject = (debugPrefix + subject)

    subjectDir = os.path.join(pathToPipeline, (subject + subjectSuffix))
    funcPath = os.path.join(subjectDir, funcDir, funcFile)
    if os.path.isfile(funcPath):
        print('got func file for subject ' + subject)
        funcFileList.append(funcPath + '\n')
    else:
        raise Exception('didn\'t find func file for subject ' + subject
                        + ' in ' + funcPath)

# Saving the functional paths to file
print('saving stuff')
f = open(funcPathsFile, 'wb')
f.writelines(funcFileList)
f.close()
