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
pathToSymlinks = '/home2/data/Projects/wave_uniform/output_dir/sym_links/'
pipelineDir = 'pipeline_HackettCity/'
strategyDir = '_compcor_ncomponents_5_linear1.motion1.compcor1.CSF_0.98_GM_0.7_WM_0.98/'
pathToPipeline = os.path.join(pathToSymlinks, pipelineDir, strategyDir)

# Relative path to functional file for each subject
subjectSuffix = '_session_1'
funcDir = 'scan_func_rest/func/bandpass_freqs_0.009.0.1'
funcFile = 'functional_mni.nii.gz'

# Input phenotypic file
pathToPheno = '/home2/surchs/secondLine/configs/wave/wave_pheno81_compcor.csv'
# Output functional path file
funcPathsFile = '/home2/surchs/secondLine/configs/wave/pathsToFuncFiles_wave_compcor.csv'

# Reading phenotypic file and creating list of subjects
pheno = pa.read_csv(pathToPheno)
pheno.astype(str)
subjectList = pheno['subject'].tolist()

# Create container for functional paths
funcFileList = []

# loop through subjects and look if funcfile is where we expect it to be
# and then append path to functional file to funcFileList
for subject in subjectList:
    # subject = ('00' + str(subject))
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
