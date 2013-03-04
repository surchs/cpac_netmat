'''
Created on Feb 11, 2013

@author: surchs

the purpose of this script is to run the SC analysis that is embedded in CPAC
and get the maps based on either MNI or native funcfiles.

Basically, we only pick up the mni or native funcfile and pass it on to the 
CPAC workflow
'''
import os
import re
import sys
import CPAC
import glob
import numpy as np
import shutil as sl
import nipype.pipeline.engine as pe


def getIndexDict(covariateFile):
    '''
    module to retrieve a dictionary where each key corresponds to a header 
    label in the csv file and each value corresponds to the number of the 
    column (starting with 0) of this label
    
    currently the subject designator is skipped so all the keys inside the
    dictionary should correspond to covariates of interest
    '''
    covariateIndex = {}
    covariateLine = open(covariateFile, 'rb').readline()
    covariateLoop = covariateLine.strip().split(',')
    
    run = 0
    for covariate in covariateLoop:
        # check if it is the subject designator which is not a phenotypic info
        if not covariate == 'subject':
            covariateIndex[covariate] = run

        run += 1
        
    return covariateIndex


def Main(searchDir, outDir, modelDir):
    '''
    Main method
    '''
    # define the paths:
    pathToRoiConfig = ''
    roiLinear = 'roi_timeseries/_scan_func_rest/_csf_threshold_0.98/_gm_threshold_0.7/_wm_threshold_0.98/_compcor_ncomponents_5_selector_pc10.linear1.wm1.global1.motion1.quadratic0.gm0.compcor0.csf1/_bandpass_freqs_0.009.0.1/_roi_rois_1mm/'
    roiCompcor = 'roi_timeseries/_scan_func_rest/_csf_threshold_0.98/_gm_threshold_0.7/_wm_threshold_0.98/_compcor_ncomponents_5_selector_pc10.linear1.wm0.global0.motion1.quadratic0.gm0.compcor1.csf0/_bandpass_freqs_0.009.0.1/_roi_rois_1mm/'
    # roiPreStrat = 'roi_timeseries/_scan_func_rest/_csf_threshold_0.98/_gm_threshold_0.7/_wm_threshold_0.98/'
    # roiPostStrat = '_bandpass_freqs_0.009.0.1/_roi_rois_1mm/'
    roiFile = 'roi_rois_1mm.1D'
    
    
    funcLinear = 'functional_mni/_scan_func_rest/_csf_threshold_0.98/_gm_threshold_0.7/_wm_threshold_0.98/_compcor_ncomponents_5_selector_pc10.linear1.wm1.global1.motion1.quadratic0.gm0.compcor0.csf1/_bandpass_freqs_0.009.0.1/'
    funcCompcor = 'functional_mni/_scan_func_rest/_csf_threshold_0.98/_gm_threshold_0.7/_wm_threshold_0.98/_compcor_ncomponents_5_selector_pc10.linear1.wm0.global0.motion1.quadratic0.gm0.compcor1.csf0/_bandpass_freqs_0.009.0.1/'
    # funcPreStrat = 'functional_mni/_scan_func_rest/_csf_threshold_0.98/_gm_threshold_0.7/_wm_threshold_0.98/'
    # funcPostStrat = '_bandpass_freqs_0.009.0.1'
    funcFile = 'bandpassed_demeaned_filtered_warp.nii.gz'
    
    funcLinearMaskMni = 'functional_brain_mask_to_standard/_scan_func_rest/'
    funcMaskMniFile = '*_3dc_tshift_RPI_3dv_automask_warp.nii.gz'
    
    # Define parameters
    FslDir = '/usr/share/fsl/4.1/'
    
    # Define output path
    pathToRoiDump = ''
    
    phenoIndex = getIndexDict(phenoFile)
    nuisanceIndex = getIndexDict(nuisanceFile)
    
    dataAge = np.array([])
    dataMeanFd = np.array([])
    subjectDict = {}
    
    # First: generate the ROI files
    print('Creating ROI files from ' + pathToRoiConfig)
    pathToRoiFile = CPAC.utils.create_seeds_(pathToRoiDump, 
                                             pathToRoiConfig, 
                                             FslDir)
    print('These are the ROI files: ' + pathToRoiFile)
    
    # Next: define the output directory for the sca
    linearOutputDir = os.path.join(outDir, 'linear')
    compcorOutputDir = os.path.join(outDir, 'compcor')
    
    # first loop for the subjects
    for subject in os.listdir(searchDir):
        subjectDir = os.path.abspath(os.path.join(searchDir, subject))
        # get the name of the subject, discard the session
        findSubBase = re.search(r'[a-zA-Z]*[0-9]*(?=_)', subject)
        subBase = findSubBase.group()
        
        # now get the paths for the mni func file
        funcLinearPath = os.path.join(subjectDir, funcLinear)
        funcCompcorPath = os.path.join(subjectDir, funcCompcor)
        
        funcMaskMniPath = os.path.join(subjectDir, funcLinearMaskMni)
        
        funcLinearFile = (funcLinearPath + funcFile)
        funcCompcorFile = (funcCompcorPath + funcFile)
        
        linearOut = os.path.join(linearOutputDir, subject)
        compcorOut = os.path.join(compcorOutputDir, subject)
        
        # now the mask files can have strange names so we have to search for
        # them        
        funcMaskMniSearch = (funcMaskMniPath + funcMaskMniFile)
        a = glob.glob(funcMaskMniSearch)
        if len(a) != 1:
            # something is wrong, either more or less than one
            print('no good functional mask for subject ' + subject)
            continue
        else:
            funcMaskMniPath = a[0]  
            # and copy this shit over!
            fileName = (subject + '_functional_mni_mask.nii.gz')
            outPath = (linearOut)
            out = os.path.join(outPath, fileName)
            sl.copyfile(funcMaskMniPath, out)
        
        print linearOut
        print compcorOut
        
        # check if the folder exists and create it if not
        if not os.path.isdir(linearOut):
            print('creating ' + linearOut)
            os.makedirs(linearOut)
            
        if not os.path.isdir(compcorOut):
            print('creating ' + compcorOut)
            os.makedirs(compcorOut)
            
        # Create SCA workflow
        sca_wf = pe.Workflow(name='sca_analysis')
        # run the timeseries extraction 
        
        # run linear workflow:
        sca_linear = CPAC.sca.create_sca((subject + '_linear'))
        sca_linear.base_dir = linearOut
        sca_linear.inputs.inputspec.functional_file = funcLinearFile
        sca_linear.inputs.inputspec.timeseries_one_d = roiLinearFile
        print('Running linear SCA for subject ' + subject)
        sca_linear.run()
        print('Done running linear SCA for subject ' + subject)
        

        # run compcor workflow
        sca_compcor = CPAC.sca.create_sca((subject + '_compcor'))
        sca_compcor.base_dir = compcorOut
        sca_compcor.inputs.inputspec.functional_file = funcCompcorFile
        sca_compcor.inputs.inputspec.timeseries_one_d = roiCompcorFile
        print('Running compcor SCA for subject ' + subject)
        sca_compcor.run()
        print('Done running compcor SCA for subject ' + subject)

    # done running the SCA analysis
    
    print('Done with everything')     

if __name__ == '__main__':
    searchDir = sys.argv[1]
    phenoFile = sys.argv[2]
    nuisanceFile = sys.argv[3]
    outDir = sys.argv[4]
    modelDir = sys.argv[5]
    Main(searchDir, outDir, modelDir)
    pass