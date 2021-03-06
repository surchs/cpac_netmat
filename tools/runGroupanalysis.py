'''
Created on Feb 12, 2013

@author: surchs

script that runs a flameo model from a specified location using the CPAC
group analysis workflow
'''
import os
import re
import sys
import time
import glob
import CPAC
import numpy as np
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as util
import nipype.interfaces.io as nio
import nipype.interfaces.fsl as nipFsl


def getModelFile(modelDir, fileEnding):
    '''
    module returns either the file or FALSE
    '''
    searchString = (modelDir + '*' + fileEnding)
    a = glob.glob(searchString)
    if len(a) == 0:
        raise Exception('\n\nHaven\'t found a file for '
                        + fileEnding + '!!!!!!\n\n')
        modelFile = False
    elif len(a) > 1:
        raise Exception('more than one ' + fileEnding + ' file in ' + modelDir)
        modelFile = a[0]
    else:
        modelFile = a[0]

    return modelFile

def Main(searchDir, modelDir, OutDir, WorkingDir):
    '''
    main module
    '''
    FslDir = '/usr/share/fsl/4.1/'

    # first fetch the model files
    contrastFile = getModelFile(modelDir, 'con')
    designFile = getModelFile(modelDir, 'mat')
    groupFile = getModelFile(modelDir, 'grp')
    subjectList = getModelFile(modelDir, 'txt')
    print('fsl ' + str(type(nipFsl)))
    print('I got this: ' + subjectList)
    time.sleep(4)
    mni = 'MNI152'
    fsl = FslDir

    # first we have to search for the number of rois that are present in the
    # subjects - assuming that they are the same for all subjects

    # for both pipelines
    pipeDict = {}
    # outer level to loop through pipelines
    for pipeline in os.listdir(searchDir):
        # first drop the compcore portion of the pipeline
        if not pipeline == 'linear':
            print('dont want compcor')
            continue

        # keep the seed regions
        seeds = {}
        masks = {}
        print('in pipeline ' + pipeline + ' now!')

        # second level to loop through subjects
        pipeDir = os.path.abspath(os.path.join(searchDir, pipeline))

        # inner loop for the subjects
        for subject in os.listdir(pipeDir):
            # get the name of the subject, discard the session
            findSubBase = re.search(r'[a-zA-Z]*[0-9]*(?=_)', subject)
            subBase = findSubBase.group()
            if subject == 'sub3211_session_1':
                print('found sub3211_session_1')

            subjectDir = os.path.join(pipeDir, subject)
            # find the functional MNI mask
            mniMaskSearch = (subjectDir + '/*.nii.gz')
            mask = glob.glob(mniMaskSearch)
            if len(mask) != 1:
                print('didn\'t find a functional mask for ' + subject)
                continue
            else:
                mniMaskFile = mask[0]
                masks[subject] = mniMaskFile

            # now find all of the ROIs
            searchString = (subjectDir + '/*/z_score/*.nii.gz')
            a = glob.glob(searchString)
            if len(a) == 0:
                    print('no file ROI file found for ' + subject)
                    continue
            else:
                # get the roiNumbers in there
                for roi in a:
                    roiNumber = re.search(r'[0-9]*(?=.nii.gz)', roi).group()

                    # now store the path in the seed dictionary
                    if not roiNumber in seeds.keys():
                        seeds[roiNumber] = {}
                    seeds[roiNumber][subject] = roi


        # end of the pipeline loop. store the seeds in the pipeline directory
        pipeDict[pipeline] = seeds

    # now grab the subject list and go through it line by line
    # throw an error if one of the subjects in the list is not inside the
    # current seed
    subjectFile = open(subjectList, 'rb')
    subjects = subjectFile.readlines()

    # now loop through the pipelines
    groupanalysis = {}
    for pipeline in pipeDict.keys():
        print('Running pipeline ' + pipeline)
        seeds = pipeDict[pipeline]

        # and through the seeds
        for seed in seeds.keys():
            # create a file list that gets populated in the same order as
            # the subject file
            fileList = []
            maskList = []
            print('Running seed ' + seed + ' in pipeline ' + pipeline)
            seedDict = seeds[seed]
            # print('sub3211_session_1 : ' + seedDict.get('sub3211_session_1'))
            # run through subjects
            for subject in subjects:
                # strip newline from subject
                subject = subject.strip()
                # check if subject is in dict for current seed
                if (not subject in seeds[seed].keys()
                    or not subject in masks.keys()):
                    print('subject ' + subject + ' is not in seed or mask '
                          + seed + ' for pipeline ' + pipeline)
                    continue
                subjectFile = seedDict[subject]
                subjectMask = masks[subject]
                # append subject filepath to the list
                fileList.append(subjectFile)
                # append subject funcmask to list
                maskList.append(subjectMask)

            if not pipeline in groupanalysis.keys():
                groupanalysis[pipeline] = {}

            groupanalysis[pipeline][seed] = (fileList, maskList)

    # done with the looping, now we can run
    # now run the different pipelines
    for pipeline in groupanalysis.keys():
        pipeDict = groupanalysis[pipeline]

        for seed in pipeDict.keys():

            (fileList, maskList) = pipeDict[seed]
            print('fileL ' + str(fileList))
            print('maskL ' + str(maskList))
#             groupOutDir = os.path.join(OutDir, pipeline)
            groupOutDir = os.path.join(OutDir, seed)


            if not os.path.isdir(groupOutDir):
                print('Creating ' + groupOutDir)
                os.makedirs(groupOutDir)

            # fwhm in mm
            fwhm = 6
            # precision in decimals
            prec = 6

            wf = pe.Workflow(name='group_analysis')
            wf.base_dir = WorkingDir

            grp_wkf = CPAC.group_analysis.create_group_analysis(ftest=False,
                                                                wf_name=seed)

            ds = pe.Node(nio.DataSink(), name='gpa_sink')
            ds.inputs.base_directory = groupOutDir
            ds.inputs.container = ''
            ds.inputs.regexp_substitutions = [(r'_cluster(.)*[/]', ''),
                                              (r'_slicer(.)*[/]', '')]

            # smoothing
            smoothing = pe.MapNode(interface=nipFsl.MultiImageMaths(),
                                   name='smoothing', iterfield=['in_file',
                                                                'operand_files'])
            # sigma for gaussian
            sigma = np.round(fwhm / 2.3548, prec)
            opString = '-kernel gauss ' + str(sigma) + ' -fmean -mas %s'
            smoothing.inputs.in_file = fileList
            smoothing.inputs.op_string = opString
            smoothing.inputs.operand_files = maskList

            grp_wkf.inputs.inputspec.mat_file = designFile
            grp_wkf.inputs.inputspec.con_file = contrastFile
            grp_wkf.inputs.inputspec.grp_file = groupFile

            # grp_wkf.inputs.inputspec.zmap_files = fileList
            grp_wkf.inputs.inputspec.z_threshold = 2.3
            grp_wkf.inputs.inputspec.p_threshold = 0.05
            grp_wkf.inputs.inputspec.parameters = (fsl, mni)

            wf.connect(smoothing, 'out_file',
                       grp_wkf, 'inputspec.zmap_files')
            wf.connect(grp_wkf, 'outputspec.cluster_threshold',
                       ds, 'thresholded')
            wf.connect(grp_wkf, 'outputspec.rendered_image',
                       ds, 'rendered')


            print('Running the group analysis for seed ' + seed
                  + ' in pipeline ' + pipeline)
            print('con ' + contrastFile)
            print('design ' + designFile)
            print('group ' + groupFile)
            wf.run()


if __name__ == '__main__':
    searchDir = sys.argv[1]
    modelDir = sys.argv[2]
    OutDir = sys.argv[3]
    WorkingDir = sys.argv[4]
    Main(searchDir, modelDir, OutDir, WorkingDir)
    pass
