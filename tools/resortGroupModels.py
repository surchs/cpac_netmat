'''
Created on Feb 5, 2013

@author: surchs
'''
import os
import glob
import sys
import distutils.dir_util as dr


def Main(searchDir, outDir):
    '''
    hierarchy:
        - searchDir
        - # pipelines (global / compcor)
        - */*/*/ (_bandpass_freqs_0.009.0.1/_roi_rois_1mm/_fwhm_6)
        - # seeds
        - outputs:
            - model_files
            - rendered
            - stats
                - threshold (threshold)
                - unthreshold (unthreshold)
        - # models
    '''
    pipeRel = '*/*/*/'

    for pipeline in os.listdir(searchDir):
        # inside one pipeline now
        print(pipeline)
        pipeName = os.path.basename(pipeline)
        pipePath = os.path.abspath(os.path.join(searchDir, pipeline))
        pipeSearchString = (pipePath + '/' + pipeRel)
        # search for all rois
        seeds = glob.glob(pipeSearchString + '_sca_roi_Z_to_standard_smooth_*')
        print(str(len(seeds)))
        # now run through the seeds
        for seed in seeds:
            seedName = os.path.basename(seed)
            seedSearchString = (seed + '/')
            modelFileP = (seedSearchString + 'model_files/*')
            renderedFileP = (seedSearchString + 'rendered/*')
            statThresFileP = (seedSearchString + 'stats/threshold/*')
            statUnthresFileP = (seedSearchString + 'stats/unthreshold/*')
            
            # now get the files in these paths
            modelFiles = glob.glob(modelFileP)
            renderedFiles = glob.glob(renderedFileP)
            statThresFiles = glob.glob(statThresFileP)
            statUnthresFiles = glob.glob(statUnthresFileP)

            # and loop over the models inside these paths
            # modelFiles:
            for model in modelFiles:
                modelName = os.path.basename(model)
                outPath = os.path.join(outDir, pipeName, modelName, 
                                       'modelfiles/')

                # copy all the files over
                dr.copy_tree(model, outPath)
                
            for rendered in renderedFiles:
                modelName = os.path.basename(rendered)
                outPath = os.path.join(outDir, pipeName, modelName, 
                                       'rendered/', seedName)
                # copy all the files over
                dr.copy_tree(rendered, outPath)
                
            for statThres in statThresFiles:
                modelName = os.path.basename(statThres)
                outPath = os.path.join(outDir, pipeName, modelName, 
                                       'stats/threshold/', seedName)
                # copy all the files over
                dr.copy_tree(statThres, outPath)
                
            for statUnthresh in statUnthresFiles:
                modelName = os.path.basename(statUnthresh)
                outPath = os.path.join(outDir, pipeName, modelName, 
                                       'stats/unthreshold/', seedName)
                # copy all the files over
                dr.copy_tree(statThres, outPath)
    print('ALl done')


if __name__ == '__main__':
    searchDir = sys.argv[1]
    outDir = sys.argv[2]
    Main(searchDir, outDir)
    pass