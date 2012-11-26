'''
Created on Oct 26, 2012

@author: sebastian

a short script to generate subject archives from the CPAC output directory
and store the result on disk in a folder
'''
import re
import os
import sys
import glob
import gzip
import cPickle
import multiprocessing as mp
import cpac_processing.preprocessing as pp


def runDerivative(args):
    (outDir, subject) = args
    # generate the derivatives from the subject
    tempSub = subject['sub']
    tempSub.makeDerivative()
    # generate the path
    outPath = os.path.join(outDir, tempSub.pipeline, tempSub.name)
    # check if dir exists
    if not os.path.isdir(outPath):
        os.makedirs(outPath)
    # generate outName
    outName = (tempSub.name + '_' + tempSub.mask.name + '.sub')
    # save the subject
    savePath = os.path.join(outPath, outName)
    dumpFile = gzip.open(savePath, 'wb')
    cPickle.dump(tempSub, dumpFile, 2)

    return (tempSub.pipeline + ' ' + tempSub.mask.name + ' ' + tempSub.name
            + ' ' ' was run successfully')


def Main(searchDir, templateFile, phenoFile, maskDir, outDir):
    # template string to search for
    pathFile = open(templateFile)
    pathLines = pathFile.readlines()

    masks = {}
    maskNames = os.listdir(maskDir)
    for mask in maskNames:
        maskPath = os.path.abspath(os.path.join(maskDir, mask))
        masks[mask] = maskPath

    pipeDict = {}

    # outer level loop for the parameters
    for line in pathLines:
        tempLine = line.strip().split()
        deriName = tempLine[0]
        deriPath = tempLine[1]
        deriFile = tempLine[2]
        # check if a mask has been given - not yet implemented
        if len(tempLine) > 3:
            maskName = tempLine[3]

        # second level loop for the pipelines
        for pipeline in os.listdir(searchDir):
            # generate pipeline dict entry
            if not pipeline in pipeDict.keys():
                pipeDict[pipeline] = {}

            subjectDir = os.path.abspath(os.path.join(searchDir, pipeline))

            # inner loop for the subjects
            for subject in os.listdir(subjectDir):
                # get the name of the subject, discard the session
                findSubBase = re.search(r'[a-zA-Z]*[0-9]*(?=_)', subject)
                subBase = findSubBase.group()

                # generate subject name
                if not subBase in pipeDict[pipeline].keys():
                    pipeDict[subBase] = {}

                # search for the derivative
                searchString = (subjectDir + '/' + subject
                                + deriPath + deriFile)
                a = glob.glob(searchString)
                # add entry to pipeline dict under correct derivative
                pipeDict[pipeline][deriName][subBase] = a[0]

                # generate derivative dict entry
                if not deriName in pipeDict[pipeline].keys():
                    pipeDict[pipeline][deriName] = {}

    # now we have the dictionaries full of the paths and can proceed to
    # generate the subject files and reorder the dict

    subPipe = {}
    phenoIndex = {}
    phenoLine = open(phenoFile, 'rb').readline()

    loopLine = phenoLine.strip().split(',')
    run = 0
    for pheno in loopLine:
        # check if it is the subject designator which is not a phenotypic info
        if not pheno == 'subject':
            phenoIndex[pheno] = run

        run += 1

    for pipeline in pipeDict.keys():

        for deriName in pipeDict[pipeline].keys():

            for subject in pipeDict[pipeline][deriName].keys():
                # look up the phenotypic information for this subject
                # and just skip it if you can't find it
                if subject in open(phenoFile).read():
                    # all is good
                    print('found ' + subject + ' in phenofile. Yeah!')
                    # loop through the file again
                    for line in open(phenoFile, 'rb'):
                        # search for the correct line
                        # currently requires the subject name to be given at
                        # the start
                        if line.startswith(subject):
                            subLine = line.strip().split(',')
                else:
                    # nope, isn't there. skip it for now...
                    print('didn\'t find ' + subject + ' in phenofile.'
                          + ' skipping for now...')
                    continue

                for mask in maskNames:
                    # reorder and store the dict
                    if not pipeline in subPipe.keys():
                        subPipe[pipeline] = {}

                    if not mask in subPipe[pipeline].keys():
                        subPipe[pipeline][mask] = {}

                    if not subject in subPipe[pipeline][mask].keys():
                        subPipe[pipeline][mask][subject] = {}

                    # fork out the subject dict for readability
                    subDict = subPipe[pipeline][mask][subject]
                    if not 'ranDeri' in subDict.keys():
                        subDict['ranDeri'] = []
                    if not 'sub' in subDict.keys():
                        # create subject
                        tempSub = pp.base.Subject(subject, pipeline)
                        # add current mask to tempSub
                        tempSub.loadMask(masks[mask])
                        # store tempSub
                        subDict['sub'] = tempSub
                    if not 'ranPheno' in subDict.keys():
                        subDict['ranPheno'] = []

                    for pheno in phenoIndex.keys():
                        if not pheno in subDict['ranPheno']:
                            tempSub.pheno[pheno] = subLine[phenoIndex[pheno]]
                            subDict['ranPheno'].append(pheno)

                    if not deriName in subDict['ranDeri']:
                        # fetch the subject
                        tempSub = subDict['sub']
                        # fetch path
                        path = pipeDict[pipeline][deriName][subject]
                        # add current derivative to tempSub
                        tempSub.addDerivativePath(deriName, path)
                        subDict['sub'] = tempSub
                        subDict['ranDeri'].append(deriName)
                        subPipe[pipeline][mask][subject] = subDict
                    else:
                        print 'something wrong'

    print subPipe.keys()

    for pipeline in subPipe.keys():
        # pipeline
        maskDict = subPipe[pipeline]

        for mask in maskDict.keys():
            # derivative
            runList = []
            subDict = maskDict[mask]

            for subject in subDict.keys():
                subFile = subDict[subject]
                runList.append((outDir, subFile))

            pool = mp.Pool(processes=1)
            print type(subDict.values())
            resultList = pool.map(runDerivative, runList)
            print resultList

    pass

if __name__ == '__main__':
    searchDir = sys.argv[1]
    templateFile = sys.argv[2]
    phenoFile = sys.argv[3]
    maskDir = sys.argv[4]
    outDir = sys.argv[5]
    Main(searchDir, templateFile, phenoFile, maskDir, outDir)
    pass
