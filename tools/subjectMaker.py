'''
Created on Nov 16, 2012

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
    (outDir, tempSub) = args
    # generate the derivatives from the subject
    tempSub.makeDerivative()
    # generate the path
    outPath = os.path.join(outDir, tempSub.pipeline, tempSub.name)
    # check if dir exists
    if not os.path.isdir(outPath):
        os.makedirs(outPath)
    # generate outName
    outName = (tempSub.name + '_.sub')
    # save the subject
    savePath = os.path.join(outPath, outName)
    dumpFile = gzip.open(savePath, 'wb')
    cPickle.dump(tempSub, dumpFile, 2)
    dumpFile.close()

    return (tempSub.pipeline + ' ' + ' ' + tempSub.name
            + ' ' ' was run successfully')


def Main(searchDir, templateFile, phenoFile, maskDir, outDir, nProcs):
    # template string to search for
    pathFile = open(templateFile)

    masks = {}
    maskNames = os.listdir(maskDir)
    for mask in maskNames:
        maskPath = os.path.abspath(os.path.join(maskDir, mask))
        masks[mask] = maskPath

    phenoIndex = {}
    phenoLine = open(phenoFile, 'rb').readline()

    loopLine = phenoLine.strip().split(',')
    run = 0
    for pheno in loopLine:
        # check if it is the subject designator which is not a phenotypic info
        if not pheno == 'subject':
            phenoIndex[pheno] = run

        run += 1

    pipeDict = {}

    # outer level to loop through pipelines
    for pipeline in os.listdir(searchDir):
        # generate pipeline dict entry
        if not pipeline in pipeDict.keys():
            pipeDict[pipeline] = {}

        # second level to loop through subjects
        subjectDir = os.path.abspath(os.path.join(searchDir, pipeline))

        # inner loop for the subjects
        for subject in os.listdir(subjectDir):
            # get the name of the subject, discard the session
            findSubBase = re.search(r'[a-zA-Z]*[0-9]*(?=_)', subject)
            subBase = findSubBase.group()

            # generate new subject file
            tempSub = pp.base.Subject(subBase, pipeline)

            # first within subject loop: add phenotypic data
            if subBase in open(phenoFile).read():
                # all is good
                print('found ' + subBase + ' in phenofile. Yeah!')
                # loop through the file again
                for line in open(phenoFile, 'rb'):
                    # search for the correct line
                    # currently requires the subject name to be given at
                    # the start
                    if line.startswith(subBase):
                        subLine = line.strip().split(',')

            else:
                # no phenotypic information, presently we skip these subjects
                print('didn\'t find ' + subBase + ' in phenofile.'
                      + ' skipping for now...')
                continue

            # now loop through the subject line and correctly assign the pheno
            # data to the subject object attribute
            for pheno in phenoIndex.keys():
                tempSub.pheno[pheno] = subLine[phenoIndex[pheno]]

            # second within subject loop: add derivatives
            # to loop through the derivative paths in the template
            # file we have to reset the file line position for every subject
            pathFile.seek(0)
            for line in pathFile.readlines():
                tempLine = line.strip().split()
                deriName = tempLine[0]
                deriPath = tempLine[1]
                deriFile = tempLine[2]
                # check if a mask has been given - not yet implemented
                if len(tempLine) > 3:
                    maskName = tempLine[3]

                # search for the derivative
                searchString = (subjectDir + '/' + subject + deriPath
                                + deriFile)
                a = glob.glob(searchString)
                if len(a) == 0:
                    print(subBase + ' ' + pipeline + ' ' + deriName
                          + ' doesn\'t exist')
                else:
                    tempDeriPath = a[0]
                    tempSub.addDerivativePath(deriName, tempDeriPath)

            # third within subject loop: add masks
            for mask in masks:
                maskPath = masks[mask]
                tempSub.loadMask(maskPath)

            # and finally generate subject name and store subject
            if not subBase in pipeDict[pipeline].keys():
                pipeDict[pipeline][subBase] = tempSub

    # now we loop through the subjects once more to make the derivatives which
    # takes some time. To save time, we will run this in parallel
    for pipeline in pipeDict.keys():
        print('\nRunning subjects inside pipeline ' + pipeline + ' now.')

        # now we can multiprocess through the subjects inside the dictionary
        runList = []
        for subject in pipeDict[pipeline].keys():
            subFile = pipeDict[pipeline][subject]
            runList.append((outDir, subFile))

        pool = mp.Pool(processes=nProcs)
        resultList = pool.map(runDerivative, runList)


if __name__ == '__main__':
    searchDir = sys.argv[1]
    templateFile = sys.argv[2]
    phenoFile = sys.argv[3]
    maskDir = sys.argv[4]
    outDir = sys.argv[5]
    nProcs = int(sys.argv[6])
    Main(searchDir, templateFile, phenoFile, maskDir, outDir, nProcs)
    pass
