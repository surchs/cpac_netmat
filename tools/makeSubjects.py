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


def Main(searchDir, templateFile, phenoPath, maskDir, outDir, nProcs):
    # template string to search for
    pathFile = open(templateFile)
    pathLines = pathFile.readlines()

    masks = {}
    maskNames = os.listdir(maskDir)
    for mask in maskNames:
        maskPath = os.path.abspath(os.path.join(maskDir, mask))
        masks[mask] = maskPath

    pipeDict = {}
    # prepare pheno data
    phenoFile = open(phenoPath, 'rb')
    # get the first line
    phenoLine = phenoFile.readline()
    # get all the remaining lines
    phenoList = phenoFile.readlines()
    # reset reader header
    phenoFile.seek(0)
    # and read the whole thing into one huge string
    phenoString = phenoFile.read()

    # prepare the indexDictionary for the phenotypic information
    phenoIndex = {}
    loopLine = phenoLine.strip().split(',')
    run = 0
    for pheno in loopLine:
        # check if it is the subject designator which is not a phenotypic info
        if not pheno == 'subject':
            phenoIndex[pheno] = run
        run += 1

    # outer level for the pipelines
    for pipeline in os.listdir(searchDir):
        # check if pipeline is in pipeline dictionary
        if not pipeline in pipeDict.keys():
            pipeDict[pipeline] = {}
        subjectDir = os.path.abspath(os.path.join(searchDir, pipeline))
        #

        # second level for subjects
        for subject in os.listdir(subjectDir):
            # get the name of the subject, discard the session
            findSubBase = re.search(r'[a-zA-Z]*[0-9]*(?=_)', subject)
            subBase = findSubBase.group()
            # create subject
            tempSub = pp.base.Subject(subBase, pipeline)

            # level 2.1 for phenotypic information
            if subBase in phenoString:
                # all is good
                print('found ' + subBase + ' in phenofile. Yeah!')

                for line in phenoList:
                    # search for the correct line
                    # currently requires the subject name to be given at
                    # the start - and only once in the document
                    if line.startswith(subject):
                        # this is the correct line
                        subLine = line.strip().split(',')
                        # loop through the index dictionary and assign the
                        # phenotypic information
                        for pheno in phenoIndex.keys():
                            tempPheno = subLine[phenoIndex[pheno]]
                            tempSub.phen[pheno] = tempPheno
                    else:
                        # not the right line, just continue
                        continue

            else:
                # nope, isn't there. skip the subject
                print('didn\'t find ' + subject + ' in phenofile.'
                      + ' skipping for now...')
                continue

            # level 2.2 for derivatives
            for line in pathLines:
                tempLine = line.strip().split()
                deriName = tempLine[0]
                deriPath = tempLine[1]
                deriFile = tempLine[2]
                # check if a mask has been given - not yet implemented
                if len(tempLine) > 3:
                    maskName = tempLine[3]

                searchString = (subjectDir + '/' + subject
                                + deriPath + deriFile)
                a = glob.glob(searchString)
                if a:
                    foundDeri = os.path.abspath(a[0])
                    tempSub.addDerivativePath(deriName, foundDeri)

                else:
                    print('subject ' + subject + 'doesn\'t have derivative '
                          + deriName + 'at path ' + deriPath)
                    continue

            # level 2.3 for masks
            for mask in masks.values():
                tempSub.loadMask(mask)

        pipeDict[pipeline][subBase] = tempSub

    print('\n\nNow we are done with adding all paths and pheno information.'
          + 'Run the subjects in parallel')

    for pipeline in pipeDict.keys():
        print('\nRunning subjects inside pipeline ' + pipeline + ' now.')

        # now we can multiprocess through the subjects inside the dictionary
        runList = []
        for subject in pipeDict[pipeline].keys():
            subFile = pipeDict[pipeline][subject]
            runList.append((outDir, subFile))

        pool = mp.Pool(processes=nProcs)
        resultList = pool.map(runDerivative, runList)
        print resultList
    pass

if __name__ == '__main__':
    searchDir = sys.argv[1]
    templateFile = sys.argv[2]
    phenoPath = sys.argv[3]
    maskDir = sys.argv[4]
    outDir = sys.argv[5]
    nProcs = int(sys.argv[6])
    Main(searchDir, templateFile, phenoPath, maskDir, outDir, nProcs)
    pass
