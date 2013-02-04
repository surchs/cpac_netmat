'''
Created on Feb 1, 2013

@author: sebastian

my little script that picks up preprocessed images and corresponding phenotypic
information and creates a 4D stack and a csv file ready for FLAMEO.

Mainly using stuff I already have
'''
import os
import re
import sys
import glob
import numpy as np
import nibabel as nib


def Main(searchDir, templateFile, phenoFile, nuisanceFile, outDir):
    '''
    method to load the preprocessed stuff and stack it while at the same time
    saving the phenotypic information

    Since I am doing a group analysis I will sort subjects into two groups here
    '''
    pathFile = open(templateFile)
    # for now we have only one line there so we can get it once
    line = pathFile.readline()
    tempLine = line.strip().split()
    deriName = tempLine[0]
    deriPath = tempLine[1]
    deriFile = tempLine[2]

    phenoIndex = {}
    phenoLine = open(phenoFile, 'rb').readline()
    phenoLoop = phenoLine.strip().split(',')

    nuisanceIndex = {}
    nuisanceLine = open(nuisanceFile, 'rb').readline()
    nuisanceLoop = nuisanceLine.strip().split(',')
    
    lastHeader = None
    lastAffine = None

    run = 0
    for pheno in phenoLoop:
        # check if it is the subject designator which is not a phenotypic info
        if not pheno == 'subject':
            phenoIndex[pheno] = run

        run += 1

    run = 0
    for nuisance in nuisanceLoop:
        # check if it is the subject designator which is not a phenotypic info
        if not pheno == 'subject':
            nuisanceIndex[nuisance] = run

        run += 1

    pipeDict = {}

    # outer level to loop through pipelines
    for pipeline in os.listdir(searchDir):
        # generate pipeline dict entry
        if not pipeline in pipeDict.keys():
            pipeDict[pipeline] = {}
            
        cpacString = 'subId, group, sex, meanFd\n'

        meanFdGroupDict = {}
        meanFdGroupDict['child'] = []
        meanFdGroupDict['adult'] = []
        childFd = np.array([])
        adultFd = np.array([])

        # second level to loop through subjects
        subjectDir = os.path.abspath(os.path.join(searchDir, pipeline))

        # inner loop for the subjects
        for subject in os.listdir(subjectDir):
            # prepare temporary dict
            tempSubPheno = {}
            # get the name of the subject, discard the session
            findSubBase = re.search(r'[a-zA-Z]*[0-9]*(?=_)', subject)
            subBase = findSubBase.group()

            # first within subject loop: add phenotypic data
            if (subBase in open(phenoFile).read() and
                subject in open(nuisanceFile).read()):
                # all is good
                print('found ' + subBase + ' in pheno and nuisance file. Ya!')
                # loop through the file again
                for line in open(phenoFile, 'rb'):
                    # search for the correct line
                    # currently requires the subject name to be given at
                    # the start
                    if line.startswith(subBase):
                        subLine = line.strip().split(',')
                # then search for the nuisance line
                for line in open(nuisanceFile, 'rb'):
                    # get the correct line
                    if line.startswith(subject):
                        nuisanceLine = line.strip().split(',')

            else:
                # no phenotypic information, presently we skip these subjects
                print('didn\'t find ' + subBase + ' in phenofile or nFile.'
                      + ' skipping for now...')
                continue

            # get the phenotypic information
            subAge = float(subLine[phenoIndex['age']])
            subSex = subLine[phenoIndex['sex']]
            subMeanFd = float(nuisanceLine[nuisanceIndex['MeanFD']])

            if subSex == 'Female':
                subSex = 1
            elif subSex == 'Male':
                subSex = -1
            else:
                print('Your subject is gender neutral you dumb bitch. Aborting')
                continue

            # second within subject loop: add derivatives
            # to loop through the derivative paths in the template
            # file we have to reset the file line position for every subject
            searchString = (subjectDir + '/' + subject + deriPath
                            + deriFile)
            a = glob.glob(searchString)
            if len(a) == 0:
                    print(subBase + ' ' + pipeline + ' ' + deriName
                          + ' doesn\'t exist')
            else:
                tempScaPath = a[0]

            # pull all the shit together
            storeStuff = (tempScaPath, subMeanFd, subSex)
            # and then store it depending on age
            if subAge < 17.0:
                meanFdGroupDict['child'].append(storeStuff)
                childFd = np.append(childFd, subMeanFd)
                cpacString = (cpacString
                              + subBase + ', ' + 'child' + ', ' + str(subSex)
                              + ', ' + str(subMeanFd) + '\n')
            else:
                meanFdGroupDict['adult'].append(storeStuff)
                adultFd = np.append(adultFd, subMeanFd)
                cpacString = (cpacString
                              + subBase + ', ' + 'adult' + ', ' + str(subSex)
                              + ', ' + str(subMeanFd) + '\n')

        # done with the pipeline loop
        # get the mean FD for both groups
        childMean = np.mean(childFd)
        adultMean = np.mean(adultFd)

        # prepare the output stuff
        tempOutDir = (outDir + pipeline)
        if not os.path.isdir(tempOutDir):
            os.makedirs(tempOutDir)

        fourDFile = (tempOutDir + '/fourDTestfile.nii.gz')
        fslModel = (tempOutDir + '/fslTestfile.csv')
        cpacModel = (tempOutDir + '/cpacTestfile.csv')

        fourDmatrix = np.array([])
        fslString = ''

        '''
        String columns for csv:
            1) Group: 1 for kids, 2 for adults
            2) Mean: 1 for all
            3) kidsfactor: 1 for kids, 0 for adults
            4) adultsfactor: 0 for kids, 1 for adults
            5) sexfactor: 1 for women, 0 for men
            6) agefactor: demeaned age (won't need this in my analysis though

        '''

        # and now create the output stuff
        for subStuff in meanFdGroupDict['child']:
            (tempScaPath, subMeanFd, subSex) = subStuff
            subMeanFd = subMeanFd - childMean
            # add to csv
            fslString = (fslString +
                           '1, 1, 0, ' + str(subSex) + ', 0, ' 
                           + str(subMeanFd) + ', 0\n')
            # load the sca map for the subject
            f = nib.load(tempScaPath)
            scaMap = f.get_data()

            # add to childMat
            if fourDmatrix.size == 0:
                fourDmatrix = scaMap[..., None]
            else:
                fourDmatrix = np.concatenate((fourDmatrix, scaMap[..., None]),
                                             axis=3)

        # now the same for the adults
        for subStuff in meanFdGroupDict['adult']:
            (tempScaPath, subMeanFd, subSex) = subStuff
            subMeanFd = subMeanFd - adultMean
            # add to csv
            fslString = (fslString +
                         '2, 0, 1, 0, ' + str(subSex) + ', 0, ' 
                         + str(subMeanFd) + '\n')
            # load the sca map for the subject
            f = nib.load(tempScaPath)
            scaMap = f.get_data()
            lastHeader = f.get_header()
            lastAffine = f.get_affine()

            # add to childMat
            if fourDmatrix.size == 0:
                fourDmatrix = scaMap[..., None]
            else:
                fourDmatrix = np.concatenate((fourDmatrix, scaMap[..., None]),
                                             axis=3)

        # now print that shit out
        outNifti = nib.Nifti1Image(fourDmatrix, lastAffine, lastHeader)
        nib.nifti1.save(outNifti, fourDFile)
        # and the csv
        f = open(fslModel, 'wb')
        f.writelines(fslString)
        f.close()
        c = open(cpacModel, 'wb')
        c.writelines(cpacString)
        c.close()


if __name__ == '__main__':
    searchDir = sys.argv[1]
    templateFile = sys.argv[2]
    phenoFile = sys.argv[3]
    nuisanceFile = sys.argv[4]
    outDir = sys.argv[5]
    Main(searchDir, templateFile, phenoFile, nuisanceFile, outDir)
    pass
