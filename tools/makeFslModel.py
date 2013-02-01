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


def Main(searchDir, templateFile, phenoFile, outDir):
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

    loopLine = phenoLine.strip().split(',')
    print(loopLine)
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

        ageGroupDict = {}
        ageGroupDict['child'] = []
        ageGroupDict['adult'] = []
        childAge = np.array([])
        adultAge = np.array([])

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

            # get the phenotypic information
            subAge = subLine[phenoIndex['age']]
            subSex = subLine[phenoIndex['sex']]

            if subSex == 'Female':
                subSex = 1
            elif subSex == 'Male':
                subSex = 0
            else:
                print('Your subject is gender neutral you dumb bitch. Aborting')
                continue

            # second within subject loop: add derivatives
            # to loop through the derivative paths in the template
            # file we have to reset the file line position for every subject
            searchString = (subjectDir + '/' + subject + deriPath
                            + deriFile)
            a = glob.glob(searchString)
            tempScaPath = a[0]

            # pull all the shit together
            storeStuff = (tempScaPath, subAge, subSex)
            # and then store it depending on age
            if subAge < 17.0:
                ageGroupDict['child'].append(storeStuff)
                childAge = np.append(childAge, subAge)
            else:
                ageGroupDict['adult'].append(storeStuff)
                adultAge = np.append(adultAge, subAge)

        # done with the pipeline loop
        # get the mean ages for both groups
        childMean = np.mean(childAge)
        adultMean = np.mean(adultAge)

        # prepare the output stuff
        tempOutDir = (outDir + pipeline)
        if not os.path.isdir(tempOutDir):
            os.makedirs(tempOutDir)

        fourDFile = (tempOutDir + '/fourDTestfile.nii.gz')
        modelFile = (tempOutDir + '/modelTestfile.csv')

        fourDmatrix = np.array([])
        csvString = ''
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
        for subStuff in ageGroupDict['child'].values():
            (tempScaPath, subAge, subSex) = subStuff
            subAge = subAge - childMean
            # add to csv
            csvString = (csvString +
                           '1, 1, 1, 0, ' + str(subSex) + ', ' + str(subAge)
                           + '\n')
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
        for subStuff in ageGroupDict['adults'].values():
            (tempScaPath, subAge, subSex) = subStuff
            subAge = subAge - adultMean
            # add to csv
            csvString = (csvString +
                           '2, 1, 0, 1, ' + str(subSex) + ', ' + str(subAge)
                           + '\n')
            # load the sca map for the subject
            f = nib.load(tempScaPath)
            scaMap = f.get_data()

            # add to childMat
            if fourDmatrix.size == 0:
                fourDmatrix = scaMap[..., None]
            else:
                fourDmatrix = np.concatenate((fourDmatrix, scaMap[..., None]),
                                             axis=3)

        # now print that shit out
        outNifti = nib.Nifti1Image(fourDmatrix, f.get_affine(), f.get_header())
        nib.nifti1.save(outNifti, fourDFile)
        # and the csv
        f = open(modelFile, 'wb')
        f.writelines(csvString)
        f.close()


if __name__ == '__main__':
    searchDir = sys.argv[1]
    templateFile = sys.argv[2]
    phenoFile = sys.argv[3]
    outDir = sys.argv[4]
    Main(searchDir, templateFile, phenoFile, outDir)
    pass
