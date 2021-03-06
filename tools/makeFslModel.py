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


def Main(searchDir, templateFile, phenoFile, nuisanceFile, outDir, mName):
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
        # fourDmatrix = np.array([])
        
        wStringDW = {}
        wStringDW['child'] = ''
        wStringDW['adolescent'] = ''
        wStringDW['adult'] = ''
        
        bStringDW = ''
        bStringDB = ''

        subjectList = {}
        subjectList['child'] = ''
        subjectList['adolescent'] = ''
        subjectList['adult'] = ''
        subjectList['between'] = ''
        # store children and adults here
        groupDict = {}
        groupDict['child'] = []
        groupDict['adolescent'] = []
        groupDict['adult'] = []
        
        # store mean fd and age here for within group demeaning
        meanFdGroupDict = {}
        meanFdGroupDict['child'] = np.array([])
        meanFdGroupDict['adolescent'] = np.array([])
        meanFdGroupDict['adult'] = np.array([])
        
        ageGroupDict = {}
        ageGroupDict['child'] = np.array([])
        ageGroupDict['adolescent'] = np.array([])
        ageGroupDict['adult'] = np.array([])

        # store mean fd and age here for between group demeaning
        dataAge = np.array([])
        betweenAge = np.array([])
        betweenFd = np.array([])

        # second level to loop through subjects
        subjectDir = os.path.abspath(os.path.join(searchDir, pipeline))

        # inner loop for the subjects
        for subject in os.listdir(subjectDir):
            # get the name of the subject, discard the session
            findSubBase = re.search(r'[a-zA-Z]*[0-9]*(?=_)', subject)
            subBase = findSubBase.group()

            # first within subject loop: add phenotypic data
            if (subBase in open(phenoFile).read() and
                subject in open(nuisanceFile).read()):
                # all is good
                # print('found ' + subBase + ' in pheno and nuisance file. Ya!')
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
            subUseAge = subAge
            subSex = subLine[phenoIndex['sex']]
            subMeanFd = float(nuisanceLine[nuisanceIndex['MeanFD']])
            dataAge = np.append(dataAge, subAge)

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
            storeStuff = (subject, tempScaPath, subMeanFd, subUseAge, subSex)
            # and then store it depending on age
            if subAge <= 12.0:
                print(subBase + ' is child with age ' + str(subAge))
                groupDict['child'].append(storeStuff)
                meanFdGroupDict['child'] = np.append(meanFdGroupDict['child'],
                                                     subMeanFd)
                ageGroupDict['child'] = np.append(ageGroupDict['child'], 
                                                  subUseAge)
                betweenAge = np.append(betweenAge, subUseAge)
                betweenFd = np.append(betweenFd, subMeanFd)

                cpacString = (cpacString
                              + subBase + ', ' + 'child' + ', ' + str(subSex)
                              + ', ' + str(subMeanFd) + '\n')
                
            elif subAge > 12.0 and subAge <= 18.0:
                print(subBase + ' is adolescent with age ' + str(subAge))
                groupDict['adolescent'].append(storeStuff)
                meanFdGroupDict['adolescent'] = np.append(meanFdGroupDict['adolescent'],
                                                     subMeanFd)
                ageGroupDict['adolescent'] = np.append(ageGroupDict['adolescent'], 
                                                  subUseAge)
                betweenAge = np.append(betweenAge, subUseAge)
                betweenFd = np.append(betweenFd, subMeanFd)

                cpacString = (cpacString
                              + subBase + ', ' + 'adolescent' + ', ' + str(subSex)
                              + ', ' + str(subMeanFd) + '\n')
                
                
            elif subAge > 18.0:
                print(subBase + ' is adult with age ' + str(subAge))
                groupDict['adult'].append(storeStuff)
                meanFdGroupDict['adult'] = np.append(meanFdGroupDict['adult'],
                                                     subMeanFd)
                ageGroupDict['adult'] = np.append(ageGroupDict['adult'], 
                                                  subUseAge)
                betweenAge = np.append(betweenAge, subUseAge)
                betweenFd = np.append(betweenFd, subMeanFd)

                cpacString = (cpacString
                              + subBase + ', ' + 'adult' + ', ' + str(subSex)
                              + ', ' + str(subMeanFd) + '\n')
            else:
                print(subBase + ' doesn\'t fit with age ' + str(subAge))
                continue

        # done with the pipeline loop
        # get the mean FD for both groups
        print('age: from ' + str(dataAge.min()) + ' to ' + str(dataAge.max())
              + ' with ' + str(len(dataAge)) + ' cases')
        avgBetweenAge = np.mean(betweenAge)
        avgBetweenFd = np.mean(betweenFd)

        # prepare the output stuff
        # tempOutDir = (outDir + pipeline)
        tempOutDir = outDir
        if not os.path.isdir(tempOutDir):
            os.makedirs(tempOutDir)

        '''
        String columns for between group:
            1) Group: 1 for kids, 2 for adolescents, 3 for adults
            2) kidsgroup: 1 for kids, 0 for adults
            3) adolescentgroup: 1 for adolescents
            4) adultsgroup: 0 for kids, 1 for adults
            5) kids-sex: yes
            6) adolescent-sex: yes
            7) adults-sex: yes
            8) kids-meanFd: demeaned
            9) adolescent-meanFD: demeaned
            10) adults-meanFd: demeaned

        String columns for within group:
            1) Group: always 1
            2) Mean: always 1
            3) sex: yes
            4) age: demeaned
            5) meanFd: demeaned
        '''

        # and now create the output stuff
        for subStuff in groupDict['child']:
            (subject, tempScaPath, subMeanFd, subUseAge, subSex) = subStuff
            # demean individual covariate by between group average
            bwSubAge = subUseAge - avgBetweenAge
            bwSubMeanFd = subMeanFd - avgBetweenFd
            
            # demean individual covariate by within group average
            wiSubAge = subUseAge - np.average(ageGroupDict['child'])
            wiSubMeanFd = subMeanFd - np.average(meanFdGroupDict['child'])

            # add to csv
            wStringDW['child'] = (wStringDW['child']
                                  +'1,1,' + str(subSex) + ',' 
                                  + str(wiSubAge) + ',' + str(wiSubMeanFd) 
                                  + '\n')
            
            bStringDW = (bStringDW
                         + '1,1,0,0,' + str(subSex) + ',0,0,' + str(wiSubMeanFd)
                         + ',0,0\n')
            
            bStringDB = (bStringDB
                         + '1,1,0,0,' + str(subSex) + ',0,0,' + str(bwSubMeanFd)
                         + ',0,0\n')
            
            subjectList['child'] = (subjectList['child'] + subject + '\n')
            
            subjectList['between'] = (subjectList['between'] + subject + '\n')
            

            # load the sca map for the subject
            '''
            f = nib.load(tempScaPath)
            scaMap = f.get_data()
            
            
            # add to childMat
            if fourDmatrix.size == 0:
                fourDmatrix = scaMap[..., None]
            else:
                fourDmatrix = np.concatenate((fourDmatrix, scaMap[..., None]),
                                             axis=3)
            '''
            
            
            
        # now the same for the adolescent
        for subStuff in groupDict['adolescent']:
            (subject, tempScaPath, subMeanFd, subUseAge, subSex) = subStuff
            # demean individual covariate by between group average
            bwSubAge = subUseAge - avgBetweenAge
            bwSubMeanFd = subMeanFd - avgBetweenFd
            
            # demean individual covariate by within group average
            wiSubAge = subUseAge - np.average(ageGroupDict['adolescent'])
            wiSubMeanFd = subMeanFd - np.average(meanFdGroupDict['adolescent'])

            # add to csv
            wStringDW['adolescent'] = (wStringDW['adolescent']
                                  +'1,1,' + str(subSex) + ',' 
                                  + str(wiSubAge) + ',' + str(wiSubMeanFd) 
                                  + '\n')
            
            bStringDW = (bStringDW
                         + '2,0,1,0,0,' + str(subSex) + ',0,0,' + str(wiSubMeanFd)
                         + ',0\n')
            
            bStringDB = (bStringDB
                         + '2,0,1,0,0,' + str(subSex) + ',0,0,' + str(bwSubMeanFd)
                         + ',0\n')
            
            subjectList['adolescent'] = (subjectList['adolescent'] + subject + '\n')
            
            subjectList['between'] = (subjectList['between'] + subject + '\n')

            
        # now the same for the adults
        for subStuff in groupDict['adult']:
            (subject, tempScaPath, subMeanFd, subUseAge, subSex) = subStuff
            # demean individual covariate by between group average
            bwSubAge = subUseAge - avgBetweenAge
            bwSubMeanFd = subMeanFd - avgBetweenFd
            
            # demean individual covariate by within group average
            wiSubAge = subUseAge - np.average(ageGroupDict['adult'])
            wiSubMeanFd = subMeanFd - np.average(meanFdGroupDict['adult'])

            # add to csv
            wStringDW['adult'] = (wStringDW['adult']
                                  +'1,1,' + str(subSex) + ',' 
                                  + str(wiSubAge) + ',' + str(wiSubMeanFd) 
                                  + '\n')
            
            bStringDW = (bStringDW
                         + '3,0,0,1,0,0,' + str(subSex) + ',0,0,' + str(wiSubMeanFd)
                         + '\n')
            
            bStringDB = (bStringDB
                         + '3,0,0,1,0,0,' + str(subSex) + ',0,0,' + str(bwSubMeanFd)
                         + '\n')
            
            subjectList['adult'] = (subjectList['adult'] + subject + '\n')
            
            subjectList['between'] = (subjectList['between'] + subject + '\n')

        # now print that shit out
        '''
        outNifti = nib.Nifti1Image(fourDmatrix, lastAffine, lastHeader)
        nib.nifti1.save(outNifti, fourDFile)
        '''

        bModelDW = (tempOutDir + '/' + mName + '_betweenModelDemeanWithin.csv')
        bModelDB = (tempOutDir + '/' + mName + '_betweenModelDemeanBetween.csv')
        wModelC = (tempOutDir + '/' + mName + '_withinModelChildren.csv')
        wModelA = (tempOutDir + '/' + mName + '_withinModelAdults.csv')
        bSubList = (tempOutDir + '/' + mName + '_subjectListBetween.txt')
        wSubListC = (tempOutDir + '/' + mName 
                     + '_subjectListWithinChildren.txt')
        wSubListA = (tempOutDir + '/' + mName + '_subjectListWithinAdults.txt')
        
        cpacModel = (tempOutDir + '/' + mName + '_cpacModel.csv')
        
        # and the csv
        bmdw = open(bModelDW, 'wb')
        bmdw.writelines(bStringDW)
        bmdw.close()
        
        bmdb = open(bModelDB, 'wb')
        bmdb.writelines(bStringDB)
        bmdb.close()
        
        wmc = open(wModelC, 'wb')
        wmc.writelines(wStringDW['child'])
        wmc.close()
        
        wma = open(wModelA, 'wb')
        wma.writelines(wStringDW['adult'])
        wma.close()
        
        bs = open(bSubList, 'wb')
        bs.writelines(subjectList['between'])
        bs.close()
        
        wsc = open(wSubListC, 'wb')
        wsc.writelines(subjectList['child'])
        wsc.close()
        
        wsa = open(wSubListA, 'wb')
        wsa.writelines(subjectList['adult'])
        wsa.close()

        c = open(cpacModel, 'wb')
        c.writelines(cpacString)
        c.close()


if __name__ == '__main__':
    if len(sys.argv) < 6:
        print('\nYou have specified too few commands!\n')
        print('This is the intended usage:\n'
              + ' 1) search directory where to look for the subject files\n'
              + ' 2) template file that contains the paths to the files\n'
              + ' 3) pheno file that contains the phenotypic information\n'
              + ' 4) nuisance file that contains the nuisance information\n'
              + ' 5) output director where to save the generated files\n'
              + ' 6) -optional: name prefix for the generated files.\n'
              + '    If none is specified, the output directory name is used')
    else:
        searchDir = sys.argv[1]
        templateFile = sys.argv[2]
        phenoFile = sys.argv[3]
        nuisanceFile = sys.argv[4]
        outDir = sys.argv[5]
        if len(sys.argv) > 6:
            mName = sys.argv[6]
        else:
            mName = os.path.basename(os.path.abspath(outDir))
        Main(searchDir, templateFile, phenoFile, nuisanceFile, outDir, mName)
    pass
