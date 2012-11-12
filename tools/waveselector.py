'''
Created on Oct 12, 2012

@author: sebastian

script that runs through a prepared QC file and checks for a couple of
conditions
'''
import os
import re
import sys
import glob
import nibabel as nib


class Scan:
    def __init__(self, scanId):
        self.scanId = scanId
        self.restId = None
        self.useAnat = None
        self.anatId = None
        self.base = None


def Main(loadFile, checkDir, toDir):
    qcFile = open(loadFile, 'rb')
    qcLines = qcFile.readlines()

    # List of passing subjects
    awesomeDict = {}

    # loop over the scan IDs
    for line in qcLines:
        # for now let's just print stuff
        '''
        SCAN_Subject_ID - 1
        QC_Rest1 - 11
        QC_Rest1_Eyes - 12
        QC_WhichAnat - 17
        DX_Status - 18
        '''

        # prepare check line (cL)
        cL = line.strip().split(',')
        scanID = cL[0]
        anatID1 = cL[5]
        anatID2 = cL[6]
        restID = cL[7]
        rest1Qc = cL[10]
        rest1Eyes = cL[11]
        anatomicalQc = cL[16]

        if rest1Qc == 'pass' and rest1Eyes == 'open':
            if anatomicalQc == '1' \
            or anatomicalQc == 'either':

                a = Scan(scanID)
                a.useAnat = 1
                a.restId = restID
                a.anatId = anatID1
                a.base = None

            elif anatomicalQc == '2':
                a = Scan(scanID)
                a.useAnat = 2
                a.restId = restID
                a.anatId = anatID2
                a.base = None
                pass
            awesomeDict[scanID] = a

    subjectDict = {}
    # read the whole file. Now I can loop over the list
    for scanID in awesomeDict.values():
        # print 'I like', scanID.scanId, 'with anat', scanID.useAnat
        # check if subject exists
        scanId = scanID.scanId
        scanPath = os.path.join(checkDir, scanId)

        # check if the scan exists (session name)
        if os.path.isdir(scanPath):
            restId = scanID.restId
            if len(restId) < 2:
                restId = ('0' + restId)

            anatId = scanID.anatId
            if len(anatId) < 2:
                anatId = ('0' + anatId)

            restPath = (scanPath + '/*/' + restId + '+Rest*')
            anatPath = (scanPath + '/*/' + anatId + '+HighResT1')

            restDirSearch = glob.glob(restPath)

            if len(restDirSearch) == 0:
                # there is no rest dir
                print 'no rest dir for scanId', scanId
                continue

            # there is a rest directory, is there also a full scan?
            restDir = restDirSearch[0]
            restSearch = glob.glob((restDir + '/*.nii.gz'))
            if len(restSearch) == 1:
                # there it is
                restFile = restSearch[0]
            else:
                # no rest File or too many
                continue

            # get the rest file
            restImg = nib.load(restFile)
            if len(restImg.shape) < 4:
                # this is not a 4D file, continue
                print 'no rest file (no 4D) for scanId', scanId
                continue

            TR = restImg.shape[3]
            if TR < 180:
                # this is an uncomplete scan
                print 'no complete rest file for scanId', scanId
                continue

            # now we do the same for the anatomical file
            anatDirSearch = glob.glob(anatPath)

            if len(anatDirSearch) == 0:
                # there is no rest dir
                print 'no anat dir for scanId', scanId, anatId
                continue

            # there is an anat directory, is there also a file in there?
            anatDir = anatDirSearch[0]
            anatSearch = glob.glob((anatDir + '/*.nii.gz'))
            if len(anatSearch) == 1:
                anatFile = anatSearch[0]
            else:
                print 'no anat file for scanId', scanId
                # no rest File or too many
                continue

            # get the anat file
            anatImg = nib.load(anatFile)
            if len(anatImg.shape) != 3:
                # this is not a 4D file, continue
                print 'no anat file (no 3D) for scanId', scanId
                continue

            # if we made it this far, then we have a full rest scan that passes
            # the NYU QC standards
            findSubBase = re.search(r'[a-zA-Z]*[0-9]*', scanId)
            findSession = re.search(r'[a-zA-Z]+(?=\Z)', scanId)

            # so we need to get the base and session name to store the stuff
            subBase = findSubBase.group()
            if findSession:
                session = findSession.group()
            else:
                session = ''

            # see if the subject base already exists
            if subBase in subjectDict.keys():
                # get the dictionary inside the subjectDict under the current
                # subject base name
                tempSessionDict = subjectDict[subBase]
            else:
                # we have to create a new dictionary
                tempSessionDict = {}

            # assign the two file paths to the dictionary
            tempSessionDict[session] = (anatFile, restFile)
            # and write the whole thing back into the subject Dict
            subjectDict[subBase] = tempSessionDict

    # outside of the session ID loop
    print len(subjectDict.keys())

    useSubjects = {}
    # now we need to loop over the subjects again to get the session with
    # the session with the smallest extension (nothing for the first)
    for subject in subjectDict.keys():
        # get the session Dictionary
        sessionDict = subjectDict[subject]
        # get the session names
        sessionNames = sessionDict.keys()
        # order them so the first one is the smallest
        sessionNames.sort()
        # use the first in the ordered sessionNames for further analysis
        session = sessionNames[0]
        # save to the new use subjects dictionary
        (anatPath, restPath) = sessionDict[session]
        useSubjects[subject] = (anatPath, restPath)

        # create a symlink to the anat file and the rest file
        toRestDir = os.path.join(toDir, subject, 'session_1/func')
        toRestPath = os.path.join(toRestDir, 'rest.nii.gz')

        toAnatDir = os.path.join(toDir, subject, 'session_1/anat')
        toAnatPath = os.path.join(toAnatDir, 'mprage.nii.gz')

        # print '\nCreating symlinks for', subject

        # see if target directories already exist
        if not os.path.isdir(toRestDir):
            # we need to create the directory
            # print 'creating new rest directory', toRestDir
            os.makedirs(toRestDir)

        if not os.path.isdir(toAnatDir):
            # we need to create the anat directory
            # print 'creating new anat directory', toAnatDir
            os.makedirs(toAnatDir)

        # print '    ', anatPath, '----->', toAnatPath
        os.symlink(anatPath, toAnatPath)
        # print '    ', restPath, '----->', toRestPath
        os.symlink(restPath, toRestPath)


if __name__ == '__main__':
    loadFile = sys.argv[1]
    checkDir = sys.argv[2]
    toDir = sys.argv[3]
    Main(loadFile, checkDir, toDir)
    pass
