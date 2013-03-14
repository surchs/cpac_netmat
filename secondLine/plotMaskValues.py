'''
Created on Mar 10, 2013

@author: surchs
'''
import numpy as np
import pandas as pa
import nibabel as nib
import multiprocessing as mp
from matplotlib import pyplot as plt


def loadShit(args):
    # Unpack args
    ((csfMaskPath, csfThresh,
      gmMaskPath, gmThresh,
      wmMaskPath, wmThresh)) = args

    # Load the mask files
    csfFile = nib.load(csfMaskPath)
    csfMask = csfFile.get_data()
    gmFile = nib.load(gmMaskPath)
    gmMask = gmFile.get_data()
    wmFile = nib.load(wmMaskPath)
    wmMask = wmFile.get_data()

    # Compute shit
    csfCount = np.sum(csfMask > 0)
    gmCount = np.sum(gmMask > 0)
    wmCount = np.sum(wmMask > 0)

    csfThreshCount = np.sum(csfMask > csfThresh)
    gmThreshCount = np.sum(gmMask > gmThresh)
    wmThreshCount = np.sum(wmMask > wmThresh)

    return ((csfCount, csfThreshCount),
            (gmCount, gmThreshCount),
            (wmCount, wmThreshCount))


def plotShit(ageStack, aboveStack, allStack, thresh, subjectList, what):
    # Prepare Containers
    list0 = []
    list20 = []
    list01 = []
    list03 = []
    list05 = []

    # Shoutout
    print('Going to show results for ' + what + ' now\n\n')

    # Check stuff:
    for i, threshCount in enumerate(aboveStack):
        age = ageStack[i]
        allCount = allStack[i]

        if threshCount == 0:
            list0.append(age)
            pass
        elif threshCount < 20 and threshCount > 0:
            list20.append(age)
            pass
        elif threshCount / allCount < 0.1:
            list01.append(age)
            pass
        elif threshCount / allCount < 0.3:
            list03.append(age)
            pass
        elif threshCount / allCount < 0.5:
            list05.append(age)
            pass

    # Now start plotting stuff
    # First tell some stories:
    print('\nThese are the results:\n')
    print(str(len(list0)) + ' subjects didn\'t pass the threshold')
    print(str(len(list20)) + ' subjects had less than 20 passing voxels')
    print(str(len(list01)) + ' subjects had less than 10 perc passing')
    print(str(len(list03)) + ' subjects had less than 30 perc passing')
    print(str(len(list05)) + ' subjects had less than 50 perc passing')

    # Compute some more shit
    median = np.median(aboveStack)
    mean = np.mean(aboveStack)
    perc10 = np.percentile(aboveStack, 10)
    perc90 = np.percentile(aboveStack, 90)

    print('\nThe median for passing voxels is: ' + str(median))
    print('The mean for passing voxels is: ' + str(mean))
    print('The 10th percentile for passing voxels is: ' + str(perc10))
    print('The 90th percentile for passing voxels is: ' + str(perc90))

    # And search for the outlier subjects
    outlier = perc90 * 2
    outlierIndex = np.where(aboveStack > outlier)[0]

    outlierString = ''
    # Make the outlier string
    for index in outlierIndex:
        subject = subjectList[index]
        above = aboveStack[index]
        age = ageStack[index]
        outlierString = (outlierString + str(subject) + ' (' + str(age) + ')'
                         + ' has ' + str(above)
                         + ' voxels exceeding threshold\n')

    # Print it
    print('\nOutlier analysis:')
    print(outlierString)

    print('\n\nPlotting now')
    # Start plotting
    plt.plot(ageStack, aboveStack, 'kx', label='passing voxels')
    plt.axhline(y=mean, xmin=5, xmax=24, color='r', label='mean')
    plt.axhline(y=median, xmin=5, xmax=24, color='b', label='median')
    plt.axhline(y=perc10, xmin=5, xmax=24, color='y', label='10%')
    plt.axhline(y=perc90, xmin=5, xmax=24, color='g', label='90%')
    plt.xlabel('age')
    plt.ylabel('voxels exceeding threshold')
    plt.legend()
    plt.title('Test for ' + what + ' at ' + str(thresh) + ' threshold')
    plt.show()
    raw_input("Press Enter to continue...")
    plt.close()


def Main():
    # Input phenotypic file
    pathToPheno = '/home2/surchs/secondLine/configs/abide/abide_across_236_pheno.csv'
    csfMaskPaths = '/home2/surchs/secondLine/configs/abide/pathsTo_CSF_mask_abide.csv'
    wmMaskPaths = '/home2/surchs/secondLine/configs/abide/pathsTo_WM_mask_abide.csv'
    gmMaskPaths = '/home2/surchs/secondLine/configs/abide/pathsTo_GM_mask_abide.csv'

    # Parameter
    csfThresh = 0.98
    wmThresh = 0.95
    gmThresh = 0.7

    nProcs = 25

    # Read the paths
    csfPathFile = open(csfMaskPaths, 'rb')
    csfPaths = csfPathFile.readlines()
    wmPathFile = open(wmMaskPaths, 'rb')
    wmPaths = wmPathFile.readlines()
    gmMaskPaths = open(gmMaskPaths, 'rb')
    gmPaths = gmMaskPaths.readlines()

    # Reading phenotypic file and creating list of subjects
    pheno = pa.read_csv(pathToPheno)
    pheno.astype(str)
    subjectList = pheno['SubID'].tolist()
    ageList = pheno['age'].tolist()

    # Prepare Containers:
    parallelList = []

    csfAboveStack = np.array([])
    wmAboveStack = np.array([])
    gmAboveStack = np.array([])

    csfAllStack = np.array([])
    wmAllStack = np.array([])
    gmAllStack = np.array([])

    ageStack = np.array([])

    for i, subject in enumerate(subjectList):
        subject = ('00' + str(subject))
        age = ageList[i]
        csfMaskPath = csfPaths[i].strip()
        gmMaskPath = gmPaths[i].strip()
        wmMaskPath = wmPaths[i].strip()

        # Check if we have the correct subject here
        if not subject in csfMaskPath:
            print('you got the wrong subject! '
                  + subject + ' not in ' + csfMaskPath)
        elif not subject in gmMaskPath:
            print('you got the wrong subject! '
                  + subject + ' not in ' + gmMaskPath)
        elif not subject in wmMaskPath:
            print('you got the wrong subject! '
                  + subject + ' not in ' + wmMaskPath)

        print('Preparing ' + subject + ' now')

        # Stack age
        ageStack = np.append(ageStack, age)

        # Prepare parallel
        parallelList.append((csfMaskPath, csfThresh,
                             gmMaskPath, gmThresh,
                             wmMaskPath, wmThresh))

    # Execute parallel run
    # Now run this in parallel
    print('prepared to run multicore')
    pool = mp.Pool(processes=nProcs)
    resultList = pool.map(loadShit, parallelList)
    print('ran multicore')

    # Getting results
    print('Getting results')
    for result in resultList:
        # unpack results
        ((csfCount, csfThreshCount),
         (gmCount, gmThreshCount),
         (wmCount, wmThreshCount)) = result

        # Stack shit:
        csfAllStack = np.append(csfAllStack, csfCount)
        gmAllStack = np.append(gmAllStack, gmCount)
        wmAllStack = np.append(wmAllStack, wmCount)

        csfAboveStack = np.append(csfAboveStack, csfThreshCount)
        gmAboveStack = np.append(gmAboveStack, gmThreshCount)
        wmAboveStack = np.append(wmAboveStack, wmThreshCount)

    print('Done, plotting stuff')
    # Now we are done with all this shit, let's plot the results
    plotShit(ageStack, csfAboveStack, csfAllStack, csfThresh, subjectList, 'CSF')
    plotShit(ageStack, wmAboveStack, wmAllStack, wmThresh, subjectList, 'WM')
    plotShit(ageStack, gmAboveStack, gmAllStack, gmThresh, subjectList, 'GM')


if __name__ == '__main__':
    Main()
    pass
