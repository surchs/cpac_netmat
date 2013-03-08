'''
Created on Mar 5, 2013

@author: surchs
'''
import os
import numpy as np
from matplotlib import pyplot as plt
from cpac_netmat.tools import meisterlein as mm


def readCsvWithHeader(pathToCsvFile):
    f = open(pathToCsvFile, 'rb')
    colDict = {}
    headerline = f.readline().strip().split(',')
    lines = f.readlines()
    for colNum, colString in enumerate(headerline):
        colDict[colString] = colNum

    print('The columns for the phenotypic file are:\n'
          + str(colDict))

    return lines, colDict


def survivorCounter(subjectDict, parameters):
    # Returns the number of subjects that would survive this selection
    # It does so by cutting the ages into the define bins and then multiplying
    # number in the smallest bin width the number of bins
    startAge, binWidth, stopAge = parameters

    # Create the bins
    lowBins = np.arange(startAge, stopAge, binWidth)
    numberBins = len(lowBins)
    maxLowerAge = np.max(lowBins)
    ageTest = np.array([])

    # Create container
    ageBins = {}
    binSurvivors = {}
    # The final list of subjects to include for these parameters
    # (evenly sampled)
    subjectList = []

    for lower in lowBins:
        # Check all subjects if they fall in this age range and if so:
        # add to list
        binList = []
        higher = lower + binWidth
        for subject in subjectDict.keys():
            age = subjectDict[subject]['age']
            if age >= lower and age <= higher:
                # include the subject
                binList.append(subject)
        # firstTemp = ages[ages > lower]
        # secondTemp = firstTemp[firstTemp < higher]
        # numSurvs = len(secondTemp)
        numSurvs = len(binList)
        ageBins[str(lower)] = binList
        binSurvivors[str(lower)] = numSurvs

    smallestBin = min(binSurvivors.values())

    if not smallestBin < 2:
        numberSurvivors = smallestBin * numberBins
        # Now go back through the bins and check if the number of survivors is
        # greater than smallestBin
        # If so, make a numpy array of the subjectList, make a random index,
        # take the first 'smallestBin' entries and pass that on
        for lower in binSurvivors.keys():
            survivors = binSurvivors[lower]
            subjects = ageBins[lower]
            # Check if we have to bring down the number first
            if survivors > smallestBin:
                npSubjects = np.array(subjects)
                np.random.shuffle(npSubjects)
                # get the number desired
                survSubjects = npSubjects[:(smallestBin)]
                # and back to list
                keepSubs = survSubjects.tolist()
            else:
                keepSubs = subjects

            # Append all the subjects to the big list
            for subject in keepSubs:
                subjectList.append(subject)
                subAge = subjectDict[subject]['age']
                ageTest = np.append(ageTest, subAge)
        # Now check:
        if np.max(ageTest) < maxLowerAge:
            print('Found messed up stuff for ' + str(parameters)
                  + ' maxlower is ' + str(maxLowerAge) + ' '
                  + ' max total is ' + str(np.max(ageTest)))
    else:
        numberSurvivors = 0

    return numberSurvivors, subjectList


def siteSurvivorCounter(siteDict, parameters):
    # Returns the number of total survivors and the sites that are included
    sites = []
    siteSurvivorList = []
    numberSurvivors = 0
    for site in siteDict.keys():
        siteSubjectDict = siteDict[site]
        (siteSurvivors,
         siteSubjectList) = survivorCounter(siteSubjectDict, parameters)
        # Check if site has any survivors
        if siteSurvivors > 0:
            sites.append(site)
            numberSurvivors += siteSurvivors
            # and now run through the subjects
            for subject in siteSubjectList:
                siteSurvivorList.append(subject)

    return numberSurvivors, sites, siteSurvivorList


def plotSurvivorSites(strategies, stratDict,
                      subjectDict, minNumBin,
                      outDir, focus):
    for strategy in strategies:
        # Get inputs for plot
        tempDict = stratDict[strategy]
        tempSiteDict = {}
        stratAges = np.array([])
        siteID = []
        survivorSubjects = tempDict['survivors']
        parameters = tempDict['parameters']
        numSurvivors = tempDict['numSurvs']
        # Check input consistency
        if not numSurvivors == len(survivorSubjects):
            print('Got different number of survivors from count and dict for '
                  + strategy
                  + ': ' + str(numSurvivors) + '/' + str(len(survivorSubjects)))
            continue
        # Load subject data for the survivor list
        for subject in survivorSubjects:
            tempDict = subjectDict[subject]
            subAge = tempDict['age']
            subSite = tempDict['site']
            stratAges = np.append(stratAges, subAge)
            siteID.append(subSite)
            # Store subject data by site in tempSiteDict
            if not subSite in tempSiteDict.keys():
                tempSiteDict[subSite] = np.array([])
            tempSiteDict[subSite] = np.append(tempSiteDict[subSite], subAge)

        # Now its time to plot the stuff for site
        # First generate a spread for the sites that are to be plotted
        siteSpread = {}
        siteNum = len(tempSiteDict.keys())
        spread = 1
        # Generate point spread for sites
        for site in tempSiteDict.keys():
            siteSpread[site] = spread
            spread += 0.1

        minSpread = 1
        maxSpread = 1 + spread

        # And now plot the sites with the spread
        for site in tempSiteDict.keys():
            spread = siteSpread[site]
            siteAges = tempSiteDict[site]
            xspread = np.ones(len(siteAges)) * spread
            plt.plot(xspread, siteAges, marker='o', linestyle='', label=site)

        # Dummyplot to spread the x-axis
        xdummy = np.arange(minSpread - 0.5, maxSpread + 0.5, 0.1)
        ydummy = np.zeros_like(xdummy) + 6
        plt.plot(xdummy, ydummy, linestyle='')

        # Now show the result
        title = (focus + ' '
                 + str(minNumBin) + ' '
                 + strategy + ': '
                 + str(numSurvivors))
        plt.title(title)
        plt.legend()
        plt.show()
        do = raw_input('Enter to go on\n')
        fileName = (focus + '_'
                    + str(minNumBin) + '_'
                    + strategy + '_'
                    + str(numSurvivors) + '.txt')
        filePath = os.path.join(outDir, fileName)
        if do == 'g':
            # keep this list
            print('Writing this to ' + filePath)
            f = open(filePath, 'wb')
            for subject in survivorSubjects:
                f.write(subject + '\n')

            f.close()
        elif do == 'break':
            print('Breaking display for ' + focus)
            break

        plt.close()


def Main():
    # Define Inputs
    pathToPhenoFile = '/home2/surchs/secondLine/configs/ABIDE_763_Phenotypics.csv'

    # Define outputs
    pathToOutDir = '/home2/surchs/secondLine/sampling/ABIDE'

    # Define parameters
    startAgeMin = 6
    startAgeMax = 10
    startAgeInc = 0.1
    startAgeParams = np.arange(startAgeMin, startAgeMax, startAgeInc)

    stopAgeMin = 15
    stopAgeMax = 24
    stopAgeInc = 0.1

    stopAgeParams = np.arange(stopAgeMin, stopAgeMax, stopAgeInc)

    binWidthMin = 2
    binWidthMax = 4
    binWidthInc = 0.1
    minNumBin = 2
    binWidthParams = np.arange(binWidthMin, binWidthMax, binWidthInc)

    # Prepare display parameters
    doPrint = True
    doPlot = True

    # Prepare Containers
    siteDict = {}
    siteSpread = {}
    subjectDict = {}
    resultDict = {}
    acrossSurvivorArray = np.array([])
    parameterList = []
    stratDict = {}
    stratDict['within'] = {}
    stratDict['across'] = {}
    # Prepare Containers
    includedSiteList = []
    includeListSurvivors = []
    withinSurvivorArray = np.array([])

    # Read the phenotypic file
    phenoLines, phenoColDict = readCsvWithHeader(pathToPhenoFile)

    # Iterate over the subjects line by line and split up by site
    for line in phenoLines:
        # Prepare the line
        useLine = line.strip().split(',')
        subject = useLine[phenoColDict['SubID']]
        site = useLine[phenoColDict['Site']]
        group = int(useLine[phenoColDict['group']])
        age = float(useLine[phenoColDict['age']])

        if group == 1:
            print(subject + ' is ASD and will be dropped')
            continue

        if not subject in subjectDict.keys():
            subjectDict[subject] = {}

        tempDict = subjectDict[subject]
        tempDict['site'] = site
        tempDict['age'] = age
        subjectDict[subject] = tempDict

        if not site in siteDict.keys():
            # Initialize the site here
            # siteDict[site] = np.array([])
            siteDict[site] = {}
        siteDict[site][subject] = tempDict
        # siteDict[site] = np.append(siteDict[site], age)
        # allAges = np.append(allAges, age)

    # Loop through the fucking parameters
    for startAge in startAgeParams:
        for binWidth in binWidthParams:
            for stopAge in stopAgeParams:
                # Prepare parameters
                binNum = 0
                while (startAge + binNum * binWidth) < stopAge:
                    # check if we exceed
                    binNum += 1

                if binNum < minNumBin:
                    continue

                maxAge = startAge + (binNum * binWidth)
                stratString = (str(startAge) + '_'
                               + str(binWidth) + '_'
                               + str(maxAge))
                if not stratString in stratDict['within'].keys():
                    # Run this strategy
                    print('Running ' + stratString)
                    stratDict['within'][stratString] = {}
                if not stratString in stratDict['across'].keys():
                    # Run this strategy
                    stratDict['across'][stratString] = {}
                    # Prepare container for within and across
                    # Make temporary dicts out of them for easier handling
                    crossDict = stratDict['across'][stratString]
                    withDict = stratDict['within'][stratString]

                    # Save the parameters
                    parameters = (startAge, binWidth, stopAge)

                    # First, get the list of sites that pass (i.e. have more
                    # than 2 subjects per bin
                    (numSiteSurvivors,
                     sites,
                     siteSurvivorList) = siteSurvivorCounter(siteDict,
                                                             parameters)
                    # Store the results
                    withinSurvivorArray = np.append(withinSurvivorArray,
                                                    numSiteSurvivors)
                    includedSiteList.append(sites)
                    includeListSurvivors.append(siteSurvivorList)

                    withDict['survivors'] = siteSurvivorList
                    withDict['numSurvs'] = numSiteSurvivors
                    withDict['parameters'] = parameters

                    # Now I need a new subject dictionary for the sites that
                    # made it
                    acrossSubDict = {}
                    for site in sites:
                        for subject in siteDict[site].keys():
                            # Get the sites subjects into the new dictionary
                            acrossSubDict[subject] = siteDict[site][subject]

                    # Now run this stuff for the across site sample
                    (numAcrossSurvivors,
                     acrossSurvivorList) = survivorCounter(acrossSubDict,
                                                           parameters)
                    # Add results to across array
                    acrossSurvivorArray = np.append(acrossSurvivorArray,
                                                    numAcrossSurvivors)

                    crossDict['survivors'] = acrossSurvivorList
                    crossDict['numSurvs'] = numAcrossSurvivors
                    crossDict['parameters'] = parameters

                    # Make a list of the strategies we run so I can find them
                    # again later by their ordering
                    parameterList.append(stratString)
                    stratDict['across'][stratString] = crossDict
                    stratDict['within'][stratString] = withDict

                else:
                    continue

    # Now look at the results
    # Across sites first:
    acrossSurvivorValues = np.unique(acrossSurvivorArray)
    maxAcrossSurvivors = acrossSurvivorValues[-1]
    sndAcrossSurvivors = acrossSurvivorValues[-2]

    # Get the corresponding key/s for across all ages
    bestAcrossIndex = np.argwhere(acrossSurvivorArray == maxAcrossSurvivors).flatten()
    sndAcrossIndex = np.argwhere(acrossSurvivorArray == sndAcrossSurvivors).flatten()
    strategies = np.array(parameterList)
    bestAcrossStrategies = strategies[bestAcrossIndex].tolist()
    sndAcrossStrategies = strategies[sndAcrossIndex].tolist()

    # Now store all this in a dict
    resultDict['across'] = {}
    resultDict['across']['maxSurv'] = maxAcrossSurvivors
    resultDict['across']['sndSurv'] = sndAcrossSurvivors
    resultDict['across']['bestStrat'] = bestAcrossStrategies
    resultDict['across']['sndStrat'] = sndAcrossStrategies


    # And now for the within site results
    withinSurvivorValues = np.unique(withinSurvivorArray)
    maxWithinSurvivors = withinSurvivorValues[-1]
    sndWithinSurvivors = withinSurvivorValues[-2]

    # Get the corresponding strategies
    bestWithinIndex = np.argwhere(withinSurvivorArray == maxWithinSurvivors).flatten()
    sndWithinIndex = np.argwhere(withinSurvivorArray == maxWithinSurvivors).flatten()
    bestWithinStrategies = strategies[bestWithinIndex]
    sndWithinStrategies = strategies[sndWithinIndex]

    # And also store it all in a dict
    resultDict['within'] = {}
    resultDict['within']['maxSurv'] = maxWithinSurvivors
    resultDict['within']['sndSurv'] = sndWithinSurvivors
    resultDict['within']['bestStrat'] = bestWithinStrategies
    resultDict['within']['sndStrat'] = sndWithinStrategies


    # Now we can start printing, plotting and saving
    if doPrint:
        for focus in resultDict.keys():
            tempDict = resultDict[focus]
            print('\n\n'
                  + 'Results for ' + focus + ' sampling:')
            maxSurv = tempDict['maxSurv']
            sndSurv = tempDict['sndSurv']
            bestStrat = tempDict['bestStrat']
            sndStrat = tempDict['sndStrat']

            print('The highest nuber of sujects was ' + str(maxSurv))
            print('This number was achieved by these strategies:')
            for strat in bestStrat:
                print('    ' + strat)

            print('\nThe second highest number of subjects was ' + str(sndSurv))
            print('This number was achieved by these strategies:')
            for strat in sndStrat:
                print('    ' + strat)

    # Plotting
    if doPlot:
        for focus in resultDict.keys():
            tempStrat = stratDict[focus]
            tempDict = resultDict[focus]
            print('\n\n'
                  + 'Plotting for ' + focus + ' sampling:')
            maxSurv = tempDict['maxSurv']
            sndSurv = tempDict['sndSurv']
            bestStrat = tempDict['bestStrat']
            sndStrat = tempDict['sndStrat']

            # Plot for the best strategies first
            print('Plotting best strategies for ' + focus
                  + ' with ' + str(maxSurv) + ' survivors')
            plotSurvivorSites(bestStrat, tempStrat,
                              subjectDict, minNumBin,
                              pathToOutDir, focus)

            # Now plot for the second best strategy
            print('Plotting second best strategies for ' + focus
                  + ' with ' + str(sndSurv) + ' survivors')
            plotSurvivorSites(sndStrat, tempStrat,
                              subjectDict, minNumBin,
                              pathToOutDir, focus)

    print('\n\nThis is it. Plotting done.\n')

if __name__ == '__main__':
    Main()
    pass
