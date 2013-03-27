'''
Created on Mar 14, 2013

@author: surchs

Script to append an existing phenotypic file by new entries from another file
'''


def readPheno(pathToPhenoFile):
    '''
    Method to read the file and make a dictionary of header indices
    '''
    # Prepare the dictionary with the index
    indexDict = {}

    # Open and read the file
    phenoFile = open(pathToPhenoFile, 'rb')
    header = phenoFile.readline().strip().split(',')
    phenoData = phenoFile.readlines()

    # Get the indices for the column names in the header
    for i, colName in enumerate(header):
        indexDict[colName] = i

    return indexDict, phenoData, header


def writePheno(pathToOutput, listOfStrings):
    '''
    Method to write the combined output
    '''
    outF = open(pathToOutput, 'wb')
    outF.writelines(listOfStrings)
    outF.close()
    print('Finished writing at ' + pathToOutput)


def Main():
    # Define inputs
    pathToOriginalPheno = '/home2/surchs/secondLine/configs/wave/wave_pheno81_uniform.csv'
    pathToAddPheno = '/home2/data/Projects/wave_uniform/output_dir/pipeline_HackettCity/scan_func_rest_threshold_0.2_all_params.csv'

    # Define parameters
    origSub = 'subject'
    addSub = 'Subject'
    listOfAddedCov = ['MeanFD',
                        'NumFD_greater_than_0.20']

    # Define outputs
    combinedPheno = []
    pathToCombinedPheno = '/home2/surchs/secondLine/configs/wave/wave_pheno81_uniform_combined.csv'

    # Read the inputs
    origIndex, origPheno, origHeader = readPheno(pathToOriginalPheno)
    addIndex, addPheno, addHeader = readPheno(pathToAddPheno)

    # Prepare the outputs
    outHeaderStr = ''
    # first the original header
    for colName in origHeader:
        outHeaderStr = (outHeaderStr + colName + ',')
    # now the added covariates
    for addCov in listOfAddedCov:
        outHeaderStr = (outHeaderStr + addCov + ',')
    # now strip the last ',' and add newline
    outHeaderStr = (outHeaderStr.strip(',') + '\n')
    # And place it as first entry in the list of strings to write
    combinedPheno.append(outHeaderStr)

    # Go through each existing subject in the original Pheno file
    for origLine in origPheno:
        origUseLine = origLine.strip().split(',')
        origSubject = origUseLine[origIndex[origSub]]

        # Now go through the add file and search for the subject
        for addLine in addPheno:
            addUseLine = addLine.strip().split(',')
            addSubject = addUseLine[addIndex[addSub]]

            # Check if the subject is in there
            if origSubject in addSubject:
                # We have the right subject
                print('Found the match for ' + origSubject + ' in '
                      + addSubject)
                # Get the covariates to add
                for covariate in listOfAddedCov:
                    targetCov = addUseLine[addIndex[covariate]]
                    # and append this to the original line
                    origLine = (origLine.strip() + ',' + targetCov)
                # Add newline and append it to the output
                origLine = (origLine + '\n')
                combinedPheno.append(origLine)
            else:
                continue

    # Now write it down
    writePheno(pathToCombinedPheno, combinedPheno)


if __name__ == '__main__':
    Main()
    pass
