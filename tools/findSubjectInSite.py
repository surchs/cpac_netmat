'''
Created on Mar 9, 2013

@author: surchs
'''
import os
import sys
import glob


def Main(subList, searchDir):

    subjects = open(subList, 'rb').readlines()

    for subject in subjects:
        subject = subject.strip()

        searchString = (os.path.abspath(searchDir) + '/*/' + subject + '/')
        search = glob.glob(searchString)
        if not len(search) == 1:
            if len(search) > 1:
                print('Your subject ' + subject + ' was found more than once:\n'
                      + '    ' + str(search) + '\n')
            else:
                print('\nI haven\'t found subject ' + subject + ' anywhere\n')
        else:
            print('Found subject ' + subject + ':' + str(search[0]))

if __name__ == '__main__':
    subList = sys.argv[1]
    searchDir = sys.argv[2]
    Main(subList, searchDir)
    pass
