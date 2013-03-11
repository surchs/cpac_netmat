'''
Created on Mar 9, 2013

@author: surchs
'''
import sys


def Main(subjectList):
    listSubjects = open(subjectList, 'rb').readlines()
    uniqueSubjects = []

    for subject in listSubjects:
        subject = subject.strip()
        if not subject in uniqueSubjects:
            uniqueSubjects.append(subject)
        else:
            print(subject + ' is a goddamn duplicate')

    print('I got ' + str(len(uniqueSubjects)) + ' unique subjects out of '
          + str(len(listSubjects)))

if __name__ == '__main__':
    subjectList = sys.argv[1]
    Main(subjectList)
    pass
