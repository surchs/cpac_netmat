'''
Created on Nov 26, 2012

@author: sebastian
'''
import sys


def Main(inFile):
    f = open(inFile, 'rb')
    print(f.read())
    f.seek(0)
    lines = f.readlines()
    for i in range(3):
        for line in lines:
            print line.strip()

    f.seek(0)

    print(lines)
    print type(f.read())
    print('\n\n')

    if 'Hallo' in lines:
        print 'Hallo'
    else:
        print('nay')

    if 'Hallo' in f.read():
        print('hui')
        f.seek(0)
        print f.read()
    else:
        print('doof')



if __name__ == '__main__':
    inFile = sys.argv[1]
    Main(inFile)
    pass
