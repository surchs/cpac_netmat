'''
Created on Nov 30, 2012

@author: sebastian

a collection of useful small stuff that I like
'''


def isNumber(testString):
    '''
    A method to check if a string can be converted to a number
    Not my creation, took it from http://stackoverflow.com/
    '''
    try:
        float(testString)
        return True
    except ValueError:
        return False

if __name__ == '__main__':
    pass
