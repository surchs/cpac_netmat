'''
Created on Nov 26, 2012

@author: sebastian
'''
import numpy as np
from random import shuffle


def Main():
    print('Main')
    x = 1
    b = []
    for y in np.arange(10):
        b.append(np.arange(15 - y, 15))

    shuffle(b)

    for x in np.arange(10):
        for y in b:
            if x in y:
                print('here' + str(x))
                break
        else:
            print('Hallo Welt' + str(x))

if __name__ == '__main__':
    Main()
    pass
