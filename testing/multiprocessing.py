'''
Created on Dec 6, 2012

@author: sebastian
'''
import time
import numpy as np
import multiprocessing as mp


class container(object):
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.mean = None

    def doSomething(self):
        time.sleep(10)
        self.mean = np.mean(self.data)


def mach(container):
    container.doSomething()
    return container


def Main():
    lulu = []
    for i in np.arange(100, 1000, 50):
        tempData = np.random.random((i, i))
        tempCon = container(str(i), tempData)
        lulu.append(tempCon)

    pool = mp.Pool(processes=10)
    resultList = pool.map(mach, lulu)
    for result in resultList:
        print(str(result.mean))

    print('\nSleeping now')
    time.sleep(240)
    print('Sleeping done')




if __name__ == '__main__':
    pass
