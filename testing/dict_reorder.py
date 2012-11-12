'''
Created on Nov 1, 2012

@author: sebastian
'''


def Main():
    pipes = ['pipe1', 'pipe2']
    subs = ['sub1', 'sub2']
    deris = ['func', 'alff', 'cent']
    masks = ['dose', 'yeo']

    firstDict = {}
    for pipe in pipes:
        if not pipe in firstDict.keys():
            firstDict[pipe] = {}

        for deri in deris:
            if not deri in firstDict[pipe].keys():
                firstDict[pipe][deri] = {}

            for sub in subs:
                if not sub in firstDict[pipe][deri].keys():
                    firstDict[pipe][deri][sub] = {}

                    for mask in masks:
                        firstDict[pipe][deri][sub][mask] = mask

    print 'firstDict complete', firstDict.keys(), '\n\n'

    '''
    The target order for the second dict is:
        1) pipes
        2) masks
        3) subs
        4) deris
    '''

    secondDict = {}

    for pipe in firstDict.keys():
        if not pipe in secondDict.keys():
            secondDict[pipe] = {}

        for deri in firstDict[pipe].keys():

            for sub in firstDict[pipe][deri].keys():

                for mask in firstDict[pipe][deri][sub].keys():
                    target = firstDict[pipe][deri][sub][mask]
                    if not mask in secondDict[pipe].keys():
                        secondDict[pipe][mask] = {}
                    if not sub in secondDict[pipe][mask].keys():
                        secondDict[pipe][mask][sub] = {}
                    if not deri in secondDict[pipe][mask][sub].keys():
                        secondDict[pipe][mask][sub][deri] = target
                    else:
                        print 'strangely, this is already here', pipe, mask

    print 'second dict complete', secondDict

    pass


if __name__ == '__main__':
    Main()
    pass
