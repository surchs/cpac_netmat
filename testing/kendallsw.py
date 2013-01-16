'''
Created on Jan 9, 2013

@author: surchs

short script to calculate kendalls w
'''
import numpy as np


def Main():
    # number of networks and subjects
    superArray = np.random.random((6, 12))
    nNetworks = superArray.shape[0]
    nSubs = superArray.shape[1]
    subjects = []
    subRanks = np.array([])
    # mean total value of ranks
    meanRank = 1 / 2 * nNetworks * (nSubs + 1)
    
    
    # calculate the rank for every subject
    for subject in subjects:
        subRatings = np.array({})
        for rater in raters:
            # get the rating from the current network
            subRating = ...
            subRatings = np.append(subRatings, subRating)
            
        
        
        

    # and the mean of all the ranks
    meanR = 1 / 2 * nNetworks * (nSubs + 1)

    # get the squared rank deviations for every subject
    subSqSums = np.array([])
    for sub in subjects:
        rankDevSq = np.square(subRank - meanRank)
        subSq = np.append(subSq, rankDevSq)

    # sum of squared deviations
    sSum = np.sum(subSq)

    kendallW = 12 * sSum / np.square(nNetworks) * (nSubs ** 3 - nSubs)


if __name__ == '__main__':
    pass
