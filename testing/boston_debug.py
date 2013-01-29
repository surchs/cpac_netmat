'''
Created on Jan 28, 2013

@author: surchs
'''
import numpy as np
from sklearn.svm import SVR
from matplotlib import pyplot as plt
from sklearn.datasets import  load_boston
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression


# exchange the loaded datasets to get a working example with load_diabetes()

# data = load_diabetes()
data = load_boston()
X = data.data
y = data.target
# also dump some stuff here
np.savetxt('bd_X_full.csv', X, fmt='%10.5f', delimiter=',')
np.savetxt('bd_Y_full.csv', y, fmt='%10.5f', delimiter=',')

# prepare the training and testing data for the model
nCases = len(y)
nTrain = np.floor(nCases / 2)
trainX = X[:nTrain]
trainY = y[:nTrain]
testX  = X[nTrain:]
testY = y[nTrain:]

# also dump the training and test sets
np.savetxt('bd_trainX.csv', trainX, fmt='%10.5f', delimiter=',')
np.savetxt('bd_trainY.csv', trainY, fmt='%10.5f', delimiter=',')
np.savetxt('bd_testX.csv', testX, fmt='%10.5f', delimiter=',')
np.savetxt('bd_testY.csv', testY, fmt='%10.5f', delimiter=',')

# define parameters for the models
# SVR is set to rbf-kernel and C=1000 by parameter estimation for diabetes data
# running grid search on boston data doesn't improve this models performance
svr = SVR(kernel='rbf', C=1000)
log = LinearRegression()

# train both models
svr.fit(trainX, trainY)
log.fit(trainX, trainY)


# predict test labels from both models
predLog = log.predict(testX)
predSvr = svr.predict(testX)

# dump the predicted labels
np.savetxt('bld_predSvrY.csv', predLog, fmt='%10.5f', delimiter=',')
np.savetxt('bld_predLogY.csv', predSvr, fmt='%10.5f', delimiter=',')

# show it on the plot
plt.plot(testY, testY, label='true data')
plt.plot(testY, predSvr, 'co', label='SVR')
plt.plot(testY, predLog, 'mo', label='LogReg')
plt.legend()
plt.show()