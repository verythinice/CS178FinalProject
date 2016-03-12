#placeholder for kernel knn experiments
'''
Created on Mar 5, 2016

@author: Ben
'''
import numpy as np
import matplotlib.pyplot as pl
import mltools as ml
from sklearn.neighbors.kde import KernelDensity

#X1=np.genfromtxt("data/kaggle.X1.train.txt", delimiter = ",")
#Y=np.genfromtxt("data/kaggle.Y.train.txt", delimiter = ",")
#Xtr,Xte,Ytr,Yte = ml.splitData(X1,Y,0.75)

data = np.genfromtxt("data/curve80.txt", delimiter=None)
X = data[:, 0]
X = X[:, np.newaxis]
Y = data[:, 1]
Xtr, Xte, Ytr, Yte = ml.splitData(X, Y, 0.75)

KernelLearner = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(Xtr)
Ye = KernelLearner.score_samples(Xtr)

def KernelMSE(Y, Yhat):
    error = 0
    for i in range(len(Y)):
        error += (Y[i] - Yhat[i])**2
    return error / len(Y)

currentMSE = KernelMSE(Ytr, Ye)

print currentMSE

