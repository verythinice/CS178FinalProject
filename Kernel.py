#placeholder for kernel SVM experiments
'''
Created on Mar 5, 2016

@author: Ben
'''
import numpy as np
import matplotlib.pyplot as pl
import mltools as ml

X1=np.genfromtxt("data/kaggle.X1.train.txt", delimiter = ",")
Y=np.genfromtxt("data/kaggle.Y.train.txt", delimiter = ",")
Xtr,Xte,Ytr,Yte = ml.splitData(X1,Y,0.75)

#Should we use the libsvm code from online?
