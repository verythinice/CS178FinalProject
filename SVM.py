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

#Ihler's code for soft margin linear SVM - which might be useful to play around with
#not prepared for the data

class linearSVM(ml.classifier):
    def predict(self, X):
        Z = self.theta[:,0].T + X.dot( self.theta[:,1:].T )
        Yhat = 2*(Z>0)-1  # np.sign(Z) without sign(0)=0
        return Yhat

sv = linearSVM()

from numpy import atleast_2d as twod
from numpy import asarray as arr

M,N = X.shape
sv.theta = np.array([[-3,-1,.5]])

alpha = 0.01
reg = 1e-4
for it in range(1000):   # 100 iterations:
    for j in range(M):
        zj = sv.theta[0,0]+twod(X[j,:]).dot(sv.theta[0,1:].T)  # compute linear response
        #print zj
        # Now, compute the gradient of the hinge loss:
        gradj = 0 if zj*Y[j] > 1.0 else -Y[j]*arr([[1,X[j,0],X[j,1]]])
        # plus the gradient of the L2 regularization term:
        gradj += reg * sv.theta
        # and update theta:
        sv.theta -= alpha/(it+1) * gradj
    if it<10: print "Error rate: {} \t(iter {})".format(sv.err(X,Y),it)
print "...\nTheta:",sv.theta