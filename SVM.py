'''
Created on Mar 5, 2016

@author: Ben
'''
import numpy as np
import matplotlib.pyplot as pl
import mltools as ml
from sklearn.svm import SVR


#X1=np.genfromtxt("data/kaggle.X1.train.txt", delimiter = ",")
#Y=np.genfromtxt("data/kaggle.Y.train.txt", delimiter = ",")
#Xtr,Xte,Ytr,Yte = ml.splitData(X1,Y,0.75)

#Trying to do this with Scikit learn

#WARNING: this takes a long time to run 

#Right now, the project is just training on every element in the training set
X1 = np.genfromtxt("data/kaggle.X1.train.txt", delimiter=",")
Y = np.genfromtxt("data/kaggle.Y.train.txt", delimiter=",")


#testing the data
Xe1 = np.genfromtxt("data/kaggle.X1.test.txt", delimiter=",")


#There's three different kind of SVR according to sklearn: SVR, NuSVR, and LinearSVR
#They are all based on libSVM.

#In SVR, there are five built-in kernels: rbf, linear, poly, sigmoid, or precomputed.
#Precomputed requires different parameters rather than just changing the parameter for kernel
#With this default code below, which I ran with rbf, I got 0.83858, which is kinda bad.

#We can also play around with C, gamma, and the other parameters in SVR

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)

Ye = svr_rbf.fit(X1, Y).predict(Xe1)

Ye = Ye.ravel()
    
fh = open('predictions.csv','w')    # open file for upload
fh.write('ID,Prediction\n')         # output header line
for i,yi in enumerate(Ye):
  fh.write('{},{}\n'.format(i+1,yi)) # output each prediction
fh.close()                          # close the file

