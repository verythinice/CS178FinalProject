'''
Created on Mar 5, 2016

@author: Ben
'''
import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from sklearn.svm import SVR

def SVMMSE(Y, Yhat):
    error = 0
    for i in range(len(Y)):
        error += (Y[i] - Yhat[i])**2
    return error / len(Y)

#WARNING: this takes a long time to run with the kaggle training data

X1=np.genfromtxt("data/kaggle.X1.train.txt", delimiter = ",")
Y=np.genfromtxt("data/kaggle.Y.train.txt", delimiter = ",")
Xtr,Xte,Ytr,Yte = ml.splitData(X1[:200],Y[:200],0.75)
#Kaggle testing data
Xe1 = np.genfromtxt("data/kaggle.X1.test.txt", delimiter=",")

# data = np.genfromtxt("data/curve80.txt", delimiter=None)
# X = data[:, 0]
# X = X[:, np.newaxis]
# Y = data[:, 1]
# Xtr, Xte, Ytr, Yte = ml.splitData(X, Y, 0.75)


# #Training and predicting with the SVM
# kernelTypes = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
# for i in range(1,10):
#     svr_rbf = SVR(kernel='rbf', C=1e3, gamma=(10**(-10)), epsilon = i*(10**(-1)))
#     Ye = svr_rbf.fit(Xtr, Ytr).predict(Xte)
#     YhatTr=svr_rbf.predict(Xtr)
#     #Ye = [4, 8.5, 15, 2, 10]
#     #Yte = [5, 8, 10, 2, 11]
#     trainingMSE=SVMMSE(Ytr, YhatTr)
#     currentMSE = SVMMSE(Yte, Ye)
#        
#     print (i," ",trainingMSE," ",currentMSE)

#for writing the things to kaggle
# svr_rbf = SVR(kernel='linear', C=10**(-7), epsilon=.3)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=10**(-10), epsilon=.3)
Ye = svr_rbf.fit(X1, Y).predict(Xe1)
Ye = Ye.ravel()
         
fh = open('predictions.csv','w')    # open file for upload
fh.write('ID,Prediction\n')         # output header line
for i,yi in enumerate(Ye):
    fh.write('{},{}\n'.format(i+1,yi)) # output each prediction
fh.close()                          # close the file

#There's three different kind of SVR according to sklearn: SVR, NuSVR, and LinearSVR
#They are all based on libSVM.

#In SVR, there are five built-in kernels: rbf, linear, poly, sigmoid, or precomputed.
#Precomputed requires different parameters rather than just changing the parameter for kernel
#With this default code below, which I ran with rbf, I got 0.83858, which is kinda bad.

#We can also play around with C, gamma, and the other parameters in SVR