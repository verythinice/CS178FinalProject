
from sklearn.datasets import load_digits
from sklearn import linear_model
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.svm import SVR

import numpy as np
import mltools as ml
%matplotlib inline

# Load the digits dataset

X = np.genfromtxt("kaggle.X1.train.txt",delimiter=",")
Y = np.genfromtxt("kaggle.Y.train.txt",delimiter=",")
[Xt,Xv,Yt,Yv] = ml.splitData(X,Y,0.999)
Xe = np.genfromtxt("kaggle.X1.test.txt",delimiter=",")

print Xv.shape, Yv.shape

# Create the RFE object and rank each pixel
#regr = linear_model.LinearRegression()
#regr.fit(xs, ys)

svc = SVR(kernel="linear")

rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
rfe.fit(Xv, Yv)
ranking = rfe.ranking_.reshape(digits.images[0].shape)

#Plot pixel ranking
plt.matshow(ranking)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()
