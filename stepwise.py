from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
from sklearn.svm import SVR

import numpy as np
import mltools as ml
%matplotlib inline

X = np.genfromtxt("kaggle.X1.train.txt",delimiter=",")
Y = np.genfromtxt("kaggle.Y.train.txt",delimiter=",")
[Xt,Xv,Yt,Yv] = ml.splitData(X,Y,0.75)
svc = SVR(kernel="linear")

# this will literally take about forever.  on 60K points of data, ~7 hours.
# features to select = features that will be kept
# step size is number of features to eliminate each recursive evaluation of the function.
# while it shortens the time somewhat, it also produces more error.

RFE(estimator=svc, n_features_to_select=70, step=10)
rfe.fit(Xt, Yt)

print rfe.ranking_.shape
ranking = rfe.ranking_.reshape((1,91))

# Plotting the ranking is helpful to see which features, corresponding to their
# indices in the feature vector, are kept.  Features with ranking #1 are kept.

plt.matshow(ranking)
plt.colorbar()
plt.title("Ranking of pixels with RFE")
plt.show()

# Now creating the feature vector and corresponding Y-values for the basic
# linear regressor.

features_list = []
Training = []
print Xv.shape
print rfe.ranking_
for i in range (len(Xv[1])):
        if rfe.ranking_[i] == 1:
                    features_list.append(Xv[i])
                            Training.append(Yv[i])

import mltools.linear
reload(ml.linear)
lr = ml.linear.linearRegress(features_list,Training, reg=0) # init and train the model

Xt = np.genfromtxt("kaggle.X1.train.txt",delimiter=",")
Yt = np.genfromtxt("kaggle.Y.train.txt",delimiter=",")
print Xt.shape, Yt.shape # 60k training examples

Xe = np.genfromtxt("kaggle.X1.test.txt",delimiter=",")
YeHat = lr.predict(Xe)

YeHat = YeHat.ravel()

open("predictions2.csv","w") # open file for upload
fh.write("ID,Prediction\n") # output header line
for i,yi in enumerate(YeHat):
        fh.write("{},{}\n".format(i+1,yi)) # output each prediction
        fh.close() # close the file
