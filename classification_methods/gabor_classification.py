import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

train = np.load("new_result/gabor_pca.npy")
[m,n] = np.shape(train)
label = [4,0]
X = np.zeros([3480,n])
Y = np.zeros([3480])
for i in range(1740):
    for j in range(n):
        X[i,j] = train[label[0]*1740+i,j]
        Y[i] = 0
        X[i+1740,j] = train[label[1]*1740+i,j]
        Y[i+1740] = 1
X = np.nan_to_num(X)
Y = np.nan_to_num(Y)
print(np.shape(X))

clf = RandomForestClassifier(n_jobs = -1,n_estimators = 100)

predicted = cross_val_predict(clf,X,Y,cv=100)

print(metrics.accuracy_score(Y,predicted))
