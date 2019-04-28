from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import random

label = [1,0]

X1_Data = np.load('new_result/parta_dis_texture_eulur_descriptor.npy')

[m,n,z] = np.shape(X1_Data)
X1 = []
Y = []
decision_matrix = []
for i in range(2):
    for j in range(n):
        X1.append(X1_Data[label[i],j,:])
        Y.append(i)
X1 = np.array(X1)
Y  = np.array(Y)
X1 = np.nan_to_num(X1)
X2_Data = np.load('new_result/parta_biosignal.npy')
X2 = []
for i in range(2):
    for j in range(n):
        X2.append(X2_Data[label[i],j,:])
X2 = np.array(X2)
X2 = np.nan_to_num(X2)


X1_train, X1_test, X2_train, X2_test, Y_train, Y_test = train_test_split(X1,X2,Y,test_size=0.2)

clf1 = RandomForestClassifier(n_jobs=-1,n_estimators=100)
clf1.fit(X1_train,Y_train)
X1_pre = clf1.predict(X1_test)
X1_decision_matrix = np.transpose(clf1.predict_proba(X1_test))
decision_matrix.append(X1_decision_matrix.tolist())


clf2 = RandomForestClassifier(n_jobs=-1,n_estimators=100)
clf2 = clf2.fit(X2_train,Y_train)
clf2.predict(X2_test)
X2_decision_matrix = np.transpose(clf2.predict_proba(X2_test))
decision_matrix.append(X2_decision_matrix.tolist())

decision_matrix = np.array(decision_matrix)


[mm,nn,zz] = np.shape(decision_matrix)
print(mm,nn,zz)

#weight
w=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.90,0.95]
#w=[0.1,0.5,0.90,0.95]
lenth = len(w)
decision = np.zeros([zz, nn])
max_weight = []
max_count = 0
for i1 in range(lenth):
    for i2 in range(lenth):
        count = 0
        weight = [[w[i1],w[i2]],[1-w[i1],1-w[i2]]]
        for i in range(zz):
            for k in range(nn):
                decision[i, k] = weight[0][k] * decision_matrix[0, k, i] + weight[1][k] * decision_matrix[1, k, i]
        y_pred = np.argmax(decision,1)
        for iii in range(zz):
            if(y_pred[iii] == Y_test[iii]):
                 count += 1
        if(count > max_count):
            max_count = count
            max_weight = weight
print('final_weight',max_weight)
print('accuracy', max_count/zz)


kf = KFold(n_splits = 10)
decision1 = np.zeros([348,nn])
final_result1 = []
final_result2 = []  #两个分类器分别的准确率
final_result = []
li = []
for i in range(3480):
    li.append(i)
random.shuffle(li)

for train_n , test_n in kf.split(X1):
    X1_train1 = []
    X1_test1 = []
    X2_train1 =[]
    X2_test1 = []
    Y_train1 = []
    Y_test1 = []
    for i in range(len(train_n)):
        X1_train1.append(X1[li[train_n[i]]])
        X2_train1.append(X2[li[train_n[i]]])
        Y_train1.append(Y[li[train_n[i]]])
    for j in range(len(test_n)):
        X1_test1.append(X1[li[test_n[j]]])
        X2_test1.append(X2[li[test_n[j]]])
        Y_test1.append(Y[li[test_n[j]]])
    print(Y_test1[1:100])
    decision_matrix = []
    clf3 = RandomForestClassifier(n_jobs=-1,n_estimators=100)
    clf3.fit(X1_train1,Y_train1)
    X3_decision_matrix = np.transpose(clf3.predict_proba(X1_test1))
    X3_pred = np.argmax(clf3.predict_proba(X1_test1),1)
    clf4 = RandomForestClassifier(n_jobs=-1,n_estimators=100)
    clf4.fit(X2_train1,Y_train1)
    X4_pred = np.argmax(clf4.predict_proba(X2_test1),1)
    print('x4_pred',X4_pred)
    X4_decision_matrix = np.transpose(clf4.predict_proba(X2_test1))
    decision_matrix.append(X3_decision_matrix)
    decision_matrix.append(X4_decision_matrix)
    decision_matrix = np.array(decision_matrix)
    [mm,nn,zz] = np.shape(decision_matrix)
    
    for i in range(zz):
        for k in range(nn):
            decision1[i, k] = max_weight[0][k] * decision_matrix[0, k, i] + max_weight[1][k] * decision_matrix[1, k, i]
    y_pred1 = np.argmax(decision1,1)
    count = 0
    count1 = 0
    count2 = 0
    for iii in range(zz):
        if(y_pred1[iii] == Y_test1[iii]):
            count += 1
        if(X3_pred[iii] == Y_test1[iii]):
            count1 += 1
        if(X4_pred[iii] == Y_test1[iii]):
            count2 += 1

    print(np.shape(y_pred1),np.shape(Y_test1))
    final_result1.append(count1/zz)
    final_result2.append(count2/zz)
    final_result.append(count/zz)
print(final_result1,final_result2,final_result)

print(np.mean(final_result1),np.mean(final_result2),np.mean(final_result))

