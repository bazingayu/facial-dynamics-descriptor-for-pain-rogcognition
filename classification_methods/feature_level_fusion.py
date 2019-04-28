from sklearn import tree
from itertools import product
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import random
dis_Data = np.load('new_result2/dis.npy')   #读取文件
Euler_Data = np.load('new_result2/Euler.npy')
gra_Data = np.load('new_result2/gra.npy')
filenames = np.load('new_result/parta_name.npy')
bio_Data = np.load('new_result/parta_biosignal.npy')
print(np.shape(bio_Data))
[m,n] = np.shape(filenames)
person = []
for i in range(m):                                       #取出人的编号生成一个87的list
    for j in range(n):
        s = filenames[i,j].split('-')
        if person == []:
            person.append(s[0])
        else:
            flag = 0
            for k in range(len(person)):
                if s[0] == person[k]:
                    flag = 1
            if flag == 0:
                person.append(s[0])
print(len(person))

dis_Data_Norm = []
Euler_Data_Norm = []
bio_Data_Norm = []
gra_Data_Norm = []
Y = []
for k in range(len(person)):                          #根据人名组织文件，生成87*100*n数组
    f3 = []
    f1 = []
    f2 = []
    f4 = []
    y = []
    for i in range(m):
        for j in range(n):
            s = filenames[i,j].split('-')
            if s[0] == person[k]:
                f2.append(gra_Data[i,j,:])
                f1.append(dis_Data[i,j,:])
                f3.append(Euler_Data[i,j,:])
                f4.append(bio_Data[i,j,:])
                y.append(i)
    gra_Data_Norm.append(f2)
    dis_Data_Norm.append(f1)
    Euler_Data_Norm.append(f3)
    bio_Data_Norm.append(f4)
    Y.append(y)
Y = np.array(Y)
dis_Data_Norm = np.array(dis_Data_Norm)
gra_Data_Norm = np.array(gra_Data_Norm)
Euler_Data_Norm = np.array(Euler_Data_Norm)
bio_Data_Norm = np.array(bio_Data_Norm)



[m,n,z1] = np.shape(dis_Data_Norm)
[m,n,z2] = np.shape(gra_Data_Norm)
[m,n,z3] = np.shape(Euler_Data_Norm)
[m,n,z4] = np.shape(bio_Data_Norm)
dis_sum = np.zeros([m,z1])
gra_sum = np.zeros([m,z2])
Euler_sum = np.zeros([m,z3])
bio_sum = np.zeros([m,z4])
for i in range(m):
    for k in range(z1):
        for j in range(n):
            dis_sum[i,k] = dis_sum[i,k] + dis_Data_Norm[i,j,k]
for i in range(m):
    for k in range(z2):
        for j in range(n):
            gra_sum[i,k] = gra_sum[i,k] + gra_Data_Norm[i,j,k]
for i in range(m):
    for k in range(z3):
        for j in range(n):
            Euler_sum[i,k] = Euler_sum[i,k] + Euler_Data_Norm[i,j,k]
for i in range(m):
    for k in range(z4):
        for j in range(n):
            bio_sum[i,k] = bio_sum[i,k] + bio_Data_Norm[i,j,k]
dis_ave = dis_sum/100
gra_ave = gra_sum/100
Euler_ave = Euler_sum/100
bio_ave = bio_sum/100

dis_sum_val = np.zeros([m,z1])
gra_sum_val = np.zeros([m,z2])
Euler_sum_val = np.zeros([m,z3])
bio_sum_val = np.zeros([m,z4])
for i in range(m):
    for k in range(z1):
        for j in range(n):
            dis_sum_val[i,k] = dis_sum_val[i,k] + (dis_Data_Norm[i,j,k] - dis_ave[i,k]) ** 2
for i in range(m):
    for k in range(z2):
        for j in range(n):
            gra_sum_val[i,k] = gra_sum_val[i,k] + (gra_Data_Norm[i,j,k] - gra_ave[i,k]) ** 2
for i in range(m):
    for k in range(z3):
        for j in range(n):
            Euler_sum_val[i,k] = Euler_sum_val[i,k] + (Euler_Data_Norm[i,j,k] - Euler_ave[i,k]) ** 2
for i in range(m):
    for k in range(z4):
        for j in range(n):
            bio_sum_val[i,k] = bio_sum_val[i,k] + (bio_Data_Norm[i,j,k] - bio_ave[i,k]) ** 2

dis_sigma = (dis_sum_val/100)**0.5
gra_sigma = (gra_sum_val/100)**0.5
Euler_sigma = (Euler_sum_val/100)**0.5
bio_sigma = (bio_sum_val/100)**0.5
for i in range(m):
    for j in range(n):
        for k in range(z1):
            if dis_sigma[i,k] == 0:
                dis_sigma[i,k] = 0.0000001
            dis_Data_Norm[i,j,k] = (dis_Data_Norm[i,j,k] - dis_ave[i,k]) / dis_sigma[i,k]
for i in range(m):
    for j in range(n):
        for k in range(z2):
            if gra_sigma[i,k] == 0:
                gra_sigma[i,k] = 0.0000001
            gra_Data_Norm[i,j,k] = (gra_Data_Norm[i,j,k] - gra_ave[i,k]) / gra_sigma[i,k]
for i in range(m):
    for j in range(n):
        for k in range(z3):
            if Euler_sigma[i,k] == 0:
                Euler_sigma[i,k] = 0.0000001
            Euler_Data_Norm[i,j,k] = (Euler_Data_Norm[i,j,k] - Euler_ave[i,k]) / Euler_sigma[i,k]
for i in range(m):
    for j in range(n):
        for k in range(z4):
            if bio_sigma[i,k] == 0:
                bio_sigma[i,k] = 0.0000001
            bio_Data_Norm[i,j,k] = (bio_Data_Norm[i,j,k] - bio_ave[i,k]) / bio_sigma[i,k]


X1 = []
Y1 = []
decision_matrix = []
for i in range(m):
    for j in range(n):
        X1.append(dis_Data_Norm[i,j,:])
        Y1.append(Y[i,j])

Y1  = np.array(Y1)

X2 = []
for i in range(m):
    for j in range(n):
        X2.append(gra_Data_Norm[i,j,:])



X3 = []
for i in range(m):
    for j in range(n):
        X3.append(Euler_Data_Norm[i,j,:])


X4 = []
for i in range(m):
    for j in range(n):
        X4.append(bio_Data_Norm[i,j,:])

X5 = np.load("new_result2/Gabor_norm_pca.npy")

X = []
for i in range(len(X1)):
    single = X1[i].tolist()
    single.extend(X2[i])
    single.extend(X3[i])
    single.extend(X4[i])
    single.extend(X5[i])
    X.append(single)
X = np.array(X)
X = np.nan_to_num(X)
print(np.shape(X))




kf = KFold(n_splits = 10)

li = []
for i in range(8700):
    li.append(i)
random.shuffle(li)
final_result = []
for train_n , test_n in kf.split(X):
    X_train1 = []
    X_test1 = []

    Y_train1 = []
    Y_test1 = []
    for i in range(len(train_n)):
        X_train1.append(X[li[train_n[i]]])

        Y_train1.append(Y1[li[train_n[i]]])
    for j in range(len(test_n)):
        X_test1.append(X[li[test_n[j]]])

        Y_test1.append(Y1[li[test_n[j]]])


    clf1 = RandomForestClassifier(n_jobs=-1,n_estimators=1000)
    clf1.fit(X_train1,Y_train1)
    X11_decision_matrix = np.transpose(clf1.predict_proba(X_test1))
    X11_pred = np.argmax(clf1.predict_proba(X_test1),1)

    [zz] = np.shape(X11_pred)


    count = 0

    for iii in range(zz):
        if(X11_pred[iii] == Y_test1[iii]):
            count += 1


    #print(np.shape(y_pred1),np.shape(Y_test1))

    final_result.append(count/zz)
print(final_result)

print(np.mean(final_result))