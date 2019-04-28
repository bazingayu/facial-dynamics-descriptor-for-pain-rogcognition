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
decision_matrix1 = []
for i in range(m):
    for j in range(n):
        X1.append(dis_Data_Norm[i,j,:])
        Y1.append(Y[i,j])
X1 = np.array(X1)
Y1  = np.array(Y1)
X1 = np.nan_to_num(X1)

X2 = []
for i in range(m):
    for j in range(n):
        X2.append(gra_Data_Norm[i,j,:])
X2 = np.array(X2)
X2 = np.nan_to_num(X2)

X3 = []
for i in range(m):
    for j in range(n):
        X3.append(Euler_Data_Norm[i,j,:])
X3 = np.array(X3)
X3 = np.nan_to_num(X3)
'''
X4 = []
for i in range(m):
    for j in range(n):
        X4.append(bio_Data_Norm[i,j,:])
X4 = np.array(X4)
X4 = np.nan_to_num(X4)
'''
X5 = np.load("new_result2/Gabor_norm_pca.npy")

max_weight1 = []
for ii in range(1):
    X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X5_train, X5_test, Y_train, Y_test = train_test_split(X1,X2,X3,X5,Y1,test_size=0.2)
    clf1 = RandomForestClassifier(n_jobs=-1,n_estimators=100)
    clf1 = clf1.fit(X1_train,Y_train)
    X1_pre = clf1.predict(X1_test)
    X1_decision_matrix = np.transpose(clf1.predict_proba(X1_test))
    decision_matrix1.append(X1_decision_matrix.tolist())
    clf2 = RandomForestClassifier(n_jobs=-1,n_estimators=100)
    clf2 = clf2.fit(X2_train,Y_train)
    X2_pre = clf2.predict(X2_test)
    X2_decision_matrix = np.transpose(clf2.predict_proba(X2_test))
    decision_matrix1.append(X2_decision_matrix.tolist())
    clf3 = RandomForestClassifier(n_jobs=-1,n_estimators=100)
    clf3 = clf3.fit(X3_train,Y_train)
    X3_pre = clf3.predict(X3_test)
    X3_decision_matrix = np.transpose(clf3.predict_proba(X3_test))

    '''
    clf4 = RandomForestClassifier(n_jobs=-1,n_estimators=100)
    clf4 = clf4.fit(X4_train,Y_train)
    X4_pre = clf4.predict(X4_test)
    X4_decision_matrix = np.transpose(clf4.predict_proba(X4_test))
    '''
    clf5 = RandomForestClassifier(n_jobs=-1,n_estimators=100)
    clf5 = clf5.fit(X5_train,Y_train)
    X5_pre = clf5.predict(X5_test)
    X5_decision_matrix = np.transpose(clf5.predict_proba(X5_test))


    decision_matrix1.append(X3_decision_matrix.tolist())
    #decision_matrix1.append(X4_decision_matrix.tolist())
    decision_matrix1.append(X5_decision_matrix.tolist())
    decision_matrix = np.array(decision_matrix1)
    [mm,nn,zz] = np.shape(decision_matrix)
    print(mm,nn,zz)

    result = [0.5, 0.2, 0.1, 0.2], [0.4, 0.1, 0.2, 0.3], [0.3, 0.3, 0.3, 0.1], [0.1, 0.1, 0.5, 0.3], [0.1, 0.5, 0.2,0.2], \
             [0.3, 0.2, 0.2, 0.2], [0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.8], [0.6, 0.4, 0, 0], [0.4, 0.2, 0.4, 0], [0.1, 0.4, 0.3, 0.2], [0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.1, 0.2]


    print(result)
    l = range(len(result))
    print(len(result))
    list(product(l, l))
    deter = list(product(l, repeat=5))
    count = 0
    max_count = 0
    max_weight = np.zeros([4,5])
    decision = np.zeros([zz,nn])

    print(len(deter))
    for i in range(len(deter)):
        print(i)
        weight1 = []
        for j in range(5):
            weight1.append(result[deter[i][j]])
        weight = np.array(weight1).transpose().tolist()
        for i in range(zz):
            for k in range(nn):
                decision[i, k] = weight[0][k] * decision_matrix[0, k, i] + weight[1][k] * decision_matrix[1, k, i] + \
                                 weight[2][k] * decision_matrix[2, k, i] + weight[3][k] * decision_matrix[3, k, i]
        y_pred = np.argmax(decision, 1)
        count = 0
        for iii in range(zz):
            if (y_pred[iii] == Y_test[iii]):
                count += 1
        if (count > max_count):
            max_count = count
            max_weight = weight
    max_weight1.append(max_weight)
max_weight1 = np.array(max_weight1)
[m, n, z] = np.shape(max_weight1)
max_weight = np.zeros([n, z])
for i in range(m):
    for j in range(n):
        for k in range(z):
            max_weight[j, k] = max_weight[j, k] + max_weight1[i, j, k]
for i in range(n):
    for j in range(z):
        max_weight[i, j] = max_weight[i, j]
print('max_weight',max_weight)

print('accuracy', max_count/1740)


kf = KFold(n_splits = 10)
decision1 = np.zeros([870,nn])
final_result1 = []
final_result2 = []  #两个分类器分别的准确率
final_result3 = []
final_result = []
final_result4 = []
final_result5 = []
li = []
for i in range(8700):
    li.append(i)
random.shuffle(li)

for train_n , test_n in kf.split(X1):
    X1_train1 = []
    X1_test1 = []
    X2_train1 =[]
    X2_test1 = []
    X3_train1 = []
    X3_test1 = []


    X5_train1 = []
    X5_test1 = []
    Y_train1 = []
    Y_test1 = []
    for i in range(len(train_n)):
        X1_train1.append(X1[li[train_n[i]]])
        X2_train1.append(X2[li[train_n[i]]])
        X3_train1.append(X3[li[train_n[i]]])
        #X4_train1.append(X4[li[train_n[i]]])
        X5_train1.append(X5[li[train_n[i]]])
        Y_train1.append(Y1[li[train_n[i]]])
    for j in range(len(test_n)):
        X1_test1.append(X1[li[test_n[j]]])
        X2_test1.append(X2[li[test_n[j]]])
        X3_test1.append(X3[li[test_n[j]]])
        #X4_test1.append(X4[li[test_n[j]]])
        X5_test1.append(X5[li[train_n[i]]])
        Y_test1.append(Y1[li[test_n[j]]])

    decision_matrix = []
    clf11 = RandomForestClassifier(n_jobs=-1,n_estimators=1000)
    clf11.fit(X1_train1,Y_train1)
    X11_decision_matrix = np.transpose(clf11.predict_proba(X1_test1))
    X11_pred = np.argmax(clf11.predict_proba(X1_test1),1)
    clf22 = RandomForestClassifier(n_jobs=-1,n_estimators=1000)
    clf22.fit(X2_train1,Y_train1)
    X22_pred = np.argmax(clf22.predict_proba(X2_test1),1)
    #print('x22_pred',X22_pred)
    X22_decision_matrix = np.transpose(clf22.predict_proba(X2_test1))
    decision_matrix.append(X11_decision_matrix)
    decision_matrix.append(X22_decision_matrix)
    clf33 = RandomForestClassifier(n_jobs=-1, n_estimators=1000)
    clf33.fit(X3_train1, Y_train1)
    X33_decision_matrix = np.transpose(clf33.predict_proba(X3_test1))
    X33_pred = np.argmax(clf33.predict_proba(X3_test1), 1)
    '''
    clf44 = RandomForestClassifier(n_jobs=-1, n_estimators=100)
    clf44.fit(X4_train1, Y_train1)
    X44_pred = np.argmax(clf44.predict_proba(X4_test1), 1)
    #print('x44_pred', X44_pred)
    X44_decision_matrix = np.transpose(clf44.predict_proba(X4_test1))
    '''
    clf55 = RandomForestClassifier(n_jobs=-1, n_estimators=1000)
    clf55.fit(X5_train1, Y_train1)
    X55_pred = np.argmax(clf55.predict_proba(X5_test1), 1)
    # print('x44_pred', X44_pred)
    X55_decision_matrix = np.transpose(clf55.predict_proba(X5_test1))
    decision_matrix.append(X33_decision_matrix)
    #decision_matrix.append(X44_decision_matrix)
    decision_matrix.append(X55_decision_matrix)
    decision_matrix = np.array(decision_matrix)
    [mm,nn,zz] = np.shape(decision_matrix)
    #print(mm,nn,zz)
    for i in range(zz):
        for k in range(nn):
            decision1[i, k] = max_weight[0][k] * decision_matrix[0, k, i] + max_weight[1][k] * decision_matrix[1, k, i] + \
                             max_weight[2][k] * decision_matrix[2, k, i] + max_weight[3][k] * decision_matrix[3, k, i]
    y_pred1 = np.argmax(decision1,1)
    count = 0
    count1 = 0
    count2 = 0
    count3 = 0

    count5 = 0
    for iii in range(zz):
        if(y_pred1[iii] == Y_test1[iii]):
            count += 1
        if(X11_pred[iii] == Y_test1[iii]):
            count1 += 1
        if(X22_pred[iii] == Y_test1[iii]):
            count2 += 1
        if (X33_pred[iii] == Y_test1[iii]):
            count3 += 1

        if (X55_pred[iii] == Y_test1[iii]):
            count5 += 1

    #print(np.shape(y_pred1),np.shape(Y_test1))
    final_result1.append(count1/zz)
    final_result2.append(count2/zz)
    final_result3.append(count3 / zz)

    final_result5.append(count5 / zz)
    final_result.append(count/zz)
print(final_result1,final_result2,final_result3,final_result5,final_result)

print(np.mean(final_result1),np.mean(final_result2),np.mean(final_result3),\
      np.mean(final_result5), np.mean(final_result))