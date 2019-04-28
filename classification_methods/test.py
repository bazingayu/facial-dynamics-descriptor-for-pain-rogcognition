from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import random
from itertools import product

dis_Data = np.load('new_result2/dis.npy')   #读取文件
Euler_Data = np.load('new_result2/Euler.npy')
gra_Data = np.load('new_result2/gra.npy')
filenames = np.load('new_result/parta_name.npy')
bio_Data = np.load('new_result/parta_biosignal.npy')
label = [0,4]

[m,n] = np.shape(filenames)
person = []
for i in range(2):                                       #取出人的编号生成一个87的list
    for j in range(n):
        s = filenames[label[i],j].split('-')
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
X1_Data_Norm = []
X2_Data_Norm = []
X3_Data_Norm = []
X4_Data_Norm = []
Y = []
for k in range(len(person)):                          #根据人名组织文件，生成87*100*n数组
    f2 = []
    f3 = []
    f4 = []
    f1 = []
    y = []
    for i in range(2):
        for j in range(n):
            s = filenames[label[i],j].split('-')
            if s[0] == person[k]:
                f1.append(dis_Data[label[i],j,:])
                f2.append(gra_Data[label[i],j,:])
                f3.append(Euler_Data[label[i], j, :])
                f4.append(bio_Data[label[i], j, :])
                y.append(i)
    X1_Data_Norm.append(f1)
    X2_Data_Norm.append(f2)
    X3_Data_Norm.append(f3)
    X4_Data_Norm.append(f4)
    Y.append(y)
Y = np.array(Y)
X1_Data_Norm = np.array(X1_Data_Norm)
X2_Data_Norm = np.array(X2_Data_Norm)
X3_Data_Norm = np.array(X3_Data_Norm)
X4_Data_Norm = np.array(X4_Data_Norm)



[m,n,z1] = np.shape(X1_Data_Norm)
[m,n,z2] = np.shape(X2_Data_Norm)
[m,n,z3] = np.shape(X3_Data_Norm)
[m,n,z4] = np.shape(X4_Data_Norm)
sum1 = np.zeros([m,z1])
sum2 = np.zeros([m,z2])
sum3 = np.zeros([m,z3])
sum4 = np.zeros([m,z4])

for i in range(m):
    for k in range(z1):
        for j in range(n):
            sum1[i,k] = sum1[i,k] + X1_Data_Norm[i,j,k]
for i in range(m):
    for k in range(z2):
        for j in range(n):
            sum2[i,k] = sum2[i,k] + X2_Data_Norm[i,j,k]
for i in range(m):
    for k in range(z3):
        for j in range(n):
            sum3[i,k] = sum3[i,k] + X3_Data_Norm[i,j,k]
for i in range(m):
    for k in range(z4):
        for j in range(n):
            sum4[i,k] = sum4[i,k] + X4_Data_Norm[i,j,k]
ave1 = sum1/40
ave2 = sum2/40
ave3 = sum3/40
ave4 = sum4/40
sum_val1 = np.zeros([m,z1])
sum_val2 = np.zeros([m,z2])
sum_val3 = np.zeros([m,z3])
sum_val4 = np.zeros([m,z4])
for i in range(m):
    for k in range(z1):
        for j in range(n):
            sum_val1[i,k] = sum_val1[i,k] + (X1_Data_Norm[i,j,k] - ave1[i,k]) ** 2
for i in range(m):
    for k in range(z2):
        for j in range(n):
            sum_val2[i,k] = sum_val2[i,k] + (X2_Data_Norm[i,j,k] - ave2[i,k]) ** 2
for i in range(m):
    for k in range(z3):
        for j in range(n):
            sum_val3[i,k] = sum_val3[i,k] + (X3_Data_Norm[i,j,k] - ave3[i,k]) ** 2
for i in range(m):
    for k in range(z4):
        for j in range(n):
            sum_val4[i,k] = sum_val4[i,k] + (X4_Data_Norm[i,j,k] - ave4[i,k]) ** 2

sigma1 = (sum_val1/40)**0.5
sigma2 = (sum_val2/40)**0.5
sigma3 = (sum_val3/40)**0.5
sigma4 = (sum_val4/40)**0.5
for i in range(m):
    for j in range(n):
        for k in range(z1):
            if sigma1[i,k] == 0:
                sigma1[i,k] = 0.0000001
            X1_Data_Norm[i,j,k] = (X1_Data_Norm[i,j,k] - ave1[i,k]) / sigma1[i,k]
for i in range(m):
    for j in range(n):
        for k in range(z2):
            if sigma2[i,k] == 0:
                sigma2[i,k] = 0.0000001
            X2_Data_Norm[i,j,k] = (X2_Data_Norm[i,j,k] - ave2[i,k]) / sigma2[i,k]
for i in range(m):
    for j in range(n):
        for k in range(z3):
            if sigma3[i,k] == 0:
                sigma3[i,k] = 0.0000001
            X3_Data_Norm[i,j,k] = (X3_Data_Norm[i,j,k] - ave3[i,k]) / sigma3[i,k]
for i in range(m):
    for j in range(n):
        for k in range(z4):
            if sigma4[i,k] == 0:
                sigma4[i,k] = 0.0000001
            X4_Data_Norm[i,j,k] = (X4_Data_Norm[i,j,k] - ave4[i,k]) / sigma4[i,k]



X1 = []
Y1 = []
decision_matrix = []
for i in range(m):
    for j in range(n):
        X1.append(X1_Data_Norm[i,j,:])
        Y1.append(Y[i,j])
X1 = np.array(X1)
Y1  = np.array(Y1)
X1 = np.nan_to_num(X1)

X2 = []
for i in range(m):
    for j in range(n):
        X2.append(X2_Data_Norm[i,j,:])
X2 = np.array(X2)
X2 = np.nan_to_num(X2)

X3 = []
for i in range(m):
    for j in range(n):
        X3.append(X3_Data_Norm[i,j,:])
X3 = np.array(X3)
X3 = np.nan_to_num(X3)

X4 = []
for i in range(m):
    for j in range(n):
        X4.append(X4_Data_Norm[i,j,:])
X4 = np.array(X4)
X4 = np.nan_to_num(X4)
'''
X5 = np.load("new_result2/Gabor_norm_pca.npy")
[m,z5] = np.shape(X5)
print("[m,z5]",[m,z5])
X5_Data_Norm = []
for i in range(2):
    for j in range(1740):
        print(label[i]*1740+j)
        X5_Data_Norm.append(X5[label[i]*1740+j])
X5 = np.array(X5_Data_Norm)
print(np.shape(X5))'''
import numpy as np
from matplotlib import pyplot as plt
from scipy import io as spio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import random
from sklearn.decomposition import pca
from sklearn.preprocessing import StandardScaler
filenames = np.load("new_result/parta_name.npy")
[m,n] = np.shape(filenames)
gabor = []
for i in range(m):
    gabor1 = []
    for j in range(n):
        filename = filenames[i,j]
        s = filename.split('-')
        if s[1] == 'BL1':
            class_name = '0'
        elif s[1] == 'PA1':
            class_name = '1'
        elif s[1] == 'PA2':
            class_name = '2'
        elif s[1] == 'PA3':
            class_name = '3'
        elif s[1] == 'PA4':
            class_name = '4'
        gabor_name = '/media/ustb/Dataset/biovid/PartA/yudata/frame_front1_gabor/' + class_name + '/' + filename + '.npy'
        #if len(s[2]) == 1:
            #gabor_name = '/media/ustb/Dataset/biovid/PartA/yudata/frame_front1_gabor/' + class_name + '/' + filename + '44.npy'
        #if len(s[2]) == 2:
            #gabor_name = '/media/ustb/Dataset/biovid/PartA/yudata/frame_front1_gabor/' + class_name + '/' + filename + '4.npy'
        single = np.load(gabor_name)
        gabor1.append(single)
    gabor.append(gabor1)
X = np.array(gabor)


[m,n] = np.shape(filenames)
person = []
for i in range(2):                                       #取出人的编号生成一个87的list3
    for j in range(n):
        s = filenames[label[i],j].split('-')
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
X1_Data_Norm = []


for k in range(len(person)):                          #根据人名组织文件，生成87*100*n数组

    f1 = []
    for i in range(2):
        for j in range(n):
            s = filenames[label[i],j].split('-')
            if s[0] == person[k]:
                f1.append(X[label[i],j,:])

    X1_Data_Norm.append(f1)


X1_Data_Norm = np.array(X1_Data_Norm)




[m,n,z1] = np.shape(X1_Data_Norm)
print([m,n,z1])

sum1 = np.zeros([m])



for i in range(m):
    for k in range(z1):
        for j in range(n):
            sum1[i] = sum1[i] + X1_Data_Norm[i,j,k]

print("done1")


ave1 = sum1/(n*z1)

sum_val1 = np.zeros([m])


for i in range(m):
    for k in range(z1):
        for j in range(n):
            sum_val1[i] = sum_val1[i] + (X1_Data_Norm[i,j,k] - ave1[i]) ** 2

sigma1 = (sum_val1/(n*z1))**0.5

print("done2")


for i in range(m):
    for j in range(n):
        for k in range(z1):
            if sigma1[i] == 0:
                sigma1[i] = 0.0000001
            X1_Data_Norm[i,j,k] = (X1_Data_Norm[i,j,k] - ave1[i]) / sigma1[i]

print("done3")

X = []

decision_matrix = []
for i in range(m):
    for j in range(n):
        X.append(X1_Data_Norm[i,j,:])

X = np.array(X)

X = np.nan_to_num(X)

#np.save("new_result2/Gabor_norm.npy",X)


print('PCA')
scaler = StandardScaler()
scaler.fit(X)
print(np.shape(X))
x_train = scaler.transform(X)

K = 250
model = pca.PCA(n_components=K).fit(x_train)
Z = model.transform(x_train)
print("x5:",np.shape(Z))
X5 = Z


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


max_weight1 = []
for ii in range(5):
    decision_matrix = []
    X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, X4_train, X4_test, X5_train, \
    X5_test, Y_train, Y_test = train_test_split(X1,X2,X3,X4,X5,Y1,test_size=0.2)
    clf1 = RandomForestClassifier(n_jobs=-1,n_estimators=500)
    clf1 = clf1.fit(X1_train,Y_train)
    X1_pre = clf1.predict(X1_test)
    X1_decision_matrix = np.transpose(clf1.predict_proba(X1_test))
    decision_matrix.append(X1_decision_matrix.tolist())
    clf2 = RandomForestClassifier(n_jobs=-1,n_estimators=500)
    clf2 = clf2.fit(X2_train,Y_train)
    X2_pre = clf2.predict(X2_test)
    X2_decision_matrix = np.transpose(clf2.predict_proba(X2_test))
    decision_matrix.append(X2_decision_matrix.tolist())
    clf3 = RandomForestClassifier(n_jobs=-1,n_estimators=500)
    clf3 = clf3.fit(X3_train,Y_train)
    X3_pre = clf3.predict(X3_test)
    X3_decision_matrix = np.transpose(clf3.predict_proba(X3_test))
    clf4 = RandomForestClassifier(n_jobs=-1,n_estimators=500)
    clf4 = clf4.fit(X4_train,Y_train)
    X4_pre = clf4.predict(X4_test)
    X4_decision_matrix = np.transpose(clf4.predict_proba(X4_test))

    clf5 = RandomForestClassifier(n_jobs=-1,n_estimators=500)
    clf5 = clf5.fit(X5_train,Y_train)
    X5_pre = clf5.predict(X5_test)
    X5_decision_matrix = np.transpose(clf5.predict_proba(X5_test))


    decision_matrix.append(X3_decision_matrix.tolist())
    decision_matrix.append(X4_decision_matrix.tolist())
    decision_matrix.append(X5_decision_matrix.tolist())
    decision_matrix = np.array(decision_matrix)
    [mm,nn,zz] = np.shape(decision_matrix)
    print(mm,nn,zz)


    result = [0.1,0.1,0.1,0.6,0.1],[0.1,0.1,0.1,0.6,0.1],[0.1,0.1,0.2,0.5,0.1],[0.1,0.1,0.5,0.2,0.1],[0.5,0.1,0.1,0.2,0.1],\
        [0.1,0.5,0.1,0.2,0.1],[0.1,0.1,0.1,0.2,0.5],[0.1,0,0.1,0.8,0],[0,0,0.2,0.8,0]


    print(result)
    l = range(len(result))
    print(len(result))
    list(product(l, l))
    deter = list(product(l, repeat=2))
    count = 0
    max_count = 0
    max_weight = np.zeros([5,2])
    decision = np.zeros([zz,nn])

    print(len(deter))


    for i in range(len(deter)):
        print(i)
        weight1 = []
        for j in range(2):
            weight1.append(result[deter[i][j]])
        weight = np.array(weight1).transpose().tolist()
        for i in range(zz):
            for k in range(nn):
                decision[i, k] = weight[0][k] * decision_matrix[0, k, i] + weight[1][k] * decision_matrix[1, k, i] + \
                                 weight[2][k] * decision_matrix[2, k, i] + weight[3][k] * decision_matrix[3, k, i] + \
                                 weight[4][k] * decision_matrix[4, k, i]
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
[m,n,z] = np.shape(max_weight1)
max_weight = np.zeros([n,z])
for i in range(m):
    for j in range(n):
        for k in range(z):
            max_weight[j,k] = max_weight[j,k] + max_weight1[i,j,k]
print(max_weight)
for i in range(n):
    for j in range(z):
        max_weight[i,j] = max_weight[i,j]/5

print('max_weight',max_weight)

print('accuracy', max_count/696)


kf = KFold(n_splits = 10)
decision1 = np.zeros([348,nn])
final_result1 = []
final_result2 = []  #两个分类器分别的准确率
final_result3 = []
final_result = []
final_result4 = []
final_result5 = []
li = []
for i in range(3480):
    li.append(i)
random.shuffle(li)
for train_n , test_n in kf.split(X1):
    X1_train1 = []
    X1_test1 = []
    X2_train1 =[]
    X2_test1 = []
    X3_train1 = []
    X3_test1 = []
    X4_train1 = []
    X4_test1 = []
    X5_train1 = []
    X5_test1 = []
    X_train1 = []
    X_test1 = []
    Y_train1 = []
    Y_test1 = []
    for i in range(len(train_n)):
        X1_train1.append(X1[li[train_n[i]]])
        X2_train1.append(X2[li[train_n[i]]])
        X3_train1.append(X3[li[train_n[i]]])
        X4_train1.append(X4[li[train_n[i]]])
        X5_train1.append(X5[li[train_n[i]]])
        X_train1.append(X[li[train_n[i]]])
        Y_train1.append(Y1[li[train_n[i]]])
    for j in range(len(test_n)):
        X1_test1.append(X1[li[test_n[j]]])
        X2_test1.append(X2[li[test_n[j]]])
        X3_test1.append(X3[li[test_n[j]]])
        X4_test1.append(X4[li[test_n[j]]])
        X5_test1.append(X5[li[test_n[j]]])
        Y_test1.append(Y1[li[test_n[j]]])
        X_test1.append(X[li[test_n[j]]])

    decision_matrix = []
    clf11 = RandomForestClassifier(n_jobs=-1,n_estimators=500)
    clf11.fit(X1_train1,Y_train1)
    X11_decision_matrix = np.transpose(clf11.predict_proba(X1_test1))
    X11_pred = np.argmax(clf11.predict_proba(X1_test1),1)

    y1_score =clf11.predict_proba(X1_test1)
    y_test2 = np.array(Y_test1)
    print(y_test2,y1_score[:,0])
    fpr1,tpr1,thresholds = metrics.roc_curve(y_test2,y1_score[:,0],pos_label=0)
    print(fpr1,tpr1)
    roc_auc1 = auc(fpr1, tpr1)

    plt.plot( fpr1,tpr1,linewidth = '1',label='distance (auc = %0.2f)' % (roc_auc1),color = '#ADD8E6',linestyle='-')

    clf22 = RandomForestClassifier(n_jobs=-1,n_estimators=500)
    clf22.fit(X2_train1,Y_train1)
    X22_pred = np.argmax(clf22.predict_proba(X2_test1),1)
    #print('x22_pred',X22_pred)
    X22_decision_matrix = np.transpose(clf22.predict_proba(X2_test1))

    y2_score = clf22.predict_proba(X2_test1)

    fpr2, tpr2, thresholds = metrics.roc_curve(y_test2, y2_score[:, 0], pos_label=0)
    roc_auc2 = auc(fpr2, tpr2)
    plt.plot(fpr2, tpr2, linewidth='1', label='gradient (auc = %0.2f)' % (roc_auc2), color='#6495ED', linestyle='-')


    decision_matrix.append(X11_decision_matrix)
    decision_matrix.append(X22_decision_matrix)
    clf33 = RandomForestClassifier(n_jobs=-1, n_estimators=500)
    clf33.fit(X3_train1, Y_train1)
    X33_decision_matrix = np.transpose(clf33.predict_proba(X3_test1))
    y3_score = clf33.predict_proba(X3_test1)

    fpr3, tpr3, thresholds = metrics.roc_curve(y_test2, y3_score[:, 0], pos_label=0)
    roc_auc3 = auc(fpr3, tpr3)
    plt.plot(fpr3, tpr3, linewidth='1', label='head pose(auc = %0.2f)' % (roc_auc3), color='#00008B', linestyle='-')
    X33_pred = np.argmax(clf33.predict_proba(X3_test1), 1)
    clf44 = RandomForestClassifier(n_jobs=-1, n_estimators=500)
    clf44.fit(X4_train1, Y_train1)
    X44_pred = np.argmax(clf44.predict_proba(X4_test1), 1)
    #print('x44_pred', X44_pred)
    y4_score = clf44.predict_proba(X4_test1)

    fpr4, tpr4, thresholds = metrics.roc_curve(y_test2, y4_score[:, 0], pos_label=0)
    roc_auc4 = auc(fpr4, tpr4)
    plt.plot(fpr4, tpr4, linewidth='1', label='biomedical signals(auc = %0.2f)' % (roc_auc4), color='#0000CD', linestyle='-')

    X44_decision_matrix = np.transpose(clf44.predict_proba(X4_test1))
    clf55 = RandomForestClassifier(n_jobs=-1, n_estimators=500)
    clf55.fit(X5_train1, Y_train1)
    X55_pred = np.argmax(clf55.predict_proba(X5_test1), 1)
    y5_score = clf55.predict_proba(X5_test1)

    fpr5, tpr5, thresholds = metrics.roc_curve(y_test2, y5_score[:, 0], pos_label=0)

    roc_auc5 = auc(fpr5, tpr5)
    plt.plot(fpr5, tpr5, linewidth='1', label='sequence Gabor(auc = %0.2f)' % (roc_auc5), color='#8A2BE2', linestyle='-')
    X55_decision_matrix = np.transpose(clf55.predict_proba(X5_test1))
    decision_matrix.append(X33_decision_matrix)
    decision_matrix.append(X44_decision_matrix)
    decision_matrix.append(X55_decision_matrix)
    decision_matrix = np.array(decision_matrix)
    [mm,nn,zz] = np.shape(decision_matrix)
    #print(mm,  nn,zz)
    for i in range(zz):
        for k in range(nn):
            decision1[i, k] = max_weight[0][k] * decision_matrix[0, k, i] + max_weight[1][k] * decision_matrix[1, k, i] + \
                             max_weight[2][k] * decision_matrix[2, k, i] + max_weight[3][k] * decision_matrix[3, k, i] + \
                              max_weight[4][k] * decision_matrix[4, k, i]

    fpr6, tpr6, thresholds = metrics.roc_curve(y_test2, decision1[:, 0], pos_label=0)
    roc_auc6 = auc(fpr6, tpr6)

    plt.plot(fpr6, tpr6, linewidth='1', label='decision-level fusion(auc = %0.2f)' % ( roc_auc6), color='red', linestyle='-')

    clf77 = RandomForestClassifier(n_jobs=-1, n_estimators=500)
    clf77.fit(X_train1, Y_train1)
    X77_pred = np.argmax(clf77.predict_proba(X_test1), 1)
    y7_score = clf77.predict_proba(X_test1)

    fpr7, tpr7, thresholds = metrics.roc_curve(y_test2, y7_score[:, 0], pos_label=0)
    roc_auc7 = auc(fpr7, tpr7)
    plt.plot(fpr7, tpr7, linewidth='1',label='feature-level fusion (auc = %0.2f)' % (roc_auc7), color='green', linestyle='-')

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.title("The binary classification's(BL vs PA4) ROC")
    plt.xlabel("FPR(False positive rate)")
    plt.ylabel("TPR(True positive rate)")
    plt.legend()
    plt.show()
    y_pred1 = np.argmax(decision1,1)
    count = 0
    count1 = 0
    count2 = 0
    count3 = 0
    count4 = 0
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
        if (X44_pred[iii] == Y_test1[iii]):
            count4 += 1
        if (X55_pred[iii] == Y_test1[iii]):
            count5 += 1

    #print(np.shape(y_pred1),np.shape(Y_test1))
    final_result1.append(count1 / zz)
    final_result2.append(count2 / zz)
    final_result3.append(count3 / zz)
    final_result4.append(count4 / zz)
    final_result5.append(count5 / zz)
    final_result.append(count / zz)
print(final_result1,final_result2,final_result3,final_result4,final_result5,final_result)
scores = classification_report(Y_test1, y_pred1, digits=3)
print(scores)
print(np.mean(final_result1),np.mean(final_result2),np.mean(final_result3),np.mean(final_result4),\
      np.mean(final_result5), np.mean(final_result))