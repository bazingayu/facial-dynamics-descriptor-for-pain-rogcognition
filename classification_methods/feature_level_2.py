from sklearn import tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
import random
from itertools import product

dis_Data = np.load('new_result2/dis.npy')   #读取文件g
Euler_Data = np.load('new_result2/Euler.npy')
gra_Data = np.load('new_result2/gra.npy')
filenames = np.load('new_result/parta_name.npy')
bio_Data = np.load('new_result/parta_biosignal.npy')
label = [1,0]

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

Y1  = np.array(Y1)

X2 = []
for i in range(m):
    for j in range(n):
        X2.append(X2_Data_Norm[i,j,:])


X3 = []
for i in range(m):
    for j in range(n):
        X3.append(X3_Data_Norm[i,j,:])


X4 = []
for i in range(m):
    for j in range(n):
        X4.append(X4_Data_Norm[i,j,:])

'''
X5_ = np.load("new_result2/Gabor_norm_pca.npy")
X5 = []
for i in range(2):
    for j in range(1740):
        X5.append(X5_[label[i]*1740+j])'''


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


kf = KFold(n_splits = 10)

li = []
for i in range(3480):
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
