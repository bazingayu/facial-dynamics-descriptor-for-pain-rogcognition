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
        print(filename)
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
np.save("new_result2/gabor.npy",gabor)

label = [0,4]
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

Y1 = []
for k in range(len(person)):                          #根据人名组织文件，生成87*100*n数组

    f1 = []
    y = []
    for i in range(2):
        for j in range(n):
            s = filenames[label[i],j].split('-')
            if s[0] == person[k]:
                f1.append(X[label[i],j,:])

                y.append(i)
    X1_Data_Norm.append(f1)

    Y1.append(y)
Y1 = np.array(Y1)
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
Y = []
decision_matrix = []
for i in range(m):
    for j in range(n):
        X.append(X1_Data_Norm[i,j,:])
        Y.append(Y1[[i],[j]])
X = np.array(X)
Y  = np.array(Y)
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
print(np.shape(Z))
#np.save('new_result2/Gabor_norm_pca.npy',Z)
print(np.sum(Y))

kf = KFold(n_splits = 10)

li = []
for i in range(3480):
    li.append(i)
random.shuffle(li)
final_result = []
for train_n , test_n in kf.split(Z):
    X_train1 = []
    X_test1 = []

    Y_train1 = []
    Y_test1 = []
    for i in range(len(train_n)):
        X_train1.append(Z[li[train_n[i]]])

        Y_train1.append(Y[li[train_n[i]]])
    for j in range(len(test_n)):
        X_test1.append(Z[li[test_n[j]]])

        Y_test1.append(Y[li[test_n[j]]])


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
