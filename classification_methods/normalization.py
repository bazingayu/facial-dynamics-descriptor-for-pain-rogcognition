#normalization
import numpy as np

descr = np.load('new_result/parta_biosignal.npy')
print(descr.shape)
[label,m,n] = np.shape(descr)
print (label,m,n)
max = [-99999] * n
min = [99999] * n

for label1 in range(label):
    for i in range(m):
        for j in range(n):
            if(descr[label1,i,j] > max[j]):
                max[j] = descr[label1,i,j]
            if(descr[label1,i,j] < min[j]):
                min[j] = descr[label1,i,j]

for label1 in range(label):
    for i in range(m):
        for j in range(n):
            descr[label1,i,j] = (descr[label1,i,j] - min[j])/(max[j] - min[j])

np.save('new_result/bio_descr_norm.npy',descr)