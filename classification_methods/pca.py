import numpy as np
from matplotlib import pyplot as plt
from scipy import io as spio
from sklearn.decomposition import pca
from sklearn.preprocessing import StandardScaler

filenames = np.load('new_result/parta_name.npy')
[m,n] = np.shape(filenames)
gabor = []
for i in range(m):
    for j in range(n):
        filename = filenames[i,j]
        print(filename)
        path = '/media/ustb/Seagate Backup Plus Drive/BioVid_copy/gabor_3_part_npy/' + str(i) + '/' + filename + '.npy'
        feature = np.load(path)
        gabor.append(feature)
gabor = np.array(gabor)
print(np.shape(gabor))
X = np.nan_to_num(gabor)
np.save('new_result/gabor_3.npy',X)


scaler = StandardScaler()
scaler.fit(X)
print(np.shape(X))
x_train = scaler.transform(X)

K = 50
model = pca.PCA(n_components=K).fit(x_train)
Z = model.transform(x_train)
print(np.shape(Z))
np.save('new_result/gabor_pca_3',Z)