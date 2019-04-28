import numpy as np
import os
import pandas as pd
from scipy.stats import iqr
from matplotlib.mlab import prctile
from statsmodels import robust
def zerocoss(x):
    s = np.sign(x)
    new = []
    new.append(s[0])
    for i in range(len(s) - 1):
        new.append(s[i] + s[i + 1])
    count = new[i == 0]
    return count
def extract_signal(dis):
    dis = dis.transpose()
    [m,n] = np.shape(dis)
    signal = np.zeros([n])
    signal1 = signal2 = signal
    seq_descr = np.zeros([m,48])
    for i in range(m):
        signal_all = np.zeros([3,n])
        signal = dis[i,:]
        signal1 = np.array(np.diff(signal))
        signal2 = np.diff(signal1)
        signal_all[0,:] = signal
        signal_all[1,:-1] = signal1
        signal_all[2,:-2] = signal2
        #   现在提出来了信号，每个信号三个  存在了signal_all中
        descr = np.zeros([16, 3])
        for k,sig in enumerate(signal_all):
            #  针对每一个信号，提取十六个特征
            s_mean = np.mean(sig)
            s_min = np.min(sig)
            s_max = np.max(sig)
            s_thresh = 0.5 * (s_min + s_mean)
            descr[0,k] = s_mean
            descr[1,k] = np.median(sig)
            descr[2,k] = s_min
            descr[3,k] = s_max
            descr[4,k] = s_max - s_min
            descr[5,k] = np.std(sig)
            descr[6,k] = iqr(sig)
            descr[7,k] = np.diff(np.percentile(sig,(10.0,90.0)))
            descr[8,k] = robust.mad(sig)
            descr[9,k] = np.argmax(sig)
            #duration
            seg_mean = sig > s_mean
            descr[10,k] = np.mean(seg_mean)
            seg_thresh = sig > s_thresh
            descr[11,k] = np.mean(seg_thresh)
            #count
            zc_mean = zerocoss(seg_mean - 0.5)
            zc_thresh = zerocoss(seg_thresh - 0.5)
            descr[12,k] = np.ceil(zc_mean/2)
            descr[13,k] = np.ceil(zc_thresh/2)
            descr[12,k] = np.sum(seg_mean)
            descr[13,k] = np.sum(seg_thresh)
            descr[14,k] = 0.001 * np.sum(sig - s_min)
            descr[15,k] = 0.01 * np.sum(sig - s_min) / (s_max - s_min)
        #print(descr)
        descr = descr.reshape([1,48])
        seq_descr[i,:] = descr
    seq_descr = seq_descr.reshape([1,48*m])
    return seq_descr

filename1 = np.load('new_result/parta_name.npy')
[m,n] = np.shape(filename1)
bio_signal_all = []
for i in range(m):
    bio_signal_single = []
    for j in range(n):
        print(filename1[i,j])
        filename = filename1[i,j]
        s = filename.split('-')
        filename_1 = s[0]
        bio_path = '/media/ustb/Dataset/biovid/PartA/biosignals_filtered/' + filename_1 + '/' + s[0] + '-' + s[
            1] + '-' + s[2]  + '_bio.csv'
        if len(s[2]) == 2:
            bio_path = '/media/ustb/Dataset/biovid/PartA/biosignals_filtered/' + filename_1 + '/' + s[0] + '-' + s[
                1] + '-' + s[2] + '4' + '_bio.csv'
        if len(s[2]) == 1:
            bio_path = '/media/ustb/Dataset/biovid/PartA/biosignals_filtered/' + filename_1 + '/' + s[0] + '-' + s[
                1] + '-' + s[2] + '44' + '_bio.csv'
        bio_path1 = '/media/ustb/Dataset/biovid/PartA/temperature/temperature/' + filename_1 + '/' + s[0] + '-' + s[
            1] + '-' + s[2] + '_temp.csv'
        if len(s[2]) == 2:
            bio_path1 = '/media/ustb/Dataset/biovid/PartA/temperature/temperature/' + filename_1 + '/' + s[0] + '-' + s[
                1] + '-' + s[2] + '4' + '_temp.csv'
        if len(s[2]) == 1:
            bio_path1 = '/media/ustb/Dataset/biovid/PartA/temperature/temperature/' + filename_1 + '/' + s[0] + '-' + s[
                1] + '-' + s[2] + '44' + '_temp.csv'
        print(bio_path)
        csv_data = np.array(pd.read_csv(bio_path,sep='\t', usecols=[1,2,3]).values)


        bio_signal1 = extract_signal(csv_data).tolist()

        #bio_signal = csv_data.reshape([8448]).tolist()
        bio_signal = bio_signal1[0]
        print(len(bio_signal1[0]),len(bio_signal))
        #bio_signal_single.append(bio_signal[0])
        bio_signal_single.append(bio_signal)
    bio_signal_all.append(bio_signal_single)
print("start converting to array")
bio_signal_all1 = np.array(bio_signal_all)
print(np.shape(bio_signal_all1))
np.save('new_result2/parta_biosignal.npy',bio_signal_all1)
