import cv2
import dlib
import numpy as np
import math
import os
import glob
import concurrent.futures
from scipy.stats import iqr
from matplotlib.mlab import prctile
from statsmodels import robust
from PIL import Image
count1 =0
def extract_landmark(path):
    #print(path)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/yjw/python/face_landmark/shape_predictor_68_face_landmarks.dat")
    dets = detector(img, 1)

    if (len(dets) == 1):
        for face in dets:
            x = face.left()
            y = face.top()
            z = face.right()
            h = face.bottom()
        img = Image.fromarray(img)
        new_img = np.array(img.crop((x, y, z, h)))
        new_img = cv2.resize(new_img,(50,50))
        new_img = np.array(new_img)
        return new_img, 1
    else:
        img = Image.fromarray(img)
        new_img = img.crop((81,81,236,236))
        new_img = np.array(new_img)
        new_img = cv2.resize(new_img,(50,50))
        return new_img, 0

def deal_with_filename(subdir):
    theta = np.linspace(0,math.pi,num=13)
    gamma = 0.5
    lamda = 2
    sigma = 0.56*lamda
    m = int(np.floor(sigma))
    tau = 2.75
    ut = 1.75
    #STGE filter equation
    j = 1
    I_1 = np.zeros([50, 50, 23])
    I = np.zeros([50, 50, 23])
    count = 0
    for filename1 in os.listdir(subdir):
        E = []
        filename = subdir + '/' +filename1
        img , deter = extract_landmark(filename)
        I[:,:,count] = img
        count += 1
        time = np.linspace(0,0.04*(22),23)
        g_new1 = np.zeros([2*m +1 , 2*m +1 , 23])
        g_new = np.zeros([2*m+1,2*m+1, 23 , 13])
        for th in range(0,13):   #theta的个数
            for t in range(0,23):    #t的个数
                g = np.zeros([2 * m + 1, 2 * m + 1])
                for x in range(-m,m+1):
                    for y in range(-m,m+1):
                        x1 = x * np.cos(theta[th]) + y*np.sin(theta[th])
                        y1 = -x * np.sin(theta[th]) + y* np.cos(theta[th])
                        g[m+x,m+y]=(gamma/(2*np.pi*sigma**2)) * np.exp(-(x1**2+(gamma**2)*y1**2)/(2*sigma**2))* np.cos(2*np.pi*x1/lamda)*(1/np.sqrt(2 * np.pi * tau)) * np.exp(-(time[t]-ut) / (2 * (tau **2)))
                g_new1[:,:,t] = g[:,:]
            g_new[:,:,:,th] = g_new1

        #E = zeros(size(theta))
        for th in range(0,13):
            c = 0
            for ii in range(23):
                I_1[:,:,ii] = cv2.filter2D(I[:,:,ii],-1,g_new[:,:,ii,th])
                I_1[:,:,ii] = [[I_1[jj][kk][ii]**2 for kk in range(len(I_1[jj,:,ii]))]for jj in range(len(I_1[ii]))]
                c = c + I_1[:,:,ii]
            c1 = np.array(c).flatten().tolist()
            E.append(c1)
        E1 = np.array(E).flatten()
    s = subdir.split('/')
    target_filename = '/media/ustb/Dataset/biovid/PartA/yudata/frame_front1_gabor/' + s[7] + '/' + s[8]
    print(target_filename)
    np.save(target_filename,E1)

with concurrent.futures.ProcessPoolExecutor() as executor:
    subfiles = glob.glob("/media/ustb/Dataset/biovid/PartA/frame_front1/4/*")
    count = 0
    for image_file in zip(subfiles,executor.map(deal_with_filename,subfiles)):
        print(count)
        count += 1
