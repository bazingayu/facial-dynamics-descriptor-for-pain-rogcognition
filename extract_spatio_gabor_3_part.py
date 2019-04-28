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

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/yjw/python/face_landmark/shape_predictor_68_face_landmarks.dat")
    dets = detector(img, 1)

    if (len(dets) == 1):

        return img, 1 , np.matrix([[p.x, p.y] for p in predictor(img, dets[0]).parts()])
    else:
        return img, 0 ,[[0,0]]
def extract_texture_part(img,landmark):
    gra = []
    pst1 = np.float32([[landmark[21,0], landmark[21,1]], [landmark[22,0], landmark[22,1]], [landmark[39,0], landmark[39,1]], [landmark[42,0], landmark[42,1]]])
    pst2 = np.float32([[0, 0], [0,49], [49,0], [49, 49]])
    M = cv2.getPerspectiveTransform(pst1, pst2)
    dst1 = cv2.warpPerspective(img, M, (50, 50))
    dst1_gray = cv2.cvtColor(dst1,cv2.COLOR_BGR2GRAY)

    pst1 = np.float32([[landmark[2, 0], landmark[2, 1]], [landmark[31, 0], landmark[31, 1]], [landmark[5, 0], landmark[5, 1]],[landmark[7, 0], landmark[7, 1]]])
    pst2 = np.float32([[0, 0], [0, 49], [49, 0], [49, 49]])
    M = cv2.getPerspectiveTransform(pst1, pst2)
    dst2 = cv2.warpPerspective(img, M, (50, 50))
    dst2_gray = cv2.cvtColor(dst2, cv2.COLOR_BGR2GRAY)

    pst1 = np.float32([[landmark[35, 0], landmark[35, 1]], [landmark[14, 0], landmark[14, 1]], [landmark[9, 0], landmark[9, 1]],[landmark[12, 0], landmark[12, 1]]])
    pst2 = np.float32([[0, 0], [0, 49], [49, 0], [49, 49]])
    M = cv2.getPerspectiveTransform(pst1, pst2)
    dst3 = cv2.warpPerspective(img, M, (50, 50))
    dst3_gray = cv2.cvtColor(dst3, cv2.COLOR_BGR2GRAY)

    return dst1_gray,dst2_gray,dst3_gray
def deal_with_filename(subdir):
    s = subdir.split('/')
    target_filename = '/media/ustb/Seagate Backup Plus Drive/BioVid_copy/gabor_3_part_npy/' + s[7] + '/' + s[8]+'.npy'
    print(target_filename)
    if not os.path.exists(target_filename):
        theta = np.linspace(0,math.pi,num=13)
        gamma = 0.5
        lamda = 2
        sigma = 0.56*lamda
        m = int(np.floor(sigma))
        tau = 2.75
        ut = 1.75
        #STGE filter equation
        j = 1
        E_final = []
        I_1 = np.zeros([ 50, 50, 23])
        I = np.zeros([3, 50, 50, 23])
        count = 0
        for filename1 in os.listdir(subdir):
            E = []
            filename = subdir + '/' +filename1
            img , deter ,landmark= extract_landmark(filename)
            if deter == 1:
                im1, im2, im3 = extract_texture_part(img,landmark)
            else:
                landmark1 = [[91,135],[92,152],[96,167],[ 99,183],[103,200],[112,214],[126,225],[142,233],[161,234],[181,231],[197,224],[211,214],[219,200],[222,184],[224,167],[226,151],[226,134],[101,122],[111,114],[123,112],[137,113],[150,118],[166,117],[179,112],[193,110],[207,112],[217,119],[158,129],[159,142],[159,154],[159,167],[143,174],[151,176],[159,178],[168,176],[176,173],[115,136],[123,135],[131,134],[142,134],[133,136],[124,137],[177,134],[187,132],[195,133],[204,134],[195,136],[187,135],[133,195],[143,191],[152,188],[159,190],[167,188],[177,190],[188,193],[178,200],[168,203],[160,204],[153,204],[143,201],[137,195],[152,195],[160,195],[167,194],[184,193],[168,194],[160,195],[153,194]]
                landmark = np.array(landmark1)
                im1, im2, im3 = extract_texture_part(img, landmark)
            I[0,:,:,count] = im1
            I[1,:,:,count] = im2
            I[2,:,:,count] = im3
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
        for im_num in range(3):
            E = []
            for th in range(0,13):
                c = 0
                for ii in range(23):
                    I_1[:,:,ii] = cv2.filter2D(I[im_num,:,:,ii],-1,g_new[:,:,ii,th])
                    I_1[:,:,ii] = [[I_1[jj][kk][ii]**2 for kk in range(len(I_1[jj,:,ii]))]for jj in range(len(I_1[ii]))]
                    c = c + I_1[:,:,ii]
                c1 = np.array(c).flatten().tolist()
                E.append(c1)
            E_final.append(E)
        E1 = np.array(E_final).flatten()
        print(np.shape(E1))
        np.save(target_filename,E1)
for i in range(1,5,1):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        filename = '/media/ustb/Dataset/biovid/PartA/frame_front1/' + str(i) + '/*'
        subfiles = glob.glob(filename)
        count = 0
        for image_file in zip(subfiles,executor.map(deal_with_filename,subfiles)):
            print(count)
            count += 1
