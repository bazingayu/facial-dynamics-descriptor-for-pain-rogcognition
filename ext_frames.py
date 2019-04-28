import numpy as np
import cv2
import face_landmark_detection
import os
import demo
import glob
import concurrent.futures
import dlib
from PIL import Image

root_dir = '/media/ustb/Dataset/biovid/PartA/video/'
target_root = '/media/ustb/Seagate Backup Plus Drive/BioVid_copy/yudata/PartA/frame138/'
video_count = 1
def extract_landmark(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/yjw/python/face_landmark/shape_predictor_68_face_landmarks.dat")
    if img is not None:
        img = img
    else:
        img = cv2.imread('/home/yjw/3.png')
    dets = detector(img)


    if len(dets) == 1:
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
def deal_with_filenanme(subdir):
    for filename1 in os.listdir(subdir):
        filename = subdir + '/' + filename1
        s = filename1.split('-')
        if (s[1] == 'BL1'):
            target_sub = target_root + '0/'
        elif (s[1] == 'PA1'):
            target_sub = target_root + '1/'
        elif (s[1] == 'PA2'):
            target_sub = target_root + '2/'
        elif (s[1] == 'PA3'):
            target_sub = target_root + '3/'
        elif (s[1] == 'PA4'):
            target_sub = target_root + '4/'
        target_sub = target_sub + s[0] + '-' + s[1] + '-' + s[2][:3]
        if not os.path.exists(target_sub):
            os.makedirs(target_sub)
        target_files = target_sub + '/*'
        images_count = len(glob.glob(target_files))
        print("images_count",images_count)
        if images_count == 138:
            print("full_full")
        else:
            cap = cv2.VideoCapture(filename)
            count = 1
            while (cap.isOpened() == True):
                target_filename = target_sub + '/' + str(count) + '.png'
                ret, frame = cap.read()

                img, deter = extract_landmark(frame)

                if ret == True:
                    cv2.imwrite(target_filename, img)
                else:
                    break
                print(filename,count)
                count += 1

with concurrent.futures.ProcessPoolExecutor() as executor:
    subfiles = glob.glob("/media/ustb/Dataset/biovid/PartA/video/*")
    for image_file in zip(subfiles,executor.map(deal_with_filenanme,subfiles)):
        print(subfiles)
