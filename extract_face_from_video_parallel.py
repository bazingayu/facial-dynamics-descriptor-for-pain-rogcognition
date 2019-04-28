import numpy as np
import cv2
import face_landmark_detection
import os
import demo
import glob
import concurrent.futures

root_dir = '/media/ustb/Dataset/biovid/PartA/video/'
target_root = '/media/ustb/Dataset/biovid/PartA/front_img2/'
video_count = 1
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
            base_path = '/home/yjw/2.png'
            count = 1
            while (cap.isOpened() == True):
                target_filename = target_sub + '/' + str(count) + '.png'
                ret, frame = cap.read()
                if ret == True:
                    im1, im2, M, landmark1, landmark2 = face_landmark_detection.face_align(base_path, frame, 0)
                    if M == [1, 1]:
                        warped_img2 = im2
                    else:
                        landmark2 = np.array(landmark2)
                        landmark1 = np.array(landmark1)
                        b = np.array(
                                [[landmark2[0], landmark2[1], landmark2[2], landmark2[3], landmark2[4], landmark2[5],
                                  landmark2[6], landmark2[7], landmark2[8], landmark2[9], landmark2[10], landmark2[11],
                                  landmark2[12], landmark2[13], landmark2[14], landmark2[15], landmark2[16],
                                  landmark2[26],
                                  landmark2[25], landmark2[24], landmark2[19], landmark2[18], landmark2[17]]],
                                dtype=np.int32)
                        im = np.zeros(im2.shape[:2], dtype="uint8")
                        cv2.polylines(im, b, 1, 255)
                        cv2.fillPoly(im, b, 255)
                        mask = im
                        masked = cv2.bitwise_and(im2, im2, mask=mask)

                        warped_im2 = face_landmark_detection.warp_im(masked, M, im1.shape)
                    front_img = demo.demo(warped_im2)
                    print(target_filename)
                    cv2.imwrite(target_filename, front_img)

                else:
                    break
                print(count)
                count += 1

with concurrent.futures.ProcessPoolExecutor() as executor:
    subfiles = glob.glob("/media/ustb/Dataset/biovid/PartA/video/*")
    for image_file in zip(subfiles,executor.map(deal_with_filenanme,subfiles)):
        print(subfiles)
