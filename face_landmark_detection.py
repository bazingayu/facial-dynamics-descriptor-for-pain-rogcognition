#coding = utf-8

import cv2
import dlib
import numpy as np
def extract_landmark(path):
	img = cv2.imread(path)
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("/home/yjw/python/face_landmark/shape_predictor_68_face_landmarks.dat")

	dets = detector(gray,1)

	for face in dets:
		shape = predictor(img,face)
		#print(shape.parts())
		'''for pt in shape.parts():
			pt_pos = (pt.x,pt.y)
			cv2.circle(img,pt_pos,2,(0,255,0),1)'''
	
	if(len(dets)==1):
		return img,1,np.matrix([[p.x,p.y]for p in predictor(img,dets[0]).parts()])
	else:
		return img,0,0


def extract_landmark1(img):

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("/home/yjw/python/face_landmark/shape_predictor_68_face_landmarks.dat")

	dets = detector(gray, 1)

	for face in dets:
		shape = predictor(img, face)
		# print(shape.parts())
		'''for pt in shape.parts():
			pt_pos = (pt.x,pt.y)
			cv2.circle(img,pt_pos,2,(0,255,0),1)'''

	if (len(dets) == 1):
		return img, 1, np.matrix([[p.x, p.y] for p in predictor(img, dets[0]).parts()])
	else:
		return img, 0, 0
	
def transformation_from_points(points1,points2):
	points1 = points1.astype(np.float64)
	points2 = points2.astype(np.float64)

	c1 = np.mean(points1,axis=0)
	c2 = np.mean(points2,axis=0)
	points1 -= c1
	points2 -= c2
	
	s1 = np.std(points1)
	s2 = np.std(points2)
	points1 /= s1
	points2 /= s2

	U,S,Vt = np.linalg.svd(points1.T * points2)
	R = (U*Vt).T
	#print('now')
	return np.vstack([np.hstack(((s2/s1)*R,c2.T-(s2/s1)*R*c1.T)),np.matrix([0.,0.,1.])])

def warp_im(im,M,dshape):
	output_im = np.zeros(dshape,dtype=im.dtype)
	cv2.warpAffine(im,M[:2],(dshape[1],dshape[0]),dst=output_im,borderMode=cv2.BORDER_TRANSPARENT,flags=cv2.WARP_INVERSE_MAP)
	return output_im
def face_align(base_path,cover_path,deter):
	im1,flag,landmarks1 = extract_landmark(base_path)
	if deter == 1:
		im2,flag,landmark2 = extract_landmark(cover_path)
	else:
		#cover path is a image now
		im2,flag,landmark2 = extract_landmark1(cover_path)
	lm2 = landmark2
	if flag == 1:
		M = transformation_from_points(landmarks1,lm2)
	else:
		M=[1,1]
		landmark2=1
	return im1,im2,M,landmarks1,lm2
	












