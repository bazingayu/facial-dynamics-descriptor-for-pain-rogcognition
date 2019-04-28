import glob
import dlib
import cv2
import numpy as np
import math
import os
from scipy.stats import iqr
from matplotlib.mlab import prctile
from statsmodels import robust
import concurrent.futures

def extract_landmark(path):
    #print(path)
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/home/yjw/python/face_landmark/shape_predictor_68_face_landmarks.dat")
    dets = detector(gray, 1)
    for face in dets:
        shape = predictor(img, face)
        # print(shape.parts())
        for pt in shape.parts():
            pt_pos = (pt.x,pt.y)
            cv2.circle(img,pt_pos,2,(0,255,0),1)
    if (len(dets) == 1):
        return img, 1, np.matrix([[p.x, p.y] for p in predictor(img, dets[0]).parts()])
    else:
        return img, 0, 0
def extract_distance(landmark):
    size = landmark[16,0] - landmark[0,0]
    debl = -((landmark[19,0] - landmark[37,0]) ** 2 + (landmark[19,1] - landmark[37,1]) ** 2) ** 0.5 / size
    debr = -((landmark[24,0] - landmark[44,0]) ** 2 + (landmark[24,1] - landmark[44,1]) ** 2) ** 0.5 / size
    del1 = -((landmark[38,0] - landmark[41,0]) ** 2 + (landmark[38,1] - landmark[41,1]) ** 2) ** 0.5 / size
    der1 = -((landmark[43,0] - landmark[46,0]) ** 2 + (landmark[43,1] - landmark[46,1]) ** 2) ** 0.5 / size
    dbml = -((landmark[19,0] - landmark[48,0]) ** 2 + (landmark[19,1] - landmark[48,1]) ** 2) ** 0.5 / size
    dbmr = -((landmark[24,0] - landmark[54,0]) ** 2 + (landmark[24,1] - landmark[54,1]) ** 2) ** 0.5 / size
    deml = -((landmark[36,0] - landmark[48,0]) ** 2 + (landmark[36,1] - landmark[48,1]) ** 2) ** 0.5 / size
    demr = -((landmark[45,0] - landmark[54,0]) ** 2 + (landmark[45,1] - landmark[54,1]) ** 2) ** 0.5 / size
    dmw1 = ((landmark[48,0] - landmark[54,0]) ** 2 + (landmark[48,1] - landmark[54,1]) ** 2) ** 0.5 / size
    dwh1 = ((landmark[51,0] - landmark[57,0]) ** 2 + (landmark[51,1] - landmark[57,1]) ** 2) ** 0.5 / size

    return debl,debr,del1,der1,dbml,dbmr,deml,demr,dmw1,dwh1
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
    signal = np.zeros([138])
    signal1 = signal2 = signal
    seq_descr = np.zeros([m,48])
    print('start extracting signal')
    for i in range(m):
        signal_all = np.zeros([3,138])
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
            descr[14,k] = 0.001 * np.sum(sig - s_min)
            descr[15,k] = 0.01 * np.sum(sig - s_min) / (s_max - s_min)
        #print(descr)
        descr = descr.reshape([1,48])
        seq_descr[i,:] = descr
    seq_descr = seq_descr.reshape([1,48*m])
    return seq_descr
def sobel_demo(image):
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0)   #对x求一阶导
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1)   #对y求一阶导
    gradx = cv2.convertScaleAbs(grad_x)  #用convertScaleAbs()函数将其转回原来的uint8形式
    grady = cv2.convertScaleAbs(grad_y)
    gradxy = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0) #图片融合
    mean_grad = np.mean(gradxy)

    return mean_grad
def extract_texture(img,landmark):
    gra = []
    pst1 = np.float32([[landmark[21,0], landmark[21,1]], [landmark[22,0], landmark[22,1]], [landmark[39,0], landmark[39,1]], [landmark[42,0], landmark[42,1]]])
    pst2 = np.float32([[0, 0], [0,49], [49,0], [49, 49]])
    M = cv2.getPerspectiveTransform(pst1, pst2)
    dst1 = cv2.warpPerspective(img, M, (49, 49))
    dst1_gray = cv2.cvtColor(dst1,cv2.COLOR_BGR2GRAY)
    gra1 = sobel_demo(dst1_gray)
    pst1 = np.float32([[landmark[2, 0], landmark[2, 1]], [landmark[31, 0], landmark[31, 1]], [landmark[5, 0], landmark[5, 1]],[landmark[7, 0], landmark[7, 1]]])
    pst2 = np.float32([[0, 0], [0, 49], [49, 0], [49, 49]])
    M = cv2.getPerspectiveTransform(pst1, pst2)
    dst2 = cv2.warpPerspective(img, M, (49, 49))
    dst2_gray = cv2.cvtColor(dst2, cv2.COLOR_BGR2GRAY)
    gra2 = sobel_demo(dst2_gray)
    pst1 = np.float32([[landmark[35, 0], landmark[35, 1]], [landmark[14, 0], landmark[14, 1]], [landmark[9, 0], landmark[9, 1]],[landmark[12, 0], landmark[12, 1]]])
    pst2 = np.float32([[0, 0], [0, 49], [49, 0], [49, 49]])
    M = cv2.getPerspectiveTransform(pst1, pst2)
    dst3 = cv2.warpPerspective(img, M, (49, 49))
    dst3_gray = cv2.cvtColor(dst3, cv2.COLOR_BGR2GRAY)
    gra3 = sobel_demo(dst3_gray)
    return gra1,gra2,gra3
def _largest_face(dets):
    if len(dets) == 1:
        return 0
    face_areas = [ (det.right()-det.left())*(det.bottom()-det.top()) for det in dets]
    largest_area = face_areas[0]
    largest_index = 0
    for index in range(1, len(dets)):
        if face_areas[index] > largest_area :
            largest_index = index
            largest_area= face_areas[index]
    print("largest_face index is {} in {} faces".format(largest_index, len(dets)))
    return largest_index
# 从dlib的检测结果抽取姿态估计需要的点坐标
def get_image_points_from_landmark_shape(landmark_shape):
    if landmark_shape.num_parts != 68:
        print("ERROR:landmark_shape.num_parts-{}".format(landmark_shape.num_parts))
        return -1, None
    #2D image points. If you change the image, you need to change vector
    image_points = np.array([ (landmark_shape.part(30).x, landmark_shape.part(30).y),
                              (landmark_shape.part(8).x, landmark_shape.part(8).y),
                              (landmark_shape.part(36).x, landmark_shape.part(36).y),
                              (landmark_shape.part(45).x, landmark_shape.part(45).y),
                              (landmark_shape.part(48).x, landmark_shape.part(48).y),
                              (landmark_shape.part(54).x, landmark_shape.part(54).y) ], dtype="double")
    return 0, image_points
# 用dlib检测关键点，返回姿态估计需要的几个点坐标
def get_image_points(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    POINTS_NUM_LANDMARK = 68
    gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY ) # 图片调整为灰色
    dets = detector( img, 0 )
    if 0 == len( dets ):
        print( "ERROR: found no face" )
        return -1, None
    largest_index = _largest_face(dets)
    face_rectangle = dets[largest_index]
    landmark_shape = predictor(img, face_rectangle)
    return get_image_points_from_landmark_shape(landmark_shape)
# 获取旋转向量和平移向量
def get_pose_estimation(img_size, image_points ):
    # 3D model points.
    model_points = np.array([ (0.0, 0.0, 0.0),
                              (0.0, -330.0, -65.0), # Chin
                             (-225.0, 170.0, -135.0),
                              (225.0, 170.0, -135.0),
                              (-150.0, -150.0, -125.0),
                              (150.0, -150.0, -125.0)  ])
    # Camera internals
    focal_length = img_size[1]
    center = (img_size[1]/2, img_size[0]/2)
    camera_matrix = np.array( [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype = "double" )
    #print("Camera Matrix :{}".format(camera_matrix))
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE )
    #print("Rotation Vector:\n {}".format(rotation_vector))
    #print("Translation Vector:\n {}".format(translation_vector))
    return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs
# 从旋转向量转换为欧拉角
def get_euler_angle(rotation_vector):
    # calculate rotation angles
    theta = cv2.norm(rotation_vector, cv2.NORM_L2) #
    #transformed to quaterniond
    w = math.cos(theta / 2)
    x = math.sin(theta / 2)*rotation_vector[0][0] / theta
    y = math.sin(theta / 2)*rotation_vector[1][0] / theta
    z = math.sin(theta / 2)*rotation_vector[2][0] / theta
    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)
    #print('t0:{}, t1:{}'.format(t0, t1))
    pitch = math.atan2(t0, t1)
    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)
    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)
    #print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))
    # 单位转换：将弧度转换为度
    #Y = int((pitch/math.pi)*180)
    #X = int((yaw/math.pi)*180)
    #Z = int((roll/math.pi)*180)
    Y = pitch
    X = yaw
    Z = roll
    return 0, Y, X, Z
def get_pose_estimation_in_euler_angle(landmark_shape, im_szie):
    try:
        ret, image_points = get_image_points_from_landmark_shape(landmark_shape)
        if ret != 0:
            print('get_image_points failed')
            return -1, None, None, None
        ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(im_szie, image_points)
        if ret != True:
            print('get_pose_estimation failed')
            return -1, None, None, None
        ret, pitch, yaw, roll = get_euler_angle(rotation_vector)
        if ret != 0:
            print('get_euler_angle failed')
            return -1, None, None, None
        euler_angle_str = 'Y:{}, X:{}, Z:{}'.format(pitch, yaw, roll)
        #print(euler_angle_str)
        return 0, pitch, yaw, roll
    except Exception as e:
        #print('get_pose_estimation_in_euler_angle exception:{}'.format(e))
        return -1, None, None, None
def extract_euler_angle(filename,img):
    im = cv2.imread(filename)
    if im is None:
        print("Problem img",filename)
        return 0,0,0,0
    size = im.shape 
    if size[0] > 700:
        h = size[0] / 3
        w = size[1] / 3
        im = cv2.resize(im, (int(w), int(h)), interpolation=cv2.INTER_CUBIC)
        size = im.shape
    ret, image_points = get_image_points(im)
    if ret != 0:
        print('get_image_points failed')
        return 0,0,0,0
    ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(size, image_points)
    if ret != True:
        print('get_pose_estimation failed')
        # continue
    ret, pitch, yaw, roll = get_euler_angle(rotation_vector)
    return 1,pitch,yaw,roll

def deal_with_filename(subdir):
    s = subdir.split('/')
    name = "/media/ustb/Dataset/biovid/PartA/yudata/"
    dis_name = name + 'dis/' + s[7] + '/' + s[8] + '.npy'
    if not os.path.exists(dis_name):
        files = os.listdir(subdir)
        files.sort(key=lambda x: int(x[:-4]))
        s_dis = []
        s_gra = []
        s_Euler = []
        s1 = subdir.split('/')
        gra1 = [0,0,0]
        dis1 = [0,0,0,0,0,0,0,0,0,0]
        Euler1 = [0,0,0]
        for filename1 in files:
            filename = subdir + '/' + filename1
            print(filename)
            angle_filename = '/media/ustb/Dataset/biovid/PartA/front_img2/' + s1[7] + '/' + s1[8] + '/' + filename1
            img ,deter ,landmark = extract_landmark(filename)
            if deter == 0:
                print('no face detected')
                gra = gra1
                dis = dis1
                Euler = Euler1
            else:
                debl, debr, del1, der1, dbml, dbmr, deml, demr, dmw1, dwh1 = extract_distance(landmark)
                dis = [debl,debr,del1,der1,dbml,dbmr,deml,demr,dmw1,dwh1]
                gra1 ,gra2 ,gra3 = extract_texture(img , landmark)
                gra = [gra1, gra2, gra3]
                    
                deter,pitch,yaw,roll = extract_euler_angle(angle_filename,img)
                if deter == 1:
                    Euler = [pitch,yaw,roll]
                else:
                    Euler = Euler1
            gra1 = gra
            dis1 = dis
            Euler1 = Euler
            s_dis.append(dis)
            s_gra.append(gra)
            s_Euler.append(Euler)
        print("start sequence deal")
        dis = np.array(s_dis)
        print("1")
        gra = np.array(s_gra)
        print("2")
        Euler = np.array(s_Euler)
        print("3")
        dis_descr = extract_signal(dis).tolist()
        print("4")
        gra_descr = extract_signal(gra).tolist()
        print("5")
        Euler_descr = extract_signal(Euler).tolist()
        print("6")
        dis_descr1 = np.array( dis_descr[0] )
        print("7")
        gra_descr1 = np.array( gra_descr[0] )
        print("8")
        Euler_descr1 = np.array( Euler_descr[0] )
        print("9")
        s = subdir.split('/')
        name = "/media/ustb/Dataset/biovid/PartA/yudata/"
        dis_name = name + 'dis/' + s[7] + '/' + s[8] + '.npy'
        print('dis_name',dis_name)
        gra_name = name + 'gra/' + s[7] + '/' + s[8] + '.npy'
        Euler_name = name + 'Euler/' + s[7] + '/' + s[8] + '.npy'
        np.save(dis_name,dis_descr1)
        np.save(gra_name,gra_descr1)
        np.save(Euler_name,Euler_descr1)
    else:
        print('alreay exists')
for i in range(5):
    with concurrent.futures.ProcessPoolExecutor(6) as executor:
        subfiles = glob.glob("/media/ustb/Dataset/biovid/PartA/frame1381/" + str(i) + "/*")
        count = 0
        for image_file in zip(subfiles,executor.map(deal_with_filename,subfiles)):
            print(count)
            count += 1
