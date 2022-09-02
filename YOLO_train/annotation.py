from turtle import width
import cv2 as cv
import sys
import numpy as np
import argparse
import copy
import os

from scipy.ndimage.filters import gaussian_filter
from scipy import spatial
import glob


refPt = []
keypoints = 0
_y = 0

def click_and_crop(event, x, y, flags, param):
    '''Grab coordinates from click and release in opencv window'''
	# grab references to the global variables
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
    
    global refPt 
    global keypoints
    if event == cv.EVENT_LBUTTONDOWN:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        print('keypoint: ',x,y)
        
        keypoints += 1
        #xy in opencv standard but yx is the corresponding numpy index 
        refPt.append((y, x))

        print(refPt)
        # draw a rectangle around the region of interest       
    elif event==cv.EVENT_LBUTTONUP:
        print('keypoint: ',x,y)
        
        keypoints += 1
        #xy in opencv standard but yx is the corresponding numpy index 
        refPt.append((y, x))

        print(refPt)

def select_keypoints(output):
    '''Get selected Keypoints from opencv window'''
    cv.namedWindow("image")
    
    cv.setMouseCallback("image", click_and_crop)
    global refPt
    global keypoints
    
    refPt = []
    keypoints=0

    while keypoints<2:
        
        if len(refPt)>=2 and keypoints%2==0:
            cv.line(output, np.flip(refPt[-2]), np.flip(refPt[-1]), 125, thickness=3)
        cv.imshow('image', output)
        if cv.waitKey(33) == ord('a'):
            break

    #cv.line(output, np.flip(refPt[-2]), np.flip(refPt[-1]), 125, thickness=3)
    #cv.imshow('image', output)
    print(refPt)
    
    return np.asarray(refPt)

def work_on_sample(filepath, file_dst, num):

    dst_path = file_dst+'/rgb/'+str(num).zfill(5)+'.jpg'
    shoe_on_dina = cv.imread(filepath)
    shoe_on_dina = cv.resize( shoe_on_dina, (800,600))
    if not os.path.exists(dst_path):
        output_ = copy.deepcopy(shoe_on_dina)
        kp = select_keypoints(output_)

        cv.imwrite(file_dst+'/imgs/'+'99'+str(num).zfill(5)+'.png',shoe_on_dina) 
        np.savetxt(file_dst+'/kp/'+'99'+str(num).zfill(5)+'.csv', kp, delimiter = ',')


def main_():
    '''Opens window to click on desired keypoints.
        Saves: outline of shoe,
               clicked keypoints,
               nearest neighbor of keypoint to outline'''

    data_dir = './datasets/rgb1'
    foot_objects = sorted(os.listdir(data_dir))
    print(foot_objects)
    dst_path = 'datasets/annotations'
    if not os.path.isdir(dst_path):  
        os.mkdir(dst_path)
        os.mkdir(dst_path+'/imgs')
        os.mkdir(dst_path+'/kp')


    for i,foot_path in enumerate(foot_objects):
        print(foot_path)
        work_on_sample(os.path.join(data_dir,foot_path), dst_path, i)

'''
shoes = sorted(os.listdir(dst_path+'/imgs'))
kps = sorted(os.listdir(dst_path+'/kp'))

for kp, shoe in zip(kps,shoes):
    outline_ = cv.imread(os.path.join(dst_path,'imgs',shoe))
    outline_ = cv.cvtColor(outline_, cv.COLOR_BGR2GRAY)
    kp_ = np.genfromtxt(os.path.join(dst_path,'kp',kp), delimiter = ',')
    new_kp = fit_kp_to_ctr(outline_, kp_)
    np.savetxt(os.path.join(dst_path,'new_kp',kp), new_kp, delimiter = ',')
'''
def vis_bbox():
    data_dir = './datasets/annotations/imgs'
    label_dir = './datasets/annotations/kp'
    imgs = sorted(os.listdir(data_dir))
    kps = sorted(os.listdir(label_dir))

    for img_p,kp_p in zip(imgs,kps):
        kp = np.genfromtxt(label_dir +'/'+ kp_p, delimiter = ',')
        img = cv.imread(data_dir+'/'+img_p)
        print(img_p, kp_p)
        try:
            if kp.shape==(2,2):
                cv.rectangle(img, np.flip(kp[0]).astype(int), 
                                np.flip(kp[1]).astype(int), (0,0,255), 2)
        except:
            import ipdb;ipdb.set_trace()
        cv.imshow('image', img)
        cv.waitKey(0)


def video2imgs():
    vidcap = cv.VideoCapture('video.mp4')
    success,image = vidcap.read()
    count = 0
    while success:
        cv.imwrite("datasets/rgb1/frame%d.jpg" % count, image)     # save frame as JPEG file      
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

main_()